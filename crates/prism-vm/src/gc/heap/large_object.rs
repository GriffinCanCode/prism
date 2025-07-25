//! Large Object Management for Heap
//!
//! This module manages large objects within the heap, providing specialized
//! handling for objects above the large object threshold.
//!
//! **Large Object Responsibilities:**
//! - Track large objects separately from size classes
//! - Manage large object free list with best-fit allocation
//! - Monitor large object fragmentation and lifetime
//! - Coordinate with compaction for large object optimization
//!
//! **NOT Large Object Responsibilities (delegated):**
//! - Raw memory allocation (handled by allocators::LargeObjectAllocator)
//! - Virtual memory management (handled by allocators::PageAllocator)
//! - Cross-generational references (handled by card table)

use super::types::*;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::{VecDeque, BTreeMap};
use std::ptr::NonNull;
use std::time::Instant;

/// Large object allocator for heap-level management
/// 
/// This manages large objects within the heap, providing efficient
/// allocation strategies and specialized tracking for objects above
/// the large object threshold.
pub struct LargeObjectAllocator {
    /// Free large objects organized by size for best-fit allocation
    free_objects: Mutex<BTreeMap<usize, VecDeque<LargeObjectEntry>>>,
    
    /// Tracked large objects for GC and statistics
    tracked_objects: RwLock<Vec<TrackedLargeObject>>,
    
    /// Configuration
    config: LargeObjectConfig,
    
    /// Statistics
    stats: LargeObjectStats,
    
    /// Fragmentation tracker
    fragmentation_tracker: Mutex<LargeObjectFragmentationTracker>,
    
    /// Lifetime tracker for optimization
    lifetime_tracker: Mutex<LargeObjectLifetimeTracker>,
}

/// Configuration for large object management
#[derive(Debug, Clone)]
pub struct LargeObjectConfig {
    /// Threshold for considering an object "large"
    pub large_object_threshold: usize,
    
    /// Maximum number of tracked large objects
    pub max_tracked_objects: usize,
    
    /// Enable lifetime tracking
    pub track_lifetimes: bool,
    
    /// Enable fragmentation monitoring
    pub monitor_fragmentation: bool,
    
    /// Best-fit vs first-fit allocation strategy
    pub use_best_fit: bool,
    
    /// Coalescing threshold (minimum gap to coalesce)
    pub coalescing_threshold: usize,
}

impl Default for LargeObjectConfig {
    fn default() -> Self {
        Self {
            large_object_threshold: LARGE_OBJECT_THRESHOLD,
            max_tracked_objects: 10000,
            track_lifetimes: true,
            monitor_fragmentation: true,
            use_best_fit: true,
            coalescing_threshold: 64, // 64 bytes minimum gap
        }
    }
}

/// Statistics for large object management
#[derive(Debug, Default)]
pub struct LargeObjectStats {
    /// Total large objects allocated
    total_allocated: AtomicUsize,
    /// Total large objects deallocated
    total_deallocated: AtomicUsize,
    /// Current live large objects
    live_objects: AtomicUsize,
    /// Total bytes in large objects
    total_bytes: AtomicUsize,
    /// Largest object size seen
    largest_object: AtomicUsize,
    /// Average object size
    average_object_size: AtomicUsize,
    /// Fragmentation waste
    fragmentation_waste: AtomicUsize,
    /// Successful coalescing operations
    coalescing_operations: AtomicUsize,
}

/// Tracked large object for GC and analysis
#[derive(Debug, Clone)]
pub struct TrackedLargeObject {
    /// Pointer to the object
    pub ptr: NonNull<u8>,
    /// Size of the object
    pub size: usize,
    /// Allocation timestamp
    pub allocated_at: Instant,
    /// Last access timestamp
    pub last_access: Instant,
    /// Object generation (for generational GC)
    pub generation: u8,
    /// Reference count (for hybrid GC)
    pub ref_count: u32,
    /// Object type identifier
    pub type_id: u32,
}

/// Fragmentation tracking for large objects
#[derive(Debug, Default)]
struct LargeObjectFragmentationTracker {
    /// Fragmentation snapshots over time
    snapshots: Vec<FragmentationSnapshot>,
    /// Current fragmentation metrics
    current_metrics: LargeObjectFragmentationMetrics,
    /// Last analysis timestamp
    last_analysis: Option<Instant>,
}

#[derive(Debug, Default)]
struct LargeObjectFragmentationMetrics {
    /// Total free space in large object area
    total_free_space: usize,
    /// Number of free blocks
    free_block_count: usize,
    /// Largest free block
    largest_free_block: usize,
    /// Average free block size
    average_free_block_size: f64,
    /// Fragmentation ratio
    fragmentation_ratio: f64,
    /// Wasted space due to alignment/fragmentation
    wasted_space: usize,
}

/// Lifetime tracking for large objects
#[derive(Debug, Default)]
struct LargeObjectLifetimeTracker {
    /// Lifetime samples
    lifetime_samples: Vec<LifetimeSample>,
    /// Average lifetime per size class
    average_lifetimes: BTreeMap<usize, std::time::Duration>,
    /// Lifetime distribution analysis
    lifetime_distribution: LifetimeDistribution,
}

#[derive(Debug)]
struct LifetimeSample {
    size: usize,
    lifetime: std::time::Duration,
    timestamp: Instant,
}

#[derive(Debug, Default)]
struct LifetimeDistribution {
    /// Short-lived objects (< 1 second)
    short_lived: usize,
    /// Medium-lived objects (1 second - 1 minute)
    medium_lived: usize,
    /// Long-lived objects (> 1 minute)
    long_lived: usize,
}

impl LargeObjectAllocator {
    /// Create a new large object allocator
    pub fn new() -> Self {
        Self::with_config(LargeObjectConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: LargeObjectConfig) -> Self {
        Self {
            free_objects: Mutex::new(BTreeMap::new()),
            tracked_objects: RwLock::new(Vec::new()),
            config,
            stats: LargeObjectStats::default(),
            fragmentation_tracker: Mutex::new(LargeObjectFragmentationTracker::default()),
            lifetime_tracker: Mutex::new(LargeObjectLifetimeTracker::default()),
        }
    }
    
    /// Try to allocate a large object from free list
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        if size < self.config.large_object_threshold {
            return None; // Not a large object
        }
        
        let ptr = if self.config.use_best_fit {
            self.allocate_best_fit(size)
        } else {
            self.allocate_first_fit(size)
        };
        
        if let Some(ptr) = ptr {
            // Track the allocation
            self.track_allocation(ptr, size);
            
            // Update statistics
            self.stats.total_allocated.fetch_add(1, Ordering::Relaxed);
            self.stats.live_objects.fetch_add(1, Ordering::Relaxed);
            self.stats.total_bytes.fetch_add(size, Ordering::Relaxed);
            
            // Update largest object
            let current_largest = self.stats.largest_object.load(Ordering::Relaxed);
            if size > current_largest {
                self.stats.largest_object.store(size, Ordering::Relaxed);
            }
            
            // Update average size
            self.update_average_size();
        }
        
        ptr
    }
    
    /// Deallocate a large object
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        if size < self.config.large_object_threshold {
            return; // Not a large object
        }
        
        // Remove from tracking
        self.untrack_allocation(ptr, size);
        
        // Add to free list with coalescing
        self.add_to_free_list_with_coalescing(ptr, size);
        
        // Update statistics
        self.stats.total_deallocated.fetch_add(1, Ordering::Relaxed);
        self.stats.live_objects.fetch_sub(1, Ordering::Relaxed);
        self.stats.total_bytes.fetch_sub(size, Ordering::Relaxed);
        
        // Update average size
        self.update_average_size();
        
        // Update fragmentation tracking
        if self.config.monitor_fragmentation {
            self.update_fragmentation_metrics();
        }
    }
    
    /// Best-fit allocation strategy
    fn allocate_best_fit(&self, size: usize) -> Option<NonNull<u8>> {
        let mut free_objects = self.free_objects.lock().unwrap();
        
        // Find the smallest block that can fit the request
        let mut best_size = None;
        for (&block_size, _) in free_objects.iter() {
            if block_size >= size {
                best_size = Some(block_size);
                break;
            }
        }
        
        if let Some(block_size) = best_size {
            if let Some(mut blocks) = free_objects.remove(&block_size) {
                if let Some(entry) = blocks.pop_front() {
                    // Put remaining blocks back if any
                    if !blocks.is_empty() {
                        free_objects.insert(block_size, blocks);
                    }
                    
                    // Split block if it's significantly larger
                    self.split_block_if_beneficial(entry, size, &mut free_objects)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// First-fit allocation strategy
    fn allocate_first_fit(&self, size: usize) -> Option<NonNull<u8>> {
        let mut free_objects = self.free_objects.lock().unwrap();
        
        // Find first block that can fit the request
        for (&block_size, blocks) in free_objects.iter_mut() {
            if block_size >= size && !blocks.is_empty() {
                let entry = blocks.pop_front().unwrap();
                
                // Split block if beneficial
                return self.split_block_if_beneficial(entry, size, &mut free_objects);
            }
        }
        
        None
    }
    
    /// Split a block if it's beneficial to do so
    fn split_block_if_beneficial(
        &self,
        entry: LargeObjectEntry,
        requested_size: usize,
        free_objects: &mut BTreeMap<usize, VecDeque<LargeObjectEntry>>,
    ) -> Option<NonNull<u8>> {
        let remaining_size = entry.size - requested_size;
        
        // Only split if remainder is large enough to be useful
        if remaining_size >= self.config.coalescing_threshold {
            // Create remainder block
            if let Some(remainder_ptr) = NonNull::new(unsafe { 
                entry.ptr.as_ptr().add(requested_size) 
            }) {
                let remainder_entry = LargeObjectEntry {
                    ptr: remainder_ptr,
                    size: remaining_size,
                    allocated_at: Instant::now(),
                    last_access: Instant::now(),
                };
                
                // Add remainder to free list
                free_objects
                    .entry(remaining_size)
                    .or_insert_with(VecDeque::new)
                    .push_back(remainder_entry);
            }
        }
        
        Some(entry.ptr)
    }
    
    /// Add block to free list with coalescing
    fn add_to_free_list_with_coalescing(&self, ptr: NonNull<u8>, size: usize) {
        let mut free_objects = self.free_objects.lock().unwrap();
        
        // Try to coalesce with adjacent blocks
        let coalesced_entry = self.try_coalesce(ptr, size, &mut free_objects);
        
        // Add to appropriate size bucket
        free_objects
            .entry(coalesced_entry.size)
            .or_insert_with(VecDeque::new)
            .push_back(coalesced_entry);
    }
    
    /// Try to coalesce with adjacent free blocks
    fn try_coalesce(
        &self,
        ptr: NonNull<u8>,
        size: usize,
        free_objects: &mut BTreeMap<usize, VecDeque<LargeObjectEntry>>,
    ) -> LargeObjectEntry {
        let mut coalesced_ptr = ptr;
        let mut coalesced_size = size;
        let ptr_addr = ptr.as_ptr() as usize;
        
        // Look for adjacent blocks to coalesce
        let mut to_remove = Vec::new();
        
        for (&block_size, blocks) in free_objects.iter_mut() {
            let mut indices_to_remove = Vec::new();
            
            for (index, entry) in blocks.iter().enumerate() {
                let entry_addr = entry.ptr.as_ptr() as usize;
                
                // Check if blocks are adjacent
                if entry_addr + entry.size == ptr_addr {
                    // Entry comes before our block
                    coalesced_ptr = entry.ptr;
                    coalesced_size += entry.size;
                    indices_to_remove.push(index);
                } else if ptr_addr + coalesced_size == entry_addr {
                    // Entry comes after our block
                    coalesced_size += entry.size;
                    indices_to_remove.push(index);
                }
            }
            
            // Remove coalesced blocks (in reverse order to maintain indices)
            for &index in indices_to_remove.iter().rev() {
                blocks.remove(index);
            }
            
            if blocks.is_empty() {
                to_remove.push(block_size);
            }
            
            if !indices_to_remove.is_empty() {
                self.stats.coalescing_operations.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Remove empty size buckets
        for size in to_remove {
            free_objects.remove(&size);
        }
        
        LargeObjectEntry {
            ptr: coalesced_ptr,
            size: coalesced_size,
            allocated_at: Instant::now(),
            last_access: Instant::now(),
        }
    }
    
    /// Track a large object allocation
    fn track_allocation(&self, ptr: NonNull<u8>, size: usize) {
        let mut tracked = self.tracked_objects.write().unwrap();
        
        // Check capacity
        if tracked.len() >= self.config.max_tracked_objects {
            // Remove oldest entry (simple LRU)
            tracked.remove(0);
        }
        
        tracked.push(TrackedLargeObject {
            ptr,
            size,
            allocated_at: Instant::now(),
            last_access: Instant::now(),
            generation: 0,
            ref_count: 1,
            type_id: 0, // Would be set by caller
        });
    }
    
    /// Untrack a large object allocation
    fn untrack_allocation(&self, ptr: NonNull<u8>, size: usize) {
        let mut tracked = self.tracked_objects.write().unwrap();
        
        // Find and remove the object
        if let Some(pos) = tracked.iter().position(|obj| obj.ptr == ptr) {
            let obj = tracked.remove(pos);
            
            // Track lifetime if enabled
            if self.config.track_lifetimes {
                let lifetime = Instant::now().duration_since(obj.allocated_at);
                self.record_lifetime(size, lifetime);
            }
        }
    }
    
    /// Record object lifetime for analysis
    fn record_lifetime(&self, size: usize, lifetime: std::time::Duration) {
        if let Ok(mut tracker) = self.lifetime_tracker.lock() {
            tracker.lifetime_samples.push(LifetimeSample {
                size,
                lifetime,
                timestamp: Instant::now(),
            });
            
            // Update distribution
            if lifetime.as_secs() < 1 {
                tracker.lifetime_distribution.short_lived += 1;
            } else if lifetime.as_secs() < 60 {
                tracker.lifetime_distribution.medium_lived += 1;
            } else {
                tracker.lifetime_distribution.long_lived += 1;
            }
            
            // Keep samples bounded
            if tracker.lifetime_samples.len() > 10000 {
                tracker.lifetime_samples.drain(0..5000);
            }
            
            // Update average lifetimes
            let count = tracker.average_lifetimes.get(&size).map_or(0, |_| 1) + 1;
            let current_avg = tracker.average_lifetimes.get(&size).copied().unwrap_or_default();
            let new_avg = (current_avg + lifetime) / count as u32;
            tracker.average_lifetimes.insert(size, new_avg);
        }
    }
    
    /// Update average object size
    fn update_average_size(&self) {
        let live_objects = self.stats.live_objects.load(Ordering::Relaxed);
        let total_bytes = self.stats.total_bytes.load(Ordering::Relaxed);
        
        if live_objects > 0 {
            let average = total_bytes / live_objects;
            self.stats.average_object_size.store(average, Ordering::Relaxed);
        }
    }
    
    /// Update fragmentation metrics
    fn update_fragmentation_metrics(&self) {
        if let Ok(mut tracker) = self.fragmentation_tracker.lock() {
            let free_objects = self.free_objects.lock().unwrap();
            
            let mut total_free_space = 0;
            let mut free_block_count = 0;
            let mut largest_free_block = 0;
            
            for (size, blocks) in free_objects.iter() {
                for _ in blocks {
                    total_free_space += size;
                    free_block_count += 1;
                    largest_free_block = largest_free_block.max(*size);
                }
            }
            
            let average_free_block_size = if free_block_count > 0 {
                total_free_space as f64 / free_block_count as f64
            } else {
                0.0
            };
            
            let fragmentation_ratio = FragmentationInfo::calculate_fragmentation_ratio(
                total_free_space,
                free_block_count,
            );
            
            tracker.current_metrics = LargeObjectFragmentationMetrics {
                total_free_space,
                free_block_count,
                largest_free_block,
                average_free_block_size,
                fragmentation_ratio,
                wasted_space: 0, // Would be calculated based on alignment waste
            };
            
            // Record snapshot
            tracker.snapshots.push(FragmentationSnapshot {
                timestamp: Instant::now(),
                fragmentation_ratio,
                largest_free_block,
                total_free_space,
            });
            
            // Keep snapshots bounded
            if tracker.snapshots.len() > 1000 {
                tracker.snapshots.drain(0..500);
            }
        }
    }
    
    /// Get all tracked large objects
    pub fn get_tracked_objects(&self) -> Vec<TrackedLargeObject> {
        self.tracked_objects.read().unwrap().clone()
    }
    
    /// Get large object statistics
    pub fn get_statistics(&self) -> LargeObjectStatistics {
        let fragmentation_metrics = self.fragmentation_tracker
            .lock()
            .unwrap()
            .current_metrics
            .clone();
        
        let lifetime_distribution = self.lifetime_tracker
            .lock()
            .unwrap()
            .lifetime_distribution
            .clone();
        
        LargeObjectStatistics {
            total_allocated: self.stats.total_allocated.load(Ordering::Relaxed),
            total_deallocated: self.stats.total_deallocated.load(Ordering::Relaxed),
            live_objects: self.stats.live_objects.load(Ordering::Relaxed),
            total_bytes: self.stats.total_bytes.load(Ordering::Relaxed),
            largest_object: self.stats.largest_object.load(Ordering::Relaxed),
            average_object_size: self.stats.average_object_size.load(Ordering::Relaxed),
            fragmentation_waste: self.stats.fragmentation_waste.load(Ordering::Relaxed),
            coalescing_operations: self.stats.coalescing_operations.load(Ordering::Relaxed),
            fragmentation_metrics,
            lifetime_distribution,
        }
    }
    
    /// Compact large object free lists
    pub fn compact(&self) -> usize {
        let mut total_coalesced = 0;
        let mut free_objects = self.free_objects.lock().unwrap();
        
        // Collect all free blocks
        let mut all_blocks = Vec::new();
        for (_, blocks) in free_objects.drain() {
            all_blocks.extend(blocks);
        }
        
        if all_blocks.is_empty() {
            return 0;
        }
        
        // Sort by address
        all_blocks.sort_by_key(|entry| entry.ptr.as_ptr() as usize);
        
        // Coalesce adjacent blocks
        let mut coalesced_blocks = Vec::new();
        let mut current = all_blocks.into_iter().next().unwrap();
        
        for next in all_blocks {
            let current_end = unsafe { current.ptr.as_ptr().add(current.size) };
            
            if current_end == next.ptr.as_ptr() {
                // Adjacent blocks - coalesce
                current.size += next.size;
                total_coalesced += 1;
            } else {
                // Not adjacent - keep current and move to next
                coalesced_blocks.push(current);
                current = next;
            }
        }
        coalesced_blocks.push(current);
        
        // Put coalesced blocks back
        for entry in coalesced_blocks {
            free_objects
                .entry(entry.size)
                .or_insert_with(VecDeque::new)
                .push_back(entry);
        }
        
        if total_coalesced > 0 {
            self.stats.coalescing_operations.fetch_add(total_coalesced, Ordering::Relaxed);
        }
        
        total_coalesced
    }
}

/// Detailed statistics for large object analysis
#[derive(Debug, Clone)]
pub struct LargeObjectStatistics {
    pub total_allocated: usize,
    pub total_deallocated: usize,
    pub live_objects: usize,
    pub total_bytes: usize,
    pub largest_object: usize,
    pub average_object_size: usize,
    pub fragmentation_waste: usize,
    pub coalescing_operations: usize,
    pub fragmentation_metrics: LargeObjectFragmentationMetrics,
    pub lifetime_distribution: LifetimeDistribution,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_object_allocator_creation() {
        let allocator = LargeObjectAllocator::new();
        let stats = allocator.get_statistics();
        
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.live_objects, 0);
    }

    #[test]
    fn test_large_object_threshold() {
        let allocator = LargeObjectAllocator::new();
        
        // Small object should return None
        let result = allocator.allocate(1024);
        assert!(result.is_none());
        
        // Large object allocation should work (after deallocating first)
        let large_ptr = NonNull::new(0x10000 as *mut u8).unwrap();
        allocator.deallocate(large_ptr, 64 * 1024);
        
        let result = allocator.allocate(64 * 1024);
        assert!(result.is_some());
    }

    #[test]
    fn test_coalescing() {
        let allocator = LargeObjectAllocator::new();
        
        // Create adjacent blocks
        let ptr1 = NonNull::new(0x10000 as *mut u8).unwrap();
        let ptr2 = NonNull::new(0x20000 as *mut u8).unwrap(); // Adjacent to ptr1
        
        allocator.deallocate(ptr1, 64 * 1024);
        allocator.deallocate(ptr2, 64 * 1024);
        
        let coalesced = allocator.compact();
        assert!(coalesced > 0);
    }

    #[test]
    fn test_tracking() {
        let allocator = LargeObjectAllocator::new();
        let large_ptr = NonNull::new(0x10000 as *mut u8).unwrap();
        
        // Deallocate first to have something to allocate
        allocator.deallocate(large_ptr, 64 * 1024);
        
        // Allocate and check tracking
        let result = allocator.allocate(64 * 1024);
        assert!(result.is_some());
        
        let tracked = allocator.get_tracked_objects();
        assert_eq!(tracked.len(), 1);
        assert_eq!(tracked[0].size, 64 * 1024);
    }
} 