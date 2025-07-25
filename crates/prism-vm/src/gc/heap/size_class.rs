//! Size Class Management for Heap Free Lists
//!
//! This module manages segregated free lists by size class within the heap.
//! It focuses on organizing already-allocated memory into efficient free lists.
//!
//! **Size Class Responsibilities:**
//! - Organize free memory blocks by size class
//! - Maintain segregated free lists for efficient allocation
//! - Track utilization and statistics per size class
//! - Coordinate with fragmentation management
//!
//! **NOT Size Class Responsibilities (delegated):**
//! - Raw memory allocation (handled by allocators::*)
//! - Thread-local caches (handled by allocators::ThreadCache)
//! - Page management (handled by allocators::PageAllocator)

use super::types::*;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::ptr::NonNull;

/// Size class allocator for managing heap free lists
/// 
/// This allocator manages segregated free lists for different object sizes,
/// providing efficient allocation from previously deallocated memory blocks.
pub struct SizeClassAllocator {
    /// Free lists for each size class
    size_classes: Vec<SizeClassFreeList>,
    
    /// Configuration
    config: SizeClassConfig,
    
    /// Global statistics
    stats: SizeClassGlobalStats,
    
    /// Utilization tracker for adaptive behavior
    utilization_tracker: Mutex<UtilizationTracker>,
}

/// Configuration for size class allocator
#[derive(Debug, Clone)]
pub struct SizeClassConfig {
    /// Size classes to use
    pub size_classes: Vec<usize>,
    /// Enable utilization tracking
    pub track_utilization: bool,
    /// Maximum objects per size class free list
    pub max_objects_per_class: usize,
    /// Enable adaptive size class adjustment
    pub adaptive_size_classes: bool,
}

impl Default for SizeClassConfig {
    fn default() -> Self {
        Self {
            size_classes: DEFAULT_SIZE_CLASSES.to_vec(),
            track_utilization: true,
            max_objects_per_class: 10000,
            adaptive_size_classes: false, // Disabled by default for stability
        }
    }
}

/// Free list for a specific size class
pub struct SizeClassFreeList {
    /// Size of objects in this class
    object_size: usize,
    
    /// Index of this size class
    class_index: usize,
    
    /// Free objects queue (FIFO for better cache behavior)
    free_objects: Mutex<VecDeque<FreeListEntry>>,
    
    /// Statistics for this size class
    stats: SizeClassStats,
    
    /// Last access time for LRU tracking
    last_access: AtomicUsize, // Timestamp as usize
}

/// Global statistics across all size classes
#[derive(Debug, Default)]
pub struct SizeClassGlobalStats {
    /// Total objects across all size classes
    total_objects: AtomicUsize,
    /// Total bytes managed
    total_bytes: AtomicUsize,
    /// Total allocations served
    total_allocations: AtomicUsize,
    /// Total deallocations received
    total_deallocations: AtomicUsize,
    /// Cache hit ratio
    cache_hit_ratio: AtomicUsize, // Stored as percentage * 100
}

/// Utilization tracking for adaptive behavior
#[derive(Debug, Default)]
struct UtilizationTracker {
    /// Utilization history per size class
    utilization_history: Vec<Vec<UtilizationSample>>,
    /// Allocation pattern analysis
    allocation_patterns: AllocationPatternAnalysis,
    /// Last analysis timestamp
    last_analysis: Option<std::time::Instant>,
}

#[derive(Debug)]
struct UtilizationSample {
    timestamp: std::time::Instant,
    utilization_ratio: f64,
    allocation_rate: f64,
    deallocation_rate: f64,
}

#[derive(Debug, Default)]
struct AllocationPatternAnalysis {
    /// Most frequently requested sizes
    hot_sizes: Vec<(usize, usize)>, // (size, frequency)
    /// Least frequently requested sizes
    cold_sizes: Vec<(usize, usize)>,
    /// Average object lifetime per size class
    average_lifetimes: Vec<std::time::Duration>,
    /// Fragmentation contribution per size class
    fragmentation_impact: Vec<f64>,
}

impl SizeClassAllocator {
    /// Create a new size class allocator
    pub fn new() -> Self {
        Self::with_config(SizeClassConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: SizeClassConfig) -> Self {
        let size_class_count = config.size_classes.len();
        let mut size_classes = Vec::with_capacity(size_class_count);
        
        // Initialize free lists for each size class
        for (index, &size) in config.size_classes.iter().enumerate() {
            size_classes.push(SizeClassFreeList::new(size, index));
        }
        
        let mut utilization_history = Vec::with_capacity(size_class_count);
        for _ in 0..size_class_count {
            utilization_history.push(Vec::new());
        }
        
        Self {
            size_classes,
            config,
            stats: SizeClassGlobalStats::default(),
            utilization_tracker: Mutex::new(UtilizationTracker {
                utilization_history,
                allocation_patterns: AllocationPatternAnalysis::default(),
                last_analysis: None,
            }),
        }
    }
    
    /// Try to allocate from free lists
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        // Find appropriate size class
        let size_class_index = self.find_size_class_index(size)?;
        
        // Try to allocate from the exact size class first
        if let Some(ptr) = self.allocate_from_size_class(size_class_index) {
            self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
            return Some(ptr);
        }
        
        // Try larger size classes if exact match failed
        for i in (size_class_index + 1)..self.size_classes.len() {
            if let Some(ptr) = self.allocate_from_size_class(i) {
                self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
                
                // Split the block if it's significantly larger
                self.handle_oversized_allocation(ptr, size, i);
                
                return Some(ptr);
            }
        }
        
        None
    }
    
    /// Deallocate memory to appropriate free list
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        if let Some(size_class_index) = self.find_size_class_index(size) {
            self.deallocate_to_size_class(ptr, size, size_class_index);
            self.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
            
            // Update utilization tracking
            if self.config.track_utilization {
                self.update_utilization_tracking(size_class_index);
            }
        }
    }
    
    /// Find the appropriate size class index for a given size
    fn find_size_class_index(&self, size: usize) -> Option<usize> {
        for (index, &class_size) in self.config.size_classes.iter().enumerate() {
            if size <= class_size {
                return Some(index);
            }
        }
        None
    }
    
    /// Allocate from a specific size class
    fn allocate_from_size_class(&self, size_class_index: usize) -> Option<NonNull<u8>> {
        if size_class_index >= self.size_classes.len() {
            return None;
        }
        
        let size_class = &self.size_classes[size_class_index];
        let mut free_objects = size_class.free_objects.lock().unwrap();
        
        if let Some(entry) = free_objects.pop_front() {
            // Update access time
            size_class.last_access.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as usize,
                Ordering::Relaxed
            );
            
            // Update statistics
            size_class.stats.allocations.fetch_add(1, Ordering::Relaxed);
            size_class.stats.free_objects.fetch_sub(1, Ordering::Relaxed);
            
            Some(entry.ptr)
        } else {
            None
        }
    }
    
    /// Deallocate to a specific size class
    fn deallocate_to_size_class(&self, ptr: NonNull<u8>, size: usize, size_class_index: usize) {
        if size_class_index >= self.size_classes.len() {
            return;
        }
        
        let size_class = &self.size_classes[size_class_index];
        let mut free_objects = size_class.free_objects.lock().unwrap();
        
        // Check if we're at capacity
        if free_objects.len() >= self.config.max_objects_per_class {
            // Could implement overflow handling here
            return;
        }
        
        // Add to free list
        free_objects.push_back(FreeListEntry { ptr, size });
        
        // Update statistics
        size_class.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        size_class.stats.free_objects.fetch_add(1, Ordering::Relaxed);
        size_class.stats.total_bytes.fetch_add(size, Ordering::Relaxed);
    }
    
    /// Handle allocation from oversized block (split if beneficial)
    fn handle_oversized_allocation(&self, ptr: NonNull<u8>, requested_size: usize, allocated_class_index: usize) {
        let allocated_size = self.config.size_classes[allocated_class_index];
        let remaining_size = allocated_size - requested_size;
        
        // Only split if the remainder is large enough to be useful
        if remaining_size >= 32 && remaining_size >= allocated_size / 4 {
            // Calculate remainder pointer
            if let Some(remainder_ptr) = NonNull::new(unsafe { ptr.as_ptr().add(requested_size) }) {
                // Find size class for remainder
                if let Some(remainder_class) = self.find_size_class_index(remaining_size) {
                    self.deallocate_to_size_class(remainder_ptr, remaining_size, remainder_class);
                }
            }
        }
    }
    
    /// Update utilization tracking for adaptive behavior
    fn update_utilization_tracking(&self, size_class_index: usize) {
        if let Ok(mut tracker) = self.utilization_tracker.try_lock() {
            let now = std::time::Instant::now();
            
            // Update utilization sample
            if size_class_index < tracker.utilization_history.len() {
                let size_class = &self.size_classes[size_class_index];
                let stats = &size_class.stats;
                
                let total_allocations = stats.allocations.load(Ordering::Relaxed);
                let total_deallocations = stats.deallocations.load(Ordering::Relaxed);
                let free_objects = stats.free_objects.load(Ordering::Relaxed);
                
                let utilization_ratio = if total_allocations > 0 {
                    (total_allocations - free_objects) as f64 / total_allocations as f64
                } else {
                    0.0
                };
                
                tracker.utilization_history[size_class_index].push(UtilizationSample {
                    timestamp: now,
                    utilization_ratio,
                    allocation_rate: 0.0, // Would calculate based on time delta
                    deallocation_rate: 0.0, // Would calculate based on time delta
                });
                
                // Keep history bounded
                let history = &mut tracker.utilization_history[size_class_index];
                if history.len() > 1000 {
                    history.drain(0..500);
                }
            }
            
            // Perform periodic analysis
            if tracker.last_analysis.is_none() || 
               now.duration_since(tracker.last_analysis.unwrap()).as_secs() > 60 {
                self.analyze_allocation_patterns(&mut tracker);
                tracker.last_analysis = Some(now);
            }
        }
    }
    
    /// Analyze allocation patterns for optimization
    fn analyze_allocation_patterns(&self, tracker: &mut UtilizationTracker) {
        let mut hot_sizes = Vec::new();
        let mut cold_sizes = Vec::new();
        
        // Analyze each size class
        for (index, size_class) in self.size_classes.iter().enumerate() {
            let allocations = size_class.stats.allocations.load(Ordering::Relaxed);
            let size = size_class.object_size;
            
            if allocations > 100 { // Threshold for "hot"
                hot_sizes.push((size, allocations));
            } else if allocations < 10 { // Threshold for "cold"
                cold_sizes.push((size, allocations));
            }
        }
        
        // Sort by frequency
        hot_sizes.sort_by(|a, b| b.1.cmp(&a.1));
        cold_sizes.sort_by(|a, b| a.1.cmp(&b.1));
        
        // Update analysis
        tracker.allocation_patterns.hot_sizes = hot_sizes;
        tracker.allocation_patterns.cold_sizes = cold_sizes;
    }
    
    /// Get statistics for all size classes
    pub fn get_statistics(&self) -> SizeClassStatistics {
        let mut per_class_stats = Vec::new();
        
        for size_class in &self.size_classes {
            per_class_stats.push(SizeClassStatDetail {
                object_size: size_class.object_size,
                class_index: size_class.class_index,
                allocations: size_class.stats.allocations.load(Ordering::Relaxed),
                deallocations: size_class.stats.deallocations.load(Ordering::Relaxed),
                free_objects: size_class.stats.free_objects.load(Ordering::Relaxed),
                total_bytes: size_class.stats.total_bytes.load(Ordering::Relaxed),
                utilization: size_class.stats.utilization.load(Ordering::Relaxed) as f64 / 10000.0,
                last_access: size_class.last_access.load(Ordering::Relaxed),
            });
        }
        
        SizeClassStatistics {
            per_class_stats,
            global_stats: SizeClassGlobalStatDetail {
                total_objects: self.stats.total_objects.load(Ordering::Relaxed),
                total_bytes: self.stats.total_bytes.load(Ordering::Relaxed),
                total_allocations: self.stats.total_allocations.load(Ordering::Relaxed),
                total_deallocations: self.stats.total_deallocations.load(Ordering::Relaxed),
                cache_hit_ratio: self.stats.cache_hit_ratio.load(Ordering::Relaxed) as f64 / 10000.0,
            },
        }
    }
    
    /// Compact free lists by coalescing adjacent blocks
    pub fn compact_free_lists(&self) -> usize {
        let mut total_coalesced = 0;
        
        for size_class in &self.size_classes {
            total_coalesced += self.compact_size_class_free_list(size_class);
        }
        
        total_coalesced
    }
    
    /// Compact a specific size class free list
    fn compact_size_class_free_list(&self, size_class: &SizeClassFreeList) -> usize {
        let mut free_objects = size_class.free_objects.lock().unwrap();
        let original_count = free_objects.len();
        
        if original_count < 2 {
            return 0;
        }
        
        // Convert to vector for sorting
        let mut objects: Vec<_> = free_objects.drain(..).collect();
        
        // Sort by address
        objects.sort_by_key(|entry| entry.ptr.as_ptr() as usize);
        
        // Coalesce adjacent blocks
        let mut coalesced = Vec::new();
        let mut current = objects.into_iter().next().unwrap();
        
        for next in objects {
            let current_end = unsafe { current.ptr.as_ptr().add(current.size) };
            
            if current_end == next.ptr.as_ptr() {
                // Adjacent blocks - coalesce
                current.size += next.size;
            } else {
                // Not adjacent - keep current and move to next
                coalesced.push(current);
                current = next;
            }
        }
        coalesced.push(current);
        
        // Put coalesced blocks back
        for entry in coalesced {
            free_objects.push_back(entry);
        }
        
        let coalesced_count = original_count - free_objects.len();
        coalesced_count
    }
    
    /// Get fragmentation information
    pub fn get_fragmentation_info(&self) -> FragmentationInfo {
        let mut total_free_space = 0;
        let mut free_block_count = 0;
        let mut largest_free_block = 0;
        
        for size_class in &self.size_classes {
            let free_objects = size_class.free_objects.lock().unwrap();
            
            for entry in free_objects.iter() {
                total_free_space += entry.size;
                free_block_count += 1;
                largest_free_block = largest_free_block.max(entry.size);
            }
        }
        
        let average_free_block_size = if free_block_count > 0 {
            total_free_space as f64 / free_block_count as f64
        } else {
            0.0
        };
        
        let fragmentation_ratio = FragmentationInfo::calculate_fragmentation_ratio(
            total_free_space, 
            free_block_count
        );
        
        FragmentationInfo {
            total_free_space,
            free_block_count,
            largest_free_block,
            average_free_block_size,
            fragmentation_ratio,
        }
    }
}

impl SizeClassFreeList {
    fn new(object_size: usize, class_index: usize) -> Self {
        Self {
            object_size,
            class_index,
            free_objects: Mutex::new(VecDeque::new()),
            stats: SizeClassStats::default(),
            last_access: AtomicUsize::new(0),
        }
    }
}

/// Detailed statistics for size class analysis
#[derive(Debug, Clone)]
pub struct SizeClassStatistics {
    pub per_class_stats: Vec<SizeClassStatDetail>,
    pub global_stats: SizeClassGlobalStatDetail,
}

#[derive(Debug, Clone)]
pub struct SizeClassStatDetail {
    pub object_size: usize,
    pub class_index: usize,
    pub allocations: usize,
    pub deallocations: usize,
    pub free_objects: usize,
    pub total_bytes: usize,
    pub utilization: f64,
    pub last_access: usize,
}

#[derive(Debug, Clone)]
pub struct SizeClassGlobalStatDetail {
    pub total_objects: usize,
    pub total_bytes: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub cache_hit_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_allocator_creation() {
        let allocator = SizeClassAllocator::new();
        let stats = allocator.get_statistics();
        
        assert_eq!(stats.global_stats.total_allocations, 0);
        assert_eq!(stats.per_class_stats.len(), DEFAULT_SIZE_CLASSES.len());
    }

    #[test]
    fn test_size_class_finding() {
        let allocator = SizeClassAllocator::new();
        
        assert_eq!(allocator.find_size_class_index(1), Some(0)); // Maps to 8
        assert_eq!(allocator.find_size_class_index(8), Some(0)); // Exact match
        assert_eq!(allocator.find_size_class_index(9), Some(1)); // Maps to 16
        assert_eq!(allocator.find_size_class_index(32768), Some(23)); // Largest
        assert_eq!(allocator.find_size_class_index(32769), None); // Too large
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let allocator = SizeClassAllocator::new();
        
        // Create a dummy pointer for testing
        let test_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        
        // Deallocate first to have something to allocate
        allocator.deallocate(test_ptr, 64);
        
        // Now try to allocate
        let result = allocator.allocate(64);
        assert!(result.is_some());
        
        let stats = allocator.get_statistics();
        assert_eq!(stats.global_stats.total_allocations, 1);
        assert_eq!(stats.global_stats.total_deallocations, 1);
    }

    #[test]
    fn test_free_list_compaction() {
        let allocator = SizeClassAllocator::new();
        
        // Add some test entries
        let ptr1 = NonNull::new(0x1000 as *mut u8).unwrap();
        let ptr2 = NonNull::new(0x1040 as *mut u8).unwrap(); // Adjacent to ptr1
        let ptr3 = NonNull::new(0x2000 as *mut u8).unwrap(); // Not adjacent
        
        allocator.deallocate(ptr1, 64);
        allocator.deallocate(ptr2, 64);
        allocator.deallocate(ptr3, 64);
        
        let coalesced = allocator.compact_free_lists();
        assert!(coalesced > 0); // Should coalesce ptr1 and ptr2
    }

    #[test]
    fn test_fragmentation_calculation() {
        let allocator = SizeClassAllocator::new();
        
        // Add some fragmented free blocks
        for i in 0..10 {
            let ptr = NonNull::new((0x1000 + i * 128) as *mut u8).unwrap();
            allocator.deallocate(ptr, 64);
        }
        
        let frag_info = allocator.get_fragmentation_info();
        assert!(frag_info.fragmentation_ratio > 0.0);
        assert_eq!(frag_info.free_block_count, 10);
        assert_eq!(frag_info.total_free_space, 640); // 10 * 64
    }
} 