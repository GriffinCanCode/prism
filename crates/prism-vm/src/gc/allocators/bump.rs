//! Bump Allocator - Fast allocation for garbage collection
//!
//! This allocator is optimized for garbage collection scenarios where objects
//! have short lifetimes and can be reclaimed in bulk. It provides:
//!
//! - Very fast allocation through pointer bumping
//! - Efficient integration with GC marking
//! - Memory layout optimized for cache locality
//! - Support for multiple allocation regions

use super::types::*;
use super::{Allocator, GcAllocator};

use std::ptr::NonNull;
use std::sync::{RwLock, Mutex, Arc};
use std::sync::atomic::{AtomicPtr, AtomicUsize, AtomicBool, Ordering};
use std::alloc::{alloc, dealloc, Layout};
use std::time::Instant;

/// Specialized bump allocator optimized for garbage collection
/// 
/// This allocator is designed to work efficiently with the tri-color collector:
/// - Fast allocation through pointer bumping
/// - Efficient integration with GC marking
/// - Memory layout optimized for cache locality
/// - Support for multiple allocation regions
pub struct BumpAllocator {
    /// Current allocation regions
    regions: RwLock<Vec<AllocationRegion>>,
    /// Current active region index
    current_region: AtomicUsize,
    /// Total allocated bytes across all regions
    total_allocated: AtomicUsize,
    /// Configuration
    config: BumpAllocatorConfig,
    /// Statistics for monitoring performance
    stats: Arc<Mutex<BumpAllocationStats>>,
    /// Peak memory usage tracking
    peak_memory: AtomicUsize,
}

impl BumpAllocator {
    pub fn new(initial_capacity: usize) -> Self {
        let config = BumpAllocatorConfig {
            initial_region_size: initial_capacity.min(1024 * 1024), // Max 1MB initial
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    pub fn with_config(config: BumpAllocatorConfig) -> Self {
        let initial_region = Self::create_region(config.initial_region_size)
            .expect("Failed to create initial allocation region");
        
        let mut regions = Vec::new();
        regions.push(initial_region);
        
        Self {
            regions: RwLock::new(regions),
            current_region: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            config,
            stats: Arc::new(Mutex::new(BumpAllocationStats {
                last_allocation_time: Instant::now(),
                ..Default::default()
            })),
            peak_memory: AtomicUsize::new(0),
        }
    }
    
    /// Try to allocate from the current active region
    fn try_allocate_from_current_region(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let current_region_index = self.current_region.load(Ordering::Acquire);
        let regions = self.regions.read().unwrap();
        
        if current_region_index >= regions.len() {
            return None;
        }
        
        let region = &regions[current_region_index];
        if !region.active.load(Ordering::Acquire) {
            return None;
        }
        
        self.try_allocate_from_region(region, size, align)
    }
    
    /// Try to allocate from a specific region
    fn try_allocate_from_region(&self, region: &AllocationRegion, size: usize, align: usize) -> Option<NonNull<u8>> {
        loop {
            let current = region.current.load(Ordering::Acquire);
            
            // Calculate aligned allocation address
            let current_addr = current as usize;
            let aligned_addr = (current_addr + align - 1) & !(align - 1);
            let aligned_ptr = aligned_addr as *mut u8;
            
            // Check if allocation fits in region
            let end_ptr = unsafe { aligned_ptr.add(size) };
            if end_ptr > region.end {
                return None; // Doesn't fit
            }
            
            // Try to update the current pointer atomically
            match region.current.compare_exchange_weak(
                current,
                end_ptr,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully allocated
                    region.object_count.fetch_add(1, Ordering::Relaxed);
                    
                    // Zero memory if configured
                    if self.config.zero_memory {
                        unsafe {
                            std::ptr::write_bytes(aligned_ptr, 0, size);
                        }
                    }
                    
                    return NonNull::new(aligned_ptr);
                }
                Err(_) => {
                    // Another thread updated the pointer, retry
                    continue;
                }
            }
        }
    }
    
    /// Allocate from a new region
    fn allocate_from_new_region(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Calculate region size needed
        let min_region_size = size + align + 1024; // Extra space for alignment and overhead
        let region_size = min_region_size
            .max(self.config.initial_region_size)
            .min(self.config.max_region_size);
        
        // Create new region
        let new_region = Self::create_region(region_size)?;
        
        // Try to allocate from the new region before adding it
        let ptr = self.try_allocate_from_region(&new_region, size, align)?;
        
        // Add the new region to our collection
        {
            let mut regions = self.regions.write().unwrap();
            let new_index = regions.len();
            regions.push(new_region);
            
            // Update current region index
            self.current_region.store(new_index, Ordering::Release);
            
            // Update statistics
            let mut stats = self.stats.lock().unwrap();
            stats.regions_created += 1;
        }
        
        Some(ptr)
    }
    
    /// Create a new allocation region
    fn create_region(size: usize) -> Option<AllocationRegion> {
        let layout = Layout::from_size_align(size, PAGE_SIZE).ok()?;
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return None;
        }
        
        let start = NonNull::new(ptr)?;
        let end = unsafe { ptr.add(size) };
        
        Some(AllocationRegion {
            start,
            current: AtomicPtr::new(ptr),
            end,
            size,
            object_count: AtomicUsize::new(0),
            active: AtomicBool::new(true),
        })
    }
    
    /// Update allocation statistics
    fn update_allocation_stats(&self, size: usize) {
        let new_total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak memory tracking
        let mut current_peak = self.peak_memory.load(Ordering::Relaxed);
        while new_total > current_peak {
            match self.peak_memory.compare_exchange_weak(
                current_peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual,
            }
        }
        
        let mut stats = self.stats.lock().unwrap();
        let now = Instant::now();
        
        stats.total_allocations += 1;
        stats.total_bytes += size;
        stats.average_allocation_size = stats.total_bytes as f64 / stats.total_allocations as f64;
        
        // Calculate allocation rate
        let time_since_last = now.duration_since(stats.last_allocation_time).as_secs_f64();
        if time_since_last > 0.0 {
            stats.allocation_rate = 1.0 / time_since_last;
        }
        stats.last_allocation_time = now;
    }
    
    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }
    
    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> usize {
        self.peak_memory.load(Ordering::Relaxed)
    }
    
    /// Get detailed memory usage breakdown
    pub fn get_memory_breakdown(&self) -> BumpMemoryBreakdown {
        let regions = self.regions.read().unwrap();
        let mut breakdown = BumpMemoryBreakdown {
            total_regions: regions.len(),
            active_region_index: self.current_region.load(Ordering::Relaxed),
            total_capacity: 0,
            allocated_bytes: self.total_allocated.load(Ordering::Relaxed),
            peak_bytes: self.peak_memory.load(Ordering::Relaxed),
            region_details: Vec::new(),
        };
        
        for (i, region) in regions.iter().enumerate() {
            let current_ptr = region.current.load(Ordering::Relaxed);
            let allocated_in_region = current_ptr as usize - region.start.as_ptr() as usize;
            
            breakdown.total_capacity += region.size;
            breakdown.region_details.push(RegionDetail {
                region_index: i,
                size: region.size,
                allocated: allocated_in_region,
                free: region.size - allocated_in_region,
                object_count: region.object_count.load(Ordering::Relaxed),
                is_active: region.active.load(Ordering::Relaxed),
            });
        }
        
        breakdown
    }
    
    /// Get allocation statistics
    pub fn get_stats(&self) -> BumpAllocationStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Reset all regions (called after GC)
    pub fn reset(&self) {
        let mut regions = self.regions.write().unwrap();
        
        // Reset all regions to their start positions
        for region in regions.iter() {
            region.current.store(region.start.as_ptr(), Ordering::Release);
            region.object_count.store(0, Ordering::Relaxed);
            region.active.store(true, Ordering::Relaxed);
        }
        
        // Reset to first region
        self.current_region.store(0, Ordering::Release);
        self.total_allocated.store(0, Ordering::Relaxed);
        
        // Note: Don't reset peak_memory as it represents historical maximum
        
        // Reset statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocations = 0;
            stats.total_bytes = 0;
            stats.allocation_failures = 0;
            stats.average_allocation_size = 0.0;
        }
    }
}

impl Allocator for BumpAllocator {
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return NonNull::new(align as *mut u8);
        }
        
        // Try to allocate from current region first
        if let Some(ptr) = self.try_allocate_from_current_region(size, align) {
            self.update_allocation_stats(size);
            return Some(ptr);
        }
        
        // If that fails, try to create a new region
        if let Some(ptr) = self.allocate_from_new_region(size, align) {
            self.update_allocation_stats(size);
            return Some(ptr);
        }
        
        // Update failure statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocation_failures += 1;
        }
        
        None
    }
    
    fn deallocate(&self, _ptr: NonNull<u8>, _size: usize) {
        // Bump allocator doesn't support individual deallocation
        // Objects are reclaimed in bulk during GC reset
    }
    
    fn stats(&self) -> AllocationStats {
        let bump_stats = self.get_stats();
        
        AllocationStats {
            total_allocated: bump_stats.total_bytes,
            total_deallocated: 0, // Bump allocator doesn't track individual deallocations
            live_bytes: self.total_allocated.load(Ordering::Relaxed),
            allocation_count: bump_stats.total_allocations,
            deallocation_count: 0,
            peak_memory: self.peak_memory.load(Ordering::Relaxed),
            large_object_count: 0,
            large_object_bytes: 0,
            page_bytes: 0,
            metadata_memory: bump_stats.regions_created * std::mem::size_of::<AllocationRegion>(),
            memory_overhead: 0,
            barrier_calls: 0,
        }
    }
    
    fn should_trigger_gc(&self) -> bool {
        // Simple heuristic: trigger GC if we have many regions or high allocation rate
        let stats = self.get_stats();
        stats.regions_created > 10 || stats.allocation_rate > 1000.0
    }
    
    fn prepare_for_gc(&self) {
        // Mark all regions as inactive to prevent new allocations
        let regions = self.regions.read().unwrap();
        for region in regions.iter() {
            region.active.store(false, Ordering::Release);
        }
    }
    
    fn get_config(&self) -> AllocatorConfig {
        AllocatorConfig {
            enable_thread_cache: false, // Bump allocator doesn't use thread caches
            gc_trigger_threshold: self.config.max_region_size * 10,
            numa_aware: false,
        }
    }
    
    fn reconfigure(&self, _config: AllocatorConfig) {
        // Bump allocator configuration is mostly immutable after creation
        // Could update some runtime parameters here if needed
    }
}

impl GcAllocator for BumpAllocator {
    fn iter_objects<F>(&self, mut callback: F) 
    where 
        F: FnMut(*const u8, usize),
    {
        let regions = self.regions.read().unwrap();
        
        for region in regions.iter() {
            let start = region.start.as_ptr();
            let current = region.current.load(Ordering::Acquire);
            
            // Simple iteration - assumes all allocations are ObjectHeader + data
            let mut ptr = start;
            while ptr < current {
                unsafe {
                    // Read object header to get size
                    let header = &*(ptr as *const ObjectHeader);
                    let object_size = header.size;
                    
                    // Call callback with object pointer and size
                    callback(ptr, object_size);
                    
                    // Move to next object (header + data, aligned)
                    let next_ptr = ptr.add(object_size);
                    let aligned_next = ((next_ptr as usize + 7) & !7) as *mut u8; // 8-byte align
                    ptr = aligned_next;
                }
            }
        }
    }
    
    fn post_gc_reset(&self) {
        self.reset();
    }
    
    fn memory_usage(&self) -> MemoryUsage {
        let regions = self.regions.read().unwrap();
        let stats = self.stats.lock().unwrap();
        
        let total_capacity: usize = regions.iter().map(|r| r.size).sum();
        let allocated = self.total_allocated.load(Ordering::Relaxed);
        let free = total_capacity.saturating_sub(allocated);
        
        MemoryUsage {
            total_capacity,
            allocated,
            free,
            regions_count: regions.len(),
            average_allocation_size: stats.average_allocation_size,
            allocation_rate: stats.allocation_rate,
            fragmentation_ratio: 0.0, // Bump allocator has minimal fragmentation
        }
    }
}

impl Drop for AllocationRegion {
    fn drop(&mut self) {
        // Deallocate the region memory
        let layout = Layout::from_size_align(self.size, PAGE_SIZE)
            .expect("Invalid layout for region deallocation");
        unsafe {
            dealloc(self.start.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bump_allocator_creation() {
        let allocator = BumpAllocator::new(1024);
        let stats = allocator.get_stats();
        assert_eq!(stats.total_allocations, 0);
    }
    
    #[test]
    fn test_basic_allocation() {
        let allocator = BumpAllocator::new(1024);
        
        let ptr1 = allocator.allocate(32, 8).expect("Failed to allocate 32 bytes");
        let ptr2 = allocator.allocate(64, 8).expect("Failed to allocate 64 bytes");
        
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
        
        let stats = allocator.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_bytes, 96);
    }
    
    #[test]
    fn test_large_allocation() {
        let allocator = BumpAllocator::new(1024);
        
        // Allocate something larger than initial region
        let ptr = allocator.allocate(2048, 8).expect("Failed to allocate large object");
        assert!(!ptr.as_ptr().is_null());
        
        let stats = allocator.get_stats();
        assert_eq!(stats.regions_created, 2); // Initial + new region
    }
    
    #[test]
    fn test_zero_size_allocation() {
        let allocator = BumpAllocator::new(1024);
        
        let ptr = allocator.allocate(0, 8).expect("Failed to allocate 0 bytes");
        assert!(!ptr.as_ptr().is_null());
    }
    
    #[test]
    fn test_alignment() {
        let allocator = BumpAllocator::new(1024);
        
        for &align in &[1, 2, 4, 8, 16, 32] {
            let ptr = allocator.allocate(64, align)
                .expect(&format!("Failed to allocate with alignment {}", align));
            
            assert_eq!(ptr.as_ptr() as usize % align, 0, 
                      "Pointer not aligned to {} bytes", align);
        }
    }
    
    #[test]
    fn test_reset() {
        let allocator = BumpAllocator::new(1024);
        
        // Make some allocations
        allocator.allocate(32, 8).unwrap();
        allocator.allocate(64, 8).unwrap();
        
        let stats_before = allocator.get_stats();
        assert_eq!(stats_before.total_allocations, 2);
        
        // Reset allocator
        allocator.reset();
        
        let stats_after = allocator.get_stats();
        assert_eq!(stats_after.total_allocations, 0);
        assert_eq!(stats_after.total_bytes, 0);
        assert_eq!(allocator.allocated_bytes(), 0);
    }
    
    #[test]
    fn test_memory_usage() {
        let allocator = BumpAllocator::new(1024);
        
        allocator.allocate(100, 8).unwrap();
        allocator.allocate(200, 8).unwrap();
        
        let usage = allocator.memory_usage();
        assert!(usage.total_capacity >= 1024);
        assert_eq!(usage.allocated, 300);
        assert_eq!(usage.regions_count, 1);
        assert_eq!(usage.fragmentation_ratio, 0.0);
    }
} 