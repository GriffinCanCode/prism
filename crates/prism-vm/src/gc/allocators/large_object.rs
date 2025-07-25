//! Large Object Allocator for objects exceeding size class thresholds
//!
//! This allocator handles objects larger than LARGE_OBJECT_THRESHOLD (32KB)
//! using direct system allocation with tracking for garbage collection.

use super::*;
use std::sync::{Mutex, Arc};
use std::collections::HashMap;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Allocator for large objects that bypass size classes
pub struct LargeObjectAllocator {
    /// Track large allocations
    allocations: Mutex<HashMap<NonNull<u8>, LargeAllocation>>,
    
    /// Total allocated bytes
    allocated_bytes: AtomicUsize,
    
    /// Number of large objects
    object_count: AtomicUsize,
    
    /// Statistics for monitoring
    stats: Arc<Mutex<LargeObjectStats>>,
}

/// Statistics specific to large object allocation
#[derive(Debug, Default)]
pub struct LargeObjectStats {
    /// Total number of large allocations
    pub total_allocations: usize,
    /// Total number of deallocations
    pub total_deallocations: usize,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Average allocation size
    pub average_size: f64,
    /// Allocation rate (objects per second)
    pub allocation_rate: f64,
    /// Last allocation timestamp
    pub last_allocation: Option<Instant>,
}

impl LargeObjectAllocator {
    /// Create a new large object allocator
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            allocated_bytes: AtomicUsize::new(0),
            object_count: AtomicUsize::new(0),
            stats: Arc::new(Mutex::new(LargeObjectStats::default())),
        }
    }
    
    /// Get statistics for this allocator
    pub fn get_stats(&self) -> LargeObjectStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Get current memory usage
    pub fn allocated_memory(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
    
    /// Get number of allocated objects
    pub fn object_count(&self) -> usize {
        self.object_count.load(Ordering::Relaxed)
    }
    
    /// Check if GC should be triggered based on large object pressure
    pub fn should_trigger_gc(&self) -> bool {
        let allocated = self.allocated_bytes.load(Ordering::Relaxed);
        let count = self.object_count.load(Ordering::Relaxed);
        
        // Trigger if we have more than 100MB in large objects or more than 1000 objects
        allocated > 100 * 1024 * 1024 || count > 1000
    }
    
    /// Prepare for garbage collection
    pub fn prepare_for_gc(&self) {
        // Large object allocator doesn't need special preparation
        // Objects are tracked individually and can be collected as needed
    }
    
    /// Iterator over all large allocations
    pub fn iter_allocations(&self) -> Vec<(*const u8, usize)> {
        let allocations = self.allocations.lock().unwrap();
        allocations
            .iter()
            .map(|(&ptr, alloc)| (ptr.as_ptr() as *const u8, alloc.size))
            .collect()
    }
}

impl Allocator for LargeObjectAllocator {
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return NonNull::new(align as *mut u8);
        }
        
        // Create layout for allocation
        let layout = Layout::from_size_align(size, align).ok()?;
        
        // Allocate using system allocator
        let ptr = unsafe { alloc(layout) };
        let non_null_ptr = NonNull::new(ptr)?;
        
        // Create allocation record
        let allocation = LargeAllocation {
            size,
            layout,
            allocated_at: Instant::now(),
        };
        
        // Track the allocation
        {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.insert(non_null_ptr, allocation);
        }
        
        // Update counters
        self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
        self.object_count.fetch_add(1, Ordering::Relaxed);
        
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocations += 1;
            stats.current_memory += size;
            stats.peak_memory = stats.peak_memory.max(stats.current_memory);
            
            // Update average size
            stats.average_size = stats.current_memory as f64 / self.object_count.load(Ordering::Relaxed) as f64;
            
            // Update allocation rate
            if let Some(last_allocation) = stats.last_allocation {
                let elapsed = last_allocation.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    stats.allocation_rate = 1.0 / elapsed;
                }
            }
            stats.last_allocation = Some(Instant::now());
        }
        
        Some(non_null_ptr)
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, _size: usize) {
        // Remove from tracking
        let allocation = {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.remove(&ptr)
        };
        
        if let Some(allocation) = allocation {
            // Deallocate using system allocator
            unsafe {
                dealloc(ptr.as_ptr(), allocation.layout);
            }
            
            // Update counters
            self.allocated_bytes.fetch_sub(allocation.size, Ordering::Relaxed);
            self.object_count.fetch_sub(1, Ordering::Relaxed);
            
            // Update statistics
            {
                let mut stats = self.stats.lock().unwrap();
                stats.total_deallocations += 1;
                stats.current_memory = stats.current_memory.saturating_sub(allocation.size);
                
                // Update average size
                let count = self.object_count.load(Ordering::Relaxed);
                if count > 0 {
                    stats.average_size = stats.current_memory as f64 / count as f64;
                } else {
                    stats.average_size = 0.0;
                }
            }
        }
    }
    
    fn stats(&self) -> AllocationStats {
        let stats = self.stats.lock().unwrap();
        let allocated = self.allocated_bytes.load(Ordering::Relaxed);
        let count = self.object_count.load(Ordering::Relaxed);
        
        AllocationStats {
            total_allocated: allocated,
            total_deallocated: 0, // We don't track total deallocated separately
            live_bytes: allocated,
            allocation_count: stats.total_allocations,
            deallocation_count: stats.total_deallocations,
            peak_memory: stats.peak_memory,
            large_object_count: count,
            large_object_bytes: allocated,
            page_bytes: 0,
            metadata_memory: std::mem::size_of::<Self>(),
            memory_overhead: std::mem::size_of::<LargeAllocation>() * count,
            barrier_calls: 0,
        }
    }
    
    fn should_trigger_gc(&self) -> bool {
        self.should_trigger_gc()
    }
    
    fn prepare_for_gc(&self) {
        self.prepare_for_gc()
    }
    
    fn get_config(&self) -> AllocatorConfig {
        AllocatorConfig::default()
    }
    
    fn reconfigure(&self, _config: AllocatorConfig) {
        // Large object allocator doesn't have much configuration
    }
}

impl GcAllocator for LargeObjectAllocator {
    fn iter_objects<F>(&self, mut callback: F) 
    where 
        F: FnMut(*const u8, usize)
    {
        let allocations = self.allocations.lock().unwrap();
        for (&ptr, allocation) in allocations.iter() {
            callback(ptr.as_ptr() as *const u8, allocation.size);
        }
    }
    
    fn post_gc_reset(&self) {
        // Large objects don't need special reset after GC
        // They are individually tracked and managed
    }
    
    fn memory_usage(&self) -> MemoryUsage {
        let stats = self.stats.lock().unwrap();
        let allocated = self.allocated_bytes.load(Ordering::Relaxed);
        let count = self.object_count.load(Ordering::Relaxed);
        
        MemoryUsage {
            total_allocated: allocated,
            live_bytes: allocated,
            metadata_overhead: std::mem::size_of::<Self>() + 
                              std::mem::size_of::<LargeAllocation>() * count,
            fragmentation_ratio: 0.0, // Large objects don't fragment internally
        }
    }
}

/// Find all large objects that haven't been accessed recently
/// This can be used by GC to identify candidates for collection
impl LargeObjectAllocator {
    pub fn find_stale_objects(&self, max_age: Duration) -> Vec<*const u8> {
        let allocations = self.allocations.lock().unwrap();
        let now = Instant::now();
        
        allocations
            .iter()
            .filter_map(|(&ptr, allocation)| {
                if now.duration_since(allocation.allocated_at) > max_age {
                    Some(ptr.as_ptr() as *const u8)
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get allocation information for a specific pointer
    pub fn get_allocation_info(&self, ptr: *const u8) -> Option<LargeAllocation> {
        let allocations = self.allocations.lock().unwrap();
        let non_null_ptr = NonNull::new(ptr as *mut u8)?;
        allocations.get(&non_null_ptr).cloned()
    }
    
    /// Compact the allocation tracking by removing entries for deallocated objects
    pub fn compact(&self) {
        // The HashMap already handles this automatically when we remove entries
        // This method is provided for consistency with other allocators
    }
}

impl Default for LargeObjectAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_large_object_allocator_creation() {
        let allocator = LargeObjectAllocator::new();
        assert_eq!(allocator.allocated_memory(), 0);
        assert_eq!(allocator.object_count(), 0);
        assert!(!allocator.should_trigger_gc());
    }
    
    #[test]
    fn test_large_allocation() {
        let allocator = LargeObjectAllocator::new();
        
        // Allocate a large object
        let size = 64 * 1024; // 64KB
        let ptr = allocator.allocate(size, 8);
        assert!(ptr.is_some());
        
        // Check tracking
        assert_eq!(allocator.allocated_memory(), size);
        assert_eq!(allocator.object_count(), 1);
        
        // Deallocate
        if let Some(ptr) = ptr {
            allocator.deallocate(ptr, size);
        }
        
        // Check cleanup
        assert_eq!(allocator.allocated_memory(), 0);
        assert_eq!(allocator.object_count(), 0);
    }
    
    #[test]
    fn test_multiple_allocations() {
        let allocator = LargeObjectAllocator::new();
        let mut ptrs = Vec::new();
        
        // Allocate multiple large objects
        for i in 1..=5 {
            let size = 32 * 1024 * i; // 32KB, 64KB, 96KB, 128KB, 160KB
            let ptr = allocator.allocate(size, 8);
            assert!(ptr.is_some());
            ptrs.push((ptr.unwrap(), size));
        }
        
        // Check total tracking
        let expected_total = (1..=5).map(|i| 32 * 1024 * i).sum::<usize>();
        assert_eq!(allocator.allocated_memory(), expected_total);
        assert_eq!(allocator.object_count(), 5);
        
        // Deallocate all
        for (ptr, size) in ptrs {
            allocator.deallocate(ptr, size);
        }
        
        // Check final state
        assert_eq!(allocator.allocated_memory(), 0);
        assert_eq!(allocator.object_count(), 0);
    }
    
    #[test]
    fn test_gc_triggering() {
        let allocator = LargeObjectAllocator::new();
        
        // Allocate many large objects to trigger GC threshold
        let mut ptrs = Vec::new();
        for i in 0..10 {
            let size = 15 * 1024 * 1024; // 15MB each
            let ptr = allocator.allocate(size, 8);
            if let Some(ptr) = ptr {
                ptrs.push((ptr, size));
            }
            
            // Should trigger GC after enough allocations
            if allocator.allocated_memory() > 100 * 1024 * 1024 {
                assert!(allocator.should_trigger_gc());
                break;
            }
        }
        
        // Clean up
        for (ptr, size) in ptrs {
            allocator.deallocate(ptr, size);
        }
    }
    
    #[test]
    fn test_stale_object_detection() {
        let allocator = LargeObjectAllocator::new();
        
        // Allocate an object
        let size = 64 * 1024;
        let ptr = allocator.allocate(size, 8).unwrap();
        
        // Check stale objects with very short max_age
        let stale = allocator.find_stale_objects(Duration::from_nanos(1));
        assert!(!stale.is_empty());
        
        // Check with very long max_age
        let stale = allocator.find_stale_objects(Duration::from_secs(3600));
        assert!(stale.is_empty());
        
        // Clean up
        allocator.deallocate(ptr, size);
    }
    
    #[test]
    fn test_allocation_info_retrieval() {
        let allocator = LargeObjectAllocator::new();
        
        let size = 64 * 1024;
        let ptr = allocator.allocate(size, 8).unwrap();
        let ptr_addr = ptr.as_ptr() as *const u8;
        
        // Get allocation info
        let info = allocator.get_allocation_info(ptr_addr);
        assert!(info.is_some());
        
        let info = info.unwrap();
        assert_eq!(info.size, size);
        assert_eq!(info.layout.size(), size);
        
        // Clean up
        allocator.deallocate(ptr, size);
        
        // Should not find info after deallocation
        let info = allocator.get_allocation_info(ptr_addr);
        assert!(info.is_none());
    }
} 