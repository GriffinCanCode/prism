use super::*;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::alloc::{alloc, dealloc, Layout};

/// High-performance bump allocator for garbage collection
/// Optimized for fast allocation with minimal overhead
pub struct BumpAllocator {
    /// Current allocation pointer
    current: AtomicPtr<u8>,
    /// End of current allocation region
    end: AtomicPtr<u8>,
    /// Total capacity
    capacity: usize,
    /// Total allocated bytes
    allocated: AtomicUsize,
    /// Allocation regions for thread-local allocation
    regions: Mutex<Vec<AllocationRegion>>,
    /// Large object allocator for objects > threshold
    large_allocator: LargeObjectAllocator,
}

/// Individual allocation region (typically 64KB - 1MB)
struct AllocationRegion {
    start: *mut u8,
    size: usize,
    allocated: usize,
}

impl BumpAllocator {
    pub fn new(capacity: usize) -> Self {
        let initial_region_size = (capacity / 16).max(64 * 1024); // At least 64KB
        let initial_region = Self::allocate_region(initial_region_size);
        
        let mut regions = Vec::new();
        let (start, end) = if let Some(region) = initial_region {
            regions.push(region);
            (region.start, unsafe { region.start.add(region.size) })
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        
        Self {
            current: AtomicPtr::new(start),
            end: AtomicPtr::new(end),
            capacity,
            allocated: AtomicUsize::new(0),
            regions: Mutex::new(regions),
            large_allocator: LargeObjectAllocator::new(),
        }
    }
    
    /// Fast path allocation - lock-free bump pointer
    pub fn allocate(&self, size: usize, align: usize) -> Option<*mut u8> {
        // Large objects go to specialized allocator
        if size > 8192 {
            return self.large_allocator.allocate(size, align);
        }
        
        // Align size to pointer boundary for better cache performance
        let aligned_size = (size + align - 1) & !(align - 1);
        
        loop {
            let current = self.current.load(Ordering::Relaxed);
            let end = self.end.load(Ordering::Relaxed);
            
            if current.is_null() || end.is_null() {
                return self.slow_path_allocate(aligned_size, align);
            }
            
            // Check if we have enough space in current region
            let aligned_current = Self::align_ptr(current, align);
            let new_current = unsafe { aligned_current.add(aligned_size) };
            
            if new_current <= end {
                // Try to atomically update the current pointer
                if self.current.compare_exchange_weak(
                    current,
                    new_current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ).is_ok() {
                    self.allocated.fetch_add(aligned_size, Ordering::Relaxed);
                    return Some(aligned_current);
                }
                // CAS failed, retry
                continue;
            } else {
                // Not enough space, need new region
                return self.slow_path_allocate(aligned_size, align);
            }
        }
    }
    
    /// Slow path when current region is exhausted
    fn slow_path_allocate(&self, size: usize, align: usize) -> Option<*mut u8> {
        let mut regions = self.regions.lock().unwrap();
        
        // Try to find a region with enough space
        for region in regions.iter_mut() {
            if region.size - region.allocated >= size + align {
                let ptr = unsafe { region.start.add(region.allocated) };
                let aligned_ptr = Self::align_ptr(ptr, align);
                let actual_size = unsafe { aligned_ptr.add(size) as usize - ptr as usize };
                
                region.allocated += actual_size;
                self.allocated.fetch_add(actual_size, Ordering::Relaxed);
                return Some(aligned_ptr);
            }
        }
        
        // Need to allocate a new region
        let region_size = (size * 4).max(64 * 1024); // At least 4x the requested size
        if let Some(new_region) = Self::allocate_region(region_size) {
            let ptr = new_region.start;
            let aligned_ptr = Self::align_ptr(ptr, align);
            let actual_size = unsafe { aligned_ptr.add(size) as usize - ptr as usize };
            
            // Update the new region
            let mut region = new_region;
            region.allocated = actual_size;
            
            // Update atomic pointers for fast path
            self.current.store(unsafe { ptr.add(actual_size) }, Ordering::Relaxed);
            self.end.store(unsafe { ptr.add(region.size) }, Ordering::Relaxed);
            
            regions.push(region);
            self.allocated.fetch_add(actual_size, Ordering::Relaxed);
            
            Some(aligned_ptr)
        } else {
            None // Out of memory
        }
    }
    
    /// Allocate a new memory region from the system
    fn allocate_region(size: usize) -> Option<AllocationRegion> {
        let layout = Layout::from_size_align(size, 4096).ok()?; // Page-aligned
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            None
        } else {
            Some(AllocationRegion {
                start: ptr,
                size,
                allocated: 0,
            })
        }
    }
    
    /// Align a pointer to the given alignment
    fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
        let addr = ptr as usize;
        let aligned_addr = (addr + align - 1) & !(align - 1);
        aligned_addr as *mut u8
    }
    
    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::Relaxed) + self.large_allocator.allocated_bytes()
    }
    
    /// Reset allocator (used after garbage collection)
    pub fn reset(&self) {
        let mut regions = self.regions.lock().unwrap();
        
        // Reset all regions
        for region in regions.iter_mut() {
            region.allocated = 0;
        }
        
        // Reset atomic pointers to first region
        if let Some(first_region) = regions.first() {
            self.current.store(first_region.start, Ordering::Relaxed);
            self.end.store(unsafe { first_region.start.add(first_region.size) }, Ordering::Relaxed);
        }
        
        self.allocated.store(0, Ordering::Relaxed);
        self.large_allocator.reset();
    }
    
    /// Compact allocator by removing unused regions
    pub fn compact(&self) -> usize {
        let mut regions = self.regions.lock().unwrap();
        let initial_count = regions.len();
        
        // Keep only regions that have some allocation
        regions.retain(|region| region.allocated > 0);
        
        // Deallocate completely unused regions
        let removed_count = initial_count - regions.len();
        
        removed_count
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        let regions = self.regions.lock().unwrap();
        
        // Deallocate all regions
        for region in regions.iter() {
            let layout = Layout::from_size_align(region.size, 4096).unwrap();
            unsafe {
                dealloc(region.start, layout);
            }
        }
    }
}

/// Specialized allocator for large objects (> 8KB)
/// Uses system allocator directly to avoid fragmentation
struct LargeObjectAllocator {
    /// Track large allocations for statistics
    allocations: Mutex<Vec<LargeAllocation>>,
    /// Total allocated bytes
    allocated: AtomicUsize,
}

struct LargeAllocation {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
}

impl LargeObjectAllocator {
    fn new() -> Self {
        Self {
            allocations: Mutex::new(Vec::new()),
            allocated: AtomicUsize::new(0),
        }
    }
    
    fn allocate(&self, size: usize, align: usize) -> Option<*mut u8> {
        let layout = Layout::from_size_align(size, align).ok()?;
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return None;
        }
        
        // Track the allocation
        {
            let mut allocations = self.allocations.lock().unwrap();
            allocations.push(LargeAllocation { ptr, size, layout });
        }
        
        self.allocated.fetch_add(size, Ordering::Relaxed);
        Some(ptr)
    }
    
    fn deallocate(&self, ptr: *mut u8) -> bool {
        let mut allocations = self.allocations.lock().unwrap();
        
        if let Some(pos) = allocations.iter().position(|alloc| alloc.ptr == ptr) {
            let allocation = allocations.swap_remove(pos);
            self.allocated.fetch_sub(allocation.size, Ordering::Relaxed);
            
            unsafe {
                dealloc(allocation.ptr, allocation.layout);
            }
            
            true
        } else {
            false
        }
    }
    
    fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
    
    fn reset(&self) {
        let mut allocations = self.allocations.lock().unwrap();
        
        // Deallocate all large objects
        for allocation in allocations.drain(..) {
            unsafe {
                dealloc(allocation.ptr, allocation.layout);
            }
        }
        
        self.allocated.store(0, Ordering::Relaxed);
    }
}

/// Thread-local allocation buffer for even better performance
/// Each thread gets its own small buffer to reduce contention
pub struct ThreadLocalAllocator {
    /// Thread-local buffer
    buffer: *mut u8,
    /// Current position in buffer
    current: *mut u8,
    /// End of buffer
    end: *mut u8,
    /// Parent allocator
    parent: *const BumpAllocator,
    /// Buffer size
    buffer_size: usize,
}

impl ThreadLocalAllocator {
    const DEFAULT_BUFFER_SIZE: usize = 32 * 1024; // 32KB per thread
    
    pub fn new(parent: &BumpAllocator) -> Self {
        Self {
            buffer: std::ptr::null_mut(),
            current: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            parent: parent as *const BumpAllocator,
            buffer_size: Self::DEFAULT_BUFFER_SIZE,
        }
    }
    
    /// Allocate from thread-local buffer
    pub fn allocate(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        // Check if we need a new buffer
        if self.buffer.is_null() || unsafe { self.current.add(size) } > self.end {
            self.refill_buffer()?;
        }
        
        let aligned_current = BumpAllocator::align_ptr(self.current, align);
        let new_current = unsafe { aligned_current.add(size) };
        
        if new_current <= self.end {
            self.current = new_current;
            Some(aligned_current)
        } else {
            // Buffer too small, allocate directly from parent
            unsafe { (*self.parent).allocate(size, align) }
        }
    }
    
    /// Refill thread-local buffer from parent allocator
    fn refill_buffer(&mut self) -> Option<()> {
        let parent = unsafe { &*self.parent };
        let buffer = parent.allocate(self.buffer_size, 8)?;
        
        self.buffer = buffer;
        self.current = buffer;
        self.end = unsafe { buffer.add(self.buffer_size) };
        
        Some(())
    }
    
    /// Reset thread-local buffer
    pub fn reset(&mut self) {
        self.buffer = std::ptr::null_mut();
        self.current = std::ptr::null_mut();
        self.end = std::ptr::null_mut();
    }
}

/// Free list allocator for reusing deallocated objects
/// Maintains size-segregated free lists for efficient reuse
pub struct FreeListAllocator {
    /// Free lists for different size classes
    size_classes: Vec<FreeList>,
    /// Minimum allocation size
    min_size: usize,
    /// Size class multiplier (e.g., 1.5 for 50% growth between classes)
    size_multiplier: f64,
}

struct FreeList {
    /// Size of objects in this free list
    object_size: usize,
    /// List of free objects
    free_objects: Vec<*mut u8>,
}

impl FreeListAllocator {
    pub fn new() -> Self {
        let mut size_classes = Vec::new();
        let mut size = 16; // Start with 16-byte objects
        
        // Create size classes up to 8KB (larger objects go to large object allocator)
        while size <= 8192 {
            size_classes.push(FreeList {
                object_size: size,
                free_objects: Vec::new(),
            });
            size = (size as f64 * 1.5) as usize; // 50% growth between size classes
        }
        
        Self {
            size_classes,
            min_size: 16,
            size_multiplier: 1.5,
        }
    }
    
    /// Allocate from appropriate size class
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        let size_class = self.find_size_class(size)?;
        
        if let Some(ptr) = self.size_classes[size_class].free_objects.pop() {
            Some(ptr)
        } else {
            None // No free objects in this size class
        }
    }
    
    /// Return object to appropriate free list
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        if let Some(size_class) = self.find_size_class(size) {
            self.size_classes[size_class].free_objects.push(ptr);
        }
    }
    
    /// Find the appropriate size class for a given size
    fn find_size_class(&self, size: usize) -> Option<usize> {
        for (i, size_class) in self.size_classes.iter().enumerate() {
            if size <= size_class.object_size {
                return Some(i);
            }
        }
        None
    }
    
    /// Get statistics about free list usage
    pub fn get_stats(&self) -> FreeListStats {
        let mut total_free_objects = 0;
        let mut total_free_bytes = 0;
        
        for size_class in &self.size_classes {
            total_free_objects += size_class.free_objects.len();
            total_free_bytes += size_class.free_objects.len() * size_class.object_size;
        }
        
        FreeListStats {
            total_free_objects,
            total_free_bytes,
            size_classes: self.size_classes.len(),
        }
    }
    
    /// Clear all free lists (used after garbage collection)
    pub fn clear(&mut self) {
        for size_class in &mut self.size_classes {
            size_class.free_objects.clear();
        }
    }
}

#[derive(Debug)]
pub struct FreeListStats {
    pub total_free_objects: usize,
    pub total_free_bytes: usize,
    pub size_classes: usize,
} 