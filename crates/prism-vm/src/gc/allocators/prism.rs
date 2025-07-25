//! PrismAllocator - High-performance size-class based allocator
//!
//! This allocator provides fast allocation and deallocation through:
//! - Size class segregation to minimize fragmentation
//! - Thread-local caches to reduce contention
//! - Efficient memory layout for cache performance
//! - Integration with garbage collection

use super::*;
use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use std::sync::{Mutex, RwLock, Arc};
use std::collections::HashMap;
use std::thread;
use std::ptr::NonNull;

/// High-performance allocator with thread-local caches and size class segregation
pub struct PrismAllocator {
    /// Configuration for this allocator
    config: PrismAllocatorConfig,
    
    /// Central caches for each size class
    central_caches: Vec<CentralCache>,
    
    /// Thread-local cache registry
    thread_caches: RwLock<HashMap<thread::ThreadId, Arc<ThreadCache>>>,
    
    /// Global allocation statistics
    stats: Arc<Mutex<AllocationStats>>,
    
    /// Memory usage tracking
    memory_usage: AtomicUsize,
    
    /// GC trigger threshold
    gc_threshold: AtomicUsize,
}

/// Central cache for a specific size class
pub struct CentralCache {
    /// Free objects for this size class
    free_objects: Mutex<Vec<NonNull<u8>>>,
    
    /// Size of objects in this cache
    object_size: usize,
    
    /// Statistics for this size class
    stats: CentralCacheStats,
    
    /// Track allocated objects for GC iteration
    allocated_objects: Mutex<Vec<NonNull<u8>>>,
    
    /// Total objects allocated from this cache (for statistics)
    total_objects_allocated: AtomicUsize,
}

/// Thread-local allocation cache
pub struct ThreadCache {
    /// Free lists for each size class
    free_lists: Vec<FreeList>,
    
    /// Total cached bytes
    cached_bytes: AtomicUsize,
    
    /// Thread ID this cache belongs to
    thread_id: thread::ThreadId,
}

/// Free list for a specific size class in thread cache
pub struct FreeList {
    /// Head of the free list
    head: AtomicPtr<u8>,
    
    /// Number of objects in this free list
    count: AtomicUsize,
    
    /// Size of objects in this free list
    object_size: usize,
}

impl PrismAllocator {
    /// Create a new PrismAllocator with default configuration
    pub fn new() -> Self {
        Self::with_config(PrismAllocatorConfig::default())
    }
    
    /// Create a new PrismAllocator with custom configuration
    pub fn with_config(config: PrismAllocatorConfig) -> Self {
        let size_class_count = SIZE_CLASSES.len();
        let mut central_caches = Vec::with_capacity(size_class_count);
        
        // Initialize central caches for each size class
        for &size in SIZE_CLASSES.iter() {
            central_caches.push(CentralCache::new(size));
        }
        
        Self {
            config,
            central_caches,
            thread_caches: RwLock::new(HashMap::new()),
            stats: Arc::new(Mutex::new(AllocationStats::new())),
            memory_usage: AtomicUsize::new(0),
            gc_threshold: AtomicUsize::new(config.base.gc_trigger_threshold),
        }
    }
    
    /// Find the appropriate size class for the given size
    fn find_size_class(&self, size: usize) -> Option<usize> {
        for (index, &class_size) in SIZE_CLASSES.iter().enumerate() {
            if size <= class_size {
                return Some(index);
            }
        }
        None
    }
    
    /// Get or create thread cache for current thread
    fn get_thread_cache(&self) -> Option<Arc<ThreadCache>> {
        if !self.config.base.enable_thread_cache {
            return None;
        }
        
        let thread_id = thread::current().id();
        
        // Try to get existing cache
        {
            let caches = self.thread_caches.read().unwrap();
            if let Some(cache) = caches.get(&thread_id) {
                return Some(cache.clone());
            }
        }
        
        // Create new cache
        let mut caches = self.thread_caches.write().unwrap();
        let cache = Arc::new(ThreadCache::new(thread_id));
        caches.insert(thread_id, cache.clone());
        Some(cache)
    }
    
    /// Allocate from thread cache if available
    fn allocate_from_thread_cache(&self, size_class_index: usize) -> Option<NonNull<u8>> {
        let cache = self.get_thread_cache()?;
        cache.allocate(size_class_index)
    }
    
    /// Allocate from central cache
    fn allocate_from_central_cache(&self, size_class_index: usize) -> Option<NonNull<u8>> {
        if size_class_index >= self.central_caches.len() {
            return None;
        }
        
        self.central_caches[size_class_index].allocate()
    }
    
    /// Deallocate to thread cache if available
    fn deallocate_to_thread_cache(&self, ptr: NonNull<u8>, size_class_index: usize) {
        if let Some(cache) = self.get_thread_cache() {
            cache.deallocate(ptr, size_class_index);
        }
    }
    
    /// Deallocate to central cache
    fn deallocate_to_central_cache(&self, ptr: NonNull<u8>, size_class_index: usize) {
        if size_class_index < self.central_caches.len() {
            self.central_caches[size_class_index].deallocate(ptr);
        }
    }
}

impl Allocator for PrismAllocator {
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return NonNull::new(align as *mut u8);
        }
        
        // Find appropriate size class
        let size_class_index = self.find_size_class(size)?;
        let actual_size = SIZE_CLASSES[size_class_index];
        
        // Try thread cache first
        let result = if let Some(ptr) = self.allocate_from_thread_cache(size_class_index) {
            Some(ptr)
        } else {
            // Fall back to central cache
            self.allocate_from_central_cache(size_class_index)
        };
        
        if result.is_some() {
            self.memory_usage.fetch_add(actual_size, Ordering::Relaxed);
            
            // Update statistics
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocated += actual_size;
            stats.allocation_count += 1;
        }
        
        result
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        if size == 0 {
            return;
        }
        
        // Find size class
        if let Some(size_class_index) = self.find_size_class(size) {
            let actual_size = SIZE_CLASSES[size_class_index];
            
            // Try thread cache first
            if self.config.base.enable_thread_cache {
                self.deallocate_to_thread_cache(ptr, size_class_index);
            } else {
                self.deallocate_to_central_cache(ptr, size_class_index);
            }
            
            self.memory_usage.fetch_sub(actual_size, Ordering::Relaxed);
            
            // Update statistics
            let mut stats = self.stats.lock().unwrap();
            stats.total_deallocated += actual_size;
            stats.deallocation_count += 1;
        }
    }
    
    fn stats(&self) -> AllocationStats {
        self.stats.lock().unwrap().clone()
    }
    
    fn should_trigger_gc(&self) -> bool {
        let usage = self.memory_usage.load(Ordering::Relaxed);
        let threshold = self.gc_threshold.load(Ordering::Relaxed);
        usage >= threshold
    }
    
    fn prepare_for_gc(&self) {
        // Flush all thread caches
        let caches = self.thread_caches.read().unwrap();
        for cache in caches.values() {
            cache.flush_to_central(&self.central_caches);
        }
    }
    
    fn get_config(&self) -> AllocatorConfig {
        self.config.base.clone()
    }
    
    fn reconfigure(&self, config: AllocatorConfig) {
        self.gc_threshold.store(config.gc_trigger_threshold, Ordering::Relaxed);
    }
}

impl GcAllocator for PrismAllocator {
    fn iter_objects<F>(&self, mut callback: F) 
    where 
        F: FnMut(*const u8, usize)
    {
        // Iterate through central caches to find allocated objects
        for (index, cache) in self.central_caches.iter().enumerate() {
            let object_size = SIZE_CLASSES[index];
            cache.iter_allocated_objects(|ptr| {
                callback(ptr, object_size);
            });
        }
    }
    
    fn post_gc_reset(&self) {
        // Clear all allocated objects from central caches
        for cache in &self.central_caches {
            cache.clear_allocated_objects();
        }
        
        // Reset statistics and clean up thread caches
        let mut caches = self.thread_caches.write().unwrap();
        caches.retain(|_, cache| Arc::strong_count(cache) > 1);
        
        // Reset memory usage counter
        self.memory_usage.store(0, Ordering::Relaxed);
        
        // Reset global statistics
        {
            let mut stats = self.stats.lock().unwrap();
            *stats = AllocationStats::new();
        }
    }
    
    fn memory_usage(&self) -> MemoryUsage {
        let stats = self.stats.lock().unwrap();
        MemoryUsage {
            total_allocated: stats.total_allocated,
            live_bytes: stats.total_allocated - stats.total_deallocated,
            metadata_overhead: std::mem::size_of::<Self>() + 
                              self.central_caches.len() * std::mem::size_of::<CentralCache>(),
            fragmentation_ratio: 0.1, // Estimated based on size class overhead
        }
    }
}

impl CentralCache {
    fn new(object_size: usize) -> Self {
        Self {
            free_objects: Mutex::new(Vec::new()),
            object_size,
            stats: CentralCacheStats::default(),
            allocated_objects: Mutex::new(Vec::new()),
            total_objects_allocated: AtomicUsize::new(0),
        }
    }
    
    fn allocate(&self) -> Option<NonNull<u8>> {
        // Try to get from free list first
        let ptr = {
            let mut free_objects = self.free_objects.lock().unwrap();
            free_objects.pop()
        };
        
        let ptr = if let Some(ptr) = ptr {
            ptr
        } else {
            // Allocate new object from system
            use std::alloc::{alloc, Layout};
            let layout = Layout::from_size_align(self.object_size, 8)
                .expect("Invalid layout for object allocation");
            let raw_ptr = unsafe { alloc(layout) };
            NonNull::new(raw_ptr)?
        };
        
        // Track the allocated object
        {
            let mut allocated = self.allocated_objects.lock().unwrap();
            allocated.push(ptr);
        }
        
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        self.total_objects_allocated.fetch_add(1, Ordering::Relaxed);
        Some(ptr)
    }
    
    fn deallocate(&self, ptr: NonNull<u8>) {
        // Remove from allocated objects tracking
        {
            let mut allocated = self.allocated_objects.lock().unwrap();
            if let Some(pos) = allocated.iter().position(|&p| p == ptr) {
                allocated.remove(pos);
            }
        }
        
        // Add to free list
        let mut free_objects = self.free_objects.lock().unwrap();
        free_objects.push(ptr);
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
    }
    
    fn iter_allocated_objects<F>(&self, mut callback: F)
    where
        F: FnMut(*const u8)
    {
        // Iterate through all currently allocated objects
        let allocated = self.allocated_objects.lock().unwrap();
        for &ptr in allocated.iter() {
            callback(ptr.as_ptr() as *const u8);
        }
    }
    
    /// Get the number of currently allocated objects
    pub fn allocated_object_count(&self) -> usize {
        self.allocated_objects.lock().unwrap().len()
    }
    
    /// Get total objects ever allocated from this cache
    pub fn total_allocated_count(&self) -> usize {
        self.total_objects_allocated.load(Ordering::Relaxed)
    }
    
    /// Clear all objects (used during GC reset)
    pub fn clear_allocated_objects(&self) {
        let mut allocated = self.allocated_objects.lock().unwrap();
        
        // Free all allocated objects to system
        for &ptr in allocated.iter() {
            use std::alloc::{dealloc, Layout};
            let layout = Layout::from_size_align(self.object_size, 8)
                .expect("Invalid layout for object deallocation");
            unsafe {
                dealloc(ptr.as_ptr(), layout);
            }
        }
        
        allocated.clear();
        
        // Also clear free objects
        let mut free_objects = self.free_objects.lock().unwrap();
        for &ptr in free_objects.iter() {
            use std::alloc::{dealloc, Layout};
            let layout = Layout::from_size_align(self.object_size, 8)
                .expect("Invalid layout for object deallocation");
            unsafe {
                dealloc(ptr.as_ptr(), layout);
            }
        }
        free_objects.clear();
    }
}

impl ThreadCache {
    fn new(thread_id: thread::ThreadId) -> Self {
        let mut free_lists = Vec::with_capacity(SIZE_CLASSES.len());
        for &size in SIZE_CLASSES.iter() {
            free_lists.push(FreeList::new(size));
        }
        
        Self {
            free_lists,
            cached_bytes: AtomicUsize::new(0),
            thread_id,
        }
    }
    
    fn allocate(&self, size_class_index: usize) -> Option<NonNull<u8>> {
        if size_class_index >= self.free_lists.len() {
            return None;
        }
        
        self.free_lists[size_class_index].pop()
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, size_class_index: usize) {
        if size_class_index < self.free_lists.len() {
            self.free_lists[size_class_index].push(ptr);
        }
    }
    
    fn flush_to_central(&self, central_caches: &[CentralCache]) {
        for (index, free_list) in self.free_lists.iter().enumerate() {
            if index < central_caches.len() {
                free_list.flush_to_central(&central_caches[index]);
            }
        }
    }
    
    fn cached_bytes(&self) -> usize {
        self.cached_bytes.load(Ordering::Relaxed)
    }
    
    fn flush(&self) {
        // Clear all free lists
        for free_list in &self.free_lists {
            free_list.clear();
        }
        self.cached_bytes.store(0, Ordering::Relaxed);
    }
}

impl FreeList {
    fn new(object_size: usize) -> Self {
        Self {
            head: AtomicPtr::new(std::ptr::null_mut()),
            count: AtomicUsize::new(0),
            object_size,
        }
    }
    
    fn push(&self, ptr: NonNull<u8>) {
        // Simple lock-free stack implementation
        let new_head = ptr.as_ptr();
        let mut current_head = self.head.load(Ordering::Acquire);
        
        loop {
            unsafe {
                // Store the current head as the next pointer
                *(new_head as *mut *mut u8) = current_head;
            }
            
            match self.head.compare_exchange_weak(
                current_head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.count.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Err(head) => current_head = head,
            }
        }
    }
    
    fn pop(&self) -> Option<NonNull<u8>> {
        loop {
            let current_head = self.head.load(Ordering::Acquire);
            if current_head.is_null() {
                return None;
            }
            
            let next = unsafe { *(current_head as *const *mut u8) };
            
            match self.head.compare_exchange_weak(
                current_head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.count.fetch_sub(1, Ordering::Relaxed);
                    return NonNull::new(current_head);
                }
                Err(_) => continue,
            }
        }
    }
    
    fn clear(&self) {
        while self.pop().is_some() {}
    }
    
    fn flush_to_central(&self, central_cache: &CentralCache) {
        // Move objects from thread cache to central cache
        while let Some(ptr) = self.pop() {
            central_cache.deallocate(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prism_allocator_creation() {
        let allocator = PrismAllocator::new();
        assert!(!allocator.should_trigger_gc());
    }
    
    #[test]
    fn test_size_class_finding() {
        let allocator = PrismAllocator::new();
        
        // Test various sizes
        assert_eq!(allocator.find_size_class(1), Some(0)); // Should map to size class 8
        assert_eq!(allocator.find_size_class(8), Some(0)); // Exact match
        assert_eq!(allocator.find_size_class(9), Some(1)); // Should map to size class 16
        assert_eq!(allocator.find_size_class(32768), Some(SIZE_CLASSES.len() - 1));
        assert_eq!(allocator.find_size_class(32769), None); // Too large
    }
    
    #[test]
    fn test_allocation_and_deallocation() {
        let allocator = PrismAllocator::new();
        
        // Allocate some memory
        let ptr = allocator.allocate(64, 8);
        assert!(ptr.is_some());
        
        // Deallocate it
        if let Some(ptr) = ptr {
            allocator.deallocate(ptr, 64);
        }
        
        let stats = allocator.stats();
        assert!(stats.allocation_count > 0);
    }
    
    #[test]
    fn test_free_list_operations() {
        let free_list = FreeList::new(64);
        
        // Create a dummy pointer (normally this would be real allocated memory)
        let dummy_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        
        // Test push and pop
        free_list.push(dummy_ptr);
        assert_eq!(free_list.count.load(Ordering::Relaxed), 1);
        
        let popped = free_list.pop();
        assert!(popped.is_some());
        assert_eq!(free_list.count.load(Ordering::Relaxed), 0);
    }
} 