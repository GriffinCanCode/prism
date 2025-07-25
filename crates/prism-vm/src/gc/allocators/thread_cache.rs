//! Thread Cache - Thread-local allocation caches for reduced contention
//!
//! This module provides thread-local caches that reduce contention on
//! central allocator structures by keeping frequently used objects
//! in thread-local storage.

use super::types::*;
use super::prism::CentralCache;

use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::thread;

/// Thread-local allocation cache for fast allocation
#[derive(Debug)]
pub struct ThreadCache {
    /// Free lists for each size class
    free_lists: Vec<FreeList>,
    /// Total cached bytes
    cached_bytes: AtomicUsize,
    /// Cache generation for invalidation
    generation: AtomicUsize,
    /// Thread ID this cache belongs to
    thread_id: thread::ThreadId,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Free list for a specific size class in thread cache
#[derive(Debug)]
pub struct FreeList {
    /// Head of the free list
    head: AtomicPtr<u8>,
    /// Number of objects in this free list
    count: AtomicUsize,
    /// Size class this free list serves
    size_class: usize,
    /// Object size for this size class
    object_size: usize,
    /// Maximum objects to keep in this free list
    max_objects: usize,
}

impl ThreadCache {
    pub fn new(thread_id: thread::ThreadId, max_cache_size: usize) -> Self {
        let mut free_lists = Vec::with_capacity(SIZE_CLASSES.len());
        for (i, &size) in SIZE_CLASSES.iter().enumerate() {
            free_lists.push(FreeList::new(size, i, max_cache_size / SIZE_CLASSES.len()));
        }
        
        Self {
            free_lists,
            cached_bytes: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            thread_id,
            max_cache_size,
        }
    }
    
    /// Allocate from thread cache
    pub fn allocate(&self, size_class_index: usize) -> Option<NonNull<u8>> {
        if size_class_index < self.free_lists.len() {
            if let Some(ptr) = self.free_lists[size_class_index].pop() {
                let object_size = SIZE_CLASSES[size_class_index];
                self.cached_bytes.fetch_sub(object_size, Ordering::Relaxed);
                return Some(ptr);
            }
        }
        None
    }
    
    /// Deallocate to thread cache
    pub fn deallocate(&self, ptr: NonNull<u8>, size_class_index: usize) {
        if size_class_index < self.free_lists.len() {
            let object_size = SIZE_CLASSES[size_class_index];
            
            // Check if we have room in the cache
            let current_bytes = self.cached_bytes.load(Ordering::Relaxed);
            if current_bytes + object_size <= self.max_cache_size {
                if self.free_lists[size_class_index].push(ptr) {
                    self.cached_bytes.fetch_add(object_size, Ordering::Relaxed);
                }
            }
            // If cache is full, the object will be lost (in a real implementation,
            // you'd return it to the central cache)
        }
    }
    
    /// Flush thread cache to central caches
    pub fn flush_to_central(&self, central_caches: &[CentralCache]) {
        for (i, free_list) in self.free_lists.iter().enumerate() {
            if i < central_caches.len() {
                let objects_to_flush = free_list.count.load(Ordering::Relaxed) / 2;
                
                for _ in 0..objects_to_flush {
                    if let Some(ptr) = free_list.pop() {
                        // In a real implementation, you'd return this to central cache
                        // For now, we just track the reduction in cached bytes
                        let object_size = SIZE_CLASSES[i];
                        self.cached_bytes.fetch_sub(object_size, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    
    /// Get current cached bytes
    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes.load(Ordering::Relaxed)
    }
    
    /// Get thread ID
    pub fn thread_id(&self) -> thread::ThreadId {
        self.thread_id
    }
    
    /// Get cache generation (for invalidation)
    pub fn generation(&self) -> usize {
        self.generation.load(Ordering::Relaxed)
    }
    
    /// Increment cache generation (invalidates cache)
    pub fn increment_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Clear all cached objects
    pub fn clear(&self) {
        for free_list in &self.free_lists {
            free_list.clear();
        }
        self.cached_bytes.store(0, Ordering::Relaxed);
    }
    
    /// Get statistics for this thread cache
    pub fn get_stats(&self) -> ThreadCacheStats {
        let mut total_objects = 0;
        let mut per_size_class = Vec::new();
        
        for free_list in &self.free_lists {
            let count = free_list.count.load(Ordering::Relaxed);
            total_objects += count;
            per_size_class.push(count);
        }
        
        ThreadCacheStats {
            thread_id: self.thread_id,
            cached_bytes: self.cached_bytes(),
            total_objects,
            per_size_class,
            generation: self.generation(),
        }
    }
}

impl FreeList {
    fn new(size_class: usize, index: usize, max_objects: usize) -> Self {
        Self {
            head: AtomicPtr::new(std::ptr::null_mut()),
            count: AtomicUsize::new(0),
            size_class,
            object_size: SIZE_CLASSES[index],
            max_objects: max_objects.max(1), // At least 1 object
        }
    }
    
    /// Push an object onto the free list
    fn push(&self, ptr: NonNull<u8>) -> bool {
        // Check if we have room
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count >= self.max_objects {
            return false; // Cache full
        }
        
        let current_head = self.head.load(Ordering::Relaxed);
        
        // Link the new object to the current head
        unsafe {
            *(ptr.as_ptr() as *mut *mut u8) = current_head;
        }
        
        // Update head to point to the new object
        self.head.store(ptr.as_ptr(), Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        
        true
    }
    
    /// Pop an object from the free list
    fn pop(&self) -> Option<NonNull<u8>> {
        let current_head = self.head.load(Ordering::Relaxed);
        if current_head.is_null() {
            return None;
        }
        
        let ptr = NonNull::new(current_head)?;
        
        // Get the next object in the list
        let next = unsafe { *(current_head as *mut *mut u8) };
        
        // Update head to point to the next object
        self.head.store(next, Ordering::Relaxed);
        self.count.fetch_sub(1, Ordering::Relaxed);
        
        Some(ptr)
    }
    
    /// Clear all objects from the free list
    fn clear(&self) {
        while self.pop().is_some() {
            // Keep popping until empty
        }
    }
    
    /// Get current count
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
    
    /// Check if the free list is empty
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
    
    /// Check if the free list is full
    pub fn is_full(&self) -> bool {
        self.count() >= self.max_objects
    }
}

/// Statistics for a thread cache
#[derive(Debug, Clone)]
pub struct ThreadCacheStats {
    /// Thread ID
    pub thread_id: thread::ThreadId,
    /// Total cached bytes
    pub cached_bytes: usize,
    /// Total cached objects
    pub total_objects: usize,
    /// Objects per size class
    pub per_size_class: Vec<usize>,
    /// Cache generation
    pub generation: usize,
}

/// Manager for all thread caches
pub struct ThreadCacheManager {
    /// Maximum cache size per thread
    max_cache_size: usize,
    /// Statistics
    stats: std::sync::Mutex<ThreadCacheManagerStats>,
}

/// Statistics for thread cache manager
#[derive(Debug, Default, Clone)]
pub struct ThreadCacheManagerStats {
    /// Number of active thread caches
    pub active_caches: usize,
    /// Total cached bytes across all threads
    pub total_cached_bytes: usize,
    /// Total cached objects across all threads
    pub total_cached_objects: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Cache flushes
    pub cache_flushes: usize,
}

impl ThreadCacheManager {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            max_cache_size,
            stats: std::sync::Mutex::new(ThreadCacheManagerStats::default()),
        }
    }
    
    /// Create a new thread cache
    pub fn create_cache(&self, thread_id: thread::ThreadId) -> ThreadCache {
        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.active_caches += 1;
        }
        
        ThreadCache::new(thread_id, self.max_cache_size)
    }
    
    /// Record cache hit
    pub fn record_hit(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_hits += 1;
        }
    }
    
    /// Record cache miss
    pub fn record_miss(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_misses += 1;
        }
    }
    
    /// Record cache flush
    pub fn record_flush(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.cache_flushes += 1;
        }
    }
    
    /// Get manager statistics
    pub fn get_stats(&self) -> ThreadCacheManagerStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Update statistics with thread cache data
    pub fn update_stats(&self, thread_caches: &[ThreadCacheStats]) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.active_caches = thread_caches.len();
            stats.total_cached_bytes = thread_caches.iter()
                .map(|tc| tc.cached_bytes)
                .sum();
            stats.total_cached_objects = thread_caches.iter()
                .map(|tc| tc.total_objects)
                .sum();
        }
    }
    
    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        if let Ok(stats) = self.stats.lock() {
            let total_requests = stats.cache_hits + stats.cache_misses;
            if total_requests > 0 {
                stats.cache_hits as f64 / total_requests as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

impl Default for ThreadCacheManager {
    fn default() -> Self {
        Self::new(THREAD_CACHE_SIZE * SIZE_CLASSES.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thread_cache_creation() {
        let thread_id = thread::current().id();
        let cache = ThreadCache::new(thread_id, 1024);
        
        assert_eq!(cache.thread_id(), thread_id);
        assert_eq!(cache.cached_bytes(), 0);
        assert_eq!(cache.generation(), 0);
    }
    
    #[test]
    fn test_free_list_operations() {
        let free_list = FreeList::new(32, 2, 10);
        
        assert!(free_list.is_empty());
        assert!(!free_list.is_full());
        
        // Create a dummy pointer (not actually allocated)
        let dummy_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        
        // Push object
        assert!(free_list.push(dummy_ptr));
        assert_eq!(free_list.count(), 1);
        assert!(!free_list.is_empty());
        
        // Pop object
        let popped = free_list.pop().unwrap();
        assert_eq!(popped, dummy_ptr);
        assert_eq!(free_list.count(), 0);
        assert!(free_list.is_empty());
    }
    
    #[test]
    fn test_free_list_capacity() {
        let free_list = FreeList::new(32, 2, 2); // Max 2 objects
        
        let ptr1 = NonNull::new(0x1000 as *mut u8).unwrap();
        let ptr2 = NonNull::new(0x2000 as *mut u8).unwrap();
        let ptr3 = NonNull::new(0x3000 as *mut u8).unwrap();
        
        // Fill to capacity
        assert!(free_list.push(ptr1));
        assert!(free_list.push(ptr2));
        assert!(free_list.is_full());
        
        // Should reject additional objects
        assert!(!free_list.push(ptr3));
        assert_eq!(free_list.count(), 2);
    }
    
    #[test]
    fn test_thread_cache_allocation() {
        let thread_id = thread::current().id();
        let cache = ThreadCache::new(thread_id, 1024);
        
        // Initially empty, allocation should fail
        assert!(cache.allocate(0).is_none());
        
        // Add an object and try to allocate
        let dummy_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        cache.deallocate(dummy_ptr, 0);
        
        // Now allocation should succeed
        let allocated = cache.allocate(0).unwrap();
        assert_eq!(allocated, dummy_ptr);
    }
    
    #[test]
    fn test_cache_statistics() {
        let thread_id = thread::current().id();
        let cache = ThreadCache::new(thread_id, 1024);
        
        let stats = cache.get_stats();
        assert_eq!(stats.thread_id, thread_id);
        assert_eq!(stats.cached_bytes, 0);
        assert_eq!(stats.total_objects, 0);
        assert_eq!(stats.generation, 0);
    }
    
    #[test]
    fn test_cache_clear() {
        let thread_id = thread::current().id();
        let cache = ThreadCache::new(thread_id, 1024);
        
        // Add some objects
        let ptr1 = NonNull::new(0x1000 as *mut u8).unwrap();
        let ptr2 = NonNull::new(0x2000 as *mut u8).unwrap();
        
        cache.deallocate(ptr1, 0);
        cache.deallocate(ptr2, 1);
        
        assert!(cache.cached_bytes() > 0);
        
        // Clear cache
        cache.clear();
        
        assert_eq!(cache.cached_bytes(), 0);
        assert!(cache.allocate(0).is_none());
        assert!(cache.allocate(1).is_none());
    }
    
    #[test]
    fn test_generation_tracking() {
        let thread_id = thread::current().id();
        let cache = ThreadCache::new(thread_id, 1024);
        
        assert_eq!(cache.generation(), 0);
        
        cache.increment_generation();
        assert_eq!(cache.generation(), 1);
        
        cache.increment_generation();
        assert_eq!(cache.generation(), 2);
    }
    
    #[test]
    fn test_thread_cache_manager() {
        let manager = ThreadCacheManager::new(1024);
        let thread_id = thread::current().id();
        
        let _cache = manager.create_cache(thread_id);
        
        let stats = manager.get_stats();
        assert_eq!(stats.active_caches, 1);
        
        // Test hit/miss tracking
        manager.record_hit();
        manager.record_hit();
        manager.record_miss();
        
        assert!((manager.hit_rate() - 0.666).abs() < 0.01); // 2/3 ≈ 0.666
    }
    
    #[test]
    fn test_cache_size_limits() {
        let thread_id = thread::current().id();
        let small_cache = ThreadCache::new(thread_id, 100); // Very small cache
        
        // Try to add more objects than the cache can hold
        let mut added_objects = 0;
        for i in 0..1000 {
            let ptr = NonNull::new((0x1000 + i * 8) as *mut u8).unwrap();
            small_cache.deallocate(ptr, 0); // All to size class 0
            
            if small_cache.cached_bytes() > 0 {
                added_objects += 1;
            }
            
            // Cache should not exceed its limit
            assert!(small_cache.cached_bytes() <= 100);
        }
        
        // Should have added some objects but not all
        assert!(added_objects > 0);
        assert!(added_objects < 1000);
    }
} 