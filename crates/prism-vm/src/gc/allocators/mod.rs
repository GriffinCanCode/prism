//! Modular Allocator Subsystem for Prism VM
//!
//! This module provides a comprehensive memory allocation system with multiple
//! specialized allocators optimized for different use cases:
//!
//! - **PrismAllocator**: High-performance size-class based allocator with thread-local caches
//! - **BumpAllocator**: Fast bump-pointer allocator optimized for garbage collection
//! - **LargeObjectAllocator**: Specialized allocator for large objects
//! - **PageAllocator**: Low-level page management
//! - **ThreadCache**: Thread-local allocation caches for reduced contention
//!
//! ## Architecture
//!
//! The allocator subsystem is designed with the following principles:
//! - **Modularity**: Each allocator type is in its own module
//! - **Composability**: Allocators can be combined and coordinated
//! - **Performance**: Optimized allocation paths with minimal overhead
//! - **Safety**: Memory-safe interfaces with proper error handling
//! - **Observability**: Comprehensive statistics and monitoring

pub mod types;
pub mod manager;
pub mod prism;
pub mod bump;
pub mod large_object;
pub mod page;
pub mod thread_cache;
pub mod statistics;

#[cfg(test)]
mod tests;

// Re-export key types and traits for convenience
pub use types::*;
pub use manager::*;
pub use prism::PrismAllocator;
pub use bump::BumpAllocator;
pub use large_object::LargeObjectAllocator;
pub use page::PageAllocator;
pub use thread_cache::ThreadCache;
pub use statistics::*;

use std::ptr::NonNull;

/// Main allocator interface that all allocators must implement
/// 
/// This trait provides a unified interface for memory allocation across
/// different allocator implementations, ensuring consistency and interoperability.
pub trait Allocator: Send + Sync {
    /// Allocate memory with the specified size and alignment
    /// 
    /// # Arguments
    /// * `size` - Size in bytes to allocate
    /// * `align` - Required alignment (must be power of 2)
    /// 
    /// # Returns
    /// * `Some(ptr)` - Successfully allocated memory pointer
    /// * `None` - Allocation failed (out of memory or invalid parameters)
    /// 
    /// # Safety
    /// The returned pointer is valid for reads and writes up to `size` bytes
    /// and is aligned to the requested alignment.
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>>;
    
    /// Deallocate previously allocated memory
    /// 
    /// # Arguments
    /// * `ptr` - Pointer to memory to deallocate
    /// * `size` - Size of the allocation (must match original allocation)
    /// 
    /// # Safety
    /// The pointer must have been returned by a previous call to `allocate`
    /// on the same allocator instance, and must not have been deallocated already.
    fn deallocate(&self, ptr: NonNull<u8>, size: usize);
    
    /// Get current allocation statistics
    fn stats(&self) -> AllocationStats;
    
    /// Check if garbage collection should be triggered
    fn should_trigger_gc(&self) -> bool;
    
    /// Prepare for garbage collection (flush caches, etc.)
    fn prepare_for_gc(&self);
    
    /// Get allocator-specific configuration
    fn get_config(&self) -> AllocatorConfig;
    
    /// Reconfigure the allocator
    fn reconfigure(&self, config: AllocatorConfig);
}

/// Specialized allocator interface for GC-aware allocators
/// 
/// This trait extends the basic Allocator interface with garbage collection
/// specific functionality, such as object iteration and root scanning.
pub trait GcAllocator: Allocator {
    /// Iterate over all allocated objects for GC marking
    /// 
    /// # Arguments
    /// * `callback` - Function called for each object with (pointer, size)
    /// 
    /// # Safety
    /// The callback must not modify the allocator state or perform allocations
    /// during iteration.
    fn iter_objects<F>(&self, callback: F) 
    where 
        F: FnMut(*const u8, usize);
    
    /// Reset allocator state after garbage collection
    /// 
    /// This is called after a GC cycle to reset internal state,
    /// such as clearing allocation regions in bump allocators.
    fn post_gc_reset(&self);
    
    /// Get memory usage information for GC heuristics
    fn memory_usage(&self) -> MemoryUsage;
}

/// Factory for creating different types of allocators
pub struct AllocatorFactory;

impl AllocatorFactory {
    /// Create a new PrismAllocator with default configuration
    pub fn new_prism_allocator() -> PrismAllocator {
        PrismAllocator::new()
    }
    
    /// Create a new PrismAllocator with custom configuration
    pub fn new_prism_allocator_with_config(config: PrismAllocatorConfig) -> PrismAllocator {
        PrismAllocator::with_config(config)
    }
    
    /// Create a new BumpAllocator with specified initial capacity
    pub fn new_bump_allocator(initial_capacity: usize) -> BumpAllocator {
        BumpAllocator::new(initial_capacity)
    }
    
    /// Create a new BumpAllocator with custom configuration
    pub fn new_bump_allocator_with_config(config: BumpAllocatorConfig) -> BumpAllocator {
        BumpAllocator::with_config(config)
    }
    
    /// Create a new LargeObjectAllocator
    pub fn new_large_object_allocator() -> LargeObjectAllocator {
        LargeObjectAllocator::new()
    }
    
    /// Create a new PageAllocator
    pub fn new_page_allocator() -> PageAllocator {
        PageAllocator::new()
    }
} 