//! Heap Management Subsystem for Prism VM
//!
//! This module provides comprehensive heap management for the Prism VM garbage collector.
//! The design is modular and performance-oriented, with specialized components for:
//!
//! - **Core Heap**: Main heap management with object tracking
//! - **Size Classes**: Segregated free lists for efficient small object allocation  
//! - **Large Objects**: Specialized handling for objects above threshold
//! - **Fragmentation**: Compaction and coalescing to reduce fragmentation
//! - **Card Table**: Support for generational garbage collection
//! - **Regional Heaps**: Specialized heap regions for different object types
//! - **Statistics**: Performance monitoring and adaptive behavior
//! - **Memory Regions**: Cache-friendly memory organization
//!
//! ## Architecture
//!
//! The heap subsystem follows a layered architecture:
//! ```
//! ┌─────────────────┐
//! │   HeapManager   │  ← Main coordinator and public interface
//! ├─────────────────┤
//! │  Core | Regional│  ← Heap implementations
//! ├─────────────────┤  
//! │ Size │ Large │ │  ← Allocation strategies
//! │Class │ Object│ │
//! ├─────────────────┤
//! │ Fragmentation   │  ← Memory compaction
//! ├─────────────────┤
//! │   Card Table    │  ← Generational GC support
//! ├─────────────────┤
//! │ Memory Regions  │  ← Memory organization
//! └─────────────────┘
//! ```

pub mod types;
pub mod manager;
pub mod core; 
pub mod size_class;
pub mod large_object;
pub mod fragmentation;
pub mod card_table;
pub mod statistics;
pub mod regional;
pub mod memory_regions;

// Re-export key types and interfaces for convenience
pub use types::*;
pub use manager::HeapManager;
pub use core::Heap;
pub use size_class::SizeClassAllocator;
pub use large_object::LargeObjectAllocator;
pub use fragmentation::FragmentationManager;
pub use card_table::CardTable;
pub use statistics::*;
pub use regional::RegionalHeap;
pub use memory_regions::MemoryRegionManager;

/// Main heap interface that provides unified access to all heap functionality
/// 
/// This trait ensures consistency across different heap implementations while
/// allowing for specialized optimizations.
pub trait HeapInterface: Send + Sync {
    /// Register a newly allocated object
    fn register_object(&mut self, ptr: *const u8, header: crate::ObjectHeader);
    
    /// Get object header for a pointer
    fn get_header(&self, ptr: *const u8) -> Option<&crate::ObjectHeader>;
    
    /// Get mutable object header for a pointer  
    fn get_header_mut(&mut self, ptr: *const u8) -> Option<&mut crate::ObjectHeader>;
    
    /// Find all unmarked objects for collection
    fn find_white_objects(&self) -> Vec<*const u8>;
    
    /// Reset object colors for next collection cycle
    fn reset_colors_to_white(&mut self);
    
    /// Deallocate an object
    fn deallocate(&mut self, ptr: *const u8);
    
    /// Try to allocate from free lists
    fn try_allocate_from_free_list(&mut self, size: usize, align: usize) -> Option<*const u8>;
    
    /// Get heap statistics
    fn get_stats(&self) -> HeapStats;
    
    /// Check if collection is needed
    fn needs_collection(&self, threshold: f64) -> bool;
    
    /// Get memory pressure indicator
    fn memory_pressure(&self) -> f64;
    
    /// Compact the heap to reduce fragmentation
    fn compact(&mut self) -> usize;
    
    /// Verify heap integrity (debug builds only)
    #[cfg(debug_assertions)]
    fn verify_integrity(&self) -> Result<(), String>;
}

/// Factory for creating different types of heaps
pub struct HeapFactory;

impl HeapFactory {
    /// Create a standard heap with default configuration
    pub fn create_standard(capacity: usize) -> Box<dyn HeapInterface> {
        Box::new(Heap::new(capacity))
    }
    
    /// Create a regional heap for generational collection
    pub fn create_regional(capacity: usize, enable_generational: bool) -> RegionalHeap {
        RegionalHeap::new(capacity, enable_generational)
    }
    
    /// Create a heap optimized for specific workload characteristics
    pub fn create_optimized(config: HeapConfig) -> Box<dyn HeapInterface> {
        match config.heap_type {
            HeapType::Standard => Box::new(Heap::with_config(config)),
            HeapType::Regional => Box::new(RegionalHeap::with_config(config)),
        }
    }
} 