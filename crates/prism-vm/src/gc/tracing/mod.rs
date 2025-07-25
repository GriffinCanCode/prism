//! Modular Tracing Subsystem for Prism VM Garbage Collector
//!
//! This module provides a comprehensive object tracing system designed for the
//! Prism VM garbage collector. The system is organized into focused modules
//! with clear separation of concerns:
//!
//! ## Architecture
//!
//! ```
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    TracingManager                           │  ← Main coordinator
//! ├─────────────────────────────────────────────────────────────┤
//! │ Registry │ Implementations │ Traversal │ Utils             │  ← Core components
//! ├─────────────────────────────────────────────────────────────┤
//! │                      Types                                  │  ← Shared types
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Modules
//!
//! - **`types`**: Core types, traits, and data structures
//! - **`registry`**: Thread-safe tracer registration and lookup
//! - **`implementations`**: Concrete tracer implementations (Array, Struct, etc.)
//! - **`manager`**: Central coordinator for all tracing operations
//! - **`traversal`**: Object graph traversal algorithms with cycle detection
//! - **`utils`**: Utility functions and common operations
//!
//! ## Design Principles
//!
//! - **Modularity**: Each component has a single, well-defined responsibility
//! - **Performance**: Optimized tracing paths with minimal overhead
//! - **Safety**: Memory-safe operations with proper error handling
//! - **Extensibility**: Easy to add new tracer types and traversal strategies
//! - **Thread Safety**: All operations are thread-safe where needed
//!
//! ## Usage
//!
//! The tracing subsystem is designed to be used primarily through the
//! `TracingManager`, which provides a high-level interface for all tracing
//! operations:
//!
//! ```rust
//! use prism_vm::gc::tracing::TracingManager;
//!
//! // Initialize the tracing manager
//! let manager = TracingManager::new()?;
//!
//! // Trace a single object
//! let references = manager.trace_object(object_ptr)?;
//!
//! // Traverse an object graph
//! let mut visitor = MyVisitor::new();
//! let stats = manager.traverse_object_graph(&roots, &mut visitor)?;
//!
//! // Register a custom tracer
//! manager.register_tracer(type_id, my_tracer)?;
//! ```
//!
//! ## Integration with GC Components
//!
//! The tracing subsystem is designed to integrate cleanly with other GC
//! components:
//!
//! - **Collectors**: Use tracing for mark phases and reachability analysis
//! - **Allocators**: Provide type information for tracer selection
//! - **Heap**: Coordinate object layout information
//! - **Roots**: Work together for complete graph traversal

pub mod types;
pub mod registry;
pub mod implementations;
pub mod manager;
pub mod traversal;
pub mod utils;

// Re-export key types and interfaces for convenience
pub use types::{
    ObjectTracer, TracingConfig, TracingContext, TracingResult, TracingError,
    TracingOperationResult, TracingStats, TracingStatsSnapshot,
    SelfTracing, NoReferences, TypedTracing
};

pub use registry::{
    init_tracer_registry, init_tracer_registry_with_config,
    get_tracer, register_tracer, unregister_tracer, has_tracer,
    registered_types, registry_stats, RegistryStats
};

pub use implementations::{
    ArrayTracer, StructTracer, ClosureTracer, PrismObjectTracer,
    CaptureType, TracerFactory, TracerBuilder
};

pub use manager::{
    TracingManager, TracingManagerStats, TracingPerformanceMetrics
};

pub use traversal::{
    ObjectVisitor, ObjectGraphTraverser, ParallelObjectGraphTraverser,
    BreadthFirstTraverser, TraversalStats, TraversalConfig,
    visitors::{CollectingVisitor, CountingVisitor, ValidationVisitor}
};

pub use utils::{
    trace_object, trace_objects_parallel, trace_objects_sequential,
    count_references, has_references, collect_reachable_objects,
    count_reachable_objects, validate_object_graph,
    calculate_object_graph_stats, ObjectGraphStats, MemoryUsageStats,
    BatchTracer, BatchTracingConfig,
    patterns::{find_objects_by_type, find_large_objects, find_shortest_path}
};

use std::sync::OnceLock;
use std::sync::Arc;

/// Global tracing manager instance
static GLOBAL_TRACING_MANAGER: OnceLock<Arc<TracingManager>> = OnceLock::new();

/// Initialize the global tracing manager
/// 
/// This must be called before any tracing operations. It's safe to call
/// multiple times - subsequent calls will be ignored.
pub fn init_tracing_subsystem() -> Result<(), TracingError> {
    init_tracing_subsystem_with_config(TracingConfig::default())
}

/// Initialize the global tracing manager with custom configuration
pub fn init_tracing_subsystem_with_config(config: TracingConfig) -> Result<(), TracingError> {
    let manager = TracingManager::with_config(config)?;
    GLOBAL_TRACING_MANAGER.get_or_init(|| Arc::new(manager));
    Ok(())
}

/// Get the global tracing manager
/// 
/// Returns the global tracing manager instance. Panics if the tracing
/// subsystem has not been initialized.
pub fn get_tracing_manager() -> &'static Arc<TracingManager> {
    GLOBAL_TRACING_MANAGER.get()
        .expect("Tracing subsystem not initialized. Call init_tracing_subsystem() first.")
}

/// Convenience function to trace a single object using the global manager
/// 
/// # Safety
/// 
/// The object pointer must be valid and point to a properly formatted object.
pub unsafe fn trace_single_object(object_ptr: *const u8) -> Result<Vec<*const u8>, TracingError> {
    get_tracing_manager().trace_object(object_ptr)
}

/// Convenience function to collect reachable objects using the global manager
pub fn collect_all_reachable_objects(roots: &[*const u8]) -> Result<Vec<*const u8>, TracingError> {
    get_tracing_manager().collect_reachable_objects(roots)
}

/// Convenience function to analyze an object graph using the global manager
pub fn analyze_graph(roots: &[*const u8]) -> Result<ObjectGraphStats, TracingError> {
    get_tracing_manager().analyze_object_graph(roots)
}

/// Macro for implementing the ObjectTracer trait for simple struct types
/// 
/// This macro generates a tracer implementation for structs with known
/// GC pointer field offsets.
/// 
/// # Example
/// 
/// ```rust
/// struct MyStruct {
///     field1: *const u8,  // GC pointer at offset 0
///     field2: i32,        // Non-GC field
///     field3: *const u8,  // GC pointer at offset 16
/// }
/// 
/// impl_struct_tracer!(MyStruct, [0, 16]);
/// ```
#[macro_export]
macro_rules! impl_struct_tracer {
    ($struct_type:ty, [$($offset:expr),*]) => {
        impl $crate::gc::tracing::SelfTracing for $struct_type {
            unsafe fn trace_self(&self) -> Vec<*const u8> {
                let mut references = Vec::new();
                let base_ptr = self as *const Self as *const u8;
                
                $(
                    let field_ptr = base_ptr.add($offset);
                    let gc_ptr = *(field_ptr as *const *const u8);
                    if !gc_ptr.is_null() {
                        references.push(gc_ptr);
                    }
                )*
                
                references
            }
            
            fn has_references(&self) -> bool {
                true
            }
        }
    };
}

/// Macro for implementing the NoReferences trait for types with no GC pointers
/// 
/// This macro marks a type as containing no GC references, allowing the
/// tracing system to skip it entirely.
/// 
/// # Example
/// 
/// ```rust
/// struct PlainData {
///     value: i32,
///     name: String,
/// }
/// 
/// impl_no_references!(PlainData);
/// ```
#[macro_export]
macro_rules! impl_no_references {
    ($type:ty) => {
        impl $crate::gc::tracing::NoReferences for $type {}
        
        impl $crate::gc::tracing::SelfTracing for $type {
            unsafe fn trace_self(&self) -> Vec<*const u8> {
                Vec::new()
            }
            
            fn has_references(&self) -> bool {
                false
            }
        }
    };
}

/// Macro for registering a tracer for a specific type
/// 
/// This macro provides a convenient way to register tracers during
/// initialization.
/// 
/// # Example
/// 
/// ```rust
/// register_tracer!(42, ArrayTracer::new(8, 0));
/// ```
#[macro_export]
macro_rules! register_tracer {
    ($type_id:expr, $tracer:expr) => {
        $crate::gc::tracing::register_tracer($type_id, std::sync::Arc::new($tracer))
            .expect("Failed to register tracer");
    };
}

/// Integration utilities for other GC components
pub mod integration {
    use super::*;
    
    /// Integration interface for collectors
    /// 
    /// This provides a clean interface for garbage collectors to use
    /// the tracing subsystem.
    pub struct CollectorIntegration;
    
    impl CollectorIntegration {
        /// Trace all objects reachable from roots for marking phase
        pub fn mark_reachable_objects(
            roots: &[*const u8],
            mark_callback: impl Fn(*const u8),
        ) -> Result<usize, TracingError> {
            let mut marked_count = 0;
            let mut visitor = MarkingVisitor::new(&mark_callback, &mut marked_count);
            
            get_tracing_manager().traverse_object_graph(roots, &mut visitor)?;
            Ok(marked_count)
        }
        
        /// Count objects and references for heap statistics
        pub fn count_heap_objects(roots: &[*const u8]) -> Result<(usize, usize), TracingError> {
            get_tracing_manager().analyze_object_graph(roots)
                .map(|stats| (stats.total_objects, stats.total_references))
        }
        
        /// Validate heap integrity
        pub fn validate_heap(roots: &[*const u8]) -> Result<bool, TracingError> {
            let errors = utils::validate_object_graph(roots)?;
            Ok(errors.is_empty())
        }
    }
    
    /// Visitor for marking objects during GC
    struct MarkingVisitor<'a, F> {
        mark_callback: &'a F,
        marked_count: &'a mut usize,
    }
    
    impl<'a, F> MarkingVisitor<'a, F>
    where
        F: Fn(*const u8),
    {
        fn new(mark_callback: &'a F, marked_count: &'a mut usize) -> Self {
            Self {
                mark_callback,
                marked_count,
            }
        }
    }
    
    impl<'a, F> ObjectVisitor for MarkingVisitor<'a, F>
    where
        F: Fn(*const u8),
    {
        fn visit_object(&mut self, object_ptr: *const u8, _object_size: usize, _depth: usize) -> bool {
            (self.mark_callback)(object_ptr);
            *self.marked_count += 1;
            true
        }
    }
    
    /// Integration interface for allocators
    /// 
    /// This provides utilities for allocators to work with the tracing system.
    pub struct AllocatorIntegration;
    
    impl AllocatorIntegration {
        /// Register tracers for common allocation patterns
        pub fn register_common_tracers() -> Result<(), TracingError> {
            // Register tracers for common allocation patterns based on allocator knowledge
            
            // Common array types
            register_tracer(1000, TracerFactory::array_of_pointers(std::mem::size_of::<*const u8>()))?; // Pointer arrays
            register_tracer(1001, TracerFactory::array_of_objects(16, 0))?; // Small object arrays
            register_tracer(1002, TracerFactory::array_of_objects(32, 8))?; // Medium object arrays
            register_tracer(1003, TracerFactory::array_of_objects(64, 16))?; // Large object arrays
            
            // Common struct patterns
            register_tracer(2000, TracerFactory::simple_struct(vec![0]))?; // Single pointer struct
            register_tracer(2001, TracerFactory::simple_struct(vec![0, 8]))?; // Two pointer struct
            register_tracer(2002, TracerFactory::simple_struct(vec![0, 8, 16]))?; // Three pointer struct
            register_tracer(2003, TracerFactory::simple_struct(vec![8, 24]))?; // Sparse pointer struct
            
            // String-like objects (length + data pointer)
            register_tracer(3000, TracerFactory::simple_struct(vec![8]))?; // String with pointer at offset 8
            
            // Hash table nodes (key, value, next pointers)
            register_tracer(4000, TracerFactory::simple_struct(vec![0, 8, 16]))?; // Hash node
            
            // Linked list nodes (data, next)
            register_tracer(5000, TracerFactory::simple_struct(vec![0, 8]))?; // List node
            
            // Tree nodes (left, right, data)
            register_tracer(6000, TracerFactory::simple_struct(vec![0, 8, 16]))?; // Binary tree node
            
            // Closure types with different capture patterns
            register_tracer(7000, TracerFactory::simple_closure(vec![0]))?; // Single capture
            register_tracer(7001, TracerFactory::simple_closure(vec![0, 8]))?; // Double capture
            register_tracer(7002, TracerFactory::simple_closure(vec![0, 8, 16, 24]))?; // Quad capture
            
            Ok(())
        }
        
        /// Get recommended tracer for an allocation size and pattern
        pub fn recommend_tracer(size: usize, pattern: AllocationPattern) -> Arc<dyn ObjectTracer> {
            match pattern {
                AllocationPattern::Array => TracerFactory::array_of_pointers(8),
                AllocationPattern::Struct => TracerFactory::conservative(),
                AllocationPattern::Unknown => TracerFactory::conservative(),
            }
        }
    }
    
    /// Allocation patterns that can influence tracer selection
    #[derive(Debug, Clone, Copy)]
    pub enum AllocationPattern {
        Array,
        Struct,
        Unknown,
    }
}

/// Testing utilities for the tracing subsystem
#[cfg(test)]
pub mod testing {
    use super::*;
    use std::sync::Mutex;
    
    /// Mock object for testing tracers
    pub struct MockObject {
        pub header: ObjectHeader,
        pub data: Vec<u8>,
    }
    
    impl MockObject {
        pub fn new(type_id: u32, size: usize) -> Self {
            Self {
                header: ObjectHeader {
                    size,
                    type_id,
                    mark_bits: 0,
                    generation: 0,
                    ref_count: 0,
                },
                data: vec![0; size.saturating_sub(std::mem::size_of::<ObjectHeader>())],
            }
        }
        
        pub fn as_ptr(&self) -> *const u8 {
            self as *const Self as *const u8
        }
        
        pub fn set_reference_at(&mut self, offset: usize, ptr: *const u8) {
            if offset + std::mem::size_of::<*const u8>() <= self.data.len() {
                unsafe {
                    let data_ptr = self.data.as_mut_ptr().add(offset);
                    *(data_ptr as *mut *const u8) = ptr;
                }
            }
        }
    }
    
    /// Test visitor that collects visited objects
    pub struct TestVisitor {
        pub visited_objects: Vec<*const u8>,
        pub visit_order: Vec<usize>,
    }
    
    impl TestVisitor {
        pub fn new() -> Self {
            Self {
                visited_objects: Vec::new(),
                visit_order: Vec::new(),
            }
        }
    }
    
    impl ObjectVisitor for TestVisitor {
        fn visit_object(&mut self, object_ptr: *const u8, _object_size: usize, depth: usize) -> bool {
            self.visited_objects.push(object_ptr);
            self.visit_order.push(depth);
            true
        }
    }
    
    /// Initialize tracing subsystem for testing
    pub fn init_test_tracing() -> Result<(), TracingError> {
        let config = TracingConfig {
            enable_tracing_stats: true,
            tracing_cache_size: 64, // Small cache for testing
            ..Default::default()
        };
        init_tracing_subsystem_with_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use testing::*;
    
    #[test]
    fn test_tracing_subsystem_initialization() {
        // Test should work with a fresh subsystem state
        let result = init_tracing_subsystem();
        assert!(result.is_ok());
        
        // Second initialization should be ignored
        let result2 = init_tracing_subsystem();
        assert!(result2.is_ok());
    }
    
    #[test]
    fn test_mock_object_creation() {
        let mock = MockObject::new(42, 64);
        assert_eq!(mock.header.type_id, 42);
        assert_eq!(mock.header.size, 64);
        
        let ptr = mock.as_ptr();
        assert!(!ptr.is_null());
    }
    
    #[test]
    fn test_global_tracing_functions() {
        init_test_tracing().unwrap();
        
        // Test that global functions work
        let roots = vec![];
        let result = collect_all_reachable_objects(&roots);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
    
    #[test]
    fn test_integration_interfaces() {
        init_test_tracing().unwrap();
        
        let roots = vec![];
        let result = integration::CollectorIntegration::count_heap_objects(&roots);
        assert!(result.is_ok());
        
        let (object_count, reference_count) = result.unwrap();
        assert_eq!(object_count, 0);
        assert_eq!(reference_count, 0);
    }
} 