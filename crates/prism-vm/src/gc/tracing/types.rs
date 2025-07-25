//! Core types and interfaces for the tracing subsystem
//!
//! This module defines the fundamental types used throughout the tracing
//! subsystem, including tracer interfaces, tracing contexts, and result types.
//!
//! ## Design Principles
//!
//! - **Safety**: All tracing operations are memory-safe with proper bounds checking
//! - **Performance**: Zero-cost abstractions with efficient tracing paths
//! - **Modularity**: Clean separation between different tracer types
//! - **Extensibility**: Easy to add new tracer implementations
//! - **Thread Safety**: All operations are thread-safe where needed

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

// Re-export ObjectHeader from the main GC module
pub use crate::gc::ObjectHeader;

/// Core trait for tracing references within objects
/// 
/// This trait provides a unified interface for tracing GC references in different
/// object types. Implementations must be thread-safe and memory-safe.
pub trait ObjectTracer: Send + Sync {
    /// Trace all GC references within an object
    /// 
    /// # Safety
    /// 
    /// - `object_ptr` must point to a valid object with proper header
    /// - `object_size` must be the actual size of the object
    /// - The object must remain valid for the duration of this call
    /// 
    /// # Returns
    /// 
    /// Vector of pointers to GC-managed objects found within this object
    unsafe fn trace_references(&self, object_ptr: *const u8, object_size: usize) -> Vec<*const u8>;
    
    /// Get the size of an object (if tracer knows the layout)
    /// 
    /// This is an optimization that allows tracers to provide size information
    /// without requiring a separate size calculation.
    fn get_object_size(&self, object_ptr: *const u8) -> Option<usize> {
        let _ = object_ptr; // Suppress unused parameter warning
        None
    }
    
    /// Check if this object type contains any GC references
    /// 
    /// This is an optimization that allows skipping tracing for types
    /// that are known to contain no references.
    fn has_references(&self) -> bool {
        true
    }
    
    /// Get a human-readable name for this tracer (for debugging)
    fn tracer_name(&self) -> &'static str {
        "ObjectTracer"
    }
    
    /// Validate that an object can be traced by this tracer
    /// 
    /// This provides additional safety checks in debug builds
    fn can_trace(&self, _object_ptr: *const u8, _object_size: usize) -> bool {
        true
    }
}

/// Configuration for tracing operations
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Enable parallel tracing for large object graphs
    pub enable_parallel_tracing: bool,
    /// Maximum depth for recursive tracing (cycle detection)
    pub max_tracing_depth: usize,
    /// Enable conservative pointer scanning as fallback
    pub enable_conservative_scanning: bool,
    /// Thread pool size for parallel tracing
    pub tracing_thread_pool_size: usize,
    /// Enable tracing statistics collection
    pub enable_tracing_stats: bool,
    /// Cache size for recently traced objects
    pub tracing_cache_size: usize,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enable_parallel_tracing: true,
            max_tracing_depth: 1000,
            enable_conservative_scanning: false,
            tracing_thread_pool_size: num_cpus::get().min(8),
            enable_tracing_stats: cfg!(debug_assertions),
            tracing_cache_size: 1024,
        }
    }
}

/// Context information for tracing operations
#[derive(Debug, Clone)]
pub struct TracingContext {
    /// Current tracing depth (for cycle detection)
    pub depth: usize,
    /// Set of objects currently being traced (cycle detection)
    pub tracing_stack: HashSet<*const u8>,
    /// Statistics collector (if enabled)
    pub stats: Option<Arc<TracingStats>>,
    /// Start time of current tracing operation
    pub start_time: Instant,
}

impl TracingContext {
    /// Create a new tracing context
    pub fn new(config: &TracingConfig) -> Self {
        Self {
            depth: 0,
            tracing_stack: HashSet::new(),
            stats: if config.enable_tracing_stats {
                Some(Arc::new(TracingStats::new()))
            } else {
                None
            },
            start_time: Instant::now(),
        }
    }
    
    /// Check if we can trace deeper (cycle and depth detection)
    pub fn can_trace_deeper(&self, object_ptr: *const u8, max_depth: usize) -> bool {
        self.depth < max_depth && !self.tracing_stack.contains(&object_ptr)
    }
    
    /// Enter tracing for an object (update context)
    pub fn enter_object(&mut self, object_ptr: *const u8) -> bool {
        if self.tracing_stack.insert(object_ptr) {
            self.depth += 1;
            true
        } else {
            false // Cycle detected
        }
    }
    
    /// Exit tracing for an object (update context)
    pub fn exit_object(&mut self, object_ptr: *const u8) {
        self.tracing_stack.remove(&object_ptr);
        self.depth = self.depth.saturating_sub(1);
    }
}

/// Statistics for tracing operations
#[derive(Debug, Default)]
pub struct TracingStats {
    /// Total objects traced
    pub objects_traced: std::sync::atomic::AtomicUsize,
    /// Total references found
    pub references_found: std::sync::atomic::AtomicUsize,
    /// Total time spent tracing
    pub total_tracing_time: std::sync::atomic::AtomicU64, // in nanoseconds
    /// Number of cycles detected
    pub cycles_detected: std::sync::atomic::AtomicUsize,
    /// Cache hits during tracing
    pub cache_hits: std::sync::atomic::AtomicUsize,
    /// Cache misses during tracing
    pub cache_misses: std::sync::atomic::AtomicUsize,
}

impl TracingStats {
    /// Create new tracing statistics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record an object being traced
    pub fn record_object_traced(&self) {
        self.objects_traced.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Record references found in an object
    pub fn record_references_found(&self, count: usize) {
        self.references_found.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Record time spent tracing
    pub fn record_tracing_time(&self, duration: std::time::Duration) {
        self.total_tracing_time.fetch_add(
            duration.as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed
        );
    }
    
    /// Record a cycle detection
    pub fn record_cycle_detected(&self) {
        self.cycles_detected.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Get current statistics snapshot
    pub fn snapshot(&self) -> TracingStatsSnapshot {
        TracingStatsSnapshot {
            objects_traced: self.objects_traced.load(std::sync::atomic::Ordering::Relaxed),
            references_found: self.references_found.load(std::sync::atomic::Ordering::Relaxed),
            total_tracing_time_ns: self.total_tracing_time.load(std::sync::atomic::Ordering::Relaxed),
            cycles_detected: self.cycles_detected.load(std::sync::atomic::Ordering::Relaxed),
            cache_hits: self.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
            cache_misses: self.cache_misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// Snapshot of tracing statistics at a point in time
#[derive(Debug, Clone)]
pub struct TracingStatsSnapshot {
    pub objects_traced: usize,
    pub references_found: usize,
    pub total_tracing_time_ns: u64,
    pub cycles_detected: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl TracingStatsSnapshot {
    /// Calculate average references per object
    pub fn avg_references_per_object(&self) -> f64 {
        if self.objects_traced > 0 {
            self.references_found as f64 / self.objects_traced as f64
        } else {
            0.0
        }
    }
    
    /// Calculate average tracing time per object in nanoseconds
    pub fn avg_tracing_time_per_object_ns(&self) -> f64 {
        if self.objects_traced > 0 {
            self.total_tracing_time_ns as f64 / self.objects_traced as f64
        } else {
            0.0
        }
    }
    
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses > 0 {
            self.cache_hits as f64 / total_accesses as f64
        } else {
            0.0
        }
    }
}

/// Result of a tracing operation
#[derive(Debug)]
pub struct TracingResult {
    /// References found during tracing
    pub references: Vec<*const u8>,
    /// Whether any cycles were detected
    pub cycles_detected: bool,
    /// Time taken for the tracing operation
    pub duration: std::time::Duration,
    /// Number of objects traced
    pub objects_traced: usize,
}

impl TracingResult {
    /// Create a new tracing result
    pub fn new(references: Vec<*const u8>) -> Self {
        Self {
            references,
            cycles_detected: false,
            duration: std::time::Duration::default(),
            objects_traced: 0,
        }
    }
    
    /// Create an empty tracing result
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }
}

/// Error types for tracing operations
#[derive(Debug, Clone)]
pub enum TracingError {
    /// Invalid object pointer provided
    InvalidObjectPointer(*const u8),
    /// Object size mismatch
    SizeMismatch { expected: usize, actual: usize },
    /// Maximum tracing depth exceeded
    MaxDepthExceeded(usize),
    /// Tracer not found for object type
    TracerNotFound(u32),
    /// Tracing operation timed out
    Timeout(std::time::Duration),
    /// Generic tracing error with message
    Generic(String),
}

impl std::fmt::Display for TracingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TracingError::InvalidObjectPointer(ptr) => {
                write!(f, "Invalid object pointer: {:p}", ptr)
            }
            TracingError::SizeMismatch { expected, actual } => {
                write!(f, "Object size mismatch: expected {}, got {}", expected, actual)
            }
            TracingError::MaxDepthExceeded(depth) => {
                write!(f, "Maximum tracing depth exceeded: {}", depth)
            }
            TracingError::TracerNotFound(type_id) => {
                write!(f, "Tracer not found for type ID: {}", type_id)
            }
            TracingError::Timeout(duration) => {
                write!(f, "Tracing operation timed out after {:?}", duration)
            }
            TracingError::Generic(msg) => {
                write!(f, "Tracing error: {}", msg)
            }
        }
    }
}

impl std::error::Error for TracingError {}

/// Type alias for tracing operation results
pub type TracingOperationResult<T> = Result<T, TracingError>;

/// Trait for objects that can provide tracing information about themselves
/// 
/// This is an optional optimization that objects can implement to provide
/// more efficient tracing than the generic ObjectTracer implementations.
pub trait SelfTracing {
    /// Trace references from this object
    /// 
    /// # Safety
    /// 
    /// This method must only be called on valid objects and must return
    /// valid GC pointers.
    unsafe fn trace_self(&self) -> Vec<*const u8>;
    
    /// Check if this object has any references
    fn has_references(&self) -> bool;
}

/// Marker trait for objects that are known to contain no GC references
/// 
/// This is a compile-time optimization that allows the tracing system
/// to skip certain object types entirely.
pub trait NoReferences: Send + Sync {}

/// Trait for providing type-specific tracing information
/// 
/// This allows the type system to provide additional information that
/// can improve tracing performance and accuracy.
pub trait TypedTracing {
    /// Get the type ID for this object type
    fn type_id() -> u32;
    
    /// Get the tracer for this object type
    fn get_tracer() -> Arc<dyn ObjectTracer>;
    
    /// Register the tracer for this type
    fn register_tracer();
} 