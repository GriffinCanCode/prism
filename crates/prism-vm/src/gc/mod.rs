use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub mod collectors;
pub mod allocator;
pub mod barriers;
pub mod heap;
pub mod metadata;
pub mod roots;
pub mod tracing;

/// Core garbage collection trait that all collectors must implement
pub trait GarbageCollector: Send + Sync {
    /// Allocate memory for an object of the given size and alignment
    fn allocate(&self, size: usize, align: usize) -> Option<*mut u8>;
    
    /// Trigger a garbage collection cycle
    fn collect(&self) -> CollectionStats;
    
    /// Check if a collection should be triggered based on current state
    fn should_collect(&self) -> bool;
    
    /// Get current heap statistics
    fn heap_stats(&self) -> HeapStats;
    
    /// Set the garbage collection strategy/configuration
    fn configure(&self, config: GcConfig);
    
    /// Register a root object that should never be collected
    fn register_root(&self, ptr: *const u8);
    
    /// Unregister a previously registered root
    fn unregister_root(&self, ptr: *const u8);
    
    /// Mark an object as reachable during tracing
    fn mark_object(&self, ptr: *const u8);
    
    /// Check if an object is currently marked as live
    fn is_marked(&self, ptr: *const u8) -> bool;
}

/// Statistics collected during a garbage collection cycle
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub duration: Duration,
    pub bytes_collected: usize,
    pub objects_collected: usize,
    pub pause_time: Duration,
    pub heap_size_before: usize,
    pub heap_size_after: usize,
    pub collection_type: CollectionType,
}

#[derive(Debug, Clone, Copy)]
pub enum CollectionType {
    Minor,      // Young generation only
    Major,      // Full heap collection
    Incremental, // Partial collection
    Concurrent,  // Background collection
}

/// Current heap statistics
#[derive(Debug, Clone)]
pub struct HeapStats {
    pub total_allocated: usize,
    pub live_objects: usize,
    pub free_space: usize,
    pub fragmentation_ratio: f64,
    pub allocation_rate: f64, // bytes per second
    pub gc_overhead: f64,     // percentage of time spent in GC
}

/// Configuration for garbage collection behavior
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Target heap size before triggering collection
    pub heap_target: usize,
    /// Maximum pause time goal (for low-latency collectors)
    pub max_pause_time: Duration,
    /// Number of worker threads for parallel collection
    pub worker_threads: usize,
    /// Enable concurrent collection
    pub concurrent: bool,
    /// Enable generational collection
    pub generational: bool,
    /// Write barrier strategy
    pub write_barrier: WriteBarrierType,
    /// Collection trigger strategy
    pub trigger_strategy: TriggerStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum WriteBarrierType {
    None,
    Incremental,  // Dijkstra-style insertion barrier
    Snapshot,     // Yuasa-style deletion barrier  
    Hybrid,       // Combined approach (like Go's hybrid barrier)
}

#[derive(Debug, Clone)]
pub enum TriggerStrategy {
    /// Trigger when heap reaches a certain size
    HeapSize(usize),
    /// Trigger based on allocation rate
    AllocationRate(f64),
    /// Trigger at regular time intervals
    Periodic(Duration),
    /// Adaptive strategy that learns from application behavior
    Adaptive,
}

/// Object metadata stored in heap headers
#[derive(Debug, Clone, Copy)]
pub struct ObjectHeader {
    /// Size of the object in bytes
    pub size: usize,
    /// Object type identifier for tracing
    pub type_id: u32,
    /// Mark bits for garbage collection
    pub mark_bits: u8,
    /// Generation (for generational collectors)
    pub generation: u8,
    /// Reference count (for hybrid collectors)
    pub ref_count: u32,
}

/// Represents different object colors in tri-color marking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectColor {
    White,  // Potentially garbage
    Gray,   // Reachable but not yet scanned
    Black,  // Reachable and fully scanned
}

impl ObjectHeader {
    pub fn new(size: usize, type_id: u32) -> Self {
        Self {
            size,
            type_id,
            mark_bits: 0,
            generation: 0,
            ref_count: 0,
        }
    }
    
    pub fn get_color(&self) -> ObjectColor {
        match self.mark_bits & 0x3 {
            0 => ObjectColor::White,
            1 => ObjectColor::Gray,
            2 => ObjectColor::Black,
            _ => ObjectColor::White,
        }
    }
    
    pub fn set_color(&mut self, color: ObjectColor) {
        self.mark_bits = (self.mark_bits & !0x3) | match color {
            ObjectColor::White => 0,
            ObjectColor::Gray => 1,
            ObjectColor::Black => 2,
        };
    }
    
    pub fn is_pinned(&self) -> bool {
        (self.mark_bits & 0x4) != 0
    }
    
    pub fn set_pinned(&mut self, pinned: bool) {
        if pinned {
            self.mark_bits |= 0x4;
        } else {
            self.mark_bits &= !0x4;
        }
    }
}

/// Factory for creating different types of garbage collectors
pub struct GcFactory;

impl GcFactory {
    /// Create a tri-color mark-sweep collector optimized for low latency
    pub fn create_tricolor_collector(config: GcConfig) -> Box<dyn GarbageCollector> {
        Box::new(collectors::TriColorCollector::new(config))
    }
    
    /// Create a generational collector optimized for high throughput
    pub fn create_generational_collector(config: GcConfig) -> Box<dyn GarbageCollector> {
        Box::new(collectors::GenerationalCollector::new(config))
    }
    
    /// Create a concurrent collector with minimal pause times
    pub fn create_concurrent_collector(config: GcConfig) -> Box<dyn GarbageCollector> {
        Box::new(collectors::ConcurrentCollector::new(config))
    }
    
    /// Create a hybrid collector combining multiple strategies
    pub fn create_hybrid_collector(config: GcConfig) -> Box<dyn GarbageCollector> {
        Box::new(collectors::HybridCollector::new(config))
    }
    
    /// Create the optimal collector for Prism's specific use case
    pub fn create_prism_optimized_collector(config: GcConfig) -> Box<dyn GarbageCollector> {
        // For Prism VM, we want:
        // 1. Low pause times for interactive execution
        // 2. Good throughput for compilation workloads  
        // 3. Efficient handling of short-lived objects (AST nodes, temporaries)
        // 4. Support for long-lived objects (compiled code, caches)
        
        let optimized_config = GcConfig {
            concurrent: true,
            generational: true,
            write_barrier: WriteBarrierType::Hybrid,
            trigger_strategy: TriggerStrategy::Adaptive,
            max_pause_time: Duration::from_millis(1), // 1ms target
            ..config
        };
        
        Box::new(collectors::PrismCollector::new(optimized_config))
    }
}

/// Default configuration optimized for Prism's workload characteristics
impl Default for GcConfig {
    fn default() -> Self {
        Self {
            heap_target: 64 * 1024 * 1024, // 64MB default heap
            max_pause_time: Duration::from_millis(2),
            worker_threads: num_cpus::get().min(8),
            concurrent: true,
            generational: true,
            write_barrier: WriteBarrierType::Hybrid,
            trigger_strategy: TriggerStrategy::Adaptive,
        }
    }
} 