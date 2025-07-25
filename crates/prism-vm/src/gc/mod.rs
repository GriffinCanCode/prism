use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub mod collectors;
pub mod allocators;
pub mod barriers;
pub mod heap;
pub mod roots;
pub mod tracing;

// Re-export key types from the modular barriers subsystem
pub use barriers::{
    WriteBarrierType, ObjectColor, BarrierConfig, BarrierStats,
    BarrierSubsystem, BarrierFactory
};

// Re-export key types from the modular tracing subsystem
pub use tracing::{
    ObjectTracer, TracingConfig, TracingError, TracingManager,
    init_tracing_subsystem, get_tracing_manager, trace_single_object,
    collect_all_reachable_objects, analyze_graph
};

// Re-export key types for convenience
pub use collectors::*;
pub use allocators::*;
pub use barriers::*;
pub use heap::*;
pub use roots::*;

/// Core garbage collection trait that all collectors must implement
/// 
/// This trait ensures memory safety through:
/// - Controlled allocation with bounds checking
/// - Automatic collection triggering
/// - Safe root management
/// - Thread-safe operations
pub trait GarbageCollector: Send + Sync {
    /// Allocate memory for an object of the given size and alignment
    /// 
    /// Safety: Returns properly aligned memory or None if allocation fails
    /// The returned pointer is valid until the next GC cycle
    fn allocate(&self, size: usize, align: usize) -> Option<*mut u8>;
    
    /// Trigger a garbage collection cycle
    /// 
    /// Safety: This operation may pause other threads and invalidate
    /// previously allocated objects that are no longer reachable
    fn collect(&self) -> CollectionStats;
    
    /// Check if a collection should be triggered based on current state
    /// 
    /// This is safe to call from any thread and provides a hint for
    /// when collection might be beneficial
    fn should_collect(&self) -> bool;
    
    /// Get current heap statistics
    /// 
    /// Thread-safe read-only operation that provides current memory usage
    fn heap_stats(&self) -> HeapStats;
    
    /// Set the garbage collection strategy/configuration
    /// 
    /// Safety: Configuration changes may affect allocation behavior
    /// and should be done when no allocations are in progress
    fn configure(&self, config: GcConfig);
    
    /// Register a root object that should never be collected
    /// 
    /// Safety: The pointer must remain valid for the lifetime of the root
    /// registration. Roots keep objects alive across GC cycles.
    fn register_root(&self, ptr: *const u8);
    
    /// Unregister a previously registered root
    /// 
    /// Safety: Only call with pointers that were previously registered
    /// as roots. After unregistration, the object may be collected.
    fn unregister_root(&self, ptr: *const u8);
    
    /// Mark an object as reachable during tracing
    /// 
    /// Safety: The pointer must point to a valid object header
    /// This is typically called by the collector itself during marking
    fn mark_object(&self, ptr: *const u8);
    
    /// Check if an object is currently marked as live
    /// 
    /// Thread-safe read operation that returns the current mark state
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

/// Configuration for garbage collection behavior with safety constraints
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Target heap size before triggering collection
    /// Must be > 0 to prevent infinite allocation
    pub heap_target: usize,
    /// Maximum pause time goal (for low-latency collectors)
    /// Must be > 0 to ensure progress
    pub max_pause_time: Duration,
    /// Number of worker threads for parallel collection
    /// Must be > 0 and <= available CPU cores
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

impl GcConfig {
    /// Validate configuration parameters for safety
    pub fn validate(&self) -> Result<(), String> {
        if self.heap_target == 0 {
            return Err("heap_target must be greater than 0".to_string());
        }
        
        if self.max_pause_time.is_zero() {
            return Err("max_pause_time must be greater than 0".to_string());
        }
        
        if self.worker_threads == 0 {
            return Err("worker_threads must be greater than 0".to_string());
        }
        
        let max_threads = num_cpus::get() * 2; // Allow some oversubscription
        if self.worker_threads > max_threads {
            return Err(format!(
                "worker_threads ({}) exceeds recommended maximum ({})",
                self.worker_threads, max_threads
            ));
        }
        
        Ok(())
    }
    
    /// Create a configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            heap_target: 32 * 1024 * 1024, // 32MB
            max_pause_time: Duration::from_millis(1), // 1ms target
            worker_threads: num_cpus::get().min(4), // Limited parallelism
            concurrent: true,
            generational: true,
            write_barrier: WriteBarrierType::Hybrid,
            trigger_strategy: TriggerStrategy::Adaptive,
        }
    }
    
    /// Create a configuration optimized for high throughput
    pub fn high_throughput() -> Self {
        Self {
            heap_target: 256 * 1024 * 1024, // 256MB
            max_pause_time: Duration::from_millis(10), // 10ms acceptable
            worker_threads: num_cpus::get(), // Use all cores
            concurrent: true,
            generational: true,
            write_barrier: WriteBarrierType::Hybrid,
            trigger_strategy: TriggerStrategy::HeapSize(200 * 1024 * 1024),
        }
    }
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            heap_target: 64 * 1024 * 1024, // 64MB default
            max_pause_time: Duration::from_millis(5), // 5ms target
            worker_threads: num_cpus::get().min(8), // Reasonable default
            concurrent: true,
            generational: false, // Disable by default for simplicity
            write_barrier: WriteBarrierType::Hybrid,
            trigger_strategy: TriggerStrategy::Adaptive,
        }
    }
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

/// Object metadata stored in heap headers with safety information
#[derive(Debug, Clone, Copy)]
pub struct ObjectHeader {
    /// Size of the object in bytes (must be > 0)
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

impl ObjectHeader {
    /// Create a new object header
    pub fn new(size: usize, type_id: u32) -> Self {
        Self {
            size,
            type_id,
            mark_bits: 0,
            generation: 0,
            ref_count: 0,
        }
    }
    
    /// Get the object's color for tri-color marking
    pub fn get_color(&self) -> ObjectColor {
        match self.mark_bits & 0x3 {
            0 => ObjectColor::White,
            1 => ObjectColor::Gray,
            2 => ObjectColor::Black,
            _ => ObjectColor::White, // Default fallback
        }
    }
    
    /// Set the object's color for tri-color marking
    pub fn set_color(&mut self, color: ObjectColor) {
        self.mark_bits = (self.mark_bits & !0x3) | (color as u8);
    }
    
    /// Check if the object is marked as live
    pub fn is_marked(&self) -> bool {
        self.get_color() != ObjectColor::White
    }
    
    /// Mark the object as live
    pub fn mark(&mut self) {
        self.set_color(ObjectColor::Black);
    }
    
    /// Clear all marks (reset to white)
    pub fn clear_mark(&mut self) {
        self.set_color(ObjectColor::White);
    }
}

/// Safe wrapper for GC operations that ensures proper lifecycle management
pub struct SafeGc<T: GarbageCollector> {
    collector: Arc<T>,
    /// Track active allocations to prevent use-after-free
    active_allocations: Arc<std::sync::RwLock<std::collections::HashSet<*const u8>>>,
}

impl<T: GarbageCollector> SafeGc<T> {
    /// Create a new safe GC wrapper
    pub fn new(collector: T) -> Self {
        Self {
            collector: Arc::new(collector),
            active_allocations: Arc::new(std::sync::RwLock::new(std::collections::HashSet::new())),
        }
    }
    
    /// Safely allocate memory with automatic tracking
    pub fn allocate(&self, size: usize, align: usize) -> Option<SafePtr> {
        // Validate parameters
        if size == 0 || align == 0 || !align.is_power_of_two() {
            return None;
        }
        
        // Perform allocation
        let ptr = self.collector.allocate(size, align)?;
        let ptr_const = ptr as *const u8;
        
        // Track the allocation
        {
            let mut allocations = self.active_allocations.write().unwrap();
            allocations.insert(ptr_const);
        }
        
        Some(SafePtr {
            ptr: ptr_const,
            size,
            allocations: self.active_allocations.clone(),
        })
    }
    
    /// Safely trigger garbage collection
    pub fn collect(&self) -> CollectionStats {
        let stats = self.collector.collect();
        
        // After collection, remove deallocated objects from tracking
        // This is a simplified approach - a real implementation would need
        // more sophisticated tracking integration with the collector
        
        stats
    }
    
    /// Get heap statistics safely
    pub fn heap_stats(&self) -> HeapStats {
        self.collector.heap_stats()
    }
    
    /// Register a root safely
    pub fn register_root(&self, ptr: *const u8) -> Result<RootHandle, String> {
        // Validate that the pointer is from a tracked allocation
        {
            let allocations = self.active_allocations.read().unwrap();
            if !allocations.contains(&ptr) {
                return Err("Cannot register root for untracked allocation".to_string());
            }
        }
        
        self.collector.register_root(ptr);
        
        Ok(RootHandle {
            ptr,
            collector: self.collector.clone(),
        })
    }
}

/// Safe pointer wrapper that prevents use-after-free
pub struct SafePtr {
    ptr: *const u8,
    size: usize,
    allocations: Arc<std::sync::RwLock<std::collections::HashSet<*const u8>>>,
}

impl SafePtr {
    /// Get the raw pointer (unsafe operation)
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    
    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Check if this pointer is still valid
    pub fn is_valid(&self) -> bool {
        let allocations = self.allocations.read().unwrap();
        allocations.contains(&self.ptr)
    }
    
    /// Convert to a typed pointer (unsafe)
    pub unsafe fn cast<U>(&self) -> *const U {
        self.ptr as *const U
    }
}

impl Drop for SafePtr {
    fn drop(&mut self) {
        // Remove from tracking when the safe pointer is dropped
        let mut allocations = self.allocations.write().unwrap();
        allocations.remove(&self.ptr);
    }
}

/// Handle for a registered root that automatically unregisters on drop
pub struct RootHandle {
    ptr: *const u8,
    collector: Arc<dyn GarbageCollector>,
}

impl Drop for RootHandle {
    fn drop(&mut self) {
        self.collector.unregister_root(self.ptr);
    }
}

/// Factory for creating different types of garbage collectors with safety validation
pub struct GcFactory;

impl GcFactory {
    /// Create a tri-color mark-sweep collector optimized for low latency
    pub fn create_tricolor_collector(config: GcConfig) -> Result<Box<dyn GarbageCollector>, String> {
        config.validate()?;
        Ok(Box::new(collectors::TriColorCollector::new(config)))
    }
    
    /// Create a generational collector optimized for high throughput
    pub fn create_generational_collector(config: GcConfig) -> Result<Box<dyn GarbageCollector>, String> {
        config.validate()?;
        Ok(Box::new(collectors::GenerationalCollector::new(config)))
    }
    
    /// Create a concurrent collector with minimal pause times
    pub fn create_concurrent_collector(config: GcConfig) -> Result<Box<dyn GarbageCollector>, String> {
        config.validate()?;
        Ok(Box::new(collectors::ConcurrentCollector::new(config)))
    }
    
    /// Create a hybrid collector combining multiple strategies
    pub fn create_hybrid_collector(config: GcConfig) -> Result<Box<dyn GarbageCollector>, String> {
        config.validate()?;
        Ok(Box::new(collectors::HybridCollector::new(config)))
    }
    
    /// Create the optimal collector for Prism's specific use case
    pub fn create_prism_optimized_collector(config: GcConfig) -> Result<Box<dyn GarbageCollector>, String> {
        // Validate and optimize configuration for Prism VM
        let mut optimized_config = config;
        optimized_config.validate()?;
        
        // For Prism VM, we want:
        // 1. Low pause times for interactive execution
        // 2. Good throughput for compilation workloads  
        // 3. Efficient handling of short-lived objects (AST nodes, temporaries)
        // 4. Support for long-lived objects (compiled code, caches)
        
        optimized_config.concurrent = true;
        optimized_config.generational = true;
        optimized_config.write_barrier = WriteBarrierType::Hybrid;
        optimized_config.trigger_strategy = TriggerStrategy::Adaptive;
        optimized_config.max_pause_time = Duration::from_millis(1); // 1ms target
        
        Ok(Box::new(collectors::PrismCollector::new(optimized_config)))
    }
    
    /// Create a safe GC wrapper around any collector
    pub fn create_safe_collector<T: GarbageCollector + 'static>(collector: T) -> SafeGc<T> {
        SafeGc::new(collector)
    }
}

/// Default configuration optimized for Prism's workload characteristics
impl Default for GcFactory {
    fn default() -> Self {
        GcFactory
    }
}

/// Thread-safe collector registry for managing multiple collectors
pub struct CollectorRegistry {
    collectors: std::sync::RwLock<std::collections::HashMap<String, Arc<dyn GarbageCollector>>>,
}

impl CollectorRegistry {
    /// Create a new collector registry
    pub fn new() -> Self {
        Self {
            collectors: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }
    
    /// Register a collector with a given name
    pub fn register(&self, name: String, collector: Arc<dyn GarbageCollector>) {
        let mut collectors = self.collectors.write().unwrap();
        collectors.insert(name, collector);
    }
    
    /// Get a collector by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn GarbageCollector>> {
        let collectors = self.collectors.read().unwrap();
        collectors.get(name).cloned()
    }
    
    /// Remove a collector by name
    pub fn unregister(&self, name: &str) -> Option<Arc<dyn GarbageCollector>> {
        let mut collectors = self.collectors.write().unwrap();
        collectors.remove(name)
    }
    
    /// List all registered collector names
    pub fn list_collectors(&self) -> Vec<String> {
        let collectors = self.collectors.read().unwrap();
        collectors.keys().cloned().collect()
    }
}

impl Default for CollectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the entire GC subsystem
/// 
/// This function initializes all GC components including tracing, barriers,
/// and other subsystems. It should be called once during application startup.
pub fn init_gc_subsystem() -> Result<(), String> {
    init_gc_subsystem_with_config(GcConfig::default())
}

/// Initialize the GC subsystem with custom configuration
pub fn init_gc_subsystem_with_config(config: GcConfig) -> Result<(), String> {
    // Validate configuration
    config.validate()?;
    
    // Initialize tracing subsystem
    let tracing_config = TracingConfig {
        enable_parallel_tracing: config.concurrent,
        max_tracing_depth: 1000,
        enable_conservative_scanning: false,
        tracing_thread_pool_size: config.worker_threads,
        enable_tracing_stats: true,
        tracing_cache_size: 1024,
    };
    
    init_tracing_subsystem_with_config(tracing_config)
        .map_err(|e| format!("Failed to initialize tracing subsystem: {}", e))?;
    
    // Initialize other subsystems as needed
    // barriers::init_barriers_subsystem()?;
    // roots::init_roots_subsystem()?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gc_config_validation() {
        let mut config = GcConfig::default();
        assert!(config.validate().is_ok());
        
        config.heap_target = 0;
        assert!(config.validate().is_err());
        
        config.heap_target = 1024;
        config.worker_threads = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_object_header() {
        let mut header = ObjectHeader::new(64, 42);
        assert_eq!(header.size, 64);
        assert_eq!(header.type_id, 42);
        assert_eq!(header.get_color(), ObjectColor::White);
        
        header.set_color(ObjectColor::Gray);
        assert_eq!(header.get_color(), ObjectColor::Gray);
        assert!(header.is_marked());
        
        header.clear_mark();
        assert_eq!(header.get_color(), ObjectColor::White);
        assert!(!header.is_marked());
    }
    
    #[test]
    fn test_collector_registry() {
        let registry = CollectorRegistry::new();
        assert_eq!(registry.list_collectors().len(), 0);
        
        // This would require a real collector implementation
        // let collector = Arc::new(DummyCollector::new());
        // registry.register("test".to_string(), collector.clone());
        // assert_eq!(registry.list_collectors().len(), 1);
        // assert!(registry.get("test").is_some());
    }
    
    #[test]
    fn test_gc_subsystem_initialization() {
        let result = init_gc_subsystem();
        assert!(result.is_ok());
    }
} 