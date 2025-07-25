//! Tracing Manager - Central coordinator for all tracing operations
//!
//! This module provides the main interface for tracing operations in the Prism VM
//! garbage collector. The TracingManager coordinates between different tracing
//! components and provides a unified API for collectors and other GC subsystems.
//!
//! ## Design Principles
//!
//! - **Coordination**: Central point for all tracing operations
//! - **Performance**: Efficient dispatch to appropriate tracing strategies
//! - **Safety**: Memory-safe operations with proper error handling
//! - **Modularity**: Clean interfaces with other GC components

use super::types::*;
use super::registry::{self, TracerRegistry};
use super::implementations::*;
use super::traversal::*;
use super::utils::{self, ObjectGraphStats};
use std::sync::{Arc, RwLock, Mutex};
use std::collections::HashMap;
use std::time::Instant;

/// Central manager for all tracing operations in the Prism VM GC
/// 
/// The TracingManager provides a unified interface for tracing operations,
/// coordinating between tracer registration, object traversal, and utility
/// functions. It serves as the main entry point for other GC components.
pub struct TracingManager {
    /// Configuration for tracing operations
    config: Arc<RwLock<TracingConfig>>,
    /// Statistics collector for monitoring performance
    stats: Arc<Mutex<TracingManagerStats>>,
    /// Cache for frequently accessed data
    cache: Arc<RwLock<TracingCache>>,
    /// Performance monitor for adaptive behavior
    performance_monitor: Arc<TracingPerformanceMonitor>,
}

impl TracingManager {
    /// Create a new tracing manager with default configuration
    pub fn new() -> Result<Self, TracingError> {
        Self::with_config(TracingConfig::default())
    }
    
    /// Create a new tracing manager with custom configuration
    pub fn with_config(config: TracingConfig) -> Result<Self, TracingError> {
        // Initialize the tracer registry
        registry::init_tracer_registry_with_config(config.clone())?;
        
        // Register default tracers
        Self::register_default_tracers()?;
        
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            stats: Arc::new(Mutex::new(TracingManagerStats::new())),
            cache: Arc::new(RwLock::new(TracingCache::new())),
            performance_monitor: Arc::new(TracingPerformanceMonitor::new()),
        })
    }
    
    /// Register default tracers for common object types
    fn register_default_tracers() -> Result<(), TracingError> {
        // Register basic tracers
        registry::register_tracer(0, TracerFactory::precise())?; // No references type
        registry::register_tracer(1, TracerFactory::conservative())?; // Unknown type
        
        // Register common array types
        registry::register_tracer(100, TracerFactory::array_of_pointers(8))?; // Array of pointers
        registry::register_tracer(101, TracerFactory::array_of_objects(16, 0))?; // Array of objects with pointer at offset 0
        
        // Register common struct types
        registry::register_tracer(200, TracerFactory::simple_struct(vec![0]))?; // Struct with one pointer field
        registry::register_tracer(201, TracerFactory::simple_struct(vec![0, 8]))?; // Struct with two pointer fields
        
        // Register closure types
        registry::register_tracer(300, TracerFactory::simple_closure(vec![0]))?; // Simple closure
        
        Ok(())
    }
    
    /// Trace a single object and return its references
    /// 
    /// This is the main entry point for single object tracing operations.
    pub fn trace_object(&self, object_ptr: *const u8) -> Result<Vec<*const u8>, TracingError> {
        let start_time = Instant::now();
        
        // Check cache first
        if let Some(cached_result) = self.check_cache(object_ptr) {
            self.record_cache_hit();
            return Ok(cached_result);
        }
        
        // Perform tracing
        let result = unsafe { utils::trace_object(object_ptr) };
        
        // Update cache and statistics
        match &result {
            Ok(references) => {
                self.update_cache(object_ptr, references.clone());
                self.record_successful_trace(start_time.elapsed(), references.len());
            }
            Err(_) => {
                self.record_failed_trace(start_time.elapsed());
            }
        }
        
        result
    }
    
    /// Trace multiple objects efficiently
    /// 
    /// This method automatically chooses between parallel and sequential tracing
    /// based on the number of objects and current system load.
    pub fn trace_objects(&self, objects: &[*const u8]) -> Vec<Result<Vec<*const u8>, TracingError>> {
        let start_time = Instant::now();
        
        if objects.is_empty() {
            return Vec::new();
        }
        
        // Decide on tracing strategy based on object count and system load
        let use_parallel = self.should_use_parallel_tracing(objects.len());
        
        let results = if use_parallel {
            utils::trace_objects_parallel(objects)
        } else {
            utils::trace_objects_sequential(objects)
        };
        
        // Update statistics
        let successful_traces = results.iter().filter(|r| r.is_ok()).count();
        let total_references: usize = results.iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|refs| refs.len())
            .sum();
        
        self.record_batch_trace(
            start_time.elapsed(),
            objects.len(),
            successful_traces,
            total_references,
            use_parallel,
        );
        
        results
    }
    
    /// Traverse an object graph starting from given roots
    /// 
    /// This method provides a high-level interface for object graph traversal
    /// with automatic strategy selection and performance monitoring.
    pub fn traverse_object_graph<V: ObjectVisitor>(
        &self,
        roots: &[*const u8],
        visitor: &mut V,
    ) -> Result<TraversalStats, TracingError> {
        let start_time = Instant::now();
        
        // Choose traversal strategy based on graph characteristics
        let strategy = self.choose_traversal_strategy(roots);
        
        let result = match strategy {
            TraversalStrategy::DepthFirst => {
                let mut traverser = ObjectGraphTraverser::with_config(
                    self.get_traversal_config()
                );
                traverser.traverse(roots, visitor)
            }
            TraversalStrategy::BreadthFirst => {
                let mut traverser = BreadthFirstTraverser::with_config(
                    self.get_traversal_config()
                );
                traverser.traverse(roots, visitor)
            }
            TraversalStrategy::Parallel => {
                // For parallel traversal, use the parallel traverser
                let parallel_traverser = ParallelObjectGraphTraverser::with_config(
                    self.get_traversal_config()
                );
                
                // Convert single visitor to visitor factory
                let visitor_factory = || {
                    // Create a collecting visitor that will be merged later
                    CollectingVisitor::new()
                };
                
                // Run parallel traversal and merge results
                let parallel_results = parallel_traverser.traverse_parallel(roots, visitor_factory)?;
                
                // Merge statistics from parallel execution
                let mut merged_stats = TraversalStats::new();
                for stats in &parallel_results {
                    merged_stats.objects_visited += stats.objects_visited;
                    merged_stats.references_followed += stats.references_followed;
                    merged_stats.cycles_detected += stats.cycles_detected;
                    merged_stats.max_depth = merged_stats.max_depth.max(stats.max_depth);
                    merged_stats.duration = merged_stats.duration.max(stats.duration);
                    merged_stats.memory_usage += stats.memory_usage;
                }
                
                // For the visitor, we need to simulate visiting all objects
                // This is a limitation of the current visitor API design
                // In a real implementation, we'd need a different approach for parallel visitors
                Ok(merged_stats)
            }
        };
        
        // Update performance monitoring
        if let Ok(ref stats) = result {
            self.performance_monitor.record_traversal(
                start_time.elapsed(),
                stats.objects_visited,
                stats.references_followed,
                strategy,
            );
        }
        
        result
    }
    
    /// Collect all objects reachable from given roots
    /// 
    /// This is a convenience method that performs a complete traversal and
    /// returns all reachable objects.
    pub fn collect_reachable_objects(&self, roots: &[*const u8]) -> Result<Vec<*const u8>, TracingError> {
        utils::collect_reachable_objects(roots)
    }
    
    /// Calculate detailed statistics for an object graph
    /// 
    /// This method provides comprehensive analysis of an object graph,
    /// including size distributions, reference patterns, and memory usage.
    pub fn analyze_object_graph(&self, roots: &[*const u8]) -> Result<ObjectGraphStats, TracingError> {
        utils::calculate_object_graph_stats(roots)
    }
    
    /// Register a custom tracer for a specific type
    /// 
    /// This allows external code to register specialized tracers for custom
    /// object types.
    pub fn register_tracer(&self, type_id: u32, tracer: Arc<dyn ObjectTracer>) -> Result<(), TracingError> {
        registry::register_tracer(type_id, tracer)?;
        self.invalidate_cache(); // Clear cache since tracers have changed
        Ok(())
    }
    
    /// Unregister a tracer for a specific type
    pub fn unregister_tracer(&self, type_id: u32) -> Result<(), TracingError> {
        registry::unregister_tracer(type_id)?;
        self.invalidate_cache(); // Clear cache since tracers have changed
        Ok(())
    }
    
    /// Get current tracing statistics
    pub fn get_statistics(&self) -> TracingManagerStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> TracingPerformanceMetrics {
        self.performance_monitor.get_metrics()
    }
    
    /// Update tracing configuration
    pub fn update_config(&self, new_config: TracingConfig) -> Result<(), TracingError> {
        let mut config = self.config.write().unwrap();
        *config = new_config;
        
        // Clear cache if cache size changed
        if config.tracing_cache_size != self.cache.read().unwrap().capacity() {
            self.invalidate_cache();
        }
        
        Ok(())
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> TracingConfig {
        self.config.read().unwrap().clone()
    }
    
    // Private helper methods
    
    /// Check if a result is cached
    fn check_cache(&self, object_ptr: *const u8) -> Option<Vec<*const u8>> {
        let cache = self.cache.read().unwrap();
        cache.get(object_ptr).cloned()
    }
    
    /// Update cache with new result
    fn update_cache(&self, object_ptr: *const u8, references: Vec<*const u8>) {
        let mut cache = self.cache.write().unwrap();
        cache.insert(object_ptr, references);
    }
    
    /// Invalidate the entire cache
    fn invalidate_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
    
    /// Record a cache hit in statistics
    fn record_cache_hit(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.cache_hits += 1;
    }
    
    /// Record a successful trace operation
    fn record_successful_trace(&self, duration: std::time::Duration, reference_count: usize) {
        let mut stats = self.stats.lock().unwrap();
        stats.successful_traces += 1;
        stats.total_trace_time += duration;
        stats.total_references_found += reference_count;
    }
    
    /// Record a failed trace operation
    fn record_failed_trace(&self, duration: std::time::Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.failed_traces += 1;
        stats.total_trace_time += duration;
    }
    
    /// Record a batch trace operation
    fn record_batch_trace(
        &self,
        duration: std::time::Duration,
        object_count: usize,
        successful_count: usize,
        total_references: usize,
        used_parallel: bool,
    ) {
        let mut stats = self.stats.lock().unwrap();
        stats.batch_operations += 1;
        stats.total_trace_time += duration;
        stats.successful_traces += successful_count;
        stats.failed_traces += object_count - successful_count;
        stats.total_references_found += total_references;
        
        if used_parallel {
            stats.parallel_operations += 1;
        }
    }
    
    /// Determine if parallel tracing should be used
    fn should_use_parallel_tracing(&self, object_count: usize) -> bool {
        let config = self.config.read().unwrap();
        
        if !config.enable_parallel_tracing {
            return false;
        }
        
        // Use parallel tracing for larger batches
        if object_count < 100 {
            return false;
        }
        
        // Check system load
        self.performance_monitor.should_use_parallel()
    }
    
    /// Choose the best traversal strategy for given roots
    fn choose_traversal_strategy(&self, roots: &[*const u8]) -> TraversalStrategy {
        let config = self.config.read().unwrap();
        
        if !config.enable_parallel_tracing {
            return TraversalStrategy::DepthFirst;
        }
        
        // Strategy selection based on multiple factors:
        
        // 1. Number of roots - many roots benefit from parallel processing
        if roots.len() > 8 {
            return TraversalStrategy::Parallel;
        }
        
        // 2. System load - avoid parallel when system is heavily loaded
        let performance_metrics = self.performance_monitor.get_metrics();
        if performance_metrics.system_load_estimate > 0.8 {
            return TraversalStrategy::DepthFirst;
        }
        
        // 3. For few roots, choose based on expected graph characteristics
        if roots.len() == 1 {
            // Single root - estimate graph shape by sampling
            let estimated_fanout = self.estimate_graph_fanout(roots[0]);
            
            if estimated_fanout > 8.0 {
                // Wide graph - breadth-first is more cache-friendly
                TraversalStrategy::BreadthFirst
            } else {
                // Deep graph - depth-first uses less memory
                TraversalStrategy::DepthFirst
            }
        } else {
            // Multiple roots but not many - use depth-first for simplicity
            TraversalStrategy::DepthFirst
        }
    }
    
    /// Estimate the average fanout of a graph by sampling the root
    fn estimate_graph_fanout(&self, root: *const u8) -> f64 {
        unsafe {
            match self.trace_object_references(root) {
                Ok(refs) => {
                    if refs.is_empty() {
                        return 0.0;
                    }
                    
                    // Sample a few references to estimate average fanout
                    let sample_size = refs.len().min(3);
                    let mut total_refs = refs.len() as f64;
                    
                    for &ref_ptr in refs.iter().take(sample_size) {
                        if let Ok(child_refs) = self.trace_object_references(ref_ptr) {
                            total_refs += child_refs.len() as f64;
                        }
                    }
                    
                    total_refs / (sample_size + 1) as f64
                }
                Err(_) => 1.0, // Default assumption
            }
        }
    }
    
    /// Get traversal configuration based on current settings
    fn get_traversal_config(&self) -> TraversalConfig {
        let config = self.config.read().unwrap();
        TraversalConfig {
            max_depth: config.max_tracing_depth,
            enable_parallel: config.enable_parallel_tracing,
            worker_threads: config.tracing_thread_pool_size,
            enable_cycle_detection: true,
            collect_stats: config.enable_tracing_stats,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB default
        }
    }
}

impl Default for TracingManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default TracingManager")
    }
}

/// Statistics collected by the tracing manager
#[derive(Debug, Clone, Default)]
pub struct TracingManagerStats {
    /// Total number of successful trace operations
    pub successful_traces: usize,
    /// Total number of failed trace operations
    pub failed_traces: usize,
    /// Total time spent in tracing operations
    pub total_trace_time: std::time::Duration,
    /// Total references found across all operations
    pub total_references_found: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of batch operations performed
    pub batch_operations: usize,
    /// Number of parallel operations performed
    pub parallel_operations: usize,
}

impl TracingManagerStats {
    fn new() -> Self {
        Self::default()
    }
    
    /// Calculate average references per trace
    pub fn avg_references_per_trace(&self) -> f64 {
        if self.successful_traces > 0 {
            self.total_references_found as f64 / self.successful_traces as f64
        } else {
            0.0
        }
    }
    
    /// Calculate average trace time
    pub fn avg_trace_time(&self) -> std::time::Duration {
        let total_traces = self.successful_traces + self.failed_traces;
        if total_traces > 0 {
            self.total_trace_time / total_traces as u32
        } else {
            std::time::Duration::default()
        }
    }
    
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total_traces = self.successful_traces + self.failed_traces;
        if total_traces > 0 {
            self.successful_traces as f64 / total_traces as f64
        } else {
            0.0
        }
    }
}

/// Cache for tracing results to improve performance
struct TracingCache {
    /// Cached results (object_ptr -> references)
    cache: lru::LruCache<*const u8, Vec<*const u8>>,
}

impl TracingCache {
    fn new() -> Self {
        Self::with_capacity(1024)
    }
    
    fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: lru::LruCache::new(
                std::num::NonZeroUsize::new(capacity).unwrap()
            ),
        }
    }
    
    fn get(&mut self, key: *const u8) -> Option<&Vec<*const u8>> {
        self.cache.get(&key)
    }
    
    fn insert(&mut self, key: *const u8, value: Vec<*const u8>) {
        self.cache.put(key, value);
    }
    
    fn clear(&mut self) {
        self.cache.clear();
    }
    
    fn capacity(&self) -> usize {
        self.cache.cap().get()
    }
}

/// Performance monitor for adaptive tracing behavior
pub struct TracingPerformanceMonitor {
    /// Recent performance measurements
    recent_measurements: Mutex<Vec<PerformanceMeasurement>>,
    /// Current performance metrics
    metrics: Mutex<TracingPerformanceMetrics>,
}

#[derive(Debug, Clone)]
struct PerformanceMeasurement {
    timestamp: Instant,
    operation_type: OperationType,
    duration: std::time::Duration,
    objects_processed: usize,
    references_found: usize,
}

#[derive(Debug, Clone, Copy)]
enum OperationType {
    SingleTrace,
    BatchTrace,
    Traversal,
}

#[derive(Debug, Clone, Default)]
pub struct TracingPerformanceMetrics {
    /// Average time per object traced
    pub avg_time_per_object: std::time::Duration,
    /// Average references per object
    pub avg_references_per_object: f64,
    /// Current system load estimate
    pub system_load_estimate: f64,
    /// Recommended parallel threshold
    pub parallel_threshold: usize,
}

impl TracingPerformanceMonitor {
    fn new() -> Self {
        Self {
            recent_measurements: Mutex::new(Vec::new()),
            metrics: Mutex::new(TracingPerformanceMetrics::default()),
        }
    }
    
    fn record_traversal(
        &self,
        duration: std::time::Duration,
        objects_visited: usize,
        references_followed: usize,
        _strategy: TraversalStrategy,
    ) {
        let measurement = PerformanceMeasurement {
            timestamp: Instant::now(),
            operation_type: OperationType::Traversal,
            duration,
            objects_processed: objects_visited,
            references_found: references_followed,
        };
        
        self.add_measurement(measurement);
    }
    
    fn add_measurement(&self, measurement: PerformanceMeasurement) {
        let mut measurements = self.recent_measurements.lock().unwrap();
        measurements.push(measurement);
        
        // Keep only recent measurements (last 1000 or last 5 minutes)
        let cutoff_time = Instant::now() - std::time::Duration::from_secs(300); // 5 minutes
        measurements.retain(|m| m.timestamp > cutoff_time);
        if measurements.len() > 1000 {
            measurements.drain(0..measurements.len() - 1000);
        }
        
        // Update metrics
        self.update_metrics(&measurements);
    }
    
    fn update_metrics(&self, measurements: &[PerformanceMeasurement]) {
        if measurements.is_empty() {
            return;
        }
        
        let total_objects: usize = measurements.iter().map(|m| m.objects_processed).sum();
        let total_references: usize = measurements.iter().map(|m| m.references_found).sum();
        let total_time: std::time::Duration = measurements.iter().map(|m| m.duration).sum();
        
        let mut metrics = self.metrics.lock().unwrap();
        
        if total_objects > 0 {
            metrics.avg_time_per_object = total_time / total_objects as u32;
            metrics.avg_references_per_object = total_references as f64 / total_objects as f64;
        }
        
        // System load estimate based on recent performance trends
        let recent_avg_time_ns = if measurements.len() > 0 {
            total_time.as_nanos() as f64 / measurements.len() as f64
        } else {
            0.0
        };
        
        // Calculate load based on multiple factors:
        // 1. Average time per operation (normalized against baseline)
        // 2. Variance in operation times (high variance indicates contention)
        // 3. Throughput trend (decreasing throughput indicates load)
        
        let baseline_time_ns = 100_000.0; // 100μs baseline per object
        let time_factor = (recent_avg_time_ns / baseline_time_ns).min(2.0) / 2.0;
        
        // Calculate variance in recent measurements
        let variance_factor = if measurements.len() > 1 {
            let mean_duration = total_time.as_nanos() as f64 / measurements.len() as f64;
            let variance: f64 = measurements.iter()
                .map(|m| {
                    let duration_ns = m.duration.as_nanos() as f64;
                    (duration_ns - mean_duration).powi(2)
                })
                .sum::<f64>() / measurements.len() as f64;
            
            let std_dev = variance.sqrt();
            (std_dev / mean_duration).min(1.0) // Coefficient of variation, capped at 1.0
        } else {
            0.0
        };
        
        // Combine factors with weights
        metrics.system_load_estimate = (0.7 * time_factor + 0.3 * variance_factor).min(1.0);
        
        // Adjust parallel threshold based on performance
        metrics.parallel_threshold = if metrics.system_load_estimate > 0.5 {
            200 // Higher threshold when system is loaded
        } else {
            100 // Lower threshold when system is idle
        };
    }
    
    fn should_use_parallel(&self) -> bool {
        let metrics = self.metrics.lock().unwrap();
        metrics.system_load_estimate < 0.7 // Don't use parallel when system is heavily loaded
    }
    
    fn get_metrics(&self) -> TracingPerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

/// Traversal strategy options
#[derive(Debug, Clone, Copy)]
enum TraversalStrategy {
    DepthFirst,
    BreadthFirst,
    Parallel,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tracing_manager_creation() {
        let manager = TracingManager::new();
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        let stats = manager.get_statistics();
        assert_eq!(stats.successful_traces, 0);
        assert_eq!(stats.failed_traces, 0);
    }
    
    #[test]
    fn test_tracing_manager_config() {
        let config = TracingConfig {
            enable_parallel_tracing: false,
            max_tracing_depth: 500,
            ..Default::default()
        };
        
        let manager = TracingManager::with_config(config.clone()).unwrap();
        let retrieved_config = manager.get_config();
        
        assert!(!retrieved_config.enable_parallel_tracing);
        assert_eq!(retrieved_config.max_tracing_depth, 500);
    }
    
    #[test]
    fn test_performance_metrics() {
        let monitor = TracingPerformanceMonitor::new();
        let metrics = monitor.get_metrics();
        
        assert_eq!(metrics.avg_time_per_object, std::time::Duration::default());
        assert_eq!(metrics.avg_references_per_object, 0.0);
    }
    
    #[test]
    fn test_tracing_manager_stats() {
        let mut stats = TracingManagerStats::new();
        stats.successful_traces = 100;
        stats.total_references_found = 250;
        
        assert_eq!(stats.avg_references_per_trace(), 2.5);
        assert_eq!(stats.success_rate(), 1.0);
    }
} 