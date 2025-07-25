//! Object graph traversal algorithms with cycle detection and visitor patterns
//!
//! This module provides efficient algorithms for traversing object graphs in the
//! Prism VM garbage collector. It includes cycle detection, visitor patterns,
//! and parallel traversal capabilities.
//!
//! ## Design Principles
//!
//! - **Cycle Safety**: Robust cycle detection prevents infinite loops
//! - **Performance**: Efficient traversal with minimal overhead
//! - **Flexibility**: Multiple traversal strategies for different use cases
//! - **Parallelism**: Support for parallel traversal of large object graphs

use super::types::*;
use super::registry;
use std::collections::{HashSet, VecDeque, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Visitor trait for object graph traversal
/// 
/// Implementors of this trait can be used with the graph traversal algorithms
/// to perform custom operations on each visited object.
pub trait ObjectVisitor {
    /// Visit an object during traversal
    /// 
    /// # Parameters
    /// 
    /// - `object_ptr`: Pointer to the object being visited
    /// - `object_size`: Size of the object in bytes
    /// - `depth`: Current traversal depth
    /// 
    /// # Returns
    /// 
    /// Whether to continue traversing from this object
    fn visit_object(&mut self, object_ptr: *const u8, object_size: usize, depth: usize) -> bool;
    
    /// Called when entering a reference cycle
    /// 
    /// This is called when the traverser detects that it's about to visit
    /// an object that's already in the current traversal path.
    fn enter_cycle(&mut self, cycle_root: *const u8) {
        let _ = cycle_root; // Default implementation does nothing
    }
    
    /// Called when exiting a reference cycle
    /// 
    /// This is called when the traverser finishes processing a cycle.
    fn exit_cycle(&mut self) {
        // Default implementation does nothing
    }
    
    /// Called when traversal is complete
    /// 
    /// This allows visitors to perform cleanup or final processing.
    fn traversal_complete(&mut self, stats: &TraversalStats) {
        let _ = stats; // Default implementation does nothing
    }
}

/// Statistics collected during object graph traversal
#[derive(Debug, Clone)]
pub struct TraversalStats {
    /// Total number of objects visited
    pub objects_visited: usize,
    /// Total number of references followed
    pub references_followed: usize,
    /// Number of cycles detected
    pub cycles_detected: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Time taken for traversal
    pub duration: std::time::Duration,
    /// Memory usage during traversal
    pub memory_usage: usize,
}

impl TraversalStats {
    /// Create new empty traversal statistics
    pub fn new() -> Self {
        Self {
            objects_visited: 0,
            references_followed: 0,
            cycles_detected: 0,
            max_depth: 0,
            duration: std::time::Duration::default(),
            memory_usage: 0,
        }
    }
    
    /// Calculate average references per object
    pub fn avg_references_per_object(&self) -> f64 {
        if self.objects_visited > 0 {
            self.references_followed as f64 / self.objects_visited as f64
        } else {
            0.0
        }
    }
}

impl Default for TraversalStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for object graph traversal
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum traversal depth (for cycle prevention)
    pub max_depth: usize,
    /// Enable parallel traversal for large graphs
    pub enable_parallel: bool,
    /// Number of worker threads for parallel traversal
    pub worker_threads: usize,
    /// Enable cycle detection
    pub enable_cycle_detection: bool,
    /// Enable traversal statistics collection
    pub collect_stats: bool,
    /// Maximum memory usage before stopping traversal
    pub max_memory_usage: usize,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 1000,
            enable_parallel: true,
            worker_threads: num_cpus::get().min(8),
            enable_cycle_detection: true,
            collect_stats: true,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Main object graph traverser with cycle detection
/// 
/// This traverser provides safe traversal of object graphs with automatic
/// cycle detection and support for custom visitor patterns.
pub struct ObjectGraphTraverser {
    /// Set of visited objects (for cycle detection)
    visited: HashSet<*const u8>,
    /// Set of objects currently being visited (for cycle detection)
    visiting: HashSet<*const u8>,
    /// Current traversal path (for cycle reporting)
    current_path: Vec<*const u8>,
    /// Traversal configuration
    config: TraversalConfig,
    /// Statistics collector
    stats: TraversalStats,
    /// Start time of current traversal
    start_time: Option<Instant>,
}

impl ObjectGraphTraverser {
    /// Create a new object graph traverser with default configuration
    pub fn new() -> Self {
        Self::with_config(TraversalConfig::default())
    }
    
    /// Create a new object graph traverser with custom configuration
    pub fn with_config(config: TraversalConfig) -> Self {
        Self {
            visited: HashSet::new(),
            visiting: HashSet::new(),
            current_path: Vec::new(),
            config,
            stats: TraversalStats::new(),
            start_time: None,
        }
    }
    
    /// Traverse object graph starting from given roots
    /// 
    /// This is the main entry point for graph traversal. It visits all objects
    /// reachable from the given roots using the provided visitor.
    pub fn traverse<V: ObjectVisitor>(
        &mut self, 
        roots: &[*const u8], 
        visitor: &mut V
    ) -> Result<TraversalStats, TracingError> {
        self.start_time = Some(Instant::now());
        self.reset_state();
        
        // Traverse from each root
        for &root in roots {
            if let Err(e) = self.traverse_from_root(root, visitor) {
                return Err(e);
            }
        }
        
        // Finalize statistics
        if let Some(start_time) = self.start_time {
            self.stats.duration = start_time.elapsed();
        }
        
        // Notify visitor of completion
        visitor.traversal_complete(&self.stats);
        
        Ok(self.stats.clone())
    }
    
    /// Traverse from a single root object
    fn traverse_from_root<V: ObjectVisitor>(
        &mut self, 
        root: *const u8, 
        visitor: &mut V
    ) -> Result<(), TracingError> {
        if self.visited.contains(&root) {
            return Ok(());
        }
        
        if self.config.enable_cycle_detection && self.visiting.contains(&root) {
            // Cycle detected
            self.stats.cycles_detected += 1;
            visitor.enter_cycle(root);
            return Ok(());
        }
        
        // Check depth limit
        if self.current_path.len() >= self.config.max_depth {
            return Err(TracingError::MaxDepthExceeded(self.config.max_depth));
        }
        
        // Check memory usage limit
        let current_memory = self.estimate_memory_usage();
        if current_memory > self.config.max_memory_usage {
            return Err(TracingError::Generic(
                format!("Memory usage limit exceeded: {} bytes", current_memory)
            ));
        }
        
        // Mark as visiting
        if self.config.enable_cycle_detection {
            self.visiting.insert(root);
        }
        self.current_path.push(root);
        
        // Get object size
        let object_size = unsafe {
            let header = &*(root as *const ObjectHeader);
            header.size
        };
        
        // Visit the object
        let should_continue = visitor.visit_object(root, object_size, self.current_path.len());
        self.stats.objects_visited += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.current_path.len());
        
        if should_continue {
            // Trace object references and continue traversal
            let references = self.trace_object_references(root)?;
            self.stats.references_followed += references.len();
            
            for ref_ptr in references {
                self.traverse_from_root(ref_ptr, visitor)?;
            }
        }
        
        // Mark as visited and remove from visiting set
        self.current_path.pop();
        if self.config.enable_cycle_detection {
            self.visiting.remove(&root);
        }
        self.visited.insert(root);
        
        // Check if we've exited all cycles
        if self.visiting.is_empty() {
            visitor.exit_cycle();
        }
        
        Ok(())
    }
    
    /// Trace references from an object
    fn trace_object_references(&self, object_ptr: *const u8) -> Result<Vec<*const u8>, TracingError> {
        unsafe {
            let header = &*(object_ptr as *const ObjectHeader);
            let tracer = registry::get_tracer(header.type_id);
            Ok(tracer.trace_references(object_ptr, header.size))
        }
    }
    
    /// Reset traverser state for a new traversal
    fn reset_state(&mut self) {
        self.visited.clear();
        self.visiting.clear();
        self.current_path.clear();
        self.stats = TraversalStats::new();
        self.start_time = None;
    }
    
    /// Estimate current memory usage of the traverser
    fn estimate_memory_usage(&self) -> usize {
        let visited_size = self.visited.len() * std::mem::size_of::<*const u8>();
        let visiting_size = self.visiting.len() * std::mem::size_of::<*const u8>();
        let path_size = self.current_path.len() * std::mem::size_of::<*const u8>();
        
        visited_size + visiting_size + path_size
    }
    
    /// Get current traversal statistics
    pub fn stats(&self) -> &TraversalStats {
        &self.stats
    }
    
    /// Reset the traverser for reuse
    pub fn reset(&mut self) {
        self.reset_state();
    }
}

impl Default for ObjectGraphTraverser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel object graph traverser for large graphs
/// 
/// This traverser uses multiple worker threads to traverse large object graphs
/// in parallel, providing better performance for complex object hierarchies.
pub struct ParallelObjectGraphTraverser {
    /// Configuration for parallel traversal
    config: TraversalConfig,
    /// Worker thread pool
    thread_pool: Option<rayon::ThreadPool>,
}

impl ParallelObjectGraphTraverser {
    /// Create a new parallel traverser
    pub fn new() -> Self {
        Self::with_config(TraversalConfig::default())
    }
    
    /// Create a new parallel traverser with custom configuration
    pub fn with_config(config: TraversalConfig) -> Self {
        let thread_pool = if config.enable_parallel {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.worker_threads)
                .build()
                .ok()
        } else {
            None
        };
        
        Self {
            config,
            thread_pool,
        }
    }
    
    /// Traverse object graph in parallel
    pub fn traverse_parallel<V: ObjectVisitor + Send>(
        &self,
        roots: &[*const u8],
        visitor_factory: impl Fn() -> V + Send + Sync,
    ) -> Result<Vec<TraversalStats>, TracingError> {
        if let Some(ref pool) = self.thread_pool {
            use rayon::prelude::*;
            
            pool.install(|| {
                roots.par_iter()
                    .map(|&root| {
                        let mut traverser = ObjectGraphTraverser::with_config(self.config.clone());
                        let mut visitor = visitor_factory();
                        traverser.traverse(&[root], &mut visitor)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
        } else {
            // Fallback to sequential traversal
            let mut results = Vec::new();
            for &root in roots {
                let mut traverser = ObjectGraphTraverser::with_config(self.config.clone());
                let mut visitor = visitor_factory();
                results.push(traverser.traverse(&[root], &mut visitor)?);
            }
            Ok(results)
        }
    }
}

impl Default for ParallelObjectGraphTraverser {
    fn default() -> Self {
        Self::new()
    }
}

/// Breadth-first traversal strategy
/// 
/// This traverser uses a breadth-first approach, which can be more memory-efficient
/// for wide graphs and provides different visiting order characteristics.
pub struct BreadthFirstTraverser {
    /// Configuration for traversal
    config: TraversalConfig,
    /// Queue for breadth-first traversal
    work_queue: VecDeque<(*const u8, usize)>, // (object_ptr, depth)
    /// Set of visited objects
    visited: HashSet<*const u8>,
    /// Statistics collector
    stats: TraversalStats,
}

impl BreadthFirstTraverser {
    /// Create a new breadth-first traverser
    pub fn new() -> Self {
        Self::with_config(TraversalConfig::default())
    }
    
    /// Create a new breadth-first traverser with custom configuration
    pub fn with_config(config: TraversalConfig) -> Self {
        Self {
            config,
            work_queue: VecDeque::new(),
            visited: HashSet::new(),
            stats: TraversalStats::new(),
        }
    }
    
    /// Traverse object graph using breadth-first strategy
    pub fn traverse<V: ObjectVisitor>(
        &mut self,
        roots: &[*const u8],
        visitor: &mut V,
    ) -> Result<TraversalStats, TracingError> {
        let start_time = Instant::now();
        self.reset_state();
        
        // Add roots to work queue
        for &root in roots {
            if !self.visited.contains(&root) {
                self.work_queue.push_back((root, 0));
            }
        }
        
        // Process work queue
        while let Some((object_ptr, depth)) = self.work_queue.pop_front() {
            if self.visited.contains(&object_ptr) {
                continue;
            }
            
            // Check depth limit
            if depth >= self.config.max_depth {
                continue;
            }
            
            // Mark as visited
            self.visited.insert(object_ptr);
            
            // Get object size
            let object_size = unsafe {
                let header = &*(object_ptr as *const ObjectHeader);
                header.size
            };
            
            // Visit the object
            let should_continue = visitor.visit_object(object_ptr, object_size, depth);
            self.stats.objects_visited += 1;
            self.stats.max_depth = self.stats.max_depth.max(depth);
            
            if should_continue {
                // Add references to work queue
                let references = self.trace_object_references(object_ptr)?;
                self.stats.references_followed += references.len();
                
                for ref_ptr in references {
                    if !self.visited.contains(&ref_ptr) {
                        self.work_queue.push_back((ref_ptr, depth + 1));
                    }
                }
            }
        }
        
        // Finalize statistics
        self.stats.duration = start_time.elapsed();
        visitor.traversal_complete(&self.stats);
        
        Ok(self.stats.clone())
    }
    
    /// Trace references from an object
    fn trace_object_references(&self, object_ptr: *const u8) -> Result<Vec<*const u8>, TracingError> {
        unsafe {
            let header = &*(object_ptr as *const ObjectHeader);
            let tracer = registry::get_tracer(header.type_id);
            Ok(tracer.trace_references(object_ptr, header.size))
        }
    }
    
    /// Reset traverser state
    fn reset_state(&mut self) {
        self.work_queue.clear();
        self.visited.clear();
        self.stats = TraversalStats::new();
    }
    
    /// Reset the traverser for reuse
    pub fn reset(&mut self) {
        self.reset_state();
    }
}

impl Default for BreadthFirstTraverser {
    fn default() -> Self {
        Self::new()
    }
}

/// Common visitor implementations for typical use cases
pub mod visitors {
    use super::*;
    
    /// Visitor that collects all reachable objects
    pub struct CollectingVisitor {
        objects: Vec<*const u8>,
        size_threshold: Option<usize>,
    }
    
    impl CollectingVisitor {
        pub fn new() -> Self {
            Self {
                objects: Vec::new(),
                size_threshold: None,
            }
        }
        
        pub fn with_size_threshold(threshold: usize) -> Self {
            Self {
                objects: Vec::new(),
                size_threshold: Some(threshold),
            }
        }
        
        pub fn objects(&self) -> &[*const u8] {
            &self.objects
        }
        
        pub fn into_objects(self) -> Vec<*const u8> {
            self.objects
        }
    }
    
    impl ObjectVisitor for CollectingVisitor {
        fn visit_object(&mut self, object_ptr: *const u8, object_size: usize, _depth: usize) -> bool {
            if let Some(threshold) = self.size_threshold {
                if object_size >= threshold {
                    self.objects.push(object_ptr);
                }
            } else {
                self.objects.push(object_ptr);
            }
            true
        }
    }
    
    /// Visitor that counts objects and references
    pub struct CountingVisitor {
        object_count: usize,
        reference_count: usize,
        size_histogram: HashMap<usize, usize>,
    }
    
    impl CountingVisitor {
        pub fn new() -> Self {
            Self {
                object_count: 0,
                reference_count: 0,
                size_histogram: HashMap::new(),
            }
        }
        
        pub fn object_count(&self) -> usize {
            self.object_count
        }
        
        pub fn reference_count(&self) -> usize {
            self.reference_count
        }
        
        pub fn size_histogram(&self) -> &HashMap<usize, usize> {
            &self.size_histogram
        }
    }
    
    impl ObjectVisitor for CountingVisitor {
        fn visit_object(&mut self, object_ptr: *const u8, object_size: usize, _depth: usize) -> bool {
            self.object_count += 1;
            
            // Update size histogram
            let size_bucket = (object_size / 64) * 64; // 64-byte buckets
            *self.size_histogram.entry(size_bucket).or_insert(0) += 1;
            
            // Count references
            unsafe {
                let header = &*(object_ptr as *const ObjectHeader);
                let tracer = registry::get_tracer(header.type_id);
                let references = tracer.trace_references(object_ptr, header.size);
                self.reference_count += references.len();
            }
            
            true
        }
    }
    
    /// Visitor that validates object graph integrity
    pub struct ValidationVisitor {
        errors: Vec<String>,
        validate_headers: bool,
        validate_references: bool,
    }
    
    impl ValidationVisitor {
        pub fn new() -> Self {
            Self {
                errors: Vec::new(),
                validate_headers: true,
                validate_references: true,
            }
        }
        
        pub fn errors(&self) -> &[String] {
            &self.errors
        }
        
        pub fn is_valid(&self) -> bool {
            self.errors.is_empty()
        }
    }
    
    impl ObjectVisitor for ValidationVisitor {
        fn visit_object(&mut self, object_ptr: *const u8, object_size: usize, _depth: usize) -> bool {
            if self.validate_headers {
                unsafe {
                    let header = &*(object_ptr as *const ObjectHeader);
                    if header.size != object_size {
                        self.errors.push(format!(
                            "Object at {:p}: header size {} != actual size {}",
                            object_ptr, header.size, object_size
                        ));
                    }
                }
            }
            
            if self.validate_references {
                unsafe {
                    let header = &*(object_ptr as *const ObjectHeader);
                    let tracer = registry::get_tracer(header.type_id);
                    let references = tracer.trace_references(object_ptr, header.size);
                    
                    for ref_ptr in references {
                        if ref_ptr.is_null() {
                            self.errors.push(format!(
                                "Object at {:p}: contains null reference",
                                object_ptr
                            ));
                        }
                        // Additional reference validation could be added here
                    }
                }
            }
            
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::visitors::*;
    
    #[test]
    fn test_collecting_visitor() {
        let mut visitor = CollectingVisitor::new();
        
        // Mock some object visits
        visitor.visit_object(0x1000 as *const u8, 64, 0);
        visitor.visit_object(0x2000 as *const u8, 128, 1);
        
        assert_eq!(visitor.objects().len(), 2);
        assert_eq!(visitor.objects()[0] as usize, 0x1000);
        assert_eq!(visitor.objects()[1] as usize, 0x2000);
    }
    
    #[test]
    fn test_counting_visitor() {
        let mut visitor = CountingVisitor::new();
        
        // Mock some object visits
        visitor.visit_object(0x1000 as *const u8, 64, 0);
        visitor.visit_object(0x2000 as *const u8, 128, 1);
        
        assert_eq!(visitor.object_count(), 2);
        assert_eq!(visitor.size_histogram().len(), 2); // Two different size buckets
    }
    
    #[test]
    fn test_traversal_config() {
        let config = TraversalConfig {
            max_depth: 500,
            enable_parallel: false,
            ..Default::default()
        };
        
        let traverser = ObjectGraphTraverser::with_config(config);
        assert_eq!(traverser.config.max_depth, 500);
        assert!(!traverser.config.enable_parallel);
    }
} 