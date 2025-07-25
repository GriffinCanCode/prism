//! Utility functions and common operations for object tracing
//!
//! This module provides convenient utility functions for common tracing operations,
//! including parallel tracing, object counting, and batch operations.
//!
//! ## Design Principles
//!
//! - **Convenience**: Easy-to-use functions for common tracing tasks
//! - **Performance**: Optimized implementations with parallel support
//! - **Safety**: Memory-safe operations with proper error handling
//! - **Composability**: Functions that work well together

use super::types::*;
use super::registry;
use super::traversal::{ObjectGraphTraverser, visitors::*};
use std::collections::HashMap;
use std::sync::Arc;

/// Trace a single object and return its references
/// 
/// This is a convenience function for tracing a single object without
/// setting up a full traversal context.
/// 
/// # Safety
/// 
/// The object pointer must be valid and point to a properly formatted object
/// with a valid header.
pub unsafe fn trace_object(object_ptr: *const u8) -> Result<Vec<*const u8>, TracingError> {
    if object_ptr.is_null() {
        return Err(TracingError::InvalidObjectPointer(object_ptr));
    }
    
    // Get object header to determine type and size
    let header = &*(object_ptr as *const ObjectHeader);
    let tracer = registry::get_tracer(header.type_id);
    
    // Validate tracer can handle this object
    if !tracer.can_trace(object_ptr, header.size) {
        return Err(TracingError::Generic(format!(
            "Tracer {} cannot trace object at {:p}",
            tracer.tracer_name(),
            object_ptr
        )));
    }
    
    Ok(tracer.trace_references(object_ptr, header.size))
}

/// Trace multiple objects in parallel
/// 
/// This function traces multiple objects concurrently, which can provide
/// significant performance benefits for large numbers of objects.
pub fn trace_objects_parallel(objects: &[*const u8]) -> Vec<Result<Vec<*const u8>, TracingError>> {
    if objects.is_empty() {
        return Vec::new();
    }
    
    // Use rayon for parallel processing
    use rayon::prelude::*;
    
    objects.par_iter()
        .map(|&obj_ptr| unsafe { trace_object(obj_ptr) })
        .collect()
}

/// Trace multiple objects sequentially
/// 
/// This function traces multiple objects one by one, which uses less memory
/// than parallel tracing but may be slower for large numbers of objects.
pub fn trace_objects_sequential(objects: &[*const u8]) -> Vec<Result<Vec<*const u8>, TracingError>> {
    objects.iter()
        .map(|&obj_ptr| unsafe { trace_object(obj_ptr) })
        .collect()
}

/// Count total references in an object
/// 
/// This is a convenience function that traces an object and returns the
/// number of references found.
/// 
/// # Safety
/// 
/// The object pointer must be valid and point to a properly formatted object.
pub unsafe fn count_references(object_ptr: *const u8) -> Result<usize, TracingError> {
    let references = trace_object(object_ptr)?;
    Ok(references.len())
}

/// Check if an object has any references
/// 
/// This function uses the tracer's `has_references` method for efficiency,
/// falling back to actual tracing if necessary.
/// 
/// # Safety
/// 
/// The object pointer must be valid and point to a properly formatted object.
pub unsafe fn has_references(object_ptr: *const u8) -> Result<bool, TracingError> {
    if object_ptr.is_null() {
        return Err(TracingError::InvalidObjectPointer(object_ptr));
    }
    
    let header = &*(object_ptr as *const ObjectHeader);
    let tracer = registry::get_tracer(header.type_id);
    
    // Use tracer's optimization if available
    if !tracer.has_references() {
        return Ok(false);
    }
    
    // Fall back to actual tracing
    let references = tracer.trace_references(object_ptr, header.size);
    Ok(!references.is_empty())
}

/// Collect all objects reachable from a set of roots
/// 
/// This function performs a complete traversal of the object graph starting
/// from the given roots and returns all reachable objects.
pub fn collect_reachable_objects(roots: &[*const u8]) -> Result<Vec<*const u8>, TracingError> {
    let mut traverser = ObjectGraphTraverser::new();
    let mut visitor = CollectingVisitor::new();
    
    traverser.traverse(roots, &mut visitor)?;
    Ok(visitor.into_objects())
}

/// Count all objects reachable from a set of roots
/// 
/// This function performs a traversal and returns statistics about the
/// reachable object graph.
pub fn count_reachable_objects(roots: &[*const u8]) -> Result<(usize, usize), TracingError> {
    let mut traverser = ObjectGraphTraverser::new();
    let mut visitor = CountingVisitor::new();
    
    traverser.traverse(roots, &mut visitor)?;
    Ok((visitor.object_count(), visitor.reference_count()))
}

/// Validate object graph integrity
/// 
/// This function traverses the object graph and validates that all objects
/// and references are properly formed.
pub fn validate_object_graph(roots: &[*const u8]) -> Result<Vec<String>, TracingError> {
    let mut traverser = ObjectGraphTraverser::new();
    let mut visitor = ValidationVisitor::new();
    
    traverser.traverse(roots, &mut visitor)?;
    Ok(visitor.errors().to_vec())
}

/// Calculate object graph statistics
/// 
/// This function provides detailed statistics about an object graph,
/// including size distributions, reference patterns, and depth information.
#[derive(Debug, Clone)]
pub struct ObjectGraphStats {
    /// Total number of objects
    pub total_objects: usize,
    /// Total number of references
    pub total_references: usize,
    /// Maximum depth of the object graph
    pub max_depth: usize,
    /// Average references per object
    pub avg_references_per_object: f64,
    /// Size distribution histogram
    pub size_histogram: HashMap<usize, usize>,
    /// Type distribution
    pub type_distribution: HashMap<u32, usize>,
    /// Memory usage breakdown
    pub memory_usage: MemoryUsageStats,
}

#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    /// Total memory used by objects
    pub total_object_memory: usize,
    /// Memory used by object headers
    pub header_memory: usize,
    /// Memory used by object data
    pub data_memory: usize,
    /// Average object size
    pub avg_object_size: f64,
}

impl ObjectGraphStats {
    /// Create empty statistics
    pub fn new() -> Self {
        Self {
            total_objects: 0,
            total_references: 0,
            max_depth: 0,
            avg_references_per_object: 0.0,
            size_histogram: HashMap::new(),
            type_distribution: HashMap::new(),
            memory_usage: MemoryUsageStats {
                total_object_memory: 0,
                header_memory: 0,
                data_memory: 0,
                avg_object_size: 0.0,
            },
        }
    }
    
    /// Calculate derived statistics
    pub fn finalize(&mut self) {
        if self.total_objects > 0 {
            self.avg_references_per_object = 
                self.total_references as f64 / self.total_objects as f64;
            
            self.memory_usage.avg_object_size = 
                self.memory_usage.total_object_memory as f64 / self.total_objects as f64;
        }
    }
}

impl Default for ObjectGraphStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Visitor for collecting detailed object graph statistics
struct StatsCollectingVisitor {
    stats: ObjectGraphStats,
}

impl StatsCollectingVisitor {
    fn new() -> Self {
        Self {
            stats: ObjectGraphStats::new(),
        }
    }
    
    fn into_stats(mut self) -> ObjectGraphStats {
        self.stats.finalize();
        self.stats
    }
}

impl ObjectVisitor for StatsCollectingVisitor {
    fn visit_object(&mut self, object_ptr: *const u8, object_size: usize, depth: usize) -> bool {
        self.stats.total_objects += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);
        
        // Update memory usage
        let header_size = std::mem::size_of::<ObjectHeader>();
        self.stats.memory_usage.total_object_memory += object_size;
        self.stats.memory_usage.header_memory += header_size;
        self.stats.memory_usage.data_memory += object_size.saturating_sub(header_size);
        
        // Update size histogram
        let size_bucket = (object_size / 64) * 64; // 64-byte buckets
        *self.stats.size_histogram.entry(size_bucket).or_insert(0) += 1;
        
        // Update type distribution and count references
        unsafe {
            let header = &*(object_ptr as *const ObjectHeader);
            *self.stats.type_distribution.entry(header.type_id).or_insert(0) += 1;
            
            let tracer = registry::get_tracer(header.type_id);
            let references = tracer.trace_references(object_ptr, header.size);
            self.stats.total_references += references.len();
        }
        
        true
    }
}

/// Calculate detailed statistics for an object graph
pub fn calculate_object_graph_stats(roots: &[*const u8]) -> Result<ObjectGraphStats, TracingError> {
    let mut traverser = ObjectGraphTraverser::new();
    let mut visitor = StatsCollectingVisitor::new();
    
    traverser.traverse(roots, &mut visitor)?;
    Ok(visitor.into_stats())
}

/// Batch tracing operations for improved performance
pub struct BatchTracer {
    /// Objects to trace
    objects: Vec<*const u8>,
    /// Configuration for batch operations
    config: BatchTracingConfig,
    /// Results cache
    results_cache: HashMap<*const u8, Vec<*const u8>>,
}

#[derive(Debug, Clone)]
pub struct BatchTracingConfig {
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable result caching
    pub enable_caching: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
}

impl Default for BatchTracingConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            batch_size: 1000,
            enable_caching: true,
            max_cache_size: 10000,
        }
    }
}

impl BatchTracer {
    /// Create a new batch tracer
    pub fn new() -> Self {
        Self::with_config(BatchTracingConfig::default())
    }
    
    /// Create a batch tracer with custom configuration
    pub fn with_config(config: BatchTracingConfig) -> Self {
        Self {
            objects: Vec::new(),
            config,
            results_cache: HashMap::new(),
        }
    }
    
    /// Add objects to the batch
    pub fn add_objects(&mut self, objects: &[*const u8]) {
        self.objects.extend_from_slice(objects);
    }
    
    /// Add a single object to the batch
    pub fn add_object(&mut self, object: *const u8) {
        self.objects.push(object);
    }
    
    /// Process all objects in the batch
    pub fn process_batch(&mut self) -> Result<HashMap<*const u8, Vec<*const u8>>, TracingError> {
        let mut results = HashMap::new();
        
        // Process in chunks for better memory usage
        for chunk in self.objects.chunks(self.config.batch_size) {
            let chunk_results = if self.config.enable_parallel {
                self.process_chunk_parallel(chunk)?
            } else {
                self.process_chunk_sequential(chunk)?
            };
            
            for (obj_ptr, references) in chunk_results {
                // Update cache if enabled
                if self.config.enable_caching && 
                   self.results_cache.len() < self.config.max_cache_size {
                    self.results_cache.insert(obj_ptr, references.clone());
                }
                
                results.insert(obj_ptr, references);
            }
        }
        
        Ok(results)
    }
    
    /// Process a chunk of objects in parallel
    fn process_chunk_parallel(
        &self, 
        chunk: &[*const u8]
    ) -> Result<HashMap<*const u8, Vec<*const u8>>, TracingError> {
        use rayon::prelude::*;
        
        let results: Vec<_> = chunk.par_iter()
            .map(|&obj_ptr| {
                // Check cache first
                if self.config.enable_caching {
                    if let Some(cached) = self.results_cache.get(&obj_ptr) {
                        return Ok((obj_ptr, cached.clone()));
                    }
                }
                
                // Trace the object
                unsafe {
                    trace_object(obj_ptr).map(|refs| (obj_ptr, refs))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(results.into_iter().collect())
    }
    
    /// Process a chunk of objects sequentially
    fn process_chunk_sequential(
        &self, 
        chunk: &[*const u8]
    ) -> Result<HashMap<*const u8, Vec<*const u8>>, TracingError> {
        let mut results = HashMap::new();
        
        for &obj_ptr in chunk {
            // Check cache first
            if self.config.enable_caching {
                if let Some(cached) = self.results_cache.get(&obj_ptr) {
                    results.insert(obj_ptr, cached.clone());
                    continue;
                }
            }
            
            // Trace the object
            let references = unsafe { trace_object(obj_ptr)? };
            results.insert(obj_ptr, references);
        }
        
        Ok(results)
    }
    
    /// Clear the batch and cache
    pub fn clear(&mut self) {
        self.objects.clear();
        self.results_cache.clear();
    }
    
    /// Get the number of objects in the batch
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl Default for BatchTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common tracing patterns
pub mod patterns {
    use super::*;
    
    /// Find all objects of a specific type in the reachable graph
    pub fn find_objects_by_type(
        roots: &[*const u8], 
        target_type_id: u32
    ) -> Result<Vec<*const u8>, TracingError> {
        struct TypeFilterVisitor {
            target_type_id: u32,
            matching_objects: Vec<*const u8>,
        }
        
        impl ObjectVisitor for TypeFilterVisitor {
            fn visit_object(&mut self, object_ptr: *const u8, _object_size: usize, _depth: usize) -> bool {
                unsafe {
                    let header = &*(object_ptr as *const ObjectHeader);
                    if header.type_id == self.target_type_id {
                        self.matching_objects.push(object_ptr);
                    }
                }
                true
            }
        }
        
        let mut traverser = ObjectGraphTraverser::new();
        let mut visitor = TypeFilterVisitor {
            target_type_id,
            matching_objects: Vec::new(),
        };
        
        traverser.traverse(roots, &mut visitor)?;
        Ok(visitor.matching_objects)
    }
    
    /// Find all objects larger than a specific size
    pub fn find_large_objects(
        roots: &[*const u8], 
        min_size: usize
    ) -> Result<Vec<(*const u8, usize)>, TracingError> {
        struct SizeFilterVisitor {
            min_size: usize,
            large_objects: Vec<(*const u8, usize)>,
        }
        
        impl ObjectVisitor for SizeFilterVisitor {
            fn visit_object(&mut self, object_ptr: *const u8, object_size: usize, _depth: usize) -> bool {
                if object_size >= self.min_size {
                    self.large_objects.push((object_ptr, object_size));
                }
                true
            }
        }
        
        let mut traverser = ObjectGraphTraverser::new();
        let mut visitor = SizeFilterVisitor {
            min_size,
            large_objects: Vec::new(),
        };
        
        traverser.traverse(roots, &mut visitor)?;
        Ok(visitor.large_objects)
    }
    
    /// Find the shortest path between two objects using BFS
    pub fn find_shortest_path(
        roots: &[*const u8],
        target: *const u8,
    ) -> Result<Option<Vec<*const u8>>, TracingError> {
        use std::collections::{VecDeque, HashMap, HashSet};
        
        // BFS implementation for shortest path
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent_map: HashMap<*const u8, *const u8> = HashMap::new();
        
        // Initialize with all roots
        for &root in roots {
            if root == target {
                return Ok(Some(vec![root]));
            }
            queue.push_back(root);
            visited.insert(root);
        }
        
        // BFS traversal
        while let Some(current) = queue.pop_front() {
            // Get references from current object
            let references = unsafe {
                match trace_object(current) {
                    Ok(refs) => refs,
                    Err(_) => continue, // Skip objects that can't be traced
                }
            };
            
            for &ref_ptr in &references {
                if visited.contains(&ref_ptr) {
                    continue;
                }
                
                visited.insert(ref_ptr);
                parent_map.insert(ref_ptr, current);
                
                if ref_ptr == target {
                    // Found target - reconstruct path
                    let mut path = Vec::new();
                    let mut current_node = target;
                    
                    // Build path backwards from target to root
                    path.push(current_node);
                    while let Some(&parent) = parent_map.get(&current_node) {
                        path.push(parent);
                        current_node = parent;
                        
                        // Stop when we reach a root
                        if roots.contains(&parent) {
                            break;
                        }
                    }
                    
                    // Reverse to get path from root to target
                    path.reverse();
                    return Ok(Some(path));
                }
                
                queue.push_back(ref_ptr);
            }
        }
        
        // Target not found
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    
    #[test]
    fn test_batch_tracer() {
        let mut batch_tracer = BatchTracer::new();
        
        // Add some mock objects
        let objects = vec![0x1000 as *const u8, 0x2000 as *const u8];
        batch_tracer.add_objects(&objects);
        
        assert_eq!(batch_tracer.len(), 2);
        assert!(!batch_tracer.is_empty());
        
        batch_tracer.clear();
        assert_eq!(batch_tracer.len(), 0);
        assert!(batch_tracer.is_empty());
    }
    
    #[test]
    fn test_object_graph_stats() {
        let mut stats = ObjectGraphStats::new();
        stats.total_objects = 100;
        stats.total_references = 250;
        stats.memory_usage.total_object_memory = 6400;
        
        stats.finalize();
        
        assert_eq!(stats.avg_references_per_object, 2.5);
        assert_eq!(stats.memory_usage.avg_object_size, 64.0);
    }
    
    #[test]
    fn test_batch_tracing_config() {
        let config = BatchTracingConfig {
            enable_parallel: false,
            batch_size: 500,
            ..Default::default()
        };
        
        assert!(!config.enable_parallel);
        assert_eq!(config.batch_size, 500);
        assert!(config.enable_caching);
    }
} 