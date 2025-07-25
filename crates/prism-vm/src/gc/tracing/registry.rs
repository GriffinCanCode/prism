//! Thread-safe tracer registry for managing type-specific object tracers
//!
//! This module provides a centralized registry for managing ObjectTracer implementations
//! for different object types. The registry is thread-safe and supports dynamic
//! registration and lookup of tracers.
//!
//! ## Design Principles
//!
//! - **Thread Safety**: All operations are thread-safe with minimal contention
//! - **Performance**: Fast lookup paths with caching for frequently used tracers
//! - **Safety**: Proper validation and error handling for tracer registration
//! - **Extensibility**: Easy to add new tracer types at runtime

use super::types::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};

/// Global tracer registry instance
static TRACER_REGISTRY: OnceLock<TracerRegistry> = OnceLock::new();

/// Initialize the global tracer registry
/// 
/// This must be called before any tracing operations. It's safe to call
/// multiple times - subsequent calls will be ignored.
pub fn init_tracer_registry() -> Result<(), TracingError> {
    init_tracer_registry_with_config(TracingConfig::default())
}

/// Initialize the global tracer registry with custom configuration
pub fn init_tracer_registry_with_config(config: TracingConfig) -> Result<(), TracingError> {
    TRACER_REGISTRY.get_or_init(|| TracerRegistry::new(config));
    Ok(())
}

/// Get a tracer for a specific type ID
/// 
/// Returns the registered tracer for the given type ID, or the default tracer
/// if no specific tracer is registered.
pub fn get_tracer(type_id: u32) -> Arc<dyn ObjectTracer> {
    get_registry().get_tracer(type_id)
}

/// Register a tracer for a specific type
/// 
/// This will replace any existing tracer for the given type ID.
pub fn register_tracer(type_id: u32, tracer: Arc<dyn ObjectTracer>) -> Result<(), TracingError> {
    get_registry().register_tracer(type_id, tracer)
}

/// Unregister a tracer for a specific type
/// 
/// After unregistration, the default tracer will be used for this type.
pub fn unregister_tracer(type_id: u32) -> Result<(), TracingError> {
    get_registry().unregister_tracer(type_id)
}

/// Check if a tracer is registered for a specific type
pub fn has_tracer(type_id: u32) -> bool {
    get_registry().has_tracer(type_id)
}

/// Get all registered type IDs
pub fn registered_types() -> Vec<u32> {
    get_registry().registered_types()
}

/// Get registry statistics
pub fn registry_stats() -> RegistryStats {
    get_registry().stats()
}

/// Get the global tracer registry
fn get_registry() -> &'static TracerRegistry {
    TRACER_REGISTRY.get().expect("Tracer registry not initialized. Call init_tracer_registry() first.")
}

/// Thread-safe registry for object tracers
/// 
/// The registry maintains a mapping from type IDs to tracer implementations,
/// with additional caching and statistics for performance monitoring.
pub struct TracerRegistry {
    /// Map from type ID to tracer implementation
    tracers: RwLock<HashMap<u32, Arc<dyn ObjectTracer>>>,
    /// Default tracer used when no specific tracer is registered
    default_tracer: Arc<dyn ObjectTracer>,
    /// Configuration for the registry
    config: TracingConfig,
    /// Cache of recently used tracers (type_id -> tracer)
    tracer_cache: RwLock<lru::LruCache<u32, Arc<dyn ObjectTracer>>>,
    /// Registry statistics
    stats: Arc<RegistryStats>,
    /// Whether the registry is initialized
    initialized: AtomicBool,
}

impl TracerRegistry {
    /// Create a new tracer registry with the given configuration
    fn new(config: TracingConfig) -> Self {
        let cache_size = config.tracing_cache_size.max(16); // Minimum cache size
        
        Self {
            tracers: RwLock::new(HashMap::new()),
            default_tracer: Arc::new(DefaultTracer::new()),
            config,
            tracer_cache: RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_size).unwrap()
            )),
            stats: Arc::new(RegistryStats::new()),
            initialized: AtomicBool::new(true),
        }
    }
    
    /// Get a tracer for the specified type ID
    pub fn get_tracer(&self, type_id: u32) -> Arc<dyn ObjectTracer> {
        // First check the cache for recently used tracers
        if let Ok(mut cache) = self.tracer_cache.try_write() {
            if let Some(tracer) = cache.get(&type_id) {
                self.stats.record_cache_hit();
                return tracer.clone();
            }
        }
        
        // Cache miss - look up in the main registry
        self.stats.record_cache_miss();
        
        let tracer = {
            let tracers = self.tracers.read().unwrap();
            tracers.get(&type_id)
                .cloned()
                .unwrap_or_else(|| self.default_tracer.clone())
        };
        
        // Update cache with the found tracer
        if let Ok(mut cache) = self.tracer_cache.try_write() {
            cache.put(type_id, tracer.clone());
        }
        
        self.stats.record_lookup();
        tracer
    }
    
    /// Register a tracer for a specific type
    pub fn register_tracer(&self, type_id: u32, tracer: Arc<dyn ObjectTracer>) -> Result<(), TracingError> {
        // Validate the tracer
        self.validate_tracer(&tracer)?;
        
        // Register the tracer
        {
            let mut tracers = self.tracers.write().unwrap();
            tracers.insert(type_id, tracer.clone());
        }
        
        // Update cache
        if let Ok(mut cache) = self.tracer_cache.try_write() {
            cache.put(type_id, tracer);
        }
        
        self.stats.record_registration();
        Ok(())
    }
    
    /// Unregister a tracer for a specific type
    pub fn unregister_tracer(&self, type_id: u32) -> Result<(), TracingError> {
        {
            let mut tracers = self.tracers.write().unwrap();
            tracers.remove(&type_id);
        }
        
        // Remove from cache
        if let Ok(mut cache) = self.tracer_cache.try_write() {
            cache.pop(&type_id);
        }
        
        self.stats.record_unregistration();
        Ok(())
    }
    
    /// Check if a tracer is registered for a specific type
    pub fn has_tracer(&self, type_id: u32) -> bool {
        let tracers = self.tracers.read().unwrap();
        tracers.contains_key(&type_id)
    }
    
    /// Get all registered type IDs
    pub fn registered_types(&self) -> Vec<u32> {
        let tracers = self.tracers.read().unwrap();
        tracers.keys().copied().collect()
    }
    
    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        self.stats.snapshot()
    }
    
    /// Validate a tracer implementation
    fn validate_tracer(&self, tracer: &Arc<dyn ObjectTracer>) -> Result<(), TracingError> {
        // Basic validation - ensure the tracer responds correctly
        let tracer_name = tracer.tracer_name();
        if tracer_name.is_empty() {
            return Err(TracingError::Generic("Tracer name cannot be empty".to_string()));
        }
        
        // Additional validation could be added here
        Ok(())
    }
    
    /// Clear all registered tracers (for testing)
    #[cfg(test)]
    pub fn clear(&self) {
        let mut tracers = self.tracers.write().unwrap();
        tracers.clear();
        
        if let Ok(mut cache) = self.tracer_cache.try_write() {
            cache.clear();
        }
    }
}

/// Default tracer implementation used when no specific tracer is registered
struct DefaultTracer {
    /// Whether this tracer should attempt conservative scanning
    conservative_scanning: bool,
}

impl DefaultTracer {
    fn new() -> Self {
        Self {
            conservative_scanning: false,
        }
    }
    
    /// Enable conservative scanning for unknown object types
    #[allow(dead_code)]
    fn with_conservative_scanning() -> Self {
        Self {
            conservative_scanning: true,
        }
    }
}

impl ObjectTracer for DefaultTracer {
    unsafe fn trace_references(&self, object_ptr: *const u8, object_size: usize) -> Vec<*const u8> {
        if !self.conservative_scanning {
            // Default behavior: assume no references
            return Vec::new();
        }
        
        // Conservative scanning: treat any pointer-aligned value as a potential reference
        let mut references = Vec::new();
        
        // Skip the object header
        let header_size = std::mem::size_of::<ObjectHeader>();
        if object_size <= header_size {
            return references;
        }
        
        let data_ptr = object_ptr.add(header_size);
        let data_size = object_size - header_size;
        
        // Scan for potential pointers
        let mut current = data_ptr;
        let end = data_ptr.add(data_size);
        
        while current < end {
            // Align to pointer boundary
            let aligned = Self::align_pointer(current);
            if aligned >= end {
                break;
            }
            
            let potential_ptr = *(aligned as *const *const u8);
            
            // Basic validation - this would be more sophisticated in practice
            if Self::could_be_gc_pointer(potential_ptr) {
                references.push(potential_ptr);
            }
            
            current = aligned.add(std::mem::size_of::<*const u8>());
        }
        
        references
    }
    
    fn has_references(&self) -> bool {
        self.conservative_scanning
    }
    
    fn tracer_name(&self) -> &'static str {
        if self.conservative_scanning {
            "DefaultConservativeTracer"
        } else {
            "DefaultNoReferencesTracer"
        }
    }
    
    fn can_trace(&self, _object_ptr: *const u8, _object_size: usize) -> bool {
        true // Default tracer can handle any object
    }
}

impl DefaultTracer {
    /// Align a pointer to the next pointer boundary
    unsafe fn align_pointer(ptr: *const u8) -> *const u8 {
        let addr = ptr as usize;
        let align = std::mem::align_of::<*const u8>();
        let aligned_addr = (addr + align - 1) & !(align - 1);
        aligned_addr as *const u8
    }
    
    /// Basic heuristic to determine if a value could be a GC pointer
    fn could_be_gc_pointer(ptr: *const u8) -> bool {
        if ptr.is_null() {
            return false;
        }
        
        let addr = ptr as usize;
        
        // Check alignment
        if addr % std::mem::align_of::<usize>() != 0 {
            return false;
        }
        
        // Filter out small values and very large values
        // This is a basic heuristic and would need refinement
        addr >= 0x1000 && addr <= 0x7fff_ffff_ffff
    }
}

/// Statistics for the tracer registry
#[derive(Debug, Default)]
pub struct RegistryStats {
    /// Total number of tracer lookups
    lookups: std::sync::atomic::AtomicUsize,
    /// Number of cache hits
    cache_hits: std::sync::atomic::AtomicUsize,
    /// Number of cache misses
    cache_misses: std::sync::atomic::AtomicUsize,
    /// Number of tracer registrations
    registrations: std::sync::atomic::AtomicUsize,
    /// Number of tracer unregistrations
    unregistrations: std::sync::atomic::AtomicUsize,
}

impl RegistryStats {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_lookup(&self) {
        self.lookups.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_registration(&self) {
        self.registrations.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_unregistration(&self) {
        self.unregistrations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Create a snapshot of current statistics
    fn snapshot(&self) -> RegistryStats {
        RegistryStats {
            lookups: std::sync::atomic::AtomicUsize::new(
                self.lookups.load(Ordering::Relaxed)
            ),
            cache_hits: std::sync::atomic::AtomicUsize::new(
                self.cache_hits.load(Ordering::Relaxed)
            ),
            cache_misses: std::sync::atomic::AtomicUsize::new(
                self.cache_misses.load(Ordering::Relaxed)
            ),
            registrations: std::sync::atomic::AtomicUsize::new(
                self.registrations.load(Ordering::Relaxed)
            ),
            unregistrations: std::sync::atomic::AtomicUsize::new(
                self.unregistrations.load(Ordering::Relaxed)
            ),
        }
    }
    
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
    
    /// Get total number of lookups
    pub fn total_lookups(&self) -> usize {
        self.lookups.load(Ordering::Relaxed)
    }
    
    /// Get total number of registrations
    pub fn total_registrations(&self) -> usize {
        self.registrations.load(Ordering::Relaxed)
    }
}

impl Clone for RegistryStats {
    fn clone(&self) -> Self {
        self.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestTracer {
        name: &'static str,
    }
    
    impl TestTracer {
        fn new(name: &'static str) -> Self {
            Self { name }
        }
    }
    
    impl ObjectTracer for TestTracer {
        unsafe fn trace_references(&self, _object_ptr: *const u8, _object_size: usize) -> Vec<*const u8> {
            Vec::new()
        }
        
        fn tracer_name(&self) -> &'static str {
            self.name
        }
    }
    
    #[test]
    fn test_registry_basic_operations() {
        let registry = TracerRegistry::new(TracingConfig::default());
        
        // Test registration and lookup
        let tracer = Arc::new(TestTracer::new("TestTracer"));
        registry.register_tracer(42, tracer.clone()).unwrap();
        
        assert!(registry.has_tracer(42));
        let retrieved = registry.get_tracer(42);
        assert_eq!(retrieved.tracer_name(), "TestTracer");
        
        // Test unregistration
        registry.unregister_tracer(42).unwrap();
        assert!(!registry.has_tracer(42));
        
        // Should now return default tracer
        let default = registry.get_tracer(42);
        assert!(default.tracer_name().contains("Default"));
    }
    
    #[test]
    fn test_registry_caching() {
        let registry = TracerRegistry::new(TracingConfig::default());
        let tracer = Arc::new(TestTracer::new("CachedTracer"));
        
        registry.register_tracer(123, tracer).unwrap();
        
        // First lookup should be a cache miss
        let _first = registry.get_tracer(123);
        
        // Second lookup should be a cache hit
        let _second = registry.get_tracer(123);
        
        let stats = registry.stats();
        assert!(stats.cache_hit_rate() > 0.0);
    }
} 