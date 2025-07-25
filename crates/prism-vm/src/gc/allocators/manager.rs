//! Allocator Manager - Central coordination for memory allocation
//!
//! This manager provides a unified interface for different allocator types,
//! handles allocator selection based on object size and type, and coordinates
//! between allocators during garbage collection.

use super::*;
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::collections::HashMap;
use std::thread;

/// Central allocator manager coordinating multiple allocator strategies
/// 
/// This manager provides:
/// - Unified allocation interface across different allocator types
/// - Intelligent allocator selection based on object characteristics
/// - Coordination during garbage collection cycles
/// - Performance monitoring and adaptive behavior
/// - Thread-safe allocation with minimal contention
pub struct AllocatorManager {
    /// Primary allocator for small objects (size-class based)
    small_object_allocator: Arc<PrismAllocator>,
    
    /// Bump allocator for young generation objects
    young_allocator: Arc<BumpAllocator>,
    
    /// Specialized allocator for large objects
    large_object_allocator: Arc<LargeObjectAllocator>,
    
    /// Low-level page allocator
    page_allocator: Arc<PageAllocator>,
    
    /// Thread-local cache manager
    thread_cache_manager: Arc<ThreadCacheManager>,
    
    /// Configuration for allocation strategy
    config: Arc<RwLock<AllocationManagerConfig>>,
    
    /// Global allocation statistics
    global_stats: Arc<Mutex<GlobalAllocationStats>>,
    
    /// Allocation pressure monitor for GC triggering
    pressure_monitor: Arc<AllocationPressureMonitor>,
    
    /// Current allocation mode (normal, gc-preparing, gc-active)
    allocation_mode: Arc<RwLock<AllocationMode>>,
}

/// Configuration for the allocation manager
#[derive(Debug, Clone)]
pub struct AllocationManagerConfig {
    /// Size threshold for large objects
    pub large_object_threshold: usize,
    
    /// Size threshold for using bump allocator vs size-class allocator
    pub bump_allocator_threshold: usize,
    
    /// Enable thread-local caching
    pub enable_thread_caches: bool,
    
    /// Maximum thread cache size
    pub max_thread_cache_size: usize,
    
    /// GC trigger threshold (bytes)
    pub gc_trigger_threshold: usize,
    
    /// Enable adaptive allocation strategies
    pub enable_adaptive_strategies: bool,
    
    /// NUMA awareness settings
    pub numa_awareness: NumaAwarenessConfig,
}

/// NUMA awareness configuration
#[derive(Debug, Clone)]
pub struct NumaAwarenessConfig {
    pub enabled: bool,
    pub preferred_node: Option<usize>,
    pub interleave_policy: bool,
}

/// Current allocation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationMode {
    /// Normal allocation mode
    Normal,
    /// Preparing for garbage collection
    GcPreparing,
    /// Garbage collection is active
    GcActive,
    /// Post-GC cleanup phase
    PostGc,
}

/// Global allocation statistics across all allocators
#[derive(Debug, Default)]
pub struct GlobalAllocationStats {
    /// Total allocations across all allocators
    pub total_allocations: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Allocations per allocator type
    pub allocator_stats: HashMap<AllocatorType, AllocatorTypeStats>,
    /// Thread cache statistics
    pub thread_cache_stats: ThreadCacheStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Statistics for a specific allocator type
#[derive(Debug, Default)]
pub struct AllocatorTypeStats {
    pub allocations: usize,
    pub bytes_allocated: usize,
    pub average_allocation_size: f64,
    pub allocation_rate: f64,
}

/// Identifies different allocator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocatorType {
    SmallObject,
    Young,
    LargeObject,
    Page,
}

/// Thread cache statistics
#[derive(Debug, Default)]
pub struct ThreadCacheStats {
    pub active_caches: usize,
    pub total_cached_bytes: usize,
    pub cache_hit_rate: f64,
    pub cache_miss_rate: f64,
}

/// Performance metrics for allocation subsystem
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub average_allocation_latency: f64,
    pub allocation_throughput: f64,
    pub contention_rate: f64,
    pub gc_allocation_impact: f64,
}

/// Monitors allocation pressure to trigger garbage collection
pub struct AllocationPressureMonitor {
    /// Current allocation rate (bytes/second)
    allocation_rate: AtomicUsize,
    /// Total allocated bytes since last GC
    bytes_since_gc: AtomicUsize,
    /// Last allocation timestamp
    last_allocation_time: Mutex<std::time::Instant>,
    /// Pressure threshold for GC triggering
    pressure_threshold: AtomicUsize,
    /// Adaptive threshold adjustment
    adaptive_factor: AtomicUsize, // Fixed-point: 1000 = 1.0
}

/// Thread cache manager for coordinating thread-local caches
pub struct ThreadCacheManager {
    /// Active thread caches
    caches: RwLock<HashMap<thread::ThreadId, Arc<ThreadCache>>>,
    /// Global cache statistics
    stats: Mutex<ThreadCacheStats>,
    /// Configuration
    config: ThreadCacheConfig,
}

/// Configuration for thread caches
#[derive(Debug, Clone)]
pub struct ThreadCacheConfig {
    pub max_cache_size: usize,
    pub batch_size: usize,
    pub gc_flush_threshold: usize,
}

impl Default for AllocationManagerConfig {
    fn default() -> Self {
        Self {
            large_object_threshold: LARGE_OBJECT_THRESHOLD,
            bump_allocator_threshold: 1024, // 1KB
            enable_thread_caches: true,
            max_thread_cache_size: THREAD_CACHE_SIZE,
            gc_trigger_threshold: 64 * 1024 * 1024, // 64MB
            enable_adaptive_strategies: true,
            numa_awareness: NumaAwarenessConfig {
                enabled: false,
                preferred_node: None,
                interleave_policy: false,
            },
        }
    }
}

impl AllocatorManager {
    /// Create a new allocator manager with default configuration
    pub fn new() -> Self {
        Self::with_config(AllocationManagerConfig::default())
    }
    
    /// Create a new allocator manager with custom configuration
    pub fn with_config(config: AllocationManagerConfig) -> Self {
        let prism_config = PrismAllocatorConfig {
            base: AllocatorConfig {
                enable_thread_cache: config.enable_thread_caches,
                gc_trigger_threshold: config.gc_trigger_threshold,
                numa_aware: config.numa_awareness.enabled,
            },
            ..Default::default()
        };
        
        let bump_config = BumpAllocatorConfig {
            initial_region_size: 2 * 1024 * 1024, // 2MB
            max_region_size: 16 * 1024 * 1024,   // 16MB
            ..Default::default()
        };
        
        Self {
            small_object_allocator: Arc::new(PrismAllocator::with_config(prism_config)),
            young_allocator: Arc::new(BumpAllocator::with_config(bump_config)),
            large_object_allocator: Arc::new(LargeObjectAllocator::new()),
            page_allocator: Arc::new(PageAllocator::new()),
            thread_cache_manager: Arc::new(ThreadCacheManager::new()),
            config: Arc::new(RwLock::new(config.clone())),
            global_stats: Arc::new(Mutex::new(GlobalAllocationStats::default())),
            pressure_monitor: Arc::new(AllocationPressureMonitor::new(config.gc_trigger_threshold)),
            allocation_mode: Arc::new(RwLock::new(AllocationMode::Normal)),
        }
    }
    
    /// Allocate memory with intelligent allocator selection
    pub fn allocate(&self, size: usize, align: usize, object_type: ObjectType) -> Option<NonNull<u8>> {
        // Check if we should block allocations during GC
        if !self.should_allow_allocation() {
            return None;
        }
        
        // Update allocation pressure monitoring
        self.pressure_monitor.record_allocation(size);
        
        // Select appropriate allocator based on size and type
        let allocator_type = self.select_allocator(size, object_type);
        
        // Perform the allocation
        let result = match allocator_type {
            AllocatorType::SmallObject => {
                self.small_object_allocator.allocate(size, align)
            }
            AllocatorType::Young => {
                self.young_allocator.allocate(size, align)
            }
            AllocatorType::LargeObject => {
                self.large_object_allocator.allocate(size, align)
            }
            AllocatorType::Page => {
                self.page_allocator.allocate_pages(
                    (size + PAGE_SIZE - 1) / PAGE_SIZE
                ).map(|page| NonNull::new(page.addr.as_ptr()).unwrap())
            }
        };
        
        // Update statistics
        if result.is_some() {
            self.update_allocation_stats(allocator_type, size);
        }
        
        result
    }
    
    /// Deallocate memory using the appropriate allocator
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize, allocator_type: AllocatorType) {
        match allocator_type {
            AllocatorType::SmallObject => {
                self.small_object_allocator.deallocate(ptr, size);
            }
            AllocatorType::Young => {
                // Bump allocator doesn't support individual deallocation
                // Objects are reclaimed during GC
            }
            AllocatorType::LargeObject => {
                self.large_object_allocator.deallocate(ptr, size);
            }
            AllocatorType::Page => {
                let page = Page {
                    addr: ptr,
                    count: (size + PAGE_SIZE - 1) / PAGE_SIZE,
                    numa_node: 0,
                };
                self.page_allocator.deallocate_pages(page);
            }
        }
        
        self.update_deallocation_stats(allocator_type, size);
    }
    
    /// Select the appropriate allocator based on object characteristics
    fn select_allocator(&self, size: usize, object_type: ObjectType) -> AllocatorType {
        let config = self.config.read().unwrap();
        
        // Large objects always go to large object allocator
        if size >= config.large_object_threshold {
            return AllocatorType::LargeObject;
        }
        
        // Check if we should use bump allocator for young objects
        match object_type {
            ObjectType::Young | ObjectType::Nursery => {
                if size <= config.bump_allocator_threshold {
                    AllocatorType::Young
                } else {
                    AllocatorType::SmallObject
                }
            }
            ObjectType::Old | ObjectType::Permanent => {
                AllocatorType::SmallObject
            }
            ObjectType::Page => {
                AllocatorType::Page
            }
        }
    }
    
    /// Check if allocation should be allowed in current mode
    fn should_allow_allocation(&self) -> bool {
        let mode = *self.allocation_mode.read().unwrap();
        match mode {
            AllocationMode::Normal | AllocationMode::PostGc => true,
            AllocationMode::GcPreparing => true, // Allow but with caution
            AllocationMode::GcActive => false,   // Block during active GC
        }
    }
    
    /// Update allocation statistics
    fn update_allocation_stats(&self, allocator_type: AllocatorType, size: usize) {
        let mut stats = self.global_stats.lock().unwrap();
        stats.total_allocations += 1;
        stats.total_bytes += size;
        
        let type_stats = stats.allocator_stats.entry(allocator_type).or_default();
        type_stats.allocations += 1;
        type_stats.bytes_allocated += size;
        type_stats.average_allocation_size = 
            type_stats.bytes_allocated as f64 / type_stats.allocations as f64;
    }
    
    /// Update deallocation statistics
    fn update_deallocation_stats(&self, allocator_type: AllocatorType, size: usize) {
        let mut stats = self.global_stats.lock().unwrap();
        // Update any deallocation-specific metrics here
    }
    
    /// Get current allocation statistics
    pub fn get_stats(&self) -> GlobalAllocationStats {
        let mut stats = self.global_stats.lock().unwrap();
        
        // Update thread cache stats
        stats.thread_cache_stats = self.thread_cache_manager.get_stats();
        
        // Update individual allocator stats
        let small_stats = self.small_object_allocator.stats();
        let large_stats = self.large_object_allocator.get_stats();
        let bump_stats = self.young_allocator.get_stats();
        
        stats.clone()
    }
    
    /// Check if garbage collection should be triggered
    pub fn should_trigger_gc(&self) -> bool {
        self.pressure_monitor.should_trigger_gc() ||
        self.small_object_allocator.should_trigger_gc() ||
        self.large_object_allocator.should_trigger_gc()
    }
    
    /// Prepare all allocators for garbage collection
    pub fn prepare_for_gc(&self) {
        *self.allocation_mode.write().unwrap() = AllocationMode::GcPreparing;
        
        self.small_object_allocator.prepare_for_gc();
        self.large_object_allocator.prepare_for_gc();
        self.thread_cache_manager.flush_all_caches();
        
        *self.allocation_mode.write().unwrap() = AllocationMode::GcActive;
    }
    
    /// Reset allocators after garbage collection
    pub fn post_gc_reset(&self) {
        *self.allocation_mode.write().unwrap() = AllocationMode::PostGc;
        
        self.young_allocator.reset();
        self.pressure_monitor.reset();
        
        *self.allocation_mode.write().unwrap() = AllocationMode::Normal;
    }
    
    /// Get memory usage information
    pub fn memory_usage(&self) -> MemoryUsage {
        let small_usage = self.small_object_allocator.memory_usage();
        let large_usage = self.large_object_allocator.memory_usage();
        let bump_usage = self.young_allocator.memory_usage();
        
        MemoryUsage {
            total_allocated: small_usage.total_allocated + large_usage.total_allocated + bump_usage.total_allocated,
            live_bytes: small_usage.live_bytes + large_usage.live_bytes + bump_usage.live_bytes,
            metadata_overhead: small_usage.metadata_overhead + large_usage.metadata_overhead + bump_usage.metadata_overhead,
            fragmentation_ratio: (small_usage.fragmentation_ratio + large_usage.fragmentation_ratio) / 2.0,
        }
    }
    
    /// Get detailed allocator breakdown
    pub fn get_allocator_breakdown(&self) -> AllocatorBreakdown {
        AllocatorBreakdown {
            small_object: self.small_object_allocator.stats(),
            young_generation: self.young_allocator.stats(),
            large_object: self.large_object_allocator.stats(),
            page_allocator: self.page_allocator.stats(),
            thread_cache_stats: self.thread_cache_manager.get_stats(),
        }
    }
    
    /// Update configuration for all allocators
    pub fn update_config(&self, new_config: AllocationManagerConfig) {
        // Update individual allocator configurations
        let prism_config = PrismAllocatorConfig {
            base: AllocatorConfig {
                enable_thread_cache: new_config.enable_thread_caches,
                gc_trigger_threshold: new_config.gc_trigger_threshold,
                numa_aware: new_config.numa_awareness.enabled,
            },
            ..Default::default()
        };
        
        self.small_object_allocator.reconfigure(prism_config.base.clone());
        
        // Update pressure monitor threshold
        self.pressure_monitor.pressure_threshold.store(
            new_config.gc_trigger_threshold, 
            Ordering::Relaxed
        );
        
        // Update page allocator NUMA settings
        if new_config.numa_awareness.enabled {
            self.page_allocator.update_numa_preferences();
        }
        
        // Update manager configuration
        *self.config.write().unwrap() = new_config;
    }
    
    /// Force garbage collection across all allocators
    pub fn force_gc(&self) {
        self.prepare_for_gc();
        
        // Trigger GC on individual allocators that support it
        // In a real implementation, this would coordinate with the GC system
        
        self.post_gc_reset();
    }
    
    /// Get allocation mode
    pub fn get_allocation_mode(&self) -> AllocationMode {
        *self.allocation_mode.read().unwrap()
    }
    
    /// Set allocation mode (for testing/debugging)
    pub fn set_allocation_mode(&self, mode: AllocationMode) {
        *self.allocation_mode.write().unwrap() = mode;
    }
}

impl AllocationPressureMonitor {
    fn new(threshold: usize) -> Self {
        Self {
            allocation_rate: AtomicUsize::new(0),
            bytes_since_gc: AtomicUsize::new(0),
            last_allocation_time: Mutex::new(std::time::Instant::now()),
            pressure_threshold: AtomicUsize::new(threshold),
            adaptive_factor: AtomicUsize::new(1000), // 1.0
        }
    }
    
    fn record_allocation(&self, size: usize) {
        self.bytes_since_gc.fetch_add(size, Ordering::Relaxed);
        
        // Update allocation rate
        let now = std::time::Instant::now();
        let mut last_time = self.last_allocation_time.lock().unwrap();
        let elapsed = now.duration_since(*last_time).as_secs_f64();
        
        if elapsed > 0.0 {
            let rate = size as f64 / elapsed;
            self.allocation_rate.store(rate as usize, Ordering::Relaxed);
        }
        
        *last_time = now;
    }
    
    fn should_trigger_gc(&self) -> bool {
        let bytes = self.bytes_since_gc.load(Ordering::Relaxed);
        let threshold = self.pressure_threshold.load(Ordering::Relaxed);
        let factor = self.adaptive_factor.load(Ordering::Relaxed);
        
        // Apply adaptive threshold
        let adjusted_threshold = (threshold * factor) / 1000;
        
        bytes >= adjusted_threshold
    }
    
    fn reset(&self) {
        self.bytes_since_gc.store(0, Ordering::Relaxed);
    }
}

impl ThreadCacheManager {
    fn new() -> Self {
        Self {
            caches: RwLock::new(HashMap::new()),
            stats: Mutex::new(ThreadCacheStats::default()),
            config: ThreadCacheConfig {
                max_cache_size: THREAD_CACHE_SIZE,
                batch_size: CENTRAL_CACHE_BATCH,
                gc_flush_threshold: THREAD_CACHE_SIZE / 2,
            },
        }
    }
    
    fn get_or_create_cache(&self, thread_id: thread::ThreadId) -> Arc<ThreadCache> {
        let mut caches = self.caches.write().unwrap();
        
        if let Some(cache) = caches.get(&thread_id) {
            cache.clone()
        } else {
            let cache = Arc::new(ThreadCache::new(thread_id, self.config.max_cache_size));
            caches.insert(thread_id, cache.clone());
            cache
        }
    }
    
    fn flush_all_caches(&self) {
        let caches = self.caches.read().unwrap();
        for cache in caches.values() {
            cache.flush();
        }
    }
    
    fn get_stats(&self) -> ThreadCacheStats {
        let caches = self.caches.read().unwrap();
        let active_caches = caches.len();
        
        let mut total_cached_bytes = 0;
        for cache in caches.values() {
            total_cached_bytes += cache.cached_bytes();
        }
        
        ThreadCacheStats {
            active_caches,
            total_cached_bytes,
            cache_hit_rate: 0.95, // Would be calculated from actual metrics
            cache_miss_rate: 0.05,
        }
    }
}

/// Object type hint for allocator selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectType {
    /// Young generation object (likely short-lived)
    Young,
    /// Nursery object (very short-lived)
    Nursery,
    /// Old generation object (long-lived)
    Old,
    /// Permanent object (never collected)
    Permanent,
    /// Page-sized allocation
    Page,
}

/// Memory usage information across all allocators
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub total_allocated: usize,
    pub live_bytes: usize,
    pub metadata_overhead: usize,
    pub fragmentation_ratio: f64,
}

/// Detailed breakdown of all allocator statistics
#[derive(Debug, Clone)]
pub struct AllocatorBreakdown {
    pub small_object: AllocationStats,
    pub young_generation: AllocationStats,
    pub large_object: LargeObjectStats,
    pub page_allocator: AllocationStats,
    pub thread_cache_stats: ThreadCacheStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocator_manager_creation() {
        let manager = AllocatorManager::new();
        assert!(!manager.should_trigger_gc());
    }
    
    #[test]
    fn test_allocator_selection() {
        let manager = AllocatorManager::new();
        
        // Small young object should use bump allocator
        assert_eq!(
            manager.select_allocator(512, ObjectType::Young),
            AllocatorType::Young
        );
        
        // Large object should use large object allocator
        assert_eq!(
            manager.select_allocator(64 * 1024, ObjectType::Young),
            AllocatorType::LargeObject
        );
        
        // Old object should use size-class allocator
        assert_eq!(
            manager.select_allocator(512, ObjectType::Old),
            AllocatorType::SmallObject
        );
    }
    
    #[test]
    fn test_allocation_pressure_monitoring() {
        let monitor = AllocationPressureMonitor::new(1024);
        
        // Should not trigger initially
        assert!(!monitor.should_trigger_gc());
        
        // Record allocations
        monitor.record_allocation(512);
        monitor.record_allocation(600);
        
        // Should trigger after crossing threshold
        assert!(monitor.should_trigger_gc());
        
        // Reset should clear pressure
        monitor.reset();
        assert!(!monitor.should_trigger_gc());
    }
} 