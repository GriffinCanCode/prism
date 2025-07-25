//! Integration layer for write barriers
//!
//! This module provides clean interfaces between the barriers subsystem
//! and other GC components, ensuring loose coupling and proper separation
//! of concerns.

use super::types::*;
use super::implementations::BarrierImplementation;
use std::sync::{Arc, RwLock, Mutex};
use std::collections::HashMap;
use std::time::Instant;

/// Integration layer that coordinates barriers with other GC components
pub struct IntegrationLayer {
    /// Allocator integration
    allocator_integration: Arc<AllocatorIntegration>,
    /// Collector integration
    collector_integration: Arc<CollectorIntegration>,
    /// Heap integration
    heap_integration: Arc<HeapIntegration>,
    /// Barrier hooks for extensibility
    hooks: Arc<BarrierHooks>,
    /// Configuration
    config: BarrierConfig,
    /// Statistics
    stats: Arc<Mutex<IntegrationStats>>,
}

#[derive(Debug, Default, Clone)]
struct IntegrationStats {
    allocator_notifications: u64,
    collector_notifications: u64,
    heap_notifications: u64,
    hook_executions: u64,
    integration_errors: u64,
}

impl IntegrationLayer {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            allocator_integration: Arc::new(AllocatorIntegration::new(config)),
            collector_integration: Arc::new(CollectorIntegration::new(config)),
            heap_integration: Arc::new(HeapIntegration::new(config)),
            hooks: Arc::new(BarrierHooks::new()),
            config: config.clone(),
            stats: Arc::new(Mutex::new(IntegrationStats::default())),
        }
    }

    /// Notify that a barrier was executed
    pub fn notify_barrier_executed(&self, slot: *mut *const u8, new_value: *const u8, old_value: *const u8) {
        // Notify allocator about pointer update
        self.allocator_integration.on_pointer_update(slot, new_value, old_value);
        
        // Notify collector about potential work
        self.collector_integration.on_barrier_executed(slot, new_value, old_value);
        
        // Notify heap about memory access pattern
        self.heap_integration.on_memory_access(slot, new_value, old_value);
        
        // Execute custom hooks
        self.hooks.execute_barrier_hooks(slot, new_value, old_value);
        
        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.allocator_notifications += 1;
            stats.collector_notifications += 1;
            stats.heap_notifications += 1;
        }
    }

    /// Notify that marking phase started
    pub fn notify_marking_started(&self) {
        self.allocator_integration.on_marking_started();
        self.collector_integration.on_marking_started();
        self.heap_integration.on_marking_started();
        self.hooks.execute_marking_started_hooks();
    }

    /// Notify that marking phase stopped
    pub fn notify_marking_stopped(&self) {
        self.allocator_integration.on_marking_stopped();
        self.collector_integration.on_marking_stopped();
        self.heap_integration.on_marking_stopped();
        self.hooks.execute_marking_stopped_hooks();
    }

    /// Prepare for garbage collection
    pub fn prepare_for_collection(&self) {
        self.allocator_integration.prepare_for_collection();
        self.collector_integration.prepare_for_collection();
        self.heap_integration.prepare_for_collection();
        self.hooks.execute_pre_collection_hooks();
    }

    /// Clean up after garbage collection
    pub fn post_collection_cleanup(&self) {
        self.allocator_integration.post_collection_cleanup();
        self.collector_integration.post_collection_cleanup();
        self.heap_integration.post_collection_cleanup();
        self.hooks.execute_post_collection_hooks();
    }

    /// Reconfigure the integration layer
    pub fn reconfigure(&self, new_config: &BarrierConfig) {
        self.allocator_integration.reconfigure(new_config);
        self.collector_integration.reconfigure(new_config);
        self.heap_integration.reconfigure(new_config);
    }

    /// Get integration statistics
    pub fn get_stats(&self) -> BarrierStats {
        let integration_stats = self.stats.lock().unwrap().clone();
        let allocator_stats = self.allocator_integration.get_stats();
        let collector_stats = self.collector_integration.get_stats();
        let heap_stats = self.heap_integration.get_stats();

        // Combine all statistics
        BarrierStats::combine(vec![allocator_stats, collector_stats, heap_stats])
    }
}

/// Integration with the memory allocator
pub struct AllocatorIntegration {
    /// Allocation tracking for GC triggers
    allocation_tracker: Arc<Mutex<AllocationTracker>>,
    /// Configuration
    config: BarrierConfig,
}

#[derive(Debug)]
struct AllocationTracker {
    /// Recent allocations
    recent_allocations: Vec<AllocationInfo>,
    /// Total allocations since last GC
    allocations_since_gc: usize,
    /// Bytes allocated since last GC
    bytes_since_gc: usize,
    /// Last GC timestamp
    last_gc: Instant,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    address: *const u8,
    size: usize,
    timestamp: Instant,
    thread_id: std::thread::ThreadId,
}

impl AllocatorIntegration {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            allocation_tracker: Arc::new(Mutex::new(AllocationTracker {
                recent_allocations: Vec::new(),
                allocations_since_gc: 0,
                bytes_since_gc: 0,
                last_gc: Instant::now(),
            })),
            config: config.clone(),
        }
    }

    /// Called when a pointer is updated via write barrier
    pub fn on_pointer_update(&self, _slot: *mut *const u8, _new_value: *const u8, _old_value: *const u8) {
        // Track pointer updates that might affect allocation patterns
        // This could be used for allocation-site profiling or GC heuristics
    }

    /// Called when marking phase starts
    pub fn on_marking_started(&self) {
        // Could pause allocation or switch to a different allocation strategy
        // For now, just reset tracking
        if let Ok(mut tracker) = self.allocation_tracker.lock() {
            tracker.allocations_since_gc = 0;
            tracker.bytes_since_gc = 0;
            tracker.last_gc = Instant::now();
        }
    }

    /// Called when marking phase stops
    pub fn on_marking_stopped(&self) {
        // Resume normal allocation
    }

    /// Prepare for garbage collection
    pub fn prepare_for_collection(&self) {
        // Could flush allocation buffers, prepare allocation metadata, etc.
        if let Ok(mut tracker) = self.allocation_tracker.lock() {
            tracker.recent_allocations.clear();
        }
    }

    /// Clean up after garbage collection
    pub fn post_collection_cleanup(&self) {
        // Reset allocation tracking
        if let Ok(mut tracker) = self.allocation_tracker.lock() {
            tracker.allocations_since_gc = 0;
            tracker.bytes_since_gc = 0;
            tracker.recent_allocations.clear();
        }
    }

    /// Reconfigure allocator integration
    pub fn reconfigure(&self, new_config: &BarrierConfig) {
        // Update configuration
        // In a real implementation, this might change allocation strategies
    }

    /// Get allocator integration statistics
    pub fn get_stats(&self) -> BarrierStats {
        let mut stats = BarrierStats::new();
        
        if let Ok(tracker) = self.allocation_tracker.lock() {
            stats.metadata_memory = tracker.recent_allocations.len() * std::mem::size_of::<AllocationInfo>();
        }
        
        stats
    }

    /// Record a new allocation
    pub fn record_allocation(&self, address: *const u8, size: usize) {
        if let Ok(mut tracker) = self.allocation_tracker.lock() {
            tracker.recent_allocations.push(AllocationInfo {
                address,
                size,
                timestamp: Instant::now(),
                thread_id: std::thread::current().id(),
            });
            
            tracker.allocations_since_gc += 1;
            tracker.bytes_since_gc += size;
            
            // Keep only recent allocations to avoid unbounded growth
            if tracker.recent_allocations.len() > 10000 {
                tracker.recent_allocations.drain(0..5000);
            }
        }
    }

    /// Check if GC should be triggered based on allocation pressure
    pub fn should_trigger_gc(&self) -> bool {
        if let Ok(tracker) = self.allocation_tracker.lock() {
            // Simple heuristic: trigger GC if we've allocated a lot since last GC
            tracker.bytes_since_gc > 64 * 1024 * 1024 || // 64MB
            tracker.allocations_since_gc > 100000 // 100k objects
        } else {
            false
        }
    }
}

/// Integration with garbage collectors
pub struct CollectorIntegration {
    /// Gray queue for objects that need scanning
    gray_queue: Arc<crossbeam::queue::SegQueue<*const u8>>,
    /// Collection state
    collection_state: Arc<RwLock<CollectionState>>,
    /// Configuration
    config: BarrierConfig,
}

#[derive(Debug, Clone)]
enum CollectionState {
    Idle,
    Marking,
    Sweeping,
    Finalizing,
}

impl CollectorIntegration {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            gray_queue: Arc::new(crossbeam::queue::SegQueue::new()),
            collection_state: Arc::new(RwLock::new(CollectionState::Idle)),
            config: config.clone(),
        }
    }

    /// Called when a barrier is executed
    pub fn on_barrier_executed(&self, _slot: *mut *const u8, new_value: *const u8, _old_value: *const u8) {
        // If we're marking and the new value is an object, add it to gray queue
        if let Ok(state) = self.collection_state.read() {
            if matches!(*state, CollectionState::Marking) && !new_value.is_null() {
                self.gray_queue.push(new_value);
            }
        }
    }

    /// Called when marking phase starts
    pub fn on_marking_started(&self) {
        if let Ok(mut state) = self.collection_state.write() {
            *state = CollectionState::Marking;
        }
    }

    /// Called when marking phase stops
    pub fn on_marking_stopped(&self) {
        if let Ok(mut state) = self.collection_state.write() {
            *state = CollectionState::Sweeping;
        }
    }

    /// Prepare for garbage collection
    pub fn prepare_for_collection(&self) {
        // Clear gray queue
        while self.gray_queue.pop().is_some() {}
        
        if let Ok(mut state) = self.collection_state.write() {
            *state = CollectionState::Idle;
        }
    }

    /// Clean up after garbage collection
    pub fn post_collection_cleanup(&self) {
        if let Ok(mut state) = self.collection_state.write() {
            *state = CollectionState::Idle;
        }
    }

    /// Reconfigure collector integration
    pub fn reconfigure(&self, _new_config: &BarrierConfig) {
        // Update configuration
    }

    /// Get collector integration statistics
    pub fn get_stats(&self) -> BarrierStats {
        let mut stats = BarrierStats::new();
        stats.memory_overhead = self.gray_queue.len() * std::mem::size_of::<*const u8>();
        stats
    }

    /// Drain gray queue for collector processing
    pub fn drain_gray_queue(&self) -> Vec<*const u8> {
        let mut objects = Vec::new();
        while let Some(obj) = self.gray_queue.pop() {
            objects.push(obj);
        }
        objects
    }

    /// Get current collection state
    pub fn get_collection_state(&self) -> CollectionState {
        self.collection_state.read().unwrap().clone()
    }
}

/// Integration with heap management
pub struct HeapIntegration {
    /// Memory access patterns for optimization
    access_patterns: Arc<Mutex<AccessPatternTracker>>,
    /// Configuration
    config: BarrierConfig,
}

#[derive(Debug)]
struct AccessPatternTracker {
    /// Recent memory accesses
    recent_accesses: Vec<MemoryAccess>,
    /// Hot memory regions
    hot_regions: HashMap<usize, HotRegion>, // region_id -> HotRegion
    /// Last analysis timestamp
    last_analysis: Instant,
}

#[derive(Debug, Clone)]
struct MemoryAccess {
    address: *const u8,
    access_type: AccessType,
    timestamp: Instant,
    thread_id: std::thread::ThreadId,
}

#[derive(Debug, Clone)]
enum AccessType {
    Read,
    Write,
    BarrierTriggered,
}

#[derive(Debug, Clone)]
struct HotRegion {
    start_address: *const u8,
    size: usize,
    access_count: usize,
    last_access: Instant,
}

impl HeapIntegration {
    pub fn new(config: &BarrierConfig) -> Self {
        Self {
            access_patterns: Arc::new(Mutex::new(AccessPatternTracker {
                recent_accesses: Vec::new(),
                hot_regions: HashMap::new(),
                last_analysis: Instant::now(),
            })),
            config: config.clone(),
        }
    }

    /// Called when memory is accessed via write barrier
    pub fn on_memory_access(&self, slot: *mut *const u8, _new_value: *const u8, _old_value: *const u8) {
        if let Ok(mut tracker) = self.access_patterns.lock() {
            tracker.recent_accesses.push(MemoryAccess {
                address: slot as *const u8,
                access_type: AccessType::BarrierTriggered,
                timestamp: Instant::now(),
                thread_id: std::thread::current().id(),
            });
            
            // Limit memory usage
            if tracker.recent_accesses.len() > 50000 {
                tracker.recent_accesses.drain(0..25000);
            }
            
            // Periodic analysis of access patterns
            if tracker.last_analysis.elapsed().as_secs() > 10 {
                self.analyze_access_patterns(&mut tracker);
                tracker.last_analysis = Instant::now();
            }
        }
    }

    /// Analyze memory access patterns to identify hot regions
    fn analyze_access_patterns(&self, tracker: &mut AccessPatternTracker) {
        // Simple analysis: group accesses by memory region
        let mut region_counts: HashMap<usize, usize> = HashMap::new();
        
        for access in &tracker.recent_accesses {
            let region_id = (access.address as usize) / (64 * 1024); // 64KB regions
            *region_counts.entry(region_id).or_insert(0) += 1;
        }
        
        // Update hot regions
        for (region_id, count) in region_counts {
            if count > 100 { // Threshold for "hot"
                let region = HotRegion {
                    start_address: (region_id * 64 * 1024) as *const u8,
                    size: 64 * 1024,
                    access_count: count,
                    last_access: Instant::now(),
                };
                tracker.hot_regions.insert(region_id, region);
            }
        }
        
        // Remove old hot regions
        tracker.hot_regions.retain(|_, region| {
            region.last_access.elapsed().as_secs() < 60 // Keep for 1 minute
        });
    }

    /// Called when marking phase starts
    pub fn on_marking_started(&self) {
        // Could prepare heap metadata for concurrent marking
    }

    /// Called when marking phase stops
    pub fn on_marking_stopped(&self) {
        // Could optimize heap layout based on access patterns
    }

    /// Prepare for garbage collection
    pub fn prepare_for_collection(&self) {
        // Could compact hot regions or prepare heap metadata
    }

    /// Clean up after garbage collection
    pub fn post_collection_cleanup(&self) {
        // Reset access pattern tracking
        if let Ok(mut tracker) = self.access_patterns.lock() {
            tracker.recent_accesses.clear();
            // Keep hot regions for next collection cycle
        }
    }

    /// Reconfigure heap integration
    pub fn reconfigure(&self, _new_config: &BarrierConfig) {
        // Update configuration
    }

    /// Get heap integration statistics
    pub fn get_stats(&self) -> BarrierStats {
        let mut stats = BarrierStats::new();
        
        if let Ok(tracker) = self.access_patterns.lock() {
            stats.metadata_memory += tracker.recent_accesses.len() * std::mem::size_of::<MemoryAccess>();
            stats.metadata_memory += tracker.hot_regions.len() * std::mem::size_of::<HotRegion>();
        }
        
        stats
    }

    /// Get hot memory regions for optimization
    pub fn get_hot_regions(&self) -> Vec<HotRegion> {
        if let Ok(tracker) = self.access_patterns.lock() {
            tracker.hot_regions.values().cloned().collect()
        } else {
            Vec::new()
        }
    }
}

/// Extensible hook system for custom barrier behavior
pub struct BarrierHooks {
    /// Hooks called when barriers are executed
    barrier_hooks: RwLock<Vec<Box<dyn Fn(*mut *const u8, *const u8, *const u8) + Send + Sync>>>,
    /// Hooks called when marking starts
    marking_started_hooks: RwLock<Vec<Box<dyn Fn() + Send + Sync>>>,
    /// Hooks called when marking stops
    marking_stopped_hooks: RwLock<Vec<Box<dyn Fn() + Send + Sync>>>,
    /// Hooks called before collection
    pre_collection_hooks: RwLock<Vec<Box<dyn Fn() + Send + Sync>>>,
    /// Hooks called after collection
    post_collection_hooks: RwLock<Vec<Box<dyn Fn() + Send + Sync>>>,
}

impl BarrierHooks {
    pub fn new() -> Self {
        Self {
            barrier_hooks: RwLock::new(Vec::new()),
            marking_started_hooks: RwLock::new(Vec::new()),
            marking_stopped_hooks: RwLock::new(Vec::new()),
            pre_collection_hooks: RwLock::new(Vec::new()),
            post_collection_hooks: RwLock::new(Vec::new()),
        }
    }

    /// Add a hook that's called when barriers are executed
    pub fn add_barrier_hook<F>(&self, hook: F) 
    where 
        F: Fn(*mut *const u8, *const u8, *const u8) + Send + Sync + 'static 
    {
        if let Ok(mut hooks) = self.barrier_hooks.write() {
            hooks.push(Box::new(hook));
        }
    }

    /// Add a hook that's called when marking starts
    pub fn add_marking_started_hook<F>(&self, hook: F) 
    where 
        F: Fn() + Send + Sync + 'static 
    {
        if let Ok(mut hooks) = self.marking_started_hooks.write() {
            hooks.push(Box::new(hook));
        }
    }

    /// Execute all barrier hooks
    pub fn execute_barrier_hooks(&self, slot: *mut *const u8, new_value: *const u8, old_value: *const u8) {
        if let Ok(hooks) = self.barrier_hooks.read() {
            for hook in hooks.iter() {
                hook(slot, new_value, old_value);
            }
        }
    }

    /// Execute all marking started hooks
    pub fn execute_marking_started_hooks(&self) {
        if let Ok(hooks) = self.marking_started_hooks.read() {
            for hook in hooks.iter() {
                hook();
            }
        }
    }

    /// Execute all marking stopped hooks
    pub fn execute_marking_stopped_hooks(&self) {
        if let Ok(hooks) = self.marking_stopped_hooks.read() {
            for hook in hooks.iter() {
                hook();
            }
        }
    }

    /// Execute all pre-collection hooks
    pub fn execute_pre_collection_hooks(&self) {
        if let Ok(hooks) = self.pre_collection_hooks.read() {
            for hook in hooks.iter() {
                hook();
            }
        }
    }

    /// Execute all post-collection hooks
    pub fn execute_post_collection_hooks(&self) {
        if let Ok(hooks) = self.post_collection_hooks.read() {
            for hook in hooks.iter() {
                hook();
            }
        }
    }
}

/// Manager for coordinating all barrier integrations
pub struct BarrierManager {
    /// Integration layer
    integration: Arc<IntegrationLayer>,
    /// Active barrier implementation
    barrier: Arc<dyn BarrierImplementation>,
}

impl BarrierManager {
    pub fn new(
        integration: Arc<IntegrationLayer>,
        barrier: Arc<dyn BarrierImplementation>,
    ) -> Self {
        Self {
            integration,
            barrier,
        }
    }

    /// Execute a coordinated write barrier
    pub fn coordinated_write_barrier(
        &self,
        slot: *mut *const u8,
        new_value: *const u8,
        old_value: *const u8,
    ) {
        // Execute the actual barrier
        self.barrier.write_barrier(slot, new_value, old_value);
        
        // Notify integration layer
        self.integration.notify_barrier_executed(slot, new_value, old_value);
    }

    /// Get the integration layer
    pub fn get_integration(&self) -> &IntegrationLayer {
        &self.integration
    }

    /// Get the barrier implementation
    pub fn get_barrier(&self) -> &dyn BarrierImplementation {
        &*self.barrier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_integration() {
        let config = BarrierConfig::default();
        let integration = AllocatorIntegration::new(&config);
        
        // Record some allocations
        integration.record_allocation(0x1000 as *const u8, 64);
        integration.record_allocation(0x2000 as *const u8, 128);
        
        let stats = integration.get_stats();
        assert!(stats.metadata_memory > 0);
    }

    #[test]
    fn test_collector_integration() {
        let config = BarrierConfig::default();
        let integration = CollectorIntegration::new(&config);
        
        // Start marking
        integration.on_marking_started();
        assert!(matches!(integration.get_collection_state(), CollectionState::Marking));
        
        // Execute barrier
        integration.on_barrier_executed(
            std::ptr::null_mut(),
            0x1000 as *const u8,
            std::ptr::null()
        );
        
        // Should have objects in gray queue
        let objects = integration.drain_gray_queue();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0], 0x1000 as *const u8);
    }

    #[test]
    fn test_heap_integration() {
        let config = BarrierConfig::default();
        let integration = HeapIntegration::new(&config);
        
        // Record memory accesses
        for i in 0..10 {
            integration.on_memory_access(
                (0x1000 + i * 8) as *mut *const u8,
                std::ptr::null(),
                std::ptr::null()
            );
        }
        
        let stats = integration.get_stats();
        assert!(stats.metadata_memory > 0);
    }

    #[test]
    fn test_barrier_hooks() {
        let hooks = BarrierHooks::new();
        
        // Add a barrier hook
        let hook_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_called_clone = hook_called.clone();
        
        hooks.add_barrier_hook(move |_, _, _| {
            hook_called_clone.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        
        // Execute hooks
        hooks.execute_barrier_hooks(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null()
        );
        
        assert!(hook_called.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn test_integration_layer() {
        let config = BarrierConfig::default();
        let integration = IntegrationLayer::new(&config);
        
        // Test barrier execution notification
        integration.notify_barrier_executed(
            std::ptr::null_mut(),
            0x1000 as *const u8,
            std::ptr::null()
        );
        
        // Test marking lifecycle
        integration.notify_marking_started();
        integration.notify_marking_stopped();
        
        // Test collection lifecycle
        integration.prepare_for_collection();
        integration.post_collection_cleanup();
        
        let stats = integration.get_stats();
        assert!(stats.barrier_calls >= 0); // Should not crash
    }
} 