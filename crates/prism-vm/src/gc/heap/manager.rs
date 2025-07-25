//! Heap Manager - Central coordinator for all heap operations
//!
//! This manager provides a unified interface for heap operations while ensuring
//! proper delegation to specialized components. It maintains clear boundaries:
//!
//! **HeapManager Responsibilities:**
//! - Coordinate between heap subsystem components
//! - Provide unified interface for heap operations
//! - Manage heap lifecycle and configuration
//! - Aggregate statistics from all heap components
//!
//! **Delegated Responsibilities:**
//! - Raw allocation: Delegated to allocators::AllocatorManager
//! - Object marking: Delegated to collectors::*
//! - Write barriers: Delegated to barriers::*
//! - Root management: Delegated to roots::RootSet

use super::*;
use super::types::*;
use super::core::Heap;
use super::size_class::SizeClassAllocator;
use super::large_object::LargeObjectAllocator as HeapLargeObjectAllocator;
use super::memory_regions::MemoryRegionManager;
use super::fragmentation::FragmentationManager;
use super::card_table::CardTable;
use super::statistics::StatisticsCollector;
use super::regional::RegionalHeap;

use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::collections::HashMap;

/// Central heap manager that coordinates all heap operations
/// 
/// This manager serves as the main entry point for heap operations,
/// coordinating between different heap components while maintaining
/// clear separation from other GC subsystems.
pub struct HeapManager {
    /// Current heap implementation
    heap: Arc<RwLock<Box<dyn HeapInterface>>>,
    
    /// Configuration for heap behavior
    config: Arc<RwLock<HeapConfig>>,
    
    /// Statistics collector for monitoring
    statistics: Arc<StatisticsCollector>,
    
    /// Fragmentation manager for compaction decisions
    fragmentation_manager: Arc<FragmentationManager>,
    
    /// Memory region manager (if enabled)
    region_manager: Option<Arc<MemoryRegionManager>>,
    
    /// Card table for generational GC (if enabled)
    card_table: Option<Arc<CardTable>>,
    
    /// Current heap state
    state: Arc<RwLock<HeapState>>,
    
    /// Performance monitoring
    performance_monitor: Arc<HeapPerformanceMonitor>,
}

/// Current state of the heap
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeapState {
    /// Normal operation
    Active,
    /// Preparing for garbage collection
    PreparingForGc,
    /// Garbage collection in progress
    GcInProgress,
    /// Post-GC cleanup
    PostGc,
    /// Heap is being compacted
    Compacting,
}

/// Performance monitor for heap operations
pub struct HeapPerformanceMonitor {
    /// Track allocation patterns
    allocation_tracker: Mutex<AllocationTracker>,
    /// Track fragmentation over time
    fragmentation_tracker: Mutex<FragmentationTracker>,
    /// Track GC impact on heap
    gc_impact_tracker: Mutex<GcImpactTracker>,
    /// Monitor creation time for uptime calculation
    creation_time: std::time::Instant,
}

#[derive(Debug, Default)]
struct AllocationTracker {
    recent_allocations: Vec<AllocationEvent>,
    allocation_rate: f64,
    peak_allocation_rate: f64,
    last_update: Option<std::time::Instant>,
}

#[derive(Debug)]
struct AllocationEvent {
    size: usize,
    timestamp: std::time::Instant,
    allocation_type: AllocationType,
}

#[derive(Debug, Clone, Copy)]
enum AllocationType {
    SmallObject,
    LargeObject,
    FromFreeList,
    NewAllocation,
}

#[derive(Debug, Default)]
struct FragmentationTracker {
    fragmentation_history: Vec<FragmentationSnapshot>,
    compaction_events: Vec<CompactionEvent>,
}

#[derive(Debug)]
struct FragmentationSnapshot {
    timestamp: std::time::Instant,
    fragmentation_ratio: f64,
    largest_free_block: usize,
    total_free_space: usize,
}

#[derive(Debug)]
struct CompactionEvent {
    timestamp: std::time::Instant,
    bytes_compacted: usize,
    duration: std::time::Duration,
    fragmentation_before: f64,
    fragmentation_after: f64,
}

#[derive(Debug, Default)]
struct GcImpactTracker {
    gc_events: Vec<GcEvent>,
    total_gc_time: std::time::Duration,
    objects_collected: usize,
    bytes_reclaimed: usize,
}

#[derive(Debug)]
struct GcEvent {
    timestamp: std::time::Instant,
    duration: std::time::Duration,
    objects_before: usize,
    objects_after: usize,
    bytes_reclaimed: usize,
}

impl HeapManager {
    /// Create a new heap manager with the given configuration
    pub fn new(config: HeapConfig) -> Self {
        let heap: Box<dyn HeapInterface> = match config.heap_type {
            HeapType::Standard => Box::new(Heap::with_config(config.clone())),
            HeapType::Regional => Box::new(RegionalHeap::with_config(config.clone())),
        };
        
        let statistics = Arc::new(StatisticsCollector::new());
        let fragmentation_manager = Arc::new(FragmentationManager::new(config.compaction_threshold));
        
        let region_manager = if config.enable_memory_regions {
            Some(Arc::new(MemoryRegionManager::new(config.max_memory_regions)))
        } else {
            None
        };
        
        let card_table = if config.enable_card_table {
            // Initialize with proper heap base address
            // In a real implementation, this would be coordinated with the allocator
            let heap_base = 0x10000000 as *const u8; // Placeholder base address
            Some(Arc::new(CardTable::new(
                heap_base,
                config.capacity,
                config.card_size,
            )))
        } else {
            None
        };
        
        Self {
            heap: Arc::new(RwLock::new(heap)),
            config: Arc::new(RwLock::new(config)),
            statistics,
            fragmentation_manager,
            region_manager,
            card_table,
            state: Arc::new(RwLock::new(HeapState::Active)),
            performance_monitor: Arc::new(HeapPerformanceMonitor::new()),
        }
    }
    
    /// Register a newly allocated object in the heap
    /// 
    /// This is called by the allocator after successful allocation
    /// to track the object in the heap's data structures.
    pub fn register_object(&self, ptr: *const u8, header: crate::ObjectHeader) -> Result<(), HeapError> {
        // Validate the allocation
        self.validate_object_registration(ptr, &header)?;
        
        // Register with the main heap
        {
            let mut heap = self.heap.write().unwrap();
            heap.register_object(ptr, header);
        }
        
        // Update memory regions if enabled
        if let Some(ref region_manager) = self.region_manager {
            region_manager.track_allocation(ptr, header.size);
        }
        
        // Mark card table if enabled
        if let Some(ref card_table) = self.card_table {
            card_table.mark_dirty(ptr);
        }
        
        // Update statistics
        self.statistics.record_object_registration(header.size);
        
        // Track allocation pattern
        self.performance_monitor.record_allocation(header.size, AllocationType::NewAllocation);
        
        Ok(())
    }
    
    /// Get object header for a pointer
    pub fn get_object_header(&self, ptr: *const u8) -> Option<crate::ObjectHeader> {
        let heap = self.heap.read().unwrap();
        heap.get_header(ptr).copied()
    }
    
    /// Try to allocate from heap free lists
    /// 
    /// This method only handles free list allocation - it does NOT
    /// perform raw memory allocation (that's handled by allocators).
    pub fn try_allocate_from_free_list(&self, size: usize, align: usize) -> Option<*const u8> {
        let mut heap = self.heap.write().unwrap();
        let result = heap.try_allocate_from_free_list(size, align);
        
        if result.is_some() {
            // Update statistics
            self.statistics.record_free_list_hit();
            
            // Track allocation pattern
            self.performance_monitor.record_allocation(size, AllocationType::FromFreeList);
        }
        
        result
    }
    
    /// Deallocate an object (add to free lists)
    /// 
    /// This method handles returning memory to heap free lists - it does NOT
    /// perform raw memory deallocation (that's handled by allocators).
    pub fn deallocate_object(&self, ptr: *const u8) -> Result<(), HeapError> {
        // Get object size before deallocation
        let object_size = {
            let heap = self.heap.read().unwrap();
            heap.get_header(ptr)
                .map(|h| h.size)
                .ok_or(HeapError::InvalidPointer)?
        };
        
        // Deallocate from main heap
        {
            let mut heap = self.heap.write().unwrap();
            heap.deallocate(ptr);
        }
        
        // Update memory regions if enabled
        if let Some(ref region_manager) = self.region_manager {
            region_manager.track_deallocation(ptr, object_size);
        }
        
        // Update statistics
        self.statistics.record_object_deallocation(object_size);
        
        Ok(())
    }
    
    /// Get current heap statistics
    pub fn get_statistics(&self) -> HeapStats {
        let heap = self.heap.read().unwrap();
        let mut stats = heap.get_stats();
        
        // Add region statistics if enabled
        if let Some(ref region_manager) = self.region_manager {
            stats.region_stats = Some(region_manager.get_statistics());
        }
        
        // Add card table statistics if enabled
        if let Some(ref card_table) = self.card_table {
            stats.card_table_stats = Some(card_table.get_statistics());
        }
        
        // Update with performance monitor data
        self.performance_monitor.update_statistics(&mut stats);
        
        stats
    }
    
    /// Prepare heap for garbage collection
    /// 
    /// This coordinates with other GC components but doesn't perform
    /// the actual collection (that's handled by collectors).
    pub fn prepare_for_gc(&self) -> Result<(), HeapError> {
        // Update state
        *self.state.write().unwrap() = HeapState::PreparingForGc;
        
        // Let heap implementation prepare
        {
            let mut heap = self.heap.write().unwrap();
            heap.reset_colors_to_white();
        }
        
        // Clear card table if enabled (will be marked dirty during GC)
        if let Some(ref card_table) = self.card_table {
            card_table.clear_all();
        }
        
        // Update state
        *self.state.write().unwrap() = HeapState::GcInProgress;
        
        Ok(())
    }
    
    /// Complete garbage collection cycle
    pub fn complete_gc(&self, collected_objects: usize, bytes_reclaimed: usize) -> Result<(), HeapError> {
        // Update state
        *self.state.write().unwrap() = HeapState::PostGc;
        
        // Update statistics
        self.statistics.record_gc_completion(collected_objects, bytes_reclaimed);
        
        // Track GC impact
        self.performance_monitor.record_gc_event(collected_objects, bytes_reclaimed);
        
        // Check if compaction is needed
        if self.should_compact()? {
            self.trigger_compaction()?;
        }
        
        // Return to active state
        *self.state.write().unwrap() = HeapState::Active;
        
        Ok(())
    }
    
    /// Find all unmarked objects for collection
    /// 
    /// This is called by collectors during the sweep phase
    pub fn find_unmarked_objects(&self) -> Vec<*const u8> {
        let heap = self.heap.read().unwrap();
        heap.find_white_objects()
    }
    
    /// Check if compaction should be triggered
    pub fn should_compact(&self) -> Result<bool, HeapError> {
        let stats = self.get_statistics();
        let config = self.config.read().unwrap();
        
        Ok(stats.fragmentation_ratio > config.compaction_threshold)
    }
    
    /// Trigger heap compaction
    pub fn trigger_compaction(&self) -> Result<usize, HeapError> {
        // Update state
        *self.state.write().unwrap() = HeapState::Compacting;
        
        let start_time = std::time::Instant::now();
        let fragmentation_before = self.get_statistics().fragmentation_ratio;
        
        // Perform compaction
        let bytes_compacted = {
            let mut heap = self.heap.write().unwrap();
            heap.compact()
        };
        
        // Let fragmentation manager handle the compaction
        self.fragmentation_manager.perform_compaction();
        
        let duration = start_time.elapsed();
        let fragmentation_after = self.get_statistics().fragmentation_ratio;
        
        // Record compaction event
        self.performance_monitor.record_compaction(
            bytes_compacted,
            duration,
            fragmentation_before,
            fragmentation_after,
        );
        
        // Return to previous state
        *self.state.write().unwrap() = HeapState::Active;
        
        Ok(bytes_compacted)
    }
    
    /// Get current heap state
    pub fn get_state(&self) -> HeapState {
        *self.state.read().unwrap()
    }
    
    /// Check if heap needs collection
    pub fn needs_collection(&self, threshold: f64) -> bool {
        let heap = self.heap.read().unwrap();
        heap.needs_collection(threshold)
    }
    
    /// Get memory pressure indicator
    pub fn memory_pressure(&self) -> f64 {
        let heap = self.heap.read().unwrap();
        heap.memory_pressure()
    }
    
    /// Reconfigure the heap
    pub fn reconfigure(&self, new_config: HeapConfig) -> Result<(), HeapError> {
        let mut config = self.config.write().unwrap();
        *config = new_config;
        
        // Reconfigure components as needed
        // Note: Some configuration changes may require heap recreation
        
        Ok(())
    }
    
    /// Validate object registration
    fn validate_object_registration(&self, ptr: *const u8, header: &crate::ObjectHeader) -> Result<(), HeapError> {
        if ptr.is_null() {
            return Err(HeapError::InvalidPointer);
        }
        
        if header.size == 0 {
            return Err(HeapError::InvalidObjectSize);
        }
        
        // Additional validation can be added here
        
        Ok(())
    }
    
    /// Verify heap integrity (debug builds only)
    #[cfg(debug_assertions)]
    pub fn verify_integrity(&self) -> Result<(), String> {
        let heap = self.heap.read().unwrap();
        heap.verify_integrity()
    }
}

impl HeapPerformanceMonitor {
    fn new() -> Self {
        Self {
            allocation_tracker: Mutex::new(AllocationTracker::default()),
            fragmentation_tracker: Mutex::new(FragmentationTracker::default()),
            gc_impact_tracker: Mutex::new(GcImpactTracker::default()),
            creation_time: std::time::Instant::now(),
        }
    }
    
    fn record_allocation(&self, size: usize, allocation_type: AllocationType) {
        let mut tracker = self.allocation_tracker.lock().unwrap();
        let now = std::time::Instant::now();
        
        tracker.recent_allocations.push(AllocationEvent {
            size,
            timestamp: now,
            allocation_type,
        });
        
        // Keep only recent allocations (last 1000)
        if tracker.recent_allocations.len() > 1000 {
            tracker.recent_allocations.drain(0..500);
        }
        
        // Update allocation rate
        self.update_allocation_rate(&mut tracker, now);
    }
    
    fn update_allocation_rate(&self, tracker: &mut AllocationTracker, now: std::time::Instant) {
        if let Some(last_update) = tracker.last_update {
            let elapsed = now.duration_since(last_update).as_secs_f64();
            if elapsed > 1.0 { // Update every second
                let recent_bytes: usize = tracker.recent_allocations
                    .iter()
                    .filter(|e| now.duration_since(e.timestamp).as_secs_f64() < 1.0)
                    .map(|e| e.size)
                    .sum();
                
                tracker.allocation_rate = recent_bytes as f64 / elapsed;
                tracker.peak_allocation_rate = tracker.peak_allocation_rate.max(tracker.allocation_rate);
                tracker.last_update = Some(now);
            }
        } else {
            tracker.last_update = Some(now);
        }
    }
    
    fn record_compaction(&self, bytes_compacted: usize, duration: std::time::Duration, 
                        fragmentation_before: f64, fragmentation_after: f64) {
        let mut tracker = self.fragmentation_tracker.lock().unwrap();
        tracker.compaction_events.push(CompactionEvent {
            timestamp: std::time::Instant::now(),
            bytes_compacted,
            duration,
            fragmentation_before,
            fragmentation_after,
        });
    }
    
    fn record_gc_event(&self, objects_collected: usize, bytes_reclaimed: usize) {
        let mut tracker = self.gc_impact_tracker.lock().unwrap();
        let now = std::time::Instant::now();
        
        tracker.gc_events.push(GcEvent {
            timestamp: now,
            duration: std::time::Duration::from_millis(0), // Would be filled by collector
            objects_before: 0, // Would be filled by collector
            objects_after: 0,  // Would be filled by collector
            bytes_reclaimed,
        });
        
        tracker.objects_collected += objects_collected;
        tracker.bytes_reclaimed += bytes_reclaimed;
    }
    
    fn update_statistics(&self, stats: &mut HeapStats) {
        let allocation_tracker = self.allocation_tracker.lock().unwrap();
        stats.allocation_rate = allocation_tracker.allocation_rate;
        
        let gc_tracker = self.gc_impact_tracker.lock().unwrap();
        if !gc_tracker.gc_events.is_empty() {
            let total_time = gc_tracker.total_gc_time.as_secs_f64();
            let uptime = self.creation_time.elapsed().as_secs_f64();
            
            // Avoid division by zero and ensure reasonable overhead calculation
            if uptime > 0.0 {
                stats.gc_overhead = ((total_time / uptime) * 100.0).min(100.0);
            } else {
                stats.gc_overhead = 0.0;
            }
        }
    }
}

/// Errors that can occur during heap operations
#[derive(Debug, Clone)]
pub enum HeapError {
    /// Invalid pointer provided
    InvalidPointer,
    /// Invalid object size
    InvalidObjectSize,
    /// Heap is in wrong state for operation
    InvalidState,
    /// Configuration error
    ConfigurationError(String),
    /// Internal heap error
    InternalError(String),
}

impl std::fmt::Display for HeapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeapError::InvalidPointer => write!(f, "Invalid pointer provided to heap"),
            HeapError::InvalidObjectSize => write!(f, "Invalid object size"),
            HeapError::InvalidState => write!(f, "Heap is in wrong state for operation"),
            HeapError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            HeapError::InternalError(msg) => write!(f, "Internal heap error: {}", msg),
        }
    }
}

impl std::error::Error for HeapError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_manager_creation() {
        let config = HeapConfig::default();
        let manager = HeapManager::new(config);
        
        assert_eq!(manager.get_state(), HeapState::Active);
        assert!(!manager.needs_collection(0.8));
    }

    #[test]
    fn test_heap_state_transitions() {
        let config = HeapConfig::default();
        let manager = HeapManager::new(config);
        
        // Test GC preparation
        manager.prepare_for_gc().unwrap();
        assert_eq!(manager.get_state(), HeapState::GcInProgress);
        
        // Test GC completion
        manager.complete_gc(10, 1024).unwrap();
        assert_eq!(manager.get_state(), HeapState::Active);
    }

    #[test]
    fn test_object_registration_validation() {
        let config = HeapConfig::default();
        let manager = HeapManager::new(config);
        
        // Test null pointer validation
        let header = crate::ObjectHeader {
            size: 64,
            type_id: 1,
            mark_bits: 0,
            generation: 0,
            ref_count: 0,
        };
        
        let result = manager.register_object(std::ptr::null(), header);
        assert!(matches!(result, Err(HeapError::InvalidPointer)));
        
        // Test zero size validation
        let header = crate::ObjectHeader {
            size: 0,
            type_id: 1,
            mark_bits: 0,
            generation: 0,
            ref_count: 0,
        };
        
        let result = manager.register_object(0x1000 as *const u8, header);
        assert!(matches!(result, Err(HeapError::InvalidObjectSize)));
    }
} 