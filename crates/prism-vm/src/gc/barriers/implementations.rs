//! Write barrier implementations
//!
//! This module contains the core write barrier algorithms:
//! - Dijkstra insertion barrier for strong tri-color invariant
//! - Yuasa deletion barrier for weak tri-color invariant  
//! - Hybrid barrier combining both approaches (Go-style)
//! - No-op barrier for stop-the-world collection

use super::types::*;
use super::{ObjectHeader, ObjectColor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use crossbeam::queue::SegQueue;

/// Trait that all write barrier implementations must satisfy
pub trait BarrierImplementation: Send + Sync {
    /// Execute a write barrier for the given pointer update
    fn write_barrier(&self, slot: *mut *const u8, new_value: *const u8, old_value: *const u8);
    
    /// Enable marking phase - barriers become active
    fn enable_marking(&self);
    
    /// Disable marking phase - barriers become inactive
    fn disable_marking(&self);
    
    /// Check if marking is currently active
    fn is_marking_active(&self) -> bool;
    
    /// Get statistics for this barrier implementation
    fn get_stats(&self) -> BarrierStats;
    
    /// Reset statistics
    fn reset_stats(&self);
    
    /// Get the barrier type
    fn barrier_type(&self) -> WriteBarrierType;
    
    /// Drain any queued objects for processing by collector
    fn drain_gray_queue(&self) -> Vec<*const u8>;
    
    /// Get estimated memory overhead of this barrier
    fn memory_overhead(&self) -> usize;
}

/// No-op barrier for stop-the-world collection
/// 
/// This barrier does nothing, which is safe when all mutator threads
/// are stopped during garbage collection.
pub struct NoBarrier {
    stats: AtomicBarrierStats,
}

impl NoBarrier {
    pub fn new() -> Self {
        Self {
            stats: AtomicBarrierStats::new(),
        }
    }
}

impl BarrierImplementation for NoBarrier {
    fn write_barrier(&self, _slot: *mut *const u8, _new_value: *const u8, _old_value: *const u8) {
        // No-op - just update statistics
        self.stats.barrier_calls.fetch_add(1, Ordering::Relaxed);
        self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
    }
    
    fn enable_marking(&self) {
        // No-op for stop-the-world
    }
    
    fn disable_marking(&self) {
        // No-op for stop-the-world
    }
    
    fn is_marking_active(&self) -> bool {
        false
    }
    
    fn get_stats(&self) -> BarrierStats {
        self.stats.to_stats()
    }
    
    fn reset_stats(&self) {
        *self = Self::new();
    }
    
    fn barrier_type(&self) -> WriteBarrierType {
        WriteBarrierType::None
    }
    
    fn drain_gray_queue(&self) -> Vec<*const u8> {
        Vec::new()
    }
    
    fn memory_overhead(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Dijkstra insertion barrier implementation
/// 
/// Maintains the strong tri-color invariant by ensuring that black objects
/// never point directly to white objects. When a black object gets a new
/// pointer to a white object, the white object is immediately marked gray.
pub struct DijkstraBarrier {
    /// Whether marking is currently active
    is_marking: AtomicBool,
    /// Queue of objects that need to be scanned (gray objects)
    gray_queue: Arc<SegQueue<*const u8>>,
    /// Statistics
    stats: AtomicBarrierStats,
    /// Configuration
    config: BarrierConfig,
}

impl DijkstraBarrier {
    pub fn new(config: BarrierConfig) -> Self {
        Self {
            is_marking: AtomicBool::new(false),
            gray_queue: Arc::new(SegQueue::new()),
            stats: AtomicBarrierStats::new(),
            config,
        }
    }
    
    /// Get object header from pointer (unsafe operation)
    unsafe fn get_object_header(&self, ptr: *const u8) -> Option<&mut ObjectHeader> {
        if ptr.is_null() {
            return None;
        }
        
        // Object header is at the beginning of each allocation
        let header_ptr = ptr as *mut ObjectHeader;
        Some(&mut *header_ptr)
    }
    
    /// Shade an object gray if it's currently white
    unsafe fn shade_object(&self, ptr: *const u8) -> bool {
        if let Some(header) = self.get_object_header(ptr) {
            let current_color = header.get_color();
            if current_color == ObjectColor::White {
                header.set_color(ObjectColor::Gray);
                self.gray_queue.push(ptr);
                return true;
            }
        }
        false
    }
}

impl BarrierImplementation for DijkstraBarrier {
    fn write_barrier(&self, _slot: *mut *const u8, new_value: *const u8, _old_value: *const u8) {
        let start_time = Instant::now();
        self.stats.barrier_calls.fetch_add(1, Ordering::Relaxed);
        self.stats.dijkstra_barriers.fetch_add(1, Ordering::Relaxed);
        
        // Only active during marking
        if !self.is_marking.load(Ordering::Acquire) {
            self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
            return;
        }
        
        // Shade the new object if it's white
        if !new_value.is_null() {
            unsafe {
                if self.shade_object(new_value) {
                    self.stats.active_barriers.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update timing statistics (simplified)
        let _duration = start_time.elapsed();
        // In a real implementation, we'd update timing stats here
    }
    
    fn enable_marking(&self) {
        self.is_marking.store(true, Ordering::Release);
    }
    
    fn disable_marking(&self) {
        self.is_marking.store(false, Ordering::Release);
    }
    
    fn is_marking_active(&self) -> bool {
        self.is_marking.load(Ordering::Acquire)
    }
    
    fn get_stats(&self) -> BarrierStats {
        self.stats.to_stats()
    }
    
    fn reset_stats(&self) {
        self.stats = AtomicBarrierStats::new();
    }
    
    fn barrier_type(&self) -> WriteBarrierType {
        WriteBarrierType::Incremental
    }
    
    fn drain_gray_queue(&self) -> Vec<*const u8> {
        let mut objects = Vec::new();
        while let Some(obj) = self.gray_queue.pop() {
            objects.push(obj);
        }
        objects
    }
    
    fn memory_overhead(&self) -> usize {
        std::mem::size_of::<Self>() + 
        self.gray_queue.len() * std::mem::size_of::<*const u8>()
    }
}

/// Yuasa deletion barrier implementation
/// 
/// Maintains the weak tri-color invariant by preserving the snapshot of
/// the heap at the start of garbage collection. When an object is about
/// to be overwritten, it's marked gray to ensure it remains reachable.
pub struct YuasaBarrier {
    /// Whether marking is currently active
    is_marking: AtomicBool,
    /// Queue of objects that need to be scanned (gray objects)
    gray_queue: Arc<SegQueue<*const u8>>,
    /// Statistics
    stats: AtomicBarrierStats,
    /// Configuration
    config: BarrierConfig,
}

impl YuasaBarrier {
    pub fn new(config: BarrierConfig) -> Self {
        Self {
            is_marking: AtomicBool::new(false),
            gray_queue: Arc::new(SegQueue::new()),
            stats: AtomicBarrierStats::new(),
            config,
        }
    }
    
    /// Get object header from pointer (unsafe operation)
    unsafe fn get_object_header(&self, ptr: *const u8) -> Option<&mut ObjectHeader> {
        if ptr.is_null() {
            return None;
        }
        
        let header_ptr = ptr as *mut ObjectHeader;
        Some(&mut *header_ptr)
    }
    
    /// Shade an object gray if it's currently white
    unsafe fn shade_object(&self, ptr: *const u8) -> bool {
        if let Some(header) = self.get_object_header(ptr) {
            let current_color = header.get_color();
            if current_color == ObjectColor::White {
                header.set_color(ObjectColor::Gray);
                self.gray_queue.push(ptr);
                return true;
            }
        }
        false
    }
}

impl BarrierImplementation for YuasaBarrier {
    fn write_barrier(&self, _slot: *mut *const u8, _new_value: *const u8, old_value: *const u8) {
        let start_time = Instant::now();
        self.stats.barrier_calls.fetch_add(1, Ordering::Relaxed);
        self.stats.yuasa_barriers.fetch_add(1, Ordering::Relaxed);
        
        // Only active during marking
        if !self.is_marking.load(Ordering::Acquire) {
            self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
            return;
        }
        
        // Shade the old object if it's white (preserve snapshot)
        if !old_value.is_null() {
            unsafe {
                if self.shade_object(old_value) {
                    self.stats.active_barriers.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
        }
        
        let _duration = start_time.elapsed();
    }
    
    fn enable_marking(&self) {
        self.is_marking.store(true, Ordering::Release);
    }
    
    fn disable_marking(&self) {
        self.is_marking.store(false, Ordering::Release);
    }
    
    fn is_marking_active(&self) -> bool {
        self.is_marking.load(Ordering::Acquire)
    }
    
    fn get_stats(&self) -> BarrierStats {
        self.stats.to_stats()
    }
    
    fn reset_stats(&self) {
        self.stats = AtomicBarrierStats::new();
    }
    
    fn barrier_type(&self) -> WriteBarrierType {
        WriteBarrierType::Snapshot
    }
    
    fn drain_gray_queue(&self) -> Vec<*const u8> {
        let mut objects = Vec::new();
        while let Some(obj) = self.gray_queue.pop() {
            objects.push(obj);
        }
        objects
    }
    
    fn memory_overhead(&self) -> usize {
        std::mem::size_of::<Self>() + 
        self.gray_queue.len() * std::mem::size_of::<*const u8>()
    }
}

/// Hybrid write barrier implementation
/// 
/// Combines the benefits of both Dijkstra and Yuasa barriers, based on
/// Go's hybrid write barrier design. This provides better performance
/// than either barrier alone while maintaining correctness.
/// 
/// The hybrid barrier:
/// 1. Shades the old value (deletion barrier component)
/// 2. Conditionally shades the new value based on the slot's location
pub struct HybridBarrier {
    /// Whether marking is currently active
    is_marking: AtomicBool,
    /// Queue of objects that need to be scanned (gray objects)
    gray_queue: Arc<SegQueue<*const u8>>,
    /// Statistics
    stats: AtomicBarrierStats,
    /// Configuration
    config: BarrierConfig,
}

impl HybridBarrier {
    pub fn new(config: BarrierConfig) -> Self {
        Self {
            is_marking: AtomicBool::new(false),
            gray_queue: Arc::new(SegQueue::new()),
            stats: AtomicBarrierStats::new(),
            config,
        }
    }
    
    /// Get object header from pointer (unsafe operation)
    unsafe fn get_object_header(&self, ptr: *const u8) -> Option<&mut ObjectHeader> {
        if ptr.is_null() {
            return None;
        }
        
        let header_ptr = ptr as *mut ObjectHeader;
        Some(&mut *header_ptr)
    }
    
    /// Shade an object gray if it's currently white
    unsafe fn shade_object(&self, ptr: *const u8) -> bool {
        if let Some(header) = self.get_object_header(ptr) {
            let current_color = header.get_color();
            if current_color == ObjectColor::White {
                header.set_color(ObjectColor::Gray);
                self.gray_queue.push(ptr);
                return true;
            }
        }
        false
    }
    
    /// Determine if a slot is on the stack or in a gray object
    /// 
    /// In Go's implementation, stack slots are treated as gray, and
    /// we only need to shade the new value when storing from a gray location.
    fn is_slot_gray(&self, slot: *mut *const u8) -> bool {
        // Simplified implementation - in practice this would:
        // 1. Check if the slot is on any thread's stack (always gray)
        // 2. Find the containing object and check its color
        // 3. Use heap metadata to determine object boundaries
        
        // For now, conservatively assume stack slots are gray
        self.is_likely_stack_address(slot)
    }
    
    /// Heuristic to determine if an address is likely on the stack
    /// 
    /// This is a simplified implementation - a real system would maintain
    /// precise stack bounds for each thread.
    fn is_likely_stack_address(&self, _addr: *mut *const u8) -> bool {
        // Conservative assumption: treat all slots as potentially gray
        // This ensures correctness at the cost of some extra work
        true
    }
}

impl BarrierImplementation for HybridBarrier {
    fn write_barrier(&self, slot: *mut *const u8, new_value: *const u8, old_value: *const u8) {
        let start_time = Instant::now();
        self.stats.barrier_calls.fetch_add(1, Ordering::Relaxed);
        self.stats.hybrid_barriers.fetch_add(1, Ordering::Relaxed);
        
        // Only active during marking
        if !self.is_marking.load(Ordering::Acquire) {
            self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
            return;
        }
        
        let mut did_work = false;
        
        unsafe {
            // Deletion barrier component: shade the old value
            if !old_value.is_null() {
                if self.shade_object(old_value) {
                    did_work = true;
                }
            }
            
            // Insertion barrier component: conditionally shade the new value
            // Only shade if the slot is in a gray location (stack or gray object)
            if !new_value.is_null() && self.is_slot_gray(slot) {
                if self.shade_object(new_value) {
                    did_work = true;
                }
            }
        }
        
        if did_work {
            self.stats.active_barriers.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.noop_barriers.fetch_add(1, Ordering::Relaxed);
        }
        
        let _duration = start_time.elapsed();
    }
    
    fn enable_marking(&self) {
        self.is_marking.store(true, Ordering::Release);
    }
    
    fn disable_marking(&self) {
        self.is_marking.store(false, Ordering::Release);
    }
    
    fn is_marking_active(&self) -> bool {
        self.is_marking.load(Ordering::Acquire)
    }
    
    fn get_stats(&self) -> BarrierStats {
        self.stats.to_stats()
    }
    
    fn reset_stats(&self) {
        self.stats = AtomicBarrierStats::new();
    }
    
    fn barrier_type(&self) -> WriteBarrierType {
        WriteBarrierType::Hybrid
    }
    
    fn drain_gray_queue(&self) -> Vec<*const u8> {
        let mut objects = Vec::new();
        while let Some(obj) = self.gray_queue.pop() {
            objects.push(obj);
        }
        objects
    }
    
    fn memory_overhead(&self) -> usize {
        std::mem::size_of::<Self>() + 
        self.gray_queue.len() * std::mem::size_of::<*const u8>()
    }
}

/// Convenience type alias for the main write barrier interface
pub type WriteBarrier = dyn BarrierImplementation;

/// Factory for creating barrier implementations
pub struct BarrierFactory;

impl BarrierFactory {
    /// Create a barrier implementation based on the given type and configuration
    pub fn create_barrier(
        barrier_type: WriteBarrierType,
        config: BarrierConfig,
    ) -> Box<dyn BarrierImplementation> {
        match barrier_type {
            WriteBarrierType::None => Box::new(NoBarrier::new()),
            WriteBarrierType::Incremental => Box::new(DijkstraBarrier::new(config)),
            WriteBarrierType::Snapshot => Box::new(YuasaBarrier::new(config)),
            WriteBarrierType::Hybrid => Box::new(HybridBarrier::new(config)),
        }
    }
    
    /// Create the optimal barrier for Prism VM's workload
    pub fn create_prism_optimized(config: BarrierConfig) -> Box<dyn BarrierImplementation> {
        // Hybrid barrier is optimal for concurrent collection with good performance
        Box::new(HybridBarrier::new(config))
    }
    
    /// Create a barrier optimized for low latency
    pub fn create_low_latency(config: BarrierConfig) -> Box<dyn BarrierImplementation> {
        // Dijkstra barrier has more predictable overhead
        Box::new(DijkstraBarrier::new(config))
    }
    
    /// Create a barrier optimized for high throughput
    pub fn create_high_throughput(config: BarrierConfig) -> Box<dyn BarrierImplementation> {
        // Hybrid barrier provides best overall performance
        Box::new(HybridBarrier::new(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn create_test_config() -> BarrierConfig {
        BarrierConfig {
            barrier_type: WriteBarrierType::Hybrid,
            enable_thread_local_buffering: false, // Simplify for testing
            enable_simd_optimizations: false,
            enable_card_marking: false,
            enable_safety_checks: false,
            enable_race_detection: false,
            enable_barrier_validation: false,
            buffer_size: 64,
            flush_threshold: 48,
            max_pause_contribution: std::time::Duration::from_millis(1),
            worker_threads: 1,
            enable_prefetching: false,
            card_size: 512,
        }
    }

    #[test]
    fn test_no_barrier() {
        let barrier = NoBarrier::new();
        
        // Should always be inactive
        assert!(!barrier.is_marking_active());
        
        // Should not do any work
        barrier.write_barrier(std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
        
        let stats = barrier.get_stats();
        assert_eq!(stats.barrier_calls, 1);
        assert_eq!(stats.noop_barriers, 1);
        assert_eq!(stats.active_barriers, 0);
    }

    #[test]
    fn test_dijkstra_barrier_inactive() {
        let barrier = DijkstraBarrier::new(create_test_config());
        
        // Should be inactive initially
        assert!(!barrier.is_marking_active());
        
        // Should not do work when inactive
        barrier.write_barrier(std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
        
        let stats = barrier.get_stats();
        assert_eq!(stats.barrier_calls, 1);
        assert_eq!(stats.noop_barriers, 1);
        assert_eq!(stats.dijkstra_barriers, 1);
    }

    #[test]
    fn test_dijkstra_barrier_active() {
        let barrier = DijkstraBarrier::new(create_test_config());
        
        // Enable marking
        barrier.enable_marking();
        assert!(barrier.is_marking_active());
        
        // Should still be no-op with null pointers
        barrier.write_barrier(std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
        
        let stats = barrier.get_stats();
        assert_eq!(stats.barrier_calls, 1);
        assert_eq!(stats.noop_barriers, 1);
        
        // Disable marking
        barrier.disable_marking();
        assert!(!barrier.is_marking_active());
    }

    #[test]
    fn test_yuasa_barrier() {
        let barrier = YuasaBarrier::new(create_test_config());
        
        assert_eq!(barrier.barrier_type(), WriteBarrierType::Snapshot);
        
        barrier.enable_marking();
        barrier.write_barrier(std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
        
        let stats = barrier.get_stats();
        assert_eq!(stats.yuasa_barriers, 1);
    }

    #[test]
    fn test_hybrid_barrier() {
        let barrier = HybridBarrier::new(create_test_config());
        
        assert_eq!(barrier.barrier_type(), WriteBarrierType::Hybrid);
        
        barrier.enable_marking();
        barrier.write_barrier(std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
        
        let stats = barrier.get_stats();
        assert_eq!(stats.hybrid_barriers, 1);
    }

    #[test]
    fn test_barrier_factory() {
        let config = create_test_config();
        
        let no_barrier = BarrierFactory::create_barrier(WriteBarrierType::None, config.clone());
        assert_eq!(no_barrier.barrier_type(), WriteBarrierType::None);
        
        let dijkstra = BarrierFactory::create_barrier(WriteBarrierType::Incremental, config.clone());
        assert_eq!(dijkstra.barrier_type(), WriteBarrierType::Incremental);
        
        let yuasa = BarrierFactory::create_barrier(WriteBarrierType::Snapshot, config.clone());
        assert_eq!(yuasa.barrier_type(), WriteBarrierType::Snapshot);
        
        let hybrid = BarrierFactory::create_barrier(WriteBarrierType::Hybrid, config.clone());
        assert_eq!(hybrid.barrier_type(), WriteBarrierType::Hybrid);
    }

    #[test]
    fn test_gray_queue_operations() {
        let barrier = HybridBarrier::new(create_test_config());
        
        // Initially empty
        let objects = barrier.drain_gray_queue();
        assert_eq!(objects.len(), 0);
        
        // Memory overhead should be reasonable
        let overhead = barrier.memory_overhead();
        assert!(overhead > 0);
        assert!(overhead < 1024); // Should be small initially
    }

    #[test]
    fn test_barrier_statistics() {
        let barrier = DijkstraBarrier::new(create_test_config());
        
        // Make several calls
        barrier.enable_marking();
        for _ in 0..10 {
            barrier.write_barrier(std::ptr::null_mut(), std::ptr::null(), std::ptr::null());
        }
        
        let stats = barrier.get_stats();
        assert_eq!(stats.barrier_calls, 10);
        assert_eq!(stats.dijkstra_barriers, 10);
        assert_eq!(stats.noop_barriers, 10); // All null pointers
        
        // Reset stats
        barrier.reset_stats();
        let new_stats = barrier.get_stats();
        assert_eq!(new_stats.barrier_calls, 0);
    }

    #[test]
    fn test_concurrent_barrier_access() {
        let barrier = Arc::new(HybridBarrier::new(create_test_config()));
        barrier.enable_marking();
        
        let mut handles = vec![];
        
        // Spawn multiple threads using the barrier
        for thread_id in 0..4 {
            let barrier_clone = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    // Simulate pointer updates
                    let slot = &mut (i as *const u8) as *mut *const u8;
                    let new_val = (thread_id * 1000 + i) as *const u8;
                    let old_val = i as *const u8;
                    
                    barrier_clone.write_barrier(slot, new_val, old_val);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Check that all calls were recorded
        let stats = barrier.get_stats();
        assert_eq!(stats.barrier_calls, 400); // 4 threads * 100 calls
        assert_eq!(stats.hybrid_barriers, 400);
    }
} 