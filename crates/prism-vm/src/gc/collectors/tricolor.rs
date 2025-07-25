use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Condvar};
use std::collections::{VecDeque, HashSet};
use std::thread;
use std::time::{Duration, Instant};
use crossbeam_queue::SegQueue;

/// Tri-color mark-sweep garbage collector implementing modern concurrent GC techniques
/// 
/// This collector is inspired by:
/// - Go's tri-color concurrent mark-sweep algorithm
/// - V8's incremental marking approach
/// - Research on write barriers and concurrent collection
/// 
/// Key features:
/// - Concurrent marking with tri-color invariants
/// - Incremental sweeping to reduce pause times
/// - Write barriers for maintaining correctness during concurrent execution
/// - Adaptive heap sizing based on allocation patterns
pub struct TriColorCollector {
    config: GcConfig,
    
    // Core GC state
    heap: Arc<RwLock<heap::Heap>>,
    roots: Arc<RwLock<roots::RootSet>>,
    allocator: Arc<allocator::BumpAllocator>,
    
    // Tri-color marking state
    gray_queue: Arc<SegQueue<*const u8>>,
    mark_stack: Arc<Mutex<Vec<*const u8>>>,
    
    // Collection state management
    collection_state: Arc<RwLock<CollectionState>>,
    
    // Background marking thread
    marking_thread: Option<thread::JoinHandle<()>>,
    should_stop: Arc<AtomicBool>,
    mark_trigger: Arc<(Mutex<bool>, Condvar)>,
    
    // Write barrier support
    write_barrier: Arc<barriers::WriteBarrier>,
    
    // Performance monitoring
    stats: Arc<Mutex<CollectionStats>>,
    allocation_counter: AtomicUsize,
    last_collection: Arc<Mutex<Instant>>,
    
    // Adaptive heap management
    heap_growth_factor: AtomicUsize, // Stored as fixed-point (1000 = 1.0)
    allocation_rate_tracker: Arc<Mutex<AllocationRateTracker>>,
}

/// Current state of the garbage collection process
#[derive(Debug, Clone, Copy, PartialEq)]
enum CollectionState {
    /// Normal allocation and mutator execution
    Idle,
    /// Preparing for collection - setting up write barriers
    Preparing,
    /// Concurrent marking phase active
    Marking,
    /// Finalizing marks and preparing for sweep
    MarkingComplete,
    /// Sweeping unreachable objects
    Sweeping,
}

/// Tracks allocation rate for adaptive heap sizing
struct AllocationRateTracker {
    samples: VecDeque<(Instant, usize)>,
    window_size: Duration,
}

impl AllocationRateTracker {
    fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            window_size: Duration::from_secs(10), // 10-second window
        }
    }
    
    fn record_allocation(&mut self, bytes: usize) {
        let now = Instant::now();
        self.samples.push_back((now, bytes));
        
        // Remove old samples outside the window
        let cutoff = now - self.window_size;
        while let Some((timestamp, _)) = self.samples.front() {
            if *timestamp < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
    
    fn current_rate(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        
        let total_bytes: usize = self.samples.iter().map(|(_, bytes)| bytes).sum();
        let time_span = self.samples.back().unwrap().0 - self.samples.front().unwrap().0;
        
        if time_span.is_zero() {
            return 0.0;
        }
        
        total_bytes as f64 / time_span.as_secs_f64()
    }
}

impl TriColorCollector {
    pub fn new(config: GcConfig) -> Self {
        let heap = Arc::new(RwLock::new(heap::Heap::new(config.heap_target)));
        let roots = Arc::new(RwLock::new(roots::RootSet::new()));
        let allocator = Arc::new(allocator::BumpAllocator::new(config.heap_target));
        let write_barrier = Arc::new(barriers::WriteBarrier::new(config.write_barrier));
        
        let collector = Self {
            config: config.clone(),
            heap: heap.clone(),
            roots: roots.clone(),
            allocator: allocator.clone(),
            
            gray_queue: Arc::new(SegQueue::new()),
            mark_stack: Arc::new(Mutex::new(Vec::new())),
            
            collection_state: Arc::new(RwLock::new(CollectionState::Idle)),
            
            marking_thread: None,
            should_stop: Arc::new(AtomicBool::new(false)),
            mark_trigger: Arc::new((Mutex::new(false), Condvar::new())),
            
            write_barrier,
            
            stats: Arc::new(Mutex::new(CollectionStats {
                duration: Duration::new(0, 0),
                bytes_collected: 0,
                objects_collected: 0,
                pause_time: Duration::new(0, 0),
                heap_size_before: 0,
                heap_size_after: 0,
                collection_type: CollectionType::Minor,
            })),
            allocation_counter: AtomicUsize::new(0),
            last_collection: Arc::new(Mutex::new(Instant::now())),
            
            heap_growth_factor: AtomicUsize::new(2000), // Start with 2.0x growth factor
            allocation_rate_tracker: Arc::new(Mutex::new(AllocationRateTracker::new())),
        };
        
        // Start background marking thread if concurrent collection is enabled
        if config.concurrent {
            collector.start_background_marking();
        }
        
        collector
    }
    
    fn start_background_marking(&self) {
        let gray_queue = self.gray_queue.clone();
        let mark_stack = self.mark_stack.clone();
        let heap = self.heap.clone();
        let collection_state = self.collection_state.clone();
        let should_stop = self.should_stop.clone();
        let mark_trigger = self.mark_trigger.clone();
        let write_barrier = self.write_barrier.clone();
        
        let handle = thread::spawn(move || {
            Self::background_marking_loop(
                gray_queue,
                mark_stack,
                heap,
                collection_state,
                should_stop,
                mark_trigger,
                write_barrier,
            );
        });
        
        // Note: In a real implementation, we'd store this handle properly
        // For now, we let it run detached
        std::mem::forget(handle);
    }
    
    fn background_marking_loop(
        gray_queue: Arc<SegQueue<*const u8>>,
        mark_stack: Arc<Mutex<Vec<*const u8>>>,
        heap: Arc<RwLock<heap::Heap>>,
        collection_state: Arc<RwLock<CollectionState>>,
        should_stop: Arc<AtomicBool>,
        mark_trigger: Arc<(Mutex<bool>, Condvar)>,
        write_barrier: Arc<barriers::WriteBarrier>,
    ) {
        let (lock, cvar) = &*mark_trigger;
        
        while !should_stop.load(Ordering::Relaxed) {
            // Wait for marking to be triggered
            let mut triggered = lock.lock().unwrap();
            while !*triggered && !should_stop.load(Ordering::Relaxed) {
                triggered = cvar.wait(triggered).unwrap();
            }
            
            if should_stop.load(Ordering::Relaxed) {
                break;
            }
            
            // Perform concurrent marking
            Self::concurrent_mark_phase(
                &gray_queue,
                &mark_stack,
                &heap,
                &collection_state,
                &write_barrier,
            );
            
            *triggered = false;
        }
    }
    
    fn concurrent_mark_phase(
        gray_queue: &SegQueue<*const u8>,
        mark_stack: &Mutex<Vec<*const u8>>,
        heap: &RwLock<heap::Heap>,
        collection_state: &RwLock<CollectionState>,
        write_barrier: &barriers::WriteBarrier,
    ) {
        // Enable write barriers for concurrent marking
        write_barrier.enable_marking();
        
        // Update collection state
        {
            let mut state = collection_state.write().unwrap();
            *state = CollectionState::Marking;
        }
        
        // Process gray objects concurrently
        let mut work_done = true;
        while work_done {
            work_done = false;
            
            // Process objects from the gray queue
            while let Some(obj_ptr) = gray_queue.pop() {
                Self::mark_object_and_children(obj_ptr, gray_queue, heap);
                work_done = true;
            }
            
            // Process objects from the mark stack (for write barrier additions)
            {
                let mut stack = mark_stack.lock().unwrap();
                while let Some(obj_ptr) = stack.pop() {
                    if let Some(header) = heap.read().unwrap().get_header(obj_ptr) {
                        if header.get_color() == ObjectColor::White {
                            Self::mark_object_and_children(obj_ptr, gray_queue, heap);
                            work_done = true;
                        }
                    }
                }
            }
            
            // Small yield to prevent monopolizing CPU
            if work_done {
                thread::yield_now();
            }
        }
        
        // Mark phase complete
        {
            let mut state = collection_state.write().unwrap();
            *state = CollectionState::MarkingComplete;
        }
        
        write_barrier.disable_marking();
    }
    
    fn mark_object_and_children(
        obj_ptr: *const u8,
        gray_queue: &SegQueue<*const u8>,
        heap: &RwLock<heap::Heap>,
    ) {
        // Mark the object as black (fully processed)
        {
            let mut heap_guard = heap.write().unwrap();
            if let Some(header) = heap_guard.get_header_mut(obj_ptr) {
                header.set_color(ObjectColor::Black);
            } else {
                return; // Object no longer exists
            }
        }
        
        // Trace object's references and mark them gray
        let references = {
            let heap_guard = heap.read().unwrap();
            if let Some(header) = heap_guard.get_header(obj_ptr) {
                // Get type-specific tracer for this object
                let tracer = tracing::get_tracer(header.type_id);
                unsafe { tracer.trace_references(obj_ptr, header.size) }
            } else {
                Vec::new()
            }
        };
        
        // Mark referenced objects as gray
        for ref_ptr in references {
            let mut should_enqueue = false;
            {
                let mut heap_guard = heap.write().unwrap();
                if let Some(ref_header) = heap_guard.get_header_mut(ref_ptr) {
                    if ref_header.get_color() == ObjectColor::White {
                        ref_header.set_color(ObjectColor::Gray);
                        should_enqueue = true;
                    }
                }
            }
            
            if should_enqueue {
                gray_queue.push(ref_ptr);
            }
        }
    }
    
    /// Perform a complete tri-color collection cycle
    fn tricolor_collect(&self) -> CollectionStats {
        let start_time = Instant::now();
        let heap_size_before = self.allocator.allocated_bytes();
        
        // Phase 1: Preparation (minimal STW)
        let pause_start = Instant::now();
        self.prepare_collection();
        let preparation_pause = pause_start.elapsed();
        
        // Phase 2: Concurrent marking
        let marking_start = Instant::now();
        self.trigger_concurrent_marking();
        self.wait_for_marking_completion();
        let marking_duration = marking_start.elapsed();
        
        // Phase 3: Final mark and sweep (brief STW)
        let final_pause_start = Instant::now();
        let (bytes_collected, objects_collected) = self.final_mark_and_sweep();
        let final_pause = final_pause_start.elapsed();
        
        let heap_size_after = self.allocator.allocated_bytes();
        let total_duration = start_time.elapsed();
        let total_pause_time = preparation_pause + final_pause;
        
        // Update adaptive heap sizing
        self.update_heap_growth_factor(heap_size_before, heap_size_after, bytes_collected);
        
        CollectionStats {
            duration: total_duration,
            bytes_collected,
            objects_collected,
            pause_time: total_pause_time,
            heap_size_before,
            heap_size_after,
            collection_type: if objects_collected < 1000 {
                CollectionType::Minor
            } else {
                CollectionType::Major
            },
        }
    }
    
    fn prepare_collection(&self) {
        // Set collection state
        {
            let mut state = self.collection_state.write().unwrap();
            *state = CollectionState::Preparing;
        }
        
        // Reset all objects to white
        {
            let mut heap = self.heap.write().unwrap();
            heap.reset_colors_to_white();
        }
        
        // Mark root objects as gray
        {
            let roots = self.roots.read().unwrap();
            let mut heap = self.heap.write().unwrap();
            
            for root_ptr in roots.iter() {
                if let Some(header) = heap.get_header_mut(*root_ptr) {
                    header.set_color(ObjectColor::Gray);
                    self.gray_queue.push(*root_ptr);
                }
            }
        }
    }
    
    fn trigger_concurrent_marking(&self) {
        let (lock, cvar) = &*self.mark_trigger;
        let mut triggered = lock.lock().unwrap();
        *triggered = true;
        cvar.notify_one();
    }
    
    fn wait_for_marking_completion(&self) {
        // Wait for concurrent marking to complete
        loop {
            {
                let state = self.collection_state.read().unwrap();
                if *state == CollectionState::MarkingComplete {
                    break;
                }
            }
            thread::sleep(Duration::from_micros(100)); // Brief sleep
        }
    }
    
    fn final_mark_and_sweep(&self) -> (usize, usize) {
        // Final mark phase - handle any objects marked by write barriers
        while let Some(obj_ptr) = self.gray_queue.pop() {
            Self::mark_object_and_children(obj_ptr, &self.gray_queue, &self.heap);
        }
        
        // Sweep phase - collect white objects
        let mut bytes_collected = 0;
        let mut objects_collected = 0;
        
        {
            let mut heap = self.heap.write().unwrap();
            let white_objects = heap.find_white_objects();
            
            for obj_ptr in white_objects {
                if let Some(header) = heap.get_header(obj_ptr) {
                    bytes_collected += header.size;
                    objects_collected += 1;
                }
                heap.deallocate(obj_ptr);
            }
            
            // Reset remaining objects to white for next collection
            heap.reset_colors_to_white();
        }
        
        // Update collection state
        {
            let mut state = self.collection_state.write().unwrap();
            *state = CollectionState::Idle;
        }
        
        (bytes_collected, objects_collected)
    }
    
    fn update_heap_growth_factor(&self, size_before: usize, size_after: usize, collected: usize) {
        let current_factor = self.heap_growth_factor.load(Ordering::Relaxed);
        
        // Calculate collection efficiency
        let collection_ratio = if size_before > 0 {
            collected as f64 / size_before as f64
        } else {
            0.0
        };
        
        // Adjust growth factor based on collection efficiency
        let new_factor = if collection_ratio > 0.5 {
            // High collection ratio - can be more aggressive
            (current_factor as f64 * 0.95).max(1200.0) as usize // Min 1.2x
        } else if collection_ratio < 0.1 {
            // Low collection ratio - be more conservative
            (current_factor as f64 * 1.05).min(4000.0) as usize // Max 4.0x
        } else {
            current_factor // Keep current factor
        };
        
        self.heap_growth_factor.store(new_factor, Ordering::Relaxed);
    }
    
    fn should_trigger_collection(&self) -> bool {
        match &self.config.trigger_strategy {
            TriggerStrategy::HeapSize(target) => {
                self.allocator.allocated_bytes() > *target
            }
            TriggerStrategy::AllocationRate(rate) => {
                let tracker = self.allocation_rate_tracker.lock().unwrap();
                tracker.current_rate() > *rate
            }
            TriggerStrategy::Periodic(interval) => {
                let last_collection = *self.last_collection.lock().unwrap();
                last_collection.elapsed() > *interval
            }
            TriggerStrategy::Adaptive => {
                let allocated = self.allocator.allocated_bytes();
                let growth_factor = self.heap_growth_factor.load(Ordering::Relaxed);
                let target = (self.config.heap_target * growth_factor) / 1000; // Convert from fixed-point
                allocated > target
            }
        }
    }
}

impl GarbageCollector for TriColorCollector {
    fn allocate(&self, size: usize, align: usize) -> Option<*mut u8> {
        // Check if we should trigger collection
        if self.should_trigger_collection() {
            // For concurrent collectors, we can trigger collection without blocking allocation
            if self.config.concurrent {
                let (lock, cvar) = &*self.mark_trigger;
                if let Ok(mut triggered) = lock.try_lock() {
                    if !*triggered {
                        *triggered = true;
                        cvar.notify_one();
                    }
                }
            } else {
                // For non-concurrent, we must collect now
                self.collect();
            }
        }
        
        // Perform allocation
        let ptr = self.allocator.allocate(size, align)?;
        self.allocation_counter.fetch_add(size, Ordering::Relaxed);
        
        // Track allocation rate
        {
            let mut tracker = self.allocation_rate_tracker.lock().unwrap();
            tracker.record_allocation(size);
        }
        
        // Initialize object header
        unsafe {
            let header = ptr as *mut ObjectHeader;
            *header = ObjectHeader::new(size, 0); // Type ID would be determined by caller
        }
        
        Some(ptr)
    }
    
    fn collect(&self) -> CollectionStats {
        let stats = self.tricolor_collect();
        *self.stats.lock().unwrap() = stats.clone();
        *self.last_collection.lock().unwrap() = Instant::now();
        stats
    }
    
    fn should_collect(&self) -> bool {
        self.should_trigger_collection()
    }
    
    fn heap_stats(&self) -> HeapStats {
        let allocated = self.allocator.allocated_bytes();
        let stats = self.stats.lock().unwrap();
        let allocation_rate = {
            let tracker = self.allocation_rate_tracker.lock().unwrap();
            tracker.current_rate()
        };
        
        HeapStats {
            total_allocated: allocated,
            live_objects: 0, // Would be calculated during collection
            free_space: self.config.heap_target.saturating_sub(allocated),
            fragmentation_ratio: 0.0, // Would be calculated based on heap layout
            allocation_rate,
            gc_overhead: stats.pause_time.as_secs_f64() / stats.duration.as_secs_f64(),
        }
    }
    
    fn configure(&self, _config: GcConfig) {
        // Would update configuration - implementation depends on specific needs
    }
    
    fn register_root(&self, ptr: *const u8) {
        self.roots.write().unwrap().add_root(ptr);
    }
    
    fn unregister_root(&self, ptr: *const u8) {
        self.roots.write().unwrap().remove_root(ptr);
    }
    
    fn mark_object(&self, ptr: *const u8) {
        // Add to mark stack for processing by background thread
        if let Ok(mut stack) = self.mark_stack.try_lock() {
            stack.push(ptr);
        } else {
            // If we can't get the lock, add to gray queue directly
            self.gray_queue.push(ptr);
        }
    }
    
    fn is_marked(&self, ptr: *const u8) -> bool {
        if let Some(header) = self.heap.read().unwrap().get_header(ptr) {
            header.get_color() != ObjectColor::White
        } else {
            false
        }
    }
}

impl Drop for TriColorCollector {
    fn drop(&mut self) {
        // Signal background thread to stop
        self.should_stop.store(true, Ordering::Relaxed);
        
        // Trigger one last time to wake up the thread
        let (lock, cvar) = &*self.mark_trigger;
        let mut triggered = lock.lock().unwrap();
        *triggered = true;
        cvar.notify_one();
        
        // Note: In a real implementation, we'd join the background thread here
        // For now, the thread will detect should_stop and exit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tricolor_collector_creation() {
        let config = GcConfig {
            heap_target: 1024 * 1024, // 1MB
            max_pause_time: Duration::from_millis(10),
            worker_threads: 1,
            concurrent: true,
            generational: false,
            write_barrier: WriteBarrierType::Hybrid,
            trigger_strategy: TriggerStrategy::Adaptive,
        };
        
        let collector = TriColorCollector::new(config);
        assert!(!collector.should_collect()); // Should not need collection initially
    }
    
    #[test]
    fn test_allocation_and_collection() {
        let config = GcConfig {
            heap_target: 1024, // Small heap to trigger collection
            max_pause_time: Duration::from_millis(10),
            worker_threads: 1,
            concurrent: false, // Synchronous for testing
            generational: false,
            write_barrier: WriteBarrierType::None,
            trigger_strategy: TriggerStrategy::HeapSize(512),
        };
        
        let collector = TriColorCollector::new(config);
        
        // Allocate some objects
        for _ in 0..10 {
            let ptr = collector.allocate(64, 8);
            assert!(ptr.is_some());
        }
        
        // Should trigger collection
        assert!(collector.should_collect());
        
        let stats = collector.collect();
        assert!(stats.duration > Duration::ZERO);
    }
} 