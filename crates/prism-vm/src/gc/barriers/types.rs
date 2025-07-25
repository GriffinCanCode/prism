//! Core types and data structures for the write barriers subsystem
//!
//! This module defines all the fundamental types used throughout the barriers
//! subsystem, including barrier types, object colors, configuration, and
//! statistics structures.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Types of write barriers supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteBarrierType {
    /// No write barrier - only safe for stop-the-world collection
    None,
    /// Dijkstra-style insertion barrier - maintains strong tri-color invariant
    /// Shades new objects being pointed to by black objects
    Incremental,
    /// Yuasa-style deletion barrier - maintains weak tri-color invariant
    /// Preserves snapshot at start of GC by shading overwritten objects
    Snapshot,
    /// Hybrid barrier combining benefits of both insertion and deletion barriers
    /// Based on Go's hybrid write barrier design for concurrent marking
    Hybrid,
}

/// Object colors in tri-color marking algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectColor {
    /// White: Potentially garbage, not yet reached by marking
    White = 0,
    /// Gray: Reachable but not yet scanned for references
    Gray = 1,
    /// Black: Reachable and fully scanned
    Black = 2,
}

impl ObjectColor {
    /// Convert from raw mark bits
    pub fn from_mark_bits(bits: u8) -> Self {
        match bits & 0x3 {
            0 => ObjectColor::White,
            1 => ObjectColor::Gray,
            2 => ObjectColor::Black,
            _ => ObjectColor::White, // Default to white for invalid values
        }
    }

    /// Convert to mark bits for storage
    pub fn to_mark_bits(self) -> u8 {
        self as u8
    }

    /// Check if this color represents a live object
    pub fn is_live(self) -> bool {
        match self {
            ObjectColor::White => false,
            ObjectColor::Gray | ObjectColor::Black => true,
        }
    }

    /// Check if this color indicates the object needs scanning
    pub fn needs_scanning(self) -> bool {
        self == ObjectColor::Gray
    }
}

/// Represents a color transition during marking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTransition {
    pub from: ObjectColor,
    pub to: ObjectColor,
    pub timestamp: Instant,
}

impl ColorTransition {
    pub fn new(from: ObjectColor, to: ObjectColor) -> Self {
        Self {
            from,
            to,
            timestamp: Instant::now(),
        }
    }

    /// Check if this is a valid color transition
    pub fn is_valid(&self) -> bool {
        match (self.from, self.to) {
            // Valid transitions
            (ObjectColor::White, ObjectColor::Gray) => true,
            (ObjectColor::Gray, ObjectColor::Black) => true,
            (ObjectColor::Black, ObjectColor::White) => true, // Reset after collection
            (ObjectColor::Gray, ObjectColor::White) => true,  // Reset after collection
            // Invalid transitions
            (ObjectColor::White, ObjectColor::Black) => false, // Must go through gray
            (ObjectColor::Black, ObjectColor::Gray) => false,  // Can't go backwards
            // Same color is allowed (no-op)
            (a, b) if a == b => true,
        }
    }
}

/// Configuration for write barriers
#[derive(Debug, Clone)]
pub struct BarrierConfig {
    /// Type of write barrier to use
    pub barrier_type: WriteBarrierType,
    
    /// Enable thread-local buffering for better performance
    pub enable_thread_local_buffering: bool,
    
    /// Enable SIMD optimizations where possible
    pub enable_simd_optimizations: bool,
    
    /// Enable card marking for large objects
    pub enable_card_marking: bool,
    
    /// Enable safety checks (recommended for debug builds)
    pub enable_safety_checks: bool,
    
    /// Enable race condition detection
    pub enable_race_detection: bool,
    
    /// Enable barrier validation
    pub enable_barrier_validation: bool,
    
    /// Size of thread-local buffers
    pub buffer_size: usize,
    
    /// Threshold for flushing buffers (as fraction of buffer_size)
    pub flush_threshold: usize,
    
    /// Maximum time barriers should contribute to pause times
    pub max_pause_contribution: Duration,
    
    /// Number of worker threads for barrier processing
    pub worker_threads: usize,
    
    /// Enable hardware prefetching hints
    pub enable_prefetching: bool,
    
    /// Card size for card marking (must be power of 2)
    pub card_size: usize,
}

impl Default for BarrierConfig {
    fn default() -> Self {
        Self {
            barrier_type: WriteBarrierType::Hybrid,
            enable_thread_local_buffering: true,
            enable_simd_optimizations: true,
            enable_card_marking: true,
            enable_safety_checks: cfg!(debug_assertions),
            enable_race_detection: cfg!(debug_assertions),
            enable_barrier_validation: cfg!(debug_assertions),
            buffer_size: 256,
            flush_threshold: 192, // 75% of buffer_size
            max_pause_contribution: Duration::from_micros(100),
            worker_threads: num_cpus::get().min(4),
            enable_prefetching: true,
            card_size: 512, // 512 bytes per card
        }
    }
}

/// Comprehensive statistics for barrier operations
#[derive(Debug, Clone)]
pub struct BarrierStats {
    /// Total number of barrier calls
    pub barrier_calls: u64,
    
    /// Number of barriers that actually did work
    pub active_barriers: u64,
    
    /// Number of barriers that were no-ops
    pub noop_barriers: u64,
    
    /// Barriers by type
    pub dijkstra_barriers: u64,
    pub yuasa_barriers: u64,
    pub hybrid_barriers: u64,
    
    /// Color transitions performed
    pub color_transitions: HashMap<(ObjectColor, ObjectColor), u64>,
    
    /// Buffer statistics
    pub buffer_flushes: u64,
    pub buffer_overflows: u64,
    pub average_buffer_occupancy: f64,
    
    /// Performance metrics
    pub total_barrier_time: Duration,
    pub average_barrier_time: Duration,
    pub max_barrier_time: Duration,
    
    /// SIMD optimization usage
    pub simd_operations: u64,
    pub scalar_operations: u64,
    
    /// Card marking statistics
    pub cards_marked: u64,
    pub cards_scanned: u64,
    
    /// Safety and validation
    pub safety_violations: u64,
    pub race_conditions_detected: u64,
    pub validation_failures: u64,
    
    /// Memory usage
    pub memory_overhead: usize,
    pub buffer_memory: usize,
    pub metadata_memory: usize,
}

impl BarrierStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            barrier_calls: 0,
            active_barriers: 0,
            noop_barriers: 0,
            dijkstra_barriers: 0,
            yuasa_barriers: 0,
            hybrid_barriers: 0,
            color_transitions: HashMap::new(),
            buffer_flushes: 0,
            buffer_overflows: 0,
            average_buffer_occupancy: 0.0,
            total_barrier_time: Duration::ZERO,
            average_barrier_time: Duration::ZERO,
            max_barrier_time: Duration::ZERO,
            simd_operations: 0,
            scalar_operations: 0,
            cards_marked: 0,
            cards_scanned: 0,
            safety_violations: 0,
            race_conditions_detected: 0,
            validation_failures: 0,
            memory_overhead: 0,
            buffer_memory: 0,
            metadata_memory: 0,
        }
    }

    /// Combine multiple statistics into one
    pub fn combine(stats_list: Vec<BarrierStats>) -> Self {
        let mut combined = BarrierStats::new();
        
        for stats in stats_list {
            combined.barrier_calls += stats.barrier_calls;
            combined.active_barriers += stats.active_barriers;
            combined.noop_barriers += stats.noop_barriers;
            combined.dijkstra_barriers += stats.dijkstra_barriers;
            combined.yuasa_barriers += stats.yuasa_barriers;
            combined.hybrid_barriers += stats.hybrid_barriers;
            
            // Combine color transitions
            for ((from, to), count) in stats.color_transitions {
                *combined.color_transitions.entry((from, to)).or_insert(0) += count;
            }
            
            combined.buffer_flushes += stats.buffer_flushes;
            combined.buffer_overflows += stats.buffer_overflows;
            combined.total_barrier_time += stats.total_barrier_time;
            combined.max_barrier_time = combined.max_barrier_time.max(stats.max_barrier_time);
            combined.simd_operations += stats.simd_operations;
            combined.scalar_operations += stats.scalar_operations;
            combined.cards_marked += stats.cards_marked;
            combined.cards_scanned += stats.cards_scanned;
            combined.safety_violations += stats.safety_violations;
            combined.race_conditions_detected += stats.race_conditions_detected;
            combined.validation_failures += stats.validation_failures;
            combined.memory_overhead += stats.memory_overhead;
            combined.buffer_memory += stats.buffer_memory;
            combined.metadata_memory += stats.metadata_memory;
        }
        
        // Calculate averages
        if combined.barrier_calls > 0 {
            combined.average_barrier_time = combined.total_barrier_time / combined.barrier_calls as u32;
        }
        
        combined
    }

    /// Calculate barrier efficiency (active barriers / total barriers)
    pub fn efficiency(&self) -> f64 {
        if self.barrier_calls == 0 {
            0.0
        } else {
            self.active_barriers as f64 / self.barrier_calls as f64
        }
    }

    /// Calculate SIMD utilization ratio
    pub fn simd_utilization(&self) -> f64 {
        let total_ops = self.simd_operations + self.scalar_operations;
        if total_ops == 0 {
            0.0
        } else {
            self.simd_operations as f64 / total_ops as f64
        }
    }
}

impl Default for BarrierStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local state for barrier operations
#[derive(Debug)]
pub struct ThreadLocalState {
    /// Thread ID this state belongs to
    pub thread_id: std::thread::ThreadId,
    
    /// Local statistics
    pub stats: BarrierStats,
    
    /// Buffer for batching barrier operations
    pub buffer: Vec<BarrierEvent>,
    
    /// Last flush timestamp
    pub last_flush: Instant,
    
    /// Generation counter for cache invalidation
    pub generation: u64,
    
    /// Whether marking is currently active for this thread
    pub marking_active: bool,
}

impl ThreadLocalState {
    pub fn new(thread_id: std::thread::ThreadId, buffer_size: usize) -> Self {
        Self {
            thread_id,
            stats: BarrierStats::new(),
            buffer: Vec::with_capacity(buffer_size),
            last_flush: Instant::now(),
            generation: 0,
            marking_active: false,
        }
    }

    /// Check if buffer should be flushed
    pub fn should_flush(&self, threshold: usize) -> bool {
        self.buffer.len() >= threshold
    }

    /// Clear the buffer and update statistics
    pub fn flush_buffer(&mut self) -> Vec<BarrierEvent> {
        let events = std::mem::take(&mut self.buffer);
        self.stats.buffer_flushes += 1;
        self.last_flush = Instant::now();
        events
    }
}

/// Events that can be buffered for batch processing
#[derive(Debug, Clone)]
pub enum BarrierEvent {
    /// A write barrier was triggered
    WriteBarrier {
        slot: *mut *const u8,
        new_value: *const u8,
        old_value: *const u8,
        timestamp: Instant,
    },
    
    /// An object's color changed
    ColorChange {
        object: *const u8,
        transition: ColorTransition,
    },
    
    /// A card was marked dirty
    CardMark {
        card_address: *const u8,
        timestamp: Instant,
    },
    
    /// Buffer overflow occurred
    BufferOverflow {
        lost_events: usize,
        timestamp: Instant,
    },
}

// Ensure BarrierEvent can be sent between threads safely
unsafe impl Send for BarrierEvent {}
unsafe impl Sync for BarrierEvent {}

/// Atomic statistics for concurrent access
#[derive(Debug)]
pub struct AtomicBarrierStats {
    pub barrier_calls: AtomicU64,
    pub active_barriers: AtomicU64,
    pub noop_barriers: AtomicU64,
    pub dijkstra_barriers: AtomicU64,
    pub yuasa_barriers: AtomicU64,
    pub hybrid_barriers: AtomicU64,
    pub buffer_flushes: AtomicU64,
    pub buffer_overflows: AtomicU64,
    pub simd_operations: AtomicU64,
    pub scalar_operations: AtomicU64,
    pub cards_marked: AtomicU64,
    pub cards_scanned: AtomicU64,
    pub safety_violations: AtomicU64,
    pub race_conditions_detected: AtomicU64,
    pub validation_failures: AtomicU64,
    pub memory_overhead: AtomicUsize,
}

impl AtomicBarrierStats {
    pub fn new() -> Self {
        Self {
            barrier_calls: AtomicU64::new(0),
            active_barriers: AtomicU64::new(0),
            noop_barriers: AtomicU64::new(0),
            dijkstra_barriers: AtomicU64::new(0),
            yuasa_barriers: AtomicU64::new(0),
            hybrid_barriers: AtomicU64::new(0),
            buffer_flushes: AtomicU64::new(0),
            buffer_overflows: AtomicU64::new(0),
            simd_operations: AtomicU64::new(0),
            scalar_operations: AtomicU64::new(0),
            cards_marked: AtomicU64::new(0),
            cards_scanned: AtomicU64::new(0),
            safety_violations: AtomicU64::new(0),
            race_conditions_detected: AtomicU64::new(0),
            validation_failures: AtomicU64::new(0),
            memory_overhead: AtomicUsize::new(0),
        }
    }

    /// Convert to regular BarrierStats
    pub fn to_stats(&self) -> BarrierStats {
        BarrierStats {
            barrier_calls: self.barrier_calls.load(Ordering::Relaxed),
            active_barriers: self.active_barriers.load(Ordering::Relaxed),
            noop_barriers: self.noop_barriers.load(Ordering::Relaxed),
            dijkstra_barriers: self.dijkstra_barriers.load(Ordering::Relaxed),
            yuasa_barriers: self.yuasa_barriers.load(Ordering::Relaxed),
            hybrid_barriers: self.hybrid_barriers.load(Ordering::Relaxed),
            color_transitions: HashMap::new(), // Would need more complex handling
            buffer_flushes: self.buffer_flushes.load(Ordering::Relaxed),
            buffer_overflows: self.buffer_overflows.load(Ordering::Relaxed),
            average_buffer_occupancy: 0.0, // Calculated elsewhere
            total_barrier_time: Duration::ZERO, // Calculated elsewhere
            average_barrier_time: Duration::ZERO, // Calculated elsewhere
            max_barrier_time: Duration::ZERO, // Calculated elsewhere
            simd_operations: self.simd_operations.load(Ordering::Relaxed),
            scalar_operations: self.scalar_operations.load(Ordering::Relaxed),
            cards_marked: self.cards_marked.load(Ordering::Relaxed),
            cards_scanned: self.cards_scanned.load(Ordering::Relaxed),
            safety_violations: self.safety_violations.load(Ordering::Relaxed),
            race_conditions_detected: self.race_conditions_detected.load(Ordering::Relaxed),
            validation_failures: self.validation_failures.load(Ordering::Relaxed),
            memory_overhead: self.memory_overhead.load(Ordering::Relaxed),
            buffer_memory: 0, // Calculated elsewhere
            metadata_memory: 0, // Calculated elsewhere
        }
    }
}

impl Default for AtomicBarrierStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_color_conversions() {
        for color in [ObjectColor::White, ObjectColor::Gray, ObjectColor::Black] {
            let bits = color.to_mark_bits();
            let recovered = ObjectColor::from_mark_bits(bits);
            assert_eq!(color, recovered);
        }
    }

    #[test]
    fn test_color_transitions() {
        let valid_transitions = [
            (ObjectColor::White, ObjectColor::Gray),
            (ObjectColor::Gray, ObjectColor::Black),
            (ObjectColor::Black, ObjectColor::White),
            (ObjectColor::Gray, ObjectColor::White),
        ];

        for (from, to) in valid_transitions {
            let transition = ColorTransition::new(from, to);
            assert!(transition.is_valid(), "Transition {:?} -> {:?} should be valid", from, to);
        }

        let invalid_transitions = [
            (ObjectColor::White, ObjectColor::Black),
            (ObjectColor::Black, ObjectColor::Gray),
        ];

        for (from, to) in invalid_transitions {
            let transition = ColorTransition::new(from, to);
            assert!(!transition.is_valid(), "Transition {:?} -> {:?} should be invalid", from, to);
        }
    }

    #[test]
    fn test_barrier_stats_combine() {
        let mut stats1 = BarrierStats::new();
        stats1.barrier_calls = 100;
        stats1.active_barriers = 80;

        let mut stats2 = BarrierStats::new();
        stats2.barrier_calls = 50;
        stats2.active_barriers = 30;

        let combined = BarrierStats::combine(vec![stats1, stats2]);
        assert_eq!(combined.barrier_calls, 150);
        assert_eq!(combined.active_barriers, 110);
    }

    #[test]
    fn test_thread_local_state() {
        let thread_id = std::thread::current().id();
        let mut state = ThreadLocalState::new(thread_id, 256);
        
        assert_eq!(state.buffer.capacity(), 256);
        assert!(!state.should_flush(100));
        
        // Fill buffer
        for _ in 0..100 {
            state.buffer.push(BarrierEvent::WriteBarrier {
                slot: std::ptr::null_mut(),
                new_value: std::ptr::null(),
                old_value: std::ptr::null(),
                timestamp: Instant::now(),
            });
        }
        
        assert!(state.should_flush(100));
        
        let events = state.flush_buffer();
        assert_eq!(events.len(), 100);
        assert_eq!(state.buffer.len(), 0);
        assert_eq!(state.stats.buffer_flushes, 1);
    }

    #[test]
    fn test_barrier_config_default() {
        let config = BarrierConfig::default();
        assert_eq!(config.barrier_type, WriteBarrierType::Hybrid);
        assert!(config.enable_thread_local_buffering);
        assert!(config.buffer_size > 0);
        assert!(config.flush_threshold < config.buffer_size);
    }
} 