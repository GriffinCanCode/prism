//! Type Inference Performance Profiling
//!
//! This module handles performance profiling and metrics collection for type inference.
//! It tracks timing, memory usage, and provides integration with the compiler's profiling system.
//!
//! **Single Responsibility**: Performance profiling and metrics
//! **What it does**: Track timing, collect metrics, provide performance data
//! **What it doesn't do**: Perform inference, manage types, handle compilation

use crate::type_inference::{InferenceStatistics, constraints::ConstraintSet};
use prism_common::Span;
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Performance profiler for type inference
#[derive(Debug)]
pub struct InferenceProfiler {
    /// Current profiling session
    current_session: Option<ProfilingSession>,
    /// Historical performance data
    performance_history: Vec<PerformanceMetrics>,
    /// Phase timings
    phase_timings: HashMap<String, Duration>,
    /// Memory usage tracking
    memory_tracker: MemoryTracker,
    /// Configuration
    config: ProfilingConfig,
}

/// A single profiling session
#[derive(Debug)]
struct ProfilingSession {
    /// Session start time
    start_time: Instant,
    /// Session name
    name: String,
    /// Phase start times
    phase_starts: HashMap<String, Instant>,
    /// Metrics collected during session
    metrics: PerformanceMetrics,
}

/// Performance metrics collected during inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total inference time
    pub total_time: Duration,
    /// Time spent on different phases
    pub phase_times: HashMap<String, Duration>,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Node processing statistics
    pub node_stats: NodeProcessingStats,
    /// Constraint processing statistics
    pub constraint_stats: ConstraintProcessingStats,
    /// Cache performance
    pub cache_stats: CacheStats,
    /// AI metadata generation time
    pub ai_metadata_time: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Current memory usage in bytes
    pub current_memory: usize,
    /// Memory usage by component
    pub component_memory: HashMap<String, usize>,
    /// Number of allocations
    pub allocation_count: usize,
}

/// Node processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProcessingStats {
    /// Total nodes processed
    pub total_nodes: usize,
    /// Nodes processed per second
    pub nodes_per_second: f64,
    /// Processing time by node type
    pub node_type_times: HashMap<String, Duration>,
    /// Average processing time per node
    pub avg_node_time: Duration,
}

/// Constraint processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintProcessingStats {
    /// Total constraints generated
    pub total_constraints: usize,
    /// Constraints solved per second
    pub constraints_per_second: f64,
    /// Time spent on constraint solving
    pub solving_time: Duration,
    /// Number of unification steps
    pub unification_steps: usize,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Total cache queries
    pub total_queries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache size in bytes
    pub cache_size: usize,
}

/// Memory usage tracker
#[derive(Debug)]
struct MemoryTracker {
    /// Current memory usage estimate
    current_usage: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Memory usage by component
    component_usage: HashMap<String, usize>,
    /// Allocation count
    allocation_count: usize,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable detailed profiling
    pub enable_detailed_profiling: bool,
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
    /// Enable cache profiling
    pub enable_cache_profiling: bool,
    /// Profiling sample rate (0.0 to 1.0)
    pub sample_rate: f64,
    /// Maximum history entries to keep
    pub max_history_entries: usize,
}

/// Performance statistics for external reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Average inference time
    pub avg_inference_time: Duration,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Throughput (nodes per second)
    pub throughput: f64,
    /// Error rate
    pub error_rate: f64,
}

impl InferenceProfiler {
    /// Create a new inference profiler
    pub fn new() -> Self {
        Self {
            current_session: None,
            performance_history: Vec::new(),
            phase_timings: HashMap::new(),
            memory_tracker: MemoryTracker::new(),
            config: ProfilingConfig::default(),
        }
    }

    /// Create a profiler with custom configuration
    pub fn with_config(config: ProfilingConfig) -> Self {
        Self {
            current_session: None,
            performance_history: Vec::new(),
            phase_timings: HashMap::new(),
            memory_tracker: MemoryTracker::new(),
            config,
        }
    }

    /// Start profiling a program inference
    pub fn start_program_inference(&mut self) {
        if !self.config.enable_detailed_profiling {
            return;
        }

        self.start_session("program_inference".to_string());
    }

    /// End program inference profiling
    pub fn end_program_inference(&mut self, elapsed: Duration) {
        if !self.config.enable_detailed_profiling {
            return;
        }

        if let Some(mut session) = self.current_session.take() {
            session.metrics.total_time = elapsed;
            self.finalize_session(session);
        }
    }

    /// Start profiling a specific phase
    pub fn start_phase(&mut self, phase_name: &str) {
        if !self.config.enable_detailed_profiling {
            return;
        }

        if let Some(ref mut session) = self.current_session {
            session.phase_starts.insert(phase_name.to_string(), Instant::now());
        }
    }

    /// End profiling a specific phase
    pub fn end_phase(&mut self, phase_name: &str) {
        if !self.config.enable_detailed_profiling {
            return;
        }

        if let Some(ref mut session) = self.current_session {
            if let Some(start_time) = session.phase_starts.remove(phase_name) {
                let elapsed = start_time.elapsed();
                session.metrics.phase_times.insert(phase_name.to_string(), elapsed);
                self.phase_timings.insert(phase_name.to_string(), elapsed);
            }
        }
    }

    /// Record node processing
    pub fn record_node_processing(&mut self, node_type: &str, processing_time: Duration) {
        if !self.config.enable_detailed_profiling {
            return;
        }

        if let Some(ref mut session) = self.current_session {
            session.metrics.node_stats.total_nodes += 1;
            session.metrics.node_stats.node_type_times
                .entry(node_type.to_string())
                .and_modify(|time| *time += processing_time)
                .or_insert(processing_time);
        }
    }

    /// Record constraint processing
    pub fn record_constraint_processing(&mut self, constraint_count: usize, solving_time: Duration) {
        if !self.config.enable_detailed_profiling {
            return;
        }

        if let Some(ref mut session) = self.current_session {
            session.metrics.constraint_stats.total_constraints += constraint_count;
            session.metrics.constraint_stats.solving_time += solving_time;
        }
    }

    /// Record memory allocation
    pub fn record_memory_allocation(&mut self, component: &str, size: usize) {
        if !self.config.enable_memory_tracking {
            return;
        }

        self.memory_tracker.record_allocation(component, size);
        
        if let Some(ref mut session) = self.current_session {
            session.metrics.memory_stats.current_memory = self.memory_tracker.current_usage;
            session.metrics.memory_stats.peak_memory = self.memory_tracker.peak_usage;
            session.metrics.memory_stats.allocation_count = self.memory_tracker.allocation_count;
        }
    }

    /// Record cache access
    pub fn record_cache_access(&mut self, hit: bool) {
        if !self.config.enable_cache_profiling {
            return;
        }

        if let Some(ref mut session) = self.current_session {
            session.metrics.cache_stats.total_queries += 1;
            if hit {
                session.metrics.cache_stats.cache_hits += 1;
            } else {
                session.metrics.cache_stats.cache_misses += 1;
            }
            
            // Update hit rate
            let total = session.metrics.cache_stats.total_queries as f64;
            let hits = session.metrics.cache_stats.cache_hits as f64;
            session.metrics.cache_stats.hit_rate = hits / total;
        }
    }

    /// Get current performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        if self.performance_history.is_empty() {
            return PerformanceStats::default();
        }

        let total_entries = self.performance_history.len() as f64;
        let avg_time = self.performance_history.iter()
            .map(|m| m.total_time)
            .sum::<Duration>() / self.performance_history.len() as u32;

        let peak_memory = self.performance_history.iter()
            .map(|m| m.memory_stats.peak_memory)
            .max()
            .unwrap_or(0);

        let avg_hit_rate = self.performance_history.iter()
            .map(|m| m.cache_stats.hit_rate)
            .sum::<f64>() / total_entries;

        let avg_throughput = self.performance_history.iter()
            .map(|m| m.node_stats.nodes_per_second)
            .sum::<f64>() / total_entries;

        PerformanceStats {
            avg_inference_time: avg_time,
            peak_memory_usage: peak_memory,
            cache_hit_rate: avg_hit_rate,
            throughput: avg_throughput,
            error_rate: 0.0, // Would be calculated from error tracking
        }
    }

    /// Get the latest performance metrics
    pub fn get_latest_metrics(&self) -> Option<&PerformanceMetrics> {
        self.performance_history.last()
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        self.current_session = None;
        self.phase_timings.clear();
        self.memory_tracker.reset();
        
        // Keep some history for analysis
        if self.performance_history.len() > self.config.max_history_entries {
            self.performance_history.drain(0..self.performance_history.len() - self.config.max_history_entries);
        }
    }

    /// Start a profiling session
    fn start_session(&mut self, name: String) {
        let session = ProfilingSession {
            start_time: Instant::now(),
            name,
            phase_starts: HashMap::new(),
            metrics: PerformanceMetrics::default(),
        };
        self.current_session = Some(session);
    }

    /// Finalize a profiling session
    fn finalize_session(&mut self, session: ProfilingSession) {
        let mut metrics = session.metrics;
        
        // Calculate derived statistics
        if metrics.total_time.as_secs_f64() > 0.0 {
            metrics.node_stats.nodes_per_second = 
                metrics.node_stats.total_nodes as f64 / metrics.total_time.as_secs_f64();
            
            metrics.constraint_stats.constraints_per_second = 
                metrics.constraint_stats.total_constraints as f64 / metrics.total_time.as_secs_f64();
        }

        if metrics.node_stats.total_nodes > 0 {
            let total_node_time: Duration = metrics.node_stats.node_type_times.values().sum();
            metrics.node_stats.avg_node_time = total_node_time / metrics.node_stats.total_nodes as u32;
        }

        // Store in history
        self.performance_history.push(metrics);
        
        // Limit history size
        if self.performance_history.len() > self.config.max_history_entries {
            self.performance_history.remove(0);
        }
    }
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            component_usage: HashMap::new(),
            allocation_count: 0,
        }
    }

    fn record_allocation(&mut self, component: &str, size: usize) {
        self.current_usage += size;
        self.allocation_count += 1;
        
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
        
        *self.component_usage.entry(component.to_string()).or_insert(0) += size;
    }

    fn reset(&mut self) {
        self.current_usage = 0;
        self.peak_usage = 0;
        self.component_usage.clear();
        self.allocation_count = 0;
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_detailed_profiling: true,
            enable_memory_tracking: true,
            enable_cache_profiling: true,
            sample_rate: 1.0,
            max_history_entries: 100,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time: Duration::ZERO,
            phase_times: HashMap::new(),
            memory_stats: MemoryStats::default(),
            node_stats: NodeProcessingStats::default(),
            constraint_stats: ConstraintProcessingStats::default(),
            cache_stats: CacheStats::default(),
            ai_metadata_time: Duration::ZERO,
        }
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            peak_memory: 0,
            current_memory: 0,
            component_memory: HashMap::new(),
            allocation_count: 0,
        }
    }
}

impl Default for NodeProcessingStats {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            nodes_per_second: 0.0,
            node_type_times: HashMap::new(),
            avg_node_time: Duration::ZERO,
        }
    }
}

impl Default for ConstraintProcessingStats {
    fn default() -> Self {
        Self {
            total_constraints: 0,
            constraints_per_second: 0.0,
            solving_time: Duration::ZERO,
            unification_steps: 0,
        }
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            total_queries: 0,
            cache_hits: 0,
            cache_misses: 0,
            cache_size: 0,
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_inference_time: Duration::ZERO,
            peak_memory_usage: 0,
            cache_hit_rate: 0.0,
            throughput: 0.0,
            error_rate: 0.0,
        }
    }
}

impl Default for InferenceProfiler {
    fn default() -> Self {
        Self::new()
    }
} 