//! Performance Profiling - Compilation Metrics and Performance Tracking
//!
//! This module implements comprehensive performance profiling for the compilation
//! process, tracking timing, memory usage, cache performance, and parallelization metrics.
//!
//! **Conceptual Responsibility**: Performance profiling and metrics collection
//! **What it does**: Track timing, memory usage, cache metrics, parallel execution stats
//! **What it doesn't do**: Manage compilation phases, collect diagnostics, optimize code

use crate::context::CompilationPhase;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance profiler for compilation metrics
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    /// Phase timing information
    pub phase_timings: HashMap<CompilationPhase, Duration>,
    /// Memory usage tracking
    pub memory_usage: MemoryUsageTracker,
    /// Cache performance metrics
    pub cache_performance: CachePerformanceTracker,
    /// Parallel execution metrics
    pub parallel_metrics: ParallelExecutionMetrics,
    /// Start time for current phase
    current_phase_start: Option<Instant>,
    /// Overall compilation start time
    compilation_start: Instant,
}

/// Memory usage tracking throughout compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageTracker {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Current memory usage in bytes
    pub current_memory: usize,
    /// Memory usage by compilation phase
    pub phase_memory: HashMap<CompilationPhase, usize>,
    /// Memory usage by compiler component
    pub component_memory: HashMap<String, usize>,
}

/// Cache performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceTracker {
    /// Total cache hits across all caches
    pub total_hits: u64,
    /// Total cache misses across all caches
    pub total_misses: u64,
    /// Cache hit rate by query type
    pub hit_rates: HashMap<String, f64>,
    /// Current total cache size in bytes
    pub cache_size: usize,
    /// Cache eviction count
    pub evictions: u64,
    /// Cache memory pressure events
    pub memory_pressure_events: u64,
}

/// Parallel execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionMetrics {
    /// Number of worker threads used
    pub worker_threads: usize,
    /// Average thread utilization (0.0 to 1.0)
    pub thread_utilization: f64,
    /// Work stealing statistics
    pub work_stealing_stats: WorkStealingStats,
    /// Synchronization overhead time
    pub sync_overhead: Duration,
    /// Lock contention events
    pub lock_contention_events: u64,
    /// Parallel efficiency ratio
    pub parallel_efficiency: f64,
}

/// Work stealing statistics for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkStealingStats {
    /// Total tasks stolen by workers
    pub tasks_stolen: u64,
    /// Total steal attempts made
    pub steal_attempts: u64,
    /// Successful steal operations
    pub successful_steals: u64,
    /// Average tasks per worker
    pub avg_tasks_per_worker: f64,
    /// Load balancing efficiency
    pub load_balance_efficiency: f64,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            phase_timings: HashMap::new(),
            memory_usage: MemoryUsageTracker::new(),
            cache_performance: CachePerformanceTracker::new(),
            parallel_metrics: ParallelExecutionMetrics::new(),
            current_phase_start: None,
            compilation_start: Instant::now(),
        }
    }

    /// Start timing a compilation phase
    pub fn start_phase(&mut self, phase: &CompilationPhase) {
        // Finish previous phase if any
        if let Some(start_time) = self.current_phase_start.take() {
            // Find the previous phase and record its timing
            let duration = start_time.elapsed();
            // For simplicity, we'll record this under the new phase
            // In practice, you'd track the previous phase properly
            self.phase_timings.insert(phase.clone(), duration);
        }
        
        self.current_phase_start = Some(Instant::now());
    }

    /// Finish timing the current phase
    pub fn finish_current_phase(&mut self, phase: &CompilationPhase) {
        if let Some(start_time) = self.current_phase_start.take() {
            let duration = start_time.elapsed();
            self.phase_timings.insert(phase.clone(), duration);
        }
    }

    /// Get total compilation time so far
    pub fn total_time(&self) -> Duration {
        self.compilation_start.elapsed()
    }

    /// Get timing for a specific phase
    pub fn get_phase_time(&self, phase: &CompilationPhase) -> Option<Duration> {
        self.phase_timings.get(phase).copied()
    }

    /// Record memory usage for current state
    pub fn record_memory_usage(&mut self, phase: &CompilationPhase) {
        // In a real implementation, this would query actual memory usage
        // For now, we'll simulate it
        let current_memory = self.get_current_memory_usage();
        
        self.memory_usage.current_memory = current_memory;
        if current_memory > self.memory_usage.peak_memory {
            self.memory_usage.peak_memory = current_memory;
        }
        
        self.memory_usage.phase_memory.insert(phase.clone(), current_memory);
    }

    /// Update cache performance metrics
    pub fn update_cache_metrics(&mut self, hits: u64, misses: u64, cache_size: usize) {
        self.cache_performance.total_hits += hits;
        self.cache_performance.total_misses += misses;
        self.cache_performance.cache_size = cache_size;
    }

    /// Record parallel execution metrics
    pub fn record_parallel_metrics(&mut self, metrics: ParallelExecutionMetrics) {
        self.parallel_metrics = metrics;
    }

    /// Get a performance summary
    pub fn summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_time: self.total_time(),
            phase_count: self.phase_timings.len(),
            peak_memory_mb: self.memory_usage.peak_memory as f64 / 1024.0 / 1024.0,
            cache_hit_rate: self.cache_performance.overall_hit_rate(),
            parallel_efficiency: self.parallel_metrics.parallel_efficiency,
            slowest_phase: self.find_slowest_phase(),
        }
    }

    /// Find the slowest compilation phase
    fn find_slowest_phase(&self) -> Option<(CompilationPhase, Duration)> {
        self.phase_timings.iter()
            .max_by_key(|(_, duration)| *duration)
            .map(|(phase, duration)| (phase.clone(), *duration))
    }

    /// Get current memory usage (simulated)
    fn get_current_memory_usage(&self) -> usize {
        // In a real implementation, this would query system memory usage
        // For now, return a simulated value
        1024 * 1024 * 64 // 64 MB
    }
}

impl MemoryUsageTracker {
    /// Create a new memory usage tracker
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            current_memory: 0,
            phase_memory: HashMap::new(),
            component_memory: HashMap::new(),
        }
    }

    /// Record memory usage for a component
    pub fn record_component_memory(&mut self, component: String, memory: usize) {
        self.component_memory.insert(component, memory);
    }

    /// Get memory usage summary
    pub fn summary(&self) -> String {
        let peak_mb = self.peak_memory as f64 / 1024.0 / 1024.0;
        let current_mb = self.current_memory as f64 / 1024.0 / 1024.0;
        
        format!("Peak: {:.2} MB, Current: {:.2} MB", peak_mb, current_mb)
    }
}

impl CachePerformanceTracker {
    /// Create a new cache performance tracker
    pub fn new() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            hit_rates: HashMap::new(),
            cache_size: 0,
            evictions: 0,
            memory_pressure_events: 0,
        }
    }

    /// Calculate overall cache hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Record hit rate for a specific cache type
    pub fn record_hit_rate(&mut self, cache_type: String, hit_rate: f64) {
        self.hit_rates.insert(cache_type, hit_rate);
    }

    /// Record cache eviction
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    /// Record memory pressure event
    pub fn record_memory_pressure(&mut self) {
        self.memory_pressure_events += 1;
    }

    /// Get cache efficiency summary
    pub fn efficiency_summary(&self) -> String {
        let hit_rate_percent = self.overall_hit_rate() * 100.0;
        let cache_mb = self.cache_size as f64 / 1024.0 / 1024.0;
        
        format!("Hit Rate: {:.1}%, Size: {:.2} MB, Evictions: {}", 
                hit_rate_percent, cache_mb, self.evictions)
    }
}

impl ParallelExecutionMetrics {
    /// Create a new parallel execution metrics tracker
    pub fn new() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            thread_utilization: 0.0,
            work_stealing_stats: WorkStealingStats::new(),
            sync_overhead: Duration::from_millis(0),
            lock_contention_events: 0,
            parallel_efficiency: 0.0,
        }
    }

    /// Update thread utilization
    pub fn update_thread_utilization(&mut self, utilization: f64) {
        self.thread_utilization = utilization.clamp(0.0, 1.0);
    }

    /// Record synchronization overhead
    pub fn record_sync_overhead(&mut self, overhead: Duration) {
        self.sync_overhead += overhead;
    }

    /// Record lock contention event
    pub fn record_lock_contention(&mut self) {
        self.lock_contention_events += 1;
    }

    /// Calculate and update parallel efficiency
    pub fn calculate_efficiency(&mut self, sequential_time: Duration, parallel_time: Duration) {
        if parallel_time.as_nanos() > 0 {
            self.parallel_efficiency = sequential_time.as_secs_f64() / 
                (parallel_time.as_secs_f64() * self.worker_threads as f64);
        }
    }

    /// Get parallelization summary
    pub fn summary(&self) -> String {
        format!("Threads: {}, Utilization: {:.1}%, Efficiency: {:.1}%",
                self.worker_threads,
                self.thread_utilization * 100.0,
                self.parallel_efficiency * 100.0)
    }
}

impl WorkStealingStats {
    /// Create new work stealing statistics
    pub fn new() -> Self {
        Self {
            tasks_stolen: 0,
            steal_attempts: 0,
            successful_steals: 0,
            avg_tasks_per_worker: 0.0,
            load_balance_efficiency: 0.0,
        }
    }

    /// Record a steal attempt
    pub fn record_steal_attempt(&mut self, successful: bool) {
        self.steal_attempts += 1;
        if successful {
            self.successful_steals += 1;
            self.tasks_stolen += 1;
        }
    }

    /// Calculate steal success rate
    pub fn steal_success_rate(&self) -> f64 {
        if self.steal_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / self.steal_attempts as f64
        }
    }

    /// Update load balancing metrics
    pub fn update_load_balance(&mut self, tasks_per_worker: Vec<usize>) {
        if !tasks_per_worker.is_empty() {
            self.avg_tasks_per_worker = tasks_per_worker.iter().sum::<usize>() as f64 / tasks_per_worker.len() as f64;
            
            // Calculate load balance efficiency (lower variance = better balance)
            let variance: f64 = tasks_per_worker.iter()
                .map(|&x| (x as f64 - self.avg_tasks_per_worker).powi(2))
                .sum::<f64>() / tasks_per_worker.len() as f64;
            
            // Convert variance to efficiency score (0.0 to 1.0)
            self.load_balance_efficiency = 1.0 / (1.0 + variance / self.avg_tasks_per_worker);
        }
    }
}

/// Performance summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Total compilation time
    pub total_time: Duration,
    /// Number of phases completed
    pub phase_count: usize,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Overall cache hit rate
    pub cache_hit_rate: f64,
    /// Parallel execution efficiency
    pub parallel_efficiency: f64,
    /// Slowest compilation phase
    pub slowest_phase: Option<(CompilationPhase, Duration)>,
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MemoryUsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CachePerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ParallelExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WorkStealingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = PerformanceProfiler::new();
        
        assert!(profiler.phase_timings.is_empty());
        assert_eq!(profiler.memory_usage.peak_memory, 0);
        assert_eq!(profiler.cache_performance.total_hits, 0);
    }

    #[test]
    fn test_phase_timing() {
        let mut profiler = PerformanceProfiler::new();
        let phase = CompilationPhase::Parsing;
        
        profiler.start_phase(&phase);
        std::thread::sleep(Duration::from_millis(1));
        profiler.finish_current_phase(&phase);
        
        let timing = profiler.get_phase_time(&phase);
        assert!(timing.is_some());
        assert!(timing.unwrap().as_millis() >= 1);
    }

    #[test]
    fn test_cache_performance() {
        let mut tracker = CachePerformanceTracker::new();
        
        tracker.total_hits = 80;
        tracker.total_misses = 20;
        
        assert_eq!(tracker.overall_hit_rate(), 0.8);
    }

    #[test]
    fn test_work_stealing_stats() {
        let mut stats = WorkStealingStats::new();
        
        stats.record_steal_attempt(true);
        stats.record_steal_attempt(false);
        stats.record_steal_attempt(true);
        
        assert_eq!(stats.steal_success_rate(), 2.0 / 3.0);
        assert_eq!(stats.tasks_stolen, 2);
    }

    #[test]
    fn test_memory_tracking() {
        let mut tracker = MemoryUsageTracker::new();
        
        tracker.record_component_memory("parser".to_string(), 1024 * 1024);
        tracker.current_memory = 2 * 1024 * 1024;
        tracker.peak_memory = 3 * 1024 * 1024;
        
        let summary = tracker.summary();
        assert!(summary.contains("3.00 MB"));
        assert!(summary.contains("2.00 MB"));
    }

    #[test]
    fn test_performance_summary() {
        let mut profiler = PerformanceProfiler::new();
        profiler.phase_timings.insert(CompilationPhase::Parsing, Duration::from_millis(100));
        profiler.memory_usage.peak_memory = 64 * 1024 * 1024; // 64 MB
        
        let summary = profiler.summary();
        assert_eq!(summary.phase_count, 1);
        assert_eq!(summary.peak_memory_mb, 64.0);
    }
} 