//! Stack Performance Analytics
//!
//! This module provides comprehensive performance monitoring and analytics for stack
//! operations, integrating with the existing prism-runtime performance systems.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};

/// Stack performance analytics system
#[derive(Debug)]
pub struct StackAnalytics {
    /// Performance metrics
    metrics: Arc<RwLock<StackPerformanceMetrics>>,
    
    /// Performance history
    history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    
    /// Configuration
    config: AnalyticsConfig,
    
    /// Operation timings
    operation_timings: Arc<RwLock<HashMap<String, OperationTiming>>>,
}

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable detailed profiling
    pub detailed_profiling: bool,
    
    /// History retention count
    pub history_retention: usize,
    
    /// Sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    
    /// Enable hotspot detection
    pub hotspot_detection: bool,
    
    /// Hotspot threshold (operations per second)
    pub hotspot_threshold: f64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            detailed_profiling: true,
            history_retention: 1000,
            sampling_interval_ms: 100,
            hotspot_detection: true,
            hotspot_threshold: 1000.0,
        }
    }
}

/// Stack performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackPerformanceMetrics {
    /// Total stack operations performed
    pub total_operations: u64,
    
    /// Operations per second
    pub operations_per_second: f64,
    
    /// Average operation latency in microseconds
    pub avg_latency_us: f64,
    
    /// Peak stack depth reached
    pub peak_stack_depth: usize,
    
    /// Average stack depth
    pub avg_stack_depth: f64,
    
    /// Frame creation rate (frames/sec)
    pub frame_creation_rate: f64,
    
    /// Frame destruction rate (frames/sec)
    pub frame_destruction_rate: f64,
    
    /// Memory efficiency (% of allocated memory actually used)
    pub memory_efficiency: f64,
    
    /// Cache hit rate for stack operations
    pub cache_hit_rate: f64,
    
    /// Hotspots detected
    pub hotspots: Vec<PerformanceHotspot>,
    
    /// Performance by operation type
    pub operation_breakdown: HashMap<String, OperationMetrics>,
}

impl Default for StackPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_per_second: 0.0,
            avg_latency_us: 0.0,
            peak_stack_depth: 0,
            avg_stack_depth: 0.0,
            frame_creation_rate: 0.0,
            frame_destruction_rate: 0.0,
            memory_efficiency: 0.0,
            cache_hit_rate: 0.0,
            hotspots: Vec::new(),
            operation_breakdown: HashMap::new(),
        }
    }
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// When the snapshot was taken
    pub timestamp: Instant,
    
    /// Stack depth at snapshot
    pub stack_depth: usize,
    
    /// Operations per second at snapshot
    pub ops_per_second: f64,
    
    /// Memory usage at snapshot
    pub memory_usage_bytes: usize,
    
    /// Active frame count
    pub active_frames: usize,
}

/// Performance hotspot detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHotspot {
    /// Function name where hotspot detected
    pub function_name: String,
    
    /// Operation type causing hotspot
    pub operation_type: String,
    
    /// Frequency (operations per second)
    pub frequency: f64,
    
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    
    /// Severity level
    pub severity: HotspotSeverity,
    
    /// Optimization suggestions
    pub suggestions: Vec<String>,
}

/// Hotspot severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotspotSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Metrics for specific operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Total count of this operation
    pub count: u64,
    
    /// Total time spent in microseconds
    pub total_time_us: u64,
    
    /// Average time per operation
    pub avg_time_us: f64,
    
    /// Minimum time observed
    pub min_time_us: u64,
    
    /// Maximum time observed
    pub max_time_us: u64,
    
    /// 95th percentile time
    pub p95_time_us: u64,
    
    /// 99th percentile time
    pub p99_time_us: u64,
}

impl Default for OperationMetrics {
    fn default() -> Self {
        Self {
            count: 0,
            total_time_us: 0,
            avg_time_us: 0.0,
            min_time_us: u64::MAX,
            max_time_us: 0,
            p95_time_us: 0,
            p99_time_us: 0,
        }
    }
}

/// Operation timing tracker
#[derive(Debug, Clone)]
pub struct OperationTiming {
    /// Recent timing samples
    pub samples: VecDeque<u64>,
    
    /// Total operations
    pub total_ops: u64,
    
    /// Last update time
    pub last_update: Instant,
}

impl OperationTiming {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(1000), // Keep last 1000 samples
            total_ops: 0,
            last_update: Instant::now(),
        }
    }

    fn add_sample(&mut self, duration_us: u64) {
        if self.samples.len() >= 1000 {
            self.samples.pop_front();
        }
        self.samples.push_back(duration_us);
        self.total_ops += 1;
        self.last_update = Instant::now();
    }

    fn calculate_metrics(&self) -> OperationMetrics {
        if self.samples.is_empty() {
            return OperationMetrics::default();
        }

        let mut sorted_samples: Vec<u64> = self.samples.iter().copied().collect();
        sorted_samples.sort_unstable();

        let total_time_us: u64 = sorted_samples.iter().sum();
        let count = sorted_samples.len() as u64;
        let avg_time_us = total_time_us as f64 / count as f64;
        let min_time_us = *sorted_samples.first().unwrap_or(&0);
        let max_time_us = *sorted_samples.last().unwrap_or(&0);
        
        let p95_index = ((count as f64 * 0.95) as usize).min(count as usize - 1);
        let p99_index = ((count as f64 * 0.99) as usize).min(count as usize - 1);
        let p95_time_us = sorted_samples.get(p95_index).copied().unwrap_or(0);
        let p99_time_us = sorted_samples.get(p99_index).copied().unwrap_or(0);

        OperationMetrics {
            count: self.total_ops,
            total_time_us,
            avg_time_us,
            min_time_us,
            max_time_us,
            p95_time_us,
            p99_time_us,
        }
    }
}

impl StackAnalytics {
    /// Create a new stack analytics system
    pub fn new() -> VMResult<Self> {
        let _span = span!(Level::INFO, "stack_analytics_init").entered();
        info!("Initializing stack performance analytics");

        Ok(Self {
            metrics: Arc::new(RwLock::new(StackPerformanceMetrics::default())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            config: AnalyticsConfig::default(),
            operation_timings: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Record a stack operation
    pub fn record_operation(&self, operation_type: &str, duration: Duration) {
        let duration_us = duration.as_micros() as u64;
        
        // Update operation timings
        {
            let mut timings = self.operation_timings.write().unwrap();
            let timing = timings.entry(operation_type.to_string())
                .or_insert_with(OperationTiming::new);
            timing.add_sample(duration_us);
        }

        // Update overall metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_operations += 1;
            
            // Update average latency using running average
            if metrics.total_operations == 1 {
                metrics.avg_latency_us = duration_us as f64;
            } else {
                let alpha = 0.1; // Exponential moving average factor
                metrics.avg_latency_us = alpha * duration_us as f64 + (1.0 - alpha) * metrics.avg_latency_us;
            }
        }

        debug!("Recorded {} operation: {}μs", operation_type, duration_us);
    }

    /// Record stack depth change
    pub fn record_stack_depth(&self, current_depth: usize) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.peak_stack_depth = metrics.peak_stack_depth.max(current_depth);
        
        // Update average stack depth using exponential moving average
        let alpha = 0.05;
        metrics.avg_stack_depth = alpha * current_depth as f64 + (1.0 - alpha) * metrics.avg_stack_depth;
    }

    /// Record frame creation
    pub fn record_frame_creation(&self, function_name: &str) {
        let start_time = Instant::now();
        
        // This would be called when frame creation completes
        self.record_operation("frame_creation", start_time.elapsed());
        
        // Update frame creation rate
        let mut metrics = self.metrics.write().unwrap();
        // This is a simplified rate calculation - in practice, you'd want a time window
        metrics.frame_creation_rate += 1.0;
        
        debug!("Recorded frame creation for: {}", function_name);
    }

    /// Record frame destruction
    pub fn record_frame_destruction(&self, function_name: &str) {
        let start_time = Instant::now();
        
        self.record_operation("frame_destruction", start_time.elapsed());
        
        // Update frame destruction rate
        let mut metrics = self.metrics.write().unwrap();
        metrics.frame_destruction_rate += 1.0;
        
        debug!("Recorded frame destruction for: {}", function_name);
    }

    /// Take a performance snapshot
    pub fn take_snapshot(&self, stack: &ExecutionStack) {
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            stack_depth: stack.frame_count(),
            ops_per_second: self.calculate_ops_per_second(),
            memory_usage_bytes: self.estimate_memory_usage(stack),
            active_frames: stack.frame_count(),
        };

        // Add to history
        {
            let mut history = self.history.write().unwrap();
            if history.len() >= self.config.history_retention {
                history.pop_front();
            }
            history.push_back(snapshot);
        }

        // Update operations per second in metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.operations_per_second = self.calculate_ops_per_second();
        }
    }

    /// Detect performance hotspots
    pub fn detect_hotspots(&self) -> Vec<PerformanceHotspot> {
        if !self.config.hotspot_detection {
            return Vec::new();
        }

        let mut hotspots = Vec::new();
        let timings = self.operation_timings.read().unwrap();

        for (operation_type, timing) in timings.iter() {
            let metrics = timing.calculate_metrics();
            let frequency = self.calculate_operation_frequency(timing);

            if frequency > self.config.hotspot_threshold {
                let severity = match frequency {
                    f if f > 10000.0 => HotspotSeverity::Critical,
                    f if f > 5000.0 => HotspotSeverity::High,
                    f if f > 2000.0 => HotspotSeverity::Medium,
                    _ => HotspotSeverity::Low,
                };

                let suggestions = self.generate_optimization_suggestions(operation_type, &metrics);

                hotspots.push(PerformanceHotspot {
                    function_name: "unknown".to_string(), // Would need more context tracking
                    operation_type: operation_type.clone(),
                    frequency,
                    avg_latency_us: metrics.avg_time_us,
                    severity,
                    suggestions,
                });
            }
        }

        // Update hotspots in metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.hotspots = hotspots.clone();
        }

        hotspots
    }

    /// Calculate operations per second
    fn calculate_ops_per_second(&self) -> f64 {
        let history = self.history.read().unwrap();
        if history.len() < 2 {
            return 0.0;
        }

        let recent = history.back().unwrap();
        let older = history.get(history.len().saturating_sub(10)).unwrap_or(history.front().unwrap());
        
        let time_diff = recent.timestamp.duration_since(older.timestamp);
        if time_diff.as_secs_f64() > 0.0 {
            let ops_diff = recent.ops_per_second - older.ops_per_second;
            ops_diff / time_diff.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate operation frequency
    fn calculate_operation_frequency(&self, timing: &OperationTiming) -> f64 {
        let elapsed = timing.last_update.elapsed();
        if elapsed.as_secs_f64() > 0.0 {
            timing.total_ops as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, stack: &ExecutionStack) -> usize {
        // Rough estimation based on stack statistics
        let stats = stack.statistics();
        let frame_size = std::mem::size_of::<StackFrame>();
        let value_size = std::mem::size_of::<StackValue>();
        
        stats.frame_count * frame_size + stats.current_size * value_size
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(&self, operation_type: &str, metrics: &OperationMetrics) -> Vec<String> {
        let mut suggestions = Vec::new();

        match operation_type {
            "frame_creation" => {
                if metrics.avg_time_us > 100.0 {
                    suggestions.push("Consider frame pooling to reduce allocation overhead".to_string());
                }
                if metrics.p99_time_us > metrics.avg_time_us * 3.0 {
                    suggestions.push("High latency variance - investigate memory pressure".to_string());
                }
            }
            "stack_push" | "stack_pop" => {
                if metrics.avg_time_us > 10.0 {
                    suggestions.push("Stack operations are slow - check for memory fragmentation".to_string());
                }
            }
            "local_access" => {
                if metrics.count > 10000 {
                    suggestions.push("High local variable access - consider optimization".to_string());
                }
            }
            _ => {
                if metrics.avg_time_us > 50.0 {
                    suggestions.push(format!("Operation {} is slower than expected", operation_type));
                }
            }
        }

        suggestions
    }

    /// Get current performance metrics
    pub fn current_metrics(&self) -> StackPerformanceMetrics {
        // Update operation breakdown before returning
        {
            let mut metrics = self.metrics.write().unwrap();
            let timings = self.operation_timings.read().unwrap();
            
            metrics.operation_breakdown.clear();
            for (op_type, timing) in timings.iter() {
                metrics.operation_breakdown.insert(op_type.clone(), timing.calculate_metrics());
            }
        }

        self.metrics.read().unwrap().clone()
    }

    /// Get performance history
    pub fn performance_history(&self, count: usize) -> Vec<PerformanceSnapshot> {
        let history = self.history.read().unwrap();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Generate performance report for AI analysis
    pub fn generate_ai_report(&self) -> StackPerformanceAIReport {
        let metrics = self.current_metrics();
        let hotspots = self.detect_hotspots();
        let history = self.performance_history(100);

        StackPerformanceAIReport {
            summary: PerformanceSummary {
                overall_health: self.assess_performance_health(&metrics),
                primary_bottlenecks: self.identify_bottlenecks(&metrics),
                optimization_priority: self.prioritize_optimizations(&metrics),
            },
            metrics,
            hotspots,
            trends: self.analyze_trends(&history),
            recommendations: self.generate_ai_recommendations(&metrics),
        }
    }

    /// Assess overall performance health
    fn assess_performance_health(&self, metrics: &StackPerformanceMetrics) -> HealthStatus {
        if metrics.avg_latency_us > 1000.0 || !metrics.hotspots.is_empty() {
            HealthStatus::Poor
        } else if metrics.avg_latency_us > 500.0 {
            HealthStatus::Fair
        } else if metrics.avg_latency_us > 100.0 {
            HealthStatus::Good
        } else {
            HealthStatus::Excellent
        }
    }

    /// Identify primary bottlenecks
    fn identify_bottlenecks(&self, metrics: &StackPerformanceMetrics) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        if metrics.avg_latency_us > 500.0 {
            bottlenecks.push("High average latency".to_string());
        }

        if metrics.memory_efficiency < 0.7 {
            bottlenecks.push("Low memory efficiency".to_string());
        }

        if metrics.cache_hit_rate < 0.8 {
            bottlenecks.push("Poor cache performance".to_string());
        }

        bottlenecks
    }

    /// Prioritize optimizations
    fn prioritize_optimizations(&self, _metrics: &StackPerformanceMetrics) -> Vec<String> {
        vec![
            "Implement frame pooling".to_string(),
            "Optimize memory layout".to_string(),
            "Improve cache locality".to_string(),
        ]
    }

    /// Analyze performance trends
    fn analyze_trends(&self, history: &[PerformanceSnapshot]) -> Vec<String> {
        let mut trends = Vec::new();

        if history.len() >= 10 {
            let recent_avg = history.iter().rev().take(5).map(|s| s.ops_per_second).sum::<f64>() / 5.0;
            let older_avg = history.iter().take(5).map(|s| s.ops_per_second).sum::<f64>() / 5.0;

            if recent_avg > older_avg * 1.1 {
                trends.push("Performance improving".to_string());
            } else if recent_avg < older_avg * 0.9 {
                trends.push("Performance degrading".to_string());
            } else {
                trends.push("Performance stable".to_string());
            }
        }

        trends
    }

    /// Generate AI recommendations
    fn generate_ai_recommendations(&self, metrics: &StackPerformanceMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        if metrics.peak_stack_depth > 1000 {
            recommendations.push("Consider tail call optimization to reduce stack depth".to_string());
        }

        if metrics.frame_creation_rate > metrics.frame_destruction_rate * 1.2 {
            recommendations.push("Frame creation rate exceeds destruction - check for leaks".to_string());
        }

        recommendations
    }
}

/// AI performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackPerformanceAIReport {
    /// Performance summary
    pub summary: PerformanceSummary,
    
    /// Detailed metrics
    pub metrics: StackPerformanceMetrics,
    
    /// Detected hotspots
    pub hotspots: Vec<PerformanceHotspot>,
    
    /// Performance trends
    pub trends: Vec<String>,
    
    /// AI recommendations
    pub recommendations: Vec<String>,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Overall health assessment
    pub overall_health: HealthStatus,
    
    /// Primary bottlenecks identified
    pub primary_bottlenecks: Vec<String>,
    
    /// Optimization priorities
    pub optimization_priority: Vec<String>,
}

/// Health status levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
} 