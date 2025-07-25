//! Statistics - Comprehensive monitoring and reporting for allocators
//!
//! This module provides detailed statistics collection and reporting
//! for all allocator types, enabling performance monitoring, debugging,
//! and optimization.

use super::types::*;
use std::time::{Duration, Instant};
use std::sync::{Mutex, Arc};
use std::collections::{HashMap, VecDeque};

/// Comprehensive statistics collector for all allocators
pub struct StatisticsCollector {
    /// Global statistics
    global_stats: Arc<Mutex<GlobalStats>>,
    /// Per-allocator statistics
    allocator_stats: Arc<Mutex<HashMap<String, AllocatorSpecificStats>>>,
    /// Collection start time
    start_time: Instant,
    /// Configuration
    config: StatisticsConfig,
    /// Timing data for latency calculations
    timing_data: Arc<Mutex<TimingDataCollector>>,
}

/// Collects timing data for performance analysis
#[derive(Debug, Default)]
struct TimingDataCollector {
    /// Recent allocation times (circular buffer)
    allocation_times: VecDeque<Duration>,
    /// Recent deallocation times (circular buffer)
    deallocation_times: VecDeque<Duration>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Cache hit/miss ratios from allocators
    cache_stats: CacheEfficiencyStats,
    /// GC timing data
    gc_stats: GcTimingStats,
}

#[derive(Debug, Default)]
struct CacheEfficiencyStats {
    total_requests: usize,
    cache_hits: usize,
    cache_misses: usize,
}

#[derive(Debug, Default)]
struct GcTimingStats {
    total_gc_time: Duration,
    gc_count: usize,
    total_runtime: Duration,
}

/// Configuration for statistics collection
#[derive(Debug, Clone)]
pub struct StatisticsConfig {
    /// Enable detailed per-size-class statistics
    pub enable_size_class_stats: bool,
    /// Enable thread-local statistics
    pub enable_thread_stats: bool,
    /// Enable timing statistics
    pub enable_timing_stats: bool,
    /// Statistics collection interval
    pub collection_interval: Duration,
    /// Maximum history entries to keep
    pub max_history_entries: usize,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            enable_size_class_stats: true,
            enable_thread_stats: true,
            enable_timing_stats: true,
            collection_interval: Duration::from_secs(1),
            max_history_entries: 1000,
        }
    }
}

/// Global statistics across all allocators
#[derive(Debug, Default, Clone)]
pub struct GlobalStats {
    /// Total allocations across all allocators
    pub total_allocations: usize,
    /// Total deallocations across all allocators
    pub total_deallocations: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Total bytes deallocated
    pub total_bytes_deallocated: usize,
    /// Current live bytes
    pub live_bytes: usize,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Number of GC triggers
    pub gc_triggers: usize,
    /// Total GC time
    pub total_gc_time: Duration,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    /// Memory efficiency (live_bytes / total_allocated)
    pub memory_efficiency: f64,
    /// Fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Statistics specific to an allocator type
#[derive(Debug, Clone)]
pub struct AllocatorSpecificStats {
    /// Allocator name
    pub name: String,
    /// Basic allocation statistics
    pub basic: AllocationStats,
    /// Size class statistics (if applicable)
    pub size_class_stats: Option<Vec<SizeClassStats>>,
    /// Thread cache statistics (if applicable)
    pub thread_cache_stats: Option<Vec<ThreadCacheStats>>,
    /// Timing statistics
    pub timing_stats: TimingStats,
    /// Historical data
    pub history: Vec<HistoricalEntry>,
}

/// Statistics for a specific size class
#[derive(Debug, Clone)]
pub struct SizeClassStats {
    /// Size class index
    pub size_class_index: usize,
    /// Object size for this class
    pub object_size: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Number of deallocations
    pub deallocations: usize,
    /// Current live objects
    pub live_objects: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Fragmentation in this size class
    pub fragmentation: f64,
}

/// Timing statistics for performance analysis
#[derive(Debug, Default, Clone)]
pub struct TimingStats {
    /// Average allocation time
    pub avg_allocation_time: Duration,
    /// Average deallocation time
    pub avg_deallocation_time: Duration,
    /// 95th percentile allocation time
    pub p95_allocation_time: Duration,
    /// 99th percentile allocation time
    pub p99_allocation_time: Duration,
    /// Fastest allocation time
    pub min_allocation_time: Duration,
    /// Slowest allocation time
    pub max_allocation_time: Duration,
    /// Total time spent in allocation
    pub total_allocation_time: Duration,
    /// Total time spent in deallocation
    pub total_deallocation_time: Duration,
}

/// Historical statistics entry
#[derive(Debug, Clone)]
pub struct HistoricalEntry {
    /// Timestamp
    pub timestamp: Instant,
    /// Snapshot of statistics at this time
    pub stats: AllocationStats,
    /// Memory usage at this time
    pub memory_usage: MemoryUsage,
}

/// Performance metrics for analysis
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Throughput (allocations per second)
    pub throughput: f64,
    /// Latency percentiles
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Cache efficiency
    pub cache_efficiency: f64,
    /// GC overhead
    pub gc_overhead: f64,
}

/// Trend analysis for statistics
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Allocation rate trend (positive = increasing)
    pub allocation_rate_trend: f64,
    /// Memory usage trend
    pub memory_usage_trend: f64,
    /// Fragmentation trend
    pub fragmentation_trend: f64,
    /// Performance trend
    pub performance_trend: f64,
    /// Confidence level of trends (0.0 to 1.0)
    pub confidence: f64,
}

impl StatisticsCollector {
    pub fn new() -> Self {
        Self::with_config(StatisticsConfig::default())
    }
    
    pub fn with_config(config: StatisticsConfig) -> Self {
        Self {
            global_stats: Arc::new(Mutex::new(GlobalStats::default())),
            allocator_stats: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            config,
            timing_data: Arc::new(Mutex::new(TimingDataCollector {
                max_samples: 10000,
                ..Default::default()
            })),
        }
    }
    
    /// Register a new allocator for statistics tracking
    pub fn register_allocator(&self, name: String) {
        if let Ok(mut stats) = self.allocator_stats.lock() {
            stats.insert(name.clone(), AllocatorSpecificStats {
                name,
                basic: AllocationStats::default(),
                size_class_stats: if self.config.enable_size_class_stats {
                    Some(Vec::new())
                } else {
                    None
                },
                thread_cache_stats: if self.config.enable_thread_stats {
                    Some(Vec::new())
                } else {
                    None
                },
                timing_stats: TimingStats::default(),
                history: Vec::new(),
            });
        }
    }
    
    /// Update statistics for a specific allocator
    pub fn update_allocator_stats(&self, name: &str, stats: AllocationStats) {
        if let Ok(mut allocator_stats) = self.allocator_stats.lock() {
            if let Some(entry) = allocator_stats.get_mut(name) {
                entry.basic = stats;
                
                // Add to history if enabled
                if entry.history.len() >= self.config.max_history_entries {
                    entry.history.remove(0);
                }
                
                entry.history.push(HistoricalEntry {
                    timestamp: Instant::now(),
                    stats,
                    memory_usage: MemoryUsage {
                        total_capacity: stats.peak_memory,
                        allocated: stats.live_bytes,
                        free: stats.peak_memory.saturating_sub(stats.live_bytes),
                        regions_count: 0, // Would be filled by specific allocators
                        average_allocation_size: if stats.allocation_count > 0 {
                            stats.total_allocated as f64 / stats.allocation_count as f64
                        } else {
                            0.0
                        },
                        allocation_rate: 0.0, // Would be calculated
                        fragmentation_ratio: if stats.total_allocated > 0 {
                            1.0 - (stats.live_bytes as f64 / stats.total_allocated as f64)
                        } else {
                            0.0
                        },
                    },
                });
            }
        }
        
        // Update global statistics
        self.update_global_stats();
    }
    
    /// Update global statistics by aggregating from all allocators
    fn update_global_stats(&self) {
        if let Ok(allocator_stats) = self.allocator_stats.lock() {
            if let Ok(mut global) = self.global_stats.lock() {
                // Reset global stats
                *global = GlobalStats::default();
                
                // Aggregate from all allocators
                for (_, allocator_stat) in allocator_stats.iter() {
                    let stats = &allocator_stat.basic;
                    
                    global.total_allocations += stats.allocation_count;
                    global.total_deallocations += stats.deallocation_count;
                    global.total_bytes_allocated += stats.total_allocated;
                    global.total_bytes_deallocated += stats.total_deallocated;
                    global.live_bytes += stats.live_bytes;
                    global.peak_memory = global.peak_memory.max(stats.peak_memory);
                }
                
                // Calculate derived metrics
                if global.total_allocations > 0 {
                    global.average_allocation_size = 
                        global.total_bytes_allocated as f64 / global.total_allocations as f64;
                }
                
                let elapsed = self.start_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    global.allocation_rate = global.total_allocations as f64 / elapsed;
                }
                
                if global.total_bytes_allocated > 0 {
                    global.memory_efficiency = global.live_bytes as f64 / global.total_bytes_allocated as f64;
                    global.fragmentation_ratio = 1.0 - global.memory_efficiency;
                }
            }
        }
    }
    
    /// Get current global statistics
    pub fn get_global_stats(&self) -> GlobalStats {
        self.global_stats.lock().unwrap().clone()
    }
    
    /// Get statistics for a specific allocator
    pub fn get_allocator_stats(&self, name: &str) -> Option<AllocatorSpecificStats> {
        self.allocator_stats.lock().unwrap().get(name).cloned()
    }
    
    /// Get all allocator statistics
    pub fn get_all_allocator_stats(&self) -> HashMap<String, AllocatorSpecificStats> {
        self.allocator_stats.lock().unwrap().clone()
    }
    
    /// Record allocation timing data
    pub fn record_allocation_time(&self, duration: Duration) {
        if self.config.enable_timing_stats {
            let mut timing_data = self.timing_data.lock().unwrap();
            timing_data.allocation_times.push_back(duration);
            
            // Keep circular buffer bounded
            if timing_data.allocation_times.len() > timing_data.max_samples {
                timing_data.allocation_times.pop_front();
            }
        }
    }
    
    /// Record deallocation timing data
    pub fn record_deallocation_time(&self, duration: Duration) {
        if self.config.enable_timing_stats {
            let mut timing_data = self.timing_data.lock().unwrap();
            timing_data.deallocation_times.push_back(duration);
            
            // Keep circular buffer bounded
            if timing_data.deallocation_times.len() > timing_data.max_samples {
                timing_data.deallocation_times.pop_front();
            }
        }
    }
    
    /// Record cache statistics
    pub fn record_cache_stats(&self, hits: usize, misses: usize) {
        let mut timing_data = self.timing_data.lock().unwrap();
        timing_data.cache_stats.total_requests += hits + misses;
        timing_data.cache_stats.cache_hits += hits;
        timing_data.cache_stats.cache_misses += misses;
    }
    
    /// Record GC timing
    pub fn record_gc_time(&self, gc_duration: Duration, total_runtime: Duration) {
        let mut timing_data = self.timing_data.lock().unwrap();
        timing_data.gc_stats.total_gc_time += gc_duration;
        timing_data.gc_stats.gc_count += 1;
        timing_data.gc_stats.total_runtime = total_runtime;
    }

    /// Calculate performance metrics with real data
    pub fn calculate_performance_metrics(&self) -> PerformanceMetrics {
        let global = self.get_global_stats();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        // Calculate basic metrics
        let throughput = if elapsed > 0.0 {
            global.total_allocations as f64 / elapsed
        } else {
            0.0
        };
        
        let memory_utilization = if global.peak_memory > 0 {
            global.live_bytes as f64 / global.peak_memory as f64
        } else {
            0.0
        };
        
        // Calculate real latency percentiles from collected timing data
        let timing_data = self.timing_data.lock().unwrap();
        let (latency_p50, latency_p95, latency_p99) = self.calculate_latency_percentiles(&timing_data.allocation_times);
        
        // Calculate cache efficiency from actual data
        let cache_efficiency = if timing_data.cache_stats.total_requests > 0 {
            timing_data.cache_stats.cache_hits as f64 / timing_data.cache_stats.total_requests as f64
        } else {
            0.0
        };
        
        // Calculate GC overhead from actual timing data
        let gc_overhead = if timing_data.gc_stats.total_runtime.as_nanos() > 0 {
            timing_data.gc_stats.total_gc_time.as_nanos() as f64 / timing_data.gc_stats.total_runtime.as_nanos() as f64
        } else {
            0.0
        };
        
        PerformanceMetrics {
            throughput,
            latency_p50,
            latency_p95,
            latency_p99,
            memory_utilization,
            cache_efficiency,
            gc_overhead,
        }
    }
    
    /// Calculate latency percentiles from timing samples
    fn calculate_latency_percentiles(&self, samples: &VecDeque<Duration>) -> (Duration, Duration, Duration) {
        if samples.is_empty() {
            return (Duration::from_nanos(0), Duration::from_nanos(0), Duration::from_nanos(0));
        }
        
        // Convert to sorted vector for percentile calculation
        let mut sorted_samples: Vec<Duration> = samples.iter().cloned().collect();
        sorted_samples.sort();
        
        let len = sorted_samples.len();
        let p50_idx = (len as f64 * 0.50) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;
        
        let p50 = sorted_samples.get(p50_idx.min(len - 1)).cloned().unwrap_or_default();
        let p95 = sorted_samples.get(p95_idx.min(len - 1)).cloned().unwrap_or_default();
        let p99 = sorted_samples.get(p99_idx.min(len - 1)).cloned().unwrap_or_default();
        
        (p50, p95, p99)
    }
    
    /// Analyze trends in the statistics with improved calculations
    pub fn analyze_trends(&self, name: &str, window_size: usize) -> Option<TrendAnalysis> {
        let allocator_stats = self.allocator_stats.lock().unwrap();
        let entry = allocator_stats.get(name)?;
        
        if entry.history.len() < window_size {
            return None; // Not enough data
        }
        
        let recent_entries = &entry.history[entry.history.len() - window_size..];
        
        // Calculate trends using simple linear regression
        let allocation_rate_trend = self.calculate_trend(
            &recent_entries.iter()
                .enumerate()
                .map(|(i, e)| (i as f64, e.stats.allocation_count as f64))
                .collect::<Vec<_>>()
        );
        
        let memory_usage_trend = self.calculate_trend(
            &recent_entries.iter()
                .enumerate()
                .map(|(i, e)| (i as f64, e.stats.live_bytes as f64))
                .collect::<Vec<_>>()
        );
        
        let fragmentation_trend = self.calculate_trend(
            &recent_entries.iter()
                .enumerate()
                .map(|(i, e)| (i as f64, e.memory_usage.fragmentation_ratio))
                .collect::<Vec<_>>()
        );
        
        // Calculate performance trend based on allocation rate and memory efficiency
        let performance_data: Vec<(f64, f64)> = recent_entries.iter()
            .enumerate()
            .map(|(i, e)| {
                let efficiency = if e.stats.total_allocated > 0 {
                    e.stats.live_bytes as f64 / e.stats.total_allocated as f64
                } else {
                    1.0
                };
                let rate_score = if e.memory_usage.allocation_rate > 0.0 {
                    1.0 / (1.0 + e.memory_usage.allocation_rate / 1000.0) // Normalize rate
                } else {
                    0.0
                };
                (i as f64, efficiency * rate_score)
            })
            .collect();
        
        let performance_trend = self.calculate_trend(&performance_data);
        
        // Calculate confidence based on data consistency and sample size
        let confidence = self.calculate_trend_confidence(recent_entries, window_size);
        
        Some(TrendAnalysis {
            allocation_rate_trend,
            memory_usage_trend,
            fragmentation_trend,
            performance_trend,
            confidence,
        })
    }
    
    /// Calculate confidence in trend analysis
    fn calculate_trend_confidence(&self, entries: &[HistoricalEntry], window_size: usize) -> f64 {
        if entries.len() < 3 {
            return 0.0;
        }
        
        // Base confidence on sample size
        let sample_confidence = (entries.len() as f64 / window_size as f64).min(1.0);
        
        // Adjust based on data variance (lower variance = higher confidence)
        let allocation_rates: Vec<f64> = entries.iter()
            .map(|e| e.memory_usage.allocation_rate)
            .collect();
        
        let mean_rate = allocation_rates.iter().sum::<f64>() / allocation_rates.len() as f64;
        let variance = allocation_rates.iter()
            .map(|&rate| (rate - mean_rate).powi(2))
            .sum::<f64>() / allocation_rates.len() as f64;
        
        let variance_confidence = if variance > 0.0 {
            1.0 / (1.0 + variance.sqrt() / mean_rate.max(1.0))
        } else {
            1.0
        };
        
        // Combine confidences
        (sample_confidence * 0.6 + variance_confidence * 0.4).clamp(0.0, 1.0)
    }
    
    /// Calculate linear trend from data points
    fn calculate_trend(&self, points: &[(f64, f64)]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }
        
        let n = points.len() as f64;
        let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = points.iter().map(|(x, _)| x * x).sum();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }
    
    /// Generate a comprehensive report
    pub fn generate_report(&self) -> StatisticsReport {
        let global = self.get_global_stats();
        let all_allocators = self.get_all_allocator_stats();
        let performance = self.calculate_performance_metrics();
        
        // Calculate trends for all allocators
        let mut trends = HashMap::new();
        for name in all_allocators.keys() {
            if let Some(trend) = self.analyze_trends(name, 10) {
                trends.insert(name.clone(), trend);
            }
        }
        
        StatisticsReport {
            timestamp: Instant::now(),
            uptime: self.start_time.elapsed(),
            global_stats: global,
            allocator_stats: all_allocators,
            performance_metrics: performance,
            trend_analysis: trends,
            recommendations: self.generate_recommendations(),
        }
    }
    
    /// Generate optimization recommendations based on statistics
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let global = self.get_global_stats();
        
        // High fragmentation
        if global.fragmentation_ratio > 0.3 {
            recommendations.push(
                "High fragmentation detected. Consider increasing GC frequency or using different size classes.".to_string()
            );
        }
        
        // Low memory efficiency
        if global.memory_efficiency < 0.7 {
            recommendations.push(
                "Low memory efficiency. Consider tuning allocator parameters or reviewing object lifetimes.".to_string()
            );
        }
        
        // High allocation rate
        if global.allocation_rate > 10000.0 {
            recommendations.push(
                "High allocation rate detected. Consider object pooling or reducing allocation frequency.".to_string()
            );
        }
        
        recommendations
    }
    
    /// Export statistics to JSON format with proper serialization
    pub fn export_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let report = self.generate_report();
        
        // For now, create a structured JSON representation manually
        // In a full implementation, you'd use serde_json with proper Serialize derives
        let json = format!(
            r#"{{
    "timestamp": "{}",
    "uptime_seconds": {},
    "global_stats": {{
        "total_allocations": {},
        "total_deallocations": {},
        "total_bytes_allocated": {},
        "live_bytes": {},
        "peak_memory": {},
        "allocation_rate": {},
        "memory_efficiency": {},
        "fragmentation_ratio": {}
    }},
    "performance_metrics": {{
        "throughput": {},
        "latency_p50_nanos": {},
        "latency_p95_nanos": {},
        "latency_p99_nanos": {},
        "memory_utilization": {},
        "cache_efficiency": {},
        "gc_overhead": {}
    }},
    "allocator_count": {},
    "recommendations": {:?}
}}"#,
            report.timestamp.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            report.uptime.as_secs(),
            report.global_stats.total_allocations,
            report.global_stats.total_deallocations,
            report.global_stats.total_bytes_allocated,
            report.global_stats.live_bytes,
            report.global_stats.peak_memory,
            report.global_stats.allocation_rate,
            report.global_stats.memory_efficiency,
            report.global_stats.fragmentation_ratio,
            report.performance_metrics.throughput,
            report.performance_metrics.latency_p50.as_nanos(),
            report.performance_metrics.latency_p95.as_nanos(),
            report.performance_metrics.latency_p99.as_nanos(),
            report.performance_metrics.memory_utilization,
            report.performance_metrics.cache_efficiency,
            report.performance_metrics.gc_overhead,
            report.allocator_stats.len(),
            report.recommendations
        );
        
        Ok(json)
    }
    
    /// Clear all historical data
    pub fn clear_history(&self) {
        if let Ok(mut allocator_stats) = self.allocator_stats.lock() {
            for (_, entry) in allocator_stats.iter_mut() {
                entry.history.clear();
            }
        }
    }
}

/// Comprehensive statistics report
#[derive(Debug, Clone)]
pub struct StatisticsReport {
    /// Report timestamp
    pub timestamp: Instant,
    /// System uptime
    pub uptime: Duration,
    /// Global statistics
    pub global_stats: GlobalStats,
    /// Per-allocator statistics
    pub allocator_stats: HashMap<String, AllocatorSpecificStats>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Trend analysis
    pub trend_analysis: HashMap<String, TrendAnalysis>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistics_collector_creation() {
        let collector = StatisticsCollector::new();
        let global_stats = collector.get_global_stats();
        
        assert_eq!(global_stats.total_allocations, 0);
        assert_eq!(global_stats.total_bytes_allocated, 0);
    }
    
    #[test]
    fn test_allocator_registration() {
        let collector = StatisticsCollector::new();
        
        collector.register_allocator("test_allocator".to_string());
        
        let stats = collector.get_allocator_stats("test_allocator");
        assert!(stats.is_some());
        
        let stats = stats.unwrap();
        assert_eq!(stats.name, "test_allocator");
        assert_eq!(stats.basic.allocation_count, 0);
    }
    
    #[test]
    fn test_statistics_update() {
        let collector = StatisticsCollector::new();
        collector.register_allocator("test".to_string());
        
        let mut test_stats = AllocationStats::default();
        test_stats.allocation_count = 100;
        test_stats.total_allocated = 1000;
        test_stats.live_bytes = 800;
        
        collector.update_allocator_stats("test", test_stats);
        
        let global = collector.get_global_stats();
        assert_eq!(global.total_allocations, 100);
        assert_eq!(global.total_bytes_allocated, 1000);
        assert_eq!(global.live_bytes, 800);
    }
    
    #[test]
    fn test_performance_metrics() {
        let collector = StatisticsCollector::new();
        collector.register_allocator("test".to_string());
        
        let mut test_stats = AllocationStats::default();
        test_stats.allocation_count = 1000;
        test_stats.total_allocated = 10000;
        test_stats.live_bytes = 8000;
        test_stats.peak_memory = 12000;
        
        collector.update_allocator_stats("test", test_stats);
        
        let metrics = collector.calculate_performance_metrics();
        assert!(metrics.throughput > 0.0);
        assert!(metrics.memory_utilization > 0.0);
        assert!(metrics.memory_utilization <= 1.0);
    }
    
    #[test]
    fn test_trend_calculation() {
        let collector = StatisticsCollector::new();
        
        // Test simple linear trend
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let trend = collector.calculate_trend(&points);
        
        assert!((trend - 1.0).abs() < 0.01); // Should be close to 1.0 (slope)
    }
    
    #[test]
    fn test_report_generation() {
        let collector = StatisticsCollector::new();
        collector.register_allocator("test".to_string());
        
        let report = collector.generate_report();
        
        assert!(report.uptime.as_nanos() > 0);
        assert_eq!(report.allocator_stats.len(), 1);
        assert!(report.allocator_stats.contains_key("test"));
    }
    
    #[test]
    fn test_recommendations() {
        let collector = StatisticsCollector::new();
        collector.register_allocator("test".to_string());
        
        // Create stats that should trigger recommendations
        let mut bad_stats = AllocationStats::default();
        bad_stats.allocation_count = 1000;
        bad_stats.total_allocated = 10000;
        bad_stats.live_bytes = 3000; // Low efficiency
        
        collector.update_allocator_stats("test", bad_stats);
        
        let report = collector.generate_report();
        assert!(!report.recommendations.is_empty());
    }
    
    #[test]
    fn test_history_management() {
        let config = StatisticsConfig {
            max_history_entries: 3,
            ..Default::default()
        };
        let collector = StatisticsCollector::with_config(config);
        collector.register_allocator("test".to_string());
        
        // Add more entries than the limit
        for i in 0..5 {
            let mut stats = AllocationStats::default();
            stats.allocation_count = i;
            collector.update_allocator_stats("test", stats);
        }
        
        let allocator_stats = collector.get_allocator_stats("test").unwrap();
        assert_eq!(allocator_stats.history.len(), 3); // Should be limited
    }
} 