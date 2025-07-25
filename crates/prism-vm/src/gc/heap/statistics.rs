//! Statistics Collection for Heap Management
//!
//! This module provides comprehensive statistics collection and analysis
//! for heap operations, supporting performance monitoring and optimization.

use super::types::*;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::{VecDeque, HashMap};
use std::time::{Instant, Duration};

/// Statistics collector for heap operations
pub struct StatisticsCollector {
    /// Current statistics snapshot
    current_stats: RwLock<HeapStats>,
    
    /// Historical statistics for trend analysis
    history: Mutex<StatisticsHistory>,
    
    /// Performance counters
    counters: PerformanceCounters,
    
    /// Configuration
    config: StatisticsConfig,
}

/// Configuration for statistics collection
#[derive(Debug, Clone)]
pub struct StatisticsConfig {
    /// Enable detailed statistics collection
    pub detailed_stats: bool,
    
    /// History retention period (seconds)
    pub history_retention: u64,
    
    /// Sampling interval for performance counters
    pub sampling_interval: Duration,
    
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    
    /// Maximum history entries to keep
    pub max_history_entries: usize,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            detailed_stats: true,
            history_retention: 3600, // 1 hour
            sampling_interval: Duration::from_secs(10),
            enable_trend_analysis: true,
            max_history_entries: 1000,
        }
    }
}

/// Performance counters for heap operations
#[derive(Debug, Default)]
pub struct PerformanceCounters {
    /// Object registrations
    object_registrations: AtomicUsize,
    /// Object deallocations
    object_deallocations: AtomicUsize,
    /// Free list hits
    free_list_hits: AtomicUsize,
    /// Free list misses
    free_list_misses: AtomicUsize,
    /// GC completions
    gc_completions: AtomicUsize,
    /// Bytes reclaimed by GC
    bytes_reclaimed: AtomicUsize,
    /// Total allocation time (nanoseconds)
    total_allocation_time: AtomicU64,
    /// Total deallocation time (nanoseconds)
    total_deallocation_time: AtomicU64,
}

/// Historical statistics for trend analysis
#[derive(Debug, Default)]
struct StatisticsHistory {
    /// Snapshots over time
    snapshots: VecDeque<StatisticsSnapshot>,
    /// Computed trends
    trends: StatisticsTrends,
    /// Last analysis timestamp
    last_analysis: Option<Instant>,
}

#[derive(Debug, Clone)]
struct StatisticsSnapshot {
    timestamp: Instant,
    total_allocated: usize,
    live_objects: usize,
    free_space: usize,
    fragmentation_ratio: f64,
    allocation_rate: f64,
    gc_overhead: f64,
}

#[derive(Debug, Default, Clone)]
struct StatisticsTrends {
    /// Allocation rate trend (bytes per second change per hour)
    allocation_rate_trend: f64,
    /// Fragmentation trend (ratio change per hour)
    fragmentation_trend: f64,
    /// GC overhead trend (percentage change per hour)
    gc_overhead_trend: f64,
    /// Memory growth trend (bytes per hour)
    memory_growth_trend: f64,
    /// Prediction confidence (0.0-1.0)
    prediction_confidence: f64,
}

impl StatisticsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        Self::with_config(StatisticsConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: StatisticsConfig) -> Self {
        Self {
            current_stats: RwLock::new(HeapStats {
                total_allocated: 0,
                live_objects: 0,
                free_space: 0,
                fragmentation_ratio: 0.0,
                allocation_rate: 0.0,
                gc_overhead: 0.0,
                region_stats: None,
                card_table_stats: None,
            }),
            history: Mutex::new(StatisticsHistory::default()),
            counters: PerformanceCounters::default(),
            config,
        }
    }
    
    /// Record an object registration
    pub fn record_object_registration(&self, size: usize) {
        self.counters.object_registrations.fetch_add(1, Ordering::Relaxed);
        
        // Update current stats
        if let Ok(mut stats) = self.current_stats.write() {
            stats.total_allocated += size;
            stats.live_objects += 1;
        }
        
        self.maybe_record_snapshot();
    }
    
    /// Record an object deallocation
    pub fn record_object_deallocation(&self, size: usize) {
        self.counters.object_deallocations.fetch_add(1, Ordering::Relaxed);
        
        // Update current stats
        if let Ok(mut stats) = self.current_stats.write() {
            stats.total_allocated = stats.total_allocated.saturating_sub(size);
            stats.live_objects = stats.live_objects.saturating_sub(1);
        }
        
        self.maybe_record_snapshot();
    }
    
    /// Record a free list hit
    pub fn record_free_list_hit(&self) {
        self.counters.free_list_hits.fetch_add(1, Ordering::Relaxed);
        self.update_allocation_rate();
    }
    
    /// Record a free list miss
    pub fn record_free_list_miss(&self) {
        self.counters.free_list_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record GC completion
    pub fn record_gc_completion(&self, objects_collected: usize, bytes_reclaimed: usize) {
        self.counters.gc_completions.fetch_add(1, Ordering::Relaxed);
        self.counters.bytes_reclaimed.fetch_add(bytes_reclaimed, Ordering::Relaxed);
        
        // Update current stats
        if let Ok(mut stats) = self.current_stats.write() {
            stats.live_objects = stats.live_objects.saturating_sub(objects_collected);
            stats.free_space += bytes_reclaimed;
        }
        
        self.update_gc_overhead();
        self.maybe_record_snapshot();
    }
    
    /// Update allocation rate using time-based sliding window
    fn update_allocation_rate(&self) {
        let now = Instant::now();
        let hits = self.counters.free_list_hits.load(Ordering::Relaxed);
        let misses = self.counters.free_list_misses.load(Ordering::Relaxed);
        let total_allocations = hits + misses;
        
        if let Ok(mut history) = self.history.lock() {
            // Add current snapshot if we have allocations
            if total_allocations > 0 {
                let snapshot = StatisticsSnapshot {
                    timestamp: now,
                    total_allocated: 0, // Will be updated by caller
                    live_objects: 0,    // Will be updated by caller
                    free_space: 0,      // Will be updated by caller
                    fragmentation_ratio: 0.0, // Will be updated by caller
                    allocation_rate: 0.0, // Will be calculated below
                    gc_overhead: 0.0,   // Will be updated by caller
                };
                
                // Calculate rate using sliding window
                let window_duration = Duration::from_secs(60); // 1-minute window
                let cutoff_time = now - window_duration;
                
                // Filter snapshots within the window
                let recent_snapshots: Vec<&StatisticsSnapshot> = history.snapshots
                    .iter()
                    .filter(|s| s.timestamp >= cutoff_time)
                    .collect();
                
                let rate = if recent_snapshots.len() >= 2 {
                    // Calculate rate based on allocation changes over time
                    let oldest = recent_snapshots.first().unwrap();
                    let newest = recent_snapshots.last().unwrap();
                    
                    let time_diff = newest.timestamp.duration_since(oldest.timestamp).as_secs_f64();
                    if time_diff > 0.0 {
                        let allocation_diff = total_allocations.saturating_sub(
                            (oldest.allocation_rate * time_diff) as usize
                        );
                        allocation_diff as f64 / time_diff
                    } else {
                        0.0
                    }
                } else {
                    // Fallback for insufficient data
                    total_allocations as f64 / window_duration.as_secs_f64()
                };
                
                // Update current stats
                if let Ok(mut stats) = self.current_stats.write() {
                    stats.allocation_rate = rate;
                }
                
                // Add to history
                let mut updated_snapshot = snapshot;
                updated_snapshot.allocation_rate = rate;
                history.snapshots.push_back(updated_snapshot);
                
                // Maintain history size
                if history.snapshots.len() > self.config.max_history_entries {
                    history.snapshots.pop_front();
                }
            }
        }
    }
    
    /// Update GC overhead using actual timing data
    fn update_gc_overhead(&self) {
        let now = Instant::now();
        
        if let Ok(mut history) = self.history.lock() {
            // Calculate overhead based on recent GC events
            let window_duration = Duration::from_secs(300); // 5-minute window
            let cutoff_time = now - window_duration;
            
            // Filter recent snapshots
            let recent_snapshots: Vec<&StatisticsSnapshot> = history.snapshots
                .iter()
                .filter(|s| s.timestamp >= cutoff_time)
                .collect();
            
            let overhead = if recent_snapshots.len() >= 2 {
                // Calculate based on GC frequency and impact
                let gc_count = self.counters.gc_completions.load(Ordering::Relaxed);
                let bytes_reclaimed = self.counters.bytes_reclaimed.load(Ordering::Relaxed);
                
                // Estimate GC time based on bytes processed
                let estimated_gc_time = if bytes_reclaimed > 0 {
                    // Rough estimate: 1ms per MB processed
                    (bytes_reclaimed as f64 / (1024.0 * 1024.0)) * 0.001
                } else {
                    0.0
                };
                
                // Calculate overhead as percentage of total time
                let total_time = window_duration.as_secs_f64();
                let total_gc_time = estimated_gc_time * gc_count as f64;
                
                if total_time > 0.0 {
                    ((total_gc_time / total_time) * 100.0).min(95.0) // Cap at 95%
                } else {
                    0.0
                }
            } else {
                // Fallback calculation for insufficient data
                let gc_count = self.counters.gc_completions.load(Ordering::Relaxed);
                if gc_count > 0 {
                    // Conservative estimate based on GC frequency
                    let time_since_start = now.duration_since(
                        history.snapshots.front()
                            .map(|s| s.timestamp)
                            .unwrap_or(now)
                    ).as_secs_f64();
                    
                    if time_since_start > 0.0 {
                        let gc_frequency = gc_count as f64 / time_since_start;
                        // Assume each GC takes 10ms on average
                        (gc_frequency * 0.01 * 100.0).min(50.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            };
            
            // Update current stats
            if let Ok(mut stats) = self.current_stats.write() {
                stats.gc_overhead = overhead;
            }
        }
    }
    
    /// Maybe record a statistics snapshot
    fn maybe_record_snapshot(&self) {
        if let Ok(mut history) = self.history.try_lock() {
            let now = Instant::now();
            
            // Check if it's time for a new snapshot
            let should_snapshot = if let Some(last_snapshot) = history.snapshots.back() {
                now.duration_since(last_snapshot.timestamp) >= self.config.sampling_interval
            } else {
                true
            };
            
            if should_snapshot {
                if let Ok(stats) = self.current_stats.read() {
                    let snapshot = StatisticsSnapshot {
                        timestamp: now,
                        total_allocated: stats.total_allocated,
                        live_objects: stats.live_objects,
                        free_space: stats.free_space,
                        fragmentation_ratio: stats.fragmentation_ratio,
                        allocation_rate: stats.allocation_rate,
                        gc_overhead: stats.gc_overhead,
                    };
                    
                    history.snapshots.push_back(snapshot);
                    
                    // Keep history bounded
                    if history.snapshots.len() > self.config.max_history_entries {
                        history.snapshots.pop_front();
                    }
                    
                    // Perform trend analysis if enabled
                    if self.config.enable_trend_analysis {
                        self.analyze_trends(&mut history);
                    }
                }
            }
        }
    }
    
    /// Analyze statistical trends
    fn analyze_trends(&self, history: &mut StatisticsHistory) {
        if history.snapshots.len() < 2 {
            return;
        }
        
        let snapshots: Vec<_> = history.snapshots.iter().collect();
        let time_span = snapshots.last().unwrap().timestamp
            .duration_since(snapshots.first().unwrap().timestamp)
            .as_secs_f64() / 3600.0; // Convert to hours
        
        if time_span <= 0.0 {
            return;
        }
        
        // Calculate allocation rate trend
        let initial_rate = snapshots.first().unwrap().allocation_rate;
        let final_rate = snapshots.last().unwrap().allocation_rate;
        let allocation_rate_trend = (final_rate - initial_rate) / time_span;
        
        // Calculate fragmentation trend
        let initial_frag = snapshots.first().unwrap().fragmentation_ratio;
        let final_frag = snapshots.last().unwrap().fragmentation_ratio;
        let fragmentation_trend = (final_frag - initial_frag) / time_span;
        
        // Calculate GC overhead trend
        let initial_gc = snapshots.first().unwrap().gc_overhead;
        let final_gc = snapshots.last().unwrap().gc_overhead;
        let gc_overhead_trend = (final_gc - initial_gc) / time_span;
        
        // Calculate memory growth trend
        let initial_memory = snapshots.first().unwrap().total_allocated;
        let final_memory = snapshots.last().unwrap().total_allocated;
        let memory_growth_trend = (final_memory as f64 - initial_memory as f64) / time_span;
        
        // Calculate prediction confidence based on trend consistency
        let prediction_confidence = self.calculate_prediction_confidence(&snapshots);
        
        history.trends = StatisticsTrends {
            allocation_rate_trend,
            fragmentation_trend,
            gc_overhead_trend,
            memory_growth_trend,
            prediction_confidence,
        };
    }
    
    /// Calculate prediction confidence based on trend consistency
    fn calculate_prediction_confidence(&self, snapshots: &[&StatisticsSnapshot]) -> f64 {
        if snapshots.len() < 3 {
            return 0.0;
        }
        
        // Analyze consistency of trends across different metrics
        let mut consistency_scores = Vec::new();
        
        // Allocation rate consistency
        let rate_consistency = self.calculate_metric_consistency(
            snapshots,
            |s| s.allocation_rate,
        );
        consistency_scores.push(rate_consistency);
        
        // Fragmentation consistency
        let frag_consistency = self.calculate_metric_consistency(
            snapshots,
            |s| s.fragmentation_ratio,
        );
        consistency_scores.push(frag_consistency);
        
        // GC overhead consistency
        let gc_consistency = self.calculate_metric_consistency(
            snapshots,
            |s| s.gc_overhead,
        );
        consistency_scores.push(gc_consistency);
        
        // Average consistency scores
        let average_consistency = consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64;
        average_consistency.clamp(0.0, 1.0)
    }
    
    /// Calculate consistency of a specific metric
    fn calculate_metric_consistency<F>(&self, snapshots: &[&StatisticsSnapshot], metric_fn: F) -> f64
    where
        F: Fn(&StatisticsSnapshot) -> f64,
    {
        if snapshots.len() < 3 {
            return 0.0;
        }
        
        // Calculate trend direction changes
        let mut direction_changes = 0;
        let mut total_comparisons = 0;
        
        for window in snapshots.windows(3) {
            let val1 = metric_fn(window[0]);
            let val2 = metric_fn(window[1]);
            let val3 = metric_fn(window[2]);
            
            let trend1 = val2 - val1;
            let trend2 = val3 - val2;
            
            if (trend1 > 0.0) != (trend2 > 0.0) && trend1.abs() > 0.001 && trend2.abs() > 0.001 {
                direction_changes += 1;
            }
            total_comparisons += 1;
        }
        
        if total_comparisons == 0 {
            return 0.0;
        }
        
        // Consistency is inverse of direction change ratio
        1.0 - (direction_changes as f64 / total_comparisons as f64)
    }
    
    /// Get current statistics
    pub fn get_current_stats(&self) -> HeapStats {
        self.current_stats.read().unwrap().clone()
    }
    
    /// Get performance counters
    pub fn get_performance_counters(&self) -> PerformanceCountersSnapshot {
        PerformanceCountersSnapshot {
            object_registrations: self.counters.object_registrations.load(Ordering::Relaxed),
            object_deallocations: self.counters.object_deallocations.load(Ordering::Relaxed),
            free_list_hits: self.counters.free_list_hits.load(Ordering::Relaxed),
            free_list_misses: self.counters.free_list_misses.load(Ordering::Relaxed),
            gc_completions: self.counters.gc_completions.load(Ordering::Relaxed),
            bytes_reclaimed: self.counters.bytes_reclaimed.load(Ordering::Relaxed),
            total_allocation_time: Duration::from_nanos(
                self.counters.total_allocation_time.load(Ordering::Relaxed)
            ),
            total_deallocation_time: Duration::from_nanos(
                self.counters.total_deallocation_time.load(Ordering::Relaxed)
            ),
        }
    }
    
    /// Get statistical trends
    pub fn get_trends(&self) -> Option<StatisticsTrends> {
        self.history.lock().unwrap().trends.clone().into()
    }
    
    /// Get historical snapshots
    pub fn get_history(&self) -> Vec<StatisticsSnapshot> {
        self.history.lock().unwrap().snapshots.iter().cloned().collect()
    }
    
    /// Update heap statistics
    pub fn update_heap_stats(&self, stats: HeapStats) {
        *self.current_stats.write().unwrap() = stats;
        self.maybe_record_snapshot();
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        // Reset counters
        self.counters.object_registrations.store(0, Ordering::Relaxed);
        self.counters.object_deallocations.store(0, Ordering::Relaxed);
        self.counters.free_list_hits.store(0, Ordering::Relaxed);
        self.counters.free_list_misses.store(0, Ordering::Relaxed);
        self.counters.gc_completions.store(0, Ordering::Relaxed);
        self.counters.bytes_reclaimed.store(0, Ordering::Relaxed);
        self.counters.total_allocation_time.store(0, Ordering::Relaxed);
        self.counters.total_deallocation_time.store(0, Ordering::Relaxed);
        
        // Reset current stats
        *self.current_stats.write().unwrap() = HeapStats {
            total_allocated: 0,
            live_objects: 0,
            free_space: 0,
            fragmentation_ratio: 0.0,
            allocation_rate: 0.0,
            gc_overhead: 0.0,
            region_stats: None,
            card_table_stats: None,
        };
        
        // Clear history
        let mut history = self.history.lock().unwrap();
        history.snapshots.clear();
        history.trends = StatisticsTrends::default();
        history.last_analysis = None;
    }
    
    /// Export statistics for external analysis
    pub fn export_statistics(&self) -> StatisticsExport {
        let current_stats = self.get_current_stats();
        let counters = self.get_performance_counters();
        let trends = self.get_trends();
        let history = self.get_history();
        
        StatisticsExport {
            timestamp: Instant::now(),
            current_stats,
            performance_counters: counters,
            trends,
            history,
            config: self.config.clone(),
        }
    }
}

/// Snapshot of performance counters
#[derive(Debug, Clone)]
pub struct PerformanceCountersSnapshot {
    pub object_registrations: usize,
    pub object_deallocations: usize,
    pub free_list_hits: usize,
    pub free_list_misses: usize,
    pub gc_completions: usize,
    pub bytes_reclaimed: usize,
    pub total_allocation_time: Duration,
    pub total_deallocation_time: Duration,
}

/// Complete statistics export for external analysis
#[derive(Debug, Clone)]
pub struct StatisticsExport {
    pub timestamp: Instant,
    pub current_stats: HeapStats,
    pub performance_counters: PerformanceCountersSnapshot,
    pub trends: Option<StatisticsTrends>,
    pub history: Vec<StatisticsSnapshot>,
    pub config: StatisticsConfig,
}

// Implement necessary traits for StatisticsTrends
impl From<StatisticsTrends> for Option<StatisticsTrends> {
    fn from(trends: StatisticsTrends) -> Self {
        Some(trends)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_collector_creation() {
        let collector = StatisticsCollector::new();
        let stats = collector.get_current_stats();
        
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.live_objects, 0);
    }

    #[test]
    fn test_object_registration_tracking() {
        let collector = StatisticsCollector::new();
        
        collector.record_object_registration(1024);
        collector.record_object_registration(2048);
        
        let stats = collector.get_current_stats();
        assert_eq!(stats.total_allocated, 3072);
        assert_eq!(stats.live_objects, 2);
        
        let counters = collector.get_performance_counters();
        assert_eq!(counters.object_registrations, 2);
    }

    #[test]
    fn test_object_deallocation_tracking() {
        let collector = StatisticsCollector::new();
        
        collector.record_object_registration(2048);
        collector.record_object_deallocation(1024);
        
        let stats = collector.get_current_stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.live_objects, 0);
        
        let counters = collector.get_performance_counters();
        assert_eq!(counters.object_deallocations, 1);
    }

    #[test]
    fn test_free_list_tracking() {
        let collector = StatisticsCollector::new();
        
        collector.record_free_list_hit();
        collector.record_free_list_hit();
        collector.record_free_list_miss();
        
        let counters = collector.get_performance_counters();
        assert_eq!(counters.free_list_hits, 2);
        assert_eq!(counters.free_list_misses, 1);
    }

    #[test]
    fn test_gc_completion_tracking() {
        let collector = StatisticsCollector::new();
        
        // Set up some objects first
        collector.record_object_registration(1024);
        collector.record_object_registration(2048);
        
        // Record GC completion
        collector.record_gc_completion(1, 1024);
        
        let stats = collector.get_current_stats();
        assert_eq!(stats.live_objects, 1); // One object collected
        assert_eq!(stats.free_space, 1024); // Bytes reclaimed
        
        let counters = collector.get_performance_counters();
        assert_eq!(counters.gc_completions, 1);
        assert_eq!(counters.bytes_reclaimed, 1024);
    }

    #[test]
    fn test_statistics_reset() {
        let collector = StatisticsCollector::new();
        
        // Generate some statistics
        collector.record_object_registration(1024);
        collector.record_free_list_hit();
        collector.record_gc_completion(1, 512);
        
        // Verify statistics exist
        let stats_before = collector.get_current_stats();
        assert!(stats_before.total_allocated > 0);
        
        // Reset statistics
        collector.reset();
        
        // Verify statistics are cleared
        let stats_after = collector.get_current_stats();
        assert_eq!(stats_after.total_allocated, 0);
        assert_eq!(stats_after.live_objects, 0);
        
        let counters = collector.get_performance_counters();
        assert_eq!(counters.object_registrations, 0);
        assert_eq!(counters.free_list_hits, 0);
    }

    #[test]
    fn test_statistics_export() {
        let collector = StatisticsCollector::new();
        
        // Generate some statistics
        collector.record_object_registration(1024);
        collector.record_free_list_hit();
        
        // Export statistics
        let export = collector.export_statistics();
        
        assert_eq!(export.current_stats.total_allocated, 1024);
        assert_eq!(export.performance_counters.object_registrations, 1);
        assert_eq!(export.performance_counters.free_list_hits, 1);
    }
} 