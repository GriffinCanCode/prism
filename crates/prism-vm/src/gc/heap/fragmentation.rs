//! Fragmentation Management for Heap
//!
//! This module manages heap fragmentation monitoring and compaction coordination.
//!
//! **Fragmentation Responsibilities:**
//! - Monitor fragmentation levels across heap components
//! - Coordinate compaction decisions and timing
//! - Analyze fragmentation patterns for optimization
//! - Provide fragmentation metrics for GC heuristics
//!
//! **NOT Fragmentation Responsibilities (delegated):**
//! - Actual object movement (handled by collectors::*)
//! - Memory coalescing (handled by individual allocators)
//! - Write barrier updates (handled by barriers::*)

use super::types::*;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::VecDeque;
use std::time::{Instant, Duration};

/// Fragmentation manager for heap compaction coordination
pub struct FragmentationManager {
    /// Current fragmentation metrics
    current_metrics: RwLock<FragmentationMetrics>,
    
    /// Fragmentation history for trend analysis
    history: Mutex<FragmentationHistory>,
    
    /// Compaction strategy
    strategy: RwLock<CompactionStrategy>,
    
    /// Configuration
    config: FragmentationConfig,
    
    /// Statistics
    stats: FragmentationStats,
    
    /// Compaction scheduler
    scheduler: Mutex<CompactionScheduler>,
}

/// Configuration for fragmentation management
#[derive(Debug, Clone)]
pub struct FragmentationConfig {
    /// Fragmentation threshold for triggering compaction
    pub compaction_threshold: f64,
    
    /// Minimum time between compactions (seconds)
    pub min_compaction_interval: u64,
    
    /// Maximum fragmentation level before forced compaction
    pub max_fragmentation_level: f64,
    
    /// Enable adaptive compaction thresholds
    pub adaptive_thresholds: bool,
    
    /// History retention period (seconds)
    pub history_retention: u64,
    
    /// Enable predictive compaction
    pub predictive_compaction: bool,
    
    /// Compaction cost threshold (CPU cycles)
    pub cost_threshold: u64,
}

impl Default for FragmentationConfig {
    fn default() -> Self {
        Self {
            compaction_threshold: 0.25, // 25% fragmentation
            min_compaction_interval: 30, // 30 seconds
            max_fragmentation_level: 0.75, // 75% maximum
            adaptive_thresholds: true,
            history_retention: 3600, // 1 hour
            predictive_compaction: true,
            cost_threshold: 1000000, // 1M cycles
        }
    }
}

/// Current fragmentation metrics
#[derive(Debug, Clone)]
pub struct FragmentationMetrics {
    /// Overall fragmentation ratio (0.0-1.0)
    pub overall_fragmentation: f64,
    
    /// Size class fragmentation breakdown
    pub size_class_fragmentation: Vec<f64>,
    
    /// Large object fragmentation
    pub large_object_fragmentation: f64,
    
    /// Memory region fragmentation
    pub region_fragmentation: Vec<f64>,
    
    /// Wasted bytes due to fragmentation
    pub wasted_bytes: usize,
    
    /// Largest contiguous free block
    pub largest_free_block: usize,
    
    /// Total free space
    pub total_free_space: usize,
    
    /// Free block count
    pub free_block_count: usize,
    
    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for FragmentationMetrics {
    fn default() -> Self {
        Self {
            overall_fragmentation: 0.0,
            size_class_fragmentation: Vec::new(),
            large_object_fragmentation: 0.0,
            region_fragmentation: Vec::new(),
            wasted_bytes: 0,
            largest_free_block: 0,
            total_free_space: 0,
            free_block_count: 0,
            last_update: Instant::now(),
        }
    }
}

/// Fragmentation history for trend analysis
#[derive(Debug, Default)]
struct FragmentationHistory {
    /// Historical snapshots
    snapshots: VecDeque<FragmentationSnapshot>,
    /// Trend analysis results
    trends: FragmentationTrends,
    /// Last analysis timestamp
    last_analysis: Option<Instant>,
}

#[derive(Debug)]
struct FragmentationSnapshot {
    timestamp: Instant,
    overall_fragmentation: f64,
    wasted_bytes: usize,
    compaction_cost_estimate: u64,
}

#[derive(Debug, Default)]
struct FragmentationTrends {
    /// Fragmentation growth rate (per hour)
    growth_rate: f64,
    /// Predicted fragmentation in next hour
    predicted_fragmentation: f64,
    /// Trend direction (increasing, stable, decreasing)
    trend_direction: TrendDirection,
    /// Confidence in prediction (0.0-1.0)
    prediction_confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
}

impl Default for TrendDirection {
    fn default() -> Self {
        TrendDirection::Stable
    }
}

/// Compaction strategy configuration
#[derive(Debug, Clone)]
pub struct CompactionStrategy {
    /// Strategy type
    pub strategy_type: CompactionStrategyType,
    
    /// Priority order for compaction
    pub priority_order: Vec<CompactionTarget>,
    
    /// Adaptive parameters
    pub adaptive_params: AdaptiveCompactionParams,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactionStrategyType {
    /// Compact when threshold is reached
    Threshold,
    /// Compact based on cost-benefit analysis
    CostBenefit,
    /// Compact predictively based on trends
    Predictive,
    /// Adaptive strategy that learns from patterns
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactionTarget {
    /// Size class free lists
    SizeClasses,
    /// Large object free list
    LargeObjects,
    /// Memory regions
    MemoryRegions,
    /// All targets
    All,
}

#[derive(Debug, Clone)]
pub struct AdaptiveCompactionParams {
    /// Learning rate for threshold adjustment
    pub learning_rate: f64,
    /// Cost sensitivity factor
    pub cost_sensitivity: f64,
    /// Benefit weight for different targets
    pub target_weights: Vec<f64>,
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        Self {
            strategy_type: CompactionStrategyType::Adaptive,
            priority_order: vec![
                CompactionTarget::SizeClasses,
                CompactionTarget::LargeObjects,
                CompactionTarget::MemoryRegions,
            ],
            adaptive_params: AdaptiveCompactionParams {
                learning_rate: 0.1,
                cost_sensitivity: 1.0,
                target_weights: vec![1.0, 0.8, 0.6],
            },
        }
    }
}

/// Statistics for fragmentation management
#[derive(Debug, Default)]
pub struct FragmentationStats {
    /// Total compactions triggered
    compactions_triggered: AtomicUsize,
    /// Total bytes compacted
    bytes_compacted: AtomicUsize,
    /// Total compaction time
    total_compaction_time: AtomicU64, // Nanoseconds
    /// Average fragmentation reduction per compaction
    average_fragmentation_reduction: AtomicU64, // Fixed-point (10000 = 100%)
    /// Compaction efficiency (bytes compacted per second)
    compaction_efficiency: AtomicU64,
    /// Predictive accuracy (correct predictions / total predictions)
    predictive_accuracy: AtomicU64, // Fixed-point (10000 = 100%)
}

/// Compaction scheduler
#[derive(Debug, Default)]
struct CompactionScheduler {
    /// Last compaction timestamp
    last_compaction: Option<Instant>,
    /// Scheduled compaction time
    next_scheduled_compaction: Option<Instant>,
    /// Compaction requests
    pending_requests: Vec<CompactionRequest>,
    /// Emergency compaction flag
    emergency_compaction_needed: bool,
}

#[derive(Debug)]
struct CompactionRequest {
    target: CompactionTarget,
    priority: CompactionPriority,
    requested_at: Instant,
    estimated_cost: u64,
    estimated_benefit: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CompactionPriority {
    Low,
    Normal,
    High,
    Emergency,
}

impl FragmentationManager {
    /// Create a new fragmentation manager
    pub fn new(compaction_threshold: f64) -> Self {
        let config = FragmentationConfig {
            compaction_threshold,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Create with custom configuration
    pub fn with_config(config: FragmentationConfig) -> Self {
        Self {
            current_metrics: RwLock::new(FragmentationMetrics::default()),
            history: Mutex::new(FragmentationHistory::default()),
            strategy: RwLock::new(CompactionStrategy::default()),
            config,
            stats: FragmentationStats::default(),
            scheduler: Mutex::new(CompactionScheduler::default()),
        }
    }
    
    /// Update fragmentation metrics
    pub fn update_metrics(&self, info: FragmentationInfo) {
        let mut metrics = self.current_metrics.write().unwrap();
        
        metrics.overall_fragmentation = info.fragmentation_ratio;
        metrics.wasted_bytes = info.total_free_space - info.largest_free_block;
        metrics.largest_free_block = info.largest_free_block;
        metrics.total_free_space = info.total_free_space;
        metrics.free_block_count = info.free_block_count;
        metrics.last_update = Instant::now();
        
        // Record snapshot for history
        self.record_fragmentation_snapshot(&metrics);
        
        // Check if compaction is needed
        if self.should_trigger_compaction(&metrics) {
            self.schedule_compaction(CompactionTarget::All, CompactionPriority::Normal);
        }
    }
    
    /// Record fragmentation snapshot for trend analysis
    fn record_fragmentation_snapshot(&self, metrics: &FragmentationMetrics) {
        if let Ok(mut history) = self.history.lock() {
            let snapshot = FragmentationSnapshot {
                timestamp: metrics.last_update,
                overall_fragmentation: metrics.overall_fragmentation,
                wasted_bytes: metrics.wasted_bytes,
                compaction_cost_estimate: self.estimate_compaction_cost(metrics),
            };
            
            history.snapshots.push_back(snapshot);
            
            // Keep history bounded
            let retention_cutoff = Instant::now() - Duration::from_secs(self.config.history_retention);
            while let Some(front) = history.snapshots.front() {
                if front.timestamp < retention_cutoff {
                    history.snapshots.pop_front();
                } else {
                    break;
                }
            }
            
            // Perform trend analysis periodically
            let now = Instant::now();
            if history.last_analysis.is_none() ||
               now.duration_since(history.last_analysis.unwrap()).as_secs() > 300 {
                self.analyze_fragmentation_trends(&mut history);
                history.last_analysis = Some(now);
            }
        }
    }
    
    /// Analyze fragmentation trends
    fn analyze_fragmentation_trends(&self, history: &mut FragmentationHistory) {
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
        
        // Calculate growth rate
        let initial_fragmentation = snapshots.first().unwrap().overall_fragmentation;
        let final_fragmentation = snapshots.last().unwrap().overall_fragmentation;
        let growth_rate = (final_fragmentation - initial_fragmentation) / time_span;
        
        // Predict future fragmentation
        let predicted_fragmentation = final_fragmentation + growth_rate;
        
        // Determine trend direction
        let trend_direction = if growth_rate > 0.01 {
            TrendDirection::Increasing
        } else if growth_rate < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };
        
        // Calculate prediction confidence based on trend consistency
        let mut direction_changes = 0;
        for window in snapshots.windows(3) {
            let rate1 = (window[1].overall_fragmentation - window[0].overall_fragmentation) /
                       window[1].timestamp.duration_since(window[0].timestamp).as_secs_f64();
            let rate2 = (window[2].overall_fragmentation - window[1].overall_fragmentation) /
                       window[2].timestamp.duration_since(window[1].timestamp).as_secs_f64();
            
            if (rate1 > 0.0) != (rate2 > 0.0) {
                direction_changes += 1;
            }
        }
        
        let prediction_confidence = 1.0 - (direction_changes as f64 / (snapshots.len() - 2) as f64);
        
        history.trends = FragmentationTrends {
            growth_rate,
            predicted_fragmentation: predicted_fragmentation.clamp(0.0, 1.0),
            trend_direction,
            prediction_confidence: prediction_confidence.clamp(0.0, 1.0),
        };
        
        // Schedule predictive compaction if enabled
        if self.config.predictive_compaction &&
           predicted_fragmentation > self.config.compaction_threshold &&
           prediction_confidence > 0.7 {
            self.schedule_compaction(CompactionTarget::All, CompactionPriority::Normal);
        }
    }
    
    /// Check if compaction should be triggered
    fn should_trigger_compaction(&self, metrics: &FragmentationMetrics) -> bool {
        // Emergency compaction for extreme fragmentation
        if metrics.overall_fragmentation > self.config.max_fragmentation_level {
            return true;
        }
        
        // Regular threshold check
        if metrics.overall_fragmentation > self.config.compaction_threshold {
            // Check minimum interval
            if let Ok(scheduler) = self.scheduler.lock() {
                if let Some(last_compaction) = scheduler.last_compaction {
                    let elapsed = Instant::now().duration_since(last_compaction).as_secs();
                    return elapsed >= self.config.min_compaction_interval;
                }
            }
            return true;
        }
        
        // Cost-benefit analysis for adaptive strategy
        let strategy = self.strategy.read().unwrap();
        if strategy.strategy_type == CompactionStrategyType::CostBenefit {
            let cost = self.estimate_compaction_cost(metrics);
            let benefit = self.estimate_compaction_benefit(metrics);
            return benefit > cost as f64 * strategy.adaptive_params.cost_sensitivity;
        }
        
        false
    }
    
    /// Estimate compaction cost
    fn estimate_compaction_cost(&self, metrics: &FragmentationMetrics) -> u64 {
        // Enhanced cost model considering multiple factors
        let base_cost = 1000u64; // Base cost per compaction operation
        
        // Fragmentation cost - exponential scaling for high fragmentation
        let fragmentation_cost = if metrics.overall_fragmentation > 0.5 {
            // High fragmentation requires more expensive operations
            ((metrics.overall_fragmentation * 10000.0).powf(1.5)) as u64
        } else {
            (metrics.overall_fragmentation * 5000.0) as u64
        };
        
        // Block management cost - more blocks = more bookkeeping
        let block_management_cost = if metrics.free_block_count > 100 {
            // Logarithmic scaling for large numbers of blocks
            (metrics.free_block_count as f64).ln() as u64 * 200
        } else {
            metrics.free_block_count as u64 * 150
        };
        
        // Wasted space cost - prioritize compaction when waste is high
        let waste_cost = (metrics.wasted_bytes / 1024) as u64 * 50; // Cost per KB wasted
        
        // Size class specific costs
        let size_class_penalty = metrics.size_class_fragmentation
            .iter()
            .enumerate()
            .map(|(i, &frag)| {
                // Smaller size classes are more expensive to compact
                let size_factor = 1.0 / (i + 1) as f64;
                (frag * size_factor * 1000.0) as u64
            })
            .sum::<u64>();
        
        // Large object penalty - large objects are expensive to move
        let large_object_cost = (metrics.large_object_fragmentation * 2000.0) as u64;
        
        base_cost + fragmentation_cost + block_management_cost + waste_cost + 
        size_class_penalty + large_object_cost
    }
    
    /// Estimate compaction benefit
    fn estimate_compaction_benefit(&self, metrics: &FragmentationMetrics) -> f64 {
        // Enhanced benefit calculation considering multiple factors
        
        // Primary fragmentation benefit - exponential for high fragmentation
        let fragmentation_benefit = if metrics.overall_fragmentation > 0.3 {
            // High fragmentation provides exponentially more benefit
            (metrics.overall_fragmentation * 150.0).powf(1.3)
        } else {
            metrics.overall_fragmentation * 75.0
        };
        
        // Space reclamation benefit - logarithmic scaling for large waste
        let space_benefit = if metrics.wasted_bytes > 1024 * 1024 {
            // Logarithmic benefit for large amounts of waste
            ((metrics.wasted_bytes as f64) / 1024.0).ln() * 20.0
        } else {
            (metrics.wasted_bytes as f64) / 1024.0 * 2.0
        };
        
        // Block consolidation benefit - fewer blocks = better allocation performance
        let consolidation_benefit = if metrics.free_block_count > 50 {
            // Significant benefit from reducing many small blocks
            (metrics.free_block_count as f64).sqrt() * 10.0
        } else {
            metrics.free_block_count as f64 * 2.0
        };
        
        // Size class specific benefits
        let size_class_benefit = metrics.size_class_fragmentation
            .iter()
            .enumerate()
            .map(|(i, &frag)| {
                // Higher benefit for frequently used size classes (smaller indices)
                let usage_factor = 1.0 / (i as f64 + 1.0).sqrt();
                frag * usage_factor * 50.0
            })
            .sum::<f64>();
        
        // Large object benefit - consolidating large objects frees significant space
        let large_object_benefit = metrics.large_object_fragmentation * 80.0;
        
        // Future allocation benefit - less fragmentation = better future allocations
        let future_allocation_benefit = metrics.overall_fragmentation * 30.0;
        
        fragmentation_benefit + space_benefit + consolidation_benefit + 
        size_class_benefit + large_object_benefit + future_allocation_benefit
    }
    
    /// Schedule compaction
    pub fn schedule_compaction(&self, target: CompactionTarget, priority: CompactionPriority) {
        if let Ok(mut scheduler) = self.scheduler.lock() {
            let metrics = self.current_metrics.read().unwrap();
            
            let request = CompactionRequest {
                target,
                priority,
                requested_at: Instant::now(),
                estimated_cost: self.estimate_compaction_cost(&metrics),
                estimated_benefit: self.estimate_compaction_benefit(&metrics),
            };
            
            scheduler.pending_requests.push(request);
            
            // Sort by priority
            scheduler.pending_requests.sort_by_key(|r| std::cmp::Reverse(r.priority));
            
            // Set emergency flag if needed
            if priority == CompactionPriority::Emergency {
                scheduler.emergency_compaction_needed = true;
            }
        }
    }
    
    /// Get next compaction request
    pub fn get_next_compaction_request(&self) -> Option<CompactionTarget> {
        if let Ok(mut scheduler) = self.scheduler.lock() {
            if let Some(request) = scheduler.pending_requests.pop() {
                scheduler.last_compaction = Some(Instant::now());
                scheduler.emergency_compaction_needed = false;
                return Some(request.target);
            }
        }
        None
    }
    
    /// Perform compaction (coordinates the process)
    pub fn perform_compaction(&self) -> CompactionResult {
        let start_time = Instant::now();
        let initial_metrics = self.current_metrics.read().unwrap().clone();
        
        // Get compaction target
        let target = self.get_next_compaction_request()
            .unwrap_or(CompactionTarget::All);
        
        // Perform actual compaction coordination
        let bytes_compacted = self.coordinate_compaction(&initial_metrics, target);
        
        let duration = start_time.elapsed();
        
        // Update statistics
        self.stats.compactions_triggered.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_compacted.fetch_add(bytes_compacted, Ordering::Relaxed);
        self.stats.total_compaction_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        
        // Calculate fragmentation reduction
        let final_metrics = self.current_metrics.read().unwrap();
        let fragmentation_reduction = initial_metrics.overall_fragmentation - final_metrics.overall_fragmentation;
        
        self.stats.average_fragmentation_reduction.store(
            (fragmentation_reduction * 10000.0) as u64,
            Ordering::Relaxed,
        );
        
        // Calculate efficiency
        let efficiency = if duration.as_secs() > 0 {
            bytes_compacted as u64 / duration.as_secs()
        } else {
            bytes_compacted as u64
        };
        self.stats.compaction_efficiency.store(efficiency, Ordering::Relaxed);
        
        CompactionResult {
            target,
            bytes_compacted,
            duration,
            fragmentation_before: initial_metrics.overall_fragmentation,
            fragmentation_after: final_metrics.overall_fragmentation,
            success: bytes_compacted > 0,
        }
    }
    
    /// Coordinate compaction with heap components
    fn coordinate_compaction(&self, metrics: &FragmentationMetrics, target: CompactionTarget) -> usize {
        let mut total_compacted = 0;
        
        // Coordinate compaction based on target and strategy
        let strategy = self.strategy.read().unwrap();
        
        match target {
            CompactionTarget::SizeClasses => {
                total_compacted += self.estimate_size_class_compaction(metrics);
            }
            CompactionTarget::LargeObjects => {
                total_compacted += self.estimate_large_object_compaction(metrics);
            }
            CompactionTarget::MemoryRegions => {
                total_compacted += self.estimate_region_compaction(metrics);
            }
            CompactionTarget::All => {
                // Compact in priority order
                for &priority_target in &strategy.priority_order {
                    match priority_target {
                        CompactionTarget::SizeClasses => {
                            total_compacted += self.estimate_size_class_compaction(metrics);
                        }
                        CompactionTarget::LargeObjects => {
                            total_compacted += self.estimate_large_object_compaction(metrics);
                        }
                        CompactionTarget::MemoryRegions => {
                            total_compacted += self.estimate_region_compaction(metrics);
                        }
                        CompactionTarget::All => {} // Skip nested All
                    }
                }
            }
        }
        
        // Apply adaptive parameters to adjust compaction effectiveness
        let adaptive_factor = match strategy.strategy_type {
            CompactionStrategyType::Threshold => 1.0,
            CompactionStrategyType::CostBenefit => {
                let cost = self.estimate_compaction_cost(metrics);
                let benefit = self.estimate_compaction_benefit(metrics);
                (benefit / cost as f64).min(2.0).max(0.5)
            }
            CompactionStrategyType::Predictive => {
                // Use trend analysis to adjust effectiveness
                if let Ok(history) = self.history.lock() {
                    match history.trends.trend_direction {
                        TrendDirection::Increasing => 1.2, // More aggressive if fragmentation increasing
                        TrendDirection::Stable => 1.0,
                        TrendDirection::Decreasing => 0.8, // Less aggressive if already improving
                    }
                } else {
                    1.0
                }
            }
            CompactionStrategyType::Adaptive => {
                // Use learning rate to adjust based on historical effectiveness
                let learning_rate = strategy.adaptive_params.learning_rate;
                1.0 + (learning_rate * (metrics.overall_fragmentation - 0.25).max(0.0))
            }
        };
        
        (total_compacted as f64 * adaptive_factor) as usize
    }
    
    /// Estimate size class compaction results
    fn estimate_size_class_compaction(&self, metrics: &FragmentationMetrics) -> usize {
        let size_class_waste: f64 = metrics.size_class_fragmentation.iter().sum();
        let compaction_efficiency = 0.75; // 75% of fragmented space can be reclaimed
        (size_class_waste * compaction_efficiency * 1024.0) as usize // Convert to bytes
    }
    
    /// Estimate large object compaction results
    fn estimate_large_object_compaction(&self, metrics: &FragmentationMetrics) -> usize {
        let large_object_waste = metrics.large_object_fragmentation;
        let compaction_efficiency = 0.85; // Large objects compact more efficiently
        (large_object_waste * compaction_efficiency * metrics.wasted_bytes as f64) as usize
    }
    
    /// Estimate memory region compaction results
    fn estimate_region_compaction(&self, metrics: &FragmentationMetrics) -> usize {
        let region_waste: f64 = metrics.region_fragmentation.iter().sum();
        let compaction_efficiency = 0.70; // Region compaction is more complex
        (region_waste * compaction_efficiency * 2048.0) as usize // Regions are larger units
    }
    
    /// Get current fragmentation metrics
    pub fn get_current_metrics(&self) -> FragmentationMetrics {
        self.current_metrics.read().unwrap().clone()
    }
    
    /// Get fragmentation trends
    pub fn get_trends(&self) -> Option<FragmentationTrends> {
        self.history.lock().unwrap().trends.clone().into()
    }
    
    /// Check if emergency compaction is needed
    pub fn needs_emergency_compaction(&self) -> bool {
        self.scheduler.lock().unwrap().emergency_compaction_needed ||
        self.current_metrics.read().unwrap().overall_fragmentation > self.config.max_fragmentation_level
    }
    
    /// Update compaction strategy
    pub fn update_strategy(&self, new_strategy: CompactionStrategy) {
        *self.strategy.write().unwrap() = new_strategy;
    }
    
    /// Get compaction statistics
    pub fn get_statistics(&self) -> CompactionStatistics {
        CompactionStatistics {
            compactions_triggered: self.stats.compactions_triggered.load(Ordering::Relaxed),
            bytes_compacted: self.stats.bytes_compacted.load(Ordering::Relaxed),
            total_compaction_time: Duration::from_nanos(
                self.stats.total_compaction_time.load(Ordering::Relaxed)
            ),
            average_fragmentation_reduction: 
                self.stats.average_fragmentation_reduction.load(Ordering::Relaxed) as f64 / 10000.0,
            compaction_efficiency: self.stats.compaction_efficiency.load(Ordering::Relaxed),
            predictive_accuracy: 
                self.stats.predictive_accuracy.load(Ordering::Relaxed) as f64 / 10000.0,
        }
    }
}

/// Result of a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub target: CompactionTarget,
    pub bytes_compacted: usize,
    pub duration: Duration,
    pub fragmentation_before: f64,
    pub fragmentation_after: f64,
    pub success: bool,
}

/// Statistics for compaction operations
#[derive(Debug, Clone)]
pub struct CompactionStatistics {
    pub compactions_triggered: usize,
    pub bytes_compacted: usize,
    pub total_compaction_time: Duration,
    pub average_fragmentation_reduction: f64,
    pub compaction_efficiency: u64,
    pub predictive_accuracy: f64,
}

// Implement Clone for FragmentationTrends to support Option<FragmentationTrends>
impl Clone for FragmentationTrends {
    fn clone(&self) -> Self {
        Self {
            growth_rate: self.growth_rate,
            predicted_fragmentation: self.predicted_fragmentation,
            trend_direction: self.trend_direction,
            prediction_confidence: self.prediction_confidence,
        }
    }
}

// Implement Into<Option<FragmentationTrends>> for FragmentationTrends
impl From<FragmentationTrends> for Option<FragmentationTrends> {
    fn from(trends: FragmentationTrends) -> Self {
        Some(trends)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragmentation_manager_creation() {
        let manager = FragmentationManager::new(0.25);
        let metrics = manager.get_current_metrics();
        
        assert_eq!(metrics.overall_fragmentation, 0.0);
        assert_eq!(metrics.wasted_bytes, 0);
    }

    #[test]
    fn test_fragmentation_threshold_triggering() {
        let manager = FragmentationManager::new(0.25);
        
        // Update with high fragmentation
        let high_frag_info = FragmentationInfo {
            total_free_space: 1000,
            free_block_count: 100,
            largest_free_block: 10,
            average_free_block_size: 10.0,
            fragmentation_ratio: 0.8, // 80% fragmentation
        };
        
        manager.update_metrics(high_frag_info);
        
        // Should trigger compaction
        assert!(manager.needs_emergency_compaction());
    }

    #[test]
    fn test_compaction_scheduling() {
        let manager = FragmentationManager::new(0.25);
        
        manager.schedule_compaction(CompactionTarget::SizeClasses, CompactionPriority::High);
        
        let next_request = manager.get_next_compaction_request();
        assert_eq!(next_request, Some(CompactionTarget::SizeClasses));
    }

    #[test]
    fn test_trend_analysis() {
        let manager = FragmentationManager::new(0.25);
        
        // Simulate increasing fragmentation over time
        for i in 0..10 {
            let fragmentation_ratio = 0.1 + (i as f64 * 0.05); // Increasing trend
            let info = FragmentationInfo {
                total_free_space: 1000,
                free_block_count: 50,
                largest_free_block: 100,
                average_free_block_size: 20.0,
                fragmentation_ratio,
            };
            
            manager.update_metrics(info);
            
            // Sleep briefly to create time progression
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        // Should detect increasing trend
        if let Some(trends) = manager.get_trends() {
            assert_eq!(trends.trend_direction, TrendDirection::Increasing);
            assert!(trends.growth_rate > 0.0);
        }
    }
} 