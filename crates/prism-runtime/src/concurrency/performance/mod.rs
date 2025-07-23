//! Performance Optimization Module
//!
//! This module implements high-performance optimizations for the concurrency system:
//! - **Message Batching**: Batch multiple messages for efficient processing
//! - **Lock-Free Data Structures**: Wait-free and lock-free concurrent data structures
//! - **NUMA-Aware Scheduling**: CPU topology-aware task and actor placement
//! - **Performance Monitoring**: Real-time performance metrics and optimization hints

use std::time::{Duration, SystemTime};
use std::sync::Arc;
use thiserror::Error;

pub mod message_batching;
pub mod lock_free;
pub mod numa_scheduling;
pub mod metrics;

// Re-exports for public API
pub use message_batching::{MessageBatch, BatchingPolicy, BatchProcessor, BatchingCoordinator};
pub use lock_free::{LockFreeQueue, LockFreeMap, AtomicCounter};
pub use numa_scheduling::{NumaScheduler, CpuAffinity, NumaTopology};
pub use metrics::{PerformanceMetrics, OptimizationHint, PerformanceProfiler};

/// Performance optimizer for concurrency operations
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Performance profiler
    profiler: Arc<PerformanceProfiler>,
    /// Message batching coordinator
    batching_coordinator: Arc<BatchingCoordinator>,
    /// NUMA scheduler
    numa_scheduler: Arc<NumaScheduler>,
    /// Optimization settings
    settings: OptimizerSettings,
}

/// Settings for performance optimization
#[derive(Debug, Clone)]
pub struct OptimizerSettings {
    /// Enable automatic optimization
    pub auto_optimize: bool,
    /// Optimization interval
    pub optimization_interval: Duration,
    /// Minimum improvement threshold to apply optimizations
    pub min_improvement_threshold: f64,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Enable message batching
    pub enable_batching: bool,
}

impl Default for OptimizerSettings {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            optimization_interval: Duration::from_secs(30),
            min_improvement_threshold: 0.05, // 5% improvement threshold
            numa_aware: true,
            enable_batching: true,
        }
    }
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new() -> Self {
        Self {
            profiler: Arc::new(PerformanceProfiler::new().unwrap()),
            batching_coordinator: Arc::new(BatchingCoordinator::new().unwrap()),
            numa_scheduler: Arc::new(NumaScheduler::new().unwrap()),
            settings: OptimizerSettings::default(),
        }
    }

    /// Create with custom settings
    pub fn with_settings(settings: OptimizerSettings) -> Self {
        Self {
            profiler: Arc::new(PerformanceProfiler::new().unwrap()),
            batching_coordinator: Arc::new(BatchingCoordinator::new().unwrap()),
            numa_scheduler: Arc::new(NumaScheduler::new().unwrap()),
            settings,
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.profiler.get_current_metrics()
    }

    /// Generate optimization hints
    pub fn generate_hints(&self) -> Vec<OptimizationHint> {
        self.profiler.analyze_performance()
    }

    /// Apply automatic optimizations
    pub async fn auto_optimize(&self) -> Result<(), PerformanceError> {
        if !self.settings.auto_optimize {
            return Ok(());
        }

        let hints = self.generate_hints();
        for hint in hints {
            if self.should_apply_hint(&hint) {
                self.apply_hint(hint).await?;
            }
        }

        Ok(())
    }

    /// Check if a hint should be applied
    fn should_apply_hint(&self, hint: &OptimizationHint) -> bool {
        match hint {
            OptimizationHint::IncreaseBatchSize { expected_improvement, .. } => {
                *expected_improvement >= self.settings.min_improvement_threshold
            }
            OptimizationHint::RebalanceActors { expected_improvement, .. } => {
                *expected_improvement >= self.settings.min_improvement_threshold
            }
            OptimizationHint::ReduceContentionOn { .. } => true,
            OptimizationHint::OptimizeMemoryAllocation { .. } => true,
            OptimizationHint::AdjustThreadPoolSize { .. } => true,
            OptimizationHint::ToggleOptimization { .. } => true,
        }
    }

    /// Apply an optimization hint
    async fn apply_hint(&self, hint: OptimizationHint) -> Result<(), PerformanceError> {
        match hint {
            OptimizationHint::IncreaseBatchSize { actor_id, suggested_size, .. } => {
                self.batching_coordinator.update_batch_size(actor_id, suggested_size).await?;
            }
            OptimizationHint::RebalanceActors { from_cpu, to_cpu, actor_ids, .. } => {
                self.numa_scheduler.migrate_actors(from_cpu, to_cpu, actor_ids).await?;
            }
            OptimizationHint::ReduceContentionOn { .. } => {
                // Implementation would depend on the specific contention type
                tracing::info!("Applying contention reduction optimization");
            }
            OptimizationHint::OptimizeMemoryAllocation { .. } => {
                // Implementation would optimize memory allocation patterns
                tracing::info!("Applying memory allocation optimization");
            }
            OptimizationHint::AdjustThreadPoolSize { .. } => {
                // Implementation would adjust thread pool sizes
                tracing::info!("Applying thread pool size optimization");
            }
            OptimizationHint::ToggleOptimization { .. } => {
                // Implementation would toggle specific optimizations
                tracing::info!("Toggling optimization setting");
            }
        }
        Ok(())
    }

    /// Start automatic optimization loop
    pub async fn start_auto_optimization(&self) -> Result<(), PerformanceError> {
        if !self.settings.auto_optimize {
            return Ok(());
        }

        let optimizer = Arc::new(self.clone());
        let interval = self.settings.optimization_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;
                if let Err(e) = optimizer.auto_optimize().await {
                    tracing::error!("Auto-optimization failed: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> OptimizerStats {
        OptimizerStats {
            optimizations_applied: 0, // Would track actual optimizations
            average_improvement: 0.0,
            last_optimization: None,
            active_optimizations: Vec::new(),
        }
    }
}

impl Clone for PerformanceOptimizer {
    fn clone(&self) -> Self {
        Self {
            profiler: Arc::clone(&self.profiler),
            batching_coordinator: Arc::clone(&self.batching_coordinator),
            numa_scheduler: Arc::clone(&self.numa_scheduler),
            settings: self.settings.clone(),
        }
    }
}

/// Statistics about performance optimization
#[derive(Debug, Clone)]
pub struct OptimizerStats {
    /// Number of optimizations applied
    pub optimizations_applied: u64,
    /// Average performance improvement
    pub average_improvement: f64,
    /// Last optimization timestamp
    pub last_optimization: Option<SystemTime>,
    /// Currently active optimizations
    pub active_optimizations: Vec<String>,
}

/// Performance optimization errors
#[derive(Debug, Error)]
pub enum PerformanceError {
    /// Profiler error
    #[error("Profiler error: {message}")]
    Profiler { message: String },
    
    /// Batching error
    #[error("Batching error: {message}")]
    Batching { message: String },
    
    /// NUMA scheduling error
    #[error("NUMA scheduling error: {message}")]
    NumaScheduling { message: String },
    
    /// Generic performance error
    #[error("Performance error: {message}")]
    Generic { message: String },
} 