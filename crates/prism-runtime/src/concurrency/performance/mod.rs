//! Performance Optimization Module
//!
//! This module implements high-performance optimizations for the concurrency system:
//! - **Message Batching**: Batch multiple messages for efficient processing
//! - **Lock-Free Data Structures**: Wait-free and lock-free concurrent data structures
//! - **NUMA-Aware Scheduling**: CPU topology-aware task and actor placement
//! - **Performance Monitoring**: Real-time performance metrics and optimization hints

pub mod message_batching;
pub mod lock_free;
pub mod numa_scheduling;
pub mod metrics;

// Re-exports for public API
pub use message_batching::{MessageBatch, BatchingPolicy, BatchProcessor};
pub use lock_free::{LockFreeQueue, LockFreeMap, AtomicCounter};
pub use numa_scheduling::{NumaScheduler, CpuAffinity, NumaTopology};
pub use metrics::{PerformanceMetrics, OptimizationHint, PerformanceProfiler};

/// Performance optimization coordinator
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Message batching coordinator
    batching: message_batching::BatchingCoordinator,
    /// NUMA-aware scheduler
    numa_scheduler: numa_scheduling::NumaScheduler,
    /// Performance metrics collector
    metrics: metrics::PerformanceProfiler,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new() -> Result<Self, PerformanceError> {
        let batching = message_batching::BatchingCoordinator::new()?;
        let numa_scheduler = numa_scheduling::NumaScheduler::new()?;
        let metrics = metrics::PerformanceProfiler::new()?;

        Ok(Self {
            batching,
            numa_scheduler,
            metrics,
        })
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.get_current_metrics()
    }

    /// Get optimization recommendations
    pub fn get_optimization_hints(&self) -> Vec<OptimizationHint> {
        self.metrics.analyze_performance()
    }

    /// Apply automatic optimizations
    pub async fn auto_optimize(&self) -> Result<(), PerformanceError> {
        let hints = self.get_optimization_hints();
        
        for hint in hints {
            match hint {
                OptimizationHint::IncreaseBatchSize { actor_id, suggested_size } => {
                    self.batching.update_batch_size(actor_id, suggested_size).await?;
                }
                OptimizationHint::RebalanceActors { from_cpu, to_cpu, actor_ids } => {
                    self.numa_scheduler.migrate_actors(from_cpu, to_cpu, actor_ids).await?;
                }
                OptimizationHint::ReduceContentionOn { resource } => {
                    // Implement contention reduction strategies
                    tracing::info!("Reducing contention on resource: {}", resource);
                }
            }
        }

        Ok(())
    }
}

/// Performance optimization errors
#[derive(Debug, thiserror::Error)]
pub enum PerformanceError {
    #[error("Batching error: {message}")]
    Batching { message: String },
    
    #[error("NUMA scheduling error: {message}")]
    NumaScheduling { message: String },
    
    #[error("Metrics collection error: {message}")]
    Metrics { message: String },
    
    #[error("Performance optimization error: {message}")]
    Generic { message: String },
} 