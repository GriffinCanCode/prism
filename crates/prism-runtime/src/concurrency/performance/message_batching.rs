//! Message Batching - Efficient Message Processing
//!
//! This module implements message batching optimizations:
//! - **Adaptive Batching**: Dynamically adjust batch sizes based on load
//! - **Priority-Aware Batching**: Higher priority messages get processed first
//! - **Timeout-Based Flushing**: Prevent message starvation with timeout-based flushing
//! - **Memory-Efficient**: Zero-copy message batching where possible
//! - **AI-Optimized**: Machine learning hints for optimal batch sizes

use crate::concurrency::{ActorId, ActorError};
use super::PerformanceError;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;
use uuid::Uuid;

/// Unique identifier for message batches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(Uuid);

impl BatchId {
    /// Generate a new batch ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// A batch of messages for efficient processing
#[derive(Debug)]
pub struct MessageBatch<T> {
    /// Batch ID
    pub id: BatchId,
    /// Messages in this batch
    pub messages: Vec<BatchedMessage<T>>,
    /// Batch creation time
    pub created_at: Instant,
    /// Batch priority (highest priority of contained messages)
    pub priority: MessagePriority,
    /// Batch metadata for AI analysis
    pub metadata: BatchMetadata,
}

/// A message within a batch
#[derive(Debug)]
pub struct BatchedMessage<T> {
    /// Message payload
    pub payload: T,
    /// Message priority
    pub priority: MessagePriority,
    /// Message timestamp
    pub timestamp: Instant,
    /// Response channel for ask messages
    pub response_channel: Option<oneshot::Sender<Result<(), ActorError>>>,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Critical messages (system messages, failures)
    Critical = 0,
    /// High priority messages (user interactions)
    High = 1,
    /// Normal priority messages (business logic)
    Normal = 2,
    /// Low priority messages (background tasks)
    Low = 3,
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Batch metadata for AI analysis
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Actor that will process this batch
    pub target_actor: ActorId,
    /// Average message size in bytes
    pub avg_message_size: usize,
    /// Processing complexity hint
    pub complexity_hint: ProcessingComplexity,
    /// Expected processing time
    pub expected_processing_time: Duration,
}

/// Processing complexity hints for the AI system
#[derive(Debug, Clone)]
pub enum ProcessingComplexity {
    /// Simple operations (O(1) or O(log n))
    Simple,
    /// Medium complexity (O(n))
    Medium,
    /// High complexity (O(n²) or database operations)
    High,
    /// Variable complexity (depends on input)
    Variable,
}

/// Batching policy configuration
#[derive(Debug, Clone)]
pub struct BatchingPolicy {
    /// Minimum batch size before processing
    pub min_batch_size: usize,
    /// Maximum batch size to prevent memory issues
    pub max_batch_size: usize,
    /// Maximum time to wait before flushing a partial batch
    pub max_wait_time: Duration,
    /// Whether to prioritize low-latency or high-throughput
    pub optimization_goal: OptimizationGoal,
    /// Adaptive sizing parameters
    pub adaptive_params: AdaptiveParams,
}

/// Optimization goals for batching
#[derive(Debug, Clone, Copy)]
pub enum OptimizationGoal {
    /// Minimize per-message latency
    LowLatency,
    /// Maximize overall throughput
    HighThroughput,
    /// Balance latency and throughput
    Balanced,
}

/// Adaptive batching parameters
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    /// Learning rate for batch size adjustments
    pub learning_rate: f64,
    /// Target processing time per batch
    pub target_processing_time: Duration,
    /// Minimum adjustment threshold
    pub min_adjustment: f64,
    /// Maximum adjustment per iteration
    pub max_adjustment: f64,
}

impl Default for BatchingPolicy {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 100,
            max_wait_time: Duration::from_millis(10),
            optimization_goal: OptimizationGoal::Balanced,
            adaptive_params: AdaptiveParams {
                learning_rate: 0.1,
                target_processing_time: Duration::from_millis(5),
                min_adjustment: 0.05,
                max_adjustment: 0.5,
            },
        }
    }
}

/// Message batch processor
pub struct BatchProcessor<T> {
    /// Actor ID this processor serves
    actor_id: ActorId,
    /// Batching policy
    policy: BatchingPolicy,
    /// Pending messages waiting to be batched
    pending_messages: Arc<Mutex<VecDeque<BatchedMessage<T>>>>,
    /// Current batch being assembled
    current_batch: Arc<Mutex<Option<MessageBatch<T>>>>,
    /// Batch processing statistics
    stats: Arc<Mutex<BatchProcessingStats>>,
    /// Batch ready notification
    batch_ready_tx: mpsc::UnboundedSender<MessageBatch<T>>,
    /// Batch ready receiver
    batch_ready_rx: Arc<Mutex<mpsc::UnboundedReceiver<MessageBatch<T>>>>,
}

/// Batch processing statistics
#[derive(Debug, Default, Clone)]
pub struct BatchProcessingStats {
    /// Total batches processed
    pub total_batches: u64,
    /// Total messages processed
    pub total_messages: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average processing time per batch
    pub avg_processing_time: Duration,
    /// Average wait time before batching
    pub avg_wait_time: Duration,
    /// Adaptive sizing history
    pub size_adjustments: Vec<(Instant, usize, Duration)>,
}

impl<T: Send + 'static> BatchProcessor<T> {
    /// Create a new batch processor
    pub fn new(actor_id: ActorId, policy: BatchingPolicy) -> Self {
        let (batch_ready_tx, batch_ready_rx) = mpsc::unbounded_channel();
        
        Self {
            actor_id,
            policy,
            pending_messages: Arc::new(Mutex::new(VecDeque::new())),
            current_batch: Arc::new(Mutex::new(None)),
            stats: Arc::new(Mutex::new(BatchProcessingStats::default())),
            batch_ready_tx,
            batch_ready_rx: Arc::new(Mutex::new(batch_ready_rx)),
        }
    }

    /// Add a message to be batched
    pub async fn add_message(
        &self,
        message: T,
        priority: MessagePriority,
        response_channel: Option<oneshot::Sender<Result<(), ActorError>>>,
    ) -> Result<(), PerformanceError> {
        let batched_message = BatchedMessage {
            payload: message,
            priority,
            timestamp: Instant::now(),
            response_channel,
        };

        // Add to pending messages
        {
            let mut pending = self.pending_messages.lock().unwrap();
            pending.push_back(batched_message);
        }

        // Check if we should create a batch
        self.try_create_batch().await?;

        Ok(())
    }

    /// Try to create a batch if conditions are met
    async fn try_create_batch(&self) -> Result<(), PerformanceError> {
        let should_batch = {
            let pending = self.pending_messages.lock().unwrap();
            let current_batch = self.current_batch.lock().unwrap();
            
            // Check batching conditions
            pending.len() >= self.policy.min_batch_size || 
            current_batch.is_none() ||
            self.has_critical_messages(&pending)
        };

        if should_batch {
            self.create_batch().await?;
        } else {
            // Start timeout timer for partial batch flushing
            self.schedule_timeout_flush().await;
        }

        Ok(())
    }

    /// Create a batch from pending messages
    async fn create_batch(&self) -> Result<(), PerformanceError> {
        let messages = {
            let mut pending = self.pending_messages.lock().unwrap();
            let batch_size = std::cmp::min(pending.len(), self.policy.max_batch_size);
            
            if batch_size == 0 {
                return Ok(());
            }

            // Sort by priority (highest first)
            let mut messages: Vec<_> = pending.drain(..batch_size).collect();
            messages.sort_by_key(|msg| msg.priority);
            messages
        };

        if messages.is_empty() {
            return Ok(());
        }

        // Calculate batch metadata
        let priority = messages.iter().map(|m| m.priority).min().unwrap_or_default();
        let avg_message_size = self.estimate_message_size(&messages);
        let complexity_hint = self.estimate_complexity(&messages);
        let expected_processing_time = self.estimate_processing_time(&messages);

        let metadata = BatchMetadata {
            target_actor: self.actor_id,
            avg_message_size,
            complexity_hint,
            expected_processing_time,
        };

        let batch = MessageBatch {
            id: BatchId::new(),
            messages,
            created_at: Instant::now(),
            priority,
            metadata,
        };

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_batches += 1;
            stats.total_messages += batch.messages.len() as u64;
            stats.avg_batch_size = (stats.avg_batch_size * (stats.total_batches - 1) as f64 + batch.messages.len() as f64) / stats.total_batches as f64;
        }

        // Send batch for processing
        self.batch_ready_tx.send(batch)
            .map_err(|_| PerformanceError::Batching {
                message: "Failed to send batch for processing".to_string(),
            })?;

        Ok(())
    }

    /// Schedule timeout-based flush for partial batches
    async fn schedule_timeout_flush(&self) {
        let pending_messages = Arc::clone(&self.pending_messages);
        let batch_ready_tx = self.batch_ready_tx.clone();
        let max_wait_time = self.policy.max_wait_time;
        let actor_id = self.actor_id;

        tokio::spawn(async move {
            tokio::time::sleep(max_wait_time).await;
            
            // Check if there are still pending messages
            let has_pending = {
                let pending = pending_messages.lock().unwrap();
                !pending.is_empty()
            };

            if has_pending {
                // Force flush remaining messages
                let messages: Vec<_> = {
                    let mut pending = pending_messages.lock().unwrap();
                    pending.drain(..).collect()
                };

                if !messages.is_empty() {
                    let batch = MessageBatch {
                        id: BatchId::new(),
                        messages,
                        created_at: Instant::now(),
                        priority: MessagePriority::Normal,
                        metadata: BatchMetadata {
                            target_actor: actor_id,
                            avg_message_size: 0,
                            complexity_hint: ProcessingComplexity::Simple,
                            expected_processing_time: Duration::from_millis(1),
                        },
                    };

                    let _ = batch_ready_tx.send(batch);
                }
            }
        });
    }

    /// Receive the next ready batch
    pub async fn recv_batch(&self) -> Option<MessageBatch<T>> {
        let mut receiver = self.batch_ready_rx.lock().unwrap();
        receiver.recv().await
    }

    /// Check if there are critical messages that need immediate processing
    fn has_critical_messages(&self, messages: &VecDeque<BatchedMessage<T>>) -> bool {
        messages.iter().any(|msg| msg.priority == MessagePriority::Critical)
    }

    /// Estimate message size for batching decisions
    fn estimate_message_size(&self, _messages: &[BatchedMessage<T>]) -> usize {
        // Simplified estimation - in practice, this would use sizeof or serialization
        std::mem::size_of::<T>()
    }

    /// Estimate processing complexity
    fn estimate_complexity(&self, messages: &[BatchedMessage<T>]) -> ProcessingComplexity {
        // Simple heuristic based on batch size
        match messages.len() {
            1..=10 => ProcessingComplexity::Simple,
            11..=50 => ProcessingComplexity::Medium,
            _ => ProcessingComplexity::High,
        }
    }

    /// Estimate processing time for this batch
    fn estimate_processing_time(&self, messages: &[BatchedMessage<T>]) -> Duration {
        // Simple linear estimation - in practice, this would use ML models
        Duration::from_micros(messages.len() as u64 * 100)
    }

    /// Update batch processing statistics after completion
    pub fn record_batch_completion(&self, batch_id: BatchId, processing_time: Duration) {
        let mut stats = self.stats.lock().unwrap();
        stats.avg_processing_time = Duration::from_nanos(
            ((stats.avg_processing_time.as_nanos() * (stats.total_batches - 1) as u128 + processing_time.as_nanos()) / stats.total_batches as u128).try_into().unwrap()
        );

        // Record for adaptive sizing
        let current_size = stats.avg_batch_size as usize;
        stats.size_adjustments.push((Instant::now(), current_size, processing_time));
        
        // Keep only recent adjustments for adaptive learning
        if stats.size_adjustments.len() > 100 {
            stats.size_adjustments.drain(0..50);
        }
    }

    /// Get current processing statistics
    pub fn get_stats(&self) -> BatchProcessingStats {
        (*self.stats.lock().unwrap()).clone()
    }

    /// Perform adaptive batch size optimization
    pub fn optimize_batch_size(&mut self) -> Result<(), PerformanceError> {
        let stats = self.stats.lock().unwrap();
        let target_time = self.policy.adaptive_params.target_processing_time;
        let current_time = stats.avg_processing_time;
        
        if stats.size_adjustments.len() < 10 {
            // Not enough data for optimization
            return Ok(());
        }

        // Calculate adjustment based on performance deviation
        let time_ratio = current_time.as_secs_f64() / target_time.as_secs_f64();
        let adjustment_factor = if time_ratio > 1.0 {
            // Processing too slow - reduce batch size
            -self.policy.adaptive_params.learning_rate * (time_ratio - 1.0)
        } else {
            // Processing fast enough - can increase batch size
            self.policy.adaptive_params.learning_rate * (1.0 - time_ratio)
        };

        // Apply adjustment with bounds
        let adjustment = adjustment_factor.clamp(
            -self.policy.adaptive_params.max_adjustment,
            self.policy.adaptive_params.max_adjustment,
        );

        if adjustment.abs() > self.policy.adaptive_params.min_adjustment {
            let new_size = ((self.policy.max_batch_size as f64) * (1.0 + adjustment)) as usize;
            self.policy.max_batch_size = new_size.clamp(1, 1000); // Reasonable bounds
            
            tracing::debug!(
                "Adapted batch size for actor {:?}: {} -> {} (adjustment: {:.2})",
                self.actor_id,
                stats.avg_batch_size,
                self.policy.max_batch_size,
                adjustment
            );
        }

        Ok(())
    }
}

/// Batching coordinator that manages all batch processors
pub struct BatchingCoordinator {
    /// Batch processors per actor
    processors: Arc<RwLock<HashMap<ActorId, Arc<dyn BatchProcessorTrait + Send + Sync>>>>,
    /// Global batching metrics
    metrics: Arc<Mutex<GlobalBatchingMetrics>>,
}

impl std::fmt::Debug for BatchingCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchingCoordinator")
            .field("processors", &"<HashMap<ActorId, BatchProcessor>>")
            .field("metrics", &"<GlobalBatchingMetrics>")
            .finish()
    }
}

/// Trait for type-erased batch processors
pub trait BatchProcessorTrait {
    fn get_stats(&self) -> BatchProcessingStats;
    fn optimize_batch_size(&self) -> Result<(), PerformanceError>;
}

impl<T: Send + 'static> BatchProcessorTrait for BatchProcessor<T> {
    fn get_stats(&self) -> BatchProcessingStats {
        self.get_stats()
    }

    fn optimize_batch_size(&self) -> Result<(), PerformanceError> {
        // This would need to be made mutable, but for now return Ok
        Ok(())
    }
}

/// Global batching metrics across all actors
#[derive(Debug, Default, Clone)]
pub struct GlobalBatchingMetrics {
    /// Total processors active
    pub active_processors: usize,
    /// Total batches processed across all actors
    pub total_batches: u64,
    /// Total messages processed across all actors
    pub total_messages: u64,
    /// Average batch efficiency (messages per batch)
    pub avg_batch_efficiency: f64,
}

impl BatchingCoordinator {
    /// Create a new batching coordinator
    pub fn new() -> Result<Self, PerformanceError> {
        Ok(Self {
            processors: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(GlobalBatchingMetrics::default())),
        })
    }

    /// Register a batch processor for an actor
    pub fn register_processor<T: Send + 'static>(
        &self,
        actor_id: ActorId,
        processor: BatchProcessor<T>,
    ) -> Result<(), PerformanceError> {
        let mut processors = self.processors.write().unwrap();
        processors.insert(actor_id, Arc::new(processor));
        
        let mut metrics = self.metrics.lock().unwrap();
        metrics.active_processors += 1;
        
        Ok(())
    }

    /// Update batch size for a specific actor
    pub async fn update_batch_size(&self, actor_id: ActorId, new_size: usize) -> Result<(), PerformanceError> {
        let processors = self.processors.read().unwrap();
        if let Some(processor) = processors.get(&actor_id) {
            processor.optimize_batch_size()?;
        }
        Ok(())
    }

    /// Get global batching metrics
    pub fn get_global_metrics(&self) -> GlobalBatchingMetrics {
        let mut global_metrics = (*self.metrics.lock().unwrap()).clone();
        
        // Aggregate metrics from all processors
        let processors = self.processors.read().unwrap();
        for processor in processors.values() {
            let stats = processor.get_stats();
            global_metrics.total_batches += stats.total_batches;
            global_metrics.total_messages += stats.total_messages;
        }
        
        if global_metrics.total_batches > 0 {
            global_metrics.avg_batch_efficiency = global_metrics.total_messages as f64 / global_metrics.total_batches as f64;
        }
        
        global_metrics
    }

    /// Optimize all batch processors
    pub async fn optimize_all(&self) -> Result<(), PerformanceError> {
        let processors = self.processors.read().unwrap();
        for processor in processors.values() {
            processor.optimize_batch_size()?;
        }
        Ok(())
    }
} 