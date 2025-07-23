//! Async Context - Execution Context for Async Operations
//!
//! This module provides the execution context that is passed to async operations:
//! - **Capability integration**: Access to authorized capabilities
//! - **Effect tracking**: Record effects during async execution
//! - **Cancellation support**: Check for cancellation during operation
//! - **AI metadata**: Collect metadata for AI analysis
//! - **Resource management**: Track resource usage during execution
//!
//! The AsyncContext serves as the bridge between the async runtime and the
//! broader Prism runtime systems (authority, resources, intelligence, etc.).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::{authority, resources, intelligence};
use crate::resources::effects::{Effect, EffectId, EffectTracker};
use crate::intelligence::metadata::AIMetadataCollector;

use super::types::{TaskId, AsyncResult, AsyncError};
use super::cancellation::CancellationToken;

/// Context passed to async operations
#[derive(Debug)]
pub struct AsyncContext {
    /// Task ID
    pub task_id: TaskId,
    /// Available capabilities
    pub capabilities: authority::CapabilitySet,
    /// Cancellation token
    pub cancellation_token: CancellationToken,
    /// Effect tracking ID
    pub effect_id: EffectId,
    /// AI metadata collector
    pub ai_collector: Arc<AIMetadataCollector>,
    /// Effect tracker for recording effects
    pub effect_tracker: Arc<EffectTracker>,
}

impl AsyncContext {
    /// Create a new async context
    pub fn new(
        task_id: TaskId,
        capabilities: authority::CapabilitySet,
        cancellation_token: CancellationToken,
        effect_tracker: Arc<EffectTracker>,
        ai_collector: Arc<AIMetadataCollector>,
    ) -> Self {
        let effect_id = EffectId::new();
        Self {
            task_id,
            capabilities,
            cancellation_token,
            effect_id,
            ai_collector,
            effect_tracker,
        }
    }

    /// Check for cancellation
    pub fn check_cancelled(&self) -> AsyncResult<()> {
        self.cancellation_token.check_cancelled()
    }

    /// Execute with timeout
    pub async fn with_timeout<F, T>(&self, duration: Duration, future: F) -> AsyncResult<T>
    where
        F: std::future::Future<Output = AsyncResult<T>> + Send,
    {
        tokio::select! {
            result = future => result,
            _ = tokio::time::sleep(duration) => Err(AsyncError::Timeout { 
                task_id: self.task_id,
                timeout: duration,
            }),
            _ = self.cancellation_token.cancelled() => Err(AsyncError::Cancelled),
        }
    }

    /// Record an effect execution
    pub async fn record_effect(&self, effect: Effect) -> AsyncResult<()> {
        // Create metadata for business correlation
        let mut metadata = HashMap::new();
        metadata.insert("task_id".to_string(), format!("{:?}", self.task_id));
        metadata.insert("timestamp".to_string(), SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .to_string());
        
        // Begin effect tracking
        let effect_id = self.effect_tracker
            .begin_effect(effect, Some(metadata))
            .map_err(|e| AsyncError::Generic {
                message: format!("Failed to begin effect tracking: {}", e)
            })?;
        
        // For demonstration, immediately complete the effect
        // In a real implementation, this would be called when the effect completes
        self.effect_tracker
            .end_effect(effect_id)
            .map_err(|e| AsyncError::Generic {
                message: format!("Failed to complete effect tracking: {}", e)
            })?;
        
        Ok(())
    }

    /// Record a computational effect
    pub async fn record_computation(&self, operation: &str, complexity: Option<&str>) -> AsyncResult<()> {
        let effect = Effect::Computation {
            operation: operation.to_string(),
            complexity: complexity.map(|s| s.to_string()),
        };
        self.record_effect(effect).await
    }

    /// Record an I/O effect
    pub async fn record_io(&self, operation: &str, size: Option<usize>) -> AsyncResult<()> {
        let effect = Effect::IO {
            operation: operation.to_string(),
            size,
        };
        self.record_effect(effect).await
    }

    /// Record a memory effect
    pub async fn record_memory(&self, operation: &str, size: usize) -> AsyncResult<()> {
        let effect = Effect::Memory {
            operation: operation.to_string(),
            size,
        };
        self.record_effect(effect).await
    }

    /// Execute an operation with automatic effect recording
    pub async fn execute_with_effect<F, T>(
        &self,
        effect: Effect,
        operation: F,
    ) -> AsyncResult<T>
    where
        F: std::future::Future<Output = AsyncResult<T>>,
    {
        // Begin effect tracking
        let mut metadata = HashMap::new();
        metadata.insert("task_id".to_string(), format!("{:?}", self.task_id));
        
        let effect_id = self.effect_tracker
            .begin_effect(effect, Some(metadata))
            .map_err(|e| AsyncError::Generic {
                message: format!("Failed to begin effect tracking: {}", e)
            })?;

        // Execute the operation
        let result = operation.await;

        // End effect tracking
        let _completed_effect = self.effect_tracker
            .end_effect(effect_id)
            .map_err(|e| AsyncError::Generic {
                message: format!("Failed to end effect tracking: {}", e)
            })?;

        result
    }
} 