//! Supervision System - Fault Tolerance and Error Recovery
//!
//! This module implements the supervision hierarchy system as specified in PLD-005, providing:
//! - **Supervisor trees**: Hierarchical fault isolation and recovery
//! - **Restart strategies**: Configurable failure recovery policies
//! - **"Let it crash" philosophy**: Fail fast with automatic recovery
//! - **Error escalation**: Structured error propagation up the hierarchy
//! - **AI metadata**: Rich metadata for understanding supervision behavior

use crate::concurrency::{ActorRef, ActorError, ActorId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Supervisor that manages child actors with fault tolerance
#[derive(Debug)]
pub struct Supervisor {
    /// Supervisor ID
    id: SupervisorId,
    /// Child actors under supervision
    children: Arc<RwLock<HashMap<ActorId, ChildSpec>>>,
    /// Supervision strategy
    strategy: SupervisionStrategy,
    /// Restart statistics
    stats: Arc<RwLock<SupervisionStats>>,
    /// AI metadata for comprehension
    ai_metadata: SupervisionAIMetadata,
}

/// Unique identifier for supervisors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SupervisorId(Uuid);

impl SupervisorId {
    /// Generate a new supervisor ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Child actor specification with restart policies
#[derive(Debug, Clone)]
pub struct ChildSpec {
    /// Actor ID
    actor_id: ActorId,
    /// Restart policy for this child
    restart_policy: RestartPolicy,
    /// Number of restarts within the time window
    restart_count: u32,
    /// Time window for restart counting
    restart_window: Duration,
    /// Last restart time
    last_restart: Option<SystemTime>,
    /// Maximum restart attempts
    max_restarts: u32,
    /// Child metadata
    metadata: ChildMetadata,
}

/// Supervision strategies for handling child failures
#[derive(Debug, Clone)]
pub enum SupervisionStrategy {
    /// Restart only the failed child
    OneForOne,
    /// Restart all children when one fails
    OneForAll,
    /// Restart children in dependency order
    RestForOne,
    /// Custom strategy with user-defined logic
    Custom(Box<dyn Fn(&ActorId, &ActorError) -> SupervisionDecision + Send + Sync>),
}

/// Restart policies for individual children
#[derive(Debug, Clone)]
pub enum RestartPolicy {
    /// Always restart on failure
    Permanent,
    /// Restart only on unexpected failures
    Transient,
    /// Never restart - let it die
    Temporary,
    /// Restart with exponential backoff
    ExponentialBackoff { 
        initial_delay: Duration, 
        max_delay: Duration, 
        multiplier: f64 
    },
}

/// Decision made by supervisor on child failure
#[derive(Debug, Clone)]
pub enum SupervisionDecision {
    /// Restart the child immediately
    Restart,
    /// Restart the child after a delay
    RestartWithDelay(Duration),
    /// Stop the child permanently
    Stop,
    /// Escalate the failure to parent supervisor
    Escalate,
    /// Ignore the failure
    Ignore,
}

/// Child actor metadata for AI comprehension
#[derive(Debug, Clone)]
pub struct ChildMetadata {
    /// Human-readable name
    pub name: String,
    /// Business purpose of the child
    pub purpose: String,
    /// Expected failure modes
    pub failure_modes: Vec<String>,
    /// Dependencies on other children
    pub dependencies: Vec<ActorId>,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Performance characteristics of a child actor
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Expected message processing rate
    pub messages_per_second: u64,
    /// Average memory usage
    pub memory_usage_mb: u64,
    /// CPU intensity level
    pub cpu_intensity: CpuIntensity,
    /// Network I/O requirements
    pub network_io: NetworkIoProfile,
}

/// CPU intensity levels
#[derive(Debug, Clone)]
pub enum CpuIntensity {
    Low,
    Medium,
    High,
    Burst,
}

/// Network I/O profile
#[derive(Debug, Clone)]
pub struct NetworkIoProfile {
    pub connections_per_second: u64,
    pub bytes_per_second: u64,
    pub connection_duration: Duration,
}

/// Supervision statistics for monitoring
#[derive(Debug, Default)]
pub struct SupervisionStats {
    /// Total child failures handled
    pub total_failures: u64,
    /// Total successful restarts
    pub successful_restarts: u64,
    /// Failed restart attempts
    pub failed_restarts: u64,
    /// Children permanently stopped
    pub permanent_stops: u64,
    /// Failures escalated to parent
    pub escalated_failures: u64,
    /// Average restart time
    pub avg_restart_time: Duration,
}

/// AI metadata for supervision behavior
#[derive(Debug, Clone)]
pub struct SupervisionAIMetadata {
    /// Supervision pattern description
    pub pattern: String,
    /// Fault tolerance guarantees
    pub fault_tolerance: Vec<String>,
    /// Recovery strategies
    pub recovery_strategies: Vec<String>,
    /// Performance impact
    pub performance_impact: String,
    /// Monitoring recommendations
    pub monitoring_recommendations: Vec<String>,
}

impl Supervisor {
    /// Create a new supervisor with the given strategy
    pub fn new(strategy: SupervisionStrategy) -> Self {
        Self {
            id: SupervisorId::new(),
            children: Arc::new(RwLock::new(HashMap::new())),
            strategy,
            stats: Arc::new(RwLock::new(SupervisionStats::default())),
            ai_metadata: SupervisionAIMetadata {
                pattern: "Hierarchical Supervision".to_string(),
                fault_tolerance: vec![
                    "Automatic failure detection".to_string(),
                    "Configurable restart policies".to_string(),
                    "Error isolation between children".to_string(),
                ],
                recovery_strategies: vec![
                    "Immediate restart".to_string(),
                    "Delayed restart with backoff".to_string(),
                    "Escalation to parent supervisor".to_string(),
                ],
                performance_impact: "Low overhead with fast failure detection".to_string(),
                monitoring_recommendations: vec![
                    "Track restart frequency per child".to_string(),
                    "Monitor escalation patterns".to_string(),
                    "Alert on repeated failures".to_string(),
                ],
            },
        }
    }

    /// Add a child actor to supervision
    pub fn supervise_child(
        &self,
        actor_id: ActorId,
        restart_policy: RestartPolicy,
        metadata: ChildMetadata,
    ) -> Result<(), SupervisionError> {
        let child_spec = ChildSpec {
            actor_id,
            restart_policy,
            restart_count: 0,
            restart_window: Duration::from_secs(60), // Default 1-minute window
            last_restart: None,
            max_restarts: 5, // Default max restarts
            metadata,
        };

        let mut children = self.children.write()
            .map_err(|_| SupervisionError::LockPoisoned)?;
        
        children.insert(actor_id, child_spec);
        Ok(())
    }

    /// Restart a child actor
    async fn restart_child(&self, child_id: &ActorId) -> Result<(), SupervisionError> {
        // In a real implementation, this would restart the actor
        // For now, just log the restart attempt
        tracing::info!("Restarting child actor {:?}", child_id);
        Ok(())
    }

    /// Stop a child actor
    fn stop_child(&self, child_id: &ActorId) -> Result<(), SupervisionError> {
        // In a real implementation, this would stop the actor
        // For now, just log the stop attempt
        tracing::info!("Stopping child actor {:?}", child_id);
        Ok(())
    }

    /// Handle child actor failure
    pub async fn handle_child_failure(
        &self,
        child_id: ActorId,
        error: ActorError,
    ) -> Result<SupervisionDecision, SupervisionError> {
        // Update statistics
        {
            let mut stats = self.stats.write()
                .map_err(|_| SupervisionError::LockPoisoned)?;
            stats.total_failures += 1;
        }

        // Get child specification
        let child_spec = {
            let children = self.children.read()
                .map_err(|_| SupervisionError::LockPoisoned)?;
            children.get(&child_id)
                .ok_or(SupervisionError::ChildNotFound(child_id))?
                .clone()
        };

        // Apply supervision strategy
        let decision = match &self.strategy {
            SupervisionStrategy::OneForOne => {
                self.decide_restart(&child_spec, &error).await
            }
            SupervisionStrategy::OneForAll => {
                // Restart all children
                self.restart_all_children().await?;
                SupervisionDecision::Restart
            }
            SupervisionStrategy::RestForOne => {
                // Restart this child and all children started after it
                self.restart_rest_for_one(&child_id).await?;
                SupervisionDecision::Restart
            }
            SupervisionStrategy::Custom(strategy_fn) => {
                strategy_fn(&child_id, &error)
            }
        };

        // Execute the decision
        self.execute_decision(&child_id, &decision).await?;

        Ok(decision)
    }

    /// Decide whether to restart a child based on its policy
    async fn decide_restart(
        &self,
        child_spec: &ChildSpec,
        error: &ActorError,
    ) -> SupervisionDecision {
        match &child_spec.restart_policy {
            RestartPolicy::Permanent => SupervisionDecision::Restart,
            RestartPolicy::Temporary => SupervisionDecision::Stop,
            RestartPolicy::Transient => {
                if error.is_expected() {
                    SupervisionDecision::Stop
                } else {
                    SupervisionDecision::Restart
                }
            }
            RestartPolicy::ExponentialBackoff { 
                initial_delay, 
                max_delay, 
                multiplier 
            } => {
                let delay = self.calculate_backoff_delay(
                    child_spec.restart_count,
                    *initial_delay,
                    *max_delay,
                    *multiplier,
                );
                SupervisionDecision::RestartWithDelay(delay)
            }
        }
    }

    /// Calculate exponential backoff delay
    fn calculate_backoff_delay(
        &self,
        restart_count: u32,
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
    ) -> Duration {
        let delay_ms = initial_delay.as_millis() as f64 * multiplier.powi(restart_count as i32);
        let capped_delay = Duration::from_millis(delay_ms as u64).min(max_delay);
        capped_delay
    }

    /// Execute a supervision decision
    async fn execute_decision(
        &self,
        child_id: &ActorId,
        decision: &SupervisionDecision,
    ) -> Result<(), SupervisionError> {
        match decision {
            SupervisionDecision::Restart => {
                self.restart_child(child_id).await?;
            }
            SupervisionDecision::RestartWithDelay(delay) => {
                tokio::time::sleep(*delay).await;
                self.restart_child(child_id).await?;
            }
            SupervisionDecision::Stop => {
                self.stop_child(child_id)?;
            }
            SupervisionDecision::Escalate => {
                // Escalate to parent supervisor
                return Err(SupervisionError::EscalatedFailure(*child_id));
            }
            SupervisionDecision::Ignore => {
                // Do nothing
            }
        }
        Ok(())
    }

    /// Restart all children (OneForAll strategy)
    async fn restart_all_children(&self) -> Result<(), SupervisionError> {
        let child_ids: Vec<ActorId> = {
            let children = self.children.read()
                .map_err(|_| SupervisionError::LockPoisoned)?;
            children.keys().copied().collect()
        };

        for child_id in child_ids {
            self.restart_child(&child_id).await?;
        }

        Ok(())
    }

    /// Restart children in dependency order (RestForOne strategy)
    async fn restart_rest_for_one(&self, failed_child_id: &ActorId) -> Result<(), SupervisionError> {
        // This is a simplified implementation
        // In a full implementation, we'd need to track child start order
        // and restart only children started after the failed one
        self.restart_child(failed_child_id).await
    }

    /// Get supervision statistics
    pub fn get_stats(&self) -> Result<SupervisionStats, SupervisionError> {
        let stats = self.stats.read()
            .map_err(|_| SupervisionError::LockPoisoned)?;
        Ok((*stats).clone())
    }

    /// Get AI metadata for this supervisor
    pub fn get_ai_metadata(&self) -> &SupervisionAIMetadata {
        &self.ai_metadata
    }
}

/// Errors that can occur in the supervision system
#[derive(Debug, thiserror::Error)]
pub enum SupervisionError {
    #[error("Child actor {0:?} not found")]
    ChildNotFound(ActorId),
    
    #[error("Lock poisoned")]
    LockPoisoned,
    
    #[error("Restart limit exceeded for child {0:?}")]
    RestartLimitExceeded(ActorId),
    
    #[error("Failed to restart child {0:?}: {1}")]
    RestartFailed(ActorId, ActorError),
    
    #[error("Failed to stop child {0:?}: {1}")]
    StopFailed(ActorId, ActorError),
    
    #[error("Failure escalated from child {0:?}")]
    EscalatedFailure(ActorId),
}

impl ActorError {
    /// Check if this error is expected/normal vs unexpected
    pub fn is_expected(&self) -> bool {
        match self {
            ActorError::Timeout => true,
            ActorError::Cancelled => true,
            ActorError::InvalidMessage => false,
            ActorError::PanickedActor => false,
            ActorError::SystemError(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_supervisor_creation() {
        let supervisor = Supervisor::new(SupervisionStrategy::OneForOne);
        assert_eq!(supervisor.children.read().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_child_supervision() {
        let supervisor = Supervisor::new(SupervisionStrategy::OneForOne);
        
        // This would need a mock ActorRef for testing
        // let actor_ref = MockActorRef::new();
        // let metadata = ChildMetadata {
        //     name: "test-child".to_string(),
        //     purpose: "Testing".to_string(),
        //     failure_modes: vec![],
        //     dependencies: vec![],
        //     performance_profile: PerformanceProfile {
        //         messages_per_second: 1000,
        //         memory_usage_mb: 10,
        //         cpu_intensity: CpuIntensity::Low,
        //         network_io: NetworkIoProfile {
        //             connections_per_second: 0,
        //             bytes_per_second: 0,
        //             connection_duration: Duration::from_millis(0),
        //         },
        //     },
        // };
        
        // supervisor.supervise_child(actor_ref, RestartPolicy::Permanent, metadata).unwrap();
        // assert_eq!(supervisor.children.read().unwrap().len(), 1);
    }
} 