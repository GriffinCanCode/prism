//! Concurrency System - PLD-005 Implementation
//!
//! This module implements the Prism Concurrency Model as specified in PLD-005, providing:
//! - **Actor System**: Capability-secured actors with supervision and message passing
//! - **Async Runtime**: Structured concurrency with async/await and cancellation
//! - **Effect Integration**: Full integration with the effect system and capabilities
//!
//! ## Design Principles from PLD-005
//!
//! 1. **Structured Concurrency by Default**: All concurrent operations have clear lifetimes
//! 2. **Capability-Secured Isolation**: Actors operate within capability boundaries  
//! 3. **Effect-Aware Composition**: Type system tracks effects of concurrent operations
//! 4. **Message-Passing over Shared State**: Actors communicate through typed messages
//! 5. **Progressive Concurrency**: From simple async/await to full actor systems
//! 6. **Semantic Transparency**: Concurrent operations express business intent
//!
//! ## Architecture
//!
//! The concurrency system is organized into three complementary models:
//! - `async_runtime/` - Async/await for I/O-bound operations
//! - `actor_system/` - Actors for stateful concurrent components  
//! - `structured/` - Structured concurrency primitives and scopes
//!
//! ## Integration with Existing Systems
//!
//! This module builds upon and integrates with:
//! - **Authority System**: All actors require explicit capabilities
//! - **Effect System**: All concurrent operations declare their effects
//! - **Resource System**: Memory and resource usage tracked across actors
//! - **Security System**: Policy enforcement applies to concurrent operations
//! - **Intelligence System**: AI metadata generated for concurrent patterns

use crate::{authority, resources, security, intelligence};
use crate::resources::effects::Effect;
use std::sync::Arc;
use thiserror::Error;

pub mod actor_system;
pub mod async_runtime;
pub mod structured;
pub mod event_bus;  // NEW: Event bus for publish-subscribe patterns
pub mod supervision;  // NEW: Supervision system for fault tolerance
pub mod performance; // NEW: Performance optimization module

#[cfg(test)]
pub mod tests;

#[cfg(test)]
pub mod test_runner;

#[cfg(test)]
pub mod simple_tests;

// Re-exports for public API
pub use actor_system::{Actor, ActorRef, ActorSystem, Message, ActorId, ActorError};
pub use async_runtime::{AsyncRuntime, AsyncHandle, AsyncResult, TaskPriority};
pub use structured::{StructuredScope, ScopeHandle};
pub use event_bus::{EventBus, EventSubscription, EventPriority, EventFilter, EventBusAIMetadata};
pub use supervision::{Supervisor, SupervisionStrategy, RestartPolicy, SupervisionDecision, ChildMetadata};
pub use performance::{PerformanceOptimizer, MessageBatch, BatchingPolicy, LockFreeQueue, LockFreeMap, NumaScheduler};

/// Complete concurrency system that coordinates all concurrency models
#[derive(Debug)]
pub struct ConcurrencySystem {
    /// Actor system for stateful concurrent components
    actor_system: Arc<actor_system::ActorSystem>,
    
    /// Async runtime for I/O-bound operations
    async_runtime: Arc<async_runtime::AsyncRuntime>,
    
    /// Structured concurrency coordinator
    structured_coordinator: Arc<structured::StructuredCoordinator>,
    
    /// Performance optimizer for high-performance operations
    performance_optimizer: Arc<performance::PerformanceOptimizer>,
}

impl ConcurrencySystem {
    /// Create a new concurrency system
    pub fn new() -> Result<Self, ConcurrencyError> {
        let actor_system = Arc::new(actor_system::ActorSystem::new()?);
        let async_runtime = Arc::new(async_runtime::AsyncRuntime::new()?);
        let structured_coordinator = Arc::new(structured::StructuredCoordinator::new()?);
        let performance_optimizer = Arc::new(performance::PerformanceOptimizer::new()
            .map_err(|e| ConcurrencyError::Generic { 
                message: format!("Failed to create performance optimizer: {}", e) 
            })?);

        Ok(Self {
            actor_system,
            async_runtime,
            structured_coordinator,
            performance_optimizer,
        })
    }

    /// Spawn an actor with capabilities
    pub fn spawn_actor<A>(
        &self,
        actor: A,
        capabilities: authority::CapabilitySet,
    ) -> Result<ActorRef<A>, ConcurrencyError> {
        Ok(self.actor_system.spawn_actor(actor, capabilities)?)
    }

    /// Create a structured concurrency scope
    pub fn create_scope(&self) -> Result<StructuredScope, ConcurrencyError> {
        Ok(self.structured_coordinator.create_scope()?)
    }

    /// Execute async code with structured concurrency
    pub async fn execute_async<F, T>(&self, future: F) -> Result<T, ConcurrencyError>
    where
        F: std::future::Future<Output = Result<T, ConcurrencyError>> + Send + 'static,
        T: Send + 'static,
    {
        self.async_runtime.execute_structured(future).await
            .map_err(|e| ConcurrencyError::Generic { 
                message: format!("Async execution failed: {}", e) 
            })
    }

    /// Get number of active actors
    pub fn actor_count(&self) -> usize {
        self.actor_system.active_count()
    }

    /// Get number of async tasks
    pub fn task_count(&self) -> usize {
        self.async_runtime.task_count()
    }

    /// Get number of structured scopes
    pub fn scope_count(&self) -> usize {
        self.structured_coordinator.scope_count()
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> performance::PerformanceMetrics {
        self.performance_optimizer.get_metrics()
    }

    /// Get performance optimization hints
    pub fn get_optimization_hints(&self) -> Vec<performance::OptimizationHint> {
        self.performance_optimizer.get_optimization_hints()
    }

    /// Apply automatic performance optimizations
    pub async fn auto_optimize(&self) -> Result<(), ConcurrencyError> {
        self.performance_optimizer.auto_optimize().await
            .map_err(|e| ConcurrencyError::Generic { 
                message: format!("Auto-optimization failed: {}", e) 
            })
    }

    /// Shutdown the concurrency coordinator
    pub async fn shutdown(&self) -> Result<(), ConcurrencyError> {
        // Shutdown all subsystems
        self.actor_system.shutdown().await
            .map_err(|e| ConcurrencyError::ActorSystem(e))?;
        
        self.async_runtime.shutdown().await
            .map_err(|e| ConcurrencyError::AsyncRuntime(format!("Async runtime shutdown failed: {:?}", e)))?;
        
        self.event_bus.shutdown().await
            .map_err(|e| ConcurrencyError::EventBus(format!("Event bus shutdown failed: {:?}", e)))?;
        
        Ok(())
    }

    /// Create a high-performance lock-free queue
    pub fn create_lock_free_queue<T>(&self) -> performance::LockFreeQueue<T> {
        performance::LockFreeQueue::new()
    }

    /// Create a high-performance lock-free map
    pub fn create_lock_free_map<K, V>(&self) -> performance::LockFreeMap<K, V> 
    where 
        K: std::hash::Hash + Eq + Clone,
        V: Clone,
    {
        performance::LockFreeMap::new()
    }
}

/// Concurrency-related errors
#[derive(Debug, Clone, Error)]
pub enum ConcurrencyError {
    /// Actor system error
    #[error("Actor system error: {0}")]
    ActorSystem(#[from] actor_system::ActorError),
    
    /// Async runtime error
    #[error("Async runtime error: {0}")]
    AsyncRuntime(String),
    
    /// Event bus error
    #[error("Event bus error: {0}")]
    EventBus(String),
    
    /// Structured concurrency error
    #[error("Structured concurrency error: {0}")]
    Structured(#[from] structured::StructuredError),
    
    /// Generic concurrency error
    #[error("Concurrency error: {message}")]
    Generic { message: String },
}

/// Result type for concurrency operations
pub type ConcurrencyResult<T> = Result<T, ConcurrencyError>; 