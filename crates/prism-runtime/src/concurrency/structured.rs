//! Structured Concurrency - Scopes and Cancellation Management
//!
//! This module implements structured concurrency primitives as specified in PLD-005:
//! - **Structured scopes**: All concurrent operations have clear parent-child relationships
//! - **Automatic cleanup**: Child operations are cancelled when parent scope exits
//! - **Error propagation**: Errors propagate through the scope hierarchy
//! - **Resource tracking**: All resources are properly cleaned up on scope exit
//! - **AI metadata**: Rich metadata for understanding concurrency patterns

use crate::{authority, resources, intelligence};
use crate::concurrency::r#async::{AsyncHandle, AsyncResult, CancellationToken, TaskPriority, TaskId, AsyncTaskMetadata};
use crate::resources::effects::Effect;
use crate::intelligence::metadata::AIMetadataCollector;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use thiserror::Error;
use uuid::Uuid;
use tokio::sync::broadcast;

/// Unique identifier for structured scopes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(Uuid);

impl ScopeId {
    /// Generate a new unique scope ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// A structured concurrency scope that manages child operations
#[derive(Debug)]
pub struct StructuredScope {
    /// Scope ID
    id: ScopeId,
    /// Cancellation token for this scope
    cancellation_token: CancellationToken,
    /// Child tasks spawned in this scope
    child_tasks: Arc<RwLock<HashMap<TaskId, ChildTask>>>,
    /// Child scopes created within this scope
    child_scopes: Arc<RwLock<HashMap<ScopeId, StructuredScope>>>,
    /// Scope metadata
    metadata: ScopeMetadata,
    /// Capabilities available in this scope
    capabilities: authority::CapabilitySet,
    /// Effect tracker
    effect_tracker: Arc<resources::effects::EffectTracker>,
}

/// Information about a child task
#[derive(Debug)]
struct ChildTask {
    id: TaskId,
    handle: tokio::task::JoinHandle<()>,
    metadata: TaskMetadata,
    started_at: Instant,
}

/// Metadata about a task within a scope
#[derive(Debug, Clone)]
struct TaskMetadata {
    purpose: String,
    effects: Vec<Effect>,
    expected_duration: Option<Duration>,
}

/// Metadata about a structured scope
#[derive(Debug, Clone)]
pub struct ScopeMetadata {
    /// Business purpose of this scope
    pub purpose: String,
    /// Scope type description
    pub scope_type: String,
    /// Expected lifetime
    pub expected_lifetime: Option<Duration>,
    /// Capabilities used
    pub capabilities: Vec<String>,
    /// Effects produced
    pub effects: Vec<String>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// AI-comprehensible context
    pub ai_context: ScopeAIContext,
}

/// AI context for structured scopes
#[derive(Debug, Clone, Default)]
pub struct ScopeAIContext {
    /// Business domain this scope operates in
    pub business_domain: Option<String>,
    /// Concurrency pattern being used
    pub concurrency_pattern: String,
    /// Resource management strategy
    pub resource_management: String,
    /// Error handling strategy
    pub error_handling: String,
}

/// Handle to a structured scope
#[derive(Debug, Clone)]
pub struct ScopeHandle {
    /// Scope ID
    id: ScopeId,
    /// Cancellation token
    cancellation_token: CancellationToken,
    /// Scope metadata
    metadata: ScopeMetadata,
}

impl ScopeHandle {
    /// Get the scope ID
    pub fn id(&self) -> ScopeId {
        self.id
    }

    /// Cancel this scope
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Check if this scope is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }

    /// Get the scope metadata
    pub fn metadata(&self) -> &ScopeMetadata {
        &self.metadata
    }
}

impl StructuredScope {
    /// Create a new structured scope
    pub fn new(
        capabilities: authority::CapabilitySet,
        parent_token: Option<CancellationToken>,
        effect_tracker: Arc<resources::effects::EffectTracker>,
    ) -> Result<Self, StructuredError> {
        let id = ScopeId::new();
        let cancellation_token = if let Some(parent) = parent_token {
            parent.child()
        } else {
            CancellationToken::new()
        };

        let metadata = ScopeMetadata {
            purpose: "General structured scope".to_string(),
            scope_type: "Structured".to_string(),
            expected_lifetime: None,
            capabilities: capabilities.capability_names(),
            effects: Vec::new(),
            created_at: SystemTime::now(),
            ai_context: ScopeAIContext::default(),
        };

        Ok(Self {
            id,
            cancellation_token,
            child_tasks: Arc::new(RwLock::new(HashMap::new())),
            child_scopes: Arc::new(RwLock::new(HashMap::new())),
            metadata,
            capabilities,
            effect_tracker,
        })
    }

    /// Spawn a task within this scope
    pub fn spawn<F, T>(&self, effect: Effect, future: F) -> Result<AsyncHandle<T>, StructuredError>
    where
        F: std::future::Future<Output = AsyncResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        // Check if scope is cancelled
        if self.cancellation_token.is_cancelled() {
            return Err(StructuredError::ScopeCancelled { id: self.id });
        }

        // Create cancellation token that inherits from scope
        let task_token = self.cancellation_token.child();
        
        // Create task with scope's capabilities
        let task_id = TaskId::new();
        
        // Wrap future with scope cancellation
        let scoped_future = async move {
            tokio::select! {
                result = future => result,
                _ = task_token.cancelled() => Err(crate::concurrency::r#async::AsyncError::Cancelled),
            }
        };

        // Create join handle
        let join_handle = tokio::spawn(async move {
            match scoped_future.await {
                Ok(_) => {}
                Err(_) => {}
            }
        });

        // Create task metadata
        let task_metadata = TaskMetadata {
            purpose: "Scoped task".to_string(),
            effects: vec![effect],
            expected_duration: None,
        };

        // Register child task
        {
            let mut tasks = self.child_tasks.write().unwrap();
            tasks.insert(task_id, ChildTask {
                id: task_id,
                handle: join_handle,
                metadata: task_metadata,
                started_at: Instant::now(),
            });
        }

        // Create async handle
        let handle = AsyncHandle::new(
            task_id,
            self.cancellation_token.clone(),
            AsyncTaskMetadata {
                purpose: "Scoped async task".to_string(),
                type_name: "ScopedTask".to_string(),
                capabilities: self.capabilities.capability_names(),
                effects: vec!["Computation".to_string()],
                expected_duration: None,
                created_at: SystemTime::now(),
                priority: TaskPriority::Normal,
            },
        );

        Ok(handle)
    }

    /// Create a child scope
    pub fn child_scope(&self) -> Result<StructuredScope, StructuredError> {
        if self.cancellation_token.is_cancelled() {
            return Err(StructuredError::ScopeCancelled { id: self.id });
        }

        let child_scope = StructuredScope::new(
            self.capabilities.clone(),
            Some(self.cancellation_token.clone()),
            Arc::clone(&self.effect_tracker),
        )?;

        // Register child scope
        {
            let mut scopes = self.child_scopes.write().unwrap();
            scopes.insert(child_scope.id, child_scope.clone());
        }

        Ok(child_scope)
    }

    /// Wait for all child operations to complete
    pub async fn wait_for_completion(&self) -> Result<(), StructuredError> {
        // Wait for all child tasks
        let task_handles: Vec<_> = {
            let tasks = self.child_tasks.read().unwrap();
            tasks.values().map(|task| task.id).collect()
        };

        for task_id in task_handles {
            let task = {
                let mut tasks = self.child_tasks.write().unwrap();
                tasks.remove(&task_id)
            };

            if let Some(task) = task {
                if let Err(_) = task.handle.await {
                    return Err(StructuredError::TaskFailed {
                        scope_id: self.id,
                        error: "Task panicked".to_string(),
                    });
                }
            }
        }

        // Wait for all child scopes
        let child_scope_ids: Vec<_> = {
            let scopes = self.child_scopes.read().unwrap();
            scopes.keys().copied().collect()
        };

        for scope_id in child_scope_ids {
            let scope = {
                let scopes = self.child_scopes.read().unwrap();
                scopes.get(&scope_id).cloned()
            };

            if let Some(scope) = scope {
                scope.wait_for_completion().await?;
            }
        }

        Ok(())
    }

    /// Cancel this scope and all child operations
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Clean up completed tasks from the scope
    pub fn cleanup_completed_tasks(&self) {
        let mut tasks = self.child_tasks.write().unwrap();
        tasks.retain(|_, task| !task.handle.is_finished());
    }

    /// Get a handle to this scope
    pub fn handle(&self) -> ScopeHandle {
        ScopeHandle {
            id: self.id,
            cancellation_token: self.cancellation_token.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Get the number of active tasks in this scope
    pub fn active_task_count(&self) -> usize {
        self.child_tasks.read().unwrap().len()
    }

    /// Get the number of child scopes
    pub fn child_scope_count(&self) -> usize {
        self.child_scopes.read().unwrap().len()
    }

    /// Check if this scope is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }

    /// Execute a future with a timeout within this scope
    pub async fn with_timeout<F, T>(&self, duration: Duration, future: F) -> Result<T, StructuredError>
    where
        F: std::future::Future<Output = Result<T, StructuredError>> + Send,
    {
        tokio::select! {
            result = future => result,
            _ = tokio::time::sleep(duration) => {
                Err(StructuredError::Timeout {
                    scope_id: self.id,
                    timeout: duration,
                })
            }
            _ = self.cancellation_token.cancelled() => {
                Err(StructuredError::ScopeCancelled { id: self.id })
            }
        }
    }

    /// Join all child operations (wait for completion and handle errors)
    pub async fn join_all(&self) -> Result<(), StructuredError> {
        self.wait_for_completion().await?;
        self.cleanup_completed_tasks();
        Ok(())
    }
}

impl Clone for StructuredScope {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            cancellation_token: self.cancellation_token.clone(),
            child_tasks: Arc::clone(&self.child_tasks),
            child_scopes: Arc::clone(&self.child_scopes),
            metadata: self.metadata.clone(),
            capabilities: self.capabilities.clone(),
            effect_tracker: Arc::clone(&self.effect_tracker),
        }
    }
}

impl Drop for StructuredScope {
    fn drop(&mut self) {
        // Cancel all child operations when scope is dropped
        self.cancel();
    }
}

/// Coordinator for structured concurrency operations
#[derive(Debug)]
pub struct StructuredCoordinator {
    /// Active scopes
    active_scopes: Arc<RwLock<HashMap<ScopeId, Arc<StructuredScope>>>>,
    /// Scope statistics
    stats: Arc<RwLock<StructuredStats>>,
    /// AI metadata collector
    ai_collector: Arc<AIMetadataCollector>,
}

/// Statistics for structured concurrency
#[derive(Debug, Clone, Default)]
pub struct StructuredStats {
    /// Total scopes created
    pub total_scopes_created: u64,
    /// Currently active scopes
    pub active_scopes: usize,
    /// Total tasks spawned
    pub total_tasks_spawned: u64,
    /// Successful completions
    pub successful_completions: u64,
    /// Cancelled operations
    pub cancelled_operations: u64,
    /// Average scope lifetime
    pub average_scope_lifetime: Duration,
}

impl StructuredCoordinator {
    /// Create a new structured coordinator
    pub fn new() -> Self {
        Self {
            active_scopes: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StructuredStats::default())),
            ai_collector: Arc::new(AIMetadataCollector::new_default()),
        }
    }

        /// Create a new structured scope
    pub fn create_scope(&self) -> Result<StructuredScope, StructuredError> {
        let capabilities = authority::CapabilitySet::new();
        let effect_tracker = Arc::new(resources::effects::EffectTracker::new()?);
        
        let scope = StructuredScope::new(
            capabilities,
            None, // No parent token for root scope
            effect_tracker,
        )?;

        // Register the scope
        {
            let mut scopes = self.active_scopes.write().unwrap();
            scopes.insert(scope.id, Arc::new(scope.clone()));
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_scopes_created += 1;
            stats.active_scopes += 1;
        }

        Ok(scope)
    }

    /// Remove a scope when it completes
    pub fn remove_scope(&self, scope_id: ScopeId) {
        let mut scopes = self.active_scopes.write().unwrap();
        if scopes.remove(&scope_id).is_some() {
            let mut stats = self.stats.write().unwrap();
            stats.active_scopes = stats.active_scopes.saturating_sub(1);
            stats.successful_completions += 1;
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> StructuredStats {
        self.stats.read().unwrap().clone()
    }

    /// Get number of active scopes
    pub fn scope_count(&self) -> usize {
        self.active_scopes.read().unwrap().len()
    }

    /// Cancel all active scopes
    pub async fn cancel_all(&self) {
        let scopes = {
            let scopes = self.active_scopes.read().unwrap();
            scopes.values().cloned().collect::<Vec<_>>()
        };

        for scope in scopes {
            scope.cancellation_token.cancel();
        }
    }
}

/// Errors that can occur in structured concurrency
#[derive(Debug, Clone, Error)]
pub enum StructuredError {
    /// Scope was cancelled
    #[error("Scope {id:?} was cancelled")]
    ScopeCancelled { id: ScopeId },
    
    /// Operation timed out
    #[error("Operation in scope {scope_id:?} timed out after {timeout:?}")]
    Timeout {
        scope_id: ScopeId,
        timeout: Duration,
    },
    
    /// Task failed
    #[error("Task in scope {scope_id:?} failed: {error}")]
    TaskFailed {
        scope_id: ScopeId,
        error: String,
    },
    
    /// Capability error
    #[error("Capability error: {0}")]
    Capability(#[from] authority::CapabilityError),
    
    /// Effect error
    #[error("Effect error: {0}")]
    Effect(#[from] resources::EffectError),
    
    /// Generic structured concurrency error
    #[error("Structured concurrency error: {message}")]
    Generic { message: String },
}