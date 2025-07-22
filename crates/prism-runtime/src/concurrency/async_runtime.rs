//! Async Runtime - Structured Concurrency for I/O Operations
//!
//! This module implements the async/await runtime as specified in PLD-005, providing:
//! - **Structured concurrency**: All async operations have clear lifetimes
//! - **Cancellation propagation**: Cancellation tokens flow through async operations
//! - **Effect integration**: Async operations declare their effects
//! - **Capability checking**: All async operations require explicit capabilities
//! - **AI metadata**: Rich metadata for AI comprehension of async patterns

use crate::{authority, resources, intelligence};
use crate::resources::effects::Effect;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, SystemTime, Instant};
use tokio::sync::{oneshot, broadcast};
use tokio::task::JoinHandle;
use thiserror::Error;
use uuid::Uuid;

/// Unique identifier for async tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(Uuid);

impl TaskId {
    /// Generate a new unique task ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Handle to a running async task
#[derive(Debug)]
pub struct AsyncHandle<T> {
    /// Task ID
    id: TaskId,
    /// Join handle for the task
    join_handle: JoinHandle<AsyncResult<T>>,
    /// Cancellation token
    cancellation_token: CancellationToken,
    /// Task metadata
    metadata: AsyncTaskMetadata,
}

impl<T> AsyncHandle<T> {
    /// Get task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Cancel the async task
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Check if task is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }

    /// Wait for the task to complete
    pub async fn await_result(self) -> AsyncResult<T> {
        match self.join_handle.await {
            Ok(result) => result,
            Err(e) => Err(AsyncError::TaskPanic { 
                id: self.id,
                message: e.to_string(),
            }),
        }
    }

    /// Get task metadata for AI analysis
    pub fn metadata(&self) -> &AsyncTaskMetadata {
        &self.metadata
    }
}

/// Cancellation token for structured concurrency
#[derive(Debug, Clone)]
pub struct CancellationToken {
    /// Cancellation broadcaster
    broadcaster: broadcast::Sender<()>,
    /// Whether this token is cancelled
    is_cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        let (broadcaster, _) = broadcast::channel(1);
        Self {
            broadcaster,
            is_cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Cancel this token and all child tokens
    pub fn cancel(&self) {
        self.is_cancelled.store(true, Ordering::SeqCst);
        let _ = self.broadcaster.send(()); // Ignore error if no receivers
    }

    /// Check if this token is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.is_cancelled.load(Ordering::SeqCst)
    }

    /// Create a child token that will be cancelled when this token is cancelled
    pub fn child(&self) -> Self {
        let child = Self::new();
        
        // Subscribe to parent cancellation
        let mut parent_receiver = self.broadcaster.subscribe();
        let child_broadcaster = child.broadcaster.clone();
        let child_cancelled = Arc::clone(&child.is_cancelled);
        
        tokio::spawn(async move {
            if let Ok(()) = parent_receiver.recv().await {
                child_cancelled.store(true, Ordering::SeqCst);
                let _ = child_broadcaster.send(());
            }
        });
        
        child
    }

    /// Wait for cancellation
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        
        let mut receiver = self.broadcaster.subscribe();
        let _ = receiver.recv().await;
    }

    /// Check for cancellation and return error if cancelled
    pub fn check_cancelled(&self) -> AsyncResult<()> {
        if self.is_cancelled() {
            Err(AsyncError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Async task metadata for AI comprehension
#[derive(Debug, Clone)]
pub struct AsyncTaskMetadata {
    /// Business purpose of this task
    pub purpose: String,
    /// Task type name
    pub type_name: String,
    /// Capabilities required
    pub capabilities: Vec<String>,
    /// Effects produced
    pub effects: Vec<String>,
    /// Expected duration
    pub expected_duration: Option<Duration>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Priority level
    pub priority: TaskPriority,
}

/// Priority levels for async tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Critical tasks (user-facing, blocking)
    Critical = 0,
    /// High priority tasks
    High = 1,
    /// Normal priority tasks
    Normal = 2,
    /// Background tasks
    Background = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Context for async task execution
pub struct AsyncContext {
    /// Task ID
    pub task_id: TaskId,
    /// Available capabilities
    pub capabilities: authority::CapabilitySet,
    /// Cancellation token
    pub cancellation_token: CancellationToken,
    /// Effect tracker handle
    pub effect_id: resources::EffectId,
    /// AI metadata collector
    pub ai_collector: Arc<intelligence::AIMetadataCollector>,
}

impl AsyncContext {
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
        // TODO: Integrate with effect tracking system
        Ok(())
    }
}

/// Async runtime that manages all async operations
#[derive(Debug)]
pub struct AsyncRuntime {
    /// Active tasks
    tasks: Arc<RwLock<HashMap<TaskId, TaskHandle>>>,
    /// Task scheduler
    scheduler: Arc<AsyncScheduler>,
    /// Runtime metrics
    metrics: Arc<Mutex<AsyncRuntimeMetrics>>,
    /// Next task ID
    next_task_id: AtomicU64,
}

/// Handle to a task in the runtime
struct TaskHandle {
    id: TaskId,
    metadata: AsyncTaskMetadata,
    cancellation_token: CancellationToken,
    started_at: Instant,
}

/// Async task scheduler
#[derive(Debug)]
struct AsyncScheduler {
    /// Task queues by priority
    priority_queues: Arc<RwLock<HashMap<TaskPriority, Vec<TaskId>>>>,
    /// Scheduler metrics
    scheduler_metrics: Arc<Mutex<SchedulerMetrics>>,
}

/// Scheduler metrics
#[derive(Debug, Default)]
struct SchedulerMetrics {
    /// Tasks scheduled by priority
    scheduled_by_priority: HashMap<TaskPriority, u64>,
    /// Average queue wait time
    average_wait_time: Duration,
    /// Total tasks scheduled
    total_scheduled: u64,
}

/// Runtime metrics
#[derive(Debug, Default)]
struct AsyncRuntimeMetrics {
    /// Total tasks created
    total_created: u64,
    /// Currently active tasks
    active_count: usize,
    /// Tasks completed successfully
    completed: u64,
    /// Tasks failed
    failed: u64,
    /// Tasks cancelled
    cancelled: u64,
    /// Average task duration
    average_duration: Duration,
}

impl AsyncRuntime {
    /// Create a new async runtime
    pub fn new() -> Result<Self, AsyncError> {
        Ok(Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(AsyncScheduler::new()),
            metrics: Arc::new(Mutex::new(AsyncRuntimeMetrics::default())),
            next_task_id: AtomicU64::new(1),
        })
    }

    /// Spawn an async task with structured concurrency
    pub fn spawn_task<F, T>(
        &self,
        future: F,
        capabilities: authority::CapabilitySet,
        cancellation_token: CancellationToken,
        priority: TaskPriority,
    ) -> AsyncHandle<T>
    where
        F: std::future::Future<Output = AsyncResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = TaskId::new();
        let child_token = cancellation_token.child();

        // Create task metadata
        let metadata = AsyncTaskMetadata {
            purpose: "Async task".to_string(), // TODO: Extract from task
            type_name: std::any::type_name::<F>().to_string(),
            capabilities: capabilities.capability_names(),
            effects: Vec::new(), // TODO: Extract effects
            expected_duration: None,
            created_at: SystemTime::now(),
            priority,
        };

        // Create async context
        let effect_id = resources::EffectId::new(); // TODO: Proper effect tracking
        let ai_collector = Arc::new(intelligence::AIMetadataCollector::new().unwrap()); // TODO: Proper error handling

        let context = AsyncContext {
            task_id,
            capabilities,
            cancellation_token: child_token.clone(),
            effect_id,
            ai_collector,
        };

        // Spawn the task
        let join_handle = tokio::spawn(async move {
            // Check cancellation before starting
            context.check_cancelled()?;
            
            // Execute the future
            let result = future.await;
            
            result
        });

        let async_handle = AsyncHandle {
            id: task_id,
            join_handle,
            cancellation_token: child_token.clone(),
            metadata: metadata.clone(),
        };

        // Register task
        {
            let mut tasks_guard = self.tasks.write().unwrap();
            tasks_guard.insert(task_id, TaskHandle {
                id: task_id,
                metadata,
                cancellation_token: child_token,
                started_at: Instant::now(),
            });
        }

        // Update metrics
        {
            let mut metrics_guard = self.metrics.lock().unwrap();
            metrics_guard.total_created += 1;
            metrics_guard.active_count += 1;
        }

        async_handle
    }

    /// Execute async code with structured concurrency
    pub async fn execute_structured<F, T>(&self, future: F) -> AsyncResult<T>
    where
        F: std::future::Future<Output = Result<T, crate::ConcurrencyError>> + Send + 'static,
        T: Send + 'static,
    {
        let cancellation_token = CancellationToken::new();
        let capabilities = authority::CapabilitySet::new(); // TODO: Get from context
        
        let handle = self.spawn_task(
            async move { 
                future.await.map_err(|e| AsyncError::Generic { 
                    message: e.to_string() 
                })
            },
            capabilities,
            cancellation_token,
            TaskPriority::Normal,
        );

        handle.await_result().await
    }

    /// Get number of active tasks
    pub fn task_count(&self) -> usize {
        self.tasks.read().unwrap().len()
    }

    /// Join multiple async operations
    pub async fn join<T1, T2>(
        &self,
        future1: impl std::future::Future<Output = AsyncResult<T1>> + Send + 'static,
        future2: impl std::future::Future<Output = AsyncResult<T2>> + Send + 'static,
    ) -> AsyncResult<(T1, T2)>
    where
        T1: Send + 'static,
        T2: Send + 'static,
    {
        let cancellation_token = CancellationToken::new();
        let capabilities = authority::CapabilitySet::new(); // TODO: Get from context

        let handle1 = self.spawn_task(future1, capabilities.clone(), cancellation_token.clone(), TaskPriority::Normal);
        let handle2 = self.spawn_task(future2, capabilities, cancellation_token, TaskPriority::Normal);

        let result1 = handle1.await_result().await?;
        let result2 = handle2.await_result().await?;

        Ok((result1, result2))
    }

    /// Race multiple async operations (return first to complete)
    pub async fn race<T>(
        &self,
        futures: Vec<impl std::future::Future<Output = AsyncResult<T>> + Send + 'static>,
    ) -> AsyncResult<T>
    where
        T: Send + 'static,
    {
        if futures.is_empty() {
            return Err(AsyncError::Generic { 
                message: "No futures to race".to_string() 
            });
        }

        let cancellation_token = CancellationToken::new();
        let capabilities = authority::CapabilitySet::new(); // TODO: Get from context

        let handles: Vec<_> = futures.into_iter().map(|future| {
            self.spawn_task(future, capabilities.clone(), cancellation_token.clone(), TaskPriority::Normal)
        }).collect();

        // Use tokio::select! to race the futures
        // For simplicity, just await the first one for now
        // TODO: Implement proper racing logic
        if let Some(handle) = handles.into_iter().next() {
            handle.await_result().await
        } else {
            Err(AsyncError::Generic { 
                message: "No handles to race".to_string() 
            })
        }
    }

    /// Shutdown the async runtime
    pub async fn shutdown(&self) -> Result<(), AsyncError> {
        // Cancel all active tasks
        {
            let tasks = self.tasks.read().unwrap();
            for task in tasks.values() {
                task.cancellation_token.cancel();
            }
        }

        // TODO: Wait for tasks to complete or timeout
        Ok(())
    }
}

impl AsyncScheduler {
    /// Create a new async scheduler
    fn new() -> Self {
        Self {
            priority_queues: Arc::new(RwLock::new(HashMap::new())),
            scheduler_metrics: Arc::new(Mutex::new(SchedulerMetrics::default())),
        }
    }
}

/// Result type for async operations
pub type AsyncResult<T> = Result<T, AsyncError>;

/// Async runtime errors
#[derive(Debug, Error)]
pub enum AsyncError {
    /// Task was cancelled
    #[error("Task was cancelled")]
    Cancelled,
    
    /// Task timed out
    #[error("Task {task_id:?} timed out after {timeout:?}")]
    Timeout {
        task_id: TaskId,
        timeout: Duration,
    },
    
    /// Task panicked
    #[error("Task {id:?} panicked: {message}")]
    TaskPanic {
        id: TaskId,
        message: String,
    },
    
    /// Capability error
    #[error("Capability error: {0}")]
    Capability(#[from] authority::CapabilityError),
    
    /// Effect error
    #[error("Effect error: {0}")]
    Effect(#[from] resources::EffectError),
    
    /// Generic async error
    #[error("Async error: {message}")]
    Generic { message: String },
} 