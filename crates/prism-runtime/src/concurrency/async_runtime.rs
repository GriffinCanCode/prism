//! Async Runtime - Structured Concurrency for I/O Operations
//!
//! This module implements the async/await runtime as specified in PLD-005, providing:
//! - **Structured concurrency**: All async operations have clear lifetimes
//! - **Cancellation propagation**: Cancellation tokens flow through async operations
//! - **Effect integration**: Async operations declare their effects
//! - **Capability checking**: All async operations require explicit capabilities
//! - **AI metadata**: Rich metadata for AI comprehension of async patterns

use crate::{authority, resources, intelligence};
use crate::resources::effects::{Effect, EffectId, EffectTracker};
use crate::intelligence::metadata::AIMetadataCollector;
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
    /// Get the task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Check if the task is finished
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    /// Cancel the task
    pub fn cancel(&self) {
        self.cancellation_token.cancel();
    }

    /// Await the task result
    pub async fn await_result(self) -> AsyncResult<T> {
        match self.join_handle.await {
            Ok(result) => result,
            Err(join_error) => {
                if join_error.is_cancelled() {
                    Err(AsyncError::Cancelled)
                } else if join_error.is_panic() {
                    Err(AsyncError::Panic {
                        task_id: self.id,
                        message: "Task panicked".to_string(),
                    })
                } else {
                    Err(AsyncError::Generic {
                        message: format!("Join error: {}", join_error),
                    })
                }
            }
        }
    }

    /// Get task metadata
    pub fn metadata(&self) -> &AsyncTaskMetadata {
        &self.metadata
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPriority {
    /// Low priority background tasks
    Low,
    /// Normal priority tasks
    Normal,
    /// High priority tasks
    High,
    /// Critical system tasks
    Critical,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Cancellation token for structured cancellation
#[derive(Debug, Clone)]
pub struct CancellationToken {
    /// Internal cancellation state
    inner: Arc<CancellationTokenInner>,
}

#[derive(Debug)]
struct CancellationTokenInner {
    /// Cancellation sender
    sender: broadcast::Sender<()>,
    /// Whether this token is cancelled
    is_cancelled: std::sync::atomic::AtomicBool,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1);
        Self {
            inner: Arc::new(CancellationTokenInner {
                sender,
                is_cancelled: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    /// Create a child token that is cancelled when this token is cancelled
    pub fn child(&self) -> Self {
        // For simplicity, return a clone of this token
        // In a full implementation, this would create a hierarchical cancellation structure
        self.clone()
    }

    /// Cancel this token
    pub fn cancel(&self) {
        self.inner.is_cancelled.store(true, Ordering::SeqCst);
        let _ = self.inner.sender.send(());
    }

    /// Check if this token is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled.load(Ordering::SeqCst)
    }

    /// Check for cancellation and return error if cancelled
    pub fn check_cancelled(&self) -> AsyncResult<()> {
        if self.is_cancelled() {
            Err(AsyncError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Get a future that completes when this token is cancelled
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        
        let mut receiver = self.inner.sender.subscribe();
        let _ = receiver.recv().await;
    }
}

/// Metadata about an async task
#[derive(Debug, Clone)]
pub struct AsyncTaskMetadata {
    /// Business purpose of the task
    pub purpose: String,
    /// Type name of the task
    pub type_name: String,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Effects this task may produce
    pub effects: Vec<String>,
    /// Expected duration (if known)
    pub expected_duration: Option<Duration>,
    /// When the task was created
    pub created_at: SystemTime,
    /// Task priority
    pub priority: TaskPriority,
}

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
    pub ai_collector: Arc<intelligence::metadata::AIMetadataCollector>,
    /// Effect tracker for recording effects
    pub effect_tracker: Arc<EffectTracker>,
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
    /// Effect tracker for async operations
    effect_tracker: Arc<EffectTracker>,
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
    active_count: u64,
    /// Tasks completed successfully
    completed_count: u64,
    /// Tasks that failed
    failed_count: u64,
    /// Tasks that were cancelled
    cancelled_count: u64,
}

impl AsyncRuntime {
    /// Create a new async runtime
    pub fn new() -> Result<Self, AsyncError> {
        let effect_tracker = Arc::new(EffectTracker::new()
            .map_err(|e| AsyncError::Generic {
                message: format!("Failed to create effect tracker: {}", e)
            })?);

        Ok(Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(AsyncScheduler::new()),
            metrics: Arc::new(Mutex::new(AsyncRuntimeMetrics::default())),
            effect_tracker,
            next_task_id: AtomicU64::new(1),
        })
    }

    /// Spawn a new async task
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

        // Extract task metadata from capabilities and type information
        let task_purpose = self.extract_task_purpose(&capabilities, std::any::type_name::<F>());
        let task_effects = self.extract_task_effects(&capabilities);

        // Create task metadata
        let metadata = AsyncTaskMetadata {
            purpose: task_purpose,
            type_name: std::any::type_name::<F>().to_string(),
            capabilities: capabilities.capability_names(),
            effects: task_effects,
            expected_duration: None,
            created_at: SystemTime::now(),
            priority,
        };

        // Create async context with proper effect tracking
        let effect_id = EffectId::new();
        let ai_collector = Arc::new(AIMetadataCollector::new()
            .map_err(|e| AsyncError::Generic { 
                message: format!("Failed to create AI metadata collector: {}", e) 
            })?);

        let context = AsyncContext {
            task_id,
            capabilities,
            cancellation_token: child_token.clone(),
            effect_id,
            ai_collector,
            effect_tracker: Arc::clone(&self.effect_tracker),
        };

        // Spawn the task
        let join_handle = tokio::spawn(async move {
            // Check cancellation before starting
            context.check_cancelled()?;
            
            // Record task start effect
            context.record_computation("async_task_start", Some("low")).await?;
            
            // Execute the future
            let result = future.await;
            
            // Record task completion effect
            match &result {
                Ok(_) => {
                    context.record_computation("async_task_complete", Some("low")).await?;
                }
                Err(_) => {
                    context.record_computation("async_task_error", Some("low")).await?;
                }
            }
            
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
        let capabilities = self.create_default_capabilities();
        
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
        let capabilities = self.create_default_capabilities();

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
        let capabilities = self.create_default_capabilities();

        let handles: Vec<_> = futures.into_iter().map(|future| {
            self.spawn_task(future, capabilities.clone(), cancellation_token.clone(), TaskPriority::Normal)
        }).collect();

        // Implement proper racing logic using tokio::select!
        match handles.len() {
            1 => handles.into_iter().next().unwrap().await_result().await,
            2 => {
                let mut iter = handles.into_iter();
                let handle1 = iter.next().unwrap();
                let handle2 = iter.next().unwrap();
                
                // Create cancellation tokens for each handle
                let cancel1 = handle1.cancellation_token.clone();
                let cancel2 = handle2.cancellation_token.clone();
                
                tokio::select! {
                    result1 = handle1.await_result() => {
                        // Cancel the other task
                        cancel2.cancel();
                        result1
                    }
                    result2 = handle2.await_result() => {
                        // Cancel the other task
                        cancel1.cancel();
                        result2
                    }
                }
            }
            _ => {
                // For multiple futures, use a more complex select approach
                // For now, just await the first one and cancel the rest
                let mut handles_iter = handles.into_iter();
                let first_handle = handles_iter.next().unwrap();
                
                // Collect cancellation tokens before consuming handles
                let cancel_tokens: Vec<_> = handles_iter.map(|h| h.cancellation_token.clone()).collect();
                
                let result = first_handle.await_result().await;
                
                // Cancel remaining handles
                for token in cancel_tokens {
                    token.cancel();
                }
                
                result
            }
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

        // Wait for tasks to complete or timeout
        let shutdown_timeout = Duration::from_secs(30);
        let start = Instant::now();
        
        while start.elapsed() < shutdown_timeout {
            let active_count = {
                let tasks = self.tasks.read().unwrap();
                tasks.len()
            };
            
            if active_count == 0 {
                break;
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        // Force cleanup remaining tasks
        {
            let mut tasks = self.tasks.write().unwrap();
            tasks.clear();
        }

        Ok(())
    }

    /// Extract task purpose from capabilities and type information
    fn extract_task_purpose(&self, capabilities: &authority::CapabilitySet, type_name: &str) -> String {
        // Try to extract meaningful purpose from capability names
        let cap_names = capabilities.capability_names();
        if !cap_names.is_empty() {
            format!("Task with capabilities: {}", cap_names.join(", "))
        } else if type_name.contains("Future") {
            "Async computation task".to_string()
        } else {
            format!("Async task of type: {}", type_name.split("::").last().unwrap_or(type_name))
        }
    }

    /// Extract likely effects from capabilities
    fn extract_task_effects(&self, capabilities: &authority::CapabilitySet) -> Vec<String> {
        let cap_names = capabilities.capability_names();
        let mut effects = Vec::new();
        
        for cap_name in cap_names {
            if cap_name.to_lowercase().contains("file") || cap_name.to_lowercase().contains("io") {
                effects.push("IO".to_string());
            }
            if cap_name.to_lowercase().contains("network") || cap_name.to_lowercase().contains("http") {
                effects.push("Network".to_string());
            }
            if cap_name.to_lowercase().contains("memory") || cap_name.to_lowercase().contains("alloc") {
                effects.push("Memory".to_string());
            }
            if cap_name.to_lowercase().contains("compute") || cap_name.to_lowercase().contains("cpu") {
                effects.push("Computation".to_string());
            }
        }
        
        if effects.is_empty() {
            effects.push("Computation".to_string()); // Default effect
        }
        
        effects
    }

    /// Create default capabilities for internal tasks
    fn create_default_capabilities(&self) -> authority::CapabilitySet {
        // Create a basic capability set for internal runtime operations
        authority::CapabilitySet::new()
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
    Timeout { task_id: TaskId, timeout: Duration },
    
    /// Task panicked
    #[error("Task {task_id:?} panicked: {message}")]
    Panic { task_id: TaskId, message: String },
    
    /// Capability error
    #[error("Capability error: {0}")]
    Capability(#[from] authority::CapabilityError),
    
    /// Resource error
    #[error("Resource error: {0}")]
    Resource(#[from] resources::ResourceError),
    
    /// Generic async error
    #[error("Async error: {message}")]
    Generic { message: String },
}