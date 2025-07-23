//! Async Runtime - Main Coordinator for Async Operations
//!
//! This module implements the main async runtime that coordinates all async operations:
//! - **Task spawning**: Create and manage async tasks with capabilities
//! - **Resource integration**: Integrate with effect tracking and resource management
//! - **Structured concurrency**: Support for structured async operations
//! - **Metrics collection**: Track runtime performance and usage
//! - **Lifecycle management**: Handle startup, shutdown, and cleanup
//!
//! The AsyncRuntime serves as the main entry point for async operations,
//! coordinating between the scheduler, context management, and resource systems.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use crate::{authority, resources, intelligence, ConcurrencyError};
use crate::resources::effects::EffectTracker;
use crate::intelligence::metadata::AIMetadataCollector;

use super::types::{TaskId, TaskPriority, AsyncTaskMetadata, AsyncHandle, AsyncResult, AsyncError};
use super::cancellation::CancellationToken;
use super::context::AsyncContext;
use super::scheduler::{AsyncScheduler, SchedulerMetrics};

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
#[derive(Debug)]
struct TaskHandle {
    id: TaskId,
    metadata: AsyncTaskMetadata,
    cancellation_token: CancellationToken,
    started_at: Instant,
}

/// Runtime metrics for performance monitoring
#[derive(Debug, Default)]
pub struct AsyncRuntimeMetrics {
    /// Total tasks created
    pub total_created: u64,
    /// Currently active tasks
    pub active_count: u64,
    /// Tasks completed successfully
    pub completed_count: u64,
    /// Tasks that failed
    pub failed_count: u64,
    /// Tasks that were cancelled
    pub cancelled_count: u64,
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
            created_at: std::time::SystemTime::now(),
            priority,
        };

        // Create async context with proper effect tracking
        let ai_collector = Arc::new(AIMetadataCollector::new()
            .unwrap_or_else(|_| AIMetadataCollector::new_default()));

        let context = AsyncContext::new(
            task_id,
            capabilities,
            child_token.clone(),
            Arc::clone(&self.effect_tracker),
            ai_collector,
        );

        // Spawn the task
        let join_handle = tokio::spawn(async move {
            // Check cancellation
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

        let async_handle = AsyncHandle::new(
            task_id,
            join_handle,
            child_token.clone(),
            metadata.clone(),
        );

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

        // Schedule task
        self.scheduler.schedule_task(task_id, priority);

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
        F: std::future::Future<Output = Result<T, ConcurrencyError>> + Send + 'static,
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
                let cancel1 = handle1.cancellation_token().clone();
                let cancel2 = handle2.cancellation_token().clone();
                
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
                // For now, just await the first one and cancel the rest
                let mut handles_iter = handles.into_iter();
                let first_handle = handles_iter.next().unwrap();
                
                // Collect cancellation tokens before consuming handles
                let cancel_tokens: Vec<_> = handles_iter.map(|h| h.cancellation_token().clone()).collect();
                
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

        // Clear scheduler queues
        self.scheduler.clear_all_queues();

        Ok(())
    }

    /// Get runtime metrics
    pub fn metrics(&self) -> AsyncRuntimeMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get scheduler metrics
    pub fn scheduler_metrics(&self) -> SchedulerMetrics {
        self.scheduler.metrics()
    }

    /// Remove completed task from tracking
    pub fn remove_completed_task(&self, task_id: TaskId) {
        let mut tasks = self.tasks.write().unwrap();
        if tasks.remove(&task_id).is_some() {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.active_count = metrics.active_count.saturating_sub(1);
            metrics.completed_count += 1;
        }
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

impl Clone for AsyncRuntimeMetrics {
    fn clone(&self) -> Self {
        Self {
            total_created: self.total_created,
            active_count: self.active_count,
            completed_count: self.completed_count,
            failed_count: self.failed_count,
            cancelled_count: self.cancelled_count,
        }
    }
} 