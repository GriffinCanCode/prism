//! Core Async Types - Fundamental Types for Async Runtime
//!
//! This module contains the core types used throughout the async runtime system:
//! - Task identifiers and metadata
//! - Priority levels and scheduling information
//! - Error types and result handling
//! - Handle types for async operations
//!
//! These types are designed to be lightweight, serializable, and AI-comprehensible.

use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::{authority, resources};
use super::cancellation::CancellationToken;

/// Unique identifier for async tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(Uuid);

impl TaskId {
    /// Generate a new unique task ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Task priority levels for scheduling
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

/// Metadata about an async task for AI analysis
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
    /// Create a new async handle
    pub fn new(
        id: TaskId,
        join_handle: JoinHandle<AsyncResult<T>>,
        cancellation_token: CancellationToken,
        metadata: AsyncTaskMetadata,
    ) -> Self {
        Self {
            id,
            join_handle,
            cancellation_token,
            metadata,
        }
    }

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

    /// Get cancellation token (for internal use)
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation_token
    }
}

/// Result type for async operations
pub type AsyncResult<T> = Result<T, AsyncError>;

/// Async runtime errors
#[derive(Debug, Clone, Error)]
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