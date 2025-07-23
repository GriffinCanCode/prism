//! Execution Error Types
//!
//! This module defines all error types related to code execution across
//! different target platforms.

use super::context::ExecutionTarget;
use thiserror::Error;

/// Execution-related errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// Unsupported target
    #[error("Unsupported execution target: {target:?}")]
    UnsupportedTarget {
        /// Target that is not supported
        target: ExecutionTarget,
    },

    /// Insufficient capabilities for execution
    #[error("Insufficient capabilities - required: {required:?}, available: {available:?}")]
    InsufficientCapabilities {
        /// Required capabilities
        required: Vec<String>,
        /// Available capabilities
        available: Vec<String>,
    },

    /// Monitoring failed
    #[error("Execution monitoring failed: {reason}")]
    MonitoringFailed {
        /// Reason for failure
        reason: String,
    },

    /// Capability error during execution
    #[error("Capability error: {0}")]
    Capability(#[from] crate::authority::CapabilityError),

    /// Resource error during execution
    #[error("Resource error: {0}")]
    Resource(#[from] crate::resources::ResourceError),

    /// Adapter initialization failed
    #[error("Adapter initialization failed: {message}")]
    AdapterInitializationFailed {
        /// Error message
        message: String,
    },

    /// Execution timeout
    #[error("Execution timed out after {timeout_secs} seconds")]
    ExecutionTimeout {
        /// Timeout in seconds
        timeout_secs: u64,
    },

    /// Memory limit exceeded
    #[error("Memory limit exceeded: used {used_bytes} bytes, limit {limit_bytes} bytes")]
    MemoryLimitExceeded {
        /// Used memory in bytes
        used_bytes: usize,
        /// Memory limit in bytes
        limit_bytes: usize,
    },

    /// Generic execution error
    #[error("Execution error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
}

/// Result type for execution operations
pub type ExecutionResult<T> = Result<T, ExecutionError>; 