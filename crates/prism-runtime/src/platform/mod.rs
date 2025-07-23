//! Platform Abstraction - Multi-Target Execution System
//!
//! This module implements the platform abstraction layer that enables Prism code to execute
//! across multiple targets (TypeScript, WebAssembly, Native) while maintaining consistent
//! security and performance characteristics. It embodies the business capability of
//! **cross-platform execution orchestration**.
//!
//! ## Business Capability: Platform Abstraction
//!
//! **Core Responsibility**: Execute Prism code consistently across different target platforms.
//!
//! **Key Business Functions**:
//! - **Target Adaptation**: Adapt execution to platform-specific requirements
//! - **Execution Orchestration**: Coordinate code execution across targets
//! - **Performance Optimization**: Apply target-specific optimizations
//! - **Context Management**: Maintain execution context across platform boundaries
//! - **Capability Integration**: Ensure capability checking works on all targets
//!
//! ## Conceptual Cohesion
//!
//! This module maintains high conceptual cohesion by focusing solely on **platform execution**.
//! It handles the complexity of multi-target execution while presenting a unified interface.
//!
//! The module does NOT handle:
//! - Authority management (handled by `authority` module)
//! - Resource tracking (handled by `resources` module)
//! - Security policy enforcement (handled by `security` module)
//! - Business intelligence (handled by `intelligence` module)

use crate::{authority, resources};
use std::sync::Arc;
use thiserror::Error;

pub mod execution;

/// Platform manager that orchestrates multi-target execution
#[derive(Debug)]
pub struct PlatformManager {
    /// Execution manager for coordinating target execution
    execution_manager: Arc<execution::ExecutionManager>,
}

impl PlatformManager {
    /// Create a new platform manager
    pub fn new() -> Result<Self, PlatformError> {
        let execution_manager = Arc::new(execution::ExecutionManager::new()
            .map_err(|e| PlatformError::ExecutionError(e))?);

        Ok(Self {
            execution_manager,
        })
    }

    /// Execute code with monitoring and capability checking across targets
    pub fn execute_monitored<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &authority::CapabilitySet,
        context: &execution::ExecutionContext,
        resource_handle: &resources::ResourceHandle,
    ) -> Result<T, PlatformError> {
        self.execution_manager
            .execute_monitored(code, capabilities, context, resource_handle)
            .map_err(PlatformError::ExecutionError)
    }

    /// Get execution statistics from the platform
    pub fn get_execution_stats(&self) -> PlatformStats {
        // In a real implementation, this would aggregate stats from execution manager
        PlatformStats {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_execution_time: std::time::Duration::ZERO,
        }
    }

    /// Check if a target is supported
    pub fn is_target_supported(&self, target: &execution::ExecutionTarget) -> bool {
        match target {
            execution::ExecutionTarget::TypeScript |
            execution::ExecutionTarget::WebAssembly |
            execution::ExecutionTarget::Native |
            execution::ExecutionTarget::PrismVM => true,
        }
    }
}

/// Platform statistics
#[derive(Debug, Clone)]
pub struct PlatformStats {
    /// Total executions across all targets
    pub total_executions: u64,
    /// Successful executions
    pub successful_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time
    pub average_execution_time: std::time::Duration,
}

/// Platform-related errors
#[derive(Debug, Error)]
pub enum PlatformError {
    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(#[from] execution::ExecutionError),

    /// Authority error
    #[error("Authority error: {0}")]
    Authority(#[from] authority::CapabilityError),

    /// Resource error
    #[error("Resource error: {0}")]
    Resource(#[from] resources::ResourceError),

    /// Generic platform error
    #[error("Platform error: {message}")]
    Generic { message: String },
} 