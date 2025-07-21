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
        let execution_manager = Arc::new(execution::ExecutionManager::new()?);

        Ok(Self {
            execution_manager,
        })
    }

    /// Execute code with monitoring across platforms
    pub fn execute_monitored<T>(
        &self,
        code: &dyn crate::Executable<T>,
        capabilities: &authority::CapabilitySet,
        context: &execution::ExecutionContext,
        resource_handle: &resources::ResourceHandle,
    ) -> Result<T, PlatformError> {
        self.execution_manager
            .execute_monitored(code, capabilities, context, &resource_handle.effect_handle)
            .map_err(PlatformError::Execution)
    }
}

/// Platform execution errors
#[derive(Debug, Error)]
pub enum PlatformError {
    /// Execution error
    #[error("Execution error: {0}")]
    Execution(#[from] execution::ExecutionError),

    /// Platform adaptation error
    #[error("Platform adaptation error: {message}")]
    Adaptation { message: String },

    /// Generic platform error
    #[error("Platform error: {message}")]
    Generic { message: String },
} 