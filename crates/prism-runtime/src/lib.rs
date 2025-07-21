//! Prism Runtime System - AI-First Capability-Based Execution
//!
//! This crate implements the Prism Runtime System as specified in PLD-005, providing
//! capability-based security, effect tracking, and zero-trust execution for the 
//! AI-first programming language Prism.
//!
//! ## Architecture Overview
//!
//! The runtime implements a four-layer security architecture:
//! - **Application Layer**: Business logic and AI-generated code
//! - **Capability Enforcement Layer**: Capability checking and effect tracking
//! - **Multi-Target Execution Layer**: Runtime adapters and memory management
//! - **Hardware Abstraction Layer**: Platform interface and secure enclaves
//!
//! ## Core Principles
//!
//! 1. **Security by Default**: Every operation requires explicit capability authorization
//! 2. **Effect Transparency**: All computational effects are explicit and auditable
//! 3. **AI-First Observability**: Structured metadata for AI analysis and debugging
//! 4. **Multi-Target Support**: Execute across TypeScript, WebAssembly, and native targets
//! 5. **Composable Security**: Security properties compose across component boundaries
//!
//! ## Modular Architecture - Conceptual Cohesion
//!
//! The runtime is organized into business capability domains for maximum conceptual cohesion:
//!
//! - **Authority Management** (`authority/`) - Capability-based security and permissions
//! - **Resource Management** (`resources/`) - Memory, effects, and resource tracking  
//! - **Platform Abstraction** (`platform/`) - Multi-target execution and adaptation
//! - **Security Enforcement** (`security/`) - Policy enforcement and threat detection
//! - **Intelligence & Analytics** (`intelligence/`) - AI metadata and business insights
//!
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Business capability modules organized by conceptual cohesion
pub mod authority;
pub mod resources; 
pub mod platform;
pub mod security;
pub mod intelligence;

// Legacy module re-exports for backward compatibility
// TODO: Remove these after migration is complete
pub mod capability {
    //! Legacy re-export - use `authority` module instead
    pub use crate::authority::*;
}
pub mod effects {
    //! Legacy re-export - use `resources::effects` module instead  
    pub use crate::resources::effects::*;
}
pub mod execution {
    //! Legacy re-export - use `platform::execution` module instead
    pub use crate::platform::execution::*;
}
pub mod memory {
    //! Legacy re-export - use `resources::memory` module instead
    pub use crate::resources::memory::*;
}
pub mod isolation {
    //! Legacy re-export - use `security::isolation` module instead
    pub use crate::security::isolation::*;
}
pub mod ai_metadata {
    //! Legacy re-export - use `intelligence::metadata` module instead
    pub use crate::intelligence::metadata::*;
}

// Public API re-exports organized by business capability
pub use authority::{Capability, CapabilityManager, CapabilityError};
pub use resources::effects::{EffectTracker, EffectHandle, EffectResult, EffectError};
pub use platform::execution::{ExecutionContext, ExecutionTarget, ExecutionError};
pub use resources::memory::{MemoryManager, MemoryError, SemanticPtr};
pub use intelligence::metadata::{AIMetadataCollector, RuntimeMetadata, AIRuntimeContext};
pub use security::isolation::{ComponentHandle, ComponentId, IsolationError};
pub use security::enforcement::{SecurityPolicy, SecurityError, PolicyDecision};

use prism_common::{Result as PrismResult, PrismError};
use prism_effects::{Effect, EffectSystem};
use std::sync::Arc;
use thiserror::Error;

/// Main Prism Runtime System
/// 
/// The central runtime that coordinates all aspects of secure, capability-based
/// execution across multiple targets while maintaining AI-comprehensible state.
#[derive(Debug)]
pub struct PrismRuntime {
    /// Authority management system
    authority_manager: Arc<authority::CapabilityManager>,
    
    /// Resource tracking and management
    resource_manager: Arc<resources::ResourceManager>,
    
    /// Multi-target platform execution
    platform_manager: Arc<platform::PlatformManager>,
    
    /// Security policy enforcement
    security_enforcer: Arc<security::SecurityEnforcer>,
    
    /// Intelligence and analytics collection
    intelligence_collector: Arc<intelligence::IntelligenceCollector>,
}

impl PrismRuntime {
    /// Create a new Prism Runtime instance
    pub fn new() -> Result<Self, RuntimeError> {
        let authority_manager = Arc::new(authority::CapabilityManager::new()?);
        let resource_manager = Arc::new(resources::ResourceManager::new()?);
        let platform_manager = Arc::new(platform::PlatformManager::new()?);
        let security_enforcer = Arc::new(security::SecurityEnforcer::new()?);
        let intelligence_collector = Arc::new(intelligence::IntelligenceCollector::new()?);

        Ok(Self {
            authority_manager,
            resource_manager,
            platform_manager,
            security_enforcer,
            intelligence_collector,
        })
    }

    /// Execute code with full capability checking and effect tracking
    pub fn execute_with_capabilities<T>(
        &self,
        code: &dyn Executable<T>,
        capabilities: &authority::CapabilitySet,
        context: &platform::execution::ExecutionContext,
    ) -> Result<T, RuntimeError> {
        // Begin resource tracking
        let resource_handle = self.resource_manager.begin_execution(context)?;
        
        // Validate capabilities
        self.authority_manager.validate_capabilities(capabilities, context)?;
        
        // Execute with monitoring
        let result = self.platform_manager.execute_monitored(
            code,
            capabilities,
            context,
            &resource_handle,
        )?;
        
        // Complete resource tracking
        self.resource_manager.end_execution(resource_handle, &result)?;
        
        // Record intelligence metadata
        self.intelligence_collector.record_execution(&result, context)?;
        
        Ok(result)
    }

    /// Get current runtime statistics for monitoring
    pub fn get_runtime_stats(&self) -> RuntimeStats {
        RuntimeStats {
            active_capabilities: self.authority_manager.active_count(),
            tracked_resources: self.resource_manager.active_count(),
            memory_usage: self.resource_manager.current_memory_usage(),
            isolated_components: self.security_enforcer.component_count(),
            security_violations: self.security_enforcer.violation_count(),
        }
    }
}

impl Default for PrismRuntime {
    fn default() -> Self {
        Self::new().expect("Failed to create default runtime")
    }
}

/// Trait for executable code that can run in the Prism runtime
pub trait Executable<T> {
    /// Execute the code with the given capabilities and context
    fn execute(
        &self,
        capabilities: &authority::CapabilitySet,
        context: &platform::execution::ExecutionContext,
    ) -> Result<T, RuntimeError>;
    
    /// Get the effects this code declares
    fn declared_effects(&self) -> Vec<Effect>;
    
    /// Get the capabilities this code requires
    fn required_capabilities(&self) -> authority::CapabilitySet;
}

/// Runtime statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Number of active capabilities
    pub active_capabilities: usize,
    /// Number of resources currently being tracked
    pub tracked_resources: usize,
    /// Current memory usage in bytes
    pub memory_usage: usize,
    /// Number of isolated components
    pub isolated_components: usize,
    /// Number of security violations detected
    pub security_violations: usize,
}

/// Main runtime error type
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Authority management error
    #[error("Authority error: {0}")]
    Authority(#[from] authority::CapabilityError),
    
    /// Resource management error
    #[error("Resource error: {0}")]
    Resource(#[from] resources::ResourceError),
    
    /// Platform execution error
    #[error("Platform error: {0}")]
    Platform(#[from] platform::PlatformError),
    
    /// Security enforcement error
    #[error("Security error: {0}")]
    Security(#[from] security::SecurityError),
    
    /// Intelligence collection error
    #[error("Intelligence error: {0}")]
    Intelligence(#[from] intelligence::IntelligenceError),
    
    /// Generic runtime error
    #[error("Runtime error: {message}")]
    Generic {
        /// Error message
        message: String,
    },
}

/// Runtime result type
pub type RuntimeResult<T> = Result<T, RuntimeError>;
