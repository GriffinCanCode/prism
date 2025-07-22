//! Prism Runtime System
//!
//! The Prism runtime provides a comprehensive execution environment for Prism programs,
//! featuring advanced concurrency, security, resource management, and AI integration.
//!
//! ## Key Features
//!
//! - **Capability-based Security**: Fine-grained permissions and isolation
//! - **Structured Concurrency**: Safe, composable parallel execution
//! - **Advanced Resource Management**: NUMA-aware allocation, quotas, and pooling
//! - **Effect System Integration**: Track and manage computational effects
//! - **AI-Enhanced Intelligence**: Runtime optimization and analysis
//! - **Cross-platform Support**: Works on major operating systems

#![warn(missing_docs)]
#![warn(unused_imports)]
#![warn(unused_variables)]
// Temporarily allow unsafe code for memory management
// #![forbid(unsafe_code)]

use std::sync::Arc;
use thiserror::Error;

// Core runtime modules
pub mod authority;
pub mod resources;
pub mod platform;
pub mod security;
pub mod concurrency;
pub mod intelligence;
// TODO: Re-enable when prism-ai dependency is available
// pub mod ai_integration;

// Public re-exports for common types
pub use authority::{
    Capability, CapabilitySet, CapabilityError, Operation, 
    ConstraintSet, ComponentId as AuthorityComponentId,
};

pub use concurrency::{
    ConcurrencySystem, ConcurrencyError,
    StructuredScope, ScopeHandle, 
    AsyncRuntime, TaskPriority,
};

pub use resources::{
    ResourceManager, ResourceError, ResourceTracker, 
    MemoryPool, PooledBuffer, QuotaManager,
    create_production_manager, create_development_manager,
};

// Resource management re-exports
pub mod resource_management {
    //! Comprehensive resource management utilities
    pub use crate::resources::tracker::*;
    pub use crate::resources::pools::*;
    pub use crate::resources::quotas::*;
    pub use crate::resources::effects::*;
    pub use crate::resources::memory::*;
}

/// Runtime system errors
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Authority/capability error
    #[error("Authority error: {0}")]
    Authority(#[from] authority::CapabilityError),
    
    /// Concurrency system error
    #[error("Concurrency error: {0}")]
    Concurrency(#[from] concurrency::ConcurrencyError),
    
    /// Resource management error
    #[error("Resource error: {0}")]
    Resource(#[from] resources::ResourceError),
    
    /// Security violation
    #[error("Security error: {0}")]
    Security(#[from] security::SecurityError),
    
    /// Platform error
    #[error("Platform error: {message}")]
    Platform { message: String },
    
    /// AI integration error
    #[error("AI error: {message}")]
    AI { message: String },
    
    /// Generic runtime error
    #[error("Runtime error: {message}")]
    Generic { message: String },
}

/// Trait for executable code that can be run by the runtime
pub trait Executable<T> {
    /// Execute the code with the given capabilities and context
    fn execute(&self, capabilities: &authority::CapabilitySet, context: &platform::execution::ExecutionContext) -> Result<T, RuntimeError>;
}
pub struct AIMetadataCollector;

impl AIMetadataCollector {
    pub fn new() -> Result<Self, RuntimeError> {
        Ok(Self)
    }
}

/// Main Prism runtime system
pub struct PrismRuntime {
    /// Authority system for capability management
    pub authority_system: Arc<authority::AuthoritySystem>,
    /// Concurrency management
    pub concurrency_system: Arc<concurrency::ConcurrencySystem>,
    /// Resource management
    pub resource_manager: Arc<resources::ResourceManager>,
    /// Security enforcement
    pub security_system: Arc<security::SecuritySystem>,
    /// AI integration
    pub ai_system: Option<Arc<intelligence::IntelligenceSystem>>,
}

impl PrismRuntime {
    /// Create a new runtime with default configuration
    pub fn new() -> Result<Self, RuntimeError> {
        let authority_system = Arc::new(authority::AuthoritySystem::new()?);
        let concurrency_system = Arc::new(concurrency::ConcurrencySystem::new()?);
        let resource_manager = Arc::new(resources::ResourceManager::new()?);
        let security_system = Arc::new(security::SecuritySystem::new()?);
        
        Ok(Self {
            authority_system,
            concurrency_system,
            resource_manager,
            security_system,
            ai_system: None,
        })
    }
    
    /// Create a production-ready runtime
    pub fn production() -> Result<Self, RuntimeError> {
        let authority_system = Arc::new(authority::AuthoritySystem::new()?);
        let concurrency_system = Arc::new(concurrency::ConcurrencySystem::new()?);
        let resource_manager = Arc::new(resources::create_production_manager()?);
        let security_system = Arc::new(security::SecuritySystem::new()?);
        
        Ok(Self {
            authority_system,
            concurrency_system,
            resource_manager,
            security_system,
            ai_system: None,
        })
    }
    
    /// Execute a future asynchronously
    pub async fn execute_async<F, T>(&self, future: F) -> Result<T, RuntimeError>
    where
        F: std::future::Future<Output = Result<T, RuntimeError>> + Send + 'static,
        T: Send + 'static,
    {
        // Convert RuntimeError to ConcurrencyError for the concurrency system
        let converted_future = async move {
            future.await.map_err(|e| concurrency::ConcurrencyError::Generic { 
                message: e.to_string() 
            })
        };
        
        self.concurrency_system.execute_async(converted_future).await
            .map_err(|e| RuntimeError::Concurrency(e))
    }
    
    /// Start the runtime system
    pub async fn start(&mut self) -> Result<(), RuntimeError> {
        // Start resource monitoring
        // Note: We'll need to implement this properly when we have the full system
        Ok(())
    }
    
    /// Shutdown the runtime system gracefully
    pub async fn shutdown(&self) -> Result<(), RuntimeError> {
        // Shutdown all subsystems
        self.concurrency_system.shutdown().await?;
        Ok(())
    }
}
