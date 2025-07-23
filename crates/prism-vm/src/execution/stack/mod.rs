//! Stack Management System
//!
//! This module extends the existing VM stack implementation with  features
//! that integrate with the broader Prism ecosystem while maintaining conceptual cohesion.
//!
//! ## Design Principles
//!
//! 1. **Extends, Not Replaces**: Builds upon existing `ExecutionStack` functionality
//! 2. **Runtime Integration**: Leverages `prism-runtime` resource management
//! 3. **Capability-Aware**: All operations respect capability-based security
//! 4. **AI-First**: Generates structured metadata for AI analysis
//! 5. **Performance Optimized**: Uses existing memory pools and NUMA awareness
//! 6. **No Logic Duplication**: Interfaces with existing systems
//!
//! ## Module Structure
//!
//! - [`memory`] - memory management integration
//! - [`security`] - Capability-aware stack operations
//! - [`analytics`] - Performance monitoring and profiling
//! - [`jit_integration`] - JIT compiler stack frame management
//! - [`ai_metadata`] - AI-comprehensible stack analysis

pub mod memory;
pub mod security;
pub mod analytics;
pub mod jit_integration;
pub mod ai_metadata;

// Re-export the existing stack types for convenience
pub use crate::execution::{ExecutionStack, StackFrame, StackValue};

use crate::{VMResult, PrismVMError};
use prism_runtime::{
    resources::{ResourceManager, MemoryPool, PooledBuffer},
    authority::capability::CapabilitySet,
};
use std::sync::Arc;
use tracing::{debug, info, span, Level};

/// Stack manager that integrates features
#[derive(Debug)]
pub struct AdvancedStackManager {
    /// Core execution stack
    execution_stack: ExecutionStack,
    
    /// Memory management integration
    memory_manager: memory::StackMemoryManager,
    
    /// Security integration
    security_manager: security::StackSecurityManager,
    
    /// Performance analytics
    analytics: analytics::StackAnalytics,
    
    /// JIT integration
    jit_integration: jit_integration::JITStackManager,
    
    /// AI metadata generator
    ai_metadata: ai_metadata::StackAIMetadata,
}

impl AdvancedStackManager {
    /// Create a new advanced stack manager
    pub fn new(
        resource_manager: Arc<ResourceManager>,
        capabilities: CapabilitySet,
    ) -> VMResult<Self> {
        let _span = span!(Level::INFO, "advanced_stack_init").entered();
        info!("Initializing advanced stack management system");

        let execution_stack = ExecutionStack::new();
        
        let memory_manager = memory::StackMemoryManager::new(
            Arc::clone(&resource_manager)
        )?;
        
        let security_manager = security::StackSecurityManager::new(capabilities)?;
        
        let analytics = analytics::StackAnalytics::new()?;
        
        let jit_integration = jit_integration::JITStackManager::new()?;
        
        let ai_metadata = ai_metadata::StackAIMetadata::new()?;

        Ok(Self {
            execution_stack,
            memory_manager,
            security_manager,
            analytics,
            jit_integration,
            ai_metadata,
        })
    }

    /// Get the underlying execution stack
    pub fn execution_stack(&self) -> &ExecutionStack {
        &self.execution_stack
    }

    /// Get mutable access to the execution stack
    pub fn execution_stack_mut(&mut self) -> &mut ExecutionStack {
        &mut self.execution_stack
    }

    /// Get memory management statistics
    pub fn memory_stats(&self) -> memory::StackMemoryStats {
        self.memory_manager.stats()
    }

    /// Get security analysis
    pub fn security_analysis(&self) -> security::StackSecurityAnalysis {
        self.security_manager.analyze(&self.execution_stack)
    }

    /// Get performance analytics
    pub fn performance_analytics(&self) -> analytics::StackPerformanceMetrics {
        self.analytics.current_metrics()
    }

    /// Generate AI metadata
    pub fn generate_ai_metadata(&self) -> ai_metadata::StackAIContext {
        self.ai_metadata.generate_context(&self.execution_stack)
    }
} 