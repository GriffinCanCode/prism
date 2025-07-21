//! Resource Management - Effects and Memory Coordination  
//!
//! This module implements the resource management system that coordinates memory allocation,
//! effect tracking, and resource consumption monitoring. It embodies the business capability
//! of **resource lifecycle management** across the entire runtime execution.
//!
//! ## Business Capability: Resource Management
//!
//! **Core Responsibility**: Manage computational resources (memory, effects) throughout their lifecycle.
//!
//! **Key Business Functions**:
//! - **Memory Management**: Allocate, track, and deallocate memory with semantic types
//! - **Effect Tracking**: Monitor and audit all computational effects
//! - **Resource Consumption**: Track resource usage patterns and limits
//! - **Performance Profiling**: Collect resource usage metrics for optimization
//! - **Resource Security**: Ensure resource access respects capability boundaries
//!
//! ## Conceptual Cohesion
//!
//! This module maintains high conceptual cohesion by focusing on **resource lifecycle management**.
//! It coordinates between memory and effects because they are intrinsically related:
//! - Memory allocations are effects that consume resources
//! - Effect execution requires memory and produces resource consumption
//! - Both require capability checking and audit trails
//! - Both contribute to performance profiles and AI metadata
//!
//! The module does NOT handle:
//! - Authority validation (handled by `authority` module)
//! - Platform-specific execution (handled by `platform` module)
//! - Security policy enforcement (handled by `security` module)
//! - Business intelligence analysis (handled by `intelligence` module)

use crate::authority;
use crate::platform::execution::ExecutionContext;
use std::sync::Arc;
use thiserror::Error;

pub mod effects;
pub mod memory;

/// Unified resource manager that coordinates memory and effects
#[derive(Debug)]
pub struct ResourceManager {
    /// Effect tracking system
    effect_tracker: Arc<effects::EffectTracker>,
    
    /// Memory management system
    memory_manager: Arc<memory::MemoryManager>,
    
    /// Resource usage coordinator
    usage_coordinator: Arc<ResourceUsageCoordinator>,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Result<Self, ResourceError> {
        let effect_tracker = Arc::new(effects::EffectTracker::new()?);
        let memory_manager = Arc::new(memory::MemoryManager::new()?);
        let usage_coordinator = Arc::new(ResourceUsageCoordinator::new());

        Ok(Self {
            effect_tracker,
            memory_manager,
            usage_coordinator,
        })
    }

    /// Begin tracking resource usage for an execution
    pub fn begin_execution(&self, context: &ExecutionContext) -> Result<ResourceHandle, ResourceError> {
        // Begin effect tracking
        let effect_handle = self.effect_tracker.begin_execution(context)?;
        
        // Create unified resource handle
        let resource_handle = ResourceHandle::new(effect_handle, context.execution_id);
        
        // Register with usage coordinator
        self.usage_coordinator.register_execution(&resource_handle, context);
        
        Ok(resource_handle)
    }

    /// End resource tracking for an execution
    pub fn end_execution<T>(&self, handle: ResourceHandle, result: &T) -> Result<(), ResourceError> {
        // End effect tracking
        self.effect_tracker.end_execution(handle.effect_handle, result)?;
        
        // Update usage coordinator
        self.usage_coordinator.complete_execution(&handle);
        
        Ok(())
    }

    /// Get current number of active resource tracking sessions
    pub fn active_count(&self) -> usize {
        self.effect_tracker.active_count()
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        self.memory_manager.current_usage()
    }

    /// Allocate semantic memory with capability checking
    pub fn allocate_semantic<T: memory::SemanticType>(
        &self,
        semantic_type: &T,
        size: usize,
        capability: &authority::Capability,
        context: &ExecutionContext,
    ) -> Result<memory::SemanticPtr<T>, ResourceError> {
        self.memory_manager
            .allocate_semantic(semantic_type, size, capability, context)
            .map_err(ResourceError::Memory)
    }

    /// Deallocate semantic memory
    pub fn deallocate_semantic<T: memory::SemanticType>(
        &self,
        ptr: memory::SemanticPtr<T>,
        context: &ExecutionContext,
    ) -> Result<(), ResourceError> {
        self.memory_manager
            .deallocate_semantic(ptr, context)
            .map_err(ResourceError::Memory)
    }
}

/// Handle for tracking resources across an execution
#[derive(Debug)]
pub struct ResourceHandle {
    /// Effect tracking handle
    pub effect_handle: effects::EffectHandle,
    /// Execution ID for correlation
    pub execution_id: crate::platform::execution::ExecutionId,
    /// Resource tracking start time
    pub started_at: std::time::Instant,
}

impl ResourceHandle {
    fn new(effect_handle: effects::EffectHandle, execution_id: crate::platform::execution::ExecutionId) -> Self {
        Self {
            effect_handle,
            execution_id,
            started_at: std::time::Instant::now(),
        }
    }
}

/// Coordinator for resource usage patterns and optimization
#[derive(Debug)]
struct ResourceUsageCoordinator {
    // Implementation would track resource usage patterns
}

impl ResourceUsageCoordinator {
    fn new() -> Self {
        Self {}
    }

    fn register_execution(&self, _handle: &ResourceHandle, _context: &ExecutionContext) {
        // Track resource usage start
    }

    fn complete_execution(&self, _handle: &ResourceHandle) {
        // Track resource usage completion
    }
}

/// Resource management errors
#[derive(Debug, Error)]
pub enum ResourceError {
    /// Effect tracking error
    #[error("Effect error: {0}")]
    Effect(#[from] effects::EffectError),

    /// Memory management error  
    #[error("Memory error: {0}")]
    Memory(#[from] memory::MemoryError),

    /// Resource coordination error
    #[error("Resource coordination error: {message}")]
    Coordination { message: String },

    /// Generic resource error
    #[error("Resource error: {message}")]
    Generic { message: String },
} 