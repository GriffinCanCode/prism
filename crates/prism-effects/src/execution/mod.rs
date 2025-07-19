//! Execution System
//!
//! Effect execution and handler management with integrated capability validation

pub mod handlers;
pub mod registry;
pub mod plan;

use crate::effects::Effect;
use crate::security::{SecureExecutionContext, Capability};
use prism_common::span::Span;
use std::collections::HashMap;
use thiserror::Error;

// Re-exports
pub use handlers::{EffectHandler, EffectResult, EffectHandlerError};
pub use registry::{HandlerRegistry, RegistryStats};
pub use plan::{ExecutionPlan, ExecutionStep, ExecutionError};

/// Complete execution system for effects
#[derive(Debug)]
pub struct ExecutionSystem {
    /// Handler registry
    pub registry: HandlerRegistry,
    /// Execution configuration
    pub config: ExecutionConfig,
    /// Active execution contexts
    pub active_contexts: HashMap<String, ExecutionContext>,
}

impl ExecutionSystem {
    /// Create new execution system
    pub fn new() -> Self {
        Self {
            registry: HandlerRegistry::new(),
            config: ExecutionConfig::default(),
            active_contexts: HashMap::new(),
        }
    }

    /// Execute an execution plan
    pub fn execute_plan(
        &mut self,
        plan: &ExecutionPlan,
        context: &SecureExecutionContext,
    ) -> Result<Vec<EffectResult>, ExecutionError> {
        let mut results = Vec::new();
        
        // Get execution order
        let execution_order = plan.get_execution_order()?;
        
        // Execute each step in order
        for &step_index in &execution_order {
            let step = &plan.steps[step_index];
            let result = self.execute_step(step, context)?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Execute a single step
    fn execute_step(
        &self,
        step: &ExecutionStep,
        context: &SecureExecutionContext,
    ) -> Result<EffectResult, ExecutionError> {
        // Get handler for this step
        let handler = self.registry.get_handler(&step.handler_name)
            .ok_or_else(|| ExecutionError::StepFailed(
                format!("Handler '{}' not found", step.handler_name)
            ))?;

        // Check capabilities
        if !self.registry.check_capabilities(&step.effect.definition, &context.available_capabilities) {
            return Err(ExecutionError::MissingCapability(
                format!("Missing capabilities for effect '{}'", step.effect.definition)
            ));
        }

        // Create effect context
        let mut effect_context = handlers::EffectContext::new(step.effect.span);
        for cap in &context.available_capabilities {
            effect_context.add_capability(cap.clone());
        }

        // Execute the effect
        handler.handle_effect(&step.effect, &context.available_capabilities, &mut effect_context)
            .map_err(|e| ExecutionError::StepFailed(e.to_string()))
    }

    /// Get handler count
    pub fn get_handler_count(&self) -> usize {
        self.registry.stats().total_handlers
    }

    /// Register a new handler
    pub fn register_handler(&mut self, handler: std::sync::Arc<dyn EffectHandler>) -> Result<(), ExecutionError> {
        let effect_name = format!("{}Effect", handler.name());
        self.registry.register_handler(effect_name, handler)
            .map_err(|e| ExecutionError::StepFailed(e.to_string()))
    }
}

impl Default for ExecutionSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum parallel executions
    pub max_parallel: usize,
    /// Timeout for individual steps
    pub step_timeout: std::time::Duration,
    /// Whether to continue on errors
    pub continue_on_error: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_parallel: 4,
            step_timeout: std::time::Duration::from_secs(30),
            continue_on_error: false,
        }
    }
}

/// Execution context for a specific execution session
#[derive(Debug)]
pub struct ExecutionContext {
    /// Context identifier
    pub id: String,
    /// Start time
    pub start_time: std::time::Instant,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

impl ExecutionContext {
    /// Create new execution context
    pub fn new(id: String) -> Self {
        Self {
            id,
            start_time: std::time::Instant::now(),
            metadata: HashMap::new(),
        }
    }
} 