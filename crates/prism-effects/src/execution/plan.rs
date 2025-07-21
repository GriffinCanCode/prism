//! Execution Plan
//!
//! Planning and orchestration of effect execution with capability validation

use crate::effects::Effect;
use crate::capability::Capability;
use crate::security::SecureExecutionContext;
use super::handlers::EffectResult;
use prism_common::span::Span;
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Execution plan for a set of effects
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Ordered steps for execution
    pub steps: Vec<ExecutionStep>,
    /// Required capabilities for the entire plan
    pub required_capabilities: Vec<String>,
    /// Execution metadata
    pub metadata: ExecutionMetadata,
    /// Security constraints
    pub security_constraints: Vec<SecurityConstraint>,
}

impl ExecutionPlan {
    /// Create a new execution plan
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            required_capabilities: Vec::new(),
            metadata: ExecutionMetadata::default(),
            security_constraints: Vec::new(),
        }
    }

    /// Add an execution step
    pub fn add_step(&mut self, step: ExecutionStep) {
        // Merge required capabilities
        for cap in &step.required_capabilities {
            if !self.required_capabilities.contains(cap) {
                self.required_capabilities.push(cap.clone());
            }
        }
        
        self.steps.push(step);
        self.metadata.total_steps += 1;
    }

    /// Validate the execution plan
    pub fn validate(&self, available_capabilities: &[Capability]) -> Result<(), ExecutionError> {
        // Check capability requirements
        let available_names: Vec<String> = available_capabilities
            .iter()
            .map(|c| c.definition.clone())
            .collect();

        for required in &self.required_capabilities {
            if !available_names.contains(required) {
                return Err(ExecutionError::MissingCapability(required.clone()));
            }
        }

        // Validate step dependencies
        for (i, step) in self.steps.iter().enumerate() {
            for dep in &step.dependencies {
                if *dep >= i {
                    return Err(ExecutionError::InvalidDependency(
                        format!("Step {} depends on future step {}", i, dep)
                    ));
                }
            }
        }

        // Check security constraints
        for constraint in &self.security_constraints {
            constraint.validate(&self.steps)?;
        }

        Ok(())
    }

    /// Get execution order considering dependencies
    pub fn get_execution_order(&self) -> Result<Vec<usize>, ExecutionError> {
        let mut order = Vec::new();
        let mut completed = vec![false; self.steps.len()];
        let mut in_progress = vec![false; self.steps.len()];

        while order.len() < self.steps.len() {
            let mut made_progress = false;

            for (i, step) in self.steps.iter().enumerate() {
                if completed[i] || in_progress[i] {
                    continue;
                }

                // Check if all dependencies are satisfied
                let can_execute = step.dependencies.iter().all(|&dep| completed[dep]);

                if can_execute {
                    in_progress[i] = true;
                    order.push(i);
                    completed[i] = true;
                    made_progress = true;
                }
            }

            if !made_progress {
                return Err(ExecutionError::CircularDependency);
            }
        }

        Ok(order)
    }

    /// Estimate execution time
    pub fn estimate_duration(&self) -> std::time::Duration {
        self.steps
            .iter()
            .map(|step| step.estimated_duration)
            .sum()
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// A single execution step in the plan
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// The effect to execute
    pub effect: Effect,
    /// Handler name for this effect
    pub handler_name: String,
    /// Required capabilities for this step
    pub required_capabilities: Vec<String>,
    /// Dependencies (indices of other steps that must complete first)
    pub dependencies: Vec<usize>,
    /// Estimated duration for this step
    pub estimated_duration: std::time::Duration,
    /// Step metadata
    pub metadata: StepMetadata,
}

impl ExecutionStep {
    /// Create a new execution step
    pub fn new(effect: Effect, handler_name: String) -> Self {
        Self {
            effect,
            handler_name,
            required_capabilities: Vec::new(),
            dependencies: Vec::new(),
            estimated_duration: std::time::Duration::from_millis(100), // Default estimate
            metadata: StepMetadata::default(),
        }
    }

    /// Add a capability requirement
    pub fn require_capability(&mut self, capability: String) {
        if !self.required_capabilities.contains(&capability) {
            self.required_capabilities.push(capability);
        }
    }

    /// Add a dependency on another step
    pub fn add_dependency(&mut self, step_index: usize) {
        if !self.dependencies.contains(&step_index) {
            self.dependencies.push(step_index);
        }
    }
}

/// Metadata for the execution plan
#[derive(Debug)]
pub struct ExecutionMetadata {
    /// Total number of steps
    pub total_steps: usize,
    /// When the plan was created
    pub created_at: std::time::SystemTime,
    /// Plan complexity score
    pub complexity_score: f64,
}

impl Default for ExecutionMetadata {
    fn default() -> Self {
        Self {
            total_steps: 0,
            created_at: std::time::SystemTime::now(),
            complexity_score: 0.0,
        }
    }
}

/// Metadata for individual execution steps
#[derive(Debug, Default, Clone)]
pub struct StepMetadata {
    /// Step priority (higher = more important)
    pub priority: u8,
    /// Whether this step can be retried on failure
    pub retryable: bool,
    /// Maximum retry attempts
    pub max_retries: u8,
}

/// Security constraints for execution
#[derive(Debug)]
pub enum SecurityConstraint {
    /// No two effects from specified categories can run simultaneously
    MutualExclusion(Vec<String>),
    /// Effect must run in isolated context
    RequireIsolation(String),
    /// Effect requires specific trust level
    RequireTrustLevel(String, u8),
}

impl SecurityConstraint {
    /// Validate constraint against execution steps
    pub fn validate(&self, steps: &[ExecutionStep]) -> Result<(), ExecutionError> {
        match self {
            SecurityConstraint::MutualExclusion(categories) => {
                // Check that no two effects from these categories can run in parallel
                // This is a simplified check - full implementation would consider actual parallelism
                for category in categories {
                    let count = steps
                        .iter()
                        .filter(|step| step.effect.definition.contains(category))
                        .count();
                    if count > 1 {
                        return Err(ExecutionError::SecurityViolation(
                            format!("Multiple effects from category '{}' would violate mutual exclusion", category)
                        ));
                    }
                }
            },
            SecurityConstraint::RequireIsolation(effect_name) => {
                // Verify isolated execution for specific effect
                if steps.iter().any(|step| &step.effect.definition == effect_name) {
                    // In a full implementation, we'd check isolation capabilities
                    // For now, just verify the constraint is acknowledged
                }
            },
            SecurityConstraint::RequireTrustLevel(effect_name, min_level) => {
                // Verify trust level requirements
                for step in steps {
                    if &step.effect.definition == effect_name {
                        // In full implementation, check against actual trust levels
                        if *min_level > 5 { // Arbitrary high trust threshold
                            return Err(ExecutionError::SecurityViolation(
                                format!("Effect '{}' requires trust level {} which exceeds maximum", effect_name, min_level)
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Errors that can occur during execution planning
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Missing required capability: {0}")]
    MissingCapability(String),
    
    #[error("Invalid dependency: {0}")]
    InvalidDependency(String),
    
    #[error("Circular dependency detected in execution plan")]
    CircularDependency,
    
    #[error("Security violation: {0}")]
    SecurityViolation(String),
    
    #[error("Execution step failed: {0}")]
    StepFailed(String),
} 