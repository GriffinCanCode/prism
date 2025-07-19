//! Effect System & Capabilities for the Prism programming language
//!
//! This crate implements Prism's capability-based security model and effect system
//! as specified in PLD-003. It provides fine-grained control over computational 
//! effects while enabling secure, composable computation in an AI-first world.
//!
//! ## AI Integration Philosophy
//!
//! **Important**: This crate does NOT execute AI models or perform ML inference directly.
//! Instead, it follows Prism's external AI integration model:
//!
//! - **AI tools interact with Prism through external servers/APIs**
//! - **Prism generates AI-comprehensible metadata** for external consumption
//! - **Static analysis creates structured data** that AI systems can understand
//! - **No embedded ML models** exist within the language runtime
//!
//! This approach maintains security, predictability, and allows users to choose
//! their preferred AI services while keeping the language runtime lean and secure.

// Core modules organized by business capability
pub mod effects;
pub mod execution;
pub mod security;
pub mod validation;
pub mod ai;

// Legacy compatibility - re-export from capability.rs and security_trust.rs until fully migrated
pub mod capability;
pub mod security_trust;

// Re-exports for public API
pub use effects::{Effect, EffectDefinition, EffectCategory, EffectRegistry};
pub use execution::{EffectHandler, EffectResult, ExecutionSystem, ExecutionPlan};
pub use security::{SecuritySystem, Capability, SecurityLevel, InformationFlowControl};
pub use validation::{EffectValidator, ValidationContext, ValidationViolation};

use prism_common::span::Span;
use std::collections::HashMap;
use thiserror::Error;

/// Complete Prism Effects System
#[derive(Debug)]
pub struct PrismEffectsSystem {
    /// Effect management
    pub effects: effects::EffectSystem,
    /// Execution system
    pub execution: execution::ExecutionSystem,
    /// Security system
    pub security: security::SecuritySystem,
    /// Validation system
    pub validation: validation::EffectValidator,
}

impl PrismEffectsSystem {
    /// Create a new Prism Effects System
    pub fn new() -> Self {
        Self {
            effects: effects::EffectSystem::new(),
            execution: execution::ExecutionSystem::new(),
            security: security::SecuritySystem::new(),
            validation: validation::EffectValidator::new(),
        }
    }

    /// Process effects with full validation and security
    pub fn process_effects(
        &mut self,
        effects: Vec<Effect>,
        context: security::SecureExecutionContext,
    ) -> Result<Vec<EffectResult>, EffectSystemError> {
        // Validate effects
        self.validation.validate_effects(
            &effects,
            &context.available_capabilities,
            &capability::CapabilityManager::new(),
        )?;

        // Create execution plan
        let mut plan = execution::ExecutionPlan::new();
        for effect in effects {
            let step = execution::ExecutionStep::new(effect, "default".to_string());
            plan.add_step(step);
        }

        // Validate execution plan
        plan.validate(&context.available_capabilities)?;

        // Execute plan
        Ok(self.execution.execute_plan(&plan, &context)?)
    }

    /// Get system statistics
    pub fn get_stats(&self) -> SystemStats {
        SystemStats {
            effects_registered: self.effects.get_effect_count(),
            handlers_registered: self.execution.get_handler_count(),
            security_validations: self.security.get_stats().total_validations,
            validation_rules: self.validation.validation_rules.len(),
        }
    }
}

impl Default for PrismEffectsSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// System statistics
#[derive(Debug)]
pub struct SystemStats {
    /// Number of registered effects
    pub effects_registered: usize,
    /// Number of registered handlers
    pub handlers_registered: usize,
    /// Number of security validations performed
    pub security_validations: usize,
    /// Number of validation rules active
    pub validation_rules: usize,
}

/// Errors that can occur in the effect system
#[derive(Debug, Error)]
pub enum EffectSystemError {
    #[error("Effect already registered: {name}")]
    EffectAlreadyRegistered { name: String },

    #[error("Effect not found: {name}")]
    EffectNotFound { name: String },

    #[error("Handler registration failed: {reason}")]
    HandlerRegistrationFailed { reason: String },

    #[error("Effect validation failed: {reason}")]
    EffectValidationFailed { reason: String },

    #[error("Execution failed: {reason}")]
    ExecutionFailed { reason: String },

    #[error("Security violation: {violation}")]
    SecurityViolation { violation: String },

    #[error("Capability error: {error}")]
    CapabilityError { error: String },

    #[error("Information flow violation: {violation}")]
    InformationFlowViolation { violation: String },

    #[error("Capability not found: {name}")]
    CapabilityNotFound { name: String },

    #[error("Capability constraint violation: {constraint}")]
    CapabilityConstraintViolation { constraint: String },
}

impl From<execution::ExecutionError> for EffectSystemError {
    fn from(err: execution::ExecutionError) -> Self {
        match err {
            execution::ExecutionError::MissingCapability(cap) => {
                EffectSystemError::CapabilityError { error: cap }
            }
            execution::ExecutionError::SecurityViolation(violation) => {
                EffectSystemError::SecurityViolation { violation }
            }
            execution::ExecutionError::StepFailed(reason) => {
                EffectSystemError::ExecutionFailed { reason }
            }
            _ => EffectSystemError::ExecutionFailed { 
                reason: err.to_string() 
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let system = PrismEffectsSystem::new();
        let stats = system.get_stats();
        
        // System should be initialized with some default effects and handlers
        assert!(stats.effects_registered > 0);
        assert!(stats.handlers_registered > 0);
        assert!(stats.validation_rules > 0);
    }

    #[test]
    fn test_effect_processing() {
        let mut system = PrismEffectsSystem::new();
        
        // Create a simple effect
        let effect = Effect::new("IO.FileSystem.Read".to_string(), Span::dummy());
        
        // Create execution context with required capability
        let mut context = security::SecureExecutionContext::new(
            "test_context".to_string(),
            security::SecurityLevel::new("Public".to_string(), 0, vec![], vec![]),
            security::trust::TrustContext {
                level: security::trust::TrustLevel {
                    level: 1,
                    categories: vec!["basic".to_string()],
                },
                metadata: HashMap::new(),
            },
        );
        
        let filesystem_cap = Capability::new(
            "FileSystem".to_string(),
            capability::CapabilityConstraints::new(),
        );
        context.add_capability(filesystem_cap);
        
        // Process the effect
        let result = system.process_effects(vec![effect], context);
        
        // Should succeed with proper capabilities
        assert!(result.is_ok());
    }
}
