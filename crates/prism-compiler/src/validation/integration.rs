//! Integration with Existing Validators
//!
//! This module provides integration interfaces to existing validators from specialized crates,
//! ensuring we delegate validation responsibilities rather than duplicating logic.
//!
//! **Conceptual Responsibility**: Validator integration and delegation
//! **What it does**: Provides references to existing validators, manages integration lifecycle
//! **What it doesn't do**: Implement validation logic (delegates to existing validators)

use crate::error::{CompilerError, CompilerResult};
use crate::context::CompilationTarget;
use prism_pir::{PrismIR, PIRValidator, ValidationResult as PIRValidationResult};
use prism_semantic::{SemanticValidator, ValidationResult as SemanticValidationResult};
use prism_constraints::{ConstraintEngine, ValidationResult as ConstraintValidationResult};
use prism_effects::{EffectValidator, ValidationContext as EffectValidationContext};
use prism_codegen::{CodeGenBackend, CodeArtifact};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// References to existing validators from specialized crates
/// 
/// This struct maintains references to validators that already exist in other crates,
/// ensuring we delegate validation responsibilities rather than duplicating logic.
#[derive(Debug)]
pub struct ExistingValidatorRefs {
    /// PIR validator from prism-pir crate
    pub pir_validator: Arc<PIRValidator>,
    /// Semantic validator from prism-semantic crate
    pub semantic_validator: Arc<SemanticValidator>,
    /// Constraint engine from prism-constraints crate
    pub constraint_engine: Arc<ConstraintEngine>,
    /// Effect validator from prism-effects crate
    pub effect_validator: Arc<EffectValidator>,
    /// Codegen backend validators (they already have validate() methods)
    pub codegen_backends: HashMap<CompilationTarget, Arc<dyn CodeGenBackend>>,
}

/// Validation delegation trait for coordinating existing validators
pub trait ValidationDelegation {
    /// Delegate PIR validation to existing PIRValidator
    async fn delegate_pir_validation(&self, pir: &PrismIR) -> CompilerResult<PIRValidationResult>;
    
    /// Delegate semantic validation to existing SemanticValidator
    async fn delegate_semantic_validation(&self, pir: &PrismIR) -> CompilerResult<SemanticValidationResult>;
    
    /// Delegate constraint validation to existing ConstraintEngine
    async fn delegate_constraint_validation(&self, pir: &PrismIR) -> CompilerResult<ConstraintValidationResult>;
    
    /// Delegate effect validation to existing EffectValidator
    async fn delegate_effect_validation(&self, pir: &PrismIR) -> CompilerResult<EffectValidationContext>;
    
    /// Delegate codegen validation to existing backend validators
    async fn delegate_codegen_validation(
        &self,
        target: &CompilationTarget,
        artifact: &CodeArtifact,
    ) -> CompilerResult<Vec<String>>;
}

/// Aggregated validation results from all existing validators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedValidationResults {
    /// PIR validation result
    pub pir_result: PIRValidationResultSummary,
    /// Semantic validation result
    pub semantic_result: SemanticValidationResultSummary,
    /// Constraint validation result
    pub constraint_result: ConstraintValidationResultSummary,
    /// Effect validation result
    pub effect_result: EffectValidationResultSummary,
    /// Codegen validation results by target
    pub codegen_results: HashMap<CompilationTarget, CodegenValidationResultSummary>,
}

/// PIR validation result summary (converted from prism-pir types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRValidationResultSummary {
    /// Validation passed
    pub passed: bool,
    /// Overall score
    pub score: f64,
    /// Number of check results
    pub check_count: usize,
    /// Validation duration
    pub duration_ms: u64,
}

/// Semantic validation result summary (converted from prism-semantic types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticValidationResultSummary {
    /// Validation passed
    pub passed: bool,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
    /// Rule violation count
    pub rule_violation_count: usize,
}

/// Constraint validation result summary (converted from prism-constraints types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidationResultSummary {
    /// Validation passed
    pub passed: bool,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
}

/// Effect validation result summary (converted from prism-effects types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectValidationResultSummary {
    /// Validation passed
    pub passed: bool,
    /// Violation count
    pub violation_count: usize,
    /// Warning count
    pub warning_count: usize,
}

/// Codegen validation result summary (from existing backend validate() methods)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodegenValidationResultSummary {
    /// Target platform
    pub target: CompilationTarget,
    /// Warning messages from backend validation
    pub warnings: Vec<String>,
    /// Validation passed (no critical issues)
    pub passed: bool,
}

impl ExistingValidatorRefs {
    /// Create new validator references
    pub fn new(
        pir_validator: Arc<PIRValidator>,
        semantic_validator: Arc<SemanticValidator>,
        constraint_engine: Arc<ConstraintEngine>,
        effect_validator: Arc<EffectValidator>,
        codegen_backends: HashMap<CompilationTarget, Arc<dyn CodeGenBackend>>,
    ) -> Self {
        Self {
            pir_validator,
            semantic_validator,
            constraint_engine,
            effect_validator,
            codegen_backends,
        }
    }

    /// Check if all required validators are available
    pub fn validate_integration(&self) -> CompilerResult<()> {
        // Verify all required validators are present
        if self.codegen_backends.is_empty() {
            return Err(CompilerError::ValidationError {
                message: "No codegen backend validators available".to_string(),
            });
        }

        Ok(())
    }

    /// Get integration status
    pub fn get_integration_status(&self) -> crate::ValidationIntegrationStatus {
        crate::ValidationIntegrationStatus {
            pir_validator: true, // Always true if we have the reference
            semantic_validator: true,
            constraint_engine: true,
            effect_validator: true,
            codegen_validators: self.codegen_backends.keys().cloned().collect(),
        }
    }
}

impl ValidationDelegation for ExistingValidatorRefs {
    /// Delegate PIR validation to existing PIRValidator
    async fn delegate_pir_validation(&self, pir: &PrismIR) -> CompilerResult<PIRValidationResult> {
        // Delegate to existing PIRValidator - no duplication
        self.pir_validator
            .validate_semantic_preservation(pir)
            .map_err(|e| CompilerError::ValidationError {
                message: format!("PIR validation failed: {}", e),
            })
    }

    /// Delegate semantic validation to existing SemanticValidator
    async fn delegate_semantic_validation(&self, pir: &PrismIR) -> CompilerResult<SemanticValidationResult> {
        // Convert PIR to format expected by semantic validator
        // This is a coordination responsibility, not duplication
        
        // For now, we'll create a placeholder since we need to coordinate between
        // PIR format and semantic validator input format
        // In a real implementation, this would involve format conversion
        
        // Delegate to existing SemanticValidator - no duplication
        // Note: This would need proper PIR -> AST conversion for semantic validation
        Err(CompilerError::ValidationError {
            message: "Semantic validation delegation requires PIR->AST conversion (not yet implemented)".to_string(),
        })
    }

    /// Delegate constraint validation to existing ConstraintEngine
    async fn delegate_constraint_validation(&self, pir: &PrismIR) -> CompilerResult<ConstraintValidationResult> {
        // Extract constraints from PIR and delegate to existing ConstraintEngine
        // This is coordination, not duplication
        
        // For now, return a placeholder result
        // In a real implementation, this would extract PIR constraints and validate them
        Err(CompilerError::ValidationError {
            message: "Constraint validation delegation requires constraint extraction (not yet implemented)".to_string(),
        })
    }

    /// Delegate effect validation to existing EffectValidator
    async fn delegate_effect_validation(&self, pir: &PrismIR) -> CompilerResult<EffectValidationContext> {
        // Extract effects from PIR and delegate to existing EffectValidator
        // This is coordination, not duplication
        
        // For now, return a placeholder result
        // In a real implementation, this would extract PIR effects and validate them
        Err(CompilerError::ValidationError {
            message: "Effect validation delegation requires effect extraction (not yet implemented)".to_string(),
        })
    }

    /// Delegate codegen validation to existing backend validators
    async fn delegate_codegen_validation(
        &self,
        target: &CompilationTarget,
        artifact: &CodeArtifact,
    ) -> CompilerResult<Vec<String>> {
        // Delegate to existing CodeGenBackend::validate() methods - no duplication
        if let Some(backend) = self.codegen_backends.get(target) {
            backend
                .validate(artifact)
                .await
                .map_err(|e| CompilerError::ValidationError {
                    message: format!("Codegen validation failed for {:?}: {}", target, e),
                })
        } else {
            Err(CompilerError::ValidationError {
                message: format!("No validator available for target: {:?}", target),
            })
        }
    }
}

/// Builder for creating validator references with proper integration
pub struct ValidatorIntegrationBuilder {
    pir_validator: Option<Arc<PIRValidator>>,
    semantic_validator: Option<Arc<SemanticValidator>>,
    constraint_engine: Option<Arc<ConstraintEngine>>,
    effect_validator: Option<Arc<EffectValidator>>,
    codegen_backends: HashMap<CompilationTarget, Arc<dyn CodeGenBackend>>,
}

impl ValidatorIntegrationBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            pir_validator: None,
            semantic_validator: None,
            constraint_engine: None,
            effect_validator: None,
            codegen_backends: HashMap::new(),
        }
    }

    /// Add PIR validator reference
    pub fn with_pir_validator(mut self, validator: Arc<PIRValidator>) -> Self {
        self.pir_validator = Some(validator);
        self
    }

    /// Add semantic validator reference
    pub fn with_semantic_validator(mut self, validator: Arc<SemanticValidator>) -> Self {
        self.semantic_validator = Some(validator);
        self
    }

    /// Add constraint engine reference
    pub fn with_constraint_engine(mut self, engine: Arc<ConstraintEngine>) -> Self {
        self.constraint_engine = Some(engine);
        self
    }

    /// Add effect validator reference
    pub fn with_effect_validator(mut self, validator: Arc<EffectValidator>) -> Self {
        self.effect_validator = Some(validator);
        self
    }

    /// Add codegen backend reference
    pub fn with_codegen_backend(
        mut self,
        target: CompilationTarget,
        backend: Arc<dyn CodeGenBackend>,
    ) -> Self {
        self.codegen_backends.insert(target, backend);
        self
    }

    /// Build validator references
    pub fn build(self) -> CompilerResult<ExistingValidatorRefs> {
        let pir_validator = self.pir_validator.ok_or_else(|| CompilerError::ValidationError {
            message: "PIR validator reference is required".to_string(),
        })?;

        let semantic_validator = self.semantic_validator.ok_or_else(|| CompilerError::ValidationError {
            message: "Semantic validator reference is required".to_string(),
        })?;

        let constraint_engine = self.constraint_engine.ok_or_else(|| CompilerError::ValidationError {
            message: "Constraint engine reference is required".to_string(),
        })?;

        let effect_validator = self.effect_validator.ok_or_else(|| CompilerError::ValidationError {
            message: "Effect validator reference is required".to_string(),
        })?;

        if self.codegen_backends.is_empty() {
            return Err(CompilerError::ValidationError {
                message: "At least one codegen backend reference is required".to_string(),
            });
        }

        let refs = ExistingValidatorRefs::new(
            pir_validator,
            semantic_validator,
            constraint_engine,
            effect_validator,
            self.codegen_backends,
        );

        // Validate integration
        refs.validate_integration()?;

        Ok(refs)
    }
}

impl Default for ValidatorIntegrationBuilder {
    fn default() -> Self {
        Self::new()
    }
} 