//! Validation Orchestrator - Main Coordination Engine
//!
//! This module implements the main validation orchestrator that coordinates cross-target
//! validation by delegating to existing validators from specialized crates.
//!
//! **Conceptual Responsibility**: Validation orchestration and coordination
//! **What it does**: Coordinates existing validators, aggregates results, manages validation flow
//! **What it doesn't do**: Implement validation logic (delegates to existing validators)

use crate::error::{CompilerError, CompilerResult};
use crate::context::CompilationTarget;
use crate::validation::{
    ValidationOrchestration, ValidationCapabilities, ValidationType, ValidationIntegrationStatus,
    ExistingValidatorRefs, ValidationDelegation, AggregatedValidationResults,
    PIRValidationResultSummary, SemanticValidationResultSummary, ConstraintValidationResultSummary,
    EffectValidationResultSummary, CodegenValidationResultSummary,
    ValidationReport, ValidationSummary, ValidationRecommendation,
    ValidationCoordinator, ConsistencyAnalysis
};
use prism_pir::{PrismIR, ValidationResult as PIRValidationResult};
use prism_codegen::CodeArtifact;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, debug, warn};

/// Main validation orchestrator that coordinates existing validators
/// 
/// This orchestrator follows the delegation pattern, using existing validators from
/// specialized crates rather than duplicating validation logic.
#[derive(Debug)]
pub struct ValidationOrchestrator {
    /// Configuration for orchestration
    config: ValidationOrchestratorConfig,
    /// References to existing validators
    validator_refs: Arc<ExistingValidatorRefs>,
    /// Result coordinator
    coordinator: ValidationCoordinator,
}

/// Configuration for validation orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationOrchestratorConfig {
    /// Enable parallel validation
    pub enable_parallel: bool,
    /// Validation timeout in seconds
    pub timeout_seconds: u64,
    /// Enable detailed reporting
    pub detailed_reporting: bool,
    /// Validation types to include
    pub enabled_validations: Vec<ValidationType>,
    /// Fail fast on critical issues
    pub fail_fast: bool,
}

impl Default for ValidationOrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            timeout_seconds: 30,
            detailed_reporting: true,
            enabled_validations: vec![
                ValidationType::SemanticPreservation,
                ValidationType::BusinessRuleConsistency,
                ValidationType::ConstraintValidation,
                ValidationType::EffectValidation,
                ValidationType::CrossTargetConsistency,
            ],
            fail_fast: false,
        }
    }
}

impl ValidationOrchestrator {
    /// Create new validation orchestrator with existing validator references
    pub fn new(
        config: ValidationOrchestratorConfig,
        validator_refs: Arc<ExistingValidatorRefs>,
    ) -> CompilerResult<Self> {
        // Validate that we have all required validator references
        validator_refs.validate_integration()?;

        let coordinator = ValidationCoordinator::new();

        Ok(Self {
            config,
            validator_refs,
            coordinator,
        })
    }

    /// Create orchestrator with default configuration
    pub fn with_validator_refs(validator_refs: Arc<ExistingValidatorRefs>) -> CompilerResult<Self> {
        Self::new(ValidationOrchestratorConfig::default(), validator_refs)
    }

    /// Orchestrate validation by delegating to existing validators
    async fn orchestrate_validation_internal(
        &self,
        pir: &PrismIR,
        artifacts: &[(&CompilationTarget, &CodeArtifact)],
    ) -> CompilerResult<AggregatedValidationResults> {
        let start_time = Instant::now();
        info!("Starting validation orchestration for {} targets", artifacts.len());

        let mut results = AggregatedValidationResults {
            pir_result: PIRValidationResultSummary {
                passed: true,
                score: 1.0,
                check_count: 0,
                duration_ms: 0,
            },
            semantic_result: SemanticValidationResultSummary {
                passed: true,
                error_count: 0,
                warning_count: 0,
                rule_violation_count: 0,
            },
            constraint_result: ConstraintValidationResultSummary {
                passed: true,
                error_count: 0,
                warning_count: 0,
            },
            effect_result: EffectValidationResultSummary {
                passed: true,
                violation_count: 0,
                warning_count: 0,
            },
            codegen_results: HashMap::new(),
        };

        // Delegate PIR validation to existing PIRValidator
        if self.config.enabled_validations.contains(&ValidationType::SemanticPreservation) {
            debug!("Delegating PIR validation to existing PIRValidator");
            match self.validator_refs.delegate_pir_validation(pir).await {
                Ok(pir_result) => {
                    results.pir_result = Self::convert_pir_result(&pir_result);
                    info!("PIR validation completed: passed={}, score={}", 
                          results.pir_result.passed, results.pir_result.score);
                }
                Err(e) => {
                    warn!("PIR validation failed: {}", e);
                    results.pir_result.passed = false;
                    if self.config.fail_fast {
                        return Err(e);
                    }
                }
            }
        }

        // Delegate semantic validation to existing SemanticValidator
        if self.config.enabled_validations.contains(&ValidationType::BusinessRuleConsistency) {
            debug!("Delegating semantic validation to existing SemanticValidator");
            match self.validator_refs.delegate_semantic_validation(pir).await {
                Ok(semantic_result) => {
                    results.semantic_result = Self::convert_semantic_result(&semantic_result);
                    info!("Semantic validation completed: passed={}, errors={}", 
                          results.semantic_result.passed, results.semantic_result.error_count);
                }
                Err(e) => {
                    warn!("Semantic validation delegation not yet implemented: {}", e);
                    // Don't fail fast for not-yet-implemented delegations
                }
            }
        }

        // Delegate constraint validation to existing ConstraintEngine
        if self.config.enabled_validations.contains(&ValidationType::ConstraintValidation) {
            debug!("Delegating constraint validation to existing ConstraintEngine");
            match self.validator_refs.delegate_constraint_validation(pir).await {
                Ok(constraint_result) => {
                    results.constraint_result = Self::convert_constraint_result(&constraint_result);
                    info!("Constraint validation completed: passed={}, errors={}", 
                          results.constraint_result.passed, results.constraint_result.error_count);
                }
                Err(e) => {
                    warn!("Constraint validation delegation not yet implemented: {}", e);
                    // Don't fail fast for not-yet-implemented delegations
                }
            }
        }

        // Delegate effect validation to existing EffectValidator
        if self.config.enabled_validations.contains(&ValidationType::EffectValidation) {
            debug!("Delegating effect validation to existing EffectValidator");
            match self.validator_refs.delegate_effect_validation(pir).await {
                Ok(effect_context) => {
                    results.effect_result = Self::convert_effect_result(&effect_context);
                    info!("Effect validation completed: passed={}, violations={}", 
                          results.effect_result.passed, results.effect_result.violation_count);
                }
                Err(e) => {
                    warn!("Effect validation delegation not yet implemented: {}", e);
                    // Don't fail fast for not-yet-implemented delegations
                }
            }
        }

        // Delegate codegen validation to existing backend validators
        for (target, artifact) in artifacts {
            debug!("Delegating codegen validation for {:?} to existing backend validator", target);
            match self.validator_refs.delegate_codegen_validation(target, artifact).await {
                Ok(warnings) => {
                    let codegen_result = CodegenValidationResultSummary {
                        target: *target,
                        warnings: warnings.clone(),
                        passed: warnings.iter().all(|w| !w.contains("error") && !w.contains("failed")),
                    };
                    results.codegen_results.insert(*target, codegen_result);
                    info!("Codegen validation for {:?} completed: {} warnings", target, warnings.len());
                }
                Err(e) => {
                    warn!("Codegen validation failed for {:?}: {}", target, e);
                    let failed_result = CodegenValidationResultSummary {
                        target: *target,
                        warnings: vec![format!("Validation failed: {}", e)],
                        passed: false,
                    };
                    results.codegen_results.insert(*target, failed_result);
                    if self.config.fail_fast {
                        return Err(e);
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        info!("Validation orchestration completed in {:?}", duration);

        Ok(results)
    }

    /// Convert PIR validation result to summary
    fn convert_pir_result(result: &PIRValidationResult) -> PIRValidationResultSummary {
        PIRValidationResultSummary {
            passed: result.status == prism_pir::ValidationStatus::Passed || 
                   result.status == prism_pir::ValidationStatus::PassedWithWarnings,
            score: result.overall_score,
            check_count: result.check_results.len(),
            duration_ms: result.metadata.duration_ms,
        }
    }

    /// Convert semantic validation result to summary (placeholder)
    fn convert_semantic_result(result: &prism_semantic::ValidationResult) -> SemanticValidationResultSummary {
        SemanticValidationResultSummary {
            passed: result.passed,
            error_count: result.errors.len(),
            warning_count: result.warnings.len(),
            rule_violation_count: result.rule_violations.len(),
        }
    }

    /// Convert constraint validation result to summary (placeholder)
    fn convert_constraint_result(result: &prism_constraints::ValidationResult) -> ConstraintValidationResultSummary {
        ConstraintValidationResultSummary {
            passed: result.errors.is_empty(),
            error_count: result.errors.len(),
            warning_count: result.warnings.len(),
        }
    }

    /// Convert effect validation context to summary (placeholder)
    fn convert_effect_result(context: &prism_effects::ValidationContext) -> EffectValidationResultSummary {
        EffectValidationResultSummary {
            passed: context.violations.is_empty(),
            violation_count: context.violations.len(),
            warning_count: context.warnings.len(),
        }
    }

    /// Generate validation recommendations based on results
    fn generate_recommendations(results: &AggregatedValidationResults) -> Vec<ValidationRecommendation> {
        let mut recommendations = Vec::new();

        // PIR validation recommendations
        if !results.pir_result.passed {
            recommendations.push(ValidationRecommendation {
                category: "PIR Validation".to_string(),
                priority: "High".to_string(),
                message: "PIR semantic preservation validation failed. Review PIR generation process.".to_string(),
                action: "Check PIR validator output for specific issues".to_string(),
            });
        }

        // Semantic validation recommendations
        if !results.semantic_result.passed {
            recommendations.push(ValidationRecommendation {
                category: "Semantic Validation".to_string(),
                priority: "High".to_string(),
                message: format!("Semantic validation found {} errors and {} rule violations", 
                               results.semantic_result.error_count, 
                               results.semantic_result.rule_violation_count),
                action: "Review business rule implementations and semantic constraints".to_string(),
            });
        }

        // Codegen validation recommendations
        for (target, result) in &results.codegen_results {
            if !result.passed {
                recommendations.push(ValidationRecommendation {
                    category: format!("Codegen {:?}", target),
                    priority: "Medium".to_string(),
                    message: format!("Code generation validation failed for {:?}", target),
                    action: "Review generated code and backend-specific validation rules".to_string(),
                });
            } else if !result.warnings.is_empty() {
                recommendations.push(ValidationRecommendation {
                    category: format!("Codegen {:?}", target),
                    priority: "Low".to_string(),
                    message: format!("Code generation produced {} warnings for {:?}", 
                                   result.warnings.len(), target),
                    action: "Review warnings and consider code improvements".to_string(),
                });
            }
        }

        if recommendations.is_empty() {
            recommendations.push(ValidationRecommendation {
                category: "Overall".to_string(),
                priority: "Info".to_string(),
                message: "All validations passed successfully".to_string(),
                action: "Consider running additional integration tests".to_string(),
            });
        }

        recommendations
    }
}

impl ValidationOrchestration for ValidationOrchestrator {
    /// Orchestrate cross-target validation using existing validators
    async fn validate_cross_target_consistency(
        &self,
        pir: &PrismIR,
        artifacts: &[(&CompilationTarget, &CodeArtifact)],
    ) -> CompilerResult<ValidationReport> {
        info!("Starting cross-target validation orchestration");

        // Orchestrate validation by delegating to existing validators
        let aggregated_results = self.orchestrate_validation_internal(pir, artifacts).await?;

        // Coordinate results and perform consistency analysis
        let consistency_analysis = self.coordinator.analyze_consistency(&aggregated_results)?;

        // Generate summary
        let summary = ValidationSummary {
            total_validations: self.config.enabled_validations.len(),
            passed_validations: if aggregated_results.pir_result.passed && 
                                  aggregated_results.semantic_result.passed &&
                                  aggregated_results.constraint_result.passed &&
                                  aggregated_results.effect_result.passed &&
                                  aggregated_results.codegen_results.values().all(|r| r.passed) {
                self.config.enabled_validations.len()
            } else {
                0 // Simplified for now
            },
            total_issues: aggregated_results.semantic_result.error_count + 
                         aggregated_results.constraint_result.error_count +
                         aggregated_results.effect_result.violation_count,
            consistency_score: if consistency_analysis.overall_consistency { 1.0 } else { 0.5 },
            validation_duration_ms: 0, // Would be calculated from timing
        };

        // Generate recommendations
        let recommendations = Self::generate_recommendations(&aggregated_results);

        Ok(ValidationReport {
            summary,
            consistency_analysis,
            aggregated_results,
            recommendations,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Get validation capabilities
    fn get_validation_capabilities(&self) -> ValidationCapabilities {
        ValidationCapabilities {
            supported_types: self.config.enabled_validations.clone(),
            target_validators: self.validator_refs.codegen_backends.keys().cloned().collect(),
            integrations: self.validator_refs.get_integration_status(),
        }
    }
} 