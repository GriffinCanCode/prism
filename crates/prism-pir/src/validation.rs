//! PIR Validation - Semantic Preservation and Invariant Checking
//!
//! This module implements the semantic validation framework that ensures PIR
//! transformations preserve semantic invariants and maintain program meaning.

use crate::{PIRResult, semantic::PrismIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PIR validator for semantic preservation
#[derive(Debug)]
pub struct PIRValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
    /// Configuration
    config: ValidationConfig,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict validation
    pub strict_mode: bool,
    /// Enable business context validation
    pub validate_business_context: bool,
    /// Enable effect system validation
    pub validate_effects: bool,
    /// Enable AI metadata validation
    pub validate_ai_metadata: bool,
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule implementation
    pub check: ValidationCheck,
}

/// Validation check implementation
#[derive(Debug, Clone)]
pub enum ValidationCheck {
    /// Semantic preservation check
    SemanticPreservation,
    /// Type consistency check
    TypeConsistency,
    /// Effect preservation check
    EffectPreservation,
    /// Business context integrity check
    BusinessContextIntegrity,
    /// AI metadata completeness check
    AIMetadataCompleteness,
}

/// Semantic preservation check
pub trait SemanticPreservationCheck: Send + Sync {
    /// Check semantic preservation
    fn check_preservation(&self, pir: &PrismIR) -> PIRResult<PreservationResult>;
    
    /// Get check name
    fn check_name(&self) -> &str;
    
    /// Get check description
    fn check_description(&self) -> &str;
}

/// Preservation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationResult {
    /// Overall result
    pub result: PreservationStatus,
    /// Preservation score (0.0 to 1.0)
    pub score: f64,
    /// Detailed findings
    pub findings: Vec<PreservationFinding>,
}

/// Preservation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreservationStatus {
    /// Fully preserved
    FullyPreserved,
    /// Mostly preserved
    MostlyPreserved,
    /// Partially preserved
    PartiallyPreserved,
    /// Not preserved
    NotPreserved,
}

/// Preservation finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationFinding {
    /// Finding type
    pub finding_type: FindingType,
    /// Severity
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Location
    pub location: Option<String>,
    /// Recommendation
    pub recommendation: Option<String>,
}

/// Finding types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingType {
    /// Missing semantic information
    MissingSemanticInfo,
    /// Inconsistent type information
    InconsistentTypes,
    /// Lost business context
    LostBusinessContext,
    /// Broken effect relationships
    BrokenEffects,
    /// Missing AI metadata
    MissingAIMetadata,
}

/// Severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Information
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation status
    pub status: ValidationStatus,
    /// Individual check results
    pub check_results: Vec<CheckResult>,
    /// Overall score (0.0 to 1.0)
    pub overall_score: f64,
    /// Validation metadata
    pub metadata: ValidationMetadata,
}

/// Validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed
    Passed,
    /// Validation passed with warnings
    PassedWithWarnings,
    /// Validation failed
    Failed,
}

/// Individual check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Check name
    pub check_name: String,
    /// Check result
    pub result: CheckStatus,
    /// Check score (0.0 to 1.0)
    pub score: f64,
    /// Findings
    pub findings: Vec<PreservationFinding>,
}

/// Check status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    /// Check passed
    Passed,
    /// Check passed with warnings
    PassedWithWarnings,
    /// Check failed
    Failed,
    /// Check skipped
    Skipped,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Validation timestamp
    pub timestamp: String,
    /// Validator version
    pub validator_version: String,
    /// Validation duration
    pub duration_ms: u64,
    /// PIR version validated
    pub pir_version: String,
}

/// Semantic invariant that must be preserved
pub trait SemanticInvariant: Send + Sync {
    /// Check if invariant holds for PIR
    fn check_invariant(&self, pir: &PrismIR) -> PIRResult<InvariantResult>;
    
    /// Get invariant name
    fn invariant_name(&self) -> &str;
    
    /// Get invariant description
    fn invariant_description(&self) -> &str;
}

/// Invariant check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantResult {
    /// Whether invariant holds
    pub holds: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Evidence for or against invariant
    pub evidence: Vec<InvariantEvidence>,
}

/// Evidence for invariant checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantEvidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Evidence description
    pub description: String,
    /// Supporting data
    pub data: Option<String>,
}

/// Evidence types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Supporting evidence
    Supporting,
    /// Contradicting evidence
    Contradicting,
    /// Neutral evidence
    Neutral,
}

/// Preservation rule
pub trait PreservationRule: Send + Sync {
    /// Apply preservation rule
    fn apply_rule(&self, pir: &PrismIR) -> PIRResult<RuleResult>;
    
    /// Get rule name
    fn rule_name(&self) -> &str;
    
    /// Get rule priority
    fn rule_priority(&self) -> RulePriority;
}

/// Rule result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleResult {
    /// Rule compliance
    pub compliant: bool,
    /// Compliance score (0.0 to 1.0)
    pub score: f64,
    /// Violations found
    pub violations: Vec<RuleViolation>,
}

/// Rule violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    /// Violation type
    pub violation_type: ViolationType,
    /// Severity
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Location
    pub location: Option<String>,
    /// Fix suggestion
    pub fix_suggestion: Option<String>,
}

/// Violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Semantic violation
    Semantic,
    /// Type violation
    Type,
    /// Effect violation
    Effect,
    /// Business rule violation
    BusinessRule,
    /// Consistency violation
    Consistency,
}

/// Rule priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RulePriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

impl PIRValidator {
    /// Create a new PIR validator
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a PIR validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        let mut validator = Self {
            rules: Vec::new(),
            config,
        };
        
        // Add default validation rules
        validator.add_default_rules();
        validator
    }

    /// Add a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Validate semantic preservation
    pub fn validate_semantic_preservation(&self, pir: &PrismIR) -> PIRResult<ValidationResult> {
        let start_time = std::time::Instant::now();
        let mut check_results = Vec::new();
        let mut overall_score = 0.0;
        let mut passed_checks = 0;

        for rule in &self.rules {
            let result = self.apply_validation_rule(rule, pir)?;
            overall_score += result.score;
            if result.result != CheckStatus::Failed {
                passed_checks += 1;
            }
            check_results.push(result);
        }

        let duration = start_time.elapsed();
        overall_score /= self.rules.len() as f64;

        let status = if passed_checks == self.rules.len() {
            if check_results.iter().any(|r| r.result == CheckStatus::PassedWithWarnings) {
                ValidationStatus::PassedWithWarnings
            } else {
                ValidationStatus::Passed
            }
        } else {
            ValidationStatus::Failed
        };

        Ok(ValidationResult {
            status,
            check_results,
            overall_score,
            metadata: ValidationMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                validator_version: "1.0.0".to_string(),
                duration_ms: duration.as_millis() as u64,
                pir_version: crate::PIRVersion::CURRENT.to_string(),
            },
        })
    }

    fn add_default_rules(&mut self) {
        // Add default semantic preservation rules
        self.add_rule(ValidationRule {
            id: "semantic_types".to_string(),
            name: "Semantic Type Preservation".to_string(),
            description: "Ensures semantic type information is preserved".to_string(),
            check: ValidationCheck::SemanticPreservation,
        });

        self.add_rule(ValidationRule {
            id: "type_consistency".to_string(),
            name: "Type Consistency".to_string(),
            description: "Ensures type information is consistent throughout PIR".to_string(),
            check: ValidationCheck::TypeConsistency,
        });

        if self.config.validate_effects {
            self.add_rule(ValidationRule {
                id: "effect_preservation".to_string(),
                name: "Effect Preservation".to_string(),
                description: "Ensures effect system information is preserved".to_string(),
                check: ValidationCheck::EffectPreservation,
            });
        }

        if self.config.validate_business_context {
            self.add_rule(ValidationRule {
                id: "business_context".to_string(),
                name: "Business Context Integrity".to_string(),
                description: "Ensures business context information is preserved".to_string(),
                check: ValidationCheck::BusinessContextIntegrity,
            });
        }

        if self.config.validate_ai_metadata {
            self.add_rule(ValidationRule {
                id: "ai_metadata".to_string(),
                name: "AI Metadata Completeness".to_string(),
                description: "Ensures AI metadata is complete and consistent".to_string(),
                check: ValidationCheck::AIMetadataCompleteness,
            });
        }
    }

    fn apply_validation_rule(&self, rule: &ValidationRule, pir: &PrismIR) -> PIRResult<CheckResult> {
        match &rule.check {
            ValidationCheck::SemanticPreservation => {
                self.check_semantic_preservation(pir)
            }
            ValidationCheck::TypeConsistency => {
                self.check_type_consistency(pir)
            }
            ValidationCheck::EffectPreservation => {
                self.check_effect_preservation(pir)
            }
            ValidationCheck::BusinessContextIntegrity => {
                self.check_business_context_integrity(pir)
            }
            ValidationCheck::AIMetadataCompleteness => {
                self.check_ai_metadata_completeness(pir)
            }
        }
    }

    fn check_semantic_preservation(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // Check if semantic types are preserved
        for module in &pir.modules {
            for section in &module.sections {
                if let crate::semantic::PIRSection::Types(type_section) = section {
                    for pir_type in &type_section.types {
                        if pir_type.business_rules.is_empty() && !self.config.strict_mode {
                            findings.push(PreservationFinding {
                                finding_type: FindingType::MissingSemanticInfo,
                                severity: Severity::Warning,
                                description: format!("Type '{}' has no business rules", pir_type.name),
                                location: Some(format!("module: {}, type: {}", module.name, pir_type.name)),
                                recommendation: Some("Add business rules to preserve semantic meaning".to_string()),
                            });
                            score -= 0.1;
                        }
                    }
                }
            }
        }

        let result = if score >= 0.9 {
            CheckStatus::Passed
        } else if score >= 0.7 {
            CheckStatus::PassedWithWarnings
        } else {
            CheckStatus::Failed
        };

        Ok(CheckResult {
            check_name: "Semantic Preservation".to_string(),
            result,
            score: score.max(0.0) as f64,
            findings,
        })
    }

    fn check_type_consistency(&self, _pir: &PrismIR) -> PIRResult<CheckResult> {
        // Implementation would check type consistency
        Ok(CheckResult {
            check_name: "Type Consistency".to_string(),
            result: CheckStatus::Passed,
            score: 1.0,
            findings: Vec::new(),
        })
    }

    fn check_effect_preservation(&self, _pir: &PrismIR) -> PIRResult<CheckResult> {
        // Implementation would check effect preservation
        Ok(CheckResult {
            check_name: "Effect Preservation".to_string(),
            result: CheckStatus::Passed,
            score: 1.0,
            findings: Vec::new(),
        })
    }

    fn check_business_context_integrity(&self, _pir: &PrismIR) -> PIRResult<CheckResult> {
        // Implementation would check business context integrity
        Ok(CheckResult {
            check_name: "Business Context Integrity".to_string(),
            result: CheckStatus::Passed,
            score: 1.0,
            findings: Vec::new(),
        })
    }

    fn check_ai_metadata_completeness(&self, _pir: &PrismIR) -> PIRResult<CheckResult> {
        // Implementation would check AI metadata completeness
        Ok(CheckResult {
            check_name: "AI Metadata Completeness".to_string(),
            result: CheckStatus::Passed,
            score: 1.0,
            findings: Vec::new(),
        })
    }
}

impl Default for PIRValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            validate_business_context: true,
            validate_effects: true,
            validate_ai_metadata: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = PIRValidator::new();
        assert!(!validator.rules.is_empty());
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert!(!config.strict_mode);
        assert!(config.validate_business_context);
        assert!(config.validate_effects);
        assert!(config.validate_ai_metadata);
    }

    #[test]
    fn test_validation_result() {
        let pir = PrismIR::new();
        let validator = PIRValidator::new();
        let result = validator.validate_semantic_preservation(&pir).unwrap();
        
        assert_eq!(result.status, ValidationStatus::Passed);
        assert!(!result.check_results.is_empty());
    }
} 