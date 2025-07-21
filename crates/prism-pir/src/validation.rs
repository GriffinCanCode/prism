//! PIR Validation - Semantic Preservation and Invariant Checking
//!
//! This module implements the semantic validation framework that ensures PIR
//! transformations preserve semantic invariants and maintain program meaning.
//!
//! **Conceptual Responsibility**: PIR-specific validation coordination
//! **What it does**: Validates PIR structure, coordinates domain validators, ensures semantic preservation
//! **What it doesn't do**: Duplicate domain validation logic (delegates to existing validators)

use crate::{PIRResult, semantic::PrismIR};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// PIR validator that coordinates with existing domain validators
#[derive(Debug)]
pub struct PIRValidator {
    /// Validation rules specific to PIR
    rules: Vec<ValidationRule>,
    /// Configuration
    config: ValidationConfig,
    /// Domain validator integrations
    domain_integrations: DomainValidatorIntegrations,
}

/// Integration with existing domain validators
#[derive(Debug)]
pub struct DomainValidatorIntegrations {
    /// Semantic validator integration
    pub semantic_integration: Option<SemanticValidatorIntegration>,
    /// Constraint engine integration  
    pub constraint_integration: Option<ConstraintEngineIntegration>,
    /// Effect validator integration
    pub effect_integration: Option<EffectValidatorIntegration>,
}

/// Integration with prism-semantic validator
#[derive(Debug)]
pub struct SemanticValidatorIntegration {
    /// Enable business rule validation delegation
    pub delegate_business_rules: bool,
    /// Enable semantic constraint validation delegation
    pub delegate_semantic_constraints: bool,
}

/// Integration with prism-constraints engine
#[derive(Debug)]
pub struct ConstraintEngineIntegration {
    /// Enable constraint validation delegation
    pub delegate_constraints: bool,
    /// Constraint validation context
    pub validation_context: Option<String>, // Would be actual context type
}

/// Integration with prism-effects validator
#[derive(Debug)]
pub struct EffectValidatorIntegration {
    /// Enable effect validation delegation
    pub delegate_effects: bool,
    /// Effect validation context
    pub validation_context: Option<String>, // Would be actual context type
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
    /// Enable delegation to domain validators
    pub enable_domain_delegation: bool,
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
    /// PIR structure validation
    PIRStructureValidation,
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
    /// PIR structure issues
    PIRStructureIssue,
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
    /// Structure violation
    Structure,
}

/// Rule priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
            domain_integrations: DomainValidatorIntegrations {
                semantic_integration: None,
                constraint_integration: None,
                effect_integration: None,
            },
        };
        
        // Add default validation rules
        validator.add_default_rules();
        validator
    }

    /// Add a validation rule
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    /// Configure domain validator integrations
    pub fn with_domain_integrations(mut self, integrations: DomainValidatorIntegrations) -> Self {
        self.domain_integrations = integrations;
        self
    }

    /// Validate semantic preservation - PIR's primary responsibility
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
                pir_version: pir.metadata.version.clone(),
            },
        })
    }

    /// Add default validation rules
    fn add_default_rules(&mut self) {
        // PIR structure validation
        self.rules.push(ValidationRule {
            id: "pir_structure".to_string(),
            name: "PIR Structure Validation".to_string(),
            description: "Validates basic PIR structure and completeness".to_string(),
            check: ValidationCheck::PIRStructureValidation,
        });

        // Semantic preservation validation
        self.rules.push(ValidationRule {
            id: "semantic_preservation".to_string(),
            name: "Semantic Preservation".to_string(),
            description: "Ensures semantic information is preserved in PIR".to_string(),
            check: ValidationCheck::SemanticPreservation,
        });

        // Type consistency validation
        self.rules.push(ValidationRule {
            id: "type_consistency".to_string(),
            name: "Type Consistency".to_string(),
            description: "Validates type consistency across PIR".to_string(),
            check: ValidationCheck::TypeConsistency,
        });

        // Effect preservation validation
        if self.config.validate_effects {
            self.rules.push(ValidationRule {
                id: "effect_preservation".to_string(),
                name: "Effect Preservation".to_string(),
                description: "Ensures effect information is preserved".to_string(),
                check: ValidationCheck::EffectPreservation,
            });
        }

        // Business context integrity validation
        if self.config.validate_business_context {
            self.rules.push(ValidationRule {
                id: "business_context_integrity".to_string(),
                name: "Business Context Integrity".to_string(),
                description: "Validates business context preservation".to_string(),
                check: ValidationCheck::BusinessContextIntegrity,
            });
        }

        // AI metadata completeness validation
        if self.config.validate_ai_metadata {
            self.rules.push(ValidationRule {
                id: "ai_metadata_completeness".to_string(),
                name: "AI Metadata Completeness".to_string(),
                description: "Ensures AI metadata is complete".to_string(),
                check: ValidationCheck::AIMetadataCompleteness,
            });
        }
    }

    fn apply_validation_rule(&self, rule: &ValidationRule, pir: &PrismIR) -> PIRResult<CheckResult> {
        match &rule.check {
            ValidationCheck::PIRStructureValidation => {
                self.check_pir_structure(pir)
            }
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

    /// Check PIR structure integrity - PIR-specific validation
    fn check_pir_structure(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // Check basic PIR structure
        if pir.modules.is_empty() {
            findings.push(PreservationFinding {
                finding_type: FindingType::PIRStructureIssue,
                severity: Severity::Error,
                description: "PIR contains no modules".to_string(),
                location: Some("PIR root".to_string()),
                recommendation: Some("Ensure PIR contains at least one module".to_string()),
            });
            score -= 0.5;
        }

        // Check module structure
        for module in &pir.modules {
            if module.sections.is_empty() {
                findings.push(PreservationFinding {
                    finding_type: FindingType::PIRStructureIssue,
                    severity: Severity::Warning,
                    description: format!("Module '{}' has no sections", module.name),
                    location: Some(format!("module: {}", module.name)),
                    recommendation: Some("Consider adding sections to the module".to_string()),
                });
                score -= 0.1;
            }
        }

        // Check type registry
        if pir.type_registry.types.is_empty() {
            findings.push(PreservationFinding {
                finding_type: FindingType::PIRStructureIssue,
                severity: Severity::Warning,
                description: "PIR type registry is empty".to_string(),
                location: Some("type_registry".to_string()),
                recommendation: Some("Ensure types are registered in PIR".to_string()),
            });
            score -= 0.1;
        }

        let result = if score >= 0.9 {
            CheckStatus::Passed
        } else if score >= 0.7 {
            CheckStatus::PassedWithWarnings
        } else {
            CheckStatus::Failed
        };

        Ok(CheckResult {
            check_name: "PIR Structure Validation".to_string(),
            result,
            score: score.max(0.0) as f64,
            findings,
        })
    }

    /// Check semantic preservation - delegates to domain validators when possible
    fn check_semantic_preservation(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // PIR-specific semantic checks
        for module in &pir.modules {
            // Check business capability preservation
            if module.capability.is_empty() {
                findings.push(PreservationFinding {
                    finding_type: FindingType::LostBusinessContext,
                    severity: Severity::Warning,
                    description: format!("Module '{}' has no business capability defined", module.name),
                    location: Some(format!("module: {}", module.name)),
                    recommendation: Some("Define business capability for the module".to_string()),
                });
                score -= 0.1;
            }

            // Check semantic type preservation
            for section in &module.sections {
                if let crate::semantic::PIRSection::Types(type_section) = section {
                    for pir_type in &type_section.types {
                        // Check business rules preservation
                        if pir_type.business_rules.is_empty() && self.config.strict_mode {
                            findings.push(PreservationFinding {
                                finding_type: FindingType::MissingSemanticInfo,
                                severity: Severity::Warning,
                                description: format!("Type '{}' has no business rules", pir_type.name),
                                location: Some(format!("module: {}, type: {}", module.name, pir_type.name)),
                                recommendation: Some("Add business rules to preserve semantic meaning".to_string()),
                            });
                            score -= 0.05;
                        }

                        // Check domain classification
                        if pir_type.domain.is_empty() {
                            findings.push(PreservationFinding {
                                finding_type: FindingType::MissingSemanticInfo,
                                severity: Severity::Info,
                                description: format!("Type '{}' has no domain classification", pir_type.name),
                                location: Some(format!("module: {}, type: {}", module.name, pir_type.name)),
                                recommendation: Some("Add domain classification for better semantic understanding".to_string()),
                            });
                            score -= 0.02;
                        }
                    }
                }
            }
        }

        // Delegate to semantic validator if configured
        if self.config.enable_domain_delegation {
            if let Some(_semantic_integration) = &self.domain_integrations.semantic_integration {
                // Would delegate to prism-semantic validator here
                // For now, we'll add a placeholder finding
                findings.push(PreservationFinding {
                    finding_type: FindingType::MissingSemanticInfo,
                    severity: Severity::Info,
                    description: "Semantic validator delegation not yet implemented".to_string(),
                    location: Some("domain_delegation".to_string()),
                    recommendation: Some("Implement semantic validator delegation".to_string()),
                });
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

    /// Check type consistency - delegates to constraint engine when possible
    fn check_type_consistency(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // PIR-specific type consistency checks
        let mut type_names = std::collections::HashSet::new();
        
        for (type_name, _) in &pir.type_registry.types {
            if !type_names.insert(type_name.clone()) {
                findings.push(PreservationFinding {
                    finding_type: FindingType::InconsistentTypes,
                    severity: Severity::Error,
                    description: format!("Duplicate type name: {}", type_name),
                    location: Some(format!("type_registry: {}", type_name)),
                    recommendation: Some("Ensure type names are unique".to_string()),
                });
                score -= 0.2;
            }
        }

        // Check type relationships consistency
        for (type_name, relationships) in &pir.type_registry.relationships {
            if !pir.type_registry.types.contains_key(type_name) {
                findings.push(PreservationFinding {
                    finding_type: FindingType::InconsistentTypes,
                    severity: Severity::Error,
                    description: format!("Type relationship references unknown type: {}", type_name),
                    location: Some(format!("type_relationships: {}", type_name)),
                    recommendation: Some("Ensure all type relationships reference valid types".to_string()),
                });
                score -= 0.2;
            }

            // Check relationship targets
            for relationship in relationships {
                if !pir.type_registry.types.contains_key(&relationship.related_type) {
                    findings.push(PreservationFinding {
                        finding_type: FindingType::InconsistentTypes,
                        severity: Severity::Error,
                        description: format!("Type relationship targets unknown type: {}", relationship.related_type),
                        location: Some(format!("type_relationship: {} -> {}", type_name, relationship.related_type)),
                        recommendation: Some("Ensure relationship targets exist".to_string()),
                    });
                    score -= 0.1;
                }
            }
        }

        // Delegate to constraint engine if configured
        if self.config.enable_domain_delegation {
            if let Some(_constraint_integration) = &self.domain_integrations.constraint_integration {
                // Would delegate to prism-constraints engine here
                findings.push(PreservationFinding {
                    finding_type: FindingType::InconsistentTypes,
                    severity: Severity::Info,
                    description: "Constraint engine delegation not yet implemented".to_string(),
                    location: Some("domain_delegation".to_string()),
                    recommendation: Some("Implement constraint engine delegation".to_string()),
                });
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
            check_name: "Type Consistency".to_string(),
            result,
            score: score.max(0.0) as f64,
            findings,
        })
    }

    /// Check effect preservation - delegates to effect validator when possible
    fn check_effect_preservation(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // PIR-specific effect preservation checks
        if pir.effect_graph.nodes.is_empty() && !pir.modules.is_empty() {
            findings.push(PreservationFinding {
                finding_type: FindingType::BrokenEffects,
                severity: Severity::Warning,
                description: "Effect graph is empty but PIR contains modules".to_string(),
                location: Some("effect_graph".to_string()),
                recommendation: Some("Ensure effects are captured in effect graph".to_string()),
            });
            score -= 0.2;
        }

        // Check effect graph consistency
        for edge in &pir.effect_graph.edges {
            if !pir.effect_graph.nodes.contains_key(&edge.source) {
                findings.push(PreservationFinding {
                    finding_type: FindingType::BrokenEffects,
                    severity: Severity::Error,
                    description: format!("Effect edge references unknown source: {}", edge.source),
                    location: Some(format!("effect_edge: {} -> {}", edge.source, edge.target)),
                    recommendation: Some("Ensure effect edges reference valid nodes".to_string()),
                });
                score -= 0.3;
            }

            if !pir.effect_graph.nodes.contains_key(&edge.target) {
                findings.push(PreservationFinding {
                    finding_type: FindingType::BrokenEffects,
                    severity: Severity::Error,
                    description: format!("Effect edge references unknown target: {}", edge.target),
                    location: Some(format!("effect_edge: {} -> {}", edge.source, edge.target)),
                    recommendation: Some("Ensure effect edges reference valid nodes".to_string()),
                });
                score -= 0.3;
            }
        }

        // Delegate to effect validator if configured
        if self.config.enable_domain_delegation {
            if let Some(_effect_integration) = &self.domain_integrations.effect_integration {
                // Would delegate to prism-effects validator here
                findings.push(PreservationFinding {
                    finding_type: FindingType::BrokenEffects,
                    severity: Severity::Info,
                    description: "Effect validator delegation not yet implemented".to_string(),
                    location: Some("domain_delegation".to_string()),
                    recommendation: Some("Implement effect validator delegation".to_string()),
                });
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
            check_name: "Effect Preservation".to_string(),
            result,
            score: score.max(0.0) as f64,
            findings,
        })
    }

    /// Check business context integrity
    fn check_business_context_integrity(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // Check module business context
        for module in &pir.modules {
            if module.business_context.domain.is_empty() {
                findings.push(PreservationFinding {
                    finding_type: FindingType::LostBusinessContext,
                    severity: Severity::Warning,
                    description: format!("Module '{}' has no business domain", module.name),
                    location: Some(format!("module: {}", module.name)),
                    recommendation: Some("Define business domain for the module".to_string()),
                });
                score -= 0.1;
            }

            if module.business_context.entities.is_empty() {
                findings.push(PreservationFinding {
                    finding_type: FindingType::LostBusinessContext,
                    severity: Severity::Info,
                    description: format!("Module '{}' has no business entities", module.name),
                    location: Some(format!("module: {}", module.name)),
                    recommendation: Some("Consider defining business entities".to_string()),
                });
                score -= 0.05;
            }

            // Check cohesion score
            if module.cohesion_score < 0.5 {
                findings.push(PreservationFinding {
                    finding_type: FindingType::LostBusinessContext,
                    severity: Severity::Warning,
                    description: format!("Module '{}' has low cohesion score: {:.2}", module.name, module.cohesion_score),
                    location: Some(format!("module: {}", module.name)),
                    recommendation: Some("Improve module cohesion by grouping related functionality".to_string()),
                });
                score -= 0.1;
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
            check_name: "Business Context Integrity".to_string(),
            result,
            score: score.max(0.0) as f64,
            findings,
        })
    }

    /// Check AI metadata completeness
    fn check_ai_metadata_completeness(&self, pir: &PrismIR) -> PIRResult<CheckResult> {
        let mut findings = Vec::new();
        let mut score: f32 = 1.0;

        // Check module AI context
        if pir.ai_metadata.module_context.is_none() {
            findings.push(PreservationFinding {
                finding_type: FindingType::MissingAIMetadata,
                severity: Severity::Warning,
                description: "PIR has no module AI context".to_string(),
                location: Some("ai_metadata".to_string()),
                recommendation: Some("Add module AI context for better AI comprehension".to_string()),
            });
            score -= 0.2;
        }

        // Check function contexts
        let mut function_count = 0;
        for module in &pir.modules {
            for section in &module.sections {
                if let crate::semantic::PIRSection::Functions(func_section) = section {
                    function_count += func_section.functions.len();
                }
            }
        }

        if function_count > 0 && pir.ai_metadata.function_contexts.is_empty() {
            findings.push(PreservationFinding {
                finding_type: FindingType::MissingAIMetadata,
                severity: Severity::Info,
                description: "PIR has functions but no function AI contexts".to_string(),
                location: Some("ai_metadata.function_contexts".to_string()),
                recommendation: Some("Add function AI contexts for better AI understanding".to_string()),
            });
            score -= 0.1;
        }

        // Check type contexts
        if !pir.type_registry.types.is_empty() && pir.ai_metadata.type_contexts.is_empty() {
            findings.push(PreservationFinding {
                finding_type: FindingType::MissingAIMetadata,
                severity: Severity::Info,
                description: "PIR has types but no type AI contexts".to_string(),
                location: Some("ai_metadata.type_contexts".to_string()),
                recommendation: Some("Add type AI contexts for better AI comprehension".to_string()),
            });
            score -= 0.1;
        }

        let result = if score >= 0.9 {
            CheckStatus::Passed
        } else if score >= 0.7 {
            CheckStatus::PassedWithWarnings
        } else {
            CheckStatus::Failed
        };

        Ok(CheckResult {
            check_name: "AI Metadata Completeness".to_string(),
            result,
            score: score.max(0.0) as f64,
            findings,
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
            enable_domain_delegation: true,
        }
    }
}

impl Default for DomainValidatorIntegrations {
    fn default() -> Self {
        Self {
            semantic_integration: Some(SemanticValidatorIntegration {
                delegate_business_rules: true,
                delegate_semantic_constraints: true,
            }),
            constraint_integration: Some(ConstraintEngineIntegration {
                delegate_constraints: true,
                validation_context: None,
            }),
            effect_integration: Some(EffectValidatorIntegration {
                delegate_effects: true,
                validation_context: None,
            }),
        }
    }
}

/// Integration helper for connecting PIR validation with the compiler's validation orchestrator
/// 
/// This helper demonstrates the proper integration pattern where PIR validation
/// coordinates with existing domain validators rather than duplicating their logic.
pub struct PIRValidationIntegration;

impl PIRValidationIntegration {
    /// Create a PIR validator integrated with existing domain validators
    /// 
    /// This method shows how PIR validation should be configured to work with
    /// the compiler's validation orchestrator system.
    pub fn create_integrated_validator() -> PIRValidator {
        let config = ValidationConfig {
            strict_mode: false,
            validate_business_context: true,
            validate_effects: true,
            validate_ai_metadata: true,
            enable_domain_delegation: true,
        };

        let integrations = DomainValidatorIntegrations {
            semantic_integration: Some(SemanticValidatorIntegration {
                delegate_business_rules: true,
                delegate_semantic_constraints: true,
            }),
            constraint_integration: Some(ConstraintEngineIntegration {
                delegate_constraints: true,
                validation_context: Some("pir_validation_context".to_string()),
            }),
            effect_integration: Some(EffectValidatorIntegration {
                delegate_effects: true,
                validation_context: Some("pir_effect_context".to_string()),
            }),
        };

        PIRValidator::with_config(config).with_domain_integrations(integrations)
    }

    /// Demonstrate how PIR validation integrates with the compiler orchestrator
    /// 
    /// This method shows the delegation pattern where PIR validation coordinates
    /// with existing validators through the compiler's validation orchestrator.
    pub fn demonstrate_orchestrator_integration() -> String {
        r#"
        // In prism-compiler/src/lib.rs - Compiler construction
        
        let pir_validator = Arc::new(PIRValidationIntegration::create_integrated_validator());
        let semantic_validator = Arc::new(prism_semantic::SemanticValidator::new());
        let constraint_engine = Arc::new(prism_constraints::ConstraintEngine::new());
        let effect_validator = Arc::new(prism_effects::EffectValidator::new());
        
        // Create validator references for orchestrator
        let validator_refs = ValidatorIntegrationBuilder::new()
            .with_pir_validator(pir_validator)           // PIR validator handles PIR-specific concerns
            .with_semantic_validator(semantic_validator) // Semantic validator handles business rules
            .with_constraint_engine(constraint_engine)   // Constraint engine handles type constraints
            .with_effect_validator(effect_validator)     // Effect validator handles effect validation
            .build()?;
        
        let validation_orchestrator = ValidationOrchestrator::with_validator_refs(
            Arc::new(validator_refs)
        )?;
        
        // The orchestrator delegates to each validator for their domain expertise:
        // - PIR validator: PIR structure, semantic preservation coordination
        // - Semantic validator: Business rules, semantic constraints  
        // - Constraint engine: Type constraints, validation predicates
        // - Effect validator: Effect system validation, capability checking
        "#.to_string()
    }

    /// Show how PIR validation delegates to domain validators
    /// 
    /// This demonstrates the delegation pattern where PIR validation coordinates
    /// domain-specific validation without duplicating logic.
    pub fn demonstrate_delegation_pattern() -> String {
        r#"
        // PIR Validation Delegation Pattern:
        
        1. PIR Structure Validation (PIR-specific)
           - Validates PIR completeness and consistency
           - Checks module organization and sections
           - Verifies type registry integrity
           
        2. Semantic Preservation (Coordinates with semantic validator)
           - PIR checks: business capability preservation, semantic type structure
           - Delegates to semantic validator: business rule validation, semantic constraints
           
        3. Type Consistency (Coordinates with constraint engine)
           - PIR checks: type name uniqueness, relationship consistency
           - Delegates to constraint engine: type constraint validation, predicate evaluation
           
        4. Effect Preservation (Coordinates with effect validator)
           - PIR checks: effect graph structure, edge consistency
           - Delegates to effect validator: effect capability validation, security policies
           
        5. Business Context Integrity (PIR-specific)
           - Validates business context preservation
           - Checks cohesion metrics and domain classification
           
        6. AI Metadata Completeness (PIR-specific)
           - Ensures AI metadata is complete for external tool integration
           - Validates AI context information
        "#.to_string()
    }

    /// Example of proper validation delegation implementation
    /// 
    /// This shows how PIR validation would delegate to existing validators
    /// in a real implementation.
    pub fn example_delegation_implementation() -> String {
        r#"
        // Example: How PIR validator would delegate to semantic validator
        
        impl PIRValidator {
            fn delegate_business_rule_validation(&self, pir: &PrismIR) -> PIRResult<Vec<PreservationFinding>> {
                let mut findings = Vec::new();
                
                if let Some(semantic_integration) = &self.domain_integrations.semantic_integration {
                    if semantic_integration.delegate_business_rules {
                        // Extract business rules from PIR
                        let business_rules = self.extract_business_rules_from_pir(pir);
                        
                        // Delegate to existing semantic validator
                        // Note: This would require access to the semantic validator instance
                        // which would be provided through the integration system
                        
                        match self.semantic_validator.validate_business_rules(&business_rules) {
                            Ok(violations) => {
                                for violation in violations {
                                    findings.push(PreservationFinding {
                                        finding_type: FindingType::LostBusinessContext,
                                        severity: Severity::Warning,
                                        description: violation.description,
                                        location: Some(violation.location),
                                        recommendation: violation.suggested_fix,
                                    });
                                }
                            }
                            Err(e) => {
                                findings.push(PreservationFinding {
                                    finding_type: FindingType::MissingSemanticInfo,
                                    severity: Severity::Error,
                                    description: format!("Business rule validation failed: {}", e),
                                    location: None,
                                    recommendation: Some("Check business rule definitions".to_string()),
                                });
                            }
                        }
                    }
                }
                
                Ok(findings)
            }
        }
        "#.to_string()
    }
}

/// Helper for extracting validation data from PIR for delegation to domain validators
pub struct PIRValidationExtractor;

impl PIRValidationExtractor {
    /// Extract business rules from PIR for semantic validator delegation
    pub fn extract_business_rules(pir: &PrismIR) -> Vec<String> {
        let mut rules = Vec::new();
        
        for module in &pir.modules {
            // Extract module-level business rules
            for rule in &module.domain_rules {
                rules.push(rule.name.clone());
            }
            
            // Extract type-level business rules
            for section in &module.sections {
                if let crate::semantic::PIRSection::Types(type_section) = section {
                    for pir_type in &type_section.types {
                        for business_rule in &pir_type.business_rules {
                            rules.push(business_rule.name.clone());
                        }
                    }
                }
            }
        }
        
        rules
    }

    /// Extract type constraints from PIR for constraint engine delegation
    pub fn extract_type_constraints(pir: &PrismIR) -> Vec<String> {
        let mut constraints = Vec::new();
        
        // Extract from type registry
        for constraint in &pir.type_registry.global_constraints {
            constraints.push(format!("{:?}", constraint));
        }
        
        // Extract from individual types
        for (_, pir_type) in &pir.type_registry.types {
            for constraint in &pir_type.constraints {
                constraints.push(format!("{:?}", constraint));
            }
        }
        
        constraints
    }

    /// Extract effects from PIR for effect validator delegation
    pub fn extract_effects(pir: &PrismIR) -> Vec<String> {
        let mut effects = Vec::new();
        
        // Extract from effect graph
        for (effect_name, _) in &pir.effect_graph.nodes {
            effects.push(effect_name.clone());
        }
        
        // Extract from modules
        for module in &pir.modules {
            for effect in &module.effects {
                effects.push(effect.name.clone());
            }
        }
        
        effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = PIRValidator::new();
        assert!(!validator.rules.is_empty());
        assert!(validator.config.enable_domain_delegation);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::default();
        assert!(!config.strict_mode);
        assert!(config.validate_business_context);
        assert!(config.validate_effects);
        assert!(config.validate_ai_metadata);
        assert!(config.enable_domain_delegation);
    }

    #[test]
    fn test_validation_result() {
        let pir = PrismIR::new();
        let validator = PIRValidator::new();
        let result = validator.validate_semantic_preservation(&pir).unwrap();
        
        // Should pass with warnings due to empty PIR
        assert!(matches!(result.status, ValidationStatus::Passed | ValidationStatus::PassedWithWarnings));
        assert!(!result.check_results.is_empty());
    }

    #[test]
    fn test_pir_structure_validation() {
        let validator = PIRValidator::new();
        let empty_pir = PrismIR::new();
        
        let result = validator.check_pir_structure(&empty_pir).unwrap();
        
        // Should have warnings about empty structure
        assert!(matches!(result.result, CheckStatus::PassedWithWarnings | CheckStatus::Failed));
        assert!(!result.findings.is_empty());
    }

    #[test]
    fn test_domain_integration_configuration() {
        let integrations = DomainValidatorIntegrations::default();
        assert!(integrations.semantic_integration.is_some());
        assert!(integrations.constraint_integration.is_some());
        assert!(integrations.effect_integration.is_some());
        
        let validator = PIRValidator::new().with_domain_integrations(integrations);
        assert!(validator.config.enable_domain_delegation);
    }

    #[test]
    fn test_pir_validation_integration_helper() {
        let validator = PIRValidationIntegration::create_integrated_validator();
        assert!(validator.config.enable_domain_delegation);
        assert!(validator.domain_integrations.semantic_integration.is_some());
        assert!(validator.domain_integrations.constraint_integration.is_some());
        assert!(validator.domain_integrations.effect_integration.is_some());
    }

    #[test]
    fn test_validation_extractor() {
        let pir = PrismIR::new();
        
        let business_rules = PIRValidationExtractor::extract_business_rules(&pir);
        let type_constraints = PIRValidationExtractor::extract_type_constraints(&pir);
        let effects = PIRValidationExtractor::extract_effects(&pir);
        
        // For empty PIR, all should be empty
        assert!(business_rules.is_empty());
        assert!(type_constraints.is_empty());
        assert!(effects.is_empty());
    }

    #[test]
    fn test_validation_finding_types() {
        use FindingType::*;
        
        // Test that all finding types are available
        let finding_types = vec![
            MissingSemanticInfo,
            InconsistentTypes,
            LostBusinessContext,
            BrokenEffects,
            MissingAIMetadata,
            PIRStructureIssue,
        ];
        
        assert_eq!(finding_types.len(), 6);
    }

    #[test]
    fn test_validation_severity_ordering() {
        use Severity::*;
        
        assert!(Critical > Error);
        assert!(Error > Warning);
        assert!(Warning > Info);
    }

    #[test]
    fn test_rule_priority_ordering() {
        use RulePriority::*;
        
        assert!(Critical > High);
        assert!(High > Medium);
        assert!(Medium > Low);
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ValidationResult {
            status: ValidationStatus::Passed,
            check_results: vec![],
            overall_score: 1.0,
            metadata: ValidationMetadata {
                timestamp: "2024-01-01T00:00:00Z".to_string(),
                validator_version: "1.0.0".to_string(),
                duration_ms: 100,
                pir_version: "1.0.0".to_string(),
            },
        };

        // Test that it can be serialized/deserialized
        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ValidationResult = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(result.status, deserialized.status);
        assert_eq!(result.overall_score, deserialized.overall_score);
    }
} 