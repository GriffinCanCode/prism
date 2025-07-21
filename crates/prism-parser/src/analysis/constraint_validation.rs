//! Constraint Validation for Semantic-Aware Parsing
//!
//! This module implements validation of semantic constraints during parsing,
//! enabling business rule enforcement and type-guided error recovery.

use crate::core::{ParseError, ParseResult};
use prism_ast::{AstNode, Type, Expr, Constraint, BusinessRule};
use prism_common::{span::Span, NodeId};
use std::collections::HashMap;

/// Configuration for constraint validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable business rule validation
    pub enable_business_rules: bool,
    /// Enable type constraint validation
    pub enable_type_constraints: bool,
    /// Enable effect constraint validation
    pub enable_effect_constraints: bool,
    /// Maximum validation depth
    pub max_validation_depth: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_business_rules: true,
            enable_type_constraints: true,
            enable_effect_constraints: true,
            max_validation_depth: 10,
        }
    }
}

/// Result of constraint validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors found
    pub errors: Vec<ValidationError>,
    /// Warnings generated
    pub warnings: Vec<ValidationWarning>,
    /// Suggested fixes
    pub suggestions: Vec<ValidationSuggestion>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Location of error
    pub span: Span,
    /// Type of constraint violated
    pub constraint_type: ConstraintType,
    /// Severity of violation
    pub severity: ValidationSeverity,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Location of warning
    pub span: Span,
    /// Type of potential issue
    pub warning_type: WarningType,
}

/// Validation suggestion for fixing issues
#[derive(Debug, Clone)]
pub struct ValidationSuggestion {
    /// Description of suggestion
    pub description: String,
    /// Location to apply suggestion
    pub span: Span,
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Confidence in suggestion (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of constraints
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Type constraint (e.g., range, format)
    TypeConstraint,
    /// Business rule constraint
    BusinessRule,
    /// Effect constraint (capabilities, side effects)
    EffectConstraint,
    /// Semantic constraint (meaning, cohesion)
    SemanticConstraint,
}

/// Validation severity levels
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

/// Types of warnings
#[derive(Debug, Clone)]
pub enum WarningType {
    PotentialTypeIssue,
    BusinessRuleViolation,
    PerformanceIssue,
    CohesionIssue,
}

/// Types of suggestions
#[derive(Debug, Clone)]
pub enum SuggestionType {
    TypeAnnotation,
    BusinessRuleCompliance,
    EffectDeclaration,
    SemanticImprovement,
}

/// Constraint validator that performs semantic validation
pub struct ConstraintValidator {
    /// Configuration
    config: ValidationConfig,
    /// Known business rules
    business_rules: HashMap<String, BusinessRule>,
    /// Type constraint database
    type_constraints: HashMap<NodeId, Vec<Constraint>>,
    /// Validation cache for performance
    validation_cache: HashMap<NodeId, ValidationResult>,
}

impl ConstraintValidator {
    /// Create a new constraint validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            business_rules: HashMap::new(),
            type_constraints: HashMap::new(),
            validation_cache: HashMap::new(),
        }
    }

    /// Validate constraints for an AST node
    pub fn validate_node(&mut self, node: &AstNode<Type>) -> ValidationResult {
        // Check cache first
        if let Some(cached_result) = self.validation_cache.get(&node.id) {
            return cached_result.clone();
        }

        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        // Validate type constraints
        if self.config.enable_type_constraints {
            self.validate_type_constraints(node, &mut result);
        }

        // Validate business rules
        if self.config.enable_business_rules {
            self.validate_business_rules(node, &mut result);
        }

        // Validate effect constraints
        if self.config.enable_effect_constraints {
            self.validate_effect_constraints(node, &mut result);
        }

        // Update validity based on errors
        result.is_valid = result.errors.is_empty();

        // Cache result
        self.validation_cache.insert(node.id, result.clone());

        result
    }

    /// Validate type-specific constraints
    fn validate_type_constraints(&self, node: &AstNode<Type>, result: &mut ValidationResult) {
        match &node.kind {
            Type::Integer(int_type) => {
                // Validate integer constraints (range, etc.)
                if let Some(constraints) = self.type_constraints.get(&node.id) {
                    for constraint in constraints {
                        match constraint {
                            Constraint::Range { min, max } => {
                                // Check if integer type respects range constraints
                                if let Some(value) = int_type.get_literal_value() {
                                    if let Some(min_val) = min {
                                        if value < *min_val {
                                            result.errors.push(ValidationError {
                                                message: format!("Value {} is below minimum {}", value, min_val),
                                                span: node.span,
                                                constraint_type: ConstraintType::TypeConstraint,
                                                severity: ValidationSeverity::Error,
                                            });
                                        }
                                    }
                                    if let Some(max_val) = max {
                                        if value > *max_val {
                                            result.errors.push(ValidationError {
                                                message: format!("Value {} is above maximum {}", value, max_val),
                                                span: node.span,
                                                constraint_type: ConstraintType::TypeConstraint,
                                                severity: ValidationSeverity::Error,
                                            });
                                        }
                                    }
                                }
                            }
                            Constraint::Format { pattern } => {
                                // Validate format constraints
                                result.suggestions.push(ValidationSuggestion {
                                    description: format!("Consider validating format against pattern: {}", pattern),
                                    span: node.span,
                                    suggestion_type: SuggestionType::TypeAnnotation,
                                    confidence: 0.7,
                                });
                            }
                        }
                    }
                }
            }
            Type::String(str_type) => {
                // Validate string constraints (length, format, etc.)
                if let Some(constraints) = self.type_constraints.get(&node.id) {
                    for constraint in constraints {
                        match constraint {
                            Constraint::Length { min, max } => {
                                if let Some(value) = str_type.get_literal_value() {
                                    let len = value.len();
                                    if let Some(min_len) = min {
                                        if len < *min_len {
                                            result.errors.push(ValidationError {
                                                message: format!("String length {} is below minimum {}", len, min_len),
                                                span: node.span,
                                                constraint_type: ConstraintType::TypeConstraint,
                                                severity: ValidationSeverity::Error,
                                            });
                                        }
                                    }
                                    if let Some(max_len) = max {
                                        if len > *max_len {
                                            result.errors.push(ValidationError {
                                                message: format!("String length {} is above maximum {}", len, max_len),
                                                span: node.span,
                                                constraint_type: ConstraintType::TypeConstraint,
                                                severity: ValidationSeverity::Error,
                                            });
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Type::Semantic(sem_type) => {
                // Validate semantic type constraints
                self.validate_semantic_type_constraints(sem_type, node.span, result);
            }
            _ => {}
        }
    }

    /// Validate business rule constraints
    fn validate_business_rules(&self, node: &AstNode<Type>, result: &mut ValidationResult) {
        // Check if type is subject to business rules
        if let Type::Semantic(sem_type) = &node.kind {
            if let Some(business_domain) = &sem_type.business_domain {
                // Look for applicable business rules
                for (rule_name, rule) in &self.business_rules {
                    if rule.applies_to_domain(business_domain) {
                        if !rule.validate_type(&node.kind) {
                            result.errors.push(ValidationError {
                                message: format!("Business rule '{}' violated", rule_name),
                                span: node.span,
                                constraint_type: ConstraintType::BusinessRule,
                                severity: ValidationSeverity::Error,
                            });

                            // Generate suggestion for compliance
                            result.suggestions.push(ValidationSuggestion {
                                description: rule.get_compliance_suggestion(),
                                span: node.span,
                                suggestion_type: SuggestionType::BusinessRuleCompliance,
                                confidence: 0.8,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Validate effect constraints
    fn validate_effect_constraints(&self, node: &AstNode<Type>, result: &mut ValidationResult) {
        if let Type::Effect(effect_type) = &node.kind {
            // Validate capability requirements
            for required_capability in &effect_type.required_capabilities {
                if !self.is_capability_available(required_capability) {
                    result.errors.push(ValidationError {
                        message: format!("Required capability '{}' is not available", required_capability),
                        span: node.span,
                        constraint_type: ConstraintType::EffectConstraint,
                        severity: ValidationSeverity::Error,
                    });

                    result.suggestions.push(ValidationSuggestion {
                        description: format!("Add capability declaration for '{}'", required_capability),
                        span: node.span,
                        suggestion_type: SuggestionType::EffectDeclaration,
                        confidence: 0.9,
                    });
                }
            }

            // Validate effect composition rules
            if effect_type.effects.len() > 1 {
                if !self.are_effects_composable(&effect_type.effects) {
                    result.warnings.push(ValidationWarning {
                        message: "Effects may not compose safely".to_string(),
                        span: node.span,
                        warning_type: WarningType::PotentialTypeIssue,
                    });
                }
            }
        }
    }

    /// Validate semantic type constraints
    fn validate_semantic_type_constraints(
        &self,
        sem_type: &prism_ast::SemanticType,
        span: Span,
        result: &mut ValidationResult,
    ) {
        // Validate conceptual cohesion
        if let Some(cohesion_score) = sem_type.cohesion_score {
            if cohesion_score < 0.7 {
                result.warnings.push(ValidationWarning {
                    message: format!("Low conceptual cohesion score: {:.2}", cohesion_score),
                    span,
                    warning_type: WarningType::CohesionIssue,
                });

                result.suggestions.push(ValidationSuggestion {
                    description: "Consider refactoring to improve conceptual cohesion".to_string(),
                    span,
                    suggestion_type: SuggestionType::SemanticImprovement,
                    confidence: 0.6,
                });
            }
        }

        // Validate business meaning consistency
        if let Some(business_meaning) = &sem_type.business_meaning {
            if !self.is_business_meaning_consistent(business_meaning) {
                result.warnings.push(ValidationWarning {
                    message: "Business meaning may be inconsistent with context".to_string(),
                    span,
                    warning_type: WarningType::BusinessRuleViolation,
                });
            }
        }
    }

    // Helper methods

    fn is_capability_available(&self, _capability: &str) -> bool {
        // Check if capability is available in current context
        // This would integrate with the capability system
        true // Placeholder
    }

    fn are_effects_composable(&self, _effects: &[prism_ast::Effect]) -> bool {
        // Check if effects can be safely composed
        // This would integrate with the effect system
        true // Placeholder
    }

    fn is_business_meaning_consistent(&self, _meaning: &str) -> bool {
        // Check if business meaning is consistent with context
        // This would integrate with business rule system
        true // Placeholder
    }

    /// Add a business rule to the validator
    pub fn add_business_rule(&mut self, name: String, rule: BusinessRule) {
        self.business_rules.insert(name, rule);
    }

    /// Add type constraints for a node
    pub fn add_type_constraints(&mut self, node_id: NodeId, constraints: Vec<Constraint>) {
        self.type_constraints.insert(node_id, constraints);
    }

    /// Clear validation cache
    pub fn clear_cache(&mut self) {
        self.validation_cache.clear();
    }
} 