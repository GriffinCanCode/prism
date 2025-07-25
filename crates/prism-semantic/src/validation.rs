//! Semantic Validation Engine
//!
//! This module embodies the single concept of "Semantic Validation".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: validating semantic constraints and business rules.
//!
//! **Conceptual Responsibility**: Semantic constraint validation
//! **What it does**: business rule validation, constraint checking, compliance verification
//! **What it doesn't do**: type inference, semantic analysis, pattern recognition

use crate::{SemanticResult, SemanticError, SemanticConfig};
use crate::type_inference::constraints::ConstraintSet;
use crate::analyzer::AnalysisResult;
use crate::types::BusinessRule;
use prism_ast::Program;
use prism_common::{span::{Span, Position}, SourceId};
use prism_constraints::{ConstraintEngine, ConstraintValue, ValidationContext};
use serde::{Serialize, Deserialize};

/// Specification for constraints to be validated in the semantic layer
#[derive(Debug, Clone)]
pub enum ConstraintSpec {
    /// Range constraint
    Range {
        value: ConstraintValue,
        min: Option<ConstraintValue>,
        max: Option<ConstraintValue>,
        inclusive: bool,
    },
    /// Pattern constraint
    Pattern {
        value: ConstraintValue,
        pattern: String,
    },
    /// Length constraint
    Length {
        value: ConstraintValue,
        min_length: Option<usize>,
        max_length: Option<usize>,
    },
    /// Format constraint
    Format {
        value: ConstraintValue,
        format_name: String,
    },
}

/// Semantic validator for business rules and constraints
#[derive(Debug)]
pub struct SemanticValidator {
    /// Configuration
    config: ValidationConfig,
    /// Business rule engine
    rule_engine: BusinessRuleEngine,
    /// Constraint checker
    constraint_checker: ConstraintChecker,
    /// Constraint engine for actual validation
    constraint_engine: ConstraintEngine,
}

/// Configuration for semantic validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable business rule validation
    pub enable_business_rules: bool,
    /// Enable constraint validation
    pub enable_constraints: bool,
    /// Enable compliance checking
    pub enable_compliance: bool,
    /// Validation strictness level
    pub strictness: ValidationStrictness,
}

/// Validation strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStrictness {
    /// Lenient validation
    Lenient,
    /// Standard validation
    Standard,
    /// Strict validation
    Strict,
    /// Pedantic validation
    Pedantic,
}

/// Result of semantic validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation passed
    pub passed: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Business rule violations
    pub rule_violations: Vec<RuleViolation>,
    /// Constraint violations
    pub constraint_violations: Vec<ConstraintViolation>,
    /// Validation metadata
    pub metadata: ValidationMetadata,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Error location
    pub location: Span,
    /// Error code
    pub code: String,
    /// Severity level
    pub severity: ErrorSeverity,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Warning location
    pub location: Span,
    /// Warning code
    pub code: String,
}

/// Business rule violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    /// Rule that was violated
    pub rule: BusinessRule,
    /// Violation description
    pub description: String,
    /// Location of violation
    pub location: Span,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Constraint that was violated
    pub constraint: String,
    /// Violation description
    pub description: String,
    /// Location of violation
    pub location: Span,
    /// Actual value
    pub actual_value: Option<String>,
    /// Expected value
    pub expected_value: Option<String>,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Critical error
    Critical,
    /// High severity error
    High,
    /// Medium severity error
    Medium,
    /// Low severity error
    Low,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Validation timestamp
    pub timestamp: String,
    /// Validation duration
    pub duration_ms: u64,
    /// Rules checked
    pub rules_checked: usize,
    /// Constraints checked
    pub constraints_checked: usize,
}

/// Business rule engine
#[derive(Debug)]
pub struct BusinessRuleEngine {
    /// Registered rules
    rules: Vec<BusinessRule>,
}

/// Constraint checker
#[derive(Debug)]
pub struct ConstraintChecker {
    /// Enable constraint checking
    enabled: bool,
}

impl SemanticValidator {
    /// Create a new semantic validator
    pub fn new(config: &SemanticConfig) -> SemanticResult<Self> {
        let validation_config = ValidationConfig {
            enable_business_rules: config.enable_business_rules,
            enable_constraints: true,
            enable_compliance: true,
            strictness: ValidationStrictness::Standard,
        };

        Ok(Self {
            config: validation_config,
            rule_engine: BusinessRuleEngine {
                rules: Vec::new(),
            },
            constraint_checker: ConstraintChecker {
                enabled: true,
            },
            constraint_engine: ConstraintEngine::new(),
        })
    }

    /// Validate a program
    pub fn validate_program(&mut self, _program: &Program, _analysis: &AnalysisResult) -> SemanticResult<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Validation implementation would go here
        // This is a stub for now
        
        let duration = start_time.elapsed();
        
        Ok(ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            rule_violations: Vec::new(),
            constraint_violations: Vec::new(),
            metadata: ValidationMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                duration_ms: duration.as_millis() as u64,
                rules_checked: 0,
                constraints_checked: 0,
            },
        })
    }

    /// Validate business rules
    pub fn validate_business_rules(&self, _rules: &[BusinessRule]) -> SemanticResult<Vec<RuleViolation>> {
        // Business rule validation would go here
        Ok(Vec::new())
    }

    /// Validate constraints
    pub fn validate_constraints(&self, constraints: &[ConstraintSpec]) -> SemanticResult<Vec<ConstraintViolation>> {
        let mut violations = Vec::new();
        let context = ValidationContext::new();
        
        for constraint in constraints {
            match constraint {
                ConstraintSpec::Range { value, min, max, inclusive } => {
                                    let result = self.constraint_engine.validate_range(
                    value, min.clone(), max.clone(), *inclusive, &context
                ).map_err(|e| SemanticError::ValidationError { 
                    message: e.to_string()
                })?;
                    
                    if !result.is_valid {
                        for error in result.errors {
                            violations.push(ConstraintViolation {
                                constraint: error.constraint_type,
                                description: error.message,
                                location: error.location.unwrap_or_else(|| Span::new(
                                    Position::new(1, 1, 0), 
                                    Position::new(1, 1, 0), 
                                    SourceId::new(0)
                                )),
                                actual_value: error.actual,
                                expected_value: error.expected,
                            });
                        }
                    }
                }
                ConstraintSpec::Pattern { value, pattern } => {
                    let result = self.constraint_engine.validate_pattern(
                        value, pattern, &context
                    ).map_err(|e| SemanticError::ValidationError { 
                        message: e.to_string()
                    })?;
                    
                    if !result.is_valid {
                        for error in result.errors {
                            violations.push(ConstraintViolation {
                                constraint: error.constraint_type,
                                description: error.message,
                                location: error.location.unwrap_or_else(|| Span::new(
                                    Position::new(1, 1, 0), 
                                    Position::new(1, 1, 0), 
                                    SourceId::new(0)
                                )),
                                actual_value: error.actual,
                                expected_value: error.expected,
                            });
                        }
                    }
                }
                ConstraintSpec::Length { value, min_length, max_length } => {
                    let result = self.constraint_engine.validate_length(
                        value, *min_length, *max_length, &context
                    ).map_err(|e| SemanticError::ValidationError { 
                        message: e.to_string()
                    })?;
                    
                    if !result.is_valid {
                        for error in result.errors {
                            violations.push(ConstraintViolation {
                                constraint: error.constraint_type,
                                description: error.message,
                                location: error.location.unwrap_or_else(|| Span::new(
                                    Position::new(1, 1, 0), 
                                    Position::new(1, 1, 0), 
                                    SourceId::new(0)
                                )),
                                actual_value: error.actual,
                                expected_value: error.expected,
                            });
                        }
                    }
                }
                ConstraintSpec::Format { value, format_name } => {
                    let result = self.constraint_engine.validate_format(
                        value, format_name, &context
                    ).map_err(|e| SemanticError::ValidationError { 
                        message: e.to_string()
                    })?;
                    
                    if !result.is_valid {
                        for error in result.errors {
                            violations.push(ConstraintViolation {
                                constraint: error.constraint_type,
                                description: error.message,
                                location: error.location.unwrap_or_else(|| Span::new(
                                    Position::new(1, 1, 0), 
                                    Position::new(1, 1, 0), 
                                    SourceId::new(0)
                                )),
                                actual_value: error.actual,
                                expected_value: error.expected,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(violations)
    }

    /// Get access to the constraint engine for advanced validation
    pub fn constraint_engine(&self) -> &ConstraintEngine {
        &self.constraint_engine
    }
} 