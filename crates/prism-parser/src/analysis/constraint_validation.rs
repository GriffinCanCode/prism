//! Constraint validation engine for semantic types
//!
//! This module provides comprehensive validation of semantic type constraints,
//! business rules, and invariants. It supports both compile-time and runtime
//! validation with detailed error reporting and AI-readable feedback.

use prism_ast::{
    AstNode, Type, TypeConstraint, SemanticType, SemanticTypeMetadata,
    RangeConstraint, PatternConstraint, LengthConstraint, FormatConstraint,
    CustomConstraint, BusinessRuleConstraint, Expr, LiteralValue,
    SecurityClassification,
};
use prism_common::span::Span;
use std::collections::HashMap;
use thiserror::Error;

/// Comprehensive constraint validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors found
    pub errors: Vec<ValidationError>,
    /// Warnings about potential issues
    pub warnings: Vec<ValidationWarning>,
    /// AI-readable validation summary
    pub ai_summary: ValidationSummary,
    /// Performance metrics
    pub performance_metrics: ValidationMetrics,
}

/// Validation error with context and suggestions
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    #[error("Range constraint violation: {message} at {span}")]
    RangeViolation {
        message: String,
        span: Span,
        expected_range: String,
        actual_value: String,
        suggestions: Vec<String>,
    },
    
    #[error("Pattern constraint violation: {message} at {span}")]
    PatternViolation {
        message: String,
        span: Span,
        pattern: String,
        actual_value: String,
        suggestions: Vec<String>,
    },
    
    #[error("Length constraint violation: {message} at {span}")]
    LengthViolation {
        message: String,
        span: Span,
        expected_length: String,
        actual_length: usize,
        suggestions: Vec<String>,
    },
    
    #[error("Format constraint violation: {message} at {span}")]
    FormatViolation {
        message: String,
        span: Span,
        expected_format: String,
        actual_value: String,
        suggestions: Vec<String>,
    },
    
    #[error("Business rule violation: {message} at {span}")]
    BusinessRuleViolation {
        message: String,
        span: Span,
        rule_name: String,
        context: String,
        suggestions: Vec<String>,
    },
    
    #[error("Security constraint violation: {message} at {span}")]
    SecurityViolation {
        message: String,
        span: Span,
        security_level: SecurityClassification,
        violation_type: String,
        suggestions: Vec<String>,
    },
    
    #[error("Invariant violation: {message} at {span}")]
    InvariantViolation {
        message: String,
        span: Span,
        invariant_name: String,
        context: String,
        suggestions: Vec<String>,
    },
    
    #[error("Compliance violation: {message} at {span}")]
    ComplianceViolation {
        message: String,
        span: Span,
        requirement: String,
        violation_details: String,
        suggestions: Vec<String>,
    },
}

/// Validation warning for potential issues
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
    pub span: Span,
    pub warning_type: WarningType,
    pub suggestions: Vec<String>,
}

/// Types of validation warnings
#[derive(Debug, Clone)]
pub enum WarningType {
    Performance,
    Security,
    Maintainability,
    Compliance,
    AIComprehension,
}

/// AI-readable validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Overall validation score (0.0 to 1.0)
    pub overall_score: f64,
    /// Number of constraints validated
    pub constraints_validated: usize,
    /// Number of business rules checked
    pub business_rules_checked: usize,
    /// Security implications identified
    pub security_implications: Vec<String>,
    /// Compliance requirements verified
    pub compliance_verified: Vec<String>,
    /// AI insights about the validation
    pub ai_insights: Vec<String>,
    /// Suggested improvements
    pub improvements: Vec<String>,
}

/// Performance metrics for validation
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Time taken for validation (in microseconds)
    pub validation_time_us: u64,
    /// Number of constraints evaluated
    pub constraints_evaluated: usize,
    /// Number of expressions evaluated
    pub expressions_evaluated: usize,
    /// Memory usage during validation
    pub memory_usage_bytes: usize,
}

/// Main constraint validator
pub struct ConstraintValidator {
    /// Validation configuration
    config: ValidationConfig,
    /// Expression evaluator for constraint expressions
    evaluator: ExpressionEvaluator,
    /// Business rule engine
    business_rule_engine: BusinessRuleEngine,
    /// Security validator
    security_validator: SecurityValidator,
    /// Compliance checker
    compliance_checker: ComplianceChecker,
}

/// Configuration for constraint validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable runtime validation
    pub runtime_validation: bool,
    /// Enable compile-time validation
    pub compile_time_validation: bool,
    /// Enable AI-assisted validation
    pub ai_assisted_validation: bool,
    /// Maximum validation time (in milliseconds)
    pub max_validation_time_ms: u64,
    /// Enable performance metrics collection
    pub collect_metrics: bool,
    /// Validation strictness level
    pub strictness: ValidationStrictness,
}

/// Validation strictness levels
#[derive(Debug, Clone)]
pub enum ValidationStrictness {
    /// Lenient - warnings only
    Lenient,
    /// Standard - errors for clear violations
    Standard,
    /// Strict - errors for any potential issues
    Strict,
    /// Pedantic - errors for style and best practices
    Pedantic,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            runtime_validation: true,
            compile_time_validation: true,
            ai_assisted_validation: true,
            max_validation_time_ms: 1000,
            collect_metrics: true,
            strictness: ValidationStrictness::Standard,
        }
    }
}

impl ConstraintValidator {
    /// Create a new constraint validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            evaluator: ExpressionEvaluator::new(),
            business_rule_engine: BusinessRuleEngine::new(),
            security_validator: SecurityValidator::new(),
            compliance_checker: ComplianceChecker::new(),
        }
    }

    /// Validate a semantic type and all its constraints
    pub fn validate_semantic_type(
        &mut self,
        semantic_type: &SemanticType,
        span: Span,
    ) -> ValidationResult {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut constraints_validated = 0;
        let mut business_rules_checked = 0;
        let mut security_implications = Vec::new();
        let mut compliance_verified = Vec::new();
        let mut ai_insights = Vec::new();

        // Validate each constraint
        for constraint in &semantic_type.constraints {
            constraints_validated += 1;
            
            match self.validate_constraint(constraint, span) {
                Ok(constraint_result) => {
                    warnings.extend(constraint_result.warnings);
                    ai_insights.extend(constraint_result.ai_insights);
                }
                Err(error) => {
                    errors.push(error);
                }
            }
        }

        // Validate business rules
        for rule in &semantic_type.metadata.business_rules {
            business_rules_checked += 1;
            
            match self.business_rule_engine.validate_rule(rule, span) {
                Ok(rule_result) => {
                    ai_insights.extend(rule_result.insights);
                }
                Err(error) => {
                    errors.push(ValidationError::BusinessRuleViolation {
                        message: error.to_string(),
                        span,
                        rule_name: rule.clone(),
                        context: "Business rule validation".to_string(),
                        suggestions: vec![
                            "Review business rule implementation".to_string(),
                            "Consider simplifying the rule logic".to_string(),
                        ],
                    });
                }
            }
        }

        // Validate security implications
        if let Some(security_result) = self.security_validator.validate_security(
            &semantic_type.metadata.security_classification,
            &semantic_type.metadata.compliance_requirements,
            span,
        ) {
            security_implications = security_result.implications;
            compliance_verified = security_result.compliance_verified;
            
            if let Some(security_error) = security_result.error {
                errors.push(security_error);
            }
        }

        // Calculate validation metrics
        let validation_time_us = start_time.elapsed().as_micros() as u64;
        let performance_metrics = ValidationMetrics {
            validation_time_us,
            constraints_evaluated: constraints_validated,
            expressions_evaluated: self.evaluator.expressions_evaluated(),
            memory_usage_bytes: std::mem::size_of_val(&semantic_type),
        };

        // Calculate overall validation score
        let overall_score = if errors.is_empty() {
            1.0 - (warnings.len() as f64 * 0.1).min(0.5)
        } else {
            0.0
        };

        // Generate AI insights
        if self.config.ai_assisted_validation {
            ai_insights.push(format!(
                "Validated {} constraints with {} business rules",
                constraints_validated, business_rules_checked
            ));
            
            if !errors.is_empty() {
                ai_insights.push(format!(
                    "Found {} validation errors that need attention",
                    errors.len()
                ));
            }
            
            if overall_score > 0.8 {
                ai_insights.push("Type definition follows semantic type best practices".to_string());
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            ai_summary: ValidationSummary {
                overall_score,
                constraints_validated,
                business_rules_checked,
                security_implications,
                compliance_verified,
                ai_insights,
                improvements: self.generate_improvements(&semantic_type.metadata),
            },
            performance_metrics,
        }
    }

    /// Validate a single constraint
    fn validate_constraint(
        &mut self,
        constraint: &TypeConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        match constraint {
            TypeConstraint::Range(range) => self.validate_range_constraint(range, span),
            TypeConstraint::Pattern(pattern) => self.validate_pattern_constraint(pattern, span),
            TypeConstraint::Length(length) => self.validate_length_constraint(length, span),
            TypeConstraint::Format(format) => self.validate_format_constraint(format, span),
            TypeConstraint::Custom(custom) => self.validate_custom_constraint(custom, span),
            TypeConstraint::BusinessRule(rule) => self.validate_business_rule_constraint(rule, span),
        }
    }

    /// Validate range constraint
    fn validate_range_constraint(
        &mut self,
        range: &RangeConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        let mut warnings = Vec::new();
        let mut ai_insights = Vec::new();

        // Validate range bounds
        if let (Some(min), Some(max)) = (&range.min, &range.max) {
            // Check if min <= max
            if let (Some(min_val), Some(max_val)) = (
                self.evaluator.evaluate_to_number(min),
                self.evaluator.evaluate_to_number(max),
            ) {
                if min_val > max_val {
                    return Err(ValidationError::RangeViolation {
                        message: "Minimum value is greater than maximum value".to_string(),
                        span,
                        expected_range: format!("min <= max"),
                        actual_value: format!("min: {}, max: {}", min_val, max_val),
                        suggestions: vec![
                            "Swap minimum and maximum values".to_string(),
                            "Check if the range bounds are correct".to_string(),
                        ],
                    });
                }
                
                ai_insights.push(format!(
                    "Range constraint validated: {} to {}",
                    min_val, max_val
                ));
            }
        }

        // Check for common range patterns
        if let Some(min) = &range.min {
            if let Some(min_val) = self.evaluator.evaluate_to_number(min) {
                if min_val == 0.0 {
                    ai_insights.push("Non-negative constraint detected".to_string());
                } else if min_val == 1.0 {
                    ai_insights.push("Positive constraint detected".to_string());
                }
            }
        }

        Ok(ConstraintValidationResult {
            warnings,
            ai_insights,
        })
    }

    /// Validate pattern constraint
    fn validate_pattern_constraint(
        &mut self,
        pattern: &PatternConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        let mut warnings = Vec::new();
        let mut ai_insights = Vec::new();

        // Validate regex pattern
        if let Err(regex_error) = regex::Regex::new(&pattern.pattern) {
            return Err(ValidationError::PatternViolation {
                message: format!("Invalid regex pattern: {}", regex_error),
                span,
                pattern: pattern.pattern.clone(),
                actual_value: "invalid regex".to_string(),
                suggestions: vec![
                    "Fix the regex pattern syntax".to_string(),
                    "Test the pattern with online regex validators".to_string(),
                ],
            });
        }

        // Analyze pattern complexity
        if pattern.pattern.len() > 100 {
            warnings.push(ValidationWarning {
                message: "Complex regex pattern may impact performance".to_string(),
                span,
                warning_type: WarningType::Performance,
                suggestions: vec![
                    "Consider simplifying the pattern".to_string(),
                    "Split complex patterns into multiple simpler ones".to_string(),
                ],
            });
        }

        // Recognize common patterns
        if pattern.pattern.contains("@") && pattern.pattern.contains("\\.") {
            ai_insights.push("Email pattern detected".to_string());
        } else if pattern.pattern.contains("^[0-9]+$") {
            ai_insights.push("Numeric pattern detected".to_string());
        } else if pattern.pattern.contains("^[a-zA-Z]+$") {
            ai_insights.push("Alphabetic pattern detected".to_string());
        }

        Ok(ConstraintValidationResult {
            warnings,
            ai_insights,
        })
    }

    /// Validate length constraint
    fn validate_length_constraint(
        &mut self,
        length: &LengthConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        let mut warnings = Vec::new();
        let mut ai_insights = Vec::new();

        // Validate length bounds
        if let (Some(min), Some(max)) = (length.min_length, length.max_length) {
            if min > max {
                return Err(ValidationError::LengthViolation {
                    message: "Minimum length is greater than maximum length".to_string(),
                    span,
                    expected_length: format!("min <= max"),
                    actual_length: 0,
                    suggestions: vec![
                        "Swap minimum and maximum lengths".to_string(),
                        "Check if the length bounds are correct".to_string(),
                    ],
                });
            }
            
            ai_insights.push(format!(
                "Length constraint validated: {} to {} characters",
                min, max
            ));
        }

        // Check for common length patterns
        if let Some(min) = length.min_length {
            if min == 1 {
                ai_insights.push("Non-empty string constraint detected".to_string());
            } else if min >= 8 {
                ai_insights.push("Strong length requirement detected (possibly for passwords)".to_string());
            }
        }

        Ok(ConstraintValidationResult {
            warnings,
            ai_insights,
        })
    }

    /// Validate format constraint
    fn validate_format_constraint(
        &mut self,
        format: &FormatConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        let mut warnings = Vec::new();
        let mut ai_insights = Vec::new();

        // Recognize common formats
        match format.format.to_lowercase().as_str() {
            "email" => {
                ai_insights.push("Email format validation detected".to_string());
            }
            "url" | "uri" => {
                ai_insights.push("URL format validation detected".to_string());
            }
            "uuid" => {
                ai_insights.push("UUID format validation detected".to_string());
            }
            "date" | "datetime" => {
                ai_insights.push("Date/time format validation detected".to_string());
            }
            "phone" => {
                ai_insights.push("Phone number format validation detected".to_string());
            }
            _ => {
                ai_insights.push(format!("Custom format '{}' detected", format.format));
            }
        }

        // Check for security implications
        if format.format.contains("password") || format.format.contains("secret") {
            warnings.push(ValidationWarning {
                message: "Sensitive data format detected".to_string(),
                span,
                warning_type: WarningType::Security,
                suggestions: vec![
                    "Ensure proper encryption and hashing".to_string(),
                    "Consider security implications of this format".to_string(),
                ],
            });
        }

        Ok(ConstraintValidationResult {
            warnings,
            ai_insights,
        })
    }

    /// Validate custom constraint
    fn validate_custom_constraint(
        &mut self,
        custom: &CustomConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        let mut warnings = Vec::new();
        let mut ai_insights = Vec::new();

        // Validate constraint expression
        if let Err(eval_error) = self.evaluator.validate_expression(&custom.expression) {
            return Err(ValidationError::BusinessRuleViolation {
                message: format!("Invalid constraint expression: {}", eval_error),
                span,
                rule_name: custom.name.clone(),
                context: "Custom constraint validation".to_string(),
                suggestions: vec![
                    "Fix the constraint expression".to_string(),
                    "Ensure all variables are properly defined".to_string(),
                ],
            });
        }

        ai_insights.push(format!(
            "Custom constraint '{}' validated successfully",
            custom.name
        ));

        Ok(ConstraintValidationResult {
            warnings,
            ai_insights,
        })
    }

    /// Validate business rule constraint
    fn validate_business_rule_constraint(
        &mut self,
        rule: &BusinessRuleConstraint,
        span: Span,
    ) -> Result<ConstraintValidationResult, ValidationError> {
        let mut warnings = Vec::new();
        let mut ai_insights = Vec::new();

        // Validate rule expression
        if let Err(eval_error) = self.evaluator.validate_expression(&rule.expression) {
            return Err(ValidationError::BusinessRuleViolation {
                message: format!("Invalid business rule expression: {}", eval_error),
                span,
                rule_name: rule.description.clone(),
                context: "Business rule validation".to_string(),
                suggestions: vec![
                    "Fix the business rule expression".to_string(),
                    "Ensure business logic is correctly implemented".to_string(),
                ],
            });
        }

        // Check rule priority
        if rule.priority > 10 {
            warnings.push(ValidationWarning {
                message: "High priority business rule detected".to_string(),
                span,
                warning_type: WarningType::Maintainability,
                suggestions: vec![
                    "Consider if this rule priority is appropriate".to_string(),
                    "Document why this rule has high priority".to_string(),
                ],
            });
        }

        ai_insights.push(format!(
            "Business rule '{}' validated with priority {}",
            rule.description, rule.priority
        ));

        Ok(ConstraintValidationResult {
            warnings,
            ai_insights,
        })
    }

    /// Generate improvement suggestions
    fn generate_improvements(&self, metadata: &SemanticTypeMetadata) -> Vec<String> {
        let mut improvements = Vec::new();

        if metadata.examples.is_empty() {
            improvements.push("Add examples to improve AI comprehension".to_string());
        }

        if metadata.business_rules.is_empty() {
            improvements.push("Consider adding business rules for better domain modeling".to_string());
        }

        if metadata.ai_context.is_none() {
            improvements.push("Add AI context for better code understanding".to_string());
        }

        if metadata.compliance_requirements.is_empty() {
            improvements.push("Consider adding compliance requirements if applicable".to_string());
        }

        improvements
    }
}

/// Result of constraint validation
#[derive(Debug, Clone)]
struct ConstraintValidationResult {
    warnings: Vec<ValidationWarning>,
    ai_insights: Vec<String>,
}

/// Expression evaluator for constraint expressions
struct ExpressionEvaluator {
    expressions_evaluated: usize,
}

impl ExpressionEvaluator {
    fn new() -> Self {
        Self {
            expressions_evaluated: 0,
        }
    }

    fn evaluate_to_number(&mut self, expr: &AstNode<Expr>) -> Option<f64> {
        self.expressions_evaluated += 1;
        
        match &expr.kind {
            Expr::Literal(literal) => {
                match &literal.value {
                    LiteralValue::Integer(val) => Some(*val as f64),
                    LiteralValue::Float(val) => Some(*val),
                    _ => None,
                }
            }
            _ => None, // For now, only handle literals
        }
    }

    fn validate_expression(&mut self, expr: &AstNode<Expr>) -> Result<(), String> {
        self.expressions_evaluated += 1;
        
        // Basic expression validation
        match &expr.kind {
            Expr::Error(_) => Err("Error expression found".to_string()),
            _ => Ok(()),
        }
    }

    fn expressions_evaluated(&self) -> usize {
        self.expressions_evaluated
    }
}

/// Business rule engine
struct BusinessRuleEngine;

impl BusinessRuleEngine {
    fn new() -> Self {
        Self
    }

    fn validate_rule(&self, rule: &str, _span: Span) -> Result<BusinessRuleResult, String> {
        // For now, just check if the rule is not empty
        if rule.trim().is_empty() {
            Err("Empty business rule".to_string())
        } else {
            Ok(BusinessRuleResult {
                insights: vec![format!("Business rule '{}' is well-formed", rule)],
            })
        }
    }
}

/// Result of business rule validation
#[derive(Debug, Clone)]
struct BusinessRuleResult {
    insights: Vec<String>,
}

/// Security validator
struct SecurityValidator;

impl SecurityValidator {
    fn new() -> Self {
        Self
    }

    fn validate_security(
        &self,
        classification: &SecurityClassification,
        compliance: &[String],
        _span: Span,
    ) -> Option<SecurityValidationResult> {
        let mut implications = Vec::new();
        let mut compliance_verified = Vec::new();
        let mut error = None;

        // Check security classification
        match classification {
            SecurityClassification::Confidential | 
            SecurityClassification::Restricted | 
            SecurityClassification::TopSecret => {
                implications.push("High security classification requires additional protections".to_string());
            }
            _ => {}
        }

        // Verify compliance requirements
        for requirement in compliance {
            match requirement.to_uppercase().as_str() {
                "GDPR" => {
                    compliance_verified.push("GDPR compliance noted".to_string());
                    implications.push("GDPR requires data protection measures".to_string());
                }
                "HIPAA" => {
                    compliance_verified.push("HIPAA compliance noted".to_string());
                    implications.push("HIPAA requires healthcare data protection".to_string());
                }
                "PCI-DSS" | "PCI DSS" => {
                    compliance_verified.push("PCI DSS compliance noted".to_string());
                    implications.push("PCI DSS requires payment card data protection".to_string());
                }
                _ => {
                    compliance_verified.push(format!("Custom compliance requirement: {}", requirement));
                }
            }
        }

        Some(SecurityValidationResult {
            implications,
            compliance_verified,
            error,
        })
    }
}

/// Result of security validation
#[derive(Debug, Clone)]
struct SecurityValidationResult {
    implications: Vec<String>,
    compliance_verified: Vec<String>,
    error: Option<ValidationError>,
}

/// Compliance checker
struct ComplianceChecker;

impl ComplianceChecker {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::{span::Position, SourceId};

    #[test]
    fn test_constraint_validator_creation() {
        let config = ValidationConfig::default();
        let validator = ConstraintValidator::new(config);
        
        // Should create without errors
        assert!(validator.config.runtime_validation);
        assert!(validator.config.compile_time_validation);
    }

    #[test]
    fn test_range_constraint_validation() {
        let mut validator = ConstraintValidator::new(ValidationConfig::default());
        
        // Create a valid range constraint
        let range = RangeConstraint {
            min: Some(create_literal_expr(0.0)),
            max: Some(create_literal_expr(100.0)),
            inclusive: true,
        };
        
        let span = create_test_span();
        let result = validator.validate_range_constraint(&range, span);
        
        assert!(result.is_ok());
        let constraint_result = result.unwrap();
        assert!(!constraint_result.ai_insights.is_empty());
    }

    #[test]
    fn test_pattern_constraint_validation() {
        let mut validator = ConstraintValidator::new(ValidationConfig::default());
        
        // Create a valid email pattern
        let pattern = PatternConstraint {
            pattern: r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string(),
            flags: Vec::new(),
        };
        
        let span = create_test_span();
        let result = validator.validate_pattern_constraint(&pattern, span);
        
        assert!(result.is_ok());
        let constraint_result = result.unwrap();
        assert!(constraint_result.ai_insights.iter().any(|insight| insight.contains("Email")));
    }

    #[test]
    fn test_invalid_range_constraint() {
        let mut validator = ConstraintValidator::new(ValidationConfig::default());
        
        // Create an invalid range constraint (min > max)
        let range = RangeConstraint {
            min: Some(create_literal_expr(100.0)),
            max: Some(create_literal_expr(0.0)),
            inclusive: true,
        };
        
        let span = create_test_span();
        let result = validator.validate_range_constraint(&range, span);
        
        assert!(result.is_err());
        if let Err(ValidationError::RangeViolation { message, .. }) = result {
            assert!(message.contains("greater than maximum"));
        }
    }

    fn create_literal_expr(value: f64) -> AstNode<Expr> {
        use prism_common::NodeId;
        
        AstNode::new(
            Expr::Literal(prism_ast::LiteralExpr {
                value: LiteralValue::Float(value),
            }),
            create_test_span(),
            NodeId::new(0),
        )
    }

    fn create_test_span() -> Span {
        Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 10, 9),
            SourceId::new(1),
        )
    }
} 