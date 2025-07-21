//! Constraint validation engine
//!
//! This module provides the main constraint engine that orchestrates
//! constraint validation using the various validators and evaluators.

use crate::{
    ConstraintValue, ConstraintResult, ConstraintValidator, ValidationResult,
    ValidationError, ValidationConfig, ExpressionEvaluator, EvaluationContext,
    builtin::BuiltinValidatorRegistry, ValidationContext,
};
use prism_ast::Expr;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

/// Main constraint validation engine
#[derive(Debug)]
pub struct ConstraintEngine {
    /// Configuration for constraint validation
    config: ValidationConfig,
    /// Registry of constraint validators
    validators: HashMap<String, Box<dyn ConstraintValidator>>,
    /// Built-in validator registry
    builtin_registry: BuiltinValidatorRegistry,
    /// Expression evaluator for predicate constraints
    expression_evaluator: ExpressionEvaluator,
    /// Validation cache for performance
    validation_cache: HashMap<String, ValidationResult>,
    /// Compile-time constraint evaluator
    compile_time_evaluator: CompileTimeConstraintEvaluator,
}

/// Compile-time constraint evaluator
#[derive(Debug)]
pub struct CompileTimeConstraintEvaluator {
    /// Static assertion cache
    static_assertion_cache: HashMap<String, StaticAssertionResult>,
    /// Configuration
    config: CompileTimeConfig,
}

/// Static assertion result
#[derive(Debug, Clone)]
pub struct StaticAssertionResult {
    /// Whether assertion passed
    pub passed: bool,
    /// Assertion message
    pub message: String,
    /// Evaluation time
    pub evaluation_time_us: u64,
}

/// Compile-time configuration
#[derive(Debug, Clone)]
pub struct CompileTimeConfig {
    /// Enable static assertion caching
    pub enable_caching: bool,
    /// Maximum proof depth
    pub max_proof_depth: usize,
    /// Proof timeout in milliseconds
    pub proof_timeout_ms: u64,
}

impl Default for CompileTimeConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_proof_depth: 100,
            proof_timeout_ms: 5000,
        }
    }
}

impl CompileTimeConstraintEvaluator {
    /// Create a new compile-time evaluator
    pub fn new() -> Self {
        Self {
            static_assertion_cache: HashMap::new(),
            config: CompileTimeConfig::default(),
        }
    }

    /// Evaluate a static assertion
    pub fn evaluate_static_assertion(
        &mut self,
        condition: &prism_ast::AstNode<Expr>,
        error_message: &str,
        context: &ValidationContext,
    ) -> ConstraintResult<StaticAssertionResult> {
        let start_time = Instant::now();
        
        // Generate cache key
        let cache_key = format!("{:?}_{}", condition, error_message);
        
        // Check cache if enabled
        if self.config.enable_caching {
            if let Some(cached_result) = self.static_assertion_cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }

        // Evaluate the assertion condition
        let evaluator = ExpressionEvaluator::new();
        let eval_context = EvaluationContext::new();
        let evaluation_result = evaluator.evaluate(condition, &eval_context)?;
        
        let result = StaticAssertionResult {
            passed: evaluation_result.value.is_truthy(),
            message: if evaluation_result.value.is_truthy() {
                "Static assertion passed".to_string()
            } else {
                error_message.to_string()
            },
            evaluation_time_us: start_time.elapsed().as_micros() as u64,
        };

        // Cache result if enabled
        if self.config.enable_caching {
            self.static_assertion_cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }
}

impl Default for CompileTimeConstraintEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintEngine {
    /// Create a new constraint engine
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new constraint engine with configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            validators: HashMap::new(),
            builtin_registry: BuiltinValidatorRegistry::new(),
            expression_evaluator: ExpressionEvaluator::new(),
            validation_cache: HashMap::new(),
            compile_time_evaluator: CompileTimeConstraintEvaluator::new(),
        }
    }

    /// Register a custom constraint validator
    pub fn register_validator(&mut self, name: impl Into<String>, validator: Box<dyn ConstraintValidator>) {
        self.validators.insert(name.into(), validator);
    }

    /// Validate a range constraint
    pub fn validate_range(
        &self,
        value: &ConstraintValue,
        min: Option<ConstraintValue>,
        max: Option<ConstraintValue>,
        inclusive: bool,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let validator = crate::builtin::RangeValidator::new(min, max, inclusive);
        validator.validate(value, context)
    }

    /// Validate a pattern constraint
    pub fn validate_pattern(
        &self,
        value: &ConstraintValue,
        pattern: &str,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let validator = crate::builtin::PatternValidator::new(pattern)?;
        validator.validate(value, context)
    }

    /// Validate a length constraint
    pub fn validate_length(
        &self,
        value: &ConstraintValue,
        min_length: Option<usize>,
        max_length: Option<usize>,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let validator = crate::builtin::LengthValidator::new(min_length, max_length);
        validator.validate(value, context)
    }

    /// Validate a format constraint
    pub fn validate_format(
        &self,
        value: &ConstraintValue,
        format_name: &str,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        if let Some(validator) = self.builtin_registry.get(format_name) {
            validator.validate(value, context)
        } else {
            Err(crate::ConstraintError::ConfigurationError {
                message: format!("Unknown format validator: {}", format_name),
            })
        }
    }

    /// Validate a business rule constraint using expression evaluation
    pub fn validate_business_rule(
        &self,
        value: &ConstraintValue,
        rule_name: &str,
        predicate_expr: &Expr,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let start_time = if self.config.track_performance {
            Some(Instant::now())
        } else {
            None
        };

        // Create evaluation context with the value as a variable
        let mut eval_context = EvaluationContext::new();
        eval_context.add_variable("value", value.clone());
        
        // Add any context variables
        for (name, val) in &context.variables {
            eval_context.add_variable(name.clone(), val.clone());
        }

        // Evaluate the predicate expression
        // Wrap the expression in an AstNode for the evaluator
        use prism_common::{NodeId, span::Span};
        use prism_ast::AstNode;
        
        let predicate_node = AstNode::new(
            predicate_expr.clone(),
            Span::dummy(),
            NodeId::new(0),
        );
        
        let eval_result = self.expression_evaluator.evaluate(&predicate_node, &eval_context)?;
        
        let mut result = ValidationResult::success();
        
        // Check if the predicate evaluated to true
        if !eval_result.value.is_truthy() {
            let error = ValidationError::new(
                format!("Business rule '{}' violated", rule_name),
                "business_rule",
            )
            .with_expected("predicate to evaluate to true")
            .with_actual(format!("{:?}", eval_result.value))
            .with_suggestion(format!("Ensure the value satisfies the business rule: {}", rule_name));

            result.add_error(error);
        }

        // Add any warnings from expression evaluation
        for warning_msg in eval_result.warnings {
            result.add_warning(crate::validators::ValidationWarning::new(warning_msg, "expression_evaluation"));
        }

        // Track performance if enabled
        if let Some(start) = start_time {
            result.metadata.validation_time_us = start.elapsed().as_micros() as u64;
        }
        result.metadata.constraints_checked = 1;

        Ok(result)
    }

    /// Validate a custom constraint using expression evaluation
    pub fn validate_custom_constraint(
        &self,
        value: &ConstraintValue,
        constraint_name: &str,
        predicate_expr: &Expr,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let start_time = if self.config.track_performance {
            Some(Instant::now())
        } else {
            None
        };

        // Create evaluation context with the value as a variable
        let mut eval_context = EvaluationContext::new();
        eval_context.add_variable("value", value.clone());
        
        // Add any context variables
        for (name, val) in &context.variables {
            eval_context.add_variable(name.clone(), val.clone());
        }

        // Evaluate the predicate expression
        // Wrap the expression in an AstNode for the evaluator
        use prism_common::{NodeId, span::Span};
        use prism_ast::AstNode;
        
        let predicate_node = AstNode::new(
            predicate_expr.clone(),
            Span::dummy(),
            NodeId::new(0),
        );
        
        let eval_result = self.expression_evaluator.evaluate(&predicate_node, &eval_context)?;
        
        let mut result = ValidationResult::success();
        
        // Check if the predicate evaluated to true
        if !eval_result.value.is_truthy() {
            let error = ValidationError::new(
                format!("Custom constraint '{}' violated", constraint_name),
                "custom",
            )
            .with_expected("predicate to evaluate to true")
            .with_actual(format!("{:?}", eval_result.value))
            .with_suggestion(format!("Ensure the value satisfies the custom constraint: {}", constraint_name));

            result.add_error(error);
        }

        // Add any warnings from expression evaluation
        for warning_msg in eval_result.warnings {
            result.add_warning(crate::validators::ValidationWarning::new(warning_msg, "expression_evaluation"));
        }

        // Track performance if enabled
        if let Some(start) = start_time {
            result.metadata.validation_time_us = start.elapsed().as_micros() as u64;
        }
        result.metadata.constraints_checked = 1;

        Ok(result)
    }

    /// Validate multiple constraints on a single value
    pub fn validate_all(
        &self,
        value: &ConstraintValue,
        constraints: &[ConstraintSpec],
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let start_time = if self.config.track_performance {
            Some(Instant::now())
        } else {
            None
        };

        let mut combined_result = ValidationResult::success();
        let mut total_constraints = 0;

        for constraint in constraints {
            let constraint_result = match constraint {
                ConstraintSpec::Range { min, max, inclusive } => {
                    self.validate_range(value, min.clone(), max.clone(), *inclusive, context)?
                }
                ConstraintSpec::Pattern { pattern } => {
                    self.validate_pattern(value, pattern, context)?
                }
                ConstraintSpec::Length { min_length, max_length } => {
                    self.validate_length(value, *min_length, *max_length, context)?
                }
                ConstraintSpec::Format { format_name } => {
                    self.validate_format(value, format_name, context)?
                }
                ConstraintSpec::BusinessRule { rule_name, predicate } => {
                    self.validate_business_rule(value, rule_name, predicate, context)?
                }
                ConstraintSpec::Custom { constraint_name, predicate } => {
                    self.validate_custom_constraint(value, constraint_name, predicate, context)?
                }
            };

            combined_result = combined_result.combine(constraint_result);
            total_constraints += 1;

            // Stop early if we have too many errors
            if combined_result.errors.len() >= self.config.max_errors {
                break;
            }
        }

        // Track performance if enabled
        if let Some(start) = start_time {
            combined_result.metadata.validation_time_us = start.elapsed().as_micros() as u64;
        }
        combined_result.metadata.constraints_checked = total_constraints;

        debug!(
            "Validated {} constraints on value {:?}: {} errors, {} warnings",
            total_constraints,
            value,
            combined_result.errors.len(),
            combined_result.warnings.len()
        );

        Ok(combined_result)
    }

    /// Clear the validation cache
    pub fn clear_cache(&mut self) {
        self.validation_cache.clear();
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> ConstraintEngineStatistics {
        ConstraintEngineStatistics {
            registered_validators: self.validators.len(),
            builtin_validators: self.builtin_registry.list_validators().len(),
            cache_size: self.validation_cache.len(),
        }
    }

    /// Validate a compile-time constraint
    pub fn validate_compile_time_constraint(
        &mut self,
        name: &str,
        predicate: &prism_ast::AstNode<Expr>,
        error_message: &str,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let start_time = if self.config.track_performance {
            Some(Instant::now())
        } else {
            None
        };

        // Evaluate the compile-time constraint
        let assertion_result = self.compile_time_evaluator.evaluate_static_assertion(
            predicate,
            error_message,
            context,
        )?;
        
        let mut result = ValidationResult::success();
        
        if !assertion_result.passed {
            let error = ValidationError::new(
                format!("Compile-time constraint '{}' failed", name),
                "compile_time",
            )
            .with_expected("constraint to be satisfied at compile time")
            .with_actual("constraint was violated")
            .with_suggestion("Check the constraint logic and input values".to_string());

            result.add_error(error);
        }

        // Track performance if enabled
        if let Some(start) = start_time {
            result.metadata.validation_time_us = start.elapsed().as_micros() as u64;
        }
        result.metadata.constraints_checked = 1;

        debug!(
            "Validated compile-time constraint '{}': passed={}",
            name, assertion_result.passed
        );

        Ok(result)
    }

    /// Validate static assertion
    pub fn validate_static_assertion(
        &mut self,
        condition: &prism_ast::AstNode<Expr>,
        error_message: &str,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        let start_time = if self.config.track_performance {
            Some(Instant::now())
        } else {
            None
        };

        // Evaluate the static assertion
        let assertion_result = self.compile_time_evaluator.evaluate_static_assertion(
            condition,
            error_message,
            context,
        )?;
        
        let mut result = ValidationResult::success();
        
        if !assertion_result.passed {
            let error = ValidationError::new(
                "Static assertion failed".to_string(),
                "static_assertion",
            )
            .with_expected("assertion condition to be true")
            .with_actual("assertion condition was false")
            .with_suggestion(error_message.to_string());

            result.add_error(error);
        }

        // Track performance if enabled
        if let Some(start) = start_time {
            result.metadata.validation_time_us = start.elapsed().as_micros() as u64;
        }
        result.metadata.constraints_checked = 1;

        Ok(result)
    }

    /// Validate enhanced constraint specification
    pub fn validate_enhanced_constraint(
        &mut self,
        constraint: &EnhancedConstraintSpec,
        context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        match constraint {
            EnhancedConstraintSpec::CompileTime { name, predicate, error_message, .. } => {
                self.validate_compile_time_constraint(name, predicate, error_message, context)
            }
            EnhancedConstraintSpec::StaticAssertion { condition, error_message, .. } => {
                self.validate_static_assertion(condition, error_message, context)
            }
            EnhancedConstraintSpec::Dependent { expression, dependencies, .. } => {
                self.validate_dependent_constraint(expression, dependencies, context)
            }
        }
    }

    /// Validate dependent constraint
    fn validate_dependent_constraint(
        &mut self,
        _expression: &prism_ast::AstNode<Expr>,
        _dependencies: &[DependentParameter],
        _context: &ValidationContext,
    ) -> ConstraintResult<ValidationResult> {
        // TODO: Implement dependent constraint validation
        // This would involve resolving dependencies and evaluating the constraint
        let mut result = ValidationResult::success();
        result.add_warning(crate::validators::ValidationWarning::new(
            "Dependent constraint validation not fully implemented".to_string(),
            "dependent_constraint",
        ));
        Ok(result)
    }
}

impl Default for ConstraintEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Specification for a constraint to be validated
#[derive(Debug, Clone)]
pub enum ConstraintSpec {
    /// Range constraint
    Range {
        min: Option<ConstraintValue>,
        max: Option<ConstraintValue>,
        inclusive: bool,
    },
    /// Pattern constraint
    Pattern {
        pattern: String,
    },
    /// Length constraint
    Length {
        min_length: Option<usize>,
        max_length: Option<usize>,
    },
    /// Format constraint
    Format {
        format_name: String,
    },
    /// Business rule constraint
    BusinessRule {
        rule_name: String,
        predicate: Expr,
    },
    /// Custom constraint
    Custom {
        constraint_name: String,
        predicate: Expr,
    },
}

/// Context for constraint validation
pub type ConstraintContext = ValidationContext;

/// Statistics about the constraint engine
#[derive(Debug, Clone)]
pub struct ConstraintEngineStatistics {
    /// Number of registered custom validators
    pub registered_validators: usize,
    /// Number of built-in validators
    pub builtin_validators: usize,
    /// Size of validation cache
    pub cache_size: usize,
}

/// Enhanced constraint specification
#[derive(Debug, Clone)]
pub enum EnhancedConstraintSpec {
    /// Compile-time constraint
    CompileTime {
        name: String,
        predicate: prism_ast::AstNode<Expr>,
        error_message: String,
        priority: ConstraintPriority,
    },
    /// Static assertion
    StaticAssertion {
        condition: prism_ast::AstNode<Expr>,
        error_message: String,
        context: StaticAssertionContext,
    },
    /// Dependent constraint
    Dependent {
        expression: prism_ast::AstNode<Expr>,
        dependencies: Vec<DependentParameter>,
        evaluation_strategy: DependentEvaluationStrategy,
    },
}

/// Dependent parameter
#[derive(Debug, Clone)]
pub struct DependentParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Relationship to dependent constraint
    pub relationship: DependentRelationship,
}

/// Dependent relationship types
#[derive(Debug, Clone)]
pub enum DependentRelationship {
    /// Size dependency
    Size,
    /// Value dependency
    Value,
    /// Constraint dependency
    Constraint,
}

/// Dependent evaluation strategy
#[derive(Debug, Clone)]
pub enum DependentEvaluationStrategy {
    /// Eager evaluation (compute at constraint definition time)
    Eager,
    /// Lazy evaluation (compute when needed)
    Lazy,
    /// Cached evaluation (compute once, cache result)
    Cached,
}

/// Static assertion context
#[derive(Debug, Clone)]
pub enum StaticAssertionContext {
    /// Type constraint validation
    TypeConstraint,
    /// Business rule validation
    BusinessRule,
    /// Security requirement
    SecurityRequirement,
    /// Performance constraint
    PerformanceConstraint,
}

/// Constraint priority for ordering evaluation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintPriority {
    /// Low priority (performance hints)
    Low,
    /// Medium priority (business rules)
    Medium,
    /// High priority (safety constraints)
    High,
    /// Critical priority (security, correctness)
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintValue;

    #[test]
    fn test_range_validation() {
        let engine = ConstraintEngine::new();
        let context = ValidationContext::new();

        // Test valid range
        let result = engine.validate_range(
            &ConstraintValue::Integer(5),
            Some(ConstraintValue::Integer(1)),
            Some(ConstraintValue::Integer(10)),
            true,
            &context,
        ).unwrap();
        assert!(result.is_valid);

        // Test invalid range (too low)
        let result = engine.validate_range(
            &ConstraintValue::Integer(0),
            Some(ConstraintValue::Integer(1)),
            Some(ConstraintValue::Integer(10)),
            true,
            &context,
        ).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());

        // Test invalid range (too high)
        let result = engine.validate_range(
            &ConstraintValue::Integer(11),
            Some(ConstraintValue::Integer(1)),
            Some(ConstraintValue::Integer(10)),
            true,
            &context,
        ).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_pattern_validation() {
        let engine = ConstraintEngine::new();
        let context = ValidationContext::new();

        // Test valid pattern
        let result = engine.validate_pattern(
            &ConstraintValue::String("hello123".to_string()),
            r"^[a-z]+\d+$",
            &context,
        ).unwrap();
        assert!(result.is_valid);

        // Test invalid pattern
        let result = engine.validate_pattern(
            &ConstraintValue::String("HELLO123".to_string()),
            r"^[a-z]+\d+$",
            &context,
        ).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_length_validation() {
        let engine = ConstraintEngine::new();
        let context = ValidationContext::new();

        // Test valid length
        let result = engine.validate_length(
            &ConstraintValue::String("hello".to_string()),
            Some(3),
            Some(10),
            &context,
        ).unwrap();
        assert!(result.is_valid);

        // Test invalid length (too short)
        let result = engine.validate_length(
            &ConstraintValue::String("hi".to_string()),
            Some(3),
            Some(10),
            &context,
        ).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());

        // Test invalid length (too long)
        let result = engine.validate_length(
            &ConstraintValue::String("this is way too long".to_string()),
            Some(3),
            Some(10),
            &context,
        ).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }
} 