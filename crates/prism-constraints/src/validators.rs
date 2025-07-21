//! Constraint validator trait and validation results
//!
//! This module defines the trait for constraint validators and the result types
//! for validation operations.

use crate::{ConstraintValue, ConstraintError, ConstraintResult};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};

/// Warning generated during constraint validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Warning category
    pub category: String,
    /// Source location of the warning
    pub location: Option<prism_common::span::Span>,
}

impl ValidationWarning {
    /// Create a new validation warning
    pub fn new(message: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            category: category.into(),
            location: None,
        }
    }

    /// Create a warning with location
    pub fn with_location(mut self, location: prism_common::span::Span) -> Self {
        self.location = Some(location);
        self
    }
}

/// Result of constraint validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the constraint was satisfied
    pub is_valid: bool,
    /// Validation errors (if any)
    pub errors: Vec<ValidationError>,
    /// Validation warnings (if any)
    pub warnings: Vec<ValidationWarning>,
    /// Additional metadata about the validation
    pub metadata: ValidationMetadata,
}

impl ValidationResult {
    /// Create a new successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: ValidationMetadata::default(),
        }
    }

    /// Create a new failed validation result with an error
    pub fn failure(error: ValidationError) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
            warnings: Vec::new(),
            metadata: ValidationMetadata::default(),
        }
    }

    /// Create a new validation result with warnings
    pub fn with_warnings(warnings: Vec<ValidationWarning>) -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings,
            metadata: ValidationMetadata::default(),
        }
    }

    /// Add an error to the validation result
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a warning to the validation result
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// Combine two validation results
    pub fn combine(mut self, other: ValidationResult) -> Self {
        self.is_valid = self.is_valid && other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.metadata.combine(other.metadata);
        self
    }

    /// Check if the validation has any issues (errors or warnings)
    pub fn has_issues(&self) -> bool {
        !self.errors.is_empty() || !self.warnings.is_empty()
    }
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Location where the error occurred
    pub location: Option<Span>,
    /// Type of constraint that failed
    pub constraint_type: String,
    /// Expected value or pattern
    pub expected: Option<String>,
    /// Actual value that was validated
    pub actual: Option<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
    /// AI explanation (if available)
    pub ai_explanation: Option<String>,
}

impl ValidationError {
    /// Create a new validation error
    pub fn new(message: impl Into<String>, constraint_type: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            location: None,
            constraint_type: constraint_type.into(),
            expected: None,
            actual: None,
            suggestions: Vec::new(),
            ai_explanation: None,
        }
    }

    /// Set the location of the error
    pub fn with_location(mut self, location: Span) -> Self {
        self.location = Some(location);
        self
    }

    /// Set the expected value
    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    /// Set the actual value
    pub fn with_actual(mut self, actual: impl Into<String>) -> Self {
        self.actual = Some(actual.into());
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Set AI explanation
    pub fn with_ai_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.ai_explanation = Some(explanation.into());
        self
    }
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Performance score (0.0 to 1.0, higher is better)
    pub performance: f64,
    /// Validation time in microseconds
    pub validation_time_us: u64,
    /// Number of constraints checked
    pub constraints_checked: usize,
    /// Additional metadata
    pub additional: std::collections::HashMap<String, String>,
}

impl Default for ValidationMetadata {
    fn default() -> Self {
        Self {
            confidence: 1.0,
            performance: 1.0,
            validation_time_us: 0,
            constraints_checked: 0,
            additional: std::collections::HashMap::new(),
        }
    }
}

impl ValidationMetadata {
    /// Combine two metadata objects
    pub fn combine(&mut self, other: ValidationMetadata) {
        self.confidence = self.confidence.min(other.confidence);
        self.performance = (self.performance + other.performance) / 2.0;
        self.validation_time_us += other.validation_time_us;
        self.constraints_checked += other.constraints_checked;
        self.additional.extend(other.additional);
    }
}

/// Trait for constraint validators
pub trait ConstraintValidator: Send + Sync + std::fmt::Debug {
    /// Validate a value against this constraint
    fn validate(&self, value: &ConstraintValue, context: &ValidationContext) -> ConstraintResult<ValidationResult>;

    /// Get the name of this constraint validator
    fn name(&self) -> &str;

    /// Get the description of this constraint validator
    fn description(&self) -> &str;

    /// Get the constraint type this validator handles
    fn constraint_type(&self) -> &str;

    /// Check if this validator can handle the given constraint type
    fn can_validate(&self, constraint_type: &str) -> bool {
        self.constraint_type() == constraint_type
    }

    /// Get examples of valid values for this constraint
    fn examples(&self) -> Vec<ConstraintValue> {
        Vec::new()
    }

    /// Get examples of invalid values for this constraint
    fn counter_examples(&self) -> Vec<ConstraintValue> {
        Vec::new()
    }
}

/// Context for constraint validation
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Variables available during validation
    pub variables: std::collections::HashMap<String, ConstraintValue>,
    /// Configuration for validation
    pub config: ValidationConfig,
    /// Current validation depth
    pub depth: usize,
    /// Maximum validation depth
    pub max_depth: usize,
}

impl ValidationContext {
    /// Create a new validation context
    pub fn new() -> Self {
        Self {
            variables: std::collections::HashMap::new(),
            config: ValidationConfig::default(),
            depth: 0,
            max_depth: 100,
        }
    }

    /// Add a variable to the context
    pub fn add_variable(&mut self, name: impl Into<String>, value: ConstraintValue) {
        self.variables.insert(name.into(), value);
    }

    /// Get a variable from the context
    pub fn get_variable(&self, name: &str) -> Option<&ConstraintValue> {
        self.variables.get(name)
    }

    /// Iterate over all variables in the context
    pub fn iter_variables(&self) -> impl Iterator<Item = (&String, &ConstraintValue)> {
        self.variables.iter()
    }

    /// Create a child context with increased depth
    pub fn child(&self) -> ConstraintResult<Self> {
        if self.depth >= self.max_depth {
            return Err(ConstraintError::InternalError {
                message: format!("Maximum validation depth {} exceeded", self.max_depth),
            });
        }

        Ok(Self {
            variables: self.variables.clone(),
            config: self.config.clone(),
            depth: self.depth + 1,
            max_depth: self.max_depth,
        })
    }
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for constraint validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable strict validation
    pub strict_mode: bool,
    /// Enable performance tracking
    pub track_performance: bool,
    /// Enable detailed error messages
    pub detailed_errors: bool,
    /// Maximum number of errors to collect
    pub max_errors: usize,
    /// Enable AI explanations
    pub enable_ai_explanations: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            track_performance: false,
            detailed_errors: true,
            max_errors: 10,
            enable_ai_explanations: false,
        }
    }
} 