//! Error types for constraint validation
//!
//! This module defines all error types that can occur during constraint validation,
//! providing rich error information for debugging and user feedback.

use prism_common::span::Span;
use thiserror::Error;

/// Result type for constraint operations
pub type ConstraintResult<T> = Result<T, ConstraintError>;

/// Errors that can occur during constraint validation
#[derive(Debug, Error)]
pub enum ConstraintError {
    /// Invalid constraint specification
    #[error("Invalid constraint at {location}: {message}")]
    InvalidConstraint {
        /// Location of the error
        location: Span,
        /// Error message
        message: String,
    },

    /// Constraint evaluation failed
    #[error("Evaluation error at {location}: {message}")]
    EvaluationError {
        /// Location of the error
        location: Span,
        /// Error message
        message: String,
    },

    /// Type mismatch in constraint evaluation
    #[error("Type mismatch at {location}: expected {expected}, found {found} (value: {value})")]
    TypeMismatch {
        /// Location of the error
        location: Span,
        /// Expected type
        expected: String,
        /// Actual type found
        found: String,
        /// The actual value
        value: String,
    },

    /// Range constraint violation
    #[error("Range constraint violation at {location}: value {value} is not within allowed range")]
    RangeViolation {
        location: Span,
        value: String,
        min: Option<String>,
        max: Option<String>,
        inclusive: bool,
    },

    /// Pattern constraint violation
    #[error("Pattern constraint violation at {location}: value '{value}' does not match pattern '{pattern}'")]
    PatternViolation {
        location: Span,
        value: String,
        pattern: String,
        examples: Vec<String>,
    },

    /// Length constraint violation  
    #[error("Length constraint violation at {location}: length {actual} is not within bounds")]
    LengthViolation {
        location: Span,
        actual: usize,
        min: Option<usize>,
        max: Option<usize>,
    },

    /// Format constraint violation
    #[error("Format constraint violation at {location}: value '{value}' does not match format '{format}'")]
    FormatViolation {
        location: Span,
        value: String,
        format: String,
        expected_format: String,
    },

    /// Business rule violation
    #[error("Business rule violation at {location}: {rule_name} - {description}")]
    BusinessRuleViolation {
        location: Span,
        rule_name: String,
        description: String,
        predicate: String,
        ai_explanation: Option<String>,
    },

    /// Custom constraint violation
    #[error("Custom constraint violation at {location}: {constraint_name} - {message}")]
    CustomConstraintViolation {
        location: Span,
        constraint_name: String,
        message: String,
        predicate: String,
    },

    /// Undefined variable in expression
    #[error("Undefined variable '{variable}' in expression at {location}")]
    UndefinedVariable {
        location: Span,
        variable: String,
        available_variables: Vec<String>,
    },

    /// Invalid function call in expression
    #[error("Invalid function call '{function}' at {location}: {reason}")]
    InvalidFunctionCall {
        location: Span,
        function: String,
        reason: String,
        available_functions: Vec<String>,
    },

    /// Division by zero in expression
    #[error("Division by zero in expression at {location}")]
    DivisionByZero {
        location: Span,
        expression: String,
    },

    /// Overflow in numeric operation
    #[error("Numeric overflow in expression at {location}: {operation}")]
    NumericOverflow {
        location: Span,
        operation: String,
    },

    /// Invalid regular expression
    #[error("Invalid regular expression '{pattern}' at {location}: {regex_error}")]
    InvalidRegex {
        location: Span,
        pattern: String,
        regex_error: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        message: String,
    },

    /// Internal constraint engine error
    #[error("Internal constraint engine error: {message}")]
    InternalError {
        message: String,
    },
}

impl ConstraintError {
    /// Create a new evaluation error
    pub fn evaluation_error(location: Span, message: impl Into<String>) -> Self {
        Self::EvaluationError {
            location,
            message: message.into(),
        }
    }

    /// Create a new type mismatch error
    pub fn type_mismatch(location: Span, expected: impl Into<String>, found: impl Into<String>, value: impl Into<String>) -> Self {
        Self::TypeMismatch {
            location,
            expected: expected.into(),
            found: found.into(),
            value: value.into(),
        }
    }

    /// Create a new range violation error
    pub fn range_violation(
        location: Span,
        value: impl Into<String>,
        min: Option<impl Into<String>>,
        max: Option<impl Into<String>>,
        inclusive: bool,
    ) -> Self {
        Self::RangeViolation {
            location,
            value: value.into(),
            min: min.map(Into::into),
            max: max.map(Into::into),
            inclusive,
        }
    }

    /// Create a new pattern violation error
    pub fn pattern_violation(
        location: Span,
        value: impl Into<String>,
        pattern: impl Into<String>,
        examples: Vec<String>,
    ) -> Self {
        Self::PatternViolation {
            location,
            value: value.into(),
            pattern: pattern.into(),
            examples,
        }
    }

    /// Create a new business rule violation error
    pub fn business_rule_violation(
        location: Span,
        rule_name: impl Into<String>,
        description: impl Into<String>,
        predicate: impl Into<String>,
        ai_explanation: Option<String>,
    ) -> Self {
        Self::BusinessRuleViolation {
            location,
            rule_name: rule_name.into(),
            description: description.into(),
            predicate: predicate.into(),
            ai_explanation,
        }
    }

    /// Get the location of this error
    pub fn location(&self) -> Option<Span> {
        match self {
            Self::InvalidConstraint { location, .. }
            | Self::EvaluationError { location, .. }
            | Self::TypeMismatch { location, .. }
            | Self::RangeViolation { location, .. }
            | Self::PatternViolation { location, .. }
            | Self::LengthViolation { location, .. }
            | Self::FormatViolation { location, .. }
            | Self::BusinessRuleViolation { location, .. }
            | Self::CustomConstraintViolation { location, .. }
            | Self::UndefinedVariable { location, .. }
            | Self::InvalidFunctionCall { location, .. }
            | Self::DivisionByZero { location, .. }
            | Self::NumericOverflow { location, .. }
            | Self::InvalidRegex { location, .. } => Some(*location),
            Self::ConfigurationError { .. } | Self::InternalError { .. } => None,
        }
    }

    /// Check if this is a user error (vs internal error)
    pub fn is_user_error(&self) -> bool {
        !matches!(self, Self::InternalError { .. })
    }

    /// Get suggested fixes for this error
    pub fn suggested_fixes(&self) -> Vec<String> {
        match self {
            Self::UndefinedVariable { available_variables, .. } => {
                if available_variables.is_empty() {
                    vec!["No variables are available in this scope".to_string()]
                } else {
                    let mut suggestions = vec!["Available variables:".to_string()];
                    suggestions.extend(available_variables.iter().map(|v| format!("  - {}", v)));
                    suggestions
                }
            }
            Self::InvalidFunctionCall { available_functions, .. } => {
                if available_functions.is_empty() {
                    vec!["No functions are available in this context".to_string()]
                } else {
                    let mut suggestions = vec!["Available functions:".to_string()];
                    suggestions.extend(available_functions.iter().map(|f| format!("  - {}", f)));
                    suggestions
                }
            }
            Self::PatternViolation { examples, .. } => {
                if examples.is_empty() {
                    vec!["Check the pattern format and try again".to_string()]
                } else {
                    let mut suggestions = vec!["Examples of valid values:".to_string()];
                    suggestions.extend(examples.iter().map(|e| format!("  - {}", e)));
                    suggestions
                }
            }
            Self::TypeMismatch { expected, .. } => {
                vec![format!("Convert the value to {} type", expected)]
            }
            _ => vec![],
        }
    }
} 

/// Helper function to format range for display
fn format_range(min: &Option<String>, max: &Option<String>, inclusive: bool) -> String {
    let separator = if inclusive { "..=" } else { ".." };
    match (min, max) {
        (Some(min), Some(max)) => format!("{}{}{}", min, separator, max),
        (Some(min), None) => format!("{}+", min),
        (None, Some(max)) => format!("..{}{}", separator, max),
        (None, None) => "any".to_string(),
    }
}

/// Helper function to format usize range for display
fn format_range_usize(min: &Option<usize>, max: &Option<usize>) -> String {
    match (min, max) {
        (Some(min), Some(max)) => format!("{}..{}", min, max),
        (Some(min), None) => format!("{}+", min),
        (None, Some(max)) => format!("..{}", max),
        (None, None) => "any".to_string(),
    }
} 