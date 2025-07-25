//! Type Inference Error Handling
//!
//! This module provides comprehensive error handling for type inference,
//! including rich diagnostics, error recovery, and user-friendly error messages.

use super::{TypeVar, InferredType, constraints::ConstraintSet};
use crate::types::SemanticType;
use prism_common::{Span, SourceId};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

/// Main type error enum
#[derive(Debug, Clone)]
pub enum TypeError {
    /// Type mismatch between expected and actual types
    TypeMismatch {
        expected: SemanticType,
        actual: SemanticType,
        location: Span,
        context: TypeContext,
        suggestion: Option<String>,
    },

    /// Unification failure
    UnificationFailure {
        left: SemanticType,
        right: SemanticType,
        location: Span,
        reason: String,
    },

    /// Occurs check failure (infinite type)
    OccursCheck {
        variable: TypeVar,
        type_expr: SemanticType,
        location: Span,
    },

    /// Undefined variable
    UndefinedVariable {
        name: String,
        location: Span,
        suggestions: Vec<String>,
    },

    /// Arity mismatch in function application
    ArityMismatch {
        expected: usize,
        actual: usize,
        location: Span,
        function_name: Option<String>,
    },

    /// Missing field in record access
    MissingField {
        field_name: String,
        record_type: SemanticType,
        location: Span,
        available_fields: Vec<String>,
    },

    /// Duplicate field in record construction
    DuplicateField {
        field_name: String,
        first_location: Span,
        duplicate_location: Span,
    },

    /// Invalid pattern match
    InvalidPattern {
        pattern_type: SemanticType,
        expression_type: SemanticType,
        location: Span,
    },

    /// Exhaustiveness error in pattern matching
    NonExhaustivePatterns {
        missing_patterns: Vec<String>,
        location: Span,
    },

    /// Unreachable pattern
    UnreachablePattern {
        location: Span,
        reason: String,
    },

    /// Constraint violation
    ConstraintViolation {
        constraint: String,
        location: Span,
        explanation: String,
    },

    /// Recursive type definition
    RecursiveType {
        type_name: String,
        location: Span,
        cycle: Vec<String>,
    },

    /// Ambiguous type
    AmbiguousType {
        variable: TypeVar,
        location: Span,
        possible_types: Vec<SemanticType>,
    },

    /// Effect system violation
    EffectViolation {
        required_effects: Vec<String>,
        available_effects: Vec<String>,
        location: Span,
    },

    /// Generic type error with custom message
    Custom {
        message: String,
        location: Span,
        kind: TypeErrorKind,
    },
}

/// Type error kind for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeErrorKind {
    /// Type system error
    Type,
    /// Constraint error
    Constraint,
    /// Pattern matching error
    Pattern,
    /// Effect system error
    Effect,
    /// Scope/binding error
    Binding,
    /// Internal error
    Internal,
}

/// Context where a type error occurred
#[derive(Debug, Clone)]
pub enum TypeContext {
    /// Function application
    FunctionApplication {
        function_name: Option<String>,
        argument_index: usize,
    },
    /// Variable assignment
    Assignment {
        variable_name: String,
    },
    /// Function return
    Return {
        function_name: Option<String>,
    },
    /// Field access
    FieldAccess {
        field_name: String,
    },
    /// Pattern matching
    PatternMatch,
    /// Type annotation
    Annotation,
    /// Binary operation
    BinaryOperation {
        operator: String,
    },
    /// List element
    ListElement {
        index: usize,
    },
    /// Record field
    RecordField {
        field_name: String,
    },
    /// Generic context
    Other(String),
}

/// Diagnostic information for type errors
#[derive(Debug, Clone)]
pub struct TypeDiagnostic {
    /// The main error
    pub error: TypeError,
    /// Primary label with location and message
    pub primary_label: DiagnosticLabel,
    /// Secondary labels with additional context
    pub secondary_labels: Vec<DiagnosticLabel>,
    /// Help messages and suggestions
    pub help: Vec<String>,
    /// Notes with additional information
    pub notes: Vec<String>,
    /// Error severity
    pub severity: DiagnosticSeverity,
    /// Error code for documentation lookup
    pub code: Option<String>,
}

/// Label in a diagnostic
#[derive(Debug, Clone)]
pub struct DiagnosticLabel {
    /// Location of this label
    pub location: Span,
    /// Message for this label
    pub message: String,
    /// Style of this label
    pub style: LabelStyle,
}

/// Style of a diagnostic label
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelStyle {
    /// Primary error location
    Primary,
    /// Secondary context location
    Secondary,
    /// Information/note location
    Info,
}

/// Severity of a diagnostic
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    /// Informational message
    Info,
    /// Warning that doesn't prevent compilation
    Warning,
    /// Error that prevents compilation
    Error,
    /// Internal compiler error
    InternalError,
}

/// Builder for creating rich type diagnostics
#[derive(Debug)]
pub struct DiagnosticBuilder {
    error: TypeError,
    primary_label: Option<DiagnosticLabel>,
    secondary_labels: Vec<DiagnosticLabel>,
    help: Vec<String>,
    notes: Vec<String>,
    severity: DiagnosticSeverity,
    code: Option<String>,
}

/// Error recovery suggestions
#[derive(Debug, Clone)]
pub struct ErrorRecovery {
    /// Suggested fixes
    pub fixes: Vec<SuggestedFix>,
    /// Possible alternative interpretations
    pub alternatives: Vec<String>,
    /// Related documentation
    pub documentation: Vec<String>,
}

/// A suggested fix for a type error
#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// Description of the fix
    pub description: String,
    /// Code changes to apply
    pub changes: Vec<TextChange>,
    /// Confidence in this fix (0.0 to 1.0)
    pub confidence: f64,
}

/// A text change for error recovery
#[derive(Debug, Clone)]
pub struct TextChange {
    /// Location to change
    pub location: Span,
    /// New text to insert
    pub new_text: String,
    /// Whether this is an insertion, deletion, or replacement
    pub change_type: ChangeType,
}

/// Type of text change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Insert text at location
    Insert,
    /// Delete text at location
    Delete,
    /// Replace text at location
    Replace,
}

impl TypeError {
    /// Create a type mismatch error
    pub fn type_mismatch(
        expected: SemanticType,
        actual: SemanticType,
        location: Span,
        context: TypeContext,
    ) -> Self {
        Self::TypeMismatch {
            expected,
            actual,
            location,
            context,
            suggestion: None,
        }
    }

    /// Create a type mismatch error with suggestion
    pub fn type_mismatch_with_suggestion(
        expected: SemanticType,
        actual: SemanticType,
        location: Span,
        context: TypeContext,
        suggestion: String,
    ) -> Self {
        Self::TypeMismatch {
            expected,
            actual,
            location,
            context,
            suggestion: Some(suggestion),
        }
    }

    /// Create an undefined variable error
    pub fn undefined_variable(name: String, location: Span, suggestions: Vec<String>) -> Self {
        Self::UndefinedVariable {
            name,
            location,
            suggestions,
        }
    }

    /// Create an arity mismatch error
    pub fn arity_mismatch(
        expected: usize,
        actual: usize,
        location: Span,
        function_name: Option<String>,
    ) -> Self {
        Self::ArityMismatch {
            expected,
            actual,
            location,
            function_name,
        }
    }

    /// Get the primary location of this error
    pub fn location(&self) -> Span {
        match self {
            TypeError::TypeMismatch { location, .. } => *location,
            TypeError::UnificationFailure { location, .. } => *location,
            TypeError::OccursCheck { location, .. } => *location,
            TypeError::UndefinedVariable { location, .. } => *location,
            TypeError::ArityMismatch { location, .. } => *location,
            TypeError::MissingField { location, .. } => *location,
            TypeError::DuplicateField { duplicate_location, .. } => *duplicate_location,
            TypeError::InvalidPattern { location, .. } => *location,
            TypeError::NonExhaustivePatterns { location, .. } => *location,
            TypeError::UnreachablePattern { location, .. } => *location,
            TypeError::ConstraintViolation { location, .. } => *location,
            TypeError::RecursiveType { location, .. } => *location,
            TypeError::AmbiguousType { location, .. } => *location,
            TypeError::EffectViolation { location, .. } => *location,
            TypeError::Custom { location, .. } => *location,
        }
    }

    /// Get the error kind
    pub fn kind(&self) -> TypeErrorKind {
        match self {
            TypeError::TypeMismatch { .. } => TypeErrorKind::Type,
            TypeError::UnificationFailure { .. } => TypeErrorKind::Type,
            TypeError::OccursCheck { .. } => TypeErrorKind::Type,
            TypeError::UndefinedVariable { .. } => TypeErrorKind::Binding,
            TypeError::ArityMismatch { .. } => TypeErrorKind::Type,
            TypeError::MissingField { .. } => TypeErrorKind::Type,
            TypeError::DuplicateField { .. } => TypeErrorKind::Type,
            TypeError::InvalidPattern { .. } => TypeErrorKind::Pattern,
            TypeError::NonExhaustivePatterns { .. } => TypeErrorKind::Pattern,
            TypeError::UnreachablePattern { .. } => TypeErrorKind::Pattern,
            TypeError::ConstraintViolation { .. } => TypeErrorKind::Constraint,
            TypeError::RecursiveType { .. } => TypeErrorKind::Type,
            TypeError::AmbiguousType { .. } => TypeErrorKind::Type,
            TypeError::EffectViolation { .. } => TypeErrorKind::Effect,
            TypeError::Custom { kind, .. } => *kind,
        }
    }

    /// Convert to a diagnostic
    pub fn to_diagnostic(&self) -> TypeDiagnostic {
        DiagnosticBuilder::new(self.clone()).build()
    }

    /// Generate suggestions for fixing this error
    pub fn generate_suggestions(&self) -> Vec<String> {
        match self {
            TypeError::TypeMismatch { expected, actual, suggestion, .. } => {
                let mut suggestions = Vec::new();
                
                if let Some(suggestion) = suggestion {
                    suggestions.push(suggestion.clone());
                }
                
                // Generate type-specific suggestions
                match (expected, actual) {
                    (SemanticType::Primitive(exp), SemanticType::Primitive(act)) => {
                        suggestions.push(format!(
                            "Convert {} to {} using appropriate conversion function",
                            format_primitive_type(act),
                            format_primitive_type(exp)
                        ));
                    }
                    (SemanticType::List(_), SemanticType::Primitive(_)) => {
                        suggestions.push("Wrap the value in a list using [value]".to_string());
                    }
                    (SemanticType::Primitive(_), SemanticType::List(_)) => {
                        suggestions.push("Extract a single element from the list".to_string());
                    }
                    _ => {
                        suggestions.push("Check the type annotation or function signature".to_string());
                    }
                }
                
                suggestions
            }
            TypeError::UndefinedVariable { suggestions, .. } => suggestions.clone(),
            TypeError::ArityMismatch { expected, actual, .. } => {
                if *actual < *expected {
                    vec![format!("Add {} more argument(s)", expected - actual)]
                } else {
                    vec![format!("Remove {} argument(s)", actual - expected)]
                }
            }
            TypeError::MissingField { available_fields, .. } => {
                let mut suggestions = vec!["Check the field name for typos".to_string()];
                if !available_fields.is_empty() {
                    suggestions.push(format!(
                        "Available fields are: {}",
                        available_fields.join(", ")
                    ));
                }
                suggestions
            }
            _ => vec!["Check the type definitions and annotations".to_string()],
        }
    }
}

impl DiagnosticBuilder {
    /// Create a new diagnostic builder
    pub fn new(error: TypeError) -> Self {
        Self {
            error,
            primary_label: None,
            secondary_labels: Vec::new(),
            help: Vec::new(),
            notes: Vec::new(),
            severity: DiagnosticSeverity::Error,
            code: None,
        }
    }

    /// Set the primary label
    pub fn with_primary_label(mut self, location: Span, message: String) -> Self {
        self.primary_label = Some(DiagnosticLabel {
            location,
            message,
            style: LabelStyle::Primary,
        });
        self
    }

    /// Add a secondary label
    pub fn with_secondary_label(mut self, location: Span, message: String) -> Self {
        self.secondary_labels.push(DiagnosticLabel {
            location,
            message,
            style: LabelStyle::Secondary,
        });
        self
    }

    /// Add help text
    pub fn with_help(mut self, help: String) -> Self {
        self.help.push(help);
        self
    }

    /// Add a note
    pub fn with_note(mut self, note: String) -> Self {
        self.notes.push(note);
        self
    }

    /// Set the severity
    pub fn with_severity(mut self, severity: DiagnosticSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the error code
    pub fn with_code(mut self, code: String) -> Self {
        self.code = Some(code);
        self
    }

    /// Build the diagnostic
    pub fn build(mut self) -> TypeDiagnostic {
        // Generate default primary label if not provided
        if self.primary_label.is_none() {
            let location = self.error.location();
            let message = match &self.error {
                TypeError::TypeMismatch { expected, actual, .. } => {
                    format!("expected {}, found {}", 
                           format_semantic_type(expected), 
                           format_semantic_type(actual))
                }
                TypeError::UndefinedVariable { name, .. } => {
                    format!("undefined variable '{}'", name)
                }
                TypeError::ArityMismatch { expected, actual, .. } => {
                    format!("expected {} arguments, found {}", expected, actual)
                }
                _ => "type error".to_string(),
            };

            self.primary_label = Some(DiagnosticLabel {
                location,
                message,
                style: LabelStyle::Primary,
            });
        }

        // Generate suggestions as help
        let suggestions = self.error.generate_suggestions();
        for suggestion in suggestions {
            self.help.push(suggestion);
        }

        TypeDiagnostic {
            error: self.error,
            primary_label: self.primary_label.unwrap(),
            secondary_labels: self.secondary_labels,
            help: self.help,
            notes: self.notes,
            severity: self.severity,
            code: self.code,
        }
    }
}

impl TypeDiagnostic {
    /// Create a simple diagnostic from an error
    pub fn from_error(error: TypeError) -> Self {
        error.to_diagnostic()
    }

    /// Get all locations referenced in this diagnostic
    pub fn all_locations(&self) -> Vec<Span> {
        let mut locations = vec![self.primary_label.location];
        locations.extend(self.secondary_labels.iter().map(|label| label.location));
        locations
    }

    /// Check if this diagnostic is an error
    pub fn is_error(&self) -> bool {
        self.severity >= DiagnosticSeverity::Error
    }

    /// Check if this diagnostic is a warning
    pub fn is_warning(&self) -> bool {
        self.severity == DiagnosticSeverity::Warning
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeError::TypeMismatch { expected, actual, .. } => {
                write!(f, "type mismatch: expected {}, found {}", 
                       format_semantic_type(expected), 
                       format_semantic_type(actual))
            }
            TypeError::UndefinedVariable { name, .. } => {
                write!(f, "undefined variable '{}'", name)
            }
            TypeError::ArityMismatch { expected, actual, function_name, .. } => {
                if let Some(name) = function_name {
                    write!(f, "function '{}' expects {} arguments, found {}", name, expected, actual)
                } else {
                    write!(f, "function expects {} arguments, found {}", expected, actual)
                }
            }
            TypeError::MissingField { field_name, record_type, .. } => {
                write!(f, "field '{}' not found in type {}", field_name, format_semantic_type(record_type))
            }
            TypeError::OccursCheck { variable, .. } => {
                write!(f, "occurs check failed for variable {:?}", variable)
            }
            TypeError::Custom { message, .. } => {
                write!(f, "{}", message)
            }
            _ => write!(f, "type error"),
        }
    }
}

impl fmt::Display for TypeContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeContext::FunctionApplication { function_name: Some(name), argument_index } => {
                write!(f, "in argument {} of function '{}'", argument_index + 1, name)
            }
            TypeContext::FunctionApplication { function_name: None, argument_index } => {
                write!(f, "in argument {}", argument_index + 1)
            }
            TypeContext::Assignment { variable_name } => {
                write!(f, "in assignment to variable '{}'", variable_name)
            }
            TypeContext::Return { function_name: Some(name) } => {
                write!(f, "in return value of function '{}'", name)
            }
            TypeContext::Return { function_name: None } => {
                write!(f, "in return value")
            }
            TypeContext::FieldAccess { field_name } => {
                write!(f, "in access to field '{}'", field_name)
            }
            TypeContext::PatternMatch => write!(f, "in pattern match"),
            TypeContext::Annotation => write!(f, "in type annotation"),
            TypeContext::BinaryOperation { operator } => {
                write!(f, "in binary operation '{}'", operator)
            }
            TypeContext::ListElement { index } => {
                write!(f, "in list element {}", index)
            }
            TypeContext::RecordField { field_name } => {
                write!(f, "in record field '{}'", field_name)
            }
            TypeContext::Other(context) => write!(f, "{}", context),
        }
    }
}

impl std::error::Error for TypeError {}

// Helper functions for formatting types

fn format_semantic_type(semantic_type: &SemanticType) -> String {
    match semantic_type {
        SemanticType::Primitive(prim) => format_primitive_type(prim),
        SemanticType::Variable(var) => format!("'{}", var),
        SemanticType::Function { params, return_type, .. } => {
            let param_str = params
                .iter()
                .map(format_semantic_type)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({}) -> {}", param_str, format_semantic_type(return_type))
        }
        SemanticType::List(element_type) => {
            format!("[{}]", format_semantic_type(element_type))
        }
        SemanticType::Record(fields) => {
            let field_str = fields
                .iter()
                .map(|(name, field_type)| format!("{}: {}", name, format_semantic_type(field_type)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{ {} }}", field_str)
        }
        SemanticType::Union(types) => {
            let type_str = types
                .iter()
                .map(format_semantic_type)
                .collect::<Vec<_>>()
                .join(" | ");
            format!("({})", type_str)
        }
        SemanticType::Generic { name, .. } => name.clone(),
        SemanticType::Complex { name, .. } => name.clone(),
    }
}

fn format_primitive_type(prim: &prism_ast::PrimitiveType) -> String {
    match prim {
        prism_ast::PrimitiveType::Integer(_) => "Int".to_string(),
        prism_ast::PrimitiveType::Float(_) => "Float".to_string(),
        prism_ast::PrimitiveType::String => "String".to_string(),
        prism_ast::PrimitiveType::Boolean => "Bool".to_string(),
        prism_ast::PrimitiveType::Char => "Char".to_string(),
        prism_ast::PrimitiveType::Unit => "()".to_string(),
        prism_ast::PrimitiveType::Never => "!".to_string(),
        prism_ast::PrimitiveType::Int32 => "Int32".to_string(),
        prism_ast::PrimitiveType::Int64 => "Int64".to_string(),
        prism_ast::PrimitiveType::Float64 => "Float64".to_string(),
    }
} 