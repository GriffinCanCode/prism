//! Prism Constraint Validation Engine
//!
//! This crate embodies the single concept of "Constraint Validation".
//! Following Prism's Conceptual Cohesion principle, this crate is responsible
//! for ONE thing: validating semantic constraints and business rules through
//! expression evaluation.
//!
//! **Conceptual Responsibility**: Constraint validation and expression evaluation
//! **What it does**: validates type constraints, evaluates predicates, checks business rules
//! **What it doesn't do**: type inference, semantic analysis, code generation
//!
//! ## Design Principles
//!
//! 1. **Zero Dependencies on Other Prism Crates**: Only depends on prism-common and prism-ast
//! 2. **Pluggable Validators**: Support for custom constraint validators
//! 3. **Expression Evaluation**: Built-in expression evaluator for constraint predicates
//! 4. **Performance Focused**: Efficient validation with caching
//! 5. **Error Rich**: Detailed error messages with suggestions

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod evaluator;
pub mod validators;
pub mod engine;
pub mod error;
pub mod builtin;

// Re-export main types
pub use engine::{ConstraintEngine, ConstraintContext};
pub use validators::{ConstraintValidator, ValidationResult, ValidationError, ValidationContext, ValidationConfig, ValidationWarning};
pub use evaluator::{ExpressionEvaluator, EvaluationContext, EvaluationResult};
pub use error::{ConstraintError, ConstraintResult};

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Value types that can be used in constraint evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Null value
    Null,
    /// Array of values
    Array(Vec<ConstraintValue>),
    /// Object/map of values
    Object(HashMap<String, ConstraintValue>),
}

impl ConstraintValue {
    /// Check if this value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Boolean(b) => *b,
            Self::Integer(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            Self::String(s) => !s.is_empty(),
            Self::Null => false,
            Self::Array(arr) => !arr.is_empty(),
            Self::Object(obj) => !obj.is_empty(),
        }
    }

    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Boolean(_) => "boolean",
            Self::Integer(_) => "integer",
            Self::Float(_) => "float",
            Self::String(_) => "string",
            Self::Null => "null",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }

    /// Try to convert to boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to convert to integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Try to convert to float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to convert to string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to convert to array
    pub fn as_array(&self) -> Option<&Vec<ConstraintValue>> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to convert to object
    pub fn as_object(&self) -> Option<&HashMap<String, ConstraintValue>> {
        match self {
            Self::Object(obj) => Some(obj),
            _ => None,
        }
    }
} 