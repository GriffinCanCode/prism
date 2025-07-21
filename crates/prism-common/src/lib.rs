//! Common types and utilities shared across the Prism compiler infrastructure
//!
//! This crate provides foundational types, utilities, and interfaces that are
//! used throughout the Prism compiler ecosystem. It maintains zero dependencies
//! on other Prism crates to prevent circular dependencies.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod diagnostics;
pub mod span;
pub mod symbol;
pub mod ai_metadata;     // NEW: Shared AI metadata interfaces
pub mod suggestion;      // NEW: Shared suggestion interfaces

// Re-export commonly used types
pub use diagnostics::{Diagnostic, Suggestion, Severity as DiagnosticSeverity};
pub use span::{Span, Position};
pub use symbol::{Symbol, SymbolTable};
pub use ai_metadata::{AIMetadata, SemanticContextEntry, BusinessRuleEntry}; // NEW
pub use suggestion::{DiagnosticCollector, SuggestionMetadata}; // NEW

use std::fmt;
use thiserror::Error;

/// A unique identifier for a source file or input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceId(u32);

impl SourceId {
    /// Create a new source ID
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "source:{}", self.0)
    }
}

/// A unique identifier for AST nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(u32);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node:{}", self.0)
    }
}

/// Common error types used throughout the Prism compiler
#[derive(Error, Debug)]
pub enum PrismError {
    /// Parse error with location information
    #[error("Parse error at {span}: {message}")]
    Parse {
        /// Source location of the error
        span: span::Span,
        /// Error message
        message: String,
    },

    /// Type checking error
    #[error("Type error at {span}: {message}")]
    Type {
        /// Source location of the error
        span: span::Span,
        /// Error message
        message: String,
    },

    /// Semantic analysis error
    #[error("Semantic error at {span}: {message}")]
    Semantic {
        /// Source location of the error
        span: span::Span,
        /// Error message
        message: String,
    },

    /// Effect system violation
    #[error("Effect error at {span}: {message}")]
    Effect {
        /// Source location of the error
        span: span::Span,
        /// Error message
        message: String,
    },

    /// Code generation error
    #[error("Codegen error: {message}")]
    Codegen {
        /// Error message
        message: String,
    },

    /// AI integration error
    #[error("AI error: {message}")]
    Ai {
        /// Error message
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error with message
    #[error("{message}")]
    Generic {
        /// Error message
        message: String,
    },
}

/// Result type alias for Prism operations
pub type Result<T> = std::result::Result<T, PrismError>;

/// Trait for types that can be converted to a span
pub trait HasSpan {
    /// Get the span for this item
    fn span(&self) -> span::Span;
}

/// Trait for types that have a node ID
pub trait HasNodeId {
    /// Get the node ID for this item
    fn node_id(&self) -> NodeId;
} 