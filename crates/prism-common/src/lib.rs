//! Common types and utilities for the Prism language.
//!
//! This crate provides shared types, utilities, and trait interfaces that are used
//! across multiple Prism crates. It serves as the foundation layer that prevents
//! circular dependencies while providing common abstractions.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod span;
pub mod symbol;
pub mod diagnostics;
pub mod suggestion;
pub mod ai_metadata;
pub mod parsing_traits;  // NEW: Trait interfaces for parsing
pub mod optimization_traits;  // NEW: Unified optimization interfaces
pub mod source_maps;  // NEW: Unified source map generation

pub use span::{Span, SourceId, Position};
pub use symbol::{Symbol, SymbolTable};
pub use diagnostics::Diagnostic;
pub use suggestion::SuggestionTrigger;
pub use ai_metadata::AIMetadata;
// NEW: Export parsing trait interfaces
pub use parsing_traits::{
    ProgramParser, EnhancedParser, SyntaxAwareParser, PIRConstructor, ParserFactory,
    ParsingConfig, PIRConstructionConfig, ParsingMetrics, ParsingDiagnostic,
    DiagnosticLevel as ParsingDiagnosticLevel,
};
// NEW: Export optimization trait interfaces
pub use optimization_traits::{
    CodeOptimizer, BundleAnalyzer, PerformanceHintGenerator,
    OptimizationConfig, OptimizationStats, OptimizationResult, OptimizationWarning,
    OptimizerCapabilities, BundleAnalysis, PerformanceHint,
};
// NEW: Export source map interfaces
pub use source_maps::{
    SourceMapGenerator, SourceMapConfig, Mapping as SourceMapMapping, SourceMap,
};

use std::fmt;
use thiserror::Error;

/// Node identifier for AST nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub fn raw(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node:{}", self.0)
    }
}

/// Common result type for Prism operations
pub type Result<T> = std::result::Result<T, PrismError>;

/// Common error type for Prism operations
#[derive(Debug, Error)]
pub enum PrismError {
    /// Generic error with message
    #[error("{message}")]
    Generic { message: String },
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Parsing error
    #[error("Parsing error: {message}")]
    Parse { message: String },
    
    /// Semantic analysis error
    #[error("Semantic error: {message}")]
    Semantic { message: String },
    
    /// Type error
    #[error("Type error: {message}")]
    Type { message: String },
    
    /// Effect system error
    #[error("Effect error: {message}")]
    Effect { message: String },
    
    /// PIR construction error
    #[error("PIR construction error: {message}")]
    PIRConstruction { message: String },
}

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