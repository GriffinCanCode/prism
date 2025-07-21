//! Semantic normalization for multi-syntax parsing.
//! 
//! This module converts parsed syntax from any style into a canonical
//! semantic representation that preserves meaning while providing a
//! consistent format for downstream processing.
//!
//! The normalization system follows Separation of Concerns (SoC) by having:
//! - A main coordinator (normalizer.rs) that orchestrates the process
//! - Style-specific normalizers for each syntax style
//! - Shared traits defining consistent interfaces
//! - Canonical form definitions for the target representation

// Core coordination
pub mod normalizer;

// Style-specific normalizers
pub mod c_like;
pub mod python_like;
pub mod rust_like;
pub mod canonical;

// Shared infrastructure
pub mod traits;
pub mod canonical_form;
pub mod metadata;

// Public exports for the main API
pub use normalizer::{
    Normalizer, ParsedSyntax, NormalizationConfig, ValidationLevel,
    NormalizationContext, NormalizationPhase, NormalizationError,
    NormalizationWarning, WarningSeverity, NormalizationMetrics, SymbolInfo
};

// Export canonical form types
pub use canonical_form::{
    CanonicalForm, CanonicalNode, CanonicalStructure, CanonicalType,
    CanonicalExpression, CanonicalStatement, SectionType, AIMetadata,
    ComplexityMetrics
};

// Export metadata types
pub use metadata::{
    MetadataPreserver, SemanticMetadata, FormattingInfo, PreservedComment
};

// Export trait interfaces for extensibility
pub use traits::{
    StyleNormalizer, NormalizerConfig, NormalizerCapabilities, 
    PerformanceCharacteristics, ConfigurationError
};

// Style-specific normalizer exports (for advanced usage)
pub use c_like::CLikeNormalizer;
pub use python_like::PythonLikeNormalizer;
pub use rust_like::RustLikeNormalizer;
pub use canonical::CanonicalNormalizer;

// Utility functions and helpers
pub use traits::NormalizationUtils; 