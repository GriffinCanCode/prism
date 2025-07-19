//! Semantic normalization for multi-syntax parsing.
//! 
//! This module converts parsed syntax from any style into a canonical
//! semantic representation that preserves meaning while providing a
//! consistent format for downstream processing.

pub mod normalizer;
pub mod canonical_form;
pub mod metadata;

pub use normalizer::{Normalizer, NormalizationResult, NormalizationContext};
pub use canonical_form::{CanonicalForm, CanonicalNode, CanonicalStructure};
pub use metadata::{MetadataPreserver, SemanticMetadata, AIMetadata}; 