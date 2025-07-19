//! Analysis and Validation
//!
//! This module contains components for analyzing and validating parsed code:
//! - Semantic context extraction for AI comprehension
//! - Token-based semantic analysis for pattern recognition
//! - Constraint validation for semantic types and business rules
//! - Parser combinators for composable parsing patterns
//!
//! These components work together to ensure that parsed code not only
//! follows correct syntax but also carries meaningful semantic information
//! and satisfies business constraints, supporting Prism's AI-first design.

pub mod semantic_context_extractor;
pub mod token_semantic_analyzer;
pub mod constraint_validation;
pub mod combinators;

// Re-export analysis types for convenience
pub use semantic_context_extractor::{SemanticContextExtractor, DomainClassifier, AISummary};
pub use token_semantic_analyzer::{
    TokenSemanticAnalyzer, TokenSemanticSummary, ModuleInfo, FunctionInfo, 
    TypeInfo, CapabilityInfo, SemanticPattern, PatternType, IdentifierUsage, SemanticRole
};
pub use constraint_validation::{ConstraintValidator, ValidationConfig, ValidationResult, ValidationError, ValidationStrictness};
pub use combinators::{Combinator, CombinatorResult}; 