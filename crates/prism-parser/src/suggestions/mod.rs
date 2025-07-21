//! Context-Guided Diagnostic Suggestions
//!
//! This module implements intelligent, context-aware error recovery suggestions
//! that integrate with Prism's semantic analysis, business rules, and AI systems.
//! 
//! ## Conceptual Cohesion
//! 
//! This module maintains conceptual cohesion around the single responsibility of
//! "generating intelligent diagnostic suggestions from contextual analysis".
//! 
//! ## Architecture
//! 
//! - `engine.rs` - Main suggestion coordination engine
//! - `context.rs` - Context extraction and analysis
//! - `providers/` - Specialized suggestion providers by domain
//! - `ranking.rs` - Suggestion ranking and confidence scoring
//! - `pattern.rs` - Pattern matching and learning from history

pub mod engine;
pub mod context;
pub mod providers;
pub mod ranking;
pub mod pattern;

// Re-export main types
pub use engine::{SuggestionEngine, SuggestionEngineConfig};
pub use context::{SuggestionContext, ContextExtractor};
pub use ranking::{SuggestionRanker, RankedSuggestion};
pub use pattern::{PatternMatcher, SuggestionPattern};

use crate::core::error::ParseError;
use prism_common::diagnostics::Suggestion;
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Result type for suggestion operations
pub type SuggestionResult<T> = Result<T, SuggestionError>;

/// Errors that can occur during suggestion generation
#[derive(Debug, Error)]
pub enum SuggestionError {
    /// Context extraction failed
    #[error("Failed to extract context: {message}")]
    ContextExtractionFailed { message: String },
    
    /// Pattern matching failed
    #[error("Pattern matching failed: {message}")]
    PatternMatchingFailed { message: String },
    
    /// Suggestion validation failed
    #[error("Suggestion validation failed: {message}")]
    ValidationFailed { message: String },
    
    /// AI analysis failed
    #[error("AI analysis failed: {message}")]
    AIAnalysisFailed { message: String },
}

/// Enhanced diagnostic suggestion with rich context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualSuggestion {
    /// Base suggestion
    pub suggestion: Suggestion,
    
    /// Suggestion type for categorization
    pub suggestion_type: SuggestionType,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Detailed explanation of why this suggestion is recommended
    pub explanation: String,
    
    /// Context that led to this suggestion
    pub context_summary: String,
    
    /// Pattern tags for learning and analysis
    pub pattern_tags: Vec<String>,
    
    /// AI-generated insights
    pub ai_insights: Vec<String>,
    
    /// Estimated effort to apply this suggestion
    pub estimated_effort: EffortLevel,
    
    /// Whether this suggestion can be auto-applied
    pub auto_applicable: bool,
}

/// Types of suggestions for categorization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Basic syntax correction
    SyntaxFix,
    /// Type inference or annotation
    TypeGuidance,
    /// Semantic completion based on context
    SemanticCompletion,
    /// Effect system guidance
    EffectGuidance,
    /// Business rule compliance
    BusinessRuleGuidance,
    /// Module cohesion improvement
    ArchitecturalGuidance,
    /// Performance optimization
    PerformanceGuidance,
    /// Security improvement
    SecurityGuidance,
}

/// Estimated effort level for applying a suggestion
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EffortLevel {
    /// Trivial change (auto-applicable)
    Trivial,
    /// Simple change (single line/token)
    Simple,
    /// Moderate change (multiple lines)
    Moderate,
    /// Complex change (architectural)
    Complex,
    /// Major refactoring required
    Major,
} 