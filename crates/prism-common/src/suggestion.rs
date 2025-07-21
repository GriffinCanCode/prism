//! Shared Suggestion and Diagnostic Interfaces
//!
//! This module provides common interfaces for diagnostic collection and suggestion
//! generation that can be used across multiple crates without circular dependencies.

use crate::diagnostics::{Diagnostic, Severity};
use crate::span::Span;
use serde::{Serialize, Deserialize};

/// Diagnostic collector interface for gathering compilation diagnostics
pub trait DiagnosticCollector {
    /// Add a diagnostic to the collection
    fn add_diagnostic(&mut self, diagnostic: Diagnostic);
    
    /// Get all collected diagnostics
    fn get_diagnostics(&self) -> &[Diagnostic];
    
    /// Check if there are any errors
    fn has_errors(&self) -> bool;
    
    /// Get error count
    fn error_count(&self) -> usize;
    
    /// Get warning count
    fn warning_count(&self) -> usize;
    
    /// Clear all diagnostics
    fn clear(&mut self);
}

/// Basic implementation of diagnostic collector
#[derive(Debug, Clone, Default)]
pub struct BasicDiagnosticCollector {
    /// Collected diagnostics
    pub diagnostics: Vec<Diagnostic>,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
}

impl DiagnosticCollector for BasicDiagnosticCollector {
    fn add_diagnostic(&mut self, diagnostic: Diagnostic) {
        match diagnostic.severity {
            Severity::Error => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            _ => {}
        }
        self.diagnostics.push(diagnostic);
    }
    
    fn get_diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
    
    fn has_errors(&self) -> bool {
        self.error_count > 0
    }
    
    fn error_count(&self) -> usize {
        self.error_count
    }
    
    fn warning_count(&self) -> usize {
        self.warning_count
    }
    
    fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
    }
}

/// Metadata for suggestion generation and ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionMetadata {
    /// Context that led to the suggestion
    pub context: String,
    /// Confidence in the suggestion (0.0 to 1.0)
    pub confidence: f64,
    /// Pattern tags for learning
    pub pattern_tags: Vec<String>,
    /// Source location related to suggestion
    pub location: Option<Span>,
    /// Whether suggestion can be auto-applied
    pub auto_applicable: bool,
}

/// Interface for suggestion providers
pub trait SuggestionProvider {
    /// Generate suggestions for a given context
    fn generate_suggestions(&mut self, context: &SuggestionContext) -> Vec<SuggestionWithMetadata>;
    
    /// Get provider name for debugging
    fn provider_name(&self) -> &'static str;
    
    /// Get statistics about suggestions generated
    fn get_statistics(&self) -> SuggestionStatistics;
}

/// Context for suggestion generation
#[derive(Debug, Clone)]
pub struct SuggestionContext {
    /// Error or issue that triggered suggestion need
    pub trigger: SuggestionTrigger,
    /// Source location
    pub location: Span,
    /// Additional context information
    pub context_data: std::collections::HashMap<String, String>,
}

/// What triggered the need for suggestions
#[derive(Debug, Clone)]
pub enum SuggestionTrigger {
    /// Parse error occurred
    ParseError(String),
    /// Semantic analysis issue
    SemanticIssue(String),
    /// Type mismatch
    TypeMismatch(String),
    /// Business rule violation
    BusinessRuleViolation(String),
    /// Performance concern
    PerformanceConcern(String),
    /// Security issue
    SecurityIssue(String),
}

/// Suggestion with associated metadata
#[derive(Debug, Clone)]
pub struct SuggestionWithMetadata {
    /// The actual suggestion
    pub suggestion: crate::diagnostics::Suggestion,
    /// Associated metadata
    pub metadata: SuggestionMetadata,
}

/// Statistics about suggestion generation
#[derive(Debug, Clone, Default)]
pub struct SuggestionStatistics {
    /// Total suggestions generated
    pub total_generated: usize,
    /// Successfully applied suggestions
    pub successful_applications: usize,
    /// Average confidence score
    pub average_confidence: f64,
}

impl Default for SuggestionMetadata {
    fn default() -> Self {
        Self {
            context: String::new(),
            confidence: 0.5,
            pattern_tags: Vec::new(),
            location: None,
            auto_applicable: false,
        }
    }
}

impl SuggestionMetadata {
    /// Create new suggestion metadata
    pub fn new(context: String, confidence: f64) -> Self {
        Self {
            context,
            confidence,
            pattern_tags: Vec::new(),
            location: None,
            auto_applicable: false,
        }
    }
    
    /// Add pattern tag
    pub fn with_pattern_tag(mut self, tag: String) -> Self {
        self.pattern_tags.push(tag);
        self
    }
    
    /// Set location
    pub fn with_location(mut self, location: Span) -> Self {
        self.location = Some(location);
        self
    }
    
    /// Mark as auto-applicable
    pub fn auto_applicable(mut self) -> Self {
        self.auto_applicable = true;
        self
    }
} 