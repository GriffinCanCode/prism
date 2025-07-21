//! Diagnostic Collection - Error, Warning, and Hint Management
//!
//! This module implements comprehensive diagnostic collection with AI-enhanced
//! error reporting and structured diagnostic information for better user experience.
//!
//! **Conceptual Responsibility**: Diagnostic collection and reporting
//! **What it does**: Collect diagnostics, format messages, provide AI suggestions
//! **What it doesn't do**: Generate code, manage compilation state, handle performance

use prism_common::span::Span;
use serde::{Serialize, Deserialize};

/// Diagnostic collector for errors, warnings, and hints
#[derive(Debug, Clone)]
pub struct DiagnosticCollector {
    /// Collected diagnostics
    pub diagnostics: Vec<Diagnostic>,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
    /// Hint count
    pub hint_count: usize,
}

/// Diagnostic message with comprehensive context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Diagnostic severity level
    pub level: DiagnosticLevel,
    /// Error code (e.g., "E001", "W042")
    pub code: Option<String>,
    /// Primary diagnostic message
    pub message: String,
    /// Source location where diagnostic occurred
    pub location: Span,
    /// Additional labels for context
    pub labels: Vec<DiagnosticLabel>,
    /// Help text for resolution
    pub help: Option<String>,
    /// AI-generated suggestions
    pub ai_suggestions: Vec<AISuggestion>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticLevel {
    /// Error (compilation fails)
    Error,
    /// Warning (compilation succeeds but issue noted)
    Warning,
    /// Hint (suggestion for improvement)
    Hint,
    /// Info (informational message)
    Info,
}

/// Diagnostic label for additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticLabel {
    /// Label text
    pub text: String,
    /// Source location for this label
    pub location: Span,
    /// Label styling/importance
    pub style: LabelStyle,
}

/// Label styling options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelStyle {
    /// Primary label (main issue location)
    Primary,
    /// Secondary label (related context)
    Secondary,
    /// Note label (additional information)
    Note,
}

/// AI-generated suggestion for resolving issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISuggestion {
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Human-readable suggestion text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Code replacement suggestion
    pub replacement: Option<String>,
    /// Additional context or explanation
    pub context: Option<String>,
}

/// Types of AI suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Fix for an error
    Fix,
    /// Performance improvement
    Performance,
    /// Style improvement
    Style,
    /// Semantic improvement
    Semantic,
    /// Security improvement
    Security,
    /// Accessibility improvement
    Accessibility,
}

impl DiagnosticCollector {
    /// Create a new diagnostic collector
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            error_count: 0,
            warning_count: 0,
            hint_count: 0,
        }
    }

    /// Add a diagnostic to the collection
    pub fn add(&mut self, diagnostic: Diagnostic) {
        match diagnostic.level {
            DiagnosticLevel::Error => self.error_count += 1,
            DiagnosticLevel::Warning => self.warning_count += 1,
            DiagnosticLevel::Hint => self.hint_count += 1,
            DiagnosticLevel::Info => {},
        }
        self.diagnostics.push(diagnostic);
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get all diagnostics
    pub fn get_diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Get diagnostics filtered by level
    pub fn get_diagnostics_by_level(&self, level: DiagnosticLevel) -> Vec<&Diagnostic> {
        self.diagnostics.iter()
            .filter(|d| d.level == level)
            .collect()
    }

    /// Get total diagnostic count
    pub fn total_count(&self) -> usize {
        self.diagnostics.len()
    }

    /// Clear all diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
        self.hint_count = 0;
    }

    /// Get a summary string of diagnostic counts
    pub fn summary(&self) -> String {
        if self.error_count > 0 {
            format!("{} errors, {} warnings, {} hints", 
                    self.error_count, self.warning_count, self.hint_count)
        } else if self.warning_count > 0 {
            format!("{} warnings, {} hints", self.warning_count, self.hint_count)
        } else if self.hint_count > 0 {
            format!("{} hints", self.hint_count)
        } else {
            "No issues".to_string()
        }
    }

    /// Get diagnostics with AI suggestions
    pub fn get_diagnostics_with_ai_suggestions(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter()
            .filter(|d| !d.ai_suggestions.is_empty())
            .collect()
    }

    /// Get high-confidence AI suggestions
    pub fn get_high_confidence_suggestions(&self, min_confidence: f64) -> Vec<&AISuggestion> {
        self.diagnostics.iter()
            .flat_map(|d| &d.ai_suggestions)
            .filter(|s| s.confidence >= min_confidence)
            .collect()
    }
}

impl Diagnostic {
    /// Create a new error diagnostic
    pub fn error(message: String, location: Span) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            code: None,
            message,
            location,
            labels: Vec::new(),
            help: None,
            ai_suggestions: Vec::new(),
        }
    }

    /// Create a new warning diagnostic
    pub fn warning(message: String, location: Span) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            code: None,
            message,
            location,
            labels: Vec::new(),
            help: None,
            ai_suggestions: Vec::new(),
        }
    }

    /// Create a new hint diagnostic
    pub fn hint(message: String, location: Span) -> Self {
        Self {
            level: DiagnosticLevel::Hint,
            code: None,
            message,
            location,
            labels: Vec::new(),
            help: None,
            ai_suggestions: Vec::new(),
        }
    }

    /// Create a new info diagnostic
    pub fn info(message: String, location: Span) -> Self {
        Self {
            level: DiagnosticLevel::Info,
            code: None,
            message,
            location,
            labels: Vec::new(),
            help: None,
            ai_suggestions: Vec::new(),
        }
    }

    /// Add an error code to this diagnostic
    pub fn with_code(mut self, code: String) -> Self {
        self.code = Some(code);
        self
    }

    /// Add a label to this diagnostic
    pub fn with_label(mut self, label: DiagnosticLabel) -> Self {
        self.labels.push(label);
        self
    }

    /// Add help text to this diagnostic
    pub fn with_help(mut self, help: String) -> Self {
        self.help = Some(help);
        self
    }

    /// Add an AI suggestion to this diagnostic
    pub fn with_ai_suggestion(mut self, suggestion: AISuggestion) -> Self {
        self.ai_suggestions.push(suggestion);
        self
    }

    /// Check if this diagnostic has AI suggestions
    pub fn has_ai_suggestions(&self) -> bool {
        !self.ai_suggestions.is_empty()
    }

    /// Get the highest confidence AI suggestion
    pub fn best_ai_suggestion(&self) -> Option<&AISuggestion> {
        self.ai_suggestions.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
    }
}

impl DiagnosticLabel {
    /// Create a primary label
    pub fn primary(text: String, location: Span) -> Self {
        Self {
            text,
            location,
            style: LabelStyle::Primary,
        }
    }

    /// Create a secondary label
    pub fn secondary(text: String, location: Span) -> Self {
        Self {
            text,
            location,
            style: LabelStyle::Secondary,
        }
    }

    /// Create a note label
    pub fn note(text: String, location: Span) -> Self {
        Self {
            text,
            location,
            style: LabelStyle::Note,
        }
    }
}

impl AISuggestion {
    /// Create a new AI suggestion
    pub fn new(suggestion_type: SuggestionType, text: String, confidence: f64) -> Self {
        Self {
            suggestion_type,
            text,
            confidence,
            replacement: None,
            context: None,
        }
    }

    /// Add a code replacement to this suggestion
    pub fn with_replacement(mut self, replacement: String) -> Self {
        self.replacement = Some(replacement);
        self
    }

    /// Add context to this suggestion
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }

    /// Check if this suggestion has a code replacement
    pub fn has_replacement(&self) -> bool {
        self.replacement.is_some()
    }

    /// Check if this is a high-confidence suggestion
    pub fn is_high_confidence(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticLevel::Error => write!(f, "error"),
            DiagnosticLevel::Warning => write!(f, "warning"),
            DiagnosticLevel::Hint => write!(f, "hint"),
            DiagnosticLevel::Info => write!(f, "info"),
        }
    }
}

impl std::fmt::Display for SuggestionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestionType::Fix => write!(f, "fix"),
            SuggestionType::Performance => write!(f, "performance"),
            SuggestionType::Style => write!(f, "style"),
            SuggestionType::Semantic => write!(f, "semantic"),
            SuggestionType::Security => write!(f, "security"),
            SuggestionType::Accessibility => write!(f, "accessibility"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_collector_creation() {
        let collector = DiagnosticCollector::new();
        
        assert_eq!(collector.error_count, 0);
        assert_eq!(collector.warning_count, 0);
        assert_eq!(collector.hint_count, 0);
        assert!(!collector.has_errors());
    }

    #[test]
    fn test_diagnostic_addition() {
        let mut collector = DiagnosticCollector::new();
        
        let diagnostic = Diagnostic::error("Test error".to_string(), Span::dummy());
        collector.add(diagnostic);
        
        assert_eq!(collector.error_count, 1);
        assert!(collector.has_errors());
        assert_eq!(collector.total_count(), 1);
    }

    #[test]
    fn test_diagnostic_filtering() {
        let mut collector = DiagnosticCollector::new();
        
        collector.add(Diagnostic::error("Error".to_string(), Span::dummy()));
        collector.add(Diagnostic::warning("Warning".to_string(), Span::dummy()));
        collector.add(Diagnostic::hint("Hint".to_string(), Span::dummy()));
        
        let errors = collector.get_diagnostics_by_level(DiagnosticLevel::Error);
        assert_eq!(errors.len(), 1);
        
        let warnings = collector.get_diagnostics_by_level(DiagnosticLevel::Warning);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_ai_suggestions() {
        let suggestion = AISuggestion::new(
            SuggestionType::Fix,
            "Consider using a different approach".to_string(),
            0.85,
        ).with_replacement("new_code()".to_string());
        
        assert!(suggestion.has_replacement());
        assert!(suggestion.is_high_confidence(0.8));
        assert!(!suggestion.is_high_confidence(0.9));
    }

    #[test]
    fn test_diagnostic_builder() {
        let diagnostic = Diagnostic::error("Test error".to_string(), Span::dummy())
            .with_code("E001".to_string())
            .with_help("Try this instead".to_string())
            .with_ai_suggestion(AISuggestion::new(
                SuggestionType::Fix,
                "AI suggestion".to_string(),
                0.9,
            ));
        
        assert_eq!(diagnostic.code, Some("E001".to_string()));
        assert!(diagnostic.help.is_some());
        assert!(diagnostic.has_ai_suggestions());
        assert_eq!(diagnostic.best_ai_suggestion().unwrap().confidence, 0.9);
    }
} 