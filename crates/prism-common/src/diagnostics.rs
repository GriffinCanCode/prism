//! Diagnostic reporting and error recovery utilities

use std::fmt;
use crate::span::Span;

/// Severity level of a diagnostic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Severity {
    /// An error that prevents compilation
    Error,
    /// A warning that doesn't prevent compilation
    Warning,
    /// An informational message
    Info,
    /// A hint or suggestion
    Hint,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
            Self::Info => write!(f, "info"),
            Self::Hint => write!(f, "hint"),
        }
    }
}

/// A diagnostic message with location and severity
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Diagnostic {
    /// Severity level
    pub severity: Severity,
    /// Main diagnostic message
    pub message: String,
    /// Source location where the diagnostic occurred
    pub span: Span,
    /// Optional additional labels for related locations
    pub labels: Vec<Label>,
    /// Optional suggestions for fixing the issue
    pub suggestions: Vec<Suggestion>,
    /// Optional help text
    pub help: Option<String>,
    /// Optional note
    pub note: Option<String>,
}

impl Diagnostic {
    /// Create a new diagnostic
    pub fn new(severity: Severity, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity,
            message: message.into(),
            span,
            labels: Vec::new(),
            suggestions: Vec::new(),
            help: None,
            note: None,
        }
    }

    /// Create an error diagnostic
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Self::new(Severity::Error, message, span)
    }

    /// Create a warning diagnostic
    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Self::new(Severity::Warning, message, span)
    }

    /// Create an info diagnostic
    pub fn info(message: impl Into<String>, span: Span) -> Self {
        Self::new(Severity::Info, message, span)
    }

    /// Create a hint diagnostic
    pub fn hint(message: impl Into<String>, span: Span) -> Self {
        Self::new(Severity::Hint, message, span)
    }

    /// Add a label to this diagnostic
    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    /// Add a suggestion to this diagnostic
    pub fn with_suggestion(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Add help text to this diagnostic
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Add a note to this diagnostic
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Check if this diagnostic is an error
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// Check if this diagnostic is a warning
    pub fn is_warning(&self) -> bool {
        self.severity == Severity::Warning
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.message)?;
        
        if !self.labels.is_empty() {
            write!(f, "\n  at {}", self.span)?;
            for label in &self.labels {
                write!(f, "\n  {}: {}", label.span, label.message)?;
            }
        }

        if let Some(help) = &self.help {
            write!(f, "\n  help: {}", help)?;
        }

        if let Some(note) = &self.note {
            write!(f, "\n  note: {}", note)?;
        }

        Ok(())
    }
}

/// A label pointing to a specific source location
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Label {
    /// Source location
    pub span: Span,
    /// Label message
    pub message: String,
    /// Label style
    pub style: LabelStyle,
}

impl Label {
    /// Create a new label
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            style: LabelStyle::Primary,
        }
    }

    /// Create a primary label
    pub fn primary(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            style: LabelStyle::Primary,
        }
    }

    /// Create a secondary label
    pub fn secondary(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            style: LabelStyle::Secondary,
        }
    }
}

/// Style of a diagnostic label
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LabelStyle {
    /// Primary label (usually the main issue)
    Primary,
    /// Secondary label (related context)
    Secondary,
}

/// A suggestion for fixing a diagnostic
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Suggestion {
    /// Description of the suggestion
    pub message: String,
    /// Specific code replacements
    pub replacements: Vec<Replacement>,
    /// Suggestion style
    pub style: SuggestionStyle,
}

impl Suggestion {
    /// Create a new suggestion
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            replacements: Vec::new(),
            style: SuggestionStyle::MaybeIncorrect,
        }
    }

    /// Create a suggestion with replacements
    pub fn with_replacements(
        message: impl Into<String>,
        replacements: Vec<Replacement>,
    ) -> Self {
        Self {
            message: message.into(),
            replacements,
            style: SuggestionStyle::MaybeIncorrect,
        }
    }

    /// Mark this suggestion as definitely correct
    pub fn definitely_correct(mut self) -> Self {
        self.style = SuggestionStyle::DefinitelyCorrect;
        self
    }

    /// Mark this suggestion as maybe incorrect
    pub fn maybe_incorrect(mut self) -> Self {
        self.style = SuggestionStyle::MaybeIncorrect;
        self
    }
}

/// Style of a suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SuggestionStyle {
    /// This suggestion is probably correct
    DefinitelyCorrect,
    /// This suggestion might be incorrect
    MaybeIncorrect,
}

/// A specific code replacement
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Replacement {
    /// Span to replace
    pub span: Span,
    /// Replacement text
    pub text: String,
}

impl Replacement {
    /// Create a new replacement
    pub fn new(span: Span, text: impl Into<String>) -> Self {
        Self {
            span,
            text: text.into(),
        }
    }
}

/// A collection of diagnostics
#[derive(Debug, Default, Clone)]
pub struct DiagnosticBag {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBag {
    /// Create a new diagnostic bag
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a diagnostic
    pub fn add(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Add an error
    pub fn error(&mut self, message: impl Into<String>, span: Span) {
        self.add(Diagnostic::error(message, span));
    }

    /// Add a warning
    pub fn warning(&mut self, message: impl Into<String>, span: Span) {
        self.add(Diagnostic::warning(message, span));
    }

    /// Add an info diagnostic
    pub fn info(&mut self, message: impl Into<String>, span: Span) {
        self.add(Diagnostic::info(message, span));
    }

    /// Add a hint
    pub fn hint(&mut self, message: impl Into<String>, span: Span) {
        self.add(Diagnostic::hint(message, span));
    }

    /// Get all diagnostics
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_error)
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        self.diagnostics.iter().any(Diagnostic::is_warning)
    }

    /// Get the number of errors
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_error()).count()
    }

    /// Get the number of warnings
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_warning()).count()
    }

    /// Check if the bag is empty
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Clear all diagnostics
    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }

    /// Merge another diagnostic bag into this one
    pub fn merge(&mut self, other: DiagnosticBag) {
        self.diagnostics.extend(other.diagnostics);
    }
}

impl fmt::Display for DiagnosticBag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, diagnostic) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", diagnostic)?;
        }
        Ok(())
    }
} 