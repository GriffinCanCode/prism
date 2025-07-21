//! Error handling for the Prism parser
//!
//! This module provides comprehensive error reporting with context,
//! suggestions, and recovery information for better developer experience.

use prism_common::span::Span;
use prism_lexer::{Token, TokenKind};
use std::fmt;
use thiserror::Error;

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;

/// Comprehensive parse error with context and suggestions
#[derive(Debug, Clone, Error, PartialEq)]
pub struct ParseError {
    /// The specific kind of error
    pub kind: ParseErrorKind,
    /// Source location where the error occurred
    pub span: Span,
    /// Human-readable error message
    pub message: String,
    /// Suggestions for fixing the error
    pub suggestions: Vec<String>,
    /// Severity level of the error
    pub severity: ErrorSeverity,
    /// Context tokens around the error
    pub context: Option<ErrorContext>,
}

impl ParseError {
    /// Create a new parse error
    pub fn new(kind: ParseErrorKind, span: Span, message: String) -> Self {
        Self {
            kind,
            span,
            message,
            suggestions: Vec::new(),
            severity: ErrorSeverity::Error,
            context: None,
        }
    }

    /// Add suggestions to the error
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    /// Set the error severity
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add context tokens
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Check if this error can be recovered from
    pub fn is_recoverable(&self) -> bool {
        match self.kind {
            ParseErrorKind::UnexpectedToken { .. } => true,
            ParseErrorKind::UnexpectedEof { .. } => false,
            ParseErrorKind::InvalidSyntax { .. } => true,
            ParseErrorKind::SemanticError { .. } => true,
            ParseErrorKind::RecoveryError { .. } => false,
            ParseErrorKind::InternalError { .. } => false,
            ParseErrorKind::InvalidDelimiter { .. } => true,
            ParseErrorKind::UnsupportedSyntaxStyle { .. } => false,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.message)?;
        
        if !self.suggestions.is_empty() {
            write!(f, "\nSuggestions:")?;
            for suggestion in &self.suggestions {
                write!(f, "\n  - {}", suggestion)?;
            }
        }
        
        Ok(())
    }
}

/// Specific kinds of parse errors
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// Unexpected token encountered
    UnexpectedToken {
        /// What token types were expected
        expected: Vec<TokenKind>,
        /// What token was actually found
        found: TokenKind,
    },
    /// Unexpected end of file
    UnexpectedEof {
        /// What token types were expected
        expected: Vec<TokenKind>,
    },
    /// Invalid syntax construct
    InvalidSyntax {
        /// Name of the construct being parsed
        construct: String,
        /// Additional context
        details: String,
    },
    /// Semantic error (type mismatch, etc.)
    SemanticError {
        /// The violated constraint
        constraint: String,
        /// Additional context
        details: String,
    },
    /// Error during recovery
    RecoveryError {
        /// The original error message that triggered recovery
        original_message: String,
        /// Recovery attempt details
        recovery_details: String,
    },
    /// Internal parser error
    InternalError {
        /// Error details
        details: String,
    },
    /// Invalid delimiter
    InvalidDelimiter {
        /// The invalid delimiter found
        found: TokenKind,
    },
    /// Unsupported syntax style
    UnsupportedSyntaxStyle {
        /// The unsupported syntax style
        style: prism_syntax::detection::SyntaxStyle,
    },
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Hints for improvement
    Hint,
    /// Informational messages
    Info,
    /// Warnings about potential issues
    Warning,
    /// Errors that prevent compilation
    Error,
    /// Fatal errors that stop parsing
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hint => write!(f, "hint"),
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Fatal => write!(f, "fatal"),
        }
    }
}

/// Context information around an error
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Tokens before the error location
    pub before: Vec<Token>,
    /// The problematic token
    pub current: Option<Token>,
    /// Tokens after the error location
    pub after: Vec<Token>,
}

impl ErrorContext {
    /// Create error context from token stream
    pub fn from_tokens(tokens: &[Token], current_index: usize, window_size: usize) -> Self {
        let start = current_index.saturating_sub(window_size);
        let end = (current_index + window_size + 1).min(tokens.len());
        
        let before = tokens[start..current_index].to_vec();
        let current = tokens.get(current_index).cloned();
        let after = tokens[(current_index + 1)..end].to_vec();
        
        Self {
            before,
            current,
            after,
        }
    }

    /// Get a snippet of the problematic code
    pub fn code_snippet(&self) -> String {
        let mut snippet = String::new();
        
        for token in &self.before {
            snippet.push_str(&format!("{:?} ", token.kind));
        }
        
        if let Some(ref current) = self.current {
            snippet.push_str(&format!(">>> {:?} <<<", current.kind));
        }
        
        for token in &self.after {
            snippet.push_str(&format!(" {:?}", token.kind));
        }
        
        snippet
    }
}

/// Error builder for convenient error construction
pub struct ErrorBuilder {
    kind: ParseErrorKind,
    span: Span,
    message: String,
    suggestions: Vec<String>,
    severity: ErrorSeverity,
    context: Option<ErrorContext>,
}

impl ErrorBuilder {
    /// Create a new error builder
    pub fn new(kind: ParseErrorKind, span: Span, message: String) -> Self {
        Self {
            kind,
            span,
            message,
            suggestions: Vec::new(),
            severity: ErrorSeverity::Error,
            context: None,
        }
    }

    /// Add a suggestion
    pub fn suggest(mut self, suggestion: String) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Set severity
    pub fn severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add context
    pub fn context(mut self, context: ErrorContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Build the error
    pub fn build(self) -> ParseError {
        ParseError {
            kind: self.kind,
            span: self.span,
            message: self.message,
            suggestions: self.suggestions,
            severity: self.severity,
            context: self.context,
        }
    }
}

/// Convenience functions for common error patterns
impl ParseError {
    /// Create an "unexpected token" error
    pub fn unexpected_token(
        expected: Vec<TokenKind>,
        found: TokenKind,
        span: Span,
    ) -> Self {
        let message = if expected.len() == 1 {
            format!("Expected {:?}, found {:?}", expected[0], found)
        } else {
            format!("Expected one of {:?}, found {:?}", expected, found)
        };

        Self::new(
            ParseErrorKind::UnexpectedToken { expected, found },
            span,
            message,
        )
    }

    /// Create an "unexpected EOF" error
    pub fn unexpected_eof(expected: Vec<TokenKind>, span: Span) -> Self {
        let message = if expected.len() == 1 {
            format!("Expected {:?}, found end of file", expected[0])
        } else {
            format!("Expected one of {:?}, found end of file", expected)
        };

        Self::new(
            ParseErrorKind::UnexpectedEof { expected },
            span,
            message,
        )
    }

    /// Create an "invalid syntax" error
    pub fn invalid_syntax(construct: String, details: String, span: Span) -> Self {
        let message = format!("Invalid syntax in {}: {}", construct, details);

        Self::new(
            ParseErrorKind::InvalidSyntax { construct, details },
            span,
            message,
        )
    }

    /// Create a semantic error
    pub fn semantic_error(constraint: String, details: String, span: Span) -> Self {
        let message = format!("Semantic error: {} ({})", constraint, details);

        Self::new(
            ParseErrorKind::SemanticError { constraint, details },
            span,
            message,
        )
    }

    /// Create an "expected token" error (convenience for single token)
    pub fn expected_token(expected: TokenKind, span: Span) -> Self {
        Self::unexpected_token(vec![expected], TokenKind::Eof, span)
    }

    /// Create an "expected literal" error
    pub fn expected_literal(span: Span) -> Self {
        Self::invalid_syntax(
            "literal".to_string(),
            "Expected a literal value (number, string, boolean, or null)".to_string(),
            span,
        )
    }

    /// Create an "expected identifier" error
    pub fn expected_identifier(span: Span) -> Self {
        Self::invalid_syntax(
            "identifier".to_string(),
            "Expected an identifier".to_string(),
            span,
        )
    }

    /// Create an "expected object key" error
    pub fn expected_object_key(span: Span) -> Self {
        Self::invalid_syntax(
            "object_key".to_string(),
            "Expected an object key (identifier, string, or computed expression)".to_string(),
            span,
        )
    }
} 