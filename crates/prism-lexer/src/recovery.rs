//! Error recovery and diagnostics for the lexer
//!
//! This module provides robust error recovery strategies and comprehensive
//! diagnostic reporting for the Prism lexer.

use crate::token::{Token, TokenKind};
use prism_common::{
    diagnostics::{Diagnostic, DiagnosticBag, Severity},
    span::{Position, Span},
    SourceId,
};
use std::collections::VecDeque;

/// Syntax styles for error recovery context
/// 
/// This is a simplified version used only for error recovery.
/// The full SyntaxStyle definition is in prism-syntax crate.
#[derive(Debug, Clone, PartialEq)]
pub enum SyntaxStyle {
    /// C-like syntax with braces and semicolons
    CLike,
    /// Python-like syntax with indentation
    PythonLike,
    /// Rust-like syntax
    RustLike,
    /// Canonical Prism syntax
    Canonical,
    /// Mixed syntax styles
    Mixed,
}

/// Error recovery strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Skip the invalid character and continue
    Skip,
    /// Insert a missing token
    Insert(TokenKind),
    /// Replace the invalid token with a valid one
    Replace(TokenKind),
    /// Synchronize to the next valid token
    Synchronize,
    /// Custom recovery with a specific action
    Custom(RecoveryAction),
}

/// Custom recovery actions
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Continue lexing normally
    Continue,
    /// Retry the current position
    Retry,
    /// Stop lexing due to unrecoverable error
    Abort,
}

/// Error recovery manager
pub struct ErrorRecovery {
    /// Recovery strategies by error type
    strategies: Vec<RecoveryRule>,
    /// Maximum number of recovery attempts
    max_attempts: usize,
    /// Current recovery state
    recovery_state: RecoveryState,
    /// Diagnostics collector
    diagnostics: DiagnosticBag,
}

/// Recovery rule mapping error patterns to strategies
#[derive(Debug, Clone)]
pub struct RecoveryRule {
    /// Pattern that triggers this rule
    pub pattern: ErrorPattern,
    /// Strategy to apply
    pub strategy: RecoveryStrategy,
    /// Priority of this rule (higher = more important)
    pub priority: u32,
    /// Description of the recovery
    pub description: String,
}

/// Error patterns that can trigger recovery
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorPattern {
    /// Invalid character at position
    InvalidCharacter(char),
    /// Unterminated string literal
    UnterminatedString,
    /// Invalid number format
    InvalidNumber,
    /// Invalid escape sequence
    InvalidEscape,
    /// Unexpected end of file
    UnexpectedEof,
    /// Missing delimiter
    MissingDelimiter(char),
    /// Unmatched delimiter
    UnmatchedDelimiter(char),
    /// Invalid regex literal
    InvalidRegex,
    /// Invalid money literal
    InvalidMoney,
    /// Invalid duration literal
    InvalidDuration,
    /// Indentation error
    IndentationError,
    /// Any error (catch-all)
    Any,
}

/// Current recovery state
#[derive(Debug, Clone)]
pub struct RecoveryState {
    /// Number of recovery attempts made
    pub attempts: usize,
    /// Whether we're currently in recovery mode
    pub in_recovery: bool,
    /// Last successful token position
    pub last_good_position: Position,
    /// Tokens generated during recovery
    pub recovery_tokens: VecDeque<Token>,
}

impl ErrorRecovery {
    /// Create a new error recovery manager
    pub fn new() -> Self {
        let mut recovery = Self {
            strategies: Vec::new(),
            max_attempts: 10,
            recovery_state: RecoveryState {
                attempts: 0,
                in_recovery: false,
                last_good_position: Position::new(1, 1, 0),
                recovery_tokens: VecDeque::new(),
            },
            diagnostics: DiagnosticBag::new(),
        };
        
        recovery.initialize_default_strategies();
        recovery
    }

    /// Initialize default recovery strategies
    fn initialize_default_strategies(&mut self) {
        // Skip invalid characters
        self.strategies.push(RecoveryRule {
            pattern: ErrorPattern::InvalidCharacter(' '),
            strategy: RecoveryStrategy::Skip,
            priority: 10,
            description: "Skip invalid character".to_string(),
        });
        
        // Try to close unterminated strings
        self.strategies.push(RecoveryRule {
            pattern: ErrorPattern::UnterminatedString,
            strategy: RecoveryStrategy::Insert(TokenKind::StringLiteral("".to_string())),
            priority: 20,
            description: "Insert empty string to close unterminated literal".to_string(),
        });
        
        // Try to recover from invalid numbers
        self.strategies.push(RecoveryRule {
            pattern: ErrorPattern::InvalidNumber,
            strategy: RecoveryStrategy::Replace(TokenKind::IntegerLiteral(0)),
            priority: 15,
            description: "Replace invalid number with 0".to_string(),
        });
        
        // Synchronize on structural tokens
        self.strategies.push(RecoveryRule {
            pattern: ErrorPattern::Any,
            strategy: RecoveryStrategy::Synchronize,
            priority: 1,
            description: "Synchronize to next structural token".to_string(),
        });
    }

    /// Attempt to recover from an error
    pub fn recover_from_error(
        &mut self,
        error_pattern: ErrorPattern,
        source_id: SourceId,
        position: Position,
        syntax_style: SyntaxStyle,
    ) -> Option<RecoveryResult> {
        if self.recovery_state.attempts >= self.max_attempts {
            return None;
        }
        
        self.recovery_state.attempts += 1;
        self.recovery_state.in_recovery = true;
        
        // Find the best recovery strategy
        let strategy = self.find_best_strategy(&error_pattern);
        
        // Apply the strategy
        let result = self.apply_strategy(&strategy, source_id, position, syntax_style);
        
        // Record the recovery attempt
        self.record_recovery_attempt(&error_pattern, &strategy, &result);
        
        result
    }

    /// Find the best recovery strategy for an error pattern
    fn find_best_strategy(&self, pattern: &ErrorPattern) -> RecoveryStrategy {
        let mut best_rule: Option<&RecoveryRule> = None;
        let mut best_priority = 0;
        
        for rule in &self.strategies {
            if self.pattern_matches(&rule.pattern, pattern) && rule.priority > best_priority {
                best_rule = Some(rule);
                best_priority = rule.priority;
            }
        }
        
        best_rule.map(|rule| rule.strategy.clone())
            .unwrap_or(RecoveryStrategy::Skip)
    }

    /// Check if an error pattern matches a rule pattern
    fn pattern_matches(&self, rule_pattern: &ErrorPattern, error_pattern: &ErrorPattern) -> bool {
        match (rule_pattern, error_pattern) {
            (ErrorPattern::Any, _) => true,
            (ErrorPattern::InvalidCharacter(_), ErrorPattern::InvalidCharacter(_)) => true,
            (ErrorPattern::UnterminatedString, ErrorPattern::UnterminatedString) => true,
            (ErrorPattern::InvalidNumber, ErrorPattern::InvalidNumber) => true,
            (ErrorPattern::InvalidEscape, ErrorPattern::InvalidEscape) => true,
            (ErrorPattern::UnexpectedEof, ErrorPattern::UnexpectedEof) => true,
            (ErrorPattern::MissingDelimiter(a), ErrorPattern::MissingDelimiter(b)) => a == b,
            (ErrorPattern::UnmatchedDelimiter(a), ErrorPattern::UnmatchedDelimiter(b)) => a == b,
            _ => false,
        }
    }

    /// Apply a recovery strategy
    fn apply_strategy(
        &mut self,
        strategy: &RecoveryStrategy,
        source_id: SourceId,
        position: Position,
        _syntax_style: SyntaxStyle,
    ) -> Option<RecoveryResult> {
        match strategy {
            RecoveryStrategy::Skip => {
                Some(RecoveryResult {
                    action: RecoveryAction::Continue,
                    tokens: Vec::new(),
                    message: "Skipped invalid character".to_string(),
                })
            }
            RecoveryStrategy::Insert(token_kind) => {
                let token = Token::new(
                    token_kind.clone(),
                    Span::new(position, position, source_id),
                );
                
                Some(RecoveryResult {
                    action: RecoveryAction::Continue,
                    tokens: vec![token],
                    message: format!("Inserted missing token: {:?}", token_kind),
                })
            }
            RecoveryStrategy::Replace(token_kind) => {
                let token = Token::new(
                    token_kind.clone(),
                    Span::new(position, position, source_id),
                );
                
                Some(RecoveryResult {
                    action: RecoveryAction::Continue,
                    tokens: vec![token],
                    message: format!("Replaced invalid token with: {:?}", token_kind),
                })
            }
            RecoveryStrategy::Synchronize => {
                // Try to synchronize to the next structural token
                Some(RecoveryResult {
                    action: RecoveryAction::Continue,
                    tokens: Vec::new(),
                    message: "Synchronized to next structural token".to_string(),
                })
            }
            RecoveryStrategy::Custom(action) => {
                Some(RecoveryResult {
                    action: action.clone(),
                    tokens: Vec::new(),
                    message: "Applied custom recovery action".to_string(),
                })
            }
        }
    }

    /// Record a recovery attempt for analysis
    fn record_recovery_attempt(
        &mut self,
        pattern: &ErrorPattern,
        strategy: &RecoveryStrategy,
        result: &Option<RecoveryResult>,
    ) {
        let message = match result {
            Some(result) => format!("Recovery successful: {}", result.message),
            None => "Recovery failed".to_string(),
        };
        
        // This could be expanded to collect statistics about recovery success rates
        println!("Recovery attempt: {:?} -> {:?} = {}", pattern, strategy, message);
    }

    /// Reset recovery state
    pub fn reset(&mut self) {
        self.recovery_state.attempts = 0;
        self.recovery_state.in_recovery = false;
        self.recovery_state.recovery_tokens.clear();
    }

    /// Check if we're currently in recovery mode
    pub fn in_recovery(&self) -> bool {
        self.recovery_state.in_recovery
    }

    /// Get the number of recovery attempts made
    pub fn recovery_attempts(&self) -> usize {
        self.recovery_state.attempts
    }

    /// Add a custom recovery rule
    pub fn add_recovery_rule(&mut self, rule: RecoveryRule) {
        self.strategies.push(rule);
        // Sort by priority (highest first)
        self.strategies.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
}

/// Result of a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Action to take after recovery
    pub action: RecoveryAction,
    /// Tokens generated during recovery
    pub tokens: Vec<Token>,
    /// Human-readable message about the recovery
    pub message: String,
}

/// Diagnostic helper for lexer errors
pub struct LexerDiagnostics {
    /// Diagnostics bag
    diagnostics: DiagnosticBag,
}

impl LexerDiagnostics {
    /// Create a new diagnostics helper
    pub fn new() -> Self {
        Self {
            diagnostics: DiagnosticBag::new(),
        }
    }

    /// Report an invalid character error
    pub fn invalid_character(&mut self, ch: char, span: Span) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Error,
            format!("Invalid character '{}'", ch),
            span,
        ).with_help("Remove this character or escape it if it's part of a string"));
    }

    /// Report an unterminated string error
    pub fn unterminated_string(&mut self, span: Span) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Error,
            "Unterminated string literal".to_string(),
            span,
        ).with_help("Add a closing quote to terminate the string"));
    }

    /// Report an invalid number error
    pub fn invalid_number(&mut self, span: Span) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Error,
            "Invalid number literal".to_string(),
            span,
        ).with_help("Check the number format - ensure it's a valid integer or float"));
    }

    /// Report an invalid escape sequence error
    pub fn invalid_escape(&mut self, span: Span) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Error,
            "Invalid escape sequence".to_string(),
            span,
        ).with_help("Use a valid escape sequence like \\n, \\t, \\r, \\\\, or \\\""));
    }

    /// Report a mixed syntax style warning
    pub fn mixed_syntax_style(&mut self, span: Span, styles: &[SyntaxStyle]) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Warning,
            format!("Mixed syntax styles detected: {:?}", styles),
            span,
        ).with_help("Consider using a consistent syntax style throughout the file"));
    }

    /// Report a naming convention warning
    pub fn naming_convention(&mut self, span: Span, identifier: &str, suggestion: &str) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Warning,
            format!("Identifier '{}' doesn't follow naming conventions", identifier),
            span,
        ).with_help(&format!("Consider using '{}' instead", suggestion)));
    }

    /// Report a semantic hint
    pub fn semantic_hint(&mut self, span: Span, hint: &str) {
        self.diagnostics.add(Diagnostic::new(
            Severity::Info,
            format!("Semantic hint: {}", hint),
            span,
        ));
    }

    /// Get all diagnostics
    pub fn into_diagnostics(self) -> DiagnosticBag {
        self.diagnostics
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.diagnostics.error_count()
    }
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LexerDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::SourceId;

    #[test]
    fn test_error_recovery_skip() {
        let mut recovery = ErrorRecovery::new();
        let result = recovery.recover_from_error(
            ErrorPattern::InvalidCharacter('$'),
            SourceId::new(1),
            Position::new(1, 1, 0),
            SyntaxStyle::Canonical,
        );
        
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.action, RecoveryAction::Continue);
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_error_recovery_insert() {
        let mut recovery = ErrorRecovery::new();
        let result = recovery.recover_from_error(
            ErrorPattern::UnterminatedString,
            SourceId::new(1),
            Position::new(1, 1, 0),
            SyntaxStyle::Canonical,
        );
        
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.action, RecoveryAction::Continue);
        assert_eq!(result.tokens.len(), 1);
        assert!(matches!(result.tokens[0].kind, TokenKind::StringLiteral(_)));
    }

    #[test]
    fn test_diagnostics() {
        let mut diag = LexerDiagnostics::new();
        let span = Span::new(Position::new(1, 1, 0), Position::new(1, 2, 1), SourceId::new(1));
        
        diag.invalid_character('$', span);
        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    #[test]
    fn test_pattern_matching() {
        let recovery = ErrorRecovery::new();
        
        assert!(recovery.pattern_matches(&ErrorPattern::Any, &ErrorPattern::InvalidCharacter('x')));
        assert!(recovery.pattern_matches(&ErrorPattern::InvalidCharacter('a'), &ErrorPattern::InvalidCharacter('b')));
        assert!(!recovery.pattern_matches(&ErrorPattern::UnterminatedString, &ErrorPattern::InvalidNumber));
    }

    #[test]
    fn test_recovery_attempts_limit() {
        let mut recovery = ErrorRecovery::new();
        recovery.max_attempts = 2;
        
        // First attempt should succeed
        let result1 = recovery.recover_from_error(
            ErrorPattern::InvalidCharacter('x'),
            SourceId::new(1),
            Position::new(1, 1, 0),
            SyntaxStyle::Canonical,
        );
        assert!(result1.is_some());
        
        // Second attempt should succeed
        let result2 = recovery.recover_from_error(
            ErrorPattern::InvalidCharacter('y'),
            SourceId::new(1),
            Position::new(1, 2, 1),
            SyntaxStyle::Canonical,
        );
        assert!(result2.is_some());
        
        // Third attempt should fail (exceeds limit)
        let result3 = recovery.recover_from_error(
            ErrorPattern::InvalidCharacter('z'),
            SourceId::new(1),
            Position::new(1, 3, 2),
            SyntaxStyle::Canonical,
        );
        assert!(result3.is_none());
    }
} 