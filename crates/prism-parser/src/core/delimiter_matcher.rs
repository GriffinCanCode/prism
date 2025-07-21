//! Multi-Syntax Delimiter Matching and Recovery
//!
//! This module provides intelligent delimiter matching across different syntax styles,
//! maintaining conceptual cohesion around "delimiter matching and recovery coordination".
//! It integrates with existing error recovery systems without duplicating functionality.

use crate::core::{ParseError, ParseErrorKind, ParseResult};
use prism_common::span::{Position, Span};
use prism_lexer::{Token, TokenKind};
use prism_syntax::detection::SyntaxStyle;
use std::collections::HashMap;

/// Multi-syntax delimiter matcher that coordinates with existing recovery systems
#[derive(Debug)]
pub struct DelimiterMatcher {
    /// Stack of open delimiters with context
    delimiter_stack: Vec<DelimiterContext>,
    /// Current syntax style for context-aware matching
    syntax_style: SyntaxStyle,
    /// Delimiter mapping for different syntax styles
    style_mappings: HashMap<SyntaxStyle, DelimiterMapping>,
}

/// Context for an open delimiter
#[derive(Debug, Clone)]
pub struct DelimiterContext {
    /// Type of delimiter
    pub delimiter_type: DelimiterType,
    /// Position where delimiter was opened
    pub opening_position: Position,
    /// Expected closing delimiter
    pub expected_closing: TokenKind,
    /// Syntax style context
    pub syntax_context: SyntaxContext,
    /// Nesting level
    pub nesting_level: usize,
}

/// Types of delimiters across syntax styles
#[derive(Debug, Clone, PartialEq)]
pub enum DelimiterType {
    /// Braces { } - C-like syntax
    Brace,
    /// Parentheses ( ) - Universal
    Paren,
    /// Brackets [ ] - Universal
    Bracket,
    /// Indentation - Python-like syntax
    Indent,
    /// Semantic block boundaries - Prism canonical
    SemanticBlock,
    /// Custom delimiter for specific contexts
    Custom(String),
}

/// Syntax context for delimiter matching
#[derive(Debug, Clone)]
pub struct SyntaxContext {
    /// Current parsing context (function, type, etc.)
    pub context_type: ContextType,
    /// Whether we're in a mixed-syntax region
    pub mixed_syntax: bool,
    /// Expected delimiter style for this context
    pub expected_style: SyntaxStyle,
}

/// Type of parsing context
#[derive(Debug, Clone, PartialEq)]
pub enum ContextType {
    Module,
    Function,
    Type,
    Expression,
    Statement,
    Parameter,
    Argument,
}

/// Delimiter mapping for different syntax styles
#[derive(Debug, Clone)]
pub struct DelimiterMapping {
    /// Opening delimiters and their expected closing counterparts
    pub pairs: HashMap<TokenKind, TokenKind>,
    /// Whether this style supports mixed delimiters
    pub supports_mixed: bool,
    /// Priority for delimiter resolution conflicts
    pub priority: u8,
}

/// Result of delimiter matching operation
#[derive(Debug)]
pub struct DelimiterMatchResult {
    /// Whether the match was successful
    pub success: bool,
    /// Matched delimiter context
    pub context: Option<DelimiterContext>,
    /// Recovery suggestions if matching failed
    pub recovery_suggestions: Vec<DelimiterRecoverySuggestion>,
}

/// Suggestion for delimiter recovery
#[derive(Debug, Clone)]
pub struct DelimiterRecoverySuggestion {
    /// Type of recovery suggested
    pub suggestion_type: DelimiterRecoveryType,
    /// Position where recovery should be applied
    pub position: Position,
    /// Suggested token to insert/replace
    pub suggested_token: TokenKind,
    /// Confidence in this suggestion (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable explanation
    pub explanation: String,
}

/// Types of delimiter recovery
#[derive(Debug, Clone)]
pub enum DelimiterRecoveryType {
    /// Insert missing closing delimiter
    InsertClosing,
    /// Insert missing opening delimiter
    InsertOpening,
    /// Replace incorrect delimiter
    ReplaceDelimiter,
    /// Fix indentation level
    FixIndentation,
    /// Convert between syntax styles
    ConvertSyntaxStyle,
}

impl DelimiterMatcher {
    /// Create a new delimiter matcher for the given syntax style
    pub fn new(syntax_style: SyntaxStyle) -> Self {
        let mut matcher = Self {
            delimiter_stack: Vec::new(),
            syntax_style,
            style_mappings: HashMap::new(),
        };
        
        matcher.initialize_style_mappings();
        matcher
    }
    
    /// Initialize delimiter mappings for different syntax styles
    fn initialize_style_mappings(&mut self) {
        // C-like syntax (braces, semicolons)
        let c_like_mapping = DelimiterMapping {
            pairs: [
                (TokenKind::LeftBrace, TokenKind::RightBrace),
                (TokenKind::LeftParen, TokenKind::RightParen),
                (TokenKind::LeftBracket, TokenKind::RightBracket),
            ].into_iter().collect(),
            supports_mixed: false,
            priority: 10,
        };
        self.style_mappings.insert(SyntaxStyle::CLike, c_like_mapping);
        
        // Python-like syntax (indentation)
        let python_like_mapping = DelimiterMapping {
            pairs: [
                (TokenKind::LeftParen, TokenKind::RightParen),
                (TokenKind::LeftBracket, TokenKind::RightBracket),
                // Note: Indentation handled specially
            ].into_iter().collect(),
            supports_mixed: true,
            priority: 8,
        };
        self.style_mappings.insert(SyntaxStyle::PythonLike, python_like_mapping);
        
        // Rust-like syntax
        let rust_like_mapping = DelimiterMapping {
            pairs: [
                (TokenKind::LeftBrace, TokenKind::RightBrace),
                (TokenKind::LeftParen, TokenKind::RightParen),
                (TokenKind::LeftBracket, TokenKind::RightBracket),
            ].into_iter().collect(),
            supports_mixed: false,
            priority: 9,
        };
        self.style_mappings.insert(SyntaxStyle::RustLike, rust_like_mapping);
        
        // Canonical Prism syntax
        let canonical_mapping = DelimiterMapping {
            pairs: [
                (TokenKind::LeftBrace, TokenKind::RightBrace),
                (TokenKind::LeftParen, TokenKind::RightParen),
                (TokenKind::LeftBracket, TokenKind::RightBracket),
            ].into_iter().collect(),
            supports_mixed: true,
            priority: 12,
        };
        self.style_mappings.insert(SyntaxStyle::Canonical, canonical_mapping);
    }
    
    /// Open a delimiter and track its context
    pub fn open_delimiter(
        &mut self,
        token: &Token,
        context_type: ContextType,
    ) -> ParseResult<()> {
        let delimiter_type = self.token_to_delimiter_type(&token.kind)?;
        let expected_closing = self.get_expected_closing(&token.kind)?;
        
        let context = DelimiterContext {
            delimiter_type,
            opening_position: token.span.start,
            expected_closing,
            syntax_context: SyntaxContext {
                context_type,
                mixed_syntax: self.is_mixed_syntax_context(),
                expected_style: self.syntax_style,
            },
            nesting_level: self.delimiter_stack.len(),
        };
        
        self.delimiter_stack.push(context);
        Ok(())
    }
    
    /// Close a delimiter with intelligent matching and recovery
    pub fn close_delimiter(&mut self, token: &Token) -> ParseResult<DelimiterMatchResult> {
        if let Some(expected_context) = self.delimiter_stack.pop() {
            if token.kind == expected_context.expected_closing {
                // Perfect match
                Ok(DelimiterMatchResult {
                    success: true,
                    context: Some(expected_context),
                    recovery_suggestions: Vec::new(),
                })
            } else {
                // Mismatch - generate recovery suggestions
                let suggestions = self.generate_recovery_suggestions(
                    &expected_context,
                    &token.kind,
                    token.span.start,
                );
                
                Ok(DelimiterMatchResult {
                    success: false,
                    context: Some(expected_context),
                    recovery_suggestions: suggestions,
                })
            }
        } else {
            // No open delimiter - suggest insertion
            let suggestions = vec![DelimiterRecoverySuggestion {
                suggestion_type: DelimiterRecoveryType::InsertOpening,
                position: token.span.start,
                suggested_token: self.get_matching_opener(&token.kind)
                    .unwrap_or(TokenKind::LeftBrace),
                confidence: 0.7,
                explanation: "Insert matching opening delimiter".to_string(),
            }];
            
            Ok(DelimiterMatchResult {
                success: false,
                context: None,
                recovery_suggestions: suggestions,
            })
        }
    }
    
    /// Generate recovery suggestions for delimiter mismatches
    fn generate_recovery_suggestions(
        &self,
        expected_context: &DelimiterContext,
        found_token: &TokenKind,
        position: Position,
    ) -> Vec<DelimiterRecoverySuggestion> {
        let mut suggestions = Vec::new();
        
        // Check for cross-syntax confusion
        if self.is_cross_syntax_confusion(&expected_context.expected_closing, found_token) {
            suggestions.push(DelimiterRecoverySuggestion {
                suggestion_type: DelimiterRecoveryType::ConvertSyntaxStyle,
                position,
                suggested_token: expected_context.expected_closing,
                confidence: 0.9,
                explanation: format!(
                    "Convert from {:?} to {:?} syntax style",
                    self.detect_token_syntax_style(found_token),
                    self.syntax_style
                ),
            });
        }
        
        // Suggest replacing with expected delimiter
        suggestions.push(DelimiterRecoverySuggestion {
            suggestion_type: DelimiterRecoveryType::ReplaceDelimiter,
            position,
            suggested_token: expected_context.expected_closing,
            confidence: 0.8,
            explanation: format!(
                "Replace {:?} with expected {:?}",
                found_token, expected_context.expected_closing
            ),
        });
        
        // Check if we need to handle indentation specially
        if matches!(expected_context.delimiter_type, DelimiterType::Indent) {
            suggestions.push(DelimiterRecoverySuggestion {
                suggestion_type: DelimiterRecoveryType::FixIndentation,
                position,
                suggested_token: TokenKind::Dedent,
                confidence: 0.85,
                explanation: "Fix indentation level".to_string(),
            });
        }
        
        suggestions
    }
    
    /// Convert token kind to delimiter type
    fn token_to_delimiter_type(&self, token_kind: &TokenKind) -> ParseResult<DelimiterType> {
        match token_kind {
            TokenKind::LeftBrace => Ok(DelimiterType::Brace),
            TokenKind::LeftParen => Ok(DelimiterType::Paren),
            TokenKind::LeftBracket => Ok(DelimiterType::Bracket),
            TokenKind::Indent => Ok(DelimiterType::Indent),
            _ => Err(ParseError {
                kind: ParseErrorKind::InvalidDelimiter {
                    found: token_kind.clone(),
                },
                span: Span::default(), // Will be filled by caller
                message: format!("Invalid delimiter: {:?}", token_kind),
                suggestions: Vec::new(),
                severity: crate::core::error::ErrorSeverity::Error,
                context: None,
            }),
        }
    }
    
    /// Get expected closing delimiter for opening delimiter
    fn get_expected_closing(&self, opening: &TokenKind) -> ParseResult<TokenKind> {
        let mapping = self.style_mappings.get(&self.syntax_style)
            .ok_or_else(|| ParseError {
                kind: ParseErrorKind::UnsupportedSyntaxStyle {
                    style: self.syntax_style,
                },
                span: Span::default(),
                message: format!("Unsupported syntax style: {:?}", self.syntax_style),
                suggestions: Vec::new(),
                severity: crate::core::error::ErrorSeverity::Error,
                context: None,
            })?;
        
        mapping.pairs.get(opening).copied()
            .ok_or_else(|| ParseError {
                kind: ParseErrorKind::InvalidDelimiter {
                    found: opening.clone(),
                },
                span: Span::default(),
                message: format!("Invalid delimiter for style {:?}: {:?}", self.syntax_style, opening),
                suggestions: Vec::new(),
                severity: crate::core::error::ErrorSeverity::Error,
                context: None,
            })
    }
    
    /// Get matching opening delimiter for closing delimiter
    fn get_matching_opener(&self, closing: &TokenKind) -> Option<TokenKind> {
        let mapping = self.style_mappings.get(&self.syntax_style)?;
        
        for (opener, closer) in &mapping.pairs {
            if closer == closing {
                return Some(opener.clone());
            }
        }
        
        None
    }
    
    /// Check if we're in a mixed syntax context
    fn is_mixed_syntax_context(&self) -> bool {
        // For now, assume no mixed syntax
        // This could be enhanced to detect mixed syntax based on context
        false
    }
    
    /// Check if error is due to cross-syntax confusion
    fn is_cross_syntax_confusion(&self, expected: &TokenKind, found: &TokenKind) -> bool {
        match (self.syntax_style, expected, found) {
            // Python-like expecting dedent but found brace
            (SyntaxStyle::PythonLike, TokenKind::Dedent, TokenKind::RightBrace) => true,
            // C-like expecting brace but found dedent
            (SyntaxStyle::CLike, TokenKind::RightBrace, TokenKind::Dedent) => true,
            _ => false,
        }
    }
    
    /// Detect syntax style of a token
    fn detect_token_syntax_style(&self, token: &TokenKind) -> SyntaxStyle {
        match token {
            TokenKind::RightBrace | TokenKind::LeftBrace => SyntaxStyle::CLike,
            TokenKind::Newline => SyntaxStyle::PythonLike, // Use newline for Python-like indentation
            _ => SyntaxStyle::Canonical,
        }
    }
    
    /// Get current nesting level
    pub fn nesting_level(&self) -> usize {
        self.delimiter_stack.len()
    }
    
    /// Check if all delimiters are properly closed
    pub fn is_balanced(&self) -> bool {
        self.delimiter_stack.is_empty()
    }
    
    /// Get unclosed delimiters for error reporting
    pub fn unclosed_delimiters(&self) -> &[DelimiterContext] {
        &self.delimiter_stack
    }
}

impl Default for DelimiterMatcher {
    fn default() -> Self {
        Self::new(SyntaxStyle::Canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::span::Position;
    
    fn create_test_token(kind: TokenKind, line: u32, col: u32) -> Token {
        Token {
            kind,
            span: Span {
                start: Position::new(line, col, 0),
                end: Position::new(line, col + 1, 1),
            },
        }
    }
    
    #[test]
    fn test_c_like_delimiter_matching() {
        let mut matcher = DelimiterMatcher::new(SyntaxStyle::CLike);
        
        // Test opening brace
        let open_brace = create_test_token(TokenKind::LeftBrace, 1, 1);
        assert!(matcher.open_delimiter(&open_brace, ContextType::Function).is_ok());
        assert_eq!(matcher.nesting_level(), 1);
        assert!(!matcher.is_balanced());
        
        // Test matching closing brace
        let close_brace = create_test_token(TokenKind::RightBrace, 1, 10);
        let result = matcher.close_delimiter(&close_brace).unwrap();
        assert!(result.success);
        assert!(result.recovery_suggestions.is_empty());
        assert!(matcher.is_balanced());
    }
    
    #[test]
    fn test_delimiter_mismatch_recovery() {
        let mut matcher = DelimiterMatcher::new(SyntaxStyle::CLike);
        
        // Open with brace
        let open_brace = create_test_token(TokenKind::LeftBrace, 1, 1);
        assert!(matcher.open_delimiter(&open_brace, ContextType::Function).is_ok());
        
        // Try to close with paren (mismatch)
        let close_paren = create_test_token(TokenKind::RightParen, 1, 10);
        let result = matcher.close_delimiter(&close_paren).unwrap();
        
        assert!(!result.success);
        assert!(!result.recovery_suggestions.is_empty());
        
        // Should suggest replacing with correct delimiter
        let suggestion = &result.recovery_suggestions[0];
        assert_eq!(suggestion.suggested_token, TokenKind::RightBrace);
        assert!(matches!(suggestion.suggestion_type, DelimiterRecoveryType::ReplaceDelimiter));
    }
    
    #[test]
    fn test_cross_syntax_confusion() {
        let mut matcher = DelimiterMatcher::new(SyntaxStyle::PythonLike);
        
        // In Python-like syntax, we might expect indentation but get braces
        let close_brace = create_test_token(TokenKind::RightBrace, 1, 1);
        let result = matcher.close_delimiter(&close_brace).unwrap();
        
        assert!(!result.success);
        assert!(!result.recovery_suggestions.is_empty());
        
        // Should suggest opening delimiter insertion
        let suggestion = &result.recovery_suggestions[0];
        assert!(matches!(suggestion.suggestion_type, DelimiterRecoveryType::InsertOpening));
    }
    
    #[test]
    fn test_nested_delimiters() {
        let mut matcher = DelimiterMatcher::new(SyntaxStyle::CLike);
        
        // Open outer brace
        let outer_brace = create_test_token(TokenKind::LeftBrace, 1, 1);
        assert!(matcher.open_delimiter(&outer_brace, ContextType::Function).is_ok());
        
        // Open inner paren
        let inner_paren = create_test_token(TokenKind::LeftParen, 1, 5);
        assert!(matcher.open_delimiter(&inner_paren, ContextType::Expression).is_ok());
        
        assert_eq!(matcher.nesting_level(), 2);
        
        // Close inner paren
        let close_paren = create_test_token(TokenKind::RightParen, 1, 10);
        let result = matcher.close_delimiter(&close_paren).unwrap();
        assert!(result.success);
        assert_eq!(matcher.nesting_level(), 1);
        
        // Close outer brace
        let close_brace = create_test_token(TokenKind::RightBrace, 1, 15);
        let result = matcher.close_delimiter(&close_brace).unwrap();
        assert!(result.success);
        assert!(matcher.is_balanced());
    }
    
    #[test]
    fn test_syntax_style_detection() {
        let matcher = DelimiterMatcher::new(SyntaxStyle::CLike);
        
        assert_eq!(matcher.detect_token_syntax_style(&TokenKind::LeftBrace), SyntaxStyle::CLike);
        assert_eq!(matcher.detect_token_syntax_style(&TokenKind::Indent), SyntaxStyle::PythonLike);
        assert_eq!(matcher.detect_token_syntax_style(&TokenKind::LeftParen), SyntaxStyle::Canonical);
    }
    
    #[test]
    fn test_unclosed_delimiters_reporting() {
        let mut matcher = DelimiterMatcher::new(SyntaxStyle::CLike);
        
        // Open some delimiters but don't close them
        let brace = create_test_token(TokenKind::LeftBrace, 1, 1);
        let paren = create_test_token(TokenKind::LeftParen, 1, 5);
        
        assert!(matcher.open_delimiter(&brace, ContextType::Function).is_ok());
        assert!(matcher.open_delimiter(&paren, ContextType::Expression).is_ok());
        
        let unclosed = matcher.unclosed_delimiters();
        assert_eq!(unclosed.len(), 2);
        assert_eq!(unclosed[0].delimiter_type, DelimiterType::Brace);
        assert_eq!(unclosed[1].delimiter_type, DelimiterType::Paren);
    }
} 