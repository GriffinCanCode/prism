//! Token Stream Navigation
//!
//! This module embodies the single concept of "Token Stream Navigation".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: managing navigation through a stream of tokens.
//!
//! **Conceptual Responsibility**: Navigate and query token streams
//! **What it does**: advance, peek, consume, check tokens
//! **What it doesn't do**: parsing logic, error recovery, semantic analysis

use crate::core::error::{ErrorContext, ParseError, ParseResult};
use prism_common::span::Span;
use prism_lexer::{Token, TokenKind};

/// Token stream navigator with position tracking
/// 
/// This struct embodies the single concept of navigating through tokens.
/// It maintains the current position and provides methods to move through
/// the token stream safely and efficiently.
#[derive(Debug)]
pub struct TokenStreamManager {
    /// The token stream to navigate
    tokens: Vec<Token>,
    /// Current position in the stream
    current: usize,
}

impl TokenStreamManager {
    /// Create a new token stream manager
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
        }
    }

    /// Check if we're at the end of the token stream
    pub fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || self.peek().kind == TokenKind::Eof
    }

    /// Get the current token without consuming it
    pub fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token {
            kind: TokenKind::Eof,
            span: Span::dummy(),
            semantic_context: None,
        })
    }

    /// Get the previous token
    pub fn previous(&self) -> &Token {
        if self.current == 0 {
            &Token {
                kind: TokenKind::Eof,
                span: Span::dummy(),
                semantic_context: None,
            }
        } else {
            &self.tokens[self.current - 1]
        }
    }

    /// Advance to the next token and return the current one
    pub fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    /// Check if the current token matches the given kind
    pub fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(&kind)
        }
    }

    /// Consume a token if it matches the expected kind
    pub fn consume(&mut self, expected: TokenKind) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    /// Expect a specific token kind and consume it
    pub fn expect(&mut self, expected: TokenKind) -> ParseResult<&Token> {
        if self.check(expected.clone()) {
            Ok(self.advance())
        } else {
            let found = self.peek().kind.clone();
            let span = self.peek().span;
            Err(ParseError::unexpected_token(
                vec![expected],
                found,
                span,
            ).with_context(ErrorContext::from_tokens(&self.tokens, self.current, 3)))
        }
    }
    
    /// Check if current token is an identifier and return true
    pub fn check_identifier(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
    }
    
    /// Expect an identifier token and return its value
    pub fn expect_identifier(&mut self) -> ParseResult<String> {
        match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => {
                let found = self.peek().kind.clone();
                let span = self.peek().span;
                Err(ParseError::unexpected_token(
                    vec![TokenKind::Identifier("identifier".to_string())],
                    found,
                    span,
                ).with_context(ErrorContext::from_tokens(&self.tokens, self.current, 3)))
            }
        }
    }
    
    /// Get the span of the previous token
    pub fn previous_span(&self) -> Span {
        self.previous().span
    }

    /// Get the current span for error reporting
    pub fn current_span(&self) -> Span {
        self.peek().span
    }

    /// Get current position in the stream
    pub fn current_position(&self) -> usize {
        self.current
    }

    /// Get all tokens (for error context)
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Insert a token at the current position (for error recovery)
    pub fn insert_token(&mut self, token: Token) {
        self.tokens.insert(self.current, token);
    }

    /// Set position (for error recovery)
    pub fn set_position(&mut self, position: usize) {
        self.current = position.min(self.tokens.len());
    }

    /// Check if current token could end a block
    pub fn check_block_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::RightBrace | TokenKind::Eof)
    }

    /// Check if current token could end a list
    pub fn check_list_end(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::RightParen
                | TokenKind::RightBracket
                | TokenKind::RightBrace
                | TokenKind::Greater
                | TokenKind::Eof
        )
    }

    /// Synchronize to next statement boundary (for error recovery)
    pub fn synchronize_to_statement(&mut self) {
        self.advance(); // Skip the problematic token

        while !self.is_at_end() {
            // Stop at statement boundaries
            if self.previous().kind == TokenKind::Semicolon {
                return;
            }

            // Stop at keywords that start new statements
            match self.peek().kind {
                TokenKind::Module
                | TokenKind::Function
                | TokenKind::Type
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return => {
                    return;
                }
                _ => {}
            }

            self.advance();
        }
    }
    /// Get the current token kind
    pub fn current_kind(&self) -> &TokenKind {
        &self.peek().kind
    }
    
    /// Get the current token
    pub fn current_token(&self) -> &Token {
        self.peek()
    }

    /// Check if current token is an identifier
    pub fn check_identifier(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
    }

    /// Expect an identifier token and return its value
    pub fn expect_identifier(&mut self) -> ParseResult<String> {
        match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let result = name.clone();
                self.advance();
                Ok(result)
            }
            _ => {
                Err(ParseError::unexpected_token(
                    self.peek().clone(),
                    "identifier".to_string(),
                ))
            }
        }
    }

    /// Get current position for lookahead/backtracking
    pub fn position(&self) -> usize {
        self.current
    }

    /// Get the current token (alias for peek, used by parsers)
    pub fn current_token(&self) -> &Token {
        self.peek()
    }

    /// Expect a specific token kind and consume it
    pub fn expect(&mut self, expected: TokenKind) -> ParseResult<&Token> {
        if self.check(expected.clone()) {
            Ok(self.advance())
        } else {
            Err(ParseError::unexpected_token(
                self.peek().clone(),
                format!("{:?}", expected),
            ))
        }
    }

    /// Get the previous token (for span calculation)
    pub fn previous(&self) -> &Token {
        if self.current > 0 {
            &self.tokens[self.current - 1]
        } else {
            self.peek()
        }
    }

    /// Get the span of the previous token
    pub fn previous_span(&self) -> Span {
        self.previous().span
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::{span::Position, SourceId};

    fn create_test_token(kind: TokenKind) -> Token {
        Token {
            kind,
            span: Span::new(
                Position::new(1, 1, 0),
                Position::new(1, 2, 1),
                SourceId::new(1),
            ),
            semantic_context: None,
        }
    }

    #[test]
    fn test_token_navigation() {
        let tokens = vec![
            create_test_token(TokenKind::Let),
            create_test_token(TokenKind::Identifier("x".to_string())),
            create_test_token(TokenKind::Assign),
            create_test_token(TokenKind::IntegerLiteral(42)),
        ];

        let mut manager = TokenStreamManager::new(tokens);

        // Test initial state
        assert!(!manager.is_at_end());
        assert_eq!(manager.peek().kind, TokenKind::Let);
        assert_eq!(manager.current_position(), 0);

        // Test advance
        manager.advance();
        assert!(matches!(manager.peek().kind, TokenKind::Identifier(_)));
        assert_eq!(manager.current_position(), 1);

        // Test consume
        let result = manager.consume(TokenKind::Identifier("x".to_string()), "Expected identifier");
        assert!(result.is_ok());
        assert_eq!(manager.current_position(), 2);
    }

    #[test]
    fn test_error_cases() {
        let tokens = vec![create_test_token(TokenKind::Let)];
        let mut manager = TokenStreamManager::new(tokens);

        // Test consume with wrong token
        let result = manager.consume(TokenKind::Function, "Expected function");
        assert!(result.is_err());

        // Test end of stream
        manager.advance();
        assert!(manager.is_at_end());
    }
} 