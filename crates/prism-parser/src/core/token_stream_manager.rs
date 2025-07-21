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

/// Token stream manager for efficient navigation
pub struct TokenStreamManager {
    /// Vector of tokens
    tokens: Vec<Token>,
    /// Current position in the token stream
    current: usize,
    /// EOF token for safe returns
    eof_token: Token,
    /// Pre-allocated cache for token lookups
    _token_cache: std::collections::HashMap<usize, Token>,
}

impl TokenStreamManager {
    /// Create a new token stream manager with caching
    pub fn new(tokens: Vec<Token>) -> Self {
        let total_tokens = tokens.len();
        let eof_token = Token {
            kind: TokenKind::Eof,
            span: Span::dummy(),
        };
        
        Self {
            tokens,
            current: 0,
            eof_token,
            // Pre-allocate cache for better performance
            _token_cache: std::collections::HashMap::with_capacity(total_tokens / 10),
        }
    }

    /// Create a new token stream manager with custom capacity
    pub fn with_capacity(tokens: Vec<Token>, cache_capacity: usize) -> Self {
        let eof_token = Token {
            kind: TokenKind::Eof,
            span: Span::dummy(),
        };
        
        Self {
            tokens,
            current: 0,
            eof_token,
            _token_cache: std::collections::HashMap::with_capacity(cache_capacity),
        }
    }

    /// Get tokens slice for efficient access
    pub fn tokens_slice(&self) -> &[Token] {
        &self.tokens
    }

    /// Get remaining tokens count efficiently
    pub fn remaining_tokens(&self) -> usize {
        self.tokens.len().saturating_sub(self.current)
    }

    /// Bulk advance for performance
    pub fn bulk_advance(&mut self, count: usize) -> usize {
        let old_pos = self.current;
        self.current = (self.current + count).min(self.tokens.len());
        self.current - old_pos
    }

    /// Peek ahead multiple tokens efficiently
    pub fn peek_ahead(&self, offset: usize) -> &Token {
        let pos = self.current + offset;
        if pos < self.tokens.len() {
            &self.tokens[pos]
        } else {
            self.tokens.last().unwrap_or(&Token::eof())
        }
    }

    /// Check multiple token kinds at once
    pub fn check_any(&self, kinds: &[TokenKind]) -> bool {
        let current_kind = &self.peek().kind;
        kinds.iter().any(|kind| std::mem::discriminant(current_kind) == std::mem::discriminant(kind))
    }

    /// Check if we're at the end of the token stream
    pub fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || self.peek().kind == TokenKind::Eof
    }

    /// Get the current token without consuming it
    pub fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&self.eof_token)
    }

    /// Get the previous token
    pub fn previous(&self) -> &Token {
        if self.current == 0 {
            &self.eof_token
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
    pub fn insert_token(&mut self, position: usize, token: Token) {
        if position <= self.tokens.len() {
            self.tokens.insert(position, token);
            // Adjust current position if we inserted before it
            if position <= self.current {
                self.current += 1;
            }
        }
    }

    /// Replace the current token (for error recovery)
    pub fn replace_current_token(&mut self, new_token: Token) {
        if self.current < self.tokens.len() {
            self.tokens[self.current] = new_token;
        }
    }

    /// Optimized synchronization with early termination
    pub fn synchronize_to_statement(&mut self) {
        // Fast path: if already at statement boundary, return immediately
        if self.current > 0 && self.tokens[self.current - 1].kind == TokenKind::Semicolon {
            return;
        }

        // Use bulk operations for better performance
        let mut pos = self.current;
        while pos < self.tokens.len() {
            match self.tokens[pos].kind {
                TokenKind::Semicolon => {
                    self.current = pos + 1;
                    return;
                }
                TokenKind::Module
                | TokenKind::Section  
                | TokenKind::Function
                | TokenKind::Fn
                | TokenKind::Type
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Match
                | TokenKind::Return
                | TokenKind::Break
                | TokenKind::Continue => {
                    self.current = pos;
                    return;
                }
                _ => pos += 1,
            }
        }
        
        self.current = self.tokens.len();
    }

    /// Find the next expression boundary for error recovery
    pub fn find_expression_boundary(&self) -> bool {
        let mut position = self.current;
        let mut paren_depth = 0;
        let mut bracket_depth = 0;
        
        while position < self.tokens.len() {
            match self.tokens[position].kind {
                TokenKind::LeftParen => paren_depth += 1,
                TokenKind::RightParen => {
                    if paren_depth == 0 {
                        return true;
                    }
                    paren_depth -= 1;
                }
                TokenKind::LeftBracket => bracket_depth += 1,
                TokenKind::RightBracket => {
                    if bracket_depth == 0 {
                        return true;
                    }
                    bracket_depth -= 1;
                }
                TokenKind::Comma | TokenKind::Semicolon 
                    if paren_depth == 0 && bracket_depth == 0 => {
                    return true;
                }
                _ => {}
            }
            position += 1;
        }
        
        false
    }

    /// Check if a delimiter insertion would be safe
    pub fn is_delimiter_insertion_safe(&self, _delimiter: &TokenKind) -> bool {
        // Simple heuristic: safe if we're not in the middle of an expression
        self.find_expression_boundary()
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
    
    /// Get the current token kind
    pub fn current_kind(&self) -> &TokenKind {
        &self.peek().kind
    }
    
    /// Check if current token is an identifier with specific value
    pub fn check_identifier_with_value(&self, value: &str) -> bool {
        match &self.peek().kind {
            TokenKind::Identifier(id) => id == value,
            _ => false,
        }
    }
    
    /// Set position in the stream (for backtracking)
    pub fn set_position(&mut self, position: usize) {
        self.current = position.min(self.tokens.len());
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