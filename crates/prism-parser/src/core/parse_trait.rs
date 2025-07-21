//! Parse trait infrastructure inspired by syn
//!
//! This module provides the core parsing traits that enable type-driven parsing
//! while maintaining compatibility with Prism's existing parser architecture.

use crate::core::{ParseError, ParseResult, TokenStreamManager, ParsingCoordinator};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;

/// A checkpoint for backtracking in the token stream
#[derive(Debug, Clone)]
pub struct Checkpoint {
    position: usize,
}

/// Input stream for parsing - similar to syn's ParseStream
/// 
/// This provides a high-level interface for parsing that wraps
/// the existing TokenStreamManager while adding syn-style convenience methods.
pub struct ParseStream<'a> {
    /// Reference to the token stream manager
    tokens: &'a mut TokenStreamManager,
    /// Reference to the parsing coordinator for creating nodes
    coordinator: &'a mut ParsingCoordinator,
}

impl<'a> ParseStream<'a> {
    /// Create a new parse stream from existing infrastructure
    pub fn new(
        tokens: &'a mut TokenStreamManager,
        coordinator: &'a mut ParsingCoordinator,
    ) -> Self {
        Self { tokens, coordinator }
    }

    /// Parse a type T from the stream
    pub fn parse<T: Parse>(&mut self) -> ParseResult<T> {
        T::parse(self)
    }

    /// Check if the next token matches the given kind
    pub fn peek<T: Peek>(&self, token: T) -> bool {
        token.peek(self.tokens.peek())
    }

    /// Consume and return the next token if it matches
    pub fn parse_token<T: ParseToken>(&mut self) -> ParseResult<T> {
        T::parse_token(self)
    }

    /// Get current span
    pub fn span(&self) -> Span {
        self.tokens.current_span()
    }

    /// Check if at end of stream
    pub fn is_empty(&self) -> bool {
        self.tokens.is_at_end()
    }

    /// Peek at the next token without consuming it
    pub fn peek_token(&self) -> &Token {
        self.tokens.peek()
    }

    /// Advance to the next token and return it
    pub fn advance(&mut self) -> &Token {
        self.tokens.advance()
    }

    /// Check if the current token matches the given kind
    pub fn check(&self, kind: TokenKind) -> bool {
        self.tokens.check(kind)
    }

    /// Access the parsing coordinator for creating AST nodes
    pub fn coordinator(&mut self) -> &mut ParsingCoordinator {
        self.coordinator
    }

    /// Create a checkpoint for potential backtracking
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            position: self.tokens.current_position(),
        }
    }

    /// Restore the stream to a previous checkpoint
    pub fn restore(&mut self, checkpoint: Checkpoint) {
        self.tokens.set_position(checkpoint.position);
    }

    /// Fork the stream for lookahead with proper checkpoint handling
    pub fn fork(&self) -> ParseStreamFork<'_> {
        ParseStreamFork {
            checkpoint: self.checkpoint(),
            original: self,
        }
    }

    /// Try to parse something, restoring position on failure
    pub fn try_parse<T: Parse>(&mut self) -> ParseResult<T> {
        let checkpoint = self.checkpoint();
        match self.parse::<T>() {
            Ok(result) => Ok(result),
            Err(error) => {
                self.restore(checkpoint);
                Err(error)
            }
        }
    }

    /// Parse with speculative execution - returns None if parsing fails
    /// without consuming input
    pub fn speculative_parse<T: Parse>(&mut self) -> Option<T> {
        let checkpoint = self.checkpoint();
        match self.parse::<T>() {
            Ok(result) => Some(result),
            Err(_) => {
                self.restore(checkpoint);
                None
            }
        }
    }
}

/// A forked stream that can be committed or discarded
pub struct ParseStreamFork<'a> {
    checkpoint: Checkpoint,
    original: &'a ParseStream<'a>,
}

impl<'a> ParseStreamFork<'a> {
    /// Commit changes from the fork to the original stream
    pub fn commit(self, original: &mut ParseStream<'a>) {
        // Changes are already in the original stream since we share the same TokenStreamManager
        // We just need to not restore the checkpoint
    }

    /// Discard changes and restore the original stream position
    pub fn discard(self, original: &mut ParseStream<'a>) {
        original.restore(self.checkpoint);
    }
}

/// Core parsing trait - similar to syn's Parse
/// 
/// Types that implement this trait can be parsed from a ParseStream.
/// This enables composable, type-driven parsing.
pub trait Parse: Sized {
    /// Parse this type from the input stream
    fn parse(input: &mut ParseStream) -> ParseResult<Self>;
}

/// Trait for types that can be peeked at
/// 
/// This allows checking if the next token in the stream matches
/// a particular pattern without consuming it.
pub trait Peek {
    /// Check if the given token matches this pattern
    fn peek(&self, token: &Token) -> bool;
}

/// Trait for parsing specific tokens
/// 
/// This is for types that represent specific tokens or token patterns
/// that can be consumed from the stream.
pub trait ParseToken: Sized {
    /// Parse and consume this token from the stream
    fn parse_token(input: &mut ParseStream) -> ParseResult<Self>;
}

// Implement Peek for TokenKind - this allows checking for specific token types
impl Peek for TokenKind {
    fn peek(&self, token: &Token) -> bool {
        std::mem::discriminant(&token.kind) == std::mem::discriminant(self)
    }
}

// Implement ParseToken for TokenKind - allows consuming specific tokens
impl ParseToken for TokenKind {
    fn parse_token(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token();
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(self) {
            let consumed = input.advance();
            Ok(consumed.kind.clone())
        } else {
            Err(ParseError::unexpected_token(
                vec![self.clone()],
                token.kind.clone(),
                token.span,
            ))
        }
    }
}

// Implement ParseToken for specific token types that carry values
impl ParseToken for String {
    fn parse_token(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token().clone(); // Clone to avoid borrowing issues
        match &token.kind {
            TokenKind::Identifier(name) => {
                input.advance();
                Ok(name.clone())
            }
            TokenKind::StringLiteral(s) => {
                input.advance();
                Ok(s.clone())
            }
            _ => Err(ParseError::expected_token(
                TokenKind::Identifier(String::new()),
                token.span,
            )),
        }
    }
}

impl ParseToken for i64 {
    fn parse_token(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token().clone(); // Clone to avoid borrowing issues
        match &token.kind {
            TokenKind::IntegerLiteral(n) => {
                input.advance();
                Ok(*n)
            }
            _ => Err(ParseError::expected_token(
                TokenKind::IntegerLiteral(0),
                token.span,
            )),
        }
    }
}

impl ParseToken for f64 {
    fn parse_token(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token().clone(); // Clone to avoid borrowing issues
        match &token.kind {
            TokenKind::FloatLiteral(f) => {
                input.advance();
                Ok(*f)
            }
            _ => Err(ParseError::expected_token(
                TokenKind::FloatLiteral(0.0),
                token.span,
            )),
        }
    }
} 