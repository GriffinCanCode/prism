//! Token stream management for multi-syntax parsing.
//!
//! This module manages token streams with rich metadata for different syntax styles,
//! maintaining conceptual cohesion around "token stream coordination and metadata preservation".

use prism_lexer::Token;
use prism_common::span::Span;

/// Enhanced token stream with metadata
#[derive(Debug, Clone)]
pub struct TokenStream {
    /// The tokens in the stream
    pub tokens: Vec<Token>,
    
    /// Current position in the stream
    pub position: usize,
    
    /// Metadata for each token position
    pub metadata: Vec<TokenMetadata>,
}

/// Position information within a token stream
#[derive(Debug, Clone)]
pub struct TokenPosition {
    /// Index in the token stream
    pub index: usize,
    
    /// Line number (1-indexed)
    pub line: usize,
    
    /// Column number (1-indexed)
    pub column: usize,
}

/// Metadata associated with a token
#[derive(Debug, Clone)]
pub struct TokenMetadata {
    /// Semantic context for this token
    pub semantic_context: Option<String>,
    
    /// AI hints for this token
    pub ai_hints: Vec<String>,
    
    /// Whether this token was generated during error recovery
    pub is_error_recovery: bool,
    
    /// Original span in source code
    pub original_span: Span,
}

impl TokenStream {
    /// Create a new token stream from tokens
    pub fn new(tokens: Vec<Token>) -> Self {
        let metadata = tokens.iter().map(|_| TokenMetadata::default()).collect();
        
        Self {
            tokens,
            position: 0,
            metadata,
        }
    }
    
    /// Get the current token
    pub fn current(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }
    
    /// Advance to the next token
    pub fn advance(&mut self) -> Option<&Token> {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
        self.current()
    }
    
    /// Peek at the next token without advancing
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position + 1)
    }
    
    /// Check if at end of stream
    pub fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }
    
    /// Get current position information
    pub fn current_position(&self) -> TokenPosition {
        TokenPosition {
            index: self.position,
            line: 1, // TODO: Calculate from token spans
            column: 1, // TODO: Calculate from token spans
        }
    }
}

impl Default for TokenMetadata {
    fn default() -> Self {
        Self {
            semantic_context: None,
            ai_hints: Vec::new(),
            is_error_recovery: false,
            original_span: Span::dummy(), // TODO: Use actual span
        }
    }
} 