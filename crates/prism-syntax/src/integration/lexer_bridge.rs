//! Bridge to prism-lexer for clean integration.
//!
//! ## Fixed Architecture Integration
//!
//! This bridge ensures clean data flow between prism-lexer and prism-syntax:
//! - Receives enriched tokens from prism-lexer
//! - Prepares tokens for syntax-specific parsing
//! - Maintains separation of concerns between modules

use prism_lexer::Token;
use thiserror::Error;

/// Bridge for clean lexer-to-syntax integration
#[derive(Debug)]
pub struct LexerBridge {
    /// Bridge configuration
    config: BridgeConfig,
}

/// Configuration for lexer bridge
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Whether to validate token consistency
    pub validate_tokens: bool,
    /// Whether to preserve all token metadata
    pub preserve_metadata: bool,
}

/// Token preparation result for syntax parsing
#[derive(Debug)]
pub struct TokenPreparation {
    /// Preparation successful
    pub success: bool,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Any warnings during preparation
    pub warnings: Vec<String>,
}

/// Lexer integration errors
#[derive(Debug, Error)]
pub enum LexerIntegrationError {
    /// Token preparation failed
    #[error("Token preparation failed: {reason}")]
    PreparationFailed { reason: String },
    /// Invalid token stream
    #[error("Invalid token stream: {reason}")]
    InvalidTokenStream { reason: String },
}

impl LexerBridge {
    /// Create new lexer bridge
    pub fn new() -> Self {
        Self { 
            config: BridgeConfig::default()
        }
    }
    
    /// Create bridge with custom configuration
    pub fn with_config(config: BridgeConfig) -> Self {
        Self { config }
    }
    
    /// Prepare tokens from lexer for syntax parsing
    /// This maintains clean separation: lexer enriches tokens, syntax parses them
    pub fn prepare_tokens(&self, tokens: Vec<Token>) -> Result<TokenPreparation, LexerIntegrationError> {
        let mut warnings = Vec::new();
        
        // Validate token stream if configured
        if self.config.validate_tokens {
            if tokens.is_empty() {
                return Err(LexerIntegrationError::InvalidTokenStream {
                    reason: "Empty token stream".to_string()
                });
            }
            
            // Check for EOF token
            if !tokens.iter().any(|t| matches!(t.kind, prism_lexer::TokenKind::Eof)) {
                warnings.push("Missing EOF token".to_string());
            }
        }
        
        Ok(TokenPreparation {
            success: true,
            tokens_processed: tokens.len(),
            warnings,
        })
    }
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            validate_tokens: true,
            preserve_metadata: true,
        }
    }
}

impl Default for LexerBridge {
    fn default() -> Self {
        Self::new()
    }
} 