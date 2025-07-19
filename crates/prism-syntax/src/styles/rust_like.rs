//! Parser for Rust-like syntax (Rust/Go style).

use crate::{
    styles::{StyleParser, StyleConfig, ParserCapabilities, ErrorRecoveryLevel, ConfigError},
    detection::SyntaxStyle,
};
use prism_lexer::Token;
use thiserror::Error;

#[derive(Debug)]
pub struct RustLikeParser {
    config: RustLikeConfig,
}

#[derive(Debug, Clone, Default)]
pub struct RustLikeConfig {
    pub enforce_snake_case: bool,
    pub require_explicit_types: bool,
}

#[derive(Debug)]
pub struct RustLikeSyntax {
    pub content: String,
}

#[derive(Debug, Error)]
pub enum RustLikeError {
    #[error("Invalid naming convention at line {line}")]
    InvalidNaming { line: usize },
}

impl StyleConfig for RustLikeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl StyleParser for RustLikeParser {
    type Output = RustLikeSyntax;
    type Error = RustLikeError;
    type Config = RustLikeConfig;
    
    fn new() -> Self {
        Self { config: RustLikeConfig::default() }
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self { config }
    }
    
    fn parse(&mut self, tokens: Vec<Token>) -> Result<Self::Output, Self::Error> {
        Ok(RustLikeSyntax {
            content: format!("Parsed {} Rust-like tokens", tokens.len()),
        })
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::RustLike
    }
    
    fn capabilities(&self) -> ParserCapabilities {
        ParserCapabilities {
            error_recovery_level: ErrorRecoveryLevel::Advanced,
            ..Default::default()
        }
    }
} 