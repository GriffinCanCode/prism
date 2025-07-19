//! Parser for Python-like syntax (Python/CoffeeScript style).

use crate::{
    styles::{StyleParser, StyleConfig, ParserCapabilities, ErrorRecoveryLevel, ConfigError},
    detection::SyntaxStyle,
};
use prism_lexer::Token;
use thiserror::Error;

#[derive(Debug)]
pub struct PythonLikeParser {
    config: PythonLikeConfig,
}

#[derive(Debug, Clone, Default)]
pub struct PythonLikeConfig {
    pub indentation_size: usize,
    pub allow_mixed_indentation: bool,
}

#[derive(Debug)]
pub struct PythonLikeSyntax {
    pub content: String,
}

#[derive(Debug, Error)]
pub enum PythonLikeError {
    #[error("Indentation error at line {line}")]
    IndentationError { line: usize },
}

impl StyleConfig for PythonLikeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl StyleParser for PythonLikeParser {
    type Output = PythonLikeSyntax;
    type Error = PythonLikeError;
    type Config = PythonLikeConfig;
    
    fn new() -> Self {
        Self { config: PythonLikeConfig::default() }
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self { config }
    }
    
    fn parse(&mut self, tokens: Vec<Token>) -> Result<Self::Output, Self::Error> {
        Ok(PythonLikeSyntax {
            content: format!("Parsed {} Python-like tokens", tokens.len()),
        })
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::PythonLike
    }
    
    fn capabilities(&self) -> ParserCapabilities {
        ParserCapabilities {
            supports_mixed_indentation: self.config.allow_mixed_indentation,
            error_recovery_level: ErrorRecoveryLevel::Intelligent,
            ..Default::default()
        }
    }
} 