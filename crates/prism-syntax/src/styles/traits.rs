//! Common traits and interfaces for style-specific parsers.
//!
//! This module defines the common interface that all style parsers must implement,
//! maintaining conceptual cohesion around "unified parser interfaces and capabilities".

use crate::detection::SyntaxStyle;
use prism_lexer::Token;
use thiserror::Error;

/// Common interface for all syntax style parsers.
/// 
/// This trait defines the contract that all style-specific parsers must
/// implement. It ensures consistency across different syntax styles while
/// allowing for style-specific optimizations.
pub trait StyleParser {
    /// The parsed output type for this style
    type Output;
    
    /// Error type for this parser
    type Error: std::error::Error;
    
    /// Configuration type for this parser
    type Config: Default;
    
    /// Creates a new parser with default configuration.
    fn new() -> Self;
    
    /// Creates a new parser with custom configuration.
    fn with_config(config: Self::Config) -> Self;
    
    /// Parses a token stream into the style-specific representation.
    fn parse(&mut self, tokens: Vec<Token>) -> Result<Self::Output, Self::Error>;
    
    /// Returns the syntax style this parser handles.
    fn syntax_style(&self) -> SyntaxStyle;
    
    /// Returns parser capabilities and limitations.
    fn capabilities(&self) -> ParserCapabilities;
}

/// Configuration for style parsers
pub trait StyleConfig: Default + Clone {
    /// Validate the configuration
    fn validate(&self) -> Result<(), ConfigError>;
}

/// Parser capabilities and feature support
#[derive(Debug, Clone)]
pub struct ParserCapabilities {
    /// Supports mixed indentation styles
    pub supports_mixed_indentation: bool,
    
    /// Supports optional semicolons
    pub supports_optional_semicolons: bool,
    
    /// Supports trailing commas
    pub supports_trailing_commas: bool,
    
    /// Supports nested comments
    pub supports_nested_comments: bool,
    
    /// Error recovery sophistication level
    pub error_recovery_level: ErrorRecoveryLevel,
    
    /// Maximum nesting depth supported
    pub max_nesting_depth: usize,
    
    /// Whether AI metadata generation is supported
    pub supports_ai_metadata: bool,
}

/// Level of error recovery sophistication
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorRecoveryLevel {
    /// No error recovery (fail fast)
    None,
    
    /// Basic error recovery (skip tokens)
    Basic,
    
    /// Advanced error recovery (insertion, replacement)
    Advanced,
    
    /// Intelligent error recovery (context-aware)
    Intelligent,
}

/// Configuration validation errors
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Invalid parameter value
    #[error("Invalid parameter {parameter}: {reason}")]
    InvalidParameter { parameter: String, reason: String },
    
    /// Conflicting configuration options
    #[error("Conflicting options: {options:?}")]
    ConflictingOptions { options: Vec<String> },
    
    /// Missing required configuration
    #[error("Missing required configuration: {required}")]
    MissingRequired { required: String },
}

impl Default for ParserCapabilities {
    fn default() -> Self {
        Self {
            supports_mixed_indentation: false,
            supports_optional_semicolons: true,
            supports_trailing_commas: true,
            supports_nested_comments: false,
            error_recovery_level: ErrorRecoveryLevel::Basic,
            max_nesting_depth: 100,
            supports_ai_metadata: true,
        }
    }
} 