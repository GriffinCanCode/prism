//! External integrations for multi-syntax parsing.
//! 
//! This module provides integration bridges with other Prism systems,
//! maintaining conceptual cohesion around "external system integration and data exchange".

pub mod ast_bridge;
pub mod lexer_bridge;
pub mod doc_bridge;

pub use ast_bridge::{AstBridge, AstConversionResult, AstIntegrationError};
pub use lexer_bridge::{LexerBridge, TokenAdaptation, LexerIntegrationError};
pub use doc_bridge::{DocumentationBridge, DocumentationResult, DocumentationError}; 