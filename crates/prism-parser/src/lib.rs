//! Parser for the Prism programming language
//!
//! This crate provides a hybrid parsing approach combining recursive descent
//! for complex language constructs with parser combinators for simpler patterns.
//! The parser is designed for AI-first development with rich error recovery
//! and semantic metadata extraction.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod parser;
pub mod precedence;
pub mod recovery;
pub mod combinators;
pub mod constraint_validation;

// Re-export main types
pub use error::{ParseError, ParseErrorKind, ParseResult};
pub use parser::Parser;
pub use constraint_validation::{ConstraintValidator, ValidationResult, ValidationError};

use prism_ast::{Program, AstNode, Item};
use prism_lexer::Token;

/// Parse a complete Prism program from a token stream
pub fn parse_program(tokens: Vec<Token>) -> ParseResult<Program> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

/// Parse a single expression from a token stream
pub fn parse_expression(tokens: Vec<Token>) -> ParseResult<AstNode<prism_ast::Expr>> {
    let mut parser = Parser::new(tokens);
    parser.parse_expression()
}

/// Parse a single statement from a token stream
pub fn parse_statement(tokens: Vec<Token>) -> ParseResult<AstNode<prism_ast::Stmt>> {
    let mut parser = Parser::new(tokens);
    parser.parse_statement()
}

/// Parse a type annotation from a token stream
pub fn parse_type(tokens: Vec<Token>) -> ParseResult<AstNode<prism_ast::Type>> {
    let mut parser = Parser::new(tokens);
    parser.parse_type()
}

/// Configuration for parser behavior
#[derive(Debug, Clone)]
pub struct ParseConfig {
    /// Enable aggressive error recovery
    pub aggressive_recovery: bool,
    /// Extract AI context during parsing
    pub extract_ai_context: bool,
    /// Maximum number of errors before stopping
    pub max_errors: usize,
    /// Enable semantic metadata extraction
    pub semantic_metadata: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            aggressive_recovery: true,
            extract_ai_context: true,
            max_errors: 100,
            semantic_metadata: true,
        }
    }
}

/// Parse a program with custom configuration
pub fn parse_program_with_config(
    tokens: Vec<Token>,
    config: ParseConfig,
) -> ParseResult<Program> {
    let mut parser = Parser::with_config(tokens, config);
    parser.parse_program()
}
