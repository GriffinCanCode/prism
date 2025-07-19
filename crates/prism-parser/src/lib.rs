//! Parser for the Prism programming language
//!
//! This crate provides a hybrid parsing approach combining recursive descent
//! for complex language constructs with parser combinators for simpler patterns.
//! The parser is designed for AI-first development with rich error recovery
//! and semantic metadata extraction.
//!
//! ## Architecture
//!
//! The parser is organized into three logical modules following conceptual cohesion:
//!
//! ### Core Infrastructure (`core/`)
//! - Token stream navigation and management
//! - Parsing coordination and orchestration
//! - Error handling and recovery mechanisms
//! - Operator precedence management
//!
//! ### Specialized Parsers (`parsers/`)
//! - Expression parsing with Pratt parser
//! - Statement parsing for control flow and declarations
//! - Type parsing with semantic constraints
//! - Module parsing for organizational structures
//!
//! ### Analysis and Validation (`analysis/`)
//! - Semantic context extraction for AI comprehension
//! - Constraint validation for business rules
//! - Parser combinators for composable patterns

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Core parsing infrastructure
pub mod core;

// Specialized parsing logic
pub mod parsers;

// Analysis and validation
pub mod analysis;

// Main parser coordinator
pub mod parser;

// Re-export main types for public API
pub use parser::{Parser, ParseConfig};
pub use core::{ParseError, ParseErrorKind, ParseResult};

// Re-export commonly used types from each module
pub use core::{
    TokenStreamManager,
    ParsingCoordinator,
    ErrorContext,
    Precedence,
};

pub use parsers::{
    ExpressionParser,
    StatementParser,
    TypeParser,
    ModuleParser,
};

pub use analysis::{
    SemanticContextExtractor,
    ConstraintValidator,
    ValidationResult,
    ValidationError,
};

use prism_ast::{Program, AstNode, Item};
use prism_lexer::Token;

/// Parse a complete program from tokens
pub fn parse_program(tokens: Vec<Token>) -> ParseResult<Program> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

/// Parse a program with custom configuration
pub fn parse_program_with_config(tokens: Vec<Token>, config: ParseConfig) -> ParseResult<Program> {
    let mut parser = Parser::with_config(tokens, config);
    parser.parse_program()
}

/// Parse a single expression from tokens
pub fn parse_expression(tokens: Vec<Token>) -> ParseResult<prism_ast::NodeId> {
    let mut parser = Parser::new(tokens);
    parser.parse_expression()
}

/// Parse a single statement from tokens
pub fn parse_statement(tokens: Vec<Token>) -> ParseResult<prism_ast::NodeId> {
    let mut parser = Parser::new(tokens);
    parser.parse_statement()
}

/// Parse a type annotation from tokens
pub fn parse_type(tokens: Vec<Token>) -> ParseResult<prism_ast::NodeId> {
    let mut parser = Parser::new(tokens);
    parser.parse_type()
}
