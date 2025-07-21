//! AST construction and semantic analysis for Prism.
//!
//! ## Clear Separation of Concerns (Fixed Architecture)
//!
//! **✅ prism-parser responsibilities:**
//! - AST construction from canonical forms or token streams
//! - Multi-token semantic analysis and relationship detection
//! - Cross-token pattern recognition and validation
//! - Semantic-aware error recovery with meaning preservation
//! - AI metadata generation from parsed structures
//! - Integration with semantic analysis systems
//!
//! **❌ NOT prism-parser responsibilities:**
//! - ❌ Character-to-token conversion (→ prism-lexer)
//! - ❌ Syntax style detection (→ prism-syntax)
//! - ❌ Single-token enrichment (→ prism-lexer)
//! - ❌ Style-specific parsing (→ prism-syntax)
//!
//! ## Data Flow Position
//! 
//! ```
//! prism-lexer (tokens) → prism-syntax (canonical) → prism-parser (AST) → prism-semantic (analysis)
//! ```
//!
//! This crate focuses on the **AST construction and multi-token analysis** layer,
//! taking normalized input and producing rich semantic structures.
//!
//! ## Architecture
//!
//! The parser is organized into logical modules following conceptual cohesion:
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
//!
//! ### Integration Layer (`integration/`)
//! - **NEW**: Complete PLT-001 integration orchestrating all subsystems
//! - Multi-syntax coordination with semantic preservation
//! - Documentation validation and cohesion analysis integration
//! - AI metadata generation from all analysis systems
//!
//! ## Examples
//!
//! ### Basic Parsing
//! ```rust
//! use prism_parser::{parse_program, ParseConfig};
//! use prism_lexer::tokenize;
//!
//! let source = "function hello() -> String { return \"Hello, World!\" }";
//! let tokens = tokenize(source)?;
//! let program = parse_program(tokens)?;
//! ```
//!
//! ### Full PLT-001 Analysis
//! ```rust
//! use prism_parser::integration::{IntegratedParser, IntegrationConfig};
//! use prism_common::SourceId;
//!
//! let source = r#"
//!     @responsibility "Manages user authentication"
//!     @module "UserAuth"
//!     @description "Secure authentication module"
//!     @author "Security Team"
//!     
//!     module UserAuth {
//!         section interface {
//!             @responsibility "Authenticates users"
//!             function authenticate(user: User) -> Result<Session, Error>
//!                 effects [Database.Query, Audit.Log]
//!                 requires user.is_verified()
//!             {
//!                 // Implementation
//!             }
//!         }
//!     }
//! "#;
//!
//! let mut parser = IntegratedParser::new();
//! let result = parser.parse_with_full_analysis(source, SourceId::new(1))?;
//!
//! // Access comprehensive analysis results
//! println!("Detected syntax: {:?}", result.detected_syntax);
//! println!("Documentation compliant: {}", result.documentation_analysis.is_some());
//! println!("Cohesion score: {:.1}", result.cohesion_analysis.as_ref().unwrap().overall_score);
//! ```

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

// Context-guided diagnostic suggestions
pub mod suggestions;

// Stream-based combinators (for ParseStream API)
pub mod stream_combinators;

// Parse trait implementations for AST types
pub mod ast_parse_impls;

// Main parser coordinator
pub mod parser;

// **NEW**: Complete PLT-001 integration layer
pub mod integration;

// Tests for the parsing system
#[cfg(test)]
mod tests;

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

pub use suggestions::{
    SuggestionEngine,
    SuggestionEngineConfig,
    ContextualSuggestion,
    SuggestionType,
    EffortLevel,
};

// **NEW**: Re-export PLT-001 integration functionality
pub use integration::{
    IntegratedParser,
    IntegrationConfig,
    IntegratedParseResult,
    IntegrationError,
    parse_with_full_analysis,
    parse_basic,
};

use prism_ast::{Program, AstNode, Item};
use prism_lexer::Token;

/// Parse a complete program from tokens (legacy API)
pub fn parse_program(tokens: Vec<Token>) -> ParseResult<Program> {
    let mut parser = Parser::new(tokens);
    let result = parser.parse_program()?;
    Ok(result.program)
}

/// Parse a program with custom configuration (legacy API)
pub fn parse_program_with_config(tokens: Vec<Token>, config: ParseConfig) -> ParseResult<Program> {
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program()?;
    Ok(result.program)
}

/// Parse a single expression from tokens (legacy API)
pub fn parse_expression(tokens: Vec<Token>) -> ParseResult<prism_common::NodeId> {
    let mut parser = Parser::new(tokens);
    parser.parse_expression_public()
}

/// Parse a single statement from tokens (legacy API)
pub fn parse_statement(tokens: Vec<Token>) -> ParseResult<prism_common::NodeId> {
    let mut parser = Parser::new(tokens);
    parser.parse_statement()
}

/// Parse a type annotation from tokens (legacy API)
pub fn parse_type(tokens: Vec<Token>) -> ParseResult<prism_common::NodeId> {
    let mut parser = Parser::new(tokens);
    parser.parse_type()
}

/// **NEW**: Parse source code with complete PLT-001 analysis (recommended API)
/// 
/// This is the recommended entry point for parsing Prism code as it provides
/// the complete PLT-001 functionality including:
/// - Multi-syntax parsing with semantic preservation (UPDATED: using factory system)
/// - Documentation validation (PSG-003)
/// - Cohesion analysis (PLD-002)
/// - Semantic type integration (PLD-001)
/// - Effect system analysis (PLD-003)
/// - AI metadata generation
///
/// # Examples
/// 
/// ```rust
/// use prism_parser::parse_source_with_full_analysis;
/// use prism_common::SourceId;
/// 
/// let source = r#"
///     @responsibility "Example module"
///     @module "Example"
///     @description "Demonstrates Prism features"
///     @author "Developer"
///     
///     module Example {
///         section interface {
///             @responsibility "Greets users"
///             function greet(name: String) -> String {
///                 return "Hello, " + name + "!"
///             }
///         }
///     }
/// "#;
/// 
/// let result = parse_source_with_full_analysis(source, SourceId::new(1))?;
/// println!("Analysis complete! Overall quality score: {:.1}", 
///          result.ai_metadata.as_ref().unwrap().quality_assessment.overall_score);
/// ```
pub fn parse_source_with_full_analysis(
    source: &str,
    source_id: prism_common::SourceId,
) -> Result<integration::IntegratedParseResult, integration::IntegrationError> {
    integration::parse_with_full_analysis(source, source_id)
}

/// **NEW**: Parse source code with basic parsing only (fast path)
/// 
/// This provides fast parsing using the new factory-based orchestrator without 
/// the full PLT-001 analysis pipeline. Use this when you only need the AST 
/// without documentation validation, cohesion analysis, or AI metadata generation.
pub fn parse_source_basic(
    source: &str,
    source_id: prism_common::SourceId,
) -> Result<Program, integration::IntegrationError> {
    integration::parse_basic(source, source_id)
}
