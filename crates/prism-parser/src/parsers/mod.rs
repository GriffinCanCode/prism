//! Specialized Parsing Logic
//!
//! This module contains the specialized parsers for different language constructs:
//! - Expression parsing with Pratt parser for operator precedence
//! - Statement parsing for all statement types and control flow
//! - Type parsing with semantic constraints and business rules
//! - Module parsing for organizational structures and capabilities
//!
//! Each parser follows the "One Concept Per File" principle, focusing
//! on a single aspect of the language grammar while delegating to
//! other specialized parsers as needed.

pub mod expression_parser;
pub mod statement_parser;
pub mod type_parser;
pub mod module_parser;
pub mod function_parser;

// Re-export parser types for convenience
pub use expression_parser::ExpressionParser;
pub use statement_parser::StatementParser;
pub use type_parser::TypeParser;
pub use module_parser::{ModuleParser, ImportKind, ExportKind};
pub use function_parser::FunctionParser; 