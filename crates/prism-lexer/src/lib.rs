//! Lexical analyzer for the Prism programming language
//!
//! This crate provides tokenization capabilities with rich semantic information
//! designed for AI-first development and analysis.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod lexer;
pub mod recovery;
pub mod semantic;
pub mod syntax;
pub mod token;

#[cfg(test)]
mod tests;

// Re-export main types
pub use lexer::{Lexer, LexerError, LexerConfig, LexerResult, SemanticLexer, SemanticSummary};
pub use recovery::{ErrorRecovery, RecoveryStrategy, RecoveryAction, ErrorPattern, LexerDiagnostics};
pub use semantic::{SemanticAnalyzer, SemanticPattern, PatternType, IdentifierUsage};
pub use syntax::{SyntaxDetector, SyntaxEvidence, StyleRules, MixedStyleWarning};
pub use token::{SemanticContext, SyntaxStyle, Token, TokenKind}; 