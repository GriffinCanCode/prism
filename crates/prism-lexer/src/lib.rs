//! Lexical analyzer for the Prism programming language
//!
//! This crate provides tokenization capabilities focused purely on
//! converting characters to tokens. It does NOT handle:
//! - Syntax style detection (belongs in parser)
//! - Semantic analysis (belongs in parser)
//! - Token relationship analysis (belongs in parser)
//!
//! ## Responsibility
//! 
//! This crate has ONE responsibility: **Character-to-Token Conversion**
//! 
//! It takes source code as a string and produces a stream of tokens,
//! handling lexical errors and providing basic error recovery.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod lexer;
pub mod recovery;
pub mod token;

#[cfg(test)]
mod tests;

// Re-export main types for clean API
pub use lexer::{Lexer, LexerError, LexerConfig, LexerResult};
pub use recovery::{ErrorRecovery, RecoveryStrategy, RecoveryAction, ErrorPattern, LexerDiagnostics};
pub use token::{Token, TokenKind}; 