//! Lexical analyzer for the Prism programming language
//!
//! This crate provides tokenization capabilities focused purely on
//! converting characters to tokens with basic semantic enrichment.
//!
//! ## Clear Separation of Concerns
//! 
//! **prism-lexer responsibilities:**
//! - ✅ Character-to-Token conversion
//! - ✅ Basic token enrichment (metadata, context hints)
//! - ✅ Lexical error recovery
//! 
//! **NOT prism-lexer responsibilities:**
//! - ❌ Syntax style detection (→ prism-syntax)
//! - ❌ Multi-token semantic analysis (→ prism-parser/prism-semantic)
//! - ❌ AST construction (→ prism-parser)
//! - ❌ Cross-token relationship analysis (→ prism-semantic)
//!
//! ## Data Flow
//! 
//! ```
//! Source Code → prism-lexer → Enriched Tokens → prism-syntax → Canonical Form → prism-parser → AST
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod lexer;
pub mod recovery;
pub mod token;
pub mod semantic;
pub mod semantic_lexer;

#[cfg(test)]
mod tests;

// Re-export main types for clean API
pub use lexer::{Lexer, LexerError, LexerConfig, LexerResult};
pub use recovery::{ErrorRecovery, RecoveryStrategy, RecoveryAction, ErrorPattern, LexerDiagnostics};
pub use token::{Token, TokenKind};
pub use semantic_lexer::{SemanticLexer, SemanticLexerConfig, SemanticLexerResult}; 