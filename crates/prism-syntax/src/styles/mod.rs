//! Syntax style implementations for multi-syntax parsing.
//! 
//! This module contains the specific parsing logic for each supported syntax
//! style. Each style implementation follows the same interface but handles
//! the unique characteristics of its syntax, maintaining conceptual cohesion
//! around "style-specific parsing with unified interfaces".

pub mod c_like;
pub mod python_like;
pub mod rust_like;
pub mod canonical;
pub mod traits;

pub use traits::{StyleParser, StyleConfig, ParserCapabilities, ErrorRecoveryLevel};
pub use c_like::CLikeParser;
pub use python_like::PythonLikeParser;
pub use rust_like::RustLikeParser;
pub use canonical::CanonicalParser; 