//! Core Parsing Infrastructure
//!
//! This module contains the foundational components for parsing:
//! - Token stream navigation and management
//! - Parsing coordination and orchestration  
//! - Error handling and recovery mechanisms
//! - Operator precedence management
//!
//! These components provide the essential infrastructure that all
//! specialized parsers depend on, following the principle of
//! separating infrastructure concerns from parsing logic.

pub mod token_stream_manager;
pub mod parsing_coordinator;
pub mod error;
pub mod precedence;
pub mod recovery;

// Re-export commonly used types for convenience
pub use token_stream_manager::TokenStreamManager;
pub use parsing_coordinator::ParsingCoordinator;
pub use error::{ParseError, ParseErrorKind, ParseResult, ErrorContext};
pub use precedence::{Precedence, Associativity, infix_precedence, prefix_precedence, associativity};
pub use recovery::{RecoveryStrategy, RecoveryContext, ParsingContext}; 