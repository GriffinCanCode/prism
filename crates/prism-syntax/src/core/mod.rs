//! Core parsing infrastructure for multi-syntax parsing.
//! 
//! This module provides the foundational components for parsing multiple
//! syntax styles into a unified canonical representation. It coordinates
//! between syntax detection, style-specific parsing, and normalization
//! while maintaining conceptual cohesion around the single responsibility
//! of "parsing coordination and syntax normalization".

pub mod parser;
pub mod token_stream;
pub mod error_recovery;
pub mod factories;  // NEW: Factory system for component creation
pub mod orchestrator;  // NEW: Orchestrator that uses factories
pub mod parser_bridge;  // NEW: Backward compatibility bridge

pub use parser::{Parser, ParseResult, ParseContext, ParseError, ValidationLevel};
pub use token_stream::{TokenStream, TokenPosition, TokenMetadata};
pub use error_recovery::{ErrorRecovery, RecoveryStrategy, RecoveryPoint};
// NEW: Export factory types
pub use factories::{
    ParserFactory, NormalizerFactory, ValidatorFactory,
    ParserConfig, NormalizerStyleConfig, StyleSpecificConfig, NormalizerStyleOptions,
    FactoryError,
};
// NEW: Export orchestrator
pub use orchestrator::{ParsingOrchestrator, OrchestratorConfig, OrchestratorError};
// NEW: Export bridge for backward compatibility
pub use parser_bridge::ParserBridge;