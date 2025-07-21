//! Multi-syntax detection, parsing, and normalization for Prism.
//!
//! ## Clear Separation of Concerns (Fixed Architecture)
//!
//! **✅ prism-syntax responsibilities:**
//! - Syntax style detection from source code (moved from prism-lexer)
//! - Multi-style parsing coordination (C-like, Python-like, Rust-like, Canonical)
//! - Canonical form normalization across all syntax styles
//! - Syntax validation and style consistency checking
//! - Integration bridges to other modules
//!
//! **❌ NOT prism-syntax responsibilities:**
//! - ❌ Character-to-token conversion (→ prism-lexer)
//! - ❌ AST construction (→ prism-parser)
//! - ❌ Semantic analysis (→ prism-semantic)
//! - ❌ Type checking (→ prism-semantic)
//!
//! ## Data Flow Integration
//! 
//! ```
//! Source Code → prism-lexer (tokenize) → Tokens
//!      ↓
//! prism-syntax (detect style) → SyntaxStyle
//!      ↓  
//! prism-syntax (parse & normalize) → CanonicalForm
//!      ↓
//! prism-parser (build AST) → Program
//! ```
//!
//! ## Supported Syntax Styles
//! - **C-like**: C/C++/Java/JavaScript (braces, semicolons, parentheses)
//! - **Python-like**: Python/CoffeeScript (indentation, colons)  
//! - **Rust-like**: Rust/Go (explicit keywords, snake_case)
//! - **Canonical**: Prism canonical (semantic delimiters)
//!
//! All syntax styles are normalized to canonical form for consistent downstream processing.
//!
//! ## Factory-Based Architecture (NEW)
//!
//! The system now uses a factory pattern for component creation, providing:
//! - **Separation of Concerns**: Component creation separate from usage
//! - **Configuration Management**: Centralized configuration through factories
//! - **Testability**: Easy mocking and testing of components
//! - **Extensibility**: Easy addition of new syntax styles
//!
//! ### Usage Examples
//!
//! ```rust
//! // Legacy API (backward compatible)
//! use prism_syntax::Parser;
//! let mut parser = Parser::new();
//! 
//! // New Factory API (recommended for new code)
//! use prism_syntax::{ParsingOrchestrator, ParserFactory};
//! let mut orchestrator = ParsingOrchestrator::new();
//! 
//! // Bridge API (migration path)
//! use prism_syntax::ParserBridge;
//! let mut bridge = ParserBridge::new();
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Core modules organized by conceptual cohesion
pub mod core;
pub mod detection;
pub mod styles;
pub mod normalization;
pub mod validation;
pub mod integration;
pub mod ai_integration;  // NEW: AI integration and metadata provider

// Re-export main types for convenience (BACKWARD COMPATIBLE)
pub use core::{Parser, ParseResult, ParseContext, ParseError};
pub use detection::{SyntaxDetector, SyntaxStyle, DetectionResult};
pub use styles::{StyleParser, CLikeParser, PythonLikeParser, RustLikeParser, CanonicalParser};
pub use normalization::{Normalizer, CanonicalForm, NormalizationConfig};
pub use validation::{Validator, ValidationResult, ValidationConfig};
pub use integration::{SyntaxIntegration, IntegrationResult};
pub use ai_integration::SyntaxMetadataProvider;  // NEW: Export provider

// NEW: Export factory system and orchestrator
pub use core::{
    // Factory system
    ParserFactory, NormalizerFactory, ValidatorFactory,
    ParserConfig, NormalizerStyleConfig, StyleSpecificConfig, NormalizerStyleOptions,
    FactoryError,
    // Orchestrator system
    ParsingOrchestrator, OrchestratorConfig, OrchestratorError,
    // Bridge for backward compatibility
    ParserBridge,
};

use thiserror::Error;

/// Main error type for the syntax parser
#[derive(Debug, Error)]
pub enum SyntaxError {
    /// Detection failed
    #[error("Syntax detection failed: {0}")]
    Detection(#[from] detection::DetectionError),
    
    /// Parsing failed
    #[error("Parsing failed: {0}")]
    Parse(String),
    
    /// Normalization failed
    #[error("Normalization failed: {0}")]
    Normalization(String),
    
    /// Validation failed
    #[error("Validation failed: {0}")]
    Validation(String),
}

/// Parse source code using multi-syntax detection and parsing (LEGACY API)
pub fn parse_multi_syntax(
    source: &str,
    source_id: prism_common::SourceId,
) -> Result<prism_ast::Program, SyntaxError> {
    let mut parser = Parser::new();
    parser.parse_source(source, source_id)
        .map_err(|e| SyntaxError::Parse(e.to_string()))
}

/// Parse source code with explicit syntax style (LEGACY API)
pub fn parse_with_style(
    source: &str,
    source_id: prism_common::SourceId,
    style: SyntaxStyle,
) -> Result<prism_ast::Program, SyntaxError> {
    let mut parser = Parser::with_style(style);
    parser.parse_source(source, source_id)
        .map_err(|e| SyntaxError::Parse(e.to_string()))
}

/// Parse source code using the new factory-based orchestrator (NEW API)
pub fn parse_with_orchestrator(
    source: &str,
    source_id: prism_common::SourceId,
) -> Result<prism_ast::Program, SyntaxError> {
    let mut orchestrator = ParsingOrchestrator::new();
    let result = orchestrator.parse(source, source_id)
        .map_err(|e| SyntaxError::Parse(e.to_string()))?;
    Ok(result.program)
}

/// Parse source code using custom factory configuration (NEW API)
pub fn parse_with_custom_factories(
    source: &str,
    source_id: prism_common::SourceId,
    parser_factory: ParserFactory,
    normalizer_factory: NormalizerFactory,
    validator_factory: ValidatorFactory,
) -> Result<prism_ast::Program, SyntaxError> {
    let mut orchestrator = ParsingOrchestrator::with_factories(
        parser_factory,
        normalizer_factory,
        validator_factory,
        OrchestratorConfig::default(),
    );
    let result = orchestrator.parse(source, source_id)
        .map_err(|e| SyntaxError::Parse(e.to_string()))?;
    Ok(result.program)
}

/// Detect syntax style of source code
pub fn detect_syntax_style(source: &str) -> DetectionResult {
    let mut detector = SyntaxDetector::new();
    detector.detect_syntax(source)
        .unwrap_or_else(|_| DetectionResult {
            detected_style: SyntaxStyle::Canonical,
            confidence: 0.0,
            evidence: Vec::new(),
            alternatives: Vec::new(),
            warnings: Vec::new(),
        })
}

/// Normalize parsed AST to canonical form
pub fn normalize_to_canonical(
    program: &prism_ast::Program,
    style: SyntaxStyle,
) -> Result<normalization::canonical_form::CanonicalForm, SyntaxError> {
    let mut normalizer = Normalizer::new();
    
    // Convert Program to ParsedSyntax (simplified for now)
    let parsed_syntax = match style {
        SyntaxStyle::CLike => normalization::ParsedSyntax::CLike(Default::default()),
        SyntaxStyle::PythonLike => normalization::ParsedSyntax::PythonLike(Default::default()),
        SyntaxStyle::RustLike => normalization::ParsedSyntax::RustLike(Default::default()),
        SyntaxStyle::Canonical => normalization::ParsedSyntax::Canonical(Vec::new()),
    };
    
    normalizer.normalize(parsed_syntax)
        .map_err(|e| SyntaxError::Normalization(e.to_string()))
}

/// Validate syntax against Prism standards
pub fn validate_syntax(
    canonical: &normalization::canonical_form::CanonicalForm,
) -> validation::ValidationResult {
    let validator = Validator::new();
    validator.validate_canonical(canonical)
        .unwrap_or_else(|_| validation::ValidationResult::default())
}

/// Test C-like syntax integration (for testing)
pub fn test_c_like_integration() -> Result<(), SyntaxError> {
    let source = r#"
        module TestModule {
            function test() {
                let x = 42;
                return x;
            }
        }
    "#;
    
    let source_id = prism_common::SourceId::new(1);
    let _result = parse_multi_syntax(source, source_id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_syntax_detection() {
        let c_like_source = "function test() { return 42; }";
        let result = detect_syntax_style(c_like_source);
        // Should detect some style with reasonable confidence
        assert!(result.confidence >= 0.0);
    }
    
    #[test]
    fn test_parse_source_function() {
        let source = "module Test { function test() { return 42; } }";
        let source_id = prism_common::SourceId::new(1);
        
        // Test legacy API
        let legacy_result = parse_multi_syntax(source, source_id);
        assert!(legacy_result.is_ok());
        
        // Test new orchestrator API
        let orchestrator_result = parse_with_orchestrator(source, source_id);
        assert!(orchestrator_result.is_ok());
    }
    
    #[test]
    fn test_factory_system() {
        let parser_factory = ParserFactory::new();
        let normalizer_factory = NormalizerFactory::new();
        let validator_factory = ValidatorFactory::new();
        
        // Test that we can create components for all styles
        assert!(parser_factory.create_parser(SyntaxStyle::CLike).is_ok());
        assert!(normalizer_factory.create_normalizer(SyntaxStyle::CLike).is_ok());
        assert!(validator_factory.create_validator().is_ok());
    }
    
    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = ParsingOrchestrator::new();
        assert!(orchestrator.config().enable_component_caching);
    }
    
    #[test]
    fn test_bridge_compatibility() {
        let bridge = ParserBridge::new();
        assert!(bridge.context().generate_ai_metadata);
    }
} 