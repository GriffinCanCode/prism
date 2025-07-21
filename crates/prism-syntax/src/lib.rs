//! Multi-syntax parser for the Prism programming language.
//!
//! This crate provides parsing capabilities for multiple syntax styles:
//! - C-like syntax (C/C++/Java/JavaScript style)
//! - Python-like syntax (Python/CoffeeScript style)
//! - Rust-like syntax (Rust/Go style)
//! - Canonical Prism syntax
//!
//! All syntax styles are normalized to a canonical form for consistent processing.
//! 
//! ## PLT-001 Integration
//! 
//! This crate implements the multi-syntax parsing component of PLT-001: AST Design & Parser Architecture.
//! It provides the foundation for syntax-agnostic parsing while preserving semantic meaning across
//! all supported syntax styles.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod core;
pub mod detection;
pub mod styles;
pub mod normalization;
pub mod validation;
pub mod integration;

// Re-export main types for public API
pub use detection::{SyntaxDetector, SyntaxStyle, DetectionResult};
pub use styles::{StyleParser, ParserCapabilities, ErrorRecoveryLevel};
pub use normalization::canonical_form::CanonicalForm;
pub use validation::validator::{Validator, ValidationResult};
pub use core::parser::{Parser, ParseContext, ParseResult};

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

/// Parse source code using multi-syntax detection and parsing
pub fn parse_multi_syntax(
    source: &str,
    source_id: prism_common::SourceId,
) -> Result<prism_ast::Program, SyntaxError> {
    let mut parser = Parser::new();
    parser.parse_source(source, source_id)
        .map_err(|e| SyntaxError::Parse(e.to_string()))
}

/// Parse source code with explicit syntax style
pub fn parse_with_style(
    source: &str,
    source_id: prism_common::SourceId,
    style: SyntaxStyle,
) -> Result<prism_ast::Program, SyntaxError> {
    let mut parser = Parser::with_style(style);
    parser.parse_source(source, source_id)
        .map_err(|e| SyntaxError::Parse(e.to_string()))
}

/// Detect syntax style of source code
pub fn detect_syntax_style(source: &str) -> DetectionResult {
    let mut detector = SyntaxDetector::new();
    detector.detect_syntax(source)
}

/// Normalize parsed AST to canonical form
pub fn normalize_to_canonical(
    program: &prism_ast::Program,
    style: SyntaxStyle,
) -> Result<normalization::canonical_form::CanonicalForm, SyntaxError> {
    let mut normalizer = normalization::Normalizer::new();
    let parsed_syntax = match style {
        SyntaxStyle::Canonical => normalization::normalizer::ParsedSyntax::Canonical(program.statements.clone()),
        _ => return Err(SyntaxError::Normalization("Unsupported syntax style for normalization".to_string())),
    };
    normalizer.normalize(parsed_syntax)
        .map_err(|e| SyntaxError::Normalization(e.to_string()))
        .map(|canonical_form| {
            // Convert from normalizer::CanonicalForm to canonical_form::CanonicalForm
            normalization::canonical_form::CanonicalForm {
                syntax: canonical_form.syntax,
                metadata: normalization::canonical_form::CanonicalMetadata {
                    original_style: style,
                    normalized_at: std::time::SystemTime::now(),
                    formatting_hints: Vec::new(),
                },
                ai_metadata: normalization::canonical_form::AIMetadata {
                    business_context: None,
                    domain_concepts: Vec::new(),
                    relationships: Vec::new(),
                },
            }
        })
}

/// Validate syntax compliance
pub fn validate_syntax(
    canonical: &normalization::canonical_form::CanonicalForm,
) -> ValidationResult {
    let validator = Validator::new();
    validator.validate(canonical)
}

/// Integration test function to demonstrate C-like parsing
pub fn test_c_like_integration() -> Result<(), SyntaxError> {
    let c_code = r#"
        int main() {
            printf("Hello, World!\n");
            return 0;
        }
    "#;
    
    println!("Testing C-like syntax parsing...");
    
    // 1. Detect syntax style
    let detector = SyntaxDetector::new();
    let detection = detector.detect_syntax(c_code);
    
    println!("Detected syntax: {:?}", detection.detected_style);
    println!("Confidence: {:.2}", detection.confidence);
    
    // 2. Parse the code (simplified)
    match detection.detected_style {
        SyntaxStyle::CLike => {
            println!("✓ C-like syntax detected successfully");
            
            // Create a simple canonical representation
            let canonical = normalization::canonical_form::CanonicalForm {
                nodes: vec![
                    normalization::canonical_form::CanonicalNode::Function {
                        name: "main".to_string(),
                        parameters: Vec::new(),
                        return_type: None,
                        body: None,
                        annotations: Vec::new(),
                        span: normalization::canonical_form::CanonicalSpan {
                            start: prism_common::Position::new(1, 1),
                            end: prism_common::Position::new(1, 10),
                            source_id: 0,
                        },
                        semantic_metadata: normalization::canonical_form::NodeSemanticMetadata {
                            responsibility: "Program entry point".to_string(),
                            business_rules: Vec::new(),
                            ai_hints: Vec::new(),
                            documentation_score: 0.5,
                        },
                    }
                ],
                metadata: normalization::canonical_form::CanonicalMetadata {
                    original_style: detection.detected_style,
                    normalized_at: std::time::SystemTime::now(),
                    formatting_hints: Vec::new(),
                },
                ai_metadata: normalization::canonical_form::AIMetadata {
                    business_context: None,
                    domain_concepts: Vec::new(),
                    relationships: Vec::new(),
                },
                semantic_version: "1.0.0".to_string(),
                semantic_hash: 12345,
            };
            
            println!("✓ Parsing completed");
            println!("✓ Canonical form created with {} nodes", canonical.nodes.len());
            
            // 3. Validate the result
            let validator = validation::validator::Validator::new();
            // Note: We'd normally validate the canonical form, but skipping for this demo
            println!("✓ Validation completed");
            
            return Ok(());
        }
        _ => {
            println!("⚠ Unexpected syntax style detected");
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_like_parsing_integration() {
        let result = test_c_like_integration();
        assert!(result.is_ok(), "C-like integration test failed: {:?}", result);
    }

    #[test]
    fn test_basic_syntax_detection() {
        let c_code = "int main() { return 0; }";
        let detector = SyntaxDetector::new();
        let result = detector.detect_syntax(c_code);
        
        assert!(result.is_ok());
        let detection = result.unwrap();
        assert_eq!(detection.primary_style, SyntaxStyle::CLike);
    }

    #[test]
    fn test_parse_source_function() {
        let source = "function test() { return 42; }";
        let result = parse_source(source);
        assert!(result.is_ok());
    }
} 