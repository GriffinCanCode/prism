//! Multi-Syntax Parser for Prism Language
//!
//! This crate implements Prism's multi-syntax parsing capability, embodying the core principle
//! of **Conceptual Cohesion** by providing a single, focused responsibility: parsing multiple
//! syntax styles (C-like, Python-like, Rust-like, Canonical) into a unified semantic 
//! representation while preserving all semantic meaning and generating AI-comprehensible metadata.
//!
//! ## Conceptual Purpose
//!
//! The `prism-syntax` crate serves a single, clear purpose: **intelligent multi-syntax parsing 
//! with semantic preservation**. This crate bridges the gap between developer syntax preferences
//! and Prism's canonical semantic representation, ensuring that regardless of input style,
//! the semantic meaning, business rules, and AI-comprehensible metadata are fully preserved.
//!
//! ## Architecture
//!
//! This crate is structured around **Conceptual Cohesion**, with each module maintaining
//! a single, clear responsibility:
//!
//! - `detection/` - Syntax style detection and confidence scoring
//! - `core/` - Core parsing infrastructure and coordination
//! - `styles/` - Style-specific parsers (C-like, Python-like, Rust-like, Canonical)
//! - `normalization/` - Canonical form conversion and semantic preservation
//! - `integration/` - Bridge components for external system integration
//! - `validation/` - Semantic validation and consistency checking
//!
//! ## Key Design Principles
//!
//! ### 1. Conceptual Cohesion
//! Each module has ONE clear responsibility and maintains tight conceptual cohesion.
//! No module tries to do multiple unrelated things.
//!
//! ### 2. Semantic Preservation
//! All syntax transformations preserve 100% of semantic meaning. No information is lost
//! during style conversion or normalization.
//!
//! ### 3. AI-First Metadata
//! Every parsed structure includes comprehensive AI-comprehensible metadata for
//! business context, domain concepts, and architectural relationships.
//!
//! ### 4. Performance Focus
//! Parsing is optimized for both speed and memory efficiency, with incremental
//! parsing support and minimal allocations.
//!
//! ## Usage
//!
//! ```rust
//! use prism_syntax::{MultiSyntaxParser, ParserConfig};
//!
//! // Create parser with default configuration
//! let mut parser = MultiSyntaxParser::new();
//!
//! // Parse code in any supported syntax style
//! let source = "function calculate(x) { return x * 2; }";
//! let result = parser.parse(source)?;
//!
//! // Access the canonical form
//! let canonical = result.canonical_form;
//! println!("Parsed {} nodes", canonical.nodes.len());
//!
//! // Access AI metadata
//! for concept in &canonical.ai_metadata.domain_concepts {
//!     println!("Domain concept: {}", concept);
//! }
//! ```
//!
//! ## Integration Points
//!
//! This crate integrates with:
//! - `prism-lexer` for tokenization
//! - `prism-ast` for AST node definitions
//! - `prism-common` for shared utilities and spans
//! - `prism-compiler` for semantic analysis integration

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod core;
pub mod detection;
pub mod styles;
pub mod normalization;
pub mod integration;
pub mod validation;

// Re-export key types for convenience
pub use detection::{SyntaxDetector, SyntaxStyle, DetectionResult};
pub use normalization::{Normalizer, CanonicalForm};
pub use core::MultiSyntaxParser;

// Re-export error types
pub use core::parser::ParseError;
pub use detection::DetectionError;
pub use normalization::normalizer::NormalizationError;

/// Configuration for the multi-syntax parser
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Whether to enable syntax auto-detection
    pub auto_detect_syntax: bool,
    
    /// Confidence threshold for syntax detection (0.0 - 1.0)
    pub detection_confidence_threshold: f64,
    
    /// Whether to generate comprehensive AI metadata
    pub generate_ai_metadata: bool,
    
    /// Whether to preserve original formatting information
    pub preserve_formatting: bool,
    
    /// Whether to enable semantic validation
    pub enable_semantic_validation: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            auto_detect_syntax: true,
            detection_confidence_threshold: 0.7,
            generate_ai_metadata: true,
            preserve_formatting: false,
            enable_semantic_validation: true,
        }
    }
}

/// Result of parsing operation
#[derive(Debug)]
pub struct ParseResult {
    /// The canonical form of the parsed code
    pub canonical_form: CanonicalForm,
    
    /// Detected syntax style
    pub detected_style: SyntaxStyle,
    
    /// Detection confidence score
    pub detection_confidence: f64,
    
    /// Any warnings generated during parsing
    pub warnings: Vec<String>,
    
    /// Performance metrics
    pub metrics: ParseMetrics,
}

/// Performance metrics for parsing operations
#[derive(Debug, Clone)]
pub struct ParseMetrics {
    /// Total parsing time in milliseconds
    pub total_time_ms: u64,
    
    /// Time spent on syntax detection
    pub detection_time_ms: u64,
    
    /// Time spent on parsing
    pub parsing_time_ms: u64,
    
    /// Time spent on normalization
    pub normalization_time_ms: u64,
    
    /// Memory usage during parsing
    pub memory_usage_bytes: u64,
    
    /// Number of nodes parsed
    pub nodes_parsed: usize,
}

impl Default for ParseMetrics {
    fn default() -> Self {
        Self {
            total_time_ms: 0,
            detection_time_ms: 0,
            parsing_time_ms: 0,
            normalization_time_ms: 0,
            memory_usage_bytes: 0,
            nodes_parsed: 0,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::{
        detection::{SyntaxDetector, patterns::PatternMatcher, confidence::ConfidenceScorer},
        styles::canonical::{CanonicalParser, CanonicalSyntax, CanonicalFunction, CanonicalStatement, CanonicalExpression, CanonicalLiteral},
        normalization::{Normalizer, normalizer::ParsedSyntax},
    };
    
    #[test]
    fn test_complete_foundation_pipeline() {
        // Test our complete foundation: detection -> parsing -> normalization
        
        // 1. Test syntax detection
        let pattern_matcher = PatternMatcher::new();
        let confidence_scorer = ConfidenceScorer::new();
        let mut detector = SyntaxDetector::new(pattern_matcher, confidence_scorer);
        
        let test_code = "function calculate(x) { return x + 1; }";
        let detection_result = detector.detect_syntax(test_code);
        
        assert!(detection_result.is_ok(), "Syntax detection should work");
        let detection = detection_result.unwrap();
        assert!(detection.confidence > 0.0, "Should have confidence score");
        
        // 2. Test canonical parsing (simulate with direct structure creation)
        let test_function = CanonicalFunction {
            name: "calculate".to_string(),
            parameters: vec![],
            return_type: None,
            body: vec![
                CanonicalStatement::Return(Some(CanonicalExpression::Literal(
                    CanonicalLiteral::Integer(42)
                )))
            ],
            span: prism_common::span::Span::dummy(),
        };
        
        let canonical_syntax = CanonicalSyntax {
            modules: vec![],
            functions: vec![test_function],
            statements: vec![],
        };
        
        // 3. Test normalization
        let mut normalizer = Normalizer::new();
        let normalization_result = normalizer.normalize(ParsedSyntax::StyleCanonical(canonical_syntax));
        
        assert!(normalization_result.is_ok(), "Normalization should succeed: {:?}", normalization_result);
        let canonical_form = normalization_result.unwrap();
        
        // Verify the complete pipeline worked
        assert_eq!(canonical_form.nodes.len(), 1, "Should have one function node");
        assert!(!canonical_form.ai_metadata.domain_concepts.is_empty(), "Should have AI metadata");
        assert!(canonical_form.semantic_hash != 0, "Should have semantic hash");
        
        println!("✅ Complete foundation pipeline test passed!");
        println!("   - Syntax detection: {:?} (confidence: {:.2})", 
                detection.detected_style, detection.confidence);
        println!("   - Parsing: {} nodes", canonical_form.nodes.len());
        println!("   - AI concepts: {:?}", canonical_form.ai_metadata.domain_concepts);
        println!("   - Semantic hash: {}", canonical_form.semantic_hash);
    }
    
    #[test]
    fn test_foundation_performance() {
        // Test that our foundation is efficient
        let start_time = std::time::Instant::now();
        
        // Create a larger test case
        let mut functions = Vec::new();
        for i in 0..50 {
            functions.push(CanonicalFunction {
                name: format!("func_{}", i),
                parameters: vec![],
                return_type: Some("Result".to_string()),
                body: vec![
                    CanonicalStatement::Return(Some(CanonicalExpression::Literal(
                        CanonicalLiteral::Integer(i as i64)
                    )))
                ],
                span: prism_common::span::Span::dummy(),
            });
        }
        
        let canonical_syntax = CanonicalSyntax {
            modules: vec![],
            functions,
            statements: vec![],
        };
        
        let mut normalizer = Normalizer::new();
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(canonical_syntax));
        
        let duration = start_time.elapsed();
        
        assert!(result.is_ok(), "Performance test should succeed");
        assert!(duration.as_millis() < 50, "Should be fast: {}ms", duration.as_millis());
        
        let canonical_form = result.unwrap();
        assert_eq!(canonical_form.nodes.len(), 50, "Should process all functions");
        
        println!("✅ Foundation performance test passed!");
        println!("   - Processed 50 functions in {}ms", duration.as_millis());
        println!("   - Average: {:.2}ms per function", duration.as_millis() as f64 / 50.0);
    }
    
    #[test]
    fn test_foundation_semantic_preservation() {
        // Test that our foundation preserves semantic meaning
        let test_function = CanonicalFunction {
            name: "businessLogic".to_string(),
            parameters: vec![],
            return_type: Some("BusinessResult".to_string()),
            body: vec![
                CanonicalStatement::Assignment {
                    name: "result".to_string(),
                    value: CanonicalExpression::Call {
                        function: "processData".to_string(),
                        arguments: vec![
                            CanonicalExpression::Literal(CanonicalLiteral::String("input".to_string()))
                        ],
                    },
                },
                CanonicalStatement::Return(Some(CanonicalExpression::Identifier("result".to_string()))),
            ],
            span: prism_common::span::Span::dummy(),
        };
        
        let canonical_syntax = CanonicalSyntax {
            modules: vec![],
            functions: vec![test_function],
            statements: vec![],
        };
        
        let mut normalizer = Normalizer::new();
        let result = normalizer.normalize(ParsedSyntax::StyleCanonical(canonical_syntax));
        
        assert!(result.is_ok(), "Semantic preservation test should succeed");
        let canonical_form = result.unwrap();
        
        // Verify semantic information is preserved
        assert!(canonical_form.ai_metadata.domain_concepts.contains(&"function".to_string()));
        assert!(canonical_form.ai_metadata.relationships.contains(&"functions contain statements".to_string()));
        assert!(canonical_form.ai_metadata.business_context.is_some());
        assert!(canonical_form.ai_metadata.complexity_metrics.cyclomatic > 0.0);
        
        // Verify the function structure is preserved
        if let crate::normalization::canonical_form::CanonicalNode::Function { name, body, .. } = &canonical_form.nodes[0] {
            assert_eq!(name, "businessLogic", "Function name should be preserved");
            assert!(body.is_some(), "Function body should be preserved");
        } else {
            panic!("Expected function node");
        }
        
        println!("✅ Foundation semantic preservation test passed!");
        println!("   - Business context: {:?}", canonical_form.ai_metadata.business_context);
        println!("   - Complexity metrics: {:.2}", canonical_form.ai_metadata.complexity_metrics.cyclomatic);
    }
} 