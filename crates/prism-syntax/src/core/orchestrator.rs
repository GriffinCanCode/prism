//! Parsing Orchestrator - Factory-Based Coordination
//!
//! This module implements the parsing orchestrator that uses the factory system
//! to coordinate multi-syntax parsing operations. It maintains conceptual cohesion
//! around "parsing orchestration and coordination" while delegating component
//! creation to specialized factories.
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Orchestration separate from component creation
//! 2. **Factory Delegation**: Uses factories for all component creation
//! 3. **Configuration Management**: Centralized configuration through factories
//! 4. **Error Coordination**: Unified error handling across all components
//! 5. **Performance Optimization**: Efficient component reuse and caching

use crate::{
    core::factories::{ParserFactory, NormalizerFactory, ValidatorFactory, FactoryError},
    detection::{SyntaxDetector, SyntaxStyle, DetectionResult},
    normalization::{ParsedSyntax, CanonicalForm, NormalizationError},
    validation::{ValidationResult, ValidationError},
    styles::StyleParser,
    SyntaxError,
};
use prism_common::{SourceId, NodeId};
use prism_lexer::{Token, Lexer};
use prism_ast::{Program, AstNode, Item, ProgramMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Parsing orchestrator that coordinates multi-syntax parsing using factories
#[derive(Debug)]
pub struct ParsingOrchestrator {
    /// Syntax style detector
    detector: SyntaxDetector,
    /// Parser factory for creating style-specific parsers
    parser_factory: ParserFactory,
    /// Normalizer factory for creating style-specific normalizers
    normalizer_factory: NormalizerFactory,
    /// Validator factory for creating validators
    validator_factory: ValidatorFactory,
    /// Orchestrator configuration
    config: OrchestratorConfig,
    /// Component cache for performance optimization
    component_cache: ComponentCache,
}

/// Configuration for the parsing orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Whether to cache created components for reuse
    pub enable_component_caching: bool,
    /// Maximum number of components to cache
    pub max_cache_size: usize,
    /// Whether to enable parallel processing where possible
    pub enable_parallel_processing: bool,
    /// Default validation level for all parsing operations
    pub default_validation_level: crate::core::parser::ValidationLevel,
    /// Whether to generate AI metadata by default
    pub generate_ai_metadata: bool,
    /// Whether to preserve formatting information
    pub preserve_formatting: bool,
    /// Enable comprehensive error recovery
    pub enable_error_recovery: bool,
}

/// Component cache for performance optimization
#[derive(Debug)]
struct ComponentCache {
    /// Cached parsers by syntax style
    parsers: HashMap<SyntaxStyle, CachedParser>,
    /// Cached normalizers by syntax style
    normalizers: HashMap<SyntaxStyle, CachedNormalizer>,
    /// Cached validators
    validators: Vec<CachedValidator>,
    /// Cache statistics
    stats: CacheStats,
}

/// Cached parser with metadata
#[derive(Debug)]
struct CachedParser {
    /// The cached parser instance
    parser: Box<dyn StyleParser<Output = crate::core::factories::ParsedOutput, Error = crate::core::factories::ParseError>>,
    /// Number of times this parser has been used
    usage_count: usize,
    /// Last used timestamp
    last_used: std::time::Instant,
}

/// Cached normalizer with metadata
#[derive(Debug)]
struct CachedNormalizer {
    /// The cached normalizer instance
    normalizer: Box<dyn crate::normalization::traits::StyleNormalizer>,
    /// Number of times this normalizer has been used
    usage_count: usize,
    /// Last used timestamp
    last_used: std::time::Instant,
}

/// Cached validator with metadata
#[derive(Debug)]
struct CachedValidator {
    /// The cached validator instance
    validator: crate::validation::Validator,
    /// Number of times this validator has been used
    usage_count: usize,
    /// Last used timestamp
    last_used: std::time::Instant,
}

/// Cache performance statistics
#[derive(Debug, Default)]
struct CacheStats {
    /// Number of cache hits
    hits: usize,
    /// Number of cache misses
    misses: usize,
    /// Number of cache evictions
    evictions: usize,
}

/// Result of a complete parsing operation
#[derive(Debug)]
pub struct ParseResult {
    /// The parsed program
    pub program: Program,
    /// Detected syntax style
    pub detected_style: SyntaxStyle,
    /// Detection confidence score
    pub confidence: f64,
    /// Validation results
    pub validation: ValidationResult,
    /// Canonical form (if requested)
    pub canonical_form: Option<CanonicalForm>,
    /// Processing metrics
    pub metrics: ProcessingMetrics,
    /// Any warnings or diagnostics
    pub diagnostics: Vec<ParseDiagnostic>,
}

/// Processing performance metrics
#[derive(Debug, Default)]
pub struct ProcessingMetrics {
    /// Time spent on syntax detection
    pub detection_time: std::time::Duration,
    /// Time spent on parsing
    pub parsing_time: std::time::Duration,
    /// Time spent on normalization
    pub normalization_time: std::time::Duration,
    /// Time spent on validation
    pub validation_time: std::time::Duration,
    /// Total processing time
    pub total_time: std::time::Duration,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Memory usage during processing
    pub memory_usage: usize,
}

/// Parse diagnostic information
#[derive(Debug, Clone)]
pub struct ParseDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    /// Diagnostic message
    pub message: String,
    /// Source location
    pub location: Option<prism_common::span::Span>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// Error that prevents successful parsing
    Error,
    /// Warning about potential issues
    Warning,
    /// Informational message
    Info,
    /// Performance or optimization hint
    Hint,
}

/// Errors that can occur during orchestration
#[derive(Debug, Error)]
pub enum OrchestratorError {
    /// Factory operation failed
    #[error("Factory error: {0}")]
    Factory(#[from] FactoryError),
    
    /// Syntax detection failed
    #[error("Syntax detection failed: {reason}")]
    DetectionFailed { reason: String },
    
    /// Parsing operation failed
    #[error("Parsing failed for {style:?}: {reason}")]
    ParsingFailed { style: SyntaxStyle, reason: String },
    
    /// Normalization failed
    #[error("Normalization failed: {0}")]
    NormalizationFailed(#[from] NormalizationError),
    
    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(#[from] ValidationError),
    
    /// Configuration error
    #[error("Configuration error: {reason}")]
    ConfigurationError { reason: String },
    
    /// Component cache error
    #[error("Component cache error: {reason}")]
    CacheError { reason: String },
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_component_caching: true,
            max_cache_size: 10,
            enable_parallel_processing: false, // Conservative default
            default_validation_level: crate::core::parser::ValidationLevel::Standard,
            generate_ai_metadata: true,
            preserve_formatting: true,
            enable_error_recovery: true,
        }
    }
}

impl Default for ComponentCache {
    fn default() -> Self {
        Self {
            parsers: HashMap::new(),
            normalizers: HashMap::new(),
            validators: Vec::new(),
            stats: CacheStats::default(),
        }
    }
}

impl ParsingOrchestrator {
    /// Create a new parsing orchestrator with default configuration
    pub fn new() -> Self {
        Self::with_config(OrchestratorConfig::default())
    }
    
    /// Create a parsing orchestrator with custom configuration
    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            detector: SyntaxDetector::new(),
            parser_factory: ParserFactory::new(),
            normalizer_factory: NormalizerFactory::new(),
            validator_factory: ValidatorFactory::new(),
            config,
            component_cache: ComponentCache::default(),
        }
    }
    
    /// Create an orchestrator with custom factories
    pub fn with_factories(
        parser_factory: ParserFactory,
        normalizer_factory: NormalizerFactory,
        validator_factory: ValidatorFactory,
        config: OrchestratorConfig,
    ) -> Self {
        Self {
            detector: SyntaxDetector::new(),
            parser_factory,
            normalizer_factory,
            validator_factory,
            config,
            component_cache: ComponentCache::default(),
        }
    }
    
    /// Parse source code with full orchestration
    pub fn parse(&mut self, source: &str, source_id: SourceId) -> Result<ParseResult, OrchestratorError> {
        let start_time = std::time::Instant::now();
        let mut metrics = ProcessingMetrics::default();
        let mut diagnostics = Vec::new();
        
        // Step 1: Detect syntax style
        let detection_start = std::time::Instant::now();
        let detection_result = self.detector.detect_syntax(source)
            .map_err(|e| OrchestratorError::DetectionFailed { 
                reason: e.to_string() 
            })?;
        metrics.detection_time = detection_start.elapsed();
        
        // Step 2: Get or create appropriate parser
        let parsing_start = std::time::Instant::now();
        let mut parser = self.get_or_create_parser(detection_result.detected_style)?;
        
        // Step 3: Tokenize source
        let tokens = self.tokenize_source(source, source_id)?;
        metrics.tokens_processed = tokens.len();
        
        // Step 4: Parse with style-specific parser
        let parsed_syntax = parser.parse(tokens)
            .map_err(|e| OrchestratorError::ParsingFailed { 
                style: detection_result.detected_style,
                reason: e.to_string() 
            })?;
        metrics.parsing_time = parsing_start.elapsed();
        
        // Step 5: Convert to Program (simplified for now)
        let program = self.convert_to_program(parsed_syntax, source_id, &detection_result)?;
        
        // Step 6: Normalize to canonical form (if requested)
        let canonical_form = if self.config.generate_ai_metadata || self.config.preserve_formatting {
            let normalization_start = std::time::Instant::now();
            let normalizer = self.get_or_create_normalizer(detection_result.detected_style)?;
            
            // Create ParsedSyntax enum based on detected style
            let parsed_syntax_enum = self.create_parsed_syntax_enum(detection_result.detected_style, parsed_syntax)?;
            let canonical = normalizer.normalize(parsed_syntax_enum)?;
            metrics.normalization_time = normalization_start.elapsed();
            Some(canonical)
        } else {
            None
        };
        
        // Step 7: Validate the result
        let validation_start = std::time::Instant::now();
        let validator = self.get_or_create_validator()?;
        let validation = if let Some(ref canonical) = canonical_form {
            validator.validate_canonical(canonical)?
        } else {
            validator.validate_program(&program)?
        };
        metrics.validation_time = validation_start.elapsed();
        
        // Step 8: Calculate total metrics
        metrics.total_time = start_time.elapsed();
        
        Ok(ParseResult {
            program,
            detected_style: detection_result.detected_style,
            confidence: detection_result.confidence,
            validation,
            canonical_form,
            metrics,
            diagnostics,
        })
    }
    
    /// Get or create a parser for the specified style
    fn get_or_create_parser(&mut self, style: SyntaxStyle) -> Result<&mut Box<dyn StyleParser<Output = crate::core::factories::ParsedOutput, Error = crate::core::factories::ParseError>>, OrchestratorError> {
        if self.config.enable_component_caching {
            if let Some(cached_parser) = self.component_cache.parsers.get_mut(&style) {
                cached_parser.usage_count += 1;
                cached_parser.last_used = std::time::Instant::now();
                self.component_cache.stats.hits += 1;
                return Ok(&mut cached_parser.parser);
            }
        }
        
        // Cache miss - create new parser
        self.component_cache.stats.misses += 1;
        let parser = self.parser_factory.create_parser(style)?;
        
        if self.config.enable_component_caching {
            let cached_parser = CachedParser {
                parser,
                usage_count: 1,
                last_used: std::time::Instant::now(),
            };
            self.component_cache.parsers.insert(style, cached_parser);
            Ok(&mut self.component_cache.parsers.get_mut(&style).unwrap().parser)
        } else {
            // If caching is disabled, we need a different approach
            // For now, we'll create a temporary parser
            // In a real implementation, you might want to store it temporarily
            Err(OrchestratorError::CacheError { 
                reason: "Non-cached parser creation not yet implemented".to_string() 
            })
        }
    }
    
    /// Get or create a normalizer for the specified style
    fn get_or_create_normalizer(&mut self, style: SyntaxStyle) -> Result<&mut Box<dyn crate::normalization::traits::StyleNormalizer>, OrchestratorError> {
        if self.config.enable_component_caching {
            if let Some(cached_normalizer) = self.component_cache.normalizers.get_mut(&style) {
                cached_normalizer.usage_count += 1;
                cached_normalizer.last_used = std::time::Instant::now();
                self.component_cache.stats.hits += 1;
                return Ok(&mut cached_normalizer.normalizer);
            }
        }
        
        // Cache miss - create new normalizer
        self.component_cache.stats.misses += 1;
        let normalizer = self.normalizer_factory.create_normalizer(style)?;
        
        if self.config.enable_component_caching {
            let cached_normalizer = CachedNormalizer {
                normalizer,
                usage_count: 1,
                last_used: std::time::Instant::now(),
            };
            self.component_cache.normalizers.insert(style, cached_normalizer);
            Ok(&mut self.component_cache.normalizers.get_mut(&style).unwrap().normalizer)
        } else {
            Err(OrchestratorError::CacheError { 
                reason: "Non-cached normalizer creation not yet implemented".to_string() 
            })
        }
    }
    
    /// Get or create a validator
    fn get_or_create_validator(&mut self) -> Result<&mut crate::validation::Validator, OrchestratorError> {
        if self.config.enable_component_caching {
            if let Some(cached_validator) = self.component_cache.validators.first_mut() {
                cached_validator.usage_count += 1;
                cached_validator.last_used = std::time::Instant::now();
                self.component_cache.stats.hits += 1;
                return Ok(&mut cached_validator.validator);
            }
        }
        
        // Cache miss - create new validator
        self.component_cache.stats.misses += 1;
        let validator = self.validator_factory.create_validator()?;
        
        if self.config.enable_component_caching {
            let cached_validator = CachedValidator {
                validator,
                usage_count: 1,
                last_used: std::time::Instant::now(),
            };
            self.component_cache.validators.push(cached_validator);
            Ok(&mut self.component_cache.validators.last_mut().unwrap().validator)
        } else {
            Err(OrchestratorError::CacheError { 
                reason: "Non-cached validator creation not yet implemented".to_string() 
            })
        }
    }
    
    /// Tokenize source code
    fn tokenize_source(&self, source: &str, source_id: SourceId) -> Result<Vec<Token>, OrchestratorError> {
        // Create a dummy symbol table for lexing
        let mut symbol_table = prism_common::symbol::SymbolTable::new();
        let lexer_config = prism_lexer::LexerConfig::default();
        let lexer = Lexer::new(source, source_id, &mut symbol_table, lexer_config);
        
        let lex_result = lexer.tokenize();
        Ok(lex_result.tokens)
    }
    
    /// Convert parsed syntax to Program (simplified implementation)
    fn convert_to_program(
        &self, 
        _parsed_syntax: crate::core::factories::ParsedOutput, 
        source_id: SourceId, 
        detection_result: &DetectionResult
    ) -> Result<Program, OrchestratorError> {
        // Simplified implementation - in reality this would convert the parsed syntax
        // to a proper Program AST based on the syntax style
        Ok(Program {
            items: Vec::new(),
            metadata: ProgramMetadata {
                primary_capability: Some(format!("Multi-syntax parsing ({})", detection_result.detected_style.to_string())),
                capabilities: vec![
                    format!("syntax_parsing_{}", detection_result.detected_style.to_string().to_lowercase()),
                    "multi_syntax_coordination".to_string(),
                    "semantic_preservation".to_string(),
                ],
                dependencies: Vec::new(),
                security_implications: Vec::new(),
                performance_notes: vec![
                    format!("Parsing confidence: {:.2}%", detection_result.confidence * 100.0),
                ],
                ai_insights: vec![
                    format!("Successfully parsed {} syntax with {:.1}% confidence", 
                        detection_result.detected_style.to_string(), detection_result.confidence * 100.0),
                ],
            },
        })
    }
    
    /// Create ParsedSyntax enum from parsed output
    fn create_parsed_syntax_enum(
        &self, 
        style: SyntaxStyle, 
        _parsed_output: crate::core::factories::ParsedOutput
    ) -> Result<ParsedSyntax, OrchestratorError> {
        // Simplified implementation - in reality this would convert the parsed output
        // to the appropriate ParsedSyntax variant based on the style
        match style {
            SyntaxStyle::CLike => Ok(ParsedSyntax::CLike(Default::default())),
            SyntaxStyle::PythonLike => Ok(ParsedSyntax::PythonLike(Default::default())),
            SyntaxStyle::RustLike => Ok(ParsedSyntax::RustLike(Default::default())),
            SyntaxStyle::Canonical => Ok(ParsedSyntax::Canonical(Vec::new())),
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        &self.component_cache.stats
    }
    
    /// Clear component cache
    pub fn clear_cache(&mut self) {
        self.component_cache.parsers.clear();
        self.component_cache.normalizers.clear();
        self.component_cache.validators.clear();
        self.component_cache.stats = CacheStats::default();
    }
    
    /// Update orchestrator configuration
    pub fn update_config(&mut self, config: OrchestratorConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }
}

impl Default for ParsingOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SyntaxStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyntaxStyle::CLike => write!(f, "C-like"),
            SyntaxStyle::PythonLike => write!(f, "Python-like"),
            SyntaxStyle::RustLike => write!(f, "Rust-like"),
            SyntaxStyle::Canonical => write!(f, "Canonical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = ParsingOrchestrator::new();
        assert!(orchestrator.config.enable_component_caching);
        assert_eq!(orchestrator.config.max_cache_size, 10);
    }
    
    #[test]
    fn test_orchestrator_with_custom_config() {
        let config = OrchestratorConfig {
            enable_component_caching: false,
            max_cache_size: 5,
            ..Default::default()
        };
        
        let orchestrator = ParsingOrchestrator::with_config(config);
        assert!(!orchestrator.config.enable_component_caching);
        assert_eq!(orchestrator.config.max_cache_size, 5);
    }
    
    #[test]
    fn test_cache_stats() {
        let orchestrator = ParsingOrchestrator::new();
        let stats = orchestrator.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
    }
} 