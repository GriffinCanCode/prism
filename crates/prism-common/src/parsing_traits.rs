//! Parsing Trait Interfaces
//!
//! This module defines trait interfaces for parsing that can be used across
//! different crates without creating circular dependencies. These traits
//! provide a clean abstraction layer between parsing implementations and
//! their consumers.
//!
//! ## Design Principles
//!
//! 1. **Dependency Inversion**: Higher-level modules depend on abstractions, not concretions
//! 2. **Interface Segregation**: Small, focused traits for specific responsibilities
//! 3. **No Circular Dependencies**: Traits live in prism-common, avoiding cycles
//! 4. **Extensibility**: Easy to add new parsing implementations

use crate::{Result as PrismResult, PrismError};
use crate::span::SourceId;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Trait for parsing source code into AST programs
/// 
/// This trait abstracts over different parsing implementations,
/// allowing consumers to work with parsers without knowing the
/// concrete implementation.
pub trait ProgramParser {
    /// The AST program type this parser produces
    type Program;
    
    /// The error type this parser can produce
    type Error: Into<PrismError>;
    
    /// Parse source code into a program AST
    fn parse_source(&mut self, source: &str, source_id: SourceId) -> Result<Self::Program, Self::Error>;
    
    /// Parse source code with custom configuration
    fn parse_source_with_config(
        &mut self, 
        source: &str, 
        source_id: SourceId,
        config: &ParsingConfig
    ) -> Result<Self::Program, Self::Error>;
}

/// Trait for enhanced parsing with analysis capabilities
/// 
/// This trait extends basic parsing with additional analysis
/// features like documentation validation, cohesion analysis,
/// and AI metadata generation.
pub trait EnhancedParser: ProgramParser {
    /// Enhanced parsing result with analysis metadata
    type EnhancedResult;
    
    /// Parse with full analysis capabilities
    fn parse_with_analysis(
        &mut self,
        source: &str,
        source_id: SourceId,
    ) -> Result<Self::EnhancedResult, Self::Error>;
}

/// Trait for syntax-aware parsing
/// 
/// This trait provides syntax style detection and
/// multi-syntax parsing capabilities.
pub trait SyntaxAwareParser: ProgramParser {
    /// Syntax style enumeration
    type SyntaxStyle;
    
    /// Detect the syntax style of source code
    fn detect_syntax(&self, source: &str) -> Result<(Self::SyntaxStyle, f64), Self::Error>;
    
    /// Parse with explicit syntax style preference
    fn parse_with_syntax(
        &mut self,
        source: &str,
        source_id: SourceId,
        preferred_syntax: Option<Self::SyntaxStyle>,
    ) -> Result<Self::Program, Self::Error>;
}

/// Configuration for parsing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingConfig {
    /// Enable AI context extraction
    pub extract_ai_context: bool,
    
    /// Enable semantic constraint validation
    pub validate_constraints: bool,
    
    /// Enable aggressive error recovery
    pub aggressive_recovery: bool,
    
    /// Maximum number of errors before stopping
    pub max_errors: usize,
    
    /// Enable documentation validation
    pub validate_documentation: bool,
    
    /// Enable cohesion analysis
    pub analyze_cohesion: bool,
    
    /// Enable performance profiling
    pub enable_profiling: bool,
}

impl Default for ParsingConfig {
    fn default() -> Self {
        Self {
            extract_ai_context: true,
            validate_constraints: true,
            aggressive_recovery: true,
            max_errors: 100,
            validate_documentation: true,
            analyze_cohesion: true,
            enable_profiling: false,
        }
    }
}

/// Parsing performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingMetrics {
    /// Total parsing time in milliseconds
    pub total_time_ms: u64,
    
    /// Time breakdown by parsing phase
    pub phase_times: HashMap<String, u64>,
    
    /// Number of tokens processed
    pub tokens_processed: usize,
    
    /// Number of AST nodes created
    pub nodes_created: usize,
    
    /// Cache hit rate (if applicable)
    pub cache_hit_rate: Option<f64>,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<u64>,
}

/// Diagnostic information from parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    
    /// Diagnostic message
    pub message: String,
    
    /// Source location (if applicable)
    pub location: Option<crate::span::Span>,
    
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
    
    /// Diagnostic code for categorization
    pub code: Option<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Factory trait for creating parsers
/// 
/// This trait allows for dependency injection of parser implementations
/// without requiring knowledge of concrete types.
pub trait ParserFactory {
    /// The parser type this factory creates
    type Parser: ProgramParser;
    
    /// Create a new parser with default configuration
    fn create_parser(&self) -> Self::Parser;
    
    /// Create a parser with custom configuration
    fn create_parser_with_config(&self, config: &ParsingConfig) -> Self::Parser;
}

/// Trait for PIR construction from parsed programs
/// 
/// This trait abstracts PIR construction, allowing different
/// implementations without tight coupling.
pub trait PIRConstructor {
    /// The program type this constructor accepts
    type Program;
    
    /// The PIR type this constructor produces
    type PIR;
    
    /// The error type for PIR construction
    type Error: Into<PrismError>;
    
    /// Construct PIR from a parsed program
    fn construct_pir(&mut self, program: Self::Program) -> Result<Self::PIR, Self::Error>;
    
    /// Construct PIR with custom configuration
    fn construct_pir_with_config(
        &mut self,
        program: Self::Program,
        config: &PIRConstructionConfig,
    ) -> Result<Self::PIR, Self::Error>;
}

/// Configuration for PIR construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRConstructionConfig {
    /// Enable semantic validation during construction
    pub enable_validation: bool,
    
    /// Enable AI metadata extraction
    pub enable_ai_metadata: bool,
    
    /// Enable cohesion analysis
    pub enable_cohesion_analysis: bool,
    
    /// Enable effect graph construction
    pub enable_effect_graph: bool,
    
    /// Enable business context extraction
    pub enable_business_context: bool,
    
    /// Maximum construction depth for recursive structures
    pub max_construction_depth: usize,
}

impl Default for PIRConstructionConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_ai_metadata: true,
            enable_cohesion_analysis: true,
            enable_effect_graph: true,
            enable_business_context: true,
            max_construction_depth: 1000,
        }
    }
} 