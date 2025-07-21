//! Main parser coordinator for multi-syntax parsing.
//!
//! This module implements the `Parser` struct which serves as the central coordinator
//! for all parsing operations. It embodies conceptual cohesion by focusing solely on
//! the responsibility of "coordinating multi-syntax parsing with semantic preservation".
//!
//! ## PLT-001 Integration
//!
//! This parser implements the core coordination layer specified in PLT-001, providing:
//! - Multi-syntax parsing (C-like, Python-like, Rust-like, Canonical)
//! - Semantic preservation across syntax styles
//! - Integration with PLD-001 (Semantic Types), PLD-002 (Modules), PLD-003 (Effects)
//! - PSG-003 documentation validation
//! - AI-readable metadata generation

use crate::{
    detection::{SyntaxDetector, SyntaxStyle, DetectionResult},
    styles::{StyleParser, CLikeParser, PythonLikeParser, RustLikeParser, CanonicalParser},
    normalization::{Normalizer, CanonicalForm},
    validation::{Validator, ValidationResult},
};
use prism_common::{SourceId, NodeId};
use prism_lexer::{Token, Lexer};
use prism_ast::{Program, AstNode, Item, ProgramMetadata};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use thiserror::Error;

/// Main parser coordinator that handles multi-syntax parsing.
/// 
/// The Parser is responsible for:
/// 1. Detecting the input syntax style with confidence scoring
/// 2. Delegating to the appropriate style-specific parser
/// 3. Normalizing the result to canonical form with semantic preservation
/// 4. Validating the parsed output against Prism standards
/// 5. Generating AI-comprehensible metadata throughout the process
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing exclusively on parsing
/// coordination. It does not implement parsing logic directly, but orchestrates
/// the interaction between specialized components, each with their own clear
/// responsibilities.
/// 
/// # PLT-001 Implementation
/// 
/// This parser implements the multi-syntax coordination layer of PLT-001:
/// - Syntax detection and delegation
/// - Semantic preservation across styles
/// - Integration with all language subsystems
/// - AI metadata generation
/// - Error recovery with semantic awareness
/// 
/// # Example
/// 
/// ```rust
/// use prism_syntax::Parser;
/// use prism_common::SourceId;
/// 
/// let source = r#"
///     @responsibility "Manages user authentication"
///     @module "UserAuth"
///     @description "Secure user authentication module"
///     @author "Security Team"
///     
///     module UserAuth {
///         section interface {
///             @responsibility "Authenticates users securely"
///             function authenticate(user: User) -> Result<Session, Error> {
///                 return processAuth(user)
///             }
///         }
///     }
/// "#;
/// 
/// let mut parser = Parser::new();
/// let result = parser.parse_source(source, SourceId::new(1)).unwrap();
/// 
/// // Verify semantic preservation across syntax styles
/// assert!(result.items.len() > 0);
/// ```
#[derive(Debug)]
pub struct Parser {
    /// Intelligent syntax style detector with confidence scoring
    detector: SyntaxDetector,
    
    /// C-like syntax parser (C/C++/Java/JavaScript style)
    c_like_parser: CLikeParser,
    
    /// Python-like syntax parser (Python/CoffeeScript style) 
    python_like_parser: PythonLikeParser,
    
    /// Rust-like syntax parser (Rust/Go style)
    rust_like_parser: RustLikeParser,
    
    /// Canonical syntax parser (Prism canonical format)
    canonical_parser: CanonicalParser,
    
    /// Semantic normalizer for canonical form conversion
    normalizer: Normalizer,
    
    /// Validation engine for Prism standards compliance
    validator: Validator,
    
    /// Current parsing context and configuration
    context: ParseContext,
}

/// Parsing context that maintains state and configuration across operations
#[derive(Debug, Clone)]
pub struct ParseContext {
    /// Source identifier for tracking and diagnostics
    pub source_id: SourceId,
    
    /// Optional file path for enhanced error reporting
    pub file_path: Option<String>,
    
    /// Project-level configuration affecting parsing behavior
    pub project_config: Option<ProjectConfig>,
    
    /// Explicit style preference (overrides detection when specified)
    pub style_preference: Option<SyntaxStyle>,
    
    /// Validation strictness level
    pub validation_level: ValidationLevel,
    
    /// Whether to generate AI metadata during parsing
    pub generate_ai_metadata: bool,
    
    /// Whether to preserve original formatting information
    pub preserve_formatting: bool,
    
    /// Enable documentation validation (PSG-003)
    pub validate_documentation: bool,
    
    /// Enable cohesion analysis (PLD-002)
    pub analyze_cohesion: bool,
    
    /// Enable semantic type analysis (PLD-001)
    pub analyze_semantic_types: bool,
    
    /// Enable effect system analysis (PLD-003)
    pub analyze_effects: bool,
}

/// Project configuration affecting parsing behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Project name for context
    pub name: String,
    
    /// Project version
    pub version: String,
    
    /// Preferred syntax style for the project
    pub preferred_syntax: Option<SyntaxStyle>,
    
    /// Documentation requirements level
    pub documentation_level: DocumentationLevel,
    
    /// Cohesion analysis settings
    pub cohesion_settings: CohesionSettings,
    
    /// AI metadata export settings
    pub ai_export_settings: AIExportSettings,
}

/// Documentation requirement levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentationLevel {
    /// Minimal documentation required
    Minimal,
    /// Standard documentation (PSG-003 compliant)
    Standard,
    /// Comprehensive documentation with examples
    Comprehensive,
    /// Full documentation with AI context
    Full,
}

/// Cohesion analysis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionSettings {
    /// Enable cohesion analysis
    pub enabled: bool,
    
    /// Minimum acceptable cohesion score (0-100)
    pub minimum_score: f64,
    
    /// Generate improvement suggestions
    pub generate_suggestions: bool,
    
    /// Include AI insights in analysis
    pub include_ai_insights: bool,
}

/// AI metadata export settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIExportSettings {
    /// Enable AI metadata generation
    pub enabled: bool,
    
    /// Export format preferences
    pub formats: Vec<AIExportFormat>,
    
    /// Include business context in exports
    pub include_business_context: bool,
    
    /// Include architectural patterns
    pub include_architectural_patterns: bool,
}

/// AI export format options
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AIExportFormat {
    /// JSON format for general AI consumption
    JSON,
    /// TOML format for human-readable exports
    TOML,
    /// Binary format for performance-critical scenarios
    Binary,
    /// Custom format specification
    Custom(String),
}

/// Validation strictness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Lenient validation (warnings only)
    Lenient,
    /// Standard validation (PSG compliance)
    Standard,
    /// Strict validation (all rules enforced)
    Strict,
    /// Pedantic validation (maximum strictness)
    Pedantic,
}

/// Parse result with comprehensive metadata
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// Successfully parsed program
    pub program: Program,
    
    /// Detected syntax style
    pub detected_style: SyntaxStyle,
    
    /// Detection confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Validation results
    pub validation: ValidationResult,
    
    /// Canonical form representation
    pub canonical_form: Option<CanonicalForm>,
    
    /// Generated AI metadata
    pub ai_metadata: Option<AIMetadata>,
    
    /// Cohesion analysis results (if enabled)
    pub cohesion_analysis: Option<CohesionAnalysis>,
    
    /// Documentation validation results (if enabled)
    pub documentation_validation: Option<DocumentationValidation>,
    
    /// Parse warnings and suggestions
    pub diagnostics: Vec<ParseDiagnostic>,
}

/// AI metadata generated during parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Business context extracted from code
    pub business_context: BusinessContext,
    
    /// Architectural patterns detected
    pub architectural_patterns: Vec<ArchitecturalPattern>,
    
    /// Semantic relationships between components
    pub semantic_relationships: Vec<SemanticRelationship>,
    
    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,
    
    /// AI comprehension hints
    pub comprehension_hints: Vec<String>,
}

/// Business context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Primary business capability
    pub capability: Option<String>,
    
    /// Business domain
    pub domain: Option<String>,
    
    /// Business rules identified
    pub business_rules: Vec<String>,
    
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
}

/// Architectural pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Pattern description
    pub description: String,
    
    /// Components involved in the pattern
    pub components: Vec<String>,
}

/// Semantic relationship between code elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelationship {
    /// Source element
    pub source: String,
    
    /// Target element
    pub target: String,
    
    /// Relationship type
    pub relationship_type: RelationshipType,
    
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
}

/// Types of semantic relationships
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Uses relationship
    Uses,
    /// Depends on relationship
    DependsOn,
    /// Implements relationship
    Implements,
    /// Extends relationship
    Extends,
    /// Composes relationship
    Composes,
    /// Aggregates relationship
    Aggregates,
}

/// Complexity metrics for AI analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: u32,
    
    /// Cognitive load score
    pub cognitive_load: f64,
    
    /// Nesting depth
    pub nesting_depth: u32,
    
    /// Branching factor
    pub branching_factor: u32,
    
    /// Lines of code
    pub lines_of_code: u32,
    
    /// Maintainability index
    pub maintainability_index: f64,
}

/// Cohesion analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysis {
    /// Overall cohesion score (0-100)
    pub overall_score: f64,
    
    /// Individual metric scores
    pub metrics: CohesionMetrics,
    
    /// Improvement suggestions
    pub suggestions: Vec<CohesionSuggestion>,
    
    /// Detected violations
    pub violations: Vec<CohesionViolation>,
}

/// Cohesion metrics breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionMetrics {
    /// Type cohesion score
    pub type_cohesion: f64,
    
    /// Data flow cohesion score
    pub data_flow_cohesion: f64,
    
    /// Semantic cohesion score
    pub semantic_cohesion: f64,
    
    /// Dependency cohesion score
    pub dependency_cohesion: f64,
    
    /// Business cohesion score
    pub business_cohesion: f64,
}

/// Cohesion improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionSuggestion {
    /// Suggestion type
    pub suggestion_type: CohesionSuggestionType,
    
    /// Description of the suggestion
    pub description: String,
    
    /// Priority level
    pub priority: SuggestionPriority,
    
    /// Estimated impact
    pub estimated_impact: f64,
}

/// Types of cohesion suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CohesionSuggestionType {
    /// Split module into multiple focused modules
    SplitModule,
    /// Merge related functionality
    MergeComponents,
    /// Extract common functionality
    ExtractCommon,
    /// Clarify responsibilities
    ClarifyResponsibilities,
    /// Improve naming consistency
    ImproveNaming,
    /// Reorganize sections
    ReorganizeSections,
}

/// Priority levels for suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionPriority {
    /// Critical issue affecting maintainability
    Critical,
    /// Important improvement opportunity
    Important,
    /// Minor enhancement
    Minor,
    /// Optional optimization
    Optional,
}

/// Cohesion violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionViolation {
    /// Violation type
    pub violation_type: String,
    
    /// Severity level
    pub severity: ViolationSeverity,
    
    /// Description of the violation
    pub description: String,
    
    /// Location of the violation
    pub location: prism_common::span::Span,
    
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Error level violation
    Error,
    /// Warning level violation
    Warning,
    /// Information level violation
    Info,
}

/// Documentation validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationValidation {
    /// Overall compliance with PSG-003
    pub psg003_compliance: bool,
    
    /// Missing required annotations
    pub missing_annotations: Vec<MissingAnnotation>,
    
    /// Invalid documentation format
    pub format_issues: Vec<FormatIssue>,
    
    /// JSDoc compatibility issues
    pub jsdoc_issues: Vec<JSDocIssue>,
    
    /// AI context completeness score
    pub ai_context_score: f64,
}

/// Missing annotation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingAnnotation {
    /// Type of missing annotation
    pub annotation_type: String,
    
    /// Location where annotation is missing
    pub location: prism_common::span::Span,
    
    /// Severity of the missing annotation
    pub severity: ViolationSeverity,
    
    /// Suggested annotation content
    pub suggested_content: Option<String>,
}

/// Documentation format issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatIssue {
    /// Issue type
    pub issue_type: String,
    
    /// Issue description
    pub description: String,
    
    /// Location of the issue
    pub location: prism_common::span::Span,
    
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// JSDoc compatibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocIssue {
    /// Issue description
    pub description: String,
    
    /// Location of the issue
    pub location: prism_common::span::Span,
    
    /// JSDoc standard reference
    pub standard_reference: Option<String>,
}

/// Parse diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    
    /// Diagnostic message
    pub message: String,
    
    /// Location of the diagnostic
    pub location: prism_common::span::Span,
    
    /// Diagnostic code for categorization
    pub code: Option<String>,
    
    /// Suggested fixes
    pub suggestions: Vec<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticLevel {
    /// Error level diagnostic
    Error,
    /// Warning level diagnostic
    Warning,
    /// Information level diagnostic
    Info,
    /// Hint level diagnostic
    Hint,
}

/// Parse error types
#[derive(Debug, Error)]
pub enum ParseError {
    /// Syntax detection failed
    #[error("Syntax detection failed: {reason}")]
    DetectionFailed { reason: String },
    
    /// Parsing failed for specific syntax
    #[error("Parsing failed for {syntax}: {reason}")]
    ParsingFailed { syntax: SyntaxStyle, reason: String },
    
    /// Normalization to canonical form failed
    #[error("Normalization failed: {reason}")]
    NormalizationFailed { reason: String },
    
    /// Validation failed
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },
    
    /// Documentation validation failed
    #[error("Documentation validation failed: {reason}")]
    DocumentationValidationFailed { reason: String },
    
    /// Cohesion analysis failed
    #[error("Cohesion analysis failed: {reason}")]
    CohesionAnalysisFailed { reason: String },
    
    /// AI metadata generation failed
    #[error("AI metadata generation failed: {reason}")]
    AIMetadataFailed { reason: String },
}

impl Parser {
    /// Create a new parser with default configuration
    pub fn new() -> Self {
        Self {
            detector: SyntaxDetector::new(),
            c_like_parser: CLikeParser::new(),
            python_like_parser: PythonLikeParser::new(),
            rust_like_parser: RustLikeParser::new(),
            canonical_parser: CanonicalParser::new(),
            normalizer: Normalizer::new(),
            validator: Validator::new(),
            context: ParseContext::default(),
        }
    }
    
    /// Create a new parser with explicit syntax style preference
    pub fn with_style(style: SyntaxStyle) -> Self {
        let mut parser = Self::new();
        parser.context.style_preference = Some(style);
        parser
    }
    
    /// Create a new parser with custom context
    pub fn with_context(context: ParseContext) -> Self {
        let mut parser = Self::new();
        parser.context = context;
        parser
    }
    
    /// Parse source code into a complete program AST
    pub fn parse_source(&mut self, source: &str, source_id: SourceId) -> Result<Program, ParseError> {
        self.context.source_id = source_id;
        
        // Step 1: Detect syntax style (unless explicitly specified)
        let detection_result = if let Some(preferred_style) = self.context.style_preference {
            DetectionResult {
                detected_style: preferred_style,
                confidence: 1.0,
                alternative_styles: Vec::new(),
                detection_metadata: HashMap::new(),
            }
        } else {
            self.detector.detect_syntax(source)
                .map_err(|e| ParseError::DetectionFailed { reason: e.to_string() })?
        };
        
        // Step 2: Parse using the appropriate style parser
        let program = match detection_result.detected_style {
            SyntaxStyle::CLike => {
                self.c_like_parser.parse_source(source, source_id)
                    .map_err(|e| ParseError::ParsingFailed { 
                        syntax: SyntaxStyle::CLike, 
                        reason: e.to_string() 
                    })?
            }
            SyntaxStyle::PythonLike => {
                self.python_like_parser.parse_source(source, source_id)
                    .map_err(|e| ParseError::ParsingFailed { 
                        syntax: SyntaxStyle::PythonLike, 
                        reason: e.to_string() 
                    })?
            }
            SyntaxStyle::RustLike => {
                self.rust_like_parser.parse_source(source, source_id)
                    .map_err(|e| ParseError::ParsingFailed { 
                        syntax: SyntaxStyle::RustLike, 
                        reason: e.to_string() 
                    })?
            }
            SyntaxStyle::Canonical => {
                self.canonical_parser.parse_source(source, source_id)
                    .map_err(|e| ParseError::ParsingFailed { 
                        syntax: SyntaxStyle::Canonical, 
                        reason: e.to_string() 
                    })?
            }
        };
        
        Ok(program)
    }
    
    /// Parse source code with full analysis and validation
    pub fn parse_with_full_analysis(&mut self, source: &str, source_id: SourceId) -> Result<ParseResult, ParseError> {
        // Enable all analysis features
        self.context.generate_ai_metadata = true;
        self.context.validate_documentation = true;
        self.context.analyze_cohesion = true;
        self.context.analyze_semantic_types = true;
        self.context.analyze_effects = true;
        
        // Parse the program
        let program = self.parse_source(source, source_id)?;
        
        // Perform additional analysis
        let mut result = ParseResult {
            program,
            detected_style: SyntaxStyle::Canonical, // Will be updated
            confidence: 1.0,
            validation: ValidationResult::default(),
            canonical_form: None,
            ai_metadata: None,
            cohesion_analysis: None,
            documentation_validation: None,
            diagnostics: Vec::new(),
        };
        
        // TODO: Implement full analysis pipeline
        // This would include:
        // - Normalization to canonical form
        // - Validation against standards
        // - AI metadata generation
        // - Cohesion analysis
        // - Documentation validation
        
        Ok(result)
    }
    
    /// Set parsing context
    pub fn set_context(&mut self, context: ParseContext) {
        self.context = context;
    }
    
    /// Get current parsing context
    pub fn context(&self) -> &ParseContext {
        &self.context
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ParseContext {
    fn default() -> Self {
        Self {
            source_id: SourceId::new(0),
            file_path: None,
            project_config: None,
            style_preference: None,
            validation_level: ValidationLevel::Standard,
            generate_ai_metadata: true,
            preserve_formatting: false,
            validate_documentation: true,
            analyze_cohesion: true,
            analyze_semantic_types: true,
            analyze_effects: true,
        }
    }
}

impl Default for CohesionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            minimum_score: 70.0,
            generate_suggestions: true,
            include_ai_insights: true,
        }
    }
}

impl Default for AIExportSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            formats: vec![AIExportFormat::JSON, AIExportFormat::TOML],
            include_business_context: true,
            include_architectural_patterns: true,
        }
    }
}

/// Parse result type alias
pub type ParseResult = Result<Program, ParseError>; 