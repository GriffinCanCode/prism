//! Main parser coordinator for multi-syntax parsing.
//!
//! This module implements the `Parser` struct which serves as the central coordinator
//! for all parsing operations. It embodies conceptual cohesion by focusing solely on
//! the responsibility of "coordinating multi-syntax parsing with semantic preservation".

use crate::{
    detection::{SyntaxDetector, SyntaxStyle, DetectionResult},
    styles::{StyleParser, CLikeParser, PythonLikeParser, RustLikeParser, CanonicalParser},
    normalization::{Normalizer, CanonicalForm},
    validation::{Validator, ValidationResult},
    SyntaxError,
};
use prism_common::{SourceId, NodeId};
use prism_lexer::{Token, Lexer, SyntaxStyle as LexerSyntaxStyle};
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
/// # Example
/// 
/// ```rust
/// use prism_syntax::Parser;
/// 
/// let source = r#"
///     module UserAuth {
///         function authenticate(user: User) -> Result<Session, Error> {
///             return processAuth(user)
///         }
///     }
/// "#;
/// 
/// let mut parser = Parser::new();
/// let result = parser.parse(source)?;
/// 
/// // Verify semantic preservation across syntax styles
/// assert_eq!(result.canonical_form.semantic_hash(), expected_hash);
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
}

/// Project configuration affecting parsing behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Preferred syntax style for the project
    pub preferred_style: Option<SyntaxStyle>,
    
    /// Whether to allow mixed styles within the same file
    pub allow_mixed_styles: bool,
    
    /// Custom validation rules for the project
    pub custom_validation_rules: Vec<String>,
    
    /// AI integration settings
    pub ai_integration: AIIntegrationConfig,
}

/// AI integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIIntegrationConfig {
    /// Whether to generate AI context during parsing
    pub generate_context: bool,
    
    /// Level of AI metadata detail
    pub metadata_detail_level: AIMetadataLevel,
    
    /// Custom AI hints for the project
    pub custom_hints: HashMap<String, String>,
}

/// Level of AI metadata generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AIMetadataLevel {
    /// Minimal metadata for performance
    Minimal,
    /// Standard metadata for most use cases
    Standard,
    /// Comprehensive metadata for AI-first development
    Comprehensive,
}

/// Validation level for parsing strictness
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationLevel {
    /// All validations must pass, including documentation requirements
    Strict,
    /// Warnings allowed, but errors still fail parsing
    Lenient,
    /// Minimal validation, focus on syntax correctness only
    Permissive,
}

/// Complete parsing result with semantic preservation
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// The canonical semantic representation
    pub canonical_form: CanonicalForm,
    
    /// Original syntax style that was detected/used
    pub original_style: SyntaxStyle,
    
    /// Confidence score for syntax detection (1.0 if explicit)
    pub detection_confidence: f64,
    
    /// Validation results and any warnings/errors
    pub validation_result: ValidationResult,
    
    /// Rich metadata for AI systems and tooling
    pub metadata: ParseMetadata,
}

/// Rich metadata generated during parsing
#[derive(Debug, Clone)]
pub struct ParseMetadata {
    /// AI-specific context and hints
    pub ai_context: AIContext,
    
    /// Semantic hints for understanding business logic
    pub semantic_hints: Vec<SemanticHint>,
    
    /// Documentation status and compliance
    pub documentation_status: DocumentationStatus,
    
    /// Conceptual cohesion metrics (if calculated)
    pub cohesion_metrics: Option<CohesionMetrics>,
    
    /// Performance metrics for the parsing operation
    pub performance_metrics: PerformanceMetrics,
}

/// AI context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIContext {
    /// Primary purpose or intent of the parsed code
    pub purpose: Option<String>,
    
    /// Business domain context
    pub domain_context: Option<String>,
    
    /// Key concepts and entities
    pub key_concepts: Vec<String>,
    
    /// Relationships between components
    pub relationships: Vec<ComponentRelationship>,
}

/// Relationship between code components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRelationship {
    /// Source component name
    pub from: String,
    
    /// Target component name  
    pub to: String,
    
    /// Type of relationship
    pub relationship_type: RelationshipType,
    
    /// Optional description of the relationship
    pub description: Option<String>,
}

/// Types of relationships between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// One component depends on another
    Dependency,
    /// Components collaborate to achieve a goal
    Collaboration,
    /// One component inherits from another
    Inheritance,
    /// Components implement the same interface
    Implementation,
    /// One component contains another
    Composition,
}

/// Semantic hints for business logic understanding
#[derive(Debug, Clone)]
pub struct SemanticHint {
    /// The code element this hint applies to
    pub element: String,
    
    /// Type of semantic information
    pub hint_type: SemanticHintType,
    
    /// Human-readable description
    pub description: String,
    
    /// Confidence score for this hint
    pub confidence: f64,
}

/// Types of semantic hints
#[derive(Debug, Clone)]
pub enum SemanticHintType {
    /// Business rule or constraint
    BusinessRule,
    /// Domain concept or entity
    DomainConcept,
    /// Workflow or process step
    WorkflowStep,
    /// Data validation rule
    ValidationRule,
    /// Security consideration
    SecurityHint,
    /// Performance consideration
    PerformanceHint,
}

/// Documentation compliance status
#[derive(Debug, Clone)]
pub struct DocumentationStatus {
    /// Overall compliance score (0.0 to 1.0)
    pub compliance_score: f64,
    
    /// Missing required documentation
    pub missing_documentation: Vec<String>,
    
    /// Documentation quality issues
    pub quality_issues: Vec<DocumentationIssue>,
    
    /// PSG-003 compliance status
    pub psg003_compliance: PSG003Compliance,
}

/// Documentation quality issue
#[derive(Debug, Clone)]
pub struct DocumentationIssue {
    /// Type of issue
    pub issue_type: DocumentationIssueType,
    
    /// Location of the issue
    pub location: String,
    
    /// Description of the issue
    pub description: String,
    
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Types of documentation issues
#[derive(Debug, Clone)]
pub enum DocumentationIssueType {
    /// Missing required annotation
    MissingAnnotation,
    /// Annotation format is incorrect
    InvalidFormat,
    /// Content is unclear or insufficient
    InsufficientContent,
    /// Inconsistent with code behavior
    Inconsistent,
}

/// PSG-003 compliance status
#[derive(Debug, Clone)]
pub struct PSG003Compliance {
    /// Required annotations present
    pub required_annotations_present: bool,
    
    /// Responsibility declarations clear
    pub responsibility_declarations_clear: bool,
    
    /// AI context sufficiently detailed
    pub ai_context_sufficient: bool,
    
    /// Examples provided where required
    pub examples_provided: bool,
}

/// Conceptual cohesion metrics
#[derive(Debug, Clone)]
pub struct CohesionMetrics {
    /// Overall cohesion score (0.0 to 1.0)
    pub overall_score: f64,
    
    /// Individual metric scores
    pub metrics: HashMap<String, f64>,
    
    /// Suggestions for improving cohesion
    pub suggestions: Vec<CohesionSuggestion>,
}

/// Suggestion for improving conceptual cohesion
#[derive(Debug, Clone)]
pub struct CohesionSuggestion {
    /// Type of suggestion
    pub suggestion_type: CohesionSuggestionType,
    
    /// Description of the suggestion
    pub description: String,
    
    /// Priority level
    pub priority: SuggestionPriority,
}

/// Types of cohesion suggestions
#[derive(Debug, Clone)]
pub enum CohesionSuggestionType {
    /// Split module into multiple focused modules
    SplitModule,
    /// Merge related functionality
    MergeComponents,
    /// Extract common functionality
    ExtractCommon,
    /// Clarify responsibilities
    ClarifyResponsibilities,
}

/// Priority levels for suggestions
#[derive(Debug, Clone)]
pub enum SuggestionPriority {
    /// Critical issue affecting maintainability
    Critical,
    /// Important improvement opportunity
    Important,
    /// Minor enhancement
    Minor,
}

/// Performance metrics for parsing operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total parsing time in milliseconds
    pub total_time_ms: u64,
    
    /// Time spent on syntax detection
    pub detection_time_ms: u64,
    
    /// Time spent on style-specific parsing
    pub parsing_time_ms: u64,
    
    /// Time spent on normalization
    pub normalization_time_ms: u64,
    
    /// Time spent on validation
    pub validation_time_ms: u64,
    
    /// Memory usage during parsing
    pub memory_usage_bytes: u64,
}

/// Parser errors with rich diagnostic information
#[derive(Debug, Error)]
pub enum ParseError {
    /// Lexical analysis failed
    #[error("Lexical analysis failed: {message}")]
    LexicalError { message: String },
    
    /// Syntax detection failed
    #[error("Could not detect syntax style: {reason}")]
    DetectionFailed { reason: String },
    
    /// Style-specific parsing failed
    #[error("Parsing failed for {style:?}: {message}")]
    StyleParsingFailed { style: SyntaxStyle, message: String },
    
    /// Normalization to canonical form failed
    #[error("Normalization failed: {message}")]
    NormalizationFailed { message: String },
    
    /// Validation failed with specific errors
    #[error("Validation failed: {errors:?}")]
    ValidationFailed { errors: Vec<String> },
    
    /// Mixed syntax styles detected (when not allowed)
    #[error("Mixed syntax styles detected: {styles:?}")]
    MixedStylesDetected { styles: Vec<SyntaxStyle> },
}

impl Default for ParseContext {
    fn default() -> Self {
        Self {
            source_id: SourceId::new(0),
            file_path: None,
            project_config: None,
            style_preference: None,
            validation_level: ValidationLevel::Lenient,
            generate_ai_metadata: true,
            preserve_formatting: false,
        }
    }
}

impl Parser {
    /// Creates a new parser with default configuration.
    /// 
    /// The parser is initialized with all style parsers, intelligent detection,
    /// and standard validation settings optimized for most use cases.
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
    
    /// Creates a parser with custom configuration.
    pub fn with_context(context: ParseContext) -> Self {
        let mut parser = Self::new();
        parser.context = context;
        parser
    }
    
    /// Parses source code with automatic syntax detection.
    /// 
    /// This is the primary parsing method that:
    /// 1. Automatically detects the syntax style with confidence scoring
    /// 2. Tokenizes using the appropriate lexer configuration
    /// 3. Parses using the detected style-specific parser
    /// 4. Normalizes to canonical form with semantic preservation
    /// 5. Validates against Prism standards and project rules
    /// 6. Generates comprehensive metadata for AI and tooling
    /// 
    /// # Arguments
    /// 
    /// * `source` - The source code to parse
    /// 
    /// # Returns
    /// 
    /// A `ParseResult` with canonical form and rich metadata, or a `ParseError`
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use prism_syntax::Parser;
    /// 
    /// let source = r#"
    ///     module PaymentProcessor {
    ///         function processPayment(amount: Money<USD>) -> Result<Transaction, PaymentError> {
    ///             return validateAndProcess(amount)
    ///         }
    ///     }
    /// "#;
    /// 
    /// let mut parser = Parser::new();
    /// let result = parser.parse(source)?;
    /// 
    /// // Access the canonical form
    /// let canonical = result.canonical_form;
    /// 
    /// // Check AI metadata
    /// if let Some(purpose) = result.metadata.ai_context.purpose {
    ///     println!("Detected purpose: {}", purpose);
    /// }
    /// ```
    pub fn parse(&mut self, source: &str) -> Result<ParseResult, ParseError> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Detect syntax style with confidence scoring
        let detection_start = std::time::Instant::now();
        let detection = if let Some(preferred_style) = self.context.style_preference {
            // Use explicit style preference
            DetectionResult {
                detected_style: preferred_style,
                confidence: 1.0,
                evidence: Vec::new(),
                alternatives: Vec::new(),
                warnings: Vec::new(),
            }
        } else {
            // Automatic detection
            self.detector.detect_syntax(source)
        };
        let detection_time = detection_start.elapsed().as_millis() as u64;
        
        // Step 2: Tokenize using appropriate lexer configuration
        let lexer_style = match detection.detected_style {
            SyntaxStyle::CLike => LexerSyntaxStyle::CLike,
            SyntaxStyle::PythonLike => LexerSyntaxStyle::PythonLike,
            SyntaxStyle::RustLike => LexerSyntaxStyle::RustLike,
            SyntaxStyle::Canonical => LexerSyntaxStyle::Canonical,
        };
        
        let tokens = self.tokenize_with_style(source, lexer_style)?;
        
        // Step 3: Parse using style-specific parser
        let parsing_start = std::time::Instant::now();
        let parsed = self.parse_tokens_with_style(tokens, detection.detected_style)?;
        let parsing_time = parsing_start.elapsed().as_millis() as u64;
        
        // Step 4: Normalize to canonical form with semantic preservation
        let normalization_start = std::time::Instant::now();
        let normalized = self.normalizer.normalize(parsed)?;
        let normalization_time = normalization_start.elapsed().as_millis() as u64;
        
        // Step 5: Validate against Prism standards
        let validation_start = std::time::Instant::now();
        let validation = self.validator.validate(&normalized)?;
        let validation_time = validation_start.elapsed().as_millis() as u64;
        
        // Step 6: Generate comprehensive metadata
        let metadata = self.generate_metadata(&detection, &normalized, &validation);
        
        let total_time = start_time.elapsed().as_millis() as u64;
        
        let performance_metrics = PerformanceMetrics {
            total_time_ms: total_time,
            detection_time_ms: detection_time,
            parsing_time_ms: parsing_time,
            normalization_time_ms: normalization_time,
            validation_time_ms: validation_time,
            memory_usage_bytes: 0, // TODO: Implement memory tracking
        };
        
        Ok(ParseResult {
            canonical_form: normalized,
            original_style: detection.detected_style,
            detection_confidence: detection.confidence,
            validation_result: validation,
            metadata: ParseMetadata {
                ai_context: metadata.ai_context,
                semantic_hints: metadata.semantic_hints,
                documentation_status: metadata.documentation_status,
                cohesion_metrics: metadata.cohesion_metrics,
                performance_metrics,
            },
        })
    }
    
    /// Parses source code with explicit syntax style.
    /// 
    /// Use this method when you know the exact syntax style to avoid detection
    /// overhead and ensure consistent parsing behavior.
    /// 
    /// # Arguments
    /// 
    /// * `source` - The source code to parse
    /// * `style` - The explicit syntax style to use
    /// 
    /// # Returns
    /// 
    /// A `ParseResult` with canonical form and metadata, or a `ParseError`
    pub fn parse_with_style(
        &mut self, 
        source: &str, 
        style: SyntaxStyle
    ) -> Result<ParseResult, ParseError> {
        // Set explicit style preference and parse
        let original_preference = self.context.style_preference;
        self.context.style_preference = Some(style);
        
        let result = self.parse(source);
        
        // Restore original preference
        self.context.style_preference = original_preference;
        
        result
    }
    
    /// Tokenizes source code with a specific lexer style
    fn tokenize_with_style(
        &mut self, 
        source: &str, 
        style: LexerSyntaxStyle
    ) -> Result<Vec<Token>, ParseError> {
        // TODO: Integrate with prism-lexer properly
        // This is a placeholder implementation
        let mut lexer = Lexer::new(source, self.context.source_id, style);
        lexer.tokenize().map_err(|e| ParseError::LexicalError {
            message: format!("{e:?}"),
        })
    }
    
    /// Parses tokens using the appropriate style-specific parser
    fn parse_tokens_with_style(
        &mut self, 
        tokens: Vec<Token>, 
        style: SyntaxStyle
    ) -> Result<ParsedSyntax, ParseError> {
        match style {
            SyntaxStyle::CLike => {
                self.c_like_parser.parse(tokens)
                    .map(ParsedSyntax::CLike)
                    .map_err(|e| ParseError::StyleParsingFailed {
                        style,
                        message: format!("{e:?}"),
                    })
            }
            SyntaxStyle::PythonLike => {
                self.python_like_parser.parse(tokens)
                    .map(ParsedSyntax::PythonLike)
                    .map_err(|e| ParseError::StyleParsingFailed {
                        style,
                        message: format!("{e:?}"),
                    })
            }
            SyntaxStyle::RustLike => {
                self.rust_like_parser.parse(tokens)
                    .map(ParsedSyntax::RustLike)
                    .map_err(|e| ParseError::StyleParsingFailed {
                        style,
                        message: format!("{e:?}"),
                    })
            }
            SyntaxStyle::Canonical => {
                self.canonical_parser.parse(tokens)
                    .map(ParsedSyntax::Canonical)
                    .map_err(|e| ParseError::StyleParsingFailed {
                        style,
                        message: format!("{e:?}"),
                    })
            }
        }
    }
    
    /// Generates comprehensive metadata for the parsing result
    fn generate_metadata(
        &self,
        detection: &DetectionResult,
        canonical: &CanonicalForm,
        validation: &ValidationResult,
    ) -> ParseMetadata {
        // TODO: Implement comprehensive metadata generation
        // This is a placeholder implementation
        
        let ai_context = AIContext {
            purpose: Some("Parsed Prism module with semantic preservation".to_string()),
            domain_context: None,
            key_concepts: Vec::new(),
            relationships: Vec::new(),
        };
        
        let documentation_status = DocumentationStatus {
            compliance_score: validation.overall_score,
            missing_documentation: Vec::new(),
            quality_issues: Vec::new(),
            psg003_compliance: PSG003Compliance {
                required_annotations_present: true,
                responsibility_declarations_clear: true,
                ai_context_sufficient: true,
                examples_provided: false,
            },
        };
        
        ParseMetadata {
            ai_context,
            semantic_hints: Vec::new(),
            documentation_status,
            cohesion_metrics: None,
            performance_metrics: PerformanceMetrics {
                total_time_ms: 0,
                detection_time_ms: 0,
                parsing_time_ms: 0,
                normalization_time_ms: 0,
                validation_time_ms: 0,
                memory_usage_bytes: 0,
            },
        }
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Intermediate parsed syntax before normalization
#[derive(Debug)]
enum ParsedSyntax {
    /// C-like syntax parse result
    CLike(CLikeSyntax),
    /// Python-like syntax parse result
    PythonLike(PythonLikeSyntax),
    /// Rust-like syntax parse result
    RustLike(RustLikeSyntax),
    /// Canonical syntax parse result
    Canonical(CanonicalSyntax),
}

// Placeholder types for parsed syntax - these will be implemented in the styles module
#[derive(Debug)]
struct CLikeSyntax;

#[derive(Debug)]
struct PythonLikeSyntax;

#[derive(Debug)]
struct RustLikeSyntax;

#[derive(Debug)]
struct CanonicalSyntax; 