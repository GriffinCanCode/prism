//! Token types for the Prism programming language
//!
//! This module defines all token types for lexical analysis.
//! It focuses purely on TOKEN REPRESENTATION and does NOT:
//! - Define syntax styles (belongs in parser)
//! - Define semantic context (belongs in parser)
//! - Analyze token relationships (belongs in parser)

use logos::Logos;
use prism_common::span::Span;
use std::fmt;

/// A token in the Prism language with comprehensive semantic metadata
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Token {
    /// The token type and value
    pub kind: TokenKind,
    /// Source location of this token
    pub span: Span,
    /// AI-readable semantic context (PLT-002 requirement)
    pub semantic_context: Option<SemanticContext>,
    /// Syntax style this token was parsed from (PSG-001 integration)
    pub source_style: Option<SyntaxStyle>,
    /// Canonical representation (PLT-002 requirement)
    pub canonical_form: Option<String>,
    /// Documentation validation status (PSG-003 integration)
    pub doc_validation: Option<DocValidationStatus>,
    /// Responsibility annotation context (PSG-002 integration)
    pub responsibility_context: Option<ResponsibilityContext>,
    /// Effect system context (PLD-003 integration)
    pub effect_context: Option<EffectContext>,
    /// Cohesion metrics contribution (PLD-002 integration)
    pub cohesion_impact: Option<CohesionImpact>,
}

impl Token {
    /// Create a new token with basic information
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self {
            kind,
            span,
            semantic_context: None,
            source_style: None,
            canonical_form: None,
            doc_validation: None,
            responsibility_context: None,
            effect_context: None,
            cohesion_impact: None,
        }
    }

    /// Create a token with full semantic context (PLT-002)
    pub fn with_full_context(
        kind: TokenKind,
        span: Span,
        source_style: SyntaxStyle,
        semantic_context: SemanticContext,
        canonical_form: Option<String>,
    ) -> Self {
        Self {
            kind,
            span,
            semantic_context: Some(semantic_context),
            source_style: Some(source_style),
            canonical_form,
            doc_validation: None,
            responsibility_context: None,
            effect_context: None,
            cohesion_impact: None,
        }
    }

    /// Create an EOF token
    pub fn eof() -> Self {
        Self {
            kind: TokenKind::Eof,
            span: Span::dummy(),
            semantic_context: None,
            source_style: None,
            canonical_form: None,
            doc_validation: None,
            responsibility_context: None,
            effect_context: None,
            cohesion_impact: None,
        }
    }

    /// Check if this token requires documentation validation (PSG-003)
    pub fn requires_doc_validation(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Function | TokenKind::Fn | TokenKind::Module | TokenKind::Type | TokenKind::Public | TokenKind::Pub
        )
    }

    /// Check if this token contributes to conceptual cohesion (PLD-002)
    pub fn affects_cohesion(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Function | TokenKind::Fn | TokenKind::Type | TokenKind::Module | TokenKind::Identifier(_)
        )
    }

    /// Get canonical representation of this token (PLT-002)
    pub fn to_canonical(&self) -> String {
        self.canonical_form.clone().unwrap_or_else(|| {
            match &self.kind {
                TokenKind::Fn => "function".to_string(),
                TokenKind::AndAnd => "and".to_string(),
                TokenKind::OrOr => "or".to_string(),
                TokenKind::Bang => "not".to_string(),
                TokenKind::Pub => "public".to_string(),
                TokenKind::Nil => "null".to_string(),
                _ => format!("{}", self.kind),
            }
        })
    }

    /// Check if this token is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::IntegerLiteral(_)
                | TokenKind::FloatLiteral(_)
                | TokenKind::StringLiteral(_)
                | TokenKind::RegexLiteral(_)
                | TokenKind::MoneyLiteral(_)
                | TokenKind::DurationLiteral(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Null
                | TokenKind::Nil
                | TokenKind::Undefined
        )
    }

    /// Check if this token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::Power
                | TokenKind::IntegerDivision
                | TokenKind::At  // Matrix multiplication in Python
                | TokenKind::WalrusOperator
                | TokenKind::Equal
                | TokenKind::NotEqual
                | TokenKind::Less
                | TokenKind::LessEqual
                | TokenKind::Greater
                | TokenKind::GreaterEqual
                | TokenKind::SemanticEqual
                | TokenKind::SemanticNotEqual
                | TokenKind::TypeCompatible
                | TokenKind::ConceptuallySimilar
                | TokenKind::Assign
                | TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::PercentAssign
                | TokenKind::Ampersand
                | TokenKind::Pipe
                | TokenKind::Caret
                | TokenKind::LeftShift
                | TokenKind::RightShift
                | TokenKind::Tilde
                | TokenKind::AndAnd
                | TokenKind::OrOr
                | TokenKind::Bang
        )
    }

    /// Check if this token is a delimiter
    pub fn is_delimiter(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::LeftParen
                | TokenKind::RightParen
                | TokenKind::LeftBracket
                | TokenKind::RightBracket
                | TokenKind::LeftBrace
                | TokenKind::RightBrace
        )
    }

    /// Check if this token is a comment or documentation
    pub fn is_comment(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::LineComment(_)
                | TokenKind::BlockComment(_)
                | TokenKind::DocComment(_)
                | TokenKind::DocBlockComment(_)
                | TokenKind::AiAnnotation(_)
                | TokenKind::ResponsibilityAnnotation(_)
                | TokenKind::EffectAnnotation(_)
                | TokenKind::CapabilityAnnotation(_)
                | TokenKind::Comment(_)
        )
    }

    /// Check if this token is an annotation (PLT-002)
    pub fn is_annotation(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::AiAnnotation(_)
                | TokenKind::ResponsibilityAnnotation(_)
                | TokenKind::EffectAnnotation(_)
                | TokenKind::CapabilityAnnotation(_)
        )
    }

    /// Check if this token is related to the effect system (PLD-003)
    pub fn is_effect_related(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Effects
                | TokenKind::Capability
                | TokenKind::Secure
                | TokenKind::Unsafe
                | TokenKind::EffectAnnotation(_)
                | TokenKind::CapabilityAnnotation(_)
        )
    }

    /// Check if this token is related to semantic types (PLD-001)
    pub fn is_semantic_type_related(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Where
                | TokenKind::With
                | TokenKind::Requires
                | TokenKind::Ensures
                | TokenKind::Invariant
                | TokenKind::SemanticEqual
                | TokenKind::SemanticNotEqual
                | TokenKind::TypeCompatible
                | TokenKind::ConceptuallySimilar
        )
    }

    /// Check if this token is related to smart modules (PLD-002)
    pub fn is_module_related(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Module
                | TokenKind::Section
                | TokenKind::Config
                | TokenKind::Types
                | TokenKind::Errors
                | TokenKind::Events
                | TokenKind::Lifecycle
                | TokenKind::Tests
                | TokenKind::Examples
                | TokenKind::Performance
        )
    }

    /// Get the precedence of this token if it's an operator
    pub fn precedence(&self) -> Option<u8> {
        match &self.kind {
            // Logical OR (lowest precedence)
            TokenKind::Or | TokenKind::OrOr => Some(1),
            
            // Logical AND
            TokenKind::And | TokenKind::AndAnd => Some(2),
            
            // Equality
            TokenKind::Equal | TokenKind::NotEqual | TokenKind::SemanticEqual 
            | TokenKind::SemanticNotEqual => Some(3),
            
            // Relational
            TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater 
            | TokenKind::GreaterEqual | TokenKind::TypeCompatible 
            | TokenKind::ConceptuallySimilar => Some(4),
            
            // Bitwise OR
            TokenKind::Pipe => Some(5),
            
            // Bitwise XOR
            TokenKind::Caret => Some(6),
            
            // Bitwise AND
            TokenKind::Ampersand => Some(7),
            
            // Shift
            TokenKind::LeftShift | TokenKind::RightShift => Some(8),
            
            // Addition/Subtraction
            TokenKind::Plus | TokenKind::Minus => Some(9),
            
            // Multiplication/Division/Modulo/Matrix
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent | TokenKind::IntegerDivision | TokenKind::At => Some(10),
            
            // Assignment (including walrus operator)
            TokenKind::WalrusOperator => Some(0),  // Very low precedence for assignment expressions
            
            // Power (highest precedence)
            TokenKind::Power => Some(11),
            
            _ => None,
        }
    }
}

/// Syntax style detection (PSG-001 integration)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SyntaxStyle {
    /// C/C++/Java/JavaScript: braces, semicolons, parentheses
    CLike,
    /// Python/CoffeeScript: indentation, colons
    PythonLike,
    /// Rust/Go: explicit keywords, snake_case
    RustLike,
    /// Prism canonical: semantic delimiters
    Canonical,
    /// Multiple styles in same file (warning)
    Mixed,
}

/// Comprehensive semantic context for AI comprehension (PLT-002)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemanticContext {
    /// Primary purpose of this token
    pub purpose: Option<String>,
    /// Business domain this token relates to
    pub domain: Option<String>,
    /// Related concepts for AI understanding
    pub related_concepts: Vec<String>,
    /// Hints to help AI systems understand this token's purpose and context
    pub ai_comprehension_hints: Vec<String>,
    /// Security implications
    pub security_implications: Vec<String>,
    /// Performance considerations
    pub performance_hints: Vec<String>,
    /// Business context information
    pub business_context: Vec<String>,
}

/// Documentation validation status (PSG-003 integration)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DocValidationStatus {
    /// Required annotations for this token
    pub required_annotations: Vec<RequiredAnnotationStatus>,
    /// Completeness score (0.0 to 100.0)
    pub completeness_score: f64,
    /// Validation errors
    pub validation_errors: Vec<String>,
    /// Score indicating how well AI systems can understand this token (0.0 to 100.0)
    pub ai_comprehension_score: f64,
}

/// Status of a required annotation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RequiredAnnotationStatus {
    /// Type of annotation required
    pub annotation_type: RequiredAnnotationType,
    /// Whether the annotation is present
    pub present: bool,
    /// Whether the annotation is valid
    pub valid: bool,
    /// Error message if invalid
    pub error_message: Option<String>,
}

/// Types of required annotations (PSG-002/003)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RequiredAnnotationType {
    /// Function responsibility annotation
    Responsibility,
    /// Parameter documentation
    Parameter,
    /// Return value documentation
    Returns,
    /// Example usage
    Example,
    /// Effect declaration
    Effects,
    /// Capability requirement
    Capabilities,
}

/// Responsibility annotation context (PSG-002 integration)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResponsibilityContext {
    /// Responsibility description
    pub responsibility: String,
    /// Business capability this relates to
    pub business_capability: Option<String>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Related responsibilities
    pub related_responsibilities: Vec<String>,
}

/// Effect system context (PLD-003 integration)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectContext {
    /// Effects this token contributes to
    pub effects: Vec<EffectType>,
    /// Capabilities required
    pub capabilities_required: Vec<CapabilityType>,
    /// Security level
    pub security_level: SecurityLevel,
    /// Audit requirements
    pub audit_required: bool,
}

/// Types of effects (PLD-003)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EffectType {
    IO,
    Network,
    System,
    Cryptography,
    Database,
    Memory,
    Computation,
    Custom(String),
}

/// Types of capabilities (PLD-003)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CapabilityType {
    FileSystem,
    Network,
    System,
    Cryptography,
    Database,
    Memory,
    UnsafeOperations,
    Custom(String),
}

/// Security levels (PLD-003)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SecurityLevel {
    /// No security implications
    Public,
    /// Internal use only
    Internal,
    /// Restricted access
    Restricted,
    /// Classified information
    Classified,
}

/// Conceptual cohesion impact (PLD-002 integration)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CohesionImpact {
    /// How this token affects type cohesion
    pub type_cohesion_impact: f64,
    /// How this token affects data flow cohesion
    pub data_flow_impact: f64,
    /// How this token affects semantic cohesion
    pub semantic_impact: f64,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// Conceptual distance from other tokens
    pub conceptual_distance: f64,
}

impl SemanticContext {
    /// Create semantic context with primary purpose
    pub fn with_purpose(purpose: &str) -> Self {
        Self {
            purpose: Some(purpose.to_string()),
            domain: None,
            related_concepts: Vec::new(),
            ai_comprehension_hints: Vec::new(),
            security_implications: Vec::new(),
            performance_hints: Vec::new(),
            business_context: Vec::new(),
        }
    }
    
    /// Add a related concept
    pub fn add_concept(&mut self, concept: &str) {
        self.related_concepts.push(concept.to_string());
    }
    
    /// Add a hint to help AI systems understand this token
    pub fn add_ai_comprehension_hint(&mut self, hint: &str) {
        self.ai_comprehension_hints.push(hint.to_string());
    }
    
    /// Add a security implication
    pub fn add_security_implication(&mut self, implication: &str) {
        self.security_implications.push(implication.to_string());
    }
    
    /// Add a performance hint
    pub fn add_performance_hint(&mut self, hint: &str) {
        self.performance_hints.push(hint.to_string());
    }
    
    /// Add business context
    pub fn add_business_context(&mut self, context: &str) {
        self.business_context.push(context.to_string());
    }
}

/// Comprehensive token types supporting all language constructs
#[derive(Debug, Clone, PartialEq, Logos)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TokenKind {
    // === STRUCTURAL KEYWORDS ===
    
    /// Module system keyword
    #[token("module")]
    Module,
    
    /// Section keyword for smart modules
    #[token("section")]
    Section,
    
    // Section kinds (PLD-002 Smart Module System)
    #[token("config")]
    Config,
    #[token("types")]
    Types,
    #[token("errors")]
    Errors,
    #[token("events")]
    Events,
    #[token("lifecycle")]
    Lifecycle,
    #[token("tests")]
    Tests,
    #[token("examples")]
    Examples,
    #[token("performance")]
    Performance,
    #[token("internal")]
    Internal,
    #[token("operations")]
    Operations,
    #[token("validation")]
    Validation,
    #[token("migration")]
    Migration,
    #[token("documentation")]
    Documentation,
    #[token("statemachine")]
    StateMachine,
    
    /// Capability keyword
    #[token("capability")]
    Capability,
    
    /// Function definition keywords
    #[token("function")]
    Function,
    #[token("fn")]
    Fn,
    
    /// Type definition keywords
    #[token("type")]
    Type,
    #[token("interface")]
    Interface,
    #[token("trait")]
    Trait,
    
    // === VARIABLE DECLARATIONS ===
    
    #[token("let")]
    Let,
    #[token("const")]
    Const,
    #[token("var")]
    Var,
    
    // === CONTROL FLOW ===
    
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("elif")]  // Python-like (PLT-002)
    ElseIf,
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("loop")]
    Loop,
    #[token("match")]
    Match,
    #[token("switch")]
    Switch,
    #[token("case")]
    Case,
    #[token("when")]  // Alternative form (PLT-002)
    When,
    #[token("default")]
    Default,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("yield")]
    Yield,
    
    // === LOGICAL OPERATORS ===
    
    #[token("and")]
    And,
    #[token("&&")]
    AndAnd,
    #[token("or")]
    Or,
    #[token("||")]
    OrOr,
    #[token("not")]
    Not,
    #[token("!")]
    Bang,
    
    // === SEMANTIC TYPE KEYWORDS (PLD-001 integration) ===
    
    #[token("where")]
    Where,
    #[token("with")]
    With,
    #[token("requires")]
    Requires,
    #[token("ensures")]
    Ensures,
    #[token("invariant")]
    Invariant,
    
    // === EFFECT SYSTEM KEYWORDS (PLD-003 integration) ===
    
    #[token("effects")]
    Effects,
    #[token("secure")]
    Secure,
    #[token("unsafe")]
    Unsafe,
    
    // === ASYNC/CONCURRENCY ===
    
    #[token("async")]
    Async,
    #[token("await")]
    Await,
    #[token("sync")]  // PLT-002 requirement
    Sync,
    #[token("actor")]
    Actor,
    #[token("spawn")]
    Spawn,
    #[token("channel")]
    Channel,
    #[token("select")]
    Select,
    
    // === ERROR HANDLING ===
    
    #[token("try")]
    Try,
    #[token("catch")]
    Catch,
    #[token("finally")]
    Finally,
    #[token("throw")]
    Throw,
    #[token("error")]
    Error,
    #[token("result")]
    Result,
    
    // === MODULE SYSTEM ===
    
    #[token("import")]
    Import,
    #[token("export")]
    Export,
    #[token("use")]
    Use,
    #[token("from")]
    From,
    
    // === VISIBILITY (PLD-003 security integration) ===
    
    #[token("public")]
    Public,
    #[token("pub")]
    Pub,
    #[token("private")]
    Private,
    #[token("protected")]  // PLT-002 requirement
    Protected,
    
    // === LITERALS ===
    
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,
    #[token("nil")]  // Alternative (PLT-002)
    Nil,
    #[token("undefined")]  // PLT-002 requirement
    Undefined,
    
    // === MISC KEYWORDS ===
    
    #[token("in")]
    In,
    #[token("as")]
    As,
    #[token("is")]
    Is,
    #[token("typeof")]  // PLT-002 requirement
    Typeof,
    #[token("sizeof")]  // PLT-002 requirement
    Sizeof,
    
    // === COMPLEX LITERALS ===
    
    /// Identifier with semantic context
    Identifier(String),
    
    /// Integer literal
    IntegerLiteral(i64),
    
    /// Float literal
    FloatLiteral(f64),
    
    /// String literal
    StringLiteral(String),
    
    /// F-string start token (f", f', f""", f''')
    FStringStart(String),
    
    /// F-string middle (text between expressions)
    FStringMiddle(String),
    
    /// F-string end token (", ', """, ''')
    FStringEnd(String),
    
    // === ENHANCED LITERALS (PLT-002 requirements) ===
    
    /// Regular expression literal
    RegexLiteral(String),
    
    /// Money literal (semantic type)
    MoneyLiteral(String),
    
    /// Duration literal (semantic type)
    DurationLiteral(String),
    
    // === OPERATORS ===
    
    // Arithmetic
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("**")]
    Power,
    #[token("//")]  // Integer division (PLT-002)
    IntegerDivision,
    
    // Comparison
    #[token("==")]
    Equal,
    #[token("!=")]
    NotEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    
    // Semantic comparison (PLD-001)
    #[token("===")]
    SemanticEqual,
    #[token("!==")]
    SemanticNotEqual,
    #[token("~=")]
    TypeCompatible,
    #[token("≈")]
    ConceptuallySimilar,
    
    // Assignment
    #[token("=")]
    Assign,
    #[token("+=")]
    PlusAssign,
    #[token("-=")]
    MinusAssign,
    #[token("*=")]
    StarAssign,
    #[token("/=")]
    SlashAssign,
    #[token("%=")]  // PLT-002 requirement
    PercentAssign,
    
    // Bitwise
    #[token("&")]
    Ampersand,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("<<")]  // Left shift (PLT-002)
    LeftShift,
    #[token(">>")]  // Right shift (PLT-002)
    RightShift,
    #[token("~")]
    Tilde,
    
    // === DELIMITERS ===
    
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    
    // === PUNCTUATION ===
    
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token("::")]
    DoubleColon,
    #[token(".")]
    Dot,
    #[token("..")]  // Range operator (PLT-002)
    DotDot,
    #[token("...")]  // Spread operator (PLT-002)
    DotDotDot,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token(":=")]  // Walrus operator (Python 3.8+)
    WalrusOperator,
    #[token("?")]
    Question,
    #[token("??")]  // Null coalescing (PLT-002)
    DoubleQuestion,
    #[token("@")]
    At,
    #[token("#")]  // PLT-002 requirement
    Hash,
    #[token("$")]  // PLT-002 requirement
    Dollar,
    
    // === SPECIAL TOKENS ===
    
    /// Newline (significant for parsing)
    #[token("\n")]
    Newline,
    
    /// Indentation token (Python-like syntax)
    Indent(usize),
    
    /// Dedentation token (Python-like syntax)
    Dedent(usize),
    
    /// Whitespace (preserved when configured)
    #[regex(r"[ \t\r]+")]
    Whitespace,
    
    // === COMMENTS & DOCUMENTATION ===
    
    /// Line comment
    LineComment(String),
    
    /// Block comment
    BlockComment(String),
    
    /// Documentation comment (PSG-003)
    DocComment(String),
    
    /// Multi-line documentation comment
    DocBlockComment(String),
    
    /// AI annotation comment (preserved for semantic analysis)
    AiAnnotation(String),
    
    /// Responsibility annotation (PSG-002/003)
    ResponsibilityAnnotation(String),
    
    /// Effect annotation (PLD-003)
    EffectAnnotation(String),
    
    /// Capability annotation (PLD-003)
    CapabilityAnnotation(String),
    
    /// Generic comment (for parser convenience)
    Comment(String),
    
    // === ERROR HANDLING ===
    
    /// Lexer error token
    LexError(String),
    
    /// End of file
    Eof,

    // === SMART MODULE SYSTEM TOKENS ===
    

    
    /// Implements keyword for trait implementation
    #[token("implements")]
    Implements,
    

    
    /// Optional keyword for optional requirements
    #[token("optional")]
    Optional,
    
    /// Must keyword for strict requirements
    #[token("must")]
    Must,
    
    /// Should keyword for recommended requirements
    #[token("should")]
    Should,
    
    /// May keyword for optional requirements
    #[token("may")]
    May,
    
    /// Version keyword for version specifications
    #[token("version")]
    Version,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module => write!(f, "module"),
            Self::Section => write!(f, "section"),
            Self::Config => write!(f, "config"),
            Self::Types => write!(f, "types"),
            Self::Errors => write!(f, "errors"),
            Self::Events => write!(f, "events"),
            Self::Lifecycle => write!(f, "lifecycle"),
            Self::Tests => write!(f, "tests"),
            Self::Examples => write!(f, "examples"),
            Self::Performance => write!(f, "performance"),
            Self::Internal => write!(f, "internal"),
            Self::Operations => write!(f, "operations"),
            Self::Validation => write!(f, "validation"),
            Self::Migration => write!(f, "migration"),
            Self::Documentation => write!(f, "documentation"),
            Self::StateMachine => write!(f, "statemachine"),
            Self::Capability => write!(f, "capability"),
            Self::Function => write!(f, "function"),
            Self::Fn => write!(f, "fn"),
            Self::Type => write!(f, "type"),
            Self::Interface => write!(f, "interface"),
            Self::Trait => write!(f, "trait"),
            Self::Let => write!(f, "let"),
            Self::Const => write!(f, "const"),
            Self::Var => write!(f, "var"),
            Self::If => write!(f, "if"),
            Self::Else => write!(f, "else"),
            Self::ElseIf => write!(f, "elseif"),
            Self::While => write!(f, "while"),
            Self::For => write!(f, "for"),
            Self::Loop => write!(f, "loop"),
            Self::Match => write!(f, "match"),
            Self::Switch => write!(f, "switch"),
            Self::Case => write!(f, "case"),
            Self::When => write!(f, "when"),
            Self::Default => write!(f, "default"),
            Self::Return => write!(f, "return"),
            Self::Break => write!(f, "break"),
            Self::Continue => write!(f, "continue"),
            Self::Yield => write!(f, "yield"),
            Self::And => write!(f, "and"),
            Self::AndAnd => write!(f, "&&"),
            Self::Or => write!(f, "or"),
            Self::OrOr => write!(f, "||"),
            Self::Not => write!(f, "not"),
            Self::Bang => write!(f, "!"),
            Self::Where => write!(f, "where"),
            Self::With => write!(f, "with"),
            Self::Requires => write!(f, "requires"),
            Self::Ensures => write!(f, "ensures"),
            Self::Invariant => write!(f, "invariant"),
            Self::Effects => write!(f, "effects"),
            Self::Secure => write!(f, "secure"),
            Self::Unsafe => write!(f, "unsafe"),
            Self::Async => write!(f, "async"),
            Self::Await => write!(f, "await"),
            Self::Sync => write!(f, "sync"),
            Self::Actor => write!(f, "actor"),
            Self::Spawn => write!(f, "spawn"),
            Self::Channel => write!(f, "channel"),
            Self::Select => write!(f, "select"),
            Self::Try => write!(f, "try"),
            Self::Catch => write!(f, "catch"),
            Self::Finally => write!(f, "finally"),
            Self::Throw => write!(f, "throw"),
            Self::Error => write!(f, "error"),
            Self::Result => write!(f, "result"),
            Self::Import => write!(f, "import"),
            Self::Export => write!(f, "export"),
            Self::Use => write!(f, "use"),
            Self::From => write!(f, "from"),
            Self::Public => write!(f, "public"),
            Self::Pub => write!(f, "pub"),
            Self::Private => write!(f, "private"),
            Self::Protected => write!(f, "protected"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Null => write!(f, "null"),
            Self::Nil => write!(f, "nil"),
            Self::Undefined => write!(f, "undefined"),
            Self::In => write!(f, "in"),
            Self::As => write!(f, "as"),
            Self::Is => write!(f, "is"),
            Self::Typeof => write!(f, "typeof"),
            Self::Sizeof => write!(f, "sizeof"),
            Self::Identifier(name) => write!(f, "{}", name),
            Self::IntegerLiteral(value) => write!(f, "{}", value),
            Self::FloatLiteral(value) => write!(f, "{}", value),
            Self::StringLiteral(value) => write!(f, "\"{}\"", value),
            Self::FStringStart(value) => write!(f, "f{}", value),
            Self::FStringMiddle(value) => write!(f, "{}", value),
            Self::FStringEnd(value) => write!(f, "{}", value),
            Self::RegexLiteral(value) => write!(f, "/{}/", value),
            Self::MoneyLiteral(value) => write!(f, "{}", value),
            Self::DurationLiteral(value) => write!(f, "{}", value),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Percent => write!(f, "%"),
            Self::Power => write!(f, "**"),
            Self::IntegerDivision => write!(f, "//"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::Less => write!(f, "<"),
            Self::LessEqual => write!(f, "<="),
            Self::Greater => write!(f, ">"),
            Self::GreaterEqual => write!(f, ">="),
            Self::SemanticEqual => write!(f, "==="),
            Self::SemanticNotEqual => write!(f, "!=="),
            Self::TypeCompatible => write!(f, "~="),
            Self::ConceptuallySimilar => write!(f, "≈"),
            Self::Assign => write!(f, "="),
            Self::PlusAssign => write!(f, "+="),
            Self::MinusAssign => write!(f, "-="),
            Self::StarAssign => write!(f, "*="),
            Self::SlashAssign => write!(f, "/="),
            Self::PercentAssign => write!(f, "%="),
            Self::Ampersand => write!(f, "&"),
            Self::Pipe => write!(f, "|"),
            Self::Caret => write!(f, "^"),
            Self::LeftShift => write!(f, "<<"),
            Self::RightShift => write!(f, ">>"),
            Self::Tilde => write!(f, "~"),
            Self::LeftParen => write!(f, "("),
            Self::RightParen => write!(f, ")"),
            Self::LeftBracket => write!(f, "["),
            Self::RightBracket => write!(f, "]"),
            Self::LeftBrace => write!(f, "{{"),
            Self::RightBrace => write!(f, "}}"),
            Self::Comma => write!(f, ","),
            Self::Semicolon => write!(f, ";"),
            Self::Colon => write!(f, ":"),
            Self::DoubleColon => write!(f, "::"),
            Self::Dot => write!(f, "."),
            Self::DotDot => write!(f, ".."),
            Self::DotDotDot => write!(f, "..."),
            Self::Arrow => write!(f, "->"),
            Self::FatArrow => write!(f, "=>"),
            Self::WalrusOperator => write!(f, ":="),
            Self::Question => write!(f, "?"),
            Self::DoubleQuestion => write!(f, "??"),
            Self::At => write!(f, "@"),
            Self::Hash => write!(f, "#"),
            Self::Dollar => write!(f, "$"),
            Self::Newline => write!(f, "\\n"),
            Self::Indent(indent) => write!(f, "{}", "  ".repeat(*indent)),
            Self::Dedent(indent) => write!(f, "{}", "  ".repeat(*indent)),
            Self::Whitespace => write!(f, " "),
            Self::LineComment(content) => write!(f, "//{}", content),
            Self::BlockComment(content) => write!(f, "/*{}*/", content),
            Self::DocComment(content) => write!(f, "///{}", content),
            Self::DocBlockComment(content) => write!(f, "/**{}*/", content),
            Self::AiAnnotation(content) => write!(f, "//@{}", content),
            Self::ResponsibilityAnnotation(content) => write!(f, "//@responsibility {}", content),
            Self::EffectAnnotation(content) => write!(f, "//@effect {}", content),
            Self::CapabilityAnnotation(content) => write!(f, "//@capability {}", content),
            Self::Comment(content) => write!(f, "// {}", content),
            Self::LexError(msg) => write!(f, "ERROR: {}", msg),
            Self::Eof => write!(f, "EOF"),
            Self::Implements => write!(f, "implements"),
            Self::Optional => write!(f, "optional"),
            Self::Must => write!(f, "must"),
            Self::Should => write!(f, "should"),
            Self::May => write!(f, "may"),
            Self::Version => write!(f, "version"),
        }
    }
} 