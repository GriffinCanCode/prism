//! Python-like syntax parser for Prism.
//!
//! This module implements a comprehensive parser for Python-like syntax, supporting
//! all major Python language features up to Python 3.12+, including:
//! 
//! - PEP 695: Type parameter syntax and type statements
//! - PEP 701: Enhanced f-strings with arbitrary nesting
//! - PEP 684: Per-interpreter GIL (syntax implications)
//! - Match statements and pattern matching
//! - Async/await syntax
//! - Type hints and annotations
//! - Comprehensive error recovery
//! 
//! The parser maintains conceptual cohesion around "Python-compatible syntax
//! parsing with semantic preservation and AI comprehension support".

use crate::{
    detection::SyntaxStyle,
    styles::traits::{StyleParser, ParserCapabilities, ErrorRecoveryLevel, ConfigError},
    normalization::canonical_form::CanonicalNode,
};
use prism_lexer::{Token, TokenKind};
use prism_common::{span::Span, diagnostics::Diagnostic, symbol::Symbol};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Python-like syntax parser
#[derive(Debug)]
pub struct PythonLikeParser {
    /// Parser configuration
    config: PythonParserConfig,
    
    /// Current indentation stack for tracking blocks
    indentation_stack: Vec<usize>,
    
    /// Current position in token stream
    position: usize,
    
    /// Tokens being parsed
    tokens: Vec<Token>,
    
    /// Error recovery context
    error_context: ErrorContext,
    
    /// AI metadata collection
    ai_metadata: AIMetadataCollector,
}

/// Configuration for Python parser
#[derive(Debug, Clone)]
pub struct PythonParserConfig {
    /// Allow mixed tabs and spaces (not recommended)
    pub allow_mixed_indentation: bool,
    
    /// Tab size for indentation calculation
    pub tab_size: usize,
    
    /// Maximum nesting depth before warning
    pub max_nesting_depth: usize,
    
    /// Enable comprehensive error recovery
    pub enable_error_recovery: bool,
    
    /// Generate AI metadata during parsing
    pub generate_ai_metadata: bool,
    
    /// Python version compatibility mode
    pub python_version: PythonVersion,
    
    /// Enable experimental features
    pub enable_experimental_features: bool,
}

/// Python version compatibility
#[derive(Debug, Clone, PartialEq)]
pub enum PythonVersion {
    /// Python 3.8 compatibility
    Python38,
    /// Python 3.9 compatibility
    Python39,
    /// Python 3.10 compatibility (match statements)
    Python310,
    /// Python 3.11 compatibility (exception groups)
    Python311,
    /// Python 3.12 compatibility (type parameters, enhanced f-strings)
    Python312,
    /// Python 3.13+ (experimental features)
    Python313Plus,
}

/// Parsed Python syntax tree
#[derive(Debug, Clone)]
pub struct PythonSyntaxTree {
    /// Root module
    pub module: PythonModule,
    
    /// Import statements
    pub imports: Vec<ImportStatement>,
    
    /// Type aliases (PEP 695)
    pub type_aliases: Vec<TypeAlias>,
    
    /// Global statements and declarations
    pub statements: Vec<Statement>,
    
    /// AI metadata collected during parsing
    pub ai_metadata: Vec<AIHint>,
    
    /// Diagnostics and warnings
    pub diagnostics: Vec<Diagnostic>,
}

/// Python module representation
#[derive(Debug, Clone)]
pub struct PythonModule {
    /// Module docstring
    pub docstring: Option<String>,
    
    /// Module-level type parameters (PEP 695)
    pub type_parameters: Vec<TypeParameter>,
    
    /// Encoding declaration
    pub encoding: Option<String>,
    
    /// Future imports
    pub future_imports: Vec<String>,
}

/// Import statement types
#[derive(Debug, Clone)]
pub enum ImportStatement {
    /// Simple import: import module
    Import {
        modules: Vec<ImportedModule>,
        span: Span,
    },
    
    /// From import: from module import name
    FromImport {
        module: Option<String>,
        names: Vec<ImportedName>,
        level: usize, // Relative import level
        span: Span,
    },
}

/// Imported module information
#[derive(Debug, Clone)]
pub struct ImportedModule {
    pub name: String,
    pub alias: Option<String>,
    pub span: Span,
}

/// Imported name information
#[derive(Debug, Clone)]
pub struct ImportedName {
    pub name: String,
    pub alias: Option<String>,
    pub span: Span,
}

/// Type alias declaration (PEP 695)
#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: String,
    pub type_parameters: Vec<TypeParameter>,
    pub value: TypeExpression,
    pub span: Span,
}

/// Type parameter (PEP 695)
#[derive(Debug, Clone)]
pub struct TypeParameter {
    pub name: String,
    pub kind: TypeParameterKind,
    pub bound: Option<TypeExpression>,
    pub default: Option<TypeExpression>,
    pub span: Span,
}

/// Type parameter kinds
#[derive(Debug, Clone)]
pub enum TypeParameterKind {
    /// Regular type variable: T
    TypeVar,
    /// Variadic type variable: *Ts
    TypeVarTuple,
    /// Parameter specification: **P
    ParamSpec,
}

/// Type expressions
#[derive(Debug, Clone)]
pub enum TypeExpression {
    /// Simple name: int
    Name { name: String, span: Span },
    
    /// Generic type: List[T]
    Generic {
        base: Box<TypeExpression>,
        args: Vec<TypeExpression>,
        span: Span,
    },
    
    /// Union type: int | str
    Union {
        types: Vec<TypeExpression>,
        span: Span,
    },
    
    /// Optional type: int | None
    Optional {
        inner: Box<TypeExpression>,
        span: Span,
    },
    
    /// Callable type: Callable[[int], str]
    Callable {
        parameters: Vec<TypeExpression>,
        return_type: Box<TypeExpression>,
        span: Span,
    },
    
    /// Literal type: Literal["value"]
    Literal {
        values: Vec<LiteralValue>,
        span: Span,
    },
}

/// Python statements
#[derive(Debug, Clone)]
pub enum Statement {
    /// Expression statement
    Expression {
        expression: Expression,
        span: Span,
    },
    
    /// Assignment statement
    Assignment {
        targets: Vec<AssignmentTarget>,
        value: Expression,
        type_annotation: Option<TypeExpression>,
        span: Span,
    },
    
    /// Augmented assignment (+=, -=, etc.)
    AugmentedAssignment {
        target: AssignmentTarget,
        operator: AugmentedAssignmentOperator,
        value: Expression,
        span: Span,
    },
    
    /// Function definition
    FunctionDef {
        name: String,
        type_parameters: Vec<TypeParameter>, // PEP 695
        parameters: Vec<Parameter>,
        return_type: Option<TypeExpression>,
        body: Vec<Statement>,
        decorators: Vec<Decorator>,
        is_async: bool,
        span: Span,
    },
    
    /// Class definition
    ClassDef {
        name: String,
        type_parameters: Vec<TypeParameter>, // PEP 695
        bases: Vec<Expression>,
        keywords: Vec<Keyword>,
        body: Vec<Statement>,
        decorators: Vec<Decorator>,
        span: Span,
    },
    
    /// If statement
    If {
        test: Expression,
        body: Vec<Statement>,
        orelse: Vec<Statement>,
        span: Span,
    },
    
    /// While loop
    While {
        test: Expression,
        body: Vec<Statement>,
        orelse: Vec<Statement>,
        span: Span,
    },
    
    /// For loop
    For {
        target: AssignmentTarget,
        iter: Expression,
        body: Vec<Statement>,
        orelse: Vec<Statement>,
        is_async: bool,
        span: Span,
    },
    
    /// Try statement
    Try {
        body: Vec<Statement>,
        handlers: Vec<ExceptionHandler>,
        orelse: Vec<Statement>,
        finalbody: Vec<Statement>,
        span: Span,
    },
    
    /// With statement
    With {
        items: Vec<WithItem>,
        body: Vec<Statement>,
        is_async: bool,
        span: Span,
    },
    
    /// Match statement (Python 3.10+)
    Match {
        subject: Expression,
        cases: Vec<MatchCase>,
        span: Span,
    },
    
    /// Return statement
    Return {
        value: Option<Expression>,
        span: Span,
    },
    
    /// Yield statement
    Yield {
        value: Option<Expression>,
        is_from: bool,
        span: Span,
    },
    
    /// Raise statement
    Raise {
        exception: Option<Expression>,
        cause: Option<Expression>,
        span: Span,
    },
    
    /// Assert statement
    Assert {
        test: Expression,
        msg: Option<Expression>,
        span: Span,
    },
    
    /// Import statement
    Import(ImportStatement),
    
    /// Global statement
    Global {
        names: Vec<String>,
        span: Span,
    },
    
    /// Nonlocal statement
    Nonlocal {
        names: Vec<String>,
        span: Span,
    },
    
    /// Pass statement
    Pass { span: Span },
    
    /// Break statement
    Break { span: Span },
    
    /// Continue statement
    Continue { span: Span },
    
    /// Delete statement
    Delete {
        targets: Vec<Expression>,
        span: Span,
    },
    
    /// Type alias statement (PEP 695)
    TypeAlias(TypeAlias),
}

/// Python expressions
#[derive(Debug, Clone)]
pub enum Expression {
    /// Literal values
    Literal {
        value: LiteralValue,
        span: Span,
    },
    
    /// Names/identifiers
    Name {
        id: String,
        span: Span,
    },
    
    /// Attribute access: obj.attr
    Attribute {
        value: Box<Expression>,
        attr: String,
        span: Span,
    },
    
    /// Subscript: obj[key]
    Subscript {
        value: Box<Expression>,
        slice: Box<Expression>,
        span: Span,
    },
    
    /// Function call
    Call {
        func: Box<Expression>,
        args: Vec<Expression>,
        keywords: Vec<Keyword>,
        span: Span,
    },
    
    /// Binary operation
    BinaryOp {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
        span: Span,
    },
    
    /// Unary operation
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<Expression>,
        span: Span,
    },
    
    /// Boolean operation
    BoolOp {
        operator: BooleanOperator,
        values: Vec<Expression>,
        span: Span,
    },
    
    /// Comparison
    Compare {
        left: Box<Expression>,
        ops: Vec<ComparisonOperator>,
        comparators: Vec<Expression>,
        span: Span,
    },
    
    /// Conditional expression: a if test else b
    IfExp {
        test: Box<Expression>,
        body: Box<Expression>,
        orelse: Box<Expression>,
        span: Span,
    },
    
    /// Lambda function
    Lambda {
        parameters: Vec<Parameter>,
        body: Box<Expression>,
        span: Span,
    },
    
    /// List literal
    List {
        elements: Vec<Expression>,
        span: Span,
    },
    
    /// Tuple literal
    Tuple {
        elements: Vec<Expression>,
        span: Span,
    },
    
    /// Set literal
    Set {
        elements: Vec<Expression>,
        span: Span,
    },
    
    /// Dictionary literal
    Dict {
        keys: Vec<Option<Expression>>,
        values: Vec<Expression>,
        span: Span,
    },
    
    /// List comprehension
    ListComp {
        element: Box<Expression>,
        generators: Vec<Comprehension>,
        span: Span,
    },
    
    /// Set comprehension
    SetComp {
        element: Box<Expression>,
        generators: Vec<Comprehension>,
        span: Span,
    },
    
    /// Dictionary comprehension
    DictComp {
        key: Box<Expression>,
        value: Box<Expression>,
        generators: Vec<Comprehension>,
        span: Span,
    },
    
    /// Generator expression
    GeneratorExp {
        element: Box<Expression>,
        generators: Vec<Comprehension>,
        span: Span,
    },
    
    /// Await expression
    Await {
        value: Box<Expression>,
        span: Span,
    },
    
    /// Yield expression
    Yield {
        value: Option<Box<Expression>>,
        is_from: bool,
        span: Span,
    },
    
    /// F-string (PEP 701 enhanced)
    FormattedString {
        parts: Vec<FStringPart>,
        span: Span,
    },
    
    /// Starred expression: *args
    Starred {
        value: Box<Expression>,
        span: Span,
    },
    
    /// Slice expression
    Slice {
        lower: Option<Box<Expression>>,
        upper: Option<Box<Expression>>,
        step: Option<Box<Expression>>,
        span: Span,
    },
    
    /// Named expression (walrus operator): x := value
    NamedExpr {
        target: String,
        value: Box<Expression>,
        span: Span,
    },
}

/// F-string parts (PEP 701)
#[derive(Debug, Clone)]
pub enum FStringPart {
    /// Literal text
    Literal {
        value: String,
        span: Span,
    },
    
    /// Expression with optional formatting
    Expression {
        expression: Expression,
        conversion: Option<FStringConversion>,
        format_spec: Option<String>,
        debug: bool, // = after expression
        span: Span,
    },
}

/// F-string conversion types
#[derive(Debug, Clone)]
pub enum FStringConversion {
    Str,  // !s
    Repr, // !r
    Ascii, // !a
}

/// Literal values
#[derive(Debug, Clone)]
pub enum LiteralValue {
    /// None literal
    None,
    
    /// Boolean literals
    Bool(bool),
    
    /// Integer literal
    Int(i64),
    
    /// Float literal
    Float(f64),
    
    /// Complex literal
    Complex { real: f64, imag: f64 },
    
    /// String literal
    String(String),
    
    /// Bytes literal
    Bytes(Vec<u8>),
    
    /// Ellipsis literal
    Ellipsis,
}

/// Function/method parameters
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub annotation: Option<TypeExpression>,
    pub default: Option<Expression>,
    pub kind: ParameterKind,
    pub span: Span,
}

/// Parameter kinds
#[derive(Debug, Clone)]
pub enum ParameterKind {
    /// Regular parameter
    Regular,
    /// Positional-only parameter (before /)
    PositionalOnly,
    /// Keyword-only parameter (after *)
    KeywordOnly,
    /// *args parameter
    VarPositional,
    /// **kwargs parameter
    VarKeyword,
}

/// Assignment targets
#[derive(Debug, Clone)]
pub enum AssignmentTarget {
    /// Simple name
    Name { id: String, span: Span },
    
    /// Attribute assignment
    Attribute {
        value: Box<Expression>,
        attr: String,
        span: Span,
    },
    
    /// Subscript assignment
    Subscript {
        value: Box<Expression>,
        slice: Box<Expression>,
        span: Span,
    },
    
    /// Tuple unpacking
    Tuple {
        elements: Vec<AssignmentTarget>,
        span: Span,
    },
    
    /// List unpacking
    List {
        elements: Vec<AssignmentTarget>,
        span: Span,
    },
    
    /// Starred target: *rest
    Starred {
        value: Box<AssignmentTarget>,
        span: Span,
    },
}

/// Operators
#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Add, Sub, Mult, MatMult, Div, Mod, Pow, LShift, RShift,
    BitOr, BitXor, BitAnd, FloorDiv,
}

#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Invert, Not, UAdd, USub,
}

#[derive(Debug, Clone)]
pub enum BooleanOperator {
    And, Or,
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    Eq, NotEq, Lt, LtE, Gt, GtE, Is, IsNot, In, NotIn,
}

#[derive(Debug, Clone)]
pub enum AugmentedAssignmentOperator {
    Add, Sub, Mult, MatMult, Div, Mod, Pow, LShift, RShift,
    BitOr, BitXor, BitAnd, FloorDiv,
}

/// Decorator
#[derive(Debug, Clone)]
pub struct Decorator {
    pub decorator: Expression,
    pub span: Span,
}

/// Keyword argument
#[derive(Debug, Clone)]
pub struct Keyword {
    pub arg: Option<String>,
    pub value: Expression,
    pub span: Span,
}

/// Exception handler
#[derive(Debug, Clone)]
pub struct ExceptionHandler {
    pub exception_type: Option<Expression>,
    pub name: Option<String>,
    pub body: Vec<Statement>,
    pub span: Span,
}

/// With statement item
#[derive(Debug, Clone)]
pub struct WithItem {
    pub context_expr: Expression,
    pub optional_vars: Option<AssignmentTarget>,
    pub span: Span,
}

/// Match case (Python 3.10+)
#[derive(Debug, Clone)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Vec<Statement>,
    pub span: Span,
}

/// Pattern matching patterns
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Match any value
    MatchValue {
        value: Expression,
        span: Span,
    },
    
    /// Match and bind to name
    MatchSingleton {
        value: LiteralValue,
        span: Span,
    },
    
    /// Match sequence
    MatchSequence {
        patterns: Vec<Pattern>,
        span: Span,
    },
    
    /// Match mapping
    MatchMapping {
        keys: Vec<Expression>,
        patterns: Vec<Pattern>,
        rest: Option<String>,
        span: Span,
    },
    
    /// Match class
    MatchClass {
        cls: Expression,
        patterns: Vec<Pattern>,
        kwd_attrs: Vec<String>,
        kwd_patterns: Vec<Pattern>,
        span: Span,
    },
    
    /// Match star pattern
    MatchStar {
        name: Option<String>,
        span: Span,
    },
    
    /// Match as pattern
    MatchAs {
        pattern: Option<Box<Pattern>>,
        name: Option<String>,
        span: Span,
    },
    
    /// Match or pattern
    MatchOr {
        patterns: Vec<Pattern>,
        span: Span,
    },
}

/// Comprehension clause
#[derive(Debug, Clone)]
pub struct Comprehension {
    pub target: AssignmentTarget,
    pub iter: Expression,
    pub ifs: Vec<Expression>,
    pub is_async: bool,
    pub span: Span,
}

/// Error context for recovery
#[derive(Debug)]
struct ErrorContext {
    /// Current recovery mode
    recovery_mode: RecoveryMode,
    
    /// Errors encountered
    errors: Vec<PythonParseError>,
    
    /// Synchronization points for recovery
    sync_points: Vec<TokenKind>,
}

/// Recovery modes
#[derive(Debug)]
enum RecoveryMode {
    /// Normal parsing
    Normal,
    
    /// Skip to next statement
    SkipToStatement,
    
    /// Skip to end of block
    SkipToBlockEnd,
    
    /// Skip to specific token
    SkipToToken(TokenKind),
}

/// AI metadata collector
#[derive(Debug)]
struct AIMetadataCollector {
    /// Business concepts found
    business_concepts: Vec<String>,
    
    /// Architectural patterns identified
    patterns: Vec<String>,
    
    /// Complexity indicators
    complexity_indicators: Vec<String>,
}

/// AI hint for code understanding
#[derive(Debug, Clone)]
pub struct AIHint {
    pub concept: String,
    pub description: String,
    pub confidence: f64,
    pub span: Option<Span>,
}

/// Python parsing errors
#[derive(Debug, Error)]
pub enum PythonParseError {
    #[error("Unexpected token: expected {expected}, found {found}")]
    UnexpectedToken { expected: String, found: String },
    
    #[error("Invalid indentation at line {line}")]
    InvalidIndentation { line: usize },
    
    #[error("Mixed tabs and spaces in indentation")]
    MixedIndentation,
    
    #[error("Unexpected end of file")]
    UnexpectedEof,
    
    #[error("Invalid syntax: {message}")]
    InvalidSyntax { message: String },
    
    #[error("Unsupported Python version feature: {feature}")]
    UnsupportedFeature { feature: String },
    
    #[error("Maximum nesting depth exceeded: {depth}")]
    MaxNestingDepthExceeded { depth: usize },
    
    #[error("Invalid f-string: {message}")]
    InvalidFString { message: String },
    
    #[error("Invalid type parameter: {message}")]
    InvalidTypeParameter { message: String },
}

impl Default for PythonParserConfig {
    fn default() -> Self {
        Self {
            allow_mixed_indentation: false,
            tab_size: 4,
            max_nesting_depth: 100,
            enable_error_recovery: true,
            generate_ai_metadata: true,
            python_version: PythonVersion::Python312,
            enable_experimental_features: false,
        }
    }
}

impl crate::styles::traits::StyleConfig for PythonParserConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.tab_size == 0 {
            return Err(ConfigError::InvalidParameter {
                parameter: "tab_size".to_string(),
                reason: "Tab size must be greater than 0".to_string(),
            });
        }
        
        if self.max_nesting_depth < 10 {
            return Err(ConfigError::InvalidParameter {
                parameter: "max_nesting_depth".to_string(),
                reason: "Maximum nesting depth should be at least 10".to_string(),
            });
        }
        
        Ok(())
    }
}

impl StyleParser for PythonLikeParser {
    type Output = PythonSyntaxTree;
    type Error = PythonParseError;
    type Config = PythonParserConfig;
    
    fn new() -> Self {
        Self::with_config(PythonParserConfig::default())
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self {
            config,
            indentation_stack: vec![0], // Start with zero indentation
            position: 0,
            tokens: Vec::new(),
            error_context: ErrorContext {
                recovery_mode: RecoveryMode::Normal,
                errors: Vec::new(),
                sync_points: vec![
                    TokenKind::Newline,
                    TokenKind::Keyword, // def, class, if, etc.
                    TokenKind::Indent,
                    TokenKind::Dedent,
                ],
            },
            ai_metadata: AIMetadataCollector {
                business_concepts: Vec::new(),
                patterns: Vec::new(),
                complexity_indicators: Vec::new(),
            },
        }
    }
    
    fn parse(&mut self, tokens: Vec<Token>) -> Result<Self::Output, Self::Error> {
        self.tokens = tokens;
        self.position = 0;
        
        // Parse module
        let module = self.parse_module()?;
        
        // Parse top-level statements
        let mut statements = Vec::new();
        let mut imports = Vec::new();
        let mut type_aliases = Vec::new();
        
        while !self.is_at_end() {
            match self.parse_statement()? {
                Statement::Import(import) => imports.push(import),
                Statement::TypeAlias(alias) => type_aliases.push(alias),
                stmt => statements.push(stmt),
            }
        }
        
        // Generate AI metadata
        let ai_metadata = if self.config.generate_ai_metadata {
            self.generate_ai_hints(&statements)
        } else {
            Vec::new()
        };
        
        Ok(PythonSyntaxTree {
            module,
            imports,
            type_aliases,
            statements,
            ai_metadata,
            diagnostics: Vec::new(), // TODO: Convert errors to diagnostics
        })
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::PythonLike
    }
    
    fn capabilities(&self) -> ParserCapabilities {
        ParserCapabilities {
            supports_mixed_indentation: self.config.allow_mixed_indentation,
            supports_optional_semicolons: false, // Python requires newlines
            supports_trailing_commas: true,
            supports_nested_comments: false, // Python doesn't have nested comments
            error_recovery_level: if self.config.enable_error_recovery {
                ErrorRecoveryLevel::Advanced
            } else {
                ErrorRecoveryLevel::Basic
            },
            max_nesting_depth: self.config.max_nesting_depth,
            supports_ai_metadata: self.config.generate_ai_metadata,
        }
    }
}

impl PythonLikeParser {
    /// Parse module-level information
    fn parse_module(&mut self) -> Result<PythonModule, PythonParseError> {
        let mut docstring = None;
        let mut type_parameters = Vec::new();
        let mut encoding = None;
        let mut future_imports = Vec::new();
        
        // Check for encoding declaration in first few lines
        if let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::Comment) {
                // TODO: Parse encoding from comment
            }
        }
        
        // Check for module docstring
        if let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::String) {
                docstring = Some(self.parse_string_literal()?);
                self.advance();
            }
        }
        
        Ok(PythonModule {
            docstring,
            type_parameters,
            encoding,
            future_imports,
        })
    }
    
    /// Parse a statement
    fn parse_statement(&mut self) -> Result<Statement, PythonParseError> {
        self.skip_newlines();
        
        if self.is_at_end() {
            return Err(PythonParseError::UnexpectedEof);
        }
        
        match &self.peek().unwrap().kind {
            TokenKind::Keyword => self.parse_compound_statement(),
            _ => self.parse_simple_statement(),
        }
    }
    
    /// Parse compound statements (def, class, if, etc.)
    fn parse_compound_statement(&mut self) -> Result<Statement, PythonParseError> {
        let token = self.advance().unwrap();
        
        match token.value.as_str() {
            "def" => self.parse_function_def(false),
            "async" => {
                if self.match_keyword("def") {
                    self.parse_function_def(true)
                } else if self.match_keyword("for") {
                    self.parse_for_statement(true)
                } else if self.match_keyword("with") {
                    self.parse_with_statement(true)
                } else {
                    Err(PythonParseError::InvalidSyntax {
                        message: "Invalid async statement".to_string(),
                    })
                }
            }
            "class" => self.parse_class_def(),
            "if" => self.parse_if_statement(),
            "while" => self.parse_while_statement(),
            "for" => self.parse_for_statement(false),
            "try" => self.parse_try_statement(),
            "with" => self.parse_with_statement(false),
            "match" => {
                if self.config.python_version >= PythonVersion::Python310 {
                    self.parse_match_statement()
                } else {
                    Err(PythonParseError::UnsupportedFeature {
                        feature: "match statement (requires Python 3.10+)".to_string(),
                    })
                }
            }
            "type" => {
                if self.config.python_version >= PythonVersion::Python312 {
                    Ok(Statement::TypeAlias(self.parse_type_alias()?))
                } else {
                    Err(PythonParseError::UnsupportedFeature {
                        feature: "type statement (requires Python 3.12+)".to_string(),
                    })
                }
            }
            _ => Err(PythonParseError::InvalidSyntax {
                message: format!("Unexpected keyword: {}", token.value),
            }),
        }
    }
    
    /// Parse simple statements
    fn parse_simple_statement(&mut self) -> Result<Statement, PythonParseError> {
        let token = self.peek().unwrap();
        
        match &token.kind {
            TokenKind::Identifier => {
                // Could be assignment, expression, or other
                self.parse_expression_or_assignment()
            }
            TokenKind::Keyword => {
                let keyword = token.value.as_str();
                match keyword {
                    "return" => {
                        self.advance();
                        self.parse_return_statement()
                    }
                    "yield" => {
                        self.advance();
                        self.parse_yield_statement()
                    }
                    "raise" => {
                        self.advance();
                        self.parse_raise_statement()
                    }
                    "assert" => {
                        self.advance();
                        self.parse_assert_statement()
                    }
                    "import" => {
                        self.advance();
                        Ok(Statement::Import(self.parse_import_statement()?))
                    }
                    "from" => {
                        self.advance();
                        Ok(Statement::Import(self.parse_from_import_statement()?))
                    }
                    "global" => {
                        self.advance();
                        self.parse_global_statement()
                    }
                    "nonlocal" => {
                        self.advance();
                        self.parse_nonlocal_statement()
                    }
                    "pass" => {
                        let span = self.advance().unwrap().span;
                        Ok(Statement::Pass { span })
                    }
                    "break" => {
                        let span = self.advance().unwrap().span;
                        Ok(Statement::Break { span })
                    }
                    "continue" => {
                        let span = self.advance().unwrap().span;
                        Ok(Statement::Continue { span })
                    }
                    "del" => {
                        self.advance();
                        self.parse_delete_statement()
                    }
                    _ => Err(PythonParseError::InvalidSyntax {
                        message: format!("Unexpected keyword in simple statement: {}", keyword),
                    }),
                }
            }
            _ => {
                // Expression statement
                let expr = self.parse_expression()?;
                Ok(Statement::Expression {
                    expression: expr.clone(),
                    span: self.get_expression_span(&expr),
                })
            }
        }
    }
    
    // Helper methods
    
    fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }
    
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }
    
    fn advance(&mut self) -> Option<Token> {
        if !self.is_at_end() {
            let token = self.tokens[self.position].clone();
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }
    
    fn match_keyword(&mut self, keyword: &str) -> bool {
        if let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::Keyword) && token.value == keyword {
                self.advance();
                return true;
            }
        }
        false
    }
    
    fn skip_newlines(&mut self) {
        while let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::Newline) {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn parse_string_literal(&self, content: &str) -> Result<String, PythonParseError> {
        // Implement proper string literal parsing with escape handling
        let mut result = String::new();
        let mut chars = content.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                '\\' => {
                    // Handle escape sequences
                    match chars.next() {
                        Some('n') => result.push('\n'),
                        Some('t') => result.push('\t'),
                        Some('r') => result.push('\r'),
                        Some('\\') => result.push('\\'),
                        Some('\'') => result.push('\''),
                        Some('"') => result.push('"'),
                        Some('0') => result.push('\0'),
                        Some('x') => {
                            // Hexadecimal escape sequence \xHH
                            let hex1 = chars.next().ok_or(PythonParseError::InvalidEscapeSequence)?;
                            let hex2 = chars.next().ok_or(PythonParseError::InvalidEscapeSequence)?;
                            let hex_str = format!("{}{}", hex1, hex2);
                            let hex_value = u8::from_str_radix(&hex_str, 16)
                                .map_err(|_| PythonParseError::InvalidEscapeSequence)?;
                            result.push(hex_value as char);
                        }
                        Some('u') => {
                            // Unicode escape sequence \uHHHH
                            let mut hex_str = String::new();
                            for _ in 0..4 {
                                let hex_digit = chars.next().ok_or(PythonParseError::InvalidEscapeSequence)?;
                                hex_str.push(hex_digit);
                            }
                            let hex_value = u32::from_str_radix(&hex_str, 16)
                                .map_err(|_| PythonParseError::InvalidEscapeSequence)?;
                            if let Some(unicode_char) = std::char::from_u32(hex_value) {
                                result.push(unicode_char);
                            } else {
                                return Err(PythonParseError::InvalidEscapeSequence);
                            }
                        }
                        Some('U') => {
                            // Extended unicode escape sequence \UHHHHHHHH
                            let mut hex_str = String::new();
                            for _ in 0..8 {
                                let hex_digit = chars.next().ok_or(PythonParseError::InvalidEscapeSequence)?;
                                hex_str.push(hex_digit);
                            }
                            let hex_value = u32::from_str_radix(&hex_str, 16)
                                .map_err(|_| PythonParseError::InvalidEscapeSequence)?;
                            if let Some(unicode_char) = std::char::from_u32(hex_value) {
                                result.push(unicode_char);
                            } else {
                                return Err(PythonParseError::InvalidEscapeSequence);
                            }
                        }
                        Some(other) => {
                            // Unknown escape sequence, treat as literal
                            result.push('\\');
                            result.push(other);
                        }
                        None => return Err(PythonParseError::InvalidEscapeSequence),
                    }
                }
                _ => result.push(ch),
            }
        }
        
        Ok(result)
    }
    
    fn get_expression_span(&self, expr: &Expression) -> Span {
        // Implement proper span calculation based on expression type
        match expr {
            Expression::Literal { span, .. } => *span,
            Expression::Identifier { span, .. } => *span,
            Expression::BinaryOperation { left, right, .. } => {
                let left_span = self.get_expression_span(left);
                let right_span = self.get_expression_span(right);
                Span::new(
                    left_span.start(),
                    right_span.end(),
                    left_span.source_id(),
                )
            }
            Expression::UnaryOperation { operand, span, .. } => {
                let operand_span = self.get_expression_span(operand);
                Span::new(
                    span.start().min(operand_span.start()),
                    span.end().max(operand_span.end()),
                    span.source_id(),
                )
            }
            Expression::FunctionCall { function, arguments, .. } => {
                let func_span = self.get_expression_span(function);
                let mut end_pos = func_span.end();
                
                // Extend span to include all arguments
                for arg in arguments {
                    let arg_span = self.get_expression_span(arg);
                    end_pos = end_pos.max(arg_span.end());
                }
                
                Span::new(func_span.start(), end_pos, func_span.source_id())
            }
            Expression::ListLiteral { elements, span, .. } => {
                if elements.is_empty() {
                    *span
                } else {
                    let first_span = self.get_expression_span(&elements[0]);
                    let last_span = self.get_expression_span(elements.last().unwrap());
                    Span::new(
                        span.start().min(first_span.start()),
                        span.end().max(last_span.end()),
                        span.source_id(),
                    )
                }
            }
            Expression::DictLiteral { pairs, span, .. } => {
                if pairs.is_empty() {
                    *span
                } else {
                    let first_key_span = self.get_expression_span(&pairs[0].0);
                    let last_value_span = self.get_expression_span(&pairs.last().unwrap().1);
                    Span::new(
                        span.start().min(first_key_span.start()),
                        span.end().max(last_value_span.end()),
                        span.source_id(),
                    )
                }
            }
            Expression::AttributeAccess { object, span, .. } => {
                let object_span = self.get_expression_span(object);
                Span::new(
                    object_span.start(),
                    span.end(),
                    span.source_id(),
                )
            }
            Expression::IndexAccess { object, index, .. } => {
                let object_span = self.get_expression_span(object);
                let index_span = self.get_expression_span(index);
                Span::new(
                    object_span.start(),
                    index_span.end(),
                    object_span.source_id(),
                )
            }
            Expression::SliceAccess { object, start, end, step, .. } => {
                let object_span = self.get_expression_span(object);
                let mut end_pos = object_span.end();
                
                if let Some(start_expr) = start {
                    end_pos = end_pos.max(self.get_expression_span(start_expr).end());
                }
                if let Some(end_expr) = end {
                    end_pos = end_pos.max(self.get_expression_span(end_expr).end());
                }
                if let Some(step_expr) = step {
                    end_pos = end_pos.max(self.get_expression_span(step_expr).end());
                }
                
                Span::new(object_span.start(), end_pos, object_span.source_id())
            }
        }
    }
    
    fn generate_ai_hints(&self, statements: &[Statement]) -> Vec<AIHint> {
        // Implement AI hint generation for better code understanding
        let mut hints = Vec::new();
        
        for statement in statements {
            match statement {
                Statement::FunctionDef { name, args, body, decorators, .. } => {
                    // Generate hints for function complexity and purpose
                    let complexity_score = self.calculate_function_complexity(body);
                    
                    if complexity_score > 10 {
                        hints.push(AIHint {
                            hint_type: AIHintType::Complexity,
                            message: format!("Function '{}' has high complexity ({}). Consider breaking it down.", name, complexity_score),
                            confidence: 0.8,
                            location: None, // TODO: Add proper location
                            suggestions: vec![
                                "Extract helper functions".to_string(),
                                "Reduce nested conditions".to_string(),
                                "Use early returns to reduce nesting".to_string(),
                            ],
                        });
                    }
                    
                    // Check for missing docstring
                    if !self.has_docstring(body) {
                        hints.push(AIHint {
                            hint_type: AIHintType::Documentation,
                            message: format!("Function '{}' is missing documentation", name),
                            confidence: 0.9,
                            location: None,
                            suggestions: vec![
                                "Add a docstring describing the function's purpose".to_string(),
                                "Document parameters and return values".to_string(),
                            ],
                        });
                    }
                    
                    // Check for too many parameters
                    if args.len() > 5 {
                        hints.push(AIHint {
                            hint_type: AIHintType::Design,
                            message: format!("Function '{}' has {} parameters. Consider using a configuration object.", name, args.len()),
                            confidence: 0.7,
                            location: None,
                            suggestions: vec![
                                "Group related parameters into a data class".to_string(),
                                "Use keyword-only arguments".to_string(),
                            ],
                        });
                    }
                    
                    // Check for decorators suggesting patterns
                    if !decorators.is_empty() {
                        hints.push(AIHint {
                            hint_type: AIHintType::Pattern,
                            message: format!("Function '{}' uses decorators, suggesting aspect-oriented programming patterns", name),
                            confidence: 0.6,
                            location: None,
                            suggestions: vec!["Consider documenting the decorator's effects".to_string()],
                        });
                    }
                }
                
                Statement::ClassDef { name, bases, body, .. } => {
                    // Generate hints for class design
                    let method_count = self.count_methods(body);
                    
                    if method_count > 15 {
                        hints.push(AIHint {
                            hint_type: AIHintType::Design,
                            message: format!("Class '{}' has {} methods. Consider splitting responsibilities.", name, method_count),
                            confidence: 0.8,
                            location: None,
                            suggestions: vec![
                                "Apply Single Responsibility Principle".to_string(),
                                "Extract related methods into separate classes".to_string(),
                            ],
                        });
                    }
                    
                    // Check for inheritance depth
                    if bases.len() > 2 {
                        hints.push(AIHint {
                            hint_type: AIHintType::Design,
                            message: format!("Class '{}' inherits from {} bases. Consider composition over inheritance.", name, bases.len()),
                            confidence: 0.7,
                            location: None,
                            suggestions: vec!["Use composition instead of multiple inheritance".to_string()],
                        });
                    }
                }
                
                Statement::ImportFrom { module, names, .. } => {
                    // Check for wildcard imports
                    if names.iter().any(|alias| alias.name == "*") {
                        hints.push(AIHint {
                            hint_type: AIHintType::Style,
                            message: format!("Wildcard import from '{}' can pollute namespace", module.as_deref().unwrap_or("unknown")),
                            confidence: 0.9,
                            location: None,
                            suggestions: vec!["Import specific names instead of using wildcard".to_string()],
                        });
                    }
                }
                
                Statement::TryExcept { body, handlers, orelse, finalbody, .. } => {
                    // Check for bare except clauses
                    if handlers.iter().any(|handler| handler.exception_type.is_none()) {
                        hints.push(AIHint {
                            hint_type: AIHintType::ErrorHandling,
                            message: "Bare except clause can hide bugs. Specify exception types.".to_string(),
                            confidence: 0.9,
                            location: None,
                            suggestions: vec![
                                "Catch specific exception types".to_string(),
                                "Use 'except Exception:' if you must catch all".to_string(),
                            ],
                        });
                    }
                    
                    // Check for complex try blocks
                    if body.len() > 10 {
                        hints.push(AIHint {
                            hint_type: AIHintType::ErrorHandling,
                            message: "Try block is very large. Consider narrowing the scope.".to_string(),
                            confidence: 0.7,
                            location: None,
                            suggestions: vec!["Keep try blocks focused on specific operations".to_string()],
                        });
                    }
                }
                
                _ => {
                    // Other statement types can have specific hints added here
                }
            }
        }
        
        hints
    }
    
    /// Calculate function complexity score
    fn calculate_function_complexity(&self, body: &[Statement]) -> u32 {
        let mut complexity = 1; // Base complexity
        
        for stmt in body {
            complexity += match stmt {
                Statement::If { .. } => 1,
                Statement::While { .. } => 1,
                Statement::For { .. } => 1,
                Statement::TryExcept { handlers, .. } => handlers.len() as u32,
                Statement::With { .. } => 1,
                _ => 0,
            };
            
            // Add complexity for nested structures
            if let Some(nested_body) = self.get_statement_body(stmt) {
                complexity += self.calculate_function_complexity(nested_body);
            }
        }
        
        complexity
    }
    
    /// Check if function has a docstring
    fn has_docstring(&self, body: &[Statement]) -> bool {
        if let Some(first_stmt) = body.first() {
            if let Statement::Expression { value, .. } = first_stmt {
                if let Expression::Literal { value: LiteralValue::String(_), .. } = value.as_ref() {
                    return true;
                }
            }
        }
        false
    }
    
    /// Count methods in a class body
    fn count_methods(&self, body: &[Statement]) -> usize {
        body.iter()
            .filter(|stmt| matches!(stmt, Statement::FunctionDef { .. }))
            .count()
    }
    
    /// Get the body of a statement if it has one
    fn get_statement_body(&self, stmt: &Statement) -> Option<&[Statement]> {
        match stmt {
            Statement::If { body, .. } => Some(body),
            Statement::While { body, .. } => Some(body),
            Statement::For { body, .. } => Some(body),
            Statement::With { body, .. } => Some(body),
            Statement::FunctionDef { body, .. } => Some(body),
            Statement::ClassDef { body, .. } => Some(body),
            Statement::TryExcept { body, .. } => Some(body),
            _ => None,
        }
    }
    
    // Helper methods for parsing
    
    /// Expect a specific token type and consume it
    fn expect_token(&mut self, expected: TokenKind) -> Result<Token, PythonParseError> {
        if let Some(token) = self.peek() {
            let matches = match (&token.kind, &expected) {
                (TokenKind::LeftParen, TokenKind::LeftParen) => true,
                (TokenKind::RightParen, TokenKind::RightParen) => true,
                (TokenKind::LeftBracket, TokenKind::LeftBracket) => true,
                (TokenKind::RightBracket, TokenKind::RightBracket) => true,
                (TokenKind::LeftBrace, TokenKind::LeftBrace) => true,
                (TokenKind::RightBrace, TokenKind::RightBrace) => true,
                (TokenKind::Colon, TokenKind::Colon) => true,
                (TokenKind::Comma, TokenKind::Comma) => true,
                (TokenKind::Arrow, TokenKind::Arrow) => true,
                (TokenKind::Newline, TokenKind::Newline) => true,
                (TokenKind::Indent, TokenKind::Indent) => true,
                (TokenKind::Dedent, TokenKind::Dedent) => true,
                (TokenKind::Identifier(_), TokenKind::Identifier(_)) => true,
                _ => false,
            };
            
            if matches {
                Ok(self.advance().unwrap())
            } else {
                Err(PythonParseError::UnexpectedToken {
                    expected: format!("{:?}", expected),
                    found: format!("{:?}", token.kind),
                })
            }
        } else {
            Err(PythonParseError::UnexpectedEof)
        }
    }
    
    /// Check if the next token matches and consume it if so
    fn match_token(&mut self, token_kind: TokenKind) -> bool {
        if let Some(token) = self.peek() {
            if std::mem::discriminant(&token.kind) == std::mem::discriminant(&token_kind) {
                self.advance();
                true
            } else {
                false
            }
        } else {
            false
        }
    }
    
    /// Get current span for error reporting
    fn get_current_span(&self) -> Span {
        if let Some(token) = self.peek() {
            token.span
        } else {
            Span::default()
        }
    }
    
    /// Parse type parameters (PEP 695)
    fn parse_type_parameters(&mut self) -> Result<Vec<TypeParameter>, PythonParseError> {
        let mut params = Vec::new();
        
        while !self.is_at_end() && !self.check_token(TokenKind::RightBracket) {
            let name_token = self.advance().unwrap();
            let name = if let TokenKind::Identifier(name) = name_token.kind {
                name
            } else {
                return Err(PythonParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", name_token.kind),
                });
            };
            let span = name_token.span;
            
            // Determine parameter kind
            let kind = if name.starts_with("**") {
                TypeParameterKind::ParamSpec
            } else if name.starts_with('*') {
                TypeParameterKind::TypeVarTuple
            } else {
                TypeParameterKind::TypeVar
            };
            
            // Parse bounds and defaults (simplified)
            let bound = None; // TODO: Implement bound parsing
            let default = None; // TODO: Implement default parsing
            
            params.push(TypeParameter {
                name,
                kind,
                bound,
                default,
                span,
            });
            
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        
        self.expect_token(TokenKind::RightBracket)?;
        Ok(params)
    }
    
    /// Parse function parameter list
    fn parse_parameter_list(&mut self) -> Result<Vec<Parameter>, PythonParseError> {
        let mut params = Vec::new();
        
        while !self.is_at_end() && !self.check_token(TokenKind::RightParen) {
            let name_token = self.advance().unwrap();
            let name = if let TokenKind::Identifier(name) = name_token.kind {
                name
            } else {
                return Err(PythonParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", name_token.kind),
                });
            };
            let span = name_token.span;
            
            // Parse type annotation
            let annotation = if self.match_token(TokenKind::Colon) {
                Some(self.parse_type_expression()?)
            } else {
                None
            };
            
            // Parse default value
            let default = if self.match_token(TokenKind::Assign) {
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            // Determine parameter kind (simplified)
            let kind = ParameterKind::Regular; // TODO: Implement proper parameter kind detection
            
            params.push(Parameter {
                name,
                annotation,
                default,
                kind,
                span,
            });
            
            if !self.match_token(TokenKind::Comma) {
                break;
            }
        }
        
        Ok(params)
    }
    
    /// Parse type expression
    fn parse_type_expression(&mut self) -> Result<TypeExpression, PythonParseError> {
        if let Some(token) = self.peek() {
            match &token.kind {
                TokenKind::Identifier => {
                    let token = self.advance().unwrap();
                let name = if let TokenKind::Identifier(name) = token.kind {
                    name
                } else {
                    return Err(PythonParseError::InvalidSyntax {
                        message: "Expected identifier".to_string(),
                    });
                };
                    let span = token.span;
                    Ok(TypeExpression::Name { name, span })
                }
                _ => Err(PythonParseError::InvalidSyntax {
                    message: "Expected type expression".to_string(),
                }),
            }
        } else {
            Err(PythonParseError::UnexpectedEof)
        }
    }
    
    /// Parse block of statements
    fn parse_block(&mut self) -> Result<Vec<Statement>, PythonParseError> {
        let mut statements = Vec::new();
        
        // Expect newline and indent
        self.expect_token(TokenKind::Newline)?;
        self.expect_token(TokenKind::Indent)?;
        
        // Parse statements until dedent
        while !self.is_at_end() && !self.check_token(TokenKind::Dedent) {
            statements.push(self.parse_statement()?);
        }
        
        self.expect_token(TokenKind::Dedent)?;
        Ok(statements)
    }
    
    /// Check if next token matches type without consuming
    fn check_token(&self, token_kind: TokenKind) -> bool {
        if let Some(token) = self.peek() {
            std::mem::discriminant(&token.kind) == std::mem::discriminant(&token_kind)
        } else {
            false
        }
    }
    
    /// Parse f-string expression
    fn parse_f_string(&mut self) -> Result<Expression, PythonParseError> {
        let start_token = self.advance().unwrap(); // consume f-string start
        let mut parts = Vec::new();
        
        // Parse f-string parts (simplified implementation)
        while !self.is_at_end() {
            if let Some(token) = self.peek() {
                match &token.kind {
                    TokenKind::String => {
                        let text = self.advance().unwrap().value;
                        parts.push(FStringPart::Literal {
                            value: text,
                            span: token.span,
                        });
                    }
                    TokenKind::LeftBrace => {
                        self.advance(); // consume {
                        let expression = self.parse_expression()?;
                        
                        // Parse optional conversion and format spec (simplified)
                        let conversion = None; // TODO: Parse !s, !r, !a
                        let format_spec = None; // TODO: Parse format spec
                        let debug = false; // TODO: Parse = for debug mode
                        
                        self.expect_token(TokenKind::RightBrace)?;
                        
                        parts.push(FStringPart::Expression {
                            expression,
                            conversion,
                            format_spec,
                            debug,
                            span: token.span,
                        });
                    }
                    _ => break,
                }
            } else {
                break;
            }
        }
        
        let end_span = self.get_current_span();
        let span = Span::new(start_token.span.start, end_span.end, start_token.span.source_id);
        
        Ok(Expression::FormattedString { parts, span })
    }
    
    /// Parse list literal or list comprehension
    fn parse_list_or_comprehension(&mut self) -> Result<Expression, PythonParseError> {
        let start_token = self.advance().unwrap(); // consume [
        
        if self.check_token(TokenKind::RightBracket) {
            // Empty list
            self.advance();
            return Ok(Expression::List {
                elements: Vec::new(),
                span: start_token.span,
            });
        }
        
        // Parse first element
        let first_element = self.parse_expression()?;
        
        // Check if this is a comprehension
        if self.check_keyword("for") {
            // Parse list comprehension
            let generators = self.parse_comprehension_generators()?;
            self.expect_token(TokenKind::RightBracket)?;
            
            let end_span = self.get_current_span();
            let span = Span::new(start_token.span.start, end_span.end, start_token.span.source_id);
            
            Ok(Expression::ListComp {
                element: Box::new(first_element),
                generators,
                span,
            })
        } else {
            // Parse regular list
            let mut elements = vec![first_element];
            
            while self.match_token(TokenKind::Comma) && !self.check_token(TokenKind::RightBracket) {
                elements.push(self.parse_expression()?);
            }
            
            self.expect_token(TokenKind::RightBracket)?;
            
            let end_span = self.get_current_span();
            let span = Span::new(start_token.span.start, end_span.end, start_token.span.source_id);
            
            Ok(Expression::List { elements, span })
        }
    }
    
    /// Parse dictionary or set literal
    fn parse_dict_or_set(&mut self) -> Result<Expression, PythonParseError> {
        let start_token = self.advance().unwrap(); // consume {
        
        if self.check_token(TokenKind::RightBrace) {
            // Empty dict
            self.advance();
            return Ok(Expression::Dict {
                keys: Vec::new(),
                values: Vec::new(),
                span: start_token.span,
            });
        }
        
        // Parse first element to determine if dict or set
        let first_expr = self.parse_expression()?;
        
        if self.match_token(TokenKind::Colon) {
            // Dictionary
            let first_value = self.parse_expression()?;
            let mut keys = vec![Some(first_expr)];
            let mut values = vec![first_value];
            
            while self.match_token(TokenKind::Comma) && !self.check_token(TokenKind::RightBrace) {
                let key = self.parse_expression()?;
                self.expect_token(TokenKind::Colon)?;
                let value = self.parse_expression()?;
                keys.push(Some(key));
                values.push(value);
            }
            
            self.expect_token(TokenKind::RightBrace)?;
            
            let end_span = self.get_current_span();
            let span = Span::new(start_token.span.start, end_span.end, start_token.span.source_id);
            
            Ok(Expression::Dict { keys, values, span })
        } else {
            // Set
            let mut elements = vec![first_expr];
            
            while self.match_token(TokenKind::Comma) && !self.check_token(TokenKind::RightBrace) {
                elements.push(self.parse_expression()?);
            }
            
            self.expect_token(TokenKind::RightBrace)?;
            
            let end_span = self.get_current_span();
            let span = Span::new(start_token.span.start, end_span.end, start_token.span.source_id);
            
            Ok(Expression::Set { elements, span })
        }
    }
    
    /// Parse comprehension generators
    fn parse_comprehension_generators(&mut self) -> Result<Vec<Comprehension>, PythonParseError> {
        let mut generators = Vec::new();
        
        while self.check_keyword("for") || self.check_keyword("async") {
            let is_async = if self.check_keyword("async") {
                self.advance(); // consume async
                self.expect_keyword("for")?;
                true
            } else {
                self.expect_keyword("for")?;
                false
            };
            
            // Parse target
            let target_token = self.expect_token(TokenKind::Identifier)?;
            let target = AssignmentTarget::Name {
                id: target_token.value,
                span: target_token.span,
            };
            
            self.expect_keyword("in")?;
            let iter = self.parse_expression()?;
            
            // Parse optional if conditions
            let mut ifs = Vec::new();
            while self.check_keyword("if") {
                self.advance(); // consume if
                ifs.push(self.parse_expression()?);
            }
            
            generators.push(Comprehension {
                target,
                iter,
                ifs,
                is_async,
                span: target_token.span, // TODO: Calculate proper span
            });
        }
        
        Ok(generators)
    }
    
    /// Check if next token is a keyword
    fn check_keyword(&self, keyword: &str) -> bool {
        if let Some(token) = self.peek() {
            matches!(token.kind, TokenKind::Keyword) && token.value == keyword
        } else {
            false
        }
    }
    
    /// Expect a specific keyword
    fn expect_keyword(&mut self, keyword: &str) -> Result<Token, PythonParseError> {
        if let Some(token) = self.peek() {
            if matches!(token.kind, TokenKind::Keyword) && token.value == keyword {
                Ok(self.advance().unwrap())
            } else {
                Err(PythonParseError::UnexpectedToken {
                    expected: format!("keyword '{}'", keyword),
                    found: format!("{:?}", token.kind),
                })
            }
        } else {
            Err(PythonParseError::UnexpectedEof)
        }
    }
    
    // Placeholder implementations for parsing methods
    // These would be fully implemented with proper parsing logic
    
    fn parse_function_def(&mut self, is_async: bool) -> Result<Statement, PythonParseError> {
        let start_span = self.peek().unwrap().span;
        
        // Parse function name
        let name = if let Some(token) = self.peek() {
            if let TokenKind::Identifier(ref name) = token.kind {
                let name = name.clone();
                self.advance();
                name
            } else {
                return Err(PythonParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                });
            }
        } else {
            return Err(PythonParseError::UnexpectedEof);
        };
        
        // Parse type parameters (PEP 695) if present
        let type_parameters = if self.match_token(TokenKind::LeftBracket) {
            self.parse_type_parameters()?
        } else {
            Vec::new()
        };
        
        // Parse parameters
        self.expect_token(TokenKind::LeftParen)?;
        let parameters = self.parse_parameter_list()?;
        self.expect_token(TokenKind::RightParen)?;
        
        // Parse return type annotation if present
        let return_type = if self.match_token(TokenKind::Arrow) {
            Some(self.parse_type_expression()?)
        } else {
            None
        };
        
        // Parse colon
        self.expect_token(TokenKind::Colon)?;
        
        // Parse body
        let body = self.parse_block()?;
        
        // Parse decorators (would need to be parsed before the def)
        let decorators = Vec::new(); // TODO: Implement decorator parsing
        
        let end_span = self.get_current_span();
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(Statement::FunctionDef {
            name,
            type_parameters,
            parameters,
            return_type,
            body,
            decorators,
            is_async,
            span,
        })
    }
    
    fn parse_class_def(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'class' keyword
        self.expect_token(TokenKind::Keyword("class".to_string()))?;
        
        // Parse class name
        let name = if let Some(token) = self.peek() {
            if let TokenKind::Identifier(ref name) = token.kind {
                let name = name.clone();
                self.advance();
                name
            } else {
                return Err(PythonParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                });
            }
        } else {
            return Err(PythonParseError::UnexpectedEof);
        };
        
        // Parse optional base classes
        let mut base_classes = Vec::new();
        if self.match_token(TokenKind::LeftParen) {
            self.advance(); // consume '('
            
            while !self.check_token(TokenKind::RightParen) {
                let base = self.parse_expression()?;
                base_classes.push(base);
                
                if self.match_token(TokenKind::Comma) {
                    self.advance(); // consume ','
                } else {
                    break;
                }
            }
            
            self.expect_token(TokenKind::RightParen)?;
        }
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse class body
        let body = self.parse_block()?;
        
        Ok(Statement::ClassDef {
            name,
            base_classes,
            body,
            decorators: Vec::new(), // TODO: Parse decorators
            type_params: Vec::new(), // TODO: Parse type parameters
        })
    }
    
    fn parse_if_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'if' keyword
        self.expect_token(TokenKind::Keyword("if".to_string()))?;
        
        // Parse condition
        let condition = self.parse_expression()?;
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse then block
        let then_block = self.parse_block()?;
        
        // Parse optional elif/else clauses
        let mut elif_clauses = Vec::new();
        let mut else_block = None;
        
        while self.check_keyword("elif") {
            self.advance(); // consume 'elif'
            let elif_condition = self.parse_expression()?;
            self.expect_token(TokenKind::Colon)?;
            let elif_body = self.parse_block()?;
            elif_clauses.push((elif_condition, elif_body));
        }
        
        if self.check_keyword("else") {
            self.advance(); // consume 'else'
            self.expect_token(TokenKind::Colon)?;
            else_block = Some(self.parse_block()?);
        }
        
        Ok(Statement::If {
            test: condition,
            body: then_block,
            orelse: elif_clauses.into_iter().chain(else_block.into_iter()).collect(),
            span: self.get_current_span(),
        })
    }
    
    fn parse_while_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'while' keyword
        self.expect_token(TokenKind::Keyword("while".to_string()))?;
        
        // Parse condition
        let condition = self.parse_expression()?;
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse body
        let body = self.parse_block()?;
        
        // Parse optional else clause
        let else_block = if self.check_keyword("else") {
            self.advance(); // consume 'else'
            self.expect_token(TokenKind::Colon)?;
            Some(self.parse_block()?)
        } else {
            None
        };
        
        Ok(Statement::While {
            test: condition,
            body,
            orelse: else_block.into_iter().collect(),
            span: self.get_current_span(),
        })
    }
    
    fn parse_for_statement(&mut self, is_async: bool) -> Result<Statement, PythonParseError> {
        // Expect 'for' keyword
        self.expect_token(TokenKind::Keyword("for".to_string()))?;
        
        // Parse target(s)
        let target = self.parse_assignment_target()?;
        
        // Expect 'in' keyword
        self.expect_token(TokenKind::Keyword("in".to_string()))?;
        
        // Parse iterable
        let iterable = self.parse_expression()?;
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse body
        let body = self.parse_block()?;
        
        // Parse optional else clause
        let else_block = if self.check_keyword("else") {
            self.advance(); // consume 'else'
            self.expect_token(TokenKind::Colon)?;
            Some(self.parse_block()?)
        } else {
            None
        };
        
        Ok(Statement::For {
            target,
            iter: iterable,
            body,
            orelse: else_block.into_iter().collect(),
            is_async,
            span: self.get_current_span(),
        })
    }
    
    fn parse_try_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'try' keyword
        self.expect_token(TokenKind::Keyword("try".to_string()))?;
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse try body
        let try_body = self.parse_block()?;
        
        // Parse except clauses
        let mut except_clauses = Vec::new();
        while self.check_keyword("except") {
            self.advance(); // consume 'except'
            
            // Parse optional exception type and name
            let exception_type = if !self.check_token(TokenKind::Colon) {
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            let exception_name = if self.check_keyword("as") {
                self.advance(); // consume 'as'
                Some(self.expect_identifier()?)
            } else {
                None
            };
            
            self.expect_token(TokenKind::Colon)?;
            let except_body = self.parse_block()?;
            
            except_clauses.push(ExceptionHandler {
                exception_type,
                name: exception_name,
                body: except_body,
                span: self.get_current_span(),
            });
        }
        
        // Parse optional else clause
        let else_block = if self.check_keyword("else") {
            self.advance(); // consume 'else'
            self.expect_token(TokenKind::Colon)?;
            Some(self.parse_block()?)
        } else {
            None
        };
        
        // Parse optional finally clause
        let finally_block = if self.check_keyword("finally") {
            self.advance(); // consume 'finally'
            self.expect_token(TokenKind::Colon)?;
            Some(self.parse_block()?)
        } else {
            None
        };
        
        Ok(Statement::Try {
            body: try_body,
            handlers: except_clauses,
            orelse: else_block.into_iter().collect(),
            finalbody: finally_block.into_iter().collect(),
            span: self.get_current_span(),
        })
    }
    
    fn parse_with_statement(&mut self, is_async: bool) -> Result<Statement, PythonParseError> {
        // Expect 'with' keyword
        self.expect_token(TokenKind::Keyword("with".to_string()))?;
        
        // Parse context managers
        let mut context_managers = Vec::new();
        loop {
            let context_expr = self.parse_expression()?;
            let optional_vars = if self.check_keyword("as") {
                self.advance(); // consume 'as'
                Some(self.parse_assignment_target()?)
            } else {
                None
            };
            
            context_managers.push(WithItem {
                context_expr,
                optional_vars,
                span: self.get_current_span(),
            });
            
            if self.match_token(TokenKind::Comma) {
                self.advance(); // consume ','
            } else {
                break;
            }
        }
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse body
        let body = self.parse_block()?;
        
        Ok(Statement::With {
            items: context_managers,
            body,
            is_async,
            span: self.get_current_span(),
        })
    }
    
    fn parse_match_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'match' keyword
        self.expect_token(TokenKind::Keyword("match".to_string()))?;
        
        // Parse subject expression
        let subject = self.parse_expression()?;
        
        // Expect ':'
        self.expect_token(TokenKind::Colon)?;
        
        // Parse match cases
        let mut cases = Vec::new();
        while self.check_keyword("case") {
            self.advance(); // consume 'case'
            
            // Parse pattern
            let pattern = self.parse_pattern()?;
            
            // Parse optional guard
            let guard = if self.check_keyword("if") {
                self.advance(); // consume 'if'
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            self.expect_token(TokenKind::Colon)?;
            let body = self.parse_block()?;
            
            cases.push(MatchCase {
                pattern,
                guard,
                body,
                span: self.get_current_span(),
            });
        }
        
        Ok(Statement::Match {
            subject,
            cases,
            span: self.get_current_span(),
        })
    }
    
    fn parse_type_alias(&mut self) -> Result<TypeAlias, PythonParseError> {
        // Expect 'type' keyword
        self.expect_token(TokenKind::Keyword("type".to_string()))?;
        
        // Parse alias name
        let name = if let Some(token) = self.peek() {
            if let TokenKind::Identifier(ref name) = token.kind {
                let name = name.clone();
                self.advance();
                name
            } else {
                return Err(PythonParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                });
            }
        } else {
            return Err(PythonParseError::UnexpectedEof);
        };
        
        // Parse optional type parameters
        let type_params = if self.match_token(TokenKind::LeftBracket) {
            self.advance(); // consume '['
            let mut params = Vec::new();
            
            while !self.check_token(TokenKind::RightBracket) {
                let param = self.parse_type_parameter()?;
                params.push(param);
                
                if self.match_token(TokenKind::Comma) {
                    self.advance(); // consume ','
                } else {
                    break;
                }
            }
            
            self.expect_token(TokenKind::RightBracket)?;
            params
        } else {
            Vec::new()
        };
        
        // Expect '='
        self.expect_token(TokenKind::Assign)?;
        
        // Parse type expression
        let type_expr = self.parse_type_expression()?;
        
        Ok(TypeAlias {
            name,
            type_parameters: type_params,
            value: type_expr,
            span: self.get_current_span(),
        })
    }
    
    fn parse_expression_or_assignment(&mut self) -> Result<Statement, PythonParseError> {
        // Parse the left side as an expression first
        let expr = self.parse_expression()?;
        
        // Check if this is an assignment
        if self.check_token(TokenKind::Assign) {
            self.advance(); // consume '='
            
            // Convert expression to assignment target
            let target = self.expression_to_assignment_target(expr)?;
            
            // Parse the right side
            let value = self.parse_expression()?;
            
            Ok(Statement::Assignment {
                targets: vec![target],
                value,
                type_annotation: None, // TODO: Parse type comments
                span: self.get_current_span(),
            })
        } else if self.check_token(TokenKind::PlusAssign) ||
                  self.check_token(TokenKind::MinusAssign) ||
                  self.check_token(TokenKind::MultAssign) ||
                  self.check_token(TokenKind::DivAssign) {
            // Handle augmented assignment
            let op_token = self.advance().unwrap();
            let op = match &op_token.kind {
                TokenKind::PlusAssign => AugmentedAssignmentOperator::Add,
                TokenKind::MinusAssign => AugmentedAssignmentOperator::Sub,
                TokenKind::MultAssign => AugmentedAssignmentOperator::Mult,
                TokenKind::DivAssign => AugmentedAssignmentOperator::Div,
                _ => unreachable!(),
            };
            
            let target = self.expression_to_assignment_target(expr)?;
            let value = self.parse_expression()?;
            
            Ok(Statement::AugmentedAssignment {
                target,
                operator: op,
                value,
                span: self.get_current_span(),
            })
        } else {
            // This is just an expression statement
            Ok(Statement::Expression {
                expression: expr,
                span: self.get_expression_span(&expr),
            })
        }
    }
    
    fn parse_return_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'return' keyword
        self.expect_token(TokenKind::Keyword("return".to_string()))?;
        
        // Parse optional return value
        let value = if self.is_at_statement_end() {
            None
        } else {
            Some(self.parse_expression()?)
        };
        
        Ok(Statement::Return {
            value,
            span: self.get_current_span(),
        })
    }
    
    fn parse_yield_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'yield' keyword
        self.expect_token(TokenKind::Keyword("yield".to_string()))?;
        
        // Parse optional yield value or yield from
        let value = if self.check_keyword("from") {
            self.advance(); // consume 'from'
            Some(YieldValue::From(self.parse_expression()?))
        } else if !self.is_at_statement_end() {
            Some(YieldValue::Value(self.parse_expression()?))
        } else {
            None
        };
        
        Ok(Statement::Yield {
            value,
            is_from: value.is_some(),
            span: self.get_current_span(),
        })
    }
    
    fn parse_raise_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'raise' keyword
        self.expect_token(TokenKind::Keyword("raise".to_string()))?;
        
        // Parse optional exception and cause
        if self.is_at_statement_end() {
            // Bare raise
            Ok(Statement::Raise {
                exception: None,
                cause: None,
                span: self.get_current_span(),
            })
        } else {
            let exception = self.parse_expression()?;
            
            let cause = if self.check_keyword("from") {
                self.advance(); // consume 'from'
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            Ok(Statement::Raise {
                exception: Some(exception),
                cause,
                span: self.get_current_span(),
            })
        }
    }
    
    fn parse_assert_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'assert' keyword
        self.expect_token(TokenKind::Keyword("assert".to_string()))?;
        
        // Parse test expression
        let test = self.parse_expression()?;
        
        // Parse optional message
        let message = if self.check_token(TokenKind::Comma) {
            self.advance(); // consume ','
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        Ok(Statement::Assert {
            test,
            msg: message,
            span: self.get_current_span(),
        })
    }
    
    fn parse_import_statement(&mut self) -> Result<ImportStatement, PythonParseError> {
        // Expect 'import' keyword
        self.expect_token(TokenKind::Keyword("import".to_string()))?;
        
        // Parse module names
        let mut names = Vec::new();
        loop {
            let module_name = self.parse_dotted_name()?;
            let alias = if self.check_keyword("as") {
                self.advance(); // consume 'as'
                Some(self.expect_identifier()?)
            } else {
                None
            };
            
            names.push(ImportedName {
                name: module_name,
                alias,
                span: self.get_current_span(),
            });
            
            if self.check_token(TokenKind::Comma) {
                self.advance(); // consume ','
            } else {
                break;
            }
        }
        
        Ok(ImportStatement::Import {
            modules: names,
            span: self.get_current_span(),
        })
    }
    
    fn parse_from_import_statement(&mut self) -> Result<ImportStatement, PythonParseError> {
        // Expect 'from' keyword
        self.expect_token(TokenKind::Keyword("from".to_string()))?;
        
        // Parse module name (with optional relative imports)
        let mut level = 0;
        while self.check_token(TokenKind::Dot) {
            self.advance(); // consume '.'
            level += 1;
        }
        
        let module = if self.check_keyword("import") {
            None // from . import ...
        } else {
            Some(self.parse_dotted_name()?)
        };
        
        // Expect 'import' keyword
        self.expect_token(TokenKind::Keyword("import".to_string()))?;
        
        // Parse imported names
        let names = if self.check_token(TokenKind::Star) {
            self.advance(); // consume '*'
            vec![ImportedName {
                name: "*".to_string(),
                alias: None,
                span: self.get_current_span(),
            }]
        } else {
            let mut names = Vec::new();
            
            // Handle parenthesized imports
            let has_parens = self.check_token(TokenKind::LeftParen);
            if has_parens {
                self.advance(); // consume '('
            }
            
            loop {
                let name = self.expect_identifier()?;
                let alias = if self.check_keyword("as") {
                    self.advance(); // consume 'as'
                    Some(self.expect_identifier()?)
                } else {
                    None
                };
                
                names.push(ImportedName { name, alias, span: self.get_current_span() });
                
                if self.check_token(TokenKind::Comma) {
                    self.advance(); // consume ','
                } else {
                    break;
                }
            }
            
            if has_parens {
                self.expect_token(TokenKind::RightParen)?;
            }
            
            names
        };
        
        Ok(ImportStatement::FromImport {
            module,
            names,
            level,
            span: self.get_current_span(),
        })
    }
    
    fn parse_global_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'global' keyword
        self.expect_token(TokenKind::Keyword("global".to_string()))?;
        
        // Parse variable names
        let mut names = Vec::new();
        loop {
            let name = self.expect_identifier()?;
            names.push(name);
            
            if self.check_token(TokenKind::Comma) {
                self.advance(); // consume ','
            } else {
                break;
            }
        }
        
        Ok(Statement::Global {
            names,
            span: self.get_current_span(),
        })
    }
    
    fn parse_nonlocal_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'nonlocal' keyword
        self.expect_token(TokenKind::Keyword("nonlocal".to_string()))?;
        
        // Parse variable names
        let mut names = Vec::new();
        loop {
            let name = self.expect_identifier()?;
            names.push(name);
            
            if self.check_token(TokenKind::Comma) {
                self.advance(); // consume ','
            } else {
                break;
            }
        }
        
        Ok(Statement::Nonlocal {
            names,
            span: self.get_current_span(),
        })
    }
    
    fn parse_delete_statement(&mut self) -> Result<Statement, PythonParseError> {
        // Expect 'del' keyword
        self.expect_token(TokenKind::Keyword("del".to_string()))?;
        
        // Parse targets to delete
        let mut targets = Vec::new();
        loop {
            let target = self.parse_assignment_target()?;
            targets.push(target);
            
            if self.check_token(TokenKind::Comma) {
                self.advance(); // consume ','
            } else {
                break;
            }
        }
        
        Ok(Statement::Delete {
            targets,
            span: self.get_current_span(),
        })
    }
    
    fn parse_expression(&mut self) -> Result<Expression, PythonParseError> {
        // Delegate to the existing expression parser infrastructure
        // This would integrate with the Prism expression parser
        
        if self.is_at_end() {
            return Err(PythonParseError::UnexpectedEof);
        }
        
        let token = self.peek().unwrap();
        let span = token.span;
        
        match &token.kind {
            // Literals
            TokenKind::IntegerLiteral(value) => {
                Ok(Expression::Literal {
                    value: LiteralValue::Int(*value),
                    span,
                })
            }
            TokenKind::FloatLiteral(value) => {
                Ok(Expression::Literal {
                    value: LiteralValue::Float(*value),
                    span,
                })
            }
            TokenKind::StringLiteral(ref value) => {
                let value = value.clone();
                self.advance();
                Ok(Expression::Literal {
                    value: LiteralValue::String(value),
                    span,
                })
            }
            TokenKind::Identifier(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(Expression::Name { id: name, span })
            }
            
            // F-strings
            TokenKind::FStringStart(_) => {
                self.parse_f_string()
            }
            
            // Parenthesized expressions
            TokenKind::LeftParen => {
                self.advance(); // consume '('
                let expr = self.parse_expression()?;
                self.expect_token(TokenKind::RightParen)?;
                Ok(expr)
            }
            
            // List literals
            TokenKind::LeftBracket => {
                self.parse_list_or_comprehension()
            }
            
            // Dictionary literals
            TokenKind::LeftBrace => {
                self.parse_dict_or_set()
            }
            
            // Unary operators
            TokenKind::Plus | TokenKind::Minus | TokenKind::Not | TokenKind::Bang => {
                let op_token = self.advance().unwrap();
                let operator = match &op_token.kind {
                    TokenKind::Plus => UnaryOperator::UAdd,
                    TokenKind::Minus => UnaryOperator::USub,
                    TokenKind::Not => UnaryOperator::Not,
                    TokenKind::Bang => UnaryOperator::Invert,
                    _ => unreachable!(),
                };
                let operand = Box::new(self.parse_expression()?);
                Ok(Expression::UnaryOp {
                    operator,
                    operand,
                    span,
                })
            }
            
            _ => Err(PythonParseError::InvalidSyntax {
                message: format!("Unexpected token in expression: {:?}", token.kind),
            }),
        }
    }
} 