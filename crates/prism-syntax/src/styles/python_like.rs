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
    
    fn parse_string_literal(&self) -> Result<String, PythonParseError> {
        // TODO: Implement string literal parsing with proper escape handling
        Ok(String::new())
    }
    
    fn get_expression_span(&self, _expr: &Expression) -> Span {
        // TODO: Implement proper span calculation
        Span::default()
    }
    
    fn generate_ai_hints(&self, _statements: &[Statement]) -> Vec<AIHint> {
        // TODO: Implement AI hint generation
        Vec::new()
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
        todo!("Implement class definition parsing")
    }
    
    fn parse_if_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement if statement parsing")
    }
    
    fn parse_while_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement while statement parsing")
    }
    
    fn parse_for_statement(&mut self, _is_async: bool) -> Result<Statement, PythonParseError> {
        todo!("Implement for statement parsing")
    }
    
    fn parse_try_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement try statement parsing")
    }
    
    fn parse_with_statement(&mut self, _is_async: bool) -> Result<Statement, PythonParseError> {
        todo!("Implement with statement parsing")
    }
    
    fn parse_match_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement match statement parsing")
    }
    
    fn parse_type_alias(&mut self) -> Result<TypeAlias, PythonParseError> {
        todo!("Implement type alias parsing")
    }
    
    fn parse_expression_or_assignment(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement expression/assignment parsing")
    }
    
    fn parse_return_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement return statement parsing")
    }
    
    fn parse_yield_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement yield statement parsing")
    }
    
    fn parse_raise_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement raise statement parsing")
    }
    
    fn parse_assert_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement assert statement parsing")
    }
    
    fn parse_import_statement(&mut self) -> Result<ImportStatement, PythonParseError> {
        todo!("Implement import statement parsing")
    }
    
    fn parse_from_import_statement(&mut self) -> Result<ImportStatement, PythonParseError> {
        todo!("Implement from import statement parsing")
    }
    
    fn parse_global_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement global statement parsing")
    }
    
    fn parse_nonlocal_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement nonlocal statement parsing")
    }
    
    fn parse_delete_statement(&mut self) -> Result<Statement, PythonParseError> {
        todo!("Implement delete statement parsing")
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