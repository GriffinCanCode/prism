//! Token types for the Prism programming language
//!
//! This module defines all token types with rich semantic information
//! for AI-first development and analysis.

use logos::Logos;
use prism_common::{span::Span, symbol::Symbol};
use std::fmt;

/// Syntax style detected or specified for the source code
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SyntaxStyle {
    /// C/C++/Java/JavaScript style: braces, semicolons, parentheses
    CLike,
    /// Python/CoffeeScript style: indentation, colons
    PythonLike,
    /// Rust/Go style: explicit keywords, snake_case
    RustLike,
    /// Prism canonical style: semantic delimiters
    Canonical,
    /// Multiple styles in same file (warning)
    Mixed,
}

impl Default for SyntaxStyle {
    fn default() -> Self {
        Self::Canonical
    }
}

/// A token in the Prism language with semantic metadata
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Token {
    /// The token type and value
    pub kind: TokenKind,
    /// Source location of this token
    pub span: Span,
    /// Syntax style this token was parsed from
    pub source_style: SyntaxStyle,
    /// AI-readable semantic context
    pub semantic_context: Option<SemanticContext>,
    /// Canonical representation of this token
    pub canonical_form: Option<String>,
}

impl Token {
    /// Create a new token with basic information
    pub fn new(kind: TokenKind, span: Span, source_style: SyntaxStyle) -> Self {
        Self {
            kind,
            span,
            source_style,
            semantic_context: None,
            canonical_form: None,
        }
    }

    /// Create a token with semantic context
    pub fn with_context(
        kind: TokenKind,
        span: Span,
        source_style: SyntaxStyle,
        context: SemanticContext,
    ) -> Self {
        Self {
            kind,
            span,
            source_style,
            semantic_context: Some(context),
            canonical_form: None,
        }
    }

    /// Get canonical representation of this token
    pub fn to_canonical(&self) -> String {
        self.canonical_form.clone().unwrap_or_else(|| {
            match &self.kind {
                TokenKind::Fn => "function".to_string(),
                TokenKind::AndAnd => "and".to_string(),
                TokenKind::OrOr => "or".to_string(),
                TokenKind::Bang => "not".to_string(),
                TokenKind::Pub => "public".to_string(),
                _ => format!("{:?}", self.kind).to_lowercase(),
            }
        })
    }

    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Module
                | TokenKind::Section
                | TokenKind::Capability
                | TokenKind::Function
                | TokenKind::Fn
                | TokenKind::Type
                | TokenKind::Interface
                | TokenKind::Trait
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return
                | TokenKind::And
                | TokenKind::Or
                | TokenKind::Not
                | TokenKind::Public
                | TokenKind::Private
                | TokenKind::Effects
                | TokenKind::Async
                | TokenKind::Import
                | TokenKind::Export
        )
    }

    /// Check if this token requires documentation validation
    pub fn requires_doc_validation(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Function | TokenKind::Fn | TokenKind::Module | TokenKind::Type | TokenKind::Public | TokenKind::Pub
        )
    }

    /// Check if this token contributes to conceptual cohesion
    pub fn affects_cohesion(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Function | TokenKind::Fn | TokenKind::Type | TokenKind::Identifier(_) | TokenKind::Module
        )
    }
}

/// Comprehensive token types supporting all syntax styles and semantic features
#[derive(Debug, Clone, PartialEq, Logos)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TokenKind {
    // === STRUCTURAL KEYWORDS ===
    
    /// Module system keyword
    #[token("module")]
    #[token("mod")]
    Module,
    
    /// Section keyword for smart modules
    #[token("section")]
    Section,
    
    /// Capability keyword
    #[token("capability")]
    #[token("cap")]
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
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("loop")]
    Loop,
    #[token("match")]
    Match,
    #[token("case")]
    Case,
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
    
    // === SEMANTIC TYPE KEYWORDS ===
    
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
    
    // === EFFECT SYSTEM KEYWORDS ===
    
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
    
    // === ERROR HANDLING ===
    
    #[token("try")]
    Try,
    #[token("catch")]
    Catch,
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
    
    // === VISIBILITY ===
    
    #[token("public")]
    Public,
    #[token("pub")]
    Pub,
    #[token("private")]
    Private,
    #[token("internal")]
    Internal,
    
    // === LITERALS ===
    
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,
    
    // === MISC KEYWORDS ===
    
    #[token("in")]
    In,
    #[token("as")]
    As,
    #[token("is")]
    Is,
    
    // === COMPLEX LITERALS ===
    
    /// Identifier with semantic context
    Identifier(String),
    
    /// Integer literal
    IntegerLiteral(i64),
    
    /// Float literal
    FloatLiteral(f64),
    
    /// String literal
    StringLiteral(String),
    
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
    
    // Bitwise
    #[token("&")]
    Ampersand,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
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
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("?")]
    Question,
    #[token("@")]
    At,
    
    // === SPECIAL TOKENS ===
    
    /// Newline (significant in Python-like syntax)
    #[token("\n")]
    Newline,
    
    /// Whitespace (preserved for formatting)
    #[regex(r"[ \t\r]+")]
    Whitespace,
    
    // === COMMENTS & DOCUMENTATION ===
    
    /// Line comment
    LineComment(String),
    
    /// Block comment
    BlockComment(String),
    
    /// Documentation comment
    DocComment(String),
    
    /// Responsibility annotation
    ResponsibilityAnnotation(String),
    
    /// Effect annotation
    EffectAnnotation(String),
    
    /// Capability annotation
    CapabilityAnnotation(String),
    
    // === ERROR HANDLING ===
    
    /// Lexer error token
    LexError(String),
    
    /// End of file
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Module => write!(f, "module"),
            TokenKind::Section => write!(f, "section"),
            TokenKind::Capability => write!(f, "capability"),
            TokenKind::Function => write!(f, "function"),
            TokenKind::Fn => write!(f, "fn"),
            TokenKind::Type => write!(f, "type"),
            TokenKind::Interface => write!(f, "interface"),
            TokenKind::Trait => write!(f, "trait"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Const => write!(f, "const"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::While => write!(f, "while"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Loop => write!(f, "loop"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::Case => write!(f, "case"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Yield => write!(f, "yield"),
            TokenKind::And => write!(f, "and"),
            TokenKind::AndAnd => write!(f, "&&"),
            TokenKind::Or => write!(f, "or"),
            TokenKind::OrOr => write!(f, "||"),
            TokenKind::Not => write!(f, "not"),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::Where => write!(f, "where"),
            TokenKind::With => write!(f, "with"),
            TokenKind::Requires => write!(f, "requires"),
            TokenKind::Ensures => write!(f, "ensures"),
            TokenKind::Invariant => write!(f, "invariant"),
            TokenKind::Effects => write!(f, "effects"),
            TokenKind::Secure => write!(f, "secure"),
            TokenKind::Unsafe => write!(f, "unsafe"),
            TokenKind::Async => write!(f, "async"),
            TokenKind::Await => write!(f, "await"),
            TokenKind::Try => write!(f, "try"),
            TokenKind::Catch => write!(f, "catch"),
            TokenKind::Throw => write!(f, "throw"),
            TokenKind::Error => write!(f, "error"),
            TokenKind::Result => write!(f, "result"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::Export => write!(f, "export"),
            TokenKind::Use => write!(f, "use"),
            TokenKind::From => write!(f, "from"),
            TokenKind::Public => write!(f, "public"),
            TokenKind::Pub => write!(f, "pub"),
            TokenKind::Private => write!(f, "private"),
            TokenKind::Internal => write!(f, "internal"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Null => write!(f, "null"),
            TokenKind::In => write!(f, "in"),
            TokenKind::As => write!(f, "as"),
            TokenKind::Is => write!(f, "is"),
            TokenKind::Identifier(name) => write!(f, "{}", name),
            TokenKind::IntegerLiteral(value) => write!(f, "{}", value),
            TokenKind::FloatLiteral(value) => write!(f, "{}", value),
            TokenKind::StringLiteral(value) => write!(f, "\"{}\"", value),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::Power => write!(f, "**"),
            TokenKind::Equal => write!(f, "=="),
            TokenKind::NotEqual => write!(f, "!="),
            TokenKind::Less => write!(f, "<"),
            TokenKind::LessEqual => write!(f, "<="),
            TokenKind::Greater => write!(f, ">"),
            TokenKind::GreaterEqual => write!(f, ">="),
            TokenKind::Assign => write!(f, "="),
            TokenKind::PlusAssign => write!(f, "+="),
            TokenKind::MinusAssign => write!(f, "-="),
            TokenKind::StarAssign => write!(f, "*="),
            TokenKind::SlashAssign => write!(f, "/="),
            TokenKind::Ampersand => write!(f, "&"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::Tilde => write!(f, "~"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::DoubleColon => write!(f, "::"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::At => write!(f, "@"),
            TokenKind::Newline => write!(f, "\\n"),
            TokenKind::Whitespace => write!(f, " "),
            TokenKind::LineComment(content) => write!(f, "//{}", content),
            TokenKind::BlockComment(content) => write!(f, "/*{}*/", content),
            TokenKind::DocComment(content) => write!(f, "///{}", content),
            TokenKind::ResponsibilityAnnotation(content) => write!(f, "@responsibility {}", content),
            TokenKind::EffectAnnotation(content) => write!(f, "@effects {}", content),
            TokenKind::CapabilityAnnotation(content) => write!(f, "@capability {}", content),
            TokenKind::LexError(msg) => write!(f, "ERROR: {}", msg),
            TokenKind::Eof => write!(f, "EOF"),
        }
    }
}

/// Comprehensive semantic context for AI comprehension
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemanticContext {
    /// Business purpose of this token
    pub purpose: Option<String>,
    /// Semantic domain (e.g., "authentication", "payment")
    pub domain: Option<String>,
    /// Related concepts for AI understanding
    pub related_concepts: Vec<String>,
    /// Security implications
    pub security_implications: Vec<String>,
    /// AI-specific hints
    pub ai_hints: Vec<String>,
}

impl SemanticContext {
    /// Create a new empty semantic context
    pub fn new() -> Self {
        Self {
            purpose: None,
            domain: None,
            related_concepts: Vec::new(),
            security_implications: Vec::new(),
            ai_hints: Vec::new(),
        }
    }

    /// Create semantic context with purpose
    pub fn with_purpose(purpose: impl Into<String>) -> Self {
        Self {
            purpose: Some(purpose.into()),
            domain: None,
            related_concepts: Vec::new(),
            security_implications: Vec::new(),
            ai_hints: Vec::new(),
        }
    }

    /// Add a related concept
    pub fn add_concept(&mut self, concept: impl Into<String>) {
        self.related_concepts.push(concept.into());
    }

    /// Add a security implication
    pub fn add_security_implication(&mut self, implication: impl Into<String>) {
        self.security_implications.push(implication.into());
    }

    /// Add an AI hint
    pub fn add_ai_hint(&mut self, hint: impl Into<String>) {
        self.ai_hints.push(hint.into());
    }
}

impl Default for SemanticContext {
    fn default() -> Self {
        Self::new()
    }
} 