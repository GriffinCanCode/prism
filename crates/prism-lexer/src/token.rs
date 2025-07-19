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

/// A token in the Prism language
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Token {
    /// The token type and value
    pub kind: TokenKind,
    /// Source location of this token
    pub span: Span,
}

impl Token {
    /// Create a new token with basic information
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self {
            kind,
            span,
        }
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

    /// Check if this token is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::IntegerLiteral(_)
                | TokenKind::FloatLiteral(_)
                | TokenKind::StringLiteral(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Null
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
                | TokenKind::Equal
                | TokenKind::NotEqual
                | TokenKind::Less
                | TokenKind::LessEqual
                | TokenKind::Greater
                | TokenKind::GreaterEqual
                | TokenKind::Assign
                | TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::Ampersand
                | TokenKind::Pipe
                | TokenKind::Caret
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
    
    /// Identifier
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
    
    /// Newline (significant for parsing)
    #[token("\n")]
    Newline,
    
    /// Whitespace (preserved when configured)
    #[regex(r"[ \t\r]+")]
    Whitespace,
    
    // === COMMENTS & DOCUMENTATION ===
    
    /// Line comment
    LineComment(String),
    
    /// Block comment
    BlockComment(String),
    
    /// Documentation comment
    DocComment(String),
    
    // === ERROR HANDLING ===
    
    /// Lexer error token
    LexError(String),
    
    /// End of file
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module => write!(f, "module"),
            Self::Section => write!(f, "section"),
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
            Self::While => write!(f, "while"),
            Self::For => write!(f, "for"),
            Self::Loop => write!(f, "loop"),
            Self::Match => write!(f, "match"),
            Self::Case => write!(f, "case"),
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
            Self::Try => write!(f, "try"),
            Self::Catch => write!(f, "catch"),
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
            Self::Internal => write!(f, "internal"),
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
            Self::Null => write!(f, "null"),
            Self::In => write!(f, "in"),
            Self::As => write!(f, "as"),
            Self::Is => write!(f, "is"),
            Self::Identifier(name) => write!(f, "{}", name),
            Self::IntegerLiteral(value) => write!(f, "{}", value),
            Self::FloatLiteral(value) => write!(f, "{}", value),
            Self::StringLiteral(value) => write!(f, "\"{}\"", value),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Star => write!(f, "*"),
            Self::Slash => write!(f, "/"),
            Self::Percent => write!(f, "%"),
            Self::Power => write!(f, "**"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::Less => write!(f, "<"),
            Self::LessEqual => write!(f, "<="),
            Self::Greater => write!(f, ">"),
            Self::GreaterEqual => write!(f, ">="),
            Self::Assign => write!(f, "="),
            Self::PlusAssign => write!(f, "+="),
            Self::MinusAssign => write!(f, "-="),
            Self::StarAssign => write!(f, "*="),
            Self::SlashAssign => write!(f, "/="),
            Self::Ampersand => write!(f, "&"),
            Self::Pipe => write!(f, "|"),
            Self::Caret => write!(f, "^"),
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
            Self::Arrow => write!(f, "->"),
            Self::FatArrow => write!(f, "=>"),
            Self::Question => write!(f, "?"),
            Self::At => write!(f, "@"),
            Self::Newline => write!(f, "\\n"),
            Self::Whitespace => write!(f, " "),
            Self::LineComment(content) => write!(f, "//{}", content),
            Self::BlockComment(content) => write!(f, "/*{}*/", content),
            Self::DocComment(content) => write!(f, "///{}", content),
            Self::LexError(msg) => write!(f, "ERROR: {}", msg),
            Self::Eof => write!(f, "EOF"),
        }
    }
} 