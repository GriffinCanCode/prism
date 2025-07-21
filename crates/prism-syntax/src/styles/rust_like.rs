//! Parser for Rust-like syntax (Rust/Swift style).
//!
//! Key characteristics:
//! - Expression-oriented syntax
//! - Pattern matching with match expressions
//! - Ownership and borrowing annotations (&, &mut)
//! - Result/Option types and ? operator

use crate::{
    styles::{StyleParser, StyleConfig, ParserCapabilities, ErrorRecoveryLevel, traits::ConfigError},
    detection::SyntaxStyle,
};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use thiserror::Error;

#[derive(Debug)]
pub struct RustLikeParser {
    config: RustLikeConfig,
    current: usize,
    tokens: Vec<Token>,
}

#[derive(Debug, Clone)]
pub struct RustLikeConfig {
    /// Allow trailing commas in all contexts
    pub allow_trailing_commas: bool,
    
    /// Require explicit return statements (vs expression-based returns)
    pub require_explicit_returns: bool,
    
    /// Enable ownership annotations parsing
    pub parse_ownership: bool,
    
    /// Enable lifetime annotations parsing
    pub parse_lifetimes: bool,
}

#[derive(Debug)]
pub struct RustLikeSyntax {
    pub modules: Vec<RustLikeModule>,
    pub functions: Vec<RustLikeFunction>,
    pub statements: Vec<RustLikeStatement>,
}

impl Default for RustLikeSyntax {
    fn default() -> Self {
        Self {
            modules: Vec::new(),
            functions: Vec::new(),
            statements: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct RustLikeModule {
    pub name: String,
    pub body: Vec<RustLikeItem>,
    pub span: Span,
}

#[derive(Debug)]
pub struct RustLikeFunction {
    pub name: String,
    pub parameters: Vec<RustLikeParameter>,
    pub return_type: Option<RustLikeType>,
    pub body: Box<RustLikeExpression>,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_const: bool,
    pub is_extern: bool,
    pub abi: Option<String>, // For extern functions like "C"
    pub visibility: Visibility,
    pub attributes: Vec<String>, // For attributes like #[naked], #[inline], etc.
    pub generics: Vec<String>, // Generic type parameters
    pub where_clause: Option<String>, // Where clauses for generic constraints
    pub span: Span,
}

#[derive(Debug)]
pub struct RustLikeParameter {
    pub name: String,
    pub param_type: RustLikeType,
    pub is_mutable: bool,
}

#[derive(Debug)]
pub enum RustLikeItem {
    Function(RustLikeFunction),
    Statement(RustLikeStatement),
    Struct(RustLikeStruct),
    Enum(RustLikeEnum),
    Impl(RustLikeImpl),
}

#[derive(Debug)]
pub struct RustLikeStruct {
    pub name: String,
    pub fields: Vec<RustLikeField>,
    pub span: Span,
}

#[derive(Debug)]
pub struct RustLikeEnum {
    pub name: String,
    pub variants: Vec<RustLikeVariant>,
    pub span: Span,
}

#[derive(Debug)]
pub struct RustLikeImpl {
    pub target: RustLikeType,
    pub trait_name: Option<String>,
    pub items: Vec<RustLikeImplItem>,
    pub span: Span,
}

#[derive(Debug)]
pub struct RustLikeField {
    pub name: String,
    pub field_type: RustLikeType,
    pub visibility: Visibility,
}

#[derive(Debug)]
pub struct RustLikeVariant {
    pub name: String,
    pub data: Option<RustLikeVariantData>,
}

#[derive(Debug)]
pub enum RustLikeVariantData {
    Tuple(Vec<RustLikeType>),
    Struct(Vec<RustLikeField>),
}

#[derive(Debug)]
pub enum RustLikeImplItem {
    Function(RustLikeFunction),
}

#[derive(Debug)]
pub enum Visibility {
    Public,
    Private,
    Crate,
}

#[derive(Debug)]
pub enum RustLikeStatement {
    Let {
        pattern: RustLikePattern,
        value: Option<RustLikeExpression>,
        is_mutable: bool,
    },
    Expression(RustLikeExpression),
    // Support for const statements in const contexts
    Const {
        name: String,
        value: RustLikeExpression,
        const_type: Option<RustLikeType>,
    },
    // Support for static statements
    Static {
        name: String,
        value: RustLikeExpression,
        static_type: RustLikeType,
        is_mutable: bool,
    },
}

#[derive(Debug)]
pub enum RustLikeExpression {
    Identifier(String),
    Literal(RustLikeLiteral),
    Block(Vec<RustLikeStatement>, Option<Box<RustLikeExpression>>),
    If {
        condition: Box<RustLikeExpression>,
        then_branch: Box<RustLikeExpression>,
        else_branch: Option<Box<RustLikeExpression>>,
    },
    // Enhanced if let with let chains support
    IfLet {
        pattern: RustLikePattern,
        value: Box<RustLikeExpression>,
        then_branch: Box<RustLikeExpression>,
        else_branch: Option<Box<RustLikeExpression>>,
    },
    // Let chains for combining multiple conditions
    LetChain {
        conditions: Vec<RustLikeLetCondition>,
        body: Box<RustLikeExpression>,
        else_branch: Option<Box<RustLikeExpression>>,
    },
    Match {
        expr: Box<RustLikeExpression>,
        arms: Vec<RustLikeMatchArm>,
    },
    Call {
        function: Box<RustLikeExpression>,
        arguments: Vec<RustLikeExpression>,
    },
    Binary {
        left: Box<RustLikeExpression>,
        operator: String,
        right: Box<RustLikeExpression>,
    },
    Unary {
        operator: String,
        operand: Box<RustLikeExpression>,
    },
    Reference {
        is_mutable: bool,
        expr: Box<RustLikeExpression>,
    },
    Dereference(Box<RustLikeExpression>),
    TryOperator(Box<RustLikeExpression>),
    FieldAccess {
        object: Box<RustLikeExpression>,
        field: String,
    },
    MethodCall {
        receiver: Box<RustLikeExpression>,
        method: String,
        arguments: Vec<RustLikeExpression>,
    },
    Array(Vec<RustLikeExpression>),
    Tuple(Vec<RustLikeExpression>),
    Struct {
        name: String,
        fields: Vec<(String, RustLikeExpression)>,
    },
    // Async/await support
    Async(Box<RustLikeExpression>),
    Await(Box<RustLikeExpression>),
    // Range expressions
    Range {
        start: Option<Box<RustLikeExpression>>,
        end: Option<Box<RustLikeExpression>>,
        inclusive: bool,
    },
    // Closure expressions
    Closure {
        parameters: Vec<RustLikeParameter>,
        body: Box<RustLikeExpression>,
        is_async: bool,
        is_move: bool,
    },
    // Loop expressions
    Loop {
        body: Box<RustLikeExpression>,
        label: Option<String>,
    },
    While {
        condition: Box<RustLikeExpression>,
        body: Box<RustLikeExpression>,
        label: Option<String>,
    },
    For {
        pattern: RustLikePattern,
        iterable: Box<RustLikeExpression>,
        body: Box<RustLikeExpression>,
        label: Option<String>,
    },
    // Break and continue with optional values and labels
    Break {
        label: Option<String>,
        value: Option<Box<RustLikeExpression>>,
    },
    Continue {
        label: Option<String>,
    },
    // Return with optional value
    Return(Option<Box<RustLikeExpression>>),
    // Unsafe expressions
    Unsafe(Box<RustLikeExpression>),
    // Const expressions
    Const(Box<RustLikeExpression>),
    // Raw pointer operations
    RawPointer {
        is_mutable: bool,
        expr: Box<RustLikeExpression>,
    },
    // Cast expressions
    Cast {
        expr: Box<RustLikeExpression>,
        target_type: RustLikeType,
    },
}

/// Condition in let chains
#[derive(Debug)]
pub enum RustLikeLetCondition {
    Let { pattern: RustLikePattern, value: RustLikeExpression },
    Expression(RustLikeExpression),
}

#[derive(Debug)]
pub struct RustLikeMatchArm {
    pub pattern: RustLikePattern,
    pub guard: Option<RustLikeExpression>,
    pub body: RustLikeExpression,
}

#[derive(Debug)]
pub enum RustLikePattern {
    Identifier(String),
    Literal(RustLikeLiteral),
    Wildcard,
    Tuple(Vec<RustLikePattern>),
    Struct {
        name: String,
        fields: Vec<(String, RustLikePattern)>,
    },
}

#[derive(Debug)]
pub enum RustLikeLiteral {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Unit, // ()
}

#[derive(Debug)]
pub enum RustLikeType {
    Identifier(String),
    Reference {
        is_mutable: bool,
        inner: Box<RustLikeType>,
    },
    Tuple(Vec<RustLikeType>),
    Array {
        element_type: Box<RustLikeType>,
        size: Option<usize>,
    },
    Function {
        parameters: Vec<RustLikeType>,
        return_type: Box<RustLikeType>,
    },
    Generic {
        base: String,
        args: Vec<RustLikeType>,
    },
    // Impl Trait syntax
    ImplTrait {
        traits: Vec<String>,
    },
    // Dyn Trait syntax for trait objects
    DynTrait {
        traits: Vec<String>,
    },
    // Raw pointer types
    RawPointer {
        is_mutable: bool,
        inner: Box<RustLikeType>,
    },
    // Associated types
    Associated {
        base: Box<RustLikeType>,
        associated: String,
    },
    // Higher-ranked trait bounds (HRTB)
    HigherRanked {
        lifetimes: Vec<String>,
        inner: Box<RustLikeType>,
    },
    // Never type
    Never,
    // Unit type
    Unit,
    // Slice type
    Slice(Box<RustLikeType>),
    // Path types with module resolution
    Path {
        segments: Vec<String>,
        generics: Vec<RustLikeType>,
    },
}

#[derive(Debug, Error)]
pub enum RustLikeError {
    /// Unexpected token
    #[error("Unexpected token: expected {expected}, found {found:?} at {span:?}")]
    UnexpectedToken { expected: String, found: String, span: Span },
    
    /// Missing semicolon (when required)
    #[error("Missing semicolon at {span:?}")]
    MissingSemicolon { span: Span },
    
    /// Invalid pattern
    #[error("Invalid pattern at {span:?}")]
    InvalidPattern { span: Span },
    
    /// Invalid type annotation
    #[error("Invalid type annotation at {span:?}")]
    InvalidType { span: Span },
    
    /// Unexpected end of input
    #[error("Unexpected end of input")]
    UnexpectedEof,
}

impl Default for RustLikeConfig {
    fn default() -> Self {
        Self {
            allow_trailing_commas: true,
            require_explicit_returns: false, // Rust is expression-oriented
            parse_ownership: true,
            parse_lifetimes: false, // Simplified for now
        }
    }
}

impl StyleConfig for RustLikeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl RustLikeParser {
    /// Check if we're at the end of tokens
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }
    
    /// Get the current token
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current)
    }
    
    /// Advance to the next token
    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.tokens.get(self.current - 1)
    }
    
    /// Check if current token matches expected kind
    fn check(&self, kind: &TokenKind) -> bool {
        self.peek().map(|t| &t.kind) == Some(kind)
    }
    
    /// Consume a token of expected kind or return error
    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token, RustLikeError> {
        if self.check(&kind) {
            Ok(self.advance().unwrap())
        } else {
            let found = self.peek()
                .map(|t| format!("{:?}", t.kind))
                .unwrap_or_else(|| "EOF".to_string());
            let span = self.peek().map(|t| t.span).unwrap_or_else(Span::dummy);
            
            Err(RustLikeError::UnexpectedToken {
                expected: message.to_string(),
                found,
                span,
            })
        }
    }
    
    /// Parse the entire program
    fn parse_program(&mut self) -> Result<RustLikeSyntax, RustLikeError> {
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        let mut statements = Vec::new();
        
        while !self.is_at_end() {
            if self.check(&TokenKind::Module) {
                modules.push(self.parse_module()?);
            } else if self.check(&TokenKind::Function) {
                functions.push(self.parse_function()?);
            } else {
                statements.push(self.parse_statement()?);
            }
        }
        
        Ok(RustLikeSyntax {
            modules,
            functions,
            statements,
        })
    }
    
    /// Parse a module
    fn parse_module(&mut self) -> Result<RustLikeModule, RustLikeError> {
        let start_span = self.consume(TokenKind::Module, "Expected 'mod'")?.span;
        let name = self.parse_identifier()?;
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after module name")?;
        
        let mut body = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            body.push(self.parse_item()?);
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after module body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(RustLikeModule { name, body, span })
    }
    
    /// Parse an item (function, struct, enum, etc.)
    fn parse_item(&mut self) -> Result<RustLikeItem, RustLikeError> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Function) => Ok(RustLikeItem::Function(self.parse_function()?)),
            Some(TokenKind::Identifier(name)) if name == "struct" => {
                self.advance(); // consume 'struct'
                Ok(RustLikeItem::Struct(self.parse_struct()?))
            }
            Some(TokenKind::Identifier(name)) if name == "enum" => {
                self.advance(); // consume 'enum'
                Ok(RustLikeItem::Enum(self.parse_enum()?))
            }
            Some(TokenKind::Identifier(name)) if name == "impl" => {
                self.advance(); // consume 'impl'
                Ok(RustLikeItem::Impl(self.parse_impl()?))
            }
            _ => Ok(RustLikeItem::Statement(self.parse_statement()?)),
        }
    }
    
    /// Parse a function with Rust-like syntax
    fn parse_function(&mut self) -> Result<RustLikeFunction, RustLikeError> {
        let start_span = self.consume(TokenKind::Function, "Expected 'fn'")?.span;
        let name = self.parse_identifier()?;
        
        // Parse parameters
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        let mut parameters = Vec::new();
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            parameters.push(self.parse_parameter()?);
            if !self.check(&TokenKind::RightParen) {
                self.consume(TokenKind::Comma, "Expected ',' between parameters")?;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        
        // Parse return type
        let return_type = if self.check(&TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        // Parse function body as block expression
        let body = Box::new(self.parse_block_expression()?);
        
        let span = Span::new(start_span.start, body.span().end, start_span.source_id);
        
        Ok(RustLikeFunction {
            name,
            parameters,
            return_type,
            body,
            is_async: false, // Simplified
            is_unsafe: false, // Simplified
            is_const: false, // Simplified
            is_extern: false, // Simplified
            abi: None, // Simplified
            visibility: Visibility::Private, // Simplified
            attributes: Vec::new(), // Simplified
            generics: Vec::new(), // Simplified
            where_clause: None, // Simplified
            span,
        })
    }
    
    /// Parse a struct
    fn parse_struct(&mut self) -> Result<RustLikeStruct, RustLikeError> {
        let start_span = self.peek().ok_or(RustLikeError::UnexpectedEof)?.span;
        let name = self.parse_identifier()?;
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after struct name")?;
        
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            fields.push(self.parse_struct_field()?);
            if !self.check(&TokenKind::RightBrace) {
                self.consume(TokenKind::Comma, "Expected ',' between struct fields")?;
            }
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after struct body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(RustLikeStruct { name, fields, span })
    }
    
    /// Parse an enum
    fn parse_enum(&mut self) -> Result<RustLikeEnum, RustLikeError> {
        let start_span = self.peek().ok_or(RustLikeError::UnexpectedEof)?.span;
        let name = self.parse_identifier()?;
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after enum name")?;
        
        let mut variants = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            variants.push(self.parse_enum_variant()?);
            if !self.check(&TokenKind::RightBrace) {
                self.consume(TokenKind::Comma, "Expected ',' between enum variants")?;
            }
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after enum body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(RustLikeEnum { name, variants, span })
    }
    
    /// Parse an impl block
    fn parse_impl(&mut self) -> Result<RustLikeImpl, RustLikeError> {
        let start_span = self.peek().ok_or(RustLikeError::UnexpectedEof)?.span;
        let target = self.parse_type()?;
        
        // Optional trait implementation
        let trait_name = if self.check(&TokenKind::For) {
            self.advance();
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after impl target")?;
        
        let mut items = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            if self.check(&TokenKind::Function) {
                items.push(RustLikeImplItem::Function(self.parse_function()?));
            } else {
                // Skip unknown items for now
                self.advance();
            }
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after impl body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(RustLikeImpl {
            target,
            trait_name,
            items,
            span,
        })
    }
    
    /// Parse a statement
    fn parse_statement(&mut self) -> Result<RustLikeStatement, RustLikeError> {
        if self.check(&TokenKind::Let) {
            self.advance();
            
            let is_mutable = if let Some(TokenKind::Identifier(name)) = self.peek().map(|t| &t.kind) {
                if name == "mut" {
                    self.advance();
                    true
                } else {
                    false
                }
            } else {
                false
            };
            
            let pattern = self.parse_pattern()?;
            
            let value = if self.check(&TokenKind::Assign) {
                self.advance();
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            // Optional semicolon for let statements
            if self.check(&TokenKind::Semicolon) {
                self.advance();
            }
            
            Ok(RustLikeStatement::Let {
                pattern,
                value,
                is_mutable,
            })
        } else if self.check(&TokenKind::Const) {
            self.advance();
            let name = self.parse_identifier()?;
            self.consume(TokenKind::Assign, "Expected '=' after const name")?;
            let value = self.parse_expression()?;
            let const_type = if self.check(&TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            Ok(RustLikeStatement::Const { name, value, const_type })
        } else if self.check(&TokenKind::Static) {
            self.advance();
            let name = self.parse_identifier()?;
            self.consume(TokenKind::Assign, "Expected '=' after static name")?;
            let value = self.parse_expression()?;
            let static_type = self.parse_type()?;
            let is_mutable = if self.check(&TokenKind::Mut) {
                self.advance();
                true
            } else {
                false
            };
            Ok(RustLikeStatement::Static { name, value, static_type, is_mutable })
        } else {
            let expr = self.parse_expression()?;
            
            // Optional semicolon for expression statements
            if self.check(&TokenKind::Semicolon) {
                self.advance();
            }
            
            Ok(RustLikeStatement::Expression(expr))
        }
    }
    
    /// Parse an expression
    fn parse_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.parse_logical_or()
    }
    
    /// Parse logical OR expressions
    fn parse_logical_or(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_logical_and()?;
        
        while self.check(&TokenKind::OrOr) {
            self.advance(); // consume '||'
            let right = self.parse_logical_and()?;
            expr = RustLikeExpression::Binary {
                left: Box::new(expr),
                operator: "||".to_string(),
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse logical AND expressions
    fn parse_logical_and(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_equality()?;
        
        while self.check(&TokenKind::AndAnd) {
            self.advance(); // consume '&&'
            let right = self.parse_equality()?;
            expr = RustLikeExpression::Binary {
                left: Box::new(expr),
                operator: "&&".to_string(),
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse equality expressions
    fn parse_equality(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_comparison()?;
        
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Equal) | Some(TokenKind::NotEqual)) {
            let operator = match &self.advance().unwrap().kind {
                TokenKind::Equal => "==".to_string(),
                TokenKind::NotEqual => "!=".to_string(),
                _ => unreachable!(),
            };
            let right = self.parse_comparison()?;
            expr = RustLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse comparison expressions
    fn parse_comparison(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_addition()?;
        
        while matches!(self.peek().map(|t| &t.kind), 
            Some(TokenKind::Less) | Some(TokenKind::Greater) | 
            Some(TokenKind::LessEqual) | Some(TokenKind::GreaterEqual)) {
            let operator = match &self.advance().unwrap().kind {
                TokenKind::Less => "<".to_string(),
                TokenKind::LessEqual => "<=".to_string(),
                TokenKind::Greater => ">".to_string(),
                TokenKind::GreaterEqual => ">=".to_string(),
                _ => unreachable!(),
            };
            let right = self.parse_addition()?;
            expr = RustLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse addition and subtraction
    fn parse_addition(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_multiplication()?;
        
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Plus) | Some(TokenKind::Minus)) {
            let operator = match &self.advance().unwrap().kind {
                TokenKind::Plus => "+".to_string(),
                TokenKind::Minus => "-".to_string(),
                _ => unreachable!(),
            };
            let right = self.parse_multiplication()?;
            expr = RustLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse multiplication and division
    fn parse_multiplication(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_unary()?;
        
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Star) | Some(TokenKind::Slash)) {
            let operator = match &self.advance().unwrap().kind {
                TokenKind::Star => "*".to_string(),
                TokenKind::Slash => "/".to_string(),
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            expr = RustLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse unary expressions
    fn parse_unary(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Minus) | Some(TokenKind::Not) => {
                let operator = match &self.advance().unwrap().kind {
                    TokenKind::Minus => "-".to_string(),
                    TokenKind::Not => "not".to_string(),
                    _ => unreachable!(),
                };
                let operand = self.parse_unary()?;
                Ok(RustLikeExpression::Unary {
                    operator,
                    operand: Box::new(operand),
                })
            }
            Some(TokenKind::Ampersand) => {
                self.advance();
                let is_mutable = if let Some(TokenKind::Identifier(name)) = self.peek().map(|t| &t.kind) {
                    if name == "mut" {
                        self.advance();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                let expr = self.parse_unary()?;
                Ok(RustLikeExpression::Reference {
                    is_mutable,
                    expr: Box::new(expr),
                })
            }
            Some(TokenKind::Star) => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(RustLikeExpression::Dereference(Box::new(expr)))
            }
            _ => self.parse_postfix(),
        }
    }
    
    /// Parse postfix expressions (calls, field access, etc.)
    fn parse_postfix(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut expr = self.parse_primary()?;
        
        loop {
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::LeftParen) => {
                    // Function call
                    self.advance();
                    let mut arguments = Vec::new();
                    
                    while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                        arguments.push(self.parse_expression()?);
                        if !self.check(&TokenKind::RightParen) {
                            self.consume(TokenKind::Comma, "Expected ',' between arguments")?;
                        }
                    }
                    
                    self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                    
                    expr = RustLikeExpression::Call {
                        function: Box::new(expr),
                        arguments,
                    };
                }
                Some(TokenKind::Dot) => {
                    // Field access or method call
                    self.advance();
                    let field_or_method = self.parse_identifier()?;
                    
                    if self.check(&TokenKind::LeftParen) {
                        // Method call
                        self.advance();
                        let mut arguments = Vec::new();
                        
                        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                            arguments.push(self.parse_expression()?);
                            if !self.check(&TokenKind::RightParen) {
                                self.consume(TokenKind::Comma, "Expected ',' between arguments")?;
                            }
                        }
                        
                        self.consume(TokenKind::RightParen, "Expected ')' after method arguments")?;
                        
                        expr = RustLikeExpression::MethodCall {
                            receiver: Box::new(expr),
                            method: field_or_method,
                            arguments,
                        };
                    } else {
                        // Field access
                        expr = RustLikeExpression::FieldAccess {
                            object: Box::new(expr),
                            field: field_or_method,
                        };
                    }
                }
                Some(TokenKind::Question) => {
                    // Try operator
                    self.advance();
                    expr = RustLikeExpression::TryOperator(Box::new(expr));
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }
    
    /// Parse primary expressions with modern Rust features
    fn parse_primary(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::If) => {
                self.advance();
                // Check for if let or let chains
                if self.check(&TokenKind::Let) {
                    self.parse_if_let_or_chain()
                } else {
                    self.parse_if_expression()
                }
            },
            Some(TokenKind::Match) => self.parse_match_expression(),
            Some(TokenKind::LeftBrace) => self.parse_block_expression(),
            Some(TokenKind::Loop) => self.parse_loop_expression(),
            Some(TokenKind::While) => self.parse_while_expression(),
            Some(TokenKind::For) => self.parse_for_expression(),
            Some(TokenKind::Break) => self.parse_break_expression(),
            Some(TokenKind::Continue) => self.parse_continue_expression(),
            Some(TokenKind::Return) => self.parse_return_expression(),
            Some(TokenKind::Unsafe) => self.parse_unsafe_expression(),
            Some(TokenKind::Async) => self.parse_async_expression(),
            Some(TokenKind::Move) => self.parse_closure_expression(true),
            Some(TokenKind::Pipe) => self.parse_closure_expression(false), // |args| body
            Some(TokenKind::LeftParen) => {
                self.advance();
                
                // Check for unit type
                if self.check(&TokenKind::RightParen) {
                    self.advance();
                    return Ok(RustLikeExpression::Literal(RustLikeLiteral::Unit));
                }
                
                // Parse tuple or parenthesized expression
                let mut elements = vec![self.parse_expression()?];
                
                while self.check(&TokenKind::Comma) {
                    self.advance();
                    if self.check(&TokenKind::RightParen) {
                        break; // Trailing comma
                    }
                    elements.push(self.parse_expression()?);
                }
                
                self.consume(TokenKind::RightParen, "Expected ')' after expression")?;
                
                if elements.len() == 1 {
                    Ok(elements.into_iter().next().unwrap())
                } else {
                    Ok(RustLikeExpression::Tuple(elements))
                }
            }
            Some(TokenKind::LeftBracket) => {
                self.advance();
                let mut elements = Vec::new();
                
                while !self.check(&TokenKind::RightBracket) && !self.is_at_end() {
                    elements.push(self.parse_expression()?);
                    if !self.check(&TokenKind::RightBracket) {
                        self.consume(TokenKind::Comma, "Expected ',' between array elements")?;
                    }
                }
                
                self.consume(TokenKind::RightBracket, "Expected ']' after array")?;
                Ok(RustLikeExpression::Array(elements))
            }
            Some(TokenKind::Identifier(_)) => {
                let name = self.parse_identifier()?;
                
                // Check for struct literal
                if self.check(&TokenKind::LeftBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    
                    while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                        let field_name = self.parse_identifier()?;
                        self.consume(TokenKind::Colon, "Expected ':' after field name")?;
                        let field_value = self.parse_expression()?;
                        fields.push((field_name, field_value));
                        
                        if !self.check(&TokenKind::RightBrace) {
                            self.consume(TokenKind::Comma, "Expected ',' between struct fields")?;
                        }
                    }
                    
                    self.consume(TokenKind::RightBrace, "Expected '}' after struct literal")?;
                    
                    Ok(RustLikeExpression::Struct { name, fields })
                } else {
                    Ok(RustLikeExpression::Identifier(name))
                }
            }
            Some(TokenKind::StringLiteral(s)) => {
                let value = s.clone();
                self.advance();
                Ok(RustLikeExpression::Literal(RustLikeLiteral::String(value)))
            }
            Some(TokenKind::IntegerLiteral(i)) => {
                let value = *i;
                self.advance();
                Ok(RustLikeExpression::Literal(RustLikeLiteral::Integer(value)))
            }
            Some(TokenKind::FloatLiteral(f)) => {
                let value = *f;
                self.advance();
                Ok(RustLikeExpression::Literal(RustLikeLiteral::Float(value)))
            }
            Some(TokenKind::BooleanLiteral(b)) => {
                let value = *b;
                self.advance();
                Ok(RustLikeExpression::Literal(RustLikeLiteral::Boolean(value)))
            }
            Some(TokenKind::DotDot) => self.parse_range_expression(),
            Some(TokenKind::DotDotEqual) => self.parse_range_expression(),
            _ => Err(RustLikeError::UnexpectedToken {
                expected: "expression".to_string(),
                found: format!("{:?}", self.peek().map(|t| &t.kind)),
                span: self.peek().map(|t| t.span).unwrap_or_else(Span::dummy),
            }),
        }
    }

    /// Parse if let or let chains (Rust 2024 feature)
    fn parse_if_let_or_chain(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let mut conditions = Vec::new();
        
        // Parse the first let condition
        self.consume(TokenKind::Let, "Expected 'let'")?;
        let pattern = self.parse_pattern()?;
        self.consume(TokenKind::Assign, "Expected '=' after pattern")?;
        let value = self.parse_expression()?;
        conditions.push(RustLikeLetCondition::Let { pattern, value });
        
        // Parse additional conditions with && (let chains)
        while self.check(&TokenKind::AndAnd) {
            self.advance();
            
            if self.check(&TokenKind::Let) {
                self.advance();
                let pattern = self.parse_pattern()?;
                self.consume(TokenKind::Assign, "Expected '=' after pattern")?;
                let value = self.parse_expression()?;
                conditions.push(RustLikeLetCondition::Let { pattern, value });
            } else {
                // Regular boolean condition
                let expr = self.parse_logical_and()?;
                conditions.push(RustLikeLetCondition::Expression(expr));
            }
        }
        
        // Parse the body
        let body = Box::new(self.parse_block_expression()?);
        
        // Parse optional else
        let else_branch = if self.check(&TokenKind::Else) {
            self.advance();
            Some(Box::new(self.parse_block_expression()?))
        } else {
            None
        };
        
        // If it's a simple if let with no chains, use IfLet variant
        if conditions.len() == 1 {
            if let RustLikeLetCondition::Let { pattern, value } = conditions.into_iter().next().unwrap() {
                Ok(RustLikeExpression::IfLet {
                    pattern,
                    value: Box::new(value),
                    then_branch: body,
                    else_branch,
                })
            } else {
                unreachable!("First condition should be Let")
            }
        } else {
            // Use LetChain for multiple conditions
            Ok(RustLikeExpression::LetChain {
                conditions,
                body,
                else_branch,
            })
        }
    }

    /// Parse range expressions (.. and ..=)
    fn parse_range_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        let start = None; // For now, simplified to handle .. and ..= without start
        
        let inclusive = if self.check(&TokenKind::DotDotEqual) {
            self.advance();
            true
        } else {
            self.advance(); // consume ..
            false
        };
        
        let end = if !self.is_at_end() && !matches!(self.peek().map(|t| &t.kind), 
            Some(TokenKind::RightParen) | Some(TokenKind::RightBracket) | Some(TokenKind::Comma)) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        
        Ok(RustLikeExpression::Range {
            start,
            end,
            inclusive,
        })
    }

    /// Parse closure expressions
    fn parse_closure_expression(&mut self, is_move: bool) -> Result<RustLikeExpression, RustLikeError> {
        if is_move {
            // Already consumed 'move', now expect |
            self.consume(TokenKind::Pipe, "Expected '|' after 'move'")?;
        } else {
            // Consume the opening |
            self.consume(TokenKind::Pipe, "Expected '|'")?;
        }
        
        let mut parameters = Vec::new();
        while !self.check(&TokenKind::Pipe) && !self.is_at_end() {
            parameters.push(self.parse_parameter()?);
            if !self.check(&TokenKind::Pipe) {
                self.consume(TokenKind::Comma, "Expected ',' between closure parameters")?;
            }
        }
        
        self.consume(TokenKind::Pipe, "Expected '|' after closure parameters")?;
        
        let is_async = if self.check(&TokenKind::Async) {
            self.advance();
            true
        } else {
            false
        };
        
        let body = Box::new(self.parse_expression()?);
        
        Ok(RustLikeExpression::Closure {
            parameters,
            body,
            is_async,
            is_move,
        })
    }

    /// Parse async expressions
    fn parse_async_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'async'
        
        if self.check(&TokenKind::LeftBrace) {
            // async block
            let block = self.parse_block_expression()?;
            Ok(RustLikeExpression::Async(Box::new(block)))
        } else if self.check(&TokenKind::Move) {
            // async move closure
            self.parse_closure_expression(true)
        } else {
            // async closure
            self.parse_closure_expression(false)
        }
    }

    /// Parse unsafe expressions
    fn parse_unsafe_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'unsafe'
        let expr = self.parse_primary()?;
        Ok(RustLikeExpression::Unsafe(Box::new(expr)))
    }

    /// Parse loop expressions with labels
    fn parse_loop_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'loop'
        
        let label = None; // Simplified - would need to handle 'label: loop
        let body = Box::new(self.parse_block_expression()?);
        
        Ok(RustLikeExpression::Loop { body, label })
    }

    /// Parse while expressions with labels
    fn parse_while_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'while'
        
        let condition = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_block_expression()?);
        let label = None; // Simplified
        
        Ok(RustLikeExpression::While { condition, body, label })
    }

    /// Parse for expressions with labels
    fn parse_for_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'for'
        
        let pattern = self.parse_pattern()?;
        self.consume(TokenKind::In, "Expected 'in' after for pattern")?;
        let iterable = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_block_expression()?);
        let label = None; // Simplified
        
        Ok(RustLikeExpression::For { pattern, iterable, body, label })
    }

    /// Parse break expressions with optional labels and values
    fn parse_break_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'break'
        
        let label = None; // Simplified - would parse 'label if present
        let value = if !self.is_at_end() && !matches!(self.peek().map(|t| &t.kind),
            Some(TokenKind::Semicolon) | Some(TokenKind::RightBrace)) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        
        Ok(RustLikeExpression::Break { label, value })
    }

    /// Parse continue expressions with optional labels
    fn parse_continue_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'continue'
        let label = None; // Simplified
        Ok(RustLikeExpression::Continue { label })
    }

    /// Parse return expressions
    fn parse_return_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.advance(); // consume 'return'
        
        let value = if !self.is_at_end() && !matches!(self.peek().map(|t| &t.kind),
            Some(TokenKind::Semicolon) | Some(TokenKind::RightBrace)) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        
        Ok(RustLikeExpression::Return(value))
    }
    
    /// Parse if expression
    fn parse_if_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.consume(TokenKind::If, "Expected 'if'")?;
        let condition = Box::new(self.parse_expression()?);
        let then_branch = Box::new(self.parse_block_expression()?);
        
        let else_branch = if self.check(&TokenKind::Else) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        
        Ok(RustLikeExpression::If {
            condition,
            then_branch,
            else_branch,
        })
    }
    
    /// Parse match expression
    fn parse_match_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.consume(TokenKind::Match, "Expected 'match'")?;
        let expr = Box::new(self.parse_expression()?);
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after match expression")?;
        
        let mut arms = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            let pattern = self.parse_pattern()?;
            
            let guard = if self.check(&TokenKind::If) {
                self.advance();
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            self.consume(TokenKind::Arrow, "Expected '=>' after match pattern")?;
            let body = self.parse_expression()?;
            
            arms.push(RustLikeMatchArm {
                pattern,
                guard,
                body,
            });
            
            if !self.check(&TokenKind::RightBrace) {
                self.consume(TokenKind::Comma, "Expected ',' after match arm")?;
            }
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after match arms")?;
        
        Ok(RustLikeExpression::Match { expr, arms })
    }
    
    /// Parse block expression
    fn parse_block_expression(&mut self) -> Result<RustLikeExpression, RustLikeError> {
        self.consume(TokenKind::LeftBrace, "Expected '{'")?;
        
        let mut statements = Vec::new();
        let mut final_expr = None;
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            // Check if this is the final expression (no semicolon)
            let checkpoint = self.current;
            let stmt_or_expr = self.parse_statement()?;
            
            // If we didn't consume a semicolon and this is the last item, treat as expression
            if !self.check(&TokenKind::RightBrace) || matches!(stmt_or_expr, RustLikeStatement::Let { .. }) {
                statements.push(stmt_or_expr);
            } else {
                // This is the final expression
                if let RustLikeStatement::Expression(expr) = stmt_or_expr {
                    final_expr = Some(Box::new(expr));
                } else {
                    statements.push(stmt_or_expr);
                }
            }
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after block")?;
        
        Ok(RustLikeExpression::Block(statements, final_expr))
    }
    
    /// Parse a pattern
    fn parse_pattern(&mut self) -> Result<RustLikePattern, RustLikeError> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Identifier(name)) if name == "_" => {
                self.advance();
                Ok(RustLikePattern::Wildcard)
            }
            Some(TokenKind::Identifier(_)) => {
                let name = self.parse_identifier()?;
                
                // Check for struct pattern
                if self.check(&TokenKind::LeftBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    
                    while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                        let field_name = self.parse_identifier()?;
                        self.consume(TokenKind::Colon, "Expected ':' after field name")?;
                        let field_pattern = self.parse_pattern()?;
                        fields.push((field_name, field_pattern));
                        
                        if !self.check(&TokenKind::RightBrace) {
                            self.consume(TokenKind::Comma, "Expected ',' between pattern fields")?;
                        }
                    }
                    
                    self.consume(TokenKind::RightBrace, "Expected '}' after struct pattern")?;
                    
                    Ok(RustLikePattern::Struct { name, fields })
                } else {
                    Ok(RustLikePattern::Identifier(name))
                }
            }
            Some(TokenKind::LeftParen) => {
                self.advance();
                let mut patterns = Vec::new();
                
                while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                    patterns.push(self.parse_pattern()?);
                    if !self.check(&TokenKind::RightParen) {
                        self.consume(TokenKind::Comma, "Expected ',' between tuple patterns")?;
                    }
                }
                
                self.consume(TokenKind::RightParen, "Expected ')' after tuple pattern")?;
                Ok(RustLikePattern::Tuple(patterns))
            }
            _ => {
                // Try to parse as literal pattern
                let expr = self.parse_primary()?;
                match expr {
                    RustLikeExpression::Literal(lit) => Ok(RustLikePattern::Literal(lit)),
                    _ => {
                        let span = self.peek().map(|t| t.span).unwrap_or_else(Span::dummy);
                        Err(RustLikeError::InvalidPattern { span })
                    }
                }
            }
        }
    }
    
    /// Parse a type
    fn parse_type(&mut self) -> Result<RustLikeType, RustLikeError> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::Ampersand) => {
                self.advance();
                let is_mutable = if let Some(TokenKind::Identifier(name)) = self.peek().map(|t| &t.kind) {
                    if name == "mut" {
                        self.advance();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                let inner = Box::new(self.parse_type()?);
                Ok(RustLikeType::Reference { is_mutable, inner })
            }
            Some(TokenKind::LeftParen) => {
                self.advance();
                let mut types = Vec::new();
                
                while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                    types.push(self.parse_type()?);
                    if !self.check(&TokenKind::RightParen) {
                        self.consume(TokenKind::Comma, "Expected ',' between tuple types")?;
                    }
                }
                
                self.consume(TokenKind::RightParen, "Expected ')' after tuple type")?;
                Ok(RustLikeType::Tuple(types))
            }
            Some(TokenKind::LeftBracket) => {
                self.advance();
                let element_type = Box::new(self.parse_type()?);
                
                let size = if self.check(&TokenKind::Semicolon) {
                    self.advance();
                    // Parse size (simplified - just expect integer)
                    if let Some(token) = self.advance() {
                        if let TokenKind::IntegerLiteral(n) = &token.kind {
                            Some(*n as usize)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                self.consume(TokenKind::RightBracket, "Expected ']' after array type")?;
                Ok(RustLikeType::Array { element_type, size })
            }
            Some(TokenKind::Identifier(_)) => {
                let name = self.parse_identifier()?;
                
                // Check for generic arguments
                if self.check(&TokenKind::Less) {
                    self.advance();
                    let mut args = Vec::new();
                    
                    while !self.check(&TokenKind::Greater) && !self.is_at_end() {
                        args.push(self.parse_type()?);
                        if !self.check(&TokenKind::Greater) {
                            self.consume(TokenKind::Comma, "Expected ',' between type arguments")?;
                        }
                    }
                    
                    self.consume(TokenKind::Greater, "Expected '>' after type arguments")?;
                    Ok(RustLikeType::Generic { base: name, args })
                } else {
                    Ok(RustLikeType::Identifier(name))
                }
            }
            _ => {
                let span = self.peek().map(|t| t.span).unwrap_or_else(Span::dummy);
                Err(RustLikeError::InvalidType { span })
            }
        }
    }
    
    /// Parse a struct field
    fn parse_struct_field(&mut self) -> Result<RustLikeField, RustLikeError> {
        let visibility = if self.check(&TokenKind::Pub) {
            self.advance();
            Visibility::Public
        } else {
            Visibility::Private
        };
        
        let name = self.parse_identifier()?;
        self.consume(TokenKind::Colon, "Expected ':' after field name")?;
        let field_type = self.parse_type()?;
        
        Ok(RustLikeField {
            name,
            field_type,
            visibility,
        })
    }
    
    /// Parse an enum variant
    fn parse_enum_variant(&mut self) -> Result<RustLikeVariant, RustLikeError> {
        let name = self.parse_identifier()?;
        
        let data = if self.check(&TokenKind::LeftParen) {
            // Tuple variant
            self.advance();
            let mut types = Vec::new();
            
            while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                types.push(self.parse_type()?);
                if !self.check(&TokenKind::RightParen) {
                    self.consume(TokenKind::Comma, "Expected ',' between variant types")?;
                }
            }
            
            self.consume(TokenKind::RightParen, "Expected ')' after variant types")?;
            Some(RustLikeVariantData::Tuple(types))
        } else if self.check(&TokenKind::LeftBrace) {
            // Struct variant
            self.advance();
            let mut fields = Vec::new();
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                fields.push(self.parse_struct_field()?);
                if !self.check(&TokenKind::RightBrace) {
                    self.consume(TokenKind::Comma, "Expected ',' between variant fields")?;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after variant fields")?;
            Some(RustLikeVariantData::Struct(fields))
        } else {
            None
        };
        
        Ok(RustLikeVariant { name, data })
    }
    
    /// Parse a function parameter
    fn parse_parameter(&mut self) -> Result<RustLikeParameter, RustLikeError> {
        let is_mutable = if let Some(TokenKind::Identifier(name)) = self.peek().map(|t| &t.kind) {
            if name == "mut" {
                self.advance();
                true
            } else {
                false
            }
        } else {
            false
        };
        
        let name = self.parse_identifier()?;
        self.consume(TokenKind::Colon, "Expected ':' after parameter name")?;
        let param_type = self.parse_type()?;
        
        Ok(RustLikeParameter {
            name,
            param_type,
            is_mutable,
        })
    }
    
    /// Parse an identifier
    fn parse_identifier(&mut self) -> Result<String, RustLikeError> {
        if let Some(token) = self.advance() {
            match &token.kind {
                TokenKind::Identifier(name) => Ok(name.clone()),
                _ => Err(RustLikeError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                    span: token.span,
                }),
            }
        } else {
            Err(RustLikeError::UnexpectedEof)
        }
    }
}

// Helper trait for getting spans from expressions
impl RustLikeExpression {
    fn span(&self) -> Span {
        // This would need to be implemented based on actual expression structure
        Span::dummy()
    }
}

impl StyleParser for RustLikeParser {
    type Output = RustLikeSyntax;
    type Error = RustLikeError;
    type Config = RustLikeConfig;
    
    fn new() -> Self {
        Self {
            config: RustLikeConfig::default(),
            current: 0,
            tokens: Vec::new(),
        }
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self {
            config,
            current: 0,
            tokens: Vec::new(),
        }
    }
    
    fn parse(&mut self, tokens: Vec<Token>) -> Result<Self::Output, Self::Error> {
        self.tokens = tokens;
        self.current = 0;
        self.parse_program()
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::RustLike
    }
    
    fn capabilities(&self) -> ParserCapabilities {
        ParserCapabilities {
            supports_mixed_indentation: false, // Rust uses braces, not indentation
            supports_optional_semicolons: true, // Expression-oriented
            supports_trailing_commas: self.config.allow_trailing_commas,
            supports_nested_comments: true,
            error_recovery_level: ErrorRecoveryLevel::Advanced,
            max_nesting_depth: 256,
            supports_ai_metadata: true,
        }
    }
} 