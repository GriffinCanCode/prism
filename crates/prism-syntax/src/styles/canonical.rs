//! Parser for Canonical Prism syntax.

use crate::{
    styles::{StyleParser, StyleConfig, ParserCapabilities, ErrorRecoveryLevel, ConfigError},
    detection::SyntaxStyle,
};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use thiserror::Error;

#[derive(Debug)]
pub struct CanonicalParser {
    config: CanonicalConfig,
    current: usize,
    tokens: Vec<Token>,
}

#[derive(Debug, Clone, Default)]
pub struct CanonicalConfig {
    pub strict_mode: bool,
    pub require_documentation: bool,
}

#[derive(Debug)]
pub struct CanonicalSyntax {
    pub modules: Vec<CanonicalModule>,
    pub functions: Vec<CanonicalFunction>,
    pub statements: Vec<CanonicalStatement>,
}

#[derive(Debug)]
pub struct CanonicalModule {
    pub name: String,
    pub items: Vec<CanonicalItem>,
    pub span: Span,
}

#[derive(Debug)]
pub struct CanonicalFunction {
    pub name: String,
    pub parameters: Vec<CanonicalParameter>,
    pub return_type: Option<String>,
    pub body: Vec<CanonicalStatement>,
    pub span: Span,
}

#[derive(Debug)]
pub struct CanonicalParameter {
    pub name: String,
    pub param_type: Option<String>,
}

#[derive(Debug)]
pub enum CanonicalItem {
    Function(CanonicalFunction),
    Statement(CanonicalStatement),
}

#[derive(Debug)]
pub enum CanonicalStatement {
    Expression(CanonicalExpression),
    Return(Option<CanonicalExpression>),
    Assignment { name: String, value: CanonicalExpression },
}

#[derive(Debug)]
pub enum CanonicalExpression {
    Identifier(String),
    Literal(CanonicalLiteral),
    Call { function: String, arguments: Vec<CanonicalExpression> },
    Binary { left: Box<CanonicalExpression>, operator: String, right: Box<CanonicalExpression> },
}

#[derive(Debug)]
pub enum CanonicalLiteral {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

#[derive(Debug, Error)]
pub enum CanonicalError {
    #[error("Missing documentation at line {line}")]
    MissingDocumentation { line: usize },
    
    #[error("Unexpected token: expected {expected}, found {found} at {span:?}")]
    UnexpectedToken { expected: String, found: String, span: Span },
    
    #[error("Unexpected end of input")]
    UnexpectedEof,
    
    #[error("Invalid syntax: {message} at {span:?}")]
    InvalidSyntax { message: String, span: Span },
}

impl StyleConfig for CanonicalConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl CanonicalParser {
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
    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token, CanonicalError> {
        if self.check(&kind) {
            Ok(self.advance().unwrap())
        } else {
            let found = self.peek()
                .map(|t| format!("{:?}", t.kind))
                .unwrap_or_else(|| "EOF".to_string());
            let span = self.peek().map(|t| t.span).unwrap_or_else(Span::dummy);
            
            Err(CanonicalError::UnexpectedToken {
                expected: format!("{:?}", kind),
                found,
                span,
            })
        }
    }
    
    /// Parse the entire input
    fn parse_program(&mut self) -> Result<CanonicalSyntax, CanonicalError> {
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
        
        Ok(CanonicalSyntax {
            modules,
            functions,
            statements,
        })
    }
    
    /// Parse a module
    fn parse_module(&mut self) -> Result<CanonicalModule, CanonicalError> {
        let start_span = self.consume(TokenKind::Module, "Expected 'module'")?.span;
        
        let name = if let Some(token) = self.advance() {
            match &token.kind {
                TokenKind::Identifier(name) => name.clone(),
                _ => return Err(CanonicalError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                    span: token.span,
                }),
            }
        } else {
            return Err(CanonicalError::UnexpectedEof);
        };
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after module name")?;
        
        let mut items = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            if self.check(&TokenKind::Function) {
                items.push(CanonicalItem::Function(self.parse_function()?));
            } else {
                items.push(CanonicalItem::Statement(self.parse_statement()?));
            }
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after module body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(CanonicalModule { name, items, span })
    }
    
    /// Parse a function
    fn parse_function(&mut self) -> Result<CanonicalFunction, CanonicalError> {
        let start_span = self.consume(TokenKind::Function, "Expected 'function'")?.span;
        
        let name = if let Some(token) = self.advance() {
            match &token.kind {
                TokenKind::Identifier(name) => name.clone(),
                _ => return Err(CanonicalError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                    span: token.span,
                }),
            }
        } else {
            return Err(CanonicalError::UnexpectedEof);
        };
        
        // Parse parameters
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        let mut parameters = Vec::new();
        
        if !self.check(&TokenKind::RightParen) {
            loop {
                parameters.push(self.parse_parameter()?);
                
                if self.check(&TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        
        // Optional return type
        let return_type = if self.check(&TokenKind::Arrow) {
            self.advance(); // consume '->'
            if let Some(token) = self.advance() {
                match &token.kind {
                    TokenKind::Identifier(type_name) => Some(type_name.clone()),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        
        // Parse body
        self.consume(TokenKind::LeftBrace, "Expected '{' before function body")?;
        let mut body = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            body.push(self.parse_statement()?);
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after function body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(CanonicalFunction {
            name,
            parameters,
            return_type,
            body,
            span,
        })
    }
    
    /// Parse a statement
    fn parse_statement(&mut self) -> Result<CanonicalStatement, CanonicalError> {
        if self.check(&TokenKind::Return) {
            self.advance(); // consume 'return'
            
            let value = if self.check(&TokenKind::Semicolon) {
                None
            } else {
                Some(self.parse_expression()?)
            };
            
            // Optional semicolon
            if self.check(&TokenKind::Semicolon) {
                self.advance();
            }
            
            Ok(CanonicalStatement::Return(value))
        } else {
            // Try to parse as expression or assignment
            let expr = self.parse_expression()?;
            
            // Optional semicolon
            if self.check(&TokenKind::Semicolon) {
                self.advance();
            }
            
            Ok(CanonicalStatement::Expression(expr))
        }
    }
    
    /// Parse an expression
    fn parse_expression(&mut self) -> Result<CanonicalExpression, CanonicalError> {
        self.parse_primary()
    }
    
    /// Parse primary expressions
    fn parse_primary(&mut self) -> Result<CanonicalExpression, CanonicalError> {
        if let Some(token) = self.advance() {
            match &token.kind {
                TokenKind::Identifier(name) => {
                    // Check for function call
                    if self.check(&TokenKind::LeftParen) {
                        self.advance(); // consume '('
                        let mut arguments = Vec::new();
                        
                        if !self.check(&TokenKind::RightParen) {
                            loop {
                                arguments.push(self.parse_expression()?);
                                if self.check(&TokenKind::Comma) {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        
                        self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                        
                        Ok(CanonicalExpression::Call {
                            function: name.clone(),
                            arguments,
                        })
                    } else {
                        Ok(CanonicalExpression::Identifier(name.clone()))
                    }
                }
                TokenKind::StringLiteral(value) => {
                    Ok(CanonicalExpression::Literal(CanonicalLiteral::String(value.clone())))
                }
                TokenKind::IntegerLiteral(value) => {
                    Ok(CanonicalExpression::Literal(CanonicalLiteral::Integer(*value)))
                }
                TokenKind::FloatLiteral(value) => {
                    Ok(CanonicalExpression::Literal(CanonicalLiteral::Float(*value)))
                }
                TokenKind::True => {
                    Ok(CanonicalExpression::Literal(CanonicalLiteral::Boolean(true)))
                }
                TokenKind::False => {
                    Ok(CanonicalExpression::Literal(CanonicalLiteral::Boolean(false)))
                }
                _ => Err(CanonicalError::UnexpectedToken {
                    expected: "expression".to_string(),
                    found: format!("{:?}", token.kind),
                    span: token.span,
                }),
            }
        } else {
            Err(CanonicalError::UnexpectedEof)
        }
    }
    
    /// Parse a simple function parameter for canonical conversion
    fn parse_parameter(&mut self) -> Result<CanonicalParameter, CanonicalError> {
        let param_name = if let Some(token) = self.advance() {
            match &token.kind {
                TokenKind::Identifier(name) => name.clone(),
                _ => return Err(CanonicalError::UnexpectedToken {
                    expected: "parameter name".to_string(),
                    found: format!("{:?}", token.kind),
                    span: token.span,
                }),
            }
        } else {
            return Err(CanonicalError::UnexpectedEof);
        };
        
        // Optional type annotation (simplified)
        let param_type = if self.check(&TokenKind::Colon) {
            self.advance(); // consume ':'
            if let Some(token) = self.advance() {
                match &token.kind {
                    TokenKind::Identifier(type_name) => Some(type_name.clone()),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(CanonicalParameter {
            name: param_name,
            param_type,
        })
    }
    

}

impl StyleParser for CanonicalParser {
    type Output = CanonicalSyntax;
    type Error = CanonicalError;
    type Config = CanonicalConfig;
    
    fn new() -> Self {
        Self { 
            config: CanonicalConfig::default(),
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
        SyntaxStyle::Canonical
    }
    
    fn capabilities(&self) -> ParserCapabilities {
        ParserCapabilities {
            supports_mixed_indentation: false,
            supports_optional_semicolons: true,
            supports_trailing_commas: false,
            supports_nested_comments: false,
            error_recovery_level: ErrorRecoveryLevel::Intelligent,
            max_nesting_depth: 100,
            supports_ai_metadata: true,
        }
    }
} 