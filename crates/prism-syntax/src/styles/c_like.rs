//! Parser for C-like syntax (C/C++/Java/JavaScript style).
//!
//! Key characteristics:
//! - Braces {} for block delimiters
//! - Semicolons for statement terminators
//! - Parentheses around conditions
//! - C-style operators (&&, ||, etc.)

use crate::{
    styles::{StyleParser, StyleConfig, ParserCapabilities, ErrorRecoveryLevel, traits::ConfigError},
    detection::SyntaxStyle,
};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use thiserror::Error;

#[derive(Debug)]
pub struct CLikeParser {
    config: CLikeConfig,
    current: usize,
    tokens: Vec<Token>,
}

#[derive(Debug, Clone)]
pub struct CLikeConfig {
    /// Whether to require semicolons
    pub require_semicolons: bool,
    
    /// Allow trailing commas in collections
    pub allow_trailing_commas: bool,
    
    /// Brace style preference
    pub brace_style: BraceStyle,
    
    /// Indentation style (for mixed syntax detection)
    pub indentation_style: IndentationStyle,
}

/// Brace placement style
#[derive(Debug, Clone)]
pub enum BraceStyle {
    /// Opening brace on same line
    SameLine,
    
    /// Opening brace on next line
    NextLine,
    
    /// Allow either style
    Flexible,
}

/// Indentation style preference
#[derive(Debug, Clone)]
pub enum IndentationStyle {
    /// Spaces for indentation
    Spaces(usize),
    
    /// Tabs for indentation
    Tabs,
    
    /// Mixed indentation allowed
    Mixed,
}

#[derive(Debug)]
pub struct CLikeSyntax {
    pub modules: Vec<CLikeModule>,
    pub functions: Vec<CLikeFunction>,
    pub statements: Vec<CLikeStatement>,
}

impl Default for CLikeSyntax {
    fn default() -> Self {
        Self {
            modules: Vec::new(),
            functions: Vec::new(),
            statements: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct CLikeModule {
    pub name: String,
    pub body: Vec<CLikeItem>,
    pub span: Span,
}

#[derive(Debug)]
pub struct CLikeFunction {
    pub name: String,
    pub parameters: Vec<CLikeParameter>,
    pub return_type: Option<String>,
    pub body: Vec<CLikeStatement>,
    pub span: Span,
}

#[derive(Debug)]
pub struct CLikeParameter {
    pub name: String,
    pub param_type: Option<String>,
}

#[derive(Debug)]
pub enum CLikeItem {
    Function(CLikeFunction),
    Statement(CLikeStatement),
    Module(CLikeModule),
    TypeDeclaration {
        name: String,
        type_def: String,
        span: Span,
    },
    VariableDeclaration {
        name: String,
        var_type: Option<String>,
        initializer: Option<CLikeExpression>,
        span: Span,
    },
}

#[derive(Debug)]
pub enum CLikeStatement {
    Expression(CLikeExpression),
    Return(Option<CLikeExpression>),
    Assignment { 
        name: String, 
        value: CLikeExpression 
    },
    If {
        condition: CLikeExpression,
        then_block: Vec<CLikeStatement>,
        else_block: Option<Vec<CLikeStatement>>,
    },
    While {
        condition: CLikeExpression,
        body: Vec<CLikeStatement>,
    },
    For {
        init: Option<Box<CLikeStatement>>,
        condition: Option<CLikeExpression>,
        increment: Option<CLikeExpression>,
        body: Vec<CLikeStatement>,
    },
    DoWhile {
        body: Vec<CLikeStatement>,
        condition: CLikeExpression,
    },
    Switch {
        expression: CLikeExpression,
        cases: Vec<CLikeSwitchCase>,
        default_case: Option<Vec<CLikeStatement>>,
    },
    Break(Option<String>), // Optional label
    Continue(Option<String>), // Optional label
    Block(Vec<CLikeStatement>),
    Try {
        body: Vec<CLikeStatement>,
        catch_blocks: Vec<CLikeCatchBlock>,
        finally_block: Option<Vec<CLikeStatement>>,
    },
    Throw(CLikeExpression),
    VariableDeclaration {
        name: String,
        var_type: Option<String>,
        initializer: Option<CLikeExpression>,
    },
    Empty, // Empty statement (just semicolon)
    Error {
        message: String,
        span: Span,
    }, // Error statement for recovery
}

#[derive(Debug)]
pub struct CLikeSwitchCase {
    pub values: Vec<CLikeExpression>, // Multiple case values
    pub statements: Vec<CLikeStatement>,
}

#[derive(Debug)]
pub struct CLikeCatchBlock {
    pub exception_type: Option<String>,
    pub exception_name: Option<String>,
    pub body: Vec<CLikeStatement>,
}

#[derive(Debug)]
pub enum CLikeExpression {
    Identifier(String),
    Literal(CLikeLiteral),
    Call { 
        function: Box<CLikeExpression>, 
        arguments: Vec<CLikeExpression> 
    },
    Binary { 
        left: Box<CLikeExpression>, 
        operator: BinaryOperator, 
        right: Box<CLikeExpression> 
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<CLikeExpression>,
    },
    Ternary {
        condition: Box<CLikeExpression>,
        true_expr: Box<CLikeExpression>,
        false_expr: Box<CLikeExpression>,
    },
    Assignment {
        left: Box<CLikeExpression>,
        operator: AssignmentOperator,
        right: Box<CLikeExpression>,
    },
    MemberAccess {
        object: Box<CLikeExpression>,
        member: String,
        safe_navigation: bool, // For optional chaining
    },
    IndexAccess {
        object: Box<CLikeExpression>,
        index: Box<CLikeExpression>,
    },
    ArrayLiteral(Vec<CLikeExpression>),
    ObjectLiteral(Vec<CLikeObjectField>),
    Lambda {
        parameters: Vec<CLikeParameter>,
        body: Box<CLikeExpression>,
    },
    Cast {
        target_type: String,
        expression: Box<CLikeExpression>,
    },
    Parenthesized(Box<CLikeExpression>),
    PostfixIncrement(Box<CLikeExpression>),
    PostfixDecrement(Box<CLikeExpression>),
    PrefixIncrement(Box<CLikeExpression>),
    PrefixDecrement(Box<CLikeExpression>),
    Error {
        message: String,
        span: Span,
    }, // Error expression for recovery
}

#[derive(Debug)]
pub struct CLikeObjectField {
    pub key: String,
    pub value: CLikeExpression,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo,
    
    // Comparison
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    
    // Logical
    LogicalAnd, LogicalOr,
    
    // Bitwise
    BitwiseAnd, BitwiseOr, BitwiseXor,
    LeftShift, RightShift,
    
    // Special
    Comma, // Comma operator
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOperator {
    Plus, Minus, LogicalNot, BitwiseNot,
    AddressOf, Dereference, // For C-style pointers
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignmentOperator {
    Assign, AddAssign, SubtractAssign, MultiplyAssign, DivideAssign,
    ModuloAssign, AndAssign, OrAssign, XorAssign,
    LeftShiftAssign, RightShiftAssign,
}

#[derive(Debug)]
pub enum CLikeLiteral {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    Character(char),
}

#[derive(Debug, Clone, Error)]
pub enum CLikeError {
    /// Unexpected token
    #[error("Unexpected token: expected {expected}, found {found:?} at {span:?}")]
    UnexpectedToken { expected: String, found: String, span: Span },
    
    /// Missing semicolon
    #[error("Missing semicolon at {span:?}")]
    MissingSemicolon { span: Span },
    
    /// Unmatched brace
    #[error("Unmatched brace at {span:?}")]
    UnmatchedBrace { span: Span },
    
    /// Missing parentheses around condition
    #[error("Missing parentheses around condition at {span:?}")]
    MissingConditionParens { span: Span },
    
    /// Unexpected end of input
    #[error("Unexpected end of input")]
    UnexpectedEof,
    
    /// Invalid assignment target
    #[error("Invalid assignment target at {span:?}")]
    InvalidAssignmentTarget { span: Span },
    
    /// Missing expression
    #[error("Missing expression at {span:?}")]
    MissingExpression { span: Span },
    
    /// Unexpected end of file
    #[error("Unexpected end of file at {span:?}")]
    UnexpectedEndOfFile { span: Span },
    
    /// Invalid syntax
    #[error("Invalid syntax: {message} at {span:?}")]
    InvalidSyntax { message: String, span: Span },
}

impl Default for BraceStyle {
    fn default() -> Self {
        BraceStyle::Flexible
    }
}

impl Default for IndentationStyle {
    fn default() -> Self {
        IndentationStyle::Spaces(4)
    }
}

impl Default for CLikeConfig {
    fn default() -> Self {
        Self {
            require_semicolons: true,
            allow_trailing_commas: true,
            brace_style: BraceStyle::default(),
            indentation_style: IndentationStyle::default(),
        }
    }
}

impl StyleConfig for CLikeConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl CLikeParser {
    /// Check if we're at the end of tokens
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }
    
    /// Get the current token
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current)
    }
    
    /// Get the token at offset from current position
    fn peek_ahead(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.current + offset)
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
    
    /// Check if current token matches any of the expected kinds
    fn check_any(&self, kinds: &[TokenKind]) -> bool {
        if let Some(token) = self.peek() {
            kinds.contains(&token.kind)
        } else {
            false
        }
    }
    
    /// Consume a token of expected kind or return error
    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token, CLikeError> {
        if self.check(&kind) {
            Ok(self.advance().unwrap())
        } else {
            let found = self.peek()
                .map(|t| format!("{:?}", t.kind))
                .unwrap_or_else(|| "EOF".to_string());
            let span = self.peek().map(|t| t.span).unwrap_or_else(Span::dummy);
            
            Err(CLikeError::UnexpectedToken {
                expected: message.to_string(),
                found,
                span,
            })
        }
    }
    
    /// Get the previous token
    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    
    /// Get the current span for error reporting
    fn current_span(&self) -> Span {
        self.peek().map(|t| t.span).unwrap_or_else(Span::dummy)
    }
    
    /// Check if current position indicates a do-while pattern
    fn is_do_while_pattern(&self) -> bool {
        // This is a simplified check - in a real implementation,
        // you'd look for patterns like "do { ... } while"
        false
    }

    /// Advance with automatic error recovery and synchronization
    fn advance_with_recovery(&mut self) -> Result<&Token, CLikeError> {
        if let Some(token) = self.advance() {
            Ok(token)
        } else {
            Err(CLikeError::UnexpectedEof)
        }
    }

    /// Intelligent error recovery system
    fn recover_from_error(&mut self, error: &CLikeError) -> Result<(), CLikeError> {
        match error {
            CLikeError::UnexpectedToken { expected, found, .. } => {
                self.recover_from_unexpected_token(expected, found)
            }
            CLikeError::UnmatchedBrace { .. } => {
                self.recover_from_unmatched_brace()
            }
            CLikeError::MissingSemicolon { .. } => {
                self.recover_from_missing_semicolon()
            }
            CLikeError::InvalidSyntax { .. } => {
                self.recover_from_invalid_syntax()
            }
            CLikeError::UnexpectedEndOfFile { .. } => {
                // Can't recover from EOF
                Err(error.clone())
            }
            CLikeError::MissingConditionParens { .. } => {
                self.recover_from_missing_semicolon()
            }
            CLikeError::UnexpectedEof => {
                Err(error.clone())
            }
            CLikeError::InvalidAssignmentTarget { .. } => {
                self.recover_from_invalid_syntax()
            }
            CLikeError::MissingExpression { .. } => {
                self.recover_from_invalid_syntax()
            }
        }
    }

    /// Recovery strategy for unexpected tokens
    fn recover_from_unexpected_token(&mut self, expected: &str, found: &str) -> Result<(), CLikeError> {
        // Strategy 1: Skip tokens until we find a synchronization point
        let sync_points = ["semicolon", "left_brace", "right_brace", "if", "for", "while", "function"];
        
        while !self.is_at_end() {
            let current_token = self.peek();
            
            // Check if we've reached a synchronization point
            if let Some(token) = current_token {
                if sync_points.iter().any(|&sync| self.token_matches_type(token, sync)) {
                    return Ok(());
                }
            }
            
            // Strategy 2: Try to insert missing tokens
            if expected == "semicolon" && self.can_insert_semicolon() {
                // Insert virtual semicolon and continue
                return Ok(());
            }
            
            if expected == "identifier" && found == "keyword" {
                // Might be using a keyword as identifier - allow it with warning
                return Ok(());
            }
            
            self.current += 1;
        }
        
        Ok(())
    }

    /// Recovery strategy for unmatched braces
    fn recover_from_unmatched_brace(&mut self) -> Result<(), CLikeError> {
        let mut brace_count = 0;
        let start_pos = self.current;
        
        // Count braces to find the matching one or end of block
        while !self.is_at_end() {
            let token = self.peek();
            
            if let Some(token) = token {
                match &token.kind {
                    TokenKind::LeftBrace => brace_count += 1,
                    TokenKind::RightBrace => {
                        brace_count -= 1;
                        if brace_count <= 0 {
                            // Found matching brace or end of block
                            return Ok(());
                        }
                    }
                    TokenKind::Semicolon => {
                        // Semicolon can be a statement boundary
                        if brace_count == 0 {
                            return Ok(());
                        }
                    }
                    _ => {}
                }
            }
            
            self.current += 1;
        }
        
        // If we couldn't find a match, reset to start and continue
        self.current = start_pos;
        Ok(())
    }

    /// Recovery strategy for missing semicolons
    fn recover_from_missing_semicolon(&mut self) -> Result<(), CLikeError> {
        // Check if we can automatically insert a semicolon
        if self.can_insert_semicolon() {
            // Virtual semicolon insertion - continue parsing
            return Ok(());
        }
        
        // Look for the next statement boundary
        while !self.is_at_end() {
            let token = self.peek();
            
            if let Some(token) = token {
                match &token.kind {
                    TokenKind::Semicolon => {
                        // Found explicit semicolon
                        return Ok(());
                    }
                    TokenKind::RightBrace | TokenKind::LeftBrace => {
                        // Brace can indicate statement boundary
                        return Ok(());
                    }
                    TokenKind::If | TokenKind::For | TokenKind::While | 
                    TokenKind::Return | TokenKind::Break | TokenKind::Continue => {
                        // Keywords like 'if', 'for', etc. start new statements
                        return Ok(());
                    }
                    _ => {}
                }
            }
            
            self.current += 1;
        }
        
        Ok(())
    }

    /// Recovery strategy for invalid syntax
    fn recover_from_invalid_syntax(&mut self) -> Result<(), CLikeError> {
        // Skip tokens until we find a statement or declaration boundary
        let recovery_points = [
            TokenKind::Semicolon,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::If,
            TokenKind::For,
            TokenKind::While,
            TokenKind::Return,
        ];
        
        while !self.is_at_end() {
            let token = self.peek();
            
            if let Some(token) = token {
                if recovery_points.contains(&token.kind) {
                    return Ok(());
                }
            }
            
            self.current += 1;
        }
        
        Ok(())
    }

    /// Check if we can insert a virtual semicolon
    fn can_insert_semicolon(&self) -> bool {
        if self.is_at_end() {
            return true;
        }
        
        let current_token = self.peek();
        
        // Automatic semicolon insertion rules (similar to JavaScript)
        if let Some(current_token) = current_token {
            match &current_token.kind {
            TokenKind::RightBrace => true,  // Before closing brace
            TokenKind::If | TokenKind::For | TokenKind::While | 
            TokenKind::Return | TokenKind::Break | TokenKind::Continue => {
                // Before control flow keywords
                true
            }
            TokenKind::Identifier(_) => {
                // Before what looks like a new statement
                true
            }
            _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a token matches a specific type description
    fn token_matches_type(&self, token: &Token, type_desc: &str) -> bool {
        match type_desc {
            "semicolon" => matches!(token.kind, TokenKind::Semicolon),
            "left_brace" => matches!(token.kind, TokenKind::LeftBrace),
            "right_brace" => matches!(token.kind, TokenKind::RightBrace),
            "identifier" => matches!(token.kind, TokenKind::Identifier(_)),
            "keyword" => matches!(token.kind, TokenKind::If | TokenKind::For | TokenKind::While | TokenKind::Return),
            "if" | "for" | "while" | "function" => {
                matches!(token.kind, TokenKind::If | TokenKind::For | TokenKind::While | TokenKind::Function) && {
                    let keyword = self.get_token_text(token);
                    keyword == type_desc
                }
            }
            _ => false,
        }
    }

    /// Enhanced parsing with error recovery for statements
    fn parse_statement_with_recovery(&mut self) -> Result<CLikeStatement, CLikeError> {
        let result = self.parse_statement();
        
        match result {
            Ok(stmt) => Ok(stmt),
            Err(error) => {
                // Attempt recovery
                self.recover_from_error(&error)?;
                
                // Try parsing again after recovery
                match self.parse_statement() {
                    Ok(stmt) => Ok(stmt),
                    Err(_) => {
                        // If still failing, create an error statement
                        Ok(CLikeStatement::Error {
                            message: format!("Recovered from error: {}", error),
                            span: self.current_span(),
                        })
                    }
                }
            }
        }
    }

    /// Enhanced parsing with error recovery for expressions
    fn parse_expression_with_recovery(&mut self) -> Result<CLikeExpression, CLikeError> {
        let result = self.parse_expression();
        
        match result {
            Ok(expr) => Ok(expr),
            Err(error) => {
                // Attempt recovery
                self.recover_from_error(&error)?;
                
                // Try parsing again after recovery
                match self.parse_expression() {
                    Ok(expr) => Ok(expr),
                    Err(_) => {
                        // If still failing, create an error expression
                        Ok(CLikeExpression::Error {
                            message: format!("Recovered from error: {}", error),
                            span: self.current_span(),
                        })
                    }
                }
            }
        }
    }

    /// Parse program with comprehensive error recovery
    fn parse_program_with_recovery(&mut self) -> CLikeSyntax {
        let mut syntax = CLikeSyntax::default();
        let mut errors = Vec::new();
        
        while !self.is_at_end() {
            match self.parse_top_level_item() {
                Ok(item) => {
                    match item {
                        CLikeItem::Function(func) => syntax.functions.push(func),
                        CLikeItem::Statement(stmt) => syntax.statements.push(stmt),
                        CLikeItem::Module(module) => syntax.modules.push(module),
                        CLikeItem::TypeDeclaration { name, type_def, .. } => {
                            // Handle type declaration - could add to types collection
                            // For now, convert to statement
                            syntax.statements.push(CLikeStatement::VariableDeclaration {
                                name: name.clone(),
                                var_type: Some(type_def.clone()),
                                initializer: None,
                            });
                        },
                        CLikeItem::VariableDeclaration { name, var_type, initializer, .. } => {
                            syntax.statements.push(CLikeStatement::VariableDeclaration {
                                name: name.clone(),
                                var_type: var_type.clone(),
                                initializer: initializer.clone(),
                            });
                        },
                    }
                }
                Err(error) => {
                    errors.push(error.clone());
                    
                    // Attempt recovery and synchronization
                    if let Err(_) = self.recover_from_error(&error) {
                        // If recovery fails, synchronize and continue
                        self.synchronize();
                    }
                }
            }
        }
        
        // If we collected errors but still have valid syntax, return it
        // The errors can be reported separately
        syntax
    }

    /// Parse a top-level item (function, statement, or module)
    fn parse_top_level_item(&mut self) -> Result<CLikeItem, CLikeError> {
        // Look ahead to determine what we're parsing
        if self.is_function_declaration() {
            Ok(CLikeItem::Function(self.parse_function()?))
        } else if self.is_module_declaration() {
            Ok(CLikeItem::Module(self.parse_module()?))
        } else {
            Ok(CLikeItem::Statement(self.parse_statement_with_recovery()?))
        }
    }

    /// Check if current position is a function declaration
    fn is_function_declaration(&self) -> bool {
        // Look for pattern: [type] identifier (
        let mut pos = self.current;
        
        // Skip potential return type
        if pos < self.tokens.len() && matches!(self.tokens[pos].kind, TokenKind::Identifier(_)) {
            pos += 1;
        }
        
        // Check for identifier followed by (
        if pos < self.tokens.len() && matches!(self.tokens[pos].kind, TokenKind::Identifier(_)) {
            pos += 1;
            if pos < self.tokens.len() && matches!(self.tokens[pos].kind, TokenKind::LeftParen) {
                return true;
            }
        }
        
        false
    }

    /// Check if current position is a module declaration
    fn is_module_declaration(&self) -> bool {
        if let Some(token) = self.tokens.get(self.current) {
            if matches!(token.kind, TokenKind::Module) {
                return true;
            }
        }
        false
    }

    /// Enhanced error reporting with context
    fn create_contextual_error(&self, error_type: CLikeError) -> CLikeError {
        // Add context information to errors for better debugging
        match error_type {
            CLikeError::UnexpectedToken { expected, found, span } => {
                let context = self.get_parsing_context();
                CLikeError::UnexpectedToken {
                    expected: format!("{} (in {})", expected, context),
                    found,
                    span,
                }
            }
            other => other,
        }
    }

    /// Get current parsing context for error reporting
    fn get_parsing_context(&self) -> String {
        // Analyze recent tokens to determine context
        let recent_keywords: Vec<String> = self.tokens
            .iter()
            .rev()
            .skip(self.tokens.len().saturating_sub(self.current))
            .take(5)
            .filter(|token| matches!(token.kind, TokenKind::If | TokenKind::For | TokenKind::While | TokenKind::Function))
            .map(|token| self.get_token_text(token))
            .collect();
        
        if let Some(keyword) = recent_keywords.first() {
            match keyword.as_str() {
                "if" => "if statement".to_string(),
                "for" => "for loop".to_string(),
                "while" => "while loop".to_string(),
                "function" | "fun" => "function declaration".to_string(),
                "struct" => "struct declaration".to_string(),
                "class" => "class declaration".to_string(),
                _ => "statement".to_string(),
            }
        } else {
            "expression".to_string()
        }
    }

    /// Get text content of a token
    fn get_token_text(&self, token: &Token) -> String {
        // This would normally extract text from the token
        // For now, return a placeholder based on token kind
        match &token.kind {
            TokenKind::Identifier(_) => "identifier".to_string(),
            TokenKind::If => "if".to_string(),
            TokenKind::For => "for".to_string(),
            TokenKind::While => "while".to_string(),
            TokenKind::Function => "function".to_string(),
            TokenKind::IntegerLiteral(_) => "number".to_string(),
            TokenKind::FloatLiteral(_) => "number".to_string(),
            TokenKind::StringLiteral(_) => "string".to_string(),
            TokenKind::LeftParen => "(".to_string(),
            TokenKind::RightParen => ")".to_string(),
            TokenKind::LeftBrace => "{".to_string(),
            TokenKind::RightBrace => "}".to_string(),
            TokenKind::Semicolon => ";".to_string(),
            _ => "token".to_string(),
        }
    }
    
    /// Skip to next statement boundary for error recovery
    fn synchronize(&mut self) {
        while !self.is_at_end() {
            if let Some(token) = self.peek() {
                match token.kind {
                    TokenKind::Semicolon => {
                        self.advance();
                        return;
                    }
                    TokenKind::LeftBrace | TokenKind::RightBrace => return,
                    TokenKind::If | TokenKind::While | TokenKind::For | 
                    TokenKind::Function | TokenKind::Return => return,
                    _ => {
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }
    }
    
    /// Parse the entire program
    fn parse_program(&mut self) -> Result<CLikeSyntax, CLikeError> {
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        let mut statements = Vec::new();
        
        while !self.is_at_end() {
            // Skip any stray semicolons or newlines
            while self.check(&TokenKind::Semicolon) || self.check(&TokenKind::Newline) {
                self.advance();
            }
            
            if self.is_at_end() {
                break;
            }
            
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::Module) => {
                    modules.push(self.parse_module()?);
                }
                Some(TokenKind::Function) | Some(TokenKind::Fn) => {
                    functions.push(self.parse_function()?);
                }
                _ => {
                    statements.push(self.parse_statement()?);
                }
            }
        }
        
        Ok(CLikeSyntax {
            modules,
            functions,
            statements,
        })
    }
    
    /// Parse a module with C-like syntax
    fn parse_module(&mut self) -> Result<CLikeModule, CLikeError> {
        let start_span = self.consume(TokenKind::Module, "Expected 'module'")?.span;
        
        let name = self.parse_identifier()?;
        self.consume(TokenKind::LeftBrace, "Expected '{' after module name")?;
        
        let mut body = Vec::new();
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            // Skip any stray semicolons
            while self.check(&TokenKind::Semicolon) || self.check(&TokenKind::Newline) {
                self.advance();
            }
            
            if self.check(&TokenKind::RightBrace) {
                break;
            }
            
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::Function) | Some(TokenKind::Fn) => {
                    body.push(CLikeItem::Function(self.parse_function()?));
                }
                Some(TokenKind::Type) => {
                    body.push(self.parse_type_declaration()?);
                }
                Some(TokenKind::Let) | Some(TokenKind::Const) | Some(TokenKind::Var) => {
                    body.push(self.parse_variable_declaration_item()?);
                }
                _ => {
                    body.push(CLikeItem::Statement(self.parse_statement()?));
                }
            }
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after module body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(CLikeModule { name, body, span })
    }
    
    /// Parse a type declaration
    fn parse_type_declaration(&mut self) -> Result<CLikeItem, CLikeError> {
        let start_span = self.consume(TokenKind::Type, "Expected 'type'")?.span;
        let name = self.parse_identifier()?;
        self.consume(TokenKind::Assign, "Expected '=' after type name")?;
        
        // For now, just parse the type as a simple identifier
        let type_def = self.parse_identifier()?;
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after type declaration")?;
        }
        
        let end_span = self.peek().map(|t| t.span).unwrap_or(start_span);
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(CLikeItem::TypeDeclaration { name, type_def, span })
    }
    
    /// Parse a variable declaration as an item
    fn parse_variable_declaration_item(&mut self) -> Result<CLikeItem, CLikeError> {
        let start_span = self.peek().unwrap().span;
        let var_decl = self.parse_variable_declaration()?;
        
        if let CLikeStatement::VariableDeclaration { name, var_type, initializer } = var_decl {
            let end_span = self.peek().map(|t| t.span).unwrap_or(start_span);
            let span = Span::new(start_span.start, end_span.end, start_span.source_id);
            
            Ok(CLikeItem::VariableDeclaration { name, var_type, initializer, span })
        } else {
            // This shouldn't happen, but handle gracefully
            Ok(CLikeItem::Statement(var_decl))
        }
    }
    
    /// Parse a function with C-like syntax
    fn parse_function(&mut self) -> Result<CLikeFunction, CLikeError> {
        let start_span = if self.check(&TokenKind::Function) {
            self.consume(TokenKind::Function, "Expected 'function'")?.span
        } else {
            self.consume(TokenKind::Fn, "Expected 'fn'")?.span
        };
        
        let name = self.parse_identifier()?;
        
        // Parse parameters with required parentheses
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        let mut parameters = Vec::new();
        
        if !self.check(&TokenKind::RightParen) {
            loop {
                parameters.push(self.parse_parameter()?);
                
                if self.check(&TokenKind::Comma) {
                    self.advance();
                    if self.config.allow_trailing_commas && self.check(&TokenKind::RightParen) {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        
        // Optional return type
        let return_type = if self.check(&TokenKind::Arrow) {
            self.advance();
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        // Parse body with required braces
        self.consume(TokenKind::LeftBrace, "Expected '{' before function body")?;
        let mut body = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            body.push(self.parse_statement()?);
        }
        
        let end_span = self.consume(TokenKind::RightBrace, "Expected '}' after function body")?.span;
        let span = Span::new(start_span.start, end_span.end, start_span.source_id);
        
        Ok(CLikeFunction {
            name,
            parameters,
            return_type,
            body,
            span,
        })
    }
    
    /// Parse a statement with C-like syntax
    fn parse_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        // Skip any stray newlines
        while self.check(&TokenKind::Newline) {
            self.advance();
        }
        
        if self.is_at_end() {
            return Err(CLikeError::UnexpectedEof);
        }
        
        match self.peek().unwrap().kind.clone() {
            TokenKind::Return => self.parse_return_statement(),
            TokenKind::If => self.parse_if_statement(),
            TokenKind::While => {
                // Check if this is a do-while by looking ahead
                if self.is_do_while_pattern() {
                    self.parse_do_while_statement()
                } else {
                    self.parse_while_statement()
                }
            }
            TokenKind::For => self.parse_for_statement(),
            TokenKind::Switch => self.parse_switch_statement(),
            TokenKind::Break => self.parse_break_statement(),
            TokenKind::Continue => self.parse_continue_statement(),
            TokenKind::Try => self.parse_try_statement(),
            TokenKind::Throw => self.parse_throw_statement(),
            TokenKind::Let | TokenKind::Const | TokenKind::Var => self.parse_variable_declaration(),
            TokenKind::LeftBrace => self.parse_block_statement(),
            TokenKind::Semicolon => {
                self.advance(); // Consume semicolon
                Ok(CLikeStatement::Empty)
            }
            _ => self.parse_expression_statement(),
        }
    }
    
    /// Parse a return statement
    fn parse_return_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'return'
        
        let value = if self.check(&TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after return statement")?;
        }
        
        Ok(CLikeStatement::Return(value))
    }
    
    /// Parse an if statement
    fn parse_if_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'if'
        
        // C-like requires parentheses around conditions
        self.consume(TokenKind::LeftParen, "Expected '(' after 'if'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenKind::RightParen, "Expected ')' after if condition")?;
        
        let then_block = self.parse_block_or_statement()?;
        
        let else_block = if self.check(&TokenKind::Else) {
            self.advance();
            Some(self.parse_block_or_statement()?)
        } else {
            None
        };
        
        Ok(CLikeStatement::If {
            condition,
            then_block,
            else_block,
        })
    }
    
    /// Parse a while statement
    fn parse_while_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'while'
        
        // C-like requires parentheses around conditions
        self.consume(TokenKind::LeftParen, "Expected '(' after 'while'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenKind::RightParen, "Expected ')' after while condition")?;
        
        let body = self.parse_block_or_statement()?;
        
        Ok(CLikeStatement::While { condition, body })
    }
    
    /// Parse a for statement
    fn parse_for_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'for'
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'for'")?;
        
        // Parse init (optional)
        let init = if self.check(&TokenKind::Semicolon) {
            None
        } else {
            Some(Box::new(self.parse_statement()?))
        };
        
        if !self.check(&TokenKind::Semicolon) {
            self.consume(TokenKind::Semicolon, "Expected ';' after for init")?;
        } else {
            self.advance(); // consume semicolon
        }
        
        // Parse condition (optional)
        let condition = if self.check(&TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        self.consume(TokenKind::Semicolon, "Expected ';' after for condition")?;
        
        // Parse increment (optional)
        let increment = if self.check(&TokenKind::RightParen) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        
        self.consume(TokenKind::RightParen, "Expected ')' after for clauses")?;
        
        let body = self.parse_block_or_statement()?;
        
        Ok(CLikeStatement::For {
            init,
            condition,
            increment,
            body,
        })
    }
    
    /// Parse a do-while statement
    fn parse_do_while_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'do'
        
        let body = self.parse_block_or_statement()?;
        
        self.consume(TokenKind::While, "Expected 'while' after do body")?;
        self.consume(TokenKind::LeftParen, "Expected '(' after 'while'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenKind::RightParen, "Expected ')' after while condition")?;
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after do-while statement")?;
        }
        
        Ok(CLikeStatement::DoWhile { body, condition })
    }
    
    /// Parse a switch statement
    fn parse_switch_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'switch'
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'switch'")?;
        let expression = self.parse_expression()?;
        self.consume(TokenKind::RightParen, "Expected ')' after switch expression")?;
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after switch expression")?;
        
        let mut cases = Vec::new();
        let mut default_case = None;
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            if self.check(&TokenKind::Case) {
                self.advance(); // consume 'case'
                let mut values = vec![self.parse_expression()?];
                self.consume(TokenKind::Colon, "Expected ':' after case value")?;
                
                // Parse statements until next case/default/end
                let mut statements = Vec::new();
                while !self.check_any(&[TokenKind::Case, TokenKind::Default, TokenKind::RightBrace]) 
                    && !self.is_at_end() {
                    statements.push(self.parse_statement()?);
                }
                
                cases.push(CLikeSwitchCase { values, statements });
            } else if self.check(&TokenKind::Default) {
                self.advance(); // consume 'default'
                self.consume(TokenKind::Colon, "Expected ':' after 'default'")?;
                
                let mut statements = Vec::new();
                while !self.check_any(&[TokenKind::Case, TokenKind::Default, TokenKind::RightBrace]) 
                    && !self.is_at_end() {
                    statements.push(self.parse_statement()?);
                }
                
                default_case = Some(statements);
            } else {
                return Err(CLikeError::UnexpectedToken {
                    expected: "case or default".to_string(),
                    found: format!("{:?}", self.peek().unwrap().kind),
                    span: self.peek().unwrap().span,
                });
            }
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after switch body")?;
        
        Ok(CLikeStatement::Switch { expression, cases, default_case })
    }
    
    /// Parse a break statement
    fn parse_break_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'break'
        
        // Optional label (for labeled breaks in some C-like languages)
        let label = if let Some(TokenKind::Identifier(_)) = self.peek().map(|t| &t.kind) {
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after break statement")?;
        }
        
        Ok(CLikeStatement::Break(label))
    }
    
    /// Parse a continue statement
    fn parse_continue_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'continue'
        
        // Optional label
        let label = if let Some(TokenKind::Identifier(_)) = self.peek().map(|t| &t.kind) {
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after continue statement")?;
        }
        
        Ok(CLikeStatement::Continue(label))
    }
    
    /// Parse a try statement
    fn parse_try_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'try'
        
        let body = self.parse_block_or_statement()?;
        
        let mut catch_blocks = Vec::new();
        while self.check(&TokenKind::Catch) {
            self.advance(); // consume 'catch'
            
            self.consume(TokenKind::LeftParen, "Expected '(' after 'catch'")?;
            
            let exception_type = if !self.check(&TokenKind::RightParen) {
                Some(self.parse_identifier()?)
            } else {
                None
            };
            
            let exception_name = if exception_type.is_some() && !self.check(&TokenKind::RightParen) {
                Some(self.parse_identifier()?)
            } else {
                None
            };
            
            self.consume(TokenKind::RightParen, "Expected ')' after catch parameters")?;
            
            let catch_body = self.parse_block_or_statement()?;
            
            catch_blocks.push(CLikeCatchBlock {
                exception_type,
                exception_name,
                body: catch_body,
            });
        }
        
        let finally_block = if self.check(&TokenKind::Finally) {
            self.advance(); // consume 'finally'
            Some(self.parse_block_or_statement()?)
        } else {
            None
        };
        
        Ok(CLikeStatement::Try { body, catch_blocks, finally_block })
    }
    
    /// Parse a throw statement
    fn parse_throw_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume 'throw'
        
        let expression = self.parse_expression()?;
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after throw statement")?;
        }
        
        Ok(CLikeStatement::Throw(expression))
    }
    
    /// Parse a variable declaration
    fn parse_variable_declaration(&mut self) -> Result<CLikeStatement, CLikeError> {
        // Consume let/const/var
        let _decl_type = self.advance().unwrap().kind.clone();
        
        let name = self.parse_identifier()?;
        
        let var_type = if self.check(&TokenKind::Colon) {
            self.advance(); // consume ':'
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        let initializer = if self.check(&TokenKind::Assign) {
            self.advance(); // consume '='
            Some(self.parse_expression()?)
        } else {
            None
        };
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after variable declaration")?;
        }
        
        Ok(CLikeStatement::VariableDeclaration { name, var_type, initializer })
    }
    
    /// Parse a block statement
    fn parse_block_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        self.advance(); // consume '{'
        let mut statements = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            statements.push(self.parse_statement()?);
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after block")?;
        Ok(CLikeStatement::Block(statements))
    }
    
    /// Parse an expression statement
    fn parse_expression_statement(&mut self) -> Result<CLikeStatement, CLikeError> {
        let expr = self.parse_expression()?;
        
        if self.config.require_semicolons {
            self.consume(TokenKind::Semicolon, "Expected ';' after expression statement")?;
        }
        
        Ok(CLikeStatement::Expression(expr))
    }
    
    /// Parse a block or single statement
    fn parse_block_or_statement(&mut self) -> Result<Vec<CLikeStatement>, CLikeError> {
        if self.check(&TokenKind::LeftBrace) {
            if let CLikeStatement::Block(statements) = self.parse_block_statement()? {
                Ok(statements)
            } else {
                unreachable!("parse_block_statement should return Block")
            }
        } else {
            Ok(vec![self.parse_statement()?])
        }
    }
    
    /// Parse an expression with full precedence handling
    fn parse_expression(&mut self) -> Result<CLikeExpression, CLikeError> {
        self.parse_ternary()
    }
    
    /// Parse ternary conditional expressions (lowest precedence)
    fn parse_ternary(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_logical_or()?;
        
        if self.check(&TokenKind::Question) {
            self.advance(); // consume '?'
            let true_expr = Box::new(self.parse_expression()?);
            self.consume(TokenKind::Colon, "Expected ':' in ternary expression")?;
            let false_expr = Box::new(self.parse_expression()?);
            
            expr = CLikeExpression::Ternary {
                condition: Box::new(expr),
                true_expr,
                false_expr,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse logical OR expressions
    fn parse_logical_or(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_logical_and()?;
        
        while self.check(&TokenKind::OrOr) {
            self.advance();
            let right = self.parse_logical_and()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::LogicalOr,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse logical AND expressions
    fn parse_logical_and(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_bitwise_or()?;
        
        while self.check(&TokenKind::AndAnd) {
            self.advance();
            let right = self.parse_bitwise_or()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::LogicalAnd,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse bitwise OR expressions
    fn parse_bitwise_or(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_bitwise_xor()?;
        
        while self.check(&TokenKind::Pipe) {
            self.advance();
            let right = self.parse_bitwise_xor()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::BitwiseOr,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse bitwise XOR expressions
    fn parse_bitwise_xor(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_bitwise_and()?;
        
        while self.check(&TokenKind::Caret) {
            self.advance();
            let right = self.parse_bitwise_and()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::BitwiseXor,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse bitwise AND expressions
    fn parse_bitwise_and(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_equality()?;
        
        while self.check(&TokenKind::Ampersand) {
            self.advance();
            let right = self.parse_equality()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::BitwiseAnd,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse equality expressions
    fn parse_equality(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_comparison()?;
        
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Equal) | Some(TokenKind::NotEqual)) {
            let operator = match self.peek().unwrap().kind {
                TokenKind::Equal => BinaryOperator::Equal,
                TokenKind::NotEqual => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_comparison()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse comparison expressions
    fn parse_comparison(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_shift()?;
        
        while matches!(self.peek().map(|t| &t.kind), 
            Some(TokenKind::Less) | Some(TokenKind::Greater) | 
            Some(TokenKind::LessEqual) | Some(TokenKind::GreaterEqual)) {
            let operator = match self.peek().unwrap().kind {
                TokenKind::Less => BinaryOperator::Less,
                TokenKind::Greater => BinaryOperator::Greater,
                TokenKind::LessEqual => BinaryOperator::LessEqual,
                TokenKind::GreaterEqual => BinaryOperator::GreaterEqual,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_shift()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse shift expressions (left shift, right shift)
    fn parse_shift(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_addition()?;
        
        // Note: We don't have specific left/right shift tokens in our TokenKind
        // This would need to be added to the lexer for full C-like support
        
        Ok(expr)
    }
    
    /// Parse addition and subtraction
    fn parse_addition(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_multiplication()?;
        
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Plus) | Some(TokenKind::Minus)) {
            let operator = match self.peek().unwrap().kind {
                TokenKind::Plus => BinaryOperator::Add,
                TokenKind::Minus => BinaryOperator::Subtract,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_multiplication()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse multiplication, division, and modulo
    fn parse_multiplication(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_unary()?;
        
        while matches!(self.peek().map(|t| &t.kind), 
            Some(TokenKind::Star) | Some(TokenKind::Slash) | Some(TokenKind::Percent)) {
            let operator = match self.peek().unwrap().kind {
                TokenKind::Star => BinaryOperator::Multiply,
                TokenKind::Slash => BinaryOperator::Divide,
                TokenKind::Percent => BinaryOperator::Modulo,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_unary()?;
            expr = CLikeExpression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    /// Parse unary expressions
    fn parse_unary(&mut self) -> Result<CLikeExpression, CLikeError> {
        if matches!(self.peek().map(|t| &t.kind), 
            Some(TokenKind::Minus) | Some(TokenKind::Plus) | Some(TokenKind::Bang) | Some(TokenKind::Tilde)) {
            let operator = match self.peek().unwrap().kind {
                TokenKind::Minus => UnaryOperator::Minus,
                TokenKind::Plus => UnaryOperator::Plus,
                TokenKind::Bang => UnaryOperator::LogicalNot,
                TokenKind::Tilde => UnaryOperator::BitwiseNot,
                _ => unreachable!(),
            };
            self.advance();
            let operand = self.parse_unary()?;
            Ok(CLikeExpression::Unary {
                operator,
                operand: Box::new(operand),
            })
        } else {
            self.parse_postfix()
        }
    }
    
    /// Parse postfix expressions (calls, member access, indexing)
    fn parse_postfix(&mut self) -> Result<CLikeExpression, CLikeError> {
        let mut expr = self.parse_primary()?;
        
        loop {
            match self.peek().map(|t| &t.kind) {
                Some(TokenKind::LeftParen) => {
                    // Function call
                    self.advance(); // consume '('
                    let mut arguments = Vec::new();
                    
                    if !self.check(&TokenKind::RightParen) {
                        loop {
                            arguments.push(self.parse_expression()?);
                            if self.check(&TokenKind::Comma) {
                                self.advance();
                                if self.config.allow_trailing_commas && self.check(&TokenKind::RightParen) {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    
                    self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
                    
                    expr = CLikeExpression::Call {
                        function: Box::new(expr),
                        arguments,
                    };
                }
                Some(TokenKind::Dot) => {
                    // Member access
                    self.advance(); // consume '.'
                    let member = self.parse_identifier()?;
                    expr = CLikeExpression::MemberAccess {
                        object: Box::new(expr),
                        member,
                        safe_navigation: false,
                    };
                }
                Some(TokenKind::LeftBracket) => {
                    // Index access
                    self.advance(); // consume '['
                    let index = self.parse_expression()?;
                    self.consume(TokenKind::RightBracket, "Expected ']' after index")?;
                    expr = CLikeExpression::IndexAccess {
                        object: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }
    
    /// Parse primary expressions
    fn parse_primary(&mut self) -> Result<CLikeExpression, CLikeError> {
        match self.peek().map(|t| &t.kind) {
            Some(TokenKind::LeftParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(TokenKind::RightParen, "Expected ')' after expression")?;
                Ok(CLikeExpression::Parenthesized(Box::new(expr)))
            }
            Some(TokenKind::Identifier(_)) => {
                let name = self.parse_identifier()?;
                Ok(CLikeExpression::Identifier(name))
            }
            Some(TokenKind::StringLiteral(_)) => {
                if let Some(token) = self.advance() {
                    if let TokenKind::StringLiteral(s) = &token.kind {
                        Ok(CLikeExpression::Literal(CLikeLiteral::String(s.clone())))
                    } else {
                        unreachable!()
                    }
                } else {
                    Err(CLikeError::UnexpectedEof)
                }
            }
            Some(TokenKind::IntegerLiteral(_)) => {
                if let Some(token) = self.advance() {
                    if let TokenKind::IntegerLiteral(n) = &token.kind {
                        Ok(CLikeExpression::Literal(CLikeLiteral::Integer(*n)))
                    } else {
                        unreachable!()
                    }
                } else {
                    Err(CLikeError::UnexpectedEof)
                }
            }
            Some(TokenKind::FloatLiteral(_)) => {
                if let Some(token) = self.advance() {
                    if let TokenKind::FloatLiteral(f) = &token.kind {
                        Ok(CLikeExpression::Literal(CLikeLiteral::Float(*f)))
                    } else {
                        unreachable!()
                    }
                } else {
                    Err(CLikeError::UnexpectedEof)
                }
            }
            Some(TokenKind::True) => {
                self.advance();
                Ok(CLikeExpression::Literal(CLikeLiteral::Boolean(true)))
            }
            Some(TokenKind::False) => {
                self.advance();
                Ok(CLikeExpression::Literal(CLikeLiteral::Boolean(false)))
            }
            Some(TokenKind::Null) => {
                self.advance();
                Ok(CLikeExpression::Literal(CLikeLiteral::Null))
            }
            Some(TokenKind::LeftBracket) => {
                // Array literal
                self.advance(); // consume '['
                let mut elements = Vec::new();
                
                if !self.check(&TokenKind::RightBracket) {
                    loop {
                        elements.push(self.parse_expression()?);
                        if self.check(&TokenKind::Comma) {
                            self.advance();
                            if self.config.allow_trailing_commas && self.check(&TokenKind::RightBracket) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
                
                self.consume(TokenKind::RightBracket, "Expected ']' after array elements")?;
                Ok(CLikeExpression::ArrayLiteral(elements))
            }
            Some(TokenKind::LeftBrace) => {
                // Object literal
                self.advance(); // consume '{'
                let mut fields = Vec::new();
                
                if !self.check(&TokenKind::RightBrace) {
                    loop {
                        let key = if let Some(TokenKind::Identifier(_)) = self.peek().map(|t| &t.kind) {
                            self.parse_identifier()?
                        } else if let Some(TokenKind::StringLiteral(_)) = self.peek().map(|t| &t.kind) {
                            if let Some(token) = self.advance() {
                                if let TokenKind::StringLiteral(s) = &token.kind {
                                    s.clone()
                                } else {
                                    unreachable!()
                                }
                            } else {
                                return Err(CLikeError::UnexpectedEof);
                            }
                        } else {
                            return Err(CLikeError::UnexpectedToken {
                                expected: "property key".to_string(),
                                found: format!("{:?}", self.peek().unwrap().kind),
                                span: self.peek().unwrap().span,
                            });
                        };
                        
                        self.consume(TokenKind::Colon, "Expected ':' after object key")?;
                        let value = self.parse_expression()?;
                        
                        fields.push(CLikeObjectField { key, value });
                        
                        if self.check(&TokenKind::Comma) {
                            self.advance();
                            if self.config.allow_trailing_commas && self.check(&TokenKind::RightBrace) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
                
                self.consume(TokenKind::RightBrace, "Expected '}' after object fields")?;
                Ok(CLikeExpression::ObjectLiteral(fields))
            }
            _ => {
                let found = self.peek()
                    .map(|t| format!("{:?}", t.kind))
                    .unwrap_or_else(|| "EOF".to_string());
                let span = self.peek().map(|t| t.span).unwrap_or_else(Span::dummy);
                
                Err(CLikeError::UnexpectedToken {
                    expected: "expression".to_string(),
                    found,
                    span,
                })
            }
        }
    }
    
    /// Parse an identifier
    fn parse_identifier(&mut self) -> Result<String, CLikeError> {
        if let Some(token) = self.advance() {
            match &token.kind {
                TokenKind::Identifier(name) => Ok(name.clone()),
                _ => Err(CLikeError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", token.kind),
                    span: token.span,
                }),
            }
        } else {
            Err(CLikeError::UnexpectedEof)
        }
    }
    
    /// Parse a function parameter
    fn parse_parameter(&mut self) -> Result<CLikeParameter, CLikeError> {
        let name = self.parse_identifier()?;
        
        let param_type = if self.check(&TokenKind::Colon) {
            self.advance();
            Some(self.parse_identifier()?)
        } else {
            None
        };
        
        Ok(CLikeParameter { name, param_type })
    }
}

impl StyleParser for CLikeParser {
    type Output = CLikeSyntax;
    type Error = CLikeError;
    type Config = CLikeConfig;
    
    fn new() -> Self {
        Self {
            config: CLikeConfig::default(),
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
        SyntaxStyle::CLike
    }
    
    fn capabilities(&self) -> ParserCapabilities {
        ParserCapabilities {
            supports_mixed_indentation: true,
            supports_optional_semicolons: !self.config.require_semicolons,
            supports_trailing_commas: self.config.allow_trailing_commas,
            supports_nested_comments: true,
            error_recovery_level: ErrorRecoveryLevel::Advanced,
            max_nesting_depth: 256,
            supports_ai_metadata: true,
        }
    }
} 

#[cfg(test)]
mod tests {
    use super::*;
    use prism_lexer::{TokenKind, Lexer};
    use prism_common::span::Span;

    fn create_test_parser(source: &str) -> CLikeParser {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        CLikeParser::new(CLikeConfig::default(), tokens)
    }

    #[test]
    fn test_basic_function_parsing() {
        let source = r#"
            int main() {
                return 0;
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_function();
        
        assert!(result.is_ok());
        let function = result.unwrap();
        assert_eq!(function.name, "main");
        assert_eq!(function.parameters.len(), 0);
        assert_eq!(function.return_type, Some("int".to_string()));
        assert_eq!(function.body.len(), 1);
    }

    #[test]
    fn test_function_with_parameters() {
        let source = r#"
            void calculate(int x, float y, char* name) {
                printf("Calculating %d + %f for %s", x, y, name);
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_function();
        
        assert!(result.is_ok());
        let function = result.unwrap();
        assert_eq!(function.name, "calculate");
        assert_eq!(function.parameters.len(), 3);
        assert_eq!(function.parameters[0].name, "x");
        assert_eq!(function.parameters[0].param_type, Some("int".to_string()));
        assert_eq!(function.parameters[1].name, "y");
        assert_eq!(function.parameters[1].param_type, Some("float".to_string()));
        assert_eq!(function.parameters[2].name, "name");
        assert_eq!(function.parameters[2].param_type, Some("char*".to_string()));
    }

    #[test]
    fn test_if_statement_parsing() {
        let source = r#"
            if (x > 0) {
                printf("Positive");
            } else if (x < 0) {
                printf("Negative");
            } else {
                printf("Zero");
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::If { condition, then_body, else_body, .. } = result.unwrap() {
            assert!(matches!(condition, CLikeExpression::Binary { .. }));
            assert_eq!(then_body.len(), 1);
            assert!(else_body.is_some());
        } else {
            panic!("Expected If statement");
        }
    }

    #[test]
    fn test_for_loop_parsing() {
        let source = r#"
            for (int i = 0; i < 10; i++) {
                printf("%d\n", i);
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::For { init, condition, update, body, .. } = result.unwrap() {
            assert!(init.is_some());
            assert!(condition.is_some());
            assert!(update.is_some());
            assert_eq!(body.len(), 1);
        } else {
            panic!("Expected For statement");
        }
    }

    #[test]
    fn test_while_loop_parsing() {
        let source = r#"
            while (count < 100) {
                count *= 2;
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::While { condition, body, .. } = result.unwrap() {
            assert!(matches!(condition, CLikeExpression::Binary { .. }));
            assert_eq!(body.len(), 1);
        } else {
            panic!("Expected While statement");
        }
    }

    #[test]
    fn test_switch_statement_parsing() {
        let source = r#"
            switch (value) {
                case 1:
                    printf("One");
                    break;
                case 2:
                    printf("Two");
                    break;
                default:
                    printf("Other");
                    break;
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::Switch { expression, cases, .. } = result.unwrap() {
            assert!(matches!(expression, CLikeExpression::Identifier { .. }));
            assert_eq!(cases.len(), 3); // 2 cases + 1 default
        } else {
            panic!("Expected Switch statement");
        }
    }

    #[test]
    fn test_try_catch_parsing() {
        let source = r#"
            try {
                risky_operation();
            } catch (Exception e) {
                handle_error(e);
            } finally {
                cleanup();
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::Try { try_body, catch_blocks, finally_body, .. } = result.unwrap() {
            assert_eq!(try_body.len(), 1);
            assert_eq!(catch_blocks.len(), 1);
            assert!(finally_body.is_some());
            assert_eq!(finally_body.unwrap().len(), 1);
        } else {
            panic!("Expected Try statement");
        }
    }

    #[test]
    fn test_binary_expression_parsing() {
        let source = "a + b * c - d / e";
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_expression();
        
        assert!(result.is_ok());
        // Should respect operator precedence: a + (b * c) - (d / e)
        if let CLikeExpression::Binary { left, op, right, .. } = result.unwrap() {
            assert!(matches!(op, BinaryOperator::Subtract));
            assert!(matches!(**left, CLikeExpression::Binary { .. }));
            assert!(matches!(**right, CLikeExpression::Binary { .. }));
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_function_call_parsing() {
        let source = "printf(\"%d %s\", 42, \"hello\")";
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_expression();
        
        assert!(result.is_ok());
        if let CLikeExpression::Call { function, arguments, .. } = result.unwrap() {
            assert_eq!(function, "printf");
            assert_eq!(arguments.len(), 3);
        } else {
            panic!("Expected FunctionCall expression");
        }
    }

    #[test]
    fn test_array_access_parsing() {
        let source = "array[index + 1]";
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_expression();
        
        assert!(result.is_ok());
        if let CLikeExpression::IndexAccess { object, index, .. } = result.unwrap() {
            assert!(matches!(**object, CLikeExpression::Identifier { .. }));
            assert!(matches!(**index, CLikeExpression::Binary { .. }));
        } else {
            panic!("Expected ArrayAccess expression");
        }
    }

    #[test]
    fn test_object_member_access_parsing() {
        let source = "object.member.submember";
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_expression();
        
        assert!(result.is_ok());
        if let CLikeExpression::MemberAccess { object, member, .. } = result.unwrap() {
            assert!(matches!(**object, CLikeExpression::MemberAccess { .. }));
            assert_eq!(member, "submember");
        } else {
            panic!("Expected MemberAccess expression");
        }
    }

    #[test]
    fn test_assignment_parsing() {
        let source = "x += 5";
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::Assignment { target, op, value, .. } = result.unwrap() {
            assert!(matches!(target, CLikeExpression::Identifier { .. }));
            assert!(matches!(op, AssignmentOperator::AddAssign));
            assert!(matches!(value, CLikeExpression::Literal { .. }));
        } else {
            panic!("Expected Assignment statement");
        }
    }

    #[test]
    fn test_variable_declaration_parsing() {
        let source = "int x = 42, y = 24;";
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_statement();
        
        assert!(result.is_ok());
        if let CLikeStatement::VariableDeclaration { var_type, declarations, .. } = result.unwrap() {
            assert_eq!(var_type, "int");
            assert_eq!(declarations.len(), 2);
            assert_eq!(declarations[0].0, "x");
            assert_eq!(declarations[1].0, "y");
        } else {
            panic!("Expected VariableDeclaration statement");
        }
    }

    #[test]
    fn test_error_recovery_missing_semicolon() {
        let source = r#"
            int x = 42
            int y = 24;
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_program();
        
        // Should recover from missing semicolon
        assert!(result.is_ok());
        let syntax = result.unwrap();
        assert_eq!(syntax.statements.len(), 2);
    }

    #[test]
    fn test_error_recovery_unmatched_braces() {
        let source = r#"
            if (true) {
                printf("test");
            // Missing closing brace
            printf("after if");
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_program();
        
        // Should attempt recovery
        assert!(result.is_ok() || matches!(result, Err(CLikeError::UnmatchedBrace { .. })));
    }

    #[test]
    fn test_complex_nested_structure() {
        let source = r#"
            struct Point {
                int x, y;
            };
            
            int distance(struct Point p1, struct Point p2) {
                int dx = p1.x - p2.x;
                int dy = p1.y - p2.y;
                return sqrt(dx * dx + dy * dy);
            }
            
            int main() {
                struct Point points[10];
                for (int i = 0; i < 10; i++) {
                    points[i].x = i;
                    points[i].y = i * 2;
                }
                return 0;
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_program();
        
        assert!(result.is_ok());
        let syntax = result.unwrap();
        assert!(!syntax.functions.is_empty());
        assert!(!syntax.statements.is_empty());
    }

    #[test]
    fn test_comment_handling() {
        let source = r#"
            // Single line comment
            int x = 42; // End of line comment
            
            /*
             * Multi-line comment
             * with multiple lines
             */
            int y = 24;
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_program();
        
        assert!(result.is_ok());
        let syntax = result.unwrap();
        assert_eq!(syntax.statements.len(), 2);
    }

    #[test]
    fn test_string_literal_parsing() {
        let source = r#""Hello, \"world\"!\n""#;
        
        let mut parser = create_test_parser(source);
        let result = parser.parse_expression();
        
        assert!(result.is_ok());
        if let CLikeExpression::Literal { value, .. } = result.unwrap() {
            if let CLikeLiteral::String(s) = value {
                assert!(s.contains("Hello"));
                assert!(s.contains("world"));
            } else {
                panic!("Expected String literal");
            }
        } else {
            panic!("Expected Literal expression");
        }
    }

    #[test]
    fn test_numeric_literal_parsing() {
        let test_cases = vec![
            ("42", CLikeLiteral::Integer(42)),
            ("3.14", CLikeLiteral::Float(3.14)),
            ("0x1A", CLikeLiteral::Integer(26)),
            ("0755", CLikeLiteral::Integer(493)),
        ];
        
        for (source, expected) in test_cases {
            let mut parser = create_test_parser(source);
            let result = parser.parse_expression();
            
            assert!(result.is_ok(), "Failed to parse: {}", source);
            if let CLikeExpression::Literal { value, .. } = result.unwrap() {
                match (&value, &expected) {
                    (CLikeLiteral::Integer(a), CLikeLiteral::Integer(b)) => assert_eq!(a, b),
                    (CLikeLiteral::Float(a), CLikeLiteral::Float(b)) => assert!((a - b).abs() < f64::EPSILON),
                    _ => panic!("Literal type mismatch for: {}", source),
                }
            } else {
                panic!("Expected Literal expression for: {}", source);
            }
        }
    }

    #[test]
    fn test_operator_precedence() {
        let test_cases = vec![
            ("a + b * c", "should be parsed as a + (b * c)"),
            ("a * b + c", "should be parsed as (a * b) + c"),
            ("a && b || c", "should be parsed as (a && b) || c"),
            ("!a && b", "should be parsed as (!a) && b"),
        ];
        
        for (source, description) in test_cases {
            let mut parser = create_test_parser(source);
            let result = parser.parse_expression();
            
            assert!(result.is_ok(), "Failed to parse: {} ({})", source, description);
        }
    }

    #[test]
    fn test_configuration_options() {
        let mut config = CLikeConfig::default();
        config.require_semicolons = false;
        config.allow_trailing_commas = true;
        
        let source = "int x = 42"; // No semicolon
        let tokens = Vec::new(); // Simplified for test
        let parser = CLikeParser::new(config, tokens);
        
        // Configuration should be applied
        assert!(!parser.config.require_semicolons);
        assert!(parser.config.allow_trailing_commas);
    }

    #[test]
    fn test_error_types() {
        let error_cases = vec![
            (CLikeError::UnexpectedToken { expected: "identifier".to_string(), found: "number".to_string(), span: Span::default() }),
            (CLikeError::UnmatchedBrace { span: Span::default() }),
            (CLikeError::InvalidSyntax { message: "test".to_string(), span: Span::default() }),
        ];
        
        for error in error_cases {
            // Verify error types can be created and formatted
            let error_string = format!("{}", error);
            assert!(!error_string.is_empty());
        }
    }
} 