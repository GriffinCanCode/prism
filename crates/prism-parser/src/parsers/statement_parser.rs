//! Statement Parsing
//!
//! This module embodies the single concept of "Statement Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing all statement types in the Prism language.
//!
//! **Conceptual Responsibility**: Parse all statement constructs
//! **What it does**: let bindings, if/else, loops, return, break, continue, yield, async/await, 
//!                   match statements, try/catch, type declarations, imports/exports
//! **What it doesn't do**: expression parsing, type parsing, token navigation

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager, parsing_coordinator::ParsingCoordinator},
    parsers::{expression_parser::ExpressionParser, type_parser::TypeParser, function_parser::FunctionParser},
};
use prism_ast::{
    AstNode, Stmt, VariableDecl, ConstDecl, Visibility, IfStmt, WhileStmt, ForStmt, MatchStmt, MatchArm, 
    ReturnStmt, BreakStmt, ContinueStmt, ThrowStmt, TryStmt, CatchClause, BlockStmt, ExpressionStmt, 
    Pattern, TypeDecl, TypeKind, ImportDecl, ImportItems, ImportItem, ExportDecl, ExportItem,
    ModuleDecl, SectionDecl, SectionKind, StabilityLevel, Attribute, AttributeArgument, ErrorStmt, Expr,
    LiteralValue, Type, FunctionDecl, ObjectPatternField
};
use prism_lexer::{Token, TokenKind};
use prism_common::{span::Span, symbol::Symbol};
use std::collections::HashMap;

/// Statement parser - handles all statement types
/// 
/// This struct embodies the single concept of parsing statements.
/// It delegates to other parsers for sub-components but owns
/// the logic for statement structure and semantics.
pub struct StatementParser<'a> {
    /// Reference to the token stream manager (no ownership)
    token_stream: &'a mut TokenStreamManager,
    /// Reference to expression parser for statement expressions
    expr_parser: &'a mut ExpressionParser,
    /// Reference to type parser for type annotations
    type_parser: &'a mut TypeParser<'a>,
    /// Reference to function parser for function declarations
    function_parser: &'a mut FunctionParser<'a>,
    /// Reference to coordinator for error handling
    coordinator: &'a mut ParsingCoordinator,
}

impl<'a> StatementParser<'a> {
    /// Create a new statement parser
    pub fn new(
        token_stream: &'a mut TokenStreamManager,
        expr_parser: &'a mut ExpressionParser,
        type_parser: &'a mut TypeParser<'a>,
        function_parser: &'a mut FunctionParser<'a>,
        coordinator: &'a mut ParsingCoordinator,
    ) -> Self {
        Self {
            token_stream,
            expr_parser,
            type_parser,
            function_parser,
            coordinator,
        }
    }
    
    /// Helper function to combine spans safely
    fn combine_spans(&self, start: Span, end: Span) -> Span {
        start.combine(&end).unwrap_or(start)
    }

    /// Parse a statement
    pub fn parse_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        // Handle attributes first if present
        let attributes = self.parse_attributes()?;
        
        // Handle visibility modifiers
        let visibility = self.parse_visibility()?;
        
        match self.token_stream.current_kind() {
            // Function declarations
            TokenKind::Function | TokenKind::Fn => {
                let func_decl = self.function_parser.parse_function()?;
                let span = func_decl.span;
                Ok(self.coordinator.create_node(
                    Stmt::Function(func_decl.kind),
                    span,
                ))
            }
            
            // Variable declarations
            TokenKind::Let => self.parse_let_statement(attributes, visibility),
            TokenKind::Const => self.parse_const_statement(attributes, visibility),
            TokenKind::Var => self.parse_var_statement(attributes, visibility),
            
            // Type declarations
            TokenKind::Type => self.parse_type_declaration(attributes, visibility),
            
            // Control flow
            TokenKind::If => self.parse_if_statement(),
            TokenKind::While => self.parse_while_statement(),
            TokenKind::For => self.parse_for_statement(),
            TokenKind::Loop => self.parse_loop_statement(),
            TokenKind::Match => self.parse_match_statement(),
            
            // Flow control
            TokenKind::Return => self.parse_return_statement(),
            TokenKind::Break => self.parse_break_statement(),
            TokenKind::Continue => self.parse_continue_statement(),
            TokenKind::Yield => self.parse_yield_statement(),
            
            // Async/await
            TokenKind::Async => self.parse_async_statement(),
            TokenKind::Await => self.parse_await_statement(),
            
            // Error handling
            TokenKind::Try => self.parse_try_statement(),
            TokenKind::Throw => self.parse_throw_statement(),
            
            // Import/Export
            TokenKind::Import => self.parse_import_statement(),
            TokenKind::Export => self.parse_export_statement(),
            
            // Module/Section declarations
            TokenKind::Module => self.parse_module_statement(attributes, visibility),
            TokenKind::Section => self.parse_section_statement(attributes, visibility),
            
            // Block statement
            TokenKind::LeftBrace => self.parse_block_statement(),
            
            // Expression statement (fallback)
            _ => self.parse_expression_statement(),
        }
    }

    /// Parse attributes (e.g., @deprecated, @test)
    fn parse_attributes(&mut self) -> ParseResult<Vec<Attribute>> {
        let mut attributes = Vec::new();
        
        while self.token_stream.check(TokenKind::At) {
            self.token_stream.advance(); // consume '@'
            
            let name = self.token_stream.expect_identifier()?;
            let mut arguments = Vec::new();
            
            // Parse attribute arguments if present
            if self.token_stream.consume(TokenKind::LeftParen) {
                while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
                    let arg = self.parse_attribute_argument()?;
                    arguments.push(arg);
                    
                    if !self.token_stream.check(TokenKind::RightParen) {
                        self.token_stream.expect(TokenKind::Comma)?;
                    }
                }
                self.token_stream.expect(TokenKind::RightParen)?;
            }
            
            attributes.push(Attribute { name, arguments });
        }
        
        Ok(attributes)
    }
    
    /// Parse an attribute argument
    fn parse_attribute_argument(&mut self) -> ParseResult<AttributeArgument> {
        // Check for named argument (name = value)
        if matches!(self.token_stream.current_kind(), TokenKind::Identifier(_)) && matches!(self.token_stream.peek().kind, TokenKind::Assign) {
            let name = self.token_stream.expect_identifier()?;
            self.token_stream.expect(TokenKind::Assign)?;
            let value = self.parse_literal_value()?;
            Ok(AttributeArgument::Named { name, value })
        } else {
            // Positional argument
            let value = self.parse_literal_value()?;
            Ok(AttributeArgument::Literal(value))
        }
    }
    
    /// Parse a literal value for attributes
    fn parse_literal_value(&mut self) -> ParseResult<LiteralValue> {
        match self.token_stream.current_kind() {
            TokenKind::IntegerLiteral(value) => {
                let value = *value;
                self.token_stream.advance();
                Ok(LiteralValue::Integer(value))
            }
            TokenKind::FloatLiteral(value) => {
                let value = *value;
                self.token_stream.advance();
                Ok(LiteralValue::Float(value))
            }
            TokenKind::StringLiteral(value) => {
                let value = value.clone();
                self.token_stream.advance();
                Ok(LiteralValue::String(value))
            }
            TokenKind::True => {
                self.token_stream.advance();
                Ok(LiteralValue::Boolean(true))
            }
            TokenKind::False => {
                self.token_stream.advance();
                Ok(LiteralValue::Boolean(false))
            }
            TokenKind::Null => {
                self.token_stream.advance();
                Ok(LiteralValue::Null)
            }
            _ => {
                let span = self.token_stream.current_span();
                Err(ParseError::unexpected_token(
                    vec![TokenKind::IntegerLiteral(0), TokenKind::StringLiteral("".to_string())],
                    self.token_stream.current_kind().clone(),
                    span,
                ))
            }
        }
    }

    /// Parse visibility modifier
    fn parse_visibility(&mut self) -> ParseResult<Visibility> {
        match self.token_stream.current_kind() {
            TokenKind::Public | TokenKind::Pub => {
                self.token_stream.advance();
                Ok(Visibility::Public)
            }
            TokenKind::Private => {
                self.token_stream.advance();
                Ok(Visibility::Private)
            }
            TokenKind::Internal => {
                self.token_stream.advance();
                Ok(Visibility::Internal)
            }
            _ => Ok(Visibility::Private), // Default visibility
        }
    }

    /// Parse a let statement: `let pattern: Type = expression;`
    fn parse_let_statement(&mut self, attributes: Vec<Attribute>, visibility: Visibility) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Let)?;
        
        // Parse pattern (for now, simplified to identifier)
        let name = Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Optional type annotation
        let type_annotation = if self.token_stream.consume(TokenKind::Colon) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Optional initializer
        let initializer = if self.token_stream.consume(TokenKind::Assign) {
            Some(self.expr_parser.parse_expression()?)
        } else {
            None
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let variable_decl = Stmt::Variable(VariableDecl {
            name,
            type_annotation,
            initializer,
            is_mutable: true, // let variables are mutable by default
            visibility,
        });
        
        Ok(self.coordinator.create_node(variable_decl, span))
    }

    /// Parse a const statement: `const IDENTIFIER: Type = expression;`
    fn parse_const_statement(&mut self, attributes: Vec<Attribute>, visibility: Visibility) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Const)?;
        
        // Parse identifier
        let name = Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Optional type annotation
        let type_annotation = if self.token_stream.consume(TokenKind::Colon) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Expect assignment
        self.token_stream.expect(TokenKind::Assign)?;
        
        // Parse value expression
        let value = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let const_decl = Stmt::Const(ConstDecl {
            name,
            type_annotation,
            value,
            visibility,
        });
        
        Ok(self.coordinator.create_node(const_decl, span))
    }

    /// Parse a var statement: `var identifier: Type = expression;`
    fn parse_var_statement(&mut self, attributes: Vec<Attribute>, visibility: Visibility) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Var)?;
        
        // Parse identifier
        let name = Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Optional type annotation
        let type_annotation = if self.token_stream.consume(TokenKind::Colon) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Optional initializer
        let initializer = if self.token_stream.consume(TokenKind::Assign) {
            Some(self.expr_parser.parse_expression()?)
        } else {
            None
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let variable_decl = Stmt::Variable(VariableDecl {
            name,
            type_annotation,
            initializer,
            is_mutable: true, // var variables are mutable
            visibility,
        });
        
        Ok(self.coordinator.create_node(variable_decl, span))
    }

    /// Parse a type declaration: `type Name<T> = Type;`
    fn parse_type_declaration(&mut self, attributes: Vec<Attribute>, visibility: Visibility) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Type)?;
        
        // Parse type name
        let name = Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Parse optional type parameters
        let mut type_parameters = Vec::new();
        if self.token_stream.consume(TokenKind::Less) {
            while !self.token_stream.check(TokenKind::Greater) && !self.token_stream.is_at_end() {
                let param_name = Symbol::intern(&self.token_stream.expect_identifier()?);
                
                // Optional bounds
                let mut bounds = Vec::new();
                if self.token_stream.consume(TokenKind::Colon) {
                    bounds.push(self.type_parser.parse_type()?);
                    
                    while self.token_stream.consume(TokenKind::Plus) {
                        bounds.push(self.type_parser.parse_type()?);
                    }
                }
                
                // Optional default
                let default = if self.token_stream.consume(TokenKind::Assign) {
                    Some(self.type_parser.parse_type()?)
                } else {
                    None
                };
                
                type_parameters.push(prism_ast::TypeParameter {
                    name: param_name,
                    bounds,
                    default,
                });
                
                if !self.token_stream.check(TokenKind::Greater) {
                    self.token_stream.expect(TokenKind::Comma)?;
                }
            }
            self.token_stream.expect(TokenKind::Greater)?;
        }
        
        // Expect assignment
        self.token_stream.expect(TokenKind::Assign)?;
        
        // Parse type definition
        let type_def = self.type_parser.parse_type()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let type_decl = Stmt::Type(TypeDecl {
            name,
            type_parameters,
            kind: TypeKind::Alias(type_def),
            visibility,
        });
        
        Ok(self.coordinator.create_node(type_decl, span))
    }

    /// Parse an if statement: `if condition { ... } else { ... }`
    fn parse_if_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::If)?;
        
        // Parse condition
        let condition = self.expr_parser.parse_expression()?;
        
        // Parse then block
        let then_branch = Box::new(self.parse_block_statement()?);
        
        // Optional else clause
        let else_branch = if self.token_stream.consume(TokenKind::Else) {
            if self.token_stream.check(TokenKind::If) {
                // else if - parse as another if statement
                Some(Box::new(self.parse_if_statement()?))
            } else {
                // else block
                Some(Box::new(self.parse_block_statement()?))
            }
        } else {
            None
        };
        
        let end_span = self.token_stream.current_span();
        let span = self.combine_spans(start_span, end_span);
        
        let if_stmt = Stmt::If(IfStmt {
            condition,
            then_branch,
            else_branch,
        });
        
        Ok(self.coordinator.create_node(if_stmt, span))
    }

    /// Parse a while statement: `while condition { ... }`
    fn parse_while_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::While)?;
        
        // Parse condition
        let condition = self.expr_parser.parse_expression()?;
        
        // Parse body
        let body = Box::new(self.parse_block_statement()?);
        
        let end_span = self.token_stream.current_span();
        let span = self.combine_spans(start_span, end_span);
        
        let while_stmt = Stmt::While(WhileStmt {
            condition,
            body,
        });
        
        Ok(self.coordinator.create_node(while_stmt, span))
    }

    /// Parse a for statement: `for pattern in iterable { ... }`
    fn parse_for_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::For)?;
        
        // Parse variable (simplified pattern for now)
        let variable = Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Expect 'in'
        self.token_stream.expect(TokenKind::In)?;
        
        // Parse iterable expression
        let iterable = self.expr_parser.parse_expression()?;
        
        // Parse body
        let body = Box::new(self.parse_block_statement()?);
        
        let end_span = self.token_stream.current_span();
        let span = self.combine_spans(start_span, end_span);
        
        let for_stmt = Stmt::For(ForStmt {
            variable,
            iterable,
            body,
        });
        
        Ok(self.coordinator.create_node(for_stmt, span))
    }

    /// Parse a loop statement: `loop { ... }`
    fn parse_loop_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Loop)?;
        
        // Parse body
        let body = Box::new(self.parse_block_statement()?);
        
        let end_span = self.token_stream.current_span();
        let span = self.combine_spans(start_span, end_span);
        
        // Represent loop as a while(true) statement
        let condition = self.coordinator.create_node(
            Expr::Literal(prism_ast::LiteralExpr {
                value: LiteralValue::Boolean(true),
            }),
            span,
        );
        
        let while_stmt = Stmt::While(WhileStmt {
            condition,
            body,
        });
        
        Ok(self.coordinator.create_node(while_stmt, span))
    }

    /// Parse a match statement: `match expression { pattern => statement, ... }`
    fn parse_match_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Match)?;
        
        // Parse scrutinee expression
        let expression = self.expr_parser.parse_expression()?;
        
        // Parse match arms
        self.token_stream.expect(TokenKind::LeftBrace)?;
        let mut arms = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            // Parse pattern
            let pattern = self.parse_pattern()?;
            
            // Optional guard
            let guard = if self.token_stream.consume(TokenKind::If) {
                Some(self.expr_parser.parse_expression()?)
            } else {
                None
            };
            
            // Expect arrow
            self.token_stream.expect(TokenKind::FatArrow)?;
            
            // Parse body
            let body = Box::new(if self.token_stream.check(TokenKind::LeftBrace) {
                self.parse_block_statement()?
            } else {
                self.parse_expression_statement()?
            });
            
            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });
            
            // Optional comma
            if !self.token_stream.check(TokenKind::RightBrace) {
                self.token_stream.expect(TokenKind::Comma)?;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let match_stmt = Stmt::Match(MatchStmt {
            expression,
            arms,
        });
        
        Ok(self.coordinator.create_node(match_stmt, span))
    }

    /// Parse a pattern for match statements
    fn parse_pattern(&mut self) -> ParseResult<AstNode<Pattern>> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            // Identifier pattern
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                // Check for constructor pattern: Name(patterns...)
                if self.token_stream.check(TokenKind::LeftParen) {
                    self.token_stream.advance(); // consume '('
                    let mut fields = Vec::new();
                    
                    while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
                        fields.push(self.parse_pattern()?);
                        if !self.token_stream.check(TokenKind::RightParen) {
                            self.token_stream.expect(TokenKind::Comma)?;
                        }
                    }
                    
                    self.token_stream.expect(TokenKind::RightParen)?;
                    let end_span = self.token_stream.previous_span();
                    let span = self.combine_spans(start_span, end_span);
                    
                                         // For now, represent constructor as a tuple pattern
                     // TODO: Add proper constructor pattern support to AST
                     Ok(self.coordinator.create_node(
                         Pattern::Tuple(fields),
                         span,
                     ))
                } else {
                                         // Simple identifier pattern
                     Ok(self.coordinator.create_node(
                         Pattern::Identifier(Symbol::intern(&name)),
                         span,
                     ))
                }
            }
            
            // Wildcard pattern
            TokenKind::Identifier(name) if name == "_" => {
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Wildcard,
                     span,
                 ))
            }
            
            // Literal patterns
            TokenKind::IntegerLiteral(value) => {
                let value = *value;
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Literal(LiteralValue::Integer(value)),
                     span,
                 ))
            }
            
            TokenKind::StringLiteral(value) => {
                let value = value.clone();
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Literal(LiteralValue::String(value)),
                     span,
                 ))
            }
            
            TokenKind::True => {
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Literal(LiteralValue::Boolean(true)),
                     span,
                 ))
            }
            
            TokenKind::False => {
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Literal(LiteralValue::Boolean(false)),
                     span,
                 ))
            }
            
            // Tuple pattern: (pattern1, pattern2, ...)
            TokenKind::LeftParen => {
                self.token_stream.advance(); // consume '('
                let mut elements = Vec::new();
                
                while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
                    elements.push(self.parse_pattern()?);
                    if !self.token_stream.check(TokenKind::RightParen) {
                        self.token_stream.expect(TokenKind::Comma)?;
                    }
                }
                
                self.token_stream.expect(TokenKind::RightParen)?;
                let end_span = self.token_stream.previous_span();
                let span = self.combine_spans(start_span, end_span);
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Tuple(elements),
                     span,
                 ))
            }
            
            // Array pattern: [pattern1, pattern2, ...]
            TokenKind::LeftBracket => {
                self.token_stream.advance(); // consume '['
                let mut elements = Vec::new();
                
                while !self.token_stream.check(TokenKind::RightBracket) && !self.token_stream.is_at_end() {
                    elements.push(self.parse_pattern()?);
                    if !self.token_stream.check(TokenKind::RightBracket) {
                        self.token_stream.expect(TokenKind::Comma)?;
                    }
                }
                
                self.token_stream.expect(TokenKind::RightBracket)?;
                let end_span = self.token_stream.previous_span();
                let span = self.combine_spans(start_span, end_span);
                
                                 Ok(self.coordinator.create_node(
                     Pattern::Array(elements),
                     span,
                 ))
            }
            
            _ => {
                let span = self.token_stream.current_span();
                Err(ParseError::unexpected_token(
                    vec![TokenKind::Identifier("pattern".to_string())],
                    self.token_stream.current_kind().clone(),
                    span,
                ))
            }
        }
    }

    /// Parse a return statement: `return expression?;`
    fn parse_return_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Return)?;
        
        // Optional return value
        let value = if self.token_stream.check(TokenKind::Semicolon) {
            None
        } else {
            Some(self.expr_parser.parse_expression()?)
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let return_stmt = Stmt::Return(ReturnStmt { value });
        
        Ok(self.coordinator.create_node(return_stmt, span))
    }

    /// Parse a break statement: `break expression?;`
    fn parse_break_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Break)?;
        
        // Optional break value
        let value = if self.token_stream.check(TokenKind::Semicolon) {
            None
        } else {
            Some(self.expr_parser.parse_expression()?)
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let break_stmt = Stmt::Break(BreakStmt { value });
        
        Ok(self.coordinator.create_node(break_stmt, span))
    }

    /// Parse a continue statement: `continue expression?;`
    fn parse_continue_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Continue)?;
        
        // Optional continue value
        let value = if self.token_stream.check(TokenKind::Semicolon) {
            None
        } else {
            Some(self.expr_parser.parse_expression()?)
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let continue_stmt = Stmt::Continue(ContinueStmt { value });
        
        Ok(self.coordinator.create_node(continue_stmt, span))
    }

    /// Parse a yield statement: `yield expression;`
    fn parse_yield_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Yield)?;
        
        // Parse yield value
        let value = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        // For now, represent yield as an expression statement
        let yield_expr = self.coordinator.create_node(
            Expr::Yield(prism_ast::YieldExpr { value: Some(Box::new(value)) }),
            span,
        );
        
        let expr_stmt = Stmt::Expression(ExpressionStmt { expression: yield_expr });
        
        Ok(self.coordinator.create_node(expr_stmt, span))
    }

    /// Parse an async statement: `async { ... }` or `async function ...`
    fn parse_async_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Async)?;
        
        match self.token_stream.current_kind() {
            TokenKind::Function | TokenKind::Fn => {
                // async function - delegate to function parser
                let func_decl = self.function_parser.parse_function()?;
                let span = func_decl.span;
                Ok(self.coordinator.create_node(
                    Stmt::Function(func_decl.kind),
                    span,
                ))
            }
            TokenKind::LeftBrace => {
                // async block
                let block = self.parse_block_statement()?;
                let end_span = self.token_stream.current_span();
                let span = self.combine_spans(start_span, end_span);
                
                                 // Wrap in async expression
                 let async_expr = self.coordinator.create_node(
                     Expr::Block(prism_ast::BlockExpr {
                         statements: match &block.kind {
                             Stmt::Block(block_stmt) => block_stmt.statements.clone(),
                             _ => vec![block],
                         },
                         final_expr: None,
                     }),
                     span,
                 );
                
                let expr_stmt = Stmt::Expression(ExpressionStmt {
                    expression: async_expr,
                });
                
                Ok(self.coordinator.create_node(expr_stmt, span))
            }
            _ => {
                let span = self.token_stream.current_span();
                Err(ParseError::unexpected_token(
                    vec![TokenKind::Function, TokenKind::LeftBrace],
                    self.token_stream.current_kind().clone(),
                    span,
                ))
            }
        }
    }

    /// Parse an await statement: `await expression;`
    fn parse_await_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Await)?;
        
        // Parse awaited expression
        let expression = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        // Wrap in await expression
        let await_expr = self.coordinator.create_node(
            Expr::Await(prism_ast::AwaitExpr {
                expression: Box::new(expression),
            }),
            span,
        );
        
        let expr_stmt = Stmt::Expression(ExpressionStmt {
            expression: await_expr,
        });
        
        Ok(self.coordinator.create_node(expr_stmt, span))
    }

    /// Parse a try statement: `try { ... } catch error { ... }`
    fn parse_try_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Try)?;
        
        // Parse try block
        let try_block = Box::new(self.parse_block_statement()?);
        
        // Parse catch clauses
        let mut catch_clauses = Vec::new();
        
        while self.token_stream.consume(TokenKind::Catch) {
            // Optional error variable and type
            let (variable, error_type) = if self.token_stream.check(TokenKind::LeftBrace) {
                (None, None)
            } else {
                let var_name = self.token_stream.expect_identifier()?;
                let var_symbol = Symbol::intern(&var_name);
                
                // Optional type annotation
                let error_type = if self.token_stream.consume(TokenKind::Colon) {
                    Some(self.type_parser.parse_type()?)
                } else {
                    None
                };
                
                (Some(var_symbol), error_type)
            };
            
            // Parse catch block
            let body = Box::new(self.parse_block_statement()?);
            
            catch_clauses.push(CatchClause {
                variable,
                error_type,
                body,
            });
        }
        
                 // Optional finally block  
         let finally_block = if matches!(self.token_stream.current_kind(), TokenKind::Identifier(name) if name == "finally") {
             self.token_stream.advance(); // consume 'finally'
             Some(Box::new(self.parse_block_statement()?))
        } else {
            None
        };
        
        let end_span = self.token_stream.current_span();
        let span = self.combine_spans(start_span, end_span);
        
        let try_stmt = Stmt::Try(TryStmt {
            try_block,
            catch_clauses,
            finally_block,
        });
        
        Ok(self.coordinator.create_node(try_stmt, span))
    }

    /// Parse a throw statement: `throw expression;`
    fn parse_throw_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Throw)?;
        
        // Parse exception expression
        let exception = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let throw_stmt = Stmt::Throw(ThrowStmt { exception });
        
        Ok(self.coordinator.create_node(throw_stmt, span))
    }

    /// Parse an import statement: `import { items } from "module";`
    fn parse_import_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Import)?;
        
        // Parse import items
        let items = if self.token_stream.consume(TokenKind::Star) {
            // import * from "module"
            ImportItems::All
        } else if self.token_stream.consume(TokenKind::LeftBrace) {
            // import { item1, item2 as alias } from "module"
            let mut specific_items = Vec::new();
            
            while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
                let name = Symbol::intern(&self.token_stream.expect_identifier()?);
                
                let alias = if self.token_stream.consume(TokenKind::As) {
                    Some(Symbol::intern(&self.token_stream.expect_identifier()?))
                } else {
                    None
                };
                
                specific_items.push(ImportItem { name, alias });
                
                if !self.token_stream.check(TokenKind::RightBrace) {
                    self.token_stream.expect(TokenKind::Comma)?;
                }
            }
            
            self.token_stream.expect(TokenKind::RightBrace)?;
            ImportItems::Specific(specific_items)
        } else {
            // Default import: import name from "module"
            let name = Symbol::intern(&self.token_stream.expect_identifier()?);
            ImportItems::Specific(vec![ImportItem { name, alias: None }])
        };
        
        // Expect 'from'
        self.token_stream.expect(TokenKind::From)?;
        
        // Parse module path
        let path = match self.token_stream.current_kind() {
            TokenKind::StringLiteral(path) => {
                let path = path.clone();
                self.token_stream.advance();
                path
            }
            _ => {
                let span = self.token_stream.current_span();
                return Err(ParseError::unexpected_token(
                    vec![TokenKind::StringLiteral("module".to_string())],
                    self.token_stream.current_kind().clone(),
                    span,
                ));
            }
        };
        
        // Optional alias for the entire import
        let alias = if self.token_stream.consume(TokenKind::As) {
            Some(Symbol::intern(&self.token_stream.expect_identifier()?))
        } else {
            None
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let import_stmt = Stmt::Import(ImportDecl {
            path,
            items,
            alias,
        });
        
        Ok(self.coordinator.create_node(import_stmt, span))
    }

    /// Parse an export statement: `export { items };`
    fn parse_export_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Export)?;
        
        let items = if self.token_stream.consume(TokenKind::LeftBrace) {
            let mut export_items = Vec::new();
            
            while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
                let name = Symbol::intern(&self.token_stream.expect_identifier()?);
                
                let alias = if self.token_stream.consume(TokenKind::As) {
                    Some(Symbol::intern(&self.token_stream.expect_identifier()?))
                } else {
                    None
                };
                
                export_items.push(ExportItem { name, alias });
                
                if !self.token_stream.check(TokenKind::RightBrace) {
                    self.token_stream.expect(TokenKind::Comma)?;
                }
            }
            
            self.token_stream.expect(TokenKind::RightBrace)?;
            export_items
        } else {
            // Single export
            let name = Symbol::intern(&self.token_stream.expect_identifier()?);
            vec![ExportItem { name, alias: None }]
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let export_stmt = Stmt::Export(ExportDecl { items });
        
        Ok(self.coordinator.create_node(export_stmt, span))
    }

    /// Parse a module statement: `module Name { ... }`
    fn parse_module_statement(&mut self, attributes: Vec<Attribute>, visibility: Visibility) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Module)?;
        
        // Parse module name
        let name = Symbol::intern(&self.token_stream.expect_identifier()?);
        
                 // Optional capability
         let capability = if self.token_stream.consume(TokenKind::Colon) {
             Some(self.token_stream.expect_identifier()?)
         } else {
             None
         };
        
        // Parse module body
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut sections = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            if self.token_stream.check(TokenKind::Section) {
                let section = self.parse_section_statement(Vec::new(), Visibility::Private)?;
                if let Stmt::Section(section_decl) = section.kind {
                    sections.push(self.coordinator.create_node(section_decl, section.span));
                }
            } else {
                // Skip non-section statements for now
                self.token_stream.advance();
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let module_stmt = Stmt::Module(ModuleDecl {
            name,
            capability,
            description: None,
            dependencies: Vec::new(),
            stability: StabilityLevel::Experimental,
            version: None,
            sections,
            ai_context: None,
            visibility,
        });
        
        Ok(self.coordinator.create_node(module_stmt, span))
    }

    /// Parse a section statement: `section Config { ... }`
    fn parse_section_statement(&mut self, attributes: Vec<Attribute>, visibility: Visibility) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Section)?;
        
        // Parse section kind
        let kind_name = self.token_stream.expect_identifier()?;
        let kind = match kind_name.as_str() {
            "Config" => SectionKind::Config,
            "Types" => SectionKind::Types,
            "Errors" => SectionKind::Errors,
            "Internal" => SectionKind::Internal,
            "Interface" => SectionKind::Interface,
            "Events" => SectionKind::Events,
            "Lifecycle" => SectionKind::Lifecycle,
            "Tests" => SectionKind::Tests,
            "Examples" => SectionKind::Examples,
            _ => SectionKind::Custom(kind_name),
        };
        
        // Parse section body
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut items = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let stmt = self.parse_statement()?;
            items.push(stmt);
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let section_stmt = Stmt::Section(SectionDecl {
            kind,
            items,
            visibility,
        });
        
        Ok(self.coordinator.create_node(section_stmt, span))
    }

    /// Parse a block statement: `{ statements... }`
    fn parse_block_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut statements = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let stmt = self.parse_statement()?;
            statements.push(stmt);
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let block_stmt = Stmt::Block(BlockStmt { statements });
        
        Ok(self.coordinator.create_node(block_stmt, span))
    }

    /// Parse an expression statement: `expression;`
    fn parse_expression_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        
        // Parse expression
        let expression = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let expr_stmt = Stmt::Expression(ExpressionStmt { expression });
        
        Ok(self.coordinator.create_node(expr_stmt, span))
    }

    /// Parse error recovery - create error statement
    pub fn parse_error_statement(&mut self, message: String) -> AstNode<Stmt> {
        let span = self.token_stream.current_span();
        
        let error_stmt = Stmt::Error(ErrorStmt {
            message,
            context: "statement parsing".to_string(),
        });
        
        self.coordinator.create_node(error_stmt, span)
    }

    /// Synchronize after error - skip to next statement boundary
    pub fn synchronize(&mut self) {
        self.token_stream.advance();
        
        while !self.token_stream.is_at_end() {
            if matches!(self.token_stream.previous().kind, TokenKind::Semicolon) {
                return;
            }
            
            match self.token_stream.current_kind() {
                TokenKind::Function | TokenKind::Fn | TokenKind::Let | TokenKind::Const |
                TokenKind::Var | TokenKind::If | TokenKind::While | TokenKind::For |
                TokenKind::Return | TokenKind::Try | TokenKind::Throw => return,
                _ => {}
            }
            
            self.token_stream.advance();
        }
    }
}
