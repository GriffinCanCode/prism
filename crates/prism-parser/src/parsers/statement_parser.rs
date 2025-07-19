//! Statement Parsing
//!
//! This module embodies the single concept of "Statement Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing all statement types in the Prism language.
//!
//! **Conceptual Responsibility**: Parse all statement constructs
//! **What it does**: let bindings, if/else, loops, return, break, continue, yield
//! **What it doesn't do**: expression parsing, type parsing, token navigation

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager, parsing_coordinator::ParsingCoordinator},
    parsers::{expression_parser::ExpressionParser, type_parser::TypeParser, function_parser::FunctionParser},
};
use prism_ast::{AstNode, Stmt, VariableDecl, ConstDecl, Visibility, IfStmt, WhileStmt, ForStmt, MatchStmt, MatchArm, ReturnStmt, BreakStmt, ContinueStmt, ThrowStmt, TryStmt, CatchClause, BlockStmt, ExpressionStmt, Pattern, PatternKind};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;

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
        match self.token_stream.current_kind() {
            // Function declarations
            TokenKind::Function | TokenKind::Fn => {
                self.function_parser.parse_function()
            }
            
            // Variable declarations
            TokenKind::Let => self.parse_let_statement(),
            TokenKind::Const => self.parse_const_statement(),
            TokenKind::Var => self.parse_var_statement(),
            
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
            
            // Error handling
            TokenKind::Try => self.parse_try_statement(),
            TokenKind::Throw => self.parse_throw_statement(),
            
            // Block statement
            TokenKind::LeftBrace => self.parse_block_statement(),
            
            // Expression statement (fallback)
            _ => self.parse_expression_statement(),
        }
    }

    /// Parse a let statement: `let pattern: Type = expression;`
    fn parse_let_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Let)?;
        
        // Parse identifier (simplified pattern for now)
        let name = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Optional type annotation
        let type_annotation = if self.token_stream.consume(TokenKind::Colon) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Expect assignment
        self.token_stream.expect(TokenKind::Assign)?;
        
        // Parse initializer expression
        let initializer = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let variable_decl = Stmt::Variable(VariableDecl {
            name,
            type_annotation,
            initializer: Some(initializer),
            is_mutable: true, // let variables are mutable by default
            visibility: Visibility::Private, // local variables are private
        });
        
        Ok(self.coordinator.create_node(variable_decl, span))
    }

    /// Parse a const statement: `const IDENTIFIER: Type = expression;`
    fn parse_const_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Const)?;
        
        // Constants must have identifiers (no destructuring)
        let name = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Type annotation is required for constants
        self.token_stream.expect(TokenKind::Colon)?;
        let type_annotation = self.type_parser.parse_type()?;
        
        // Expect assignment
        self.token_stream.expect(TokenKind::Assign)?;
        
        // Parse initializer expression
        let value = self.expr_parser.parse_expression()?;
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let const_decl = Stmt::Const(ConstDecl {
            name,
            type_annotation: Some(type_annotation),
            value,
            visibility: Visibility::Private,
        });
        
        Ok(self.coordinator.create_node(const_decl, span))
    }

    /// Parse a var statement: `var identifier: Type = expression;`
    fn parse_var_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Var)?;
        
        // Parse identifier
        let name = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
        
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
            visibility: Visibility::Private,
        });
        
        Ok(self.coordinator.create_node(variable_decl, span))
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
        let variable = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
        
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
        
        // For now, represent loop as a while(true) statement
        let while_stmt = Stmt::While(WhileStmt {
            condition: AstNode::new(
                prism_ast::Expr::Literal(prism_ast::LiteralExpr {
                    value: prism_ast::LiteralValue::Boolean(true),
                }),
                span,
                prism_common::NodeId::new(0),
            ),
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
            // Parse pattern (simplified for now)
            let pattern = self.parse_simple_pattern()?;
            
            // Expect arrow
            self.token_stream.expect(TokenKind::Arrow)?;
            
            // Parse statement or expression
            let body = Box::new(if self.token_stream.check(TokenKind::LeftBrace) {
                self.parse_block_statement()?
            } else {
                self.parse_expression_statement()?
            });
            
            arms.push(MatchArm {
                pattern,
                guard: None, // TODO: Add guard support
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
        
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let return_stmt = Stmt::Return(ReturnStmt {
            value,
        });
        
        Ok(self.coordinator.create_node(return_stmt, span))
    }

    /// Parse a break statement: `break label?;`
    fn parse_break_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Break)?;
        
        // Optional break value (simplified - no labels for now)
        let value = if self.token_stream.check(TokenKind::Semicolon) {
            None
        } else {
            Some(self.expr_parser.parse_expression()?)
        };
        
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let break_stmt = Stmt::Break(BreakStmt {
            value,
        });
        
        Ok(self.coordinator.create_node(break_stmt, span))
    }

    /// Parse a continue statement: `continue label?;`
    fn parse_continue_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Continue)?;
        
        // Optional continue value (simplified - no labels for now)
        let value = if self.token_stream.check(TokenKind::Semicolon) {
            None
        } else {
            Some(self.expr_parser.parse_expression()?)
        };
        
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let continue_stmt = Stmt::Continue(ContinueStmt {
            value,
        });
        
        Ok(self.coordinator.create_node(continue_stmt, span))
    }

    /// Parse a yield statement: `yield expression;`
    fn parse_yield_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Yield)?;
        
        // Parse yield value
        let value = self.expr_parser.parse_expression()?;
        
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        // For now, represent yield as return statement
        let return_stmt = Stmt::Return(ReturnStmt {
            value: Some(value),
        });
        
        Ok(self.coordinator.create_node(return_stmt, span))
    }

    /// Parse a try statement: `try { ... } catch pattern { ... }`
    fn parse_try_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Try)?;
        
        // Parse try block
        let try_block = Box::new(self.parse_block_statement()?);
        
        // Parse catch clauses
        let mut catch_clauses = Vec::new();
        
        while self.token_stream.consume(TokenKind::Catch) {
            // Optional pattern for error binding
            let (variable, error_type) = if self.token_stream.check(TokenKind::LeftBrace) {
                (None, None)
            } else {
                let var_name = self.token_stream.expect_identifier()?;
                let var_symbol = prism_common::symbol::Symbol::intern(&var_name);
                // TODO: Parse optional type annotation
                (Some(var_symbol), None)
            };
            
            // Parse catch block
            let body = Box::new(self.parse_block_statement()?);
            
            catch_clauses.push(CatchClause {
                variable,
                error_type,
                body,
            });
        }
        
        let end_span = self.token_stream.current_span();
        let span = self.combine_spans(start_span, end_span);
        
        let try_stmt = Stmt::Try(TryStmt {
            try_block,
            catch_clauses,
            finally_block: None, // TODO: Add finally support
        });
        
        Ok(self.coordinator.create_node(try_stmt, span))
    }

    /// Parse a throw statement: `throw expression;`
    fn parse_throw_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Throw)?;
        
        // Parse error expression
        let exception = self.expr_parser.parse_expression()?;
        
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let throw_stmt = Stmt::Throw(ThrowStmt {
            exception,
        });
        
        Ok(self.coordinator.create_node(throw_stmt, span))
    }

    /// Parse a block statement: `{ statement* }`
    pub fn parse_block_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut statements = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            statements.push(self.parse_statement()?);
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = self.combine_spans(start_span, end_span);
        
        let block_stmt = Stmt::Block(BlockStmt {
            statements,
        });
        
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
        
        let expr_stmt = Stmt::Expression(ExpressionStmt {
            expression,
        });
        
        Ok(self.coordinator.create_node(expr_stmt, span))
    }

    /// Parse a simple pattern (for now, just identifiers and literals)
    fn parse_simple_pattern(&mut self) -> ParseResult<AstNode<Pattern>> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            // Identifier pattern
            TokenKind::Identifier(_) => {
                let name = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
                let span = self.token_stream.previous_span();
                
                Ok(AstNode::new(
                    Pattern {
                        kind: PatternKind::Identifier(name),
                    },
                    span,
                    prism_common::NodeId::new(0),
                ))
            }
            
            // Wildcard pattern
            TokenKind::Underscore => {
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                Ok(AstNode::new(
                    Pattern {
                        kind: PatternKind::Wildcard,
                    },
                    span,
                    prism_common::NodeId::new(0),
                ))
            }
            
            // Literal patterns
            TokenKind::IntegerLiteral(value) => {
                let value = *value;
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                Ok(AstNode::new(
                    Pattern {
                        kind: PatternKind::Literal(prism_ast::LiteralValue::Integer(value)),
                    },
                    span,
                    prism_common::NodeId::new(0),
                ))
            }
            
            TokenKind::StringLiteral(value) => {
                let value = value.clone();
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                Ok(AstNode::new(
                    Pattern {
                        kind: PatternKind::Literal(prism_ast::LiteralValue::String(value)),
                    },
                    span,
                    prism_common::NodeId::new(0),
                ))
            }
            
            TokenKind::True => {
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                Ok(AstNode::new(
                    Pattern {
                        kind: PatternKind::Literal(prism_ast::LiteralValue::Boolean(true)),
                    },
                    span,
                    prism_common::NodeId::new(0),
                ))
            }
            
            TokenKind::False => {
                self.token_stream.advance();
                let span = self.token_stream.previous_span();
                
                Ok(AstNode::new(
                    Pattern {
                        kind: PatternKind::Literal(prism_ast::LiteralValue::Boolean(false)),
                    },
                    span,
                    prism_common::NodeId::new(0),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test utilities would be defined here or in a test module

    #[test]
    fn test_placeholder() {
        // Placeholder test to prevent compilation errors
        assert!(true);
    }
} 