//! Expression Parsing
//!
//! This module embodies the single concept of "Expression Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing expressions using a Pratt parser approach.
//!
//! **Conceptual Responsibility**: Parse all expression types with correct precedence
//! **What it does**: prefix parsing, infix parsing, precedence handling
//! **What it doesn't do**: statement parsing, type parsing, token navigation

use crate::core::{
    error::{ParseError, ParseResult},
    token_stream_manager::TokenStreamManager,
    precedence::{associativity, infix_precedence, prefix_precedence, Precedence},
    parsing_coordinator::ParsingCoordinator,
};
use prism_ast::{AstNode, Expr, LiteralExpr, LiteralValue, BinaryExpr, UnaryExpr, CallExpr, MemberExpr, IndexExpr, IfExpr, BlockExpr, ErrorExpr, VariableExpr, BinaryOperator, UnaryOperator};
use prism_common::span::Span;
use prism_lexer::TokenKind;

/// Expression parser using Pratt parsing for correct precedence handling
/// 
/// This struct embodies the single concept of parsing expressions.
/// It uses the Pratt parsing technique to handle operator precedence
/// and associativity correctly for all Prism expression types.
pub struct ExpressionParser;

impl ExpressionParser {
    /// Parse an expression with default precedence (for statement parser compatibility)
    pub fn parse_expression(&mut self) -> ParseResult<AstNode<Expr>> {
        // This is a placeholder - in the real implementation, we would need
        // access to the coordinator. For now, create an error expression.
        let span = prism_common::span::Span::dummy();
        Ok(AstNode::new(
            Expr::Error(ErrorExpr {
                message: "Expression parsing not yet fully implemented".to_string(),
            }),
            span,
            prism_common::NodeId::new(0),
        ))
    }
    
    /// Parse an expression with minimum precedence
    /// 
    /// This is the main entry point for expression parsing using Pratt parser.
    /// It handles precedence and associativity correctly for all operators.
    pub fn parse_expression_with_precedence(
        coordinator: &mut ParsingCoordinator,
        min_precedence: Precedence,
    ) -> ParseResult<AstNode<Expr>> {
        let mut left = Self::parse_prefix(coordinator)?;

        while !coordinator.token_manager().is_at_end() {
            let token = &coordinator.token_manager().peek().kind;
            
            if let Some(precedence) = infix_precedence(token) {
                if precedence < min_precedence {
                    break;
                }
                
                left = Self::parse_infix(coordinator, left, precedence)?;
            } else {
                break;
            }
        }

        Ok(left)
    }
    
    /// Static method for parsing expressions (for compatibility with statement parser)
    pub fn parse_expression_static(
        coordinator: &mut ParsingCoordinator,
        precedence: Precedence,
    ) -> ParseResult<AstNode<Expr>> {
        Self::parse_expression_with_precedence(coordinator, precedence)
    }

    /// Parse a prefix expression (literals, identifiers, unary operators)
    fn parse_prefix(coordinator: &mut ParsingCoordinator) -> ParseResult<AstNode<Expr>> {
        let token = coordinator.token_manager().peek().clone();
        let span = token.span;

        match &token.kind {
            // Literals
            TokenKind::IntegerLiteral(value) => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Integer(*value),
                    }),
                    span,
                ))
            }
            TokenKind::FloatLiteral(value) => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Float(*value),
                    }),
                    span,
                ))
            }
            TokenKind::StringLiteral(value) => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::String(value.clone()),
                    }),
                    span,
                ))
            }
            TokenKind::True => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Boolean(true),
                    }),
                    span,
                ))
            }
            TokenKind::False => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Boolean(false),
                    }),
                    span,
                ))
            }
            TokenKind::Null => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Literal(LiteralExpr {
                        value: LiteralValue::Null,
                    }),
                    span,
                ))
            }

            // Identifiers
            TokenKind::Identifier(name) => {
                coordinator.token_manager_mut().advance();
                Ok(coordinator.create_node(
                    Expr::Variable(VariableExpr {
                        name: prism_common::symbol::Symbol::intern(name),
                    }),
                    span,
                ))
            }

            // Unary operators
            TokenKind::Bang | TokenKind::Minus | TokenKind::Plus => {
                let operator = Self::token_to_unary_operator(&token.kind)?;
                coordinator.token_manager_mut().advance();
                
                let operand = Self::parse_expression_with_precedence(coordinator, Precedence::Unary)?;
                let end_span = operand.span;
                let full_span = Span::new(span.start, end_span.end, span.source_id);
                
                Ok(coordinator.create_node(
                    Expr::Unary(UnaryExpr {
                        operator,
                        operand: Box::new(operand),
                    }),
                    full_span,
                ))
            }

            // Parenthesized expressions
            TokenKind::LeftParen => {
                coordinator.token_manager_mut().advance(); // consume '('
                let expr = Self::parse_expression_with_precedence(coordinator, Precedence::Assignment)?;
                coordinator.token_manager_mut().expect(TokenKind::RightParen)?;
                Ok(expr)
            }

            // Block expressions
            TokenKind::LeftBrace => {
                Self::parse_block_expression(coordinator)
            }

            // Conditional expressions (if expressions)
            TokenKind::If => {
                Self::parse_conditional_expression(coordinator)
            }

            _ => {
                // Create error expression for recovery
                coordinator.add_error(ParseError::invalid_syntax(
                    "expression".to_string(),
                    format!("Unexpected token: {:?}", token.kind),
                    span,
                ));
                
                Ok(coordinator.create_node(
                    Expr::Error(ErrorExpr {
                        message: format!("Invalid expression starting with {:?}", token.kind),
                    }),
                    span,
                ))
            }
        }
    }

    /// Parse an infix expression (binary operators, calls, member access)
    fn parse_infix(
        coordinator: &mut ParsingCoordinator,
        left: AstNode<Expr>,
        precedence: Precedence,
    ) -> ParseResult<AstNode<Expr>> {
        let token = coordinator.token_manager().peek().clone();
        let token_kind = token.kind.clone();

        match token_kind {
            // Binary operators
            TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash |
            TokenKind::Percent | TokenKind::Equal | TokenKind::NotEqual |
            TokenKind::Less | TokenKind::Greater | TokenKind::LessEqual | TokenKind::GreaterEqual |
            TokenKind::AndAnd | TokenKind::OrOr | TokenKind::Ampersand | TokenKind::Pipe |
            TokenKind::Caret | TokenKind::Power => {
                let operator = Self::token_to_binary_operator(&token_kind)?;
                coordinator.token_manager_mut().advance();
                
                let assoc = associativity(&token_kind);
                let next_precedence = match assoc {
                    crate::precedence::Associativity::Left => Self::next_precedence(precedence),
                    crate::precedence::Associativity::Right => precedence,
                    crate::precedence::Associativity::None => Self::next_precedence(precedence),
                };
                
                let right = Self::parse_expression(coordinator, next_precedence)?;
                let end_span = right.span;
                let full_span = Span::new(left.span.start, end_span.end, left.span.source_id);
                
                Ok(coordinator.create_node(
                    Expr::Binary(BinaryExpr {
                        left: Box::new(left),
                        operator,
                        right: Box::new(right),
                    }),
                    full_span,
                ))
            }

            // Function calls
            TokenKind::LeftParen => {
                Self::parse_call_expression(coordinator, left)
            }

            // Member access
            TokenKind::Dot => {
                Self::parse_member_access(coordinator, left)
            }

            // Array/index access
            TokenKind::LeftBracket => {
                Self::parse_index_expression(coordinator, left)
            }

            _ => {
                // This shouldn't happen if precedence tables are correct
                Ok(left)
            }
        }
    }

    /// Parse a function call expression
    fn parse_call_expression(
        coordinator: &mut ParsingCoordinator,
        function: AstNode<Expr>,
    ) -> ParseResult<AstNode<Expr>> {
        coordinator.token_manager_mut().advance(); // consume '('
        
        let mut arguments = Vec::new();
        
        if !coordinator.token_manager().check(TokenKind::RightParen) {
            loop {
                let arg = Self::parse_expression(coordinator, Precedence::Assignment)?;
                arguments.push(arg);
                
                if coordinator.token_manager().check(TokenKind::Comma) {
                    coordinator.token_manager_mut().advance();
                } else {
                    break;
                }
            }
        }
        
        let end_token = coordinator.token_manager_mut().consume(TokenKind::RightParen, "Expected ')' after arguments")?;
        let full_span = Span::new(function.span.start, end_token.span.end, function.span.source_id);
        
        Ok(coordinator.create_node(
            Expr::Call(CallExpr {
                callee: Box::new(function),
                arguments,
                type_arguments: None,
                call_style: prism_ast::CallStyle::Function,
            }),
            full_span,
        ))
    }

    /// Parse member access expression (obj.member)
    fn parse_member_access(
        coordinator: &mut ParsingCoordinator,
        object: AstNode<Expr>,
    ) -> ParseResult<AstNode<Expr>> {
        coordinator.token_manager_mut().advance(); // consume '.'
        
        let member_token = coordinator.token_manager().peek().clone();
        if let TokenKind::Identifier(member_name) = &member_token.kind {
            coordinator.token_manager_mut().advance();
            let full_span = Span::new(object.span.start, member_token.span.end, object.span.source_id);
            
            Ok(coordinator.create_node(
                Expr::Member(MemberExpr {
                    object: Box::new(object),
                    member: prism_common::symbol::Symbol::intern(member_name),
                    safe_navigation: false,
                }),
                full_span,
            ))
        } else {
            Err(ParseError::invalid_syntax(
                "member_access".to_string(),
                "Expected identifier after '.'".to_string(),
                member_token.span,
            ))
        }
    }

    /// Parse index expression (array[index])
    fn parse_index_expression(
        coordinator: &mut ParsingCoordinator,
        object: AstNode<Expr>,
    ) -> ParseResult<AstNode<Expr>> {
        coordinator.token_manager_mut().advance(); // consume '['
        
        let index = Self::parse_expression(coordinator, Precedence::Assignment)?;
        let end_token = coordinator.token_manager_mut().consume(TokenKind::RightBracket, "Expected ']' after index")?;
        let full_span = Span::new(object.span.start, end_token.span.end, object.span.source_id);
        
        Ok(coordinator.create_node(
            Expr::Index(IndexExpr {
                object: Box::new(object),
                index: Box::new(index),
            }),
            full_span,
        ))
    }

    /// Parse block expression { ... }
    fn parse_block_expression(coordinator: &mut ParsingCoordinator) -> ParseResult<AstNode<Expr>> {
        let start_token = coordinator.token_manager_mut().consume(TokenKind::LeftBrace, "Expected '{'")?;
        let mut statements = Vec::new();
        
        while !coordinator.token_manager().check(TokenKind::RightBrace) && !coordinator.token_manager().is_at_end() {
            // Delegate to statement parser (to be implemented)
            // For now, create a placeholder
            break;
        }
        
        let end_token = coordinator.token_manager_mut().consume(TokenKind::RightBrace, "Expected '}'")?;
        let full_span = Span::new(start_token.span.start, end_token.span.end, start_token.span.source_id);
        
        Ok(coordinator.create_node(
            Expr::Block(BlockExpr {
                statements,
                final_expression: None,
            }),
            full_span,
        ))
    }

    /// Parse conditional expression (if-then-else)
    fn parse_conditional_expression(coordinator: &mut ParsingCoordinator) -> ParseResult<AstNode<Expr>> {
        let start_token = coordinator.token_manager_mut().consume(TokenKind::If, "Expected 'if'")?;
        
        let condition = Self::parse_expression(coordinator, Precedence::Assignment)?;
        let then_branch = Self::parse_expression(coordinator, Precedence::Assignment)?;
        
        let else_branch = if coordinator.token_manager().check(TokenKind::Else) {
            coordinator.token_manager_mut().advance();
            Some(Box::new(Self::parse_expression(coordinator, Precedence::Assignment)?))
        } else {
            None
        };
        
        let end_span = else_branch.as_ref().map(|e| e.span).unwrap_or(then_branch.span);
        let full_span = Span::new(start_token.span.start, end_span.end, start_token.span.source_id);
        
        Ok(coordinator.create_node(
            Expr::If(IfExpr {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch,
            }),
            full_span,
        ))
    }

    // Helper methods for operator conversion

    /// Convert token to unary operator
    fn token_to_unary_operator(token: &TokenKind) -> ParseResult<UnaryOperator> {
        match token {
            TokenKind::Bang => Ok(UnaryOperator::Not),
            TokenKind::Minus => Ok(UnaryOperator::Negate),
            _ => Err(ParseError::invalid_syntax(
                "unary_operator".to_string(),
                format!("Invalid unary operator: {:?}", token),
                Span::dummy(),
            )),
        }
    }

    /// Convert token to binary operator
    fn token_to_binary_operator(token: &TokenKind) -> ParseResult<BinaryOperator> {
        match token {
            TokenKind::Plus => Ok(BinaryOperator::Add),
            TokenKind::Minus => Ok(BinaryOperator::Subtract),
            TokenKind::Star => Ok(BinaryOperator::Multiply),
            TokenKind::Slash => Ok(BinaryOperator::Divide),
            TokenKind::Percent => Ok(BinaryOperator::Modulo),
            TokenKind::Equal => Ok(BinaryOperator::Equal),
            TokenKind::NotEqual => Ok(BinaryOperator::NotEqual),
            TokenKind::Less => Ok(BinaryOperator::Less),
            TokenKind::Greater => Ok(BinaryOperator::Greater),
            TokenKind::LessEqual => Ok(BinaryOperator::LessEqual),
            TokenKind::GreaterEqual => Ok(BinaryOperator::GreaterEqual),
            TokenKind::AndAnd => Ok(BinaryOperator::And),
            TokenKind::OrOr => Ok(BinaryOperator::Or),
            TokenKind::Ampersand => Ok(BinaryOperator::BitAnd),
            TokenKind::Pipe => Ok(BinaryOperator::BitOr),
            TokenKind::Caret => Ok(BinaryOperator::BitXor),
            TokenKind::Power => Ok(BinaryOperator::Power),
            _ => Err(ParseError::invalid_syntax(
                "binary_operator".to_string(),
                format!("Invalid binary operator: {:?}", token),
                Span::dummy(),
            )),
        }
    }

    /// Get next precedence level for left-associative operators
    fn next_precedence(current: Precedence) -> Precedence {
        match current {
            Precedence::None => Precedence::Assignment,
            Precedence::Assignment => Precedence::Or,
            Precedence::Or => Precedence::And,
            Precedence::And => Precedence::BitOr,
            Precedence::BitOr => Precedence::BitXor,
            Precedence::BitXor => Precedence::BitAnd,
            Precedence::BitAnd => Precedence::Equality,
            Precedence::Equality => Precedence::Comparison,
            Precedence::Comparison => Precedence::Shift,
            Precedence::Shift => Precedence::Term,
            Precedence::Term => Precedence::Factor,
            Precedence::Factor => Precedence::Unary,
            Precedence::Unary => Precedence::Power,
            Precedence::Power => Precedence::Call,
            Precedence::Call => Precedence::Primary,
            Precedence::Primary => Precedence::Primary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::parsing_coordinator::ParsingCoordinator;
    use prism_common::{span::Position, SourceId};

    fn create_test_token(kind: TokenKind) -> prism_lexer::Token {
        prism_lexer::Token {
            kind,
            span: Span::new(
                Position::new(1, 1, 0),
                Position::new(1, 2, 1),
                SourceId::new(1),
            ),
            semantic_context: None,
        }
    }

    #[test]
    fn test_parse_literal() {
        let tokens = vec![create_test_token(TokenKind::IntegerLiteral(42))];
        let mut coordinator = ParsingCoordinator::new(tokens);

        let expr = ExpressionParser::parse_expression(&mut coordinator, Precedence::Assignment);
        assert!(expr.is_ok());
        
        if let Ok(expr) = expr {
            if let Expr::Literal(literal) = &expr.kind {
                if let LiteralValue::Integer(value) = &literal.value {
                    assert_eq!(*value, 42);
                } else {
                    panic!("Expected integer literal");
                }
            } else {
                panic!("Expected literal expression");
            }
        }
    }

    #[test]
    fn test_parse_binary_expression() {
        let tokens = vec![
            create_test_token(TokenKind::IntegerLiteral(1)),
            create_test_token(TokenKind::Plus),
            create_test_token(TokenKind::IntegerLiteral(2)),
        ];
        let mut coordinator = ParsingCoordinator::new(tokens);

        let expr = ExpressionParser::parse_expression(&mut coordinator, Precedence::Assignment);
        assert!(expr.is_ok());
        
        if let Ok(expr) = expr {
            assert!(matches!(expr.kind, Expr::Binary(_)));
        }
    }
} 