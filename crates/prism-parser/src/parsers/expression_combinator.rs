//! Expression Parsing with Stream Combinators
//!
//! This module bridges the existing Pratt parser with the new stream-based Parse trait system.
//! It provides Parse implementations for complex expressions while maintaining compatibility
//! with the existing precedence handling.
//!
//! **Conceptual Responsibility**: Bridge Pratt parser with stream-based parsing
//! **What it does**: Parse trait implementations for complex expressions, precedence integration
//! **What it doesn't do**: token navigation, AST node creation (delegates to ExpressionParser)

use crate::core::{Parse, ParseStream, ParseResult, Precedence};
use crate::parsers::expression_parser::ExpressionParser;
use crate::stream_combinators::{comma_separated, parenthesized, bracketed, optional, alternative};
use prism_ast::{
    AstNode, Expr, BinaryExpr, UnaryExpr, CallExpr, MemberExpr, IndexExpr, 
    ArrayExpr, ObjectExpr, ObjectField, BinaryOperator, UnaryOperator, CallStyle
};
use prism_lexer::TokenKind;
use prism_common::{span::Span, symbol::Symbol, NodeId};

/// Parse implementation for binary expressions
impl Parse for BinaryExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // This is a simplified implementation - in practice, we'd delegate to the Pratt parser
        // through the coordinator, but for demonstration we'll parse a simple binary expression
        
        // Parse left operand (this would normally be handled by precedence)
        let left = input.parse::<AstNode<Expr>>()?;
        
        // Parse operator
        let operator = input.parse::<BinaryOperator>()?;
        
        // Parse right operand
        let right = input.parse::<AstNode<Expr>>()?;
        
        Ok(BinaryExpr {
            left: Box::new(left),
            operator,
            right: Box::new(right),
        })
    }
}

/// Parse implementation for unary expressions
impl Parse for UnaryExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let operator = input.parse::<UnaryOperator>()?;
        let operand = input.parse::<AstNode<Expr>>()?;
        
        Ok(UnaryExpr {
            operator,
            operand: Box::new(operand),
        })
    }
}

/// Parse implementation for binary operators
impl Parse for BinaryOperator {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token();
        match &token.kind {
            TokenKind::Plus => {
                input.advance();
                Ok(BinaryOperator::Add)
            }
            TokenKind::Minus => {
                input.advance();
                Ok(BinaryOperator::Subtract)
            }
            TokenKind::Star => {
                input.advance();
                Ok(BinaryOperator::Multiply)
            }
            TokenKind::Slash => {
                input.advance();
                Ok(BinaryOperator::Divide)
            }
            TokenKind::Percent => {
                input.advance();
                Ok(BinaryOperator::Modulo)
            }
            TokenKind::Equal => {
                input.advance();
                Ok(BinaryOperator::Equal)
            }
            TokenKind::NotEqual => {
                input.advance();
                Ok(BinaryOperator::NotEqual)
            }
            TokenKind::Less => {
                input.advance();
                Ok(BinaryOperator::Less)
            }
            TokenKind::Greater => {
                input.advance();
                Ok(BinaryOperator::Greater)
            }
            TokenKind::LessEqual => {
                input.advance();
                Ok(BinaryOperator::LessEqual)
            }
            TokenKind::GreaterEqual => {
                input.advance();
                Ok(BinaryOperator::GreaterEqual)
            }
            TokenKind::AndAnd => {
                input.advance();
                Ok(BinaryOperator::And)
            }
            TokenKind::OrOr => {
                input.advance();
                Ok(BinaryOperator::Or)
            }
            TokenKind::Ampersand => {
                input.advance();
                Ok(BinaryOperator::BitAnd)
            }
            TokenKind::Pipe => {
                input.advance();
                Ok(BinaryOperator::BitOr)
            }
            TokenKind::Caret => {
                input.advance();
                Ok(BinaryOperator::BitXor)
            }
            TokenKind::Power => {
                input.advance();
                Ok(BinaryOperator::Power)
            }
            _ => Err(crate::core::ParseError::invalid_syntax(
                "binary_operator".to_string(),
                format!("Expected binary operator, found {:?}", token.kind),
                token.span,
            )),
        }
    }
}

/// Parse implementation for unary operators
impl Parse for UnaryOperator {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token();
        match &token.kind {
            TokenKind::Bang => {
                input.advance();
                Ok(UnaryOperator::Not)
            }
            TokenKind::Minus => {
                input.advance();
                Ok(UnaryOperator::Negate)
            }
            TokenKind::Plus => {
                input.advance();
                Ok(UnaryOperator::Plus)
            }
            _ => Err(crate::core::ParseError::invalid_syntax(
                "unary_operator".to_string(),
                format!("Expected unary operator, found {:?}", token.kind),
                token.span,
            )),
        }
    }
}

/// Parse implementation for function call expressions
impl Parse for CallExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // This assumes we're parsing the call part after we already have the callee
        // In practice, this would be handled by the infix parsing in the Pratt parser
        
        // For demonstration, we'll create a placeholder
        // In real usage, this would be called from the expression parser when it sees '('
        Err(crate::core::ParseError::invalid_syntax(
            "call_expression".to_string(),
            "CallExpr parsing should be handled by the Pratt parser".to_string(),
            input.span(),
        ))
    }
}

/// Parse implementation for member access expressions  
impl Parse for MemberExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Similar to CallExpr, this would normally be handled by infix parsing
        Err(crate::core::ParseError::invalid_syntax(
            "member_expression".to_string(),
            "MemberExpr parsing should be handled by the Pratt parser".to_string(),
            input.span(),
        ))
    }
}

/// Parse implementation for index expressions
impl Parse for IndexExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Similar to CallExpr, this would normally be handled by infix parsing
        Err(crate::core::ParseError::invalid_syntax(
            "index_expression".to_string(),
            "IndexExpr parsing should be handled by the Pratt parser".to_string(),
            input.span(),
        ))
    }
}

/// Enhanced Parse implementation for ObjectExpr with proper field parsing
impl Parse for ObjectExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Parse { field1: value1, field2: value2, ... }
        if !input.check(TokenKind::LeftBrace) {
            return Err(crate::core::ParseError::expected_token(
                TokenKind::LeftBrace,
                input.span(),
            ));
        }
        
        input.advance(); // consume '{'
        
        let mut fields = Vec::new();
        
        // Handle empty object
        if input.check(TokenKind::RightBrace) {
            input.advance(); // consume '}'
            return Ok(ObjectExpr { fields });
        }
        
        // Parse fields separated by commas
        loop {
            let field = input.parse::<ObjectField>()?;
            fields.push(field);
            
            if input.check(TokenKind::Comma) {
                input.advance(); // consume ','
                
                // Check for trailing comma
                if input.check(TokenKind::RightBrace) {
                    break;
                }
            } else if input.check(TokenKind::RightBrace) {
                break;
            } else {
                return Err(crate::core::ParseError::unexpected_token(
                    vec![TokenKind::Comma, TokenKind::RightBrace],
                    input.peek_token().kind.clone(),
                    input.span(),
                ));
            }
        }
        
        if !input.check(TokenKind::RightBrace) {
            return Err(crate::core::ParseError::expected_token(
                TokenKind::RightBrace,
                input.span(),
            ));
        }
        input.advance(); // consume '}'
        
        Ok(ObjectExpr { fields })
    }
}

/// A wrapper for integrating with the existing Pratt parser
pub struct PrattExpression(pub AstNode<Expr>);

impl Parse for PrattExpression {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // This is where we bridge to the existing Pratt parser
        // We use the coordinator to delegate to ExpressionParser
        let coordinator = input.coordinator();
        let expr = ExpressionParser::parse_expression_with_precedence(coordinator, Precedence::Assignment)?;
        Ok(PrattExpression(expr))
    }
}

/// Helper function to create AST nodes with proper metadata
fn create_expr_node(expr: Expr, span: Span) -> AstNode<Expr> {
    AstNode::new(
        expr,
        span,
        NodeId::new(0), // Would use proper ID generation in real implementation
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TokenStreamManager, ParsingCoordinator};
    use prism_lexer::Token;
    use prism_common::{span::Position, SourceId};

    fn create_test_tokens(kinds: Vec<TokenKind>) -> Vec<Token> {
        let source_id = SourceId::new(0);
        kinds.into_iter().enumerate().map(|(i, kind)| {
            let start = Position::new(i as u32, i as u32, i as u32);
            let end = Position::new(i as u32, i as u32 + 1, i as u32 + 1);
            Token::new(kind, Span::new(start, end, source_id))
        }).collect()
    }

    #[test]
    fn test_binary_operator_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::Plus]);
        let mut token_manager = TokenStreamManager::new(tokens.clone());
        let mut coordinator = ParsingCoordinator::new(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<BinaryOperator>();
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), BinaryOperator::Add));
    }

    #[test]
    fn test_unary_operator_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::Bang]);
        let mut token_manager = TokenStreamManager::new(tokens.clone());
        let mut coordinator = ParsingCoordinator::new(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<UnaryOperator>();
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), UnaryOperator::Not));
    }

    #[test]
    fn test_object_expression_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
        ]);
        let mut token_manager = TokenStreamManager::new(tokens.clone());
        let mut coordinator = ParsingCoordinator::new(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ObjectExpr>();
        assert!(result.is_ok());
        let obj = result.unwrap();
        assert!(obj.fields.is_empty());
    }
} 