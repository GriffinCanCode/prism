//! Parse implementations for AST types
//!
//! This module implements the Parse trait for core AST types,
//! enabling type-driven parsing. Located in prism-parser to avoid circular dependencies.

use crate::core::{Parse, ParseStream, ParseResult, ParseToken, ParseError};
use crate::stream_combinators::{delimited, separated_list, comma_separated, optional, alternative};
use crate::parsers::expression_combinator::PrattExpression;

use prism_ast::{
    AstNode, Expr, LiteralExpr, LiteralValue, VariableExpr, BinaryExpr, BinaryOperator,
    UnaryExpr, UnaryOperator, CallExpr, MemberExpr, IndexExpr, ArrayExpr, ObjectExpr, 
    ObjectField, ObjectKey,
};

use prism_lexer::TokenKind;
use prism_common::{symbol::Symbol, NodeId};

impl Parse for LiteralExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token().clone(); // Clone to avoid borrowing issues
        
        let value = match &token.kind {
            TokenKind::IntegerLiteral(n) => {
                input.advance();
                LiteralValue::Integer(*n)
            }
            TokenKind::FloatLiteral(f) => {
                input.advance();
                LiteralValue::Float(*f)
            }
            TokenKind::StringLiteral(s) => {
                input.advance();
                LiteralValue::String(s.clone())
            }
            TokenKind::True => {
                input.advance();
                LiteralValue::Boolean(true)
            }
            TokenKind::False => {
                input.advance();
                LiteralValue::Boolean(false)
            }
            TokenKind::Null => {
                input.advance();
                LiteralValue::Null
            }
            _ => return Err(ParseError::expected_literal(input.span())),
        };

        Ok(LiteralExpr { value })
    }
}

impl Parse for VariableExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token().clone(); // Clone to avoid borrowing issues
        
        match &token.kind {
            TokenKind::Identifier(name) => {
                input.advance();
                Ok(VariableExpr {
                    name: Symbol::intern(name),
                })
            }
            _ => Err(ParseError::expected_identifier(input.span())),
        }
    }
}

impl Parse for ObjectKey {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.peek_token().clone(); // Clone to avoid borrowing issues
        
        match &token.kind {
            TokenKind::Identifier(name) => {
                input.advance();
                Ok(ObjectKey::Identifier(Symbol::intern(name)))
            }
            TokenKind::StringLiteral(s) => {
                input.advance();
                Ok(ObjectKey::String(s.clone()))
            }
            TokenKind::LeftBracket => {
                // Computed key: [expression]
                let _bracket = input.advance();
                let expr = input.parse::<AstNode<Expr>>()?;
                let _close_bracket = input.parse_token::<TokenKind>()?; // Should be RightBracket
                Ok(ObjectKey::Computed(expr))
            }
            _ => Err(ParseError::expected_object_key(input.span())),
        }
    }
}

impl Parse for ObjectField {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let key = input.parse::<ObjectKey>()?;
        let _colon = input.parse_token::<TokenKind>()?; // Should be Colon
        let value = input.parse::<AstNode<Expr>>()?;
        
        Ok(ObjectField { key, value })
    }
}

/// Enhanced ArrayExpr parsing using the new stream combinators
impl Parse for ArrayExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Expect opening bracket
        if !input.check(TokenKind::LeftBracket) {
            return Err(ParseError::expected_token(TokenKind::LeftBracket, input.span()));
        }
        input.advance(); // consume [
        
        // Handle empty array
        if input.check(TokenKind::RightBracket) {
            input.advance(); // consume ]
            return Ok(ArrayExpr { elements: Vec::new() });
        }
        
        // Parse comma-separated expressions
        let elements = comma_separated::<AstNode<Expr>>(input)?;
        
        // Expect closing bracket
        if !input.check(TokenKind::RightBracket) {
            return Err(ParseError::expected_token(TokenKind::RightBracket, input.span()));
        }
        input.advance(); // consume ]
        
        Ok(ArrayExpr { elements })
    }
}

// Helper function to create AST nodes with proper metadata
fn create_ast_node<T>(kind: T, span: prism_common::span::Span) -> AstNode<T> {
    use prism_ast::metadata::NodeMetadata;
    AstNode {
        kind,
        span,
        id: NodeId::new(0), // Would use proper ID generation in real implementation
        metadata: NodeMetadata::default(),
    }
}

/// Enhanced expression parsing that integrates with the Pratt parser
impl Parse for AstNode<Expr> {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Try different expression types using the checkpoint system for safe backtracking
        
        // First try simple expressions that don't require precedence handling
        let simple_expressions = [
            // Literals
            |input: &mut ParseStream| -> ParseResult<AstNode<Expr>> {
                let literal = input.parse::<LiteralExpr>()?;
                Ok(create_ast_node(Expr::Literal(literal), input.span()))
            },
            // Variables
            |input: &mut ParseStream| -> ParseResult<AstNode<Expr>> {
                let variable = input.parse::<VariableExpr>()?;
                Ok(create_ast_node(Expr::Variable(variable), input.span()))
            },
            // Arrays
            |input: &mut ParseStream| -> ParseResult<AstNode<Expr>> {
                let array = input.parse::<ArrayExpr>()?;
                Ok(create_ast_node(Expr::Array(array), input.span()))
            },
            // Objects
            |input: &mut ParseStream| -> ParseResult<AstNode<Expr>> {
                let object = input.parse::<ObjectExpr>()?;
                Ok(create_ast_node(Expr::Object(object), input.span()))
            },
        ];
        
        // Try simple expressions first
        if let Ok(expr) = alternative(input, &simple_expressions) {
            return Ok(expr);
        }
        
        // For complex expressions with precedence, delegate to the Pratt parser
        let pratt_expr = input.parse::<PrattExpression>()?;
        Ok(pratt_expr.0)
    }
}

/// Parse implementation for parenthesized expressions
pub struct ParenthesizedExpr(pub AstNode<Expr>);

impl Parse for ParenthesizedExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        if !input.check(TokenKind::LeftParen) {
            return Err(ParseError::expected_token(TokenKind::LeftParen, input.span()));
        }
        
        input.advance(); // consume '('
        let expr = input.parse::<AstNode<Expr>>()?;
        
        if !input.check(TokenKind::RightParen) {
            return Err(ParseError::expected_token(TokenKind::RightParen, input.span()));
        }
        input.advance(); // consume ')'
        
        Ok(ParenthesizedExpr(expr))
    }
}

/// Parse implementation for expression lists (used in function calls, etc.)
pub struct ExpressionList(pub Vec<AstNode<Expr>>);

impl Parse for ExpressionList {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let expressions = comma_separated::<AstNode<Expr>>(input)?;
        Ok(ExpressionList(expressions))
    }
}

/// Parse implementation for optional expression lists
pub struct OptionalExpressionList(pub Vec<AstNode<Expr>>);

impl Parse for OptionalExpressionList {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let expressions = optional::<ExpressionList>(input)?
            .map(|list| list.0)
            .unwrap_or_default();
        Ok(OptionalExpressionList(expressions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TokenStreamManager, ParsingCoordinator};
    use prism_lexer::Token;
    use prism_common::{span::{Position, Span}, SourceId};

    fn create_test_tokens(kinds: Vec<TokenKind>) -> Vec<Token> {
        let source_id = SourceId::new(0);
        kinds.into_iter().enumerate().map(|(i, kind)| {
            let start = Position::new(i as u32, i as u32, i as u32);
            let end = Position::new(i as u32, i as u32 + 1, i as u32 + 1);
            Token::new(kind, Span::new(start, end, source_id))
        }).collect()
    }

    fn create_parse_stream(tokens: Vec<Token>) -> (TokenStreamManager, ParsingCoordinator) {
        let token_manager = TokenStreamManager::new(tokens.clone());
        let coordinator = ParsingCoordinator::new(tokens);
        (token_manager, coordinator)
    }

    #[test]
    fn test_enhanced_array_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftBracket,
            TokenKind::IntegerLiteral(1),
            TokenKind::Comma,
            TokenKind::IntegerLiteral(2),
            TokenKind::Comma,
            TokenKind::IntegerLiteral(3),
            TokenKind::RightBracket,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ArrayExpr>();
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert_eq!(array.elements.len(), 3);
    }

    #[test]
    fn test_empty_array_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ArrayExpr>();
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert!(array.elements.is_empty());
    }

    #[test]
    fn test_parenthesized_expression() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftParen,
            TokenKind::IntegerLiteral(42),
            TokenKind::RightParen,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ParenthesizedExpr>();
        assert!(result.is_ok());
        
        let paren_expr = result.unwrap();
        match paren_expr.0.kind {
            Expr::Literal(literal) => {
                match literal.value {
                    LiteralValue::Integer(n) => assert_eq!(n, 42),
                    _ => panic!("Expected integer literal"),
                }
            }
            _ => panic!("Expected literal expression"),
        }
    }

    #[test]
    fn test_expression_list_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::IntegerLiteral(1),
            TokenKind::Comma,
            TokenKind::StringLiteral("hello".to_string()),
            TokenKind::Comma,
            TokenKind::True,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ExpressionList>();
        assert!(result.is_ok());
        
        let expr_list = result.unwrap();
        assert_eq!(expr_list.0.len(), 3);
    }
} 