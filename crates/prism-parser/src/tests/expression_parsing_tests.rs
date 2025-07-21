//! Comprehensive tests for expression parsing capabilities
//!
//! This module tests the integration of:
//! - Stream-based parsing with checkpoint recovery
//! - Complex expression parsing with precedence
//! - Function call parsing with type arguments
//! - Binary and unary operator parsing
//! - Array and object literal parsing

use crate::core::{Parse, ParseStream, TokenStreamManager, ParsingCoordinator};
use crate::parsers::{PrattExpression, CallArgument, TypeArgumentList};
use crate::stream_combinators::{comma_separated, optional, alternative};
use crate::ast_parse_impls::{ParenthesizedExpr, ExpressionList};
use prism_ast::{
    AstNode, Expr, LiteralExpr, LiteralValue, VariableExpr, ArrayExpr, ObjectExpr,
    BinaryOperator, UnaryOperator, BinaryExpr, UnaryExpr
};
use prism_lexer::{Token, TokenKind};
use prism_common::{span::{Position, Span}, SourceId, symbol::Symbol};

/// Helper function to create test tokens
fn create_test_tokens(kinds: Vec<TokenKind>) -> Vec<Token> {
    let source_id = SourceId::new(0);
    kinds.into_iter().enumerate().map(|(i, kind)| {
        let start = Position::new(i as u32, i as u32, i as u32);
        let end = Position::new(i as u32, i as u32 + 1, i as u32 + 1);
        Token::new(kind, Span::new(start, end, source_id))
    }).collect()
}

/// Helper function to create a parse stream
fn create_parse_stream(tokens: Vec<Token>) -> (TokenStreamManager, ParsingCoordinator) {
    let token_manager = TokenStreamManager::new(tokens.clone());
    let coordinator = ParsingCoordinator::new(tokens);
    (token_manager, coordinator)
}

#[cfg(test)]
mod literal_parsing_tests {
    use super::*;

    #[test]
    fn test_integer_literal_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::IntegerLiteral(42)]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<LiteralExpr>();
        assert!(result.is_ok());
        
        let literal = result.unwrap();
        match literal.value {
            LiteralValue::Integer(n) => assert_eq!(n, 42),
            _ => panic!("Expected integer literal"),
        }
    }

    #[test]
    fn test_float_literal_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::FloatLiteral(3.14)]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<LiteralExpr>();
        assert!(result.is_ok());
        
        let literal = result.unwrap();
        match literal.value {
            LiteralValue::Float(f) => assert!((f - 3.14).abs() < f64::EPSILON),
            _ => panic!("Expected float literal"),
        }
    }

    #[test]
    fn test_string_literal_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::StringLiteral("hello world".to_string())]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<LiteralExpr>();
        assert!(result.is_ok());
        
        let literal = result.unwrap();
        match literal.value {
            LiteralValue::String(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected string literal"),
        }
    }

    #[test]
    fn test_boolean_literal_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::True]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<LiteralExpr>();
        assert!(result.is_ok());
        
        let literal = result.unwrap();
        match literal.value {
            LiteralValue::Boolean(b) => assert!(b),
            _ => panic!("Expected boolean literal"),
        }
    }

    #[test]
    fn test_null_literal_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::Null]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<LiteralExpr>();
        assert!(result.is_ok());
        
        let literal = result.unwrap();
        match literal.value {
            LiteralValue::Null => (), // Success
            _ => panic!("Expected null literal"),
        }
    }
}

#[cfg(test)]
mod operator_parsing_tests {
    use super::*;

    #[test]
    fn test_binary_operator_parsing() {
        let operators = vec![
            (TokenKind::Plus, BinaryOperator::Add),
            (TokenKind::Minus, BinaryOperator::Subtract),
            (TokenKind::Star, BinaryOperator::Multiply),
            (TokenKind::Slash, BinaryOperator::Divide),
            (TokenKind::Equal, BinaryOperator::Equal),
            (TokenKind::NotEqual, BinaryOperator::NotEqual),
            (TokenKind::Less, BinaryOperator::Less),
            (TokenKind::Greater, BinaryOperator::Greater),
            (TokenKind::AndAnd, BinaryOperator::And),
            (TokenKind::OrOr, BinaryOperator::Or),
        ];

        for (token_kind, expected_op) in operators {
            let tokens = create_test_tokens(vec![token_kind]);
            let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
            let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

            let result = stream.parse::<BinaryOperator>();
            assert!(result.is_ok(), "Failed to parse {:?}", expected_op);
            
            let parsed_op = result.unwrap();
            assert_eq!(
                std::mem::discriminant(&parsed_op),
                std::mem::discriminant(&expected_op),
                "Expected {:?}, got {:?}", expected_op, parsed_op
            );
        }
    }

    #[test]
    fn test_unary_operator_parsing() {
        let operators = vec![
            (TokenKind::Bang, UnaryOperator::Not),
            (TokenKind::Minus, UnaryOperator::Negate),
            (TokenKind::Plus, UnaryOperator::Plus),
        ];

        for (token_kind, expected_op) in operators {
            let tokens = create_test_tokens(vec![token_kind]);
            let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
            let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

            let result = stream.parse::<UnaryOperator>();
            assert!(result.is_ok(), "Failed to parse {:?}", expected_op);
            
            let parsed_op = result.unwrap();
            assert_eq!(
                std::mem::discriminant(&parsed_op),
                std::mem::discriminant(&expected_op),
                "Expected {:?}, got {:?}", expected_op, parsed_op
            );
        }
    }
}

#[cfg(test)]
mod array_parsing_tests {
    use super::*;

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
    fn test_single_element_array_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftBracket,
            TokenKind::IntegerLiteral(42),
            TokenKind::RightBracket,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ArrayExpr>();
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert_eq!(array.elements.len(), 1);
    }

    #[test]
    fn test_multi_element_array_parsing() {
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
    fn test_array_with_trailing_comma() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftBracket,
            TokenKind::IntegerLiteral(1),
            TokenKind::Comma,
            TokenKind::IntegerLiteral(2),
            TokenKind::Comma,
            TokenKind::RightBracket,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ArrayExpr>();
        assert!(result.is_ok());
        
        let array = result.unwrap();
        assert_eq!(array.elements.len(), 2);
    }
}

#[cfg(test)]
mod object_parsing_tests {
    use super::*;

    #[test]
    fn test_empty_object_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<ObjectExpr>();
        assert!(result.is_ok());
        
        let object = result.unwrap();
        assert!(object.fields.is_empty());
    }
}

#[cfg(test)]
mod call_argument_parsing_tests {
    use super::*;

    #[test]
    fn test_positional_argument_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::IntegerLiteral(42)]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<CallArgument>();
        assert!(result.is_ok());
        
        match result.unwrap() {
            CallArgument::Positional(_) => (), // Success
            _ => panic!("Expected positional argument"),
        }
    }

    #[test]
    fn test_named_argument_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::Identifier("param".to_string()),
            TokenKind::Assign,
            TokenKind::IntegerLiteral(42),
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<CallArgument>();
        assert!(result.is_ok());
        
        match result.unwrap() {
            CallArgument::Named { name, value: _ } => {
                assert_eq!(name.as_str(), "param");
            }
            _ => panic!("Expected named argument"),
        }
    }

    #[test]
    fn test_spread_argument_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::DotDotDot,
            TokenKind::Identifier("args".to_string()),
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<CallArgument>();
        assert!(result.is_ok());
        
        match result.unwrap() {
            CallArgument::Spread(_) => (), // Success
            _ => panic!("Expected spread argument"),
        }
    }
}

#[cfg(test)]
mod checkpoint_recovery_tests {
    use super::*;

    #[test]
    fn test_checkpoint_restore_on_parsing_failure() {
        let tokens = create_test_tokens(vec![
            TokenKind::IntegerLiteral(42),
            TokenKind::Plus,
            TokenKind::StringLiteral("hello".to_string()),
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        // Try to parse something that will fail
        let checkpoint = stream.checkpoint();
        let result = stream.try_parse::<ArrayExpr>();
        assert!(result.is_err());
        
        // Position should be restored
        assert_eq!(stream.peek_token().kind, TokenKind::IntegerLiteral(42));
        
        // Should be able to parse a literal now
        let literal_result = stream.parse::<LiteralExpr>();
        assert!(literal_result.is_ok());
    }

    #[test]
    fn test_speculative_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::IntegerLiteral(42),
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        // Speculative parsing should not consume input on failure
        let array_result = stream.speculative_parse::<ArrayExpr>();
        assert!(array_result.is_none());
        
        // Should still be able to parse the literal
        let literal_result = stream.speculative_parse::<LiteralExpr>();
        assert!(literal_result.is_some());
        
        // And position should be restored
        assert_eq!(stream.peek_token().kind, TokenKind::IntegerLiteral(42));
    }
}

#[cfg(test)]
mod combinator_integration_tests {
    use super::*;

    #[test]
    fn test_comma_separated_parsing() {
        let tokens = create_test_tokens(vec![
            TokenKind::IntegerLiteral(1),
            TokenKind::Comma,
            TokenKind::IntegerLiteral(2),
            TokenKind::Comma,
            TokenKind::IntegerLiteral(3),
        ]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = comma_separated::<LiteralExpr>(&mut stream);
        assert!(result.is_ok());
        
        let literals = result.unwrap();
        assert_eq!(literals.len(), 3);
    }

    #[test]
    fn test_optional_parsing_with_content() {
        let tokens = create_test_tokens(vec![TokenKind::IntegerLiteral(42)]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = optional::<LiteralExpr>(&mut stream);
        assert!(result.is_ok());
        
        let optional_literal = result.unwrap();
        assert!(optional_literal.is_some());
    }

    #[test]
    fn test_optional_parsing_without_content() {
        let tokens = create_test_tokens(vec![TokenKind::LeftBrace]); // Not a literal
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = optional::<LiteralExpr>(&mut stream);
        assert!(result.is_ok());
        
        let optional_literal = result.unwrap();
        assert!(optional_literal.is_none());
    }

    #[test]
    fn test_alternative_parsing() {
        let tokens = create_test_tokens(vec![TokenKind::StringLiteral("test".to_string())]);
        let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        // Try alternatives - should pick the string literal
        let alternatives = [
            |input: &mut ParseStream| -> crate::core::ParseResult<LiteralExpr> {
                // Try integer first (should fail)
                let token = input.peek_token();
                match token.kind {
                    TokenKind::IntegerLiteral(n) => {
                        input.advance();
                        Ok(LiteralExpr { value: LiteralValue::Integer(n) })
                    }
                    _ => Err(crate::core::ParseError::expected_literal(input.span())),
                }
            },
            |input: &mut ParseStream| -> crate::core::ParseResult<LiteralExpr> {
                // Try string (should succeed)
                input.parse::<LiteralExpr>()
            },
        ];

        let result = alternative(&mut stream, &alternatives);
        assert!(result.is_ok());
        
        let literal = result.unwrap();
        match literal.value {
            LiteralValue::String(s) => assert_eq!(s, "test"),
            _ => panic!("Expected string literal"),
        }
    }
}

#[cfg(test)]
mod expression_list_tests {
    use super::*;

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

    #[test]
    fn test_parenthesized_expression_parsing() {
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
} 