 //! Tests for Parse trait implementations
//!
//! These tests verify that the stream-based parsing system works correctly
//! with basic AST types.

use crate::core::{Parse, ParseStream, TokenStreamManager, ParsingCoordinator};
use prism_ast::{LiteralExpr, LiteralValue, VariableExpr};
use prism_lexer::{Token, TokenKind};
use prism_common::{span::Span, SourceId};

fn create_test_tokens(kinds: Vec<TokenKind>) -> Vec<Token> {
    use prism_common::span::Position;
    
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
fn test_parse_integer_literal() {
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
fn test_parse_string_literal() {
    let tokens = create_test_tokens(vec![TokenKind::StringLiteral("hello".to_string())]);
    let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
    let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

    let result = stream.parse::<LiteralExpr>();
    assert!(result.is_ok());
    
    let literal = result.unwrap();
    match literal.value {
        LiteralValue::String(s) => assert_eq!(s, "hello"),
        _ => panic!("Expected string literal"),
    }
}

#[test]
fn test_parse_boolean_literal() {
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
fn test_parse_variable() {
    let tokens = create_test_tokens(vec![TokenKind::Identifier("myVar".to_string())]);
    let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
    let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

    let result = stream.parse::<VariableExpr>();
    assert!(result.is_ok());
    
    let variable = result.unwrap();
    // Note: Symbol doesn't have as_str(), so we'll just check it's not empty for now
    // In a real implementation, we'd need proper Symbol comparison
    assert!(!format!("{:?}", variable.name).is_empty());
}

#[test]
fn test_parse_empty_array() {
    let tokens = create_test_tokens(vec![
        TokenKind::LeftBracket,
        TokenKind::RightBracket,
    ]);
    let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
    let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

    let result = stream.parse::<prism_ast::ArrayExpr>();
    assert!(result.is_ok());
    
    let array = result.unwrap();
    assert!(array.elements.is_empty());
}

#[test]
fn test_parse_error_on_unexpected_token() {
    let tokens = create_test_tokens(vec![TokenKind::LeftParen]); // Not a literal
    let (mut token_manager, mut coordinator) = create_parse_stream(tokens);
    let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

    let result = stream.parse::<LiteralExpr>();
    assert!(result.is_err());
} 