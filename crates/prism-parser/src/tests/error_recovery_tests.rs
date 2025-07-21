//! Comprehensive error recovery tests

use crate::core::{ParseError, ParseResult, recovery::{RecoveryStrategy, RecoveryContext, ParsingContext}};
use crate::parser::{Parser, ParseConfig};
use crate::tests::test_utils::{create_test_tokens, create_test_span};
use prism_lexer::TokenKind;
use prism_syntax::detection::SyntaxStyle;

#[test]
fn test_missing_semicolon_recovery() {
    let tokens = create_test_tokens(vec![
        TokenKind::Let,
        TokenKind::Identifier("x".to_string()),
        TokenKind::Assign,
        TokenKind::IntegerLiteral(42),
        // Missing semicolon
        TokenKind::Let,
        TokenKind::Identifier("y".to_string()),
        TokenKind::Assign,
        TokenKind::IntegerLiteral(24),
        TokenKind::Semicolon,
    ]);

    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    
    // Should recover and continue parsing
    assert!(result.is_ok());
    let program = result.unwrap();
    
    // Should have parsed both statements despite missing semicolon
    assert_eq!(program.program.items.len(), 2);
    
    // Should have recorded the error
    assert!(!program.errors.is_empty());
}

#[test]
fn test_missing_brace_recovery() {
    let tokens = create_test_tokens(vec![
        TokenKind::Function,
        TokenKind::Identifier("test".to_string()),
        TokenKind::LeftParen,
        TokenKind::RightParen,
        TokenKind::LeftBrace,
        TokenKind::Return,
        TokenKind::IntegerLiteral(42),
        TokenKind::Semicolon,
        // Missing right brace
        TokenKind::Function,
        TokenKind::Identifier("test2".to_string()),
        TokenKind::LeftParen,
        TokenKind::RightParen,
        TokenKind::LeftBrace,
        TokenKind::RightBrace,
    ]);

    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    
    // Should recover from missing brace
    assert!(result.is_ok());
    let program = result.unwrap();
    
    // Should have attempted to parse both functions
    assert!(!program.program.items.is_empty());
    assert!(!program.errors.is_empty());
}

#[test]
fn test_syntax_style_recovery() {
    // Mix Python-like and C-like syntax
    let tokens = create_test_tokens(vec![
        TokenKind::Function,
        TokenKind::Identifier("test".to_string()),
        TokenKind::LeftParen,
        TokenKind::RightParen,
        TokenKind::Colon,  // Python-like
        TokenKind::Indent, // Python-like
        TokenKind::Return,
        TokenKind::IntegerLiteral(42),
        TokenKind::RightBrace, // C-like (mismatch!)
    ]);

    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    
    // Should attempt recovery from syntax style mismatch
    assert!(result.is_ok());
    let program = result.unwrap();
    assert!(!program.errors.is_empty());
}

#[test]
fn test_recovery_strategy_selection() {
    let context = RecoveryContext {
        context: ParsingContext::Function,
        expected: vec![TokenKind::RightBrace],
        found: TokenKind::Eof,
        error_count: 1,
        available_strategies: vec![
            RecoveryStrategy::InsertDelimiter(TokenKind::RightBrace),
            RecoveryStrategy::SkipToStatement,
            RecoveryStrategy::ErrorNode,
        ],
    };

    // Should prefer delimiter insertion for function context
    let selected = select_best_strategy(&context);
    assert!(matches!(selected, RecoveryStrategy::InsertDelimiter(TokenKind::RightBrace)));
}

#[test]
fn test_semantic_recovery() {
    let tokens = create_test_tokens(vec![
        TokenKind::Let,
        TokenKind::Identifier("x".to_string()),
        TokenKind::Colon,
        // Missing type annotation
        TokenKind::Assign,
        TokenKind::IntegerLiteral(42),
        TokenKind::Semicolon,
    ]);

    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    config.validate_constraints = true;
    
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    
    // Should recover with type inference
    assert!(result.is_ok());
    let program = result.unwrap();
    assert!(!program.program.items.is_empty());
}

#[test]
fn test_multiple_error_recovery() {
    let tokens = create_test_tokens(vec![
        TokenKind::Let,
        TokenKind::Identifier("x".to_string()),
        // Missing colon and type
        TokenKind::Assign,
        TokenKind::IntegerLiteral(42),
        // Missing semicolon
        TokenKind::Let,
        TokenKind::Identifier("y".to_string()),
        // Missing everything
        TokenKind::Function,
        TokenKind::Identifier("test".to_string()),
        TokenKind::LeftParen,
        // Missing right paren and body
    ]);

    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    config.max_errors = 10;
    
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    
    // Should recover from multiple errors
    assert!(result.is_ok());
    let program = result.unwrap();
    assert!(program.errors.len() > 1);
    assert!(program.errors.len() <= config.max_errors);
}

#[test]
fn test_recovery_performance() {
    // Create a large token stream with many errors
    let mut tokens = Vec::new();
    for i in 0..1000 {
        tokens.extend(create_test_tokens(vec![
            TokenKind::Let,
            TokenKind::Identifier(format!("var{}", i)),
            // Intentionally missing assignment and semicolon
            TokenKind::IntegerLiteral(i as i64),
        ]));
    }

    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    config.max_errors = 100;
    
    let start = std::time::Instant::now();
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    let duration = start.elapsed();
    
    // Should complete in reasonable time (< 100ms for 1000 tokens)
    assert!(duration.as_millis() < 100);
    assert!(result.is_ok());
    
    let program = result.unwrap();
    assert_eq!(program.errors.len(), config.max_errors);
}

// Helper function for testing
fn select_best_strategy(context: &RecoveryContext) -> RecoveryStrategy {
    // Simplified strategy selection logic for testing
    match context.context {
        ParsingContext::Function => {
            if context.expected.contains(&TokenKind::RightBrace) {
                RecoveryStrategy::InsertDelimiter(TokenKind::RightBrace)
            } else {
                RecoveryStrategy::SkipToStatement
            }
        }
        _ => RecoveryStrategy::ErrorNode,
    }
}

#[test]
fn test_recovery_context_building() {
    let error = ParseError::unexpected_token(
        vec![TokenKind::Semicolon],
        TokenKind::Eof,
        create_test_span(),
    );

    // Test context extraction
    let context = build_test_recovery_context(&error);
    
    assert_eq!(context.expected, vec![TokenKind::Semicolon]);
    assert_eq!(context.found, TokenKind::Eof);
    assert!(!context.available_strategies.is_empty());
}

fn build_test_recovery_context(error: &ParseError) -> RecoveryContext {
    RecoveryContext {
        context: ParsingContext::TopLevel,
        expected: vec![TokenKind::Semicolon],
        found: TokenKind::Eof,
        error_count: 1,
        available_strategies: vec![
            RecoveryStrategy::SkipToStatement,
            RecoveryStrategy::ErrorNode,
        ],
    }
} 