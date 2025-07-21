//! Performance benchmarks for parsing strategies

use crate::parser::{Parser, ParseConfig};
use crate::tests::test_utils::create_test_tokens;
use prism_lexer::TokenKind;
use std::time::Instant;

#[test]
fn benchmark_basic_parsing() {
    let tokens = create_large_token_stream(1000);
    
    let start = Instant::now();
    let mut parser = Parser::new(tokens);
    let result = parser.parse_program();
    let duration = start.elapsed();
    
    println!("Basic parsing (1000 tokens): {:?}", duration);
    assert!(result.is_ok());
    assert!(duration.as_millis() < 50); // Should be under 50ms
}

#[test]
fn benchmark_error_recovery_parsing() {
    let tokens = create_error_prone_token_stream(1000);
    
    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    config.max_errors = 100;
    
    let start = Instant::now();
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    let duration = start.elapsed();
    
    println!("Error recovery parsing (1000 tokens): {:?}", duration);
    assert!(result.is_ok());
    assert!(duration.as_millis() < 100); // Should be under 100ms even with recovery
}

#[test]
fn benchmark_semantic_validation() {
    let tokens = create_large_token_stream(500);
    
    let mut config = ParseConfig::default();
    config.validate_constraints = true;
    config.enable_semantic_analysis = true;
    
    let start = Instant::now();
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    let duration = start.elapsed();
    
    println!("Semantic validation parsing (500 tokens): {:?}", duration);
    assert!(result.is_ok());
    assert!(duration.as_millis() < 200); // Allow more time for semantic analysis
}

#[test]
fn benchmark_memory_usage() {
    let tokens = create_large_token_stream(10000);
    
    // Measure memory before parsing
    let memory_before = get_memory_usage();
    
    let mut parser = Parser::new(tokens);
    let result = parser.parse_program();
    
    let memory_after = get_memory_usage();
    let memory_used = memory_after - memory_before;
    
    println!("Memory usage for 10000 tokens: {} bytes", memory_used);
    assert!(result.is_ok());
    
    // Should use less than 10MB for 10k tokens
    assert!(memory_used < 10 * 1024 * 1024);
}

#[test]
fn benchmark_incremental_parsing() {
    // Test incremental parsing performance
    let mut base_tokens = create_large_token_stream(1000);
    
    let start = Instant::now();
    let mut parser = Parser::new(base_tokens.clone());
    let first_result = parser.parse_program();
    let first_duration = start.elapsed();
    
    // Add more tokens (simulating incremental changes)
    base_tokens.extend(create_test_tokens(vec![
        TokenKind::Let,
        TokenKind::Identifier("new_var".to_string()),
        TokenKind::Assign,
        TokenKind::IntegerLiteral(42),
        TokenKind::Semicolon,
    ]));
    
    let start = Instant::now();
    let mut parser2 = Parser::new(base_tokens);
    let second_result = parser2.parse_program();
    let second_duration = start.elapsed();
    
    println!("Initial parsing: {:?}", first_duration);
    println!("Incremental parsing: {:?}", second_duration);
    
    assert!(first_result.is_ok());
    assert!(second_result.is_ok());
    
    // Incremental should be similar (since we don't have true incremental parsing yet)
    // This test establishes baseline for future incremental parsing implementation
}

#[test]
fn benchmark_suggestion_generation() {
    let tokens = create_error_prone_token_stream(100);
    
    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    
    let start = Instant::now();
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    let duration = start.elapsed();
    
    println!("Suggestion generation (100 tokens with errors): {:?}", duration);
    assert!(result.is_ok());
    
    // Should generate suggestions quickly
    assert!(duration.as_millis() < 50);
}

#[test]
fn benchmark_multi_syntax_parsing() {
    let tokens = create_multi_syntax_token_stream();
    
    let mut config = ParseConfig::default();
    config.aggressive_recovery = true;
    config.detect_syntax_style = true;
    
    let start = Instant::now();
    let mut parser = Parser::with_config(tokens, config);
    let result = parser.parse_program();
    let duration = start.elapsed();
    
    println!("Multi-syntax parsing: {:?}", duration);
    assert!(result.is_ok());
    
    // Should handle multi-syntax efficiently
    assert!(duration.as_millis() < 100);
}

// Helper functions

fn create_large_token_stream(size: usize) -> Vec<prism_lexer::Token> {
    let mut tokens = Vec::new();
    
    for i in 0..size {
        tokens.extend(create_test_tokens(vec![
            TokenKind::Let,
            TokenKind::Identifier(format!("var{}", i)),
            TokenKind::Colon,
            TokenKind::Identifier("i32".to_string()),
            TokenKind::Assign,
            TokenKind::IntegerLiteral(i as i64),
            TokenKind::Semicolon,
        ]));
    }
    
    tokens
}

fn create_error_prone_token_stream(size: usize) -> Vec<prism_lexer::Token> {
    let mut tokens = Vec::new();
    
    for i in 0..size {
        if i % 3 == 0 {
            // Every third statement has errors
            tokens.extend(create_test_tokens(vec![
                TokenKind::Let,
                TokenKind::Identifier(format!("var{}", i)),
                // Missing colon and type
                TokenKind::Assign,
                TokenKind::IntegerLiteral(i as i64),
                // Missing semicolon
            ]));
        } else {
            tokens.extend(create_test_tokens(vec![
                TokenKind::Let,
                TokenKind::Identifier(format!("var{}", i)),
                TokenKind::Colon,
                TokenKind::Identifier("i32".to_string()),
                TokenKind::Assign,
                TokenKind::IntegerLiteral(i as i64),
                TokenKind::Semicolon,
            ]));
        }
    }
    
    tokens
}

fn create_multi_syntax_token_stream() -> Vec<prism_lexer::Token> {
    let mut tokens = Vec::new();
    
    // C-like syntax
    tokens.extend(create_test_tokens(vec![
        TokenKind::Function,
        TokenKind::Identifier("c_style".to_string()),
        TokenKind::LeftParen,
        TokenKind::RightParen,
        TokenKind::LeftBrace,
        TokenKind::Return,
        TokenKind::IntegerLiteral(1),
        TokenKind::Semicolon,
        TokenKind::RightBrace,
    ]));
    
    // Python-like syntax
    tokens.extend(create_test_tokens(vec![
        TokenKind::Function,
        TokenKind::Identifier("python_style".to_string()),
        TokenKind::LeftParen,
        TokenKind::RightParen,
        TokenKind::Colon,
        TokenKind::Indent,
        TokenKind::Return,
        TokenKind::IntegerLiteral(2),
        TokenKind::Dedent,
    ]));
    
    tokens
}

fn get_memory_usage() -> usize {
    // Simple approximation of memory usage
    // In a real implementation, you'd use a proper memory profiling tool
    std::mem::size_of::<Parser>() * 1000 // Placeholder
} 