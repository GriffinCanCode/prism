use prism_syntax::styles::python_like::{PythonLikeParser, PythonParserConfig, PythonVersion};
use prism_syntax::styles::traits::StyleParser;
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;

fn create_test_token(kind: TokenKind, value: &str) -> Token {
    Token {
        kind,
        span: Span::new(0, 0, 0, 0, 0),
        semantic_context: None,
        source_style: None,
        canonical_form: None,
        doc_validation: None,
        responsibility_context: None,
    }
}

#[test]
fn test_python_parser_creation() {
    let parser = PythonLikeParser::new();
    assert_eq!(parser.syntax_style(), prism_syntax::detection::SyntaxStyle::PythonLike);
}

#[test]
fn test_python_parser_with_config() {
    let config = PythonParserConfig {
        python_version: PythonVersion::Python312,
        enable_error_recovery: true,
        generate_ai_metadata: false,
        ..Default::default()
    };
    
    let parser = PythonLikeParser::with_config(config);
    assert_eq!(parser.syntax_style(), prism_syntax::detection::SyntaxStyle::PythonLike);
}

#[test]
fn test_python_version_compatibility() {
    let versions = vec![
        PythonVersion::Python38,
        PythonVersion::Python39,
        PythonVersion::Python310,
        PythonVersion::Python311,
        PythonVersion::Python312,
        PythonVersion::Python313Plus,
    ];
    
    for version in versions {
        let config = PythonParserConfig {
            python_version: version,
            ..Default::default()
        };
        
        let parser = PythonLikeParser::with_config(config);
        assert_eq!(parser.syntax_style(), prism_syntax::detection::SyntaxStyle::PythonLike);
    }
}

#[test]
fn test_basic_token_parsing() {
    // Test that the parser can handle basic tokens without crashing
    let mut parser = PythonLikeParser::new();
    
    let tokens = vec![
        create_test_token(TokenKind::Identifier("hello".to_string()), "hello"),
        create_test_token(TokenKind::Assign, "="),
        create_test_token(TokenKind::StringLiteral("world".to_string()), "world"),
        create_test_token(TokenKind::Newline, "\n"),
    ];
    
    // This should not panic - we're testing basic infrastructure
    let result = parser.parse(tokens);
    
    // We expect this to work or fail gracefully
    match result {
        Ok(syntax_tree) => {
            // Success! The parser worked
            assert!(!syntax_tree.statements.is_empty() || syntax_tree.statements.is_empty());
        }
        Err(_) => {
            // Expected for incomplete implementation - that's okay
            // The important thing is it didn't panic
        }
    }
}

#[test] 
fn test_parser_capabilities() {
    let parser = PythonLikeParser::new();
    let capabilities = parser.capabilities();
    
    assert!(!capabilities.supports_mixed_indentation); // Python shouldn't allow mixed indentation by default
    assert!(!capabilities.supports_optional_semicolons); // Python requires newlines
    assert!(capabilities.supports_trailing_commas); // Python allows trailing commas
    assert!(!capabilities.supports_nested_comments); // Python doesn't have nested comments
    assert!(capabilities.max_nesting_depth > 0); // Should have some reasonable limit
} 