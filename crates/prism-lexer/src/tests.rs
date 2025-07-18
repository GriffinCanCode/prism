//! Comprehensive tests for the Prism lexer
//!
//! This module contains extensive tests covering all aspects of the lexer
//! including multi-syntax support, semantic analysis, and error recovery.

use crate::{
    lexer::{Lexer, LexerConfig, SemanticLexer},
    syntax::{SyntaxDetector, SyntaxStyle},
    token::{Token, TokenKind},
};
use prism_common::{span::Position, symbol::SymbolTable, SourceId};

#[cfg(test)]
mod syntax_detection_tests {
    use super::*;

    #[test]
    fn test_c_like_syntax_detection() {
        let source = r#"
        module Test {
            function foo() {
                if (condition) {
                    return value;
                }
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::CLike);
        assert!(detector.confidence > 0.3);
        
        // Should detect brace patterns
        assert!(detector.evidence.iter().any(|e| e.pattern.contains("brace")));
    }
    
    #[test]
    fn test_python_like_syntax_detection() {
        let source = r#"
        module Test:
            function foo():
                if condition:
                    return value
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::PythonLike);
        assert!(detector.confidence > 0.3);
        
        // Should detect indentation patterns
        assert!(detector.evidence.iter().any(|e| e.pattern.contains("indentation")));
    }
    
    #[test]
    fn test_rust_like_syntax_detection() {
        let source = r#"
        mod test {
            fn foo() -> i32 {
                if condition {
                    return 42;
                }
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::RustLike);
        assert!(detector.confidence > 0.3);
        
        // Should detect Rust keywords
        assert!(detector.evidence.iter().any(|e| e.pattern.contains("Rust")));
    }
    
    #[test]
    fn test_canonical_syntax_detection() {
        let source = r#"
        module UserManagement {
            section types {
                type User = String
            }
            
            function createUser() -> User {
                return "test"
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert_eq!(detector.detected_style, SyntaxStyle::Canonical);
        assert!(detector.confidence > 0.3);
        
        // Should detect canonical keywords
        assert!(detector.evidence.iter().any(|e| e.pattern.contains("canonical")));
    }
    
    #[test]
    fn test_mixed_style_warning() {
        let source = r#"
        module Test {
            fn foo() {  // Rust-like fn
                if condition and other:  // English operators
                    return value
            }
        }
        "#;
        
        let detector = SyntaxDetector::detect_syntax(source);
        assert!(!detector.mixed_style_warnings.is_empty());
        
        let warning = &detector.mixed_style_warnings[0];
        assert!(warning.conflicting_styles.len() > 1);
    }
}

#[cfg(test)]
mod basic_tokenization_tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let source = "module section function type let const if else while for return";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        let expected_tokens = vec![
            TokenKind::Module,
            TokenKind::Section,
            TokenKind::Function,
            TokenKind::Type,
            TokenKind::Let,
            TokenKind::Const,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::While,
            TokenKind::For,
            TokenKind::Return,
        ];
        
        for (i, expected) in expected_tokens.iter().enumerate() {
            assert_eq!(result.tokens[i].kind, *expected);
        }
    }
    
    #[test]
    fn test_identifiers() {
        let source = "userName user_name UserName _private";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.tokens.len(), 4);
        
        for token in &result.tokens {
            assert!(matches!(token.kind, TokenKind::Identifier(_)));
        }
        
        // Check specific identifier names
        if let TokenKind::Identifier(name) = &result.tokens[0].kind {
            assert_eq!(name, "userName");
        }
        if let TokenKind::Identifier(name) = &result.tokens[1].kind {
            assert_eq!(name, "user_name");
        }
    }
    
    #[test]
    fn test_string_literals() {
        let source = r#""hello world" 'single quotes' "escaped \"quote\"" "line\nbreak""#;
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.tokens.len(), 4);
        
        // Check string contents
        if let TokenKind::StringLiteral(value) = &result.tokens[0].kind {
            assert_eq!(value, "hello world");
        }
        if let TokenKind::StringLiteral(value) = &result.tokens[1].kind {
            assert_eq!(value, "single quotes");
        }
        if let TokenKind::StringLiteral(value) = &result.tokens[2].kind {
            assert_eq!(value, "escaped \"quote\"");
        }
        if let TokenKind::StringLiteral(value) = &result.tokens[3].kind {
            assert_eq!(value, "line\nbreak");
        }
    }
    
    #[test]
    fn test_number_literals() {
        let source = "42 3.14 0 123.456";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.tokens.len(), 4);
        
        // Check number values
        assert_eq!(result.tokens[0].kind, TokenKind::IntegerLiteral(42));
        assert_eq!(result.tokens[1].kind, TokenKind::FloatLiteral(3.14));
        assert_eq!(result.tokens[2].kind, TokenKind::IntegerLiteral(0));
        assert_eq!(result.tokens[3].kind, TokenKind::FloatLiteral(123.456));
    }
    
    #[test]
    fn test_operators() {
        let source = "+ - * / % == != < > <= >= = += -= && || !";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        let expected_tokens = vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::Equal,
            TokenKind::NotEqual,
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::LessEqual,
            TokenKind::GreaterEqual,
            TokenKind::Assign,
            TokenKind::PlusAssign,
            TokenKind::MinusAssign,
            TokenKind::AndAnd,
            TokenKind::OrOr,
            TokenKind::Bang,
        ];
        
        for (i, expected) in expected_tokens.iter().enumerate() {
            assert_eq!(result.tokens[i].kind, *expected);
        }
    }
    
    #[test]
    fn test_delimiters() {
        let source = "( ) [ ] { } , ; : :: . -> => ? @";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        let expected_tokens = vec![
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::Comma,
            TokenKind::Semicolon,
            TokenKind::Colon,
            TokenKind::DoubleColon,
            TokenKind::Dot,
            TokenKind::Arrow,
            TokenKind::FatArrow,
            TokenKind::Question,
            TokenKind::At,
        ];
        
        for (i, expected) in expected_tokens.iter().enumerate() {
            assert_eq!(result.tokens[i].kind, *expected);
        }
    }
}

#[cfg(test)]
mod semantic_analysis_tests {
    use super::*;

    #[test]
    fn test_module_semantic_context() {
        let source = "module UserManagement { }";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        // Find the module token
        let module_token = result.tokens.iter()
            .find(|t| matches!(t.kind, TokenKind::Module))
            .unwrap();
            
        assert!(module_token.semantic_context.is_some());
        let context = module_token.semantic_context.as_ref().unwrap();
        
        assert!(context.purpose.is_some());
        assert_eq!(context.domain, Some("Module System".to_string()));
        assert!(context.related_concepts.contains(&"Conceptual Cohesion".to_string()));
        assert!(context.ai_hints.iter().any(|h| h.contains("business capabilities")));
    }
    
    #[test]
    fn test_function_semantic_context() {
        let source = "function getUserById() { }";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        // Find the function token
        let function_token = result.tokens.iter()
            .find(|t| matches!(t.kind, TokenKind::Function))
            .unwrap();
            
        assert!(function_token.semantic_context.is_some());
        let context = function_token.semantic_context.as_ref().unwrap();
        
        assert!(context.purpose.is_some());
        assert_eq!(context.domain, Some("Function Definition".to_string()));
        assert!(context.related_concepts.contains(&"Single Responsibility".to_string()));
        assert!(context.ai_hints.iter().any(|h| h.contains("single responsibility")));
    }
    
    #[test]
    fn test_type_semantic_context() {
        let source = "type User = String";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        // Find the type token
        let type_token = result.tokens.iter()
            .find(|t| matches!(t.kind, TokenKind::Type))
            .unwrap();
            
        assert!(type_token.semantic_context.is_some());
        let context = type_token.semantic_context.as_ref().unwrap();
        
        assert!(context.purpose.is_some());
        assert_eq!(context.domain, Some("Type System".to_string()));
        assert!(context.related_concepts.contains(&"Semantic Types".to_string()));
        assert!(context.ai_hints.iter().any(|h| h.contains("business meaning")));
    }
    
    #[test]
    fn test_capability_semantic_context() {
        let source = "capability FileSystem { }";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        // Find the capability token
        let capability_token = result.tokens.iter()
            .find(|t| matches!(t.kind, TokenKind::Capability))
            .unwrap();
            
        assert!(capability_token.semantic_context.is_some());
        let context = capability_token.semantic_context.as_ref().unwrap();
        
        assert!(context.purpose.is_some());
        assert_eq!(context.domain, Some("Security".to_string()));
        assert!(context.related_concepts.contains(&"Capability-Based Security".to_string()));
        assert!(!context.security_implications.is_empty());
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_unterminated_string() {
        let source = r#""unterminated string"#;
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert!(result.diagnostics.has_errors());
        assert!(result.diagnostics.errors().any(|e| e.message.contains("Unterminated")));
    }
    
    #[test]
    fn test_invalid_number() {
        let source = "123.456.789";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        // Should handle this gracefully, possibly as separate tokens
        assert!(!result.tokens.is_empty());
    }
    
    #[test]
    fn test_invalid_escape_sequence() {
        let source = r#""invalid \x escape""#;
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert!(result.diagnostics.has_errors());
        assert!(result.diagnostics.errors().any(|e| e.message.contains("escape")));
    }
    
    #[test]
    fn test_error_recovery() {
        let source = "module Test { $ invalid # chars }";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig {
            aggressive_recovery: true,
            ..Default::default()
        };
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        // Should have some valid tokens despite errors
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Module)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::LeftBrace)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::RightBrace)));
        
        // Should have error diagnostics
        assert!(result.diagnostics.has_errors());
    }
}

#[cfg(test)]
mod multi_syntax_tests {
    use super::*;

    #[test]
    fn test_c_like_syntax() {
        let source = r#"
        module Test {
            function foo() {
                if (condition) {
                    return value;
                }
            }
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.syntax_style, SyntaxStyle::CLike);
        assert!(!result.tokens.is_empty());
        
        // Should have semicolon
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Semicolon)));
    }
    
    #[test]
    fn test_rust_like_syntax() {
        let source = r#"
        mod test {
            fn foo() -> i32 {
                if condition {
                    return 42;
                }
            }
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.syntax_style, SyntaxStyle::RustLike);
        assert!(!result.tokens.is_empty());
        
        // Should have Rust-like tokens
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Fn)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Arrow)));
    }
    
    #[test]
    fn test_canonical_syntax() {
        let source = r#"
        module UserManagement {
            section types {
                type User = String
            }
            
            function createUser() -> User {
                return "test"
            }
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.syntax_style, SyntaxStyle::Canonical);
        assert!(!result.tokens.is_empty());
        
        // Should have canonical tokens
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Section)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Function)));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_module_parsing() {
        let source = r#"
        @capability "User Management"
        @description "Handles user lifecycle operations"
        
        module UserManagement {
            section types {
                type UserId = PositiveInteger
                type Email = String matching emailRegex
                type User = {
                    id: UserId,
                    email: Email,
                    created: Timestamp
                }
            }
            
            section interface {
                function createUser(email: Email) -> Result<User, UserError>
                    effects [Database.Write, Audit.Log]
                    requires email.isValid()
                    ensures result.isOk() implies User.exists(result.value.id)
                
                function getUser(id: UserId) -> Result<User, UserError>
                    effects [Database.Read]
                    requires id.isValid()
            }
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        // Should successfully tokenize
        assert!(!result.tokens.is_empty());
        assert!(!result.diagnostics.has_errors());
        
        // Should detect canonical syntax
        assert_eq!(result.syntax_style, SyntaxStyle::Canonical);
        
        // Should have semantic summary
        assert!(result.semantic_summary.is_some());
        
        // Should have key tokens
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Module)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Section)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Function)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Type)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Effects)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Requires)));
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::Ensures)));
        
        // Should have semantic context on key tokens
        let module_tokens: Vec<_> = result.tokens.iter()
            .filter(|t| matches!(t.kind, TokenKind::Module))
            .collect();
        assert!(!module_tokens.is_empty());
        assert!(module_tokens[0].semantic_context.is_some());
    }
    
    #[test]
    fn test_performance_with_large_input() {
        // Generate a large input to test performance
        let mut source = String::new();
        source.push_str("module LargeModule {\n");
        
        for i in 0..1000 {
            source.push_str(&format!(
                "    function func{}() {{ return {}; }}\n",
                i, i
            ));
        }
        
        source.push_str("}\n");
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let start = std::time::Instant::now();
        let lexer = Lexer::new(&source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        let duration = start.elapsed();
        
        // Should complete in reasonable time (less than 1 second)
        assert!(duration.as_secs() < 1);
        
        // Should have many tokens
        assert!(result.tokens.len() > 5000);
        
        // Should not have errors
        assert!(!result.diagnostics.has_errors());
    }
}

#[cfg(test)]
mod canonical_conversion_tests {
    use super::*;

    #[test]
    fn test_token_canonical_conversion() {
        let test_cases = vec![
            (TokenKind::Fn, "function"),
            (TokenKind::AndAnd, "and"),
            (TokenKind::OrOr, "or"),
            (TokenKind::Bang, "not"),
            (TokenKind::Pub, "public"),
        ];
        
        for (token_kind, expected_canonical) in test_cases {
            let token = Token::new(
                token_kind,
                Span::new(SourceId::new(1), Position::new(1, 1), Position::new(1, 2)),
                SyntaxStyle::RustLike,
            );
            
            assert_eq!(token.to_canonical(), expected_canonical);
        }
    }
    
    #[test]
    fn test_multi_syntax_equivalence() {
        let rust_source = "fn test() { if condition && other { return; } }";
        let canonical_source = "function test() { if condition and other { return } }";
        
        let source_id = SourceId::new(1);
        let mut symbol_table1 = SymbolTable::new();
        let mut symbol_table2 = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer1 = Lexer::new(rust_source, source_id, &mut symbol_table1, config.clone());
        let lexer2 = Lexer::new(canonical_source, source_id, &mut symbol_table2, config);
        
        let result1 = lexer1.tokenize();
        let result2 = lexer2.tokenize();
        
        // Convert to canonical forms
        let canonical1: Vec<String> = result1.tokens.iter()
            .map(|t| t.to_canonical())
            .collect();
        let canonical2: Vec<String> = result2.tokens.iter()
            .map(|t| t.to_canonical())
            .collect();
        
        // Should be equivalent when converted to canonical form
        assert_eq!(canonical1, canonical2);
    }
} 

#[cfg(test)]
mod semantic_type_constraint_tests {
    use super::*;

    #[test]
    fn test_semantic_type_constraints() {
        let input = r#"
        Money where {
            precision: 2,
            currency: "USD",
            non_negative: true,
            min_value: 0.01,
            max_value: 1000000.00
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let mut lexer = SemanticLexer::new(input, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        assert!(result.errors.is_empty());
        
        // Check for semantic type tokens
        let token_kinds: Vec<_> = result.tokens.iter().map(|t| &t.kind).collect();
        assert!(token_kinds.contains(&&TokenKind::Identifier("Money".to_string())));
        assert!(token_kinds.contains(&&TokenKind::Where));
        assert!(token_kinds.contains(&&TokenKind::LeftBrace));
        assert!(token_kinds.contains(&&TokenKind::Identifier("precision".to_string())));
        assert!(token_kinds.contains(&&TokenKind::Identifier("currency".to_string())));
        assert!(token_kinds.contains(&&TokenKind::Identifier("non_negative".to_string())));
        assert!(token_kinds.contains(&&TokenKind::Identifier("min_value".to_string())));
        assert!(token_kinds.contains(&&TokenKind::Identifier("max_value".to_string())));
        assert!(token_kinds.contains(&&TokenKind::RightBrace));
        
        // Check for semantic patterns
        assert!(result.patterns.iter().any(|p| matches!(p.pattern_type, PatternType::TypeDefinition)));
        assert!(result.patterns.iter().any(|p| p.description.contains("Financial type")));
        assert!(result.patterns.iter().any(|p| p.description.contains("Semantic type constraints")));
        
        // Check for AI hints
        let financial_pattern = result.patterns.iter().find(|p| p.description.contains("Financial type"));
        assert!(financial_pattern.is_some());
        let pattern = financial_pattern.unwrap();
        assert!(pattern.ai_hints.iter().any(|h| h.contains("currency")));
        assert!(pattern.ai_hints.iter().any(|h| h.contains("decimal arithmetic")));
    }

    #[test]
    fn test_constraint_keywords_semantic_analysis() {
        let input = r#"
        EmailAddress where {
            pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
            min_length: 5,
            max_length: 254,
            format: "email",
            validated: true
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let mut lexer = SemanticLexer::new(input, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        assert!(result.errors.is_empty());
        
        // Check for email pattern detection
        assert!(result.patterns.iter().any(|p| p.description.contains("Email type")));
        
        // Check for constraint keywords
        let has_pattern = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "pattern")
        });
        assert!(has_pattern);
        
        let has_min_length = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "min_length")
        });
        assert!(has_min_length);
        
        let has_validated = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "validated")
        });
        assert!(has_validated);
    }

    #[test]
    fn test_business_rule_constraints() {
        let input = r#"
        Age where {
            min_value: 0,
            max_value: 150,
            business_rule: "must_be_realistic_human_age",
            invariant: "age >= 0 && age <= 150"
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let mut lexer = SemanticLexer::new(input, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        assert!(result.errors.is_empty());
        
        // Check for business rule keyword
        let has_business_rule = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "business_rule")
        });
        assert!(has_business_rule);
        
        // Check for invariant keyword
        let has_invariant = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Invariant)
        });
        assert!(has_invariant);
        
        // Check for formal verification pattern
        assert!(result.patterns.iter().any(|p| p.description.contains("Formal verification constraint")));
    }

    #[test]
    fn test_security_classification_constraints() {
        let input = r#"
        SocialSecurityNumber where {
            pattern: "^\\d{3}-\\d{2}-\\d{4}$",
            security_classification: "PII",
            compliance: "GDPR",
            immutable: true,
            ai_context: "Personally identifiable information requiring special handling"
        }
        "#;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let mut lexer = SemanticLexer::new(input, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        assert!(result.errors.is_empty());
        
        // Check for security-related keywords
        let has_security_classification = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "security_classification")
        });
        assert!(has_security_classification);
        
        let has_compliance = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "compliance")
        });
        assert!(has_compliance);
        
        let has_ai_context = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "ai_context")
        });
        assert!(has_ai_context);
        
        // Check for immutable keyword
        let has_immutable = result.tokens.iter().any(|t| {
            matches!(&t.kind, TokenKind::Identifier(name) if name == "immutable")
        });
        assert!(has_immutable);
    }
} 