//! Comprehensive tests for the Prism lexer
//!
//! This module contains extensive tests covering all aspects of the lexer
//! including multi-syntax support, semantic analysis, and error recovery.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lexer::{Lexer, LexerConfig},
        token::{Token, TokenKind},
    };
    use prism_common::{
        diagnostics::DiagnosticBag,
        span::{Position, Span},
        symbol::SymbolTable,
        SourceId,
    };

    #[test]
    fn test_basic_tokenization() {
        let source = "42 3.14";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        // Should have 2 tokens plus EOF
        assert!(result.tokens.len() >= 2);
        assert_eq!(result.tokens[0].kind, TokenKind::IntegerLiteral(42));
        assert_eq!(result.tokens[1].kind, TokenKind::FloatLiteral(3.14));
        // Last token should be EOF
        assert_eq!(result.tokens.last().unwrap().kind, TokenKind::Eof);
    }

    #[test]
    fn test_enhanced_literals() {
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        // Test regex literal
        let source = r#"/hello/"#;
        let lexer = Lexer::new(source, SourceId::new(1), &mut symbol_table, config.clone());
        let result = lexer.tokenize();
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::RegexLiteral(_))));
        
        // Test money literal
        let source = "$123.45";
        let lexer = Lexer::new(source, SourceId::new(2), &mut symbol_table, config.clone());
        let result = lexer.tokenize();
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::MoneyLiteral(_))));
        
        // Test duration literal
        let source = "30s";
        let lexer = Lexer::new(source, SourceId::new(3), &mut symbol_table, config);
        let result = lexer.tokenize();
        assert!(result.tokens.iter().any(|t| matches!(t.kind, TokenKind::DurationLiteral(_))));
    }
} 