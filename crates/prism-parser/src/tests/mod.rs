//! Tests for the parsing system

pub mod parse_trait_tests;
pub mod minimal_parse_test;
pub mod expression_parsing_tests;
pub mod error_recovery_tests;
pub mod performance_benchmarks;
pub mod property_based_tests;
pub mod integration_tests;

#[cfg(test)]
mod test_utils {
    use prism_lexer::{Token, TokenKind};
    use prism_common::{span::{Position, Span}, SourceId};
    
    /// Create test tokens for testing
    pub fn create_test_tokens(kinds: Vec<TokenKind>) -> Vec<Token> {
        let source_id = SourceId::new(0);
        kinds.into_iter().enumerate().map(|(i, kind)| {
            let start = Position::new(i as u32, i as u32, i as u32);
            let end = Position::new(i as u32, i as u32 + 1, i as u32 + 1);
            Token::new(kind, Span::new(start, end, source_id))
        }).collect()
    }
    
    /// Create a test span
    pub fn create_test_span() -> Span {
        let source_id = SourceId::new(0);
        Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 10, 9),
            source_id,
        )
    }
} 