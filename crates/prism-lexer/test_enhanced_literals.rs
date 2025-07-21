use prism_lexer::{Lexer, LexerConfig};
use prism_lexer::token::TokenKind;
use prism_common::{SourceId, symbol::SymbolTable};

fn main() {
    let mut symbol_table = SymbolTable::new();
    let config = LexerConfig::default();
    
    // Test regex literal
    println!("Testing regex literal...");
    let source = r#"/hello world/"#;
    let lexer = Lexer::new(source, SourceId::new(1), &mut symbol_table, config.clone());
    let result = lexer.tokenize();
    println!("Tokens: {:?}", result.tokens.iter().map(|t| &t.kind).collect::<Vec<_>>());
    
    // Test money literal
    println!("\nTesting money literal...");
    let source = "$123.45";
    let lexer = Lexer::new(source, SourceId::new(2), &mut symbol_table, config.clone());
    let result = lexer.tokenize();
    println!("Tokens: {:?}", result.tokens.iter().map(|t| &t.kind).collect::<Vec<_>>());
    
    // Test duration literal
    println!("\nTesting duration literal...");
    let source = "30s";
    let lexer = Lexer::new(source, SourceId::new(3), &mut symbol_table, config.clone());
    let result = lexer.tokenize();
    println!("Tokens: {:?}", result.tokens.iter().map(|t| &t.kind).collect::<Vec<_>>());
    
    println!("\nAll tests completed!");
}
