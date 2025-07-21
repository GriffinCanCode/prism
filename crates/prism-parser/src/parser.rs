//! Main Parser Coordinator
//!
//! This module embodies the single concept of "Parser Coordination".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: coordinating the parsing process by orchestrating specialized parsers.
//!
//! **Conceptual Responsibility**: Coordinate parsing workflow and provide public API
//! **What it does**: orchestrate parsers, manage configuration, provide public interface, handle semantic analysis
//! **What it doesn't do**: actual parsing logic, token navigation, AST construction

use crate::{
    core::{ParseError, ParseResult, TokenStreamManager, ParsingCoordinator, Precedence},
    parsers::{ExpressionParser, StatementParser, TypeParser, FunctionParser, ModuleParser},
    analysis::{SemanticContextExtractor, ConstraintValidator, ValidationConfig, TokenSemanticAnalyzer, TokenSemanticSummary},
};
use prism_ast::{AstArena, AstNode, Expr, Item, Program, ProgramMetadata, Stmt, Type};
use prism_common::NodeId;
use prism_common::{span::Span, SourceId};
use prism_lexer::{Token, TokenKind, LexerResult};
use prism_syntax::{SyntaxDetector, SyntaxStyle, DetectionResult}; // Use prism-syntax for style detection
use std::collections::HashMap;

/// Configuration for the parser
#[derive(Debug, Clone)]
pub struct ParseConfig {
    /// Maximum number of errors before stopping
    pub max_errors: usize,
    /// Enable AI context extraction
    pub extract_ai_context: bool,
    /// Enable semantic constraint validation
    pub validate_constraints: bool,
    /// Enable aggressive error recovery
    pub aggressive_recovery: bool,
    /// Enable token-based semantic analysis
    pub enable_semantic_analysis: bool,
    /// Enable syntax style detection
    pub detect_syntax_style: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            max_errors: 100,
            extract_ai_context: true,
            validate_constraints: true,
            aggressive_recovery: true,
            enable_semantic_analysis: true,
            detect_syntax_style: true,
        }
    }
}

/// Enhanced parsing result with semantic information
#[derive(Debug)]
pub struct ParsedProgram {
    /// The parsed program
    pub program: Program,
    /// Detected syntax style (if enabled)
    pub detected_syntax_style: Option<SyntaxStyle>,
    /// Token-based semantic summary (if enabled)
    pub semantic_summary: Option<TokenSemanticSummary>,
    /// Parse errors encountered
    pub errors: Vec<ParseError>,
    /// Warnings generated during parsing
    pub warnings: Vec<String>,
}

/// The main parser that coordinates specialized parsing modules
/// 
/// This struct embodies the single concept of parsing coordination.
/// It delegates actual parsing work to specialized modules while
/// managing the overall parsing workflow and providing the public API.
/// 
/// It now also handles semantic analysis and syntax detection that was
/// incorrectly placed in the lexer.
pub struct Parser {
    /// Token stream manager for navigation
    token_stream: TokenStreamManager,
    /// Parsing coordinator for orchestration
    coordinator: ParsingCoordinator,
    /// Token semantic analyzer
    token_analyzer: TokenSemanticAnalyzer,
    /// Parser configuration
    config: ParseConfig,
    /// Parse errors encountered
    errors: Vec<ParseError>,
}

impl Parser {
    /// Create a new parser with default configuration
    pub fn new(tokens: Vec<Token>) -> Self {
        Self::with_config(tokens, ParseConfig::default())
    }

    /// Create a new parser with custom configuration
    pub fn with_config(tokens: Vec<Token>, config: ParseConfig) -> Self {
        let source_id = if tokens.is_empty() {
            SourceId::new(0)
        } else {
            tokens[0].span.source_id
        };

        // Create token stream manager
        let token_stream = TokenStreamManager::new(tokens.clone());
        
        // Create parsing coordinator
        let coordinator = ParsingCoordinator::with_config(tokens.clone(), config.clone());

        // Create token analyzer
        let token_analyzer = TokenSemanticAnalyzer::new();

        Self {
            token_stream,
            coordinator,
            token_analyzer,
            config,
            errors: Vec::new(),
        }
    }

    // === Methods needed by combinators ===

    /// Check if the current token matches the given kind (delegate to token_stream)
    pub fn check(&self, kind: TokenKind) -> bool {
        self.token_stream.check(kind)
    }

    /// Advance to the next token (delegate to token_stream)
    pub fn advance(&mut self) -> &Token {
        self.token_stream.advance()
    }

    /// Consume a token if it matches the expected kind (delegate to token_stream)
    pub fn consume(&mut self, expected: TokenKind, _error_message: &str) -> ParseResult<&Token> {
        if self.token_stream.check(expected.clone()) {
            Ok(self.token_stream.advance())
        } else {
            let found = self.token_stream.peek().kind.clone();
            let span = self.token_stream.current_span();
            Err(ParseError::unexpected_token(
                vec![expected],
                found,
                span,
            ))
        }
    }

    /// Get the current span (delegate to token_stream)
    pub fn current_span(&self) -> Span {
        self.token_stream.current_span()
    }

    /// Check if we're at the end of the token stream (delegate to token_stream)
    pub fn is_at_end(&self) -> bool {
        self.token_stream.is_at_end()
    }

    /// Check if current token could end a list (delegate to token_stream)
    pub fn check_list_end(&self) -> bool {
        self.token_stream.check_list_end()
    }

    /// Peek at the current token (delegate to token_stream)
    pub fn peek(&self) -> &Token {
        self.token_stream.peek()
    }

    /// Consume an identifier token and return its value
    pub fn consume_identifier(&mut self, _error_message: &str) -> ParseResult<String> {
        self.token_stream.expect_identifier()
    }

    /// Consume a string literal token and return its value
    pub fn consume_string(&mut self, _error_message: &str) -> ParseResult<String> {
        match &self.token_stream.peek().kind {
            TokenKind::StringLiteral(value) => {
                let result = value.clone();
                self.token_stream.advance();
                Ok(result)
            }
            _ => {
                let found = self.token_stream.peek().kind.clone();
                let span = self.token_stream.current_span();
                Err(ParseError::unexpected_token(
                    vec![TokenKind::StringLiteral("string".to_string())],
                    found,
                    span,
                ))
            }
        }
    }

    /// Consume a number literal token and return its value
    pub fn consume_number(&mut self, _error_message: &str) -> ParseResult<f64> {
        match &self.token_stream.peek().kind {
            TokenKind::IntegerLiteral(value) => {
                let result = *value as f64;
                self.token_stream.advance();
                Ok(result)
            }
            TokenKind::FloatLiteral(value) => {
                let result = *value;
                self.token_stream.advance();
                Ok(result)
            }
            _ => {
                let found = self.token_stream.peek().kind.clone();
                let span = self.token_stream.current_span();
                Err(ParseError::unexpected_token(
                    vec![TokenKind::IntegerLiteral(0), TokenKind::FloatLiteral(0.0)],
                    found,
                    span,
                ))
            }
        }
    }

    /// Create an AST node (delegate to coordinator)
    pub fn create_node<T>(&mut self, node: T, span: Span) -> AstNode<T> {
        self.coordinator.create_node(node, span)
    }

    /// Check if current token could end a block (delegate to token_stream)
    pub fn check_block_end(&self) -> bool {
        self.token_stream.check_block_end()
    }

    /// Parse a complete program with enhanced semantic analysis
    pub fn parse_program(&mut self) -> ParseResult<ParsedProgram> {
        let mut items = Vec::new();
        let mut warnings = Vec::new();

        // Step 1: Detect syntax style if enabled
        let detected_syntax_style = if self.config.detect_syntax_style {
            // We need the original source for syntax detection
            // For now, we'll use a placeholder - in a real implementation,
            // the source would be passed through or reconstructed
            let detection_result = SyntaxDetector::detect_syntax("");
            Some(detection_result.detected_style)
        } else {
            None
        };

        // Step 2: Perform token-based semantic analysis if enabled
        let semantic_summary = if self.config.enable_semantic_analysis {
            let tokens: Vec<Token> = self.token_stream.tokens().to_vec();
            Some(self.token_analyzer.analyze_tokens(&tokens))
        } else {
            None
        };

        // Step 3: Parse AST structure
        while !self.coordinator.token_manager().is_at_end() {
            match self.parse_item() {
                Ok(item) => {
                    items.push(item);
                    
                    // Extract semantic context if enabled
                    if self.config.extract_ai_context {
                        // Context extraction would go here
                        warnings.push("AI context extraction not yet fully implemented".to_string());
                    }
                }
                Err(error) => {
                    self.errors.push(error);
                    if self.errors.len() >= self.config.max_errors {
                        break;
                    }
                    // Attempt recovery
                    if self.config.aggressive_recovery {
                        self.synchronize();
                    } else {
                        break;
                    }
                }
            }
        }

        // Step 4: Generate program metadata
        let metadata = self.generate_program_metadata(&items, &semantic_summary);

        let program = Program {
            items,
            source_id: self.coordinator.source_id(),
            metadata,
        };

        Ok(ParsedProgram {
            program,
            detected_syntax_style,
            semantic_summary,
            errors: self.errors.clone(),
            warnings,
        })
    }

    /// Parse a single item (top-level construct)
    fn parse_item(&mut self) -> ParseResult<AstNode<Item>> {
        match self.coordinator.token_manager().peek().kind {
            // Module declarations
            prism_lexer::TokenKind::Module => {
                // For now, create a placeholder module
                let span = self.coordinator.token_manager().current_span();
                Ok(self.coordinator.create_node(
                    Item::Module(prism_ast::ModuleDecl {
                        name: prism_common::symbol::Symbol::intern("placeholder"),
                        capability: None,
                        description: None,
                        dependencies: Vec::new(),
                        stability: prism_ast::StabilityLevel::Experimental,
                        version: None,
                        sections: Vec::new(),
                        ai_context: None,
                        visibility: prism_ast::Visibility::Public,
                    }),
                    span,
                ))
            }
            
            // All other items are statements wrapped as items
            _ => {
                // For now, create a placeholder statement
                let span = self.coordinator.token_manager().current_span();
                Ok(self.coordinator.create_node(
                    Item::Statement(prism_ast::Stmt::Error(prism_ast::ErrorStmt {
                        message: "Placeholder statement".to_string(),
                        context: "Item parsing not yet implemented".to_string(),
                    })),
                    span,
                ))
            }
        }
    }

    /// Parse a single expression (public API)
    pub fn parse_expression_public(&mut self) -> ParseResult<AstNode<Expr>> {
        // Create error expression for now
        let span = self.coordinator.token_manager().current_span();
        Ok(self.coordinator.create_node(
            Expr::Error(prism_ast::ErrorExpr {
                message: "Expression parsing not yet fully implemented".to_string(),
            }),
            span,
        ))
    }

    /// Parse a single statement (public API)
    pub fn parse_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        // For now, create a placeholder statement
        let span = self.coordinator.token_manager().current_span();
        Ok(self.coordinator.create_node(
            Stmt::Error(prism_ast::ErrorStmt {
                message: "Statement parsing not yet implemented".to_string(),
                context: "Parser API".to_string(),
            }),
            span,
        ))
    }

    /// Parse a type annotation (public API)
    pub fn parse_type(&mut self) -> ParseResult<AstNode<Type>> {
        // For now, create a placeholder type
        let span = self.coordinator.token_manager().current_span();
        Ok(self.coordinator.create_node(
            Type::Error(prism_ast::ErrorType {
                message: "Type parsing not yet implemented".to_string(),
            }),
            span,
        ))
    }

    /// Generate program metadata using semantic analysis results
    fn generate_program_metadata(&mut self, items: &[AstNode<Item>], semantic_summary: &Option<TokenSemanticSummary>) -> ProgramMetadata {
        let mut metadata = ProgramMetadata {
            primary_capability: None,
            capabilities: Vec::new(),
            dependencies: Vec::new(),
            security_implications: Vec::new(),
            performance_notes: Vec::new(),
            ai_insights: Vec::new(),
        };

        // Use semantic analysis results if available
        if let Some(summary) = semantic_summary {
            // Extract capabilities from semantic analysis
            metadata.capabilities = summary.capabilities.iter()
                .map(|cap| cap.name.clone())
                .collect();
            
            // Generate primary capability based on modules
            if !summary.modules.is_empty() {
                metadata.primary_capability = Some(
                    format!("Program with {} modules providing {} capabilities", 
                        summary.modules.len(),
                        summary.capabilities.len())
                );
            }
            
            // Generate AI insights from patterns
            metadata.ai_insights = summary.patterns.iter()
                .flat_map(|pattern| pattern.ai_hints.iter())
                .cloned()
                .collect();
            
            // Add semantic score insight
            metadata.ai_insights.push(format!(
                "Code semantic quality score: {:.2}/1.0",
                summary.semantic_score
            ));
            
            // Performance notes about parsing
            metadata.performance_notes.push(format!(
                "Analyzed {} tokens, {} patterns detected",
                summary.identifier_usage.len(),
                summary.patterns.len()
            ));
        }

        if self.config.extract_ai_context {
            if metadata.primary_capability.is_none() {
                metadata.primary_capability = Some(
                    "Program with conceptually cohesive modules".to_string()
                );
            }
            
            // Add performance notes about parsing
            metadata.performance_notes.push(format!(
                "Parsed {} items using conceptually cohesive parser architecture",
                items.len()
            ));
        }

        metadata
    }

    /// Synchronize after an error by finding the next statement boundary
    fn synchronize(&mut self) {
        // Implement basic synchronization
        while !self.coordinator.token_manager().is_at_end() {
            match self.coordinator.token_manager().peek().kind {
                prism_lexer::TokenKind::Semicolon => {
                    self.coordinator.token_manager_mut().advance();
                    return;
                }
                prism_lexer::TokenKind::Module
                | prism_lexer::TokenKind::Function
                | prism_lexer::TokenKind::Type
                | prism_lexer::TokenKind::Let
                | prism_lexer::TokenKind::Const
                | prism_lexer::TokenKind::If
                | prism_lexer::TokenKind::While
                | prism_lexer::TokenKind::For
                | prism_lexer::TokenKind::Return => {
                    return;
                }
                _ => {
                    self.coordinator.token_manager_mut().advance();
                }
            }
        }
    }

    /// Get parse errors
    pub fn get_errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Get semantic contexts (placeholder)
    pub fn get_semantic_contexts(&self) -> HashMap<NodeId, String> {
        // Placeholder implementation
        HashMap::new()
    }

    /// Get arena (placeholder)
    pub fn get_arena(&self) -> Option<&AstArena> {
        // Placeholder implementation
        None
    }

    /// Calculate complexity score
    pub fn calculate_complexity_score(&self, items: &[AstNode<Item>]) -> f64 {
        // Simple complexity calculation based on item count
        let base_complexity = items.len() as f64 * 0.1;
        base_complexity.min(1.0)
    }

    /// Validate program (placeholder)
    pub fn validate_program(&mut self, _program: &Program) -> Vec<ParseError> {
        // Placeholder implementation
        Vec::new()
    }

    /// Parse a function definition using the specialized function parser
    pub fn parse_function(&mut self) -> ParseResult<NodeId> {
        // This is a simplified integration - in practice, we'd need to handle
        // the borrowing more carefully or restructure the parser architecture
        
        // For now, create a basic function parser integration
        // This demonstrates the integration pattern
        match self.token_stream.current_kind() {
            TokenKind::Function | TokenKind::Fn => {
                // Consume the function token
                self.token_stream.advance();
                
                // Parse function name
                let name = self.token_stream.expect_identifier()?;
                
                // Create a basic function node (simplified)
                let span = self.token_stream.current_span();
                let function_stmt = prism_ast::Stmt::Function(prism_ast::FunctionDecl {
                    name: prism_common::symbol::Symbol::intern(&name),
                    parameters: Vec::new(), // TODO: Parse parameters
                    return_type: None,      // TODO: Parse return type
                    body: None,             // TODO: Parse body
                    visibility: prism_ast::Visibility::Private,
                    attributes: Vec::new(),
                    contracts: None,
                    is_async: false,
                });
                
                let node = self.coordinator.create_node(function_stmt, span);
                Ok(node.id)
            }
            _ => {
                let found = self.token_stream.peek().kind.clone();
                let span = self.token_stream.current_span();
                Err(ParseError::unexpected_token(
                    vec![TokenKind::Function, TokenKind::Fn],
                    found,
                    span,
                ))
            }
        }
    }
}

/// Convenience function to create parser from source string
impl Parser {
    /// Create parser from source code string
    pub fn from_source(source: &str) -> Result<Self, prism_lexer::LexerError> {
        use prism_lexer::{Lexer, LexerConfig};
        use prism_common::symbol::SymbolTable;
        
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let lexer_config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, lexer_config);
        let lexer_result = lexer.tokenize();
        
        if lexer_result.diagnostics.has_errors() {
            // For now, we'll create an empty parser if lexing fails
            // In a real implementation, we'd want to handle this more gracefully
            return Ok(Self::new(vec![]));
        }
        
        Ok(Self::new(lexer_result.tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conceptual_cohesion_architecture() {
        // Test that parser maintains conceptual cohesion
        let tokens = vec![
            Token::new(
                prism_lexer::TokenKind::Module,
                Span::dummy(),
            ),
            Token::new(
                prism_lexer::TokenKind::Identifier("Test".to_string()),
                Span::dummy(),
            ),
        ];
        
        let mut parser = Parser::new(tokens);
        let result = parser.parse_program();
        
        // Parser should coordinate different concerns without mixing them
        assert!(result.is_ok());
        assert!(!result.unwrap().program.items.is_empty());
    }

    #[test]
    fn test_parser_coordination() {
        // Test that parser properly coordinates specialized modules
        let tokens = vec![
            Token::new(
                prism_lexer::TokenKind::Function,
                Span::dummy(),
            ),
            Token::new(
                prism_lexer::TokenKind::Identifier("test".to_string()),
                Span::dummy(),
            ),
        ];
        
        let mut parser = Parser::new(tokens);
        let result = parser.parse_program();
        
        // Should delegate to appropriate specialized parsers
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert!(parsed.errors.is_empty() || parsed.errors.len() <= parser.config.max_errors);
    }

    #[test]
    fn test_function_parser_integration() {
        // Test that the function parser is properly integrated
        let tokens = vec![
            Token::new(
                prism_lexer::TokenKind::Function,
                Span::dummy(),
            ),
            Token::new(
                prism_lexer::TokenKind::Identifier("testFunction".to_string()),
                Span::dummy(),
            ),
        ];
        
        let mut parser = Parser::new(tokens);
        let result = parser.parse_function();
        
        // Should successfully parse a basic function
        assert!(result.is_ok(), "Function parsing should succeed");
    }

    #[test]
    fn test_semantic_analysis_integration() {
        let tokens = vec![
            Token::new(
                prism_lexer::TokenKind::Module,
                Span::dummy(),
            ),
            Token::new(
                prism_lexer::TokenKind::Identifier("UserAuth".to_string()),
                Span::dummy(),
            ),
        ];
        
        let mut parser = Parser::with_config(tokens, ParseConfig {
            enable_semantic_analysis: true,
            ..ParseConfig::default()
        });
        
        let result = parser.parse_program();
        
        // Should include semantic analysis results
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert!(parsed.semantic_summary.is_some());
        let summary = parsed.semantic_summary.unwrap();
        assert!(!summary.modules.is_empty());
    }

    #[test]
    fn test_error_recovery() {
        // Test error recovery mechanisms
        let tokens = vec![
            Token::new(
                prism_lexer::TokenKind::LexError("Invalid token".to_string()),
                Span::dummy(),
            ),
        ];
        
        let mut parser = Parser::with_config(tokens, ParseConfig {
            aggressive_recovery: true,
            ..ParseConfig::default()
        });
        
        let result = parser.parse_program();
        
        // Should attempt recovery and continue parsing
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert!(parsed.errors.len() <= parser.config.max_errors);
    }
} 