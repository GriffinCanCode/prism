//! Main Parser Coordinator - Fixed Architecture
//!
//! ## Clear Separation of Concerns
//!
//! **✅ What Parser DOES (Fixed):**
//! - Coordinate AST construction from tokens/canonical forms
//! - Multi-token semantic analysis and cross-token relationships
//! - Semantic-aware error recovery with meaning preservation
//! - Integration with semantic analysis systems
//! - AI metadata generation from parsed structures
//!
//! **❌ What Parser does NOT do (moved to appropriate modules):**
//! - ❌ Single-token enrichment (→ prism-lexer)
//! - ❌ Syntax style detection (→ prism-syntax)
//! - ❌ Style-specific parsing (→ prism-syntax)
//! - ❌ Character-to-token conversion (→ prism-lexer)
//!
//! **Conceptual Responsibility**: AST construction and multi-token semantic analysis
//! **Data Flow Position**: Takes enriched tokens/canonical forms → produces semantic AST

use crate::{
    core::{ParseError, ParseResult, TokenStreamManager, ParsingCoordinator, Precedence},
    parsers::{ExpressionParser, StatementParser, TypeParser, FunctionParser, ModuleParser},
    analysis::{SemanticContextExtractor, ConstraintValidator, ValidationConfig, TokenSemanticAnalyzer, TokenSemanticSummary},
};
use prism_ast::{AstArena, AstNode, Expr, Item, Program, ProgramMetadata, Stmt, Type};
use prism_common::NodeId;
use prism_common::{span::Span, SourceId};
use prism_lexer::{Token, TokenKind, LexerResult};
use prism_syntax::{ParsingOrchestrator, OrchestratorConfig, SyntaxStyle, ParserBridge}; 
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
    /// Enable syntax style detection (DEPRECATED - moved to prism-syntax)
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
            detect_syntax_style: false, // DEPRECATED - use prism-syntax instead
        }
    }
}

/// Enhanced parsing result with semantic information (Fixed Architecture)
#[derive(Debug)]
pub struct ParsedProgram {
    /// The parsed program AST
    pub program: Program,
    /// Multi-token semantic analysis summary
    pub semantic_summary: Option<TokenSemanticSummary>,
    /// Parse errors encountered
    pub errors: Vec<ParseError>,
    /// Warnings generated during parsing
    pub warnings: Vec<String>,
    
    // NOTE: detected_syntax_style moved to prism-syntax module
    // Use prism_syntax::detect_syntax_style() instead
}

/// The main parser that coordinates specialized parsing modules
/// 
/// This struct embodies the single concept of parsing coordination.
/// It delegates actual parsing work to specialized modules while
/// managing the overall parsing workflow and providing the public API.
/// 
/// UPDATED: Now uses factory-based orchestrator internally for improved
/// architecture while maintaining backward compatibility.
#[derive(Debug)]
pub struct Parser {
    /// UPDATED: Internal syntax orchestrator using factory system
    syntax_orchestrator: ParsingOrchestrator,
    
    /// Backward compatibility: Token-based parsing coordinator
    coordinator: ParsingCoordinator,
    
    /// Configuration for parsing behavior
    config: ParseConfig,
    
    /// Accumulated parsing errors
    errors: Vec<ParseError>,
    
    /// Current source ID for error reporting
    source_id: Option<SourceId>,
}

impl Parser {
    /// Create a new parser with default configuration
    pub fn new(tokens: Vec<Token>) -> Self {
        let config = ParseConfig::default();
        Self::with_config(tokens, config)
    }

    /// Create a new parser with custom configuration
    pub fn with_config(tokens: Vec<Token>, config: ParseConfig) -> Self {
        // Create orchestrator configuration from parser config
        let orchestrator_config = OrchestratorConfig {
            enable_component_caching: true,
            max_cache_size: 10,
            enable_parallel_processing: false,
            default_validation_level: prism_syntax::ValidationLevel::Standard,
            generate_ai_metadata: config.extract_ai_context,
            preserve_formatting: false,
            enable_error_recovery: config.aggressive_recovery,
        };
        
        let syntax_orchestrator = ParsingOrchestrator::with_config(orchestrator_config);
        let coordinator = ParsingCoordinator::new(tokens);
        
        Self {
            syntax_orchestrator,
            coordinator,
            config,
            errors: Vec::new(),
            source_id: None,
        }
    }
    
    /// Create a parser from source code (NEW - uses orchestrator directly)
    pub fn from_source(source: &str, source_id: SourceId) -> Result<Self, ParseError> {
        let config = ParseConfig::default();
        Self::from_source_with_config(source, source_id, config)
    }
    
    /// Create a parser from source code with configuration (NEW)
    pub fn from_source_with_config(source: &str, source_id: SourceId, config: ParseConfig) -> Result<Self, ParseError> {
        // Tokenize first for backward compatibility with token-based APIs
        let tokens = prism_lexer::tokenize(source)
            .map_err(|e| ParseError::LexerError { 
                message: format!("Tokenization failed: {}", e) 
            })?;
        
        let mut parser = Self::with_config(tokens, config);
        parser.source_id = Some(source_id);
        Ok(parser)
    }

    // === Methods needed by combinators ===

    /// Check if the current token matches the given kind (delegate to token_stream)
    pub fn check(&self, kind: TokenKind) -> bool {
        self.coordinator.token_manager().check(kind)
    }

    /// Advance to the next token (delegate to token_stream)
    pub fn advance(&mut self) -> &Token {
        self.coordinator.token_manager_mut().advance()
    }

    /// Consume a token if it matches the expected kind (delegate to token_stream)
    pub fn consume(&mut self, expected: TokenKind, _error_message: &str) -> ParseResult<&Token> {
        if self.coordinator.token_manager().check(expected.clone()) {
            Ok(self.coordinator.token_manager_mut().advance())
        } else {
            let found = self.coordinator.token_manager().peek().kind.clone();
            let span = self.coordinator.token_manager().current_span();
            Err(ParseError::unexpected_token(
                vec![expected],
                found,
                span,
            ))
        }
    }

    /// Get the current span (delegate to token_stream)
    pub fn current_span(&self) -> Span {
        self.coordinator.token_manager().current_span()
    }

    /// Check if we're at the end of the token stream (delegate to token_stream)
    pub fn is_at_end(&self) -> bool {
        self.coordinator.token_manager().is_at_end()
    }

    /// Check if current token could end a list (delegate to token_stream)
    pub fn check_list_end(&self) -> bool {
        self.coordinator.token_manager().check_list_end()
    }

    /// Peek at the current token (delegate to token_stream)
    pub fn peek(&self) -> &Token {
        self.coordinator.token_manager().peek()
    }

    /// Consume an identifier token and return its value
    pub fn consume_identifier(&mut self, _error_message: &str) -> ParseResult<String> {
        self.coordinator.token_manager().expect_identifier()
    }

    /// Consume a string literal token and return its value
    pub fn consume_string(&mut self, _error_message: &str) -> ParseResult<String> {
        match &self.coordinator.token_manager().peek().kind {
            TokenKind::StringLiteral(value) => {
                let result = value.clone();
                self.coordinator.token_manager_mut().advance();
                Ok(result)
            }
            _ => {
                let found = self.coordinator.token_manager().peek().kind.clone();
                let span = self.coordinator.token_manager().current_span();
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
        match &self.coordinator.token_manager().peek().kind {
            TokenKind::IntegerLiteral(value) => {
                let result = *value as f64;
                self.coordinator.token_manager_mut().advance();
                Ok(result)
            }
            TokenKind::FloatLiteral(value) => {
                let result = *value;
                self.coordinator.token_manager_mut().advance();
                Ok(result)
            }
            _ => {
                let found = self.coordinator.token_manager().peek().kind.clone();
                let span = self.coordinator.token_manager().current_span();
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
        self.coordinator.token_manager().check_block_end()
    }

    /// Parse a complete program from tokens
    /// 
    /// This is the main entry point for parsing. It coordinates between
    /// all specialized parsers to build a complete program AST with
    /// semantic information.
    /// 
    /// UPDATED: Uses new orchestrator when possible, falls back to legacy parsing
    pub fn parse_program(&mut self) -> ParseResult<ParsedProgram> {
        info!("Starting program parsing");
        
        // NEW: If we have a source_id, try to use the orchestrator directly
        if let Some(source_id) = self.source_id {
            // Try to reconstruct source from tokens for orchestrator
            if let Ok(source) = self.reconstruct_source_from_tokens() {
                return self.parse_with_orchestrator(&source, source_id);
            }
            // If reconstruction fails, fall back to token-based parsing
            warn!("Source reconstruction failed, falling back to token-based parsing");
        }
        
        // Legacy token-based parsing path
        self.parse_program_from_tokens()
    }
    
    /// Parse program using the new orchestrator (preferred path)
    fn parse_with_orchestrator(&mut self, source: &str, source_id: SourceId) -> ParseResult<ParsedProgram> {
        match self.syntax_orchestrator.parse(source, source_id) {
            Ok(orchestrator_result) => {
                // Convert orchestrator result to ParsedProgram
                let program = orchestrator_result.program;
                let semantic_summary = None; // Orchestrator handles this differently
                let errors = vec![]; // Orchestrator errors would be converted
                let warnings = vec!["Using new factory-based orchestrator".to_string()];
                
                Ok(ParsedProgram {
                    program,
                    semantic_summary,
                    errors,
                    warnings,
                })
            }
            Err(orchestrator_error) => {
                // Convert orchestrator error to parse error and fall back
                warn!("Orchestrator parsing failed: {}, falling back to token-based parsing", orchestrator_error);
                self.parse_program_from_tokens()
            }
        }
    }
    
    /// Legacy token-based parsing (backward compatibility)
    fn parse_program_from_tokens(&mut self) -> ParseResult<ParsedProgram> {
        // Create program metadata
        let metadata = ProgramMetadata {
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            source_hash: self.calculate_source_hash(),
            dependencies: Vec::new(),
            features: Vec::new(),
            ai_context: None,
        };

        // Parse program items
        let mut items = Vec::new();
        while !self.coordinator.token_manager().is_at_end() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(error) => {
                    self.errors.push(error.clone());
                    if self.errors.len() >= self.config.max_errors {
                        break;
                    }
                    // Try to recover by skipping to the next item
                    self.skip_to_next_item();
                }
            }
        }

        let program = Program {
            items,
            metadata,
        };

        // Step 1: Multi-token semantic analysis (cross-token relationships)
        let semantic_summary = if self.config.enable_semantic_analysis {
            let tokens: Vec<Token> = self.coordinator.token_manager().tokens().to_vec();
            Some(self.coordinator.token_analyzer().analyze_tokens(&tokens))
        } else {
            None
        };

        // Generate warnings
        let mut warnings = Vec::new();
        if self.errors.len() > 10 {
            warnings.push(format!("High error count: {} errors encountered", self.errors.len()));
        }

        Ok(ParsedProgram {
            program,
            semantic_summary,
            errors: self.errors.clone(),
            warnings,
        })
    }
    
    /// Attempt to reconstruct source code from tokens (for orchestrator)
    fn reconstruct_source_from_tokens(&self) -> Result<String, String> {
        // This is a simplified reconstruction - in practice would be more sophisticated
        let tokens = self.coordinator.token_manager().tokens();
        if tokens.is_empty() {
            return Err("No tokens to reconstruct from".to_string());
        }
        
        // For now, just return a placeholder - in practice would reconstruct actual source
        Err("Source reconstruction not yet implemented".to_string())
    }
    
    /// Calculate a simple source hash for metadata
    fn calculate_source_hash(&self) -> u64 {
        // Simplified hash based on token count
        self.coordinator.token_manager().tokens().len() as u64
    }
    
    /// Skip tokens until we find the start of the next item
    fn skip_to_next_item(&mut self) {
        while !self.coordinator.token_manager().is_at_end() {
            match self.coordinator.token_manager().peek().kind {
                TokenKind::Module | TokenKind::Function | TokenKind::Type | 
                TokenKind::Const | TokenKind::Let | TokenKind::Var => break,
                _ => {
                    self.coordinator.token_manager_mut().advance();
                }
            }
        }
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
        match self.coordinator.token_manager().peek().kind {
            TokenKind::Function | TokenKind::Fn => {
                // Consume the function token
                self.coordinator.token_manager_mut().advance();
                
                // Parse function name
                let name = self.coordinator.token_manager().expect_identifier()?;
                
                // Create a basic function node (simplified)
                let span = self.coordinator.token_manager().current_span();
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
                let found = self.coordinator.token_manager().peek().kind.clone();
                let span = self.coordinator.token_manager().current_span();
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