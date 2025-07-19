//! Parsing Coordination
//!
//! This module embodies the single concept of "Parsing Coordination".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: coordinating the parsing process by delegating to specialized parsers.
//!
//! **Conceptual Responsibility**: Orchestrate parsing workflow
//! **What it does**: coordinate parsers, manage errors, handle recovery
//! **What it doesn't do**: actual parsing logic, token navigation, semantic analysis

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager},
    analysis::constraint_validation::{ConstraintValidator, ValidationConfig},
    parser::ParseConfig,
};
use prism_ast::{AstArena, AstNode, Program, ProgramMetadata, Item, Expr, Stmt, Type};
use prism_common::NodeId;
use prism_common::{span::Span, SourceId};
use prism_lexer::Token;

/// Parsing coordinator that orchestrates the parsing process
/// 
/// This struct embodies the single concept of coordinating parsing.
/// It delegates to specialized parsers but contains no parsing logic itself.
/// It manages the overall parsing workflow, error collection, and recovery.
#[derive(Debug)]
pub struct ParsingCoordinator {
    /// Token stream manager for navigation
    token_manager: TokenStreamManager,
    /// Memory arena for AST nodes
    arena: AstArena,
    /// Parse errors encountered
    errors: Vec<ParseError>,
    /// Parser configuration
    config: ParseConfig,
    /// Source ID for span creation
    source_id: SourceId,
    /// Recovery mode flag
    recovery_mode: bool,
    /// Next node ID
    next_id: u32,
    /// Constraint validator for semantic types
    constraint_validator: ConstraintValidator,
}

impl ParsingCoordinator {
    /// Create a new parsing coordinator
    pub fn new(tokens: Vec<Token>) -> Self {
        Self::with_config(tokens, ParseConfig::default())
    }

    /// Create a new parsing coordinator with custom configuration
    pub fn with_config(tokens: Vec<Token>, config: ParseConfig) -> Self {
        let source_id = if tokens.is_empty() {
            SourceId::new(0)
        } else {
            tokens[0].span.source_id
        };

        Self {
            token_manager: TokenStreamManager::new(tokens),
            arena: AstArena::new(source_id),
            errors: Vec::new(),
            config,
            source_id,
            recovery_mode: false,
            next_id: 0,
            constraint_validator: ConstraintValidator::new(ValidationConfig::default()),
        }
    }

    /// Coordinate parsing of a complete program
    /// 
    /// This method orchestrates the parsing process by:
    /// 1. Delegating item parsing to specialized parsers
    /// 2. Collecting errors and managing recovery
    /// 3. Generating program metadata
    /// 4. Ensuring conceptual cohesion throughout
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let mut items = Vec::new();

        while !self.token_manager.is_at_end() {
            match self.coordinate_item_parsing() {
                Ok(item) => items.push(item),
                Err(error) => {
                    self.errors.push(error);
                    if self.errors.len() >= self.config.max_errors {
                        break;
                    }
                    self.coordinate_error_recovery();
                }
            }
        }

        let metadata = self.generate_program_metadata(&items);

        if self.errors.is_empty() {
            Ok(Program {
                items,
                source_id: self.source_id,
                metadata,
            })
        } else {
            Err(self.errors[0].clone())
        }
    }

    /// Coordinate parsing of a single expression
    pub fn parse_expression(&mut self) -> ParseResult<AstNode<Expr>> {
        // For now, create a placeholder expression
        let span = self.token_manager.current_span();
        Ok(self.create_node(
            Expr::Error(prism_ast::ErrorExpr {
                message: "Expression parsing not yet implemented".to_string(),
            }),
            span,
        ))
    }

    /// Coordinate parsing of a single statement
    pub fn parse_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        // For now, create a placeholder statement
        let span = self.token_manager.current_span();
        Ok(self.create_node(
            Stmt::Error(prism_ast::ErrorStmt {
                message: "Statement parsing not yet implemented".to_string(),
            }),
            span,
        ))
    }

    /// Coordinate parsing of a type annotation
    pub fn parse_type(&mut self) -> ParseResult<AstNode<Type>> {
        // For now, create a placeholder type
        let span = self.token_manager.current_span();
        Ok(self.create_node(
            Type::Error(prism_ast::ErrorType {
                message: "Type parsing not yet implemented".to_string(),
            }),
            span,
        ))
    }

    // Private coordination methods

    /// Coordinate parsing of a single item by delegating to appropriate parser
    fn coordinate_item_parsing(&mut self) -> ParseResult<AstNode<Item>> {
        let current_token = &self.token_manager.peek().kind;
        let span = self.token_manager.current_span();

        match current_token {
            prism_lexer::TokenKind::Module => {
                // For now, create a placeholder module
                self.token_manager.advance(); // consume 'module'
                Ok(self.create_node(
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
            _ => {
                // For now, create a placeholder statement wrapped as item
                self.token_manager.advance(); // consume current token
                Ok(self.create_node(
                    Item::Statement(prism_ast::Stmt::Error(prism_ast::ErrorStmt {
                        message: "Item parsing not yet implemented".to_string(),
                    })),
                    span,
                ))
            }
        }
    }

    /// Coordinate error recovery by delegating to recovery strategies
    fn coordinate_error_recovery(&mut self) {
        self.recovery_mode = true;
        self.token_manager.synchronize_to_statement();
        self.recovery_mode = false;
    }

    /// Generate program metadata using AI context extraction
    fn generate_program_metadata(&self, items: &[AstNode<Item>]) -> ProgramMetadata {
        ProgramMetadata {
            primary_capability: if self.config.extract_ai_context {
                Some("Prism program with AI-first design".to_string())
            } else {
                None
            },
            capabilities: Vec::new(),
            dependencies: Vec::new(),
            security_implications: Vec::new(),
            performance_notes: Vec::new(),
            ai_insights: if self.config.extract_ai_context {
                Some(format!(
                    "Program with {} items, complexity score: {}",
                    items.len(),
                    self.calculate_complexity_score(items)
                ))
            } else {
                None
            },
        }
    }

    /// Calculate complexity score for the program
    fn calculate_complexity_score(&self, items: &[AstNode<Item>]) -> f64 {
        // Simple complexity metric based on item count and nesting
        let base_score = items.len() as f64 * 0.1;
        base_score.min(1.0)
    }

    /// Create a new AST node with arena allocation
    pub fn create_node<T>(&mut self, kind: T, span: Span) -> AstNode<T> {
        let id = self.next_node_id();
        AstNode::new(kind, span, id)
    }

    /// Get the next node ID
    fn next_node_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        NodeId::new(id)
    }

    // Accessor methods for specialized parsers

    /// Get mutable reference to token manager
    pub fn token_manager_mut(&mut self) -> &mut TokenStreamManager {
        &mut self.token_manager
    }

    /// Get reference to token manager
    pub fn token_manager(&self) -> &TokenStreamManager {
        &self.token_manager
    }

    /// Get configuration
    pub fn config(&self) -> &ParseConfig {
        &self.config
    }

    /// Get source ID
    pub fn source_id(&self) -> SourceId {
        self.source_id
    }

    /// Check if in recovery mode
    pub fn is_in_recovery_mode(&self) -> bool {
        self.recovery_mode
    }

    /// Add error to collection
    pub fn add_error(&mut self, error: ParseError) {
        self.errors.push(error);
    }

    /// Get all errors
    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Get mutable reference to constraint validator
    pub fn constraint_validator_mut(&mut self) -> &mut ConstraintValidator {
        &mut self.constraint_validator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::{span::Position, SourceId};

    fn create_test_token(kind: prism_lexer::TokenKind) -> Token {
        Token {
            kind,
            span: Span::new(
                Position::new(1, 1, 0),
                Position::new(1, 2, 1),
                SourceId::new(1),
            ),
            semantic_context: None,
        }
    }

    #[test]
    fn test_coordinator_creation() {
        let tokens = vec![
            create_test_token(prism_lexer::TokenKind::Module),
            create_test_token(prism_lexer::TokenKind::Identifier("Test".to_string())),
        ];

        let coordinator = ParsingCoordinator::new(tokens);
        
        assert_eq!(coordinator.errors().len(), 0);
        assert!(!coordinator.is_in_recovery_mode());
        assert_eq!(coordinator.source_id(), SourceId::new(1));
    }

    #[test]
    fn test_node_creation() {
        let tokens = vec![create_test_token(prism_lexer::TokenKind::Let)];
        let mut coordinator = ParsingCoordinator::new(tokens);

        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 4, 3),
            SourceId::new(1),
        );

        let node = coordinator.create_node("test_content", span);
        assert_eq!(node.kind, "test_content");
        assert_eq!(node.span, span);
    }
} 