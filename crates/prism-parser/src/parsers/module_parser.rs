//! Module and Section Parsing
//!
//! This module embodies the single concept of "Module and Section Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing module structure, sections, and capability declarations.
//!
//! **Conceptual Responsibility**: Parse module organization and boundaries
//! **What it does**: modules, sections, capabilities, imports, exports
//! **What it doesn't do**: function parsing, type parsing, expression parsing

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager, parsing_coordinator::ParsingCoordinator},
    parsers::{statement_parser::StatementParser, type_parser::TypeParser},
};
use prism_ast::{AstNode, ModuleDecl, SectionDecl, SectionKind};
use prism_common::NodeId;
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use std::collections::HashMap;

/// Module parser - handles module structure and organization
/// 
/// This struct embodies the single concept of parsing module boundaries.
/// It understands Prism's conceptual cohesion principles and ensures
/// that each module represents a single business capability.
pub struct ModuleParser<'a> {
    /// Reference to the token stream manager (no ownership)
    token_stream: &'a mut TokenStreamManager,
    /// Reference to coordinator for node creation and error handling
    coordinator: &'a mut ParsingCoordinator,
    /// Reference to statement parser for module contents
    stmt_parser: &'a mut StatementParser<'a>,
    /// Reference to type parser for capability signatures
    type_parser: &'a mut TypeParser<'a>,
}

impl<'a> ModuleParser<'a> {
    /// Create a new module parser
    pub fn new(
        token_stream: &'a mut TokenStreamManager,
        coordinator: &'a mut ParsingCoordinator,
        stmt_parser: &'a mut StatementParser<'a>,
        type_parser: &'a mut TypeParser<'a>,
    ) -> Self {
        Self {
            token_stream,
            coordinator,
            stmt_parser,
            type_parser,
        }
    }

    /// Parse a module declaration
    pub fn parse_module(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            TokenKind::Module => {
                self.token_stream.advance(); // consume 'module'
                
                // Parse module name
                let name = self.token_stream.expect_identifier()?;
                let name_symbol = prism_common::symbol::Symbol::intern(&name);
                
                // Parse optional module metadata
                let metadata = self.parse_module_metadata()?;
                
                // Parse optional capability declaration
                let capability = if self.token_stream.check(TokenKind::Capability) {
                    Some(self.parse_capability()?)
                } else {
                    None
                };
                
                // Parse module body (sections, imports, exports)
                let mut sections = Vec::new();
                let mut dependencies = Vec::new();
                
                if self.token_stream.check(TokenKind::LeftBrace) {
                    self.token_stream.advance(); // consume '{'
                    
                    while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
                        match self.token_stream.current_kind() {
                            TokenKind::Section => {
                                let section = self.parse_section()?;
                                sections.push(section);
                            }
                            TokenKind::Import => {
                                let import = self.parse_import()?;
                                dependencies.push(import);
                            }
                            TokenKind::Export => {
                                let _export = self.parse_export()?;
                                // Handle exports
                            }
                            _ => {
                                // Skip unknown tokens for now
                                self.token_stream.advance();
                            }
                        }
                    }
                    
                    self.token_stream.expect(TokenKind::RightBrace)?;
                }
                
                let end_span = self.token_stream.current_span();
                let span = self.combine_spans(start_span, end_span);
                
                // Create module declaration
                let module_decl = prism_ast::ModuleDecl {
                    name: name_symbol,
                    capability,
                    description: metadata.get("description").cloned(),
                    dependencies: Vec::new(), // TODO: Convert imports to dependencies
                    stability: prism_ast::StabilityLevel::Experimental,
                    version: metadata.get("version").cloned(),
                    sections,
                    ai_context: metadata.get("ai_context").cloned(),
                    visibility: prism_ast::Visibility::Public,
                };
                
                let module_item = prism_ast::Item::Module(module_decl);
                let node = self.coordinator.create_node(module_item, span);
                Ok(node.id)
            }
            _ => return Err(ParseError::unexpected_token(
                vec![TokenKind::Module],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse a section within a module
    fn parse_section(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Section)?;
        
        // Parse section name
        let name = self.token_stream.expect_identifier()?;
        
        // Optional section purpose/description
        let purpose = if matches!(self.token_stream.current_kind(), TokenKind::StringLiteral(_)) {
            Some(self.parse_string_literal()?)
        } else {
            None
        };
        
        // Parse section body
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut statements = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            statements.push(self.stmt_parser.parse_statement()?);
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(&start_span, &end_span).unwrap_or(start_span);
        
        Ok(self.coordinator.create_node(
            SectionKind {
                name,
                purpose,
                statements,
            },
            span,
        ))
    }

    /// Parse a capability declaration
    fn parse_capability(&mut self) -> ParseResult<AstNode<Item>> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            TokenKind::Capability => {
                self.token_stream.advance(); // consume 'capability'
                
                let name = self.token_stream.expect_identifier()?;
                
                // Parse capability requirements
                let requirements = self.parse_capability_requirements()?;
                
                // Parse capability effects
                let effects = self.parse_capability_effects()?;
                
                // Parse security level
                let security_level = if matches!(self.token_stream.current_kind(), TokenKind::Identifier(name) if name == "security") {
                    Some(self.parse_security_level()?)
                } else {
                    None
                };
                
                let end_span = self.token_stream.current_span();
                let span = self.combine_spans(start_span, end_span);
                
                // Create a placeholder capability item
                let capability_stmt = prism_ast::Stmt::Error(prism_ast::ErrorStmt {
                    message: format!("Capability '{}' not yet fully implemented", name),
                    context: "capability parsing".to_string(),
                });
                
                let capability_item = prism_ast::Item::Statement(capability_stmt);
                Ok(self.coordinator.create_node(capability_item, span))
            }
            _ => return Err(ParseError::unexpected_token(
                vec![TokenKind::Capability],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Helper function to combine spans safely
    fn combine_spans(&self, start: Span, end: Span) -> Span {
        start.combine(&end).unwrap_or(start)
    }

    /// Parse an import statement
    fn parse_import(&mut self) -> ParseResult<AstNode<Item>> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            TokenKind::Import | TokenKind::Use => {
                self.token_stream.advance(); // consume import/use keyword
                
                let path = self.parse_import_path()?;
                
                // Check for alias
                let alias = if self.token_stream.check(TokenKind::As) {
                    self.token_stream.advance(); // consume 'as'
                    Some(self.token_stream.expect_identifier()?)
                } else {
                    None
                };
                
                let end_span = self.token_stream.current_span();
                let span = self.combine_spans(start_span, end_span);
                
                // Create import declaration
                let import_decl = prism_ast::ImportDecl {
                    path: path.clone(),
                    alias: alias.map(|a| prism_common::symbol::Symbol::intern(&a)),
                    items: prism_ast::ImportItems::All, // For now, import everything
                };
                
                let import_item = prism_ast::Item::Import(import_decl);
                Ok(self.coordinator.create_node(import_item, span))
            }
            _ => return Err(ParseError::unexpected_token(
                vec![TokenKind::Import, TokenKind::Use],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse an export statement
    fn parse_export(&mut self) -> ParseResult<AstNode<Item>> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            TokenKind::Export | TokenKind::Pub | TokenKind::Public => {
                self.token_stream.advance(); // consume export keyword
                
                // For now, create a placeholder export
                let item_name = if self.token_stream.check_identifier() {
                    self.token_stream.expect_identifier()?
                } else {
                    "unknown".to_string()
                };
                
                let end_span = self.token_stream.current_span();
                let span = self.combine_spans(start_span, end_span);
                
                // Create export declaration - use ExportItems::All for now
                let export_decl = prism_ast::ExportDecl {
                    items: prism_ast::ExportItems::All,
                };
                
                let export_item = prism_ast::Item::Export(export_decl);
                Ok(self.coordinator.create_node(export_item, span))
            }
            _ => return Err(ParseError::unexpected_token(
                vec![TokenKind::Export, TokenKind::Pub, TokenKind::Public],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse module metadata (annotations)
    fn parse_module_metadata(&mut self) -> ParseResult<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        // Parse annotation-style metadata
        while self.token_stream.check_identifier() {
            let key = self.token_stream.expect_identifier()?;
            
            if self.token_stream.consume(TokenKind::Assign) {
                let value = self.parse_string_literal()?;
                metadata.insert(key, value);
            } else {
                // Boolean flag
                metadata.insert(key, "true".to_string());
            }
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        Ok(metadata)
    }

    /// Parse capability requirements
    fn parse_capability_requirements(&mut self) -> ParseResult<Vec<String>> {
        let mut requirements = Vec::new();
        
        // Parse requirement list
        while self.token_stream.check_identifier() {
            requirements.push(self.token_stream.expect_identifier()?);
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        Ok(requirements)
    }

    /// Parse capability effects
    fn parse_capability_effects(&mut self) -> ParseResult<Vec<String>> {
        let mut effects = Vec::new();
        
        // Parse effect list
        while self.token_stream.check_identifier() {
            effects.push(self.token_stream.expect_identifier()?);
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        Ok(effects)
    }

    /// Parse security level specification (placeholder for now)
    fn parse_security_level(&mut self) -> Result<String, ParseError> {
        // For now, just consume an identifier as security level
        match self.token_stream.current_kind() {
            TokenKind::Identifier(level) => {
                let level_str = level.clone();
                self.token_stream.advance();
                Ok(level_str)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::Identifier("security level".to_string())],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse import path
    fn parse_import_path(&mut self) -> ParseResult<String> {
        // For now, just parse a string literal or identifier
        match self.token_stream.current_kind() {
            TokenKind::StringLiteral(path) => {
                let path_str = path.clone();
                self.token_stream.advance();
                Ok(path_str)
            }
            TokenKind::Identifier(path) => {
                let path_str = path.clone();
                self.token_stream.advance();
                Ok(path_str)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::StringLiteral("path".to_string()), TokenKind::Identifier("path".to_string())],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match self.token_stream.current_kind() {
            TokenKind::StringLiteral(value) => {
                let result = value.clone();
                self.token_stream.advance();
                Ok(result)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::StringLiteral("string".to_string())],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Validate module conceptual cohesion
    pub fn validate_module_cohesion(&self, module_node: NodeId) -> ParseResult<()> {
        // Validate that the module follows conceptual cohesion principles
        // This would analyze the module structure to ensure:
        // 1. Single responsibility principle
        // 2. Clear capability boundaries  
        // 3. Minimal inter-module dependencies
        // 4. Semantic consistency
        
        // Placeholder implementation
        Ok(())
    }

    /// Extract AI-comprehensible module metadata
    pub fn extract_module_metadata(&self, module_node: NodeId) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Extract semantic meaning from module structure
        metadata.insert("concept".to_string(), "Module Definition".to_string());
        metadata.insert("purpose".to_string(), "Encapsulate business capability".to_string());
        metadata.insert("ai_hint".to_string(), "Modules represent conceptual boundaries".to_string());
        metadata.insert("cohesion_principle".to_string(), "One concept per module".to_string());
        
        metadata
    }
}

// Helper types for AST node creation
#[derive(Debug, Clone)]
pub struct ImportKind {
    pub path: String,
    pub alias: Option<String>,
    pub from_module: Option<String>,
    pub is_use_syntax: bool,
}

#[derive(Debug, Clone)]
pub struct ExportKind {
    pub visibility: String,
    pub item: NodeId,
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test utilities would be defined here or in a test module

    #[test]
    fn test_module_definition() {
        let source = r#"
            module UserAuth {
                section Authentication {
                    function login() {}
                }
                
                capability Authenticate : AuthService {
                    function verify_credentials() {}
                }
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_module();
        assert!(result.is_ok());
    }

    #[test]
    fn test_section_parsing() {
        let source = r#"
            section DataValidation "Validates user input" {
                function validate_email() {}
                function validate_password() {}
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_section();
        assert!(result.is_ok());
    }

    #[test]
    fn test_capability_parsing() {
        let source = r#"
            capability DatabaseAccess 
                requires connection_pool, transaction_manager
                effects read, write
                secure high {
                
                function query() {}
                function update() {}
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_capability();
        assert!(result.is_ok());
    }

    #[test]
    fn test_import_statements() {
        let sources = vec![
            "import std.collections;",
            "use std.io as IO;",
            "import { HashMap, Vec } from std.collections;",
        ];
        
        for source in sources {
            let mut parser = create_test_parser(source);
            let result = parser.module_parser.parse_import();
            assert!(result.is_ok(), "Failed to parse import: {}", source);
        }
    }

    #[test]
    fn test_export_statements() {
        let sources = vec![
            "export function public_api() {}",
            "pub type UserData = {};",
            "public const MAX_USERS = 1000;",
        ];
        
        for source in sources {
            let mut parser = create_test_parser(source);
            let result = parser.module_parser.parse_export();
            assert!(result.is_ok(), "Failed to parse export: {}", source);
        }
    }
} 