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
use prism_ast::{AstNode, ModuleKind, SectionKind, CapabilityKind, NodeId};
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

    /// Parse a module definition
    pub fn parse_module(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        // Parse module keyword (either 'module' or 'mod')
        match self.token_stream.current_kind() {
            TokenKind::Module | TokenKind::Mod => {
                self.token_stream.advance();
            }
            _ => return Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "module or mod".to_string(),
            )),
        }
        
        // Parse module name
        let name = self.token_stream.expect_identifier()?;
        
        // Optional module metadata
        let mut metadata = HashMap::new();
        if self.token_stream.consume(TokenKind::At) {
            metadata = self.parse_module_metadata()?;
        }
        
        // Parse module body
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut sections = Vec::new();
        let mut capabilities = Vec::new();
        let mut imports = Vec::new();
        let mut exports = Vec::new();
        let mut statements = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            match self.token_stream.current_kind() {
                // Section declarations
                TokenKind::Section => {
                    sections.push(self.parse_section()?);
                }
                
                // Capability declarations
                TokenKind::Capability => {
                    capabilities.push(self.parse_capability()?);
                }
                
                // Import statements
                TokenKind::Import | TokenKind::Use => {
                    imports.push(self.parse_import()?);
                }
                
                // Export statements
                TokenKind::Export | TokenKind::Pub => {
                    exports.push(self.parse_export()?);
                }
                
                // Regular statements
                _ => {
                    statements.push(self.stmt_parser.parse_statement()?);
                }
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_module_node(
            ModuleKind {
                name,
                sections,
                capabilities,
                imports,
                exports,
                statements,
                metadata,
            },
            span,
        ))
    }

    /// Parse a section within a module
    fn parse_section(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Section)?;
        
        // Parse section name
        let name = self.token_stream.expect_identifier()?;
        
        // Optional section purpose/description
        let purpose = if self.token_stream.check(TokenKind::StringLiteral(_)) {
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
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_section_node(
            SectionKind {
                name,
                purpose,
                statements,
            },
            span,
        ))
    }

    /// Parse a capability declaration
    fn parse_capability(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        // Parse capability keyword (either 'capability' or 'cap')
        match self.token_stream.current_kind() {
            TokenKind::Capability => {
                self.token_stream.advance();
            }
            _ => return Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "capability".to_string(),
            )),
        }
        
        // Parse capability name
        let name = self.token_stream.expect_identifier()?;
        
        // Optional capability signature/interface
        let signature = if self.token_stream.consume(TokenKind::Colon) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Optional capability requirements
        let mut requirements = Vec::new();
        if self.token_stream.consume(TokenKind::Requires) {
            requirements = self.parse_capability_requirements()?;
        }
        
        // Optional capability effects
        let mut effects = Vec::new();
        if self.token_stream.consume(TokenKind::Effects) {
            effects = self.parse_capability_effects()?;
        }
        
        // Optional capability security level
        let security_level = if self.token_stream.consume(TokenKind::Secure) {
            Some(self.parse_security_level()?)
        } else {
            None
        };
        
        // Parse capability body
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut operations = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            // Parse capability operations (functions, types, etc.)
            operations.push(self.stmt_parser.parse_statement()?);
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_capability_node(
            CapabilityKind {
                name,
                signature,
                requirements,
                effects,
                security_level,
                operations,
            },
            span,
        ))
    }

    /// Parse an import statement
    fn parse_import(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        // Parse import keyword
        let is_use_syntax = match self.token_stream.current_kind() {
            TokenKind::Import => {
                self.token_stream.advance();
                false
            }
            TokenKind::Use => {
                self.token_stream.advance();
                true
            }
            _ => return Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "import or use".to_string(),
            )),
        };
        
        // Parse import path
        let path = self.parse_import_path()?;
        
        // Optional import alias
        let alias = if self.token_stream.consume(TokenKind::As) {
            Some(self.token_stream.expect_identifier()?)
        } else {
            None
        };
        
        // Optional from clause (for selective imports)
        let from_module = if self.token_stream.consume(TokenKind::From) {
            Some(self.parse_import_path()?)
        } else {
            None
        };
        
        // Expect semicolon
        self.token_stream.expect(TokenKind::Semicolon)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_import_node(
            ImportKind {
                path,
                alias,
                from_module,
                is_use_syntax,
            },
            span,
        ))
    }

    /// Parse an export statement
    fn parse_export(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        // Parse export keyword
        let visibility = match self.token_stream.current_kind() {
            TokenKind::Export => {
                self.token_stream.advance();
                "export"
            }
            TokenKind::Pub => {
                self.token_stream.advance();
                "pub"
            }
            TokenKind::Public => {
                self.token_stream.advance();
                "public"
            }
            _ => return Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "export, pub, or public".to_string(),
            )),
        };
        
        // Parse the item being exported
        let exported_item = self.stmt_parser.parse_statement()?;
        
        let end_span = self.token_stream.current_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_export_node(
            ExportKind {
                visibility: visibility.to_string(),
                item: exported_item,
            },
            span,
        ))
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

    /// Parse security level
    fn parse_security_level(&mut self) -> ParseResult<String> {
        // Parse security level (identifier or string)
        if self.token_stream.check_identifier() {
            Ok(self.token_stream.expect_identifier()?)
        } else if self.token_stream.check(TokenKind::StringLiteral(_)) {
            self.parse_string_literal()
        } else {
            Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "security level".to_string(),
            ))
        }
    }

    /// Parse import path (dot-separated identifiers)
    fn parse_import_path(&mut self) -> ParseResult<String> {
        let mut path_parts = Vec::new();
        
        // Parse first identifier
        path_parts.push(self.token_stream.expect_identifier()?);
        
        // Parse additional path components
        while self.token_stream.consume(TokenKind::Dot) {
            path_parts.push(self.token_stream.expect_identifier()?);
        }
        
        Ok(path_parts.join("."))
    }

    /// Parse a string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match self.token_stream.current_kind() {
            TokenKind::StringLiteral(value) => {
                let result = value.clone();
                self.token_stream.advance();
                Ok(result)
            }
            _ => Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "string literal".to_string(),
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