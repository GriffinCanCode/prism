//! Module and Section Parsing
//!
//! This module embodies the single concept of "Module and Section Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing module structure, sections, and capability declarations.
//!
//! **Conceptual Responsibility**: Parse module organization and boundaries
//! **What it does**: modules, sections, capabilities, imports, exports, AI context, lifecycle hooks
//! **What it doesn't do**: function parsing, type parsing, expression parsing

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager, parsing_coordinator::ParsingCoordinator},
    parsers::{statement_parser::StatementParser, type_parser::TypeParser},
};
use prism_ast::{
    AstNode, ModuleDecl, SectionDecl, SectionKind, StabilityLevel, Visibility,
    ModuleDependency, DependencyItems, DependencyItem, AiContext, AiHint, AiHintCategory,
    CriticalPath, PerformanceSla, LifecycleHook, LifecycleEvent, EventDecl, EventPriority,
    SectionRequirement, RequirementKind, InjectionConfig, InjectionBinding, InjectionScope,
    CompositionTrait, Item, Stmt, Expr
};
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

    /// Parse a module declaration with complete PLD-002 support
    pub fn parse_module(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        // Parse module attributes/annotations
        let attributes = self.parse_module_attributes()?;
        
        match self.token_stream.current_kind() {
            TokenKind::Module => {
                self.token_stream.advance(); // consume 'module'
                
                // Parse module name
                let name = self.token_stream.expect_identifier()?;
                let name_symbol = prism_common::symbol::Symbol::intern(&name);
                
                // Parse optional version (@version or @2.1.0)
                let version = if self.token_stream.check(TokenKind::At) {
                    self.token_stream.advance(); // consume '@'
                    Some(self.parse_version_specification()?)
                } else {
                    None
                };
                
                // Parse optional trait implementations
                let implemented_traits = if self.token_stream.check(TokenKind::Implements) {
                    self.token_stream.advance(); // consume 'implements'
                    self.parse_trait_implementation_list()?
                } else {
                    Vec::new()
                };
                
                // Parse module body
                let mut sections = Vec::new();
                let mut dependencies = Vec::new();
                let mut submodules = Vec::new();
                let mut lifecycle_hooks = Vec::new();
                let mut events = Vec::new();
                let mut injection_config = None;
                let mut composition_traits = Vec::new();
                let mut ai_context = None;
                
                if self.token_stream.check(TokenKind::LeftBrace) {
                    self.token_stream.advance(); // consume '{'
                    
                    while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
                        match self.token_stream.current_kind() {
                            // AI Context block
                            TokenKind::At if self.peek_ai_context() => {
                                ai_context = Some(self.parse_ai_context()?);
                            }
                            // Dependency injection
                            TokenKind::Identifier(s) if s == "inject" => {
                                injection_config = Some(self.parse_injection_config()?);
                            }
                            // Section declarations (enhanced with capability requirements)
                            TokenKind::Section => {
                                let section = self.parse_section_with_capabilities()?;
                                sections.push(section);
                            }
                            // Import statements
                            TokenKind::Import => {
                                let import = self.parse_import()?;
                                dependencies.push(self.convert_import_to_dependency(import)?);
                            }
                            // Export statements
                            TokenKind::Export => {
                                let _export = self.parse_export()?;
                                // Handle exports (add to module interface)
                            }
                            // Submodules
                            TokenKind::Module => {
                                let submodule = self.parse_module()?;
                                submodules.push(self.coordinator.get_node(submodule).unwrap().clone());
                            }
                            // Lifecycle hooks
                            TokenKind::Identifier(s) if s == "on" => {
                                let hook = self.parse_lifecycle_hook()?;
                                lifecycle_hooks.push(hook);
                            }
                            // Events
                            TokenKind::Identifier(s) if s == "event" => {
                                let event = self.parse_event()?;
                                events.push(event);
                            }
                            // Composition traits
                            TokenKind::Trait => {
                                let trait_impl = self.parse_composition_trait()?;
                                composition_traits.push(trait_impl);
                            }
                            _ => {
                                // Skip unknown tokens for error recovery
                                self.token_stream.advance();
                            }
                        }
                    }
                    
                    self.token_stream.expect(TokenKind::RightBrace)?;
                }
                
                let end_span = self.token_stream.current_span();
                let span = self.combine_spans(start_span, end_span);
                
                // Extract metadata from attributes
                let (capability, description, stability, dependency_list) = self.extract_module_metadata(&attributes);
                
                // Merge parsed dependencies with declared dependencies
                dependencies.extend(dependency_list);
                
                // Create enhanced module declaration (first without cohesion metadata)
                let mut module_decl = ModuleDecl {
                    name: name_symbol,
                    capability,
                    description,
                    dependencies,
                    stability: stability.unwrap_or(StabilityLevel::Experimental),
                    version,
                    sections,
                    ai_context,
                    visibility: Visibility::Public, // Default for now
                    attributes,
                    submodules,
                    implemented_traits,
                    lifecycle_hooks,
                    events,
                    cohesion_metadata: None, // Will be set below
                    injection_config,
                    composition_traits,
                };
                
                // Extract comprehensive business context
                let business_context = self.extract_business_context(&module_decl)?;
                
                // Perform real-time cohesion analysis
                let cohesion_metadata = self.analyze_module_cohesion(&module_decl).unwrap_or(None);
                module_decl.cohesion_metadata = cohesion_metadata;
                
                let module_item = Item::Module(module_decl);
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

    /// Parse version specification (@2.1.0 or @version "2.1.0")
    fn parse_version_specification(&mut self) -> ParseResult<String> {
        match self.token_stream.current_kind() {
            // Direct version number: @2.1.0
            TokenKind::FloatLiteral(version_str) => {
                let version = version_str.clone();
                self.token_stream.advance();
                Ok(version)
            }
            // Version keyword: @version "2.1.0"
            TokenKind::Version => {
                self.token_stream.advance(); // consume 'version'
                let version_string = self.parse_string_literal()?;
                Ok(version_string)
            }
            // Integer version: @1
            TokenKind::IntegerLiteral(version_num) => {
                let version = version_num.to_string();
                self.token_stream.advance();
                Ok(version)
            }
            // String version: @"2.1.0"
            TokenKind::StringLiteral(version_str) => {
                let version = version_str.clone();
                self.token_stream.advance();
                Ok(version)
            }
            _ => Err(ParseError::unexpected_token(
                vec![
                    TokenKind::Version, 
                    TokenKind::StringLiteral("version".to_string()),
                    TokenKind::FloatLiteral("version".to_string()),
                    TokenKind::IntegerLiteral(1)
                ],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse trait implementation list: Repository<User>, EventSourced<AuthEvent>
    fn parse_trait_implementation_list(&mut self) -> ParseResult<Vec<prism_common::symbol::Symbol>> {
        let mut traits = Vec::new();
        
        loop {
            // Parse trait name
            let trait_name = self.token_stream.expect_identifier()?;
            traits.push(prism_common::symbol::Symbol::intern(&trait_name));
            
            // Parse optional generic parameters: <User>
            if self.token_stream.consume(TokenKind::LeftAngleBracket) {
                // Skip generic parameters for now - just consume until >
                let mut depth = 1;
                while depth > 0 && !self.token_stream.is_at_end() {
                    match self.token_stream.current_kind() {
                        TokenKind::LeftAngleBracket => depth += 1,
                        TokenKind::RightAngleBracket => depth -= 1,
                        _ => {}
                    }
                    self.token_stream.advance();
                }
            }
            
            // Check for more traits
            if self.token_stream.consume(TokenKind::Comma) {
                continue;
            } else {
                break;
            }
        }
        
        Ok(traits)
    }

    /// Parse a section with capability requirements
    fn parse_section_with_capabilities(&mut self) -> ParseResult<AstNode<SectionDecl>> {
        let start_span = self.token_stream.current_span();
        self.token_stream.expect(TokenKind::Section)?;
        
        // Parse section kind
        let kind = self.parse_section_kind()?;
        
        // Parse optional section name (for custom sections)
        let name = if matches!(&kind, SectionKind::Custom(_)) {
            if let SectionKind::Custom(n) = &kind {
                Some(n.clone())
            } else {
                None
            }
        } else {
            None
        };
        
        // Parse optional section purpose/description
        let purpose = if matches!(self.token_stream.current_kind(), TokenKind::StringLiteral(_)) {
            Some(self.parse_string_literal()?)
        } else {
            None
        };
        
        // Parse capability requirements: requires capability("high_performance")
        let requirements = if self.token_stream.check(TokenKind::Requires) {
            self.parse_capability_requirements()?
        } else {
            Vec::new()
        };
        
        // Parse section attributes
        let attributes = self.parse_section_attributes()?;
        
        // Parse section body
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut items = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            items.push(self.stmt_parser.parse_statement()?);
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(&start_span, &end_span).unwrap_or(start_span);
        
        let section_decl = SectionDecl {
            kind,
            name,
            purpose,
            items,
            visibility: Visibility::Public, // Default for now
            requirements,
            attributes,
            ai_context: None, // Could be enhanced later
            performance_hints: Vec::new(),
            security_notes: Vec::new(),
        };
        
        Ok(self.coordinator.create_node(section_decl, span))
    }

    /// Parse section kind with all PLD-002 section types
    fn parse_section_kind(&mut self) -> ParseResult<SectionKind> {
        let kind = match self.token_stream.current_kind() {
            TokenKind::Config => {
                self.token_stream.advance();
                SectionKind::Config
            }
            TokenKind::Types => {
                self.token_stream.advance();
                SectionKind::Types
            }
            TokenKind::Errors => {
                self.token_stream.advance();
                SectionKind::Errors
            }
            TokenKind::Internal => {
                self.token_stream.advance();
                SectionKind::Internal
            }
            TokenKind::Interface => {
                self.token_stream.advance();
                SectionKind::Interface
            }
            TokenKind::Performance => {
                self.token_stream.advance();
                SectionKind::Performance
            }
            TokenKind::Events => {
                self.token_stream.advance();
                SectionKind::Events
            }
            TokenKind::Lifecycle => {
                self.token_stream.advance();
                SectionKind::Lifecycle
            }
            TokenKind::Tests => {
                self.token_stream.advance();
                SectionKind::Tests
            }
            TokenKind::Examples => {
                self.token_stream.advance();
                SectionKind::Examples
            }
            TokenKind::StateMachine => {
                self.token_stream.advance();
                SectionKind::StateMachine
            }
            TokenKind::Operations => {
                self.token_stream.advance();
                SectionKind::Operations
            }
            TokenKind::Validation => {
                self.token_stream.advance();
                SectionKind::Validation
            }
            TokenKind::Migration => {
                self.token_stream.advance();
                SectionKind::Migration
            }
            TokenKind::Documentation => {
                self.token_stream.advance();
                SectionKind::Documentation
            }
            TokenKind::Identifier(name) => {
                let custom_name = name.clone();
                self.token_stream.advance();
                SectionKind::Custom(custom_name)
            }
            _ => return Err(ParseError::unexpected_token(
                vec![
                    TokenKind::Config, TokenKind::Types, TokenKind::Errors,
                    TokenKind::Internal, TokenKind::Interface, TokenKind::Performance,
                    TokenKind::Events, TokenKind::Lifecycle, TokenKind::Tests,
                    TokenKind::Examples, TokenKind::StateMachine, TokenKind::Operations,
                    TokenKind::Validation, TokenKind::Migration, TokenKind::Documentation,
                    TokenKind::Identifier("custom".to_string())
                ],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        };
        
        Ok(kind)
    }

    /// Parse module attributes/annotations
    fn parse_module_attributes(&mut self) -> ParseResult<Vec<prism_ast::Attribute>> {
        let mut attributes = Vec::new();
        
        while self.token_stream.check(TokenKind::At) {
            self.token_stream.advance(); // consume '@'
            
            let name = self.token_stream.expect_identifier()?;
            let value = if matches!(self.token_stream.current_kind(), TokenKind::StringLiteral(_)) {
                Some(self.parse_string_literal()?)
            } else {
                None
            };
            
            attributes.push(prism_ast::Attribute {
                name: prism_common::symbol::Symbol::intern(&name),
                value: value.map(|v| prism_ast::AttributeValue::String(v)),
                span: self.token_stream.current_span(),
            });
        }
        
        Ok(attributes)
    }

    /// Parse AI context block
    fn parse_ai_context(&mut self) -> ParseResult<AiContext> {
        self.token_stream.expect(TokenKind::At)?;
        
        let context_type = self.token_stream.expect_identifier()?;
        if context_type != "aiContext" {
            return Err(ParseError::unexpected_token(
                vec![TokenKind::Identifier("aiContext".to_string())],
                TokenKind::Identifier(context_type),
                self.token_stream.current_span(),
            ));
        }
        
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut purpose = String::new();
        let mut compliance = Vec::new();
        let mut critical_paths = Vec::new();
        let mut error_handling = None;
        let mut performance_notes = Vec::new();
        let mut security_notes = Vec::new();
        let mut ai_hints = Vec::new();
        let mut business_context = Vec::new();
        let mut architecture_patterns = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let field_name = self.token_stream.expect_identifier()?;
            self.token_stream.expect(TokenKind::Colon)?;
            
            match field_name.as_str() {
                "purpose" => {
                    purpose = self.parse_string_literal()?;
                }
                "compliance" => {
                    compliance = self.parse_string_array()?;
                }
                "criticalPaths" => {
                    critical_paths = self.parse_critical_paths()?;
                }
                "errorHandling" => {
                    error_handling = Some(self.parse_string_literal()?);
                }
                "performance" => {
                    performance_notes = self.parse_string_array()?;
                }
                "security" => {
                    security_notes = self.parse_string_array()?;
                }
                "aiHints" => {
                    ai_hints = self.parse_ai_hints()?;
                }
                "business" => {
                    business_context = self.parse_string_array()?;
                }
                "architecture" => {
                    architecture_patterns = self.parse_string_array()?;
                }
                _ => {
                    // Skip unknown fields
                    self.skip_value()?;
                }
            }
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        Ok(AiContext {
            purpose,
            compliance,
            critical_paths,
            error_handling,
            performance_notes,
            security_notes,
            ai_hints,
            business_context,
            architecture_patterns,
        })
    }

    /// Parse dependency injection configuration
    fn parse_injection_config(&mut self) -> ParseResult<InjectionConfig> {
        self.token_stream.advance(); // consume 'inject'
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut bindings = Vec::new();
        let mut scope = InjectionScope::Module;
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let binding_name = self.token_stream.expect_identifier()?;
            self.token_stream.expect(TokenKind::Colon)?;
            let binding_type = self.type_parser.parse_type()?;
            
            let is_singleton = if self.token_stream.check(TokenKind::At) {
                self.token_stream.advance();
                let annotation = self.token_stream.expect_identifier()?;
                annotation == "singleton"
            } else {
                false
            };
            
            bindings.push(InjectionBinding {
                name: prism_common::symbol::Symbol::intern(&binding_name),
                binding_type,
                is_singleton,
            });
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        Ok(InjectionConfig { bindings, scope })
    }

    /// Parse lifecycle hook
    fn parse_lifecycle_hook(&mut self) -> ParseResult<LifecycleHook> {
        self.token_stream.advance(); // consume 'on'
        
        let event_name = self.token_stream.expect_identifier()?;
        let event = match event_name.as_str() {
            "load" => LifecycleEvent::Load,
            "unload" => LifecycleEvent::Unload,
            "initialize" => LifecycleEvent::Initialize,
            "shutdown" => LifecycleEvent::Shutdown,
            "hotReload" => LifecycleEvent::HotReload,
            "dependencyResolved" => LifecycleEvent::DependencyResolved,
            _ => return Err(ParseError::unexpected_token(
                vec![TokenKind::Identifier("lifecycle_event".to_string())],
                TokenKind::Identifier(event_name),
                self.token_stream.current_span(),
            )),
        };
        
        let body = Box::new(self.stmt_parser.parse_statement()?);
        
        Ok(LifecycleHook {
            event,
            body,
            priority: None,
        })
    }

    /// Parse event declaration
    fn parse_event(&mut self) -> ParseResult<EventDecl> {
        self.token_stream.advance(); // consume 'event'
        
        let name = self.token_stream.expect_identifier()?;
        let name_symbol = prism_common::symbol::Symbol::intern(&name);
        
        self.token_stream.expect(TokenKind::LeftParen)?;
        
        let mut parameters = Vec::new();
        while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
            let param_name = self.token_stream.expect_identifier()?;
            self.token_stream.expect(TokenKind::Colon)?;
            let param_type = self.type_parser.parse_type()?;
            
            parameters.push(prism_ast::Parameter {
                name: prism_common::symbol::Symbol::intern(&param_name),
                type_annotation: Some(param_type),
                default_value: None,
                is_mutable: false,
            });
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightParen)?;
        
        Ok(EventDecl {
            name: name_symbol,
            parameters,
            description: None,
            priority: None,
        })
    }

    /// Parse section requirements
    fn parse_section_requirements(&mut self) -> ParseResult<Vec<SectionRequirement>> {
        self.token_stream.expect(TokenKind::Requires)?;
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut requirements = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let kind = self.parse_requirement_kind()?;
            let name = self.token_stream.expect_identifier()?;
            let name_symbol = prism_common::symbol::Symbol::intern(&name);
            
            let mut description = None;
            if matches!(self.token_stream.current_kind(), TokenKind::StringLiteral(_)) {
                description = Some(self.parse_string_literal()?);
            }
            
            requirements.push(SectionRequirement {
                kind,
                name: name_symbol,
                description,
            });
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        Ok(requirements)
    }

    /// Parse requirement kind
    fn parse_requirement_kind(&mut self) -> ParseResult<RequirementKind> {
        match self.token_stream.current_kind() {
            TokenKind::Requires => {
                self.token_stream.advance();
                RequirementKind::Required
            }
            TokenKind::Optional => {
                self.token_stream.advance();
                RequirementKind::Optional
            }
            TokenKind::Must => {
                self.token_stream.advance();
                RequirementKind::Must
            }
            TokenKind::Should => {
                self.token_stream.advance();
                RequirementKind::Should
            }
            TokenKind::May => {
                self.token_stream.advance();
                RequirementKind::May
            }
            TokenKind::Identifier(name) => {
                let custom_name = name.clone();
                self.token_stream.advance();
                RequirementKind::Custom(custom_name)
            }
            _ => return Err(ParseError::unexpected_token(
                vec![
                    TokenKind::Requires, TokenKind::Optional, TokenKind::Must,
                    TokenKind::Should, TokenKind::May, TokenKind::Identifier("custom".to_string())
                ],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse section attributes
    fn parse_section_attributes(&mut self) -> ParseResult<Vec<prism_ast::Attribute>> {
        let mut attributes = Vec::new();
        
        while self.token_stream.check(TokenKind::At) {
            self.token_stream.advance(); // consume '@'
            
            let name = self.token_stream.expect_identifier()?;
            let value = if matches!(self.token_stream.current_kind(), TokenKind::StringLiteral(_)) {
                Some(self.parse_string_literal()?)
            } else {
                None
            };
            
            attributes.push(prism_ast::Attribute {
                name: prism_common::symbol::Symbol::intern(&name),
                value: value.map(|v| prism_ast::AttributeValue::String(v)),
                span: self.token_stream.current_span(),
            });
        }
        
        Ok(attributes)
    }

    /// Parse composition trait implementation
    fn parse_composition_trait(&mut self) -> ParseResult<CompositionTrait> {
        self.token_stream.expect(TokenKind::Trait)?;
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut traits = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let trait_name = self.token_stream.expect_identifier()?;
            self.token_stream.expect(TokenKind::Colon)?;
            let trait_type = self.type_parser.parse_type()?;
            
            let is_optional = if self.token_stream.check(TokenKind::At) {
                self.token_stream.advance();
                let annotation = self.token_stream.expect_identifier()?;
                annotation == "optional"
            } else {
                false
            };
            
            traits.push(CompositionTrait {
                trait_name: prism_common::symbol::Symbol::intern(&trait_name),
                trait_type,
                is_optional,
            });
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        Ok(CompositionTrait {
            trait_name: traits[0].trait_name, // Assuming only one trait for now
            trait_type: traits[0].trait_type,
            is_optional: traits[0].is_optional,
        })
    }

    /// Parse import statement
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
                
                let import_item = Item::Import(import_decl);
                Ok(self.coordinator.create_node(import_item, span))
            }
            _ => return Err(ParseError::unexpected_token(
                vec![TokenKind::Import, TokenKind::Use],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse export statement
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
                
                let export_item = Item::Export(export_decl);
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
    fn parse_module_metadata(&self, attributes: &[prism_ast::Attribute]) -> (Option<String>, Option<String>, Option<StabilityLevel>, Vec<ModuleDependency>) {
        let mut capability = None;
        let mut description = None;
        let mut stability = None;
        let mut dependencies = Vec::new();
        
        for attr in attributes {
            let name = attr.name.as_str();
            if let Some(prism_ast::AttributeValue::String(value)) = &attr.value {
                match name {
                    "capability" => capability = Some(value.clone()),
                    "description" => description = Some(value.clone()),
                    "stability" => {
                        stability = match value.as_str() {
                            "Experimental" => Some(StabilityLevel::Experimental),
                            "Alpha" => Some(StabilityLevel::Alpha),
                            "Beta" => Some(StabilityLevel::Beta),
                            "Stable" => Some(StabilityLevel::Stable),
                            "Deprecated" => Some(StabilityLevel::Deprecated),
                            _ => None,
                        };
                    }
                    _ => {}
                }
            }
        }
        
        (capability, description, stability, dependencies)
    }

    /// Parse capability requirements: requires capability("high_performance"), permission("admin")
    fn parse_capability_requirements(&mut self) -> ParseResult<Vec<SectionRequirement>> {
        self.token_stream.expect(TokenKind::Requires)?;
        
        let mut requirements = Vec::new();
        
        loop {
            // Parse requirement type
            let req_kind = if self.token_stream.check(TokenKind::Capability) {
                self.token_stream.advance(); // consume 'capability'
                RequirementKind::Capability
            } else if self.token_stream.check_identifier_with_value("permission") {
                self.token_stream.advance(); // consume 'permission'
                RequirementKind::Permission
            } else if self.token_stream.check_identifier_with_value("security") {
                self.token_stream.advance(); // consume 'security'
                RequirementKind::SecurityLevel
            } else {
                // Default to capability
                RequirementKind::Capability
            };
            
            // Parse requirement value: ("high_performance")
            self.token_stream.expect(TokenKind::LeftParen)?;
            let requirement_value = self.parse_string_literal()?;
            self.token_stream.expect(TokenKind::RightParen)?;
            
            requirements.push(SectionRequirement {
                kind: req_kind,
                value: requirement_value,
                description: None,
            });
            
            // Check for more requirements
            if self.token_stream.consume(TokenKind::Comma) {
                continue;
            } else {
                break;
            }
        }
        
        Ok(requirements)
    }

    /// Extract module metadata from attributes
    fn extract_module_metadata(&self, attributes: &[prism_ast::Attribute]) -> (Option<String>, Option<String>, Option<StabilityLevel>, Vec<ModuleDependency>) {
        let mut capability = None;
        let mut description = None;
        let mut stability = None;
        let mut dependencies = Vec::new();
        
        for attr in attributes {
            let name = attr.name.as_str();
            if let Some(prism_ast::AttributeValue::String(value)) = &attr.value {
                match name {
                    "capability" => capability = Some(value.clone()),
                    "description" => description = Some(value.clone()),
                    "stability" => {
                        stability = match value.as_str() {
                            "Experimental" => Some(StabilityLevel::Experimental),
                            "Alpha" => Some(StabilityLevel::Alpha),
                            "Beta" => Some(StabilityLevel::Beta),
                            "Stable" => Some(StabilityLevel::Stable),
                            "Deprecated" => Some(StabilityLevel::Deprecated),
                            _ => None,
                        };
                    }
                    _ => {}
                }
            }
        }
        
        (capability, description, stability, dependencies)
    }

    /// Check if current token is an identifier with specific value
    fn check_identifier_with_value(&self, value: &str) -> bool {
        match self.token_stream.current_kind() {
            TokenKind::Identifier(id) => id == value,
            _ => false,
        }
    }

    /// Combine two spans into a single span
    fn combine_spans(&self, start: Span, end: Span) -> Span {
        Span::combine(&start, &end).unwrap_or(start)
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

    /// Check if next tokens indicate AI context
    fn peek_ai_context(&mut self) -> bool {
        if let TokenKind::Identifier(name) = self.token_stream.peek_kind() {
            name == "aiContext"
        } else {
            false
        }
    }

    /// Parse string array
    fn parse_string_array(&mut self) -> ParseResult<Vec<String>> {
        self.token_stream.expect(TokenKind::LeftBracket)?;
        let mut strings = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBracket) && !self.token_stream.is_at_end() {
            strings.push(self.parse_string_literal()?);
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBracket)?;
        Ok(strings)
    }

    /// Parse critical paths
    fn parse_critical_paths(&mut self) -> ParseResult<Vec<CriticalPath>> {
        self.token_stream.expect(TokenKind::LeftBracket)?;
        let mut paths = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBracket) && !self.token_stream.is_at_end() {
            self.token_stream.expect(TokenKind::LeftBrace)?;
            
            let mut name = String::new();
            let mut description = String::new();
            let mut requirements = Vec::new();
            let mut sla = None;
            
            while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
                let field_name = self.token_stream.expect_identifier()?;
                self.token_stream.expect(TokenKind::Colon)?;
                
                match field_name.as_str() {
                    "name" => name = self.parse_string_literal()?,
                    "description" => description = self.parse_string_literal()?,
                    "requirements" => requirements = self.parse_string_array()?,
                    "sla" => sla = Some(self.parse_performance_sla()?),
                    _ => self.skip_value()?,
                }
                
                if !self.token_stream.consume(TokenKind::Comma) {
                    break;
                }
            }
            
            self.token_stream.expect(TokenKind::RightBrace)?;
            
            paths.push(CriticalPath {
                name,
                description,
                requirements,
                sla,
            });
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBracket)?;
        Ok(paths)
    }

    /// Parse performance SLA
    fn parse_performance_sla(&mut self) -> ParseResult<PerformanceSla> {
        self.token_stream.expect(TokenKind::LeftBrace)?;
        
        let mut max_response_time = String::new();
        let mut max_throughput = None;
        let mut availability = None;
        
        while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
            let field_name = self.token_stream.expect_identifier()?;
            self.token_stream.expect(TokenKind::Colon)?;
            
            match field_name.as_str() {
                "responseTime" => max_response_time = self.parse_string_literal()?,
                "throughput" => max_throughput = Some(self.parse_string_literal()?),
                "availability" => availability = Some(self.parse_string_literal()?),
                _ => self.skip_value()?,
            }
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBrace)?;
        
        Ok(PerformanceSla {
            max_response_time,
            max_throughput,
            availability,
        })
    }

    /// Parse AI hints
    fn parse_ai_hints(&mut self) -> ParseResult<Vec<AiHint>> {
        self.token_stream.expect(TokenKind::LeftBracket)?;
        let mut hints = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightBracket) && !self.token_stream.is_at_end() {
            self.token_stream.expect(TokenKind::LeftBrace)?;
            
            let mut category = AiHintCategory::Business;
            let mut content = String::new();
            
            while !self.token_stream.check(TokenKind::RightBrace) && !self.token_stream.is_at_end() {
                let field_name = self.token_stream.expect_identifier()?;
                self.token_stream.expect(TokenKind::Colon)?;
                
                match field_name.as_str() {
                    "category" => {
                        let cat_name = self.parse_string_literal()?;
                        category = match cat_name.as_str() {
                            "performance" => AiHintCategory::Performance,
                            "security" => AiHintCategory::Security,
                            "testing" => AiHintCategory::Testing,
                            "business" => AiHintCategory::Business,
                            "architecture" => AiHintCategory::Architecture,
                            "maintenance" => AiHintCategory::Maintenance,
                            "debugging" => AiHintCategory::Debugging,
                            _ => AiHintCategory::Business,
                        };
                    }
                    "content" => content = self.parse_string_literal()?,
                    _ => self.skip_value()?,
                }
                
                if !self.token_stream.consume(TokenKind::Comma) {
                    break;
                }
            }
            
            self.token_stream.expect(TokenKind::RightBrace)?;
            
            hints.push(AiHint { category, content });
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBracket)?;
        Ok(hints)
    }

    /// Skip unknown value in parsing
    fn skip_value(&mut self) -> ParseResult<()> {
        match self.token_stream.current_kind() {
            TokenKind::LeftBrace => {
                self.token_stream.advance();
                let mut depth = 1;
                while depth > 0 && !self.token_stream.is_at_end() {
                    match self.token_stream.current_kind() {
                        TokenKind::LeftBrace => depth += 1,
                        TokenKind::RightBrace => depth -= 1,
                        _ => {}
                    }
                    self.token_stream.advance();
                }
            }
            TokenKind::LeftBracket => {
                self.token_stream.advance();
                let mut depth = 1;
                while depth > 0 && !self.token_stream.is_at_end() {
                    match self.token_stream.current_kind() {
                        TokenKind::LeftBracket => depth += 1,
                        TokenKind::RightBracket => depth -= 1,
                        _ => {}
                    }
                    self.token_stream.advance();
                }
            }
            _ => {
                self.token_stream.advance();
            }
        }
        Ok(())
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

    /// Integrate with prism-cohesion for real-time analysis
    pub fn analyze_module_cohesion(&self, module_decl: &ModuleDecl) -> ParseResult<Option<CohesionMetadata>> {
        use prism_cohesion::{CohesionSystem, CohesionConfig, AnalysisDepth};
        
        // Create cohesion system for real-time analysis
        let config = CohesionConfig {
            analysis_depth: AnalysisDepth::Quick, // Fast analysis during parsing
            enable_violation_detection: true,
            enable_suggestions: true,
            enable_ai_insights: false, // Disable for performance during parsing
            metric_weights: prism_cohesion::MetricWeights::default(),
            violation_thresholds: prism_cohesion::ViolationThresholds::default(),
        };
        
        let mut cohesion_system = CohesionSystem::with_config(config);
        
        // Create a temporary AST node for the module
        let temp_span = prism_common::span::Span::new(
            prism_common::span::Position::new(0, 0),
            prism_common::span::Position::new(0, 0),
            prism_common::SourceId::new(0)
        );
        let module_item = self.coordinator.create_node(
            prism_ast::Item::Module(module_decl.clone()),
            temp_span,
        );
        
        // Analyze the module in real-time
        match cohesion_system.analyze_module(&module_item) {
            Ok(analysis) => {
                Ok(Some(CohesionMetadata {
                    overall_score: analysis.overall_score,
                    type_cohesion: analysis.metrics.type_cohesion,
                    data_flow_cohesion: analysis.metrics.data_flow_cohesion,
                    semantic_cohesion: analysis.metrics.semantic_cohesion,
                    dependency_cohesion: analysis.metrics.dependency_cohesion,
                    strengths: analysis.metrics.analysis.strengths,
                    suggestions: analysis.improvement_suggestions.into_iter().map(|s| s.description).collect(),
                    analyzed_at: Some(chrono::Utc::now().to_rfc3339()),
                    trend: Some(prism_ast::CohesionTrend::Unknown), // No historical data during parsing
                }))
            }
            Err(err) => {
                // Log error but don't fail parsing
                tracing::warn!("Cohesion analysis failed during parsing: {}", err);
                
                // Return minimal metadata
                Ok(Some(CohesionMetadata {
                    overall_score: 50.0, // Neutral score when analysis fails
                    type_cohesion: 50.0,
                    data_flow_cohesion: 50.0,
                    semantic_cohesion: 50.0,
                    dependency_cohesion: 50.0,
                    strengths: vec!["Module structure is parseable".to_string()],
                    suggestions: vec!["Unable to analyze cohesion - check module structure".to_string()],
                    analyzed_at: Some(chrono::Utc::now().to_rfc3339()),
                    trend: Some(prism_ast::CohesionTrend::Unknown),
                }))
            }
        }
    }

    /// Generate AI-comprehensible module metadata
    pub fn generate_module_ai_metadata(&self, module_decl: &ModuleDecl) -> ModuleAiContext {
        ModuleAiContext {
            purpose: module_decl.capability.clone().unwrap_or_else(|| {
                format!("Module {} provides core functionality", module_decl.name)
            }),
            compliance: Vec::new(), // TODO: Extract from attributes
            critical_paths: Vec::new(), // TODO: Analyze section contents
            error_handling: module_decl.description.clone(),
            performance_notes: Vec::new(), // TODO: Extract from performance section
            security_notes: Vec::new(), // TODO: Extract from attributes and sections
            ai_hints: Vec::new(), // TODO: Generate based on module structure
            business_context: Vec::new(), // TODO: Extract business context
            architecture_patterns: Vec::new(), // TODO: Detect patterns
        }
    }

    /// Convert import to dependency
    fn convert_import_to_dependency(&self, import_node: AstNode<Item>) -> ParseResult<ModuleDependency> {
        let import_decl = match import_node.item {
            Item::Import(import_decl) => import_decl,
            _ => return Err(ParseError::unexpected_item_type(
                "ImportDecl",
                import_node.item.kind(),
                import_node.span,
            )),
        };

        let path = import_decl.path.clone();
        let alias = import_decl.alias.map(|a| a.as_str().to_string());
        let from_module = if alias.is_some() {
            Some(path.split('.').next().unwrap().to_string())
        } else {
            None
        };

        let dependency_item = match alias {
            Some(alias) => DependencyItem::Named(DependencyItems::Named(alias, path)),
            None => DependencyItem::Path(DependencyItems::Path(path)),
        };

        Ok(ModuleDependency {
            name: prism_common::symbol::Symbol::intern(&path.split('.').last().unwrap().to_string()),
            items: vec![dependency_item],
            is_optional: false,
            is_singleton: false,
        })
    }

    /// Parse trait list (for implements)
    fn parse_trait_list(&mut self) -> ParseResult<Vec<String>> {
        let mut traits = Vec::new();
        self.token_stream.expect(TokenKind::LeftBracket)?;
        
        while !self.token_stream.check(TokenKind::RightBracket) && !self.token_stream.is_at_end() {
            let trait_name = self.token_stream.expect_identifier()?;
            traits.push(trait_name);
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        self.token_stream.expect(TokenKind::RightBracket)?;
        Ok(traits)
    }

    /// Parse version
    fn parse_version(&mut self) -> ParseResult<String> {
        self.token_stream.expect(TokenKind::At)?;
        self.token_stream.expect(TokenKind::Version)?;
        self.token_stream.expect(TokenKind::StringLiteral(self.token_stream.expect_identifier()?))?;
        Ok(self.token_stream.previous_token().unwrap().value.clone())
    }

    /// Extract comprehensive business context from module declaration and content
    pub fn extract_business_context(&self, module_decl: &ModuleDecl) -> ParseResult<BusinessContext> {
        let mut business_context = BusinessContext {
            primary_capability: module_decl.capability.clone()
                .unwrap_or_else(|| module_decl.name.to_string()),
            domain: self.extract_domain_from_module(module_decl),
            responsibility: module_decl.description.clone(),
            business_rules: self.extract_business_rules_from_sections(&module_decl.sections),
            entities: self.extract_business_entities_from_sections(&module_decl.sections),
            processes: self.extract_business_processes_from_sections(&module_decl.sections),
        };

        // Enhance with AI context if available
        if let Some(ai_context) = &module_decl.ai_context {
            business_context.enhance_with_ai_context(ai_context);
        }

        // Analyze naming patterns for additional insights
        business_context.enhance_with_naming_analysis(&module_decl.name.to_string());

        Ok(business_context)
    }

    /// Extract domain information from module structure and naming
    fn extract_domain_from_module(&self, module_decl: &ModuleDecl) -> Option<String> {
        // Extract domain from module name patterns
        let module_name = module_decl.name.to_string();
        
        // Common domain patterns
        let domain_patterns = vec![
            ("User", "UserManagement"),
            ("Payment", "PaymentProcessing"),
            ("Order", "OrderManagement"),
            ("Inventory", "InventoryManagement"),
            ("Auth", "Authentication"),
            ("Security", "SecurityManagement"),
            ("Notification", "NotificationService"),
            ("Analytics", "DataAnalytics"),
            ("Report", "ReportingService"),
            ("Audit", "AuditingService"),
        ];

        for (pattern, domain) in domain_patterns {
            if module_name.contains(pattern) {
                return Some(domain.to_string());
            }
        }

        // Check AI context for domain hints
        if let Some(ai_context) = &module_decl.ai_context {
            for business_hint in &ai_context.business_context {
                if business_hint.to_lowercase().contains("domain:") {
                    let domain = business_hint.split(':').nth(1)
                        .map(|s| s.trim().to_string());
                    if domain.is_some() {
                        return domain;
                    }
                }
            }
        }

        // Extract from capability if available
        if let Some(capability) = &module_decl.capability {
            Some(format!("{}Domain", capability))
        } else {
            None
        }
    }

    /// Extract business rules from module sections
    fn extract_business_rules_from_sections(&self, sections: &[AstNode<SectionDecl>]) -> Vec<String> {
        let mut business_rules = Vec::new();

        for section in sections {
            match &section.kind.kind {
                SectionKind::Validation => {
                    // Extract validation rules as business rules
                    business_rules.extend(self.extract_validation_rules_from_section(section));
                }
                SectionKind::Operations => {
                    // Extract business logic patterns
                    business_rules.extend(self.extract_operation_rules_from_section(section));
                }
                SectionKind::Interface => {
                    // Extract API contract rules
                    business_rules.extend(self.extract_interface_rules_from_section(section));
                }
                _ => {}
            }
        }

        business_rules
    }

    /// Extract business entities from type definitions in sections
    fn extract_business_entities_from_sections(&self, sections: &[AstNode<SectionDecl>]) -> Vec<String> {
        let mut entities = Vec::new();

        for section in sections {
            match &section.kind.kind {
                SectionKind::Types => {
                    entities.extend(self.extract_entities_from_types_section(section));
                }
                SectionKind::Interface => {
                    entities.extend(self.extract_entities_from_interface_section(section));
                }
                _ => {}
            }
        }

        // Remove duplicates and sort
        entities.sort();
        entities.dedup();
        entities
    }

    /// Extract business processes from operations and lifecycle sections
    fn extract_business_processes_from_sections(&self, sections: &[AstNode<SectionDecl>]) -> Vec<String> {
        let mut processes = Vec::new();

        for section in sections {
            match &section.kind.kind {
                SectionKind::Operations => {
                    processes.extend(self.extract_processes_from_operations_section(section));
                }
                SectionKind::Lifecycle => {
                    processes.extend(self.extract_processes_from_lifecycle_section(section));
                }
                SectionKind::Events => {
                    processes.extend(self.extract_processes_from_events_section(section));
                }
                _ => {}
            }
        }

        // Remove duplicates and sort
        processes.sort();
        processes.dedup();
        processes
    }

    /// Extract validation rules from validation section
    fn extract_validation_rules_from_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut rules = Vec::new();

        // Analyze section items for validation patterns
        for item in &section.kind.items {
            if let Some(rule) = self.extract_business_rule_from_statement(item) {
                rules.push(rule);
            }
        }

        rules
    }

    /// Extract operation rules from operations section
    fn extract_operation_rules_from_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut rules = Vec::new();

        // Look for function definitions with business logic patterns
        for item in &section.kind.items {
            if let Some(rule) = self.extract_operation_rule_from_statement(item) {
                rules.push(rule);
            }
        }

        rules
    }

    /// Extract interface rules from interface section
    fn extract_interface_rules_from_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut rules = Vec::new();

        // Look for API contracts and constraints
        for item in &section.kind.items {
            if let Some(rule) = self.extract_interface_rule_from_statement(item) {
                rules.push(rule);
            }
        }

        rules
    }

    /// Extract entities from types section
    fn extract_entities_from_types_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut entities = Vec::new();

        for item in &section.kind.items {
            if let Some(entity) = self.extract_entity_from_statement(item) {
                entities.push(entity);
            }
        }

        entities
    }

    /// Extract entities from interface section
    fn extract_entities_from_interface_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut entities = Vec::new();

        // Look for parameter types and return types that represent entities
        for item in &section.kind.items {
            entities.extend(self.extract_entities_from_function_signature(item));
        }

        entities
    }

    /// Extract processes from operations section
    fn extract_processes_from_operations_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut processes = Vec::new();

        for item in &section.kind.items {
            if let Some(process) = self.extract_process_from_statement(item) {
                processes.push(process);
            }
        }

        processes
    }

    /// Extract processes from lifecycle section
    fn extract_processes_from_lifecycle_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut processes = Vec::new();

        // Lifecycle sections define state transitions and processes
        for item in &section.kind.items {
            if let Some(process) = self.extract_lifecycle_process_from_statement(item) {
                processes.push(process);
            }
        }

        processes
    }

    /// Extract processes from events section
    fn extract_processes_from_events_section(&self, section: &AstNode<SectionDecl>) -> Vec<String> {
        let mut processes = Vec::new();

        // Events often represent business processes or triggers
        for item in &section.kind.items {
            if let Some(process) = self.extract_event_process_from_statement(item) {
                processes.push(process);
            }
        }

        processes
    }

    // Helper methods for extracting specific patterns from statements

    fn extract_business_rule_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        // This would analyze the statement AST to extract business rules
        // For now, return a placeholder based on statement type
        match &stmt.kind {
            Stmt::Function(func) => {
                if func.name.to_string().contains("validate") {
                    Some(format!("Validation rule: {}", func.name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_operation_rule_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        match &stmt.kind {
            Stmt::Function(func) => {
                // Look for business operation patterns
                let name = func.name.to_string();
                if name.contains("create") || name.contains("update") || name.contains("delete") {
                    Some(format!("Business operation: {}", name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_interface_rule_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        match &stmt.kind {
            Stmt::Function(func) => {
                // Look for API contract patterns
                if func.visibility == Visibility::Public {
                    Some(format!("API contract: {}", func.name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_entity_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        match &stmt.kind {
            Stmt::Type(type_decl) => {
                // Type declarations often represent business entities
                Some(type_decl.name.to_string())
            }
            Stmt::Struct(struct_decl) => {
                Some(struct_decl.name.to_string())
            }
            _ => None,
        }
    }

    fn extract_entities_from_function_signature(&self, stmt: &AstNode<Stmt>) -> Vec<String> {
        let mut entities = Vec::new();

        match &stmt.kind {
            Stmt::Function(func) => {
                // Extract entity names from parameter types and return type
                for param in &func.parameters {
                    if let Some(entity) = self.extract_entity_from_type_name(&param.param_type) {
                        entities.push(entity);
                    }
                }
                
                if let Some(return_type) = &func.return_type {
                    if let Some(entity) = self.extract_entity_from_type_name(return_type) {
                        entities.push(entity);
                    }
                }
            }
            _ => {}
        }

        entities
    }

    fn extract_process_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        match &stmt.kind {
            Stmt::Function(func) => {
                let name = func.name.to_string();
                // Look for process-indicating patterns
                if name.contains("process") || name.contains("handle") || name.contains("execute") {
                    Some(format!("Business process: {}", name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_lifecycle_process_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        match &stmt.kind {
            Stmt::Function(func) => {
                let name = func.name.to_string();
                if name.contains("init") || name.contains("start") || name.contains("stop") || name.contains("cleanup") {
                    Some(format!("Lifecycle process: {}", name))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_event_process_from_statement(&self, stmt: &AstNode<Stmt>) -> Option<String> {
        match &stmt.kind {
            Stmt::Event(event) => {
                Some(format!("Event process: {}", event.name))
            }
            _ => None,
        }
    }

    fn extract_entity_from_type_name(&self, type_expr: &Expr) -> Option<String> {
        // This would analyze the type expression to extract entity names
        // For now, return a placeholder
        match type_expr {
            Expr::Identifier(ident) => {
                let name = ident.name.to_string();
                // Filter out primitive types
                if ["string", "number", "boolean", "void"].contains(&name.as_str()) {
                    None
                } else {
                    Some(name)
                }
            }
            _ => None,
        }
    }
}

/// Business context information extracted during parsing
#[derive(Debug, Clone)]
pub struct BusinessContext {
    /// Primary business capability
    pub primary_capability: String,
    /// Business domain
    pub domain: Option<String>,
    /// Responsibility description
    pub responsibility: Option<String>,
    /// Business rules this module enforces
    pub business_rules: Vec<String>,
    /// Key business entities
    pub entities: Vec<String>,
    /// Business processes supported
    pub processes: Vec<String>,
}

impl BusinessContext {
    /// Enhance business context with AI context information
    pub fn enhance_with_ai_context(&mut self, ai_context: &AiContext) {
        // Extract additional business information from AI context
        self.business_rules.extend(ai_context.business_context.iter().cloned());
        
        // Look for entity and process hints in AI hints
        for hint in &ai_context.ai_hints {
            if hint.category == AiHintCategory::Business {
                if hint.content.to_lowercase().contains("entity:") {
                    if let Some(entity) = hint.content.split(':').nth(1) {
                        self.entities.push(entity.trim().to_string());
                    }
                }
                if hint.content.to_lowercase().contains("process:") {
                    if let Some(process) = hint.content.split(':').nth(1) {
                        self.processes.push(process.trim().to_string());
                    }
                }
            }
        }
    }

    /// Enhance business context with naming pattern analysis
    pub fn enhance_with_naming_analysis(&mut self, module_name: &str) {
        // Analyze module name for business insights
        let name_lower = module_name.to_lowercase();
        
        // Common business entity patterns
        let entity_patterns = vec![
            "user", "customer", "product", "order", "payment", "invoice",
            "account", "transaction", "inventory", "report", "audit",
        ];
        
        for pattern in entity_patterns {
            if name_lower.contains(pattern) {
                let entity_name = pattern.chars()
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>() + &pattern[1..])
                    .unwrap_or_else(|| pattern.to_string());
                
                if !self.entities.contains(&entity_name) {
                    self.entities.push(entity_name);
                }
            }
        }

        // Common business process patterns
        let process_patterns = vec![
            ("management", "Management Process"),
            ("service", "Service Process"),
            ("handler", "Event Handling Process"),
            ("processor", "Data Processing"),
            ("validator", "Validation Process"),
            ("analyzer", "Analysis Process"),
        ];

        for (pattern, process_name) in process_patterns {
            if name_lower.contains(pattern) {
                if !self.processes.contains(&process_name.to_string()) {
                    self.processes.push(process_name.to_string());
                }
            }
        }
    }
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
        let result = parser.module_parser.parse_section_with_capabilities();
        assert!(result.is_ok());
    }

    /// Test Phase 1 Smart Module System features
    #[test]
    fn test_phase1_smart_module_features() {
        let source = r#"
            @capability "User Management"
            @description "Handles user lifecycle operations"
            @stability "Stable"
            module UserAuth@2.1.0 implements Repository<User>, EventSourced<AuthEvent> {
                section config {
                    const MIN_PASSWORD_LENGTH = 8;
                }
                
                section performance requires capability("high_performance"), permission("admin") {
                    function optimized_hash() {}
                }
                
                section interface {
                    function login() -> Result<Session, Error>;
                }
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_module();
        assert!(result.is_ok(), "Should parse Phase 1 Smart Module features");
        
        // TODO: Add more specific assertions once we have a working parser
    }

    #[test]
    fn test_version_specification_parsing() {
        let source = r#"
            module TestModule@1.0.0 {
                section interface {}
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_module();
        assert!(result.is_ok(), "Should parse version specification");
    }

    #[test]
    fn test_trait_implementation_parsing() {
        let source = r#"
            module TestModule implements Repository<User>, Service {
                section interface {}
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_module();
        assert!(result.is_ok(), "Should parse trait implementations");
    }

    #[test]
    fn test_capability_requirements_parsing() {
        let source = r#"
            module TestModule {
                section performance requires capability("high_performance") {
                    function fast_operation() {}
                }
                
                section admin requires permission("admin"), security("level_5") {
                    function admin_operation() {}
                }
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.module_parser.parse_module();
        assert!(result.is_ok(), "Should parse capability requirements");
    }

    // Helper function to create test parser (placeholder)
    fn create_test_parser(source: &str) -> TestParser {
        // TODO: Implement proper test parser creation
        TestParser::new(source)
    }

    // Placeholder test parser struct
    struct TestParser {
        source: String,
    }

    impl TestParser {
        fn new(source: &str) -> Self {
            Self {
                source: source.to_string(),
            }
        }
    }
} 