//! Scope Builder - Construction and Initialization Utilities
//!
//! This module provides utilities for constructing and initializing scopes
//! from AST nodes, following PLT-004 specifications and integrating with
//! the existing AST infrastructure.
//!
//! **Conceptual Responsibility**: Scope construction and initialization
//! **What it does**: Scope building from AST, initialization utilities, configuration
//! **What it doesn't do**: Scope hierarchy management, symbol resolution, AST parsing

use crate::error::{CompilerError, CompilerResult};
use crate::scope::{
    ScopeTree, ScopeKind, ScopeMetadata, AIScopeContext, ScopeDocumentation,
    SectionType, BlockType, ControlFlowType, SecurityLevel,
};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_ast::{AstNode, Item, Stmt, Expr};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Configuration for the scope builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeBuilderConfig {
    /// Enable AI metadata generation during scope building
    pub enable_ai_metadata: bool,
    
    /// Enable documentation extraction from AST
    pub enable_documentation_extraction: bool,
    
    /// Enable effect and capability inference
    pub enable_effect_inference: bool,
    
    /// Enable cohesion analysis
    pub enable_cohesion_analysis: bool,
    
    /// Default security level for scopes
    pub default_security_level: SecurityLevel,
    
    /// Custom section type mappings
    pub custom_section_mappings: HashMap<String, SectionType>,
}

/// Scope builder for constructing scopes from AST nodes
#[derive(Debug)]
pub struct ScopeBuilder {
    /// Configuration
    config: ScopeBuilderConfig,
    
    /// Current scope context during building
    current_context: BuildContext,
    
    /// Mapping from AST nodes to created scopes
    node_to_scope_map: HashMap<NodeId, crate::scope::ScopeId>,
    
    /// Statistics about the building process
    build_stats: BuildStatistics,
}

/// Context maintained during scope building
#[derive(Debug, Clone)]
pub struct BuildContext {
    /// Current module being processed
    pub current_module: Option<String>,
    
    /// Current section being processed
    pub current_section: Option<SectionType>,
    
    /// Available capabilities in current context
    pub available_capabilities: Vec<String>,
    
    /// Current effects being tracked
    pub current_effects: Vec<String>,
    
    /// Nesting depth
    pub depth: usize,
    
    /// Business domain context
    pub business_domain: Option<String>,
}

/// Statistics collected during scope building
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildStatistics {
    /// Total scopes created
    pub scopes_created: usize,
    
    /// Scopes with AI metadata
    pub scopes_with_ai_metadata: usize,
    
    /// Scopes with documentation
    pub scopes_with_documentation: usize,
    
    /// Build time in milliseconds
    pub build_time_ms: u64,
    
    /// Errors encountered during building
    pub errors_encountered: usize,
    
    /// Warnings generated
    pub warnings_generated: usize,
}

impl Default for ScopeBuilderConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_documentation_extraction: true,
            enable_effect_inference: true,
            enable_cohesion_analysis: true,
            default_security_level: SecurityLevel::Normal,
            custom_section_mappings: HashMap::new(),
        }
    }
}

impl Default for BuildContext {
    fn default() -> Self {
        Self {
            current_module: None,
            current_section: None,
            available_capabilities: Vec::new(),
            current_effects: Vec::new(),
            depth: 0,
            business_domain: None,
        }
    }
}

impl ScopeBuilder {
    /// Create a new scope builder with default configuration
    pub fn new() -> Self {
        Self::with_config(ScopeBuilderConfig::default())
    }
    
    /// Create a new scope builder with custom configuration
    pub fn with_config(config: ScopeBuilderConfig) -> Self {
        Self {
            config,
            current_context: BuildContext::default(),
            node_to_scope_map: HashMap::new(),
            build_stats: BuildStatistics::default(),
        }
    }
    
    /// Build scopes from an AST program
    pub fn build_from_program(
        &mut self,
        program: &prism_ast::Program,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<crate::scope::ScopeId> {
        let start_time = std::time::Instant::now();
        
        // Create global scope
        let global_scope = scope_tree.create_global_scope()?;
        self.build_stats.scopes_created += 1;
        
        // Process top-level items
        for item in &program.items {
            self.build_from_item(item, global_scope, scope_tree)?;
        }
        
        // Update statistics
        self.build_stats.build_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(global_scope)
    }
    
    /// Build scopes from an AST item
    pub fn build_from_item(
        &mut self,
        item: &AstNode<Item>,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        match &item.kind {
            Item::Module(module_decl) => {
                self.build_module_scope(item, module_decl, parent_scope, scope_tree)
            }
            Item::Function(func_decl) => {
                self.build_function_scope(item, func_decl, parent_scope, scope_tree)
            }
            Item::Type(type_decl) => {
                self.build_type_scope(item, type_decl, parent_scope, scope_tree)
            }
            _ => {
                // For other items, we might not create a scope
                Ok(None)
            }
        }
    }
    
    /// Build a module scope
    fn build_module_scope(
        &mut self,
        item: &AstNode<Item>,
        module_decl: &prism_ast::ModuleDeclaration,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        // Extract module information
        let module_name = module_decl.name.to_string();
        let sections = self.extract_sections_from_module(module_decl)?;
        let capabilities = self.extract_capabilities_from_module(module_decl)?;
        let effects = self.extract_effects_from_module(module_decl)?;
        
        // Create scope kind
        let scope_kind = ScopeKind::Module {
            module_name: module_name.clone(),
            sections: sections.iter().map(|s| format!("{:?}", s)).collect(),
            capabilities: capabilities.clone(),
            effects: effects.clone(),
        };
        
        // Create the scope
        let scope_id = scope_tree.create_scope(scope_kind, Some(parent_scope))?;
        self.build_stats.scopes_created += 1;
        
        // Map AST node to scope
        self.node_to_scope_map.insert(item.id, scope_id);
        
        // Update context
        let previous_module = self.current_context.current_module.clone();
        self.current_context.current_module = Some(module_name.clone());
        self.current_context.available_capabilities.extend(capabilities);
        self.current_context.current_effects.extend(effects);
        
        // Generate metadata if enabled
        if self.config.enable_ai_metadata {
            let metadata = self.generate_module_metadata(&module_name, &sections)?;
            scope_tree.update_metadata(scope_id, |scope_metadata| {
                *scope_metadata = metadata;
            })?;
            self.build_stats.scopes_with_ai_metadata += 1;
        }
        
        // Process module body (sections, items, etc.)
        self.process_module_body(module_decl, scope_id, scope_tree)?;
        
        // Restore context
        self.current_context.current_module = previous_module;
        
        Ok(Some(scope_id))
    }
    
    /// Build a function scope
    fn build_function_scope(
        &mut self,
        item: &AstNode<Item>,
        func_decl: &prism_ast::FunctionDeclaration,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        // Extract function information
        let function_name = func_decl.name.to_string();
        let parameters = self.extract_parameters_from_function(func_decl)?;
        let is_async = self.is_async_function(func_decl)?;
        let effects = self.extract_effects_from_function(func_decl)?;
        let required_capabilities = self.extract_capabilities_from_function(func_decl)?;
        let contracts = self.extract_contracts_from_function(func_decl)?;
        
        // Create scope kind
        let scope_kind = ScopeKind::Function {
            function_name: function_name.clone(),
            parameters: parameters.clone(),
            is_async,
            effects: effects.clone(),
            required_capabilities: required_capabilities.clone(),
            contracts,
        };
        
        // Create the scope
        let scope_id = scope_tree.create_scope(scope_kind, Some(parent_scope))?;
        self.build_stats.scopes_created += 1;
        
        // Map AST node to scope
        self.node_to_scope_map.insert(item.id, scope_id);
        
        // Generate metadata if enabled
        if self.config.enable_ai_metadata {
            let metadata = self.generate_function_metadata(&function_name, &parameters, is_async)?;
            scope_tree.update_metadata(scope_id, |scope_metadata| {
                *scope_metadata = metadata;
            })?;
            self.build_stats.scopes_with_ai_metadata += 1;
        }
        
        // Process function body
        if let Some(body) = &func_decl.body {
            self.build_from_statement_block(body, scope_id, scope_tree)?;
        }
        
        Ok(Some(scope_id))
    }
    
    /// Build a type scope
    fn build_type_scope(
        &mut self,
        item: &AstNode<Item>,
        type_decl: &prism_ast::TypeDeclaration,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        // Extract type information
        let type_name = type_decl.name.to_string();
        let type_category = self.determine_type_category(type_decl)?;
        let generic_parameters = self.extract_generic_parameters(type_decl)?;
        
        // Create scope kind
        let scope_kind = ScopeKind::Type {
            type_name: type_name.clone(),
            type_category: type_category.clone(),
            generic_parameters: generic_parameters.clone(),
        };
        
        // Create the scope
        let scope_id = scope_tree.create_scope(scope_kind, Some(parent_scope))?;
        self.build_stats.scopes_created += 1;
        
        // Map AST node to scope
        self.node_to_scope_map.insert(item.id, scope_id);
        
        // Generate metadata if enabled
        if self.config.enable_ai_metadata {
            let metadata = self.generate_type_metadata(&type_name, &type_category)?;
            scope_tree.update_metadata(scope_id, |scope_metadata| {
                *scope_metadata = metadata;
            })?;
            self.build_stats.scopes_with_ai_metadata += 1;
        }
        
        Ok(Some(scope_id))
    }
    
    /// Build scopes from a statement block
    pub fn build_from_statement_block(
        &mut self,
        statements: &[AstNode<Stmt>],
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Vec<crate::scope::ScopeId>> {
        let mut created_scopes = Vec::new();
        
        for stmt in statements {
            if let Some(scope_id) = self.build_from_statement(stmt, parent_scope, scope_tree)? {
                created_scopes.push(scope_id);
            }
        }
        
        Ok(created_scopes)
    }
    
    /// Build scope from a statement
    fn build_from_statement(
        &mut self,
        stmt: &AstNode<Stmt>,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        match &stmt.kind {
            Stmt::Block(block_stmt) => {
                self.build_block_scope(stmt, block_stmt, parent_scope, scope_tree)
            }
            Stmt::If(if_stmt) => {
                self.build_control_flow_scope(
                    stmt,
                    ControlFlowType::If,
                    parent_scope,
                    scope_tree,
                )
            }
            Stmt::While(while_stmt) => {
                self.build_control_flow_scope(
                    stmt,
                    ControlFlowType::While,
                    parent_scope,
                    scope_tree,
                )
            }
            Stmt::For(for_stmt) => {
                self.build_control_flow_scope(
                    stmt,
                    ControlFlowType::For,
                    parent_scope,
                    scope_tree,
                )
            }
            _ => Ok(None),
        }
    }
    
    /// Build a block scope
    fn build_block_scope(
        &mut self,
        stmt: &AstNode<Stmt>,
        _block_stmt: &prism_ast::BlockStatement,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        let scope_kind = ScopeKind::Block {
            block_type: BlockType::Regular,
        };
        
        let scope_id = scope_tree.create_scope(scope_kind, Some(parent_scope))?;
        self.build_stats.scopes_created += 1;
        
        self.node_to_scope_map.insert(stmt.id, scope_id);
        
        Ok(Some(scope_id))
    }
    
    /// Build a control flow scope
    fn build_control_flow_scope(
        &mut self,
        stmt: &AstNode<Stmt>,
        control_type: ControlFlowType,
        parent_scope: crate::scope::ScopeId,
        scope_tree: &mut ScopeTree,
    ) -> CompilerResult<Option<crate::scope::ScopeId>> {
        let scope_kind = ScopeKind::ControlFlow {
            control_type,
            condition: None, // TODO: Extract condition from AST
        };
        
        let scope_id = scope_tree.create_scope(scope_kind, Some(parent_scope))?;
        self.build_stats.scopes_created += 1;
        
        self.node_to_scope_map.insert(stmt.id, scope_id);
        
        Ok(Some(scope_id))
    }
    
    /// Get the scope ID for an AST node
    pub fn get_scope_for_node(&self, node_id: NodeId) -> Option<crate::scope::ScopeId> {
        self.node_to_scope_map.get(&node_id).copied()
    }
    
    /// Get build statistics
    pub fn get_statistics(&self) -> &BuildStatistics {
        &self.build_stats
    }
    
    // Helper methods for extracting information from AST nodes
    
    fn extract_sections_from_module(
        &self,
        _module_decl: &prism_ast::ModuleDeclaration,
    ) -> CompilerResult<Vec<SectionType>> {
        // TODO: Implement actual section extraction from AST
        Ok(vec![SectionType::Interface])
    }
    
    fn extract_capabilities_from_module(
        &self,
        _module_decl: &prism_ast::ModuleDeclaration,
    ) -> CompilerResult<Vec<String>> {
        // TODO: Implement actual capability extraction from AST
        Ok(vec![])
    }
    
    fn extract_effects_from_module(
        &self,
        _module_decl: &prism_ast::ModuleDeclaration,
    ) -> CompilerResult<Vec<String>> {
        // TODO: Implement actual effect extraction from AST
        Ok(vec![])
    }
    
    fn extract_parameters_from_function(
        &self,
        _func_decl: &prism_ast::FunctionDeclaration,
    ) -> CompilerResult<Vec<String>> {
        // TODO: Implement actual parameter extraction from AST
        Ok(vec![])
    }
    
    fn is_async_function(&self, _func_decl: &prism_ast::FunctionDeclaration) -> CompilerResult<bool> {
        // TODO: Implement actual async detection from AST
        Ok(false)
    }
    
    fn extract_effects_from_function(
        &self,
        _func_decl: &prism_ast::FunctionDeclaration,
    ) -> CompilerResult<Vec<String>> {
        // TODO: Implement actual effect extraction from AST
        Ok(vec![])
    }
    
    fn extract_capabilities_from_function(
        &self,
        _func_decl: &prism_ast::FunctionDeclaration,
    ) -> CompilerResult<Vec<String>> {
        // TODO: Implement actual capability extraction from AST
        Ok(vec![])
    }
    
    fn extract_contracts_from_function(
        &self,
        _func_decl: &prism_ast::FunctionDeclaration,
    ) -> CompilerResult<Option<String>> {
        // TODO: Implement actual contract extraction from AST
        Ok(None)
    }
    
    fn determine_type_category(
        &self,
        _type_decl: &prism_ast::TypeDeclaration,
    ) -> CompilerResult<String> {
        // TODO: Implement actual type category determination from AST
        Ok("struct".to_string())
    }
    
    fn extract_generic_parameters(
        &self,
        _type_decl: &prism_ast::TypeDeclaration,
    ) -> CompilerResult<Vec<String>> {
        // TODO: Implement actual generic parameter extraction from AST
        Ok(vec![])
    }
    
    fn process_module_body(
        &mut self,
        _module_decl: &prism_ast::ModuleDeclaration,
        _scope_id: crate::scope::ScopeId,
        _scope_tree: &mut ScopeTree,
    ) -> CompilerResult<()> {
        // TODO: Implement module body processing
        Ok(())
    }
    
    // Metadata generation methods
    
    fn generate_module_metadata(
        &self,
        module_name: &str,
        sections: &[SectionType],
    ) -> CompilerResult<ScopeMetadata> {
        let mut metadata = ScopeMetadata::new(Some(format!("Module: {}", module_name)));
        
        if self.config.enable_ai_metadata {
            let ai_context = AIScopeContext::new(format!("Module {} implementation", module_name));
            metadata = metadata.with_ai_context(ai_context);
        }
        
        if self.config.enable_documentation_extraction {
            let doc = ScopeDocumentation::new(format!("Module {}", module_name));
            metadata = metadata.with_documentation(doc);
        }
        
        Ok(metadata)
    }
    
    fn generate_function_metadata(
        &self,
        function_name: &str,
        parameters: &[String],
        is_async: bool,
    ) -> CompilerResult<ScopeMetadata> {
        let async_str = if is_async { "async " } else { "" };
        let mut metadata = ScopeMetadata::new(Some(format!(
            "{}Function: {} with {} parameters",
            async_str,
            function_name,
            parameters.len()
        )));
        
        if self.config.enable_ai_metadata {
            let ai_context = AIScopeContext::new(format!(
                "{}Function {} implementation",
                async_str,
                function_name
            ));
            metadata = metadata.with_ai_context(ai_context);
        }
        
        Ok(metadata)
    }
    
    fn generate_type_metadata(
        &self,
        type_name: &str,
        type_category: &str,
    ) -> CompilerResult<ScopeMetadata> {
        let mut metadata = ScopeMetadata::new(Some(format!("Type: {} ({})", type_name, type_category)));
        
        if self.config.enable_ai_metadata {
            let ai_context = AIScopeContext::new(format!("Type {} definition", type_name));
            metadata = metadata.with_ai_context(ai_context);
        }
        
        Ok(metadata)
    }
}

impl Default for ScopeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::ScopeTree;
    
    #[test]
    fn test_scope_builder_creation() {
        let builder = ScopeBuilder::new();
        
        assert!(builder.config.enable_ai_metadata);
        assert_eq!(builder.build_stats.scopes_created, 0);
    }
    
    #[test]
    fn test_build_context() {
        let context = BuildContext::default();
        
        assert!(context.current_module.is_none());
        assert!(context.current_section.is_none());
        assert_eq!(context.depth, 0);
        assert!(context.available_capabilities.is_empty());
    }
    
    #[test]
    fn test_metadata_generation() {
        let builder = ScopeBuilder::new();
        
        let metadata = builder.generate_module_metadata("TestModule", &[SectionType::Interface]).unwrap();
        
        assert!(metadata.responsibility.is_some());
        assert!(metadata.ai_context.is_some());
        assert!(metadata.documentation.is_some());
    }
} 