//! Symbol Extraction Pipeline - AST to Symbol Table
//!
//! This module embodies the single concept of "Symbol Extraction from AST".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: extracting symbol information from AST nodes and populating
//! the symbol table using existing visitor patterns.
//!
//! **Conceptual Responsibility**: AST symbol extraction and registration
//! **What it does**: AST traversal, symbol extraction, symbol table population, metadata generation
//! **What it doesn't do**: symbol resolution, semantic analysis, code generation

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolTable, SymbolData, SymbolKind, SymbolBuilder};
use crate::symbols::metadata::{SymbolMetadata, AISymbolContext, BusinessContext};
use crate::semantic::SemanticDatabase;
use prism_ast::{
    Program, AstNode, Item, Stmt, Expr, Type, Pattern,
    visitor::{AstVisitor, DefaultVisitor},
    ModuleDecl, FunctionDecl, TypeDecl, VariableDecl, ConstDecl,
    SectionDecl, SectionKind, StabilityLevel
};
use prism_common::{span::Span, symbol::Symbol, NodeId};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use tracing::{info, debug, warn};

/// Symbol extraction pipeline that traverses AST and populates symbol table
/// 
/// This leverages existing AST visitor patterns to extract comprehensive
/// symbol information while maintaining integration with existing systems.
pub struct SymbolExtractor {
    /// Symbol table to populate
    symbol_table: Arc<SymbolTable>,
    /// Semantic database integration
    semantic_db: Arc<SemanticDatabase>,
    /// Extraction configuration
    config: ExtractionConfig,
    /// Current extraction context
    context: ExtractionContext,
    /// Statistics for monitoring
    stats: ExtractionStats,
}

/// Configuration for symbol extraction
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Enable AI metadata generation during extraction
    pub enable_ai_metadata: bool,
    /// Enable business context extraction
    pub enable_business_context: bool,
    /// Enable semantic type integration
    pub enable_semantic_integration: bool,
    /// Enable comprehensive documentation extraction
    pub enable_doc_extraction: bool,
    /// Enable effect and capability tracking
    pub enable_effect_tracking: bool,
    /// Maximum extraction depth for nested structures
    pub max_extraction_depth: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_business_context: true,
            enable_semantic_integration: true,
            enable_doc_extraction: true,
            enable_effect_tracking: true,
            max_extraction_depth: 50,
        }
    }
}

/// Current extraction context tracking
#[derive(Debug, Clone)]
struct ExtractionContext {
    /// Current module being processed
    current_module: Option<String>,
    /// Current section being processed
    current_section: Option<SectionKind>,
    /// Current function being processed
    current_function: Option<String>,
    /// Current nesting depth
    depth: usize,
    /// Available capabilities in current context
    available_capabilities: HashSet<String>,
    /// Current effects being tracked
    current_effects: HashSet<String>,
    /// Business context accumulator
    business_context: BusinessContext,
}

/// Extraction statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Total symbols extracted
    pub symbols_extracted: usize,
    /// Functions extracted
    pub functions_extracted: usize,
    /// Types extracted
    pub types_extracted: usize,
    /// Variables extracted
    pub variables_extracted: usize,
    /// Modules extracted
    pub modules_extracted: usize,
    /// Symbols with AI metadata
    pub symbols_with_ai_metadata: usize,
    /// Extraction time in milliseconds
    pub extraction_time_ms: u64,
}

impl SymbolExtractor {
    /// Create a new symbol extractor with existing integrations
    pub fn new(
        symbol_table: Arc<SymbolTable>,
        semantic_db: Arc<SemanticDatabase>,
        config: ExtractionConfig,
    ) -> Self {
        Self {
            symbol_table,
            semantic_db,
            config,
            context: ExtractionContext {
                current_module: None,
                current_section: None,
                current_function: None,
                depth: 0,
                available_capabilities: HashSet::new(),
                current_effects: HashSet::new(),
                business_context: BusinessContext::default(),
            },
            stats: ExtractionStats::default(),
        }
    }

    /// Extract symbols from a complete program
    pub async fn extract_program_symbols(&mut self, program: &Program) -> CompilerResult<ExtractionStats> {
        let start_time = std::time::Instant::now();
        info!("Starting symbol extraction for program with {} items", program.items.len());

        // Reset extraction state
        self.reset_extraction_state();

        // Extract symbols from all top-level items
        for item in &program.items {
            self.extract_item_symbols(item).await?;
        }

        // Record extraction time
        self.stats.extraction_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "Symbol extraction completed: {} symbols extracted in {}ms",
            self.stats.symbols_extracted, self.stats.extraction_time_ms
        );

        Ok(self.stats.clone())
    }

    /// Extract symbols from a top-level item
    async fn extract_item_symbols(&mut self, item: &AstNode<Item>) -> CompilerResult<()> {
        // Check depth limit
        if self.context.depth >= self.config.max_extraction_depth {
            warn!("Maximum extraction depth reached, skipping nested item");
            return Ok(());
        }

        match &item.kind {
            Item::Module(module_decl) => self.extract_module_symbols(item, module_decl).await?,
            Item::Function(func_decl) => self.extract_function_symbols(item, func_decl).await?,
            Item::Type(type_decl) => self.extract_type_symbols(item, type_decl).await?,
            Item::Variable(var_decl) => self.extract_variable_symbols(item, var_decl).await?,
            Item::Const(const_decl) => self.extract_const_symbols(item, const_decl).await?,
            Item::Import(import_decl) => self.extract_import_symbols(item, import_decl).await?,
            Item::Export(export_decl) => self.extract_export_symbols(item, export_decl).await?,
            Item::Statement(stmt) => self.extract_statement_symbols(stmt).await?,
        }

        Ok(())
    }

    /// Extract module symbols with comprehensive metadata
    async fn extract_module_symbols(
        &mut self,
        item: &AstNode<Item>,
        module_decl: &ModuleDecl,
    ) -> CompilerResult<()> {
        debug!("Extracting module symbols: {}", module_decl.name);

        // Update context
        let previous_module = self.context.current_module.clone();
        self.context.current_module = Some(module_decl.name.to_string());
        self.context.depth += 1;

        // Extract capabilities and effects
        let capabilities = self.extract_module_capabilities(module_decl);
        let effects = self.extract_module_effects(module_decl);
        
        self.context.available_capabilities.extend(capabilities.clone());
        self.context.current_effects.extend(effects.clone());

        // Build symbol using existing builder pattern
        let mut symbol_builder = SymbolBuilder::new(
            Symbol::intern(&module_decl.name.to_string()),
            SymbolKind::Module {
                sections: module_decl.sections.iter()
                    .map(|s| format!("{:?}", s.kind))
                    .collect(),
                capabilities: capabilities.clone(),
                cohesion_score: None, // Will be computed by cohesion system
            },
            item.span,
            item.id,
        );

        // Add visibility based on stability
        symbol_builder = match module_decl.stability {
            StabilityLevel::Stable | StabilityLevel::Beta => symbol_builder.public(),
            StabilityLevel::Experimental => symbol_builder.internal(),
            StabilityLevel::Deprecated => symbol_builder.private(),
        };

        // Add business context if enabled
        if self.config.enable_business_context {
            if let Some(description) = &module_decl.description {
                symbol_builder = symbol_builder.with_responsibility(description);
            }
            
            // Extract business context from module structure
            let business_context = self.extract_module_business_context(module_decl);
            symbol_builder = symbol_builder.with_business_context(business_context);
        }

        // Add AI metadata if enabled
        if self.config.enable_ai_metadata {
            let ai_context = self.generate_module_ai_context(module_decl);
            symbol_builder = symbol_builder.with_ai_context(ai_context);
            self.stats.symbols_with_ai_metadata += 1;
        }

        // Build and register symbol
        let symbol_data = symbol_builder.build()?;
        self.symbol_table.register_symbol(symbol_data).await?;

        self.stats.symbols_extracted += 1;
        self.stats.modules_extracted += 1;

        // Extract symbols from module sections
        for section in &module_decl.sections {
            self.extract_section_symbols(section).await?;
        }

        // Restore context
        self.context.current_module = previous_module;
        self.context.depth -= 1;

        debug!("Module symbol extraction completed: {}", module_decl.name);
        Ok(())
    }

    /// Extract function symbols with comprehensive metadata
    async fn extract_function_symbols(
        &mut self,
        item: &AstNode<Item>,
        func_decl: &FunctionDecl,
    ) -> CompilerResult<()> {
        debug!("Extracting function symbols: {}", func_decl.name);

        // Update context
        let previous_function = self.context.current_function.clone();
        self.context.current_function = Some(func_decl.name.to_string());
        self.context.depth += 1;

        // Extract function parameters for symbol kind
        let parameters = func_decl.parameters.iter()
            .map(|param| crate::symbols::kinds::ParameterSymbol {
                name: param.name.to_string(),
                param_type: None, // Would be resolved during semantic analysis
                default_value: param.default_value.is_some(),
                semantic_constraints: Vec::new(), // Would be extracted from type annotations
            })
            .collect();

        // Extract effects from function
        let effects = self.extract_function_effects(func_decl);

        // Build symbol using existing builder pattern
        let mut symbol_builder = SymbolBuilder::new(
            Symbol::intern(&func_decl.name.to_string()),
            SymbolKind::Function {
                parameters,
                return_type: None, // Would be resolved during semantic analysis
                effects,
                contracts: None, // Would be extracted from contracts if present
            },
            item.span,
            item.id,
        );

        // Add visibility based on current section
        symbol_builder = match self.context.current_section {
            Some(SectionKind::Interface) => symbol_builder.public(),
            Some(SectionKind::Tests) | Some(SectionKind::Examples) => symbol_builder.internal(),
            _ => symbol_builder.private(),
        };

        // Add business context if enabled
        if self.config.enable_business_context {
            let business_context = self.extract_function_business_context(func_decl);
            symbol_builder = symbol_builder.with_business_context(business_context);
        }

        // Add AI metadata if enabled
        if self.config.enable_ai_metadata {
            let ai_context = self.generate_function_ai_context(func_decl);
            symbol_builder = symbol_builder.with_ai_context(ai_context);
            self.stats.symbols_with_ai_metadata += 1;
        }

        // Build and register symbol
        let symbol_data = symbol_builder.build()?;
        self.symbol_table.register_symbol(symbol_data).await?;

        self.stats.symbols_extracted += 1;
        self.stats.functions_extracted += 1;

        // Extract parameter symbols
        for parameter in &func_decl.parameters {
            self.extract_parameter_symbol(parameter, &func_decl.name).await?;
        }

        // Extract symbols from function body if present
        if let Some(body) = &func_decl.body {
            self.extract_statement_symbols(body).await?;
        }

        // Restore context
        self.context.current_function = previous_function;
        self.context.depth -= 1;

        debug!("Function symbol extraction completed: {}", func_decl.name);
        Ok(())
    }

    /// Extract type symbols with semantic integration
    async fn extract_type_symbols(
        &mut self,
        item: &AstNode<Item>,
        type_decl: &TypeDecl,
    ) -> CompilerResult<()> {
        debug!("Extracting type symbols: {}", type_decl.name);

        // Extract semantic constraints and business rules
        let semantic_constraints = if self.config.enable_semantic_integration {
            self.extract_type_semantic_constraints(type_decl)
        } else {
            Vec::new()
        };

        let business_rules = if self.config.enable_business_context {
            self.extract_type_business_rules(type_decl)
        } else {
            Vec::new()
        };

        // Build symbol using existing builder pattern
        let mut symbol_builder = SymbolBuilder::new(
            Symbol::intern(&type_decl.name.to_string()),
            SymbolKind::Type {
                type_kind: crate::symbols::kinds::TypeKind::Custom, // Would be refined during analysis
                semantic_constraints,
                business_rules,
            },
            item.span,
            item.id,
        );

        // Add visibility based on current context
        symbol_builder = match self.context.current_section {
            Some(SectionKind::Types) | Some(SectionKind::Interface) => symbol_builder.public(),
            _ => symbol_builder.internal(),
        };

        // Add AI metadata if enabled
        if self.config.enable_ai_metadata {
            let ai_context = self.generate_type_ai_context(type_decl);
            symbol_builder = symbol_builder.with_ai_context(ai_context);
            self.stats.symbols_with_ai_metadata += 1;
        }

        // Build and register symbol
        let symbol_data = symbol_builder.build()?;
        self.symbol_table.register_symbol(symbol_data).await?;

        self.stats.symbols_extracted += 1;
        self.stats.types_extracted += 1;

        debug!("Type symbol extraction completed: {}", type_decl.name);
        Ok(())
    }

    /// Extract variable symbols
    async fn extract_variable_symbols(
        &mut self,
        item: &AstNode<Item>,
        var_decl: &VariableDecl,
    ) -> CompilerResult<()> {
        debug!("Extracting variable symbols: {}", var_decl.name);

        // Build symbol using existing builder pattern
        let mut symbol_builder = SymbolBuilder::new(
            Symbol::intern(&var_decl.name.to_string()),
            SymbolKind::Variable {
                is_mutable: var_decl.is_mutable,
                initialization: var_decl.initializer.is_some().into(),
                semantic_type: None, // Would be resolved during semantic analysis
            },
            item.span,
            item.id,
        );

        // Add visibility based on current context
        symbol_builder = match self.context.current_section {
            Some(SectionKind::Interface) => symbol_builder.public(),
            _ => symbol_builder.private(),
        };

        // Add AI metadata if enabled
        if self.config.enable_ai_metadata {
            let ai_context = self.generate_variable_ai_context(var_decl);
            symbol_builder = symbol_builder.with_ai_context(ai_context);
            self.stats.symbols_with_ai_metadata += 1;
        }

        // Build and register symbol
        let symbol_data = symbol_builder.build()?;
        self.symbol_table.register_symbol(symbol_data).await?;

        self.stats.symbols_extracted += 1;
        self.stats.variables_extracted += 1;

        debug!("Variable symbol extraction completed: {}", var_decl.name);
        Ok(())
    }

    /// Extract constant symbols
    async fn extract_const_symbols(
        &mut self,
        item: &AstNode<Item>,
        const_decl: &ConstDecl,
    ) -> CompilerResult<()> {
        debug!("Extracting constant symbols: {}", const_decl.name);

        // Build symbol using existing builder pattern
        let mut symbol_builder = SymbolBuilder::new(
            Symbol::intern(&const_decl.name.to_string()),
            SymbolKind::Constant {
                value: None, // Would be resolved during semantic analysis
                semantic_type: None, // Would be resolved during semantic analysis
            },
            item.span,
            item.id,
        );

        // Constants are typically public in interface sections
        symbol_builder = match self.context.current_section {
            Some(SectionKind::Interface) => symbol_builder.public(),
            _ => symbol_builder.internal(),
        };

        // Add AI metadata if enabled
        if self.config.enable_ai_metadata {
            let ai_context = self.generate_const_ai_context(const_decl);
            symbol_builder = symbol_builder.with_ai_context(ai_context);
            self.stats.symbols_with_ai_metadata += 1;
        }

        // Build and register symbol
        let symbol_data = symbol_builder.build()?;
        self.symbol_table.register_symbol(symbol_data).await?;

        self.stats.symbols_extracted += 1;

        debug!("Constant symbol extraction completed: {}", const_decl.name);
        Ok(())
    }

    // Helper methods for extraction (implementation details)

    fn extract_module_capabilities(&self, module_decl: &ModuleDecl) -> HashSet<String> {
        // Extract capabilities from module declaration
        // This would parse @capability annotations and module structure
        let mut capabilities = HashSet::new();
        
        // Add default capabilities based on sections
        for section in &module_decl.sections {
            match section.kind {
                SectionKind::Interface => capabilities.insert("PublicAPI".to_string()),
                SectionKind::Config => capabilities.insert("Configuration".to_string()),
                SectionKind::Errors => capabilities.insert("ErrorHandling".to_string()),
                SectionKind::Events => capabilities.insert("EventHandling".to_string()),
                SectionKind::Tests => capabilities.insert("Testing".to_string()),
                _ => false,
            };
        }
        
        capabilities
    }

    fn extract_module_effects(&self, _module_decl: &ModuleDecl) -> HashSet<String> {
        // Extract effects from module declaration
        // This would parse @effects annotations
        HashSet::new() // Placeholder
    }

    fn extract_module_business_context(&self, module_decl: &ModuleDecl) -> BusinessContext {
        BusinessContext {
            primary_capability: module_decl.name.to_string(),
            domain: None,
            responsibility: module_decl.description.clone(),
            business_rules: Vec::new(),
            entities: Vec::new(),
            processes: Vec::new(),
        }
    }

    fn generate_module_ai_context(&self, module_decl: &ModuleDecl) -> AISymbolContext {
        AISymbolContext {
            purpose: Some(format!("Module providing {}", module_decl.name)),
            conceptual_role: Some("Module".to_string()),
            ai_hints: vec![
                format!("Module with {} sections", module_decl.sections.len()),
                format!("Stability level: {:?}", module_decl.stability),
            ],
        }
    }

    fn extract_function_effects(&self, _func_decl: &FunctionDecl) -> Vec<crate::symbols::kinds::Effect> {
        // Extract effects from function declaration
        // This would parse @effects annotations and function body
        Vec::new() // Placeholder
    }

    fn extract_function_business_context(&self, func_decl: &FunctionDecl) -> BusinessContext {
        BusinessContext {
            primary_capability: func_decl.name.to_string(),
            domain: self.context.current_module.clone(),
            responsibility: None, // Would extract from documentation
            business_rules: Vec::new(),
            entities: Vec::new(),
            processes: Vec::new(),
        }
    }

    fn generate_function_ai_context(&self, func_decl: &FunctionDecl) -> AISymbolContext {
        AISymbolContext {
            purpose: Some(format!("Function {}", func_decl.name)),
            conceptual_role: Some("Function".to_string()),
            ai_hints: vec![
                format!("Function with {} parameters", func_decl.parameters.len()),
                format!("Returns: {}", func_decl.return_type.is_some()),
            ],
        }
    }

    fn extract_type_semantic_constraints(&self, _type_decl: &TypeDecl) -> Vec<crate::symbols::kinds::SemanticConstraint> {
        // Extract semantic constraints from type declaration
        Vec::new() // Placeholder
    }

    fn extract_type_business_rules(&self, _type_decl: &TypeDecl) -> Vec<crate::symbols::kinds::BusinessRule> {
        // Extract business rules from type declaration
        Vec::new() // Placeholder
    }

    fn generate_type_ai_context(&self, type_decl: &TypeDecl) -> AISymbolContext {
        AISymbolContext {
            purpose: Some(format!("Type definition for {}", type_decl.name)),
            conceptual_role: Some("Type".to_string()),
            ai_hints: vec!["Custom type definition".to_string()],
        }
    }

    fn generate_variable_ai_context(&self, var_decl: &VariableDecl) -> AISymbolContext {
        AISymbolContext {
            purpose: Some(format!("Variable {}", var_decl.name)),
            conceptual_role: Some("Variable".to_string()),
            ai_hints: vec![
                format!("Mutable: {}", var_decl.is_mutable),
                format!("Initialized: {}", var_decl.initializer.is_some()),
            ],
        }
    }

    fn generate_const_ai_context(&self, const_decl: &ConstDecl) -> AISymbolContext {
        AISymbolContext {
            purpose: Some(format!("Constant {}", const_decl.name)),
            conceptual_role: Some("Constant".to_string()),
            ai_hints: vec!["Immutable constant value".to_string()],
        }
    }

    // Additional extraction methods would be implemented here...
    
    async fn extract_section_symbols(&mut self, _section: &SectionDecl) -> CompilerResult<()> {
        // Implementation for section symbol extraction
        Ok(())
    }

    async fn extract_parameter_symbol(&mut self, _parameter: &prism_ast::Parameter, _function_name: &str) -> CompilerResult<()> {
        // Implementation for parameter symbol extraction
        Ok(())
    }

    async fn extract_statement_symbols(&mut self, _stmt: &AstNode<Stmt>) -> CompilerResult<()> {
        // Implementation for statement symbol extraction
        Ok(())
    }

    async fn extract_import_symbols(&mut self, _item: &AstNode<Item>, _import_decl: &prism_ast::ImportDecl) -> CompilerResult<()> {
        // Implementation for import symbol extraction
        Ok(())
    }

    async fn extract_export_symbols(&mut self, _item: &AstNode<Item>, _export_decl: &prism_ast::ExportDecl) -> CompilerResult<()> {
        // Implementation for export symbol extraction
        Ok(())
    }

    fn reset_extraction_state(&mut self) {
        self.context = ExtractionContext {
            current_module: None,
            current_section: None,
            current_function: None,
            depth: 0,
            available_capabilities: HashSet::new(),
            current_effects: HashSet::new(),
            business_context: BusinessContext::default(),
        };
        self.stats = ExtractionStats::default();
    }
}

impl Default for BusinessContext {
    fn default() -> Self {
        Self {
            primary_capability: String::new(),
            domain: None,
            responsibility: None,
            business_rules: Vec::new(),
            entities: Vec::new(),
            processes: Vec::new(),
        }
    }
} 