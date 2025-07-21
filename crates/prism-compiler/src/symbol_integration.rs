//! Symbol System Integration - High-Level API for PLT-004
//!
//! This module embodies the single concept of "Symbol System Integration".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: providing a cohesive high-level API that integrates all
//! symbol table components while maintaining their individual responsibilities.
//!
//! **Conceptual Responsibility**: Symbol system coordination and integration
//! **What it does**: system initialization, component coordination, high-level APIs
//! **What it doesn't do**: symbol storage, scope management, resolution (delegates to specialized modules)

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolTable, SymbolData, SymbolTableConfig, SymbolKind};
use crate::scope::{ScopeTree, ScopeId, ScopeKind, ScopeTreeConfig};
use crate::resolution::{SymbolResolver, ResolutionContext, ResolvedSymbol, ResolverConfig};
use crate::semantic::SemanticDatabase;
use crate::cache::CompilationCache;
use prism_common::{span::Span, symbol::Symbol};
use prism_effects::effects::EffectRegistry;
use prism_ast::{AstNode, Item, Program};
use std::sync::Arc;
use std::collections::HashMap;

/// Integrated symbol system providing a high-level API for PLT-004
/// 
/// This coordinates all symbol table components while maintaining
/// their individual responsibilities and conceptual cohesion
pub struct SymbolSystem {
    /// Symbol table component
    symbol_table: Arc<SymbolTable>,
    /// Scope tree component
    scope_tree: Arc<ScopeTree>,
    /// Symbol resolver component
    resolver: Arc<SymbolResolver>,
    /// Current resolution context
    current_context: ResolutionContext,
    /// System configuration
    config: SymbolSystemConfig,
}

/// Configuration for the integrated symbol system
#[derive(Debug, Clone)]
pub struct SymbolSystemConfig {
    /// Symbol table configuration
    pub symbol_table: SymbolTableConfig,
    /// Scope tree configuration
    pub scope_tree: ScopeTreeConfig,
    /// Resolver configuration
    pub resolver: ResolverConfig,
    /// Enable system-wide AI metadata
    pub enable_ai_metadata: bool,
    /// Enable comprehensive validation
    pub enable_validation: bool,
}

impl Default for SymbolSystemConfig {
    fn default() -> Self {
        Self {
            symbol_table: SymbolTableConfig::default(),
            scope_tree: ScopeTreeConfig::default(),
            resolver: ResolverConfig::default(),
            enable_ai_metadata: true,
            enable_validation: true,
        }
    }
}

/// Builder for constructing the symbol system with proper dependencies
pub struct SymbolSystemBuilder {
    semantic_db: Option<Arc<SemanticDatabase>>,
    effect_registry: Option<Arc<EffectRegistry>>,
    cache: Option<Arc<CompilationCache>>,
    config: SymbolSystemConfig,
}

impl SymbolSystemBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            semantic_db: None,
            effect_registry: None,
            cache: None,
            config: SymbolSystemConfig::default(),
        }
    }

    /// Set the semantic database integration
    pub fn with_semantic_database(mut self, semantic_db: Arc<SemanticDatabase>) -> Self {
        self.semantic_db = Some(semantic_db);
        self
    }

    /// Set the effect registry integration
    pub fn with_effect_registry(mut self, effect_registry: Arc<EffectRegistry>) -> Self {
        self.effect_registry = Some(effect_registry);
        self
    }

    /// Set the cache integration
    pub fn with_cache(mut self, cache: Arc<CompilationCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set custom configuration
    pub fn with_config(mut self, config: SymbolSystemConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the integrated symbol system
    pub fn build(self) -> CompilerResult<SymbolSystem> {
        // Require semantic database for integration
        let semantic_db = self.semantic_db.ok_or_else(|| CompilerError::InvalidInput {
            message: "Semantic database is required for symbol system".to_string(),
        })?;

        // Require effect registry for effect resolution
        let effect_registry = self.effect_registry.ok_or_else(|| CompilerError::InvalidInput {
            message: "Effect registry is required for symbol system".to_string(),
        })?;

        // Require cache for resolution caching
        let cache = self.cache.ok_or_else(|| CompilerError::InvalidInput {
            message: "Compilation cache is required for symbol system".to_string(),
        })?;

        // Build components with proper dependencies
        let symbol_table = Arc::new(SymbolTable::with_config(
            semantic_db.clone(),
            self.config.symbol_table,
        )?);

        let scope_tree = Arc::new(ScopeTree::with_config(self.config.scope_tree)?);

        let resolver = Arc::new(SymbolResolver::new(
            symbol_table.clone(),
            scope_tree.clone(),
            semantic_db,
            effect_registry,
            cache,
        )?);

        Ok(SymbolSystem {
            symbol_table,
            scope_tree,
            resolver,
            current_context: ResolutionContext::default(),
            config: self.config,
        })
    }
}

impl Default for SymbolSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolSystem {
    /// Create a builder for constructing the symbol system
    pub fn builder() -> SymbolSystemBuilder {
        SymbolSystemBuilder::new()
    }

    /// Register a symbol in the system
    pub fn register_symbol(&self, symbol_data: SymbolData, scope_id: ScopeId) -> CompilerResult<()> {
        // Register in symbol table
        self.symbol_table.register_symbol(symbol_data.clone())?;

        // Add to scope
        self.scope_tree.add_symbol_to_scope(scope_id, symbol_data.symbol)?;

        Ok(())
    }

    /// Create a new scope
    pub fn create_scope(&self, kind: ScopeKind, location: Span, parent: Option<ScopeId>) -> CompilerResult<ScopeId> {
        // Need mutable access to scope tree - this is a design limitation we'd need to address
        // For now, we'll return an error indicating this needs to be called during initialization
        Err(CompilerError::InvalidInput {
            message: "Scope creation must be done during system initialization".to_string(),
        })
    }

    /// Resolve a symbol by name
    pub async fn resolve_symbol(&self, name: &str) -> CompilerResult<ResolvedSymbol> {
        self.resolver.resolve_symbol(name, &self.current_context).await
    }

    /// Resolve a symbol with custom context
    pub async fn resolve_symbol_with_context(&self, name: &str, context: &ResolutionContext) -> CompilerResult<ResolvedSymbol> {
        self.resolver.resolve_symbol(name, context).await
    }

    /// Update the current resolution context
    pub fn set_context(&mut self, context: ResolutionContext) {
        self.current_context = context;
    }

    /// Get the current resolution context
    pub fn context(&self) -> &ResolutionContext {
        &self.current_context
    }

    /// Build symbol system from AST program
    pub fn build_from_ast(&self, program: &Program) -> CompilerResult<SymbolSystemSnapshot> {
        let mut snapshot = SymbolSystemSnapshot::new();

        // This would need mutable access to components, which our current design doesn't allow
        // This is a design decision - we could either:
        // 1. Make components internally mutable (current approach)
        // 2. Rebuild the system for each compilation unit
        // 3. Use a different pattern for initialization

        // For now, return a placeholder
        Ok(snapshot)
    }

    /// Get system statistics
    pub fn stats(&self) -> SymbolSystemStats {
        SymbolSystemStats {
            symbol_table: self.symbol_table.stats(),
            scope_tree: self.scope_tree.stats(),
            resolver: self.resolver.stats(),
        }
    }

    /// Get symbol table component
    pub fn symbol_table(&self) -> &Arc<SymbolTable> {
        &self.symbol_table
    }

    /// Get scope tree component
    pub fn scope_tree(&self) -> &Arc<ScopeTree> {
        &self.scope_tree
    }

    /// Get resolver component
    pub fn resolver(&self) -> &Arc<SymbolResolver> {
        &self.resolver
    }

    /// Get configuration
    pub fn config(&self) -> &SymbolSystemConfig {
        &self.config
    }
}

/// Snapshot of symbol system state after processing
#[derive(Debug, Clone)]
pub struct SymbolSystemSnapshot {
    /// Registered symbols
    pub symbols: HashMap<Symbol, SymbolData>,
    /// Created scopes
    pub scopes: HashMap<ScopeId, String>, // Simplified for now
    /// Resolution mappings
    pub resolutions: HashMap<String, Symbol>,
    /// AI metadata generated
    pub ai_metadata: Option<String>,
}

impl SymbolSystemSnapshot {
    /// Create a new empty snapshot
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            scopes: HashMap::new(),
            resolutions: HashMap::new(),
            ai_metadata: None,
        }
    }
}

impl Default for SymbolSystemSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined statistics from all symbol system components
#[derive(Debug, Clone)]
pub struct SymbolSystemStats {
    /// Symbol table statistics
    pub symbol_table: crate::symbols::table::SymbolTableStats,
    /// Scope tree statistics
    pub scope_tree: crate::scope::ScopeTreeStats,
    /// Resolver statistics
    pub resolver: crate::resolution::ResolverStats,
}

/// High-level API for common symbol operations
impl SymbolSystem {
    /// Find all symbols of a specific kind
    pub fn find_symbols_by_kind(&self, kind_filter: impl Fn(&SymbolKind) -> bool) -> Vec<SymbolData> {
        self.symbol_table.get_symbols_by_kind(kind_filter)
    }

    /// Find all functions in the system
    pub fn find_functions(&self) -> Vec<SymbolData> {
        self.find_symbols_by_kind(|kind| matches!(kind, SymbolKind::Function { .. }))
    }

    /// Find all modules in the system
    pub fn find_modules(&self) -> Vec<SymbolData> {
        self.find_symbols_by_kind(|kind| matches!(kind, SymbolKind::Module { .. }))
    }

    /// Find all types in the system
    pub fn find_types(&self) -> Vec<SymbolData> {
        self.find_symbols_by_kind(|kind| matches!(kind, SymbolKind::Type { .. }))
    }

    /// Get all scopes of a specific kind
    pub fn find_scopes_by_kind(&self, kind_filter: impl Fn(&ScopeKind) -> bool) -> Vec<crate::scope::ScopeData> {
        self.scope_tree.get_scopes_by_kind(kind_filter)
    }

    /// Get all module scopes
    pub fn find_module_scopes(&self) -> Vec<crate::scope::ScopeData> {
        self.find_scopes_by_kind(|kind| matches!(kind, ScopeKind::Module { .. }))
    }

    /// Get all function scopes
    pub fn find_function_scopes(&self) -> Vec<crate::scope::ScopeData> {
        self.find_scopes_by_kind(|kind| matches!(kind, ScopeKind::Function { .. }))
    }

    /// Validate the entire symbol system for consistency
    pub fn validate(&self) -> CompilerResult<ValidationResult> {
        if !self.config.enable_validation {
            return Ok(ValidationResult::skipped());
        }

        let mut issues = Vec::new();

        // Validate symbol table consistency
        let symbol_stats = self.symbol_table.stats();
        if symbol_stats.total_symbols == 0 {
            issues.push("No symbols registered in symbol table".to_string());
        }

        // Validate scope tree consistency
        let scope_stats = self.scope_tree.stats();
        if scope_stats.total_scopes == 0 {
            issues.push("No scopes created in scope tree".to_string());
        }

        // Check for orphaned symbols (symbols not in any scope)
        // This would require cross-referencing symbol table and scope tree

        Ok(ValidationResult {
            is_valid: issues.is_empty(),
            issues,
            warnings: Vec::new(),
        })
    }
}

/// Result of symbol system validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the system is valid
    pub is_valid: bool,
    /// Critical issues found
    pub issues: Vec<String>,
    /// Non-critical warnings
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a result indicating validation was skipped
    pub fn skipped() -> Self {
        Self {
            is_valid: true,
            issues: Vec::new(),
            warnings: vec!["Validation was skipped".to_string()],
        }
    }
} 