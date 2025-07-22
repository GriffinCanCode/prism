//! Symbol System Integration Layer - Orchestrating Symbol Management
//!
//! This module embodies the single concept of "Symbol System Orchestration".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: orchestrating symbol extraction, registration, and integration
//! with existing systems like the Smart Module Registry and Semantic Database.
//!
//! **Conceptual Responsibility**: Symbol system coordination and workflow orchestration
//! **What it does**: coordinate extraction, integrate with module registry, orchestrate symbol workflows
//! **What it doesn't do**: symbol storage, AST parsing, code generation

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolTable, SymbolExtractor, ExtractionConfig, ExtractionStats};
use crate::module_registry::{SmartModuleRegistry, ModuleRegistryConfig};
use crate::semantic::SemanticDatabase;
use crate::cache::CompilationCache;
use crate::scope::ScopeTree;
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::{NodeId, span::Span};
use std::sync::Arc;
use std::collections::HashMap;
use tracing::{info, debug, warn};

/// Integrated symbol management system that orchestrates all symbol-related operations
/// 
/// This provides a high-level interface that coordinates symbol extraction with
/// existing systems while maintaining their individual responsibilities.
pub struct IntegratedSymbolSystem {
    /// Symbol table for storage and retrieval
    symbol_table: Arc<SymbolTable>,
    /// Symbol extractor for AST processing
    symbol_extractor: SymbolExtractor,
    /// Smart Module Registry integration
    module_registry: Arc<SmartModuleRegistry>,
    /// Semantic database integration
    semantic_db: Arc<SemanticDatabase>,
    /// Scope tree integration
    scope_tree: Arc<ScopeTree>,
    /// Compilation cache integration
    cache: Arc<CompilationCache>,
    /// System configuration
    config: IntegratedSystemConfig,
}

/// Configuration for the integrated symbol system
#[derive(Debug, Clone)]
pub struct IntegratedSystemConfig {
    /// Symbol extraction configuration
    pub extraction: ExtractionConfig,
    /// Module registry configuration
    pub module_registry: ModuleRegistryConfig,
    /// Enable cross-system integration
    pub enable_cross_system_integration: bool,
    /// Enable real-time synchronization
    pub enable_real_time_sync: bool,
    /// Enable comprehensive validation
    pub enable_comprehensive_validation: bool,
}

impl Default for IntegratedSystemConfig {
    fn default() -> Self {
        Self {
            extraction: ExtractionConfig::default(),
            module_registry: ModuleRegistryConfig::default(),
            enable_cross_system_integration: true,
            enable_real_time_sync: true,
            enable_comprehensive_validation: true,
        }
    }
}

/// Result of integrated symbol processing
#[derive(Debug, Clone)]
pub struct IntegratedProcessingResult {
    /// Symbol extraction statistics
    pub extraction_stats: ExtractionStats,
    /// Modules processed
    pub modules_processed: usize,
    /// Cross-system integrations performed
    pub integrations_performed: usize,
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Validation result for symbol processing
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation type
    pub validation_type: ValidationType,
    /// Success status
    pub success: bool,
    /// Issues found
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Types of validation performed
#[derive(Debug, Clone)]
pub enum ValidationType {
    /// Symbol consistency validation
    SymbolConsistency,
    /// Module coherence validation
    ModuleCoherence,
    /// Cross-reference validation
    CrossReference,
    /// Semantic integration validation
    SemanticIntegration,
}

impl IntegratedSymbolSystem {
    /// Create a new integrated symbol system
    pub fn new(
        symbol_table: Arc<SymbolTable>,
        module_registry: Arc<SmartModuleRegistry>,
        semantic_db: Arc<SemanticDatabase>,
        scope_tree: Arc<ScopeTree>,
        cache: Arc<CompilationCache>,
        config: IntegratedSystemConfig,
    ) -> CompilerResult<Self> {
        // Create symbol extractor with integrated configuration
        let symbol_extractor = SymbolExtractor::new(
            symbol_table.clone(),
            semantic_db.clone(),
            config.extraction.clone(),
        );

        Ok(Self {
            symbol_table,
            symbol_extractor,
            module_registry,
            semantic_db,
            scope_tree,
            cache,
            config,
        })
    }

    /// Process program and extract all symbols with cross-crate resolution
    pub async fn process_program(&mut self, program: &Program) -> CompilerResult<IntegrationResult> {
        let start_time = std::time::Instant::now();
        let mut result = IntegrationResult::default();
        
        info!("Processing program for symbol integration: {} items", program.items.len());

        // Step 1: Extract symbols from AST
        let extraction_result = self.symbol_extractor.extract_from_program(program).await?;
        result.symbols_extracted = extraction_result.stats.total_symbols;

        // Step 2: Register symbols in symbol table
        for symbol_data in &extraction_result.symbols {
            self.symbol_table.register_symbol(symbol_data.clone())?;
        }

        // Step 3: Process modules and register with Smart Module Registry
        for item in &program.items {
            if let Item::Module(module_decl) = &item.inner {
                self.process_module_registration(module_decl, &item.node_id).await?;
                result.modules_registered += 1;
            }
        }

        // Step 4: Perform cross-crate symbol resolution
        let resolution_results = self.perform_cross_crate_resolution(&extraction_result.symbols).await?;
        result.symbols_resolved = resolution_results.len();

        // Step 5: Update semantic database with resolved symbols
        self.update_semantic_database(&resolution_results).await?;

        // Step 6: Generate integration statistics
        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        result.integration_diagnostics = extraction_result.diagnostics;

        Ok(result)
    }

    /// Register a module with the Smart Module Registry
    async fn process_module_registration(
        &self, 
        module_decl: &ModuleDecl, 
        node_id: &NodeId
    ) -> CompilerResult<()> {
        debug!("Registering module with Smart Module Registry: {}", module_decl.name);

        // Create a dummy scope ID and file path for now
        // TODO: Get actual scope ID from scope tree integration
        let scope_id = prism_compiler::scope::ScopeId::new(node_id.0 as u32);
        let file_path = std::path::PathBuf::from(format!("{}.prism", module_decl.name));

        self.module_registry
            .register_module(module_decl, file_path, scope_id, *node_id)
            .await?;

        Ok(())
    }

    /// Perform cross-crate symbol resolution for extracted symbols
    async fn perform_cross_crate_resolution(
        &self,
        symbols: &[SymbolData],
    ) -> CompilerResult<Vec<CrossCrateResolutionResult>> {
        let mut resolution_results = Vec::new();

        info!("Performing cross-crate resolution for {} symbols", symbols.len());

        for symbol_data in symbols {
            // Create resolution context for this symbol
            let context = self.create_resolution_context(symbol_data).await?;

            // Attempt to resolve symbol references within the symbol
            let resolution_result = self.resolve_symbol_references(symbol_data, &context).await?;

            resolution_results.push(resolution_result);
        }

        Ok(resolution_results)
    }

    /// Create resolution context for a symbol
    async fn create_resolution_context(&self, symbol_data: &SymbolData) -> CompilerResult<ResolutionContext> {
        // Extract current scope from symbol location
        let current_scope = self.scope_tree.find_scope_for_location(&symbol_data.location);

        // Extract current module from symbol metadata
        let current_module = symbol_data.metadata.as_ref()
            .and_then(|meta| meta.module_context.as_ref())
            .map(|ctx| ctx.module_name.clone());

        // Get available capabilities from effect system
        let available_capabilities = self.extract_available_capabilities(symbol_data).await?;

        Ok(ResolutionContext {
            current_scope,
            current_module,
            available_capabilities,
            effect_context: None, // TODO: Extract from effect registry
            syntax_style: "canonical".to_string(),
            preferences: crate::resolution::ResolutionPreferences::default(),
        })
    }

    /// Resolve symbol references within a symbol's definition
    async fn resolve_symbol_references(
        &self,
        symbol_data: &SymbolData,
        context: &ResolutionContext,
    ) -> CompilerResult<CrossCrateResolutionResult> {
        let mut resolved_references = Vec::new();
        let mut resolution_errors = Vec::new();

        // Extract symbol references from the symbol's definition
        let symbol_references = self.extract_symbol_references(symbol_data)?;

        for reference in symbol_references {
            match self.resolve_single_reference(&reference, context).await {
                Ok(resolved) => {
                    resolved_references.push(resolved);
                }
                Err(e) => {
                    warn!("Failed to resolve reference '{}': {}", reference, e);
                    resolution_errors.push(format!("Failed to resolve '{}': {}", reference, e));
                }
            }
        }

        Ok(CrossCrateResolutionResult {
            original_symbol: symbol_data.symbol,
            resolved_references,
            resolution_errors,
            cross_crate_dependencies: self.identify_cross_crate_dependencies(&resolved_references),
        })
    }

    /// Extract symbol references from a symbol's definition
    fn extract_symbol_references(&self, symbol_data: &SymbolData) -> CompilerResult<Vec<String>> {
        let mut references = Vec::new();

        // Extract from symbol metadata if available
        if let Some(metadata) = &symbol_data.metadata {
            // Extract from documentation references
            if let Some(doc) = &metadata.documentation {
                references.extend(doc.see_also.clone());
                references.extend(doc.related_symbols.clone());
            }

            // Extract from business context
            if let Some(business) = &metadata.business_context {
                references.extend(business.related_entities.clone());
                references.extend(business.dependencies.clone());
            }
        }

        // TODO: Extract from AST node analysis
        // This would require AST traversal to find identifier references

        Ok(references)
    }

    /// Resolve a single symbol reference using the integrated resolver
    async fn resolve_single_reference(
        &self,
        reference: &str,
        context: &ResolutionContext,
    ) -> CompilerResult<ResolvedReference> {
        // Use the existing SymbolResolver infrastructure
        // Note: This would require the SymbolResolver to be available in the integration
        // For now, we'll create a placeholder implementation

        Ok(ResolvedReference {
            original_name: reference.to_string(),
            resolved_symbol: None, // TODO: Actual resolution
            resolution_path: "placeholder".to_string(),
            confidence: 0.5,
            cross_crate_source: None,
        })
    }

    /// Extract available capabilities for a symbol
    async fn extract_available_capabilities(&self, symbol_data: &SymbolData) -> CompilerResult<Vec<String>> {
        let mut capabilities = Vec::new();

        // Extract from symbol metadata
        if let Some(metadata) = &symbol_data.metadata {
            if let Some(effects) = &metadata.effects {
                capabilities.extend(effects.required_capabilities.clone());
                capabilities.extend(effects.provided_capabilities.clone());
            }
        }

        // TODO: Query effect registry for additional capabilities
        
        Ok(capabilities)
    }

    /// Identify cross-crate dependencies from resolved references
    fn identify_cross_crate_dependencies(&self, resolved_refs: &[ResolvedReference]) -> Vec<CrossCrateDependency> {
        let mut dependencies = Vec::new();

        for resolved_ref in resolved_refs {
            if let Some(source) = &resolved_ref.cross_crate_source {
                dependencies.push(CrossCrateDependency {
                    target_crate: source.clone(),
                    symbol_name: resolved_ref.original_name.clone(),
                    dependency_type: DependencyType::Symbol,
                });
            }
        }

        dependencies
    }

    /// Update semantic database with resolution results
    async fn update_semantic_database(&self, results: &[CrossCrateResolutionResult]) -> CompilerResult<()> {
        for result in results {
            // Update semantic database with resolved symbol information
            // This would integrate with the existing SemanticDatabase
            debug!("Updating semantic database for symbol: {:?}", result.original_symbol);
            
            // TODO: Actual semantic database update
            // self.semantic_db.update_symbol_resolution(result).await?;
        }
        
        Ok(())
    }

    /// Get comprehensive symbol statistics
    pub async fn get_symbol_statistics(&self) -> CompilerResult<SymbolStatistics> {
        // Gather statistics from all integrated systems
        Ok(SymbolStatistics {
            total_symbols: self.symbol_table.get_symbol_count().await?,
            modules_registered: self.module_registry.list_modules().await.len(),
            semantic_entries: 0, // Would query semantic database
            scope_nodes: 0, // Would query scope tree
        })
    }

    /// Find symbols by capability
    pub async fn find_symbols_by_capability(&self, capability: &str) -> CompilerResult<Vec<String>> {
        // Use module registry to find modules with capability, then get their symbols
        let modules = self.module_registry.discover_by_capability(capability).await?;
        
        let mut symbols = Vec::new();
        for module in modules {
            // Get symbols from this module
            // This would query the symbol table for module symbols
            symbols.push(module.module_info.name);
        }

        Ok(symbols)
    }

    /// Find symbols by business context
    pub async fn find_symbols_by_business_context(&self, context: &str) -> CompilerResult<Vec<String>> {
        // Use module registry business context index
        let modules = self.module_registry.find_by_business_context(context).await?;
        
        let mut symbols = Vec::new();
        for module in modules {
            symbols.push(module.module_info.name);
        }

        Ok(symbols)
    }
}

/// Comprehensive symbol statistics
#[derive(Debug, Clone)]
pub struct SymbolStatistics {
    /// Total symbols in symbol table
    pub total_symbols: usize,
    /// Modules registered in module registry
    pub modules_registered: usize,
    /// Entries in semantic database
    pub semantic_entries: usize,
    /// Nodes in scope tree
    pub scope_nodes: usize,
}

/// Result of integrated symbol processing
#[derive(Debug, Default)]
pub struct IntegrationResult {
    /// Number of symbols extracted
    pub symbols_extracted: usize,
    /// Number of symbols resolved
    pub symbols_resolved: usize,
    /// Number of modules registered
    pub modules_registered: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Integration diagnostics
    pub integration_diagnostics: Vec<String>,
}

/// Result of cross-crate symbol resolution
#[derive(Debug)]
pub struct CrossCrateResolutionResult {
    /// Original symbol that was processed
    pub original_symbol: Symbol,
    /// Successfully resolved references
    pub resolved_references: Vec<ResolvedReference>,
    /// Resolution errors encountered
    pub resolution_errors: Vec<String>,
    /// Cross-crate dependencies identified
    pub cross_crate_dependencies: Vec<CrossCrateDependency>,
}

/// A resolved symbol reference
#[derive(Debug)]
pub struct ResolvedReference {
    /// Original reference name
    pub original_name: String,
    /// Resolved symbol (if successful)
    pub resolved_symbol: Option<Symbol>,
    /// Resolution path taken
    pub resolution_path: String,
    /// Confidence in resolution (0.0 to 1.0)
    pub confidence: f64,
    /// Cross-crate source (if applicable)
    pub cross_crate_source: Option<String>,
}

/// Cross-crate dependency information
#[derive(Debug)]
pub struct CrossCrateDependency {
    /// Target crate name
    pub target_crate: String,
    /// Symbol name
    pub symbol_name: String,
    /// Type of dependency
    pub dependency_type: DependencyType,
}

/// Types of cross-crate dependencies
#[derive(Debug)]
pub enum DependencyType {
    Symbol,
    Module,
    Type,
    Effect,
    Capability,
}

/// Integration statistics
#[derive(Debug)]
pub struct IntegrationStatistics {
    /// Total symbols processed
    pub total_symbols_processed: usize,
    /// Successful resolutions
    pub successful_resolutions: usize,
    /// Failed resolutions
    pub failed_resolutions: usize,
    /// Cross-crate dependencies found
    pub cross_crate_dependencies: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Builder for IntegratedSymbolSystem
pub struct IntegratedSymbolSystemBuilder {
    symbol_table: Option<Arc<SymbolTable>>,
    module_registry: Option<Arc<SmartModuleRegistry>>,
    semantic_db: Option<Arc<SemanticDatabase>>,
    scope_tree: Option<Arc<ScopeTree>>,
    cache: Option<Arc<CompilationCache>>,
    config: IntegratedSystemConfig,
}

impl IntegratedSymbolSystemBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            symbol_table: None,
            module_registry: None,
            semantic_db: None,
            scope_tree: None,
            cache: None,
            config: IntegratedSystemConfig::default(),
        }
    }

    /// Set symbol table
    pub fn with_symbol_table(mut self, symbol_table: Arc<SymbolTable>) -> Self {
        self.symbol_table = Some(symbol_table);
        self
    }

    /// Set module registry
    pub fn with_module_registry(mut self, module_registry: Arc<SmartModuleRegistry>) -> Self {
        self.module_registry = Some(module_registry);
        self
    }

    /// Set semantic database
    pub fn with_semantic_db(mut self, semantic_db: Arc<SemanticDatabase>) -> Self {
        self.semantic_db = Some(semantic_db);
        self
    }

    /// Set scope tree
    pub fn with_scope_tree(mut self, scope_tree: Arc<ScopeTree>) -> Self {
        self.scope_tree = Some(scope_tree);
        self
    }

    /// Set cache
    pub fn with_cache(mut self, cache: Arc<CompilationCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: IntegratedSystemConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the integrated symbol system
    pub fn build(self) -> CompilerResult<IntegratedSymbolSystem> {
        let symbol_table = self.symbol_table.ok_or_else(|| CompilerError::InvalidInput {
            message: "Symbol table is required".to_string(),
        })?;

        let module_registry = self.module_registry.ok_or_else(|| CompilerError::InvalidInput {
            message: "Module registry is required".to_string(),
        })?;

        let semantic_db = self.semantic_db.ok_or_else(|| CompilerError::InvalidInput {
            message: "Semantic database is required".to_string(),
        })?;

        let scope_tree = self.scope_tree.ok_or_else(|| CompilerError::InvalidInput {
            message: "Scope tree is required".to_string(),
        })?;

        let cache = self.cache.ok_or_else(|| CompilerError::InvalidInput {
            message: "Cache is required".to_string(),
        })?;

        IntegratedSymbolSystem::new(
            symbol_table,
            module_registry,
            semantic_db,
            scope_tree,
            cache,
            self.config,
        )
    }
}

impl Default for IntegratedSymbolSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
} 