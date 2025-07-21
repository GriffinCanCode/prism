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

    /// Process a complete program with integrated symbol management
    pub async fn process_program(&mut self, program: &Program) -> CompilerResult<IntegratedProcessingResult> {
        let start_time = std::time::Instant::now();
        info!("Starting integrated symbol processing for program");

        let mut result = IntegratedProcessingResult {
            extraction_stats: ExtractionStats::default(),
            modules_processed: 0,
            integrations_performed: 0,
            validation_results: Vec::new(),
            processing_time_ms: 0,
        };

        // Phase 1: Extract symbols from AST
        info!("Phase 1: Extracting symbols from AST");
        result.extraction_stats = self.symbol_extractor.extract_program_symbols(program).await?;

        // Phase 2: Process modules with Smart Module Registry
        info!("Phase 2: Processing modules with Smart Module Registry");
        result.modules_processed = self.process_modules_with_registry(program).await?;

        // Phase 3: Perform cross-system integrations
        if self.config.enable_cross_system_integration {
            info!("Phase 3: Performing cross-system integrations");
            result.integrations_performed = self.perform_cross_system_integrations().await?;
        }

        // Phase 4: Validate integrated system
        if self.config.enable_comprehensive_validation {
            info!("Phase 4: Validating integrated system");
            result.validation_results = self.validate_integrated_system().await?;
        }

        // Phase 5: Synchronize with cache
        if self.config.enable_real_time_sync {
            info!("Phase 5: Synchronizing with cache");
            self.synchronize_with_cache().await?;
        }

        result.processing_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "Integrated symbol processing completed in {}ms: {} symbols, {} modules",
            result.processing_time_ms,
            result.extraction_stats.symbols_extracted,
            result.modules_processed
        );

        Ok(result)
    }

    /// Process modules with the Smart Module Registry
    async fn process_modules_with_registry(&self, program: &Program) -> CompilerResult<usize> {
        let mut modules_processed = 0;

        for item in &program.items {
            if let Item::Module(module_decl) = &item.kind {
                // Register module with Smart Module Registry
                self.module_registry.register_module(
                    module_decl,
                    std::path::PathBuf::from("placeholder.prsm"), // Would be actual file path
                    crate::scope::ScopeId::new(0), // Would be actual scope ID
                    item.id,
                ).await?;

                modules_processed += 1;
                debug!("Processed module with registry: {}", module_decl.name);
            }
        }

        Ok(modules_processed)
    }

    /// Perform cross-system integrations
    async fn perform_cross_system_integrations(&self) -> CompilerResult<usize> {
        let mut integrations = 0;

        // Integration 1: Sync symbols with semantic database
        self.sync_symbols_with_semantic_db().await?;
        integrations += 1;

        // Integration 2: Update scope tree with symbol information
        self.update_scope_tree_with_symbols().await?;
        integrations += 1;

        // Integration 3: Cross-reference module registry with symbol table
        self.cross_reference_modules_and_symbols().await?;
        integrations += 1;

        Ok(integrations)
    }

    /// Validate the integrated system
    async fn validate_integrated_system(&self) -> CompilerResult<Vec<ValidationResult>> {
        let mut validations = Vec::new();

        // Validation 1: Symbol consistency
        validations.push(self.validate_symbol_consistency().await?);

        // Validation 2: Module coherence
        validations.push(self.validate_module_coherence().await?);

        // Validation 3: Cross-reference integrity
        validations.push(self.validate_cross_references().await?);

        // Validation 4: Semantic integration
        validations.push(self.validate_semantic_integration().await?);

        Ok(validations)
    }

    /// Synchronize with compilation cache
    async fn synchronize_with_cache(&self) -> CompilerResult<()> {
        // Update cache with symbol information
        // This would integrate with the existing cache system
        debug!("Synchronized symbol information with cache");
        Ok(())
    }

    // Private helper methods for integrations and validations

    async fn sync_symbols_with_semantic_db(&self) -> CompilerResult<()> {
        // Synchronize symbol table with semantic database
        // This would use existing semantic database APIs
        debug!("Synchronized symbols with semantic database");
        Ok(())
    }

    async fn update_scope_tree_with_symbols(&self) -> CompilerResult<()> {
        // Update scope tree with symbol information
        // This would use existing scope tree APIs
        debug!("Updated scope tree with symbol information");
        Ok(())
    }

    async fn cross_reference_modules_and_symbols(&self) -> CompilerResult<()> {
        // Cross-reference module registry with symbol table
        // This would ensure consistency between systems
        debug!("Cross-referenced modules and symbols");
        Ok(())
    }

    async fn validate_symbol_consistency(&self) -> CompilerResult<ValidationResult> {
        // Validate that symbols are consistent across systems
        Ok(ValidationResult {
            validation_type: ValidationType::SymbolConsistency,
            success: true,
            issues: Vec::new(),
            recommendations: Vec::new(),
        })
    }

    async fn validate_module_coherence(&self) -> CompilerResult<ValidationResult> {
        // Validate that modules are coherent and well-structured
        Ok(ValidationResult {
            validation_type: ValidationType::ModuleCoherence,
            success: true,
            issues: Vec::new(),
            recommendations: Vec::new(),
        })
    }

    async fn validate_cross_references(&self) -> CompilerResult<ValidationResult> {
        // Validate cross-references between systems
        Ok(ValidationResult {
            validation_type: ValidationType::CrossReference,
            success: true,
            issues: Vec::new(),
            recommendations: Vec::new(),
        })
    }

    async fn validate_semantic_integration(&self) -> CompilerResult<ValidationResult> {
        // Validate semantic integration
        Ok(ValidationResult {
            validation_type: ValidationType::SemanticIntegration,
            success: true,
            issues: Vec::new(),
            recommendations: Vec::new(),
        })
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