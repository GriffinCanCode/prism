//! Comprehensive Symbol Management System Usage Example
//!
//! This example demonstrates how to use the enhanced symbol management system
//! that integrates symbol extraction, module registry, semantic analysis, and
//! AI metadata generation while maintaining separation of concerns.

use prism_compiler::{
    // Core compiler components
    PrismCompiler, CompilationConfig,
    // Symbol management system
    IntegratedSymbolSystem, IntegratedSymbolSystemBuilder, IntegratedSystemConfig,
    SymbolExtractor, ExtractionConfig, SymbolMetadataProvider, SymbolProviderConfig,
    // Existing infrastructure
    symbols::{SymbolTable, SymbolTableConfig},
    module_registry::{SmartModuleRegistry, ModuleRegistryConfig},
    semantic::{SemanticDatabase, SemanticConfig},
    scope::{ScopeTree, ScopeTreeConfig},
    cache::{CompilationCache, CacheConfig},
    error::CompilerResult,
};
use prism_ast::{Program, AstNode, Item};
use prism_common::SourceId;
use std::sync::Arc;
use tokio;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> CompilerResult<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting Symbol Management System Usage Example");

    // Example 1: Basic Symbol Extraction
    basic_symbol_extraction().await?;

    // Example 2: Integrated Symbol System Usage
    integrated_symbol_system_usage().await?;

    // Example 3: AI Metadata Provider Usage
    ai_metadata_provider_usage().await?;

    // Example 4: Advanced Capability-Based Discovery
    capability_based_discovery().await?;

    info!("Symbol Management System examples completed successfully");
    Ok(())
}

/// Example 1: Basic symbol extraction from AST
async fn basic_symbol_extraction() -> CompilerResult<()> {
    info!("=== Example 1: Basic Symbol Extraction ===");

    // Create semantic database
    let semantic_config = SemanticConfig::default();
    let semantic_db = Arc::new(SemanticDatabase::new(&semantic_config)?);

    // Create symbol table
    let symbol_table_config = SymbolTableConfig::default();
    let symbol_table = Arc::new(SymbolTable::with_config(semantic_db.clone(), symbol_table_config)?);

    // Create symbol extractor
    let extraction_config = ExtractionConfig {
        enable_ai_metadata: true,
        enable_business_context: true,
        enable_semantic_integration: true,
        enable_doc_extraction: true,
        enable_effect_tracking: true,
        max_extraction_depth: 50,
    };

    let mut symbol_extractor = SymbolExtractor::new(
        symbol_table.clone(),
        semantic_db.clone(),
        extraction_config,
    );

    // Create a sample AST program (in a real scenario, this would come from parsing)
    let program = create_sample_program();

    // Extract symbols from the program
    let extraction_stats = symbol_extractor.extract_program_symbols(&program).await?;

    info!("Symbol extraction completed:");
    info!("  - Total symbols extracted: {}", extraction_stats.symbols_extracted);
    info!("  - Functions extracted: {}", extraction_stats.functions_extracted);
    info!("  - Types extracted: {}", extraction_stats.types_extracted);
    info!("  - Variables extracted: {}", extraction_stats.variables_extracted);
    info!("  - Modules extracted: {}", extraction_stats.modules_extracted);
    info!("  - Symbols with AI metadata: {}", extraction_stats.symbols_with_ai_metadata);
    info!("  - Extraction time: {}ms", extraction_stats.extraction_time_ms);

    Ok(())
}

/// Example 2: Integrated symbol system usage
async fn integrated_symbol_system_usage() -> CompilerResult<()> {
    info!("=== Example 2: Integrated Symbol System Usage ===");

    // Create all required components
    let semantic_db = Arc::new(SemanticDatabase::new(&SemanticConfig::default())?);
    let symbol_table = Arc::new(SymbolTable::with_config(
        semantic_db.clone(),
        SymbolTableConfig::default(),
    )?);
    let scope_tree = Arc::new(ScopeTree::with_config(ScopeTreeConfig::default())?);
    let cache = Arc::new(CompilationCache::new(CacheConfig::default())?);
    
    // Create Smart Module Registry
    let module_registry = Arc::new(SmartModuleRegistry::new(
        ModuleRegistryConfig::default(),
        symbol_table.clone(),
        scope_tree.clone(),
        semantic_db.clone(),
        cache.clone(),
    )?);

    // Configure integrated system
    let integrated_config = IntegratedSystemConfig {
        extraction: ExtractionConfig::default(),
        module_registry: ModuleRegistryConfig::default(),
        enable_cross_system_integration: true,
        enable_real_time_sync: true,
        enable_comprehensive_validation: true,
    };

    // Build integrated symbol system
    let mut integrated_system = IntegratedSymbolSystemBuilder::new()
        .with_symbol_table(symbol_table.clone())
        .with_module_registry(module_registry.clone())
        .with_semantic_db(semantic_db.clone())
        .with_scope_tree(scope_tree.clone())
        .with_cache(cache.clone())
        .with_config(integrated_config)
        .build()?;

    // Create a sample program
    let program = create_sample_program();

    // Process the program with integrated symbol management
    let processing_result = integrated_system.process_program(&program).await?;

    info!("Integrated processing completed:");
    info!("  - Symbols extracted: {}", processing_result.extraction_stats.symbols_extracted);
    info!("  - Modules processed: {}", processing_result.modules_processed);
    info!("  - Integrations performed: {}", processing_result.integrations_performed);
    info!("  - Validations: {}", processing_result.validation_results.len());
    info!("  - Processing time: {}ms", processing_result.processing_time_ms);

    // Demonstrate capability-based discovery
    let database_symbols = integrated_system.find_symbols_by_capability("Database").await?;
    info!("Found {} symbols with Database capability", database_symbols.len());

    // Demonstrate business context discovery
    let auth_symbols = integrated_system.find_symbols_by_business_context("authentication").await?;
    info!("Found {} symbols related to authentication", auth_symbols.len());

    // Get comprehensive statistics
    let stats = integrated_system.get_symbol_statistics().await?;
    info!("System statistics:");
    info!("  - Total symbols: {}", stats.total_symbols);
    info!("  - Modules registered: {}", stats.modules_registered);
    info!("  - Semantic entries: {}", stats.semantic_entries);
    info!("  - Scope nodes: {}", stats.scope_nodes);

    Ok(())
}

/// Example 3: AI metadata provider usage
async fn ai_metadata_provider_usage() -> CompilerResult<()> {
    info!("=== Example 3: AI Metadata Provider Usage ===");

    // Create required components
    let semantic_db = Arc::new(SemanticDatabase::new(&SemanticConfig::default())?);
    let symbol_table = Arc::new(SymbolTable::with_config(
        semantic_db.clone(),
        SymbolTableConfig::default(),
    )?);
    let scope_tree = Arc::new(ScopeTree::with_config(ScopeTreeConfig::default())?);
    let cache = Arc::new(CompilationCache::new(CacheConfig::default())?);
    let module_registry = Arc::new(SmartModuleRegistry::new(
        ModuleRegistryConfig::default(),
        symbol_table.clone(),
        scope_tree.clone(),
        semantic_db.clone(),
        cache.clone(),
    )?);

    // Extract some symbols first
    let mut symbol_extractor = SymbolExtractor::new(
        symbol_table.clone(),
        semantic_db.clone(),
        ExtractionConfig::default(),
    );
    let program = create_sample_program();
    let _ = symbol_extractor.extract_program_symbols(&program).await?;

    // Create AI metadata provider
    let provider_config = SymbolProviderConfig {
        enable_symbol_export: true,
        enable_business_context: true,
        enable_capability_export: true,
        enable_relationship_export: true,
        max_export_batch_size: 100,
        cache_refresh_interval: 300,
    };

    let mut ai_provider = SymbolMetadataProvider::new(provider_config)
        .with_symbol_table(symbol_table.clone())
        .with_module_registry(module_registry.clone());

    // Get provider information
    let provider_info = ai_provider.provider_info();
    info!("AI Provider: {} v{}", provider_info.name, provider_info.version);
    info!("Description: {}", provider_info.description);
    info!("Capabilities: {:?}", provider_info.capabilities);
    info!("Enabled: {}", provider_info.enabled);

    // Get domain metadata for AI consumption
    let context = prism_ai::providers::ProviderContext::default();
    match ai_provider.get_domain_metadata(&context).await {
        Ok(domain_metadata) => {
            info!("Successfully extracted domain metadata for AI consumption");
            // In a real scenario, this metadata would be consumed by external AI tools
        }
        Err(e) => {
            info!("AI metadata extraction failed (expected in example): {:?}", e);
        }
    }

    Ok(())
}

/// Example 4: Advanced capability-based discovery
async fn capability_based_discovery() -> CompilerResult<()> {
    info!("=== Example 4: Advanced Capability-Based Discovery ===");

    // This example demonstrates how the symbol management system enables
    // sophisticated capability-based discovery across modules and symbols

    // Create integrated system (abbreviated setup)
    let semantic_db = Arc::new(SemanticDatabase::new(&SemanticConfig::default())?);
    let symbol_table = Arc::new(SymbolTable::with_config(
        semantic_db.clone(),
        SymbolTableConfig::default(),
    )?);
    let scope_tree = Arc::new(ScopeTree::with_config(ScopeTreeConfig::default())?);
    let cache = Arc::new(CompilationCache::new(CacheConfig::default())?);
    let module_registry = Arc::new(SmartModuleRegistry::new(
        ModuleRegistryConfig::default(),
        symbol_table.clone(),
        scope_tree.clone(),
        semantic_db.clone(),
        cache.clone(),
    )?);

    let mut integrated_system = IntegratedSymbolSystemBuilder::new()
        .with_symbol_table(symbol_table.clone())
        .with_module_registry(module_registry.clone())
        .with_semantic_db(semantic_db.clone())
        .with_scope_tree(scope_tree.clone())
        .with_cache(cache.clone())
        .build()?;

    // Process a program with multiple modules
    let program = create_multi_module_program();
    let _ = integrated_system.process_program(&program).await?;

    // Demonstrate various discovery capabilities
    info!("Capability-based discovery results:");

    // Find authentication-related symbols
    let auth_symbols = integrated_system.find_symbols_by_capability("Authentication").await?;
    info!("  - Authentication symbols: {:?}", auth_symbols);

    // Find database-related symbols
    let db_symbols = integrated_system.find_symbols_by_capability("Database").await?;
    info!("  - Database symbols: {:?}", db_symbols);

    // Find symbols by business context
    let user_mgmt_symbols = integrated_system.find_symbols_by_business_context("user_management").await?;
    info!("  - User management symbols: {:?}", user_mgmt_symbols);

    // This demonstrates how the integrated symbol system enables:
    // 1. Cross-module symbol discovery
    // 2. Capability-based filtering
    // 3. Business context-aware search
    // 4. AI-ready metadata export

    Ok(())
}

/// Create a sample AST program for demonstration
fn create_sample_program() -> Program {
    use prism_ast::*;
    use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

    let source_id = SourceId::new(1);
    let dummy_span = Span::dummy();

    // Create a sample module with functions and types
    let module_decl = ModuleDecl {
        name: Symbol::intern("UserAuthentication"),
        version: Some("1.0.0".to_string()),
        description: Some("Handles user authentication and authorization".to_string()),
        stability: StabilityLevel::Stable,
        sections: vec![
            AstNode::new(
                SectionDecl {
                    kind: SectionKind::Interface,
                    items: vec![
                        // Sample function
                        AstNode::new(
                            Item::Function(FunctionDecl {
                                name: Symbol::intern("authenticate_user"),
                                parameters: vec![
                                    Parameter {
                                        name: Symbol::intern("username"),
                                        type_annotation: None,
                                        default_value: None,
                                        is_mutable: false,
                                    },
                                    Parameter {
                                        name: Symbol::intern("password"),
                                        type_annotation: None,
                                        default_value: None,
                                        is_mutable: false,
                                    },
                                ],
                                return_type: None,
                                body: None,
                                visibility: Visibility::Public,
                                attributes: Vec::new(),
                                contracts: None,
                            }),
                            dummy_span,
                            NodeId::new(2),
                        ),
                    ],
                },
                dummy_span,
                NodeId::new(1),
            ),
        ],
    };

    let module_item = AstNode::new(
        Item::Module(module_decl),
        dummy_span,
        NodeId::new(0),
    );

    Program {
        items: vec![module_item],
        source_id,
        metadata: ProgramMetadata::default(),
    }
}

/// Create a multi-module program for advanced examples
fn create_multi_module_program() -> Program {
    use prism_ast::*;
    use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};

    let source_id = SourceId::new(1);
    let dummy_span = Span::dummy();

    // Create multiple modules with different capabilities
    let auth_module = AstNode::new(
        Item::Module(ModuleDecl {
            name: Symbol::intern("Authentication"),
            version: Some("1.0.0".to_string()),
            description: Some("Authentication services".to_string()),
            stability: StabilityLevel::Stable,
            sections: vec![],
        }),
        dummy_span,
        NodeId::new(0),
    );

    let db_module = AstNode::new(
        Item::Module(ModuleDecl {
            name: Symbol::intern("Database"),
            version: Some("1.0.0".to_string()),
            description: Some("Database access layer".to_string()),
            stability: StabilityLevel::Stable,
            sections: vec![],
        }),
        dummy_span,
        NodeId::new(1),
    );

    let user_module = AstNode::new(
        Item::Module(ModuleDecl {
            name: Symbol::intern("UserManagement"),
            version: Some("1.0.0".to_string()),
            description: Some("User management functionality".to_string()),
            stability: StabilityLevel::Beta,
            sections: vec![],
        }),
        dummy_span,
        NodeId::new(2),
    );

    Program {
        items: vec![auth_module, db_module, user_module],
        source_id,
        metadata: ProgramMetadata::default(),
    }
} 