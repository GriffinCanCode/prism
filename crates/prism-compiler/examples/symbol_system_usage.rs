//! PLT-004 Symbol Table & Scope Resolution - Usage Example
//!
//! This example demonstrates how to use the integrated symbol table system
//! following Prism's principles of Conceptual Cohesion and proper integration
//! with existing infrastructure.

use prism_compiler::{
    CompilerResult, CompilerConfig,
    SymbolSystem, SymbolSystemConfig,
    SymbolData, SymbolKind, SymbolBuilder,
    ScopeKind, ResolutionContext,
    SemanticDatabase, CompilationCache, CacheConfig,
};
use prism_compiler::symbols::data::SymbolVisibility;
use prism_compiler::symbols::kinds::TypeCategory;
use prism_common::{span::Span, symbol::Symbol};
use prism_effects::effects::EffectRegistry;
use std::sync::Arc;

/// Example: Building and using the symbol table system
#[tokio::main]
async fn main() -> CompilerResult<()> {
    println!("🔧 PLT-004 Symbol Table System - Usage Example");
    println!("================================================");

    // Step 1: Setup required dependencies
    println!("\n📋 Step 1: Setting up dependencies...");
    let dependencies = setup_dependencies().await?;
    println!("   ✅ Semantic database initialized");
    println!("   ✅ Effect registry initialized");
    println!("   ✅ Compilation cache initialized");

    // Step 2: Build the integrated symbol system
    println!("\n🔨 Step 2: Building symbol system...");
    let symbol_system = SymbolSystem::builder()
        .with_semantic_database(dependencies.semantic_db)
        .with_effect_registry(dependencies.effect_registry)
        .with_cache(dependencies.cache)
        .with_config(create_custom_config())
        .build()?;
    println!("   ✅ Symbol system built successfully");

    // Step 3: Register some example symbols
    println!("\n📝 Step 3: Registering symbols...");
    register_example_symbols(&symbol_system)?;
    println!("   ✅ Example symbols registered");

    // Step 4: Demonstrate symbol queries
    println!("\n🔍 Step 4: Querying symbols...");
    demonstrate_symbol_queries(&symbol_system)?;

    // Step 5: Demonstrate symbol resolution
    println!("\n🎯 Step 5: Demonstrating resolution...");
    demonstrate_symbol_resolution(&symbol_system).await?;

    // Step 6: Show system statistics
    println!("\n📊 Step 6: System statistics...");
    show_system_statistics(&symbol_system)?;

    // Step 7: Validate system consistency
    println!("\n✅ Step 7: Validating system...");
    validate_system(&symbol_system)?;

    println!("\n🎉 Example completed successfully!");
    Ok(())
}

/// Dependencies needed for the symbol system
struct Dependencies {
    semantic_db: Arc<SemanticDatabase>,
    effect_registry: Arc<EffectRegistry>,
    cache: Arc<CompilationCache>,
}

/// Setup all required dependencies
async fn setup_dependencies() -> CompilerResult<Dependencies> {
    // Setup semantic database
    let compiler_config = CompilerConfig::default();
    let semantic_db = Arc::new(SemanticDatabase::new(&compiler_config)?);

    // Setup effect registry
    let effect_registry = Arc::new(EffectRegistry::new());

    // Setup compilation cache
    let cache_config = CacheConfig::default();
    let cache = Arc::new(CompilationCache::new(cache_config)?);

    Ok(Dependencies {
        semantic_db,
        effect_registry,
        cache,
    })
}

/// Create custom configuration for the symbol system
fn create_custom_config() -> SymbolSystemConfig {
    SymbolSystemConfig {
        enable_ai_metadata: true,
        enable_validation: true,
        ..Default::default()
    }
}

/// Register example symbols to demonstrate the system
fn register_example_symbols(symbol_system: &SymbolSystem) -> CompilerResult<()> {
    let symbol_table = symbol_system.symbol_table();

    // Register a module symbol using the builder API
    let module_symbol = SymbolBuilder::module("UserManagement", Span::dummy())
        .public()
        .with_responsibility("User account lifecycle and authentication management")
        .with_capability("UserDatabase")
        .build()?;
    symbol_table.register_symbol(module_symbol)?;

    // Register a function symbol using the builder API
    let function_symbol = SymbolBuilder::function("authenticateUser", Span::dummy())
        .public()
        .with_responsibility("Verify user credentials securely")
        .with_capability("UserDatabase")
        .with_capability("PasswordHashing")
        .build()?;
    symbol_table.register_symbol(function_symbol)?;

    // Register a type symbol using the builder API
    use prism_compiler::symbols::kinds::{TypeCategory, PrimitiveType};
    let semantic_type_category = TypeCategory::Semantic {
        base_type: "UserRecord".to_string(),
        domain: "UserManagement".to_string(),
        constraints: vec!["email.is_validated()".to_string(), "id.is_unique()".to_string()],
        business_rules: Vec::new(),
    };
    
    let type_symbol = SymbolBuilder::type_symbol("User", Span::dummy(), semantic_type_category)
        .public()
        .with_responsibility("Store user identity with validation constraints")
        .build()?;
    symbol_table.register_symbol(type_symbol)?;

    println!("   📦 Registered module: UserManagement");
    println!("   🔧 Registered function: authenticateUser");
    println!("   📋 Registered type: User");

    Ok(())
}

/// Demonstrate various symbol queries
fn demonstrate_symbol_queries(symbol_system: &SymbolSystem) -> CompilerResult<()> {
    // Query all functions
    let functions = symbol_system.find_functions();
    println!("   🔍 Found {} function(s):", functions.len());
    for func in &functions {
        println!("     - {}: {}", func.name, func.responsibility.as_deref().unwrap_or("No responsibility"));
    }

    // Query all modules
    let modules = symbol_system.find_modules();
    println!("   🔍 Found {} module(s):", modules.len());
    for module in &modules {
        println!("     - {}: {}", module.name, module.responsibility.as_deref().unwrap_or("No responsibility"));
    }

    // Query all types
    let types = symbol_system.find_types();
    println!("   🔍 Found {} type(s):", types.len());
    for type_symbol in &types {
        println!("     - {}: {}", type_symbol.name, type_symbol.responsibility.as_deref().unwrap_or("No responsibility"));
    }

    // Query by visibility
    let public_symbols = symbol_system.symbol_table().get_symbols_by_visibility(SymbolVisibility::Public);
    println!("   🔍 Found {} public symbol(s)", public_symbols.len());

    Ok(())
}

/// Demonstrate symbol resolution
async fn demonstrate_symbol_resolution(symbol_system: &SymbolSystem) -> CompilerResult<()> {
    // Create a resolution context
    let context = ResolutionContext {
        current_scope: None, // Would be set in real compilation
        current_module: Some("UserManagement".to_string()),
        available_capabilities: vec![
            "UserDatabase".to_string(),
            "PasswordHashing".to_string(),
            "Authentication".to_string(),
        ],
        effect_context: Some("secure_context".to_string()),
        syntax_style: "canonical".to_string(),
        preferences: Default::default(),
    };

    // Try to resolve symbols
    let symbols_to_resolve = vec!["authenticateUser", "User", "UserManagement", "nonexistent"];

    for symbol_name in symbols_to_resolve {
        match symbol_system.resolve_symbol_with_context(symbol_name, &context).await {
            Ok(resolved) => {
                println!("   ✅ Resolved '{}' via {:?} with confidence {:.2}", 
                         symbol_name, resolved.resolution_kind, resolved.confidence);
                if let Some(ai_metadata) = &resolved.ai_metadata {
                    println!("      💡 AI: {}", ai_metadata);
                }
            }
            Err(e) => {
                println!("   ❌ Failed to resolve '{}': {}", symbol_name, e);
            }
        }
    }

    Ok(())
}

/// Show system statistics
fn show_system_statistics(symbol_system: &SymbolSystem) -> CompilerResult<()> {
    let stats = symbol_system.stats();

    println!("   📊 Symbol Table:");
    println!("      - Total symbols: {}", stats.symbol_table.total_symbols);
    println!("      - By kind: {:?}", stats.symbol_table.symbols_by_kind);
    println!("      - By visibility: {:?}", stats.symbol_table.symbols_by_visibility);

    println!("   📊 Scope Tree:");
    println!("      - Total scopes: {}", stats.scope_tree.total_scopes);
    println!("      - Max depth: {}", stats.scope_tree.max_depth);
    println!("      - Average symbols per scope: {:.2}", stats.scope_tree.average_symbols_per_scope);

    println!("   📊 Resolver:");
    println!("      - Total resolutions: {}", stats.resolver.total_resolutions);
    println!("      - Cache hits: {}", stats.resolver.cache_hits);
    println!("      - Cache misses: {}", stats.resolver.cache_misses);

    Ok(())
}

/// Validate system consistency
fn validate_system(symbol_system: &SymbolSystem) -> CompilerResult<()> {
    let validation = symbol_system.validate()?;

    if validation.is_valid {
        println!("   ✅ System validation passed");
    } else {
        println!("   ⚠️  System validation found issues:");
        for issue in &validation.issues {
            println!("      - ❌ {}", issue);
        }
    }

    if !validation.warnings.is_empty() {
        println!("   ⚠️  Warnings:");
        for warning in &validation.warnings {
            println!("      - ⚠️  {}", warning);
        }
    }

    Ok(())
} 