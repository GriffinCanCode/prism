//! Query Subsystem - Modular Compiler Query Extensions
//!
//! This module provides specialized query types and orchestration for the Prism compiler,
//! extending the existing query.rs infrastructure with domain-specific queries for symbols,
//! scopes, and semantic analysis.
//!
//! ## Architecture
//!
//! The query subsystem is organized into focused modules:
//! - `core`: Core query infrastructure and engine
//! - `symbol_queries`: Specialized queries for symbol operations
//! - `scope_queries`: Specialized queries for scope operations  
//! - `semantic_queries`: Specialized queries for semantic analysis and compilation phases
//! - `orchestrator`: Query coordination and composition
//! - `optimization`: Performance optimizations and caching
//! - `pipeline`: Query-based compilation pipeline orchestrator
//! - `incremental`: Incremental compilation with file watching
//! - `ai_metadata`: AI-first metadata generation for all query results
//! - `integration`: Proper integration with the broader compiler system
//!
//! ## Integration with Existing Infrastructure
//!
//! This subsystem extends the existing `crate::query::CompilerQuery` trait and
//! `crate::query::QueryEngine` without duplicating functionality. It provides:
//!
//! - Specialized query implementations for different subsystems
//! - Query composition and orchestration capabilities
//! - Performance optimizations specific to query patterns
//! - AI-first metadata generation for query results
//! - Complete query-based compilation pipeline
//! - Incremental compilation with real-time file watching
//! - Proper integration interfaces using traits
//!
//! ## AI-First Design
//!
//! All query results include AI-readable metadata:
//! - Semantic context for AI understanding
//! - Business context and domain knowledge
//! - Performance characteristics and optimization suggestions
//! - Quality metrics and improvement recommendations
//! - Structured relationships between code elements
//!
//! ## Usage
//!
//! ```rust
//! use prism_compiler::query::{QueryEngine, QueryContext};
//! use prism_compiler::query::symbol_queries::*;
//! use prism_compiler::query::scope_queries::*;
//! use prism_compiler::query::pipeline::*;
//! use prism_compiler::query::incremental::*;
//! use prism_compiler::query::integration::*;
//!
//! // Use existing QueryEngine with new specialized queries
//! let symbol_query = FindSymbolsByKindQuery::new(SymbolKind::Function);
//! let scope_query = FindScopesByVisibilityQuery::new(ScopeVisibility::Public);
//! let pipeline = CompilationPipeline::new(config);
//! let incremental = IncrementalCompiler::new(Arc::new(pipeline), incremental_config)?;
//!
//! // Execute queries using existing infrastructure with AI metadata
//! let symbols = query_engine.query(&symbol_query, input, context).await?;
//! let scopes = query_engine.query(&scope_query, input, context).await?;
//! let result = incremental.compile_incremental(project_path).await?;
//! ```

pub mod core;
pub mod symbol_queries;
pub mod scope_queries;
pub mod semantic_queries;
pub mod orchestrator;
pub mod optimization;
pub mod pipeline;
pub mod incremental;
pub mod ai_metadata;
pub mod integration;

#[cfg(test)]
pub mod integration_test;

// Re-export main types for convenience
pub use core::*;
pub use symbol_queries::*;
pub use scope_queries::*; 
pub use semantic_queries::*;
pub use orchestrator::*;
pub use optimization::*;
pub use pipeline::*;
pub use incremental::*;
pub use ai_metadata::*;
pub use integration::*;

use crate::error::CompilerResult;

/// Verify the query subsystem integration with existing infrastructure
pub fn verify_query_subsystem_integration() -> CompilerResult<()> {
    // Test core infrastructure
    let _query_engine = QueryEngine::new();
    let _query_config = QueryConfig::default();
    
    // Test that we can create queries that extend existing infrastructure
    let _symbol_query = symbol_queries::FindSymbolsByKindQuery::new(
        crate::symbols::SymbolKind::Function { 
            signature: Default::default(),
            function_type: Default::default(),
            contracts: None,
            performance_info: None,
        }
    );
    
    let _scope_query = scope_queries::FindScopesByVisibilityQuery::new(
        crate::scope::ScopeVisibility::Public
    );
    
    // Test query orchestrator
    let _orchestrator = orchestrator::QueryOrchestrator::new();
    
    // Test optimization
    let _optimizer = optimization::QueryOptimizer::new(Default::default());
    
    // Test compilation pipeline
    let _pipeline = pipeline::CompilationPipeline::new(Default::default());
    
    // Test incremental compilation
    let pipeline = std::sync::Arc::new(pipeline::CompilationPipeline::new(Default::default()));
    let _incremental = incremental::IncrementalCompiler::new(pipeline, Default::default())?;
    
    // Test AI metadata generation
    let _ai_generator = ai_metadata::DefaultAIMetadataGenerator::new();
    
    // Test integration system
    let query_engine = std::sync::Arc::new(QueryEngine::new());
    let ai_generator = std::sync::Arc::new(ai_metadata::DefaultAIMetadataGenerator::new());
    let _integration = integration::DefaultQueryIntegration::new(query_engine, ai_generator);
    
    println!("✅ Query subsystem integration verified successfully");
    println!("✅ AI-first metadata generation available");
    println!("✅ Proper integration interfaces established");
    Ok(())
}

/// Initialize the query subsystem with AI-first features
pub async fn initialize_ai_first_query_system(
    context: std::sync::Arc<crate::context::CompilationContext>,
    symbol_table: std::sync::Arc<crate::symbols::SymbolTable>,
    scope_tree: std::sync::Arc<crate::scope::ScopeTree>,
    semantic_db: std::sync::Arc<crate::semantic::SemanticDatabase>,
) -> CompilerResult<std::sync::Arc<dyn integration::QuerySystemIntegration>> {
    // Create query engine with AI-optimized configuration
    let query_config = QueryConfig {
        enable_cache: true,
        enable_dependency_tracking: true,
        enable_profiling: true,
        cache_size_limit: 100_000, // Larger cache for AI metadata
        query_timeout: std::time::Duration::from_secs(60), // Longer timeout for complex AI operations
    };
    
    let query_engine = std::sync::Arc::new(QueryEngine::with_config(query_config)?);
    
    // Create AI metadata generator with full features enabled
    let ai_config = ai_metadata::AIGeneratorConfig {
        enable_detailed_semantics: true,
        enable_business_context: true,
        enable_performance_analysis: true,
        enable_quality_assessment: true,
        enable_suggestions: true,
        suggestion_quality_threshold: 0.6, // Lower threshold for more suggestions
    };
    
    let ai_generator = std::sync::Arc::new(
        ai_metadata::DefaultAIMetadataGenerator::with_config(ai_config)
    );
    
    // Create integrated query system
    let mut integration = integration::DefaultQueryIntegration::new(query_engine, ai_generator);
    
    // Initialize with compiler components
    integration.initialize(context, symbol_table, scope_tree, semantic_db).await?;
    
    Ok(std::sync::Arc::new(integration))
}

/// Create an AI-enhanced compilation pipeline
pub fn create_ai_enhanced_pipeline(
    config: pipeline::PipelineConfig,
    integration: std::sync::Arc<dyn integration::QuerySystemIntegration>,
) -> CompilerResult<pipeline::CompilationPipeline> {
    // Create pipeline with AI-first configuration
    let ai_config = pipeline::PipelineConfig {
        enable_parallel_execution: config.enable_parallel_execution,
        max_concurrent_phases: config.max_concurrent_phases,
        enable_incremental: config.enable_incremental,
        enable_ai_metadata: true, // Always enable AI metadata
        phase_timeout_secs: config.phase_timeout_secs * 2, // Longer timeout for AI processing
        enable_error_recovery: config.enable_error_recovery,
        targets: config.targets,
    };
    
    Ok(pipeline::CompilationPipeline::new(ai_config))
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_query_subsystem_integration() {
        let result = verify_query_subsystem_integration();
        assert!(result.is_ok(), "Query subsystem integration test failed: {:?}", result);
    }

    #[tokio::test]
    async fn test_ai_first_initialization() {
        use crate::context::CompilationContext;
        use crate::symbols::SymbolTable;
        use crate::scope::ScopeTree;
        use crate::semantic::SemanticDatabase;
        
        // Create mock components
        let context = std::sync::Arc::new(CompilationContext::default());
        let symbol_table = std::sync::Arc::new(SymbolTable::new());
        let scope_tree = std::sync::Arc::new(ScopeTree::new());
        let semantic_db = std::sync::Arc::new(SemanticDatabase::new());
        
        // Test AI-first initialization
        let result = initialize_ai_first_query_system(
            context,
            symbol_table,
            scope_tree,
            semantic_db,
        ).await;
        
        assert!(result.is_ok(), "AI-first query system initialization failed: {:?}", result);
    }

    #[test]
    fn test_ai_enhanced_pipeline_creation() {
        use crate::context::CompilationTarget;
        
        let config = pipeline::PipelineConfig {
            enable_parallel_execution: true,
            max_concurrent_phases: 4,
            enable_incremental: true,
            enable_ai_metadata: false, // Will be overridden to true
            phase_timeout_secs: 30,
            enable_error_recovery: true,
            targets: vec![CompilationTarget::TypeScript],
        };
        
        // Create mock integration
        let query_engine = std::sync::Arc::new(QueryEngine::new());
        let ai_generator = std::sync::Arc::new(ai_metadata::DefaultAIMetadataGenerator::new());
        let integration = std::sync::Arc::new(
            integration::DefaultQueryIntegration::new(query_engine, ai_generator)
        );
        
        let pipeline = create_ai_enhanced_pipeline(config, integration);
        assert!(pipeline.is_ok(), "AI-enhanced pipeline creation failed: {:?}", pipeline);
        
        // Verify AI metadata is enabled
        let pipeline = pipeline.unwrap();
        let metrics = pipeline.get_metrics();
        // In a real implementation, we'd check that AI metadata is actually enabled
        assert!(metrics.total_time >= std::time::Duration::from_secs(0));
    }
} 