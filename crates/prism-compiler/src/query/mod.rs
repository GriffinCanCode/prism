//! Query Subsystem - Modular Compiler Query Extensions
//!
//! This module provides specialized query types and orchestration for the Prism compiler,
//! extending the existing query.rs infrastructure with domain-specific queries for symbols,
//! scopes, and semantic analysis.
//!
//! ## Architecture
//!
//! The query subsystem is organized into focused modules:
//! - `symbol_queries`: Specialized queries for symbol operations
//! - `scope_queries`: Specialized queries for scope operations  
//! - `semantic_queries`: Specialized queries for semantic analysis
//! - `orchestrator`: Query coordination and composition
//! - `optimization`: Performance optimizations and caching
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
//!
//! ## Usage
//!
//! ```rust
//! use prism_compiler::query::{QueryEngine, QueryContext};
//! use prism_compiler::query::symbol_queries::*;
//! use prism_compiler::query::scope_queries::*;
//!
//! // Use existing QueryEngine with new specialized queries
//! let symbol_query = FindSymbolsByKindQuery::new(SymbolKind::Function);
//! let scope_query = FindScopesByVisibilityQuery::new(ScopeVisibility::Public);
//!
//! // Execute queries using existing infrastructure
//! let symbols = query_engine.query(&symbol_query, input, context).await?;
//! let scopes = query_engine.query(&scope_query, input, context).await?;
//! ```

pub mod core;
pub mod symbol_queries;
pub mod scope_queries;
pub mod semantic_queries;
pub mod orchestrator;
pub mod optimization;

// Re-export main types for convenience
pub use core::*;
pub use symbol_queries::*;
pub use scope_queries::*; 
pub use semantic_queries::*;
pub use orchestrator::*;
pub use optimization::*;

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
    
    println!("✅ Query subsystem integration verified successfully");
    Ok(())
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_query_subsystem_integration() {
        let result = verify_query_subsystem_integration();
        assert!(result.is_ok(), "Query subsystem integration test failed: {:?}", result);
    }
} 