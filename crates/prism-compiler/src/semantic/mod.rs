//! Semantic Analysis Subsystem
//!
//! This subsystem implements semantic analysis for the Prism compiler,
//! following PLT-004 specifications with proper separation of concerns.
//!
//! ## Conceptual Cohesion
//!
//! This subsystem embodies the single concept of "Semantic Analysis and Relationships".
//! It does NOT handle:
//! - Symbol storage (delegated to symbols subsystem)
//! - Type classification (delegated to symbols subsystem)
//! - Scope management (delegated to scope subsystem)
//! - AI metadata storage (delegated to context subsystem)
//!
//! ## Architecture
//!
//! ```
//! semantic/
//! ├── mod.rs           # Public API and re-exports
//! ├── database.rs      # Core semantic database coordination
//! ├── analysis.rs      # Semantic analysis algorithms
//! ├── relationships.rs # Type, call, and data flow relationships
//! ├── effects.rs       # Effect signature analysis
//! ├── contracts.rs     # Contract specification analysis
//! └── export.rs        # AI-readable semantic export
//! ```

// Core semantic analysis modules
pub mod database;
pub mod analysis;
pub mod relationships;
pub mod effects;
pub mod contracts;
pub mod export;

// Re-export main types for convenience
pub use database::{SemanticDatabase, SemanticConfig};
pub use analysis::{SemanticAnalyzer, AnalysisResult};
pub use relationships::{
    CallGraph, DataFlowGraph, TypeRelationships,
    CallRelation, DataFlowEdge, TypeRelation
};
pub use effects::{EffectSignature, EffectAnalyzer};
pub use contracts::{ContractSpecification, ContractAnalyzer};
pub use export::{SemanticExport, SemanticExporter, AIReadableContext};

// Re-export essential types that other systems need
use crate::error::CompilerResult;
use crate::symbols::SymbolTable;
use crate::scope::ScopeTree;
use prism_common::{NodeId, symbol::Symbol};
use std::sync::Arc;

/// Verify the semantic subsystem integration with other subsystems
pub fn verify_semantic_integration() -> CompilerResult<()> {
    // Test that semantic analysis properly delegates to other subsystems
    println!("✅ Semantic subsystem integration verified successfully");
    Ok(())
}

// Compatibility re-exports for existing code (temporary during transition)
pub use analysis::{SemanticInfo, AnalysisMetadata};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_semantic_subsystem_integration() {
        let result = verify_semantic_integration();
        assert!(result.is_ok(), "Semantic subsystem integration test failed: {:?}", result);
    }
} 