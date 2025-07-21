//! PIR Construction Subsystem - Query-Based Incremental PIR Building
//!
//! This subsystem implements efficient PIR construction using the established CompilerQuery
//! interface from prism-compiler. It provides:
//!
//! ## Design Philosophy
//!
//! **Separation of Concerns:**
//! - `queries/` - PIR-specific CompilerQuery implementations
//! - `builder/` - High-level construction orchestration  
//! - `phases/` - Multi-phase transformation pipeline
//! - `cache/` - PIR-specific caching strategies
//! - `validation/` - Construction-time semantic validation
//!
//! **Integration Strategy:**
//! - Reuses existing `CompilerQuery` trait and `QueryEngine` from prism-compiler
//! - Implements PIR-specific queries that compose naturally
//! - Provides incremental compilation through query dependencies
//! - Maintains semantic preservation throughout construction
//!
//! ## Architecture Inspiration
//!
//! Inspired by architecture of Rust compiler (HIR/MIR), Swift (SIL), and LLVM IR:
//! - **Query-Driven**: All PIR construction is demand-driven through queries
//! - **Incremental**: Only rebuild what changed based on dependencies
//! - **Composable**: Simple queries compose into complex transformations
//! - **Cached**: Aggressive caching with semantic-aware invalidation
//! - **Parallel**: Natural parallelization through independent queries

pub mod queries;
pub mod builder;  
pub mod phases;
pub mod cache;
pub mod validation;
pub mod transformation;
pub mod compiler_integration;
pub mod semantic_preservation;
pub mod business_extraction;
pub mod effect_integration;
pub mod ai_extraction;

// Re-export main types
pub use builder::{PIRConstructionBuilder, ConstructionConfig, ConstructionResult};
pub use queries::{
    ASTToPIRQuery, PIRModuleQuery, PIRFunctionQuery, PIRTypeQuery,
    PIRValidationQuery, PIROptimizationQuery
};
pub use phases::{
    ConstructionPhase, SemanticExtractionPhase, BusinessContextPhase,
    EffectAnalysisPhase, ValidationPhase, OptimizationPhase
};
pub use cache::{PIRCacheStrategy, PIRCacheKey, PIRCacheManager};
pub use validation::{
    ConstructionValidator, ValidationHook, ConstructionValidationResult,
    ConstructionValidationContext, ConstructionValidationConfig,
    ConstructionValidationFinding, ConstructionValidationSeverity, ConstructionValidationCategory
}; 