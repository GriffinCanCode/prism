//! Type Inference Engine Subsystem
//!
//! This subsystem implements the core type inference engine with proper separation
//! of concerns. Each module handles a specific aspect of type inference:
//!
//! - `core` - Main inference orchestration and coordination
//! - `orchestrator` - High-level inference workflow management
//! - `expression_inference` - Type inference for expressions
//! - `statement_inference` - Type inference for statements  
//! - `pattern_inference` - Type inference for patterns
//! - `builtins` - Built-in type and function management
//! - `compiler_integration` - Integration with compiler performance profiling
//! - `profiling` - Performance metrics and timing
//! - `pir_integration` - PIR metadata generation
//! - `effect_inference` - Effect analysis and inference
//! - `ast_resolution` - AST type to semantic type conversion
//!
//! ## Design Principles
//!
//! 1. **Single Responsibility** - Each module has one clear purpose
//! 2. **Interface Segregation** - Clean interfaces between modules
//! 3. **Dependency Inversion** - Depend on abstractions, not concretions
//! 4. **Open/Closed** - Open for extension, closed for modification

pub mod core;
pub mod orchestrator;
pub mod expression_inference;
pub mod statement_inference;
pub mod pattern_inference;
pub mod builtins;
pub mod compiler_integration;
pub mod profiling;
pub mod pir_integration;
pub mod effect_inference;
pub mod ast_resolution;

// Re-export main types for backwards compatibility
pub use core::{TypeInferenceEngine, InferenceConfig, InferenceResult};
pub use orchestrator::InferenceOrchestrator;
pub use expression_inference::ExpressionInferenceEngine;
pub use statement_inference::StatementInferenceEngine;
pub use pattern_inference::PatternInferenceEngine;
pub use builtins::BuiltinTypeManager;
pub use compiler_integration::CompilerIntegration;
pub use profiling::InferenceProfiler;
pub use pir_integration::PIRMetadataGenerator;
pub use effect_inference::EffectInferenceEngine;
pub use ast_resolution::ASTTypeResolver;

use crate::SemanticResult;
use prism_common::{NodeId, Span};
use std::collections::HashMap;

/// Common interface for all inference engines
pub trait InferenceEngine {
    /// The input type for this inference engine
    type Input;
    /// The output type for this inference engine
    type Output;
    
    /// Perform inference on the input
    fn infer(&mut self, input: Self::Input) -> SemanticResult<Self::Output>;
    
    /// Get the name of this inference engine for profiling
    fn engine_name(&self) -> &'static str;
    
    /// Reset the engine state
    fn reset(&mut self);
}

/// Common interface for engines that need compiler integration
pub trait CompilerIntegratable {
    /// Report performance metrics to the compiler
    fn report_performance(&self, metrics: profiling::PerformanceMetrics);
    
    /// Get current performance statistics
    fn get_performance_stats(&self) -> profiling::PerformanceStats;
}

/// Common interface for engines that generate PIR metadata
pub trait PIRIntegratable {
    /// Generate PIR-compatible metadata
    fn generate_pir_metadata(&self) -> SemanticResult<pir_integration::PIRMetadata>;
}

/// Common interface for engines that need effect analysis
pub trait EffectAware: InferenceEngine {
    /// Infer effects from the given input
    fn infer_effects(&self, input: &Self::Input) -> SemanticResult<Vec<String>>;
} 