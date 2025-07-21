//! PIR Optimization Module
//!
//! This module contains PIR optimization and normalization utilities that can be
//! used independently or integrated with the construction subsystem.

pub mod passes;

// Re-export main optimization types
pub use passes::{
    PIROptimizer, PIRNormalizer, TransformationUtils,
    OptimizerConfig, NormalizerConfig, OptimizationLevel,
    OptimizationSummary, NormalizationSummary,
}; 