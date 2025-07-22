//! Unified Optimization Interfaces
//!
//! This module defines common optimization traits and interfaces that can be used
//! across all code generation backends to prevent duplicate responsibilities and
//! ensure consistent optimization behavior.

use crate::{Result as PrismResult, PrismError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common optimization configuration that can be specialized per backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization level (0-3)
    pub level: u8,
    /// Enable minification
    pub minify: bool,
    /// Enable tree shaking
    pub tree_shaking: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable function inlining
    pub function_inlining: bool,
    /// Remove debug statements in production
    pub remove_debug_statements: bool,
    /// Backend-specific optimization options
    pub backend_specific: HashMap<String, String>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: 2,
            minify: true,
            tree_shaking: true,
            dead_code_elimination: true,
            constant_folding: true,
            function_inlining: false, // Conservative default
            remove_debug_statements: true,
            backend_specific: HashMap::new(),
        }
    }
}

/// Optimization statistics that all backends should track
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Number of constant folding operations
    pub constant_foldings: usize,
    /// Number of dead code eliminations
    pub dead_code_eliminations: usize,
    /// Number of function inlinings
    pub function_inlinings: usize,
    /// Number of tree shaking operations
    pub tree_shakings: usize,
    /// Original code size in bytes
    pub original_size: usize,
    /// Optimized code size in bytes
    pub optimized_size: usize,
    /// Optimization time in milliseconds
    pub optimization_time_ms: u64,
}

impl OptimizationStats {
    /// Calculate optimization ratio (0.0 to 1.0, where 1.0 is no reduction)
    pub fn optimization_ratio(&self) -> f64 {
        if self.original_size == 0 {
            1.0
        } else {
            self.optimized_size as f64 / self.original_size as f64
        }
    }
    
    /// Calculate size reduction percentage
    pub fn size_reduction_percentage(&self) -> f64 {
        (1.0 - self.optimization_ratio()) * 100.0
    }
}

/// Result of an optimization operation
#[derive(Debug, Clone)]
pub struct OptimizationResult<T> {
    /// The optimized output
    pub output: T,
    /// Optimization statistics
    pub stats: OptimizationStats,
    /// Warnings generated during optimization
    pub warnings: Vec<OptimizationWarning>,
}

/// Optimization warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Location information (optional)
    pub location: Option<String>,
}

/// Common trait for all code optimizers
pub trait CodeOptimizer<T>: Send + Sync {
    /// The error type for this optimizer
    type Error: Into<PrismError>;
    
    /// Optimize the given code
    fn optimize(&mut self, input: &T, config: &OptimizationConfig) -> Result<OptimizationResult<T>, Self::Error>;
    
    /// Get optimizer capabilities
    fn capabilities(&self) -> OptimizerCapabilities;
    
    /// Check if optimization is applicable to the input
    fn is_applicable(&self, input: &T) -> bool;
    
    /// Get current optimization statistics
    fn get_stats(&self) -> &OptimizationStats;
    
    /// Reset optimization statistics
    fn reset_stats(&mut self);
}

/// Optimizer capabilities
#[derive(Debug, Clone)]
pub struct OptimizerCapabilities {
    /// Supports minification
    pub supports_minification: bool,
    /// Supports tree shaking
    pub supports_tree_shaking: bool,
    /// Supports dead code elimination
    pub supports_dead_code_elimination: bool,
    /// Supports constant folding
    pub supports_constant_folding: bool,
    /// Supports function inlining
    pub supports_function_inlining: bool,
    /// Maximum optimization level supported
    pub max_optimization_level: u8,
    /// Target-specific capabilities
    pub target_specific: HashMap<String, bool>,
}

/// Bundle size analysis (common across JavaScript-like targets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleAnalysis {
    /// Original size in bytes
    pub original_size: usize,
    /// Minified size in bytes
    pub minified_size: usize,
    /// Estimated gzip size in bytes
    pub gzip_size: usize,
    /// Optimization ratio (0.0 to 1.0)
    pub optimization_ratio: f64,
    /// Breakdown by optimization type
    pub optimization_breakdown: HashMap<String, usize>,
}

/// Trait for bundle analysis (JavaScript, TypeScript)
pub trait BundleAnalyzer<T>: Send + Sync {
    /// Analyze bundle size and optimization potential
    fn analyze_bundle(&self, code: &T) -> BundleAnalysis;
    
    /// Estimate gzip compression ratio
    fn estimate_gzip_size(&self, code: &T) -> usize;
}

/// Performance optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHint {
    /// Hint type (e.g., "memory", "cpu", "io")
    pub hint_type: String,
    /// Hint message
    pub message: String,
    /// Severity level (0-3)
    pub severity: u8,
    /// Code location (optional)
    pub location: Option<String>,
}

/// Trait for performance hint generation
pub trait PerformanceHintGenerator<T>: Send + Sync {
    /// Generate performance hints for the given code
    fn generate_hints(&self, code: &T, config: &OptimizationConfig) -> Vec<PerformanceHint>;
} 