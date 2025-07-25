//! Modern JIT Static Analysis Infrastructure
//!
//! This module provides comprehensive static analysis capabilities for the JIT compiler,
//! with a unified architecture that ensures consistency and proper interoperability.
//!
//! ## Architecture
//!
//! The analysis system is built around a pipeline architecture with shared types and interfaces:
//!
//! - [`shared`] - Common data structures, traits, and interfaces
//! - [`pipeline`] - Analysis pipeline coordinator with dependency management
//! - [`control_flow`] - Control flow graph construction and analysis
//! - [`data_flow`] - Data flow analysis including liveness and reaching definitions
//! - [`loop_analysis`] - Loop detection, nesting analysis, and optimization opportunities
//! - [`type_analysis`] - Type inference and propagation analysis
//! - [`effect_analysis`] - Effect system analysis for optimization safety
//! - [`hotness`] - Hotness analysis and profiling data integration
//! - [`capability_analysis`] - Capability flow analysis for security
//!
//! ## Usage
//!
//! ```rust,ignore
//! use crate::execution::jit::analysis::{pipeline::*, shared::*};
//! 
//! // Create pipeline with configuration
//! let config = PipelineConfig::default();
//! let mut pipeline = AnalysisPipeline::new(config);
//! 
//! // Register analyzers
//! pipeline.register_analyzer(ControlFlowAnalyzer::new(&config)?)?;
//! pipeline.register_analyzer(DataFlowAnalyzer::new(&config)?)?;
//! // ... register other analyzers
//! 
//! // Analyze function
//! let context = pipeline.analyze_function(function)?;
//! let opportunities = context.optimization_opportunities;
//! ```

// Core infrastructure
pub mod shared;
pub mod pipeline;

// Analysis modules
pub mod control_flow;
pub mod data_flow;
pub mod loop_analysis;
pub mod type_analysis;
pub mod effect_analysis;
pub mod hotness;
pub mod capability_analysis;
pub mod capability_aware_inlining;

use crate::{VMResult, PrismVMError, bytecode::FunctionDefinition};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Re-export core infrastructure
pub use shared::*;
pub use pipeline::{AnalysisPipeline, PipelineConfig, AnalysisContext, OptimizationConfig};

/// Legacy configuration structure for backward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable control flow analysis
    pub enable_cfg_analysis: bool,
    /// Enable data flow analysis
    pub enable_dataflow_analysis: bool,
    /// Enable loop analysis
    pub enable_loop_analysis: bool,
    /// Enable type analysis
    pub enable_type_analysis: bool,
    /// Enable effect analysis
    pub enable_effect_analysis: bool,
    /// Enable hotness analysis
    pub enable_hotness_analysis: bool,
    /// Enable optimization opportunity detection
    pub enable_optimization_detection: bool,
    /// Enable capability analysis
    pub enable_capability_analysis: bool,
    /// Analysis timeout
    pub analysis_timeout: Duration,
    /// Maximum analysis iterations
    pub max_iterations: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_cfg_analysis: true,
            enable_dataflow_analysis: true,
            enable_loop_analysis: true,
            enable_type_analysis: true,
            enable_effect_analysis: true,
            enable_hotness_analysis: true,
            enable_optimization_detection: true,
            enable_capability_analysis: true,
            analysis_timeout: Duration::from_millis(500),
            max_iterations: 100,
        }
    }
}

impl From<AnalysisConfig> for PipelineConfig {
    fn from(config: AnalysisConfig) -> Self {
        let mut enabled_analyses = std::collections::HashSet::new();
        
        if config.enable_cfg_analysis {
            enabled_analyses.insert(AnalysisKind::ControlFlow);
        }
        if config.enable_dataflow_analysis {
            enabled_analyses.insert(AnalysisKind::DataFlow);
        }
        if config.enable_loop_analysis {
            enabled_analyses.insert(AnalysisKind::Loop);
        }
        if config.enable_type_analysis {
            enabled_analyses.insert(AnalysisKind::Type);
        }
        if config.enable_effect_analysis {
            enabled_analyses.insert(AnalysisKind::Effect);
        }
        if config.enable_hotness_analysis {
            enabled_analyses.insert(AnalysisKind::Hotness);
        }
        if config.enable_capability_analysis {
            enabled_analyses.insert(AnalysisKind::Capability);
        
        // Register capability-aware inlining analyzer
        enabled_analyses.insert(AnalysisKind::CapabilityAwareInlining);
        }

        PipelineConfig {
            enabled_analyses,
            analysis_timeout: config.analysis_timeout,
            max_iterations: config.max_iterations,
            enable_parallel: true,
            optimization_config: OptimizationConfig {
                enable_detection: config.enable_optimization_detection,
                min_benefit_threshold: 0.05,
                max_cost_threshold: 0.2,
                enable_speculative: true,
                aggressiveness: 0.5,
            },
        }
    }
}

/// Comprehensive static analyzer - now a wrapper around the pipeline
pub struct StaticAnalyzer {
    /// Internal analysis pipeline
    pipeline: AnalysisPipeline,
}

/// Legacy analysis result structure for backward compatibility
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Function being analyzed
    pub function_id: u32,
    /// Control flow graph
    pub cfg: Option<control_flow::ControlFlowGraph>,
    /// Data flow analysis results
    pub dataflow: Option<data_flow::DataFlowAnalysis>,
    /// Loop analysis results
    pub loops: Option<loop_analysis::LoopAnalysis>,
    /// Type analysis results
    pub types: Option<type_analysis::TypeAnalysis>,
    /// Effect analysis results
    pub effects: Option<effect_analysis::EffectAnalysis>,
    /// Hotness analysis results
    pub hotness: Option<hotness::HotnessAnalysis>,
    /// Detected optimization opportunities
    pub optimizations: Vec<OptimizationOpportunity>,
    /// Capability analysis results
    pub capabilities: Option<capability_analysis::CapabilityAnalysis>,
    
    /// Capability-aware inlining analysis results
    pub capability_aware_inlining: Option<capability_aware_inlining::CapabilityAwareInliningAnalysis>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}



impl StaticAnalyzer {
    /// Create new static analyzer with configuration (legacy interface)
    pub fn new(config: AnalysisConfig) -> VMResult<Self> {
        let pipeline_config = PipelineConfig::from(config);
        let mut pipeline = AnalysisPipeline::new(pipeline_config);
        
        // Register analyzers based on enabled analyses
        // Note: This is a placeholder - actual analyzers will be registered
        // when we refactor the individual analysis modules
        
        Ok(Self { pipeline })
    }

    /// Analyze a function comprehensively (legacy interface)
    pub fn analyze_function(&mut self, function: &FunctionDefinition) -> VMResult<AnalysisResult> {
        // Use the new pipeline to analyze the function
        let context = self.pipeline.analyze_function(function.clone())?;
        
        // Convert the new context back to the legacy result format
        let result = AnalysisResult {
            function_id: function.id,
            cfg: context.get_result(AnalysisKind::ControlFlow),
            dataflow: context.get_result(AnalysisKind::DataFlow),
            loops: context.get_result(AnalysisKind::Loop),
            types: context.get_result(AnalysisKind::Type),
            effects: context.get_result(AnalysisKind::Effect),
            hotness: context.get_result(AnalysisKind::Hotness),
            capabilities: context.get_result(AnalysisKind::Capability),
            capability_aware_inlining: context.get_result(AnalysisKind::CapabilityAwareInlining),
            optimizations: context.optimization_opportunities,
            metadata: context.metadata,
        };
        
        Ok(result)
    }

    /// Get the underlying pipeline for advanced usage
    pub fn pipeline(&mut self) -> &mut AnalysisPipeline {
        &mut self.pipeline
    }

    /// Calculate analysis confidence based on completed passes (legacy method)
    fn calculate_confidence(&self, result: &AnalysisResult) -> f64 {
        result.metadata.confidence
    }
} 