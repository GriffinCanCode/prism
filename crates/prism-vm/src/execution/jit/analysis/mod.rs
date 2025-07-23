//! Modern JIT Static Analysis Infrastructure
//!
//! This module provides comprehensive static analysis capabilities for the JIT compiler,
//! intelligently separated into focused sub-modules that each handle specific aspects
//! of program analysis.
//!
//! ## Module Structure
//!
//! - [`control_flow`] - Control flow graph construction and analysis
//! - [`data_flow`] - Data flow analysis including liveness and reaching definitions
//! - [`loop_analysis`] - Loop detection, nesting analysis, and optimization opportunities
//! - [`type_analysis`] - Type inference and propagation analysis
//! - [`effect_analysis`] - Effect system analysis for optimization safety
//! - [`hotness`] - Hotness analysis and profiling data integration
//! - [`optimization_opportunities`] - Detection of optimization opportunities
//! - [`capability_analysis`] - Capability flow analysis for security

pub mod control_flow;
pub mod data_flow;
pub mod loop_analysis;
pub mod type_analysis;
pub mod effect_analysis;
pub mod hotness;
pub mod optimization_opportunities;
pub mod capability_analysis;

use crate::{VMResult, PrismVMError, bytecode::{PrismBytecode, FunctionDefinition}};
use prism_runtime::authority::capability::CapabilitySet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// Re-export key types from sub-modules
pub use control_flow::{ControlFlowGraph, BasicBlock, CFGEdge, DominanceInfo};
pub use data_flow::{DataFlowAnalysis, LivenessAnalysis, ReachingDefinitions, AvailableExpressions};
pub use loop_analysis::{LoopAnalysis, LoopInfo, LoopNest, LoopOptimizationOpportunity};
pub use type_analysis::{TypeAnalysis, TypeInference, TypeConstraint, TypeEnvironment};
pub use effect_analysis::{EffectAnalysis, EffectFlow, EffectConstraint, SafetyAnalysis};
pub use hotness::{HotnessAnalysis, HotSpot, ProfileData, ExecutionFrequency};
pub use optimization_opportunities::{OptimizationFinder, OptimizationOpportunity, OptimizationKind};
pub use capability_analysis::{CapabilityAnalysis, CapabilityFlow, SecurityConstraint};

/// Comprehensive static analyzer that coordinates all analysis passes
#[derive(Debug)]
pub struct StaticAnalyzer {
    /// Configuration for analysis passes
    config: AnalysisConfig,
    
    /// Control flow analysis
    cfg_analyzer: control_flow::CFGAnalyzer,
    
    /// Data flow analysis
    dataflow_analyzer: data_flow::DataFlowAnalyzer,
    
    /// Loop analysis
    loop_analyzer: loop_analysis::LoopAnalyzer,
    
    /// Type analysis
    type_analyzer: type_analysis::TypeAnalyzer,
    
    /// Effect analysis
    effect_analyzer: effect_analysis::EffectAnalyzer,
    
    /// Hotness analysis
    hotness_analyzer: hotness::HotnessAnalyzer,
    
    /// Optimization opportunity finder
    optimization_finder: optimization_opportunities::OptimizationFinder,
    
    /// Capability analysis
    capability_analyzer: capability_analysis::CapabilityAnalyzer,
}

/// Configuration for static analysis
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

/// Comprehensive analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Function being analyzed
    pub function_id: u32,
    
    /// Control flow graph
    pub cfg: Option<ControlFlowGraph>,
    
    /// Data flow analysis results
    pub dataflow: Option<DataFlowAnalysis>,
    
    /// Loop analysis results
    pub loops: Option<LoopAnalysis>,
    
    /// Type analysis results
    pub types: Option<TypeAnalysis>,
    
    /// Effect analysis results
    pub effects: Option<EffectAnalysis>,
    
    /// Hotness analysis results
    pub hotness: Option<HotnessAnalysis>,
    
    /// Detected optimization opportunities
    pub optimizations: Vec<OptimizationOpportunity>,
    
    /// Capability analysis results
    pub capabilities: Option<CapabilityAnalysis>,
    
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Analysis metadata
#[derive(Debug, Clone, Default)]
pub struct AnalysisMetadata {
    /// Analysis duration
    pub analysis_time: Duration,
    
    /// Analysis passes run
    pub passes_run: Vec<String>,
    
    /// Analysis warnings
    pub warnings: Vec<String>,
    
    /// Analysis confidence (0.0 to 1.0)
    pub confidence: f64,
}

impl StaticAnalyzer {
    /// Create new static analyzer with configuration
    pub fn new(config: AnalysisConfig) -> VMResult<Self> {
        Ok(Self {
            cfg_analyzer: control_flow::CFGAnalyzer::new(&config)?,
            dataflow_analyzer: data_flow::DataFlowAnalyzer::new(&config)?,
            loop_analyzer: loop_analysis::LoopAnalyzer::new(&config)?,
            type_analyzer: type_analysis::TypeAnalyzer::new(&config)?,
            effect_analyzer: effect_analysis::EffectAnalyzer::new(&config)?,
            hotness_analyzer: hotness::HotnessAnalyzer::new(&config)?,
            optimization_finder: optimization_opportunities::OptimizationFinder::new(&config)?,
            capability_analyzer: capability_analysis::CapabilityAnalyzer::new(&config)?,
            config,
        })
    }

    /// Analyze a function comprehensively
    pub fn analyze_function(&mut self, function: &FunctionDefinition) -> VMResult<AnalysisResult> {
        let start_time = std::time::Instant::now();
        let mut result = AnalysisResult {
            function_id: function.id,
            cfg: None,
            dataflow: None,
            loops: None,
            types: None,
            effects: None,
            hotness: None,
            optimizations: Vec::new(),
            capabilities: None,
            metadata: AnalysisMetadata::default(),
        };

        // Run analysis passes in dependency order
        if self.config.enable_cfg_analysis {
            result.cfg = Some(self.cfg_analyzer.analyze(function)?);
            result.metadata.passes_run.push("cfg".to_string());
        }

        if self.config.enable_dataflow_analysis {
            if let Some(ref cfg) = result.cfg {
                result.dataflow = Some(self.dataflow_analyzer.analyze(function, cfg)?);
                result.metadata.passes_run.push("dataflow".to_string());
            } else {
                result.metadata.warnings.push("Dataflow analysis skipped: no CFG".to_string());
            }
        }

        if self.config.enable_loop_analysis {
            if let Some(ref cfg) = result.cfg {
                result.loops = Some(self.loop_analyzer.analyze(function, cfg)?);
                result.metadata.passes_run.push("loops".to_string());
            }
        }

        if self.config.enable_type_analysis {
            result.types = Some(self.type_analyzer.analyze(function)?);
            result.metadata.passes_run.push("types".to_string());
        }

        if self.config.enable_effect_analysis {
            result.effects = Some(self.effect_analyzer.analyze(function)?);
            result.metadata.passes_run.push("effects".to_string());
        }

        if self.config.enable_hotness_analysis {
            result.hotness = Some(self.hotness_analyzer.analyze(function)?);
            result.metadata.passes_run.push("hotness".to_string());
        }

        if self.config.enable_optimization_detection {
            result.optimizations = self.optimization_finder.find_opportunities(
                function,
                &result
            )?;
            result.metadata.passes_run.push("optimizations".to_string());
        }

        if self.config.enable_capability_analysis {
            result.capabilities = Some(self.capability_analyzer.analyze(function)?);
            result.metadata.passes_run.push("capabilities".to_string());
        }

        result.metadata.analysis_time = start_time.elapsed();
        result.metadata.confidence = self.calculate_confidence(&result);

        Ok(result)
    }

    /// Calculate analysis confidence based on completed passes
    fn calculate_confidence(&self, result: &AnalysisResult) -> f64 {
        let total_passes = 8; // Total number of possible passes
        let completed_passes = result.metadata.passes_run.len();
        
        let base_confidence = completed_passes as f64 / total_passes as f64;
        
        // Adjust confidence based on warnings
        let warning_penalty = result.metadata.warnings.len() as f64 * 0.1;
        
        (base_confidence - warning_penalty).max(0.0).min(1.0)
    }
} 