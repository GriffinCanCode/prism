//! Module Restructuring Integration
//!
//! This module embodies the single concept of "Cohesion-Driven Module Restructuring".
//! Following Prism's Conceptual Cohesion principle, this module is responsible
//! for ONE thing: translating cohesion analysis results into concrete module
//! restructuring actions and coordinating their safe execution.
//!
//! **Conceptual Responsibility**: Bridge cohesion metrics to module restructuring
//! **What it does**: action planning, safety validation, execution coordination
//! **What it doesn't do**: cohesion analysis, module storage, AST manipulation

use crate::{CohesionResult, CohesionMetrics, CohesionViolation, ConceptualBoundary, CohesionSuggestion};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::{span::Span, symbol::Symbol, NodeId};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::SystemTime;

// Sub-modules for focused responsibilities
pub mod engine;
pub mod actions;
pub mod executor;
pub mod safety;
pub mod integration;

// Re-export main types
pub use engine::{RestructuringEngine, RestructuringConfig, RestructuringResult};
pub use actions::{ModuleAction, ActionType, ActionPlan, ActionImpact};
pub use executor::{ActionExecutor, ExecutionContext, ExecutionResult};
pub use safety::{SafetyValidator, SafetyCheck, SafetyLevel};
pub use integration::{CompilerIntegration, IntegrationStatus};

/// Cohesion-driven module restructuring coordinator
/// 
/// This is the main entry point for the restructuring system. It coordinates
/// between cohesion analysis and module system changes while maintaining
/// safety guarantees and proper separation of concerns.
#[derive(Debug)]
pub struct CohesionRestructuringSystem {
    /// Core restructuring engine
    engine: RestructuringEngine,
    
    /// Action executor for safe operations
    executor: ActionExecutor,
    
    /// Safety validator for operation validation
    safety_validator: SafetyValidator,
    
    /// System configuration
    config: RestructuringSystemConfig,
}

/// Configuration for the restructuring system
#[derive(Debug, Clone)]
pub struct RestructuringSystemConfig {
    /// Enable automatic restructuring (vs. suggestions only)
    pub enable_automatic_restructuring: bool,
    
    /// Minimum cohesion threshold for triggering restructuring
    pub min_cohesion_threshold: f64,
    
    /// Maximum number of actions per restructuring session
    pub max_actions_per_session: usize,
    
    /// Enable dry-run mode for testing
    pub dry_run_mode: bool,
    
    /// Safety level for operations
    pub safety_level: SafetyLevel,
    
    /// Enable rollback capabilities
    pub enable_rollback: bool,
}

impl Default for RestructuringSystemConfig {
    fn default() -> Self {
        Self {
            enable_automatic_restructuring: false, // Conservative default
            min_cohesion_threshold: 0.6,
            max_actions_per_session: 10,
            dry_run_mode: true, // Safe default
            safety_level: SafetyLevel::Strict,
            enable_rollback: true,
        }
    }
}

/// Complete restructuring analysis and action plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestructuringAnalysis {
    /// Cohesion metrics that triggered this analysis
    pub triggering_metrics: CohesionMetrics,
    
    /// Detected violations that need addressing
    pub violations: Vec<CohesionViolation>,
    
    /// Conceptual boundaries identified
    pub boundaries: Vec<ConceptualBoundary>,
    
    /// Generated action plan
    pub action_plan: ActionPlan,
    
    /// Safety analysis results
    pub safety_analysis: SafetyAnalysis,
    
    /// Estimated impact of restructuring
    pub impact_estimate: ImpactEstimate,
    
    /// Analysis timestamp
    pub analyzed_at: SystemTime,
}

/// Safety analysis for restructuring operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAnalysis {
    /// Overall safety level
    pub safety_level: SafetyLevel,
    
    /// Identified risks
    pub risks: Vec<RestructuringRisk>,
    
    /// Required preconditions
    pub preconditions: Vec<String>,
    
    /// Recommended safeguards
    pub safeguards: Vec<String>,
    
    /// Rollback feasibility
    pub rollback_feasible: bool,
}

/// Restructuring risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestructuringRisk {
    /// Risk category
    pub category: RiskCategory,
    
    /// Risk description
    pub description: String,
    
    /// Risk severity (0-1)
    pub severity: f64,
    
    /// Mitigation strategies
    pub mitigations: Vec<String>,
}

/// Categories of restructuring risks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskCategory {
    /// Breaking changes to public APIs
    ApiBreaking,
    
    /// Dependency cycles creation
    DependencyCycles,
    
    /// Data loss or corruption
    DataLoss,
    
    /// Performance degradation
    Performance,
    
    /// Security boundary violations
    Security,
    
    /// Build system failures
    BuildSystem,
}

/// Impact estimate for restructuring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEstimate {
    /// Estimated cohesion improvement
    pub cohesion_improvement: f64,
    
    /// Number of files affected
    pub files_affected: usize,
    
    /// Number of modules affected
    pub modules_affected: usize,
    
    /// Estimated effort (hours)
    pub estimated_effort_hours: f64,
    
    /// Confidence in estimates (0-1)
    pub confidence: f64,
}

impl CohesionRestructuringSystem {
    /// Create new restructuring system
    pub fn new(config: RestructuringSystemConfig) -> CohesionResult<Self> {
        let engine = RestructuringEngine::new(config.clone().into())?;
        let executor = ActionExecutor::new(config.safety_level.clone())?;
        let safety_validator = SafetyValidator::new(config.safety_level.clone())?;
        
        Ok(Self {
            engine,
            executor,
            safety_validator,
            config,
        })
    }
    
    /// Analyze program and generate restructuring plan
    pub fn analyze_restructuring_opportunities(
        &self,
        program: &Program,
        metrics: &CohesionMetrics,
        violations: &[CohesionViolation],
        boundaries: &[ConceptualBoundary],
    ) -> CohesionResult<RestructuringAnalysis> {
        // Generate action plan based on cohesion analysis
        let action_plan = self.engine.generate_action_plan(
            program, metrics, violations, boundaries
        )?;
        
        // Perform safety analysis
        let safety_analysis = self.safety_validator.analyze_safety(
            program, &action_plan
        )?;
        
        // Estimate impact
        let impact_estimate = self.estimate_impact(program, &action_plan)?;
        
        Ok(RestructuringAnalysis {
            triggering_metrics: metrics.clone(),
            violations: violations.to_vec(),
            boundaries: boundaries.to_vec(),
            action_plan,
            safety_analysis,
            impact_estimate,
            analyzed_at: SystemTime::now(),
        })
    }
    
    /// Execute restructuring plan (if automatic mode enabled)
    pub fn execute_restructuring(
        &mut self,
        analysis: &RestructuringAnalysis,
        program: &Program,
    ) -> CohesionResult<ExecutionResult> {
        if !self.config.enable_automatic_restructuring {
            return Err(crate::CohesionError::ConfigurationError(
                "Automatic restructuring is disabled".to_string()
            ));
        }
        
        // Final safety check
        if analysis.safety_analysis.safety_level < self.config.safety_level {
            return Err(crate::CohesionError::SafetyViolation(
                "Operation does not meet minimum safety requirements".to_string()
            ));
        }
        
        // Execute the action plan
        let execution_context = ExecutionContext {
            dry_run: self.config.dry_run_mode,
            enable_rollback: self.config.enable_rollback,
            safety_level: self.config.safety_level.clone(),
        };
        
        self.executor.execute_plan(&analysis.action_plan, &execution_context)
    }
    
    /// Estimate impact of restructuring actions
    fn estimate_impact(
        &self,
        program: &Program,
        action_plan: &ActionPlan,
    ) -> CohesionResult<ImpactEstimate> {
        let mut files_affected = HashSet::new();
        let mut modules_affected = HashSet::new();
        let mut estimated_effort = 0.0;
        
        for action in &action_plan.actions {
            // Count affected files and modules
            if let Some(location) = &action.location {
                files_affected.insert(location.file_path.clone());
            }
            
            match &action.action_type {
                ActionType::SplitModule { source_module, .. } => {
                    modules_affected.insert(source_module.clone());
                    estimated_effort += 4.0; // Hours
                }
                ActionType::MergeModules { source_modules, .. } => {
                    modules_affected.extend(source_modules.iter().cloned());
                    estimated_effort += 2.0 * source_modules.len() as f64;
                }
                ActionType::MoveFunction { source_module, target_module, .. } => {
                    modules_affected.insert(source_module.clone());
                    modules_affected.insert(target_module.clone());
                    estimated_effort += 1.0;
                }
                ActionType::ReorganizeSections { module, .. } => {
                    modules_affected.insert(module.clone());
                    estimated_effort += 0.5;
                }
            }
        }
        
        // Estimate cohesion improvement based on action types
        let cohesion_improvement = action_plan.actions.iter()
            .map(|action| action.estimated_improvement)
            .sum::<f64>() / action_plan.actions.len() as f64;
        
        Ok(ImpactEstimate {
            cohesion_improvement,
            files_affected: files_affected.len(),
            modules_affected: modules_affected.len(),
            estimated_effort_hours: estimated_effort,
            confidence: 0.8, // Conservative confidence
        })
    }
} 