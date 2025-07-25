//! Restructuring Engine - Core Logic for Cohesion-Driven Module Changes
//!
//! This module embodies the single concept of "Restructuring Decision Making".
//! It analyzes cohesion metrics, violations, and boundaries to generate
//! concrete action plans for improving module organization.
//!
//! **Conceptual Responsibility**: Convert cohesion analysis to restructuring actions
//! **What it does**: action planning, decision algorithms, optimization strategies
//! **What it doesn't do**: action execution, safety validation, AST manipulation

use crate::{CohesionResult, CohesionMetrics, CohesionViolation, ConceptualBoundary, ViolationType};
use crate::restructuring::actions::{ModuleAction, ActionType, ActionPlan, ActionPriority};
use prism_ast::{Program, AstNode, Item, ModuleDecl, SectionDecl};
use prism_common::{span::Span, symbol::Symbol, NodeId};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Core restructuring engine that converts cohesion analysis to actions
#[derive(Debug)]
pub struct RestructuringEngine {
    /// Engine configuration
    config: RestructuringConfig,
    
    /// Decision algorithms
    decision_algorithms: DecisionAlgorithms,
    
    /// Action optimization strategies
    optimizer: ActionOptimizer,
}

/// Configuration for the restructuring engine
#[derive(Debug, Clone)]
pub struct RestructuringConfig {
    /// Minimum violation severity to act on
    pub min_violation_severity: f64,
    
    /// Minimum boundary strength to consider
    pub min_boundary_strength: f64,
    
    /// Maximum actions per plan
    pub max_actions_per_plan: usize,
    
    /// Enable aggressive restructuring
    pub enable_aggressive_mode: bool,
    
    /// Prioritize cohesion over stability
    pub prioritize_cohesion: bool,
    
    /// Enable cross-module analysis
    pub enable_cross_module_analysis: bool,
}

impl Default for RestructuringConfig {
    fn default() -> Self {
        Self {
            min_violation_severity: 0.7,
            min_boundary_strength: 0.6,
            max_actions_per_plan: 5,
            enable_aggressive_mode: false,
            prioritize_cohesion: false,
            enable_cross_module_analysis: true,
        }
    }
}

/// Decision algorithms for restructuring
#[derive(Debug)]
struct DecisionAlgorithms {
    /// Algorithm for split decisions
    split_algorithm: SplitDecisionAlgorithm,
    
    /// Algorithm for merge decisions
    merge_algorithm: MergeDecisionAlgorithm,
    
    /// Algorithm for move decisions
    move_algorithm: MoveDecisionAlgorithm,
}

/// Algorithm for deciding when to split modules
#[derive(Debug)]
struct SplitDecisionAlgorithm {
    /// Minimum module size for split consideration
    min_size_threshold: usize,
    
    /// Minimum cohesion drop for split
    min_cohesion_drop: f64,
}

/// Algorithm for deciding when to merge modules
#[derive(Debug)]
struct MergeDecisionAlgorithm {
    /// Maximum combined size for merge
    max_combined_size: usize,
    
    /// Minimum cohesion gain for merge
    min_cohesion_gain: f64,
}

/// Algorithm for deciding when to move functions
#[derive(Debug)]
struct MoveDecisionAlgorithm {
    /// Minimum cohesion improvement for move
    min_improvement_threshold: f64,
    
    /// Maximum dependency impact allowed
    max_dependency_impact: f64,
}

/// Action optimizer for improving action plans
#[derive(Debug)]
struct ActionOptimizer {
    /// Enable action deduplication
    enable_deduplication: bool,
    
    /// Enable action ordering optimization
    enable_ordering: bool,
    
    /// Enable impact-based prioritization
    enable_impact_prioritization: bool,
}

/// Restructuring result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestructuringResult {
    /// Generated action plan
    pub action_plan: ActionPlan,
    
    /// Decision rationale
    pub rationale: DecisionRationale,
    
    /// Alternative plans considered
    pub alternatives: Vec<ActionPlan>,
    
    /// Confidence in recommendations
    pub confidence: f64,
}

/// Rationale for restructuring decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRationale {
    /// Primary factors influencing decisions
    pub primary_factors: Vec<String>,
    
    /// Violations addressed by actions
    pub violations_addressed: Vec<String>,
    
    /// Boundaries respected by actions
    pub boundaries_respected: Vec<String>,
    
    /// Trade-offs made in decisions
    pub tradeoffs: Vec<String>,
}

impl RestructuringEngine {
    /// Create new restructuring engine
    pub fn new(config: RestructuringConfig) -> CohesionResult<Self> {
        let decision_algorithms = DecisionAlgorithms {
            split_algorithm: SplitDecisionAlgorithm {
                min_size_threshold: 10,
                min_cohesion_drop: 0.3,
            },
            merge_algorithm: MergeDecisionAlgorithm {
                max_combined_size: 50,
                min_cohesion_gain: 0.2,
            },
            move_algorithm: MoveDecisionAlgorithm {
                min_improvement_threshold: 0.1,
                max_dependency_impact: 0.3,
            },
        };
        
        let optimizer = ActionOptimizer {
            enable_deduplication: true,
            enable_ordering: true,
            enable_impact_prioritization: true,
        };
        
        Ok(Self {
            config,
            decision_algorithms,
            optimizer,
        })
    }
    
    /// Generate action plan from cohesion analysis
    pub fn generate_action_plan(
        &self,
        program: &Program,
        metrics: &CohesionMetrics,
        violations: &[CohesionViolation],
        boundaries: &[ConceptualBoundary],
    ) -> CohesionResult<ActionPlan> {
        let mut actions = Vec::new();
        
        // Extract modules for analysis
        let modules = self.extract_modules(program);
        
        // Generate actions based on violations
        actions.extend(self.generate_violation_based_actions(&modules, violations)?);
        
        // Generate actions based on boundaries
        actions.extend(self.generate_boundary_based_actions(&modules, boundaries)?);
        
        // Generate actions based on metrics
        actions.extend(self.generate_metric_based_actions(&modules, metrics)?);
        
        // Optimize the action plan
        let optimized_actions = self.optimizer.optimize_actions(actions)?;
        
        // Limit actions based on configuration
        let final_actions = optimized_actions
            .into_iter()
            .take(self.config.max_actions_per_plan)
            .collect();
        
        Ok(ActionPlan {
            actions: final_actions,
            estimated_duration_hours: self.estimate_plan_duration(&actions),
            estimated_improvement: self.estimate_plan_improvement(&actions),
            dependencies: self.analyze_action_dependencies(&actions),
        })
    }
    
    /// Extract modules from program
    fn extract_modules(&self, program: &Program) -> Vec<(NodeId, &ModuleDecl)> {
        program.items.iter()
            .filter_map(|item| {
                if let Item::Module(module_decl) = &item.kind {
                    Some((item.node_id, module_decl))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Generate actions based on violations
    fn generate_violation_based_actions(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        violations: &[CohesionViolation],
    ) -> CohesionResult<Vec<ModuleAction>> {
        let mut actions = Vec::new();
        
        for violation in violations {
            if violation.impact_score < self.config.min_violation_severity {
                continue;
            }
            
            match violation.violation_type {
                ViolationType::MultipleResponsibilities => {
                    if let Some(action) = self.generate_split_action(modules, violation)? {
                        actions.push(action);
                    }
                }
                ViolationType::ScatteredFunctionality => {
                    if let Some(action) = self.generate_merge_action(modules, violation)? {
                        actions.push(action);
                    }
                }
                ViolationType::PoorSectionOrganization => {
                    if let Some(action) = self.generate_reorganize_action(modules, violation)? {
                        actions.push(action);
                    }
                }
                ViolationType::GodModule => {
                    if let Some(action) = self.generate_split_action(modules, violation)? {
                        actions.push(action);
                    }
                }
                _ => {
                    // Handle other violation types with generic improvements
                    if let Some(action) = self.generate_generic_improvement_action(modules, violation)? {
                        actions.push(action);
                    }
                }
            }
        }
        
        Ok(actions)
    }
    
    /// Generate actions based on conceptual boundaries
    fn generate_boundary_based_actions(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        boundaries: &[ConceptualBoundary],
    ) -> CohesionResult<Vec<ModuleAction>> {
        let mut actions = Vec::new();
        
        for boundary in boundaries {
            if boundary.strength < self.config.min_boundary_strength {
                continue;
            }
            
            // Generate actions that respect or enforce boundaries
            match &boundary.suggested_action {
                crate::boundaries::BoundaryAction::Split { .. } => {
                    if let Some(action) = self.generate_boundary_split_action(modules, boundary)? {
                        actions.push(action);
                    }
                }
                crate::boundaries::BoundaryAction::Merge { .. } => {
                    if let Some(action) = self.generate_boundary_merge_action(modules, boundary)? {
                        actions.push(action);
                    }
                }
                crate::boundaries::BoundaryAction::Isolate { .. } => {
                    if let Some(action) = self.generate_isolation_action(modules, boundary)? {
                        actions.push(action);
                    }
                }
                _ => {}
            }
        }
        
        Ok(actions)
    }
    
    /// Generate actions based on overall metrics
    fn generate_metric_based_actions(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        metrics: &CohesionMetrics,
    ) -> CohesionResult<Vec<ModuleAction>> {
        let mut actions = Vec::new();
        
        // If overall cohesion is low, suggest structural improvements
        if metrics.overall_score < 0.6 {
            // Analyze each dimension and suggest improvements
            if metrics.type_cohesion < 0.5 {
                actions.extend(self.generate_type_cohesion_improvements(modules)?);
            }
            
            if metrics.semantic_cohesion < 0.5 {
                actions.extend(self.generate_semantic_improvements(modules)?);
            }
            
            if metrics.business_cohesion < 0.5 {
                actions.extend(self.generate_business_improvements(modules)?);
            }
        }
        
        Ok(actions)
    }
    
    /// Generate split action for a violation
    fn generate_split_action(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        violation: &CohesionViolation,
    ) -> CohesionResult<Option<ModuleAction>> {
        // Find the module associated with this violation
        if let Some(location) = &violation.location {
            // Find module containing this location
            for (node_id, module) in modules {
                // Check if module size justifies splitting
                let section_count = module.sections.len();
                if section_count >= self.decision_algorithms.split_algorithm.min_size_threshold {
                    return Ok(Some(ModuleAction {
                        action_type: ActionType::SplitModule {
                            source_module: module.name.to_string(),
                            target_modules: vec![
                                format!("{}_core", module.name),
                                format!("{}_utils", module.name),
                            ],
                            split_strategy: "by_responsibility".to_string(),
                        },
                        priority: ActionPriority::High,
                        rationale: format!("Split {} to address {}", module.name, violation.description),
                        estimated_improvement: 0.3,
                        location: Some(crate::restructuring::actions::ActionLocation {
                            file_path: std::path::PathBuf::from(format!("source_{}.prism", location.source_id.as_u32())),
                            span: location.clone(),
                        }),
                        dependencies: Vec::new(),
                    }));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Generate merge action for a violation
    fn generate_merge_action(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        violation: &CohesionViolation,
    ) -> CohesionResult<Option<ModuleAction>> {
        // Look for small, related modules that could be merged
        let small_modules: Vec<_> = modules.iter()
            .filter(|(_, module)| module.sections.len() < 5)
            .collect();
        
        if small_modules.len() >= 2 {
            let source_modules: Vec<String> = small_modules.iter()
                .take(2)
                .map(|(_, module)| module.name.to_string())
                .collect();
            
            return Ok(Some(ModuleAction {
                action_type: ActionType::MergeModules {
                    source_modules: source_modules.clone(),
                    target_module: format!("{}_combined", source_modules[0]),
                    merge_strategy: "by_functionality".to_string(),
                },
                priority: ActionPriority::Medium,
                rationale: format!("Merge related modules to address scattered functionality"),
                estimated_improvement: 0.2,
                location: violation.location.as_ref().map(|loc| crate::restructuring::actions::ActionLocation {
                    file_path: loc.file_path.clone(),
                    span: loc.clone(),
                }),
                dependencies: Vec::new(),
            }));
        }
        
        Ok(None)
    }
    
    /// Generate reorganize action for a violation
    fn generate_reorganize_action(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        violation: &CohesionViolation,
    ) -> CohesionResult<Option<ModuleAction>> {
        if let Some(location) = &violation.location {
            for (_, module) in modules {
                if module.sections.len() > 3 {
                    return Ok(Some(ModuleAction {
                        action_type: ActionType::ReorganizeSections {
                            module: module.name.to_string(),
                            new_organization: vec![
                                "types".to_string(),
                                "core".to_string(),
                                "utilities".to_string(),
                            ],
                        },
                        priority: ActionPriority::Low,
                        rationale: format!("Reorganize sections in {} for better cohesion", module.name),
                        estimated_improvement: 0.15,
                        location: Some(crate::restructuring::actions::ActionLocation {
                            file_path: location.file_path.clone(),
                            span: location.clone(),
                        }),
                        dependencies: Vec::new(),
                    }));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Generate generic improvement action
    fn generate_generic_improvement_action(
        &self,
        _modules: &[(NodeId, &ModuleDecl)],
        violation: &CohesionViolation,
    ) -> CohesionResult<Option<ModuleAction>> {
        // For now, return None for generic violations
        // This can be expanded based on specific violation types
        Ok(None)
    }
    
    /// Generate boundary-based split action
    fn generate_boundary_split_action(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        boundary: &ConceptualBoundary,
    ) -> CohesionResult<Option<ModuleAction>> {
        // Implementation similar to generate_split_action but based on boundary analysis
        Ok(None)
    }
    
    /// Generate boundary-based merge action
    fn generate_boundary_merge_action(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        boundary: &ConceptualBoundary,
    ) -> CohesionResult<Option<ModuleAction>> {
        // Implementation for boundary-driven merges
        Ok(None)
    }
    
    /// Generate isolation action
    fn generate_isolation_action(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
        boundary: &ConceptualBoundary,
    ) -> CohesionResult<Option<ModuleAction>> {
        // Implementation for isolating concerns
        Ok(None)
    }
    
    /// Generate type cohesion improvements
    fn generate_type_cohesion_improvements(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
    ) -> CohesionResult<Vec<ModuleAction>> {
        // Implementation for type-based improvements
        Ok(Vec::new())
    }
    
    /// Generate semantic improvements
    fn generate_semantic_improvements(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
    ) -> CohesionResult<Vec<ModuleAction>> {
        // Implementation for semantic improvements
        Ok(Vec::new())
    }
    
    /// Generate business improvements
    fn generate_business_improvements(
        &self,
        modules: &[(NodeId, &ModuleDecl)],
    ) -> CohesionResult<Vec<ModuleAction>> {
        // Implementation for business cohesion improvements
        Ok(Vec::new())
    }
    
    /// Estimate plan duration
    fn estimate_plan_duration(&self, actions: &[ModuleAction]) -> f64 {
        actions.iter()
            .map(|action| match &action.action_type {
                ActionType::SplitModule { .. } => 4.0,
                ActionType::MergeModules { .. } => 2.0,
                ActionType::MoveFunction { .. } => 1.0,
                ActionType::ReorganizeSections { .. } => 0.5,
            })
            .sum()
    }
    
    /// Estimate plan improvement
    fn estimate_plan_improvement(&self, actions: &[ModuleAction]) -> f64 {
        if actions.is_empty() {
            0.0
        } else {
            actions.iter()
                .map(|action| action.estimated_improvement)
                .sum::<f64>() / actions.len() as f64
        }
    }
    
    /// Analyze action dependencies
    fn analyze_action_dependencies(&self, actions: &[ModuleAction]) -> Vec<String> {
        // Simple dependency analysis - can be expanded
        let mut dependencies = Vec::new();
        
        for action in actions {
            match &action.action_type {
                ActionType::SplitModule { source_module, .. } => {
                    dependencies.push(format!("Split {} requires backup", source_module));
                }
                ActionType::MergeModules { source_modules, .. } => {
                    dependencies.push(format!("Merge {:?} requires compatibility check", source_modules));
                }
                _ => {}
            }
        }
        
        dependencies
    }
}

impl ActionOptimizer {
    /// Optimize actions for better execution
    fn optimize_actions(&self, mut actions: Vec<ModuleAction>) -> CohesionResult<Vec<ModuleAction>> {
        if self.enable_deduplication {
            actions = self.deduplicate_actions(actions)?;
        }
        
        if self.enable_impact_prioritization {
            actions.sort_by(|a, b| {
                b.estimated_improvement.partial_cmp(&a.estimated_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        
        if self.enable_ordering {
            actions = self.optimize_action_order(actions)?;
        }
        
        Ok(actions)
    }
    
    /// Remove duplicate actions
    fn deduplicate_actions(&self, actions: Vec<ModuleAction>) -> CohesionResult<Vec<ModuleAction>> {
        let mut seen = HashSet::new();
        let mut unique_actions = Vec::new();
        
        for action in actions {
            let key = format!("{:?}", action.action_type);
            if !seen.contains(&key) {
                seen.insert(key);
                unique_actions.push(action);
            }
        }
        
        Ok(unique_actions)
    }
    
    /// Optimize action execution order
    fn optimize_action_order(&self, mut actions: Vec<ModuleAction>) -> CohesionResult<Vec<ModuleAction>> {
        // Sort by priority, then by estimated improvement
        actions.sort_by(|a, b| {
            a.priority.cmp(&b.priority)
                .then_with(|| b.estimated_improvement.partial_cmp(&a.estimated_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal))
        });
        
        Ok(actions)
    }
} 