//! Module Restructuring Actions
//!
//! This module embodies the single concept of "Restructuring Action Definitions".
//! It defines the concrete actions that can be taken to improve module organization
//! based on cohesion analysis.
//!
//! **Conceptual Responsibility**: Define restructuring action types and metadata
//! **What it does**: action type definitions, action planning, impact estimation
//! **What it doesn't do**: action execution, safety validation, cohesion analysis

use prism_common::{span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// A concrete module restructuring action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleAction {
    /// Type of action to perform
    pub action_type: ActionType,
    
    /// Priority level for this action
    pub priority: ActionPriority,
    
    /// Rationale for why this action is recommended
    pub rationale: String,
    
    /// Estimated improvement in cohesion score (0-1)
    pub estimated_improvement: f64,
    
    /// Location where this action applies
    pub location: Option<ActionLocation>,
    
    /// Dependencies that must be satisfied before this action
    pub dependencies: Vec<String>,
}

/// Types of restructuring actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// Split a module into multiple modules
    SplitModule {
        /// Source module to split
        source_module: String,
        /// Target modules to create
        target_modules: Vec<String>,
        /// Strategy for splitting (by_responsibility, by_layer, etc.)
        split_strategy: String,
    },
    
    /// Merge multiple modules into one
    MergeModules {
        /// Source modules to merge
        source_modules: Vec<String>,
        /// Target module to create
        target_module: String,
        /// Strategy for merging (by_functionality, by_domain, etc.)
        merge_strategy: String,
    },
    
    /// Move a function between modules
    MoveFunction {
        /// Source module containing the function
        source_module: String,
        /// Target module to move to
        target_module: String,
        /// Function to move
        function_name: String,
        /// Reason for the move
        move_reason: String,
    },
    
    /// Reorganize sections within a module
    ReorganizeSections {
        /// Module to reorganize
        module: String,
        /// New section organization
        new_organization: Vec<String>,
    },
}

/// Priority levels for actions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ActionPriority {
    /// Critical action that should be performed immediately
    Critical,
    /// High priority action
    High,
    /// Medium priority action
    Medium,
    /// Low priority action
    Low,
    /// Optional improvement
    Optional,
}

/// Location information for an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionLocation {
    /// File path where the action applies
    pub file_path: PathBuf,
    /// Specific span within the file
    pub span: Span,
}

/// Complete action plan for restructuring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    /// List of actions to perform
    pub actions: Vec<ModuleAction>,
    
    /// Estimated total duration in hours
    pub estimated_duration_hours: f64,
    
    /// Estimated overall improvement in cohesion
    pub estimated_improvement: f64,
    
    /// Dependencies between actions
    pub dependencies: Vec<String>,
}

/// Impact analysis for an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionImpact {
    /// Cohesion improvement estimate
    pub cohesion_improvement: f64,
    
    /// Number of files that will be affected
    pub files_affected: usize,
    
    /// Number of modules that will be affected
    pub modules_affected: usize,
    
    /// Breaking changes that may occur
    pub breaking_changes: Vec<String>,
    
    /// Performance impact estimate
    pub performance_impact: PerformanceImpact,
    
    /// Maintenance impact estimate
    pub maintenance_impact: MaintenanceImpact,
}

/// Performance impact of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Compilation time change (negative = improvement)
    pub compilation_time_change_percent: f64,
    
    /// Runtime performance change (negative = improvement)
    pub runtime_performance_change_percent: f64,
    
    /// Memory usage change (negative = improvement)
    pub memory_usage_change_percent: f64,
}

/// Maintenance impact of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceImpact {
    /// Code readability improvement (0-1)
    pub readability_improvement: f64,
    
    /// Testing effort change (negative = less effort needed)
    pub testing_effort_change_percent: f64,
    
    /// Documentation update requirements
    pub documentation_updates_required: Vec<String>,
}

impl ModuleAction {
    /// Create a new split module action
    pub fn split_module(
        source_module: String,
        target_modules: Vec<String>,
        split_strategy: String,
        rationale: String,
        priority: ActionPriority,
    ) -> Self {
        Self {
            action_type: ActionType::SplitModule {
                source_module,
                target_modules,
                split_strategy,
            },
            priority,
            rationale,
            estimated_improvement: 0.3, // Default estimate
            location: None,
            dependencies: Vec::new(),
        }
    }
    
    /// Create a new merge modules action
    pub fn merge_modules(
        source_modules: Vec<String>,
        target_module: String,
        merge_strategy: String,
        rationale: String,
        priority: ActionPriority,
    ) -> Self {
        Self {
            action_type: ActionType::MergeModules {
                source_modules,
                target_module,
                merge_strategy,
            },
            priority,
            rationale,
            estimated_improvement: 0.2, // Default estimate
            location: None,
            dependencies: Vec::new(),
        }
    }
    
    /// Create a new move function action
    pub fn move_function(
        source_module: String,
        target_module: String,
        function_name: String,
        move_reason: String,
        rationale: String,
        priority: ActionPriority,
    ) -> Self {
        Self {
            action_type: ActionType::MoveFunction {
                source_module,
                target_module,
                function_name,
                move_reason,
            },
            priority,
            rationale,
            estimated_improvement: 0.15, // Default estimate
            location: None,
            dependencies: Vec::new(),
        }
    }
    
    /// Create a new reorganize sections action
    pub fn reorganize_sections(
        module: String,
        new_organization: Vec<String>,
        rationale: String,
        priority: ActionPriority,
    ) -> Self {
        Self {
            action_type: ActionType::ReorganizeSections {
                module,
                new_organization,
            },
            priority,
            rationale,
            estimated_improvement: 0.1, // Default estimate
            location: None,
            dependencies: Vec::new(),
        }
    }
    
    /// Get a human-readable description of this action
    pub fn description(&self) -> String {
        match &self.action_type {
            ActionType::SplitModule { source_module, target_modules, .. } => {
                format!("Split {} into {}", source_module, target_modules.join(", "))
            }
            ActionType::MergeModules { source_modules, target_module, .. } => {
                format!("Merge {} into {}", source_modules.join(", "), target_module)
            }
            ActionType::MoveFunction { source_module, target_module, function_name, .. } => {
                format!("Move {} from {} to {}", function_name, source_module, target_module)
            }
            ActionType::ReorganizeSections { module, new_organization } => {
                format!("Reorganize {} sections: {}", module, new_organization.join(", "))
            }
        }
    }
    
    /// Get the modules affected by this action
    pub fn affected_modules(&self) -> Vec<String> {
        match &self.action_type {
            ActionType::SplitModule { source_module, target_modules, .. } => {
                let mut modules = vec![source_module.clone()];
                modules.extend(target_modules.clone());
                modules
            }
            ActionType::MergeModules { source_modules, target_module, .. } => {
                let mut modules = source_modules.clone();
                modules.push(target_module.clone());
                modules
            }
            ActionType::MoveFunction { source_module, target_module, .. } => {
                vec![source_module.clone(), target_module.clone()]
            }
            ActionType::ReorganizeSections { module, .. } => {
                vec![module.clone()]
            }
        }
    }
    
    /// Estimate the risk level of this action
    pub fn risk_level(&self) -> ActionRiskLevel {
        match &self.action_type {
            ActionType::SplitModule { .. } => ActionRiskLevel::High,
            ActionType::MergeModules { .. } => ActionRiskLevel::Medium,
            ActionType::MoveFunction { .. } => ActionRiskLevel::Medium,
            ActionType::ReorganizeSections { .. } => ActionRiskLevel::Low,
        }
    }
}

/// Risk levels for actions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ActionRiskLevel {
    /// Very low risk action
    VeryLow,
    /// Low risk action
    Low,
    /// Medium risk action
    Medium,
    /// High risk action
    High,
    /// Very high risk action
    VeryHigh,
}

impl ActionPlan {
    /// Create a new empty action plan
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            estimated_duration_hours: 0.0,
            estimated_improvement: 0.0,
            dependencies: Vec::new(),
        }
    }
    
    /// Add an action to the plan
    pub fn add_action(&mut self, action: ModuleAction) {
        self.estimated_improvement = (self.estimated_improvement * self.actions.len() as f64 + action.estimated_improvement) / (self.actions.len() + 1) as f64;
        self.actions.push(action);
        self.recalculate_duration();
    }
    
    /// Remove an action from the plan
    pub fn remove_action(&mut self, index: usize) -> Option<ModuleAction> {
        if index < self.actions.len() {
            let removed = self.actions.remove(index);
            self.recalculate_estimates();
            Some(removed)
        } else {
            None
        }
    }
    
    /// Get actions by priority
    pub fn actions_by_priority(&self, priority: ActionPriority) -> Vec<&ModuleAction> {
        self.actions.iter()
            .filter(|action| action.priority == priority)
            .collect()
    }
    
    /// Get the highest priority action
    pub fn highest_priority_action(&self) -> Option<&ModuleAction> {
        self.actions.iter()
            .min_by_key(|action| &action.priority)
    }
    
    /// Check if the plan has any high-risk actions
    pub fn has_high_risk_actions(&self) -> bool {
        self.actions.iter()
            .any(|action| action.risk_level() >= ActionRiskLevel::High)
    }
    
    /// Get total estimated improvement
    pub fn total_estimated_improvement(&self) -> f64 {
        if self.actions.is_empty() {
            0.0
        } else {
            self.actions.iter()
                .map(|action| action.estimated_improvement)
                .sum::<f64>() / self.actions.len() as f64
        }
    }
    
    /// Recalculate duration estimate
    fn recalculate_duration(&mut self) {
        self.estimated_duration_hours = self.actions.iter()
            .map(|action| match &action.action_type {
                ActionType::SplitModule { .. } => 4.0,
                ActionType::MergeModules { .. } => 2.0,
                ActionType::MoveFunction { .. } => 1.0,
                ActionType::ReorganizeSections { .. } => 0.5,
            })
            .sum();
    }
    
    /// Recalculate all estimates
    fn recalculate_estimates(&mut self) {
        self.recalculate_duration();
        self.estimated_improvement = self.total_estimated_improvement();
    }
}

impl Default for ActionPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionImpact {
    /// Create a new impact analysis with default values
    pub fn new() -> Self {
        Self {
            cohesion_improvement: 0.0,
            files_affected: 0,
            modules_affected: 0,
            breaking_changes: Vec::new(),
            performance_impact: PerformanceImpact {
                compilation_time_change_percent: 0.0,
                runtime_performance_change_percent: 0.0,
                memory_usage_change_percent: 0.0,
            },
            maintenance_impact: MaintenanceImpact {
                readability_improvement: 0.0,
                testing_effort_change_percent: 0.0,
                documentation_updates_required: Vec::new(),
            },
        }
    }
    
    /// Calculate overall impact score (0-1, higher is better)
    pub fn overall_impact_score(&self) -> f64 {
        let cohesion_weight = 0.4;
        let performance_weight = 0.3;
        let maintenance_weight = 0.3;
        
        let performance_score = 1.0 - (
            (self.performance_impact.compilation_time_change_percent.abs() +
             self.performance_impact.runtime_performance_change_percent.abs() +
             self.performance_impact.memory_usage_change_percent.abs()) / 300.0
        ).min(1.0);
        
        let maintenance_score = (
            self.maintenance_impact.readability_improvement +
            (1.0 - (self.maintenance_impact.testing_effort_change_percent.abs() / 100.0).min(1.0))
        ) / 2.0;
        
        cohesion_weight * self.cohesion_improvement +
        performance_weight * performance_score +
        maintenance_weight * maintenance_score
    }
}

impl Default for ActionImpact {
    fn default() -> Self {
        Self::new()
    }
} 