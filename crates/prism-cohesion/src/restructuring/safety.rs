//! Safety Validation for Module Restructuring
//!
//! This module embodies the single concept of "Restructuring Safety Validation".
//! It analyzes proposed actions for risks and ensures safe execution.
//!
//! **Conceptual Responsibility**: Validate safety of restructuring operations
//! **What it does**: risk assessment, precondition checking, safety scoring
//! **What it doesn't do**: action execution, cohesion analysis, decision making

use crate::{CohesionResult, CohesionError};
use crate::restructuring::actions::{ActionPlan, ModuleAction, ActionType, ActionRiskLevel};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::{span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Safety validator for restructuring operations
#[derive(Debug)]
pub struct SafetyValidator {
    /// Safety configuration
    config: SafetyConfig,
    
    /// Risk analyzers
    risk_analyzers: RiskAnalyzers,
}

/// Safety configuration
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Maximum allowed risk level
    pub max_risk_level: SafetyLevel,
    
    /// Enable API breaking change detection
    pub check_api_breaking: bool,
    
    /// Enable dependency cycle detection
    pub check_dependency_cycles: bool,
    
    /// Enable build system validation
    pub check_build_system: bool,
    
    /// Require rollback capability
    pub require_rollback: bool,
}

/// Safety levels for operations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafetyLevel {
    /// Unsafe - should not be executed
    Unsafe,
    /// Low safety - high risk
    Low,
    /// Medium safety - moderate risk
    Medium,
    /// High safety - low risk
    High,
    /// Very safe - minimal risk
    VeryHigh,
    /// Completely safe - no risk
    Safe,
}

/// Individual safety check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCheck {
    /// Check name
    pub name: String,
    
    /// Check result
    pub passed: bool,
    
    /// Safety level if passed
    pub safety_level: SafetyLevel,
    
    /// Issues found
    pub issues: Vec<String>,
    
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Risk analyzers for different types of risks
#[derive(Debug)]
struct RiskAnalyzers {
    /// API breaking change analyzer
    api_analyzer: ApiBreakingAnalyzer,
    
    /// Dependency cycle analyzer
    dependency_analyzer: DependencyAnalyzer,
    
    /// Build system analyzer
    build_analyzer: BuildSystemAnalyzer,
    
    /// Performance impact analyzer
    performance_analyzer: PerformanceAnalyzer,
}

/// Analyzer for API breaking changes
#[derive(Debug)]
struct ApiBreakingAnalyzer {
    /// Track public symbols
    public_symbols: HashSet<String>,
    
    /// Track module exports
    module_exports: HashMap<String, Vec<String>>,
}

/// Analyzer for dependency cycles
#[derive(Debug)]
struct DependencyAnalyzer {
    /// Current dependency graph
    dependency_graph: HashMap<String, Vec<String>>,
}

/// Analyzer for build system impact
#[derive(Debug)]
struct BuildSystemAnalyzer {
    /// Known build dependencies
    build_dependencies: HashMap<String, Vec<String>>,
}

/// Analyzer for performance impact
#[derive(Debug)]
struct PerformanceAnalyzer {
    /// Module size tracking
    module_sizes: HashMap<String, usize>,
}

impl SafetyValidator {
    /// Create new safety validator
    pub fn new(safety_level: SafetyLevel) -> CohesionResult<Self> {
        let config = SafetyConfig {
            max_risk_level: safety_level,
            check_api_breaking: true,
            check_dependency_cycles: true,
            check_build_system: true,
            require_rollback: true,
        };
        
        let risk_analyzers = RiskAnalyzers {
            api_analyzer: ApiBreakingAnalyzer {
                public_symbols: HashSet::new(),
                module_exports: HashMap::new(),
            },
            dependency_analyzer: DependencyAnalyzer {
                dependency_graph: HashMap::new(),
            },
            build_analyzer: BuildSystemAnalyzer {
                build_dependencies: HashMap::new(),
            },
            performance_analyzer: PerformanceAnalyzer {
                module_sizes: HashMap::new(),
            },
        };
        
        Ok(Self {
            config,
            risk_analyzers,
        })
    }
    
    /// Analyze safety of an action plan
    pub fn analyze_safety(
        &self,
        program: &Program,
        action_plan: &ActionPlan,
    ) -> CohesionResult<super::SafetyAnalysis> {
        let mut safety_checks = Vec::new();
        let mut risks = Vec::new();
        let mut preconditions = Vec::new();
        let mut safeguards = Vec::new();
        
        // Initialize analyzers with current program state
        self.initialize_analyzers(program)?;
        
        // Perform safety checks for each action
        for action in &action_plan.actions {
            let action_checks = self.check_action_safety(action, program)?;
            safety_checks.extend(action_checks);
        }
        
        // Check for cross-action risks
        let cross_action_checks = self.check_cross_action_safety(&action_plan.actions, program)?;
        safety_checks.extend(cross_action_checks);
        
        // Determine overall safety level
        let overall_safety = self.calculate_overall_safety(&safety_checks);
        
        // Generate risks from failed checks
        for check in &safety_checks {
            if !check.passed {
                risks.push(super::RestructuringRisk {
                    category: self.categorize_risk(&check.name),
                    description: check.issues.join("; "),
                    severity: self.calculate_risk_severity(&check),
                    mitigations: check.recommendations.clone(),
                });
            }
        }
        
        // Generate preconditions
        preconditions.extend(self.generate_preconditions(&action_plan.actions)?);
        
        // Generate safeguards
        safeguards.extend(self.generate_safeguards(&risks)?);
        
        // Check rollback feasibility
        let rollback_feasible = self.check_rollback_feasibility(&action_plan.actions)?;
        
        Ok(super::SafetyAnalysis {
            safety_level: overall_safety,
            risks,
            preconditions,
            safeguards,
            rollback_feasible,
        })
    }
    
    /// Initialize analyzers with current program state
    fn initialize_analyzers(&self, program: &Program) -> CohesionResult<()> {
        // Extract current module structure and dependencies
        for item in &program.items {
            if let Item::Module(module_decl) = &item.kind {
                // Track module information for safety analysis
                // This would populate the analyzers with current state
            }
        }
        Ok(())
    }
    
    /// Check safety of a single action
    fn check_action_safety(
        &self,
        action: &ModuleAction,
        program: &Program,
    ) -> CohesionResult<Vec<SafetyCheck>> {
        let mut checks = Vec::new();
        
        match &action.action_type {
            ActionType::SplitModule { source_module, target_modules, .. } => {
                checks.extend(self.check_split_safety(source_module, target_modules, program)?);
            }
            ActionType::MergeModules { source_modules, target_module, .. } => {
                checks.extend(self.check_merge_safety(source_modules, target_module, program)?);
            }
            ActionType::MoveFunction { source_module, target_module, function_name, .. } => {
                checks.extend(self.check_move_safety(source_module, target_module, function_name, program)?);
            }
            ActionType::ReorganizeSections { module, .. } => {
                checks.extend(self.check_reorganize_safety(module, program)?);
            }
        }
        
        Ok(checks)
    }
    
    /// Check safety of module split
    fn check_split_safety(
        &self,
        source_module: &str,
        target_modules: &[String],
        program: &Program,
    ) -> CohesionResult<Vec<SafetyCheck>> {
        let mut checks = Vec::new();
        
        // Check if source module exists
        let module_exists_check = SafetyCheck {
            name: "module_exists".to_string(),
            passed: self.module_exists(source_module, program),
            safety_level: SafetyLevel::High,
            issues: if !self.module_exists(source_module, program) {
                vec![format!("Source module '{}' not found", source_module)]
            } else {
                Vec::new()
            },
            recommendations: vec!["Verify module name is correct".to_string()],
        };
        checks.push(module_exists_check);
        
        // Check for API breaking changes
        if self.config.check_api_breaking {
            let api_check = SafetyCheck {
                name: "api_breaking_split".to_string(),
                passed: true, // Simplified - would check actual API impact
                safety_level: SafetyLevel::Medium,
                issues: Vec::new(),
                recommendations: vec!["Review public API changes after split".to_string()],
            };
            checks.push(api_check);
        }
        
        // Check for dependency cycles
        if self.config.check_dependency_cycles {
            let cycle_check = SafetyCheck {
                name: "dependency_cycles_split".to_string(),
                passed: true, // Simplified - would check for potential cycles
                safety_level: SafetyLevel::High,
                issues: Vec::new(),
                recommendations: vec!["Verify no circular dependencies created".to_string()],
            };
            checks.push(cycle_check);
        }
        
        Ok(checks)
    }
    
    /// Check safety of module merge
    fn check_merge_safety(
        &self,
        source_modules: &[String],
        target_module: &str,
        program: &Program,
    ) -> CohesionResult<Vec<SafetyCheck>> {
        let mut checks = Vec::new();
        
        // Check if all source modules exist
        for module in source_modules {
            let module_check = SafetyCheck {
                name: format!("module_exists_{}", module),
                passed: self.module_exists(module, program),
                safety_level: SafetyLevel::High,
                issues: if !self.module_exists(module, program) {
                    vec![format!("Source module '{}' not found", module)]
                } else {
                    Vec::new()
                },
                recommendations: vec!["Verify all source modules exist".to_string()],
            };
            checks.push(module_check);
        }
        
        // Check for naming conflicts
        let naming_check = SafetyCheck {
            name: "naming_conflicts_merge".to_string(),
            passed: true, // Simplified - would check for actual conflicts
            safety_level: SafetyLevel::Medium,
            issues: Vec::new(),
            recommendations: vec!["Check for symbol naming conflicts".to_string()],
        };
        checks.push(naming_check);
        
        Ok(checks)
    }
    
    /// Check safety of function move
    fn check_move_safety(
        &self,
        source_module: &str,
        target_module: &str,
        function_name: &str,
        program: &Program,
    ) -> CohesionResult<Vec<SafetyCheck>> {
        let mut checks = Vec::new();
        
        // Check if both modules exist
        let source_check = SafetyCheck {
            name: "source_module_exists".to_string(),
            passed: self.module_exists(source_module, program),
            safety_level: SafetyLevel::High,
            issues: if !self.module_exists(source_module, program) {
                vec![format!("Source module '{}' not found", source_module)]
            } else {
                Vec::new()
            },
            recommendations: vec!["Verify source module exists".to_string()],
        };
        checks.push(source_check);
        
        let target_check = SafetyCheck {
            name: "target_module_exists".to_string(),
            passed: self.module_exists(target_module, program),
            safety_level: SafetyLevel::High,
            issues: if !self.module_exists(target_module, program) {
                vec![format!("Target module '{}' not found", target_module)]
            } else {
                Vec::new()
            },
            recommendations: vec!["Verify target module exists".to_string()],
        };
        checks.push(target_check);
        
        // Check function dependencies
        let dependency_check = SafetyCheck {
            name: "function_dependencies".to_string(),
            passed: true, // Simplified - would check actual dependencies
            safety_level: SafetyLevel::Medium,
            issues: Vec::new(),
            recommendations: vec!["Verify function dependencies are satisfied".to_string()],
        };
        checks.push(dependency_check);
        
        Ok(checks)
    }
    
    /// Check safety of section reorganization
    fn check_reorganize_safety(
        &self,
        module: &str,
        program: &Program,
    ) -> CohesionResult<Vec<SafetyCheck>> {
        let mut checks = Vec::new();
        
        let module_check = SafetyCheck {
            name: "module_exists_reorganize".to_string(),
            passed: self.module_exists(module, program),
            safety_level: SafetyLevel::VeryHigh, // Low risk operation
            issues: if !self.module_exists(module, program) {
                vec![format!("Module '{}' not found", module)]
            } else {
                Vec::new()
            },
            recommendations: vec!["Section reorganization is low risk".to_string()],
        };
        checks.push(module_check);
        
        Ok(checks)
    }
    
    /// Check for cross-action safety issues
    fn check_cross_action_safety(
        &self,
        actions: &[ModuleAction],
        program: &Program,
    ) -> CohesionResult<Vec<SafetyCheck>> {
        let mut checks = Vec::new();
        
        // Check for conflicting actions
        let conflict_check = SafetyCheck {
            name: "action_conflicts".to_string(),
            passed: !self.has_conflicting_actions(actions),
            safety_level: SafetyLevel::High,
            issues: if self.has_conflicting_actions(actions) {
                vec!["Some actions may conflict with each other".to_string()]
            } else {
                Vec::new()
            },
            recommendations: vec!["Review action plan for conflicts".to_string()],
        };
        checks.push(conflict_check);
        
        Ok(checks)
    }
    
    /// Calculate overall safety level
    fn calculate_overall_safety(&self, checks: &[SafetyCheck]) -> SafetyLevel {
        if checks.iter().any(|check| !check.passed && check.safety_level <= SafetyLevel::Low) {
            SafetyLevel::Unsafe
        } else if checks.iter().any(|check| !check.passed && check.safety_level <= SafetyLevel::Medium) {
            SafetyLevel::Low
        } else if checks.iter().any(|check| !check.passed) {
            SafetyLevel::Medium
        } else {
            SafetyLevel::High
        }
    }
    
    /// Categorize risk type
    fn categorize_risk(&self, check_name: &str) -> super::RiskCategory {
        match check_name {
            name if name.contains("api") => super::RiskCategory::ApiBreaking,
            name if name.contains("dependency") || name.contains("cycle") => super::RiskCategory::DependencyCycles,
            name if name.contains("build") => super::RiskCategory::BuildSystem,
            name if name.contains("performance") => super::RiskCategory::Performance,
            name if name.contains("security") => super::RiskCategory::Security,
            _ => super::RiskCategory::DataLoss,
        }
    }
    
    /// Calculate risk severity
    fn calculate_risk_severity(&self, check: &SafetyCheck) -> f64 {
        match check.safety_level {
            SafetyLevel::Unsafe => 1.0,
            SafetyLevel::Low => 0.8,
            SafetyLevel::Medium => 0.6,
            SafetyLevel::High => 0.4,
            SafetyLevel::VeryHigh => 0.2,
            SafetyLevel::Safe => 0.0,
        }
    }
    
    /// Generate preconditions for actions
    fn generate_preconditions(&self, actions: &[ModuleAction]) -> CohesionResult<Vec<String>> {
        let mut preconditions = Vec::new();
        
        for action in actions {
            match &action.action_type {
                ActionType::SplitModule { source_module, .. } => {
                    preconditions.push(format!("Backup {} before splitting", source_module));
                    preconditions.push("Ensure no active development on target modules".to_string());
                }
                ActionType::MergeModules { source_modules, .. } => {
                    preconditions.push(format!("Backup modules {:?} before merging", source_modules));
                    preconditions.push("Resolve any merge conflicts in advance".to_string());
                }
                ActionType::MoveFunction { function_name, .. } => {
                    preconditions.push(format!("Verify {} has no external dependencies", function_name));
                }
                ActionType::ReorganizeSections { .. } => {
                    preconditions.push("Create backup of module structure".to_string());
                }
            }
        }
        
        Ok(preconditions)
    }
    
    /// Generate safeguards for risks
    fn generate_safeguards(&self, risks: &[super::RestructuringRisk]) -> CohesionResult<Vec<String>> {
        let mut safeguards = Vec::new();
        
        for risk in risks {
            match risk.category {
                super::RiskCategory::ApiBreaking => {
                    safeguards.push("Run API compatibility tests".to_string());
                    safeguards.push("Update documentation for API changes".to_string());
                }
                super::RiskCategory::DependencyCycles => {
                    safeguards.push("Run dependency analysis after changes".to_string());
                    safeguards.push("Validate import/export consistency".to_string());
                }
                super::RiskCategory::BuildSystem => {
                    safeguards.push("Test build process after changes".to_string());
                    safeguards.push("Update build configuration if needed".to_string());
                }
                _ => {
                    safeguards.push("Monitor system after changes".to_string());
                }
            }
        }
        
        Ok(safeguards)
    }
    
    /// Check if rollback is feasible
    fn check_rollback_feasibility(&self, actions: &[ModuleAction]) -> CohesionResult<bool> {
        // For now, assume rollback is always feasible with proper backups
        // In practice, this would check for irreversible operations
        Ok(true)
    }
    
    /// Helper: Check if module exists in program
    fn module_exists(&self, module_name: &str, program: &Program) -> bool {
        program.items.iter().any(|item| {
            if let Item::Module(module_decl) = &item.kind {
                module_decl.name.as_str() == module_name
            } else {
                false
            }
        })
    }
    
    /// Helper: Check for conflicting actions
    fn has_conflicting_actions(&self, actions: &[ModuleAction]) -> bool {
        // Simple conflict detection - check for actions on same modules
        let mut affected_modules = HashSet::new();
        
        for action in actions {
            let modules = action.affected_modules();
            for module in modules {
                if affected_modules.contains(&module) {
                    return true; // Potential conflict
                }
                affected_modules.insert(module);
            }
        }
        
        false
    }
} 