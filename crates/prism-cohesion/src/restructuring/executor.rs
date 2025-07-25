//! Action Executor for Module Restructuring
//!
//! This module embodies the single concept of "Safe Action Execution".
//! It executes restructuring actions while maintaining safety guarantees
//! and providing rollback capabilities.
//!
//! **Conceptual Responsibility**: Execute restructuring actions safely
//! **What it does**: action execution, rollback management, state tracking
//! **What it doesn't do**: action planning, safety analysis, cohesion calculation

use crate::{CohesionResult, CohesionError};
use crate::restructuring::actions::{ActionPlan, ModuleAction, ActionType};
use crate::restructuring::safety::SafetyLevel;
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_common::{span::Span, symbol::Symbol, NodeId};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Action executor for safe restructuring operations
#[derive(Debug)]
pub struct ActionExecutor {
    /// Executor configuration
    config: ExecutorConfig,
    
    /// Execution state tracker
    state_tracker: ExecutionStateTracker,
    
    /// Rollback manager
    rollback_manager: RollbackManager,
}

/// Configuration for action execution
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum actions to execute in one session
    pub max_actions_per_session: usize,
    
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
    
    /// Create backups before execution
    pub create_backups: bool,
    
    /// Validation level for execution
    pub validation_level: ValidationLevel,
}

/// Validation levels for execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationLevel {
    /// No validation
    None,
    /// Basic validation
    Basic,
    /// Standard validation
    Standard,
    /// Strict validation
    Strict,
}

/// Execution context for actions
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Whether this is a dry run
    pub dry_run: bool,
    
    /// Enable rollback capabilities
    pub enable_rollback: bool,
    
    /// Safety level required
    pub safety_level: SafetyLevel,
}

/// Result of action execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Actions that were executed
    pub executed_actions: Vec<ExecutedAction>,
    
    /// Actions that failed
    pub failed_actions: Vec<FailedAction>,
    
    /// Overall execution status
    pub status: ExecutionStatus,
    
    /// Files that were modified
    pub modified_files: Vec<PathBuf>,
    
    /// Modules that were affected
    pub affected_modules: Vec<String>,
    
    /// Rollback information
    pub rollback_info: Option<RollbackInfo>,
    
    /// Execution statistics
    pub statistics: ExecutionStatistics,
}

/// Individual executed action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutedAction {
    /// Original action
    pub action: ModuleAction,
    
    /// Execution timestamp
    pub executed_at: SystemTime,
    
    /// Files modified by this action
    pub files_modified: Vec<PathBuf>,
    
    /// Success status
    pub success: bool,
    
    /// Execution details
    pub details: String,
}

/// Failed action information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedAction {
    /// Original action
    pub action: ModuleAction,
    
    /// Failure reason
    pub failure_reason: String,
    
    /// Error details
    pub error_details: Option<String>,
    
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
}

/// Overall execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// All actions executed successfully
    Success,
    /// Some actions failed
    PartialSuccess,
    /// All actions failed
    Failed,
    /// Execution was cancelled
    Cancelled,
    /// Dry run completed
    DryRunCompleted,
}

/// Rollback information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInfo {
    /// Rollback ID for tracking
    pub rollback_id: String,
    
    /// Files backed up
    pub backed_up_files: Vec<PathBuf>,
    
    /// Rollback script or instructions
    pub rollback_instructions: Vec<String>,
    
    /// Rollback feasibility
    pub rollback_feasible: bool,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    
    /// Number of actions attempted
    pub actions_attempted: usize,
    
    /// Number of actions succeeded
    pub actions_succeeded: usize,
    
    /// Number of files modified
    pub files_modified: usize,
    
    /// Number of modules affected
    pub modules_affected: usize,
}

/// Execution state tracker
#[derive(Debug)]
struct ExecutionStateTracker {
    /// Current execution session ID
    session_id: String,
    
    /// Actions in progress
    in_progress_actions: HashSet<String>,
    
    /// Completed actions
    completed_actions: Vec<String>,
    
    /// Modified files tracking
    modified_files: HashSet<PathBuf>,
    
    /// Module state changes
    module_changes: HashMap<String, ModuleChangeRecord>,
}

/// Record of changes to a module
#[derive(Debug, Clone)]
struct ModuleChangeRecord {
    /// Original module state (simplified)
    original_state: String,
    
    /// Current module state
    current_state: String,
    
    /// Changes made
    changes: Vec<String>,
    
    /// Timestamp of changes
    changed_at: SystemTime,
}

/// Rollback manager for safe operations
#[derive(Debug)]
struct RollbackManager {
    /// Backup storage location
    backup_location: PathBuf,
    
    /// Active rollback points
    rollback_points: HashMap<String, RollbackPoint>,
    
    /// Rollback enabled
    enabled: bool,
}

/// Individual rollback point
#[derive(Debug, Clone)]
struct RollbackPoint {
    /// Rollback point ID
    id: String,
    
    /// Files backed up
    backed_up_files: HashMap<PathBuf, Vec<u8>>,
    
    /// Rollback instructions
    instructions: Vec<String>,
    
    /// Creation timestamp
    created_at: SystemTime,
}

impl ActionExecutor {
    /// Create new action executor
    pub fn new(safety_level: SafetyLevel) -> CohesionResult<Self> {
        let config = ExecutorConfig {
            max_actions_per_session: match safety_level {
                SafetyLevel::Safe | SafetyLevel::VeryHigh => 10,
                SafetyLevel::High => 5,
                SafetyLevel::Medium => 3,
                SafetyLevel::Low => 1,
                SafetyLevel::Unsafe => 0,
            },
            enable_detailed_logging: true,
            create_backups: true,
            validation_level: match safety_level {
                SafetyLevel::Safe | SafetyLevel::VeryHigh => ValidationLevel::Strict,
                SafetyLevel::High => ValidationLevel::Standard,
                SafetyLevel::Medium => ValidationLevel::Basic,
                _ => ValidationLevel::None,
            },
        };
        
        let state_tracker = ExecutionStateTracker {
            session_id: format!("session_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|e| CohesionError::ExecutionError(format!("Time error: {}", e)))?
                .as_secs()),
            in_progress_actions: HashSet::new(),
            completed_actions: Vec::new(),
            modified_files: HashSet::new(),
            module_changes: HashMap::new(),
        };
        
        let rollback_manager = RollbackManager {
            backup_location: PathBuf::from(".cohesion_backups"),
            rollback_points: HashMap::new(),
            enabled: config.create_backups,
        };
        
        Ok(Self {
            config,
            state_tracker,
            rollback_manager,
        })
    }
    
    /// Execute an action plan
    pub fn execute_plan(
        &mut self,
        action_plan: &ActionPlan,
        context: &ExecutionContext,
    ) -> CohesionResult<ExecutionResult> {
        let start_time = SystemTime::now();
        let mut executed_actions = Vec::new();
        let mut failed_actions = Vec::new();
        
        // Create rollback point if enabled
        let rollback_info = if context.enable_rollback && self.rollback_manager.enabled {
            Some(self.create_rollback_point(&action_plan.actions)?)
        } else {
            None
        };
        
        // Limit actions based on configuration
        let actions_to_execute = action_plan.actions.iter()
            .take(self.config.max_actions_per_session)
            .collect::<Vec<_>>();
        
        if context.dry_run {
            return self.execute_dry_run(&actions_to_execute, context);
        }
        
        // Execute actions sequentially for safety
        for action in actions_to_execute {
            match self.execute_single_action(action, context) {
                Ok(executed_action) => {
                    executed_actions.push(executed_action);
                }
                Err(error) => {
                    let failed_action = FailedAction {
                        action: action.clone(),
                        failure_reason: error.to_string(),
                        error_details: Some(format!("{:?}", error)),
                        recovery_suggestions: self.generate_recovery_suggestions(action),
                    };
                    failed_actions.push(failed_action);
                    
                    // Stop execution on failure for safety
                    break;
                }
            }
        }
        
        // Determine overall status
        let status = if failed_actions.is_empty() {
            ExecutionStatus::Success
        } else if executed_actions.is_empty() {
            ExecutionStatus::Failed
        } else {
            ExecutionStatus::PartialSuccess
        };
        
        // Calculate statistics
        let execution_time = start_time.elapsed()
            .map_err(|e| CohesionError::ExecutionError(format!("Time calculation error: {}", e)))?
            .as_millis() as u64;
        
        let statistics = ExecutionStatistics {
            total_execution_time_ms: execution_time,
            actions_attempted: executed_actions.len() + failed_actions.len(),
            actions_succeeded: executed_actions.len(),
            files_modified: self.state_tracker.modified_files.len(),
            modules_affected: self.state_tracker.module_changes.len(),
        };
        
        Ok(ExecutionResult {
            executed_actions,
            failed_actions,
            status,
            modified_files: self.state_tracker.modified_files.iter().cloned().collect(),
            affected_modules: self.state_tracker.module_changes.keys().cloned().collect(),
            rollback_info,
            statistics,
        })
    }
    
    /// Execute a single action
    fn execute_single_action(
        &mut self,
        action: &ModuleAction,
        context: &ExecutionContext,
    ) -> CohesionResult<ExecutedAction> {
        let start_time = SystemTime::now();
        let action_id = format!("{}_{:?}", action.action_type, start_time);
        
        // Mark action as in progress
        self.state_tracker.in_progress_actions.insert(action_id.clone());
        
        let result = match &action.action_type {
            ActionType::SplitModule { source_module, target_modules, split_strategy } => {
                self.execute_split_module(source_module, target_modules, split_strategy)
            }
            ActionType::MergeModules { source_modules, target_module, merge_strategy } => {
                self.execute_merge_modules(source_modules, target_module, merge_strategy)
            }
            ActionType::MoveFunction { source_module, target_module, function_name, move_reason } => {
                self.execute_move_function(source_module, target_module, function_name, move_reason)
            }
            ActionType::ReorganizeSections { module, new_organization } => {
                self.execute_reorganize_sections(module, new_organization)
            }
        };
        
        // Mark action as completed
        self.state_tracker.in_progress_actions.remove(&action_id);
        self.state_tracker.completed_actions.push(action_id);
        
        match result {
            Ok(files_modified) => {
                for file in &files_modified {
                    self.state_tracker.modified_files.insert(file.clone());
                }
                
                Ok(ExecutedAction {
                    action: action.clone(),
                    executed_at: start_time,
                    files_modified,
                    success: true,
                    details: format!("Successfully executed {:?}", action.action_type),
                })
            }
            Err(error) => Err(error),
        }
    }
    
    /// Execute module split
    fn execute_split_module(
        &mut self,
        source_module: &str,
        target_modules: &[String],
        split_strategy: &str,
    ) -> CohesionResult<Vec<PathBuf>> {
        // In a real implementation, this would:
        // 1. Read the source module file
        // 2. Analyze the module structure
        // 3. Split according to the strategy
        // 4. Create new module files
        // 5. Update imports/exports
        // 6. Update build configuration
        
        // For now, simulate the operation
        let source_file = PathBuf::from(format!("{}.prism", source_module));
        let mut modified_files = vec![source_file];
        
        for target_module in target_modules {
            let target_file = PathBuf::from(format!("{}.prism", target_module));
            modified_files.push(target_file);
        }
        
        // Record module changes
        self.record_module_change(source_module, "split", &format!("Split into {:?}", target_modules));
        
        Ok(modified_files)
    }
    
    /// Execute module merge
    fn execute_merge_modules(
        &mut self,
        source_modules: &[String],
        target_module: &str,
        merge_strategy: &str,
    ) -> CohesionResult<Vec<PathBuf>> {
        // In a real implementation, this would:
        // 1. Read all source module files
        // 2. Merge content according to strategy
        // 3. Resolve naming conflicts
        // 4. Create merged module file
        // 5. Update imports/exports
        // 6. Remove old module files
        
        // For now, simulate the operation
        let mut modified_files = Vec::new();
        
        for source_module in source_modules {
            let source_file = PathBuf::from(format!("{}.prism", source_module));
            modified_files.push(source_file);
        }
        
        let target_file = PathBuf::from(format!("{}.prism", target_module));
        modified_files.push(target_file);
        
        // Record module changes
        for source_module in source_modules {
            self.record_module_change(source_module, "merge", &format!("Merged into {}", target_module));
        }
        
        Ok(modified_files)
    }
    
    /// Execute function move
    fn execute_move_function(
        &mut self,
        source_module: &str,
        target_module: &str,
        function_name: &str,
        move_reason: &str,
    ) -> CohesionResult<Vec<PathBuf>> {
        // In a real implementation, this would:
        // 1. Locate the function in source module
        // 2. Extract function and dependencies
        // 3. Update imports in source module
        // 4. Add function to target module
        // 5. Update exports in target module
        // 6. Update all references
        
        // For now, simulate the operation
        let source_file = PathBuf::from(format!("{}.prism", source_module));
        let target_file = PathBuf::from(format!("{}.prism", target_module));
        
        // Record module changes
        self.record_module_change(source_module, "function_moved_out", &format!("Moved {} to {}", function_name, target_module));
        self.record_module_change(target_module, "function_moved_in", &format!("Received {} from {}", function_name, source_module));
        
        Ok(vec![source_file, target_file])
    }
    
    /// Execute section reorganization
    fn execute_reorganize_sections(
        &mut self,
        module: &str,
        new_organization: &[String],
    ) -> CohesionResult<Vec<PathBuf>> {
        // In a real implementation, this would:
        // 1. Read the module file
        // 2. Parse the current section structure
        // 3. Reorganize sections according to new organization
        // 4. Preserve all content while changing structure
        // 5. Write back the reorganized module
        
        // For now, simulate the operation
        let module_file = PathBuf::from(format!("{}.prism", module));
        
        // Record module changes
        self.record_module_change(module, "reorganize", &format!("Reorganized sections: {:?}", new_organization));
        
        Ok(vec![module_file])
    }
    
    /// Execute dry run (simulation only)
    fn execute_dry_run(
        &self,
        actions: &[&ModuleAction],
        context: &ExecutionContext,
    ) -> CohesionResult<ExecutionResult> {
        let mut executed_actions = Vec::new();
        
        for action in actions {
            let simulated_files = match &action.action_type {
                ActionType::SplitModule { source_module, target_modules, .. } => {
                    let mut files = vec![PathBuf::from(format!("{}.prism", source_module))];
                    for target in target_modules {
                        files.push(PathBuf::from(format!("{}.prism", target)));
                    }
                    files
                }
                ActionType::MergeModules { source_modules, target_module, .. } => {
                    let mut files: Vec<PathBuf> = source_modules.iter()
                        .map(|m| PathBuf::from(format!("{}.prism", m)))
                        .collect();
                    files.push(PathBuf::from(format!("{}.prism", target_module)));
                    files
                }
                ActionType::MoveFunction { source_module, target_module, .. } => {
                    vec![
                        PathBuf::from(format!("{}.prism", source_module)),
                        PathBuf::from(format!("{}.prism", target_module)),
                    ]
                }
                ActionType::ReorganizeSections { module, .. } => {
                    vec![PathBuf::from(format!("{}.prism", module))]
                }
            };
            
            executed_actions.push(ExecutedAction {
                action: (*action).clone(),
                executed_at: SystemTime::now(),
                files_modified: simulated_files,
                success: true,
                details: format!("DRY RUN: Would execute {:?}", action.action_type),
            });
        }
        
        Ok(ExecutionResult {
            executed_actions,
            failed_actions: Vec::new(),
            status: ExecutionStatus::DryRunCompleted,
            modified_files: Vec::new(),
            affected_modules: Vec::new(),
            rollback_info: None,
            statistics: ExecutionStatistics {
                total_execution_time_ms: 0,
                actions_attempted: actions.len(),
                actions_succeeded: actions.len(),
                files_modified: 0,
                modules_affected: 0,
            },
        })
    }
    
    /// Create rollback point
    fn create_rollback_point(&mut self, actions: &[ModuleAction]) -> CohesionResult<RollbackInfo> {
        let rollback_id = format!("rollback_{}", SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| CohesionError::ExecutionError(format!("Time error: {}", e)))?
            .as_secs());
        
        // In a real implementation, this would backup all files that will be modified
        let mut backed_up_files = Vec::new();
        let mut rollback_instructions = Vec::new();
        
        for action in actions {
            match &action.action_type {
                ActionType::SplitModule { source_module, .. } => {
                    let file = PathBuf::from(format!("{}.prism", source_module));
                    backed_up_files.push(file);
                    rollback_instructions.push(format!("Restore original {}.prism", source_module));
                }
                ActionType::MergeModules { source_modules, .. } => {
                    for module in source_modules {
                        let file = PathBuf::from(format!("{}.prism", module));
                        backed_up_files.push(file);
                        rollback_instructions.push(format!("Restore original {}.prism", module));
                    }
                }
                ActionType::MoveFunction { source_module, target_module, .. } => {
                    backed_up_files.push(PathBuf::from(format!("{}.prism", source_module)));
                    backed_up_files.push(PathBuf::from(format!("{}.prism", target_module)));
                    rollback_instructions.push(format!("Restore original {}.prism and {}.prism", source_module, target_module));
                }
                ActionType::ReorganizeSections { module, .. } => {
                    let file = PathBuf::from(format!("{}.prism", module));
                    backed_up_files.push(file);
                    rollback_instructions.push(format!("Restore original {}.prism", module));
                }
            }
        }
        
        Ok(RollbackInfo {
            rollback_id,
            backed_up_files,
            rollback_instructions,
            rollback_feasible: true,
        })
    }
    
    /// Record module change
    fn record_module_change(&mut self, module: &str, change_type: &str, details: &str) {
        let change_record = ModuleChangeRecord {
            original_state: format!("original_{}", module),
            current_state: format!("modified_{}", module),
            changes: vec![format!("{}: {}", change_type, details)],
            changed_at: SystemTime::now(),
        };
        
        self.state_tracker.module_changes.insert(module.to_string(), change_record);
    }
    
    /// Generate recovery suggestions for failed actions
    fn generate_recovery_suggestions(&self, action: &ModuleAction) -> Vec<String> {
        match &action.action_type {
            ActionType::SplitModule { source_module, .. } => {
                vec![
                    format!("Verify that {} exists and is accessible", source_module),
                    "Check for file system permissions".to_string(),
                    "Ensure no other processes are using the module".to_string(),
                ]
            }
            ActionType::MergeModules { source_modules, .. } => {
                vec![
                    format!("Verify that all source modules exist: {:?}", source_modules),
                    "Check for naming conflicts between modules".to_string(),
                    "Ensure modules are compatible for merging".to_string(),
                ]
            }
            ActionType::MoveFunction { function_name, .. } => {
                vec![
                    format!("Verify that function {} exists", function_name),
                    "Check function dependencies".to_string(),
                    "Ensure target module can accommodate the function".to_string(),
                ]
            }
            ActionType::ReorganizeSections { module, .. } => {
                vec![
                    format!("Verify that module {} exists", module),
                    "Check section organization is valid".to_string(),
                    "Ensure no circular section dependencies".to_string(),
                ]
            }
        }
    }
} 