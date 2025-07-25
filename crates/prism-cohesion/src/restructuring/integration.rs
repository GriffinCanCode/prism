//! Compiler Integration for Module Restructuring
//!
//! This module embodies the single concept of "Compiler System Integration".
//! It connects the cohesion-driven restructuring system to the existing
//! compiler infrastructure without duplicating functionality.
//!
//! **Conceptual Responsibility**: Bridge restructuring system to compiler
//! **What it does**: integration coordination, system updates, incremental compilation
//! **What it doesn't do**: restructuring logic, cohesion analysis, action execution

use crate::{CohesionResult, CohesionError};
use crate::restructuring::{RestructuringAnalysis, CohesionRestructuringSystem};
use crate::restructuring::executor::ExecutionResult;
use prism_ast::Program;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::collections::HashMap;

/// Integration status with compiler systems
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationStatus {
    /// Not integrated
    NotIntegrated,
    /// Integration in progress
    Integrating,
    /// Successfully integrated
    Integrated,
    /// Integration failed
    Failed(String),
    /// Integration partially successful
    PartiallyIntegrated(Vec<String>),
}

/// Compiler integration for restructuring system
#[derive(Debug)]
pub struct CompilerIntegration {
    /// Integration configuration
    config: IntegrationConfig,
    
    /// Current integration status
    status: IntegrationStatus,
    
    /// Integration adapters for different compiler systems
    adapters: IntegrationAdapters,
}

/// Configuration for compiler integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable module registry updates
    pub enable_module_registry_updates: bool,
    
    /// Enable query engine integration
    pub enable_query_engine_integration: bool,
    
    /// Enable semantic database updates
    pub enable_semantic_database_updates: bool,
    
    /// Enable incremental compilation triggering
    pub enable_incremental_compilation: bool,
    
    /// Integration timeout in seconds
    pub integration_timeout_seconds: u64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_module_registry_updates: true,
            enable_query_engine_integration: true,
            enable_semantic_database_updates: true,
            enable_incremental_compilation: true,
            integration_timeout_seconds: 30,
        }
    }
}

/// Integration adapters for different compiler systems
#[derive(Debug)]
struct IntegrationAdapters {
    /// Module registry adapter
    module_registry_adapter: ModuleRegistryAdapter,
    
    /// Query engine adapter
    query_engine_adapter: QueryEngineAdapter,
    
    /// Semantic database adapter
    semantic_database_adapter: SemanticDatabaseAdapter,
}

/// Adapter for module registry integration
#[derive(Debug)]
struct ModuleRegistryAdapter {
    /// Whether adapter is enabled
    enabled: bool,
    
    /// Pending module updates
    pending_updates: HashMap<String, ModuleUpdate>,
}

/// Adapter for query engine integration
#[derive(Debug)]
struct QueryEngineAdapter {
    /// Whether adapter is enabled
    enabled: bool,
    
    /// Pending query invalidations
    pending_invalidations: Vec<String>,
}

/// Adapter for semantic database integration
#[derive(Debug)]
struct SemanticDatabaseAdapter {
    /// Whether adapter is enabled
    enabled: bool,
    
    /// Pending semantic updates
    pending_updates: Vec<SemanticUpdate>,
}

/// Module update information
#[derive(Debug, Clone)]
struct ModuleUpdate {
    /// Module name
    module_name: String,
    
    /// Update type
    update_type: ModuleUpdateType,
    
    /// Update details
    details: String,
}

/// Types of module updates
#[derive(Debug, Clone)]
enum ModuleUpdateType {
    /// Module was split
    Split { target_modules: Vec<String> },
    
    /// Module was merged
    Merge { source_modules: Vec<String> },
    
    /// Module was modified
    Modified { changes: Vec<String> },
    
    /// Module was reorganized
    Reorganized { new_structure: Vec<String> },
}

/// Semantic update information
#[derive(Debug, Clone)]
struct SemanticUpdate {
    /// Update type
    update_type: String,
    
    /// Affected symbols
    affected_symbols: Vec<String>,
    
    /// Update details
    details: String,
}

/// Integration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Integration status
    pub status: IntegrationStatus,
    
    /// Systems successfully updated
    pub updated_systems: Vec<String>,
    
    /// Systems that failed to update
    pub failed_systems: Vec<String>,
    
    /// Integration messages
    pub messages: Vec<String>,
    
    /// Integration statistics
    pub statistics: IntegrationStatistics,
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStatistics {
    /// Total integration time in milliseconds
    pub total_integration_time_ms: u64,
    
    /// Number of systems updated
    pub systems_updated: usize,
    
    /// Number of modules updated
    pub modules_updated: usize,
    
    /// Number of queries invalidated
    pub queries_invalidated: usize,
    
    /// Number of semantic updates applied
    pub semantic_updates_applied: usize,
}

impl CompilerIntegration {
    /// Create new compiler integration
    pub fn new(config: IntegrationConfig) -> Self {
        let adapters = IntegrationAdapters {
            module_registry_adapter: ModuleRegistryAdapter {
                enabled: config.enable_module_registry_updates,
                pending_updates: HashMap::new(),
            },
            query_engine_adapter: QueryEngineAdapter {
                enabled: config.enable_query_engine_integration,
                pending_invalidations: Vec::new(),
            },
            semantic_database_adapter: SemanticDatabaseAdapter {
                enabled: config.enable_semantic_database_updates,
                pending_updates: Vec::new(),
            },
        };
        
        Self {
            config,
            status: IntegrationStatus::NotIntegrated,
            adapters,
        }
    }
    
    /// Integrate restructuring results with compiler systems
    pub fn integrate_restructuring_results(
        &mut self,
        execution_result: &ExecutionResult,
        program: &Program,
    ) -> CohesionResult<IntegrationResult> {
        let start_time = std::time::SystemTime::now();
        self.status = IntegrationStatus::Integrating;
        
        let mut updated_systems = Vec::new();
        let mut failed_systems = Vec::new();
        let mut messages = Vec::new();
        
        // Update module registry if enabled
        if self.config.enable_module_registry_updates {
            match self.update_module_registry(execution_result, program) {
                Ok(_) => {
                    updated_systems.push("module_registry".to_string());
                    messages.push("Module registry updated successfully".to_string());
                }
                Err(error) => {
                    failed_systems.push("module_registry".to_string());
                    messages.push(format!("Module registry update failed: {}", error));
                }
            }
        }
        
        // Update query engine if enabled
        if self.config.enable_query_engine_integration {
            match self.update_query_engine(execution_result) {
                Ok(_) => {
                    updated_systems.push("query_engine".to_string());
                    messages.push("Query engine updated successfully".to_string());
                }
                Err(error) => {
                    failed_systems.push("query_engine".to_string());
                    messages.push(format!("Query engine update failed: {}", error));
                }
            }
        }
        
        // Update semantic database if enabled
        if self.config.enable_semantic_database_updates {
            match self.update_semantic_database(execution_result) {
                Ok(_) => {
                    updated_systems.push("semantic_database".to_string());
                    messages.push("Semantic database updated successfully".to_string());
                }
                Err(error) => {
                    failed_systems.push("semantic_database".to_string());
                    messages.push(format!("Semantic database update failed: {}", error));
                }
            }
        }
        
        // Trigger incremental compilation if enabled
        if self.config.enable_incremental_compilation {
            match self.trigger_incremental_compilation(execution_result) {
                Ok(_) => {
                    updated_systems.push("incremental_compilation".to_string());
                    messages.push("Incremental compilation triggered successfully".to_string());
                }
                Err(error) => {
                    failed_systems.push("incremental_compilation".to_string());
                    messages.push(format!("Incremental compilation failed: {}", error));
                }
            }
        }
        
        // Determine final status
        self.status = if failed_systems.is_empty() {
            IntegrationStatus::Integrated
        } else if updated_systems.is_empty() {
            IntegrationStatus::Failed("All systems failed to update".to_string())
        } else {
            IntegrationStatus::PartiallyIntegrated(failed_systems.clone())
        };
        
        // Calculate statistics
        let integration_time = start_time.elapsed()
            .map_err(|e| CohesionError::IntegrationError(format!("Time calculation error: {}", e)))?
            .as_millis() as u64;
        
        let statistics = IntegrationStatistics {
            total_integration_time_ms: integration_time,
            systems_updated: updated_systems.len(),
            modules_updated: execution_result.affected_modules.len(),
            queries_invalidated: self.adapters.query_engine_adapter.pending_invalidations.len(),
            semantic_updates_applied: self.adapters.semantic_database_adapter.pending_updates.len(),
        };
        
        Ok(IntegrationResult {
            status: self.status.clone(),
            updated_systems,
            failed_systems,
            messages,
            statistics,
        })
    }
    
    /// Update module registry with restructuring results
    fn update_module_registry(
        &mut self,
        execution_result: &ExecutionResult,
        program: &Program,
    ) -> CohesionResult<()> {
        if !self.adapters.module_registry_adapter.enabled {
            return Ok(());
        }
        
        // Process each executed action and update module registry accordingly
        for executed_action in &execution_result.executed_actions {
            match &executed_action.action.action_type {
                crate::restructuring::actions::ActionType::SplitModule { source_module, target_modules, .. } => {
                    let update = ModuleUpdate {
                        module_name: source_module.clone(),
                        update_type: ModuleUpdateType::Split { target_modules: target_modules.clone() },
                        details: format!("Module {} split into {:?}", source_module, target_modules),
                    };
                    self.adapters.module_registry_adapter.pending_updates.insert(source_module.clone(), update);
                }
                crate::restructuring::actions::ActionType::MergeModules { source_modules, target_module, .. } => {
                    let update = ModuleUpdate {
                        module_name: target_module.clone(),
                        update_type: ModuleUpdateType::Merge { source_modules: source_modules.clone() },
                        details: format!("Modules {:?} merged into {}", source_modules, target_module),
                    };
                    self.adapters.module_registry_adapter.pending_updates.insert(target_module.clone(), update);
                }
                crate::restructuring::actions::ActionType::MoveFunction { source_module, target_module, function_name, .. } => {
                    let source_update = ModuleUpdate {
                        module_name: source_module.clone(),
                        update_type: ModuleUpdateType::Modified { 
                            changes: vec![format!("Function {} moved out", function_name)] 
                        },
                        details: format!("Function {} moved from {} to {}", function_name, source_module, target_module),
                    };
                    self.adapters.module_registry_adapter.pending_updates.insert(source_module.clone(), source_update);
                    
                    let target_update = ModuleUpdate {
                        module_name: target_module.clone(),
                        update_type: ModuleUpdateType::Modified { 
                            changes: vec![format!("Function {} moved in", function_name)] 
                        },
                        details: format!("Function {} moved from {} to {}", function_name, source_module, target_module),
                    };
                    self.adapters.module_registry_adapter.pending_updates.insert(target_module.clone(), target_update);
                }
                crate::restructuring::actions::ActionType::ReorganizeSections { module, new_organization } => {
                    let update = ModuleUpdate {
                        module_name: module.clone(),
                        update_type: ModuleUpdateType::Reorganized { new_structure: new_organization.clone() },
                        details: format!("Module {} sections reorganized: {:?}", module, new_organization),
                    };
                    self.adapters.module_registry_adapter.pending_updates.insert(module.clone(), update);
                }
            }
        }
        
        // In a real implementation, this would call the actual module registry API
        // For now, we just clear the pending updates to simulate successful processing
        self.adapters.module_registry_adapter.pending_updates.clear();
        
        Ok(())
    }
    
    /// Update query engine with restructuring results
    fn update_query_engine(&mut self, execution_result: &ExecutionResult) -> CohesionResult<()> {
        if !self.adapters.query_engine_adapter.enabled {
            return Ok(());
        }
        
        // Invalidate queries related to modified modules
        for module in &execution_result.affected_modules {
            self.adapters.query_engine_adapter.pending_invalidations.push(
                format!("module_analysis_{}", module)
            );
            self.adapters.query_engine_adapter.pending_invalidations.push(
                format!("symbol_resolution_{}", module)
            );
            self.adapters.query_engine_adapter.pending_invalidations.push(
                format!("cohesion_metrics_{}", module)
            );
        }
        
        // In a real implementation, this would call the actual query engine API
        // For now, we just clear the pending invalidations to simulate successful processing
        self.adapters.query_engine_adapter.pending_invalidations.clear();
        
        Ok(())
    }
    
    /// Update semantic database with restructuring results
    fn update_semantic_database(&mut self, execution_result: &ExecutionResult) -> CohesionResult<()> {
        if !self.adapters.semantic_database_adapter.enabled {
            return Ok(());
        }
        
        // Generate semantic updates for each executed action
        for executed_action in &execution_result.executed_actions {
            let semantic_update = match &executed_action.action.action_type {
                crate::restructuring::actions::ActionType::SplitModule { source_module, target_modules, .. } => {
                    SemanticUpdate {
                        update_type: "module_split".to_string(),
                        affected_symbols: vec![source_module.clone()],
                        details: format!("Module {} split into {:?}", source_module, target_modules),
                    }
                }
                crate::restructuring::actions::ActionType::MergeModules { source_modules, target_module, .. } => {
                    SemanticUpdate {
                        update_type: "module_merge".to_string(),
                        affected_symbols: source_modules.clone(),
                        details: format!("Modules {:?} merged into {}", source_modules, target_module),
                    }
                }
                crate::restructuring::actions::ActionType::MoveFunction { source_module, target_module, function_name, .. } => {
                    SemanticUpdate {
                        update_type: "function_move".to_string(),
                        affected_symbols: vec![function_name.clone()],
                        details: format!("Function {} moved from {} to {}", function_name, source_module, target_module),
                    }
                }
                crate::restructuring::actions::ActionType::ReorganizeSections { module, .. } => {
                    SemanticUpdate {
                        update_type: "section_reorganization".to_string(),
                        affected_symbols: vec![module.clone()],
                        details: format!("Module {} sections reorganized", module),
                    }
                }
            };
            
            self.adapters.semantic_database_adapter.pending_updates.push(semantic_update);
        }
        
        // In a real implementation, this would call the actual semantic database API
        // For now, we just clear the pending updates to simulate successful processing
        self.adapters.semantic_database_adapter.pending_updates.clear();
        
        Ok(())
    }
    
    /// Trigger incremental compilation after restructuring
    fn trigger_incremental_compilation(&self, execution_result: &ExecutionResult) -> CohesionResult<()> {
        if !self.config.enable_incremental_compilation {
            return Ok(());
        }
        
        // In a real implementation, this would:
        // 1. Identify all files that need recompilation
        // 2. Trigger incremental compilation for affected modules
        // 3. Update dependency graphs
        // 4. Regenerate any cached analysis results
        
        // For now, we just simulate successful incremental compilation
        if !execution_result.modified_files.is_empty() {
            // Simulate incremental compilation trigger
            return Ok(());
        }
        
        Ok(())
    }
    
    /// Get current integration status
    pub fn get_status(&self) -> &IntegrationStatus {
        &self.status
    }
    
    /// Check if integration is ready
    pub fn is_ready(&self) -> bool {
        matches!(self.status, IntegrationStatus::Integrated)
    }
    
    /// Reset integration state
    pub fn reset(&mut self) {
        self.status = IntegrationStatus::NotIntegrated;
        self.adapters.module_registry_adapter.pending_updates.clear();
        self.adapters.query_engine_adapter.pending_invalidations.clear();
        self.adapters.semantic_database_adapter.pending_updates.clear();
    }
}

/// Helper function to create a default compiler integration
pub fn create_compiler_integration() -> CompilerIntegration {
    CompilerIntegration::new(IntegrationConfig::default())
}

/// Helper function to create a minimal compiler integration (for testing)
pub fn create_minimal_compiler_integration() -> CompilerIntegration {
    let config = IntegrationConfig {
        enable_module_registry_updates: true,
        enable_query_engine_integration: false,
        enable_semantic_database_updates: false,
        enable_incremental_compilation: false,
        integration_timeout_seconds: 10,
    };
    
    CompilerIntegration::new(config)
} 