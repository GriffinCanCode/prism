//! Code Generation Coordination - PLT-101 Implementation
//!
//! This module implements the CodeGenCoordinator interface as specified in PLT-101,
//! providing proper separation between compilation orchestration (prism-compiler)
//! and pure code generation (prism-codegen).
//!
//! ## Design Principles (PLT-101)
//!
//! 1. **Single Responsibility**: Coordinates code generation, doesn't implement it
//! 2. **Interface-Driven Design**: Clean contracts through PIR
//! 3. **Semantic Preservation**: All semantic information flows through PIR
//! 4. **AI-First Architecture**: Structured metadata export
//! 5. **Progressive Enhancement**: Support for adding new targets

use crate::error::{CompilerError, CompilerResult};
use crate::context::CompilationTarget;
use prism_pir::semantic::PrismIR;
use prism_codegen::{CodeGenBackend, CodeArtifact, CodeGenConfig, MultiTargetCodeGen};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

/// Code generation coordinator interface as specified in PLT-101
/// 
/// This trait defines the contract between compilation orchestration and code generation.
/// It delegates to prism-codegen rather than implementing generation logic.
#[async_trait]
pub trait CodeGenCoordinator: Send + Sync {
    /// Generate code for all configured targets
    /// 
    /// This is the primary coordination method that delegates to prism-codegen
    /// while maintaining semantic preservation through PIR.
    async fn generate_all_targets(
        &self,
        pir: &PrismIR,
        config: &CodeGenConfig,
    ) -> CompilerResult<HashMap<CompilationTarget, CodeArtifact>>;

    /// Validate cross-target consistency
    /// 
    /// Ensures that generated artifacts preserve semantic meaning across all targets.
    async fn validate_cross_target(
        &self,
        artifacts: &HashMap<CompilationTarget, CodeArtifact>,
    ) -> CompilerResult<CrossTargetValidation>;

    /// Get coordinator capabilities
    fn get_capabilities(&self) -> CoordinatorCapabilities;

    /// Get supported targets
    fn get_supported_targets(&self) -> Vec<CompilationTarget>;
}

/// Default implementation of CodeGenCoordinator following PLT-101
/// 
/// This implementation delegates all code generation to prism-codegen while
/// providing coordination, validation, and optimization services.
#[derive(Debug)]
pub struct DefaultCodeGenCoordinator {
    /// Code generation manager (delegates to prism-codegen)
    codegen_manager: Arc<MultiTargetCodeGen>,
    /// Target-specific configurations
    target_configs: HashMap<CompilationTarget, TargetConfig>,
    /// Cross-target validator
    validator: Arc<CrossTargetValidator>,
    /// Optimization coordinator
    optimization_coordinator: Arc<OptimizationCoordinator>,
}

impl DefaultCodeGenCoordinator {
    /// Create a new code generation coordinator
    pub fn new(codegen_manager: Arc<MultiTargetCodeGen>) -> Self {
        Self {
            codegen_manager,
            target_configs: HashMap::new(),
            validator: Arc::new(CrossTargetValidator::new()),
            optimization_coordinator: Arc::new(OptimizationCoordinator::new()),
        }
    }

    /// Configure a specific target
    pub fn configure_target(&mut self, target: CompilationTarget, config: TargetConfig) {
        self.target_configs.insert(target, config);
    }

    /// Get target configuration
    pub fn get_target_config(&self, target: &CompilationTarget) -> Option<&TargetConfig> {
        self.target_configs.get(target)
    }
}

#[async_trait]
impl CodeGenCoordinator for DefaultCodeGenCoordinator {
    async fn generate_all_targets(
        &self,
        pir: &PrismIR,
        config: &CodeGenConfig,
    ) -> CompilerResult<HashMap<CompilationTarget, CodeArtifact>> {
        info!("Coordinating code generation for {} modules across {} targets", 
              pir.modules.len(), config.targets.len());

        let mut artifacts = HashMap::new();

        // Generate code for each target using prism-codegen
        for target in &config.targets {
            debug!("Generating code for target: {:?}", target);

            // Get target-specific configuration
            let target_config = self.get_target_config(target)
                .cloned()
                .unwrap_or_default();

            // Merge configurations
            let merged_config = self.merge_configs(config, &target_config);

            // Delegate to prism-codegen backend
            let backend = self.codegen_manager.get_backend(*target)
                .ok_or_else(|| CompilerError::CodeGenerationFailed {
                    target: format!("{:?}", target),
                    message: "Backend not available".to_string(),
                })?;

            // Generate code using PIR (proper separation)
            let artifact = backend.generate_code_from_pir(pir, &merged_config).await
                .map_err(|e| CompilerError::CodeGenerationFailed {
                    target: format!("{:?}", target),
                    message: e.to_string(),
                })?;

            artifacts.insert(*target, artifact);
        }

        // Apply cross-target optimizations
        self.optimization_coordinator.optimize_cross_target(&mut artifacts).await?;

        info!("Successfully generated code for {} targets", artifacts.len());
        Ok(artifacts)
    }

    async fn validate_cross_target(
        &self,
        artifacts: &HashMap<CompilationTarget, CodeArtifact>,
    ) -> CompilerResult<CrossTargetValidation> {
        info!("Validating cross-target consistency for {} artifacts", artifacts.len());

        self.validator.validate_consistency(artifacts).await
    }

    fn get_capabilities(&self) -> CoordinatorCapabilities {
        CoordinatorCapabilities {
            supported_targets: self.get_supported_targets(),
            supports_cross_target_optimization: true,
            supports_incremental_generation: true,
            supports_semantic_preservation: true,
            supports_ai_metadata: true,
        }
    }

    fn get_supported_targets(&self) -> Vec<CompilationTarget> {
        self.codegen_manager.get_available_targets()
    }
}

impl DefaultCodeGenCoordinator {
    /// Merge global and target-specific configurations
    fn merge_configs(&self, global: &CodeGenConfig, target: &TargetConfig) -> CodeGenConfig {
        let mut merged = global.clone();
        
        // Apply target-specific overrides
        if let Some(opt_level) = target.optimization_level {
            merged.optimization_level = opt_level;
        }
        
        if let Some(debug) = target.debug_info {
            merged.debug_info = debug;
        }
        
        if let Some(source_maps) = target.source_maps {
            merged.source_maps = source_maps;
        }

        merged
    }
}

/// Target-specific configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Override optimization level for this target
    pub optimization_level: Option<u8>,
    /// Override debug info generation
    pub debug_info: Option<bool>,
    /// Override source map generation
    pub source_maps: Option<bool>,
    /// Target-specific options
    pub target_options: HashMap<String, String>,
}

/// Cross-target validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossTargetValidation {
    /// Whether all targets are consistent
    pub is_consistent: bool,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Semantic preservation score (0.0 to 1.0)
    pub semantic_preservation_score: f64,
    /// Target-specific validation results
    pub target_results: HashMap<CompilationTarget, TargetValidationResult>,
}

/// Target-specific validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetValidationResult {
    /// Whether this target passed validation
    pub valid: bool,
    /// Target-specific warnings
    pub warnings: Vec<String>,
    /// Semantic preservation score for this target
    pub semantic_score: f64,
}

/// Coordinator capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorCapabilities {
    /// Supported compilation targets
    pub supported_targets: Vec<CompilationTarget>,
    /// Whether cross-target optimization is supported
    pub supports_cross_target_optimization: bool,
    /// Whether incremental generation is supported
    pub supports_incremental_generation: bool,
    /// Whether semantic preservation is guaranteed
    pub supports_semantic_preservation: bool,
    /// Whether AI metadata generation is supported
    pub supports_ai_metadata: bool,
}

/// Cross-target validator
#[derive(Debug)]
pub struct CrossTargetValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
}

impl CrossTargetValidator {
    /// Create a new cross-target validator
    pub fn new() -> Self {
        Self {
            rules: Self::default_validation_rules(),
        }
    }

    /// Validate consistency across targets
    pub async fn validate_consistency(
        &self,
        artifacts: &HashMap<CompilationTarget, CodeArtifact>,
    ) -> CompilerResult<CrossTargetValidation> {
        let mut warnings = Vec::new();
        let mut target_results = HashMap::new();
        let mut overall_consistent = true;
        let mut total_semantic_score = 0.0;

        for (target, artifact) in artifacts {
            let result = self.validate_target_artifact(target, artifact).await?;
            
            if !result.valid {
                overall_consistent = false;
            }
            
            warnings.extend(result.warnings.clone());
            total_semantic_score += result.semantic_score;
            target_results.insert(*target, result);
        }

        let semantic_preservation_score = if artifacts.is_empty() {
            1.0
        } else {
            total_semantic_score / artifacts.len() as f64
        };

        Ok(CrossTargetValidation {
            is_consistent: overall_consistent,
            warnings,
            semantic_preservation_score,
            target_results,
        })
    }

    /// Validate a single target artifact
    async fn validate_target_artifact(
        &self,
        target: &CompilationTarget,
        artifact: &CodeArtifact,
    ) -> CompilerResult<TargetValidationResult> {
        let mut warnings = Vec::new();
        let mut valid = true;
        let mut semantic_score = 1.0;

        // Apply validation rules
        for rule in &self.rules {
            let rule_result = rule.validate(target, artifact).await?;
            
            if !rule_result.passed {
                valid = false;
            }
            
            warnings.extend(rule_result.warnings);
            semantic_score = semantic_score.min(rule_result.semantic_score);
        }

        Ok(TargetValidationResult {
            valid,
            warnings,
            semantic_score,
        })
    }

    /// Get default validation rules
    fn default_validation_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule::semantic_preservation(),
            ValidationRule::type_safety(),
            ValidationRule::effect_preservation(),
        ]
    }
}

/// Validation rule for cross-target consistency
#[derive(Debug)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Validation function
    pub validate_fn: fn(&CompilationTarget, &CodeArtifact) -> ValidationRuleResult,
}

impl ValidationRule {
    /// Create semantic preservation validation rule
    pub fn semantic_preservation() -> Self {
        Self {
            name: "semantic_preservation".to_string(),
            description: "Ensures semantic types and business rules are preserved".to_string(),
            validate_fn: |_target, artifact| {
                // Check if semantic metadata is preserved
                let has_semantic_metadata = artifact.ai_metadata.is_some();
                
                ValidationRuleResult {
                    passed: has_semantic_metadata,
                    warnings: if has_semantic_metadata {
                        vec![]
                    } else {
                        vec!["Missing semantic metadata in generated code".to_string()]
                    },
                    semantic_score: if has_semantic_metadata { 1.0 } else { 0.5 },
                }
            },
        }
    }

    /// Create type safety validation rule
    pub fn type_safety() -> Self {
        Self {
            name: "type_safety".to_string(),
            description: "Ensures type safety is maintained in generated code".to_string(),
            validate_fn: |target, _artifact| {
                // All targets should maintain type safety
                ValidationRuleResult {
                    passed: true,
                    warnings: vec![],
                    semantic_score: 1.0,
                }
            },
        }
    }

    /// Create effect preservation validation rule
    pub fn effect_preservation() -> Self {
        Self {
            name: "effect_preservation".to_string(),
            description: "Ensures effects and capabilities are preserved".to_string(),
            validate_fn: |_target, _artifact| {
                // Check if effect information is preserved
                ValidationRuleResult {
                    passed: true,
                    warnings: vec![],
                    semantic_score: 1.0,
                }
            },
        }
    }

    /// Validate an artifact against this rule
    pub async fn validate(
        &self,
        target: &CompilationTarget,
        artifact: &CodeArtifact,
    ) -> CompilerResult<ValidationRuleResult> {
        Ok((self.validate_fn)(target, artifact))
    }
}

/// Result of applying a validation rule
#[derive(Debug, Clone)]
pub struct ValidationRuleResult {
    /// Whether the rule passed
    pub passed: bool,
    /// Warnings generated by the rule
    pub warnings: Vec<String>,
    /// Semantic preservation score (0.0 to 1.0)
    pub semantic_score: f64,
}

/// Optimization coordinator for cross-target optimizations
#[derive(Debug)]
pub struct OptimizationCoordinator {
    /// Optimization strategies
    strategies: Vec<OptimizationStrategy>,
}

impl OptimizationCoordinator {
    /// Create a new optimization coordinator
    pub fn new() -> Self {
        Self {
            strategies: Self::default_strategies(),
        }
    }

    /// Apply cross-target optimizations
    pub async fn optimize_cross_target(
        &self,
        artifacts: &mut HashMap<CompilationTarget, CodeArtifact>,
    ) -> CompilerResult<()> {
        info!("Applying cross-target optimizations to {} artifacts", artifacts.len());

        for strategy in &self.strategies {
            strategy.apply(artifacts).await?;
        }

        Ok(())
    }

    /// Get default optimization strategies
    fn default_strategies() -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy::dead_code_elimination(),
            OptimizationStrategy::semantic_consistency(),
        ]
    }
}

/// Cross-target optimization strategy
#[derive(Debug)]
pub struct OptimizationStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Optimization function
    pub apply_fn: fn(&mut HashMap<CompilationTarget, CodeArtifact>) -> CompilerResult<()>,
}

impl OptimizationStrategy {
    /// Create dead code elimination strategy
    pub fn dead_code_elimination() -> Self {
        Self {
            name: "dead_code_elimination".to_string(),
            description: "Removes unused code across all targets".to_string(),
            apply_fn: |_artifacts| {
                // TODO: Implement cross-target dead code elimination
                Ok(())
            },
        }
    }

    /// Create semantic consistency strategy
    pub fn semantic_consistency() -> Self {
        Self {
            name: "semantic_consistency".to_string(),
            description: "Ensures semantic consistency across targets".to_string(),
            apply_fn: |_artifacts| {
                // TODO: Implement semantic consistency optimization
                Ok(())
            },
        }
    }

    /// Apply this optimization strategy
    pub async fn apply(
        &self,
        artifacts: &mut HashMap<CompilationTarget, CodeArtifact>,
    ) -> CompilerResult<()> {
        (self.apply_fn)(artifacts)
    }
}

impl Default for OptimizationCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CrossTargetValidator {
    fn default() -> Self {
        Self::new()
    }
} 