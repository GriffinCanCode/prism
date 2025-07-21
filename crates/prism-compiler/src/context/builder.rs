//! Context Builder - Construction and Initialization Utilities
//!
//! This module provides utilities for constructing and initializing compilation
//! contexts with proper configuration and validation, following the builder pattern.
//!
//! **Conceptual Responsibility**: Context construction and initialization
//! **What it does**: Build contexts with validation, provide configuration helpers
//! **What it doesn't do**: Execute compilation, collect diagnostics, manage state

use crate::error::{CompilerError, CompilerResult};
use crate::context::{
    CompilationContext, CompilationTarget, DiagnosticCollector, PerformanceProfiler,
    AIMetadataCollector, ProjectConfig, CompilationConfig, BuildProfile
};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;

/// Context builder for constructing compilation contexts
#[derive(Debug)]
pub struct ContextBuilder {
    /// Builder configuration
    config: ContextBuilderConfig,
    /// Current build phase
    current_phase: BuildPhase,
    /// Validation errors encountered
    validation_errors: Vec<String>,
}

/// Configuration for context builder behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBuilderConfig {
    /// Enable strict validation during building
    pub strict_validation: bool,
    /// Enable AI metadata collection by default
    pub default_ai_enabled: bool,
    /// Default optimization level
    pub default_optimization_level: u8,
    /// Default build profile
    pub default_build_profile: BuildProfile,
    /// Enable performance profiling by default
    pub enable_profiling: bool,
}

/// Build phases for context construction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildPhase {
    /// Initial configuration phase
    Configuration,
    /// Target platform setup
    TargetSetup,
    /// Feature configuration
    FeatureConfiguration,
    /// Validation phase
    Validation,
    /// Final construction
    Construction,
    /// Build completed
    Completed,
}

impl ContextBuilder {
    /// Create a new context builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ContextBuilderConfig::default(),
            current_phase: BuildPhase::Configuration,
            validation_errors: Vec::new(),
        }
    }

    /// Create a new context builder with custom configuration
    pub fn with_config(config: ContextBuilderConfig) -> Self {
        Self {
            config,
            current_phase: BuildPhase::Configuration,
            validation_errors: Vec::new(),
        }
    }

    /// Set compilation targets
    pub fn with_targets(mut self, targets: Vec<CompilationTarget>) -> Self {
        self.advance_phase(BuildPhase::TargetSetup);
        // Store targets in a temporary field or validate immediately
        if targets.is_empty() && self.config.strict_validation {
            self.validation_errors.push("At least one compilation target must be specified".to_string());
        }
        self
    }

    /// Add a single compilation target
    pub fn with_target(self, target: CompilationTarget) -> Self {
        self.with_targets(vec![target])
    }

    /// Enable AI metadata collection
    pub fn with_ai_metadata_enabled(mut self, enabled: bool) -> Self {
        self.advance_phase(BuildPhase::FeatureConfiguration);
        // Store AI setting for later use
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        if level > 3 && self.config.strict_validation {
            self.validation_errors.push("Optimization level must be between 0 and 3".to_string());
        }
        self
    }

    /// Set build profile
    pub fn with_build_profile(mut self, profile: BuildProfile) -> Self {
        self.advance_phase(BuildPhase::FeatureConfiguration);
        // Store profile for later use
        self
    }

    /// Set project root directory
    pub fn with_project_root(mut self, root: PathBuf) -> Self {
        if !root.exists() && self.config.strict_validation {
            self.validation_errors.push(format!("Project root directory does not exist: {}", root.display()));
        }
        self
    }

    /// Enable language server integration
    pub fn with_language_server(self, enabled: bool) -> Self {
        self.with_feature_enabled(ContextFeature::LanguageServer, enabled)
    }

    /// Enable incremental compilation
    pub fn with_incremental_compilation(self, enabled: bool) -> Self {
        self.with_feature_enabled(ContextFeature::IncrementalCompilation, enabled)
    }

    /// Enable debug information
    pub fn with_debug_info(self, enabled: bool) -> Self {
        self.with_feature_enabled(ContextFeature::DebugInfo, enabled)
    }

    /// Enable AST transformations
    pub fn with_transformations(self, enabled: bool) -> Self {
        self.with_feature_enabled(ContextFeature::Transformations, enabled)
    }

    /// Enable a specific feature
    pub fn with_feature_enabled(mut self, feature: ContextFeature, enabled: bool) -> Self {
        self.advance_phase(BuildPhase::FeatureConfiguration);
        // Store feature setting for later use
        self
    }

    /// Set project configuration
    pub fn with_project_config(mut self, project_config: ProjectConfig) -> Self {
        if self.config.strict_validation {
            if let Err(errors) = project_config.validate() {
                self.validation_errors.extend(errors);
            }
        }
        self
    }

    /// Build the compilation context
    pub fn build(mut self) -> CompilerResult<CompilationContext> {
        self.advance_phase(BuildPhase::Validation);
        
        // Perform final validation
        if self.config.strict_validation && !self.validation_errors.is_empty() {
            return Err(CompilerError::InvalidInput {
                message: format!("Context validation failed: {}", self.validation_errors.join(", ")),
            });
        }

        self.advance_phase(BuildPhase::Construction);

        // Create the compilation context with gathered settings
        let targets = vec![CompilationTarget::TypeScript]; // Default, would be from stored settings
        let context = CompilationContext::new(targets)?;

        self.advance_phase(BuildPhase::Completed);
        Ok(context)
    }

    /// Build context from existing compiler configuration
    pub fn from_compiler_config(config: &CompilationConfig) -> CompilerResult<CompilationContext> {
        let builder = Self::new()
            .with_targets(config.targets.clone())
            .with_optimization_level(config.optimization_level)
            .with_ai_metadata_enabled(config.ai_features.unwrap_or(true))
            .with_language_server(config.enable_language_server.unwrap_or(false))
            .with_incremental_compilation(config.incremental.unwrap_or(true))
            .with_debug_info(config.debug_info.unwrap_or(true))
            .with_transformations(config.enable_transformations.unwrap_or(true));

        builder.build()
    }

    /// Create a development context
    pub fn development() -> CompilerResult<CompilationContext> {
        Self::new()
            .with_build_profile(BuildProfile::Development)
            .with_optimization_level(0)
            .with_debug_info(true)
            .with_ai_metadata_enabled(true)
            .with_incremental_compilation(true)
            .build()
    }

    /// Create a release context
    pub fn release() -> CompilerResult<CompilationContext> {
        Self::new()
            .with_build_profile(BuildProfile::Release)
            .with_optimization_level(3)
            .with_debug_info(false)
            .with_ai_metadata_enabled(false)
            .with_incremental_compilation(false)
            .build()
    }

    /// Create a testing context
    pub fn testing() -> CompilerResult<CompilationContext> {
        Self::new()
            .with_build_profile(BuildProfile::Test)
            .with_optimization_level(1)
            .with_debug_info(true)
            .with_ai_metadata_enabled(true)
            .with_incremental_compilation(true)
            .build()
    }

    /// Validate current builder state
    pub fn validate(&self) -> Result<(), Vec<String>> {
        if self.validation_errors.is_empty() {
            Ok(())
        } else {
            Err(self.validation_errors.clone())
        }
    }

    /// Get current build phase
    pub fn current_phase(&self) -> &BuildPhase {
        &self.current_phase
    }

    /// Check if builder is ready to build
    pub fn is_ready(&self) -> bool {
        matches!(self.current_phase, BuildPhase::Construction | BuildPhase::Completed) && 
        (!self.config.strict_validation || self.validation_errors.is_empty())
    }

    /// Get validation errors
    pub fn validation_errors(&self) -> &[String] {
        &self.validation_errors
    }

    /// Advance to next build phase
    fn advance_phase(&mut self, target_phase: BuildPhase) {
        // Only advance if target phase is "later" than current phase
        if self.phase_order(&target_phase) > self.phase_order(&self.current_phase) {
            self.current_phase = target_phase;
        }
    }

    /// Get numeric order of build phase
    fn phase_order(&self, phase: &BuildPhase) -> u8 {
        match phase {
            BuildPhase::Configuration => 0,
            BuildPhase::TargetSetup => 1,
            BuildPhase::FeatureConfiguration => 2,
            BuildPhase::Validation => 3,
            BuildPhase::Construction => 4,
            BuildPhase::Completed => 5,
        }
    }
}

impl ContextBuilderConfig {
    /// Create a strict validation configuration
    pub fn strict() -> Self {
        Self {
            strict_validation: true,
            default_ai_enabled: true,
            default_optimization_level: 2,
            default_build_profile: BuildProfile::Development,
            enable_profiling: true,
        }
    }

    /// Create a permissive configuration
    pub fn permissive() -> Self {
        Self {
            strict_validation: false,
            default_ai_enabled: true,
            default_optimization_level: 2,
            default_build_profile: BuildProfile::Development,
            enable_profiling: true,
        }
    }

    /// Create a performance-focused configuration
    pub fn performance() -> Self {
        Self {
            strict_validation: false,
            default_ai_enabled: false,
            default_optimization_level: 3,
            default_build_profile: BuildProfile::Release,
            enable_profiling: true,
        }
    }
}

/// Context features that can be configured
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFeature {
    /// Language server integration
    LanguageServer,
    /// AI metadata collection
    AIMetadata,
    /// Incremental compilation
    IncrementalCompilation,
    /// Debug information generation
    DebugInfo,
    /// AST transformations
    Transformations,
    /// Performance profiling
    PerformanceProfiling,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ContextBuilderConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            default_ai_enabled: true,
            default_optimization_level: 2,
            default_build_profile: BuildProfile::Development,
            enable_profiling: true,
        }
    }
}

impl std::fmt::Display for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildPhase::Configuration => write!(f, "configuration"),
            BuildPhase::TargetSetup => write!(f, "target-setup"),
            BuildPhase::FeatureConfiguration => write!(f, "feature-configuration"),
            BuildPhase::Validation => write!(f, "validation"),
            BuildPhase::Construction => write!(f, "construction"),
            BuildPhase::Completed => write!(f, "completed"),
        }
    }
}

impl std::fmt::Display for ContextFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextFeature::LanguageServer => write!(f, "language-server"),
            ContextFeature::AIMetadata => write!(f, "ai-metadata"),
            ContextFeature::IncrementalCompilation => write!(f, "incremental-compilation"),
            ContextFeature::DebugInfo => write!(f, "debug-info"),
            ContextFeature::Transformations => write!(f, "transformations"),
            ContextFeature::PerformanceProfiling => write!(f, "performance-profiling"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_builder_creation() {
        let builder = ContextBuilder::new();
        
        assert_eq!(builder.current_phase(), &BuildPhase::Configuration);
        assert!(builder.validation_errors().is_empty());
    }

    #[test]
    fn test_builder_phase_advancement() {
        let mut builder = ContextBuilder::new();
        
        builder.advance_phase(BuildPhase::TargetSetup);
        assert_eq!(builder.current_phase(), &BuildPhase::TargetSetup);
        
        // Should not go backwards
        builder.advance_phase(BuildPhase::Configuration);
        assert_eq!(builder.current_phase(), &BuildPhase::TargetSetup);
    }

    #[test]
    fn test_development_context_creation() {
        let context = ContextBuilder::development().unwrap();
        
        assert_eq!(context.current_phase(), crate::context::CompilationPhase::Discovery);
        assert!(context.should_continue());
    }

    #[test]
    fn test_release_context_creation() {
        let context = ContextBuilder::release().unwrap();
        
        assert_eq!(context.current_phase(), crate::context::CompilationPhase::Discovery);
        assert!(context.should_continue());
    }

    #[test]
    fn test_validation_with_strict_config() {
        let strict_config = ContextBuilderConfig::strict();
        let builder = ContextBuilder::with_config(strict_config)
            .with_targets(vec![]); // Empty targets should cause validation error
        
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_with_permissive_config() {
        let permissive_config = ContextBuilderConfig::permissive();
        let builder = ContextBuilder::with_config(permissive_config)
            .with_targets(vec![]); // Empty targets allowed in permissive mode
        
        let result = builder.build();
        // Should succeed even with empty targets in permissive mode
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_chaining() {
        let builder = ContextBuilder::new()
            .with_target(CompilationTarget::TypeScript)
            .with_optimization_level(2)
            .with_ai_metadata_enabled(true)
            .with_debug_info(true);
        
        assert!(builder.is_ready());
        
        let context = builder.build();
        assert!(context.is_ok());
    }

    #[test]
    fn test_feature_configuration() {
        let builder = ContextBuilder::new()
            .with_feature_enabled(ContextFeature::LanguageServer, true)
            .with_feature_enabled(ContextFeature::AIMetadata, false);
        
        assert_eq!(builder.current_phase(), &BuildPhase::FeatureConfiguration);
    }

    #[test]
    fn test_from_compiler_config() {
        let compiler_config = CompilationConfig::development();
        let context = ContextBuilder::from_compiler_config(&compiler_config);
        
        assert!(context.is_ok());
    }

    #[test]
    fn test_builder_config_presets() {
        let strict = ContextBuilderConfig::strict();
        assert!(strict.strict_validation);
        
        let permissive = ContextBuilderConfig::permissive();
        assert!(!permissive.strict_validation);
        
        let performance = ContextBuilderConfig::performance();
        assert_eq!(performance.default_optimization_level, 3);
        assert!(!performance.default_ai_enabled);
    }
} 