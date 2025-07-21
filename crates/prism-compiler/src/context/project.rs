//! Project Configuration - Compilation Settings and Target Management
//!
//! This module implements project configuration management, handling compilation
//! settings, target platforms, and project metadata for the Prism compiler.
//!
//! **Conceptual Responsibility**: Project configuration and settings management
//! **What it does**: Manage project settings, target configurations, build options
//! **What it doesn't do**: Execute compilation, collect diagnostics, profile performance

use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Project configuration for compilation
#[derive(Debug, Clone)]
pub struct ProjectConfig {
    /// Project name
    pub name: String,
    /// Project version
    pub version: String,
    /// Root directory path
    pub root_dir: PathBuf,
    /// Source directories to compile
    pub source_dirs: Vec<PathBuf>,
    /// Output directory for generated artifacts
    pub output_dir: PathBuf,
    /// Compilation configuration
    pub compilation_config: CompilationConfig,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

/// Compilation configuration settings
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Project root directory
    pub project_root: PathBuf,
    /// Target platforms to compile for
    pub targets: Vec<crate::context::CompilationTarget>,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable language server integration
    pub enable_language_server: Option<bool>,
    /// Export AI context for external tools
    pub export_ai_context: bool,
    /// Enable incremental compilation
    pub incremental: Option<bool>,
    /// Enable AI features and metadata collection
    pub ai_features: Option<bool>,
    /// Include debug information in output
    pub debug_info: Option<bool>,
    /// Enable AST transformations
    pub enable_transformations: Option<bool>,
    /// Transformation configuration
    pub transformation_config: Option<TransformationConfig>,
    /// Additional compiler flags
    pub compiler_flags: HashMap<String, String>,
    /// Build profile settings
    pub build_profile: BuildProfile,
    /// Dependency management settings
    pub dependency_config: DependencyConfig,
}

/// AST transformation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationConfig {
    /// Enable optimization transformations
    pub enable_optimizations: bool,
    /// Enable semantic transformations
    pub enable_semantic_transforms: bool,
    /// Enable code generation transformations
    pub enable_codegen_transforms: bool,
    /// Custom transformation passes
    pub custom_passes: Vec<String>,
    /// Transformation settings
    pub settings: HashMap<String, serde_json::Value>,
}

/// Build profile settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildProfile {
    /// Development profile - fast compilation, debugging enabled
    Development,
    /// Release profile - optimized compilation, minimal debug info
    Release,
    /// Test profile - optimized for testing with coverage
    Test,
    /// Custom profile with specific settings
    Custom {
        /// Profile name
        name: String,
        /// Custom settings
        settings: HashMap<String, serde_json::Value>,
    },
}

/// Dependency management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyConfig {
    /// Dependency resolution strategy
    pub resolution_strategy: DependencyResolution,
    /// Package registry URLs
    pub registries: Vec<String>,
    /// Local dependency paths
    pub local_dependencies: HashMap<String, PathBuf>,
    /// Version constraints
    pub version_constraints: HashMap<String, String>,
    /// Enable dependency caching
    pub enable_caching: bool,
}

/// Dependency resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyResolution {
    /// Resolve to latest compatible versions
    Latest,
    /// Resolve to exact specified versions
    Exact,
    /// Resolve using semantic versioning rules
    Semantic,
    /// Custom resolution strategy
    Custom(String),
}

impl ProjectConfig {
    /// Create a default project configuration
    pub fn default() -> Self {
        Self {
            name: "prism-project".to_string(),
            version: "0.1.0".to_string(),
            root_dir: PathBuf::from("."),
            source_dirs: vec![PathBuf::from("src")],
            output_dir: PathBuf::from("target"),
            compilation_config: CompilationConfig::default(),
            custom_fields: HashMap::new(),
        }
    }

    /// Create project config from compilation config
    pub fn from_compilation_config(config: &CompilationConfig) -> Self {
        Self {
            name: "prism-project".to_string(),
            version: "0.1.0".to_string(),
            root_dir: config.project_root.clone(),
            source_dirs: vec![PathBuf::from("src")],
            output_dir: PathBuf::from("target"),
            compilation_config: config.clone(),
            custom_fields: HashMap::new(),
        }
    }

    /// Create a new project configuration
    pub fn new(name: String, version: String, root_dir: PathBuf) -> Self {
        Self {
            name,
            version,
            root_dir: root_dir.clone(),
            source_dirs: vec![root_dir.join("src")],
            output_dir: root_dir.join("target"),
            compilation_config: CompilationConfig::default(),
            custom_fields: HashMap::new(),
        }
    }

    /// Add a source directory
    pub fn add_source_dir(&mut self, dir: PathBuf) {
        if !self.source_dirs.contains(&dir) {
            self.source_dirs.push(dir);
        }
    }

    /// Set output directory
    pub fn set_output_dir(&mut self, dir: PathBuf) {
        self.output_dir = dir;
    }

    /// Add custom field
    pub fn add_custom_field(&mut self, key: String, value: String) {
        self.custom_fields.insert(key, value);
    }

    /// Get custom field value
    pub fn get_custom_field(&self, key: &str) -> Option<&String> {
        self.custom_fields.get(key)
    }

    /// Validate project configuration
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate project name
        if self.name.is_empty() {
            errors.push("Project name cannot be empty".to_string());
        }

        // Validate version format
        if self.version.is_empty() {
            errors.push("Project version cannot be empty".to_string());
        }

        // Validate root directory exists
        if !self.root_dir.exists() {
            errors.push(format!("Root directory does not exist: {}", self.root_dir.display()));
        }

        // Validate source directories
        for src_dir in &self.source_dirs {
            if !src_dir.exists() {
                errors.push(format!("Source directory does not exist: {}", src_dir.display()));
            }
        }

        // Validate compilation config
        if let Err(config_errors) = self.compilation_config.validate() {
            errors.extend(config_errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get project summary
    pub fn summary(&self) -> String {
        format!("{} v{} ({})", 
                self.name, 
                self.version, 
                self.root_dir.display())
    }

    /// Check if development mode is enabled
    pub fn is_development_mode(&self) -> bool {
        matches!(self.compilation_config.build_profile, BuildProfile::Development)
    }

    /// Check if release mode is enabled
    pub fn is_release_mode(&self) -> bool {
        matches!(self.compilation_config.build_profile, BuildProfile::Release)
    }
}

impl CompilationConfig {
    /// Create default compilation configuration
    pub fn default() -> Self {
        Self {
            project_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            targets: vec![crate::context::CompilationTarget::TypeScript],
            optimization_level: 2,
            enable_language_server: Some(false),
            export_ai_context: false,
            incremental: Some(true),
            ai_features: Some(true),
            debug_info: Some(true),
            enable_transformations: Some(true),
            transformation_config: None,
            compiler_flags: HashMap::new(),
            build_profile: BuildProfile::Development,
            dependency_config: DependencyConfig::default(),
        }
    }

    /// Create a development configuration
    pub fn development() -> Self {
        Self {
            optimization_level: 0,
            debug_info: Some(true),
            build_profile: BuildProfile::Development,
            ..Self::default()
        }
    }

    /// Create a release configuration
    pub fn release() -> Self {
        Self {
            optimization_level: 3,
            debug_info: Some(false),
            build_profile: BuildProfile::Release,
            ..Self::default()
        }
    }

    /// Add a compilation target
    pub fn add_target(&mut self, target: crate::context::CompilationTarget) {
        if !self.targets.contains(&target) {
            self.targets.push(target);
        }
    }

    /// Remove a compilation target
    pub fn remove_target(&mut self, target: &crate::context::CompilationTarget) {
        self.targets.retain(|t| t != target);
    }

    /// Add a compiler flag
    pub fn add_compiler_flag(&mut self, key: String, value: String) {
        self.compiler_flags.insert(key, value);
    }

    /// Get compiler flag value
    pub fn get_compiler_flag(&self, key: &str) -> Option<&String> {
        self.compiler_flags.get(key)
    }

    /// Enable feature
    pub fn enable_feature(&mut self, feature: CompilerFeature) {
        match feature {
            CompilerFeature::LanguageServer => self.enable_language_server = Some(true),
            CompilerFeature::AIContext => self.export_ai_context = true,
            CompilerFeature::Incremental => self.incremental = Some(true),
            CompilerFeature::AIFeatures => self.ai_features = Some(true),
            CompilerFeature::DebugInfo => self.debug_info = Some(true),
            CompilerFeature::Transformations => self.enable_transformations = Some(true),
        }
    }

    /// Disable feature
    pub fn disable_feature(&mut self, feature: CompilerFeature) {
        match feature {
            CompilerFeature::LanguageServer => self.enable_language_server = Some(false),
            CompilerFeature::AIContext => self.export_ai_context = false,
            CompilerFeature::Incremental => self.incremental = Some(false),
            CompilerFeature::AIFeatures => self.ai_features = Some(false),
            CompilerFeature::DebugInfo => self.debug_info = Some(false),
            CompilerFeature::Transformations => self.enable_transformations = Some(false),
        }
    }

    /// Validate compilation configuration
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate project root
        if !self.project_root.exists() {
            errors.push(format!("Project root does not exist: {}", self.project_root.display()));
        }

        // Validate optimization level
        if self.optimization_level > 3 {
            errors.push("Optimization level must be between 0 and 3".to_string());
        }

        // Validate targets
        if self.targets.is_empty() {
            errors.push("At least one compilation target must be specified".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get effective settings based on build profile
    pub fn effective_settings(&self) -> EffectiveSettings {
        match &self.build_profile {
            BuildProfile::Development => EffectiveSettings {
                optimization_level: 0,
                debug_info: true,
                ai_features: self.ai_features.unwrap_or(true),
                incremental: self.incremental.unwrap_or(true),
            },
            BuildProfile::Release => EffectiveSettings {
                optimization_level: 3,
                debug_info: false,
                ai_features: self.ai_features.unwrap_or(false),
                incremental: self.incremental.unwrap_or(false),
            },
            BuildProfile::Test => EffectiveSettings {
                optimization_level: 1,
                debug_info: true,
                ai_features: self.ai_features.unwrap_or(true),
                incremental: self.incremental.unwrap_or(true),
            },
            BuildProfile::Custom { settings, .. } => {
                // Extract settings from custom profile
                EffectiveSettings {
                    optimization_level: settings.get("optimization_level")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u8)
                        .unwrap_or(self.optimization_level),
                    debug_info: settings.get("debug_info")
                        .and_then(|v| v.as_bool())
                        .or(self.debug_info)
                        .unwrap_or(true),
                    ai_features: settings.get("ai_features")
                        .and_then(|v| v.as_bool())
                        .or(self.ai_features)
                        .unwrap_or(true),
                    incremental: settings.get("incremental")
                        .and_then(|v| v.as_bool())
                        .or(self.incremental)
                        .unwrap_or(true),
                }
            }
        }
    }
}

impl TransformationConfig {
    /// Create default transformation configuration
    pub fn default() -> Self {
        Self {
            enable_optimizations: true,
            enable_semantic_transforms: true,
            enable_codegen_transforms: true,
            custom_passes: Vec::new(),
            settings: HashMap::new(),
        }
    }

    /// Create minimal transformation configuration
    pub fn minimal() -> Self {
        Self {
            enable_optimizations: false,
            enable_semantic_transforms: true,
            enable_codegen_transforms: true,
            custom_passes: Vec::new(),
            settings: HashMap::new(),
        }
    }
}

impl DependencyConfig {
    /// Create default dependency configuration
    pub fn default() -> Self {
        Self {
            resolution_strategy: DependencyResolution::Semantic,
            registries: vec!["https://registry.prsm-lang.org".to_string()],
            local_dependencies: HashMap::new(),
            version_constraints: HashMap::new(),
            enable_caching: true,
        }
    }
}

/// Compiler features that can be toggled
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerFeature {
    /// Language server integration
    LanguageServer,
    /// AI context export
    AIContext,
    /// Incremental compilation
    Incremental,
    /// AI features and metadata
    AIFeatures,
    /// Debug information
    DebugInfo,
    /// AST transformations
    Transformations,
}

/// Effective compilation settings after profile resolution
#[derive(Debug, Clone)]
pub struct EffectiveSettings {
    /// Effective optimization level
    pub optimization_level: u8,
    /// Effective debug info setting
    pub debug_info: bool,
    /// Effective AI features setting
    pub ai_features: bool,
    /// Effective incremental compilation setting
    pub incremental: bool,
}

impl Default for BuildProfile {
    fn default() -> Self {
        BuildProfile::Development
    }
}

impl std::fmt::Display for BuildProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildProfile::Development => write!(f, "development"),
            BuildProfile::Release => write!(f, "release"),
            BuildProfile::Test => write!(f, "test"),
            BuildProfile::Custom { name, .. } => write!(f, "custom({})", name),
        }
    }
}

impl std::fmt::Display for DependencyResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DependencyResolution::Latest => write!(f, "latest"),
            DependencyResolution::Exact => write!(f, "exact"),
            DependencyResolution::Semantic => write!(f, "semantic"),
            DependencyResolution::Custom(strategy) => write!(f, "custom({})", strategy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_config_creation() {
        let config = ProjectConfig::default();
        
        assert_eq!(config.name, "prism-project");
        assert_eq!(config.version, "0.1.0");
        assert!(!config.source_dirs.is_empty());
    }

    #[test]
    fn test_compilation_config_profiles() {
        let dev_config = CompilationConfig::development();
        assert_eq!(dev_config.optimization_level, 0);
        assert_eq!(dev_config.debug_info, Some(true));
        
        let release_config = CompilationConfig::release();
        assert_eq!(release_config.optimization_level, 3);
        assert_eq!(release_config.debug_info, Some(false));
    }

    #[test]
    fn test_target_management() {
        let mut config = CompilationConfig::default();
        
        config.add_target(crate::context::CompilationTarget::WebAssembly);
        assert!(config.targets.contains(&crate::context::CompilationTarget::WebAssembly));
        
        config.remove_target(&crate::context::CompilationTarget::TypeScript);
        assert!(!config.targets.contains(&crate::context::CompilationTarget::TypeScript));
    }

    #[test]
    fn test_feature_toggling() {
        let mut config = CompilationConfig::default();
        
        config.disable_feature(CompilerFeature::AIFeatures);
        assert_eq!(config.ai_features, Some(false));
        
        config.enable_feature(CompilerFeature::LanguageServer);
        assert_eq!(config.enable_language_server, Some(true));
    }

    #[test]
    fn test_effective_settings() {
        let dev_config = CompilationConfig::development();
        let settings = dev_config.effective_settings();
        
        assert_eq!(settings.optimization_level, 0);
        assert!(settings.debug_info);
        assert!(settings.ai_features);
    }

    #[test]
    fn test_project_validation() {
        let mut config = ProjectConfig::default();
        config.name = "".to_string(); // Invalid empty name
        
        let validation_result = config.validate();
        assert!(validation_result.is_err());
        
        let errors = validation_result.unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| e.contains("Project name cannot be empty")));
    }

    #[test]
    fn test_custom_fields() {
        let mut config = ProjectConfig::default();
        
        config.add_custom_field("author".to_string(), "John Doe".to_string());
        assert_eq!(config.get_custom_field("author"), Some(&"John Doe".to_string()));
        
        assert_eq!(config.get_custom_field("nonexistent"), None);
    }

    #[test]
    fn test_source_directory_management() {
        let mut config = ProjectConfig::default();
        let new_dir = PathBuf::from("tests");
        
        config.add_source_dir(new_dir.clone());
        assert!(config.source_dirs.contains(&new_dir));
        
        // Adding same directory again should not duplicate
        config.add_source_dir(new_dir.clone());
        assert_eq!(config.source_dirs.iter().filter(|&d| d == &new_dir).count(), 1);
    }
} 