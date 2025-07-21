//! Prism AI Integration
//!
//! This crate provides the central coordination point for AI metadata export and integration
//! functionality across the Prism language ecosystem. It orchestrates metadata collection
//! from multiple crates and provides unified export interfaces for external AI tools.
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: AI functionality is separated from core language processing
//! 2. **Modular Integration**: Each crate can contribute metadata independently
//! 3. **Unified Export**: Single interface for all AI metadata export needs
//! 4. **External Focus**: Designed for external AI tool consumption, not internal AI execution
//! 5. **Performance Aware**: Minimal overhead when AI features are disabled

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use thiserror::Error;

pub mod metadata;
pub mod export;
pub mod integration;
pub mod context;
pub mod providers;  // NEW: Metadata provider system

// Re-export common types for convenience
pub use metadata::*;
pub use export::*;
pub use integration::*;
pub use context::*;
pub use providers::*;  // NEW: Provider traits and types

/// Central AI integration coordinator
#[derive(Debug)]
pub struct AIIntegrationCoordinator {
    /// Configuration for AI features
    config: AIIntegrationConfig,
    /// Metadata collectors from various crates (legacy)
    collectors: HashMap<String, Box<dyn MetadataCollector>>,
    /// Export formatters
    exporters: HashMap<ExportFormat, Box<dyn MetadataExporter>>,
    /// Context extractors
    context_extractors: Vec<Box<dyn ContextExtractor>>,
    /// New provider registry for real metadata collection
    provider_registry: ProviderRegistry,
}

/// Configuration for AI integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIIntegrationConfig {
    /// Enable AI metadata collection
    pub enabled: bool,
    /// Export formats to generate
    pub export_formats: Vec<ExportFormat>,
    /// Include business context in exports
    pub include_business_context: bool,
    /// Include performance metrics
    pub include_performance_metrics: bool,
    /// Include architectural patterns
    pub include_architectural_patterns: bool,
    /// Minimum confidence threshold for metadata quality
    pub min_confidence_threshold: f64,
    /// Output directory for exports
    pub output_directory: Option<PathBuf>,
}

/// Export formats supported by the AI system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format for general AI consumption
    Json,
    /// YAML format for human-readable exports
    Yaml,
    /// XML format for structured data exchange
    Xml,
    /// Binary format for performance-critical scenarios
    Binary,
    /// Protocol Buffers for efficient serialization
    Protobuf,
    /// OpenAPI specification for API documentation
    OpenApi,
    /// GraphQL schema for query interfaces
    GraphQL,
    /// Custom format with specified name
    Custom(String),
}

/// Comprehensive AI metadata from all sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAIMetadata {
    /// Metadata version
    pub version: String,
    /// Export timestamp
    pub exported_at: String,
    /// Project information
    pub project_info: ProjectInfo,
    /// AST and syntax metadata
    pub syntax_metadata: Option<prism_common::ai_metadata::AIMetadata>,
    /// Semantic analysis metadata
    pub semantic_metadata: Option<SemanticAIMetadata>,
    /// Runtime metadata
    pub runtime_metadata: Option<RuntimeAIMetadata>,
    /// PIR metadata
    pub pir_metadata: Option<PIRAIMetadata>,
    /// Effects system metadata
    pub effects_metadata: Option<EffectsAIMetadata>,
    /// Business context information
    pub business_context: Option<BusinessContext>,
    /// Cross-system relationships
    pub relationships: Vec<CrossSystemRelationship>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Project information for AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    /// Project name
    pub name: String,
    /// Project version
    pub version: Option<String>,
    /// Project root path
    pub root_path: PathBuf,
    /// Source files included
    pub source_files: Vec<SourceFileInfo>,
    /// Dependencies
    pub dependencies: Vec<DependencyInfo>,
}

/// Source file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFileInfo {
    /// File path relative to project root
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub last_modified: String,
    /// Detected language/syntax style
    pub language: String,
    /// File hash for integrity checking
    pub hash: String,
}

/// Dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// Dependency name
    pub name: String,
    /// Version specification
    pub version: String,
    /// Dependency type (direct, transitive, etc.)
    pub dependency_type: DependencyType,
    /// Source of the dependency
    pub source: String,
}

/// Types of dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Direct,
    Transitive,
    Development,
    Optional,
}

/// Cross-system relationship between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSystemRelationship {
    /// Source component
    pub source: ComponentReference,
    /// Target component
    pub target: ComponentReference,
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Strength of relationship (0.0 to 1.0)
    pub strength: f64,
    /// Description of the relationship
    pub description: String,
}

/// Reference to a component across systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentReference {
    /// System that owns the component
    pub system: String,
    /// Component identifier
    pub component_id: String,
    /// Component type
    pub component_type: String,
}

/// Types of relationships between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// One component depends on another
    Dependency,
    /// One component is composed of another
    Composition,
    /// One component inherits from another
    Inheritance,
    /// One component uses another
    Usage,
    /// Components are associated
    Association,
    /// Components implement the same interface
    Implementation,
    /// Data flows between components
    DataFlow,
    /// Control flows between components
    ControlFlow,
}

/// Quality metrics for the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Lines of code
    pub lines_of_code: u64,
    /// Cyclomatic complexity
    pub cyclomatic_complexity: u32,
    /// Cognitive complexity
    pub cognitive_complexity: u32,
    /// Test coverage percentage
    pub test_coverage: f64,
    /// Documentation coverage percentage
    pub documentation_coverage: f64,
    /// Technical debt ratio
    pub technical_debt_ratio: f64,
}

/// AI integration errors
#[derive(Debug, Error)]
pub enum AIIntegrationError {
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Metadata collection failed: {source}")]
    MetadataCollectionFailed { source: String },

    #[error("Export failed: {format:?} - {reason}")]
    ExportFailed { format: ExportFormat, reason: String },

    #[error("Context extraction failed: {reason}")]
    ContextExtractionFailed { reason: String },

    #[error("Integration error: {message}")]
    IntegrationError { message: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl Default for AIIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            export_formats: vec![ExportFormat::Json],
            include_business_context: true,
            include_performance_metrics: true,
            include_architectural_patterns: true,
            min_confidence_threshold: 0.7,
            output_directory: None,
        }
    }
}

impl AIIntegrationCoordinator {
    /// Create a new AI integration coordinator
    pub fn new(config: AIIntegrationConfig) -> Self {
        Self {
            config,
            collectors: HashMap::new(),
            exporters: HashMap::new(),
            context_extractors: Vec::new(),
            provider_registry: ProviderRegistry::new(),
        }
    }

    /// Register a metadata collector (legacy system)
    pub fn register_collector(&mut self, name: String, collector: Box<dyn MetadataCollector>) {
        self.collectors.insert(name, collector);
    }
    
    /// Register a metadata provider (new system)
    pub fn register_provider(&mut self, provider: Box<dyn MetadataProvider>) {
        self.provider_registry.register_provider(provider);
    }

    /// Register an exporter for a specific format
    pub fn register_exporter(&mut self, format: ExportFormat, exporter: Box<dyn MetadataExporter>) {
        self.exporters.insert(format, exporter);
    }

    /// Register a context extractor
    pub fn register_context_extractor(&mut self, extractor: Box<dyn ContextExtractor>) {
        self.context_extractors.push(extractor);
    }

    /// Collect all metadata from registered collectors and providers
    pub async fn collect_metadata(&self, project_root: &PathBuf) -> Result<ComprehensiveAIMetadata, AIIntegrationError> {
        if !self.config.enabled {
            return Err(AIIntegrationError::ConfigurationError {
                message: "AI integration is disabled".to_string(),
            });
        }

        let project_info = self.collect_project_info(project_root).await?;
        
        // Try new provider system first
        let mut syntax_metadata = None;
        let mut semantic_metadata = None;
        let mut runtime_metadata = None;
        let mut pir_metadata = None;
        let mut effects_metadata = None;
        
        // Create provider context
        let provider_context = ProviderContext {
            project_root: project_root.clone(),
            compilation_artifacts: None, // Would be populated by compiler integration
            runtime_info: None, // Would be populated by runtime integration
            provider_config: ProviderConfig {
                detailed: true,
                include_performance: self.config.include_performance_metrics,
                include_business_context: self.config.include_business_context,
                custom_settings: std::collections::HashMap::new(),
            },
        };
        
        // Collect from providers first (preferred)
        match self.provider_registry.collect_all_metadata(&provider_context).await {
            Ok(domain_metadata) => {
                for metadata in domain_metadata {
                    match metadata {
                        DomainMetadata::Syntax(syntax_meta) => {
                            syntax_metadata = Some(syntax_meta.ai_context);
                        }
                        DomainMetadata::Semantic(_semantic_meta) => {
                            // Convert semantic provider metadata
                            semantic_metadata = Some(SemanticAIMetadata { placeholder: false });
                        }
                        DomainMetadata::Runtime(_runtime_meta) => {
                            runtime_metadata = Some(RuntimeAIMetadata { placeholder: false });
                        }
                        DomainMetadata::Pir(_pir_meta) => {
                            pir_metadata = Some(PIRAIMetadata { placeholder: false });
                        }
                        DomainMetadata::Effects(_effects_meta) => {
                            effects_metadata = Some(EffectsAIMetadata { placeholder: false });
                        }
                        _ => {
                            // Handle other domain metadata types
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Provider system failed: {}, falling back to legacy collectors", e);
            }
        }
        
        // Fall back to legacy collectors for any missing metadata
        for (name, collector) in &self.collectors {
            match collector.collect_metadata(project_root).await {
                Ok(metadata) => {
                    match name.as_str() {
                        "syntax" if syntax_metadata.is_none() => {
                            syntax_metadata = Some(metadata.downcast_syntax());
                        }
                        "semantic" if semantic_metadata.is_none() => {
                            semantic_metadata = Some(metadata.downcast_semantic());
                        }
                        "runtime" if runtime_metadata.is_none() => {
                            runtime_metadata = Some(metadata.downcast_runtime());
                        }
                        "pir" if pir_metadata.is_none() => {
                            pir_metadata = Some(metadata.downcast_pir());
                        }
                        "effects" if effects_metadata.is_none() => {
                            effects_metadata = Some(metadata.downcast_effects());
                        }
                        _ => {} // Unknown collector or already have data
                    }
                }
                Err(e) => {
                    return Err(AIIntegrationError::MetadataCollectionFailed {
                        source: format!("Collector '{}': {}", name, e),
                    });
                }
            }
        }

        // Extract business context if enabled
        let business_context = if self.config.include_business_context {
            self.extract_business_context(&syntax_metadata, &semantic_metadata).await?
        } else {
            None
        };

        // Analyze cross-system relationships
        let relationships = self.analyze_relationships(
            &syntax_metadata,
            &semantic_metadata,
            &runtime_metadata,
            &pir_metadata,
            &effects_metadata,
        ).await?;

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(
            &project_info,
            &syntax_metadata,
            &semantic_metadata,
        ).await?;

        Ok(ComprehensiveAIMetadata {
            version: "1.0.0".to_string(),
            exported_at: chrono::Utc::now().to_rfc3339(),
            project_info,
            syntax_metadata,
            semantic_metadata,
            runtime_metadata,
            pir_metadata,
            effects_metadata,
            business_context,
            relationships,
            quality_metrics,
        })
    }

    /// Export metadata in specified formats
    pub async fn export_metadata(
        &self,
        metadata: &ComprehensiveAIMetadata,
        formats: &[ExportFormat],
    ) -> Result<HashMap<ExportFormat, String>, AIIntegrationError> {
        let mut results = HashMap::new();

        for format in formats {
            if let Some(exporter) = self.exporters.get(format) {
                match exporter.export(metadata).await {
                    Ok(exported) => {
                        results.insert(format.clone(), exported);
                    }
                    Err(e) => {
                        return Err(AIIntegrationError::ExportFailed {
                            format: format.clone(),
                            reason: e.to_string(),
                        });
                    }
                }
            } else {
                return Err(AIIntegrationError::ExportFailed {
                    format: format.clone(),
                    reason: "No exporter registered for format".to_string(),
                });
            }
        }

        Ok(results)
    }

    /// Collect project information
    async fn collect_project_info(&self, project_root: &PathBuf) -> Result<ProjectInfo, AIIntegrationError> {
        // This would scan the project directory and collect information
        // For now, return a basic implementation
        Ok(ProjectInfo {
            name: project_root.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            version: None,
            root_path: project_root.clone(),
            source_files: Vec::new(), // Would be populated by scanning
            dependencies: Vec::new(), // Would be populated from Cargo.toml, etc.
        })
    }

    /// Extract business context from metadata
    async fn extract_business_context(
        &self,
        _syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        _semantic_metadata: &Option<SemanticAIMetadata>,
    ) -> Result<Option<BusinessContext>, AIIntegrationError> {
        // This would analyze the metadata to extract business context
        // For now, return None
        Ok(None)
    }

    /// Analyze relationships between components
    async fn analyze_relationships(
        &self,
        _syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        _semantic_metadata: &Option<SemanticAIMetadata>,
        _runtime_metadata: &Option<RuntimeAIMetadata>,
        _pir_metadata: &Option<PIRAIMetadata>,
        _effects_metadata: &Option<EffectsAIMetadata>,
    ) -> Result<Vec<CrossSystemRelationship>, AIIntegrationError> {
        // This would analyze cross-system relationships
        // For now, return empty vector
        Ok(Vec::new())
    }

    /// Calculate quality metrics
    async fn calculate_quality_metrics(
        &self,
        _project_info: &ProjectInfo,
        _syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        _semantic_metadata: &Option<SemanticAIMetadata>,
    ) -> Result<QualityMetrics, AIIntegrationError> {
        // This would calculate quality metrics from the metadata
        // For now, return default metrics
        Ok(QualityMetrics {
            lines_of_code: 0,
            cyclomatic_complexity: 0,
            cognitive_complexity: 0,
            test_coverage: 0.0,
            documentation_coverage: 0.0,
            technical_debt_ratio: 0.0,
        })
    }
}

// Placeholder types for metadata from different systems
// These would be properly defined in their respective modules

/// Semantic AI metadata placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAIMetadata {
    pub placeholder: bool,
}

/// Runtime AI metadata placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAIMetadata {
    pub placeholder: bool,
}

/// PIR AI metadata placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRAIMetadata {
    pub placeholder: bool,
}

/// Effects AI metadata placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsAIMetadata {
    pub placeholder: bool,
}

/// Business context placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    pub domain: Option<String>,
    pub capabilities: Vec<String>,
    pub rules: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_ai_integration_coordinator_creation() {
        let config = AIIntegrationConfig::default();
        let coordinator = AIIntegrationCoordinator::new(config);
        
        // Test that coordinator is created successfully
        assert!(coordinator.collectors.is_empty());
        assert!(coordinator.exporters.is_empty());
        assert!(coordinator.context_extractors.is_empty());
    }

    #[tokio::test]
    async fn test_metadata_collection_disabled() {
        let mut config = AIIntegrationConfig::default();
        config.enabled = false;
        
        let coordinator = AIIntegrationCoordinator::new(config);
        let project_root = PathBuf::from(".");
        
        let result = coordinator.collect_metadata(&project_root).await;
        assert!(result.is_err());
        
        if let Err(AIIntegrationError::ConfigurationError { message }) = result {
            assert_eq!(message, "AI integration is disabled");
        } else {
            panic!("Expected ConfigurationError");
        }
    }

    #[tokio::test]
    async fn test_basic_metadata_collection() {
        let config = AIIntegrationConfig::default();
        let coordinator = AIIntegrationCoordinator::new(config);
        let project_root = PathBuf::from(".");
        
        let result = coordinator.collect_metadata(&project_root).await;
        assert!(result.is_ok());
        
        let metadata = result.unwrap();
        assert_eq!(metadata.version, "1.0.0");
        assert!(!metadata.exported_at.is_empty());
        assert_eq!(metadata.project_info.name, ".");
    }

    #[tokio::test]
    async fn test_export_format_parsing() {
        use crate::export::*;
        
        let exporter = JsonExporter::new();
        assert_eq!(exporter.format(), ExportFormat::Json);
        assert_eq!(exporter.file_extension(), "json");
        assert_eq!(exporter.mime_type(), "application/json");
        
        let yaml_exporter = YamlExporter::new();
        assert_eq!(yaml_exporter.format(), ExportFormat::Yaml);
        assert_eq!(yaml_exporter.file_extension(), "yaml");
    }

    #[tokio::test]
    async fn test_context_extraction() {
        use crate::context::*;
        
        let extractor = ProjectStructureExtractor::new();
        let project_root = PathBuf::from(".");
        
        let result = extractor.extract_context(&project_root).await;
        assert!(result.is_ok());
        
        let context = result.unwrap();
        assert_eq!(context.source, "project_structure");
        assert_eq!(context.confidence, 1.0);
    }
}
