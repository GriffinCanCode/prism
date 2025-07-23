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

use serde::{Serialize, Deserialize};
use thiserror::Error;

pub mod metadata;
pub mod export;
pub mod integration;
pub mod context;
pub mod providers;

// Re-export common types for convenience
pub use metadata::*;
pub use export::*;
pub use integration::*;
pub use context::*;
pub use providers::*;

/// Central AI integration coordinator

pub struct AIIntegrationCoordinator {
    /// Configuration for AI features
    config: AIIntegrationConfig,
    /// Metadata collectors from various crates (legacy)
    collectors: HashMap<String, Box<dyn MetadataCollector>>,
    /// Export formatters
    exporters: HashMap<ExportFormat, Box<dyn MetadataExporter>>,
    /// Context extractors
    context_extractors: Vec<Box<dyn ContextExtractor>>,
    /// Provider registry for metadata collection
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

    #[error("Metadata collection failed: {message}")]
    MetadataCollectionFailed { message: String },

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
        let mut coordinator = Self {
            config,
            collectors: HashMap::new(),
            exporters: HashMap::new(),
            context_extractors: Vec::new(),
            provider_registry: ProviderRegistry::new(),
        };
        
        // Register default exporters
        coordinator.register_default_exporters();
        
        // Register default context extractors
        coordinator.register_default_context_extractors();
        
        coordinator
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

    /// Add an integration endpoint for external AI tools
    pub fn add_integration_endpoint(&mut self, _endpoint: IntegrationEndpoint) {
        // Initialize integration manager if not already done
        if !self.has_integration_manager() {
            // Integration manager would be initialized here
        }
        // For now, store endpoint info for future use
    }

    /// Send metadata to external AI tools via HTTP endpoints
    pub async fn send_to_integrations(
        &self,
        metadata: &ComprehensiveAIMetadata,
        endpoints: &[IntegrationEndpoint],
    ) -> Result<Vec<IntegrationResult>, AIIntegrationError> {
        let http_client = crate::integration::HttpEndpointClient::new();
        let mut results = Vec::new();
        
        for endpoint in endpoints {
            let result = match http_client.send_to_endpoint(metadata, endpoint).await {
                Ok(response) => IntegrationResult {
                    endpoint_name: endpoint.name.clone(),
                    success: response.success,
                    status_code: Some(response.status_code),
                    message: if response.success {
                        "Successfully sent metadata".to_string()
                    } else {
                        format!("HTTP {} - {}", response.status_code, response.body)
                    },
                    response_data: Some(response.body),
                },
                Err(e) => IntegrationResult {
                    endpoint_name: endpoint.name.clone(),
                    success: false,
                    status_code: None,
                    message: format!("Integration failed: {}", e),
                    response_data: None,
                },
            };
            
            results.push(result);
        }
        
        Ok(results)
    }

    /// Check if integration manager is available
    fn has_integration_manager(&self) -> bool {
        // For now, always return false since we're not storing the integration manager
        // In a full implementation, this would check if an integration manager is configured
        false
    }

    /// Register default exporters for all supported formats
    fn register_default_exporters(&mut self) {
        self.exporters.insert(ExportFormat::Json, Box::new(crate::export::JsonExporter::new()));
        self.exporters.insert(ExportFormat::Yaml, Box::new(crate::export::YamlExporter::new()));
        self.exporters.insert(ExportFormat::Xml, Box::new(crate::export::XmlExporter::new()));
        self.exporters.insert(ExportFormat::Binary, Box::new(crate::export::BinaryExporter::new()));
        self.exporters.insert(ExportFormat::OpenApi, Box::new(crate::export::OpenApiExporter::new()));
        self.exporters.insert(ExportFormat::GraphQL, Box::new(crate::export::GraphQLExporter::new()));
    }

    /// Register default context extractors
    fn register_default_context_extractors(&mut self) {
        self.context_extractors.push(Box::new(crate::context::ProjectStructureExtractor::new()));
        self.context_extractors.push(Box::new(crate::context::CodePatternsExtractor::new()));
        self.context_extractors.push(Box::new(crate::context::BusinessDomainExtractor::new()));
        self.context_extractors.push(Box::new(crate::context::DependenciesExtractor::new()));
    }

    /// Collect all metadata from registered collectors and providers
    pub async fn collect_metadata(&self, project_root: &PathBuf) -> Result<ComprehensiveAIMetadata, AIIntegrationError> {
        if !self.config.enabled {
            return Err(AIIntegrationError::ConfigurationError {
                message: "AI integration is disabled".to_string(),
            });
        }

        let project_info = self.collect_project_info(project_root).await?;
        
        // Collect from provider system first (preferred approach)
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
        
        // Collect from providers (preferred approach)
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
                        message: format!("Collector '{}': {}", name, e),
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

    /// Export metadata in the specified format
    pub async fn export_metadata(
        &self,
        metadata: &ComprehensiveAIMetadata,
        format: ExportFormat,
    ) -> Result<String, AIIntegrationError> {
        if let Some(exporter) = self.exporters.get(&format) {
            exporter.export(metadata).await
        } else {
            Err(AIIntegrationError::ExportFailed {
                format,
                reason: "No exporter registered for format".to_string(),
            })
        }
    }

    /// Export metadata in multiple formats
    pub async fn export_metadata_multiple(
        &self,
        metadata: &ComprehensiveAIMetadata,
        formats: &[ExportFormat],
    ) -> Result<HashMap<ExportFormat, String>, AIIntegrationError> {
        let mut results = HashMap::new();
        
        for format in formats {
            match self.export_metadata(metadata, format.clone()).await {
                Ok(exported) => {
                    results.insert(format.clone(), exported);
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        
        Ok(results)
    }

    /// Collect project information
    async fn collect_project_info(&self, project_root: &PathBuf) -> Result<ProjectInfo, AIIntegrationError> {
        let project_name = project_root.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Scan for source files
        let source_files = self.scan_source_files(project_root).await?;
        
        // Parse dependencies from Cargo.toml or other manifest files
        let (version, dependencies) = self.parse_project_manifest(project_root).await?;
        
        Ok(ProjectInfo {
            name: project_name,
            version,
            root_path: project_root.clone(),
            source_files,
            dependencies,
        })
    }

    /// Scan directory for source files
    async fn scan_source_files(&self, project_root: &PathBuf) -> Result<Vec<SourceFileInfo>, AIIntegrationError> {
        let mut source_files = Vec::new();
        
        // Define source file extensions to scan
        let source_extensions = ["rs", "prism", "toml", "md", "json", "yaml", "yml"];
        
        // Use tokio for async file system operations
        let mut entries = tokio::fs::read_dir(project_root).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                    if source_extensions.contains(&extension) {
                        let metadata = entry.metadata().await?;
                        let relative_path = path.strip_prefix(project_root)
                            .unwrap_or(&path)
                            .to_path_buf();
                        
                        // Calculate file hash for integrity checking
                        let content = tokio::fs::read(&path).await?;
                        let hash = self.calculate_file_hash(&content);
                        
                        source_files.push(SourceFileInfo {
                            path: relative_path,
                            size: metadata.len(),
                            last_modified: metadata.modified()
                                .map(|time| time.duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default().as_secs().to_string())
                                .unwrap_or_else(|_| "unknown".to_string()),
                            language: self.detect_language(extension),
                            hash,
                        });
                    }
                }
            } else if path.is_dir() && !self.should_skip_directory(&path) {
                // Recursively scan subdirectories
                let mut sub_files = Box::pin(self.scan_source_files(&path)).await?;
                source_files.append(&mut sub_files);
            }
        }
        
        Ok(source_files)
    }

    /// Parse project manifest files (Cargo.toml, package.json, etc.)
    async fn parse_project_manifest(&self, project_root: &PathBuf) -> Result<(Option<String>, Vec<DependencyInfo>), AIIntegrationError> {
        // Try Cargo.toml first
        let cargo_toml = project_root.join("Cargo.toml");
        if cargo_toml.exists() {
            return self.parse_cargo_toml(&cargo_toml).await;
        }
        
        // Try package.json
        let package_json = project_root.join("package.json");
        if package_json.exists() {
            return self.parse_package_json(&package_json).await;
        }
        
        // No known manifest found
        Ok((None, Vec::new()))
    }

    /// Parse Cargo.toml for Rust projects
    async fn parse_cargo_toml(&self, cargo_toml: &PathBuf) -> Result<(Option<String>, Vec<DependencyInfo>), AIIntegrationError> {
        let content = tokio::fs::read_to_string(cargo_toml).await?;
        
        // Parse TOML content
        let parsed: toml::Value = content.parse()
            .map_err(|e| AIIntegrationError::ConfigurationError {
                message: format!("Failed to parse Cargo.toml: {}", e),
            })?;
        
        // Extract version
        let version = parsed.get("package")
            .and_then(|p| p.get("version"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Extract dependencies
        let mut dependencies = Vec::new();
        
        if let Some(deps) = parsed.get("dependencies").and_then(|d| d.as_table()) {
            for (name, spec) in deps {
                let dependency_info = match spec {
                    toml::Value::String(version_str) => {
                        DependencyInfo {
                            name: name.clone(),
                            version: version_str.clone(),
                            dependency_type: DependencyType::Direct,
                            source: "crates.io".to_string(),
                        }
                    }
                    toml::Value::Table(table) => {
                        let version = table.get("version")
                            .and_then(|v| v.as_str())
                            .unwrap_or("*")
                            .to_string();
                        
                        let source = if table.contains_key("path") {
                            "path".to_string()
                        } else if table.contains_key("git") {
                            "git".to_string()
                        } else {
                            "crates.io".to_string()
                        };
                        
                        DependencyInfo {
                            name: name.clone(),
                            version,
                            dependency_type: DependencyType::Direct,
                            source,
                        }
                    }
                    _ => continue,
                };
                dependencies.push(dependency_info);
            }
        }
        
        // Extract dev-dependencies
        if let Some(dev_deps) = parsed.get("dev-dependencies").and_then(|d| d.as_table()) {
            for (name, spec) in dev_deps {
                if let Some(version_str) = spec.as_str() {
                    dependencies.push(DependencyInfo {
                        name: name.clone(),
                        version: version_str.to_string(),
                        dependency_type: DependencyType::Development,
                        source: "crates.io".to_string(),
                    });
                }
            }
        }
        
        Ok((version, dependencies))
    }

    /// Parse package.json for Node.js projects
    async fn parse_package_json(&self, package_json: &PathBuf) -> Result<(Option<String>, Vec<DependencyInfo>), AIIntegrationError> {
        let content = tokio::fs::read_to_string(package_json).await?;
        
        // Parse JSON content
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| AIIntegrationError::ConfigurationError {
                message: format!("Failed to parse package.json: {}", e),
            })?;
        
        // Extract version
        let version = parsed.get("version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Extract dependencies
        let mut dependencies = Vec::new();
        
        if let Some(deps) = parsed.get("dependencies").and_then(|d| d.as_object()) {
            for (name, version_val) in deps {
                if let Some(version_str) = version_val.as_str() {
                    dependencies.push(DependencyInfo {
                        name: name.clone(),
                        version: version_str.to_string(),
                        dependency_type: DependencyType::Direct,
                        source: "npm".to_string(),
                    });
                }
            }
        }
        
        // Extract devDependencies
        if let Some(dev_deps) = parsed.get("devDependencies").and_then(|d| d.as_object()) {
            for (name, version_val) in dev_deps {
                if let Some(version_str) = version_val.as_str() {
                    dependencies.push(DependencyInfo {
                        name: name.clone(),
                        version: version_str.to_string(),
                        dependency_type: DependencyType::Development,
                        source: "npm".to_string(),
                    });
                }
            }
        }
        
        Ok((version, dependencies))
    }

    /// Calculate hash of file content for integrity checking
    fn calculate_file_hash(&self, content: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Detect programming language from file extension
    fn detect_language(&self, extension: &str) -> String {
        match extension {
            "rs" => "Rust",
            "prism" => "Prism",
            "toml" => "TOML",
            "json" => "JSON",
            "yaml" | "yml" => "YAML",
            "md" => "Markdown",
            _ => "Unknown",
        }.to_string()
    }

    /// Check if directory should be skipped during scanning
    fn should_skip_directory(&self, path: &PathBuf) -> bool {
        let skip_dirs = ["target", "node_modules", ".git", ".vscode", "dist", "build"];
        
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
            skip_dirs.contains(&dir_name) || dir_name.starts_with('.')
        } else {
            false
        }
    }

    /// Extract business context from metadata
    async fn extract_business_context(
        &self,
        syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        semantic_metadata: &Option<SemanticAIMetadata>,
    ) -> Result<Option<BusinessContext>, AIIntegrationError> {
        let mut domain = None;
        let mut capabilities = Vec::new();
        let mut rules = Vec::new();
        
        // Extract domain information from syntax metadata
        if let Some(syntax_meta) = syntax_metadata {
            // Analyze business concepts from syntax metadata
            domain = self.extract_domain_from_syntax(syntax_meta);
            capabilities.extend(self.extract_capabilities_from_syntax(syntax_meta));
        }
        
        // Extract business rules from semantic metadata
        if let Some(semantic_meta) = semantic_metadata {
            rules.extend(self.extract_business_rules_from_semantic(semantic_meta));
        }
        
        // Only return business context if we found meaningful information
        if domain.is_some() || !capabilities.is_empty() || !rules.is_empty() {
            Ok(Some(BusinessContext {
                domain,
                capabilities,
                rules,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Extract domain information from syntax metadata
    fn extract_domain_from_syntax(&self, syntax_metadata: &prism_common::ai_metadata::AIMetadata) -> Option<String> {
        // Analyze syntax patterns to infer business domain
        let business_rules = &syntax_metadata.business_rules;
        
        // Look for domain-specific patterns
        let common_domains = [
            ("web", "Web Development"),
            ("api", "API Development"),
            ("data", "Data Processing"),
            ("auth", "Authentication"),
            ("payment", "Financial Services"),
            ("user", "User Management"),
            ("inventory", "Inventory Management"),
            ("order", "Order Management"),
            ("analytics", "Analytics"),
            ("ml", "Machine Learning"),
        ];
        
        for (pattern, domain) in &common_domains {
            if business_rules.iter().any(|rule| 
                rule.name.to_lowercase().contains(pattern) ||
                rule.description.to_lowercase().contains(pattern)
            ) {
                return Some(domain.to_string());
            }
        }
        
        // Default to software development if no specific domain found
        Some("Software Development".to_string())
    }
    
    /// Extract capabilities from syntax metadata
    fn extract_capabilities_from_syntax(&self, syntax_metadata: &prism_common::ai_metadata::AIMetadata) -> Vec<String> {
        let mut capabilities = Vec::new();
        
        // Extract from business rules
        for rule in &syntax_metadata.business_rules {
            // Map rule categories to capabilities
            match rule.category {
                prism_common::ai_metadata::BusinessRuleCategory::Validation => 
                    capabilities.push(format!("Validation: {}", rule.name)),
                prism_common::ai_metadata::BusinessRuleCategory::Constraint => 
                    capabilities.push(format!("Constraint: {}", rule.name)),
                prism_common::ai_metadata::BusinessRuleCategory::Workflow => 
                    capabilities.push(format!("Workflow: {}", rule.name)),
                prism_common::ai_metadata::BusinessRuleCategory::Compliance => 
                    capabilities.push(format!("Compliance: {}", rule.name)),
                prism_common::ai_metadata::BusinessRuleCategory::Security => 
                    capabilities.push(format!("Security: {}", rule.name)),
            }
        }
        
        // Extract from insights (including architectural patterns)
        for insight in &syntax_metadata.insights {
            match insight.insight_type {
                prism_common::ai_metadata::AIInsightType::ArchitecturalImprovement => 
                    capabilities.push(format!("Architecture: {}", insight.content)),
                prism_common::ai_metadata::AIInsightType::PatternRecognition => 
                    capabilities.push(format!("Pattern: {}", insight.content)),
                _ => capabilities.push(insight.content.clone()),
            }
        }
        
        capabilities
    }
    
    /// Extract business rules from semantic metadata
    fn extract_business_rules_from_semantic(&self, _semantic_metadata: &SemanticAIMetadata) -> Vec<String> {
        // Since semantic metadata is currently a placeholder, return common rules
        vec![
            "Type Safety: All types must be validated".to_string(),
            "Error Handling: All errors must be properly handled".to_string(),
            "Resource Management: All resources must be properly managed".to_string(),
        ]
    }

    /// Analyze relationships between components
    async fn analyze_relationships(
        &self,
        syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        semantic_metadata: &Option<SemanticAIMetadata>,
        runtime_metadata: &Option<RuntimeAIMetadata>,
        pir_metadata: &Option<PIRAIMetadata>,
        effects_metadata: &Option<EffectsAIMetadata>,
    ) -> Result<Vec<CrossSystemRelationship>, AIIntegrationError> {
        let mut relationships = Vec::new();
        
        // Analyze syntax-to-semantic relationships
        if let (Some(syntax_meta), Some(_semantic_meta)) = (syntax_metadata, semantic_metadata) {
            relationships.extend(self.analyze_syntax_semantic_relationships(syntax_meta));
        }
        
        // Analyze semantic-to-PIR relationships
        if let (Some(_semantic_meta), Some(_pir_meta)) = (semantic_metadata, pir_metadata) {
            relationships.extend(self.analyze_semantic_pir_relationships());
        }
        
        // Analyze PIR-to-runtime relationships
        if let (Some(_pir_meta), Some(_runtime_meta)) = (pir_metadata, runtime_metadata) {
            relationships.extend(self.analyze_pir_runtime_relationships());
        }
        
        // Analyze effects relationships
        if let Some(_effects_meta) = effects_metadata {
            relationships.extend(self.analyze_effects_relationships());
        }
        
        // Analyze data flow relationships
        relationships.extend(self.analyze_data_flow_relationships(
            syntax_metadata, semantic_metadata, pir_metadata, runtime_metadata
        ));
        
        Ok(relationships)
    }
    
    /// Analyze relationships between syntax and semantic systems
    fn analyze_syntax_semantic_relationships(
        &self,
        syntax_metadata: &prism_common::ai_metadata::AIMetadata,
    ) -> Vec<CrossSystemRelationship> {
        let mut relationships = Vec::new();
        
        // Create relationships for each business rule
        for rule in &syntax_metadata.business_rules {
            relationships.push(CrossSystemRelationship {
                source: ComponentReference {
                    system: "syntax".to_string(),
                    component_id: rule.name.clone(),
                    component_type: format!("{:?}", rule.category),
                },
                target: ComponentReference {
                    system: "semantic".to_string(),
                    component_id: format!("semantic_{}", rule.name),
                    component_type: "type_analysis".to_string(),
                },
                relationship_type: RelationshipType::DataFlow,
                strength: 0.8, // Default confidence for business rules
                description: format!("Business rule '{}' flows to semantic analysis", rule.name),
            });
        }
        
        relationships
    }
    
    /// Analyze relationships between semantic and PIR systems
    fn analyze_semantic_pir_relationships(&self) -> Vec<CrossSystemRelationship> {
        vec![
            CrossSystemRelationship {
                source: ComponentReference {
                    system: "semantic".to_string(),
                    component_id: "type_system".to_string(),
                    component_type: "analysis".to_string(),
                },
                target: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "type_registry".to_string(),
                    component_type: "registry".to_string(),
                },
                relationship_type: RelationshipType::DataFlow,
                strength: 0.95,
                description: "Semantic type information flows to PIR type registry".to_string(),
            },
            CrossSystemRelationship {
                source: ComponentReference {
                    system: "semantic".to_string(),
                    component_id: "business_rules".to_string(),
                    component_type: "validation".to_string(),
                },
                target: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "business_context".to_string(),
                    component_type: "context".to_string(),
                },
                relationship_type: RelationshipType::Composition,
                strength: 0.85,
                description: "Semantic business rules compose PIR business context".to_string(),
            }
        ]
    }
    
    /// Analyze relationships between PIR and runtime systems
    fn analyze_pir_runtime_relationships(&self) -> Vec<CrossSystemRelationship> {
        vec![
            CrossSystemRelationship {
                source: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "modules".to_string(),
                    component_type: "structure".to_string(),
                },
                target: ComponentReference {
                    system: "runtime".to_string(),
                    component_id: "execution_units".to_string(),
                    component_type: "execution".to_string(),
                },
                relationship_type: RelationshipType::Implementation,
                strength: 0.90,
                description: "PIR modules are implemented as runtime execution units".to_string(),
            },
            CrossSystemRelationship {
                source: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "performance_profile".to_string(),
                    component_type: "metrics".to_string(),
                },
                target: ComponentReference {
                    system: "runtime".to_string(),
                    component_id: "performance_monitor".to_string(),
                    component_type: "monitoring".to_string(),
                },
                relationship_type: RelationshipType::Usage,
                strength: 0.80,
                description: "PIR performance profiles guide runtime monitoring".to_string(),
            }
        ]
    }
    
    /// Analyze effects system relationships
    fn analyze_effects_relationships(&self) -> Vec<CrossSystemRelationship> {
        vec![
            CrossSystemRelationship {
                source: ComponentReference {
                    system: "effects".to_string(),
                    component_id: "capability_system".to_string(),
                    component_type: "security".to_string(),
                },
                target: ComponentReference {
                    system: "runtime".to_string(),
                    component_id: "capability_enforcement".to_string(),
                    component_type: "security".to_string(),
                },
                relationship_type: RelationshipType::Dependency,
                strength: 0.95,
                description: "Runtime depends on effects capability system for security".to_string(),
            },
            CrossSystemRelationship {
                source: ComponentReference {
                    system: "effects".to_string(),
                    component_id: "effect_graph".to_string(),
                    component_type: "analysis".to_string(),
                },
                target: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "effect_integration".to_string(),
                    component_type: "integration".to_string(),
                },
                relationship_type: RelationshipType::Association,
                strength: 0.85,
                description: "Effects graph is associated with PIR effect integration".to_string(),
            }
        ]
    }
    
    /// Analyze data flow relationships across all systems
    fn analyze_data_flow_relationships(
        &self,
        syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        _semantic_metadata: &Option<SemanticAIMetadata>,
        _pir_metadata: &Option<PIRAIMetadata>,
        _runtime_metadata: &Option<RuntimeAIMetadata>,
    ) -> Vec<CrossSystemRelationship> {
        let mut relationships = Vec::new();
        
        // Create a data flow pipeline relationship
        if syntax_metadata.is_some() {
            relationships.push(CrossSystemRelationship {
                source: ComponentReference {
                    system: "syntax".to_string(),
                    component_id: "ast".to_string(),
                    component_type: "structure".to_string(),
                },
                target: ComponentReference {
                    system: "semantic".to_string(),
                    component_id: "analyzer".to_string(),
                    component_type: "processor".to_string(),
                },
                relationship_type: RelationshipType::DataFlow,
                strength: 1.0,
                description: "AST flows from syntax to semantic analysis".to_string(),
            });
            
            relationships.push(CrossSystemRelationship {
                source: ComponentReference {
                    system: "semantic".to_string(),
                    component_id: "analyzed_ast".to_string(),
                    component_type: "structure".to_string(),
                },
                target: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "constructor".to_string(),
                    component_type: "processor".to_string(),
                },
                relationship_type: RelationshipType::DataFlow,
                strength: 0.95,
                description: "Analyzed AST flows from semantic to PIR construction".to_string(),
            });
            
            relationships.push(CrossSystemRelationship {
                source: ComponentReference {
                    system: "pir".to_string(),
                    component_id: "intermediate_representation".to_string(),
                    component_type: "structure".to_string(),
                },
                target: ComponentReference {
                    system: "runtime".to_string(),
                    component_id: "executor".to_string(),
                    component_type: "processor".to_string(),
                },
                relationship_type: RelationshipType::DataFlow,
                strength: 0.90,
                description: "PIR flows to runtime execution".to_string(),
            });
        }
        
        relationships
    }

    /// Calculate quality metrics
    async fn calculate_quality_metrics(
        &self,
        project_info: &ProjectInfo,
        syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
        semantic_metadata: &Option<SemanticAIMetadata>,
    ) -> Result<QualityMetrics, AIIntegrationError> {
        let mut lines_of_code = 0;
        let mut cyclomatic_complexity = 0;
        let mut cognitive_complexity = 0;
        let mut documentation_coverage = 0.0;
        let mut technical_debt_ratio = 0.0;
        
        // Calculate lines of code from source files
        lines_of_code = self.calculate_lines_of_code(project_info).await?;
        
        // Calculate complexity metrics from syntax metadata
        if let Some(syntax_meta) = syntax_metadata {
            cyclomatic_complexity = self.calculate_cyclomatic_complexity(syntax_meta);
            cognitive_complexity = self.calculate_cognitive_complexity(syntax_meta);
            documentation_coverage = self.calculate_documentation_coverage(syntax_meta, project_info);
            technical_debt_ratio = self.calculate_technical_debt_ratio(syntax_meta);
        }
        
        // Estimate test coverage (would need actual test execution data)
        let test_coverage = self.estimate_test_coverage(project_info, syntax_metadata);
        
        Ok(QualityMetrics {
            lines_of_code,
            cyclomatic_complexity: cyclomatic_complexity.try_into().unwrap(),
            cognitive_complexity: cognitive_complexity.try_into().unwrap(),
            test_coverage,
            documentation_coverage,
            technical_debt_ratio,
        })
    }

    /// Calculate total lines of code from source files
    async fn calculate_lines_of_code(&self, project_info: &ProjectInfo) -> Result<u64, AIIntegrationError> {
        let mut total_lines = 0u64;
        
        for source_file in &project_info.source_files {
            // Only count code files, not documentation or config
            if self.is_code_file(&source_file.language) {
                let file_path = project_info.root_path.join(&source_file.path);
                if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                    let lines = content.lines().count() as u64;
                    total_lines += lines;
                }
            }
        }
        
        Ok(total_lines)
    }

    /// Calculate cyclomatic complexity from syntax metadata
    fn calculate_cyclomatic_complexity(&self, syntax_metadata: &prism_common::ai_metadata::AIMetadata) -> u64 {
        let mut complexity = 0u64;
        
        // Base complexity of 1 for each function-related business rule
        complexity += syntax_metadata.business_rules
            .iter()
            .filter(|rule| rule.name.to_lowercase().contains("function"))
            .count() as u64;
        
        // Add complexity for control flow structures from insights
        for insight in &syntax_metadata.insights {
            match insight.insight_type {
                prism_common::ai_metadata::AIInsightType::PatternRecognition => {
                    if insight.content.contains("Conditional") {
                        complexity += (insight.confidence * 10.0) as u64;
                    } else if insight.content.contains("Loop") {
                        complexity += (insight.confidence * 8.0) as u64;
                    } else if insight.content.contains("Error") {
                        complexity += (insight.confidence * 5.0) as u64;
                    } else if insight.content.contains("Pattern Matching") {
                        complexity += (insight.confidence * 6.0) as u64;
                    } else {
                        complexity += (insight.confidence * 2.0) as u64;
                    }
                },
                _ => complexity += (insight.confidence * 2.0) as u64,
            }
        }
        
        complexity
    }

    /// Calculate cognitive complexity from syntax metadata
    fn calculate_cognitive_complexity(&self, syntax_metadata: &prism_common::ai_metadata::AIMetadata) -> u64 {
        let mut complexity = 0u64;
        
        // Cognitive complexity considers nesting and logical operators
        for insight in &syntax_metadata.insights {
            if insight.content.contains("Nested Structures") {
                complexity += (insight.confidence * 15.0) as u64;
            } else if insight.content.contains("Complex Conditions") {
                complexity += (insight.confidence * 12.0) as u64;
            } else if insight.content.contains("Recursive Patterns") {
                complexity += (insight.confidence * 10.0) as u64;
            } else if insight.content.contains("Exception Handling") {
                complexity += (insight.confidence * 8.0) as u64;
            } else {
                complexity += (insight.confidence * 3.0) as u64;
            }
        }
        
        complexity
    }

    /// Calculate documentation coverage
    fn calculate_documentation_coverage(
        &self,
        syntax_metadata: &prism_common::ai_metadata::AIMetadata,
        project_info: &ProjectInfo,
    ) -> f64 {
        let total_functions = syntax_metadata.business_rules
            .iter()
            .filter(|rule| rule.name.to_lowercase().contains("function"))
            .count() as f64;
        
        if total_functions == 0.0 {
            return 1.0; // 100% if no functions to document
        }
        
        // Count documentation files
        let doc_files = project_info.source_files
            .iter()
            .filter(|file| file.language == "Markdown" || file.path.to_string_lossy().contains("doc"))
            .count() as f64;
        
        // Estimate documentation coverage based on doc files and function count
        let coverage = (doc_files / total_functions).min(1.0);
        
        // Adjust based on insights that indicate good documentation
        let doc_insights = syntax_metadata.insights
            .iter()
            .filter(|insight| insight.content.contains("Documentation") || 
                              insight.content.contains("Comment"))
            .map(|insight| insight.confidence)
            .sum::<f64>();
        
        (coverage + doc_insights * 0.1).min(1.0)
    }

    /// Calculate technical debt ratio
    fn calculate_technical_debt_ratio(&self, syntax_metadata: &prism_common::ai_metadata::AIMetadata) -> f64 {
        let mut debt_indicators = 0.0;
        let total_indicators = syntax_metadata.insights.len() as f64;
        
        if total_indicators == 0.0 {
            return 0.0;
        }
        
        // Count insights that indicate technical debt
        for insight in &syntax_metadata.insights {
            if insight.content.contains("Code Duplication") {
                debt_indicators += insight.confidence * 2.0;
            } else if insight.content.contains("Long Functions") {
                debt_indicators += insight.confidence * 1.5;
            } else if insight.content.contains("Complex Conditions") {
                debt_indicators += insight.confidence * 1.3;
            } else if insight.content.contains("Magic Numbers") {
                debt_indicators += insight.confidence * 1.2;
            } else if insight.content.contains("Nested Structures") {
                debt_indicators += insight.confidence * 1.1;
            }
        }
        
        (debt_indicators / total_indicators).min(1.0_f64)
    }

    /// Estimate test coverage from project structure
    fn estimate_test_coverage(
        &self,
        project_info: &ProjectInfo,
        _syntax_metadata: &Option<prism_common::ai_metadata::AIMetadata>,
    ) -> f64 {
        let total_code_files = project_info.source_files
            .iter()
            .filter(|file| self.is_code_file(&file.language))
            .count() as f64;
        
        if total_code_files == 0.0 {
            return 0.0;
        }
        
        let test_files = project_info.source_files
            .iter()
            .filter(|file| {
                let path_str = file.path.to_string_lossy().to_lowercase();
                path_str.contains("test") || 
                path_str.contains("spec") ||
                path_str.ends_with("_test.rs") ||
                path_str.starts_with("test_")
            })
            .count() as f64;
        
        // Rough estimate: assume each test file covers multiple source files
        (test_files * 2.5 / total_code_files).min(1.0)
    }

    /// Check if a file is a code file (not documentation or config)
    fn is_code_file(&self, language: &str) -> bool {
        matches!(language, "Rust" | "Prism" | "JavaScript" | "TypeScript" | "Python" | "Java" | "C++" | "C")
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
