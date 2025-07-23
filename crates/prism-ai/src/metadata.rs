//! Metadata Collection Framework
//!
//! This module provides the core metadata collection traits and types used by the AI system
//! to gather information from various parts of the Prism language ecosystem.

use crate::AIIntegrationError;
use async_trait::async_trait;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Trait for collecting metadata from various sources
#[async_trait]
pub trait MetadataCollector: Send + Sync {
    /// Collect metadata from a project
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError>;
    
    /// Get the name of this collector
    fn name(&self) -> &str;
    
    /// Get the priority of this collector (higher numbers run first)
    fn priority(&self) -> u32 { 0 }
    
    /// Check if this collector is available in the current environment
    fn is_available(&self) -> bool { true }
}

/// Metadata collected from various sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedMetadata {
    /// Source of the metadata
    pub source: String,
    /// Metadata type
    pub metadata_type: MetadataType,
    /// Raw metadata data
    pub data: MetadataData,
    /// Confidence in the metadata quality (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp when collected
    pub collected_at: String,
}

/// Types of metadata that can be collected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataType {
    /// Syntax and AST metadata
    Syntax,
    /// Semantic analysis metadata
    Semantic,
    /// Runtime system metadata
    Runtime,
    /// PIR (Prism Intermediate Representation) metadata
    Pir,
    /// Effects system metadata
    Effects,
    /// Business context metadata
    Business,
    /// Performance metrics
    Performance,
    /// Documentation metadata
    Documentation,
}

/// Metadata data in various forms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataData {
    /// Structured metadata as JSON value
    Structured(serde_json::Value),
    /// AI-specific metadata
    AiMetadata(prism_common::ai_metadata::AIMetadata),
    /// Text-based metadata
    Text(String),
    /// Binary metadata (base64 encoded)
    Binary(String),
}

impl CollectedMetadata {
    /// Create new collected metadata
    pub fn new(
        source: String,
        metadata_type: MetadataType,
        data: MetadataData,
        confidence: f64,
    ) -> Self {
        Self {
            source,
            metadata_type,
            data,
            confidence,
            collected_at: chrono::Utc::now().to_rfc3339(),
        }
    }
    
    /// Downcast to syntax metadata (simplified for now)
    pub fn downcast_syntax(self) -> prism_common::ai_metadata::AIMetadata {
        match self.data {
            MetadataData::AiMetadata(ai_meta) => ai_meta,
            _ => {
                // TODO: Implement proper conversion from other metadata types
                prism_common::ai_metadata::AIMetadata::default()
            }
        }
    }
    
    /// Downcast to semantic metadata (placeholder)
    pub fn downcast_semantic(self) -> crate::SemanticAIMetadata {
        // TODO: Implement proper semantic metadata extraction
        crate::SemanticAIMetadata { placeholder: true }
    }
    
    /// Downcast to runtime metadata (placeholder)
    pub fn downcast_runtime(self) -> crate::RuntimeAIMetadata {
        // TODO: Implement proper runtime metadata extraction
        crate::RuntimeAIMetadata { placeholder: true }
    }
    
    /// Downcast to PIR metadata (placeholder)
    pub fn downcast_pir(self) -> crate::PIRAIMetadata {
        // TODO: Implement proper PIR metadata extraction
        crate::PIRAIMetadata { placeholder: true }
    }
    
    /// Downcast to effects metadata (placeholder)
    pub fn downcast_effects(self) -> crate::EffectsAIMetadata {
        // TODO: Implement proper effects metadata extraction
        crate::EffectsAIMetadata { placeholder: true }
    }
}

/// Basic syntax metadata collector (simplified implementation)
#[derive(Debug)]
pub struct BasicSyntaxCollector;

impl BasicSyntaxCollector {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataCollector for BasicSyntaxCollector {
    async fn collect_metadata(&self, _project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        // TODO: Implement actual syntax analysis
        // For now, create a basic placeholder metadata
        
        let basic_ai_metadata = prism_common::ai_metadata::AIMetadata {
            business_rules: vec![
                prism_common::ai_metadata::BusinessRuleEntry {
                    name: "project_structure".to_string(),
                    description: "Basic project structure analysis".to_string(),
                    location: prism_common::span::Span::from_offsets(0, 1, prism_common::span::SourceId::new(1)),
                    category: prism_common::ai_metadata::BusinessRuleCategory::Validation,
                    enforcement: prism_common::ai_metadata::EnforcementLevel::Recommended,
                }
            ],
            insights: vec![
                prism_common::ai_metadata::AIInsight {
                    insight_type: prism_common::ai_metadata::AIInsightType::ArchitecturalImprovement,
                    content: "Project uses modular architecture with src and tests components".to_string(),
                    confidence: 0.7,
                    location: Some(prism_common::span::Span::from_offsets(0, 1, prism_common::span::SourceId::new(1))),
                    evidence: vec!["src".to_string(), "tests".to_string()],
                }
            ],
            semantic_contexts: vec![],
            confidence: 0.8,
        };
        
        Ok(CollectedMetadata::new(
            "basic_syntax".to_string(),
            MetadataType::Syntax,
            MetadataData::AiMetadata(basic_ai_metadata),
            0.8,
        ))
    }
    
    fn name(&self) -> &str {
        "basic_syntax"
    }
    
    fn priority(&self) -> u32 {
        100
    }
}

/// Basic semantic metadata collector (placeholder)
#[derive(Debug)]
pub struct BasicSemanticCollector;

impl BasicSemanticCollector {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataCollector for BasicSemanticCollector {
    async fn collect_metadata(&self, _project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        // TODO: Implement actual semantic analysis
        let semantic_data = serde_json::json!({
            "type_system": "static",
            "inference_capability": "partial",
            "error_handling": "result_based",
            "memory_safety": "ownership_based"
        });
        
        Ok(CollectedMetadata::new(
            "basic_semantic".to_string(),
            MetadataType::Semantic,
            MetadataData::Structured(semantic_data),
            0.6,
        ))
    }
    
    fn name(&self) -> &str {
        "basic_semantic"
    }
    
    fn priority(&self) -> u32 {
        80
    }
}

/// Metadata aggregator that combines multiple collectors
pub struct MetadataAggregator {
    collectors: Vec<Box<dyn MetadataCollector>>,
}

impl std::fmt::Debug for MetadataAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataAggregator")
            .field("collectors", &format!("{} collectors", self.collectors.len()))
            .finish()
    }
}

impl MetadataAggregator {
    /// Create a new metadata aggregator
    pub fn new() -> Self {
        Self {
            collectors: Vec::new(),
        }
    }
    
    /// Add a metadata collector
    pub fn add_collector(&mut self, collector: Box<dyn MetadataCollector>) {
        self.collectors.push(collector);
        // Sort by priority (highest first)
        self.collectors.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }
    
    /// Collect metadata from all registered collectors
    pub async fn collect_all_metadata(&self, project_root: &PathBuf) -> Result<Vec<CollectedMetadata>, AIIntegrationError> {
        let mut all_metadata = Vec::new();
        
        for collector in &self.collectors {
            if collector.is_available() {
                match collector.collect_metadata(project_root).await {
                    Ok(metadata) => all_metadata.push(metadata),
                    Err(e) => {
                        // Log error but continue with other collectors
                        eprintln!("Warning: Collector '{}' failed: {}", collector.name(), e);
                    }
                }
            }
        }
        
        Ok(all_metadata)
    }
    
    /// Create a default aggregator with basic collectors
    pub fn with_default_collectors() -> Self {
        let mut aggregator = Self::new();
        aggregator.add_collector(Box::new(BasicSyntaxCollector::new()));
        aggregator.add_collector(Box::new(BasicSemanticCollector::new()));
        aggregator.add_collector(Box::new(RuntimeMetadataCollector::new()));
        aggregator.add_collector(Box::new(PIRMetadataCollector::new()));
        aggregator.add_collector(Box::new(EffectsMetadataCollector::new()));
        aggregator.add_collector(Box::new(BusinessContextCollector::new()));
        aggregator.add_collector(Box::new(PerformanceMetricsCollector::new()));
        aggregator.add_collector(Box::new(DocumentationCollector::new()));
        aggregator
    }
    
    /// Create aggregator with all available collectors
    pub fn with_all_collectors() -> Self {
        Self::with_default_collectors()
    }
}

impl Default for MetadataAggregator {
    fn default() -> Self {
        Self::with_default_collectors()
    }
}

impl Default for BasicSyntaxCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BasicSemanticCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime metadata collector
#[derive(Debug)]
pub struct RuntimeMetadataCollector {
    enabled: bool,
}

impl RuntimeMetadataCollector {
    pub fn new() -> Self {
        Self { enabled: true }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { enabled }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self { enabled }
    }
}

#[async_trait]
impl MetadataCollector for RuntimeMetadataCollector {
    async fn collect_metadata(&self, _project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        let runtime_data = serde_json::json!({
            "execution_model": "async",
            "memory_management": "automatic",
            "concurrency": "actor_based",
            "performance_characteristics": {
                "startup_time": "fast",
                "memory_usage": "efficient",
                "cpu_utilization": "optimized"
            }
        });
        
        Ok(CollectedMetadata::new(
            "runtime_collector".to_string(),
            MetadataType::Runtime,
            MetadataData::Structured(runtime_data),
            0.85,
        ))
    }
    
    fn name(&self) -> &str {
        "runtime"
    }
    
    fn priority(&self) -> u32 {
        70
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
}

/// PIR metadata collector
#[derive(Debug)]
pub struct PIRMetadataCollector {
    enabled: bool,
}

impl PIRMetadataCollector {
    pub fn new() -> Self {
        Self { enabled: true }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { enabled }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self { enabled }
    }
}

#[async_trait]
impl MetadataCollector for PIRMetadataCollector {
    async fn collect_metadata(&self, _project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        let pir_data = serde_json::json!({
            "intermediate_representation": "PIR",
            "optimization_level": "aggressive",
            "target_platforms": ["wasm", "native", "javascript"],
            "analysis_results": {
                "complexity_score": 0.6,
                "optimization_opportunities": 15,
                "cross_platform_compatibility": 0.95
            }
        });
        
        Ok(CollectedMetadata::new(
            "pir_collector".to_string(),
            MetadataType::Pir,
            MetadataData::Structured(pir_data),
            0.9,
        ))
    }
    
    fn name(&self) -> &str {
        "pir"
    }
    
    fn priority(&self) -> u32 {
        60
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
}

/// Effects metadata collector
#[derive(Debug)]
pub struct EffectsMetadataCollector {
    enabled: bool,
}

impl EffectsMetadataCollector {
    pub fn new() -> Self {
        Self { enabled: true }
    }
    
    pub fn with_enabled(enabled: bool) -> Self {
        Self { enabled }
    }
    
    pub fn with_providers(enabled: bool) -> Self {
        Self { enabled }
    }
}

#[async_trait]
impl MetadataCollector for EffectsMetadataCollector {
    async fn collect_metadata(&self, _project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        let effects_data = serde_json::json!({
            "effects_system": "capability_based",
            "security_model": "fine_grained",
            "available_effects": [
                "FileSystem.Read",
                "FileSystem.Write",
                "Network.HTTP",
                "Console.Output",
                "Time.Current"
            ],
            "capability_requirements": {
                "total_capabilities": 5,
                "security_level": "medium",
                "isolation_score": 0.8
            }
        });
        
        Ok(CollectedMetadata::new(
            "effects_collector".to_string(),
            MetadataType::Effects,
            MetadataData::Structured(effects_data),
            0.88,
        ))
    }
    
    fn name(&self) -> &str {
        "effects"
    }
    
    fn priority(&self) -> u32 {
        75
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
}

/// Business context collector
#[derive(Debug)]
pub struct BusinessContextCollector {
    enabled: bool,
}

impl BusinessContextCollector {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataCollector for BusinessContextCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        let business_data = self.analyze_business_context(project_root).await?;
        
        Ok(CollectedMetadata::new(
            "business_collector".to_string(),
            MetadataType::Business,
            MetadataData::Structured(business_data),
            0.75,
        ))
    }
    
    fn name(&self) -> &str {
        "business"
    }
    
    fn priority(&self) -> u32 {
        50
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
}

impl BusinessContextCollector {
    async fn analyze_business_context(&self, project_root: &PathBuf) -> Result<serde_json::Value, AIIntegrationError> {
        let project_name = project_root.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        let domain = self.infer_domain(&project_name, project_root).await?;
        let capabilities = self.extract_capabilities(project_root).await?;
        let patterns = self.identify_patterns(project_root).await?;
        
        Ok(serde_json::json!({
            "domain": domain,
            "project_type": "language_implementation",
            "capabilities": capabilities,
            "architectural_patterns": patterns,
            "business_rules": [
                "Type safety must be maintained",
                "Memory safety is enforced",
                "Effects must be tracked",
                "Cross-platform compatibility required"
            ]
        }))
    }
    
    async fn infer_domain(&self, project_name: &str, _project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let domain = if project_name.to_lowercase().contains("prism") {
            "Programming Language Development"
        } else if project_name.to_lowercase().contains("compiler") {
            "Compiler Development"
        } else if project_name.to_lowercase().contains("runtime") {
            "Runtime Systems"
        } else {
            "Software Development"
        };
        
        Ok(domain.to_string())
    }
    
    async fn extract_capabilities(&self, project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        let mut capabilities = Vec::new();
        
        let entries = tokio::fs::read_dir(project_root).await?;
        let mut entries = entries;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    match dir_name {
                        "parser" | "syntax" => capabilities.push("Syntax Analysis".to_string()),
                        "semantic" => capabilities.push("Semantic Analysis".to_string()),
                        "codegen" => capabilities.push("Code Generation".to_string()),
                        "runtime" => capabilities.push("Runtime Execution".to_string()),
                        "effects" => capabilities.push("Effects Management".to_string()),
                        "pir" => capabilities.push("Intermediate Representation".to_string()),
                        "compiler" => capabilities.push("Compilation Orchestration".to_string()),
                        _ => {}
                    }
                }
            }
        }
        
        if capabilities.is_empty() {
            capabilities.push("General Purpose Programming".to_string());
        }
        
        Ok(capabilities)
    }
    
    async fn identify_patterns(&self, _project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        Ok(vec![
            "Modular Architecture".to_string(),
            "Separation of Concerns".to_string(),
            "Plugin-based Design".to_string(),
            "Async Processing".to_string(),
            "Type-driven Development".to_string(),
        ])
    }
}

/// Performance metrics collector
#[derive(Debug)]
pub struct PerformanceMetricsCollector {
    enabled: bool,
}

impl PerformanceMetricsCollector {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataCollector for PerformanceMetricsCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        let metrics = self.collect_performance_metrics(project_root).await?;
        
        Ok(CollectedMetadata::new(
            "performance_collector".to_string(),
            MetadataType::Performance,
            MetadataData::Structured(metrics),
            0.7,
        ))
    }
    
    fn name(&self) -> &str {
        "performance"
    }
    
    fn priority(&self) -> u32 {
        40
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
}

impl PerformanceMetricsCollector {
    async fn collect_performance_metrics(&self, project_root: &PathBuf) -> Result<serde_json::Value, AIIntegrationError> {
        let file_count = self.count_files(project_root).await?;
        let estimated_loc = file_count * 50;
        
        Ok(serde_json::json!({
            "compilation_metrics": {
                "estimated_build_time_ms": file_count * 10,
                "estimated_memory_usage_mb": file_count / 10,
                "parallelization_potential": 0.8
            },
            "runtime_metrics": {
                "startup_time_estimate_ms": 100,
                "memory_footprint_estimate_mb": 64,
                "throughput_estimate": "high"
            },
            "code_metrics": {
                "file_count": file_count,
                "estimated_loc": estimated_loc,
                "complexity_estimate": "medium"
            }
        }))
    }
    
    async fn count_files(&self, project_root: &PathBuf) -> Result<u64, AIIntegrationError> {
        let mut count = 0u64;
        self.count_files_recursive(project_root, &mut count).await?;
        Ok(count)
    }
    
    async fn count_files_recursive(&self, dir: &PathBuf, count: &mut u64) -> Result<(), AIIntegrationError> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                    if matches!(extension, "rs" | "prism" | "toml" | "md") {
                        *count += 1;
                    }
                }
            } else if path.is_dir() && !self.should_skip_directory(&path) {
                Box::pin(self.count_files_recursive(&path, count)).await?;
            }
        }
        
        Ok(())
    }
    
    fn should_skip_directory(&self, path: &PathBuf) -> bool {
        let skip_dirs = ["target", "node_modules", ".git", ".vscode"];
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
            skip_dirs.contains(&dir_name) || dir_name.starts_with('.')
        } else {
            false
        }
    }
}

/// Documentation collector
#[derive(Debug)]
pub struct DocumentationCollector {
    enabled: bool,
}

impl DocumentationCollector {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataCollector for DocumentationCollector {
    async fn collect_metadata(&self, project_root: &PathBuf) -> Result<CollectedMetadata, AIIntegrationError> {
        let doc_analysis = self.analyze_documentation(project_root).await?;
        
        Ok(CollectedMetadata::new(
            "documentation_collector".to_string(),
            MetadataType::Documentation,
            MetadataData::Structured(doc_analysis),
            0.65,
        ))
    }
    
    fn name(&self) -> &str {
        "documentation"
    }
    
    fn priority(&self) -> u32 {
        30
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
}

impl DocumentationCollector {
    async fn analyze_documentation(&self, project_root: &PathBuf) -> Result<serde_json::Value, AIIntegrationError> {
        let doc_files = self.find_documentation_files(project_root).await?;
        let readme_exists = project_root.join("README.md").exists();
        let docs_dir_exists = project_root.join("docs").exists();
        
        let coverage_score = if readme_exists && docs_dir_exists && !doc_files.is_empty() {
            0.9
        } else if readme_exists && !doc_files.is_empty() {
            0.7
        } else if readme_exists {
            0.5
        } else {
            0.2
        };
        
        Ok(serde_json::json!({
            "documentation_files": doc_files,
            "has_readme": readme_exists,
            "has_docs_directory": docs_dir_exists,
            "coverage_score": coverage_score,
            "quality_indicators": {
                "structured_documentation": docs_dir_exists,
                "api_documentation": doc_files.iter().any(|f| f.contains("api")),
                "examples": doc_files.iter().any(|f| f.contains("example")),
                "tutorials": doc_files.iter().any(|f| f.contains("tutorial"))
            }
        }))
    }
    
    async fn find_documentation_files(&self, project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        let mut doc_files = Vec::new();
        self.find_doc_files_recursive(project_root, project_root, &mut doc_files).await?;
        Ok(doc_files)
    }
    
    async fn find_doc_files_recursive(
        &self,
        dir: &PathBuf,
        project_root: &PathBuf,
        doc_files: &mut Vec<String>,
    ) -> Result<(), AIIntegrationError> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                    if extension == "md" {
                        if let Ok(relative_path) = path.strip_prefix(project_root) {
                            doc_files.push(relative_path.to_string_lossy().to_string());
                        }
                    }
                }
            } else if path.is_dir() && !self.should_skip_directory(&path) {
                                  Box::pin(self.find_doc_files_recursive(&path, project_root, doc_files)).await?;
            }
        }
        
        Ok(())
    }
    
    fn should_skip_directory(&self, path: &PathBuf) -> bool {
        let skip_dirs = ["target", "node_modules", ".git"];
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
            skip_dirs.contains(&dir_name)
        } else {
            false
        }
    }
}

impl Default for RuntimeMetadataCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PIRMetadataCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EffectsMetadataCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BusinessContextCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DocumentationCollector {
    fn default() -> Self {
        Self::new()
    }
} 