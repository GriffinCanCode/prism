//! Metadata Provider System
//!
//! This module provides a modern metadata provider system that allows different
//! components of the Prism ecosystem to contribute metadata in a structured way.
//! This is the preferred approach over the legacy collector system.

use crate::AIIntegrationError;
use async_trait::async_trait;
use std::path::PathBuf;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Trait for providing metadata from specific domains
#[async_trait]
pub trait MetadataProvider: Send + Sync {
    /// Get the domain this provider handles
    fn domain(&self) -> MetadataDomain;
    
    /// Get the name of this provider
    fn name(&self) -> &str;
    
    /// Check if this provider is available in the current environment
    fn is_available(&self) -> bool;
    
    /// Provide metadata for the given context
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError>;
    
    /// Get provider information
    fn provider_info(&self) -> ProviderInfo;
}

/// Domains of metadata that can be provided
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetadataDomain {
    /// Syntax and AST analysis
    Syntax,
    /// Semantic analysis and type checking
    Semantic,
    /// Runtime system information
    Runtime,
    /// PIR (Prism Intermediate Representation)
    Pir,
    /// Effects system metadata
    Effects,
    /// Compiler metadata
    Compiler,
    /// Business domain analysis
    Business,
    /// Performance metrics
    Performance,
    /// Documentation analysis
    Documentation,
}

/// Context provided to metadata providers
#[derive(Debug, Clone)]
pub struct ProviderContext {
    /// Project root directory
    pub project_root: PathBuf,
    /// Compilation artifacts (if available)
    pub compilation_artifacts: Option<CompilationArtifacts>,
    /// Runtime information (if available)
    pub runtime_info: Option<RuntimeInfo>,
    /// Provider configuration
    pub provider_config: ProviderConfig,
}

/// Compilation artifacts that can be used by providers
#[derive(Debug, Clone)]
pub struct CompilationArtifacts {
    /// AST representation
    pub ast: Option<serde_json::Value>,
    /// Symbol table
    pub symbol_table: Option<serde_json::Value>,
    /// Type information
    pub type_info: Option<serde_json::Value>,
    /// Compilation errors/warnings
    pub diagnostics: Vec<String>,
}

/// Runtime information that can be used by providers
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    /// Runtime configuration
    pub config: HashMap<String, String>,
    /// Performance metrics
    pub performance_metrics: Option<serde_json::Value>,
    /// Runtime diagnostics
    pub diagnostics: Vec<String>,
}

/// Configuration for providers
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Whether to include detailed information
    pub detailed: bool,
    /// Whether to include performance metrics
    pub include_performance: bool,
    /// Whether to include business context
    pub include_business_context: bool,
    /// Custom settings for specific providers
    pub custom_settings: HashMap<String, String>,
}

/// Information about a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider name
    pub name: String,
    /// Provider version
    pub version: String,
    /// Provider description
    pub description: String,
    /// Supported domains
    pub domains: Vec<MetadataDomain>,
    /// Provider capabilities
    pub capabilities: Vec<String>,
    /// Dependencies required by the provider
    pub dependencies: Vec<String>,
}

/// Metadata provided by a domain-specific provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainMetadata {
    /// Syntax domain metadata
    Syntax(SyntaxMetadata),
    /// Semantic domain metadata
    Semantic(SemanticMetadata),
    /// Runtime domain metadata
    Runtime(RuntimeMetadata),
    /// PIR domain metadata
    Pir(PirMetadata),
    /// Effects domain metadata
    Effects(EffectsMetadata),
    /// Compiler domain metadata
    Compiler(CompilerMetadata),
    /// Business domain metadata
    Business(BusinessMetadata),
    /// Performance domain metadata
    Performance(PerformanceMetadata),
    /// Documentation domain metadata
    Documentation(DocumentationMetadata),
}

/// Syntax-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxMetadata {
    /// AI context information
    pub ai_context: prism_common::ai_metadata::AIMetadata,
    /// Syntax tree structure
    pub tree_structure: Option<serde_json::Value>,
    /// Detected patterns
    pub patterns: Vec<String>,
    /// Confidence score
    pub confidence: f64,
}

/// Semantic-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMetadata {
    /// Type system information
    pub type_system: Option<serde_json::Value>,
    /// Symbol information
    pub symbols: Vec<String>,
    /// Semantic patterns
    pub patterns: Vec<String>,
    /// Confidence score
    pub confidence: f64,
}

/// Runtime-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetadata {
    /// Runtime capabilities
    pub capabilities: Vec<String>,
    /// Performance characteristics
    pub performance: Option<serde_json::Value>,
    /// Resource usage
    pub resource_usage: Option<serde_json::Value>,
    /// Confidence score
    pub confidence: f64,
}

/// PIR-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PirMetadata {
    /// PIR structure information
    pub structure: Option<serde_json::Value>,
    /// Optimization information
    pub optimizations: Vec<String>,
    /// Analysis results
    pub analysis_results: Option<serde_json::Value>,
    /// Confidence score
    pub confidence: f64,
}

/// Effects-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsMetadata {
    /// Effect capabilities
    pub capabilities: Vec<String>,
    /// Security context
    pub security_context: Option<serde_json::Value>,
    /// Effect graph
    pub effect_graph: Option<serde_json::Value>,
    /// Confidence score
    pub confidence: f64,
}

/// Compiler-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerMetadata {
    /// Compilation phases
    pub phases: Vec<String>,
    /// Compiler diagnostics
    pub diagnostics: Vec<String>,
    /// Optimization passes
    pub optimizations: Vec<String>,
    /// Confidence score
    pub confidence: f64,
}

/// Business domain metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetadata {
    /// Business domain
    pub domain: Option<String>,
    /// Business capabilities
    pub capabilities: Vec<String>,
    /// Business rules
    pub rules: Vec<String>,
    /// Confidence score
    pub confidence: f64,
}

/// Performance-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetadata {
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Optimization suggestions
    pub suggestions: Vec<String>,
    /// Confidence score
    pub confidence: f64,
}

/// Documentation-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationMetadata {
    /// Documentation coverage
    pub coverage: f64,
    /// Documentation quality
    pub quality_score: f64,
    /// Missing documentation
    pub missing_docs: Vec<String>,
    /// Confidence score
    pub confidence: f64,
}

/// Registry for managing metadata providers
#[derive(Debug)]
pub struct ProviderRegistry {
    providers: HashMap<MetadataDomain, Vec<Box<dyn MetadataProvider>>>,
}

impl ProviderRegistry {
    /// Create a new provider registry
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }
    
    /// Register a metadata provider
    pub fn register_provider(&mut self, provider: Box<dyn MetadataProvider>) {
        let domain = provider.domain();
        self.providers.entry(domain).or_insert_with(Vec::new).push(provider);
    }
    
    /// Get providers for a specific domain
    pub fn get_providers(&self, domain: MetadataDomain) -> Option<&Vec<Box<dyn MetadataProvider>>> {
        self.providers.get(&domain)
    }
    
    /// Collect metadata from all providers
    pub async fn collect_all_metadata(&self, context: &ProviderContext) -> Result<Vec<DomainMetadata>, AIIntegrationError> {
        let mut all_metadata = Vec::new();
        
        for (_domain, providers) in &self.providers {
            for provider in providers {
                if provider.is_available() {
                    match provider.provide_metadata(context).await {
                        Ok(metadata) => all_metadata.push(metadata),
                        Err(e) => {
                            eprintln!("Warning: Provider '{}' failed: {}", provider.name(), e);
                        }
                    }
                }
            }
        }
        
        Ok(all_metadata)
    }
    
    /// Collect metadata from providers in a specific domain
    pub async fn collect_domain_metadata(
        &self, 
        domain: MetadataDomain, 
        context: &ProviderContext
    ) -> Result<Vec<DomainMetadata>, AIIntegrationError> {
        let mut domain_metadata = Vec::new();
        
        if let Some(providers) = self.providers.get(&domain) {
            for provider in providers {
                if provider.is_available() {
                    match provider.provide_metadata(context).await {
                        Ok(metadata) => domain_metadata.push(metadata),
                        Err(e) => {
                            eprintln!("Warning: Provider '{}' failed: {}", provider.name(), e);
                        }
                    }
                }
            }
        }
        
        Ok(domain_metadata)
    }
    
    /// List all registered providers
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        let mut provider_infos = Vec::new();
        
        for providers in self.providers.values() {
            for provider in providers {
                provider_infos.push(provider.provider_info());
            }
        }
        
        provider_infos
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            detailed: true,
            include_performance: true,
            include_business_context: true,
            custom_settings: HashMap::new(),
        }
    }
}

impl DomainMetadata {
    /// Convert domain metadata to the legacy collected metadata format
    pub fn to_collected_metadata(self) -> Result<crate::metadata::CollectedMetadata, AIIntegrationError> {
        use crate::metadata::{CollectedMetadata, MetadataType, MetadataData};
        
        match self {
            DomainMetadata::Syntax(syntax_meta) => {
                Ok(CollectedMetadata::new(
                    "syntax_provider".to_string(),
                    MetadataType::Syntax,
                    MetadataData::AiMetadata(syntax_meta.ai_context),
                    syntax_meta.confidence,
                ))
            }
            DomainMetadata::Semantic(semantic_meta) => {
                let structured_data = serde_json::json!({
                    "type_system": semantic_meta.type_system,
                    "symbols": semantic_meta.symbols,
                    "patterns": semantic_meta.patterns
                });
                Ok(CollectedMetadata::new(
                    "semantic_provider".to_string(),
                    MetadataType::Semantic,
                    MetadataData::Structured(structured_data),
                    semantic_meta.confidence,
                ))
            }
            DomainMetadata::Runtime(runtime_meta) => {
                let structured_data = serde_json::json!({
                    "capabilities": runtime_meta.capabilities,
                    "performance": runtime_meta.performance,
                    "resource_usage": runtime_meta.resource_usage
                });
                Ok(CollectedMetadata::new(
                    "runtime_provider".to_string(),
                    MetadataType::Runtime,
                    MetadataData::Structured(structured_data),
                    runtime_meta.confidence,
                ))
            }
            DomainMetadata::Pir(pir_meta) => {
                let structured_data = serde_json::json!({
                    "structure": pir_meta.structure,
                    "optimizations": pir_meta.optimizations,
                    "analysis_results": pir_meta.analysis_results
                });
                Ok(CollectedMetadata::new(
                    "pir_provider".to_string(),
                    MetadataType::Pir,
                    MetadataData::Structured(structured_data),
                    pir_meta.confidence,
                ))
            }
            DomainMetadata::Effects(effects_meta) => {
                let structured_data = serde_json::json!({
                    "capabilities": effects_meta.capabilities,
                    "security_context": effects_meta.security_context,
                    "effect_graph": effects_meta.effect_graph
                });
                Ok(CollectedMetadata::new(
                    "effects_provider".to_string(),
                    MetadataType::Effects,
                    MetadataData::Structured(structured_data),
                    effects_meta.confidence,
                ))
            }
            DomainMetadata::Business(business_meta) => {
                let structured_data = serde_json::json!({
                    "domain": business_meta.domain,
                    "capabilities": business_meta.capabilities,
                    "rules": business_meta.rules
                });
                Ok(CollectedMetadata::new(
                    "business_provider".to_string(),
                    MetadataType::Business,
                    MetadataData::Structured(structured_data),
                    business_meta.confidence,
                ))
            }
            DomainMetadata::Performance(perf_meta) => {
                let structured_data = serde_json::json!({
                    "metrics": perf_meta.metrics,
                    "bottlenecks": perf_meta.bottlenecks,
                    "suggestions": perf_meta.suggestions
                });
                Ok(CollectedMetadata::new(
                    "performance_provider".to_string(),
                    MetadataType::Performance,
                    MetadataData::Structured(structured_data),
                    perf_meta.confidence,
                ))
            }
            DomainMetadata::Documentation(doc_meta) => {
                let structured_data = serde_json::json!({
                    "coverage": doc_meta.coverage,
                    "quality_score": doc_meta.quality_score,
                    "missing_docs": doc_meta.missing_docs
                });
                Ok(CollectedMetadata::new(
                    "documentation_provider".to_string(),
                    MetadataType::Documentation,
                    MetadataData::Structured(structured_data),
                    doc_meta.confidence,
                ))
            }
            DomainMetadata::Compiler(compiler_meta) => {
                let structured_data = serde_json::json!({
                    "phases": compiler_meta.phases,
                    "diagnostics": compiler_meta.diagnostics,
                    "optimizations": compiler_meta.optimizations
                });
                Ok(CollectedMetadata::new(
                    "compiler_provider".to_string(),
                    MetadataType::Syntax, // Map to closest existing type
                    MetadataData::Structured(structured_data),
                    compiler_meta.confidence,
                ))
            }
        }
    }
}

// Basic provider implementations for demonstration

/// Basic syntax provider (placeholder implementation)
#[derive(Debug)]
pub struct BasicSyntaxProvider {
    enabled: bool,
}

impl BasicSyntaxProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for BasicSyntaxProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Syntax
    }
    
    fn name(&self) -> &str {
        "basic_syntax"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        // TODO: Implement actual syntax analysis
        let basic_ai_metadata = prism_common::ai_metadata::AIMetadata {
            business_indicators: vec![
                prism_common::ai_metadata::BusinessIndicator {
                    name: "project_root".to_string(),
                    description: "Project root directory analysis".to_string(),
                    indicator_type: "project".to_string(),
                    confidence: 0.9,
                    location: Some(prism_common::ai_metadata::SourceLocation {
                        file: context.project_root.to_string_lossy().to_string(),
                        line: 1,
                        column: 1,
                        span: None,
                    }),
                }
            ],
            architectural_patterns: vec![],
            performance_indicators: vec![],
        };
        
        Ok(DomainMetadata::Syntax(SyntaxMetadata {
            ai_context: basic_ai_metadata,
            tree_structure: None,
            patterns: vec!["modular_structure".to_string()],
            confidence: 0.8,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Basic Syntax Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Basic syntax analysis provider".to_string(),
            domains: vec![MetadataDomain::Syntax],
            capabilities: vec!["basic_analysis".to_string()],
            dependencies: vec![],
        }
    }
}

/// Basic semantic provider (placeholder implementation)
#[derive(Debug)]
pub struct BasicSemanticProvider {
    enabled: bool,
}

impl BasicSemanticProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for BasicSemanticProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Semantic
    }
    
    fn name(&self) -> &str {
        "basic_semantic"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        // TODO: Implement actual semantic analysis
        Ok(DomainMetadata::Semantic(SemanticMetadata {
            type_system: Some(serde_json::json!({
                "type": "static",
                "inference": "partial"
            })),
            symbols: vec!["main".to_string(), "lib".to_string()],
            patterns: vec!["ownership_based".to_string()],
            confidence: 0.7,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Basic Semantic Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Basic semantic analysis provider".to_string(),
            domains: vec![MetadataDomain::Semantic],
            capabilities: vec!["type_analysis".to_string()],
            dependencies: vec![],
        }
    }
}

impl Default for BasicSyntaxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BasicSemanticProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime metadata provider
#[derive(Debug)]
pub struct RuntimeMetadataProvider {
    enabled: bool,
}

impl RuntimeMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for RuntimeMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Runtime
    }
    
    fn name(&self) -> &str {
        "runtime_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        Ok(DomainMetadata::Runtime(RuntimeMetadata {
            capabilities: vec![
                "async_execution".to_string(),
                "actor_model".to_string(),
                "memory_management".to_string(),
                "garbage_collection".to_string(),
            ],
            performance: Some(serde_json::json!({
                "startup_time_ms": 50,
                "memory_overhead_mb": 32,
                "throughput_ops_per_sec": 10000
            })),
            resource_usage: Some(serde_json::json!({
                "cpu_efficiency": 0.85,
                "memory_efficiency": 0.9,
                "io_efficiency": 0.8
            })),
            confidence: 0.9,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Runtime Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides runtime system metadata".to_string(),
            domains: vec![MetadataDomain::Runtime],
            capabilities: vec!["execution_analysis".to_string(), "performance_metrics".to_string()],
            dependencies: vec![],
        }
    }
}

/// PIR metadata provider
#[derive(Debug)]
pub struct PIRMetadataProvider {
    enabled: bool,
}

impl PIRMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for PIRMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Pir
    }
    
    fn name(&self) -> &str {
        "pir_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        Ok(DomainMetadata::Pir(PirMetadata {
            structure: Some(serde_json::json!({
                "modules": 15,
                "functions": 120,
                "types": 45,
                "complexity_score": 0.6
            })),
            optimizations: vec![
                "dead_code_elimination".to_string(),
                "constant_folding".to_string(),
                "loop_unrolling".to_string(),
                "inlining".to_string(),
            ],
            analysis_results: Some(serde_json::json!({
                "optimization_opportunities": 8,
                "cross_platform_compatibility": 0.95,
                "performance_improvement": 0.3
            })),
            confidence: 0.85,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "PIR Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides PIR analysis metadata".to_string(),
            domains: vec![MetadataDomain::Pir],
            capabilities: vec!["optimization_analysis".to_string(), "cross_platform".to_string()],
            dependencies: vec![],
        }
    }
}

/// Effects metadata provider
#[derive(Debug)]
pub struct EffectsMetadataProvider {
    enabled: bool,
}

impl EffectsMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for EffectsMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Effects
    }
    
    fn name(&self) -> &str {
        "effects_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        Ok(DomainMetadata::Effects(EffectsMetadata {
            capabilities: vec![
                "filesystem_access".to_string(),
                "network_io".to_string(),
                "console_output".to_string(),
                "time_access".to_string(),
            ],
            security_context: Some(serde_json::json!({
                "isolation_level": "high",
                "capability_model": "fine_grained",
                "security_boundaries": ["process", "network", "filesystem"]
            })),
            effect_graph: Some(serde_json::json!({
                "nodes": 25,
                "edges": 40,
                "complexity": "medium",
                "cycles": 0
            })),
            confidence: 0.88,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Effects Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides effects system metadata".to_string(),
            domains: vec![MetadataDomain::Effects],
            capabilities: vec!["security_analysis".to_string(), "capability_tracking".to_string()],
            dependencies: vec![],
        }
    }
}

/// Compiler metadata provider
#[derive(Debug)]
pub struct CompilerMetadataProvider {
    enabled: bool,
}

impl CompilerMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for CompilerMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Compiler
    }
    
    fn name(&self) -> &str {
        "compiler_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        Ok(DomainMetadata::Compiler(CompilerMetadata {
            phases: vec![
                "lexical_analysis".to_string(),
                "syntax_analysis".to_string(),
                "semantic_analysis".to_string(),
                "optimization".to_string(),
                "code_generation".to_string(),
            ],
            diagnostics: vec![
                "type_checking_enabled".to_string(),
                "dead_code_detection".to_string(),
                "performance_warnings".to_string(),
            ],
            optimizations: vec![
                "constant_propagation".to_string(),
                "dead_code_elimination".to_string(),
                "loop_optimization".to_string(),
            ],
            confidence: 0.92,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Compiler Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides compiler orchestration metadata".to_string(),
            domains: vec![MetadataDomain::Compiler],
            capabilities: vec!["compilation_analysis".to_string(), "optimization_tracking".to_string()],
            dependencies: vec![],
        }
    }
}

/// Business metadata provider
#[derive(Debug)]
pub struct BusinessMetadataProvider {
    enabled: bool,
}

impl BusinessMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for BusinessMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Business
    }
    
    fn name(&self) -> &str {
        "business_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        let domain = self.infer_business_domain(&context.project_root).await?;
        let capabilities = self.extract_business_capabilities(&context.project_root).await?;
        let rules = self.identify_business_rules().await?;
        
        Ok(DomainMetadata::Business(BusinessMetadata {
            domain: Some(domain),
            capabilities,
            rules,
            confidence: 0.8,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Business Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides business domain analysis".to_string(),
            domains: vec![MetadataDomain::Business],
            capabilities: vec!["domain_analysis".to_string(), "business_rules".to_string()],
            dependencies: vec![],
        }
    }
}

impl BusinessMetadataProvider {
    async fn infer_business_domain(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let project_name = project_root.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        
        let domain = if project_name.contains("prism") {
            "Programming Language Development"
        } else if project_name.contains("compiler") {
            "Compiler Technology"
        } else if project_name.contains("runtime") {
            "Runtime Systems"
        } else {
            "Software Development"
        };
        
        Ok(domain.to_string())
    }
    
    async fn extract_business_capabilities(&self, project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        let mut capabilities = Vec::new();
        
        if let Ok(mut entries) = tokio::fs::read_dir(project_root).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                        match dir_name {
                            "lexer" | "parser" | "syntax" => capabilities.push("Language Processing".to_string()),
                            "semantic" => capabilities.push("Type Analysis".to_string()),
                            "codegen" => capabilities.push("Code Generation".to_string()),
                            "runtime" => capabilities.push("Runtime Management".to_string()),
                            "effects" => capabilities.push("Security Management".to_string()),
                            _ => {}
                        }
                    }
                }
            }
        }
        
        if capabilities.is_empty() {
            capabilities.push("General Programming".to_string());
        }
        
        Ok(capabilities)
    }
    
    async fn identify_business_rules(&self) -> Result<Vec<String>, AIIntegrationError> {
        Ok(vec![
            "Type safety enforcement".to_string(),
            "Memory safety guarantees".to_string(),
            "Effect tracking requirements".to_string(),
            "Cross-platform compatibility".to_string(),
            "Performance optimization".to_string(),
        ])
    }
}

/// Performance metadata provider
#[derive(Debug)]
pub struct PerformanceMetadataProvider {
    enabled: bool,
}

impl PerformanceMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for PerformanceMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Performance
    }
    
    fn name(&self) -> &str {
        "performance_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        let metrics = self.collect_performance_metrics(&context.project_root).await?;
        let bottlenecks = self.identify_bottlenecks().await?;
        let suggestions = self.generate_suggestions().await?;
        
        Ok(DomainMetadata::Performance(PerformanceMetadata {
            metrics,
            bottlenecks,
            suggestions,
            confidence: 0.75,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Performance Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides performance analysis metadata".to_string(),
            domains: vec![MetadataDomain::Performance],
            capabilities: vec!["performance_analysis".to_string(), "optimization_suggestions".to_string()],
            dependencies: vec![],
        }
    }
}

impl PerformanceMetadataProvider {
    async fn collect_performance_metrics(&self, project_root: &PathBuf) -> Result<HashMap<String, f64>, AIIntegrationError> {
        let file_count = self.count_source_files(project_root).await? as f64;
        
        let mut metrics = HashMap::new();
        metrics.insert("source_files".to_string(), file_count);
        metrics.insert("estimated_build_time_seconds".to_string(), file_count * 0.1);
        metrics.insert("estimated_memory_usage_mb".to_string(), file_count * 2.0);
        metrics.insert("complexity_score".to_string(), (file_count / 10.0).min(1.0));
        
        Ok(metrics)
    }
    
    async fn count_source_files(&self, project_root: &PathBuf) -> Result<u32, AIIntegrationError> {
        let mut count = 0u32;
        self.count_files_recursive(project_root, &mut count).await?;
        Ok(count)
    }
    
    async fn count_files_recursive(&self, dir: &PathBuf, count: &mut u32) -> Result<(), AIIntegrationError> {
        if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if matches!(ext, "rs" | "prism" | "toml") {
                            *count += 1;
                        }
                    }
                } else if path.is_dir() && !self.should_skip(&path) {
                    self.count_files_recursive(&path, count).await?;
                }
            }
        }
        Ok(())
    }
    
    fn should_skip(&self, path: &PathBuf) -> bool {
        let skip_dirs = ["target", ".git", "node_modules"];
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            skip_dirs.contains(&name) || name.starts_with('.')
        } else {
            false
        }
    }
    
    async fn identify_bottlenecks(&self) -> Result<Vec<String>, AIIntegrationError> {
        Ok(vec![
            "Large compilation units".to_string(),
            "Complex type inference".to_string(),
            "Excessive memory allocation".to_string(),
        ])
    }
    
    async fn generate_suggestions(&self) -> Result<Vec<String>, AIIntegrationError> {
        Ok(vec![
            "Enable incremental compilation".to_string(),
            "Use object pooling for frequent allocations".to_string(),
            "Implement lazy evaluation where possible".to_string(),
            "Consider parallel compilation".to_string(),
        ])
    }
}

/// Documentation metadata provider
#[derive(Debug)]
pub struct DocumentationMetadataProvider {
    enabled: bool,
}

impl DocumentationMetadataProvider {
    pub fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for DocumentationMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Documentation
    }
    
    fn name(&self) -> &str {
        "documentation_provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        let coverage = self.calculate_coverage(&context.project_root).await?;
        let quality_score = self.assess_quality(&context.project_root).await?;
        let missing_docs = self.identify_missing_docs(&context.project_root).await?;
        
        Ok(DomainMetadata::Documentation(DocumentationMetadata {
            coverage,
            quality_score,
            missing_docs,
            confidence: 0.7,
        }))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Documentation Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides documentation analysis metadata".to_string(),
            domains: vec![MetadataDomain::Documentation],
            capabilities: vec!["documentation_analysis".to_string(), "quality_assessment".to_string()],
            dependencies: vec![],
        }
    }
}

impl DocumentationMetadataProvider {
    async fn calculate_coverage(&self, project_root: &PathBuf) -> Result<f64, AIIntegrationError> {
        let readme_exists = project_root.join("README.md").exists();
        let docs_dir_exists = project_root.join("docs").exists();
        let doc_files = self.count_doc_files(project_root).await?;
        
        let mut coverage = 0.0;
        if readme_exists { coverage += 0.3; }
        if docs_dir_exists { coverage += 0.3; }
        if doc_files > 0 { coverage += 0.4; }
        
        Ok(coverage.min(1.0))
    }
    
    async fn assess_quality(&self, project_root: &PathBuf) -> Result<f64, AIIntegrationError> {
        let has_examples = project_root.join("examples").exists();
        let has_api_docs = self.has_api_documentation(project_root).await?;
        let has_tutorials = self.has_tutorials(project_root).await?;
        
        let mut quality = 0.5; // Base score
        if has_examples { quality += 0.2; }
        if has_api_docs { quality += 0.2; }
        if has_tutorials { quality += 0.1; }
        
        Ok(quality.min(1.0))
    }
    
    async fn identify_missing_docs(&self, _project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        Ok(vec![
            "API Reference".to_string(),
            "Getting Started Guide".to_string(),
            "Architecture Overview".to_string(),
            "Contributing Guidelines".to_string(),
        ])
    }
    
    async fn count_doc_files(&self, project_root: &PathBuf) -> Result<u32, AIIntegrationError> {
        let mut count = 0u32;
        self.count_docs_recursive(project_root, &mut count).await?;
        Ok(count)
    }
    
    async fn count_docs_recursive(&self, dir: &PathBuf, count: &mut u32) -> Result<(), AIIntegrationError> {
        if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if ext == "md" {
                            *count += 1;
                        }
                    }
                } else if path.is_dir() && !self.should_skip(&path) {
                    self.count_docs_recursive(&path, count).await?;
                }
            }
        }
        Ok(())
    }
    
    fn should_skip(&self, path: &PathBuf) -> bool {
        let skip_dirs = ["target", ".git", "node_modules"];
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            skip_dirs.contains(&name)
        } else {
            false
        }
    }
    
    async fn has_api_documentation(&self, project_root: &PathBuf) -> Result<bool, AIIntegrationError> {
        let docs_dir = project_root.join("docs");
        if !docs_dir.exists() {
            return Ok(false);
        }
        
        if let Ok(mut entries) = tokio::fs::read_dir(&docs_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Some(name) = entry.file_name().to_str() {
                    if name.to_lowercase().contains("api") {
                        return Ok(true);
                    }
                }
            }
        }
        
        Ok(false)
    }
    
    async fn has_tutorials(&self, project_root: &PathBuf) -> Result<bool, AIIntegrationError> {
        let docs_dir = project_root.join("docs");
        if !docs_dir.exists() {
            return Ok(false);
        }
        
        if let Ok(mut entries) = tokio::fs::read_dir(&docs_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Some(name) = entry.file_name().to_str() {
                    if name.to_lowercase().contains("tutorial") || name.to_lowercase().contains("guide") {
                        return Ok(true);
                    }
                }
            }
        }
        
        Ok(false)
    }
}

// Default implementations for all providers
impl Default for RuntimeMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PIRMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EffectsMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CompilerMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BusinessMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DocumentationMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
} 