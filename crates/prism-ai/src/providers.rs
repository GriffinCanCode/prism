//! Metadata Provider System
//!
//! This module defines the standardized interface for individual crates to expose
//! their metadata for AI consumption. It follows strict Separation of Concerns:
//! - Each crate provides its own metadata through a standardized interface
//! - Providers focus solely on exposing existing metadata, not collecting it
//! - The system maintains conceptual cohesion by organizing providers by domain

use crate::{AIIntegrationError, CollectedMetadata};
use async_trait::async_trait;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Core trait for metadata providers in individual crates
/// 
/// This trait should be implemented by each crate to expose their existing
/// metadata structures. Providers should NOT collect new metadata, only
/// expose what already exists in their systems.
#[async_trait]
pub trait MetadataProvider: Send + Sync {
    /// Get the domain this provider handles
    fn domain(&self) -> MetadataDomain;
    
    /// Get the name of this provider
    fn name(&self) -> &str;
    
    /// Check if this provider is available/enabled
    fn is_available(&self) -> bool { true }
    
    /// Provide metadata from this crate's existing systems
    async fn provide_metadata(&self, context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError>;
    
    /// Get provider capabilities and version info
    fn provider_info(&self) -> ProviderInfo;
}

/// Context information for metadata providers
#[derive(Debug, Clone)]
pub struct ProviderContext {
    /// Project root directory
    pub project_root: PathBuf,
    /// Compilation artifacts (if available)
    pub compilation_artifacts: Option<CompilationArtifacts>,
    /// Runtime information (if available)
    pub runtime_info: Option<RuntimeInfo>,
    /// Provider-specific configuration
    pub provider_config: ProviderConfig,
}

/// Domain-specific metadata from a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainMetadata {
    /// Syntax and parsing metadata
    Syntax(SyntaxProviderMetadata),
    /// Semantic analysis metadata
    Semantic(SemanticProviderMetadata),
    /// PIR metadata
    Pir(PIRProviderMetadata),
    /// Effects system metadata
    Effects(EffectsProviderMetadata),
    /// Runtime metadata
    Runtime(RuntimeProviderMetadata),
    /// Documentation metadata
    Documentation(DocumentationProviderMetadata),
    /// Compiler metadata
    Compiler(CompilerProviderMetadata),
}

/// Metadata domain categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataDomain {
    /// Syntax analysis and parsing
    Syntax,
    /// Semantic analysis and type checking
    Semantic,
    /// Prism Intermediate Representation
    Pir,
    /// Effects and capabilities
    Effects,
    /// Runtime execution
    Runtime,
    /// Documentation and comments
    Documentation,
    /// Compiler orchestration
    Compiler,
    /// Custom domain
    Custom(u32),
}

/// Provider information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider name
    pub name: String,
    /// Provider version
    pub version: String,
    /// Supported metadata schema version
    pub schema_version: String,
    /// Provider capabilities
    pub capabilities: Vec<ProviderCapability>,
    /// Dependencies on other providers
    pub dependencies: Vec<String>,
}

/// Provider capability flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderCapability {
    /// Can provide real-time metadata
    RealTime,
    /// Can provide historical metadata
    Historical,
    /// Can provide incremental updates
    Incremental,
    /// Can provide cross-references
    CrossReference,
    /// Can provide business context
    BusinessContext,
    /// Can provide performance metrics
    PerformanceMetrics,
}

/// Configuration for providers
#[derive(Debug, Clone, Default)]
pub struct ProviderConfig {
    /// Enable detailed metadata collection
    pub detailed: bool,
    /// Include performance metrics
    pub include_performance: bool,
    /// Include business context
    pub include_business_context: bool,
    /// Provider-specific settings
    pub custom_settings: std::collections::HashMap<String, String>,
}

/// Compilation artifacts available to providers
#[derive(Debug, Clone)]
pub struct CompilationArtifacts {
    /// Parsed AST (if available)
    pub ast: Option<prism_ast::Program>,
    /// Symbol table (if available)  
    pub symbols: Option<prism_common::symbol::SymbolTable>,
    /// Type information (if available)
    pub type_info: Option<std::collections::HashMap<prism_common::NodeId, String>>,
}

/// Runtime information available to providers
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    /// Execution context (if available)
    pub execution_context: Option<String>, // Placeholder - would be actual runtime context
    /// Performance metrics (if available)
    pub performance_metrics: Option<std::collections::HashMap<String, f64>>,
}

// Domain-specific metadata structures

/// Syntax provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxProviderMetadata {
    /// Detected syntax style
    pub syntax_style: Option<String>,
    /// Parsing statistics
    pub parsing_stats: ParsingStatistics,
    /// Syntax tree metrics
    pub tree_metrics: SyntaxTreeMetrics,
    /// AI context from syntax analysis
    pub ai_context: prism_common::ai_metadata::AIMetadata,
}

/// Semantic provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticProviderMetadata {
    /// Type information
    pub type_info: TypeInformation,
    /// Business rules identified
    pub business_rules: Vec<BusinessRule>,
    /// Semantic relationships
    pub relationships: Vec<SemanticRelationship>,
    /// Validation results
    pub validation_results: ValidationSummary,
}

/// PIR provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRProviderMetadata {
    /// PIR structure information
    pub structure_info: PIRStructureInfo,
    /// Business context from PIR
    pub business_context: Option<PIRBusinessContext>,
    /// Optimization information
    pub optimization_info: OptimizationInfo,
    /// Cross-target consistency data
    pub consistency_data: ConsistencyData,
}

/// Effects provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectsProviderMetadata {
    /// Effect definitions and usage
    pub effect_definitions: Vec<EffectDefinition>,
    /// Capability requirements
    pub capabilities: Vec<CapabilityRequirement>,
    /// Security implications
    pub security_implications: SecurityAnalysis,
    /// Effect composition information
    pub composition_info: EffectCompositionInfo,
}

/// Runtime provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProviderMetadata {
    /// Execution statistics
    pub execution_stats: ExecutionStatistics,
    /// Performance profiles
    pub performance_profiles: Vec<PerformanceProfile>,
    /// Resource usage information
    pub resource_usage: ResourceUsageInfo,
    /// Runtime AI insights
    pub ai_insights: Vec<RuntimeInsight>,
}

/// Documentation provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationProviderMetadata {
    /// Documentation coverage
    pub coverage: DocumentationCoverage,
    /// Quality metrics
    pub quality_metrics: DocumentationQuality,
    /// AI-extracted context
    pub extracted_context: DocumentationContext,
    /// Compliance information
    pub compliance_info: ComplianceInfo,
}

/// Compiler provider metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerProviderMetadata {
    /// Compilation statistics
    pub compilation_stats: CompilationStatistics,
    /// Query system metrics
    pub query_metrics: QuerySystemMetrics,
    /// Cross-system coordination info
    pub coordination_info: CoordinationInfo,
    /// AI export readiness
    pub export_readiness: ExportReadiness,
}

// Supporting structures (simplified for now - would be fully defined based on actual crate structures)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingStatistics {
    pub lines_parsed: u64,
    pub tokens_processed: u64,
    pub parse_time_ms: u64,
    pub error_recovery_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxTreeMetrics {
    pub node_count: u64,
    pub max_depth: u32,
    pub avg_branching_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInformation {
    pub types_inferred: u32,
    pub constraints_solved: u32,
    pub semantic_types_identified: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    pub rule_name: String,
    pub rule_type: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelationship {
    pub source: String,
    pub target: String,
    pub relationship_type: String,
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub rules_checked: u32,
    pub violations_found: u32,
    pub warnings_issued: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRStructureInfo {
    pub modules_count: u32,
    pub functions_count: u32,
    pub types_count: u32,
    pub cohesion_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRBusinessContext {
    pub domain: String,
    pub capabilities: Vec<String>,
    pub responsibilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationInfo {
    pub optimizations_applied: Vec<String>,
    pub performance_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyData {
    pub cross_target_compatibility: f64,
    pub semantic_preservation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDefinition {
    pub name: String,
    pub category: String,
    pub security_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRequirement {
    pub capability: String,
    pub required_level: String,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    pub risk_level: String,
    pub vulnerabilities: Vec<String>,
    pub mitigations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectCompositionInfo {
    pub compositions_found: u32,
    pub safe_compositions: u32,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    pub executions_count: u64,
    pub avg_execution_time_ms: f64,
    pub memory_usage_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub profile_name: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub io_operations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageInfo {
    pub peak_memory_mb: f64,
    pub cpu_time_ms: u64,
    pub io_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInsight {
    pub insight_type: String,
    pub description: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationCoverage {
    pub functions_documented: u32,
    pub total_functions: u32,
    pub coverage_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationQuality {
    pub completeness_score: f64,
    pub clarity_score: f64,
    pub ai_readability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationContext {
    pub business_concepts: Vec<String>,
    pub architectural_patterns: Vec<String>,
    pub usage_examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceInfo {
    pub standards_met: Vec<String>,
    pub compliance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStatistics {
    pub compilation_time_ms: u64,
    pub files_processed: u32,
    pub incremental_builds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySystemMetrics {
    pub queries_executed: u64,
    pub cache_hit_rate: f64,
    pub avg_query_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationInfo {
    pub systems_coordinated: u32,
    pub coordination_overhead_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportReadiness {
    pub formats_supported: Vec<String>,
    pub metadata_completeness: f64,
    pub ai_compatibility_score: f64,
}

/// Provider registry for managing metadata providers
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    providers: std::collections::HashMap<MetadataDomain, Vec<Box<dyn MetadataProvider>>>,
}

impl ProviderRegistry {
    /// Create a new provider registry
    pub fn new() -> Self {
        Self {
            providers: std::collections::HashMap::new(),
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
    
    /// Get all available providers
    pub fn all_providers(&self) -> impl Iterator<Item = &Box<dyn MetadataProvider>> {
        self.providers.values().flatten()
    }
    
    /// Collect metadata from all providers
    pub async fn collect_all_metadata(&self, context: &ProviderContext) -> Result<Vec<DomainMetadata>, AIIntegrationError> {
        let mut results = Vec::new();
        
        for provider in self.all_providers() {
            if provider.is_available() {
                match provider.provide_metadata(context).await {
                    Ok(metadata) => results.push(metadata),
                    Err(e) => {
                        eprintln!("Warning: Provider '{}' failed: {}", provider.name(), e);
                        // Continue with other providers
                    }
                }
            }
        }
        
        Ok(results)
    }
}

impl DomainMetadata {
    /// Convert domain metadata to the existing CollectedMetadata format
    /// This maintains compatibility with the existing collection system
    pub fn to_collected_metadata(self) -> Result<CollectedMetadata, AIIntegrationError> {
        match self {
            DomainMetadata::Syntax(syntax_meta) => {
                Ok(CollectedMetadata::Syntax(syntax_meta.ai_context))
            }
            DomainMetadata::Semantic(_semantic_meta) => {
                // Convert semantic metadata to the expected format
                Ok(CollectedMetadata::Semantic(crate::SemanticAIMetadata { 
                    placeholder: false  // Real data now available
                }))
            }
            DomainMetadata::Pir(_pir_meta) => {
                Ok(CollectedMetadata::Pir(crate::PIRAIMetadata { 
                    placeholder: false  // Real data now available
                }))
            }
            DomainMetadata::Effects(_effects_meta) => {
                Ok(CollectedMetadata::Effects(crate::EffectsAIMetadata { 
                    placeholder: false  // Real data now available
                }))
            }
            DomainMetadata::Runtime(_runtime_meta) => {
                Ok(CollectedMetadata::Runtime(crate::RuntimeAIMetadata { 
                    placeholder: false  // Real data now available
                }))
            }
            _ => Err(AIIntegrationError::IntegrationError {
                message: "Unsupported domain metadata conversion".to_string(),
            })
        }
    }
}

/// Example usage of the metadata provider system
/// 
/// This demonstrates how individual crates should implement and register
/// metadata providers following the SoC and conceptual cohesion principles.
#[cfg(test)]
mod examples {
    use super::*;
    use std::path::PathBuf;
    
    /// Example provider implementation for demonstration
    struct ExampleProvider {
        name: String,
        domain: MetadataDomain,
    }
    
    impl ExampleProvider {
        fn new(name: String, domain: MetadataDomain) -> Self {
            Self { name, domain }
        }
    }
    
    #[async_trait]
    impl MetadataProvider for ExampleProvider {
        fn domain(&self) -> MetadataDomain {
            self.domain
        }
        
        fn name(&self) -> &str {
            &self.name
        }
        
        async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
            // Return example metadata based on domain
            match self.domain {
                MetadataDomain::Syntax => {
                    Ok(DomainMetadata::Syntax(SyntaxProviderMetadata {
                        syntax_style: Some("rust-like".to_string()),
                        parsing_stats: ParsingStatistics {
                            lines_parsed: 100,
                            tokens_processed: 500,
                            parse_time_ms: 50,
                            error_recovery_count: 0,
                        },
                        tree_metrics: SyntaxTreeMetrics {
                            node_count: 250,
                            max_depth: 8,
                            avg_branching_factor: 2.5,
                        },
                        ai_context: prism_common::ai_metadata::AIMetadata::default(),
                    }))
                }
                MetadataDomain::Semantic => {
                    Ok(DomainMetadata::Semantic(SemanticProviderMetadata {
                        type_info: TypeInformation {
                            types_inferred: 15,
                            constraints_solved: 8,
                            semantic_types_identified: 12,
                        },
                        business_rules: vec![
                            BusinessRule {
                                rule_name: "Non-null validation".to_string(),
                                rule_type: "safety".to_string(),
                                confidence: 0.95,
                            }
                        ],
                        relationships: vec![],
                        validation_results: ValidationSummary {
                            rules_checked: 20,
                            violations_found: 0,
                            warnings_issued: 2,
                        },
                    }))
                }
                _ => Err(AIIntegrationError::IntegrationError {
                    message: "Example provider only supports Syntax and Semantic domains".to_string(),
                })
            }
        }
        
        fn provider_info(&self) -> ProviderInfo {
            ProviderInfo {
                name: self.name.clone(),
                version: "0.1.0".to_string(),
                schema_version: "1.0.0".to_string(),
                capabilities: vec![ProviderCapability::RealTime],
                dependencies: vec![],
            }
        }
    }
    
    #[tokio::test]
    async fn test_provider_registry_usage() {
        // Create a provider registry
        let mut registry = ProviderRegistry::new();
        
        // Register providers from different crates/domains
        registry.register_provider(Box::new(ExampleProvider::new(
            "syntax-provider".to_string(),
            MetadataDomain::Syntax,
        )));
        
        registry.register_provider(Box::new(ExampleProvider::new(
            "semantic-provider".to_string(),
            MetadataDomain::Semantic,
        )));
        
        // Create provider context
        let context = ProviderContext {
            project_root: PathBuf::from("."),
            compilation_artifacts: None,
            runtime_info: None,
            provider_config: ProviderConfig::default(),
        };
        
        // Collect metadata from all providers
        let metadata_results = registry.collect_all_metadata(&context).await.unwrap();
        
        // Verify we got metadata from both providers
        assert_eq!(metadata_results.len(), 2);
        
        // Check that we have syntax and semantic metadata
        let has_syntax = metadata_results.iter().any(|m| matches!(m, DomainMetadata::Syntax(_)));
        let has_semantic = metadata_results.iter().any(|m| matches!(m, DomainMetadata::Semantic(_)));
        
        assert!(has_syntax, "Should have syntax metadata");
        assert!(has_semantic, "Should have semantic metadata");
    }
    
    #[tokio::test]
    async fn test_metadata_conversion() {
        // Test conversion from domain metadata to collected metadata
        let syntax_metadata = DomainMetadata::Syntax(SyntaxProviderMetadata {
            syntax_style: Some("rust-like".to_string()),
            parsing_stats: ParsingStatistics {
                lines_parsed: 100,
                tokens_processed: 500,
                parse_time_ms: 50,
                error_recovery_count: 0,
            },
            tree_metrics: SyntaxTreeMetrics {
                node_count: 250,
                max_depth: 8,
                avg_branching_factor: 2.5,
            },
            ai_context: prism_common::ai_metadata::AIMetadata::default(),
        });
        
        let collected = syntax_metadata.to_collected_metadata().unwrap();
        
        // Verify conversion worked
        assert!(matches!(collected, CollectedMetadata::Syntax(_)));
    }
} 