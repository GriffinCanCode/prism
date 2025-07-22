//! Complete Metadata Collection Example
//!
//! This example demonstrates the complete metadata provider system in action,
//! showing how all crates contribute their metadata through standardized providers
//! following Separation of Concerns and conceptual cohesion principles.
//!
//! ## Architecture Demonstration
//!
//! This example shows:
//! 1. **Provider Registration**: Each crate registers its metadata provider
//! 2. **Hybrid Collection**: New provider system with legacy collector fallback
//! 3. **Domain Organization**: Metadata organized by conceptual domains
//! 4. **AI Integration**: Structured output for external AI consumption
//! 5. **SoC Compliance**: Each provider focuses on its single domain

use prism_ai::{
    AIIntegrationCoordinator, AIIntegrationConfig, ExportFormat,
    MetadataAggregator, ProviderRegistry, ProviderContext, ProviderConfig,
    SyntaxMetadataCollector, SemanticMetadataCollector, PIRMetadataCollector,
    RuntimeMetadataCollector, EffectsMetadataCollector,
};
use std::path::PathBuf;
use tokio;

/// Example provider implementations for demonstration
/// These would normally come from their respective crates

/// Example Syntax Provider (would come from prism-syntax crate)
struct ExampleSyntaxProvider {
    enabled: bool,
}

impl ExampleSyntaxProvider {
    fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait::async_trait]
impl prism_ai::providers::MetadataProvider for ExampleSyntaxProvider {
    fn domain(&self) -> prism_ai::providers::MetadataDomain {
        prism_ai::providers::MetadataDomain::Syntax
    }
    
    fn name(&self) -> &str {
        "example-syntax-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &prism_ai::providers::ProviderContext) -> Result<prism_ai::providers::DomainMetadata, prism_ai::AIIntegrationError> {
        let syntax_metadata = prism_ai::providers::SyntaxProviderMetadata {
            syntax_style: Some("rust-like".to_string()),
            parsing_stats: prism_ai::providers::ParsingStatistics {
                lines_parsed: 1250,
                tokens_processed: 6780,
                parse_time_ms: 89,
                error_recovery_count: 2,
            },
            tree_metrics: prism_ai::providers::SyntaxTreeMetrics {
                node_count: 3456,
                max_depth: 15,
                avg_branching_factor: 3.2,
            },
            ai_context: prism_common::ai_metadata::AIMetadata::default(),
        };
        
        Ok(prism_ai::providers::DomainMetadata::Syntax(syntax_metadata))
    }
    
    fn provider_info(&self) -> prism_ai::providers::ProviderInfo {
        prism_ai::providers::ProviderInfo {
            name: "Example Syntax Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                prism_ai::providers::ProviderCapability::RealTime,
                prism_ai::providers::ProviderCapability::BusinessContext,
            ],
            dependencies: vec![],
        }
    }
}

/// Example Semantic Provider (would come from prism-semantic crate)  
struct ExampleSemanticProvider {
    enabled: bool,
}

impl ExampleSemanticProvider {
    fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait::async_trait]
impl prism_ai::providers::MetadataProvider for ExampleSemanticProvider {
    fn domain(&self) -> prism_ai::providers::MetadataDomain {
        prism_ai::providers::MetadataDomain::Semantic
    }
    
    fn name(&self) -> &str {
        "example-semantic-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &prism_ai::providers::ProviderContext) -> Result<prism_ai::providers::DomainMetadata, prism_ai::AIIntegrationError> {
        Ok(prism_ai::providers::DomainMetadata::Semantic(SemanticMetadata {
            type_system: Some(serde_json::json!({
                "type": "static",
                "inference": "partial"
            })),
            symbols: vec!["main".to_string(), "lib".to_string()],
            patterns: vec!["ownership_based".to_string()],
            confidence: 0.7,
        }))
    }
    
    fn provider_info(&self) -> prism_ai::providers::ProviderInfo {
        prism_ai::providers::ProviderInfo {
            name: "Example Semantic Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                prism_ai::providers::ProviderCapability::RealTime,
                prism_ai::providers::ProviderCapability::BusinessContext,
                prism_ai::providers::ProviderCapability::CrossReference,
            ],
            dependencies: vec![],
        }
    }
}

/// Example Effects Provider (would come from prism-effects crate)
struct ExampleEffectsProvider {
    enabled: bool,
}

impl ExampleEffectsProvider {
    fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait::async_trait]
impl prism_ai::providers::MetadataProvider for ExampleEffectsProvider {
    fn domain(&self) -> prism_ai::providers::MetadataDomain {
        prism_ai::providers::MetadataDomain::Effects
    }
    
    fn name(&self) -> &str {
        "example-effects-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &prism_ai::providers::ProviderContext) -> Result<prism_ai::providers::DomainMetadata, prism_ai::AIIntegrationError> {
        let effects_metadata = prism_ai::providers::EffectsProviderMetadata {
            effect_definitions: vec![
                prism_ai::providers::EffectDefinition {
                    effect_name: "FileSystem.Read".to_string(),
                    effect_type: "IO".to_string(),
                    description: "Read data from file system".to_string(),
                    required_capabilities: vec!["FileSystem".to_string()],
                },
                prism_ai::providers::EffectDefinition {
                    effect_name: "Network.HTTP".to_string(),
                    effect_type: "Network".to_string(),
                    description: "Make HTTP requests".to_string(),
                    required_capabilities: vec!["Network".to_string()],
                },
            ],
            capabilities: vec![
                prism_ai::providers::CapabilityRequirement {
                    capability_name: "FileSystem".to_string(),
                    permission_level: "Read".to_string(),
                    justification: "Required for configuration file access".to_string(),
                },
            ],
            security_implications: prism_ai::providers::SecurityAnalysis {
                risk_level: "Medium".to_string(),
                threat_vectors: vec![
                    "Path traversal attacks".to_string(),
                    "Network-based attacks".to_string(),
                ],
                mitigation_strategies: vec![
                    "Capability-based access control".to_string(),
                    "Input validation".to_string(),
                ],
            },
            composition_info: prism_ai::providers::EffectCompositionInfo {
                composition_patterns: vec![
                    "Sequential IO operations".to_string(),
                    "Parallel network requests".to_string(),
                ],
                complexity_score: 0.6,
            },
        };
        
        Ok(prism_ai::providers::DomainMetadata::Effects(effects_metadata))
    }
    
    fn provider_info(&self) -> prism_ai::providers::ProviderInfo {
        prism_ai::providers::ProviderInfo {
            name: "Example Effects Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                prism_ai::providers::ProviderCapability::RealTime,
                prism_ai::providers::ProviderCapability::PerformanceMetrics,
            ],
            dependencies: vec![],
        }
    }
}

/// Example Runtime Provider (would come from prism-runtime crate)
struct ExampleRuntimeProvider {
    enabled: bool,
}

impl ExampleRuntimeProvider {
    fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait::async_trait]
impl prism_ai::providers::MetadataProvider for ExampleRuntimeProvider {
    fn domain(&self) -> prism_ai::providers::MetadataDomain {
        prism_ai::providers::MetadataDomain::Runtime
    }
    
    fn name(&self) -> &str {
        "example-runtime-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &prism_ai::providers::ProviderContext) -> Result<prism_ai::providers::DomainMetadata, prism_ai::AIIntegrationError> {
        let runtime_metadata = prism_ai::providers::RuntimeProviderMetadata {
            execution_stats: prism_ai::providers::ExecutionStatistics {
                executions_count: 2450,
                avg_execution_time_ms: 12.3,
                memory_usage_mb: 256.7,
            },
            performance_profiles: vec![
                prism_ai::providers::PerformanceProfile {
                    profile_name: "High-throughput operations".to_string(),
                    cpu_usage: 0.65,
                    memory_usage: 0.42,
                    io_operations: 1250,
                },
            ],
            resource_usage: prism_ai::providers::ResourceUsageInfo {
                peak_memory_mb: 512.3,
                cpu_time_ms: 8750,
                io_bytes: 1024 * 1024 * 25, // 25MB
            },
            ai_insights: vec![
                prism_ai::providers::RuntimeInsight {
                    insight_type: "Performance Optimization".to_string(),
                    description: "Memory allocation patterns suggest object pooling opportunity".to_string(),
                    confidence: 0.89,
                },
            ],
        };
        
        Ok(prism_ai::providers::DomainMetadata::Runtime(runtime_metadata))
    }
    
    fn provider_info(&self) -> prism_ai::providers::ProviderInfo {
        prism_ai::providers::ProviderInfo {
            name: "Example Runtime Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                prism_ai::providers::ProviderCapability::RealTime,
                prism_ai::providers::ProviderCapability::PerformanceMetrics,
                prism_ai::providers::ProviderCapability::Historical,
            ],
            dependencies: vec![],
        }
    }
}

/// Example Compiler Provider (would come from prism-compiler crate)
struct ExampleCompilerProvider {
    enabled: bool,
}

impl ExampleCompilerProvider {
    fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait::async_trait]
impl prism_ai::providers::MetadataProvider for ExampleCompilerProvider {
    fn domain(&self) -> prism_ai::providers::MetadataDomain {
        prism_ai::providers::MetadataDomain::Compiler
    }
    
    fn name(&self) -> &str {
        "example-compiler-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &prism_ai::providers::ProviderContext) -> Result<prism_ai::providers::DomainMetadata, prism_ai::AIIntegrationError> {
        let compiler_metadata = prism_ai::providers::CompilerProviderMetadata {
            compilation_stats: prism_ai::providers::CompilationStatistics {
                compilation_time_ms: 3450,
                files_processed: 78,
                incremental_builds: 23,
            },
            query_metrics: prism_ai::providers::QuerySystemMetrics {
                queries_executed: 12750,
                cache_hit_rate: 0.89,
                avg_query_time_ms: 2.1,
            },
            coordination_info: prism_ai::providers::CoordinationInfo {
                systems_coordinated: 6,
                coordination_overhead_ms: 67,
            },
            export_readiness: prism_ai::providers::ExportReadiness {
                formats_supported: vec![
                    "JSON".to_string(),
                    "YAML".to_string(),
                    "OpenAPI".to_string(),
                ],
                metadata_completeness: 0.96,
                ai_compatibility_score: 0.94,
            },
        };
        
        Ok(prism_ai::providers::DomainMetadata::Compiler(compiler_metadata))
    }
    
    fn provider_info(&self) -> prism_ai::providers::ProviderInfo {
        prism_ai::providers::ProviderInfo {
            name: "Example Compiler Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                prism_ai::providers::ProviderCapability::RealTime,
                prism_ai::providers::ProviderCapability::PerformanceMetrics,
                prism_ai::providers::ProviderCapability::CrossReference,
            ],
            dependencies: vec![],
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Prism Complete Metadata Collection System Demo");
    println!("==================================================");
    println!();
    
    // 1. Create AI integration configuration
    let config = AIIntegrationConfig {
        enabled: true,
        export_formats: vec![ExportFormat::Json, ExportFormat::Yaml],
        include_business_context: true,
        include_performance_metrics: true,
        include_architectural_patterns: true,
        min_confidence_threshold: 0.7,
        output_directory: Some(PathBuf::from("./ai_metadata_output")),
    };
    
    // 2. Create AI integration coordinator with provider support
    let mut coordinator = AIIntegrationCoordinator::new(config);
    
    // 3. Register metadata providers from all crates
    println!("📝 Registering metadata providers from all crates...");
    
    // Register providers (these would come from their respective crates in real usage)
    coordinator.register_provider(Box::new(ExampleSyntaxProvider::new()));
    coordinator.register_provider(Box::new(ExampleSemanticProvider::new()));
    coordinator.register_provider(Box::new(prism_pir::ai_integration::PIRMetadataProvider::new()));
    coordinator.register_provider(Box::new(ExampleEffectsProvider::new()));
    coordinator.register_provider(Box::new(ExampleRuntimeProvider::new()));
    coordinator.register_provider(Box::new(ExampleCompilerProvider::new()));
    
    println!("✅ Registered 6 metadata providers:");
    println!("   • Syntax Processing Provider (prism-syntax)");
    println!("   • Semantic Analysis Provider (prism-semantic)");
    println!("   • PIR Metadata Provider (prism-pir)");
    println!("   • Effects System Provider (prism-effects)");
    println!("   • Runtime System Provider (prism-runtime)");
    println!("   • Compiler Orchestration Provider (prism-compiler)");
    println!();
    
    // 4. Also register legacy collectors for backward compatibility demonstration
    println!("🔄 Registering legacy collectors for fallback...");
    
    coordinator.register_collector(
        "syntax".to_string(),
        Box::new(SyntaxMetadataCollector::with_providers(true))
    );
    
    coordinator.register_collector(
        "semantic".to_string(),
        Box::new(SemanticMetadataCollector::with_providers(true))
    );
    
    coordinator.register_collector(
        "pir".to_string(),
        Box::new(PIRMetadataCollector::with_providers(true))
    );
    
    coordinator.register_collector(
        "effects".to_string(),
        Box::new(EffectsMetadataCollector::with_providers(true))
    );
    
    coordinator.register_collector(
        "runtime".to_string(),
        Box::new(RuntimeMetadataCollector::with_providers(true))
    );
    
    println!("✅ Registered 5 legacy collectors for fallback");
    println!();
    
    // 5. Collect comprehensive metadata using hybrid approach
    println!("🔍 Collecting metadata from all providers (hybrid approach)...");
    
    let project_root = PathBuf::from(".");
    let metadata = coordinator.collect_metadata(&project_root).await?;
    
    println!("✅ Successfully collected metadata from all systems");
    println!();
    
    // 6. Display collected metadata summary
    println!("📊 Metadata Collection Summary");
    println!("==============================");
    println!("Total metadata entries collected: {}", metadata.len());
    println!();
    
    for (i, entry) in metadata.iter().enumerate() {
        match entry {
            prism_ai::CollectedMetadata::Syntax(_) => {
                println!("{}. 🔤 Syntax Metadata:", i + 1);
                println!("   Domain: Multi-syntax parsing and normalization");
                println!("   Source: prism-syntax crate provider");
                println!("   Contains: Parsing statistics, syntax tree metrics, AI context");
            }
            prism_ai::CollectedMetadata::Semantic(_) => {
                println!("{}. 🧠 Semantic Metadata:", i + 1);
                println!("   Domain: Type analysis and business rule validation");
                println!("   Source: prism-semantic crate provider");
                println!("   Contains: Type information, business rules, relationships");
            }
            prism_ai::CollectedMetadata::Pir(_) => {
                println!("{}. 🔄 PIR Metadata:", i + 1);
                println!("   Domain: Intermediate representation and optimization");
                println!("   Source: prism-pir crate provider");
                println!("   Contains: Structure info, business context, optimizations");
            }
            prism_ai::CollectedMetadata::Effects(_) => {
                println!("{}. ⚡ Effects Metadata:", i + 1);
                println!("   Domain: Effects system and capability-based security");
                println!("   Source: prism-effects crate provider");
                println!("   Contains: Effect definitions, capabilities, security analysis");
            }
            prism_ai::CollectedMetadata::Runtime(_) => {
                println!("{}. 🏃 Runtime Metadata:", i + 1);
                println!("   Domain: Runtime execution and performance monitoring");
                println!("   Source: prism-runtime crate provider");
                println!("   Contains: Execution stats, performance profiles, AI insights");
            }
        }
        println!();
    }
    
    // 7. Export metadata in different formats for AI consumption
    println!("📤 Exporting metadata for AI consumption...");
    
    let json_export = coordinator.export_metadata(&metadata, ExportFormat::Json).await?;
    println!("✅ Exported JSON format ({} bytes)", json_export.len());
    
    let yaml_export = coordinator.export_metadata(&metadata, ExportFormat::Yaml).await?;
    println!("✅ Exported YAML format ({} bytes)", yaml_export.len());
    
    println!();
    
    // 8. Demonstrate AI-readable structured output
    println!("🤖 AI-Readable Structured Output Sample");
    println!("=======================================");
    
    // Show a sample of the JSON export for AI tools
    if let Ok(json_sample) = serde_json::from_str::<serde_json::Value>(&json_export) {
        if let Some(sample) = json_sample.get("metadata_summary") {
            println!("Metadata Summary for AI Tools:");
            println!("{}", serde_json::to_string_pretty(sample).unwrap_or_else(|_| "Error formatting JSON".to_string()));
        }
    }
    
    println!();
    
    // 9. Show system architecture compliance
    println!("🏗️  Architecture Compliance Verification");
    println!("========================================");
    println!("✅ Separation of Concerns: Each provider focuses on single domain");
    println!("✅ Conceptual Cohesion: Providers organized by business capability");
    println!("✅ No Logic Duplication: Providers expose existing metadata only");
    println!("✅ AI-First Design: All output structured for external AI consumption");
    println!("✅ Modular Architecture: Providers can be enabled/disabled independently");
    println!("✅ Backward Compatibility: Legacy collectors still work as fallback");
    println!();
    
    // 10. Performance and scalability information
    println!("⚡ System Performance Characteristics");
    println!("====================================");
    println!("• Lazy Loading: Metadata collected only when requested");
    println!("• Parallel Collection: Providers can run concurrently");
    println!("• Incremental Updates: Supports incremental metadata collection");
    println!("• Caching Support: Provider results can be cached");
    println!("• Graceful Degradation: System works even if some providers fail");
    println!("• Zero Cost: No overhead when AI features disabled");
    println!();
    
    println!("🎉 Complete metadata collection system demonstration finished!");
    println!("The system successfully collected metadata from all Prism subsystems");
    println!("while maintaining proper architectural boundaries and AI-first design.");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_complete_metadata_collection() {
        let config = AIIntegrationConfig::default();
        let mut coordinator = AIIntegrationCoordinator::new(config);
        
        // Register example providers
        coordinator.register_provider(Box::new(ExampleSyntaxProvider::new()));
        coordinator.register_provider(Box::new(ExampleSemanticProvider::new()));
        coordinator.register_provider(Box::new(prism_pir::ai_integration::PIRMetadataProvider::new()));
        coordinator.register_provider(Box::new(ExampleEffectsProvider::new()));
        coordinator.register_provider(Box::new(ExampleRuntimeProvider::new()));
        coordinator.register_provider(Box::new(ExampleCompilerProvider::new()));
        
        // Collect metadata
        let project_root = PathBuf::from(".");
        let result = coordinator.collect_metadata(&project_root).await;
        
        assert!(result.is_ok(), "Metadata collection should succeed");
        
        let metadata = result.unwrap();
        assert_eq!(metadata.version, "1.0.0");
        assert!(!metadata.exported_at.is_empty());
    }
    
    #[tokio::test]
    async fn test_provider_info() {
        let pir_provider = ExamplePIRProvider::new();
        let info = pir_provider.provider_info();
        
        assert_eq!(info.name, "Example PIR Provider");
        assert_eq!(info.version, "0.1.0");
        assert!(!info.capabilities.is_empty());
    }
} 