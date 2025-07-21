//! Complete Metadata Collection Example
//!
//! This example demonstrates how to use the new metadata provider system
//! to collect comprehensive AI metadata from multiple Prism crates while
//! following Separation of Concerns and Conceptual Cohesion principles.

use prism_ai::{
    AIIntegrationCoordinator, AIIntegrationConfig, ExportFormat,
    providers::{MetadataProvider, MetadataDomain, ProviderContext, ProviderConfig, DomainMetadata, ProviderInfo, ProviderCapability},
    metadata::{MetadataAggregator, SyntaxMetadataCollector, SemanticMetadataCollector, PIRMetadataCollector},
    AIIntegrationError,
};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio;

/// Example PIR provider (similar to what would be in prism-pir crate)
#[derive(Debug)]
struct ExamplePIRProvider {
    enabled: bool,
}

impl ExamplePIRProvider {
    fn new() -> Self {
        Self { enabled: true }
    }
}

#[async_trait]
impl MetadataProvider for ExamplePIRProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Pir
    }
    
    fn name(&self) -> &str {
        "example-pir-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        use prism_ai::providers::*;
        
        let pir_metadata = PIRProviderMetadata {
            structure_info: PIRStructureInfo {
                modules_count: 3,
                functions_count: 15,
                types_count: 8,
                cohesion_score: 0.85,
            },
            business_context: Some(PIRBusinessContext {
                domain: "Financial Services".to_string(),
                capabilities: vec![
                    "Payment Processing".to_string(),
                    "Risk Assessment".to_string(),
                ],
                responsibilities: vec![
                    "Transaction Validation".to_string(),
                    "Compliance Checking".to_string(),
                ],
            }),
            optimization_info: OptimizationInfo {
                optimizations_applied: vec![
                    "Dead Code Elimination".to_string(),
                    "Constant Folding".to_string(),
                    "Inline Expansion".to_string(),
                ],
                performance_improvement: 0.23, // 23% improvement
            },
            consistency_data: ConsistencyData {
                cross_target_compatibility: 0.97,
                semantic_preservation_score: 0.99,
            },
        };
        
        Ok(DomainMetadata::Pir(pir_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Example PIR Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                ProviderCapability::RealTime,
                ProviderCapability::BusinessContext,
                ProviderCapability::PerformanceMetrics,
                ProviderCapability::CrossReference,
            ],
            dependencies: vec![],
        }
    }
}

/// Example Effects provider (similar to what would be in prism-effects crate)
#[derive(Debug)]
struct ExampleEffectsProvider;

#[async_trait]
impl MetadataProvider for ExampleEffectsProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Effects
    }
    
    fn name(&self) -> &str {
        "example-effects-provider"
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        use prism_ai::providers::*;
        
        let effects_metadata = EffectsProviderMetadata {
            effect_definitions: vec![
                EffectDefinition {
                    name: "Network.HTTP".to_string(),
                    category: "IO".to_string(),
                    security_level: "Medium".to_string(),
                },
                EffectDefinition {
                    name: "Database.Query".to_string(),
                    category: "IO".to_string(),
                    security_level: "High".to_string(),
                },
                EffectDefinition {
                    name: "Cryptography.Hash".to_string(),
                    category: "Crypto".to_string(),
                    security_level: "High".to_string(),
                },
            ],
            capabilities: vec![
                CapabilityRequirement {
                    capability: "Network.Connect".to_string(),
                    required_level: "Trusted".to_string(),
                    justification: "Required for payment processing".to_string(),
                },
                CapabilityRequirement {
                    capability: "Database.Read".to_string(),
                    required_level: "Verified".to_string(),
                    justification: "Required for user data access".to_string(),
                },
            ],
            security_implications: SecurityAnalysis {
                risk_level: "Medium".to_string(),
                vulnerabilities: vec![
                    "Potential SQL injection in database queries".to_string(),
                ],
                mitigations: vec![
                    "Use parameterized queries".to_string(),
                    "Input validation and sanitization".to_string(),
                ],
            },
            composition_info: EffectCompositionInfo {
                compositions_found: 5,
                safe_compositions: 4,
                warnings: vec![
                    "Network + Crypto composition may have timing vulnerabilities".to_string(),
                ],
            },
        };
        
        Ok(DomainMetadata::Effects(effects_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Example Effects Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                ProviderCapability::RealTime,
                ProviderCapability::BusinessContext,
                ProviderCapability::CrossReference,
            ],
            dependencies: vec![],
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Prism AI Metadata Collection Example");
    println!("==========================================");
    
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
    
    // 2. Create AI integration coordinator
    let mut coordinator = AIIntegrationCoordinator::new(config);
    
    // 3. Register metadata providers from different crates
    println!("📝 Registering metadata providers...");
    
    // Register PIR provider (would come from prism-pir crate)
    coordinator.register_provider(Box::new(ExamplePIRProvider::new()));
    
    // Register Effects provider (would come from prism-effects crate)
    coordinator.register_provider(Box::new(ExampleEffectsProvider));
    
    // 4. Also register legacy collectors for backward compatibility
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
    
    // 5. Collect comprehensive metadata
    println!("🚀 Collecting metadata from all providers...");
    
    let project_root = PathBuf::from(".");
    let metadata = coordinator.collect_metadata(&project_root).await?;
    
    // 6. Display collected metadata summary
    println!("\n📊 Metadata Collection Results:");
    println!("================================");
    println!("Version: {}", metadata.version);
    println!("Exported at: {}", metadata.exported_at);
    println!("Project: {}", metadata.project_info.name);
    
    if let Some(syntax_meta) = &metadata.syntax_metadata {
        println!("✅ Syntax metadata: {} contexts, {} rules, {} insights", 
                syntax_meta.semantic_contexts.len(),
                syntax_meta.business_rules.len(),
                syntax_meta.insights.len());
    }
    
    if let Some(semantic_meta) = &metadata.semantic_metadata {
        println!("✅ Semantic metadata: placeholder = {}", semantic_meta.placeholder);
    }
    
    if let Some(pir_meta) = &metadata.pir_metadata {
        println!("✅ PIR metadata: placeholder = {}", pir_meta.placeholder);
    }
    
    if let Some(effects_meta) = &metadata.effects_metadata {
        println!("✅ Effects metadata: placeholder = {}", effects_meta.placeholder);
    }
    
    println!("📈 Quality metrics:");
    println!("   - Lines of code: {}", metadata.quality_metrics.lines_of_code);
    println!("   - Cyclomatic complexity: {}", metadata.quality_metrics.cyclomatic_complexity);
    println!("   - Test coverage: {:.1}%", metadata.quality_metrics.test_coverage);
    
    // 7. Export metadata in different formats
    println!("\n💾 Exporting metadata...");
    
    let export_results = coordinator.export_metadata(
        &metadata,
        &[ExportFormat::Json, ExportFormat::Yaml],
    ).await?;
    
    for (format, content) in export_results {
        match format {
            ExportFormat::Json => {
                println!("📄 JSON export: {} characters", content.len());
                // Could write to file here
            }
            ExportFormat::Yaml => {
                println!("📄 YAML export: {} characters", content.len());
                // Could write to file here
            }
            _ => {}
        }
    }
    
    println!("\n✅ Metadata collection complete!");
    println!("\n🏗️  Architecture Summary:");
    println!("========================");
    println!("✓ Separation of Concerns: Each provider handles only its domain");
    println!("✓ Conceptual Cohesion: Providers focus on exposing existing metadata");
    println!("✓ Modularity: Plug-and-play provider architecture");
    println!("✓ Backward Compatibility: Legacy collectors still work");
    println!("✓ No Duplication: Leverages existing metadata structures");
    println!("✓ Performance Aware: Minimal overhead, optional providers");
    
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
        coordinator.register_provider(Box::new(ExamplePIRProvider::new()));
        coordinator.register_provider(Box::new(ExampleEffectsProvider));
        
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