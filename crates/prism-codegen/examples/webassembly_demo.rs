//! WebAssembly Backend Enhancement Demonstration
//!
//! This example demonstrates the enhanced WebAssembly backend capabilities:
//! - PIR semantic preservation with business rules
//! - Capability-based security integration
//! - Effect tracking and validation
//! - Runtime integration patterns
//! - AI metadata preservation

use prism_codegen::backends::{
    WebAssemblyBackend, CodeGenConfig, CompilationContext, CompilationTarget,
    WasmRuntimeTarget, WasmFeatures, AIMetadataLevel,
};
use prism_pir::semantic::representation::{
    PrismIR, PIRModule, PIRSemanticType, PIRTypeInfo, PIRPrimitiveType,
    PIRTypeAIContext, SecurityClassification, ValidationPredicate,
    EffectGraph, EffectNode, SemanticTypeRegistry,
    CohesionMetrics, PIRMetadata, PIRFunction, PIRFunctionType,
    PIRParameter, PIRExpression, PIRLiteral, EffectSignature, Effect,
    Capability, PIRPerformanceContract, PIRCondition, PIRPerformanceGuarantee,
    PIRPerformanceType,
};
use prism_business::BusinessRule;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Prism WebAssembly Backend Enhancement Demo");
    println!("===============================================");
    
    // Create enhanced WebAssembly backend configuration
    let config = CodeGenConfig {
        optimization_level: 2,
        debug_info: true,
        source_maps: true,
        target_options: HashMap::new(),
        ai_metadata_level: AIMetadataLevel::Comprehensive,
    };
    
    // Configure WebAssembly backend with advanced features
    let backend = WebAssemblyBackend::new(config.clone())
        .with_runtime_target(WasmRuntimeTarget::WASI)
        .with_features(WasmFeatures {
            multi_value: true,
            bulk_memory: true,
            simd: false,
            threads: false,
            tail_calls: true,
            reference_types: true,
        });
    
    println!("✅ Created WebAssembly backend with WASI runtime target");
    
    // Create a comprehensive PIR example
    let pir = create_example_pir();
    println!("✅ Created example PIR with semantic types and effects");
    
    // Create compilation context
    let context = CompilationContext {
        current_phase: "CodeGeneration".to_string(),
        targets: vec![CompilationTarget::WebAssembly],
    };
    
    // Generate WebAssembly code with full semantic preservation
    println!("\n🔄 Generating WebAssembly with semantic preservation...");
    
    match backend.generate_code_from_pir(&pir, &context, &config).await {
        Ok(artifact) => {
            println!("✅ Successfully generated WebAssembly code!");
            println!("📊 Generation Statistics:");
            println!("   - Lines generated: {}", artifact.stats.lines_generated);
            println!("   - Generation time: {}ms", artifact.stats.generation_time);
            println!("   - Optimizations applied: {}", artifact.stats.optimizations_applied);
            println!("   - Memory usage: {} bytes", artifact.stats.memory_usage);
            
            // Show a snippet of the generated code
            println!("\n📄 Generated WebAssembly Code Preview:");
            println!("{}", "-".repeat(60));
            let lines: Vec<&str> = artifact.content.lines().collect();
            for (i, line) in lines.iter().take(50).enumerate() {
                println!("{:3}: {}", i + 1, line);
            }
            if lines.len() > 50 {
                println!("... ({} more lines)", lines.len() - 50);
            }
            println!("{}", "-".repeat(60));
            
            // Validate the generated code
            println!("\n🔍 Validating generated WebAssembly...");
            match backend.validate(&artifact).await {
                Ok(warnings) => {
                    if warnings.is_empty() {
                        println!("✅ Validation passed with no warnings");
                    } else {
                        println!("⚠️  Validation passed with {} warnings:", warnings.len());
                        for warning in warnings {
                            println!("   - {}", warning);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Validation failed: {}", e);
                }
            }
            
            // Show runtime support capabilities
            println!("\n🚀 Runtime Support Features:");
            match backend.generate_runtime_support(&pir, &config).await {
                Ok(runtime_support) => {
                    println!("✅ Runtime support generated successfully");
                    let lines: Vec<&str> = runtime_support.lines().collect();
                    for line in lines.iter().take(20) {
                        if line.starts_with(";;") && line.contains("===") {
                            println!("   {}", line);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Runtime support generation failed: {}", e);
                }
            }
            
            // Show backend capabilities
            let capabilities = backend.capabilities();
            println!("\n🛠️  Backend Capabilities:");
            println!("   - Source maps: {}", capabilities.source_maps);
            println!("   - Debug info: {}", capabilities.debug_info);
            println!("   - Incremental: {}", capabilities.incremental);
            println!("   - Parallel: {}", capabilities.parallel);
            println!("   - Optimization levels: {:?}", capabilities.optimization_levels);
        }
        Err(e) => {
            println!("❌ Code generation failed: {}", e);
            return Err(e.into());
        }
    }
    
    println!("\n🎉 WebAssembly Backend Enhancement Demo Complete!");
    Ok(())
}

fn create_example_pir() -> PrismIR {
    let mut pir = PrismIR::new();
    
    // Add semantic types with business rules
    let user_id_type = PIRSemanticType {
        name: "UserId".to_string(),
        base_type: PIRTypeInfo::Primitive(PIRPrimitiveType::Integer { signed: false, width: 32 }),
        domain: "User Management".to_string(),
        business_rules: vec![
            BusinessRule {
                name: "PositiveUserId".to_string(),
                description: "User IDs must be positive integers".to_string(),
                expression: "value > 0".to_string(),
                severity: "Error".to_string(),
                category: "Validation".to_string(),
            }
        ],
        validation_predicates: vec![
            ValidationPredicate {
                name: "ValidRange".to_string(),
                expression: "value >= 1 && value <= 2147483647".to_string(),
                description: Some("User ID must be in valid 32-bit positive range".to_string()),
            }
        ],
        constraints: vec![],
        ai_context: PIRTypeAIContext {
            intent: Some("Unique identifier for users in the system".to_string()),
            examples: vec!["12345".to_string(), "67890".to_string()],
            common_mistakes: vec!["Using negative numbers".to_string(), "Using zero as ID".to_string()],
            best_practices: vec!["Always validate user ID exists in database".to_string()],
        },
        security_classification: SecurityClassification::Internal,
    };
    
    let email_type = PIRSemanticType {
        name: "EmailAddress".to_string(),
        base_type: PIRTypeInfo::Primitive(PIRPrimitiveType::String),
        domain: "Communication".to_string(),
        business_rules: vec![
            BusinessRule {
                name: "ValidEmailFormat".to_string(),
                description: "Email must be in valid format".to_string(),
                expression: "matches_regex(value, '^[^@]+@[^@]+\\.[^@]+$')".to_string(),
                severity: "Error".to_string(),
                category: "Format".to_string(),
            }
        ],
        validation_predicates: vec![
            ValidationPredicate {
                name: "NotEmpty".to_string(),
                expression: "length(value) > 0".to_string(),
                description: Some("Email address cannot be empty".to_string()),
            }
        ],
        constraints: vec![],
        ai_context: PIRTypeAIContext {
            intent: Some("Email address for user communication".to_string()),
            examples: vec!["user@example.com".to_string(), "admin@company.org".to_string()],
            common_mistakes: vec!["Missing @ symbol".to_string(), "Invalid domain".to_string()],
            best_practices: vec!["Validate email format before storing".to_string()],
        },
        security_classification: SecurityClassification::Confidential,
    };
    
    pir.type_registry.types.insert("UserId".to_string(), user_id_type);
    pir.type_registry.types.insert("EmailAddress".to_string(), email_type);
    
    // Add effects with capabilities
    let database_effect = EffectNode {
        name: "DatabaseRead".to_string(),
        effect_type: "IO".to_string(),
        capabilities: vec!["Database".to_string(), "Read".to_string()],
        side_effects: vec!["ReadOperation".to_string()],
    };
    
    let network_effect = EffectNode {
        name: "NetworkRequest".to_string(),
        effect_type: "Network".to_string(),
        capabilities: vec!["Network".to_string(), "HTTP".to_string()],
        side_effects: vec!["ExternalCall".to_string()],
    };
    
    pir.effect_graph.nodes.insert("DatabaseRead".to_string(), database_effect);
    pir.effect_graph.nodes.insert("NetworkRequest".to_string(), network_effect);
    
    // Add a sample function with effects and capabilities
    let get_user_function = PIRFunction {
        name: "getUserById".to_string(),
        signature: PIRFunctionType {
            parameters: vec![
                PIRParameter {
                    name: "userId".to_string(),
                    param_type: PIRTypeInfo::Primitive(PIRPrimitiveType::Integer { signed: false, width: 32 }),
                    default_value: None,
                    business_meaning: Some("ID of the user to retrieve".to_string()),
                }
            ],
            return_type: Box::new(PIRTypeInfo::Primitive(PIRPrimitiveType::String)),
            effects: EffectSignature {
                input_effects: vec![],
                output_effects: vec![
                    Effect {
                        name: "DatabaseRead".to_string(),
                        effect_type: "IO".to_string(),
                        description: Some("Read user data from database".to_string()),
                    }
                ],
                effect_dependencies: vec![],
            },
            contracts: PIRPerformanceContract {
                preconditions: vec![
                    PIRCondition {
                        name: "ValidUserId".to_string(),
                        expression: PIRExpression::Literal(PIRLiteral::Boolean(true)),
                        error_message: "User ID must be valid".to_string(),
                    }
                ],
                postconditions: vec![],
                performance_guarantees: vec![
                    PIRPerformanceGuarantee {
                        guarantee_type: PIRPerformanceType::MaxExecutionTime,
                        bound: PIRExpression::Literal(PIRLiteral::Integer(1000)), // 1 second
                        description: "Function must complete within 1 second".to_string(),
                    }
                ],
            },
        },
        body: PIRExpression::Literal(PIRLiteral::String("User data".to_string())),
        responsibility: Some("Retrieve user information by ID".to_string()),
        algorithm: Some("Database lookup with caching".to_string()),
        complexity: None,
        capabilities_required: vec![
            Capability {
                name: "Database".to_string(),
                description: Some("Access to user database".to_string()),
                permissions: vec!["read".to_string()],
            }
        ],
        performance_characteristics: vec!["O(1) with caching".to_string()],
        ai_hints: vec!["Commonly used for user profile retrieval".to_string()],
    };
    
    // Create a module containing our function
    let user_module = PIRModule {
        name: "UserManagement".to_string(),
        capability: "User data management and retrieval".to_string(),
        sections: vec![],
        dependencies: vec![],
        business_context: prism_business::BusinessContext {
            domain: "User Management".to_string(),
            stakeholders: vec!["Users".to_string(), "Administrators".to_string()],
            business_rules: vec![],
            compliance_requirements: vec![],
        },
        smart_module_metadata: prism_pir::semantic::representation::SmartModuleMetadata {
            discovery_info: prism_pir::semantic::representation::ModuleDiscoveryInfo {
                module_id: "user_mgmt_001".to_string(),
                version: Some("1.0.0".to_string()),
                stability: prism_pir::semantic::representation::StabilityLevel::Stable,
                visibility: prism_pir::semantic::representation::ModuleVisibility::Internal,
                tags: vec!["user".to_string(), "database".to_string()],
                classification: prism_pir::semantic::representation::ModuleClassification {
                    domain: Some("User Management".to_string()),
                    layer: Some("Business Logic".to_string()),
                    pattern: Some("Repository".to_string()),
                    technology_stack: vec!["WebAssembly".to_string(), "Database".to_string()],
                },
                search_keywords: vec!["user".to_string(), "profile".to_string(), "database".to_string()],
            },
            capability_mapping: prism_pir::semantic::representation::CapabilityMapping {
                provides: vec![],
                requires: vec![],
                optional: vec![],
                compatibility_matrix: HashMap::new(),
                version_compatibility: prism_pir::semantic::representation::VersionCompatibility {
                    semantic_versioning: true,
                    backward_compatible_versions: vec!["1.0.0".to_string()],
                    forward_compatible_versions: vec![],
                    evolution_strategy: "Semantic versioning".to_string(),
                },
            },
            business_analysis: prism_pir::semantic::representation::BusinessAnalysis {
                primary_domain: "User Management".to_string(),
                secondary_domains: vec![],
                value_proposition: "Efficient user data management".to_string(),
                business_entities: vec![],
                business_processes: vec![],
                business_rules: vec![],
                stakeholders: vec![],
            },
            cohesion_analysis: prism_pir::semantic::representation::CohesionAnalysis {
                cohesion_score: 0.85,
                cohesion_factors: vec!["Single responsibility".to_string()],
                coupling_analysis: prism_pir::semantic::representation::CouplingAnalysis {
                    afferent_coupling: 2,
                    efferent_coupling: 1,
                    instability: 0.33,
                    abstractness: 0.0,
                },
                violations: vec![],
                suggestions: vec![],
            },
            ai_insights: prism_pir::semantic::representation::AIInsights {
                module_summary: "User management module with database integration".to_string(),
                architectural_insights: vec!["Well-structured repository pattern".to_string()],
                performance_insights: vec!["Consider caching for frequently accessed users".to_string()],
                security_insights: vec!["Ensure proper authentication before user data access".to_string()],
                maintainability_insights: vec!["Clear separation of concerns".to_string()],
                reusability_assessment: prism_pir::semantic::representation::ReusabilityAssessment {
                    score: 0.8,
                    enhancing_factors: vec!["Generic design".to_string()],
                    limiting_factors: vec!["Database-specific implementation".to_string()],
                    improvement_suggestions: vec!["Abstract database interface".to_string()],
                },
                suggested_improvements: vec![],
                related_modules: vec!["Authentication".to_string()],
                learning_resources: vec![],
            },
            lifecycle_info: prism_pir::semantic::representation::ModuleLifecycleInfo {
                current_stage: prism_pir::semantic::representation::LifecycleStage::Production,
                created_at: "2024-01-01T00:00:00Z".to_string(),
                last_modified: "2024-01-01T00:00:00Z".to_string(),
                version_history: vec![],
                deprecation_info: None,
                migration_info: None,
            },
            integration_patterns: prism_pir::semantic::representation::IntegrationPatterns {
                patterns: vec![],
                relationships: vec![],
                communication_patterns: vec![],
                data_flow_patterns: vec![],
            },
            quality_metrics: prism_pir::semantic::representation::QualityMetrics {
                overall_score: 0.85,
                code_quality: prism_pir::semantic::representation::CodeQualityMetrics {
                    cyclomatic_complexity: 2.0,
                    lines_of_code: 50,
                    duplication_percentage: 0.0,
                    technical_debt_ratio: 0.1,
                    maintainability_index: 80.0,
                },
                design_quality: prism_pir::semantic::representation::DesignQualityMetrics {
                    coupling: prism_pir::semantic::representation::CouplingMetrics {
                        afferent: HashMap::new(),
                        efferent: HashMap::new(),
                        instability: HashMap::new(),
                    },
                    cohesion: 0.85,
                    abstraction_level: 0.7,
                    interface_quality: 0.9,
                    pattern_adherence: 0.8,
                },
                documentation_quality: prism_pir::semantic::representation::DocumentationQualityMetrics {
                    coverage_percentage: 90.0,
                    quality_score: 0.85,
                    up_to_date_percentage: 95.0,
                    completeness_score: 0.9,
                },
                test_coverage: prism_pir::semantic::representation::TestCoverageMetrics {
                    line_coverage: 85.0,
                    branch_coverage: 80.0,
                    function_coverage: 100.0,
                    integration_coverage: 70.0,
                },
                performance_metrics: prism_pir::semantic::representation::PerformanceMetrics {
                    avg_response_time: 50.0,
                    throughput: 1000.0,
                    memory_usage: 10.0,
                    cpu_usage: 5.0,
                    error_rate: 0.1,
                },
                security_metrics: prism_pir::semantic::representation::SecurityMetrics {
                    security_score: 0.9,
                    vulnerability_count: 0,
                    security_test_coverage: 80.0,
                    compliance_score: 0.95,
                },
            },
        },
        domain_rules: vec![],
        effects: vec![
            Effect {
                name: "DatabaseRead".to_string(),
                effect_type: "IO".to_string(),
                description: Some("Database read operations".to_string()),
            }
        ],
        capabilities: vec![
            Capability {
                name: "Database".to_string(),
                description: Some("Database access capability".to_string()),
                permissions: vec!["read".to_string()],
            }
        ],
        performance_profile: prism_quality::PerformanceProfile {
            cpu_usage: prism_quality::ResourceUsage {
                intensity: prism_quality::UsageIntensity::Low,
                allocation_pattern: prism_quality::AllocationPattern::Steady,
                peak_usage: 10.0,
                average_usage: 5.0,
            },
            memory_usage: prism_quality::ResourceUsage {
                intensity: prism_quality::UsageIntensity::Low,
                allocation_pattern: prism_quality::AllocationPattern::Steady,
                peak_usage: 1024.0, // 1MB
                average_usage: 512.0, // 512KB
            },
            io_characteristics: prism_quality::IOCharacteristics {
                read_heavy: true,
                write_heavy: false,
                sequential_access: true,
                random_access: false,
            },
        },
        cohesion_score: 0.85,
    };
    
    pir.modules.push(user_module);
    
    // Set overall cohesion metrics
    pir.cohesion_metrics = CohesionMetrics {
        overall_score: 0.85,
        module_scores: {
            let mut scores = HashMap::new();
            scores.insert("UserManagement".to_string(), 0.85);
            scores
        },
        coupling_metrics: prism_pir::semantic::representation::CouplingMetrics {
            afferent: HashMap::new(),
            efferent: HashMap::new(),
            instability: HashMap::new(),
        },
    };
    
    pir
} 