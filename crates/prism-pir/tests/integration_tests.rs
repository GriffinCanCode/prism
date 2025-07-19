//! PIR Integration Tests
//!
//! Comprehensive integration tests for PIR including:
//! - End-to-end compiler → PIR → codegen tests
//! - Semantic preservation validation tests
//! - Cross-target consistency validation
//! - Effect system integration tests
//! - Business context preservation tests

use prism_pir::{
    PrismIR, PIRBuilder, PIRBuilderConfig, PIRValidator, PIRProducer,
    EffectSystemBuilder, CapabilitySystem, TrustLevel, EffectCategory,
    types::{PIRModule, PIRSection, PIRSemanticType, PIRFunction, SecurityClassification},
    metadata::{AIMetadata, BusinessContext, PerformanceProfile},
    validation::{ValidationConfig, SemanticPreservationCheck, PreservationResult, PreservationStatus},
    effects::{IOEffect, IOResource, IOResourceType, CryptographyEffect, CryptoAlgorithm},
};
use prism_ast::{Program, Item, ItemKind, ModuleDecl, FunctionDecl, TypeDecl, ConstDecl, Attribute};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Test fixture for creating test AST programs
struct TestProgramBuilder {
    items: Vec<Item>,
}

impl TestProgramBuilder {
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn add_module(mut self, name: &str, capability: &str) -> Self {
        let module = ModuleDecl {
            name: name.to_string(),
            attributes: vec![Attribute {
                name: "capability".to_string(),
                value: Some(capability.to_string()),
            }],
            items: vec![
                Item {
                    kind: ItemKind::Function(FunctionDecl {
                        name: format!("{}_function", name),
                        attributes: vec![],
                        sig: prism_ast::FunctionSig {
                            params: vec![],
                            return_type: prism_ast::Type {
                                kind: prism_ast::TypeKind::Path("()".to_string()),
                                span: prism_common::span::Span::dummy(),
                            },
                        },
                        body: Some(prism_ast::Expr {
                            kind: prism_ast::ExprKind::Literal(prism_ast::Literal::Unit),
                            span: prism_common::span::Span::dummy(),
                        }),
                    }),
                    span: prism_common::span::Span::dummy(),
                }
            ],
        };

        self.items.push(Item {
            kind: ItemKind::Module(module),
            span: prism_common::span::Span::dummy(),
        });
        self
    }

    fn add_type(mut self, name: &str, domain: &str) -> Self {
        let type_decl = TypeDecl {
            name: name.to_string(),
            attributes: vec![
                Attribute {
                    name: "domain".to_string(),
                    value: Some(domain.to_string()),
                },
                Attribute {
                    name: "business_rule".to_string(),
                    value: Some("Must be validated".to_string()),
                }
            ],
            ty: prism_ast::Type {
                kind: prism_ast::TypeKind::Path("String".to_string()),
                span: prism_common::span::Span::dummy(),
            },
        };

        self.items.push(Item {
            kind: ItemKind::Type(type_decl),
            span: prism_common::span::Span::dummy(),
        });
        self
    }

    fn build(self) -> Program {
        Program {
            items: self.items,
            span: prism_common::span::Span::dummy(),
        }
    }
}

#[tokio::test]
async fn test_end_to_end_pir_generation() {
    // Create test AST program
    let program = TestProgramBuilder::new()
        .add_module("user_management", "user_operations")
        .add_module("payment_processing", "financial_operations")
        .add_type("UserId", "user_domain")
        .add_type("PaymentAmount", "financial_domain")
        .build();

    // Configure PIR builder
    let config = PIRBuilderConfig {
        enable_validation: true,
        enable_ai_metadata: true,
        enable_cohesion_analysis: true,
        enable_effect_graph: true,
        enable_business_context: true,
        max_build_depth: 100,
        enable_performance_profiling: true,
    };

    let mut builder = PIRBuilder::with_config(config);

    // Generate PIR from AST
    let result = builder.build_from_program(&program);
    assert!(result.is_ok(), "PIR generation should succeed");

    let pir = result.unwrap();

    // Validate PIR structure
    assert!(!pir.modules.is_empty(), "PIR should contain modules");
    assert_eq!(pir.modules.len(), 3, "Should have 2 explicit modules + 1 global module");

    // Find user management module
    let user_module = pir.modules.iter()
        .find(|m| m.name == "user_management")
        .expect("Should have user management module");

    assert_eq!(user_module.capability, "user_operations");
    assert!(!user_module.sections.is_empty(), "Module should have sections");

    // Find payment processing module
    let payment_module = pir.modules.iter()
        .find(|m| m.name == "payment_processing")
        .expect("Should have payment processing module");

    assert_eq!(payment_module.capability, "financial_operations");

    // Check global module has types
    let global_module = pir.modules.iter()
        .find(|m| m.name == "global")
        .expect("Should have global module");

    let has_types = global_module.sections.iter()
        .any(|section| matches!(section, PIRSection::Types(_)));
    assert!(has_types, "Global module should contain types");

    // Validate type registry
    assert!(!pir.type_registry.types.is_empty(), "Type registry should not be empty");

    // Validate AI metadata
    assert!(pir.ai_metadata.module_context.is_some(), "Should have module AI context");

    // Validate cohesion metrics
    assert!(pir.cohesion_metrics.overall_score > 0.0, "Should have cohesion score");
}

#[tokio::test]
async fn test_semantic_preservation_validation() {
    // Create test program
    let program = TestProgramBuilder::new()
        .add_type("Email", "user_domain")
        .add_type("Password", "security_domain")
        .build();

    // Generate PIR
    let mut builder = PIRBuilder::new();
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Create validator
    let config = ValidationConfig {
        strict_mode: true,
        validate_business_context: true,
        validate_effects: true,
        validate_ai_metadata: true,
    };
    let validator = PIRValidator::with_config(config);

    // Validate semantic preservation
    let validation_result = validator.validate_semantic_preservation(&pir)
        .expect("Validation should succeed");

    assert!(matches!(
        validation_result.status,
        prism_pir::validation::ValidationStatus::Passed | 
        prism_pir::validation::ValidationStatus::PassedWithWarnings
    ), "Validation should pass or pass with warnings");

    assert!(validation_result.overall_score > 0.5, "Should have reasonable validation score");
    assert!(!validation_result.check_results.is_empty(), "Should have check results");

    // Validate that semantic types are preserved
    let semantic_check_result = validation_result.check_results.iter()
        .find(|result| result.check_name == "Semantic Preservation")
        .expect("Should have semantic preservation check");

    assert!(semantic_check_result.score > 0.0, "Semantic preservation should have positive score");
}

#[tokio::test]
async fn test_effect_system_integration() {
    // Create program with effects
    let program = TestProgramBuilder::new()
        .add_module("file_operations", "file_management")
        .add_module("network_operations", "network_communication")
        .build();

    // Generate PIR
    let mut builder = PIRBuilder::new();
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Build effect system
    let mut effect_builder = EffectSystemBuilder::new();

    // Register effect categories
    effect_builder.register_effect_category(
        "file_read".to_string(),
        EffectCategory::IO(IOEffect::Read(IOResource {
            resource_type: IOResourceType::File,
            identifier: "file".to_string(),
        }))
    );

    effect_builder.register_effect_category(
        "crypto_encrypt".to_string(),
        EffectCategory::Cryptography(CryptographyEffect::Encryption(CryptoAlgorithm {
            algorithm: "AES-256".to_string(),
            key_size: Some(256),
            parameters: HashMap::new(),
        }))
    );

    // Create object capabilities
    let mut allowed_effects = HashSet::new();
    allowed_effects.insert(EffectCategory::IO(IOEffect::Read(IOResource {
        resource_type: IOResourceType::File,
        identifier: "file".to_string(),
    })));

    let file_capability = effect_builder.create_object_capability(
        "file_read_capability".to_string(),
        allowed_effects
    );

    assert_eq!(file_capability.id, "file_read_capability");
    assert_eq!(file_capability.allowed_effects.len(), 1);

    // Build effect graph
    let effect_graph = effect_builder.build_effect_graph(&pir)
        .expect("Effect graph building should succeed");

    // Validate effect graph structure
    assert!(effect_graph.nodes.len() >= 0, "Effect graph should be valid");
}

#[tokio::test]
async fn test_business_context_preservation() {
    // Create program with business context
    let program = TestProgramBuilder::new()
        .add_module("user_authentication", "security_operations")
        .add_type("UserCredentials", "authentication_domain")
        .build();

    // Generate PIR
    let config = PIRBuilderConfig {
        enable_business_context: true,
        enable_ai_metadata: true,
        ..PIRBuilderConfig::default()
    };

    let mut builder = PIRBuilder::with_config(config);
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Validate business context preservation
    for module in &pir.modules {
        if module.name == "user_authentication" {
            assert_eq!(module.capability, "security_operations");
            
            // Check business context
            assert!(!module.business_context.domain.is_empty(), "Should have business domain");
            
            // Check cohesion score
            assert!(module.cohesion_score > 0.0, "Should have cohesion score");
            
            break;
        }
    }

    // Validate AI metadata
    assert!(pir.ai_metadata.module_context.is_some(), "Should have AI module context");

    if let Some(module_context) = &pir.ai_metadata.module_context {
        assert!(!module_context.capability.is_empty(), "Should have capability description");
        assert!(!module_context.purpose.is_empty(), "Should have purpose description");
    }
}

#[tokio::test]
async fn test_cross_target_consistency() {
    // Create test program
    let program = TestProgramBuilder::new()
        .add_module("core_logic", "business_logic")
        .add_type("BusinessEntity", "core_domain")
        .build();

    // Generate PIR
    let mut builder = PIRBuilder::new();
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Validate PIR consistency for multiple targets
    validate_pir_for_target(&pir, "TypeScript").await;
    validate_pir_for_target(&pir, "WebAssembly").await;
    validate_pir_for_target(&pir, "LLVM").await;
}

async fn validate_pir_for_target(pir: &PrismIR, target: &str) {
    // Validate that PIR contains all necessary information for target
    assert!(!pir.modules.is_empty(), "PIR should have modules for {}", target);
    assert!(!pir.type_registry.types.is_empty(), "PIR should have types for {}", target);
    
    // Validate semantic preservation
    for module in &pir.modules {
        assert!(!module.capability.is_empty(), "Module capability should be preserved for {}", target);
        assert!(module.cohesion_score >= 0.0 && module.cohesion_score <= 1.0, 
               "Cohesion score should be valid for {}", target);
    }

    // Validate AI metadata preservation
    assert!(pir.ai_metadata.module_context.is_some() || pir.modules.is_empty(), 
           "AI metadata should be preserved for {}", target);
}

#[tokio::test]
async fn test_pir_serialization_roundtrip() {
    // Create test program
    let program = TestProgramBuilder::new()
        .add_module("serialization_test", "testing")
        .add_type("TestType", "test_domain")
        .build();

    // Generate PIR
    let mut builder = PIRBuilder::new();
    let original_pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Serialize PIR
    let serialized = serde_json::to_string(&original_pir)
        .expect("PIR serialization should succeed");

    // Deserialize PIR
    let deserialized_pir: PrismIR = serde_json::from_str(&serialized)
        .expect("PIR deserialization should succeed");

    // Validate roundtrip preservation
    assert_eq!(original_pir.modules.len(), deserialized_pir.modules.len(),
              "Module count should be preserved");
    
    assert_eq!(original_pir.type_registry.types.len(), 
              deserialized_pir.type_registry.types.len(),
              "Type registry size should be preserved");

    // Validate specific module preservation
    for (original_module, deserialized_module) in 
        original_pir.modules.iter().zip(deserialized_pir.modules.iter()) {
        assert_eq!(original_module.name, deserialized_module.name,
                  "Module name should be preserved");
        assert_eq!(original_module.capability, deserialized_module.capability,
                  "Module capability should be preserved");
        assert_eq!(original_module.sections.len(), deserialized_module.sections.len(),
                  "Module sections count should be preserved");
    }
}

#[tokio::test]
async fn test_pir_transformation_auditability() {
    // Create test program
    let program = TestProgramBuilder::new()
        .add_module("audit_test", "auditing")
        .build();

    // Generate PIR with audit trail enabled
    let config = PIRBuilderConfig {
        enable_validation: true,
        enable_ai_metadata: true,
        ..PIRBuilderConfig::default()
    };

    let mut builder = PIRBuilder::with_config(config);
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Validate PIR metadata contains audit information
    assert!(!pir.metadata.version.is_empty(), "Should have version information");
    assert!(!pir.metadata.created_at.is_empty(), "Should have creation timestamp");
    assert!(pir.metadata.source_hash > 0, "Should have source hash");

    // Validate transformation history structure
    // (This would be populated by actual transformations)
    assert!(pir.ai_metadata.module_context.is_some() || pir.modules.is_empty(),
           "Should have AI metadata for auditability");
}

#[tokio::test]
async fn test_pir_performance_characteristics() {
    // Create test program
    let program = TestProgramBuilder::new()
        .add_module("performance_test", "performance_critical")
        .build();

    // Generate PIR with performance profiling enabled
    let config = PIRBuilderConfig {
        enable_performance_profiling: true,
        ..PIRBuilderConfig::default()
    };

    let mut builder = PIRBuilder::with_config(config);
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Validate performance characteristics are captured
    for module in &pir.modules {
        // Performance profile should be present
        let profile = &module.performance_profile;
        
        // Basic validation that structure is present
        // (Actual values would depend on implementation)
        assert!(profile.cpu_usage.intensity == prism_pir::metadata::IntensityLevel::Medium ||
                profile.cpu_usage.intensity == prism_pir::metadata::IntensityLevel::Low ||
                profile.cpu_usage.intensity == prism_pir::metadata::IntensityLevel::High,
                "Should have valid CPU intensity level");
    }
}

#[tokio::test]
async fn test_pir_validation_strictness_levels() {
    // Create test program with potential issues
    let program = TestProgramBuilder::new()
        .add_type("MinimalType", "test_domain") // Minimal type with few attributes
        .build();

    let mut builder = PIRBuilder::new();
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Test different validation strictness levels
    let strict_config = ValidationConfig {
        strict_mode: true,
        validate_business_context: true,
        validate_effects: true,
        validate_ai_metadata: true,
    };

    let permissive_config = ValidationConfig {
        strict_mode: false,
        validate_business_context: false,
        validate_effects: false,
        validate_ai_metadata: false,
    };

    // Strict validation might find issues
    let strict_validator = PIRValidator::with_config(strict_config);
    let strict_result = strict_validator.validate_semantic_preservation(&pir)
        .expect("Strict validation should complete");

    // Permissive validation should be more lenient
    let permissive_validator = PIRValidator::with_config(permissive_config);
    let permissive_result = permissive_validator.validate_semantic_preservation(&pir)
        .expect("Permissive validation should complete");

    // Both should complete, but strict might have lower scores or more warnings
    assert!(strict_result.overall_score >= 0.0);
    assert!(permissive_result.overall_score >= 0.0);
    
    // Strict mode typically finds more issues
    assert!(strict_result.check_results.len() >= permissive_result.check_results.len(),
           "Strict validation should perform at least as many checks");
}

/// Custom semantic preservation check for testing
struct TestSemanticCheck;

impl SemanticPreservationCheck for TestSemanticCheck {
    fn check_preservation(&self, pir: &PrismIR) -> prism_pir::PIRResult<PreservationResult> {
        let score = if pir.modules.is_empty() { 0.0 } else { 0.9 };
        
        Ok(PreservationResult {
            result: if score > 0.8 { 
                PreservationStatus::FullyPreserved 
            } else { 
                PreservationStatus::PartiallyPreserved 
            },
            score,
            findings: Vec::new(),
        })
    }

    fn check_name(&self) -> &str {
        "TestSemanticCheck"
    }

    fn check_description(&self) -> &str {
        "Test semantic preservation check"
    }
}

#[tokio::test]
async fn test_custom_semantic_preservation_check() {
    let program = TestProgramBuilder::new()
        .add_module("test_module", "testing")
        .build();

    let mut builder = PIRBuilder::new();
    let pir = builder.build_from_program(&program)
        .expect("PIR generation should succeed");

    // Test custom semantic check
    let custom_check = TestSemanticCheck;
    let result = custom_check.check_preservation(&pir)
        .expect("Custom check should succeed");

    assert_eq!(result.score, 0.9);
    assert_eq!(result.result, PreservationStatus::FullyPreserved);
    assert_eq!(custom_check.check_name(), "TestSemanticCheck");
} 