//! Comprehensive demonstration of the Prism documentation validation engine.
//!
//! This example shows how to use the complete PSG-003 documentation validation
//! system, including semantic type integration, business rule validation,
//! and AI context checking.

use prism_documentation::{
    DocumentationSystem, DocumentationConfig, ValidationConfig, ValidationStrictness,
    AIIntegrationConfig, AIDetailLevel, GenerationConfig, OutputFormat,
    JSDocCompatibility, RequirementConfig, RequirementStrictness,
};
use prism_documentation::validation::{
    DocumentationValidator, SemanticValidator, SemanticValidationConfig,
};
use prism_documentation::extraction::{
    DocumentationExtractor, ExtractionConfig, DocumentationElement,
    DocumentationElementType, ElementVisibility, ExtractedAnnotation,
};
use prism_common::span::Span;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Prism Documentation Validation Engine Demo");
    println!("===============================================");
    
    // 1. Create validation configuration
    let validation_config = ValidationConfig {
        strictness: ValidationStrictness::Standard,
        check_jsdoc_compatibility: true,
        check_ai_context: true,
        require_examples: true,
        require_performance_annotations: false,
        custom_rules: vec![],
        excluded_rules: std::collections::HashSet::new(),
    };

    // 2. Create documentation validator
    let mut validator = DocumentationValidator::new(validation_config);
    
    println!("\n📋 Testing Module Documentation Validation");
    println!("-------------------------------------------");
    
    // Test complete module documentation
    let complete_module = create_complete_module_example();
    let result = validator.validate_documentation_element(&complete_module)?;
    
    println!("✅ Complete Module Documentation:");
    println!("   Compliant: {}", result.is_compliant);
    println!("   Violations: {}", result.violations.len());
    println!("   Warnings: {}", result.warnings.len());
    println!("   Suggestions: {}", result.suggestions.len());

    // Test incomplete module documentation
    let incomplete_module = create_incomplete_module_example();
    let result = validator.validate_documentation_element(&incomplete_module)?;
    
    println!("\n❌ Incomplete Module Documentation:");
    println!("   Compliant: {}", result.is_compliant);
    println!("   Violations: {}", result.violations.len());
    
    for violation in &result.violations {
        println!("   - {}: {}", violation.rule_id, violation.message);
        if let Some(fix) = &violation.suggested_fix {
            println!("     Suggested fix: {}", fix);
        }
    }

    println!("\n🔧 Testing Function Documentation Validation");
    println!("--------------------------------------------");
    
    // Test complete function documentation
    let complete_function = create_complete_function_example();
    let result = validator.validate_documentation_element(&complete_function)?;
    
    println!("✅ Complete Function Documentation:");
    println!("   Compliant: {}", result.is_compliant);
    println!("   Violations: {}", result.violations.len());
    println!("   Suggestions: {}", result.suggestions.len());

    // Test function with semantic issues
    let semantic_function = create_semantic_function_example();
    let result = validator.validate_documentation_element(&semantic_function)?;
    
    println!("\n🤖 Function with Semantic Analysis:");
    println!("   Compliant: {}", result.is_compliant);
    println!("   Suggestions: {}", result.suggestions.len());
    
    for suggestion in &result.suggestions {
        println!("   - {}", suggestion);
    }

    println!("\n📊 Testing Validation Statistics");
    println!("--------------------------------");
    
    // Create extracted documentation with multiple elements
    let docs = create_extracted_documentation_example();
    let result = validator.validate_extracted(&docs)?;
    
    println!("Documentation Statistics:");
    println!("   Total items: {}", result.statistics.total_items);
    println!("   Documented items: {}", result.statistics.documented_items);
    println!("   Compliance: {:.1}%", result.statistics.compliance_percentage);
    println!("   JSDoc compatibility: {:.1}%", result.statistics.jsdoc_compatibility_percentage);

    println!("\n🎯 Testing Specialized Checkers");
    println!("-------------------------------");
    
    // Test JSDoc compatibility
    let mut jsdoc_config = ValidationConfig::strict();
    jsdoc_config.check_jsdoc_compatibility = true;
    let jsdoc_validator = DocumentationValidator::new(jsdoc_config);
    
    let prism_specific = create_prism_specific_example();
    let result = jsdoc_validator.validate_documentation_element(&prism_specific)?;
    
    println!("JSDoc Compatibility Check:");
    println!("   Warnings: {}", result.warnings.len());
    for warning in &result.warnings {
        println!("   - {}", warning);
    }

    println!("\n🏢 Testing Business Rule Validation");
    println!("-----------------------------------");
    
    let business_function = create_business_function_example();
    let result = validator.validate_documentation_element(&business_function)?;
    
    println!("Business Rule Analysis:");
    println!("   Suggestions: {}", result.suggestions.len());
    for suggestion in &result.suggestions {
        println!("   - {}", suggestion);
    }

    println!("\n🔬 Testing Content Quality Analysis");
    println!("-----------------------------------");
    
    let quality_issues = create_quality_issues_example();
    let result = validator.validate_documentation_element(&quality_issues)?;
    
    println!("Content Quality Issues:");
    println!("   Violations: {}", result.violations.len());
    for violation in &result.violations {
        println!("   - {}: {}", violation.rule_id, violation.message);
    }

    println!("\n🎉 Documentation Validation Demo Complete!");
    println!("==========================================");
    println!("The Prism documentation validation engine provides:");
    println!("✅ PSG-003 compliance checking");
    println!("✅ Semantic type integration (PLD-001)");
    println!("✅ Business rule validation");
    println!("✅ JSDoc compatibility analysis");
    println!("✅ AI context completeness checking");
    println!("✅ Content quality assessment");
    println!("✅ Comprehensive violation reporting");
    println!("✅ Actionable improvement suggestions");

    Ok(())
}

fn create_complete_module_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Module,
        name: "UserAuthentication".to_string(),
        content: Some("Comprehensive user authentication module with secure session management".to_string()),
        annotations: vec![
            ExtractedAnnotation {
                name: "responsibility".to_string(),
                value: Some("Handles secure user authentication and session management".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "module".to_string(),
                value: Some("UserAuthentication".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "description".to_string(),
                value: Some("Provides secure authentication services with MFA support".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "author".to_string(),
                value: Some("Security Team".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
        ],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_incomplete_module_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Module,
        name: "IncompleteModule".to_string(),
        content: Some("Basic module".to_string()),
        annotations: vec![],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_complete_function_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Function,
        name: "authenticate".to_string(),
        content: Some("Authenticates user credentials with comprehensive security validation".to_string()),
        annotations: vec![
            ExtractedAnnotation {
                name: "responsibility".to_string(),
                value: Some("Authenticates user credentials securely".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "param".to_string(),
                value: Some("email User's email address".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "param".to_string(),
                value: Some("password User's password".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "returns".to_string(),
                value: Some("Authentication result or error".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "throws".to_string(),
                value: Some("AuthenticationError when credentials are invalid".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "example".to_string(),
                value: Some("authenticate('user@example.com', 'password123')".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
        ],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_semantic_function_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Function,
        name: "transferMoney".to_string(),
        content: Some("Transfers money between accounts with validation".to_string()),
        annotations: vec![
            ExtractedAnnotation {
                name: "responsibility".to_string(),
                value: Some("Transfers funds between user accounts".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "param".to_string(),
                value: Some("amount Money amount to transfer".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "param".to_string(),
                value: Some("fromAccount Source account ID".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
        ],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_prism_specific_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Function,
        name: "processPayment".to_string(),
        content: Some("Processes payment with effects tracking".to_string()),
        annotations: vec![
            ExtractedAnnotation {
                name: "responsibility".to_string(),
                value: Some("Processes secure payment transactions".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "effects".to_string(),
                value: Some("Database.Write, Network.Send, Audit.Log".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
            ExtractedAnnotation {
                name: "capability".to_string(),
                value: Some("Payment.Process".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
        ],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_business_function_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Function,
        name: "calculateCustomerDiscount".to_string(),
        content: Some("Calculates discount based on customer tier and purchase history".to_string()),
        annotations: vec![
            ExtractedAnnotation {
                name: "responsibility".to_string(),
                value: Some("Calculates appropriate customer discount".to_string()),
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
        ],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_quality_issues_example() -> DocumentationElement {
    DocumentationElement {
        element_type: DocumentationElementType::Function,
        name: "badFunction".to_string(),
        content: Some("TODO: Add proper documentation...".to_string()),
        annotations: vec![
            ExtractedAnnotation {
                name: "responsibility".to_string(),
                value: Some("".to_string()), // Empty value
                arguments: vec![],
                location: Span::new(0, 0, 1, 1),
            },
        ],
        location: Span::new(0, 0, 1, 1),
        visibility: ElementVisibility::Public,
        ai_context: None,
        jsdoc_info: None,
    }
}

fn create_extracted_documentation_example() -> prism_documentation::extraction::ExtractedDocumentation {
    use prism_documentation::extraction::{ExtractedDocumentation, ModuleDocumentation, ExtractionStatistics};
    
    ExtractedDocumentation {
        elements: vec![
            create_complete_module_example(),
            create_complete_function_example(),
            create_incomplete_module_example(),
        ],
        module_documentation: Some(ModuleDocumentation {
            name: "TestModule".to_string(),
            description: Some("Test module for validation".to_string()),
            responsibility: Some("Provides test functionality".to_string()),
            author: Some("Test Team".to_string()),
            version: Some("1.0.0".to_string()),
            stability: Some("Stable".to_string()),
            dependencies: vec!["prism-common".to_string()],
        }),
        statistics: ExtractionStatistics {
            total_elements: 3,
            documented_elements: 2,
            elements_with_required_annotations: 1,
            missing_documentation_count: 1,
            missing_annotation_count: 2,
            documentation_coverage: 66.7,
        },
    }
} 