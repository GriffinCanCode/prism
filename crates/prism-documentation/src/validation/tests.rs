//! Tests for the documentation validation engine.
//!
//! These tests demonstrate the complete functionality of the PSG-003
//! documentation validation system, including semantic type integration,
//! business rule validation, and AI context checking.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::{DocumentationElement, DocumentationElementType, ElementVisibility, ExtractedAnnotation, AIContextInfo};
    use crate::validation::{DocumentationValidator, ValidationConfig, ValidationStrictness};
    use prism_common::span::Span;

    fn create_test_span() -> Span {
        Span::new(0, 0, 1, 1)
    }

    fn create_test_element(name: &str, element_type: DocumentationElementType) -> DocumentationElement {
        DocumentationElement {
            element_type,
            name: name.to_string(),
            content: Some("Test documentation content".to_string()),
            annotations: Vec::new(),
            location: create_test_span(),
            visibility: ElementVisibility::Public,
            ai_context: None,
            jsdoc_info: None,
        }
    }

    fn create_annotation(name: &str, value: Option<&str>) -> ExtractedAnnotation {
        ExtractedAnnotation {
            name: name.to_string(),
            value: value.map(|v| v.to_string()),
            arguments: Vec::new(),
            location: create_test_span(),
        }
    }

    #[test]
    fn test_module_validation_complete() {
        let validator = DocumentationValidator::strict();
        
        // Test complete module documentation
        let mut module_element = create_test_element("TestModule", DocumentationElementType::Module);
        module_element.annotations = vec![
            create_annotation("responsibility", Some("Handles user authentication and session management")),
            create_annotation("module", Some("TestModule")),
            create_annotation("description", Some("Provides secure authentication services")),
            create_annotation("author", Some("Security Team")),
        ];

        let result = validator.validate_documentation_element(&module_element).unwrap();
        assert!(result.is_compliant, "Complete module documentation should be compliant");
        assert!(result.violations.is_empty(), "No violations expected for complete documentation");
    }

    #[test]
    fn test_module_validation_missing_annotations() {
        let validator = DocumentationValidator::strict();
        
        // Test module with missing required annotations
        let module_element = create_test_element("TestModule", DocumentationElementType::Module);

        let result = validator.validate_documentation_element(&module_element).unwrap();
        assert!(!result.is_compliant, "Incomplete module documentation should not be compliant");
        
        // Should have violations for missing required annotations
        let violation_types: Vec<_> = result.violations.iter()
            .map(|v| &v.rule_id)
            .collect();
        
        assert!(violation_types.iter().any(|id| id.contains("RESPONSIBILITY")));
        assert!(violation_types.iter().any(|id| id.contains("DESCRIPTION")));
        assert!(violation_types.iter().any(|id| id.contains("AUTHOR")));
    }

    #[test]
    fn test_function_validation_complete() {
        let validator = DocumentationValidator::strict();
        
        // Test complete function documentation
        let mut function_element = create_test_element("authenticate", DocumentationElementType::Function);
        function_element.annotations = vec![
            create_annotation("responsibility", Some("Authenticates user credentials securely")),
            create_annotation("param", Some("email User email address")),
            create_annotation("param", Some("password User password")),
            create_annotation("returns", Some("Authentication result or error")),
            create_annotation("throws", Some("AuthenticationError when credentials are invalid")),
            create_annotation("example", Some("authenticate('user@example.com', 'password123')")),
            create_annotation("effects", Some("Database.Query, Audit.Log")),
        ];

        let result = validator.validate_documentation_element(&function_element).unwrap();
        assert!(result.is_compliant, "Complete function documentation should be compliant");
    }

    #[test]
    fn test_responsibility_length_validation() {
        let validator = DocumentationValidator::strict();
        
        // Test responsibility annotation that's too long
        let mut element = create_test_element("TestFunction", DocumentationElementType::Function);
        element.annotations = vec![
            create_annotation("responsibility", Some(
                "This is a very long responsibility statement that exceeds the maximum allowed character limit of 80 characters and should trigger a validation error"
            )),
        ];

        let result = validator.validate_documentation_element(&element).unwrap();
        assert!(!result.is_compliant, "Long responsibility should not be compliant");
        
        let has_length_violation = result.violations.iter()
            .any(|v| v.rule_id.contains("RESPONSIBILITY_LENGTH"));
        assert!(has_length_violation, "Should have responsibility length violation");
    }

    #[test]
    fn test_content_quality_validation() {
        let validator = DocumentationValidator::strict();
        
        // Test empty content
        let mut element = create_test_element("TestFunction", DocumentationElementType::Function);
        element.content = Some("".to_string());

        let result = validator.validate_documentation_element(&element).unwrap();
        let has_empty_content = result.violations.iter()
            .any(|v| v.rule_id == "CONTENT_EMPTY");
        assert!(has_empty_content, "Should detect empty content");

        // Test placeholder content
        element.content = Some("TODO: Add documentation...".to_string());
        let result = validator.validate_documentation_element(&element).unwrap();
        let has_placeholder = result.violations.iter()
            .any(|v| v.rule_id == "CONTENT_PLACEHOLDER");
        assert!(has_placeholder, "Should detect placeholder content");
    }

    #[test]
    fn test_jsdoc_compatibility_checking() {
        let mut config = ValidationConfig::strict();
        config.check_jsdoc_compatibility = true;
        let validator = DocumentationValidator::new(config);
        
        // Test Prism-specific annotation that's not JSDoc compatible
        let mut element = create_test_element("TestFunction", DocumentationElementType::Function);
        element.annotations = vec![
            create_annotation("responsibility", Some("Test function")),
            create_annotation("effects", Some("Database.Write")),
        ];

        let result = validator.validate_documentation_element(&element).unwrap();
        
        // Should have warnings about non-JSDoc annotations
        assert!(!result.warnings.is_empty(), "Should have JSDoc compatibility warnings");
        assert!(result.suggestions.iter().any(|s| s.contains("JSDoc")), "Should suggest JSDoc alternatives");
    }

    #[test]
    fn test_ai_context_validation() {
        let mut config = ValidationConfig::strict();
        config.check_ai_context = true;
        let validator = DocumentationValidator::new(config);
        
        // Test public function without AI context
        let element = create_test_element("processPayment", DocumentationElementType::Function);

        let result = validator.validate_documentation_element(&element).unwrap();
        
        // Should suggest AI context improvements
        assert!(result.suggestions.iter().any(|s| s.contains("AI context")), 
                "Should suggest adding AI context");
        assert!(result.suggestions.iter().any(|s| s.contains("examples")), 
                "Should suggest adding examples");
    }

    #[test]
    fn test_business_rule_validation() {
        let validator = DocumentationValidator::strict();
        
        // Test element with business-related name but no business annotations
        let element = create_test_element("calculateCustomerDiscount", DocumentationElementType::Function);

        let result = validator.validate_documentation_element(&element).unwrap();
        
        // Should suggest business context documentation
        assert!(result.suggestions.iter().any(|s| s.contains("business")), 
                "Should suggest business context for business-related function");
    }

    #[test]
    fn test_semantic_type_validation() {
        let validator = DocumentationValidator::strict();
        
        // Test Money type documentation
        let mut element = create_test_element("Money", DocumentationElementType::Type);
        element.annotations = vec![
            create_annotation("responsibility", Some("Represents monetary value with currency safety")),
        ];

        let result = validator.validate_documentation_element(&element).unwrap();
        
        // Should pass basic validation
        assert!(result.is_compliant || result.violations.iter().all(|v| v.severity != crate::validation::ViolationSeverity::Error));
        
        // May have suggestions for constraint documentation
        assert!(result.suggestions.iter().any(|s| s.contains("constraint") || s.contains("business")));
    }

    #[test]
    fn test_validation_statistics() {
        let validator = DocumentationValidator::strict();
        
        // Create multiple elements with varying documentation quality
        let complete_element = {
            let mut elem = create_test_element("CompleteFunction", DocumentationElementType::Function);
            elem.annotations = vec![
                create_annotation("responsibility", Some("Well documented function")),
                create_annotation("param", Some("input The input parameter")),
                create_annotation("returns", Some("The result")),
            ];
            elem
        };

        let incomplete_element = create_test_element("IncompleteFunction", DocumentationElementType::Function);

        let complete_result = validator.validate_documentation_element(&complete_element).unwrap();
        let incomplete_result = validator.validate_documentation_element(&incomplete_element).unwrap();

        // Complete element should be compliant
        assert!(complete_result.is_compliant, "Complete documentation should be compliant");
        
        // Incomplete element should not be compliant
        assert!(!incomplete_result.is_compliant, "Incomplete documentation should not be compliant");
        
        // Statistics should reflect the validation results
        assert_eq!(complete_result.statistics.total_items, 1);
        assert_eq!(incomplete_result.statistics.total_items, 1);
    }

    #[test]
    fn test_consistency_checking() {
        let validator = DocumentationValidator::strict();
        
        // Create extracted documentation with multiple elements
        use crate::extraction::{ExtractedDocumentation, ModuleDocumentation, ExtractionStatistics};
        
        let elements = vec![
            {
                let mut elem = create_test_element("Function1", DocumentationElementType::Function);
                elem.annotations = vec![create_annotation("responsibility", Some("Handles user input"))];
                elem
            },
            {
                let mut elem = create_test_element("Function2", DocumentationElementType::Function);
                elem.annotations = vec![create_annotation("responsibility", Some("Processes user data"))];
                elem
            },
        ];

        let docs = ExtractedDocumentation {
            elements,
            module_documentation: Some(ModuleDocumentation {
                name: "TestModule".to_string(),
                description: Some("Test module".to_string()),
                responsibility: Some("Handles user management".to_string()),
                author: Some("Test Team".to_string()),
                version: None,
                stability: None,
                dependencies: Vec::new(),
            }),
            statistics: ExtractionStatistics::new(),
        };

        let result = validator.validate_extracted(&docs).unwrap();
        
        // Should detect consistent responsibility patterns
        // (both functions have "user" in their responsibility)
        assert!(!result.warnings.is_empty() || !result.suggestions.is_empty(), 
               "Should provide feedback on documentation patterns");
    }

    #[test]
    fn test_validation_config_levels() {
        // Test different strictness levels
        let lenient = DocumentationValidator::lenient();
        let strict = DocumentationValidator::strict();
        
        let incomplete_element = create_test_element("TestFunction", DocumentationElementType::Function);

        let lenient_result = lenient.validate_documentation_element(&incomplete_element).unwrap();
        let strict_result = strict.validate_documentation_element(&incomplete_element).unwrap();

        // Strict validator should find more violations
        assert!(strict_result.violations.len() >= lenient_result.violations.len(),
                "Strict validation should find at least as many violations as lenient");
    }
} 