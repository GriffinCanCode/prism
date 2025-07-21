//! JSDoc compatibility and conversion
//!
//! This module embodies the single concept of "JSDoc Compatibility".
//! Following Prism's Conceptual Cohesion principle, this module is responsible
//! for ONE thing: providing seamless JSDoc compatibility and conversion.

use crate::{DocumentationError, DocumentationResult};
use crate::extraction::{ExtractedDocumentation, DocumentationElement, ExtractedAnnotation};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// JSDoc processor that handles compatibility and conversion
#[derive(Debug)]
pub struct JSDocProcessor {
    /// JSDoc compatibility configuration
    config: JSDocCompatibility,
}

/// JSDoc compatibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocCompatibility {
    /// Enable strict JSDoc validation
    pub strict_validation: bool,
    /// Convert Prism annotations to JSDoc format
    pub convert_to_jsdoc: bool,
    /// Convert JSDoc annotations to Prism format
    pub convert_from_jsdoc: bool,
    /// Include JSDoc-specific tags
    pub include_jsdoc_tags: bool,
    /// Generate JSDoc-compatible output
    pub generate_jsdoc_output: bool,
}

/// Result of JSDoc processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Successfully processed elements
    pub processed_elements: Vec<JSDocElement>,
    /// Conversion warnings
    pub warnings: Vec<String>,
    /// Compatibility issues found
    pub compatibility_issues: Vec<CompatibilityIssue>,
    /// Processing statistics
    pub statistics: ProcessingStatistics,
}

/// JSDoc-compatible documentation element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocElement {
    /// Element name
    pub name: String,
    /// JSDoc description
    pub description: Option<String>,
    /// JSDoc tags
    pub tags: Vec<JSDocTag>,
    /// Original Prism annotations
    pub prism_annotations: Vec<ExtractedAnnotation>,
    /// Conversion notes
    pub conversion_notes: Vec<String>,
}

/// JSDoc tag representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocTag {
    /// Tag name (e.g., "param", "returns", "throws")
    pub name: String,
    /// Tag type information
    pub tag_type: Option<String>,
    /// Tag description
    pub description: Option<String>,
    /// Tag value (for simple tags)
    pub value: Option<String>,
    /// Whether this tag is optional
    pub optional: bool,
    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

/// JSDoc compatibility issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Issue type
    pub issue_type: CompatibilityIssueType,
    /// Issue description
    pub description: String,
    /// Element where issue was found
    pub element_name: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
    /// Severity level
    pub severity: IssueSeverity,
}

/// Types of compatibility issues
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityIssueType {
    /// Missing JSDoc tag
    MissingJSDocTag,
    /// Invalid tag format
    InvalidTagFormat,
    /// Unsupported Prism annotation
    UnsupportedPrismAnnotation,
    /// Type information mismatch
    TypeMismatch,
    /// Description format issue
    DescriptionFormatIssue,
    /// Deprecated JSDoc usage
    DeprecatedUsage,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Error level issue
    Error,
    /// Warning level issue
    Warning,
    /// Information level issue
    Info,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Total elements processed
    pub total_elements: usize,
    /// Elements successfully converted
    pub converted_elements: usize,
    /// Elements with compatibility issues
    pub problematic_elements: usize,
    /// Conversion success rate
    pub conversion_success_rate: f64,
}

impl JSDocProcessor {
    /// Create a new JSDoc processor with default configuration
    pub fn new(config: JSDocCompatibility) -> Self {
        Self { config }
    }

    /// Process extracted documentation for JSDoc compatibility
    pub fn process_extracted(&self, docs: &ExtractedDocumentation) -> DocumentationResult<ProcessingResult> {
        let mut processed_elements = Vec::new();
        let mut warnings = Vec::new();
        let mut compatibility_issues = Vec::new();
        let mut stats = ProcessingStatistics::new();

        stats.total_elements = docs.elements.len();

        for element in &docs.elements {
            match self.process_element(element) {
                Ok(jsdoc_element) => {
                    processed_elements.push(jsdoc_element);
                    stats.converted_elements += 1;
                }
                Err(e) => {
                    warnings.push(format!("Failed to process element '{}': {}", element.name, e));
                    stats.problematic_elements += 1;
                    
                    // Create compatibility issue
                    compatibility_issues.push(CompatibilityIssue {
                        issue_type: CompatibilityIssueType::InvalidTagFormat,
                        description: format!("Processing failed: {}", e),
                        element_name: element.name.clone(),
                        suggested_fix: None,
                        severity: IssueSeverity::Warning,
                    });
                }
            }
        }

        // Calculate success rate
        stats.calculate_success_rate();

        Ok(ProcessingResult {
            processed_elements,
            warnings,
            compatibility_issues,
            statistics: stats,
        })
    }

    /// Process a single documentation element
    fn process_element(&self, element: &DocumentationElement) -> DocumentationResult<JSDocElement> {
        let mut jsdoc_tags = Vec::new();
        let mut conversion_notes = Vec::new();

        // Convert Prism annotations to JSDoc tags
        for annotation in &element.annotations {
            match self.convert_annotation_to_jsdoc(annotation) {
                Ok(tags) => {
                    jsdoc_tags.extend(tags);
                }
                Err(e) => {
                    conversion_notes.push(format!("Failed to convert annotation '{}': {}", annotation.name, e));
                }
            }
        }

        // Add standard JSDoc tags based on element type
        self.add_standard_jsdoc_tags(&mut jsdoc_tags, element);

        // Extract description from content or annotations
        let description = self.extract_description(element);

        Ok(JSDocElement {
            name: element.name.clone(),
            description,
            tags: jsdoc_tags,
            prism_annotations: element.annotations.clone(),
            conversion_notes,
        })
    }

    /// Convert a Prism annotation to JSDoc tags
    fn convert_annotation_to_jsdoc(&self, annotation: &ExtractedAnnotation) -> DocumentationResult<Vec<JSDocTag>> {
        let mut tags = Vec::new();

        match annotation.name.as_str() {
            // Direct mappings
            "param" => {
                tags.push(JSDocTag {
                    name: "param".to_string(),
                    tag_type: None, // TODO: Extract type information
                    description: annotation.value.clone(),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "returns" | "return" => {
                tags.push(JSDocTag {
                    name: "returns".to_string(),
                    tag_type: None, // TODO: Extract type information
                    description: annotation.value.clone(),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "throws" | "throw" => {
                tags.push(JSDocTag {
                    name: "throws".to_string(),
                    tag_type: None, // TODO: Extract exception type
                    description: annotation.value.clone(),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "example" => {
                tags.push(JSDocTag {
                    name: "example".to_string(),
                    tag_type: None,
                    description: annotation.value.clone(),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "since" => {
                tags.push(JSDocTag {
                    name: "since".to_string(),
                    tag_type: None,
                    description: None,
                    value: annotation.value.clone(),
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "deprecated" => {
                tags.push(JSDocTag {
                    name: "deprecated".to_string(),
                    tag_type: None,
                    description: annotation.value.clone(),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "author" => {
                tags.push(JSDocTag {
                    name: "author".to_string(),
                    tag_type: None,
                    description: None,
                    value: annotation.value.clone(),
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "version" => {
                tags.push(JSDocTag {
                    name: "version".to_string(),
                    tag_type: None,
                    description: None,
                    value: annotation.value.clone(),
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "see" => {
                tags.push(JSDocTag {
                    name: "see".to_string(),
                    tag_type: None,
                    description: None,
                    value: annotation.value.clone(),
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            "todo" => {
                tags.push(JSDocTag {
                    name: "todo".to_string(),
                    tag_type: None,
                    description: annotation.value.clone(),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                });
            }
            
            // Prism-specific annotations - convert with notes
            "responsibility" => {
                // Convert to custom JSDoc tag
                tags.push(JSDocTag {
                    name: "description".to_string(),
                    tag_type: None,
                    description: Some(format!("Responsibility: {}", annotation.value.as_deref().unwrap_or("Not specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), "responsibility".to_string());
                        attrs
                    },
                });
            }
            "effects" => {
                // Convert to custom JSDoc tag
                tags.push(JSDocTag {
                    name: "note".to_string(),
                    tag_type: None,
                    description: Some(format!("Effects: {}", annotation.value.as_deref().unwrap_or("None specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), "effects".to_string());
                        attrs
                    },
                });
            }
            "aiContext" => {
                // Convert to description with AI context
                tags.push(JSDocTag {
                    name: "note".to_string(),
                    tag_type: None,
                    description: Some(format!("AI Context: {}", annotation.value.as_deref().unwrap_or("Not specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), "aiContext".to_string());
                        attrs
                    },
                });
            }
            "performance" => {
                tags.push(JSDocTag {
                    name: "note".to_string(),
                    tag_type: None,
                    description: Some(format!("Performance: {}", annotation.value.as_deref().unwrap_or("Not specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), "performance".to_string());
                        attrs
                    },
                });
            }
            "security" => {
                tags.push(JSDocTag {
                    name: "note".to_string(),
                    tag_type: None,
                    description: Some(format!("Security: {}", annotation.value.as_deref().unwrap_or("Not specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), "security".to_string());
                        attrs
                    },
                });
            }
            "compliance" => {
                tags.push(JSDocTag {
                    name: "note".to_string(),
                    tag_type: None,
                    description: Some(format!("Compliance: {}", annotation.value.as_deref().unwrap_or("Not specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), "compliance".to_string());
                        attrs
                    },
                });
            }
            
            // Unknown annotations - convert as custom tags
            _ => {
                tags.push(JSDocTag {
                    name: "note".to_string(),
                    tag_type: None,
                    description: Some(format!("{}: {}", annotation.name, annotation.value.as_deref().unwrap_or("Not specified"))),
                    value: None,
                    optional: false,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("prism_annotation".to_string(), annotation.name.clone());
                        attrs
                    },
                });
            }
        }

        Ok(tags)
    }

    /// Add standard JSDoc tags based on element type
    fn add_standard_jsdoc_tags(&self, tags: &mut Vec<JSDocTag>, element: &DocumentationElement) {
        match element.element_type {
            crate::extraction::DocumentationElementType::Module => {
                // Add @module tag if not already present
                if !tags.iter().any(|tag| tag.name == "module") {
                    tags.push(JSDocTag {
                        name: "module".to_string(),
                        tag_type: None,
                        description: None,
                        value: Some(element.name.clone()),
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
            crate::extraction::DocumentationElementType::Function => {
                // Add @function tag if not already present
                if !tags.iter().any(|tag| tag.name == "function") {
                    tags.push(JSDocTag {
                        name: "function".to_string(),
                        tag_type: None,
                        description: None,
                        value: Some(element.name.clone()),
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
            crate::extraction::DocumentationElementType::Type => {
                // Add @typedef tag if not already present
                if !tags.iter().any(|tag| tag.name == "typedef") {
                    tags.push(JSDocTag {
                        name: "typedef".to_string(),
                        tag_type: None,
                        description: None,
                        value: Some(element.name.clone()),
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
            crate::extraction::DocumentationElementType::Constant => {
                // Add @constant tag if not already present
                if !tags.iter().any(|tag| tag.name == "constant") {
                    tags.push(JSDocTag {
                        name: "constant".to_string(),
                        tag_type: None,
                        description: None,
                        value: Some(element.name.clone()),
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
            _ => {} // Other types don't need special JSDoc tags
        }

        // Add visibility tags
        match element.visibility {
            crate::extraction::ElementVisibility::Public => {
                if !tags.iter().any(|tag| tag.name == "public") {
                    tags.push(JSDocTag {
                        name: "public".to_string(),
                        tag_type: None,
                        description: None,
                        value: None,
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
            crate::extraction::ElementVisibility::Private => {
                if !tags.iter().any(|tag| tag.name == "private") {
                    tags.push(JSDocTag {
                        name: "private".to_string(),
                        tag_type: None,
                        description: None,
                        value: None,
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
            crate::extraction::ElementVisibility::Internal => {
                if !tags.iter().any(|tag| tag.name == "protected") {
                    tags.push(JSDocTag {
                        name: "protected".to_string(),
                        tag_type: None,
                        description: None,
                        value: None,
                        optional: false,
                        attributes: HashMap::new(),
                    });
                }
            }
        }
    }

    /// Extract description from element
    fn extract_description(&self, element: &DocumentationElement) -> Option<String> {
        // First try to get description from content
        if let Some(content) = &element.content {
            return Some(content.clone());
        }

        // Then try to get from responsibility annotation
        for annotation in &element.annotations {
            if annotation.name == "responsibility" {
                if let Some(value) = &annotation.value {
                    return Some(format!("Responsibility: {}", value));
                }
            }
        }

        // Finally try to get from description annotation
        for annotation in &element.annotations {
            if annotation.name == "description" {
                return annotation.value.clone();
            }
        }

        None
    }

    /// Generate JSDoc-formatted output
    pub fn generate_jsdoc_output(&self, element: &JSDocElement) -> String {
        let mut output = String::new();
        
        output.push_str("/**\n");
        
        // Add description
        if let Some(description) = &element.description {
            for line in description.lines() {
                output.push_str(&format!(" * {}\n", line));
            }
            output.push_str(" *\n");
        }
        
        // Add tags
        for tag in &element.tags {
            let tag_line = self.format_jsdoc_tag(tag);
            output.push_str(&format!(" * {}\n", tag_line));
        }
        
        output.push_str(" */\n");
        
        output
    }

    /// Format a JSDoc tag
    fn format_jsdoc_tag(&self, tag: &JSDocTag) -> String {
        let mut formatted = format!("@{}", tag.name);
        
        if let Some(tag_type) = &tag.tag_type {
            formatted.push_str(&format!(" {{{}}}", tag_type));
        }
        
        if let Some(value) = &tag.value {
            formatted.push_str(&format!(" {}", value));
        }
        
        if let Some(description) = &tag.description {
            formatted.push_str(&format!(" - {}", description));
        }
        
        formatted
    }

    /// Validate JSDoc compatibility
    pub fn validate_jsdoc_compatibility(&self, element: &DocumentationElement) -> Vec<CompatibilityIssue> {
        let mut issues = Vec::new();

        // Check for required JSDoc elements based on element type
        match element.element_type {
            crate::extraction::DocumentationElementType::Function => {
                // Functions should have @param for each parameter and @returns
                if element.visibility == crate::extraction::ElementVisibility::Public {
                    if !element.annotations.iter().any(|ann| ann.name == "param") {
                        issues.push(CompatibilityIssue {
                            issue_type: CompatibilityIssueType::MissingJSDocTag,
                            description: "Public function missing @param documentation".to_string(),
                            element_name: element.name.clone(),
                            suggested_fix: Some("Add @param annotations for all parameters".to_string()),
                            severity: IssueSeverity::Warning,
                        });
                    }
                    
                    if !element.annotations.iter().any(|ann| ann.name == "returns" || ann.name == "return") {
                        issues.push(CompatibilityIssue {
                            issue_type: CompatibilityIssueType::MissingJSDocTag,
                            description: "Public function missing @returns documentation".to_string(),
                            element_name: element.name.clone(),
                            suggested_fix: Some("Add @returns annotation".to_string()),
                            severity: IssueSeverity::Warning,
                        });
                    }
                }
            }
            _ => {} // Other types have different requirements
        }

        // Check for description
        if element.content.is_none() {
            issues.push(CompatibilityIssue {
                issue_type: CompatibilityIssueType::DescriptionFormatIssue,
                description: "Element missing description".to_string(),
                element_name: element.name.clone(),
                suggested_fix: Some("Add a description comment".to_string()),
                severity: IssueSeverity::Info,
            });
        }

        issues
    }
}

impl ProcessingStatistics {
    /// Create new processing statistics
    pub fn new() -> Self {
        Self {
            total_elements: 0,
            converted_elements: 0,
            problematic_elements: 0,
            conversion_success_rate: 0.0,
        }
    }

    /// Calculate conversion success rate
    pub fn calculate_success_rate(&mut self) {
        if self.total_elements > 0 {
            self.conversion_success_rate = (self.converted_elements as f64 / self.total_elements as f64) * 100.0;
        }
    }
}

impl Default for JSDocCompatibility {
    fn default() -> Self {
        Self {
            strict_validation: false,
            convert_to_jsdoc: true,
            convert_from_jsdoc: true,
            include_jsdoc_tags: true,
            generate_jsdoc_output: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::{DocumentationElement, DocumentationElementType, ExtractedAnnotation, ElementVisibility};
    use prism_common::span::Span;

    #[test]
    fn test_basic_jsdoc_conversion() {
        let processor = JSDocProcessor::new(JSDocCompatibility::default());
        
        let element = DocumentationElement {
            element_type: DocumentationElementType::Function,
            name: "testFunction".to_string(),
            content: Some("Test function description".to_string()),
            annotations: vec![
                ExtractedAnnotation {
                    name: "param".to_string(),
                    value: Some("input - The input parameter".to_string()),
                    arguments: vec![],
                    location: Span::dummy(),
                },
                ExtractedAnnotation {
                    name: "returns".to_string(),
                    value: Some("The result value".to_string()),
                    arguments: vec![],
                    location: Span::dummy(),
                },
            ],
            location: Span::dummy(),
            visibility: ElementVisibility::Public,
            ai_context: None,
            jsdoc_info: None,
        };

        let result = processor.process_element(&element).unwrap();
        
        assert_eq!(result.name, "testFunction");
        assert_eq!(result.description, Some("Test function description".to_string()));
        assert!(result.tags.iter().any(|tag| tag.name == "param"));
        assert!(result.tags.iter().any(|tag| tag.name == "returns"));
        assert!(result.tags.iter().any(|tag| tag.name == "function"));
        assert!(result.tags.iter().any(|tag| tag.name == "public"));
    }

    #[test]
    fn test_prism_annotation_conversion() {
        let processor = JSDocProcessor::new(JSDocCompatibility::default());
        
        let element = DocumentationElement {
            element_type: DocumentationElementType::Function,
            name: "prismFunction".to_string(),
            content: None,
            annotations: vec![
                ExtractedAnnotation {
                    name: "responsibility".to_string(),
                    value: Some("Handles user authentication".to_string()),
                    arguments: vec![],
                    location: Span::dummy(),
                },
                ExtractedAnnotation {
                    name: "effects".to_string(),
                    value: Some("Database.Query, Audit.Log".to_string()),
                    arguments: vec![],
                    location: Span::dummy(),
                },
            ],
            location: Span::dummy(),
            visibility: ElementVisibility::Public,
            ai_context: None,
            jsdoc_info: None,
        };

        let result = processor.process_element(&element).unwrap();
        
        // Should convert Prism-specific annotations to JSDoc notes
        let description_tags: Vec<_> = result.tags.iter()
            .filter(|tag| tag.name == "description")
            .collect();
        let note_tags: Vec<_> = result.tags.iter()
            .filter(|tag| tag.name == "note")
            .collect();
        
        assert!(!description_tags.is_empty() || !note_tags.is_empty());
    }

    #[test]
    fn test_jsdoc_output_generation() {
        let processor = JSDocProcessor::new(JSDocCompatibility::default());
        
        let element = JSDocElement {
            name: "testFunction".to_string(),
            description: Some("Test function description".to_string()),
            tags: vec![
                JSDocTag {
                    name: "param".to_string(),
                    tag_type: Some("string".to_string()),
                    description: Some("The input parameter".to_string()),
                    value: Some("input".to_string()),
                    optional: false,
                    attributes: HashMap::new(),
                },
                JSDocTag {
                    name: "returns".to_string(),
                    tag_type: Some("boolean".to_string()),
                    description: Some("The result".to_string()),
                    value: None,
                    optional: false,
                    attributes: HashMap::new(),
                },
            ],
            prism_annotations: vec![],
            conversion_notes: vec![],
        };

        let output = processor.generate_jsdoc_output(&element);
        
        assert!(output.contains("/**"));
        assert!(output.contains("Test function description"));
        assert!(output.contains("@param"));
        assert!(output.contains("@returns"));
        assert!(output.contains("*/"));
    }
} 