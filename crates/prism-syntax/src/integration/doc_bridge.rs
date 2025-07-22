//! Bridge to documentation system for PSG-003 compliance.
//!
//! This module provides integration with the prism-documentation system,
//! extracting and validating documentation from parsed code to ensure
//! compliance with PSG-003 documentation standards.

use crate::normalization::CanonicalForm;
use prism_common::Span;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Documentation bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationBridgeConfig {
    /// Enable JSDoc compatibility
    pub jsdoc_compatibility: bool,
    /// Enable AI metadata extraction
    pub ai_metadata_extraction: bool,
    /// Validate documentation completeness
    pub validate_completeness: bool,
    /// Required documentation coverage percentage
    pub required_coverage: f64,
}

impl Default for DocumentationBridgeConfig {
    fn default() -> Self {
        Self {
            jsdoc_compatibility: true,
            ai_metadata_extraction: true,
            validate_completeness: true,
            required_coverage: 80.0, // 80% coverage required by default
        }
    }
}

/// Bridge for documentation integration
#[derive(Debug)]
pub struct DocumentationBridge {
    /// Configuration
    config: DocumentationBridgeConfig,
}

/// Documentation processing result
#[derive(Debug)]
pub struct DocumentationResult {
    /// Extracted documentation entries
    pub documentation: Vec<DocumentationEntry>,
    /// Validation results
    pub validation: ValidationResult,
    /// Processing statistics
    pub stats: DocumentationStats,
    /// Warnings generated during processing
    pub warnings: Vec<DocumentationWarning>,
}

/// Documentation entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationEntry {
    /// Entry type (function, type, module, etc.)
    pub entry_type: DocumentationType,
    /// Entry name
    pub name: String,
    /// Documentation content
    pub content: String,
    /// Source location
    pub location: Option<Span>,
    /// JSDoc tags (if applicable)
    pub jsdoc_tags: HashMap<String, String>,
    /// AI metadata
    pub ai_metadata: HashMap<String, String>,
}

/// Documentation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentationType {
    /// Function documentation
    Function,
    /// Type documentation
    Type,
    /// Module documentation
    Module,
    /// Parameter documentation
    Parameter,
    /// Return value documentation
    Return,
    /// Example documentation
    Example,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub is_valid: bool,
    /// Coverage percentage
    pub coverage_percentage: f64,
    /// Missing documentation items
    pub missing_items: Vec<String>,
    /// Validation errors
    pub errors: Vec<ValidationError>,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Location of the error
    pub location: Option<Span>,
    /// Severity level
    pub severity: ValidationSeverity,
}

/// Validation severity
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    /// Error - must be fixed
    Error,
    /// Warning - should be fixed
    Warning,
    /// Info - informational
    Info,
}

/// Documentation processing statistics
#[derive(Debug, Default, Clone)]
pub struct DocumentationStats {
    /// Number of functions documented
    pub functions_documented: usize,
    /// Number of types documented
    pub types_documented: usize,
    /// Number of modules documented
    pub modules_documented: usize,
    /// Total documentation entries
    pub total_entries: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Documentation warning
#[derive(Debug, Clone)]
pub struct DocumentationWarning {
    /// Warning message
    pub message: String,
    /// Location of the warning
    pub location: Option<Span>,
}

/// Documentation integration errors
#[derive(Debug, Error)]
pub enum DocumentationError {
    /// Documentation processing failed
    #[error("Documentation processing failed: {reason}")]
    ProcessingFailed { reason: String },
}

impl DocumentationBridge {
    /// Create new documentation bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(DocumentationBridgeConfig::default())
    }
    
    /// Create new documentation bridge with custom configuration
    pub fn with_config(config: DocumentationBridgeConfig) -> Self {
        Self { config }
    }
    
    /// Process documentation from canonical form using the existing prism-documentation system
    pub fn process_documentation(&self, canonical: &CanonicalForm) -> Result<DocumentationResult, DocumentationError> {
        let start_time = std::time::Instant::now();
        let mut stats = DocumentationStats::default();
        let mut warnings = Vec::new();
        let mut documentation_entries = Vec::new();
        
        // Create documentation system with appropriate configuration
        let doc_config = self.create_documentation_config();
        let mut doc_system = prism_documentation::DocumentationSystem::with_config(doc_config);
        
        // Extract documentation from canonical form
        for node in &canonical.nodes {
            match self.extract_documentation_from_node(node, &mut stats, &mut warnings) {
                Ok(mut entries) => documentation_entries.append(&mut entries),
                Err(e) => warnings.push(DocumentationWarning {
                    message: format!("Failed to extract documentation: {}", e),
                    location: None,
                }),
            }
        }
        
        // Validate documentation using the existing validation system
        let validation = self.validate_documentation(&documentation_entries)?;
        
        stats.processing_time_ms = start_time.elapsed().as_millis() as u64;
        stats.total_entries = documentation_entries.len();
        
        Ok(DocumentationResult {
            documentation: documentation_entries,
            validation,
            stats,
            warnings,
        })
    }
    
    /// Process documentation from AST program (integration with existing system)
    pub fn process_program(&self, program: &prism_ast::Program) -> Result<DocumentationResult, DocumentationError> {
        let start_time = std::time::Instant::now();
        
        // Create documentation system
        let doc_config = self.create_documentation_config();
        let mut doc_system = prism_documentation::DocumentationSystem::with_config(doc_config);
        
        // Process using the existing system
        let processing_result = doc_system.process_program(program)
            .map_err(|e| DocumentationError::ProcessingFailed {
                reason: format!("Documentation system processing failed: {}", e),
            })?;
        
        // Convert to our bridge format
        let mut documentation_entries = Vec::new();
        for element in &processing_result.extracted_documentation.elements {
            documentation_entries.push(self.convert_documentation_element(element));
        }
        
        let validation = ValidationResult {
            is_valid: processing_result.validation_result.is_compliant,
            coverage_percentage: self.calculate_coverage(&processing_result),
            missing_items: processing_result.validation_result.violations.iter()
                .filter(|v| matches!(v.violation_type, prism_documentation::ViolationType::MissingRequiredAnnotation))
                .map(|v| v.message.clone())
                .collect(),
            errors: processing_result.validation_result.violations.iter()
                .map(|v| ValidationError {
                    message: v.message.clone(),
                    location: Some(v.location),
                    severity: match v.severity {
                        prism_documentation::ViolationSeverity::Error => ValidationSeverity::Error,
                        prism_documentation::ViolationSeverity::Warning => ValidationSeverity::Warning,
                        prism_documentation::ViolationSeverity::Info => ValidationSeverity::Info,
                    },
                })
                .collect(),
        };
        
        let mut stats = DocumentationStats::default();
        stats.processing_time_ms = start_time.elapsed().as_millis() as u64;
        stats.total_entries = documentation_entries.len();
        stats.functions_documented = documentation_entries.iter()
            .filter(|e| matches!(e.entry_type, DocumentationType::Function))
            .count();
        stats.types_documented = documentation_entries.iter()
            .filter(|e| matches!(e.entry_type, DocumentationType::Type))
            .count();
        stats.modules_documented = documentation_entries.iter()
            .filter(|e| matches!(e.entry_type, DocumentationType::Module))
            .count();
        
        let warnings = processing_result.validation_result.warnings.iter()
            .map(|w| DocumentationWarning {
                message: w.clone(),
                location: None,
            })
            .collect();
        
        Ok(DocumentationResult {
            documentation: documentation_entries,
            validation,
            stats,
            warnings,
        })
    }
    
    /// Create documentation configuration from bridge configuration
    fn create_documentation_config(&self) -> prism_documentation::DocumentationConfig {
        prism_documentation::DocumentationConfig {
            validation: prism_documentation::ValidationConfig {
                strictness: if self.config.validate_completeness {
                    prism_documentation::ValidationStrictness::Standard
                } else {
                    prism_documentation::ValidationStrictness::Lenient
                },
                check_jsdoc_compatibility: self.config.jsdoc_compatibility,
                check_ai_context: self.config.ai_metadata_extraction,
                require_examples: self.config.validate_completeness,
                require_performance_annotations: self.config.validate_completeness,
                custom_rules: Vec::new(),
                excluded_rules: std::collections::HashSet::new(),
            },
            jsdoc_compatibility: prism_documentation::JSDocCompatibility {
                strict_validation: self.config.jsdoc_compatibility,
                convert_to_jsdoc: self.config.jsdoc_compatibility,
                convert_from_jsdoc: self.config.jsdoc_compatibility,
                include_jsdoc_tags: self.config.jsdoc_compatibility,
                generate_jsdoc_output: self.config.jsdoc_compatibility,
            },
            ai_integration: prism_documentation::AIIntegrationConfig {
                enabled: self.config.ai_metadata_extraction,
                include_business_context: self.config.ai_metadata_extraction,
                include_architectural_patterns: self.config.ai_metadata_extraction,
                include_semantic_relationships: self.config.ai_metadata_extraction,
                detail_level: prism_documentation::AIDetailLevel::Standard,
            },
            generation: prism_documentation::GenerationConfig {
                output_formats: vec![prism_documentation::OutputFormat::JSON],
                include_private: false,
                include_ai_metadata: self.config.ai_metadata_extraction,
                include_examples: true,
                generate_toc: true,
                template_directory: None,
                output_directory: "docs".to_string(),
                custom_options: HashMap::new(),
            },
            custom_requirements: HashMap::new(),
        }
    }
    
    /// Extract documentation from a canonical node
    fn extract_documentation_from_node(
        &self, 
        node: &CanonicalNode, 
        stats: &mut DocumentationStats,
        warnings: &mut Vec<DocumentationWarning>
    ) -> Result<Vec<DocumentationEntry>, DocumentationError> {
        let mut entries = Vec::new();
        
        // Extract documentation based on node structure
        match &node.structure {
            crate::normalization::CanonicalStructure::Function { name, .. } => {
                if let Some(doc_comment) = node.metadata.get("doc_comment") {
                    entries.push(DocumentationEntry {
                        entry_type: DocumentationType::Function,
                        name: name.clone(),
                        content: doc_comment.as_str().unwrap_or("").to_string(),
                        location: None, // TODO: Extract from node
                        jsdoc_tags: self.extract_jsdoc_tags(doc_comment.as_str().unwrap_or("")),
                        ai_metadata: self.extract_ai_metadata(node),
                    });
                    stats.functions_documented += 1;
                } else if self.config.validate_completeness {
                    warnings.push(DocumentationWarning {
                        message: format!("Function '{}' is missing documentation", name),
                        location: None,
                    });
                }
            }
            
            crate::normalization::CanonicalStructure::Type { name, .. } => {
                if let Some(doc_comment) = node.metadata.get("doc_comment") {
                    entries.push(DocumentationEntry {
                        entry_type: DocumentationType::Type,
                        name: name.clone(),
                        content: doc_comment.as_str().unwrap_or("").to_string(),
                        location: None,
                        jsdoc_tags: self.extract_jsdoc_tags(doc_comment.as_str().unwrap_or("")),
                        ai_metadata: self.extract_ai_metadata(node),
                    });
                    stats.types_documented += 1;
                } else if self.config.validate_completeness {
                    warnings.push(DocumentationWarning {
                        message: format!("Type '{}' is missing documentation", name),
                        location: None,
                    });
                }
            }
            
            _ => {
                // Handle other node types as needed
            }
        }
        
        Ok(entries)
    }
    
    /// Extract JSDoc tags from documentation content
    fn extract_jsdoc_tags(&self, content: &str) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        
        if !self.config.jsdoc_compatibility {
            return tags;
        }
        
        // Simple JSDoc tag extraction (in a real implementation, this would be more sophisticated)
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('@') {
                if let Some(space_pos) = line.find(' ') {
                    let tag_name = &line[1..space_pos];
                    let tag_value = &line[space_pos + 1..];
                    tags.insert(tag_name.to_string(), tag_value.to_string());
                } else {
                    let tag_name = &line[1..];
                    tags.insert(tag_name.to_string(), String::new());
                }
            }
        }
        
        tags
    }
    
    /// Extract AI metadata from a canonical node
    fn extract_ai_metadata(&self, node: &CanonicalNode) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        if !self.config.ai_metadata_extraction {
            return metadata;
        }
        
        // Extract AI-relevant information from node metadata
        for (key, value) in &node.metadata {
            if let Some(str_value) = value.as_str() {
                metadata.insert(key.clone(), str_value.to_string());
            }
        }
        
        // Add semantic information
        if let Some(complexity) = &node.ai_metadata.complexity_metrics {
            metadata.insert("complexity_score".to_string(), complexity.overall_score.to_string());
        }
        
        metadata
    }
    
    /// Validate documentation entries
    fn validate_documentation(&self, entries: &[DocumentationEntry]) -> Result<ValidationResult, DocumentationError> {
        let total_items = entries.len();
        let documented_items = entries.iter().filter(|e| !e.content.is_empty()).count();
        let coverage_percentage = if total_items > 0 {
            (documented_items as f64 / total_items as f64) * 100.0
        } else {
            100.0
        };
        
        let is_valid = coverage_percentage >= self.config.required_coverage;
        
        let mut missing_items = Vec::new();
        let mut errors = Vec::new();
        
        for entry in entries {
            if entry.content.is_empty() {
                missing_items.push(format!("{}: {}", entry.entry_type.name(), entry.name));
                
                if self.config.validate_completeness {
                    errors.push(ValidationError {
                        message: format!("{} '{}' is missing documentation", entry.entry_type.name(), entry.name),
                        location: entry.location,
                        severity: ValidationSeverity::Warning,
                    });
                }
            }
        }
        
        Ok(ValidationResult {
            is_valid,
            coverage_percentage,
            missing_items,
            errors,
        })
    }
    
    /// Convert from prism-documentation element to bridge element
    fn convert_documentation_element(&self, element: &prism_documentation::DocumentationElement) -> DocumentationEntry {
        DocumentationEntry {
            entry_type: match element.element_type {
                prism_documentation::DocumentationElementType::Module => DocumentationType::Module,
                prism_documentation::DocumentationElementType::Function => DocumentationType::Function,
                prism_documentation::DocumentationElementType::Type => DocumentationType::Type,
                prism_documentation::DocumentationElementType::Constant => DocumentationType::Type,
                prism_documentation::DocumentationElementType::Variable => DocumentationType::Type,
                prism_documentation::DocumentationElementType::Section => DocumentationType::Module,
            },
            name: element.name.clone(),
            content: element.content.clone().unwrap_or_default(),
            location: Some(element.location),
            jsdoc_tags: element.jsdoc_info.as_ref()
                .map(|info| info.tags.iter()
                    .map(|tag| (tag.name.clone(), tag.content.clone()))
                    .collect())
                .unwrap_or_default(),
            ai_metadata: element.ai_context.as_ref()
                .map(|ctx| {
                    let mut metadata = HashMap::new();
                    metadata.insert("purpose".to_string(), ctx.purpose.clone());
                    metadata.insert("complexity".to_string(), ctx.complexity_score.to_string());
                    metadata
                })
                .unwrap_or_default(),
        }
    }
    
    /// Calculate coverage percentage from processing result
    fn calculate_coverage(&self, result: &prism_documentation::ProcessingResult) -> f64 {
        let total_elements = result.extracted_documentation.elements.len();
        let documented_elements = result.extracted_documentation.elements.iter()
            .filter(|e| e.content.is_some() && !e.content.as_ref().unwrap().is_empty())
            .count();
        
        if total_elements > 0 {
            (documented_elements as f64 / total_elements as f64) * 100.0
        } else {
            100.0
        }
    }
}

impl DocumentationType {
    fn name(&self) -> &'static str {
        match self {
            DocumentationType::Function => "Function",
            DocumentationType::Type => "Type",
            DocumentationType::Module => "Module",
            DocumentationType::Parameter => "Parameter",
            DocumentationType::Return => "Return",
            DocumentationType::Example => "Example",
        }
    }
}

impl Default for DocumentationBridge {
    fn default() -> Self {
        Self::new()
    }
} 