//! Documentation extraction from AST
//!
//! This module embodies the single concept of "Documentation Extraction".
//! Following Prism's Conceptual Cohesion principle, this module is responsible
//! for ONE thing: extracting documentation elements from AST nodes.

use crate::{DocumentationError, DocumentationResult};
use prism_ast::{Program, AstNode, Item, Stmt, FunctionDecl, TypeDecl, ModuleDecl, Attribute};
use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Documentation extractor that extracts documentation from AST nodes
#[derive(Debug)]
pub struct DocumentationExtractor {
    /// Configuration for extraction
    config: ExtractionConfig,
}

/// Configuration for documentation extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Extract documentation comments
    pub extract_doc_comments: bool,
    /// Extract annotations
    pub extract_annotations: bool,
    /// Extract AI context
    pub extract_ai_context: bool,
    /// Extract JSDoc compatible information
    pub extract_jsdoc_info: bool,
    /// Include private items
    pub include_private_items: bool,
}

/// Extracted documentation from a program or module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDocumentation {
    /// Documentation elements found
    pub elements: Vec<DocumentationElement>,
    /// Module-level documentation
    pub module_documentation: Option<ModuleDocumentation>,
    /// Statistics about extraction
    pub statistics: ExtractionStatistics,
}

/// Individual documentation element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationElement {
    /// Element type
    pub element_type: DocumentationElementType,
    /// Element name/identifier
    pub name: String,
    /// Documentation content
    pub content: Option<String>,
    /// Annotations found
    pub annotations: Vec<ExtractedAnnotation>,
    /// Source location
    pub location: Span,
    /// Visibility
    pub visibility: ElementVisibility,
    /// AI context if available
    pub ai_context: Option<AIContextInfo>,
    /// JSDoc information if available
    pub jsdoc_info: Option<JSDocInfo>,
}

/// Types of documentation elements
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentationElementType {
    /// Module documentation
    Module,
    /// Function documentation
    Function,
    /// Type documentation
    Type,
    /// Constant documentation
    Constant,
    /// Variable documentation
    Variable,
    /// Section documentation
    Section,
}

/// Extracted annotation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedAnnotation {
    /// Annotation name (e.g., "responsibility", "param", "returns")
    pub name: String,
    /// Annotation value
    pub value: Option<String>,
    /// Annotation arguments
    pub arguments: Vec<AnnotationArgument>,
    /// Location of the annotation
    pub location: Span,
}

/// Annotation argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationArgument {
    /// Argument name (for named arguments)
    pub name: Option<String>,
    /// Argument value
    pub value: String,
}

/// Element visibility
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementVisibility {
    /// Public element
    Public,
    /// Private element
    Private,
    /// Internal element
    Internal,
}

/// AI context information extracted from code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIContextInfo {
    /// Primary purpose or intent
    pub purpose: Option<String>,
    /// Business context
    pub business_context: Option<String>,
    /// Constraints and limitations
    pub constraints: Vec<String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// AI-specific hints
    pub ai_hints: Vec<String>,
}

/// JSDoc compatible information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocInfo {
    /// Description
    pub description: Option<String>,
    /// Parameters
    pub params: Vec<JSDocParam>,
    /// Return information
    pub returns: Option<JSDocReturn>,
    /// Throws information
    pub throws: Vec<JSDocThrows>,
    /// Examples
    pub examples: Vec<String>,
    /// See also references
    pub see_also: Vec<String>,
    /// Since version
    pub since: Option<String>,
    /// Deprecated information
    pub deprecated: Option<String>,
}

/// JSDoc parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocParam {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: Option<String>,
    /// Parameter description
    pub description: Option<String>,
    /// Whether parameter is optional
    pub optional: bool,
}

/// JSDoc return information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocReturn {
    /// Return type
    pub return_type: Option<String>,
    /// Return description
    pub description: Option<String>,
}

/// JSDoc throws information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSDocThrows {
    /// Exception type
    pub exception_type: Option<String>,
    /// Exception description
    pub description: Option<String>,
}

/// Module-level documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDocumentation {
    /// Module name
    pub name: String,
    /// Module description
    pub description: Option<String>,
    /// Module responsibility
    pub responsibility: Option<String>,
    /// Module author
    pub author: Option<String>,
    /// Module version
    pub version: Option<String>,
    /// Module stability level
    pub stability: Option<String>,
    /// Module dependencies
    pub dependencies: Vec<String>,
}

/// Statistics about the extraction process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionStatistics {
    /// Total elements processed
    pub total_elements: usize,
    /// Elements with documentation
    pub documented_elements: usize,
    /// Elements with required annotations
    pub elements_with_required_annotations: usize,
    /// Missing documentation count
    pub missing_documentation_count: usize,
    /// Missing annotation count
    pub missing_annotation_count: usize,
    /// Documentation coverage percentage
    pub documentation_coverage: f64,
}

impl DocumentationExtractor {
    /// Create a new documentation extractor with default configuration
    pub fn new() -> Self {
        Self {
            config: ExtractionConfig::default(),
        }
    }

    /// Create a new documentation extractor with custom configuration
    pub fn with_config(config: ExtractionConfig) -> Self {
        Self { config }
    }

    /// Extract documentation from a complete program
    pub fn extract_from_program(&self, program: &Program) -> DocumentationResult<ExtractedDocumentation> {
        let mut elements = Vec::new();
        let mut module_documentation = None;
        let mut stats = ExtractionStatistics::new();

        // Process each top-level item
        for item in &program.items {
            match &item.kind {
                Item::Module(module_decl) => {
                    // Extract module documentation
                    let module_doc = self.extract_module_documentation(item, module_decl)?;
                    module_documentation = Some(module_doc);
                    
                    // Extract module as documentation element
                    let element = self.extract_from_module(item, module_decl)?;
                    elements.push(element);
                    stats.total_elements += 1;
                }
                Item::Function(func_decl) => {
                    let element = self.extract_from_function(item, func_decl)?;
                    elements.push(element);
                    stats.total_elements += 1;
                }
                Item::Type(type_decl) => {
                    let element = self.extract_from_type(item, type_decl)?;
                    elements.push(element);
                    stats.total_elements += 1;
                }
                Item::Const(const_decl) => {
                    let element = self.extract_from_const(item, const_decl)?;
                    elements.push(element);
                    stats.total_elements += 1;
                }
                Item::Variable(var_decl) => {
                    if self.config.include_private_items || var_decl.visibility == prism_ast::Visibility::Public {
                        let element = self.extract_from_variable(item, var_decl)?;
                        elements.push(element);
                        stats.total_elements += 1;
                    }
                }
                _ => {
                    // Skip other item types for now
                }
            }
        }

        // Calculate statistics
        stats.calculate_coverage(&elements);

        Ok(ExtractedDocumentation {
            elements,
            module_documentation,
            statistics: stats,
        })
    }

    /// Extract documentation from a single item
    pub fn extract_from_item(&self, item: &AstNode<Item>) -> DocumentationResult<ExtractedDocumentation> {
        let mut elements = Vec::new();
        let mut stats = ExtractionStatistics::new();

        match &item.kind {
            Item::Function(func_decl) => {
                let element = self.extract_from_function(item, func_decl)?;
                elements.push(element);
                stats.total_elements = 1;
            }
            Item::Type(type_decl) => {
                let element = self.extract_from_type(item, type_decl)?;
                elements.push(element);
                stats.total_elements = 1;
            }
            Item::Module(module_decl) => {
                let element = self.extract_from_module(item, module_decl)?;
                elements.push(element);
                stats.total_elements = 1;
            }
            _ => {
                return Err(DocumentationError::ExtractionFailed {
                    reason: "Unsupported item type for documentation extraction".to_string(),
                });
            }
        }

        stats.calculate_coverage(&elements);

        Ok(ExtractedDocumentation {
            elements,
            module_documentation: None,
            statistics: stats,
        })
    }

    /// Extract documentation from a function declaration
    fn extract_from_function(&self, item: &AstNode<Item>, func_decl: &FunctionDecl) -> DocumentationResult<DocumentationElement> {
        let annotations = self.extract_annotations_from_attributes(&func_decl.attributes)?;
        let doc_content = self.extract_doc_comment_from_metadata(&item.metadata);
        let ai_context = self.extract_ai_context_from_metadata(&item.metadata);
        let jsdoc_info = self.extract_jsdoc_info(&annotations, &doc_content);

        Ok(DocumentationElement {
            element_type: DocumentationElementType::Function,
            name: func_decl.name.to_string(),
            content: doc_content,
            annotations,
            location: item.span,
            visibility: self.map_visibility(&func_decl.visibility),
            ai_context,
            jsdoc_info,
        })
    }

    /// Extract documentation from a type declaration
    fn extract_from_type(&self, item: &AstNode<Item>, type_decl: &TypeDecl) -> DocumentationResult<DocumentationElement> {
        let annotations = self.extract_annotations_from_attributes(&type_decl.attributes);
        let doc_content = self.extract_doc_comment_from_metadata(&item.metadata);
        let ai_context = self.extract_ai_context_from_metadata(&item.metadata);
        let jsdoc_info = self.extract_jsdoc_info(&annotations.as_ref().unwrap_or(&vec![]), &doc_content);

        Ok(DocumentationElement {
            element_type: DocumentationElementType::Type,
            name: type_decl.name.to_string(),
            content: doc_content,
            annotations: annotations.unwrap_or_default(),
            location: item.span,
            visibility: self.map_visibility(&type_decl.visibility),
            ai_context,
            jsdoc_info,
        })
    }

    /// Extract documentation from a module declaration
    fn extract_from_module(&self, item: &AstNode<Item>, module_decl: &ModuleDecl) -> DocumentationResult<DocumentationElement> {
        let annotations = self.extract_annotations_from_attributes(&module_decl.attributes)?;
        let doc_content = self.extract_doc_comment_from_metadata(&item.metadata);
        let ai_context = self.extract_ai_context_from_metadata(&item.metadata);
        let jsdoc_info = self.extract_jsdoc_info(&annotations, &doc_content);

        Ok(DocumentationElement {
            element_type: DocumentationElementType::Module,
            name: module_decl.name.to_string(),
            content: doc_content,
            annotations,
            location: item.span,
            visibility: ElementVisibility::Public, // Modules are typically public
            ai_context,
            jsdoc_info,
        })
    }

    /// Extract documentation from a constant declaration
    fn extract_from_const(&self, item: &AstNode<Item>, const_decl: &prism_ast::ConstDecl) -> DocumentationResult<DocumentationElement> {
        let annotations = self.extract_annotations_from_attributes(&const_decl.attributes)?;
        let doc_content = self.extract_doc_comment_from_metadata(&item.metadata);
        let ai_context = self.extract_ai_context_from_metadata(&item.metadata);
        let jsdoc_info = self.extract_jsdoc_info(&annotations, &doc_content);

        Ok(DocumentationElement {
            element_type: DocumentationElementType::Constant,
            name: const_decl.name.to_string(),
            content: doc_content,
            annotations,
            location: item.span,
            visibility: self.map_visibility(&const_decl.visibility),
            ai_context,
            jsdoc_info,
        })
    }

    /// Extract documentation from a variable declaration
    fn extract_from_variable(&self, item: &AstNode<Item>, var_decl: &prism_ast::VariableDecl) -> DocumentationResult<DocumentationElement> {
        let annotations = Vec::new(); // Variables typically don't have attributes in current AST
        let doc_content = self.extract_doc_comment_from_metadata(&item.metadata);
        let ai_context = self.extract_ai_context_from_metadata(&item.metadata);
        let jsdoc_info = self.extract_jsdoc_info(&annotations, &doc_content);

        Ok(DocumentationElement {
            element_type: DocumentationElementType::Variable,
            name: var_decl.name.to_string(),
            content: doc_content,
            annotations,
            location: item.span,
            visibility: self.map_visibility(&var_decl.visibility),
            ai_context,
            jsdoc_info,
        })
    }

    /// Extract module-level documentation
    fn extract_module_documentation(&self, item: &AstNode<Item>, module_decl: &ModuleDecl) -> DocumentationResult<ModuleDocumentation> {
        let annotations = self.extract_annotations_from_attributes(&module_decl.attributes)?;
        
        // Extract specific module annotations
        let responsibility = self.find_annotation_value(&annotations, "responsibility");
        let description = self.find_annotation_value(&annotations, "description");
        let author = self.find_annotation_value(&annotations, "author");
        let version = self.find_annotation_value(&annotations, "version");
        let stability = self.find_annotation_value(&annotations, "stability");
        
        // Extract dependencies from annotations or metadata
        let dependencies = self.extract_dependencies(&annotations);

        Ok(ModuleDocumentation {
            name: module_decl.name.to_string(),
            description,
            responsibility,
            author,
            version,
            stability,
            dependencies,
        })
    }

    /// Extract annotations from attributes
    fn extract_annotations_from_attributes(&self, attributes: &[Attribute]) -> DocumentationResult<Vec<ExtractedAnnotation>> {
        let mut annotations = Vec::new();

        for attr in attributes {
            let annotation = ExtractedAnnotation {
                name: attr.name.to_string(),
                value: self.extract_primary_value_from_attribute(attr),
                arguments: self.extract_arguments_from_attribute(attr),
                location: Span::dummy(), // TODO: Get actual location from attribute
            };
            annotations.push(annotation);
        }

        Ok(annotations)
    }

    /// Extract primary value from attribute
    fn extract_primary_value_from_attribute(&self, attr: &Attribute) -> Option<String> {
        // For attributes like @responsibility "value", extract the first string argument
        attr.arguments.first().and_then(|arg| match arg {
            prism_ast::AttributeArgument::Literal(prism_ast::AttributeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
    }

    /// Extract arguments from attribute
    fn extract_arguments_from_attribute(&self, attr: &Attribute) -> Vec<AnnotationArgument> {
        attr.arguments.iter().map(|arg| {
            match arg {
                prism_ast::AttributeArgument::Literal(value) => {
                    AnnotationArgument {
                        name: None,
                        value: self.attribute_value_to_string(value),
                    }
                }
                prism_ast::AttributeArgument::Named { name, value } => {
                    AnnotationArgument {
                        name: Some(name.to_string()),
                        value: self.attribute_value_to_string(value),
                    }
                }
            }
        }).collect()
    }

    /// Convert literal value to string
    fn literal_to_string(&self, literal: &prism_ast::LiteralValue) -> String {
        match literal {
            prism_ast::LiteralValue::String(s) => s.clone(),
            prism_ast::LiteralValue::Integer(i) => i.to_string(),
            prism_ast::LiteralValue::Float(f) => f.to_string(),
            prism_ast::LiteralValue::Boolean(b) => b.to_string(),
            prism_ast::LiteralValue::Null => "null".to_string(),
            prism_ast::LiteralValue::Money { amount, currency } => format!("{} {}", amount, currency),
            prism_ast::LiteralValue::Duration { value, unit } => format!("{} {}", value, unit),
            prism_ast::LiteralValue::Regex(pattern) => format!("/{}/", pattern),
        }
    }

    /// Convert attribute value to string
    fn attribute_value_to_string(&self, value: &prism_ast::AttributeValue) -> String {
        match value {
            prism_ast::AttributeValue::String(s) => s.clone(),
            prism_ast::AttributeValue::Integer(i) => i.to_string(),
            prism_ast::AttributeValue::Float(f) => f.to_string(),
            prism_ast::AttributeValue::Boolean(b) => b.to_string(),
            prism_ast::AttributeValue::Array(arr) => {
                let values: Vec<String> = arr.iter().map(|v| self.attribute_value_to_string(v)).collect();
                format!("[{}]", values.join(", "))
            }
            prism_ast::AttributeValue::Object(obj) => {
                let pairs: Vec<String> = obj.iter().map(|(k, v)| format!("{}: {}", k, self.attribute_value_to_string(v))).collect();
                format!("{{{}}}", pairs.join(", "))
            }
        }
    }

    /// Extract documentation comment from node metadata
    fn extract_doc_comment_from_metadata(&self, metadata: &prism_ast::NodeMetadata) -> Option<String> {
        metadata.documentation.clone()
    }

    /// Extract AI context from node metadata
    fn extract_ai_context_from_metadata(&self, metadata: &prism_ast::NodeMetadata) -> Option<AIContextInfo> {
        metadata.ai_context.as_ref().map(|ai_ctx| {
            AIContextInfo {
                purpose: ai_ctx.purpose.clone(),
                business_context: ai_ctx.domain.clone(),
                constraints: ai_ctx.preconditions.clone(),
                examples: metadata.examples.clone(),
                ai_hints: ai_ctx.testing_recommendations.clone(),
            }
        })
    }

    /// Extract JSDoc compatible information
    fn extract_jsdoc_info(&self, annotations: &[ExtractedAnnotation], doc_content: &Option<String>) -> Option<JSDocInfo> {
        let mut params = Vec::new();
        let mut returns = None;
        let mut throws = Vec::new();
        let mut examples = Vec::new();
        let mut see_also = Vec::new();
        let mut since = None;
        let mut deprecated = None;

        // Process annotations
        for annotation in annotations {
            match annotation.name.as_str() {
                "param" => {
                    if let Some(value) = &annotation.value {
                        params.push(JSDocParam {
                            name: value.clone(),
                            param_type: None, // TODO: Extract type information
                            description: None, // TODO: Extract description
                            optional: false, // TODO: Determine if optional
                        });
                    }
                }
                "returns" | "return" => {
                    returns = annotation.value.as_ref().map(|desc| JSDocReturn {
                        return_type: None, // TODO: Extract type information
                        description: Some(desc.clone()),
                    });
                }
                "throws" | "throw" => {
                    if let Some(value) = &annotation.value {
                        throws.push(JSDocThrows {
                            exception_type: None, // TODO: Extract exception type
                            description: Some(value.clone()),
                        });
                    }
                }
                "example" => {
                    if let Some(value) = &annotation.value {
                        examples.push(value.clone());
                    }
                }
                "see" => {
                    if let Some(value) = &annotation.value {
                        see_also.push(value.clone());
                    }
                }
                "since" => {
                    since = annotation.value.clone();
                }
                "deprecated" => {
                    deprecated = annotation.value.clone();
                }
                _ => {} // Ignore other annotations for JSDoc
            }
        }

        // Only create JSDoc info if we have relevant information
        if !params.is_empty() || returns.is_some() || !throws.is_empty() || 
           !examples.is_empty() || !see_also.is_empty() || since.is_some() || deprecated.is_some() {
            Some(JSDocInfo {
                description: doc_content.clone(),
                params,
                returns,
                throws,
                examples,
                see_also,
                since,
                deprecated,
            })
        } else {
            None
        }
    }

    /// Find annotation value by name
    fn find_annotation_value(&self, annotations: &[ExtractedAnnotation], name: &str) -> Option<String> {
        annotations.iter()
            .find(|ann| ann.name == name)
            .and_then(|ann| ann.value.clone())
    }

    /// Extract dependencies from annotations
    fn extract_dependencies(&self, annotations: &[ExtractedAnnotation]) -> Vec<String> {
        annotations.iter()
            .find(|ann| ann.name == "dependencies")
            .map(|ann| {
                // Parse dependencies from annotation value or arguments
                ann.arguments.iter()
                    .map(|arg| arg.value.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Map AST visibility to documentation visibility
    fn map_visibility(&self, visibility: &prism_ast::Visibility) -> ElementVisibility {
        match visibility {
            prism_ast::Visibility::Public => ElementVisibility::Public,
            prism_ast::Visibility::Private => ElementVisibility::Private,
            prism_ast::Visibility::Internal => ElementVisibility::Internal,
        }
    }
}

impl ExtractionStatistics {
    /// Create new extraction statistics
    pub fn new() -> Self {
        Self {
            total_elements: 0,
            documented_elements: 0,
            elements_with_required_annotations: 0,
            missing_documentation_count: 0,
            missing_annotation_count: 0,
            documentation_coverage: 0.0,
        }
    }

    /// Calculate coverage statistics
    pub fn calculate_coverage(&mut self, elements: &[DocumentationElement]) {
        self.documented_elements = elements.iter()
            .filter(|elem| elem.content.is_some())
            .count();

        self.elements_with_required_annotations = elements.iter()
            .filter(|elem| self.has_required_annotations(elem))
            .count();

        self.missing_documentation_count = self.total_elements - self.documented_elements;
        
        if self.total_elements > 0 {
            self.documentation_coverage = (self.documented_elements as f64 / self.total_elements as f64) * 100.0;
        }
    }

    /// Check if element has required annotations
    fn has_required_annotations(&self, element: &DocumentationElement) -> bool {
        match element.element_type {
            DocumentationElementType::Module => {
                element.annotations.iter().any(|ann| ann.name == "responsibility")
            }
            DocumentationElementType::Function => {
                element.annotations.iter().any(|ann| ann.name == "responsibility")
            }
            DocumentationElementType::Type => {
                element.annotations.iter().any(|ann| ann.name == "responsibility")
            }
            _ => true, // Other elements may not require specific annotations
        }
    }
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            extract_doc_comments: true,
            extract_annotations: true,
            extract_ai_context: true,
            extract_jsdoc_info: true,
            include_private_items: false,
        }
    }
}

impl Default for DocumentationExtractor {
    fn default() -> Self {
        Self::new()
    }
} 