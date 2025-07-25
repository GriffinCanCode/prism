//! Documentation generation and export
//!
//! This module embodies the single concept of "Documentation Generation".
//! Following Prism's Conceptual Cohesion principle, this module is responsible
//! for ONE thing: generating documentation output in various formats.

use crate::{DocumentationError, DocumentationResult};
use crate::extraction::{ExtractedDocumentation, DocumentationElement, DocumentationElementType, ModuleDocumentation};
use crate::ai_integration::AIDocumentationMetadata;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Documentation generator for multiple output formats
#[derive(Debug)]
pub struct DocumentationGenerator {
    /// Generation configuration
    config: GenerationConfig,
    /// Format-specific generators
    generators: HashMap<OutputFormat, Box<dyn FormatGenerator>>,
}

/// Configuration for documentation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Output formats to generate
    pub output_formats: Vec<OutputFormat>,
    /// Include private elements in output
    pub include_private: bool,
    /// Include AI metadata in output
    pub include_ai_metadata: bool,
    /// Include examples in output
    pub include_examples: bool,
    /// Generate table of contents
    pub generate_toc: bool,
    /// Custom templates directory
    pub template_directory: Option<String>,
    /// Output directory
    pub output_directory: String,
    /// Custom generation options
    pub custom_options: HashMap<String, String>,
}

/// Output formats supported by the generator
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    /// HTML documentation
    HTML,
    /// Markdown documentation
    Markdown,
    /// JSON structured data
    JSON,
    /// YAML structured data
    YAML,
    /// XML documentation
    XML,
    /// PDF documentation
    PDF,
    /// Plain text documentation
    PlainText,
    /// Custom format
    Custom(String),
}

/// Format generator trait
pub trait FormatGenerator: Send + Sync + std::fmt::Debug {
    /// Generate documentation in the specific format
    fn generate(&self, docs: &ExtractedDocumentation, config: &GenerationConfig) -> DocumentationResult<GeneratedOutput>;
    
    /// Get the format this generator handles
    fn format(&self) -> OutputFormat;
    
    /// Get file extension for this format
    fn file_extension(&self) -> &str;
}

/// Generated documentation output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedDocumentation {
    /// Generated outputs by format
    pub outputs: HashMap<OutputFormat, GeneratedOutput>,
    /// Generation metadata
    pub metadata: GenerationMetadata,
    /// Generation statistics
    pub statistics: GenerationStatistics,
}

/// Individual generated output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedOutput {
    /// Output format
    pub format: OutputFormat,
    /// Generated content
    pub content: String,
    /// File name (if applicable)
    pub filename: Option<String>,
    /// Content type/MIME type
    pub content_type: String,
    /// Generation notes
    pub notes: Vec<String>,
}

/// Generation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Generation timestamp
    pub generated_at: String,
    /// Generator version
    pub generator_version: String,
    /// Source documentation version
    pub source_version: String,
    /// Generation configuration used
    pub config_used: GenerationConfig,
}

/// Generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStatistics {
    /// Total elements processed
    pub total_elements: usize,
    /// Elements included in output
    pub included_elements: usize,
    /// Formats generated
    pub formats_generated: Vec<OutputFormat>,
    /// Generation time (milliseconds)
    pub generation_time_ms: u64,
    /// Output sizes by format
    pub output_sizes: HashMap<OutputFormat, usize>,
}

/// HTML generator
#[derive(Debug)]
pub struct HTMLGenerator;

/// Markdown generator
#[derive(Debug)]
pub struct MarkdownGenerator;

/// JSON generator
#[derive(Debug)]
pub struct JSONGenerator;

impl DocumentationGenerator {
    /// Create a new documentation generator
    pub fn new(config: GenerationConfig) -> Self {
        let mut generator = Self {
            config,
            generators: HashMap::new(),
        };
        
        // Register default generators
        generator.register_default_generators();
        generator
    }

    /// Generate documentation from extracted documentation
    pub fn generate_from_extracted(&self, docs: &ExtractedDocumentation) -> DocumentationResult<GeneratedDocumentation> {
        let start_time = std::time::Instant::now();
        let mut outputs = HashMap::new();
        let mut output_sizes = HashMap::new();

        // Generate documentation in each requested format
        for format in &self.config.output_formats {
            if let Some(generator) = self.generators.get(format) {
                match generator.generate(docs, &self.config) {
                    Ok(output) => {
                        output_sizes.insert(format.clone(), output.content.len());
                        outputs.insert(format.clone(), output);
                    }
                    Err(e) => {
                        return Err(DocumentationError::ExtractionFailed {
                            reason: format!("Failed to generate {} format: {}", format.name(), e),
                        });
                    }
                }
            } else {
                return Err(DocumentationError::ExtractionFailed {
                    reason: format!("No generator available for format: {}", format.name()),
                });
            }
        }

        let generation_time = start_time.elapsed().as_millis() as u64;

        let metadata = GenerationMetadata {
            generated_at: chrono::Utc::now().to_rfc3339(),
            generator_version: "1.0.0".to_string(),
            source_version: "1.0.0".to_string(),
            config_used: self.config.clone(),
        };

        let statistics = GenerationStatistics {
            total_elements: docs.elements.len(),
            included_elements: self.count_included_elements(docs),
            formats_generated: self.config.output_formats.clone(),
            generation_time_ms: generation_time,
            output_sizes,
        };

        Ok(GeneratedDocumentation {
            outputs,
            metadata,
            statistics,
        })
    }

    /// Register default format generators
    fn register_default_generators(&mut self) {
        self.generators.insert(OutputFormat::HTML, Box::new(HTMLGenerator));
        self.generators.insert(OutputFormat::Markdown, Box::new(MarkdownGenerator));
        self.generators.insert(OutputFormat::JSON, Box::new(JSONGenerator));
    }

    /// Count elements that would be included in output
    fn count_included_elements(&self, docs: &ExtractedDocumentation) -> usize {
        if self.config.include_private {
            docs.elements.len()
        } else {
            docs.elements.iter()
                .filter(|elem| elem.visibility == crate::extraction::ElementVisibility::Public)
                .count()
        }
    }
}

impl FormatGenerator for HTMLGenerator {
    fn generate(&self, docs: &ExtractedDocumentation, config: &GenerationConfig) -> DocumentationResult<GeneratedOutput> {
        let mut html = String::new();
        
        // HTML header
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        
        // Title from module documentation
        let title = docs.module_documentation.as_ref()
            .map(|m| format!("{} Documentation", m.name))
            .unwrap_or_else(|| "Documentation".to_string());
        
        html.push_str(&format!("    <title>{}</title>\n", title));
        html.push_str("    <style>\n");
        html.push_str(include_str!("../templates/default.css"));
        html.push_str("    </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        
        // Main content
        html.push_str("    <div class=\"container\">\n");
        
        // Header
        html.push_str(&format!("        <h1>{}</h1>\n", title));
        
        // Module documentation
        if let Some(module_doc) = &docs.module_documentation {
            self.generate_module_section(&mut html, module_doc);
        }

        // Table of contents (if enabled)
        if config.generate_toc {
            self.generate_toc(&mut html, docs, config);
        }

        // Elements
        self.generate_elements_section(&mut html, docs, config)?;
        
        html.push_str("    </div>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");

        Ok(GeneratedOutput {
            format: OutputFormat::HTML,
            content: html,
            filename: Some("documentation.html".to_string()),
            content_type: "text/html".to_string(),
            notes: vec![],
        })
    }

    fn format(&self) -> OutputFormat {
        OutputFormat::HTML
    }

    fn file_extension(&self) -> &str {
        "html"
    }
}

impl HTMLGenerator {
    fn generate_module_section(&self, html: &mut String, module_doc: &ModuleDocumentation) {
        html.push_str("        <section class=\"module-info\">\n");
        
        if let Some(description) = &module_doc.description {
            html.push_str(&format!("            <p class=\"description\">{}</p>\n", self.escape_html(description)));
        }
        
        if let Some(responsibility) = &module_doc.responsibility {
            html.push_str("            <div class=\"responsibility\">\n");
            html.push_str("                <h3>Responsibility</h3>\n");
            html.push_str(&format!("                <p>{}</p>\n", self.escape_html(responsibility)));
            html.push_str("            </div>\n");
        }
        
        // Module metadata
        html.push_str("            <div class=\"metadata\">\n");
        if let Some(author) = &module_doc.author {
            html.push_str(&format!("                <div><strong>Author:</strong> {}</div>\n", self.escape_html(author)));
        }
        if let Some(version) = &module_doc.version {
            html.push_str(&format!("                <div><strong>Version:</strong> {}</div>\n", self.escape_html(version)));
        }
        if let Some(stability) = &module_doc.stability {
            html.push_str(&format!("                <div><strong>Stability:</strong> {}</div>\n", self.escape_html(stability)));
        }
        html.push_str("            </div>\n");
        
        html.push_str("        </section>\n");
    }

    fn generate_toc(&self, html: &mut String, docs: &ExtractedDocumentation, config: &GenerationConfig) {
        html.push_str("        <nav class=\"table-of-contents\">\n");
        html.push_str("            <h2>Table of Contents</h2>\n");
        html.push_str("            <ul>\n");
        
        // Group elements by type
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut constants = Vec::new();
        let mut modules = Vec::new();
        
        for element in &docs.elements {
            if !config.include_private && element.visibility != crate::extraction::ElementVisibility::Public {
                continue;
            }
            
            match element.element_type {
                DocumentationElementType::Function => functions.push(element),
                DocumentationElementType::Type => types.push(element),
                DocumentationElementType::Constant => constants.push(element),
                DocumentationElementType::Module => modules.push(element),
                _ => {}
            }
        }
        
        if !modules.is_empty() {
            html.push_str("                <li><a href=\"#modules\">Modules</a></li>\n");
        }
        if !types.is_empty() {
            html.push_str("                <li><a href=\"#types\">Types</a></li>\n");
        }
        if !functions.is_empty() {
            html.push_str("                <li><a href=\"#functions\">Functions</a></li>\n");
        }
        if !constants.is_empty() {
            html.push_str("                <li><a href=\"#constants\">Constants</a></li>\n");
        }
        
        html.push_str("            </ul>\n");
        html.push_str("        </nav>\n");
    }

    fn generate_elements_section(&self, html: &mut String, docs: &ExtractedDocumentation, config: &GenerationConfig) -> DocumentationResult<()> {
        // Group elements by type
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut constants = Vec::new();
        let mut modules = Vec::new();
        
        for element in &docs.elements {
            if !config.include_private && element.visibility != crate::extraction::ElementVisibility::Public {
                continue;
            }
            
            match element.element_type {
                DocumentationElementType::Function => functions.push(element),
                DocumentationElementType::Type => types.push(element),
                DocumentationElementType::Constant => constants.push(element),
                DocumentationElementType::Module => modules.push(element),
                _ => {}
            }
        }

        // Generate sections
        if !modules.is_empty() {
            self.generate_element_type_section(html, "Modules", "modules", &modules, config)?;
        }
        if !types.is_empty() {
            self.generate_element_type_section(html, "Types", "types", &types, config)?;
        }
        if !functions.is_empty() {
            self.generate_element_type_section(html, "Functions", "functions", &functions, config)?;
        }
        if !constants.is_empty() {
            self.generate_element_type_section(html, "Constants", "constants", &constants, config)?;
        }

        Ok(())
    }

    fn generate_element_type_section(
        &self,
        html: &mut String,
        title: &str,
        id: &str,
        elements: &[&DocumentationElement],
        config: &GenerationConfig,
    ) -> DocumentationResult<()> {
        html.push_str(&format!("        <section id=\"{}\" class=\"element-section\">\n", id));
        html.push_str(&format!("            <h2>{}</h2>\n", title));
        
        for element in elements {
            self.generate_element_documentation(html, element, config)?;
        }
        
        html.push_str("        </section>\n");
        Ok(())
    }

    fn generate_element_documentation(
        &self,
        html: &mut String,
        element: &DocumentationElement,
        config: &GenerationConfig,
    ) -> DocumentationResult<()> {
        let element_id = self.generate_element_id(&element.name);
        
        html.push_str(&format!("            <div id=\"{}\" class=\"element\">\n", element_id));
        html.push_str(&format!("                <h3 class=\"element-name\">{}</h3>\n", self.escape_html(&element.name)));
        
        // Element description
        if let Some(content) = &element.content {
            html.push_str(&format!("                <p class=\"element-description\">{}</p>\n", self.escape_html(content)));
        }

        // Annotations
        if !element.annotations.is_empty() {
            html.push_str("                <div class=\"annotations\">\n");
            for annotation in &element.annotations {
                let annotation_class = format!("annotation-{}", annotation.name);
                html.push_str(&format!("                    <div class=\"annotation {}\">\n", annotation_class));
                html.push_str(&format!("                        <strong>@{}:</strong> ", annotation.name));
                if let Some(value) = &annotation.value {
                    html.push_str(&self.escape_html(value));
                }
                html.push_str("\n                    </div>\n");
            }
            html.push_str("                </div>\n");
        }

        // Examples (if enabled and available)
        if config.include_examples {
            if let Some(jsdoc) = &element.jsdoc_info {
                if !jsdoc.examples.is_empty() {
                    html.push_str("                <div class=\"examples\">\n");
                    html.push_str("                    <h4>Examples</h4>\n");
                    for example in &jsdoc.examples {
                        html.push_str("                    <pre><code>");
                        html.push_str(&self.escape_html(example));
                        html.push_str("</code></pre>\n");
                    }
                    html.push_str("                </div>\n");
                }
            }
        }

        html.push_str("            </div>\n");
        Ok(())
    }

    fn escape_html(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }

    fn generate_element_id(&self, name: &str) -> String {
        name.to_lowercase().replace(' ', "-")
    }
}

impl FormatGenerator for MarkdownGenerator {
    fn generate(&self, docs: &ExtractedDocumentation, config: &GenerationConfig) -> DocumentationResult<GeneratedOutput> {
        let mut md = String::new();
        
        // Title
        let title = docs.module_documentation.as_ref()
            .map(|m| m.name.clone())
            .unwrap_or_else(|| "Documentation".to_string());
        
        md.push_str(&format!("# {}\n\n", title));
        
        // Module documentation
        if let Some(module_doc) = &docs.module_documentation {
            self.generate_module_section(&mut md, module_doc);
        }

        // Table of contents (if enabled)
        if config.generate_toc {
            self.generate_toc(&mut md, docs, config);
        }

        // Elements
        self.generate_elements_section(&mut md, docs, config)?;

        Ok(GeneratedOutput {
            format: OutputFormat::Markdown,
            content: md,
            filename: Some("documentation.md".to_string()),
            content_type: "text/markdown".to_string(),
            notes: vec![],
        })
    }

    fn format(&self) -> OutputFormat {
        OutputFormat::Markdown
    }

    fn file_extension(&self) -> &str {
        "md"
    }
}

impl MarkdownGenerator {
    fn generate_module_section(&self, md: &mut String, module_doc: &ModuleDocumentation) {
        if let Some(description) = &module_doc.description {
            md.push_str(&format!("{}\n\n", description));
        }
        
        if let Some(responsibility) = &module_doc.responsibility {
            md.push_str("## Responsibility\n\n");
            md.push_str(&format!("{}\n\n", responsibility));
        }
        
        // Module metadata
        md.push_str("## Module Information\n\n");
        if let Some(author) = &module_doc.author {
            md.push_str(&format!("**Author:** {}\n\n", author));
        }
        if let Some(version) = &module_doc.version {
            md.push_str(&format!("**Version:** {}\n\n", version));
        }
        if let Some(stability) = &module_doc.stability {
            md.push_str(&format!("**Stability:** {}\n\n", stability));
        }
    }

    fn generate_toc(&self, md: &mut String, docs: &ExtractedDocumentation, config: &GenerationConfig) {
        md.push_str("## Table of Contents\n\n");
        
        // Group elements by type
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut constants = Vec::new();
        let mut modules = Vec::new();
        
        for element in &docs.elements {
            if !config.include_private && element.visibility != crate::extraction::ElementVisibility::Public {
                continue;
            }
            
            match element.element_type {
                DocumentationElementType::Function => functions.push(element),
                DocumentationElementType::Type => types.push(element),
                DocumentationElementType::Constant => constants.push(element),
                DocumentationElementType::Module => modules.push(element),
                _ => {}
            }
        }
        
        if !modules.is_empty() {
            md.push_str("- [Modules](#modules)\n");
        }
        if !types.is_empty() {
            md.push_str("- [Types](#types)\n");
        }
        if !functions.is_empty() {
            md.push_str("- [Functions](#functions)\n");
        }
        if !constants.is_empty() {
            md.push_str("- [Constants](#constants)\n");
        }
        
        md.push_str("\n");
    }

    fn generate_elements_section(&self, md: &mut String, docs: &ExtractedDocumentation, config: &GenerationConfig) -> DocumentationResult<()> {
        // Group elements by type
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut constants = Vec::new();
        let mut modules = Vec::new();
        
        for element in &docs.elements {
            if !config.include_private && element.visibility != crate::extraction::ElementVisibility::Public {
                continue;
            }
            
            match element.element_type {
                DocumentationElementType::Function => functions.push(element),
                DocumentationElementType::Type => types.push(element),
                DocumentationElementType::Constant => constants.push(element),
                DocumentationElementType::Module => modules.push(element),
                _ => {}
            }
        }

        // Generate sections
        if !modules.is_empty() {
            self.generate_element_type_section(md, "Modules", &modules, config)?;
        }
        if !types.is_empty() {
            self.generate_element_type_section(md, "Types", &types, config)?;
        }
        if !functions.is_empty() {
            self.generate_element_type_section(md, "Functions", &functions, config)?;
        }
        if !constants.is_empty() {
            self.generate_element_type_section(md, "Constants", &constants, config)?;
        }

        Ok(())
    }

    fn generate_element_type_section(
        &self,
        md: &mut String,
        title: &str,
        elements: &[&DocumentationElement],
        config: &GenerationConfig,
    ) -> DocumentationResult<()> {
        md.push_str(&format!("## {}\n\n", title));
        
        for element in elements {
            self.generate_element_documentation(md, element, config)?;
        }
        
        Ok(())
    }

    fn generate_element_documentation(
        &self,
        md: &mut String,
        element: &DocumentationElement,
        config: &GenerationConfig,
    ) -> DocumentationResult<()> {
        md.push_str(&format!("### {}\n\n", element.name));
        
        // Element description
        if let Some(content) = &element.content {
            md.push_str(&format!("{}\n\n", content));
        }

        // Annotations
        if !element.annotations.is_empty() {
            for annotation in &element.annotations {
                match annotation.name.as_str() {
                    "responsibility" => {
                        if let Some(value) = &annotation.value {
                            md.push_str(&format!("**Responsibility:** {}\n\n", value));
                        }
                    }
                    "param" => {
                        if let Some(value) = &annotation.value {
                            md.push_str(&format!("**Parameter:** {}\n\n", value));
                        }
                    }
                    "returns" => {
                        if let Some(value) = &annotation.value {
                            md.push_str(&format!("**Returns:** {}\n\n", value));
                        }
                    }
                    "throws" => {
                        if let Some(value) = &annotation.value {
                            md.push_str(&format!("**Throws:** {}\n\n", value));
                        }
                    }
                    _ => {
                        if let Some(value) = &annotation.value {
                            md.push_str(&format!("**@{}:** {}\n\n", annotation.name, value));
                        }
                    }
                }
            }
        }

        // Examples (if enabled and available)
        if config.include_examples {
            if let Some(jsdoc) = &element.jsdoc_info {
                if !jsdoc.examples.is_empty() {
                    md.push_str("**Examples:**\n\n");
                    for example in &jsdoc.examples {
                        md.push_str("```prism\n");
                        md.push_str(example);
                        md.push_str("\n```\n\n");
                    }
                }
            }
        }

        md.push_str("---\n\n");
        Ok(())
    }
}

impl FormatGenerator for JSONGenerator {
    fn generate(&self, docs: &ExtractedDocumentation, _config: &GenerationConfig) -> DocumentationResult<GeneratedOutput> {
        let json = serde_json::to_string_pretty(docs)
            .map_err(|e| DocumentationError::ExtractionFailed {
                reason: format!("JSON serialization failed: {}", e),
            })?;

        Ok(GeneratedOutput {
            format: OutputFormat::JSON,
            content: json,
            filename: Some("documentation.json".to_string()),
            content_type: "application/json".to_string(),
            notes: vec![],
        })
    }

    fn format(&self) -> OutputFormat {
        OutputFormat::JSON
    }

    fn file_extension(&self) -> &str {
        "json"
    }
}

impl OutputFormat {
    /// Get the name of the output format
    pub fn name(&self) -> String {
        match self {
            OutputFormat::HTML => "HTML".to_string(),
            OutputFormat::Markdown => "Markdown".to_string(),
            OutputFormat::JSON => "JSON".to_string(),
            OutputFormat::YAML => "YAML".to_string(),
            OutputFormat::XML => "XML".to_string(),
            OutputFormat::PDF => "PDF".to_string(),
            OutputFormat::PlainText => "Plain Text".to_string(),
            OutputFormat::Custom(name) => name.clone(),
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            output_formats: vec![OutputFormat::HTML, OutputFormat::Markdown],
            include_private: false,
            include_ai_metadata: true,
            include_examples: true,
            generate_toc: true,
            template_directory: None,
            output_directory: "docs".to_string(),
            custom_options: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::{DocumentationElement, DocumentationElementType, ExtractedAnnotation, ElementVisibility, ExtractionStatistics};

    #[test]
    fn test_html_generation() {
        let generator = HTMLGenerator;
        
        let docs = create_test_documentation();
        let config = GenerationConfig::default();
        
        let output = generator.generate(&docs, &config).unwrap();
        
        assert_eq!(output.format, OutputFormat::HTML);
        assert!(output.content.contains("<!DOCTYPE html>"));
        assert!(output.content.contains("Test Module Documentation"));
        assert!(output.content.contains("testFunction"));
        assert_eq!(output.content_type, "text/html");
    }

    #[test]
    fn test_markdown_generation() {
        let generator = MarkdownGenerator;
        
        let docs = create_test_documentation();
        let config = GenerationConfig::default();
        
        let output = generator.generate(&docs, &config).unwrap();
        
        assert_eq!(output.format, OutputFormat::Markdown);
        assert!(output.content.contains("# Test Module"));
        assert!(output.content.contains("## Functions"));
        assert!(output.content.contains("### testFunction"));
        assert_eq!(output.content_type, "text/markdown");
    }

    #[test]
    fn test_json_generation() {
        let generator = JSONGenerator;
        
        let docs = create_test_documentation();
        let config = GenerationConfig::default();
        
        let output = generator.generate(&docs, &config).unwrap();
        
        assert_eq!(output.format, OutputFormat::JSON);
        assert!(output.content.contains("\"name\": \"testFunction\""));
        assert_eq!(output.content_type, "application/json");
        
        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&output.content).unwrap();
        assert!(parsed.is_object());
    }

    fn create_test_documentation() -> ExtractedDocumentation {
        use crate::extraction::{ModuleDocumentation};
        use prism_common::span::Span;
        
        ExtractedDocumentation {
            elements: vec![
                DocumentationElement {
                    element_type: DocumentationElementType::Function,
                    name: "testFunction".to_string(),
                    content: Some("Test function description".to_string()),
                    annotations: vec![
                        ExtractedAnnotation {
                            name: "responsibility".to_string(),
                            value: Some("Tests functionality".to_string()),
                            arguments: vec![],
                            location: Span::dummy(),
                        },
                    ],
                    location: Span::dummy(),
                    visibility: ElementVisibility::Public,
                    ai_context: None,
                    jsdoc_info: None,
                },
            ],
            module_documentation: Some(ModuleDocumentation {
                name: "Test Module".to_string(),
                description: Some("Test module for documentation generation".to_string()),
                responsibility: Some("Testing documentation system".to_string()),
                author: Some("Test Author".to_string()),
                version: Some("1.0.0".to_string()),
                stability: Some("Stable".to_string()),
                dependencies: vec![],
            }),
            statistics: ExtractionStatistics {
                total_elements: 1,
                documented_elements: 1,
                elements_with_required_annotations: 1,
                missing_documentation_count: 0,
                missing_annotation_count: 0,
                documentation_coverage: 100.0,
            },
        }
    }
} 