//! Metadata Export Framework
//!
//! This module provides export functionality for AI metadata in various formats
//! optimized for different AI tool consumption patterns.

use crate::{AIIntegrationError, ComprehensiveAIMetadata, ExportFormat};
use async_trait::async_trait;
use base64::Engine;
use serde_json;
use serde_yaml;
use std::collections::HashMap;

/// Trait for exporting metadata in a specific format
#[async_trait]
pub trait MetadataExporter: Send + Sync {
    /// Export metadata to a string representation
    async fn export(&self, metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError>;
    
    /// Get the format this exporter handles
    fn format(&self) -> ExportFormat;
    
    /// Get the file extension for this format
    fn file_extension(&self) -> &str;
    
    /// Get the MIME type for this format
    fn mime_type(&self) -> &str;
}

/// JSON exporter for AI metadata
#[derive(Debug)]
pub struct JsonExporter {
    pretty_print: bool,
}

impl JsonExporter {
    pub fn new() -> Self {
        Self { pretty_print: true }
    }
    
    pub fn with_pretty_print(pretty_print: bool) -> Self {
        Self { pretty_print }
    }
}

#[async_trait]
impl MetadataExporter for JsonExporter {
    async fn export(&self, metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        if self.pretty_print {
            serde_json::to_string_pretty(metadata)
                .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
        } else {
            serde_json::to_string(metadata)
                .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
        }
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::Json
    }
    
    fn file_extension(&self) -> &str {
        "json"
    }
    
    fn mime_type(&self) -> &str {
        "application/json"
    }
}

/// YAML exporter for AI metadata
#[derive(Debug)]
pub struct YamlExporter;

impl YamlExporter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataExporter for YamlExporter {
    async fn export(&self, metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        serde_yaml::to_string(metadata)
            .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::Yaml
    }
    
    fn file_extension(&self) -> &str {
        "yaml"
    }
    
    fn mime_type(&self) -> &str {
        "application/x-yaml"
    }
}

/// XML exporter for AI metadata
#[derive(Debug)]
pub struct XmlExporter;

impl XmlExporter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataExporter for XmlExporter {
    async fn export(&self, metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        let mut xml_output = String::new();
        
        // XML header
        xml_output.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml_output.push('\n');
        
        // Root element
        xml_output.push_str("<prism_ai_metadata>\n");
        
        // Metadata header
        xml_output.push_str("  <header>\n");
        xml_output.push_str(&format!("    <version>{}</version>\n", self.escape_xml(&metadata.version)));
        xml_output.push_str(&format!("    <exported_at>{}</exported_at>\n", self.escape_xml(&metadata.exported_at)));
        xml_output.push_str("    <format>xml</format>\n");
        xml_output.push_str("  </header>\n");
        
        // Project information
        xml_output.push_str("  <project_info>\n");
        xml_output.push_str(&format!("    <name>{}</name>\n", self.escape_xml(&metadata.project_info.name)));
        if let Some(ref version) = metadata.project_info.version {
            xml_output.push_str(&format!("    <project_version>{}</project_version>\n", self.escape_xml(version)));
        }
        xml_output.push_str(&format!("    <root_path>{}</root_path>\n", self.escape_xml(&metadata.project_info.root_path.to_string_lossy())));
        
        // Source files
        xml_output.push_str("    <source_files>\n");
        for source_file in &metadata.project_info.source_files {
            xml_output.push_str("      <file>\n");
            xml_output.push_str(&format!("        <path>{}</path>\n", self.escape_xml(&source_file.path.to_string_lossy())));
            xml_output.push_str(&format!("        <size>{}</size>\n", source_file.size));
            xml_output.push_str(&format!("        <language>{}</language>\n", self.escape_xml(&source_file.language)));
            xml_output.push_str(&format!("        <hash>{}</hash>\n", self.escape_xml(&source_file.hash)));
            xml_output.push_str("      </file>\n");
        }
        xml_output.push_str("    </source_files>\n");
        
        // Dependencies
        xml_output.push_str("    <dependencies>\n");
        for dependency in &metadata.project_info.dependencies {
            xml_output.push_str("      <dependency>\n");
            xml_output.push_str(&format!("        <name>{}</name>\n", self.escape_xml(&dependency.name)));
            xml_output.push_str(&format!("        <version>{}</version>\n", self.escape_xml(&dependency.version)));
            xml_output.push_str(&format!("        <type>{:?}</type>\n", dependency.dependency_type));
            xml_output.push_str(&format!("        <source>{}</source>\n", self.escape_xml(&dependency.source)));
            xml_output.push_str("      </dependency>\n");
        }
        xml_output.push_str("    </dependencies>\n");
        xml_output.push_str("  </project_info>\n");
        
        // Quality metrics
        xml_output.push_str("  <quality_metrics>\n");
        xml_output.push_str(&format!("    <lines_of_code>{}</lines_of_code>\n", metadata.quality_metrics.lines_of_code));
        xml_output.push_str(&format!("    <cyclomatic_complexity>{}</cyclomatic_complexity>\n", metadata.quality_metrics.cyclomatic_complexity));
        xml_output.push_str(&format!("    <cognitive_complexity>{}</cognitive_complexity>\n", metadata.quality_metrics.cognitive_complexity));
        xml_output.push_str(&format!("    <test_coverage>{}</test_coverage>\n", metadata.quality_metrics.test_coverage));
        xml_output.push_str(&format!("    <documentation_coverage>{}</documentation_coverage>\n", metadata.quality_metrics.documentation_coverage));
        xml_output.push_str(&format!("    <technical_debt_ratio>{}</technical_debt_ratio>\n", metadata.quality_metrics.technical_debt_ratio));
        xml_output.push_str("  </quality_metrics>\n");
        
        // Business context (if available)
        if let Some(ref business_context) = metadata.business_context {
            xml_output.push_str("  <business_context>\n");
            if let Some(ref domain) = business_context.domain {
                xml_output.push_str(&format!("    <domain>{}</domain>\n", self.escape_xml(domain)));
            }
            
            xml_output.push_str("    <capabilities>\n");
            for capability in &business_context.capabilities {
                xml_output.push_str(&format!("      <capability>{}</capability>\n", self.escape_xml(capability)));
            }
            xml_output.push_str("    </capabilities>\n");
            
            xml_output.push_str("    <rules>\n");
            for rule in &business_context.rules {
                xml_output.push_str(&format!("      <rule>{}</rule>\n", self.escape_xml(rule)));
            }
            xml_output.push_str("    </rules>\n");
            xml_output.push_str("  </business_context>\n");
        }
        
        // Cross-system relationships
        xml_output.push_str("  <relationships>\n");
        for relationship in &metadata.relationships {
            xml_output.push_str("    <relationship>\n");
            xml_output.push_str("      <source>\n");
            xml_output.push_str(&format!("        <system>{}</system>\n", self.escape_xml(&relationship.source.system)));
            xml_output.push_str(&format!("        <component_id>{}</component_id>\n", self.escape_xml(&relationship.source.component_id)));
            xml_output.push_str(&format!("        <component_type>{}</component_type>\n", self.escape_xml(&relationship.source.component_type)));
            xml_output.push_str("      </source>\n");
            xml_output.push_str("      <target>\n");
            xml_output.push_str(&format!("        <system>{}</system>\n", self.escape_xml(&relationship.target.system)));
            xml_output.push_str(&format!("        <component_id>{}</component_id>\n", self.escape_xml(&relationship.target.component_id)));
            xml_output.push_str(&format!("        <component_type>{}</component_type>\n", self.escape_xml(&relationship.target.component_type)));
            xml_output.push_str("      </target>\n");
            xml_output.push_str(&format!("      <relationship_type>{:?}</relationship_type>\n", relationship.relationship_type));
            xml_output.push_str(&format!("      <strength>{}</strength>\n", relationship.strength));
            xml_output.push_str(&format!("      <description>{}</description>\n", self.escape_xml(&relationship.description)));
            xml_output.push_str("    </relationship>\n");
        }
        xml_output.push_str("  </relationships>\n");
        
        // Close root element
        xml_output.push_str("</prism_ai_metadata>\n");
        
        Ok(xml_output)
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::Xml
    }
    
    fn file_extension(&self) -> &str {
        "xml"
    }
    
    fn mime_type(&self) -> &str {
        "application/xml"
    }
}

impl XmlExporter {
    /// Escape XML special characters
    fn escape_xml(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

/// Binary exporter for AI metadata (using MessagePack)
#[derive(Debug)]
pub struct BinaryExporter;

impl BinaryExporter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataExporter for BinaryExporter {
    async fn export(&self, metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        // Serialize to MessagePack binary format
        let binary_data = rmp_serde::to_vec(metadata)
            .map_err(|e| AIIntegrationError::SerializationError(
                format!("MessagePack serialization failed: {}", e)
            ))?;
        
        // Encode as base64 string for text-based transport
        let base64_encoded = base64::engine::general_purpose::STANDARD.encode(&binary_data);
        
        // Wrap in a structured format for AI consumption
        let wrapped_output = serde_json::json!({
            "format": "binary",
            "encoding": "messagepack+base64",
            "size_bytes": binary_data.len(),
            "data": base64_encoded,
            "metadata": {
                "compressed": false,
                "version": metadata.version,
                "exported_at": metadata.exported_at
            }
        });
        
        serde_json::to_string_pretty(&wrapped_output)
            .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::Binary
    }
    
    fn file_extension(&self) -> &str {
        "msgpack.json"
    }
    
    fn mime_type(&self) -> &str {
        "application/json"
    }
}

/// OpenAPI exporter for AI metadata
#[derive(Debug)]
pub struct OpenApiExporter;

impl OpenApiExporter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataExporter for OpenApiExporter {
    async fn export(&self, metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        // Generate OpenAPI specification that describes the metadata structure
        let openapi_spec = serde_json::json!({
            "openapi": "3.0.0",
            "info": {
                "title": "Prism AI Metadata API",
                                             "version": metadata.version,
                             "description": "API specification for Prism AI metadata export"
                         },
                         "paths": {
                             "/metadata": {
                                 "get": {
                                     "summary": "Get comprehensive AI metadata",
                                     "responses": {
                                         "200": {
                                             "description": "Comprehensive AI metadata",
                                             "content": {
                                                 "application/json": {
                                                     "schema": {
                                                         "$ref": "#/components/schemas/ComprehensiveAIMetadata"
                                                     }
                                                 }
                                             }
                                         }
                                     }
                                 }
                             }
                         },
                         "components": {
                             "schemas": {
                                 "ComprehensiveAIMetadata": {
                                     "type": "object",
                                     "properties": {
                                         "version": { "type": "string" },
                                         "exported_at": { "type": "string", "format": "date-time" },
                                         "project_info": { "$ref": "#/components/schemas/ProjectInfo" },
                                         "quality_metrics": { "$ref": "#/components/schemas/QualityMetrics" }
                                     }
                                 },
                                 "ProjectInfo": {
                                     "type": "object",
                                     "properties": {
                                         "name": { "type": "string" },
                                         "version": { "type": "string" },
                                         "root_path": { "type": "string" }
                                     }
                                 },
                                 "QualityMetrics": {
                                     "type": "object",
                                     "properties": {
                                         "lines_of_code": { "type": "integer", "minimum": 0 },
                                         "cyclomatic_complexity": { "type": "integer", "minimum": 0 },
                                         "test_coverage": { "type": "number", "minimum": 0, "maximum": 100 }
                                     }
                                 }
                             }
                         }
        });
        
        serde_json::to_string_pretty(&openapi_spec)
            .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::OpenApi
    }
    
    fn file_extension(&self) -> &str {
        "openapi.json"
    }
    
    fn mime_type(&self) -> &str {
        "application/json"
    }
}

/// GraphQL exporter for AI metadata
#[derive(Debug)]
pub struct GraphQLExporter;

impl GraphQLExporter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MetadataExporter for GraphQLExporter {
    async fn export(&self, _metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        // Generate GraphQL schema for the metadata
        let schema = r#"
# Prism AI Metadata GraphQL Schema

 type Query {
   metadata: ComprehensiveAIMetadata
   projectInfo: ProjectInfo
   qualityMetrics: QualityMetrics
 }
 
 type ComprehensiveAIMetadata {
   version: String!
   exportedAt: String!
   projectInfo: ProjectInfo!
   qualityMetrics: QualityMetrics!
   relationships: [CrossSystemRelationship!]!
 }

type ProjectInfo {
  name: String!
  version: String
  rootPath: String!
  sourceFiles: [SourceFileInfo!]!
  dependencies: [DependencyInfo!]!
}

type SourceFileInfo {
  path: String!
  size: Int!
  lastModified: String!
  language: String!
  hash: String!
}

type DependencyInfo {
  name: String!
  version: String!
  dependencyType: DependencyType!
  source: String!
}

 enum DependencyType {
   DIRECT
   TRANSITIVE
   DEVELOPMENT
   OPTIONAL
 }
 
 type QualityMetrics {
  linesOfCode: Int!
  cyclomaticComplexity: Int!
  cognitiveComplexity: Int!
  testCoverage: Float!
  documentationCoverage: Float!
  technicalDebtRatio: Float!
}

type CrossSystemRelationship {
  source: ComponentReference!
  target: ComponentReference!
  relationshipType: RelationshipType!
  strength: Float!
  description: String!
}

type ComponentReference {
  system: String!
  componentId: String!
  componentType: String!
}

enum RelationshipType {
  DEPENDENCY
  COMPOSITION
  INHERITANCE
  USAGE
  ASSOCIATION
  IMPLEMENTATION
  DATA_FLOW
  CONTROL_FLOW
}
"#;
        
        Ok(schema.to_string())
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::GraphQL
    }
    
    fn file_extension(&self) -> &str {
        "graphql"
    }
    
    fn mime_type(&self) -> &str {
        "application/graphql"
    }
}

/// Export manager that coordinates multiple exporters
pub struct ExportManager {
    exporters: HashMap<ExportFormat, Box<dyn MetadataExporter>>,
}

impl ExportManager {
    /// Create a new export manager with default exporters
    pub fn new() -> Self {
        let mut exporters: HashMap<ExportFormat, Box<dyn MetadataExporter>> = HashMap::new();
        
        exporters.insert(ExportFormat::Json, Box::new(JsonExporter::new()));
        exporters.insert(ExportFormat::Yaml, Box::new(YamlExporter::new()));
        exporters.insert(ExportFormat::Xml, Box::new(XmlExporter::new()));
        exporters.insert(ExportFormat::Binary, Box::new(BinaryExporter::new()));
        exporters.insert(ExportFormat::OpenApi, Box::new(OpenApiExporter::new()));
        exporters.insert(ExportFormat::GraphQL, Box::new(GraphQLExporter::new()));
        
        Self { exporters }
    }
    
    /// Register a custom exporter
    pub fn register_exporter(&mut self, format: ExportFormat, exporter: Box<dyn MetadataExporter>) {
        self.exporters.insert(format, exporter);
    }
    
    /// Export metadata in multiple formats
    pub async fn export_multiple(
        &self,
        metadata: &ComprehensiveAIMetadata,
        formats: &[ExportFormat],
    ) -> Result<HashMap<ExportFormat, String>, AIIntegrationError> {
        let mut results = HashMap::new();
        
        for format in formats {
            if let Some(exporter) = self.exporters.get(format) {
                match exporter.export(metadata).await {
                    Ok(exported) => {
                        results.insert(format.clone(), exported);
                    }
                    Err(e) => {
                        return Err(AIIntegrationError::ExportFailed {
                            format: format.clone(),
                            reason: e.to_string(),
                        });
                    }
                }
            } else {
                return Err(AIIntegrationError::ExportFailed {
                    format: format.clone(),
                    reason: "No exporter registered for format".to_string(),
                });
            }
        }
        
        Ok(results)
    }
    
    /// Export metadata in a single format
    pub async fn export_single(
        &self,
        metadata: &ComprehensiveAIMetadata,
        format: &ExportFormat,
    ) -> Result<String, AIIntegrationError> {
        if let Some(exporter) = self.exporters.get(format) {
            exporter.export(metadata).await
        } else {
            Err(AIIntegrationError::ExportFailed {
                format: format.clone(),
                reason: "No exporter registered for format".to_string(),
            })
        }
    }
    
    /// Get file extension for a format
    pub fn get_file_extension(&self, format: &ExportFormat) -> Option<&str> {
        self.exporters.get(format).map(|e| e.file_extension())
    }
    
    /// Get MIME type for a format
    pub fn get_mime_type(&self, format: &ExportFormat) -> Option<&str> {
        self.exporters.get(format).map(|e| e.mime_type())
    }
}

impl Default for ExportManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for JsonExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for YamlExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for XmlExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BinaryExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OpenApiExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for GraphQLExporter {
    fn default() -> Self {
        Self::new()
    }
} 