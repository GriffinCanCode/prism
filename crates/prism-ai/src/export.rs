//! Metadata Export Framework
//!
//! This module provides export functionality for AI metadata in various formats
//! optimized for different AI tool consumption patterns.

use crate::{AIIntegrationError, ComprehensiveAIMetadata, ExportFormat};
use async_trait::async_trait;
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
        // For now, convert to JSON and then wrap in XML
        // A proper implementation would use a dedicated XML serialization library
        let json_data = serde_json::to_string_pretty(metadata)
            .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))?;
            
        Ok(format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<prism_ai_metadata>
    <format>xml</format>
    <version>{}</version>
    <exported_at>{}</exported_at>
    <json_representation><![CDATA[{}]]></json_representation>
</prism_ai_metadata>"#,
            metadata.version,
            metadata.exported_at,
            json_data
        ))
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
    async fn export(&self, _metadata: &ComprehensiveAIMetadata) -> Result<String, AIIntegrationError> {
        // For now, return an error as binary export requires additional dependencies
        Err(AIIntegrationError::SerializationError(
            "Binary export not yet implemented - requires MessagePack or similar binary format".to_string()
        ))
    }
    
    fn format(&self) -> ExportFormat {
        ExportFormat::Binary
    }
    
    fn file_extension(&self) -> &str {
        "bin"
    }
    
    fn mime_type(&self) -> &str {
        "application/octet-stream"
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