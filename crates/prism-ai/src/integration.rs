//! Integration Utilities
//!
//! This module provides utilities for integrating the AI metadata export system
//! with external AI tools and development environments.

use crate::{AIIntegrationError, ComprehensiveAIMetadata, ExportFormat};
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Integration configuration for external AI tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIToolIntegration {
    /// Tool name
    pub tool_name: String,
    /// Tool version
    pub tool_version: Option<String>,
    /// Preferred export formats
    pub preferred_formats: Vec<ExportFormat>,
    /// Integration endpoints
    pub endpoints: Vec<IntegrationEndpoint>,
    /// Authentication configuration
    pub auth_config: Option<AuthConfig>,
}

/// Integration endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEndpoint {
    /// Endpoint name
    pub name: String,
    /// Endpoint URL
    pub url: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Headers to include
    pub headers: Vec<(String, String)>,
    /// Expected format
    pub format: ExportFormat,
}

/// HTTP methods for integration endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// Configuration parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    /// No authentication
    None,
    /// API key authentication
    ApiKey,
    /// Bearer token authentication
    BearerToken,
    /// Basic authentication
    Basic,
    /// OAuth 2.0
    OAuth2,
}

/// Integration manager for external AI tools
#[derive(Debug)]
pub struct IntegrationManager {
    /// Registered integrations
    integrations: Vec<AIToolIntegration>,
    /// Output directory for exports
    output_directory: Option<PathBuf>,
}

impl IntegrationManager {
    /// Create a new integration manager
    pub fn new() -> Self {
        Self {
            integrations: Vec::new(),
            output_directory: None,
        }
    }
    
    /// Set the output directory for exports
    pub fn set_output_directory(&mut self, directory: PathBuf) {
        self.output_directory = Some(directory);
    }
    
    /// Register an AI tool integration
    pub fn register_integration(&mut self, integration: AIToolIntegration) {
        self.integrations.push(integration);
    }
    
    /// Export metadata for all registered integrations
    pub async fn export_for_all_integrations(
        &self,
        metadata: &ComprehensiveAIMetadata,
    ) -> Result<Vec<IntegrationResult>, AIIntegrationError> {
        let mut results = Vec::new();
        
        for integration in &self.integrations {
            match self.export_for_integration(metadata, integration).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    results.push(IntegrationResult {
                        tool_name: integration.tool_name.clone(),
                        success: false,
                        error: Some(e.to_string()),
                        exported_files: Vec::new(),
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Export metadata for a specific integration
    async fn export_for_integration(
        &self,
        metadata: &ComprehensiveAIMetadata,
        integration: &AIToolIntegration,
    ) -> Result<IntegrationResult, AIIntegrationError> {
        let mut exported_files = Vec::new();
        
        for format in &integration.preferred_formats {
            let exported_content = match format {
                ExportFormat::Json => serde_json::to_string_pretty(metadata)
                    .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))?,
                ExportFormat::Yaml => serde_yaml::to_string(metadata)
                    .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))?,
                _ => {
                    return Err(AIIntegrationError::ExportFailed {
                        format: format.clone(),
                        reason: "Format not supported for integration".to_string(),
                    });
                }
            };
            
            // Write to file if output directory is set
            if let Some(output_dir) = &self.output_directory {
                let file_extension = match format {
                    ExportFormat::Json => "json",
                    ExportFormat::Yaml => "yaml",
                    _ => "txt",
                };
                
                let filename = format!("{}_metadata.{}", integration.tool_name, file_extension);
                let file_path = output_dir.join(filename);
                
                tokio::fs::write(&file_path, exported_content).await?;
                exported_files.push(file_path);
            }
        }
        
        Ok(IntegrationResult {
            tool_name: integration.tool_name.clone(),
            success: true,
            error: None,
            exported_files,
        })
    }
}

/// Result of an integration export
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// Tool name
    pub tool_name: String,
    /// Whether the export was successful
    pub success: bool,
    /// Error message if export failed
    pub error: Option<String>,
    /// Files that were exported
    pub exported_files: Vec<PathBuf>,
}

impl Default for IntegrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Common AI tool integrations
impl AIToolIntegration {
    /// Create integration for VS Code AI extensions
    pub fn vscode_ai() -> Self {
        Self {
            tool_name: "vscode-ai".to_string(),
            tool_version: None,
            preferred_formats: vec![ExportFormat::Json],
            endpoints: Vec::new(),
            auth_config: None,
        }
    }
    
    /// Create integration for GitHub Copilot
    pub fn github_copilot() -> Self {
        Self {
            tool_name: "github-copilot".to_string(),
            tool_version: None,
            preferred_formats: vec![ExportFormat::Json],
            endpoints: Vec::new(),
            auth_config: None,
        }
    }
    
    /// Create integration for language servers
    pub fn language_server() -> Self {
        Self {
            tool_name: "language-server".to_string(),
            tool_version: None,
            preferred_formats: vec![ExportFormat::Json, ExportFormat::Yaml],
            endpoints: Vec::new(),
            auth_config: None,
        }
    }
    
    /// Create integration for static analysis tools
    pub fn static_analysis() -> Self {
        Self {
            tool_name: "static-analysis".to_string(),
            tool_version: None,
            preferred_formats: vec![ExportFormat::Json, ExportFormat::Xml],
            endpoints: Vec::new(),
            auth_config: None,
        }
    }
} 