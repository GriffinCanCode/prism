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

/// Integration manager for coordinating with external AI tools
pub struct IntegrationManager {
    /// Registered endpoints
    endpoints: Vec<IntegrationEndpoint>,
    /// Output directory for exports
    output_directory: Option<PathBuf>,
    /// HTTP client for sending metadata to external endpoints
    http_client: HttpEndpointClient,
}

impl IntegrationManager {
    /// Create a new integration manager
    pub fn new() -> Self {
        Self {
            endpoints: Vec::new(),
            output_directory: None,
            http_client: HttpEndpointClient::new(),
        }
    }

    /// Add an integration endpoint
    pub fn add_endpoint(&mut self, endpoint: IntegrationEndpoint) {
        self.endpoints.push(endpoint);
    }

    /// Send metadata to all registered endpoints
    pub async fn send_to_all_endpoints(
        &self,
        metadata: &ComprehensiveAIMetadata,
    ) -> Result<Vec<IntegrationResult>, AIIntegrationError> {
        let mut results = Vec::new();
        
        for endpoint in &self.endpoints {
            let result = match self.http_client.send_to_endpoint(metadata, endpoint).await {
                Ok(response) => IntegrationResult {
                    endpoint_name: endpoint.name.clone(),
                    success: response.success,
                    status_code: Some(response.status_code),
                    message: if response.success {
                        "Successfully sent metadata".to_string()
                    } else {
                        format!("HTTP {} - {}", response.status_code, response.body)
                    },
                    response_data: Some(response.body),
                },
                Err(e) => IntegrationResult {
                    endpoint_name: endpoint.name.clone(),
                    success: false,
                    status_code: None,
                    message: format!("Integration failed: {}", e),
                    response_data: None,
                },
            };
            
            results.push(result);
        }
        
        Ok(results)
    }

    /// Send metadata to a specific endpoint by name
    pub async fn send_to_endpoint(
        &self,
        metadata: &ComprehensiveAIMetadata,
        endpoint_name: &str,
    ) -> Result<IntegrationResult, AIIntegrationError> {
        let endpoint = self.endpoints
            .iter()
            .find(|e| e.name == endpoint_name)
            .ok_or_else(|| AIIntegrationError::IntegrationError {
                message: format!("Endpoint '{}' not found", endpoint_name),
            })?;
        
        match self.http_client.send_to_endpoint(metadata, endpoint).await {
            Ok(response) => Ok(IntegrationResult {
                endpoint_name: endpoint.name.clone(),
                success: response.success,
                status_code: Some(response.status_code),
                message: if response.success {
                    "Successfully sent metadata".to_string()
                } else {
                    format!("HTTP {} - {}", response.status_code, response.body)
                },
                response_data: Some(response.body),
            }),
            Err(e) => Ok(IntegrationResult {
                endpoint_name: endpoint.name.clone(),
                success: false,
                status_code: None,
                message: format!("Integration failed: {}", e),
                response_data: None,
            }),
        }
    }

    /// Get all registered endpoints
    pub fn get_endpoints(&self) -> &[IntegrationEndpoint] {
        &self.endpoints
    }

    /// Remove an endpoint by name
    pub fn remove_endpoint(&mut self, endpoint_name: &str) -> bool {
        let initial_len = self.endpoints.len();
        self.endpoints.retain(|e| e.name != endpoint_name);
        self.endpoints.len() != initial_len
    }
}

/// Result of an integration attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Name of the endpoint
    pub endpoint_name: String,
    /// Whether the integration was successful
    pub success: bool,
    /// HTTP status code (if applicable)
    pub status_code: Option<u16>,
    /// Result message
    pub message: String,
    /// Response data from the endpoint
    pub response_data: Option<String>,
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

/// HTTP client for sending metadata to external endpoints
pub struct HttpEndpointClient {
    client: reqwest::Client,
}

impl HttpEndpointClient {
    /// Create a new HTTP endpoint client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
    
    /// Send metadata to an integration endpoint
    pub async fn send_to_endpoint(
        &self,
        metadata: &ComprehensiveAIMetadata,
        endpoint: &IntegrationEndpoint,
    ) -> Result<EndpointResponse, AIIntegrationError> {
        // Export metadata in the required format
        let exported_data = self.export_for_endpoint(metadata, endpoint)?;
        
        // Prepare request
        let mut request_builder = match endpoint.method {
            HttpMethod::Get => self.client.get(&endpoint.url),
            HttpMethod::Post => self.client.post(&endpoint.url),
            HttpMethod::Put => self.client.put(&endpoint.url),
            HttpMethod::Patch => self.client.patch(&endpoint.url),
            HttpMethod::Delete => self.client.delete(&endpoint.url),
        };
        
        // Add headers
        for (key, value) in &endpoint.headers {
            request_builder = request_builder.header(key, value);
        }
        
        // Add body for POST/PUT/PATCH requests
        let request = match endpoint.method {
            HttpMethod::Post | HttpMethod::Put | HttpMethod::Patch => {
                request_builder.body(exported_data)
            }
            _ => request_builder,
        };
        
        // Send request
        let response = request.send().await
            .map_err(|e| AIIntegrationError::IntegrationError {
                message: format!("HTTP request failed: {}", e),
            })?;
        
        let status_code = response.status().as_u16();
        let response_body = response.text().await
            .map_err(|e| AIIntegrationError::IntegrationError {
                message: format!("Failed to read response body: {}", e),
            })?;
        
        Ok(EndpointResponse {
            status_code,
            body: response_body,
            success: status_code >= 200 && status_code < 300,
        })
    }
    
    /// Export metadata in the format required by the endpoint
    fn export_for_endpoint(
        &self,
        metadata: &ComprehensiveAIMetadata,
        endpoint: &IntegrationEndpoint,
    ) -> Result<String, AIIntegrationError> {
        match endpoint.format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(metadata)
                    .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
            }
            ExportFormat::Yaml => {
                serde_yaml::to_string(metadata)
                    .map_err(|e| AIIntegrationError::SerializationError(e.to_string()))
            }
            _ => {
                Err(AIIntegrationError::ExportFailed {
                    format: endpoint.format.clone(),
                    reason: "Format not supported for HTTP endpoints".to_string(),
                })
            }
        }
    }
}

/// Response from an integration endpoint
#[derive(Debug, Clone)]
pub struct EndpointResponse {
    /// HTTP status code
    pub status_code: u16,
    /// Response body
    pub body: String,
    /// Whether the request was successful
    pub success: bool,
} 