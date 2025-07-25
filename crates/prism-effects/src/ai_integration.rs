//! AI Integration - Effects Metadata Provider
//!
//! This module implements the AI integration interface for the effects system,
//! following Prism's external AI integration model. It provides structured
//! metadata export for external AI tools while maintaining separation of concerns.
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Only exposes existing effects metadata, doesn't collect new data
//! 2. **Conceptual Cohesion**: Focuses solely on effects and capabilities domain metadata
//! 3. **No Logic Duplication**: Leverages existing effects system infrastructure
//! 4. **AI-First**: Generates structured metadata for external AI consumption

use crate::effects::definition::EffectDefinition;
use prism_ai::providers::{
    MetadataProvider, MetadataDomain, ProviderContext, DomainMetadata, ProviderInfo
};
use prism_ai::AIIntegrationError;
use async_trait::async_trait;

// Local type definitions for AI integration
#[derive(Debug, Clone, serde::Serialize)]
pub struct CapabilityRequirement {
    pub capability_name: String,
    pub permission_level: String,
    pub justification: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SecurityAnalysis {
    pub security_level: String,
    pub threats: Vec<String>,
    pub mitigations: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EffectCompositionInfo {
    pub composition_type: String,
    pub effects: Vec<String>,
    pub dependencies: Vec<String>,
}

/// Effects metadata provider that exposes effects system metadata to the prism-ai system
/// 
/// This provider follows Separation of Concerns by:
/// - Only exposing existing effects and capabilities metadata, not collecting new data
/// - Focusing solely on effects system domain metadata
/// - Maintaining conceptual cohesion around effects and capabilities
#[derive(Debug)]
pub struct EffectsMetadataProvider {
    /// Whether this provider is enabled
    enabled: bool,
    /// Reference to effects system (would be actual system in real implementation)
    _effect_system: Option<EffectSystemRef>,
    /// Reference to security system
    _security_system: Option<SecuritySystemRef>,
    /// Reference to effect validator
    _validator: Option<EffectValidatorRef>,
}

/// Placeholder references for effects system components
/// In a real implementation, these would be actual references to the effects systems
#[derive(Debug)]
pub struct EffectSystemRef {
    // Would contain reference to actual effect system
}

#[derive(Debug)]
pub struct SecuritySystemRef {
    // Would contain reference to actual security system
}

#[derive(Debug)]
pub struct EffectValidatorRef {
    // Would contain reference to actual validator
}

impl EffectsMetadataProvider {
    /// Create a new effects metadata provider
    pub fn new() -> Self {
        Self {
            enabled: true,
            _effect_system: None,
            _security_system: None,
            _validator: None,
        }
    }
    
    /// Create provider with effects system references
    pub fn with_effects_systems(
        effect_system: EffectSystemRef,
        security_system: SecuritySystemRef,
        validator: EffectValidatorRef,
    ) -> Self {
        Self {
            enabled: true,
            _effect_system: Some(effect_system),
            _security_system: Some(security_system),
            _validator: Some(validator),
        }
    }
    
    /// Enable or disable this provider
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Extract effect definitions from effects system
    fn extract_effect_definitions(&self) -> Vec<EffectDefinition> {
        // In a real implementation, this would extract from self._effect_system
        vec![
            EffectDefinition {
                name: "FileSystem.Read".to_string(),
                description: "Read data from file system".to_string(),
                category: crate::effects::definition::EffectCategory::IO,
                parent_effect: None,
                ai_context: Some("File system read operation with capability-based access control".to_string()),
                security_implications: vec!["Requires FileSystem capability".to_string()],
                business_rules: vec![],
                capability_requirements: {
                    let mut caps = std::collections::HashMap::new();
                    caps.insert("FileSystem".to_string(), vec!["Read".to_string()]);
                    caps
                },
                parameters: vec![],
                examples: vec!["Reading configuration files".to_string()],
                common_mistakes: vec!["Not checking file permissions".to_string()],
            },
            EffectDefinition {
                name: "Network.Connect".to_string(),
                description: "Establish network connection".to_string(),
                category: crate::effects::definition::EffectCategory::Network,
                parent_effect: None,
                ai_context: Some("Network connection establishment with security controls".to_string()),
                security_implications: vec!["Requires Network capability".to_string()],
                business_rules: vec![],
                capability_requirements: {
                    let mut caps = std::collections::HashMap::new();
                    caps.insert("Network".to_string(), vec!["Connect".to_string()]);
                    caps
                },
                parameters: vec![],
                examples: vec!["HTTP client connections".to_string()],
                common_mistakes: vec!["Not validating connection targets".to_string()],
            },
            EffectDefinition {
                name: "Database.Query".to_string(),
                description: "Execute database query".to_string(),
                category: crate::effects::definition::EffectCategory::Database,
                parent_effect: None,
                ai_context: Some("Database query execution with access controls".to_string()),
                security_implications: vec!["Requires Database and Network capabilities".to_string()],
                business_rules: vec![],
                capability_requirements: {
                    let mut caps = std::collections::HashMap::new();
                    caps.insert("Database".to_string(), vec!["Query".to_string()]);
                    caps.insert("Network".to_string(), vec!["Connect".to_string()]);
                    caps
                },
                parameters: vec![],
                examples: vec!["SELECT queries".to_string(), "INSERT operations".to_string()],
                common_mistakes: vec!["SQL injection vulnerabilities".to_string()],
            },
        ]
    }
    
    /// Extract capability requirements from security system
    fn extract_capabilities(&self) -> Vec<CapabilityRequirement> {
        // In a real implementation, this would extract from self._security_system
        vec![
            CapabilityRequirement {
                capability_name: "FileSystem".to_string(),
                permission_level: "Read".to_string(),
                justification: "Required for configuration file access".to_string(),
            },
            CapabilityRequirement {
                capability_name: "Network".to_string(),
                permission_level: "Connect".to_string(),
                justification: "Required for external service communication".to_string(),
            },
            CapabilityRequirement {
                capability_name: "Database".to_string(),
                permission_level: "Query".to_string(),
                justification: "Required for data persistence operations".to_string(),
            },
        ]
    }
    
    /// Extract security implications from security analysis
    fn extract_security_implications(&self) -> SecurityAnalysis {
        // In a real implementation, this would extract from security analysis results
        SecurityAnalysis {
            security_level: "Medium".to_string(),
            threats: vec![
                "Unauthorized file access".to_string(),
                "Network-based attacks".to_string(),
                "SQL injection via database queries".to_string(),
            ],
            mitigations: vec![
                "Capability-based access control".to_string(),
                "Input validation and sanitization".to_string(),
                "Least privilege principle enforcement".to_string(),
            ],
        }
    }
    
    /// Extract effect composition information
    fn extract_composition_info(&self) -> EffectCompositionInfo {
        // In a real implementation, this would analyze effect composition patterns
        EffectCompositionInfo {
            composition_type: "Sequential IO operations".to_string(),
            effects: vec![
                "Sequential IO operations".to_string(),
                "Parallel network requests".to_string(),
                "Transactional database operations".to_string(),
            ],
            dependencies: vec![],
        }
    }
}

impl Default for EffectsMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetadataProvider for EffectsMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Effects
    }
    
    fn name(&self) -> &str {
        "effects-metadata-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        if !self.enabled {
            return Err(AIIntegrationError::ConfigurationError {
                message: "Effects metadata provider is disabled".to_string(),
            });
        }
        
        // Extract metadata from existing effects system structures
        let effect_definitions = self.extract_effect_definitions();
        let capabilities = self.extract_capabilities();
        let security_implications = self.extract_security_implications();
        let composition_info = self.extract_composition_info();
        
        // Create effects metadata structure
        use serde_json::json;
        use prism_ai::providers::EffectsMetadata;
        
        let security_context = json!({
            "security_implications": security_implications,
        });
        
        let effect_graph = json!({
            "effect_definitions": effect_definitions,
            "composition_info": composition_info,
        });
        
        let effects_metadata = EffectsMetadata {
            capabilities: capabilities.iter().map(|c| c.capability_name.clone()).collect(),
            security_context: Some(security_context),
            effect_graph: Some(effect_graph),
            confidence: 0.85, // High confidence in effects system metadata
        };
        
        Ok(DomainMetadata::Effects(effects_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Effects System Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides metadata about the effects system".to_string(),
            domains: vec![MetadataDomain::Effects],
            capabilities: vec![
                "RealTime".to_string(),
                "BusinessContext".to_string(),
                "CrossReference".to_string(),
                "PerformanceMetrics".to_string(),
            ],
            dependencies: vec![], // Effects system doesn't depend on other providers
        }
    }
} 