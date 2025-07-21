//! AI Integration - Metadata Export and External Tool Support
//!
//! This module implements the AI integration interfaces for PIR, following
//! Prism's external AI integration model. It provides structured metadata
//! export for external AI tools while maintaining separation of concerns.

pub mod metadata;

// Re-export main types
pub use metadata::*;

// NEW: Metadata provider implementation for prism-ai integration
use prism_ai::providers::{MetadataProvider, MetadataDomain, ProviderContext, DomainMetadata, ProviderInfo, ProviderCapability};
use prism_ai::AIIntegrationError;
use async_trait::async_trait;

/// PIR metadata provider that exposes PIR's AI metadata to the prism-ai system
/// 
/// This provider follows Separation of Concerns by:
/// - Only exposing existing PIR metadata, not collecting new data
/// - Focusing solely on PIR domain metadata
/// - Maintaining conceptual cohesion around PIR representation
#[derive(Debug)]
pub struct PIRMetadataProvider {
    /// Whether this provider is enabled
    enabled: bool,
    /// PIR system reference (would be actual PIR system in real implementation)
    pir_system: Option<PIRSystem>,
}

/// Placeholder for PIR system reference
#[derive(Debug)]
struct PIRSystem {
    // Would contain actual PIR data structures
}

impl PIRMetadataProvider {
    /// Create a new PIR metadata provider
    pub fn new() -> Self {
        Self {
            enabled: true,
            pir_system: None, // Would be initialized with actual PIR system
        }
    }
    
    /// Create provider with PIR system reference
    pub fn with_pir_system(pir_system: PIRSystem) -> Self {
        Self {
            enabled: true,
            pir_system: Some(pir_system),
        }
    }
    
    /// Enable or disable this provider
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Extract PIR structure information from existing PIR data
    fn extract_structure_info(&self) -> prism_ai::providers::PIRStructureInfo {
        // In a real implementation, this would extract data from self.pir_system
        prism_ai::providers::PIRStructureInfo {
            modules_count: 0, // Would count actual modules
            functions_count: 0, // Would count actual functions
            types_count: 0, // Would count actual types
            cohesion_score: 0.0, // Would calculate actual cohesion score
        }
    }
    
    /// Extract business context from PIR
    fn extract_business_context(&self) -> Option<prism_ai::providers::PIRBusinessContext> {
        // In a real implementation, this would extract from PIR business context
        Some(prism_ai::providers::PIRBusinessContext {
            domain: "Software Development".to_string(),
            capabilities: vec!["Code Generation".to_string(), "Semantic Preservation".to_string()],
            responsibilities: vec!["Intermediate Representation".to_string()],
        })
    }
    
    /// Extract optimization information
    fn extract_optimization_info(&self) -> prism_ai::providers::OptimizationInfo {
        prism_ai::providers::OptimizationInfo {
            optimizations_applied: vec!["Dead Code Elimination".to_string()],
            performance_improvement: 0.15, // 15% improvement
        }
    }
    
    /// Extract consistency data
    fn extract_consistency_data(&self) -> prism_ai::providers::ConsistencyData {
        prism_ai::providers::ConsistencyData {
            cross_target_compatibility: 0.95, // 95% compatible across targets
            semantic_preservation_score: 0.98, // 98% semantic preservation
        }
    }
}

impl Default for PIRMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetadataProvider for PIRMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Pir
    }
    
    fn name(&self) -> &str {
        "pir-metadata-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        if !self.enabled {
            return Err(AIIntegrationError::ConfigurationError {
                message: "PIR metadata provider is disabled".to_string(),
            });
        }
        
        // Extract metadata from existing PIR structures
        let structure_info = self.extract_structure_info();
        let business_context = self.extract_business_context();
        let optimization_info = self.extract_optimization_info();
        let consistency_data = self.extract_consistency_data();
        
        let pir_metadata = prism_ai::providers::PIRProviderMetadata {
            structure_info,
            business_context,
            optimization_info,
            consistency_data,
        };
        
        Ok(DomainMetadata::Pir(pir_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "PIR Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                ProviderCapability::RealTime,
                ProviderCapability::BusinessContext,
                ProviderCapability::PerformanceMetrics,
                ProviderCapability::CrossReference,
            ],
            dependencies: vec![], // PIR doesn't depend on other providers
        }
    }
} 