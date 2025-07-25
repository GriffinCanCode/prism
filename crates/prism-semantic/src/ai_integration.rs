//! AI Integration - Semantic Metadata Provider
//!
//! This module implements the AI integration interface for semantic analysis,
//! following Prism's external AI integration model. It provides structured
//! metadata export for external AI tools while maintaining separation of concerns.
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Only exposes existing semantic metadata, doesn't collect new data
//! 2. **Conceptual Cohesion**: Focuses solely on semantic analysis domain metadata
//! 3. **No Logic Duplication**: Leverages existing semantic analysis infrastructure
//! 4. **AI-First**: Generates structured metadata for external AI consumption

use crate::analyzer::AnalysisResult;
use crate::database::SemanticInfo;
use crate::type_inference::constraints::ConstraintSet;
use prism_ai::providers::{
    MetadataProvider, MetadataDomain, ProviderContext, DomainMetadata, ProviderInfo, 
    SemanticMetadata, BusinessMetadata
};
use prism_ai::AIIntegrationError;
use async_trait::async_trait;
use std::sync::Arc;

/// Semantic metadata provider that exposes semantic analysis metadata to the prism-ai system
/// 
/// This provider follows Separation of Concerns by:
/// - Only exposing existing semantic analysis metadata, not collecting new data
/// - Focusing solely on semantic analysis domain metadata
/// - Maintaining conceptual cohesion around semantic understanding
#[derive(Debug)]
pub struct SemanticMetadataProvider {
    /// Whether this provider is enabled
    enabled: bool,
    /// Reference to semantic database for data extraction
    database: Option<Arc<crate::SemanticDatabase>>,
    /// Cached analysis results for performance
    cached_analysis: Option<AnalysisResult>,
    /// Cached semantic info for performance
    cached_semantic_info: Option<SemanticInfo>,
}

impl SemanticMetadataProvider {
    /// Create a new semantic metadata provider
    pub fn new() -> Self {
        Self {
            enabled: true,
            database: None,
            cached_analysis: None,
            cached_semantic_info: None,
        }
    }
    
    /// Create provider with semantic database reference
    pub fn with_database(database: Arc<crate::SemanticDatabase>) -> Self {
        Self {
            enabled: true,
            database: Some(database),
            cached_analysis: None,
            cached_semantic_info: None,
        }
    }
    
    /// Update cached analysis results
    pub fn update_analysis_cache(&mut self, analysis: AnalysisResult) {
        self.cached_analysis = Some(analysis);
    }
    
    /// Update cached semantic info
    pub fn update_semantic_cache(&mut self, semantic_info: SemanticInfo) {
        self.cached_semantic_info = Some(semantic_info);
    }
    
    /// Enable or disable this provider
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Extract type information from semantic analysis
    fn extract_type_information(&self) -> SemanticMetadata {
        if let Some(ref analysis) = self.cached_analysis {
            SemanticMetadata {
                type_system: Some(serde_json::json!({
                    "types_inferred": analysis.types.len(),
                    "constraints_solved": analysis.metadata.types_analyzed,
                    "semantic_types_identified": analysis.symbols.len()
                })),
                symbols: analysis.symbols.keys().map(|sym| sym.resolve().unwrap_or_else(|| "unknown".to_string())).collect(),
                patterns: vec!["semantic_analysis".to_string()],
                confidence: 0.9,
            }
        } else if let Some(ref semantic_info) = self.cached_semantic_info {
            SemanticMetadata {
                type_system: Some(serde_json::json!({
                    "types_inferred": semantic_info.types.len(),
                    "constraints_solved": semantic_info.analysis_metadata.types_analyzed,
                    "semantic_types_identified": semantic_info.symbols.len()
                })),
                symbols: semantic_info.symbols.keys().map(|sym| sym.resolve().unwrap_or_else(|| "unknown".to_string())).collect(),
                patterns: vec!["cached_semantic_info".to_string()],
                confidence: 0.8,
            }
        } else if let Some(ref database) = self.database {
            let stats = database.get_statistics();
            SemanticMetadata {
                type_system: Some(serde_json::json!({
                    "types_inferred": stats.type_count,
                    "constraints_solved": stats.type_count,
                    "semantic_types_identified": stats.semantic_type_count
                })),
                symbols: vec!["database_symbol".to_string()], // Simplified
                patterns: vec!["database_analysis".to_string()],
                confidence: 0.7,
            }
        } else {
            // Fallback to minimal data
            SemanticMetadata {
                type_system: Some(serde_json::json!({
                    "types_inferred": 0,
                    "constraints_solved": 0,
                    "semantic_types_identified": 0
                })),
                symbols: vec![],
                patterns: vec!["fallback".to_string()],
                confidence: 0.1,
            }
        }
    }
    
    /// Extract business rules from semantic analysis
    fn extract_business_rules(&self) -> Vec<BusinessMetadata> {
        let mut rules = Vec::new();
        
        // Extract from cached semantic info if available
        if let Some(ref semantic_info) = self.cached_semantic_info {
            // Extract business rules from symbols
            for symbol_info in semantic_info.symbols.values() {
                if let Some(ref business_context) = symbol_info.business_context {
                    rules.push(BusinessMetadata {
                        domain: Some(format!("{} business context", symbol_info.name)),
                        capabilities: vec!["business_context".to_string()],
                        rules: vec![format!("Business context for {}", symbol_info.name)],
                        confidence: 0.85,
                    });
                }
            }
            
            // Add validation rules if available
            if let Some(ref validation_result) = semantic_info.validation_result {
                rules.push(BusinessMetadata {
                    domain: Some("Semantic validation".to_string()),
                    capabilities: vec![if validation_result.passed { "validated" } else { "validation_failed" }.to_string()],
                    rules: vec!["Semantic validation rule".to_string()],
                    confidence: if validation_result.passed { 0.95 } else { 0.5 },
                });
            }
        }
        
        // Extract from database if available
        if let Some(ref database) = self.database {
            let db_rules = database.get_all_business_rules();
            for rule in db_rules {
                rules.push(BusinessMetadata {
                    domain: Some(rule.domain),
                    capabilities: vec![rule.name.clone()],
                    rules: vec![format!("Business rule: {}", rule.name)],
                    confidence: 0.9, // Business rules from database are high confidence
                });
            }
        }
        
        // Provide default rules if no data available
        if rules.is_empty() {
            rules.extend(vec![
                BusinessMetadata {
                    domain: Some("Type safety".to_string()),
                    capabilities: vec!["safety".to_string()],
                    rules: vec!["Type safety rule".to_string()],
                    confidence: 0.95,
                },
                BusinessMetadata {
                    domain: Some("Semantic consistency".to_string()),
                    capabilities: vec!["correctness".to_string()],
                    rules: vec!["Semantic consistency rule".to_string()],
                    confidence: 0.90,
                },
            ]);
        }
        
        rules
    }
    
    /// Extract semantic relationships from analysis
    fn extract_semantic_relationships(&self) -> Vec<String> {
        let mut relationships = Vec::new();
        
        // Extract from cached analysis if available
        if let Some(ref analysis) = self.cached_analysis {
            // Create relationships between symbols and types
            for (symbol, symbol_info) in &analysis.symbols {
                for (type_id, type_info) in &analysis.types {
                    if symbol_info.location.overlaps(&type_info.location) {
                        relationships.push(format!("{}:has_type:{}", 
                            symbol_info.name, 
                            type_info.ai_description.clone().unwrap_or_else(|| "Unknown type".to_string())
                        ));
                    }
                }
            }
        }
        
        // Extract from cached semantic info if available
        if let Some(ref semantic_info) = self.cached_semantic_info {
            for symbol_info in semantic_info.symbols.values() {
                if !symbol_info.ai_hints.is_empty() {
                    relationships.push(format!("{}:analyzed_by:AI_Analysis", symbol_info.name));
                }
            }
        }
        
        // Provide default relationships if no data available
        if relationships.is_empty() {
            relationships.push("SemanticAnalysis:integrates_with:TypeSystem".to_string());
        }
        
        relationships
    }
    
    /// Extract validation summary from semantic validation
    fn extract_validation_summary(&self) -> String {
        if let Some(ref semantic_info) = self.cached_semantic_info {
            if let Some(ref validation_result) = semantic_info.validation_result {
                format!("rules_checked:{}, violations_found:{}, warnings_issued:{}", 
                    (validation_result.errors.len() + validation_result.warnings.len()),
                    validation_result.errors.len(),
                    validation_result.warnings.len())
            } else {
                format!("rules_checked:{}, violations_found:0, warnings_issued:0", 
                    semantic_info.symbols.len())
            }
        } else if let Some(ref analysis) = self.cached_analysis {
            format!("rules_checked:{}, violations_found:{}, warnings_issued:{}", 
                (analysis.metadata.symbols_analyzed + analysis.metadata.types_analyzed),
                analysis.metadata.warnings.len(),
                analysis.metadata.warnings.len())
        } else {
            "rules_checked:0, violations_found:0, warnings_issued:0".to_string()
        }
    }
}

impl Default for SemanticMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetadataProvider for SemanticMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Semantic
    }
    
    fn name(&self) -> &str {
        "semantic-metadata-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        if !self.enabled {
            return Err(AIIntegrationError::ConfigurationError {
                message: "Semantic metadata provider is disabled".to_string(),
            });
        }
        
        // Extract metadata from existing semantic analysis structures
        let type_info = self.extract_type_information();
        let business_rules = self.extract_business_rules();
        let relationships = self.extract_semantic_relationships();
        let validation_results = self.extract_validation_summary();
        
        let semantic_metadata = prism_ai::providers::SemanticMetadata {
            type_system: Some(serde_json::json!({
                "type_info": type_info,
                "business_rules": business_rules,
                "validation_results": validation_results
            })),
            symbols: relationships,
            patterns: vec!["semantic_analysis".to_string()],
            confidence: 0.8,
        };
        
        Ok(DomainMetadata::Semantic(semantic_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Semantic Analysis Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            description: "Provides semantic analysis metadata for AI understanding".to_string(),
            domains: vec![
                prism_ai::MetadataDomain::Semantic,
                prism_ai::MetadataDomain::Business,
            ],
            capabilities: vec![
                "real_time_analysis".to_string(),
                "business_context".to_string(),
                "cross_reference".to_string(),
                "incremental_analysis".to_string(),
            ],
            dependencies: vec![], // Semantic analysis doesn't depend on other providers
        }
    }
}

// Helper trait for span overlap checking
trait SpanOverlap {
    fn overlaps(&self, other: &Self) -> bool;
}

impl SpanOverlap for prism_common::span::Span {
    fn overlaps(&self, other: &Self) -> bool {
        // Simple overlap check - would need proper implementation
        self.start <= other.end && other.start <= self.end
    }
} 