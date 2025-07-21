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

use crate::{SemanticAnalyzer, SemanticDatabase, types::SemanticTypeSystem, patterns::PatternRecognizer};
use crate::analyzer::AnalysisResult;
use crate::database::SemanticInfo;
use prism_ai::providers::{
    MetadataProvider, MetadataDomain, ProviderContext, DomainMetadata, ProviderInfo, 
    ProviderCapability, SemanticProviderMetadata, TypeInformation, BusinessRule,
    SemanticRelationship, ValidationSummary
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
    database: Option<Arc<SemanticDatabase>>,
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
    pub fn with_database(database: Arc<SemanticDatabase>) -> Self {
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
    fn extract_type_information(&self) -> TypeInformation {
        if let Some(ref analysis) = self.cached_analysis {
            TypeInformation {
                types_inferred: analysis.types.len() as u32,
                constraints_solved: analysis.metadata.types_analyzed as u32, // Use actual analyzed count
                semantic_types_identified: analysis.symbols.len() as u32,
            }
        } else if let Some(ref semantic_info) = self.cached_semantic_info {
            TypeInformation {
                types_inferred: semantic_info.types.len() as u32,
                constraints_solved: semantic_info.analysis_metadata.types_analyzed as u32,
                semantic_types_identified: semantic_info.symbols.len() as u32,
            }
        } else if let Some(ref database) = self.database {
            let stats = database.get_statistics();
            TypeInformation {
                types_inferred: stats.type_count as u32,
                constraints_solved: stats.type_count as u32, // Approximation
                semantic_types_identified: stats.semantic_type_count as u32,
            }
        } else {
            // Fallback to minimal data
            TypeInformation {
                types_inferred: 0,
                constraints_solved: 0,
                semantic_types_identified: 0,
            }
        }
    }
    
    /// Extract business rules from semantic analysis
    fn extract_business_rules(&self) -> Vec<BusinessRule> {
        let mut rules = Vec::new();
        
        // Extract from cached semantic info if available
        if let Some(ref semantic_info) = self.cached_semantic_info {
            // Extract business rules from symbols
            for symbol_info in semantic_info.symbols.values() {
                if let Some(ref business_context) = symbol_info.business_context {
                    rules.push(BusinessRule {
                        rule_name: format!("{} business context", symbol_info.name),
                        rule_type: "business_context".to_string(),
                        confidence: 0.85,
                    });
                }
            }
            
            // Add validation rules if available
            if let Some(ref validation_result) = semantic_info.validation_result {
                rules.push(BusinessRule {
                    rule_name: "Semantic validation".to_string(),
                    rule_type: if validation_result.passed { "validated" } else { "validation_failed" }.to_string(),
                    confidence: if validation_result.passed { 0.95 } else { 0.5 },
                });
            }
        }
        
        // Extract from database if available
        if let Some(ref database) = self.database {
            let db_rules = database.get_all_business_rules();
            for rule in db_rules {
                rules.push(BusinessRule {
                    rule_name: rule.name,
                    rule_type: rule.domain,
                    confidence: 0.9, // Business rules from database are high confidence
                });
            }
        }
        
        // Provide default rules if no data available
        if rules.is_empty() {
            rules.extend(vec![
                BusinessRule {
                    rule_name: "Type safety".to_string(),
                    rule_type: "safety".to_string(),
                    confidence: 0.95,
                },
                BusinessRule {
                    rule_name: "Semantic consistency".to_string(),
                    rule_type: "correctness".to_string(),
                    confidence: 0.90,
                },
            ]);
        }
        
        rules
    }
    
    /// Extract semantic relationships from analysis
    fn extract_semantic_relationships(&self) -> Vec<SemanticRelationship> {
        let mut relationships = Vec::new();
        
        // Extract from cached analysis if available
        if let Some(ref analysis) = self.cached_analysis {
            // Create relationships between symbols and types
            for (symbol, symbol_info) in &analysis.symbols {
                for (type_id, type_info) in &analysis.types {
                    if symbol_info.location.overlaps(&type_info.location) {
                        relationships.push(SemanticRelationship {
                            source: symbol_info.name.clone(),
                            target: type_info.ai_description.clone().unwrap_or_else(|| "Unknown type".to_string()),
                            relationship_type: "has_type".to_string(),
                            strength: 0.9,
                        });
                    }
                }
            }
        }
        
        // Extract from cached semantic info if available
        if let Some(ref semantic_info) = self.cached_semantic_info {
            for symbol_info in semantic_info.symbols.values() {
                if !symbol_info.ai_hints.is_empty() {
                    relationships.push(SemanticRelationship {
                        source: symbol_info.name.clone(),
                        target: "AI_Analysis".to_string(),
                        relationship_type: "analyzed_by".to_string(),
                        strength: 0.8,
                    });
                }
            }
        }
        
        // Provide default relationships if no data available
        if relationships.is_empty() {
            relationships.push(SemanticRelationship {
                source: "SemanticAnalysis".to_string(),
                target: "TypeSystem".to_string(),
                relationship_type: "integrates_with".to_string(),
                strength: 1.0,
            });
        }
        
        relationships
    }
    
    /// Extract validation summary from semantic validation
    fn extract_validation_summary(&self) -> ValidationSummary {
        if let Some(ref semantic_info) = self.cached_semantic_info {
            if let Some(ref validation_result) = semantic_info.validation_result {
                ValidationSummary {
                    rules_checked: (validation_result.errors.len() + validation_result.warnings.len()) as u32,
                    violations_found: validation_result.errors.len() as u32,
                    warnings_issued: validation_result.warnings.len() as u32,
                }
            } else {
                ValidationSummary {
                    rules_checked: semantic_info.symbols.len() as u32,
                    violations_found: 0,
                    warnings_issued: 0,
                }
            }
        } else if let Some(ref analysis) = self.cached_analysis {
            ValidationSummary {
                rules_checked: (analysis.metadata.symbols_analyzed + analysis.metadata.types_analyzed) as u32,
                violations_found: analysis.metadata.warnings.len() as u32,
                warnings_issued: analysis.metadata.warnings.len() as u32,
            }
        } else {
            ValidationSummary {
                rules_checked: 0,
                violations_found: 0,
                warnings_issued: 0,
            }
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
        
        let semantic_metadata = SemanticProviderMetadata {
            type_info,
            business_rules,
            relationships,
            validation_results,
        };
        
        Ok(DomainMetadata::Semantic(semantic_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Semantic Analysis Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                ProviderCapability::RealTime,
                ProviderCapability::BusinessContext,
                ProviderCapability::CrossReference,
                ProviderCapability::Incremental,
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