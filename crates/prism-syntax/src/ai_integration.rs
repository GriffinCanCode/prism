//! AI Integration - Syntax Metadata Provider
//!
//! This module implements the AI integration interface for the syntax system,
//! following Prism's external AI integration model. It provides structured
//! metadata export for external AI tools while maintaining separation of concerns.
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Only exposes existing syntax metadata, doesn't collect new data
//! 2. **Conceptual Cohesion**: Focuses solely on syntax parsing and normalization domain metadata
//! 3. **No Logic Duplication**: Leverages existing syntax processing infrastructure
//! 4. **AI-First**: Generates structured metadata for external AI consumption

use crate::{Parser, SyntaxDetector, Normalizer, Validator};
use prism_ai::providers::{
    MetadataProvider, MetadataDomain, ProviderContext, DomainMetadata, ProviderInfo, 
    ProviderCapability, SyntaxProviderMetadata, ParsingStatistics, SyntaxTreeMetrics
};
use prism_ai::AIIntegrationError;
use async_trait::async_trait;

/// Syntax metadata provider that exposes syntax processing metadata to the prism-ai system
/// 
/// This provider follows Separation of Concerns by:
/// - Only exposing existing syntax processing metadata, not collecting new data
/// - Focusing solely on syntax parsing and normalization domain metadata
/// - Maintaining conceptual cohesion around syntax understanding and processing
#[derive(Debug)]
pub struct SyntaxMetadataProvider {
    /// Whether this provider is enabled
    enabled: bool,
    /// Reference to parser (would be actual parser in real implementation)
    parser: Option<ParserRef>,
    /// Reference to syntax detector
    detector: Option<DetectorRef>,
    /// Reference to normalizer
    normalizer: Option<NormalizerRef>,
}

/// Placeholder references for syntax system components
/// In a real implementation, these would be actual references to the syntax systems
#[derive(Debug)]
struct ParserRef {
    // Would contain reference to actual parser
}

#[derive(Debug)]
struct DetectorRef {
    // Would contain reference to actual syntax detector
}

#[derive(Debug)]
struct NormalizerRef {
    // Would contain reference to actual normalizer
}

impl SyntaxMetadataProvider {
    /// Create a new syntax metadata provider
    pub fn new() -> Self {
        Self {
            enabled: true,
            parser: None,
            detector: None,
            normalizer: None,
        }
    }
    
    /// Create provider with syntax system references
    pub fn with_syntax_systems(
        parser: ParserRef,
        detector: DetectorRef,
        normalizer: NormalizerRef,
    ) -> Self {
        Self {
            enabled: true,
            parser: Some(parser),
            detector: Some(detector),
            normalizer: Some(normalizer),
        }
    }
    
    /// Enable or disable this provider
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Extract syntax style information from detector
    fn extract_syntax_style(&self) -> Option<String> {
        // In a real implementation, this would extract from self.detector
        Some("rust-like".to_string()) // Would detect actual syntax style
    }
    
    /// Extract parsing statistics from parser
    fn extract_parsing_statistics(&self) -> ParsingStatistics {
        // In a real implementation, this would extract from self.parser
        ParsingStatistics {
            lines_parsed: 450,        // Would count actual lines parsed
            tokens_processed: 2340,   // Would count actual tokens processed
            parse_time_ms: 125,       // Would measure actual parse time
            error_recovery_count: 3,  // Would count actual error recoveries
        }
    }
    
    /// Extract syntax tree metrics from parsing results
    fn extract_tree_metrics(&self) -> SyntaxTreeMetrics {
        // In a real implementation, this would analyze the actual syntax tree
        SyntaxTreeMetrics {
            node_count: 890,          // Would count actual AST nodes
            max_depth: 12,            // Would measure actual tree depth
            avg_branching_factor: 2.8, // Would calculate actual branching factor
        }
    }
    
    /// Extract AI context from normalization and analysis
    fn extract_ai_context(&self) -> prism_common::ai_metadata::AIMetadata {
        // In a real implementation, this would extract from normalization results
        let mut ai_metadata = prism_common::ai_metadata::AIMetadata::new();
        
        // Add semantic contexts discovered during parsing
        ai_metadata.add_semantic_context(prism_common::ai_metadata::SemanticContextEntry {
            location: prism_common::span::Span::new(0, 0, 0, 0),
            context_type: prism_common::ai_metadata::SemanticContextType::BusinessLogic,
            semantic_info: "Multi-syntax parsing with semantic preservation".to_string(),
            related_concepts: vec![
                "syntax normalization".to_string(),
                "cross-language compatibility".to_string(),
                "semantic preservation".to_string(),
            ],
            confidence: 0.92,
        });
        
        // Add business rules identified
        ai_metadata.add_business_rule(prism_common::ai_metadata::BusinessRuleEntry {
            rule_name: "Syntax style consistency".to_string(),
            rule_type: "parsing".to_string(),
            description: "All syntax styles must normalize to the same canonical form".to_string(),
            enforcement_level: "compile-time".to_string(),
            confidence: 0.98,
        });
        
        // Add AI insights about the syntax processing
        ai_metadata.add_insight(prism_common::ai_metadata::AIInsight {
            insight_type: "Architectural Pattern".to_string(),
            description: "Multi-syntax parser uses strategy pattern with normalization pipeline".to_string(),
            confidence: 0.95,
            business_impact: Some("Enables flexible syntax support without code duplication".to_string()),
        });
        
        ai_metadata.calculate_confidence();
        ai_metadata
    }
}

impl Default for SyntaxMetadataProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetadataProvider for SyntaxMetadataProvider {
    fn domain(&self) -> MetadataDomain {
        MetadataDomain::Syntax
    }
    
    fn name(&self) -> &str {
        "syntax-metadata-provider"
    }
    
    fn is_available(&self) -> bool {
        self.enabled
    }
    
    async fn provide_metadata(&self, _context: &ProviderContext) -> Result<DomainMetadata, AIIntegrationError> {
        if !self.enabled {
            return Err(AIIntegrationError::ConfigurationError {
                message: "Syntax metadata provider is disabled".to_string(),
            });
        }
        
        // Extract metadata from existing syntax processing structures
        let syntax_style = self.extract_syntax_style();
        let parsing_stats = self.extract_parsing_statistics();
        let tree_metrics = self.extract_tree_metrics();
        let ai_context = self.extract_ai_context();
        
        let syntax_metadata = SyntaxProviderMetadata {
            syntax_style,
            parsing_stats,
            tree_metrics,
            ai_context,
        };
        
        Ok(DomainMetadata::Syntax(syntax_metadata))
    }
    
    fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Syntax Processing Metadata Provider".to_string(),
            version: "0.1.0".to_string(),
            schema_version: "1.0.0".to_string(),
            capabilities: vec![
                ProviderCapability::RealTime,
                ProviderCapability::BusinessContext,
                ProviderCapability::CrossReference,
                ProviderCapability::Incremental,
            ],
            dependencies: vec![], // Syntax processing doesn't depend on other providers
        }
    }
} 