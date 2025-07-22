//! AI Metadata Generation for Query Results
//!
//! This module provides AI-first metadata generation for all query results,
//! making compilation information easily consumable by external AI tools
//! and development environments.
//!
//! **Single Responsibility**: AI metadata generation for query results

use crate::error::{CompilerError, CompilerResult};
use crate::query::core::{QueryId, CacheKey};
use prism_common::{span::Span, NodeId};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// AI-readable metadata for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIQueryMetadata {
    /// Query identification
    pub query_info: QueryInfo,
    /// Semantic context for AI understanding
    pub semantic_context: AISemanticContext,
    /// Business context extracted from code
    pub business_context: AIBusinessContext,
    /// Performance characteristics
    pub performance_profile: AIPerformanceProfile,
    /// Relationships to other code elements
    pub relationships: AIRelationships,
    /// Quality metrics
    pub quality_metrics: AIQualityMetrics,
    /// Suggestions for improvement
    pub suggestions: Vec<AISuggestion>,
    /// Metadata generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Query identification information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryInfo {
    /// Unique query identifier
    pub query_id: String,
    /// Query type name
    pub query_type: String,
    /// Cache key for reproducibility
    pub cache_key: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Whether result was cached
    pub was_cached: bool,
}

/// AI-comprehensible semantic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISemanticContext {
    /// High-level purpose of the query result
    pub purpose: String,
    /// Domain concepts involved
    pub domain_concepts: Vec<String>,
    /// Patterns identified
    pub patterns: Vec<String>,
    /// Type summary for AI understanding
    pub type_summary: String,
    /// Effects summary
    pub effects_summary: String,
    /// Complexity level assessment
    pub complexity_level: ComplexityLevel,
}

/// Business context for AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIBusinessContext {
    /// Business capability this relates to
    pub capability: Option<String>,
    /// Business rules identified
    pub rules: Vec<String>,
    /// Domain terminology
    pub terminology: Vec<String>,
    /// Stakeholder concerns
    pub stakeholder_concerns: Vec<String>,
    /// Compliance considerations
    pub compliance: Vec<String>,
}

/// Performance characteristics for AI analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIPerformanceProfile {
    /// Time complexity estimate
    pub time_complexity: String,
    /// Space complexity estimate
    pub space_complexity: String,
    /// Identified bottlenecks
    pub bottlenecks: Vec<String>,
    /// Optimization opportunities
    pub optimizations: Vec<String>,
    /// Resource usage patterns
    pub resource_usage: Vec<ResourceUsage>,
}

/// Resource usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Type of resource
    pub resource_type: String,
    /// Usage pattern description
    pub pattern: String,
    /// Intensity level
    pub intensity: String,
}

/// Relationships to other code elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRelationships {
    /// Dependencies
    pub depends_on: Vec<String>,
    /// Dependents
    pub depended_by: Vec<String>,
    /// Related elements
    pub related_to: Vec<String>,
    /// Architectural role
    pub architectural_role: String,
}

/// Quality metrics for AI assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIQualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Maintainability score (0.0 to 1.0)
    pub maintainability: f64,
    /// Readability score (0.0 to 1.0)
    pub readability: f64,
    /// Testability score (0.0 to 1.0)
    pub testability: f64,
    /// Documentation completeness (0.0 to 1.0)
    pub documentation_completeness: f64,
    /// Identified code smells
    pub code_smells: Vec<String>,
}

/// AI improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISuggestion {
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Description of the suggestion
    pub description: String,
    /// Priority level
    pub priority: Priority,
    /// Effort required
    pub effort: Effort,
    /// Expected benefits
    pub benefits: Vec<String>,
    /// Location where suggestion applies
    pub location: Option<Span>,
}

/// Types of suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Performance improvement
    Performance,
    /// Readability improvement
    Readability,
    /// Maintainability improvement
    Maintainability,
    /// Security improvement
    Security,
    /// Error handling improvement
    ErrorHandling,
    /// Documentation improvement
    Documentation,
    /// Testing improvement
    Testing,
}

/// Suggestion priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    /// Critical priority
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
}

/// Effort levels for suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effort {
    /// Minimal effort required
    Minimal,
    /// Low effort required
    Low,
    /// Medium effort required
    Medium,
    /// High effort required
    High,
    /// Extensive effort required
    Extensive,
}

/// Complexity levels for AI understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    /// Simple, straightforward code
    Simple,
    /// Moderate complexity
    Medium,
    /// High complexity
    High,
    /// Very complex, needs careful analysis
    VeryHigh,
}

/// Trait for generating AI metadata
pub trait AIMetadataGenerator: Send + Sync {
    /// Generate AI metadata for a query result
    fn generate_metadata(
        &self,
        query_id: QueryId,
        query_type: &str,
        result: &dyn AIMetadataSource,
        context: &AIGenerationContext,
    ) -> CompilerResult<AIQueryMetadata>;

    /// Extract semantic context from result
    fn extract_semantic_context(
        &self,
        result: &dyn AIMetadataSource,
        context: &AIGenerationContext,
    ) -> CompilerResult<AISemanticContext>;

    /// Extract business context from result
    fn extract_business_context(
        &self,
        result: &dyn AIMetadataSource,
        context: &AIGenerationContext,
    ) -> CompilerResult<AIBusinessContext>;

    /// Generate quality metrics
    fn assess_quality(
        &self,
        result: &dyn AIMetadataSource,
        context: &AIGenerationContext,
    ) -> CompilerResult<AIQualityMetrics>;

    /// Generate improvement suggestions
    fn generate_suggestions(
        &self,
        result: &dyn AIMetadataSource,
        context: &AIGenerationContext,
    ) -> CompilerResult<Vec<AISuggestion>>;
}

/// Source of AI metadata information
pub trait AIMetadataSource: Send + Sync {
    /// Get the primary name/identifier
    fn get_name(&self) -> String;

    /// Get the type/kind of element
    fn get_type(&self) -> String;

    /// Get source location if available
    fn get_location(&self) -> Option<Span>;

    /// Get semantic annotations
    fn get_semantic_annotations(&self) -> Vec<String>;

    /// Get business annotations
    fn get_business_annotations(&self) -> Vec<String>;

    /// Get complexity indicators
    fn get_complexity_indicators(&self) -> Vec<String>;

    /// Get relationship information
    fn get_relationships(&self) -> Vec<String>;
}

/// Context for AI metadata generation
#[derive(Debug, Clone)]
pub struct AIGenerationContext {
    /// Current compilation phase
    pub compilation_phase: String,
    /// Available semantic information
    pub semantic_database: Option<std::sync::Arc<crate::semantic::SemanticDatabase>>,
    /// Business context registry
    pub business_context: HashMap<String, String>,
    /// Performance profiling data
    pub performance_data: HashMap<String, f64>,
    /// Quality metrics from other analyses
    pub quality_context: HashMap<String, f64>,
}

/// Default AI metadata generator implementation
#[derive(Debug, Default)]
pub struct DefaultAIMetadataGenerator {
    /// Configuration for metadata generation
    config: AIGeneratorConfig,
}

/// Configuration for AI metadata generation
#[derive(Debug, Clone)]
pub struct AIGeneratorConfig {
    /// Enable detailed semantic analysis
    pub enable_detailed_semantics: bool,
    /// Enable business context extraction
    pub enable_business_context: bool,
    /// Enable performance profiling
    pub enable_performance_analysis: bool,
    /// Enable quality assessment
    pub enable_quality_assessment: bool,
    /// Enable suggestion generation
    pub enable_suggestions: bool,
    /// Minimum quality threshold for suggestions
    pub suggestion_quality_threshold: f64,
}

impl Default for AIGeneratorConfig {
    fn default() -> Self {
        Self {
            enable_detailed_semantics: true,
            enable_business_context: true,
            enable_performance_analysis: true,
            enable_quality_assessment: true,
            enable_suggestions: true,
            suggestion_quality_threshold: 0.7,
        }
    }
}

impl DefaultAIMetadataGenerator {
    /// Create a new default AI metadata generator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration
    pub fn with_config(config: AIGeneratorConfig) -> Self {
        Self { config }
    }
}

impl AIMetadataGenerator for DefaultAIMetadataGenerator {
    fn generate_metadata(
        &self,
        query_id: QueryId,
        query_type: &str,
        result: &dyn AIMetadataSource,
        context: &AIGenerationContext,
    ) -> CompilerResult<AIQueryMetadata> {
        let query_info = QueryInfo {
            query_id: query_id.to_string(),
            query_type: query_type.to_string(),
            cache_key: format!("{}:{}", query_type, result.get_name()),
            execution_time_ms: 0, // Would be provided by caller
            was_cached: false,    // Would be provided by caller
        };

        let semantic_context = if self.config.enable_detailed_semantics {
            self.extract_semantic_context(result, context)?
        } else {
            AISemanticContext {
                purpose: result.get_type(),
                domain_concepts: Vec::new(),
                patterns: Vec::new(),
                type_summary: result.get_type(),
                effects_summary: "No effects analysis".to_string(),
                complexity_level: ComplexityLevel::Medium,
            }
        };

        let business_context = if self.config.enable_business_context {
            self.extract_business_context(result, context)?
        } else {
            AIBusinessContext {
                capability: None,
                rules: Vec::new(),
                terminology: Vec::new(),
                stakeholder_concerns: vec!["Performance".to_string(), "Correctness".to_string()],
                compliance: Vec::new(),
            }
        };

        let performance_profile = if self.config.enable_performance_analysis {
            AIPerformanceProfile {
                time_complexity: "O(1)".to_string(),
                space_complexity: "O(1)".to_string(),
                bottlenecks: Vec::new(),
                optimizations: Vec::new(),
                resource_usage: Vec::new(),
            }
        } else {
            AIPerformanceProfile {
                time_complexity: "Not analyzed".to_string(),
                space_complexity: "Not analyzed".to_string(),
                bottlenecks: Vec::new(),
                optimizations: Vec::new(),
                resource_usage: Vec::new(),
            }
        };

        let relationships = AIRelationships {
            depends_on: result.get_relationships(),
            depended_by: Vec::new(),
            related_to: Vec::new(),
            architectural_role: result.get_type(),
        };

        let quality_metrics = if self.config.enable_quality_assessment {
            self.assess_quality(result, context)?
        } else {
            AIQualityMetrics {
                overall_score: 0.5,
                maintainability: 0.5,
                readability: 0.5,
                testability: 0.5,
                documentation_completeness: 0.5,
                code_smells: if result.get_complexity_indicators().len() > 2 {
                    vec!["High complexity detected".to_string()]
                } else {
                    Vec::new()
                },
            }
        };

        let suggestions = if self.config.enable_suggestions {
            self.generate_suggestions(result, context)?
        } else {
            Vec::new()
        };

        Ok(AIQueryMetadata {
            query_info,
            semantic_context,
            business_context,
            performance_profile,
            relationships,
            quality_metrics,
            suggestions,
            generated_at: Utc::now(),
        })
    }

    fn extract_semantic_context(
        &self,
        result: &dyn AIMetadataSource,
        _context: &AIGenerationContext,
    ) -> CompilerResult<AISemanticContext> {
        let annotations = result.get_semantic_annotations();
        let complexity_indicators = result.get_complexity_indicators();
        
        let complexity_level = if complexity_indicators.len() > 3 {
            ComplexityLevel::VeryHigh
        } else if complexity_indicators.len() > 1 {
            ComplexityLevel::High
        } else if complexity_indicators.len() > 0 {
            ComplexityLevel::Medium
        } else {
            ComplexityLevel::Simple
        };

        Ok(AISemanticContext {
            purpose: format!("Process {}", result.get_name()),
            domain_concepts: annotations.clone(),
            patterns: vec!["Query Pattern".to_string()],
            type_summary: result.get_type(),
            effects_summary: "Query execution effects".to_string(),
            complexity_level,
        })
    }

    fn extract_business_context(
        &self,
        result: &dyn AIMetadataSource,
        _context: &AIGenerationContext,
    ) -> CompilerResult<AIBusinessContext> {
        let business_annotations = result.get_business_annotations();
        
        Ok(AIBusinessContext {
            capability: Some(result.get_name()),
            rules: business_annotations.clone(),
            terminology: Vec::new(),
            stakeholder_concerns: vec!["Performance".to_string(), "Correctness".to_string()],
            compliance: Vec::new(),
        })
    }

    fn assess_quality(
        &self,
        result: &dyn AIMetadataSource,
        _context: &AIGenerationContext,
    ) -> CompilerResult<AIQualityMetrics> {
        let complexity_indicators = result.get_complexity_indicators();
        
        // Simple quality assessment based on complexity
        let base_score = if complexity_indicators.is_empty() { 0.9 } else { 0.7 };
        let complexity_penalty = complexity_indicators.len() as f64 * 0.1;
        let overall_score = (base_score - complexity_penalty).max(0.1);

        Ok(AIQualityMetrics {
            overall_score,
            maintainability: overall_score,
            readability: overall_score + 0.1,
            testability: overall_score - 0.1,
            documentation_completeness: 0.6,
            code_smells: if complexity_indicators.len() > 2 {
                vec!["High complexity detected".to_string()]
            } else {
                Vec::new()
            },
        })
    }

    fn generate_suggestions(
        &self,
        result: &dyn AIMetadataSource,
        _context: &AIGenerationContext,
    ) -> CompilerResult<Vec<AISuggestion>> {
        let mut suggestions = Vec::new();
        let complexity_indicators = result.get_complexity_indicators();

        if complexity_indicators.len() > 2 {
            suggestions.push(AISuggestion {
                suggestion_type: SuggestionType::Maintainability,
                description: "Consider breaking down complex logic into smaller functions".to_string(),
                priority: Priority::Medium,
                effort: Effort::Medium,
                benefits: vec![
                    "Improved maintainability".to_string(),
                    "Better testability".to_string(),
                    "Enhanced readability".to_string(),
                ],
                location: result.get_location(),
            });
        }

        if result.get_business_annotations().is_empty() {
            suggestions.push(AISuggestion {
                suggestion_type: SuggestionType::Documentation,
                description: "Add business context documentation".to_string(),
                priority: Priority::Low,
                effort: Effort::Low,
                benefits: vec![
                    "Better AI understanding".to_string(),
                    "Improved maintainability".to_string(),
                ],
                location: result.get_location(),
            });
        }

        Ok(suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockAIMetadataSource {
        name: String,
        type_name: String,
    }

    impl AIMetadataSource for MockAIMetadataSource {
        fn get_name(&self) -> String {
            self.name.clone()
        }

        fn get_type(&self) -> String {
            self.type_name.clone()
        }

        fn get_location(&self) -> Option<Span> {
            None
        }

        fn get_semantic_annotations(&self) -> Vec<String> {
            vec!["test_annotation".to_string()]
        }

        fn get_business_annotations(&self) -> Vec<String> {
            vec!["business_rule".to_string()]
        }

        fn get_complexity_indicators(&self) -> Vec<String> {
            vec!["complex_logic".to_string()]
        }

        fn get_relationships(&self) -> Vec<String> {
            vec!["related_component".to_string()]
        }
    }

    #[test]
    fn test_default_ai_metadata_generator() {
        let generator = DefaultAIMetadataGenerator::new();
        let source = MockAIMetadataSource {
            name: "test_query".to_string(),
            type_name: "TestQuery".to_string(),
        };
        let context = AIGenerationContext {
            compilation_phase: "test".to_string(),
            semantic_database: None,
            business_context: HashMap::new(),
            performance_data: HashMap::new(),
            quality_context: HashMap::new(),
        };

        let result = generator.generate_metadata(
            QueryId::new(),
            "TestQuery",
            &source,
            &context,
        );

        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert_eq!(metadata.query_info.query_type, "TestQuery");
        assert_eq!(metadata.semantic_context.purpose, "Process test_query");
    }

    #[test]
    fn test_ai_generator_config() {
        let config = AIGeneratorConfig {
            enable_detailed_semantics: false,
            enable_business_context: false,
            enable_performance_analysis: false,
            enable_quality_assessment: false,
            enable_suggestions: false,
            suggestion_quality_threshold: 0.8,
        };

        let generator = DefaultAIMetadataGenerator::with_config(config);
        assert!(!generator.config.enable_detailed_semantics);
        assert!(!generator.config.enable_business_context);
    }
} 