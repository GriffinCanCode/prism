//! Main Suggestion Engine
//!
//! This module provides the central coordination for context-guided suggestion generation.
//! It integrates with existing semantic analysis, error recovery, and AI systems.

use super::{
    context::{ContextExtractor, SuggestionContext},
    providers::SuggestionProviders,
    ranking::{SuggestionRanker, RankedSuggestion},
    pattern::{PatternMatcher, SuggestionPattern},
    ContextualSuggestion, SuggestionError, SuggestionResult,
};
use crate::core::error::{ParseError, ErrorContext};
use prism_common::diagnostics::Suggestion;
use prism_semantic::SemanticEngine;
use prism_common::ai_metadata::AIMetadata;
use prism_common::{ai_metadata::AIMetadataCollector, suggestion::DiagnosticCollector};
use std::sync::Arc;

/// Configuration for the suggestion engine
#[derive(Debug, Clone)]
pub struct SuggestionEngineConfig {
    /// Enable pattern learning from successful fixes
    pub enable_pattern_learning: bool,
    
    /// Maximum number of suggestions to generate
    pub max_suggestions: usize,
    
    /// Minimum confidence threshold for suggestions
    pub min_confidence_threshold: f64,
    
    /// Enable business rule integration
    pub enable_business_rules: bool,
    
    /// Enable semantic context analysis
    pub enable_semantic_context: bool,
}

impl Default for SuggestionEngineConfig {
    fn default() -> Self {
        Self {
            enable_pattern_learning: true,
            max_suggestions: 5,
            min_confidence_threshold: 0.3,
            enable_business_rules: true,
            enable_semantic_context: true,
        }
    }
}

/// Main suggestion engine that coordinates context-guided suggestion generation
pub struct SuggestionEngine {
    /// Configuration
    config: SuggestionEngineConfig,
    
    /// Context extractor for analyzing error context
    context_extractor: ContextExtractor,
    
    /// Specialized suggestion providers
    providers: SuggestionProviders,
    
    /// Suggestion ranking and confidence scoring
    ranker: SuggestionRanker,
    
    /// Pattern matcher for learning from history
    pattern_matcher: PatternMatcher,
    
    /// Integration with semantic analysis
    semantic_engine: Option<Arc<SemanticEngine>>,
}

impl SuggestionEngine {
    /// Create a new suggestion engine with default configuration
    pub fn new() -> SuggestionResult<Self> {
        Self::with_config(SuggestionEngineConfig::default())
    }
    
    /// Create a suggestion engine with custom configuration
    pub fn with_config(config: SuggestionEngineConfig) -> SuggestionResult<Self> {
        Ok(Self {
            context_extractor: ContextExtractor::new(&config)?,
            providers: SuggestionProviders::new(&config)?,
            ranker: SuggestionRanker::new(&config),
            pattern_matcher: PatternMatcher::new()?,
            semantic_engine: None,
            config,
        })
    }
    
    /// Set semantic engine for enhanced context analysis
    pub fn with_semantic_engine(mut self, semantic_engine: Arc<SemanticEngine>) -> Self {
        self.semantic_engine = Some(semantic_engine);
        self
    }
    
    /// Generate context-guided suggestions for a parse error
    pub fn generate_suggestions(
        &mut self,
        error: &ParseError,
        error_context: Option<&ErrorContext>,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        // Step 1: Extract comprehensive context
        let suggestion_context = self.context_extractor.extract_context(
            error,
            error_context,
            self.semantic_engine.as_ref(),
        )?;
        
        // Step 2: Find matching patterns from history
        let matching_patterns = self.pattern_matcher.find_matching_patterns(
            error,
            &suggestion_context,
        )?;
        
        // Step 3: Generate suggestions from all providers
        let mut raw_suggestions = Vec::new();
        
        // Syntax-level suggestions
        if let Some(syntax_suggestions) = self.providers.syntax_provider.generate_suggestions(
            error,
            &suggestion_context,
        )? {
            raw_suggestions.extend(syntax_suggestions);
        }
        
        // Semantic-level suggestions
        if self.config.enable_semantic_context {
            if let Some(semantic_suggestions) = self.providers.semantic_provider.generate_suggestions(
                error,
                &suggestion_context,
            )? {
                raw_suggestions.extend(semantic_suggestions);
            }
        }
        
        // Business rule suggestions
        if self.config.enable_business_rules {
            if let Some(business_suggestions) = self.providers.business_provider.generate_suggestions(
                error,
                &suggestion_context,
            )? {
                raw_suggestions.extend(business_suggestions);
            }
        }
        
        // AI-powered suggestions
        // if self.config.enable_ai_suggestions {
        //     if let Some(ai_suggestions) = self.providers.ai_provider.generate_suggestions(
        //         error,
        //         &suggestion_context,
        //         &matching_patterns,
        //     )? {
        //         raw_suggestions.extend(ai_suggestions);
        //     }
        // }
        
        // Step 4: Rank and filter suggestions
        let ranked_suggestions = self.ranker.rank_suggestions(
            raw_suggestions,
            &suggestion_context,
        )?;
        
        // Step 5: Apply confidence threshold and limit
        let filtered_suggestions: Vec<ContextualSuggestion> = ranked_suggestions
            .into_iter()
            .filter(|s| s.confidence >= self.config.min_confidence_threshold)
            .take(self.config.max_suggestions)
            .map(|r| r.suggestion)
            .collect();
        
        // Step 6: Record patterns for learning (if enabled)
        if self.config.enable_pattern_learning {
            self.pattern_matcher.record_suggestion_attempt(
                error,
                &suggestion_context,
                &filtered_suggestions,
            )?;
        }
        
        Ok(filtered_suggestions)
    }
    
    /// Learn from successful suggestion application
    pub fn learn_from_success(
        &mut self,
        original_error: &ParseError,
        applied_suggestion: &ContextualSuggestion,
        outcome_quality: f64,
    ) -> SuggestionResult<()> {
        if self.config.enable_pattern_learning {
            self.pattern_matcher.record_successful_application(
                original_error,
                applied_suggestion,
                outcome_quality,
            )?;
        }
        Ok(())
    }
    
    /// Get suggestion statistics for analysis
    pub fn get_statistics(&self) -> SuggestionStatistics {
        SuggestionStatistics {
            total_patterns: self.pattern_matcher.pattern_count(),
            successful_applications: self.pattern_matcher.success_count(),
            average_confidence: self.ranker.average_confidence(),
            provider_statistics: self.providers.get_statistics(),
        }
    }
}

impl Default for SuggestionEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default suggestion engine")
    }
}

/// Statistics about suggestion engine performance
#[derive(Debug, Clone)]
pub struct SuggestionStatistics {
    /// Total number of learned patterns
    pub total_patterns: usize,
    
    /// Number of successful applications
    pub successful_applications: usize,
    
    /// Average confidence score of generated suggestions
    pub average_confidence: f64,
    
    /// Statistics from individual providers
    pub provider_statistics: ProviderStatistics,
}

/// Statistics from suggestion providers
#[derive(Debug, Clone)]
pub struct ProviderStatistics {
    /// Number of syntax suggestions generated
    pub syntax_suggestions_generated: usize,
    /// Number of semantic suggestions generated
    pub semantic_suggestions_generated: usize,
    /// Number of business rule suggestions generated
    pub business_suggestions_generated: usize,
} 