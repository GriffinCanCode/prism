//! Suggestion Ranking and Confidence Scoring
//!
//! This module implements intelligent ranking of suggestions based on context,
//! confidence scores, and user preferences.

use super::{
    context::SuggestionContext,
    ContextualSuggestion,
    SuggestionType,
    EffortLevel,
    SuggestionError,
    SuggestionResult,
};

/// Ranked suggestion with computed priority
#[derive(Debug, Clone)]
pub struct RankedSuggestion {
    /// The suggestion
    pub suggestion: ContextualSuggestion,
    
    /// Computed ranking score (higher is better)
    pub ranking_score: f64,
    
    /// Confidence in this suggestion
    pub confidence: f64,
}

/// Suggestion ranker that prioritizes suggestions
pub struct SuggestionRanker {
    /// Configuration for ranking
    config: RankingConfig,
    
    /// Statistics for tracking
    total_suggestions_ranked: usize,
    
    /// Running average of confidence scores
    confidence_sum: f64,
}

/// Configuration for suggestion ranking
#[derive(Debug, Clone)]
pub struct RankingConfig {
    /// Weight for confidence score in ranking
    pub confidence_weight: f64,
    
    /// Weight for effort level in ranking (lower effort = higher score)
    pub effort_weight: f64,
    
    /// Weight for suggestion type priority
    pub type_weight: f64,
    
    /// Weight for auto-applicability
    pub auto_applicable_weight: f64,
    
    /// Boost for suggestions with pattern tags
    pub pattern_boost: f64,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            confidence_weight: 0.4,
            effort_weight: 0.2,
            type_weight: 0.2,
            auto_applicable_weight: 0.1,
            pattern_boost: 0.1,
        }
    }
}

impl SuggestionRanker {
    /// Create a new suggestion ranker
    pub fn new(_config: &super::engine::SuggestionEngineConfig) -> Self {
        Self {
            config: RankingConfig::default(),
            total_suggestions_ranked: 0,
            confidence_sum: 0.0,
        }
    }
    
    /// Rank a list of suggestions by priority
    pub fn rank_suggestions(
        &mut self,
        suggestions: Vec<ContextualSuggestion>,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<RankedSuggestion>> {
        let mut ranked_suggestions = Vec::new();
        
        for suggestion in suggestions {
            let ranking_score = self.calculate_ranking_score(&suggestion, context);
            
            ranked_suggestions.push(RankedSuggestion {
                confidence: suggestion.confidence,
                suggestion,
                ranking_score,
            });
            
            self.total_suggestions_ranked += 1;
            self.confidence_sum += ranking_score;
        }
        
        // Sort by ranking score (descending)
        ranked_suggestions.sort_by(|a, b| {
            b.ranking_score.partial_cmp(&a.ranking_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(ranked_suggestions)
    }
    
    /// Calculate ranking score for a suggestion
    fn calculate_ranking_score(
        &self,
        suggestion: &ContextualSuggestion,
        context: &SuggestionContext,
    ) -> f64 {
        let mut score = 0.0;
        
        // Confidence component
        score += suggestion.confidence * self.config.confidence_weight;
        
        // Effort component (inverted - lower effort is better)
        let effort_score = match suggestion.estimated_effort {
            EffortLevel::Trivial => 1.0,
            EffortLevel::Simple => 0.8,
            EffortLevel::Moderate => 0.6,
            EffortLevel::Complex => 0.4,
            EffortLevel::Major => 0.2,
        };
        score += effort_score * self.config.effort_weight;
        
        // Type priority component
        let type_score = self.get_suggestion_type_priority(&suggestion.suggestion_type);
        score += type_score * self.config.type_weight;
        
        // Auto-applicability component
        if suggestion.auto_applicable {
            score += self.config.auto_applicable_weight;
        }
        
        // Pattern tag boost
        if !suggestion.pattern_tags.is_empty() {
            score += self.config.pattern_boost;
        }
        
        // Context-specific adjustments
        score = self.apply_context_adjustments(score, suggestion, context);
        
        // Ensure score is between 0.0 and 1.0
        score.clamp(0.0, 1.0)
    }
    
    /// Get priority score for suggestion type
    fn get_suggestion_type_priority(&self, suggestion_type: &SuggestionType) -> f64 {
        match suggestion_type {
            SuggestionType::SyntaxFix => 0.9,           // High priority - fixes immediate issues
            SuggestionType::TypeGuidance => 0.8,       // Important for correctness
            SuggestionType::EffectGuidance => 0.7,     // Important for safety
            SuggestionType::SemanticCompletion => 0.6, // Helpful for development
            SuggestionType::BusinessRuleGuidance => 0.5, // Important but not urgent
            SuggestionType::SecurityGuidance => 0.8,   // High priority for security
            SuggestionType::PerformanceGuidance => 0.4, // Lower priority
            SuggestionType::ArchitecturalGuidance => 0.3, // Lowest priority - long term
        }
    }
    
    /// Apply context-specific adjustments to ranking score
    fn apply_context_adjustments(
        &self,
        mut score: f64,
        suggestion: &ContextualSuggestion,
        context: &SuggestionContext,
    ) -> f64 {
        // Boost suggestions that match error severity
        match context.error_info.severity {
            crate::core::error::ErrorSeverity::Error => {
                // Prioritize syntax fixes for errors
                if matches!(suggestion.suggestion_type, SuggestionType::SyntaxFix) {
                    score += 0.1;
                }
            }
            crate::core::error::ErrorSeverity::Warning => {
                // Prioritize guidance for warnings
                if matches!(suggestion.suggestion_type, 
                    SuggestionType::BusinessRuleGuidance | 
                    SuggestionType::ArchitecturalGuidance
                ) {
                    score += 0.05;
                }
            }
            crate::core::error::ErrorSeverity::Info => {
                // All suggestions equally valid for info
            }
        }
        
        // Boost suggestions with AI insights
        if !suggestion.ai_insights.is_empty() {
            score += 0.05;
        }
        
        // Boost suggestions that match business context
        if let Some(business_context) = &context.business_context {
            if matches!(suggestion.suggestion_type, SuggestionType::BusinessRuleGuidance) {
                score += 0.1;
            }
        }
        
        // Boost suggestions that match semantic context
        if context.semantic_context.is_some() {
            if matches!(suggestion.suggestion_type, 
                SuggestionType::TypeGuidance | 
                SuggestionType::SemanticCompletion |
                SuggestionType::EffectGuidance
            ) {
                score += 0.05;
            }
        }
        
        score
    }
    
    /// Get average confidence of all ranked suggestions
    pub fn average_confidence(&self) -> f64 {
        if self.total_suggestions_ranked == 0 {
            0.0
        } else {
            self.confidence_sum / self.total_suggestions_ranked as f64
        }
    }
    
    /// Get total number of suggestions ranked
    pub fn total_ranked(&self) -> usize {
        self.total_suggestions_ranked
    }
} 