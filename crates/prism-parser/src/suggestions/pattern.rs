//! Pattern Matching and Learning for Suggestions
//!
//! This module implements pattern recognition and learning from successful
//! suggestion applications to improve future suggestions.

use super::{
    context::SuggestionContext,
    ContextualSuggestion,
    SuggestionError,
    SuggestionResult,
};
use crate::core::error::{ParseError, ParseErrorKind};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Pattern matcher that learns from successful suggestions
pub struct PatternMatcher {
    /// Learned patterns database
    patterns: Vec<SuggestionPattern>,
    
    /// Pattern matching statistics
    pattern_applications: HashMap<String, PatternStats>,
    
    /// Configuration
    config: PatternMatcherConfig,
}

/// Configuration for pattern matching
#[derive(Debug, Clone)]
pub struct PatternMatcherConfig {
    /// Minimum confidence threshold for pattern creation
    pub min_confidence_threshold: f64,
    
    /// Maximum number of patterns to store
    pub max_patterns: usize,
    
    /// Minimum success rate to keep a pattern
    pub min_success_rate: f64,
}

impl Default for PatternMatcherConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.6,
            max_patterns: 1000,
            min_success_rate: 0.3,
        }
    }
}

/// A learned suggestion pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionPattern {
    /// Unique pattern identifier
    pub id: String,
    
    /// Pattern description
    pub description: String,
    
    /// Error pattern this applies to
    pub error_pattern: ErrorPattern,
    
    /// Context pattern this applies to
    pub context_pattern: ContextPattern,
    
    /// Suggested solution
    pub solution_template: SolutionTemplate,
    
    /// Pattern confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Number of successful applications
    pub success_count: usize,
    
    /// Total number of applications
    pub total_applications: usize,
    
    /// Pattern tags for categorization
    pub tags: Vec<String>,
}

/// Pattern for matching errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Error kind pattern
    pub error_kind: ErrorKindPattern,
    
    /// Error message pattern (regex)
    pub message_pattern: Option<String>,
    
    /// Error severity pattern
    pub severity_pattern: Option<String>,
}

/// Pattern for error kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorKindPattern {
    /// Specific error kind
    Specific(String),
    
    /// Any error kind in category
    Category(String),
    
    /// Any error kind
    Any,
}

/// Pattern for matching contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPattern {
    /// Syntax style pattern
    pub syntax_style: Option<String>,
    
    /// Token patterns around error
    pub token_patterns: Vec<String>,
    
    /// Semantic context patterns
    pub semantic_patterns: Vec<String>,
    
    /// Business context patterns
    pub business_patterns: Vec<String>,
}

/// Template for generating solutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionTemplate {
    /// Suggestion type
    pub suggestion_type: String,
    
    /// Message template with placeholders
    pub message_template: String,
    
    /// Explanation template
    pub explanation_template: String,
    
    /// Estimated effort level
    pub effort_level: String,
    
    /// Whether auto-applicable
    pub auto_applicable: bool,
}

/// Statistics for pattern application
#[derive(Debug, Clone)]
pub struct PatternStats {
    /// Number of times pattern was applied
    pub applications: usize,
    
    /// Number of successful applications
    pub successes: usize,
    
    /// Average confidence of applications
    pub average_confidence: f64,
    
    /// Last application timestamp
    pub last_used: std::time::SystemTime,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new() -> SuggestionResult<Self> {
        Ok(Self {
            patterns: Vec::new(),
            pattern_applications: HashMap::new(),
            config: PatternMatcherConfig::default(),
        })
    }
    
    /// Find patterns matching the given error and context
    pub fn find_matching_patterns(
        &self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<SuggestionPattern>> {
        let mut matching_patterns = Vec::new();
        
        for pattern in &self.patterns {
            if self.pattern_matches(pattern, error, context)? {
                matching_patterns.push(pattern.clone());
            }
        }
        
        // Sort by success rate
        matching_patterns.sort_by(|a, b| {
            let success_rate_a = if a.total_applications > 0 {
                a.success_count as f64 / a.total_applications as f64
            } else {
                0.0
            };
            let success_rate_b = if b.total_applications > 0 {
                b.success_count as f64 / b.total_applications as f64
            } else {
                0.0
            };
            success_rate_b.partial_cmp(&success_rate_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(matching_patterns)
    }
    
    /// Check if a pattern matches the error and context
    fn pattern_matches(
        &self,
        pattern: &SuggestionPattern,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<bool> {
        // Match error pattern
        if !self.error_pattern_matches(&pattern.error_pattern, error)? {
            return Ok(false);
        }
        
        // Match context pattern
        if !self.context_pattern_matches(&pattern.context_pattern, context)? {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Check if error pattern matches
    fn error_pattern_matches(
        &self,
        pattern: &ErrorPattern,
        error: &ParseError,
    ) -> SuggestionResult<bool> {
        // Check error kind
        match &pattern.error_kind {
            ErrorKindPattern::Specific(kind) => {
                let error_kind_str = format!("{:?}", error.kind);
                if !error_kind_str.contains(kind) {
                    return Ok(false);
                }
            }
            ErrorKindPattern::Category(category) => {
                // Simple category matching - could be more sophisticated
                let error_kind_str = format!("{:?}", error.kind);
                if !error_kind_str.contains(category) {
                    return Ok(false);
                }
            }
            ErrorKindPattern::Any => {
                // Always matches
            }
        }
        
        // Check message pattern if specified
        if let Some(message_pattern) = &pattern.message_pattern {
            if !error.message.contains(message_pattern) {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Check if context pattern matches
    fn context_pattern_matches(
        &self,
        pattern: &ContextPattern,
        context: &SuggestionContext,
    ) -> SuggestionResult<bool> {
        // Check syntax style if specified
        if let Some(style_pattern) = &pattern.syntax_style {
            if let Some(syntax_style) = &context.syntactic_context.syntax_style {
                let style_str = format!("{:?}", syntax_style);
                if !style_str.contains(style_pattern) {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        
        // Check token patterns
        for token_pattern in &pattern.token_patterns {
            let tokens_match = context.syntactic_context.preceding_tokens
                .iter()
                .chain(context.syntactic_context.following_tokens.iter())
                .any(|token| token.contains(token_pattern));
            
            if !tokens_match {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Record a suggestion attempt for pattern learning
    pub fn record_suggestion_attempt(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
        suggestions: &[ContextualSuggestion],
    ) -> SuggestionResult<()> {
        // For now, just log the attempt
        // A full implementation would analyze patterns and create new ones
        Ok(())
    }
    
    /// Record successful application of a suggestion
    pub fn record_successful_application(
        &mut self,
        error: &ParseError,
        suggestion: &ContextualSuggestion,
        outcome_quality: f64,
    ) -> SuggestionResult<()> {
        // Find matching patterns and update their success rates
        for pattern in &mut self.patterns {
            // Simple matching for now - could be more sophisticated
            if suggestion.pattern_tags.iter().any(|tag| pattern.tags.contains(tag)) {
                pattern.success_count += 1;
                pattern.total_applications += 1;
                
                // Update confidence based on outcome quality
                pattern.confidence = (pattern.confidence + outcome_quality) / 2.0;
            }
        }
        
        // Learn new patterns from successful applications
        if outcome_quality > 0.8 {
            self.learn_new_pattern(error, suggestion)?;
        }
        
        Ok(())
    }
    
    /// Learn a new pattern from successful application
    fn learn_new_pattern(
        &mut self,
        error: &ParseError,
        suggestion: &ContextualSuggestion,
    ) -> SuggestionResult<()> {
        // Create a new pattern based on the successful suggestion
        let pattern = SuggestionPattern {
            id: format!("learned_pattern_{}", self.patterns.len()),
            description: format!("Learned from successful {}", suggestion.suggestion_type),
            error_pattern: ErrorPattern {
                error_kind: ErrorKindPattern::Specific(format!("{:?}", error.kind)),
                message_pattern: Some(error.message.clone()),
                severity_pattern: Some(format!("{:?}", error.severity)),
            },
            context_pattern: ContextPattern {
                syntax_style: None, // Would extract from context
                token_patterns: Vec::new(), // Would extract from context
                semantic_patterns: Vec::new(),
                business_patterns: Vec::new(),
            },
            solution_template: SolutionTemplate {
                suggestion_type: format!("{:?}", suggestion.suggestion_type),
                message_template: suggestion.suggestion.message.clone(),
                explanation_template: suggestion.explanation.clone(),
                effort_level: format!("{:?}", suggestion.estimated_effort),
                auto_applicable: suggestion.auto_applicable,
            },
            confidence: suggestion.confidence,
            success_count: 1,
            total_applications: 1,
            tags: suggestion.pattern_tags.clone(),
        };
        
        // Add pattern if we haven't reached the limit
        if self.patterns.len() < self.config.max_patterns {
            self.patterns.push(pattern);
        }
        
        Ok(())
    }
    
    /// Get number of learned patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
    
    /// Get total number of successful applications
    pub fn success_count(&self) -> usize {
        self.patterns.iter().map(|p| p.success_count).sum()
    }
    
    /// Clean up patterns with low success rates
    pub fn cleanup_patterns(&mut self) {
        self.patterns.retain(|pattern| {
            if pattern.total_applications == 0 {
                return true; // Keep new patterns
            }
            
            let success_rate = pattern.success_count as f64 / pattern.total_applications as f64;
            success_rate >= self.config.min_success_rate
        });
    }
}

impl SuggestionPattern {
    /// Get pattern description
    pub fn description(&self) -> &str {
        &self.description
    }
    
    /// Get pattern confidence
    pub fn confidence(&self) -> f64 {
        self.confidence
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_applications == 0 {
            0.0
        } else {
            self.success_count as f64 / self.total_applications as f64
        }
    }
} 