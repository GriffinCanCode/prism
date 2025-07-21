//! Business Rule Suggestion Provider
//!
//! This provider generates suggestions based on business rules and domain context,
//! integrating with existing business rule validation infrastructure.

use super::SuggestionProvider;
use crate::{
    core::error::ParseError,
    suggestions::{
        context::{SuggestionContext, BusinessContext},
        ContextualSuggestion,
        SuggestionType,
        EffortLevel,
        SuggestionError,
        SuggestionResult,
    },
};
use prism_common::diagnostics::Suggestion;

/// Provider for business rule suggestions
pub struct BusinessSuggestionProvider {
    /// Statistics
    suggestions_generated: usize,
}

impl BusinessSuggestionProvider {
    /// Create a new business suggestion provider
    pub fn new(_config: &crate::suggestions::engine::SuggestionEngineConfig) -> SuggestionResult<Self> {
        Ok(Self {
            suggestions_generated: 0,
        })
    }
    
    /// Generate business rule suggestions
    fn generate_business_suggestions(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Some(business_context) = &context.business_context {
            // Business capability suggestions
            suggestions.extend(self.generate_capability_suggestions(error, business_context)?);
            
            // Business rule compliance suggestions
            suggestions.extend(self.generate_rule_compliance_suggestions(error, business_context)?);
            
            // Domain-specific suggestions
            suggestions.extend(self.generate_domain_suggestions(error, business_context)?);
        }
        
        self.suggestions_generated += suggestions.len();
        Ok(suggestions)
    }
    
    /// Generate business capability suggestions
    fn generate_capability_suggestions(
        &self,
        _error: &ParseError,
        business_context: &BusinessContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Some(capability) = &business_context.business_capability {
            suggestions.push(ContextualSuggestion {
                suggestion: Suggestion::new(format!("Align with business capability: {}", capability)),
                suggestion_type: SuggestionType::BusinessRuleGuidance,
                confidence: 0.8,
                explanation: format!("This code should align with the '{}' business capability", capability),
                context_summary: "Business capability alignment".to_string(),
                pattern_tags: vec!["business_capability".to_string()],
                ai_insights: vec![format!("Business context: {}", capability)],
                estimated_effort: EffortLevel::Moderate,
                auto_applicable: false,
            });
        }
        
        Ok(suggestions)
    }
    
    /// Generate business rule compliance suggestions
    fn generate_rule_compliance_suggestions(
        &self,
        _error: &ParseError,
        business_context: &BusinessContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        for rule in &business_context.business_rules {
            suggestions.push(ContextualSuggestion {
                suggestion: Suggestion::new(format!("Consider business rule: {}", rule.name)),
                suggestion_type: SuggestionType::BusinessRuleGuidance,
                confidence: 0.7,
                explanation: format!("Business rule '{}': {}", rule.name, rule.description),
                context_summary: "Business rule compliance".to_string(),
                pattern_tags: vec!["business_rule".to_string()],
                ai_insights: vec![format!("Rule category: {:?}", rule.category)],
                estimated_effort: EffortLevel::Moderate,
                auto_applicable: false,
            });
        }
        
        Ok(suggestions)
    }
    
    /// Generate domain-specific suggestions
    fn generate_domain_suggestions(
        &self,
        _error: &ParseError,
        business_context: &BusinessContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Some(domain) = &business_context.domain {
            suggestions.push(ContextualSuggestion {
                suggestion: Suggestion::new(format!("Apply {}-specific patterns", domain)),
                suggestion_type: SuggestionType::BusinessRuleGuidance,
                confidence: 0.6,
                explanation: format!("Consider domain-specific patterns for {}", domain),
                context_summary: format!("Domain: {}", domain),
                pattern_tags: vec!["domain_specific".to_string()],
                ai_insights: vec![format!("Domain context: {}", domain)],
                estimated_effort: EffortLevel::Complex,
                auto_applicable: false,
            });
        }
        
        Ok(suggestions)
    }
}

impl SuggestionProvider for BusinessSuggestionProvider {
    fn generate_suggestions(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Option<Vec<ContextualSuggestion>>> {
        let suggestions = self.generate_business_suggestions(error, context)?;
        Ok(if suggestions.is_empty() { None } else { Some(suggestions) })
    }
    
    fn suggestions_generated(&self) -> usize {
        self.suggestions_generated
    }
    
    fn provider_name(&self) -> &'static str {
        "BusinessSuggestionProvider"
    }
} 