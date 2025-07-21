//! Semantic-Level Suggestion Provider
//!
//! This provider generates suggestions based on semantic analysis, integrating with
//! the existing semantic engine and type inference systems.

use super::SuggestionProvider;
use crate::{
    core::error::ParseError,
    suggestions::{
        context::{SuggestionContext, EnrichedSemanticContext},
        ContextualSuggestion,
        SuggestionType,
        EffortLevel,
        SuggestionError,
        SuggestionResult,
    },
};
use prism_common::diagnostics::Suggestion;

/// Provider for semantic-level suggestions
pub struct SemanticSuggestionProvider {
    /// Statistics
    suggestions_generated: usize,
}

impl SemanticSuggestionProvider {
    /// Create a new semantic suggestion provider
    pub fn new(_config: &crate::suggestions::engine::SuggestionEngineConfig) -> SuggestionResult<Self> {
        Ok(Self {
            suggestions_generated: 0,
        })
    }
    
    /// Generate semantic suggestions based on context
    fn generate_semantic_suggestions(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Some(semantic_context) = &context.semantic_context {
            // Type inference suggestions
            suggestions.extend(self.generate_type_suggestions(error, semantic_context)?);
            
            // Function context suggestions
            suggestions.extend(self.generate_function_suggestions(error, semantic_context)?);
            
            // Module context suggestions
            suggestions.extend(self.generate_module_suggestions(error, semantic_context)?);
            
            // Effect system suggestions
            suggestions.extend(self.generate_effect_suggestions(error, semantic_context)?);
        }
        
        self.suggestions_generated += suggestions.len();
        Ok(suggestions)
    }
    
    /// Generate type-related suggestions
    fn generate_type_suggestions(
        &self,
        _error: &ParseError,
        semantic_context: &EnrichedSemanticContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Suggest available types in scope
        for (name, type_info) in &semantic_context.type_context {
            suggestions.push(ContextualSuggestion {
                suggestion: Suggestion::new(format!("Consider using type '{}'", name)),
                suggestion_type: SuggestionType::TypeGuidance,
                confidence: 0.6,
                explanation: format!("Type '{}' is available in current scope: {}", name, type_info),
                context_summary: "Type available in scope".to_string(),
                pattern_tags: vec!["type_available".to_string(), "semantic".to_string()],
                ai_insights: vec![format!("Semantic analysis found type: {}", name)],
                estimated_effort: EffortLevel::Simple,
                auto_applicable: false,
            });
        }
        
        Ok(suggestions)
    }
    
    /// Generate function-related suggestions
    fn generate_function_suggestions(
        &self,
        _error: &ParseError,
        semantic_context: &EnrichedSemanticContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Suggest available functions
        for function_info in &semantic_context.function_context {
            suggestions.push(ContextualSuggestion {
                suggestion: Suggestion::new(format!("Consider function '{}'", function_info.name)),
                suggestion_type: SuggestionType::SemanticCompletion,
                confidence: 0.7,
                explanation: format!("Function '{}' with signature: {}", 
                    function_info.name, function_info.signature),
                context_summary: "Function available in scope".to_string(),
                pattern_tags: vec!["function_available".to_string(), "semantic".to_string()],
                ai_insights: vec![format!("Semantic analysis found function: {}", function_info.name)],
                estimated_effort: EffortLevel::Simple,
                auto_applicable: false,
            });
        }
        
        Ok(suggestions)
    }
    
    /// Generate module-related suggestions
    fn generate_module_suggestions(
        &self,
        _error: &ParseError,
        semantic_context: &EnrichedSemanticContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        if let Some(module_info) = &semantic_context.module_context {
            // Check module cohesion
            if let Some(cohesion_score) = module_info.cohesion_score {
                if cohesion_score < 0.7 {
                    suggestions.push(ContextualSuggestion {
                        suggestion: Suggestion::new("Consider improving module cohesion"),
                        suggestion_type: SuggestionType::ArchitecturalGuidance,
                        confidence: 0.8,
                        explanation: format!("Module cohesion score is {:.2}, below recommended 0.7", cohesion_score),
                        context_summary: "Low module cohesion detected".to_string(),
                        pattern_tags: vec!["cohesion".to_string(), "architecture".to_string()],
                        ai_insights: vec!["Module cohesion analysis suggests improvement".to_string()],
                        estimated_effort: EffortLevel::Major,
                        auto_applicable: false,
                    });
                }
            }
        }
        
        Ok(suggestions)
    }
    
    /// Generate effect system suggestions
    fn generate_effect_suggestions(
        &self,
        _error: &ParseError,
        semantic_context: &EnrichedSemanticContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Suggest relevant effects
        for effect in &semantic_context.effect_context {
            suggestions.push(ContextualSuggestion {
                suggestion: Suggestion::new(format!("Consider effect '{}'", effect)),
                suggestion_type: SuggestionType::EffectGuidance,
                confidence: 0.6,
                explanation: format!("Effect '{}' is available in current context", effect),
                context_summary: "Effect available in context".to_string(),
                pattern_tags: vec!["effect_available".to_string(), "semantic".to_string()],
                ai_insights: vec![format!("Semantic analysis found effect: {}", effect)],
                estimated_effort: EffortLevel::Moderate,
                auto_applicable: false,
            });
        }
        
        Ok(suggestions)
    }
}

impl SuggestionProvider for SemanticSuggestionProvider {
    fn generate_suggestions(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Option<Vec<ContextualSuggestion>>> {
        let suggestions = self.generate_semantic_suggestions(error, context)?;
        Ok(if suggestions.is_empty() { None } else { Some(suggestions) })
    }
    
    fn suggestions_generated(&self) -> usize {
        self.suggestions_generated
    }
    
    fn provider_name(&self) -> &'static str {
        "SemanticSuggestionProvider"
    }
} 