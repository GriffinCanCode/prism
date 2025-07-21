//! Syntax-Level Suggestion Provider
//!
//! This provider generates suggestions for syntax errors by extending the existing
//! error suggestion functionality in the parser recovery system.

use super::SuggestionProvider;
use crate::{
    core::error::{ParseError, ParseErrorKind},
    core::recovery::RecoveryStrategy,
    suggestions::{
        context::SuggestionContext,
        ContextualSuggestion,
        SuggestionType,
        EffortLevel,
        SuggestionError,
        SuggestionResult,
    },
};
use prism_common::diagnostics::{Suggestion, Replacement, SuggestionStyle};
use prism_lexer::TokenKind;

/// Provider for syntax-level suggestions
pub struct SyntaxSuggestionProvider {
    /// Statistics
    suggestions_generated: usize,
}

impl SyntaxSuggestionProvider {
    /// Create a new syntax suggestion provider
    pub fn new(_config: &crate::suggestions::engine::SuggestionEngineConfig) -> SuggestionResult<Self> {
        Ok(Self {
            suggestions_generated: 0,
        })
    }
    
    /// Generate syntax-specific suggestions
    fn generate_syntax_suggestions(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        match &error.kind {
            ParseErrorKind::UnexpectedToken { expected, found } => {
                suggestions.extend(self.handle_unexpected_token(expected, found, context)?);
            }
            
            ParseErrorKind::UnexpectedEof { expected } => {
                suggestions.extend(self.handle_unexpected_eof(expected, context)?);
            }
            
            ParseErrorKind::InvalidSyntax { construct, details } => {
                suggestions.extend(self.handle_invalid_syntax(construct, details, context)?);
            }
            
            _ => {
                // For other error types, provide generic syntax suggestions
                suggestions.extend(self.generate_generic_syntax_suggestions(error, context)?);
            }
        }
        
        self.suggestions_generated += suggestions.len();
        Ok(suggestions)
    }
    
    /// Handle unexpected token errors
    fn handle_unexpected_token(
        &self,
        expected: &[TokenKind],
        found: &TokenKind,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        if expected.len() == 1 {
            let expected_token = &expected[0];
            
            // Direct replacement suggestion
            let replacement_text = self.token_to_text(expected_token);
            let suggestion = ContextualSuggestion {
                suggestion: Suggestion::with_replacements(
                    format!("Replace '{:?}' with '{}'", found, replacement_text),
                    vec![Replacement::new(context.error_info.span, replacement_text.clone())],
                ).definitely_correct(),
                suggestion_type: SuggestionType::SyntaxFix,
                confidence: 0.9,
                explanation: format!(
                    "The parser expected '{}' but found '{:?}'. This is a direct token replacement.",
                    replacement_text, found
                ),
                context_summary: format!("Expected token mismatch in {:?} context", 
                    context.syntactic_context.syntax_style),
                pattern_tags: vec!["token_replacement".to_string(), "syntax_fix".to_string()],
                ai_insights: vec![
                    format!("High confidence direct replacement from {:?} to {}", found, replacement_text)
                ],
                estimated_effort: EffortLevel::Trivial,
                auto_applicable: true,
            };
            suggestions.push(suggestion);
        } else {
            // Multiple expected tokens - provide options
            for expected_token in expected {
                let replacement_text = self.token_to_text(expected_token);
                let suggestion = ContextualSuggestion {
                    suggestion: Suggestion::with_replacements(
                        format!("Try using '{}'", replacement_text),
                        vec![Replacement::new(context.error_info.span, replacement_text.clone())],
                    ).maybe_incorrect(),
                    suggestion_type: SuggestionType::SyntaxFix,
                    confidence: 0.7 / expected.len() as f64, // Lower confidence for multiple options
                    explanation: format!(
                        "One of several possible tokens. '{}' is valid in this context.",
                        replacement_text
                    ),
                    context_summary: format!("Multiple token options available"),
                    pattern_tags: vec!["multiple_options".to_string(), "syntax_fix".to_string()],
                    ai_insights: vec![
                        format!("Alternative token option: {}", replacement_text)
                    ],
                    estimated_effort: EffortLevel::Simple,
                    auto_applicable: false, // Requires user choice
                };
                suggestions.push(suggestion);
            }
        }
        
        // Context-specific suggestions based on syntax style
        if let Some(style_suggestion) = self.generate_style_specific_suggestion(expected, found, context)? {
            suggestions.push(style_suggestion);
        }
        
        Ok(suggestions)
    }
    
    /// Handle unexpected EOF errors
    fn handle_unexpected_eof(
        &self,
        expected: &[TokenKind],
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        for expected_token in expected {
            let missing_text = self.token_to_text(expected_token);
            let suggestion = ContextualSuggestion {
                suggestion: Suggestion::new(
                    format!("Add missing '{}'", missing_text)
                ),
                suggestion_type: SuggestionType::SyntaxFix,
                confidence: 0.8,
                explanation: format!(
                    "The file ended unexpectedly. Adding '{}' may complete the syntax.",
                    missing_text
                ),
                context_summary: "Unexpected end of file".to_string(),
                pattern_tags: vec!["missing_delimiter".to_string(), "eof_error".to_string()],
                ai_insights: vec![
                    format!("EOF reached while expecting: {}", missing_text)
                ],
                estimated_effort: EffortLevel::Simple,
                auto_applicable: true,
            };
            suggestions.push(suggestion);
        }
        
        Ok(suggestions)
    }
    
    /// Handle invalid syntax construct errors
    fn handle_invalid_syntax(
        &self,
        construct: &str,
        details: &str,
        context: &SuggestionContext,
    ) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Generate construct-specific suggestions
        let suggestion = match construct {
            "function" => self.suggest_function_syntax(details, context)?,
            "type" => self.suggest_type_syntax(details, context)?,
            "module" => self.suggest_module_syntax(details, context)?,
            _ => self.suggest_generic_construct_fix(construct, details, context)?,
        };
        
        if let Some(s) = suggestion {
            suggestions.push(s);
        }
        
        Ok(suggestions)
    }
    
    /// Generate style-specific suggestions
    fn generate_style_specific_suggestion(
        &self,
        expected: &[TokenKind],
        found: &TokenKind,
        context: &SuggestionContext,
    ) -> SuggestionResult<Option<ContextualSuggestion>> {
        if let Some(syntax_style) = &context.syntactic_context.syntax_style {
            match syntax_style {
                prism_syntax::SyntaxStyle::CLike => {
                    // C-like specific suggestions (braces, semicolons)
                    if expected.contains(&TokenKind::RightBrace) {
                        return Ok(Some(ContextualSuggestion {
                            suggestion: Suggestion::new("Add missing closing brace '}'"),
                            suggestion_type: SuggestionType::SyntaxFix,
                            confidence: 0.85,
                            explanation: "C-like syntax requires closing braces for blocks".to_string(),
                            context_summary: "C-like syntax style detected".to_string(),
                            pattern_tags: vec!["c_like".to_string(), "brace_matching".to_string()],
                            ai_insights: vec!["C-like syntax pattern detected".to_string()],
                            estimated_effort: EffortLevel::Trivial,
                            auto_applicable: true,
                        }));
                    }
                }
                
                prism_syntax::SyntaxStyle::PythonLike => {
                    // Python-like specific suggestions (indentation, colons)
                    if expected.contains(&TokenKind::Colon) {
                        return Ok(Some(ContextualSuggestion {
                            suggestion: Suggestion::new("Add colon ':' for Python-like syntax"),
                            suggestion_type: SuggestionType::SyntaxFix,
                            confidence: 0.85,
                            explanation: "Python-like syntax requires colons before indented blocks".to_string(),
                            context_summary: "Python-like syntax style detected".to_string(),
                            pattern_tags: vec!["python_like".to_string(), "colon_required".to_string()],
                            ai_insights: vec!["Python-like syntax pattern detected".to_string()],
                            estimated_effort: EffortLevel::Trivial,
                            auto_applicable: true,
                        }));
                    }
                }
                
                _ => {} // Other syntax styles
            }
        }
        
        Ok(None)
    }
    
    /// Convert token to text representation
    fn token_to_text(&self, token: &TokenKind) -> String {
        match token {
            TokenKind::LeftBrace => "{".to_string(),
            TokenKind::RightBrace => "}".to_string(),
            TokenKind::LeftParen => "(".to_string(),
            TokenKind::RightParen => ")".to_string(),
            TokenKind::LeftBracket => "[".to_string(),
            TokenKind::RightBracket => "]".to_string(),
            TokenKind::Semicolon => ";".to_string(),
            TokenKind::Comma => ",".to_string(),
            TokenKind::Colon => ":".to_string(),
            TokenKind::Arrow => "->".to_string(),
            _ => format!("{:?}", token),
        }
    }
    
    /// Suggest function syntax fixes
    fn suggest_function_syntax(&self, _details: &str, _context: &SuggestionContext) -> SuggestionResult<Option<ContextualSuggestion>> {
        Ok(Some(ContextualSuggestion {
            suggestion: Suggestion::new("Function syntax: function name(params) -> ReturnType { body }"),
            suggestion_type: SuggestionType::SyntaxFix,
            confidence: 0.7,
            explanation: "Functions require proper declaration syntax".to_string(),
            context_summary: "Function declaration error".to_string(),
            pattern_tags: vec!["function_syntax".to_string()],
            ai_insights: vec!["Function syntax template provided".to_string()],
            estimated_effort: EffortLevel::Moderate,
            auto_applicable: false,
        }))
    }
    
    /// Suggest type syntax fixes
    fn suggest_type_syntax(&self, _details: &str, _context: &SuggestionContext) -> SuggestionResult<Option<ContextualSuggestion>> {
        Ok(Some(ContextualSuggestion {
            suggestion: Suggestion::new("Type syntax: type Name = TypeDefinition"),
            suggestion_type: SuggestionType::TypeGuidance,
            confidence: 0.7,
            explanation: "Types require proper declaration syntax".to_string(),
            context_summary: "Type declaration error".to_string(),
            pattern_tags: vec!["type_syntax".to_string()],
            ai_insights: vec!["Type syntax template provided".to_string()],
            estimated_effort: EffortLevel::Moderate,
            auto_applicable: false,
        }))
    }
    
    /// Suggest module syntax fixes
    fn suggest_module_syntax(&self, _details: &str, _context: &SuggestionContext) -> SuggestionResult<Option<ContextualSuggestion>> {
        Ok(Some(ContextualSuggestion {
            suggestion: Suggestion::new("Module syntax: module Name { sections }"),
            suggestion_type: SuggestionType::ArchitecturalGuidance,
            confidence: 0.7,
            explanation: "Modules require proper declaration syntax".to_string(),
            context_summary: "Module declaration error".to_string(),
            pattern_tags: vec!["module_syntax".to_string()],
            ai_insights: vec!["Module syntax template provided".to_string()],
            estimated_effort: EffortLevel::Moderate,
            auto_applicable: false,
        }))
    }
    
    /// Generic construct fix suggestion
    fn suggest_generic_construct_fix(&self, construct: &str, details: &str, _context: &SuggestionContext) -> SuggestionResult<Option<ContextualSuggestion>> {
        Ok(Some(ContextualSuggestion {
            suggestion: Suggestion::new(format!("Fix {} syntax: {}", construct, details)),
            suggestion_type: SuggestionType::SyntaxFix,
            confidence: 0.5,
            explanation: format!("Generic syntax issue with {}", construct),
            context_summary: format!("Invalid {} construct", construct),
            pattern_tags: vec!["generic_syntax".to_string()],
            ai_insights: vec![format!("Syntax error in {}: {}", construct, details)],
            estimated_effort: EffortLevel::Moderate,
            auto_applicable: false,
        }))
    }
    
    /// Generate generic syntax suggestions
    fn generate_generic_syntax_suggestions(&self, error: &ParseError, _context: &SuggestionContext) -> SuggestionResult<Vec<ContextualSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Add a generic "check syntax" suggestion
        suggestions.push(ContextualSuggestion {
            suggestion: Suggestion::new("Check syntax against language documentation"),
            suggestion_type: SuggestionType::SyntaxFix,
            confidence: 0.3,
            explanation: "This appears to be a syntax error. Refer to documentation for correct syntax.".to_string(),
            context_summary: "Generic syntax error".to_string(),
            pattern_tags: vec!["generic".to_string()],
            ai_insights: vec!["Generic syntax error detected".to_string()],
            estimated_effort: EffortLevel::Complex,
            auto_applicable: false,
        });
        
        Ok(suggestions)
    }
}

impl SuggestionProvider for SyntaxSuggestionProvider {
    fn generate_suggestions(
        &mut self,
        error: &ParseError,
        context: &SuggestionContext,
    ) -> SuggestionResult<Option<Vec<ContextualSuggestion>>> {
        let suggestions = self.generate_syntax_suggestions(error, context)?;
        Ok(if suggestions.is_empty() { None } else { Some(suggestions) })
    }
    
    fn suggestions_generated(&self) -> usize {
        self.suggestions_generated
    }
    
    fn provider_name(&self) -> &'static str {
        "SyntaxSuggestionProvider"
    }
} 