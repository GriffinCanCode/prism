//! Parser Bridge - Backward Compatibility Layer
//!
//! This module provides a bridge between the existing Parser API and the new
//! factory-based orchestrator system. It maintains backward compatibility while
//! internally using the improved architecture.
//!
//! ## Design Principles
//!
//! 1. **Backward Compatibility**: Existing Parser API remains unchanged
//! 2. **Internal Modernization**: Uses factory-based orchestrator internally
//! 3. **Gradual Migration**: Allows gradual migration to new API
//! 4. **Performance**: No performance penalty for compatibility layer

use crate::{
    core::{
        orchestrator::{ParsingOrchestrator, OrchestratorConfig, ParseResult as OrchestratorParseResult},
        factories::{ParserFactory, NormalizerFactory, ValidatorFactory},
        parser::{ParseContext, ParseError, ValidationLevel},
    },
    detection::{SyntaxDetector, SyntaxStyle, DetectionResult},
    normalization::{Normalizer, CanonicalForm},
    validation::{Validator, ValidationResult},
};
use prism_common::SourceId;
use prism_ast::Program;
use std::collections::HashMap;

/// Bridge that maintains backward compatibility with the original Parser API
#[derive(Debug)]
pub struct ParserBridge {
    /// Internal orchestrator using factory system
    orchestrator: ParsingOrchestrator,
    /// Current parsing context for compatibility
    context: ParseContext,
}

impl ParserBridge {
    /// Create a new parser bridge with default configuration
    pub fn new() -> Self {
        Self {
            orchestrator: ParsingOrchestrator::new(),
            context: ParseContext::default(),
        }
    }
    
    /// Create a parser bridge with explicit syntax style preference
    pub fn with_style(style: SyntaxStyle) -> Self {
        let mut bridge = Self::new();
        bridge.context.style_preference = Some(style);
        bridge
    }
    
    /// Create a parser bridge with custom context
    pub fn with_context(context: ParseContext) -> Self {
        let mut bridge = Self::new();
        bridge.context = context;
        bridge
    }
    
    /// Parse source code into a complete program AST (backward compatible)
    pub fn parse_source(&mut self, source: &str, source_id: SourceId) -> Result<Program, ParseError> {
        // Update internal context
        self.update_orchestrator_from_context();
        
        // Use orchestrator to parse
        let result = self.orchestrator.parse(source, source_id)
            .map_err(|e| ParseError::ParsingFailed { 
                syntax: SyntaxStyle::Canonical, // Default fallback
                reason: e.to_string() 
            })?;
        
        Ok(result.program)
    }
    
    /// Parse with full analysis (backward compatible)
    pub fn parse_with_full_analysis(&mut self, source: &str, source_id: SourceId) -> Result<crate::core::parser::ParseResult, ParseError> {
        // Enable all analysis features in context
        self.context.generate_ai_metadata = true;
        self.context.validate_documentation = true;
        self.context.analyze_cohesion = true;
        self.context.analyze_semantic_types = true;
        self.context.analyze_effects = true;
        
        self.update_orchestrator_from_context();
        
        // Use orchestrator to parse
        let orchestrator_result = self.orchestrator.parse(source, source_id)
            .map_err(|e| ParseError::ParsingFailed { 
                syntax: SyntaxStyle::Canonical,
                reason: e.to_string() 
            })?;
        
        // Convert orchestrator result to legacy ParseResult format
        Ok(crate::core::parser::ParseResult {
            program: orchestrator_result.program,
            detected_style: orchestrator_result.detected_style,
            confidence: orchestrator_result.confidence,
            validation: orchestrator_result.validation,
            canonical_form: orchestrator_result.canonical_form,
            ai_metadata: None, // Would be populated from canonical_form
            cohesion_analysis: None, // Would be populated from canonical_form
            documentation_validation: None, // Would be populated from validation
            diagnostics: orchestrator_result.diagnostics.into_iter().map(|d| {
                crate::core::parser::ParseDiagnostic {
                    level: match d.level {
                        crate::core::orchestrator::DiagnosticLevel::Error => crate::core::parser::DiagnosticLevel::Error,
                        crate::core::orchestrator::DiagnosticLevel::Warning => crate::core::parser::DiagnosticLevel::Warning,
                        crate::core::orchestrator::DiagnosticLevel::Info => crate::core::parser::DiagnosticLevel::Info,
                        crate::core::orchestrator::DiagnosticLevel::Hint => crate::core::parser::DiagnosticLevel::Hint,
                    },
                    message: d.message,
                    location: d.location.unwrap_or_default(),
                    code: None,
                    suggestions: d.suggestions,
                }
            }).collect(),
        })
    }
    
    /// Set parsing context (backward compatible)
    pub fn set_context(&mut self, context: ParseContext) {
        self.context = context;
        self.update_orchestrator_from_context();
    }
    
    /// Get current parsing context (backward compatible)
    pub fn context(&self) -> &ParseContext {
        &self.context
    }
    
    /// Update orchestrator configuration from parsing context
    fn update_orchestrator_from_context(&mut self) {
        let orchestrator_config = OrchestratorConfig {
            enable_component_caching: true, // Always enabled for performance
            max_cache_size: 10,
            enable_parallel_processing: false, // Conservative for compatibility
            default_validation_level: self.context.validation_level,
            generate_ai_metadata: self.context.generate_ai_metadata,
            preserve_formatting: self.context.preserve_formatting,
            enable_error_recovery: true, // Always enabled
        };
        
        self.orchestrator.update_config(orchestrator_config);
    }
    
    /// Get access to the internal orchestrator (for advanced usage)
    pub fn orchestrator(&mut self) -> &mut ParsingOrchestrator {
        &mut self.orchestrator
    }
    
    /// Get orchestrator performance metrics
    pub fn performance_metrics(&self) -> Option<&crate::core::orchestrator::CacheStats> {
        Some(self.orchestrator.cache_stats())
    }
}

impl Default for ParserBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parser_bridge_creation() {
        let bridge = ParserBridge::new();
        assert!(bridge.context.generate_ai_metadata);
    }
    
    #[test]
    fn test_parser_bridge_with_style() {
        let bridge = ParserBridge::with_style(SyntaxStyle::CLike);
        assert_eq!(bridge.context.style_preference, Some(SyntaxStyle::CLike));
    }
    
    #[test]
    fn test_context_updates() {
        let mut bridge = ParserBridge::new();
        let mut context = ParseContext::default();
        context.generate_ai_metadata = false;
        
        bridge.set_context(context);
        assert!(!bridge.context().generate_ai_metadata);
    }
} 