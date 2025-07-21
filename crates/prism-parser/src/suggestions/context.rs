//! Context Extraction for Suggestions
//!
//! This module extracts comprehensive context from errors to guide suggestion generation.
//! It integrates with existing semantic analysis, business rules, and AI metadata systems.

use super::{SuggestionError, SuggestionResult};
use crate::core::error::{ParseError, ErrorContext, ParseErrorKind};
use crate::analysis::semantic_context_extractor::SemanticContextExtractor;
use prism_common::span::Span;
use prism_semantic::{SemanticEngine, SemanticContext};
use prism_common::{ai_metadata::{AIMetadata, AIMetadataCollector, SemanticContextEntry, BusinessRuleEntry}};
use prism_syntax::{SyntaxStyle, detection::DetectionResult};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::collections::HashMap;

/// Comprehensive context for suggestion generation
#[derive(Debug, Clone)]
pub struct SuggestionContext {
    /// Error information
    pub error_info: ErrorInfo,
    
    /// Syntactic context
    pub syntactic_context: SyntacticContext,
    
    /// Semantic context from analysis
    pub semantic_context: Option<EnrichedSemanticContext>,
    
    /// Business context and rules
    pub business_context: Option<BusinessContext>,
    
    /// Historical context from similar errors
    pub historical_context: HistoricalContext,
}

/// Error information extracted from ParseError
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Type of error
    pub error_kind: ParseErrorKind,
    
    /// Error span
    pub span: Span,
    
    /// Error message
    pub message: String,
    
    /// Error severity
    pub severity: crate::core::error::ErrorSeverity,
    
    /// Whether error is recoverable
    pub recoverable: bool,
}

/// Syntactic context around the error
#[derive(Debug, Clone)]
pub struct SyntacticContext {
    /// Detected syntax style
    pub syntax_style: Option<SyntaxStyle>,
    
    /// Tokens before error
    pub preceding_tokens: Vec<String>,
    
    /// Tokens after error
    pub following_tokens: Vec<String>,
    
    /// Current statement context
    pub statement_context: Option<String>,
    
    /// Nesting level
    pub nesting_level: usize,
    
    /// Open delimiters
    pub open_delimiters: Vec<String>,
}

/// Enriched semantic context from semantic analysis
#[derive(Debug, Clone)]
pub struct EnrichedSemanticContext {
    /// Base semantic context
    pub base_context: SemanticContext,
    
    /// Inferred types in scope
    pub type_context: HashMap<String, String>,
    
    /// Available functions
    pub function_context: Vec<FunctionInfo>,
    
    /// Module context
    pub module_context: Option<ModuleInfo>,
    
    /// Effect context
    pub effect_context: Vec<String>,
}

/// Business context extracted from rules and annotations
#[derive(Debug, Clone)]
pub struct BusinessContext {
    /// Business capability being implemented
    pub business_capability: Option<String>,
    
    /// Applicable business rules
    pub business_rules: Vec<BusinessRuleEntry>,
    
    /// Domain context
    pub domain: Option<String>,
    
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    
    /// Architectural patterns in use
    pub architectural_patterns: Vec<String>,
}

/// Historical context from similar errors
#[derive(Debug, Clone)]
pub struct HistoricalContext {
    /// Similar errors seen before
    pub similar_errors: Vec<SimilarError>,
    
    /// Common patterns for this error type
    pub common_patterns: Vec<String>,
    
    /// Success rates of different fixes
    pub fix_success_rates: HashMap<String, f64>,
}

/// Information about a similar error
#[derive(Debug, Clone)]
pub struct SimilarError {
    /// Error signature
    pub signature: String,
    
    /// Context similarity score
    pub similarity: f64,
    
    /// How it was resolved
    pub resolution: Option<String>,
    
    /// Success of the resolution
    pub resolution_success: bool,
}

/// Function information for context
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    
    /// Function signature
    pub signature: String,
    
    /// Business purpose
    pub purpose: Option<String>,
}

/// Module information for context
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    
    /// Module capability
    pub capability: Option<String>,
    
    /// Module cohesion score
    pub cohesion_score: Option<f64>,
}

/// Context extractor that integrates with existing systems
pub struct ContextExtractor {
    /// Configuration
    config: ContextExtractorConfig,
    
    /// Integration with existing semantic context extractor
    semantic_extractor: Option<SemanticContextExtractor>,
    
    /// Historical error database (simplified for now)
    historical_db: HistoricalErrorDatabase,
}

/// Configuration for context extraction
#[derive(Debug, Clone)]
pub struct ContextExtractorConfig {
    /// Enable semantic context extraction
    pub enable_semantic_context: bool,
    
    /// Enable business context extraction
    pub enable_business_context: bool,
    
    /// Enable historical context lookup
    pub enable_historical_context: bool,
    
    /// Context window size for tokens
    pub token_context_window: usize,
}

impl ContextExtractor {
    /// Create a new context extractor
    pub fn new(config: &super::engine::SuggestionEngineConfig) -> SuggestionResult<Self> {
        let extractor_config = ContextExtractorConfig {
            enable_semantic_context: config.enable_semantic_context,
            enable_business_context: config.enable_business_rules,
            enable_historical_context: config.enable_pattern_learning,
            token_context_window: 5,
        };
        
        Ok(Self {
            config: extractor_config,
            semantic_extractor: Some(SemanticContextExtractor::new()),
            historical_db: HistoricalErrorDatabase::new(),
        })
    }
    
    /// Extract comprehensive context for suggestion generation
    pub fn extract_context(
        &self,
        error: &ParseError,
        error_context: Option<&ErrorContext>,
        semantic_engine: Option<&Arc<SemanticEngine>>,
    ) -> SuggestionResult<SuggestionContext> {
        // Extract error information
        let error_info = self.extract_error_info(error);
        
        // Extract syntactic context
        let syntactic_context = self.extract_syntactic_context(error, error_context)?;
        
        // Extract semantic context (if enabled and available)
        let semantic_context = if self.config.enable_semantic_context && semantic_engine.is_some() {
            self.extract_semantic_context(error, semantic_engine.unwrap())?
        } else {
            None
        };
        
        // Extract business context (if enabled)
        let business_context = if self.config.enable_business_context {
            self.extract_business_context(error, &semantic_context)?
        } else {
            None
        };
        
        // Extract historical context (if enabled)
        let historical_context = if self.config.enable_historical_context {
            self.extract_historical_context(error)?
        } else {
            HistoricalContext {
                similar_errors: Vec::new(),
                common_patterns: Vec::new(),
                fix_success_rates: HashMap::new(),
            }
        };
        
        Ok(SuggestionContext {
            error_info,
            syntactic_context,
            semantic_context,
            business_context,
            historical_context,
        })
    }
    
    /// Extract error information from ParseError
    fn extract_error_info(&self, error: &ParseError) -> ErrorInfo {
        ErrorInfo {
            error_kind: error.kind.clone(),
            span: error.span,
            message: error.message.clone(),
            severity: error.severity,
            recoverable: error.is_recoverable(),
        }
    }
    
    /// Extract syntactic context around the error
    fn extract_syntactic_context(
        &self,
        error: &ParseError,
        error_context: Option<&ErrorContext>,
    ) -> SuggestionResult<SyntacticContext> {
        let mut context = SyntacticContext {
            syntax_style: None, // Would be detected from source
            preceding_tokens: Vec::new(),
            following_tokens: Vec::new(),
            statement_context: None,
            nesting_level: 0,
            open_delimiters: Vec::new(),
        };
        
        // Extract token context if available
        if let Some(error_ctx) = error_context {
            context.preceding_tokens = error_ctx.before
                .iter()
                .map(|token| format!("{:?}", token.kind))
                .collect();
                
            context.following_tokens = error_ctx.after
                .iter()
                .map(|token| format!("{:?}", token.kind))
                .collect();
        }
        
        Ok(context)
    }
    
    /// Extract semantic context using semantic engine
    fn extract_semantic_context(
        &self,
        error: &ParseError,
        semantic_engine: &SemanticEngine,
    ) -> SuggestionResult<Option<EnrichedSemanticContext>> {
        // This would integrate with the actual semantic engine
        // For now, return a placeholder
        Ok(Some(EnrichedSemanticContext {
            base_context: SemanticContext::new(),
            type_context: HashMap::new(),
            function_context: Vec::new(),
            module_context: None,
            effect_context: Vec::new(),
        }))
    }
    
    /// Extract business context from rules and annotations
    fn extract_business_context(
        &self,
        error: &ParseError,
        semantic_context: &Option<EnrichedSemanticContext>,
    ) -> SuggestionResult<Option<BusinessContext>> {
        // This would integrate with business rule validation
        // For now, return a placeholder
        Ok(Some(BusinessContext {
            business_capability: None,
            business_rules: Vec::new(),
            domain: None,
            compliance_requirements: Vec::new(),
            architectural_patterns: Vec::new(),
        }))
    }
    

    
    /// Extract historical context from similar errors
    fn extract_historical_context(&self, error: &ParseError) -> SuggestionResult<HistoricalContext> {
        let similar_errors = self.historical_db.find_similar_errors(error)?;
        
        Ok(HistoricalContext {
            similar_errors,
            common_patterns: Vec::new(),
            fix_success_rates: HashMap::new(),
        })
    }
}

/// Simplified historical error database
pub struct HistoricalErrorDatabase {
    // This would be a proper database in production
    errors: Vec<SimilarError>,
}

impl HistoricalErrorDatabase {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
        }
    }
    
    pub fn find_similar_errors(&self, _error: &ParseError) -> SuggestionResult<Vec<SimilarError>> {
        // This would perform actual similarity matching
        Ok(Vec::new())
    }
} 