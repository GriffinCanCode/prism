//! AI Context Extraction
//!
//! This module embodies the single concept of "AI Context Extraction".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: extracting AI-comprehensible context and metadata.

use crate::{SemanticResult, SemanticConfig};
use crate::analyzer::AnalysisResult;
use crate::database::SemanticDatabase;
use prism_ast::Program;
use prism_common::span::Span;
use serde::{Serialize, Deserialize};

/// AI context extractor
#[derive(Debug)]
pub struct AIContextExtractor {
    /// Configuration
    config: ContextConfig,
}

/// Configuration for context extraction
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
}

/// AI metadata for semantic elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Business context
    pub business_context: Option<String>,
    /// Domain concepts
    pub domain_concepts: Vec<String>,
    /// AI comprehension hints
    pub comprehension_hints: Vec<String>,
}

/// Semantic context for AI understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    /// Purpose of this context
    pub purpose: Option<String>,
    /// Conceptual role
    pub conceptual_role: Option<String>,
    /// AI hints
    pub ai_hints: Vec<String>,
}

impl SemanticContext {
    /// Create a new semantic context
    pub fn new() -> Self {
        Self {
            purpose: None,
            conceptual_role: None,
            ai_hints: Vec::new(),
        }
    }

    /// Add an AI hint
    pub fn add_ai_hint(&mut self, hint: &str) {
        self.ai_hints.push(hint.to_string());
    }
}

impl Default for SemanticContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AIContextExtractor {
    /// Create a new AI context extractor
    pub fn new(config: &SemanticConfig) -> SemanticResult<Self> {
        let context_config = ContextConfig {
            enable_ai_metadata: config.enable_ai_metadata,
        };

        Ok(Self {
            config: context_config,
        })
    }

    /// Extract context from a program
    pub fn extract_context(&mut self, _program: &Program, _analysis: &AnalysisResult) -> SemanticResult<AIMetadata> {
        // Context extraction implementation would go here
        Ok(AIMetadata {
            business_context: Some("Program analysis context".to_string()),
            domain_concepts: vec!["Programming".to_string(), "Semantics".to_string()],
            comprehension_hints: vec!["This program has been analyzed for semantic content".to_string()],
        })
    }

    /// Export AI context for external tools
    pub fn export_ai_context(&self, _location: Span, _database: &SemanticDatabase) -> SemanticResult<AIMetadata> {
        // AI context export would go here
        Ok(AIMetadata {
            business_context: Some("Exported context".to_string()),
            domain_concepts: Vec::new(),
            comprehension_hints: Vec::new(),
        })
    }
} 