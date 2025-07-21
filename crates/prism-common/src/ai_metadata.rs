//! Shared AI Metadata Interfaces
//!
//! This module provides common interfaces for AI metadata that can be used
//! across the parser, compiler, and other crates without creating circular dependencies.

use crate::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Core AI metadata interface used across multiple crates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Semantic contexts discovered
    pub semantic_contexts: Vec<SemanticContextEntry>,
    /// Business rules identified
    pub business_rules: Vec<BusinessRuleEntry>,
    /// AI-generated insights
    pub insights: Vec<AIInsight>,
    /// Confidence in the metadata
    pub confidence: f64,
}

/// Semantic context entry for AI understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContextEntry {
    /// Location in source code
    pub location: Span,
    /// Type of semantic context
    pub context_type: SemanticContextType,
    /// Semantic information extracted
    pub semantic_info: String,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// Confidence in this context
    pub confidence: f64,
}

/// Types of semantic contexts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticContextType {
    /// Business logic context
    BusinessLogic,
    /// Data validation context
    DataValidation,
    /// Error handling context
    ErrorHandling,
    /// Performance critical context
    PerformanceCritical,
    /// Security sensitive context
    SecuritySensitive,
    /// User interface context
    UserInterface,
}

/// Business rule entry for compliance and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRuleEntry {
    /// Rule name or identifier
    pub name: String,
    /// Rule description
    pub description: String,
    /// Source location where rule applies
    pub location: Span,
    /// Rule category
    pub category: BusinessRuleCategory,
    /// Enforcement level
    pub enforcement: EnforcementLevel,
}

/// Categories of business rules
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessRuleCategory {
    /// Data validation rule
    Validation,
    /// Business constraint
    Constraint,
    /// Workflow rule
    Workflow,
    /// Compliance requirement
    Compliance,
    /// Security policy
    Security,
}

/// Rule enforcement levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced (compilation error if violated)
    Required,
    /// Should be enforced (warning if violated)
    Recommended,
    /// Optional enforcement (hint if violated)
    Optional,
}

/// AI-generated insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInsight {
    /// Type of insight
    pub insight_type: AIInsightType,
    /// Insight content
    pub content: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Related source location
    pub location: Option<Span>,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Types of AI insights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AIInsightType {
    /// Code pattern recognition
    PatternRecognition,
    /// Business logic interpretation
    BusinessLogic,
    /// Performance optimization opportunity
    PerformanceOptimization,
    /// Security vulnerability detection
    SecurityVulnerability,
    /// Architectural improvement suggestion
    ArchitecturalImprovement,
    /// Code quality assessment
    CodeQuality,
}

/// AI metadata collector interface
pub trait AIMetadataCollector {
    /// Add a semantic context entry
    fn add_semantic_context(&mut self, entry: SemanticContextEntry);
    
    /// Add a business rule entry
    fn add_business_rule(&mut self, rule: BusinessRuleEntry);
    
    /// Add an AI insight
    fn add_insight(&mut self, insight: AIInsight);
    
    /// Get all collected metadata
    fn get_metadata(&self) -> &AIMetadata;
    
    /// Check if collection is enabled
    fn is_enabled(&self) -> bool;
}

impl Default for AIMetadata {
    fn default() -> Self {
        Self {
            semantic_contexts: Vec::new(),
            business_rules: Vec::new(),
            insights: Vec::new(),
            confidence: 1.0,
        }
    }
}

impl AIMetadata {
    /// Create new empty AI metadata
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add semantic context
    pub fn add_semantic_context(&mut self, entry: SemanticContextEntry) {
        self.semantic_contexts.push(entry);
    }
    
    /// Add business rule
    pub fn add_business_rule(&mut self, rule: BusinessRuleEntry) {
        self.business_rules.push(rule);
    }
    
    /// Add AI insight
    pub fn add_insight(&mut self, insight: AIInsight) {
        self.insights.push(insight);
    }
    
    /// Calculate overall confidence based on entries
    pub fn calculate_confidence(&mut self) {
        if self.semantic_contexts.is_empty() && self.business_rules.is_empty() && self.insights.is_empty() {
            self.confidence = 0.0;
            return;
        }
        
        let mut total_confidence = 0.0;
        let mut count = 0;
        
        for context in &self.semantic_contexts {
            total_confidence += context.confidence;
            count += 1;
        }
        
        for insight in &self.insights {
            total_confidence += insight.confidence;
            count += 1;
        }
        
        self.confidence = if count > 0 { total_confidence / count as f64 } else { 0.0 };
    }
} 