//! Scope Metadata and AI Context
//!
//! This module defines metadata structures for scopes that enable
//! AI comprehension and external tool integration, following PLT-004
//! specifications for AI-first design.
//!
//! **Conceptual Responsibility**: Scope metadata and AI context
//! **What it does**: Define metadata structures, AI context, documentation integration
//! **What it doesn't do**: Scope hierarchy management, symbol resolution, visibility control

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Comprehensive metadata for scopes enabling AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeMetadata {
    /// AI-readable scope context
    pub ai_context: Option<AIScopeContext>,
    
    /// Business responsibility of this scope
    pub responsibility: Option<String>,
    
    /// Documentation for this scope (PSG-003 integration)
    pub documentation: Option<ScopeDocumentation>,
    
    /// Performance characteristics and notes
    pub performance_notes: Vec<String>,
    
    /// Security implications and considerations
    pub security_notes: Vec<String>,
    
    /// Business domain this scope operates in
    pub business_domain: Option<String>,
    
    /// Conceptual cohesion metrics (PLD-002 integration)
    pub cohesion_metrics: Option<CohesionMetrics>,
    
    /// Creation and modification timestamps
    pub timestamps: ScopeTimestamps,
    
    /// Custom metadata fields for extensibility
    pub custom_fields: HashMap<String, String>,
}

/// AI-comprehensible context for scopes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIScopeContext {
    /// High-level purpose of this scope
    pub purpose: String,
    
    /// Business domain this scope operates in
    pub business_domain: Option<String>,
    
    /// Architectural pattern this scope implements
    pub architectural_pattern: Option<String>,
    
    /// Key concepts and entities handled in this scope
    pub key_concepts: Vec<String>,
    
    /// Relationships to other scopes or modules
    pub relationships: Vec<ScopeRelationship>,
    
    /// Design decisions and rationale
    pub design_decisions: Vec<DesignDecision>,
    
    /// Usage patterns and examples
    pub usage_patterns: Vec<String>,
    
    /// Common operations performed in this scope
    pub common_operations: Vec<String>,
    
    /// Data flow patterns within this scope
    pub data_flow_patterns: Vec<DataFlowPattern>,
    
    /// Error handling strategy
    pub error_handling_strategy: Option<String>,
    
    /// Performance characteristics
    pub performance_characteristics: Option<PerformanceCharacteristics>,
}

/// Documentation structure for scopes (PSG-003 integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeDocumentation {
    /// Brief description of the scope's purpose
    pub summary: String,
    
    /// Detailed description
    pub description: Option<String>,
    
    /// Usage examples
    pub examples: Vec<String>,
    
    /// Notes and additional information
    pub notes: Vec<String>,
    
    /// See-also references
    pub see_also: Vec<String>,
    
    /// Author information
    pub author: Option<String>,
    
    /// Version information
    pub version: Option<String>,
    
    /// Documentation status
    pub status: DocumentationStatus,
    
    /// Last updated timestamp
    pub last_updated: Option<SystemTime>,
}

/// Relationship between scopes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeRelationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    
    /// Target scope or module name
    pub target: String,
    
    /// Description of the relationship
    pub description: String,
    
    /// Strength of the relationship (0.0 to 1.0)
    pub strength: f64,
}

/// Types of relationships between scopes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Contains or owns
    Contains,
    
    /// Is contained by or owned by
    ContainedBy,
    
    /// Uses or depends on
    Uses,
    
    /// Is used by or depended on by
    UsedBy,
    
    /// Collaborates with
    Collaborates,
    
    /// Extends or inherits from
    Extends,
    
    /// Is extended by or inherited by
    ExtendedBy,
    
    /// Implements
    Implements,
    
    /// Is implemented by
    ImplementedBy,
    
    /// Custom relationship
    Custom(String),
}

/// Design decision documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignDecision {
    /// Title of the decision
    pub title: String,
    
    /// Context that led to this decision
    pub context: String,
    
    /// The decision made
    pub decision: String,
    
    /// Rationale behind the decision
    pub rationale: String,
    
    /// Alternatives considered
    pub alternatives: Vec<String>,
    
    /// Consequences of this decision
    pub consequences: Vec<String>,
    
    /// Date the decision was made
    pub date: Option<SystemTime>,
}

/// Data flow pattern within a scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowPattern {
    /// Name of the pattern
    pub pattern_name: String,
    
    /// Description of the data flow
    pub description: String,
    
    /// Input data types or sources
    pub inputs: Vec<String>,
    
    /// Output data types or destinations
    pub outputs: Vec<String>,
    
    /// Transformations applied to the data
    pub transformations: Vec<String>,
    
    /// Error conditions that can occur
    pub error_conditions: Vec<String>,
}

/// Performance characteristics of a scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Expected time complexity
    pub time_complexity: Option<String>,
    
    /// Expected space complexity
    pub space_complexity: Option<String>,
    
    /// Performance bottlenecks
    pub bottlenecks: Vec<String>,
    
    /// Optimization opportunities
    pub optimizations: Vec<String>,
    
    /// Performance requirements
    pub requirements: Vec<String>,
    
    /// Benchmark results or estimates
    pub benchmarks: Vec<BenchmarkResult>,
}

/// Benchmark result for performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    
    /// Measured value
    pub value: f64,
    
    /// Unit of measurement
    pub unit: String,
    
    /// Context or conditions of the benchmark
    pub context: String,
    
    /// Timestamp when benchmark was run
    pub timestamp: SystemTime,
}

/// Conceptual cohesion metrics (PLD-002 integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionMetrics {
    /// Overall cohesion score (0.0 to 1.0)
    pub overall_score: f64,
    
    /// Type cohesion (how related are the types used)
    pub type_cohesion: f64,
    
    /// Data flow cohesion (how well data flows through the scope)
    pub data_flow_cohesion: f64,
    
    /// Semantic cohesion (how related are the concepts)
    pub semantic_cohesion: f64,
    
    /// Dependency cohesion (how focused are external dependencies)
    pub dependency_cohesion: f64,
    
    /// Detailed analysis of cohesion factors
    pub analysis: CohesionAnalysis,
    
    /// Recommendations for improving cohesion
    pub recommendations: Vec<String>,
}

/// Detailed cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysis {
    /// Factors that contribute positively to cohesion
    pub positive_factors: Vec<String>,
    
    /// Factors that detract from cohesion
    pub negative_factors: Vec<String>,
    
    /// Suggestions for improvement
    pub improvement_suggestions: Vec<String>,
    
    /// Related scopes with similar patterns
    pub similar_scopes: Vec<String>,
}

/// Documentation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentationStatus {
    /// Documentation is complete and up-to-date
    Complete,
    
    /// Documentation is partial or incomplete
    Incomplete,
    
    /// Documentation is outdated
    Outdated,
    
    /// Documentation is missing
    Missing,
    
    /// Documentation is under review
    UnderReview,
}

/// Timestamps for scope lifecycle tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeTimestamps {
    /// When the scope was created
    pub created: SystemTime,
    
    /// When the scope was last modified
    pub last_modified: SystemTime,
    
    /// When the scope was last analyzed
    pub last_analyzed: Option<SystemTime>,
    
    /// When the scope metadata was last updated
    pub metadata_updated: SystemTime,
}

impl Default for ScopeMetadata {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            ai_context: None,
            responsibility: None,
            documentation: None,
            performance_notes: Vec::new(),
            security_notes: Vec::new(),
            business_domain: None,
            cohesion_metrics: None,
            timestamps: ScopeTimestamps {
                created: now,
                last_modified: now,
                last_analyzed: None,
                metadata_updated: now,
            },
            custom_fields: HashMap::new(),
        }
    }
}

impl ScopeMetadata {
    /// Create new metadata with basic information
    pub fn new(responsibility: Option<String>) -> Self {
        let mut metadata = Self::default();
        metadata.responsibility = responsibility;
        metadata
    }
    
    /// Add AI context to this metadata
    pub fn with_ai_context(mut self, ai_context: AIScopeContext) -> Self {
        self.ai_context = Some(ai_context);
        self
    }
    
    /// Add documentation to this metadata
    pub fn with_documentation(mut self, documentation: ScopeDocumentation) -> Self {
        self.documentation = Some(documentation);
        self
    }
    
    /// Add a performance note
    pub fn add_performance_note(&mut self, note: String) {
        self.performance_notes.push(note);
        self.touch_modified();
    }
    
    /// Add a security note
    pub fn add_security_note(&mut self, note: String) {
        self.security_notes.push(note);
        self.touch_modified();
    }
    
    /// Set cohesion metrics
    pub fn set_cohesion_metrics(&mut self, metrics: CohesionMetrics) {
        self.cohesion_metrics = Some(metrics);
        self.touch_modified();
    }
    
    /// Update the last modified timestamp
    pub fn touch_modified(&mut self) {
        self.timestamps.last_modified = SystemTime::now();
        self.timestamps.metadata_updated = SystemTime::now();
    }
    
    /// Check if documentation is complete
    pub fn has_complete_documentation(&self) -> bool {
        self.documentation
            .as_ref()
            .map(|doc| doc.status == DocumentationStatus::Complete)
            .unwrap_or(false)
    }
    
    /// Get a summary of this scope for AI consumption
    pub fn ai_summary(&self) -> String {
        let mut summary = Vec::new();
        
        if let Some(responsibility) = &self.responsibility {
            summary.push(format!("Responsibility: {}", responsibility));
        }
        
        if let Some(domain) = &self.business_domain {
            summary.push(format!("Domain: {}", domain));
        }
        
        if let Some(ai_context) = &self.ai_context {
            summary.push(format!("Purpose: {}", ai_context.purpose));
        }
        
        if let Some(cohesion) = &self.cohesion_metrics {
            summary.push(format!("Cohesion: {:.2}", cohesion.overall_score));
        }
        
        summary.join("; ")
    }
}

impl AIScopeContext {
    /// Create a new AI context with basic information
    pub fn new(purpose: String) -> Self {
        Self {
            purpose,
            business_domain: None,
            architectural_pattern: None,
            key_concepts: Vec::new(),
            relationships: Vec::new(),
            design_decisions: Vec::new(),
            usage_patterns: Vec::new(),
            common_operations: Vec::new(),
            data_flow_patterns: Vec::new(),
            error_handling_strategy: None,
            performance_characteristics: None,
        }
    }
    
    /// Add a key concept to this context
    pub fn add_key_concept(&mut self, concept: String) {
        if !self.key_concepts.contains(&concept) {
            self.key_concepts.push(concept);
        }
    }
    
    /// Add a relationship to this context
    pub fn add_relationship(&mut self, relationship: ScopeRelationship) {
        self.relationships.push(relationship);
    }
    
    /// Add a design decision to this context
    pub fn add_design_decision(&mut self, decision: DesignDecision) {
        self.design_decisions.push(decision);
    }
}

impl ScopeDocumentation {
    /// Create new documentation with a summary
    pub fn new(summary: String) -> Self {
        Self {
            summary,
            description: None,
            examples: Vec::new(),
            notes: Vec::new(),
            see_also: Vec::new(),
            author: None,
            version: None,
            status: DocumentationStatus::Incomplete,
            last_updated: Some(SystemTime::now()),
        }
    }
    
    /// Mark documentation as complete
    pub fn mark_complete(mut self) -> Self {
        self.status = DocumentationStatus::Complete;
        self.last_updated = Some(SystemTime::now());
        self
    }
    
    /// Add an example to the documentation
    pub fn add_example(&mut self, example: String) {
        self.examples.push(example);
        self.last_updated = Some(SystemTime::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scope_metadata_creation() {
        let metadata = ScopeMetadata::new(Some("Handle user authentication".to_string()));
        
        assert_eq!(metadata.responsibility, Some("Handle user authentication".to_string()));
        assert!(metadata.performance_notes.is_empty());
        assert!(metadata.security_notes.is_empty());
        assert!(!metadata.has_complete_documentation());
    }
    
    #[test]
    fn test_ai_context() {
        let mut context = AIScopeContext::new("Process user requests".to_string());
        
        context.add_key_concept("User".to_string());
        context.add_key_concept("Request".to_string());
        
        assert_eq!(context.purpose, "Process user requests");
        assert_eq!(context.key_concepts.len(), 2);
        assert!(context.key_concepts.contains(&"User".to_string()));
        assert!(context.key_concepts.contains(&"Request".to_string()));
    }
    
    #[test]
    fn test_scope_documentation() {
        let mut doc = ScopeDocumentation::new("Handles user authentication".to_string());
        
        doc.add_example("let user = authenticate(credentials);".to_string());
        
        assert_eq!(doc.summary, "Handles user authentication");
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.status, DocumentationStatus::Incomplete);
        
        let complete_doc = doc.mark_complete();
        assert_eq!(complete_doc.status, DocumentationStatus::Complete);
    }
    
    #[test]
    fn test_cohesion_metrics() {
        let metrics = CohesionMetrics {
            overall_score: 0.85,
            type_cohesion: 0.9,
            data_flow_cohesion: 0.8,
            semantic_cohesion: 0.9,
            dependency_cohesion: 0.8,
            analysis: CohesionAnalysis {
                positive_factors: vec!["Related types".to_string()],
                negative_factors: vec!["Mixed responsibilities".to_string()],
                improvement_suggestions: vec!["Split into smaller scopes".to_string()],
                similar_scopes: vec!["UserService".to_string()],
            },
            recommendations: vec!["Consider refactoring".to_string()],
        };
        
        assert_eq!(metrics.overall_score, 0.85);
        assert_eq!(metrics.analysis.positive_factors.len(), 1);
        assert_eq!(metrics.recommendations.len(), 1);
    }
    
    #[test]
    fn test_metadata_updates() {
        let mut metadata = ScopeMetadata::default();
        let initial_modified = metadata.timestamps.last_modified;
        
        // Small delay to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        metadata.add_performance_note("Fast execution".to_string());
        
        assert_eq!(metadata.performance_notes.len(), 1);
        assert!(metadata.timestamps.last_modified > initial_modified);
    }
} 