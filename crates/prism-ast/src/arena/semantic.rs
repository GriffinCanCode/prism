//! Semantic Integration for Arena System
//!
//! This module integrates the arena system with Prism's semantic type system,
//! AI metadata generation, and business context preservation.

use crate::{AstNode, AstNodeRef, NodeMetadata, AiContext};
use super::{ArenaError, ArenaResult};
use prism_common::{NodeId, SourceId, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};

/// Enhanced AST node with semantic arena metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticArenaNode<T> {
    /// The underlying AST node
    pub node: AstNode<T>,
    /// Semantic metadata for AI analysis
    pub semantic_metadata: SemanticMetadata,
    /// Arena-specific metadata
    pub arena_metadata: ArenaNodeMetadata,
}

/// Semantic metadata attached to arena nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMetadata {
    /// Business domain this node belongs to
    pub business_domain: Option<String>,
    /// Semantic importance score (0.0 to 1.0)
    pub importance_score: f64,
    /// Related concepts for AI understanding
    pub related_concepts: Vec<String>,
    /// Semantic relationships to other nodes
    pub relationships: Vec<SemanticRelationship>,
    /// Performance characteristics
    pub performance_hints: Vec<PerformanceHint>,
    /// Security implications
    pub security_implications: Vec<SecurityImplication>,
}

/// Relationship between semantic nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Target node reference
    pub target_node: AstNodeRef,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Description of the relationship
    pub description: String,
}

/// Types of semantic relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Dependency relationship
    DependsOn,
    /// Composition relationship
    Contains,
    /// Implementation relationship
    Implements,
    /// Usage relationship
    Uses,
    /// Semantic equivalence
    EquivalentTo,
    /// Conceptual similarity
    SimilarTo,
    /// Business rule relationship
    EnforcesRule,
}

/// Performance hints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHint {
    /// Type of performance hint
    pub hint_type: PerformanceHintType,
    /// Estimated impact (0.0 to 1.0)
    pub impact: f64,
    /// Description of the hint
    pub description: String,
}

/// Types of performance hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceHintType {
    /// Frequently accessed node
    HotPath,
    /// Rarely accessed node
    ColdPath,
    /// Cache-friendly access pattern
    CacheFriendly,
    /// Memory-intensive operation
    MemoryIntensive,
    /// CPU-intensive operation
    CpuIntensive,
    /// I/O intensive operation
    IoIntensive,
}

/// Security implications of a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplication {
    /// Type of security implication
    pub implication_type: SecurityImplicationType,
    /// Severity level (0.0 to 1.0)
    pub severity: f64,
    /// Description of the implication
    pub description: String,
}

/// Types of security implications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityImplicationType {
    /// Requires capability checking
    RequiresCapability,
    /// Handles sensitive data
    SensitiveData,
    /// Performs authentication
    Authentication,
    /// Performs authorization
    Authorization,
    /// Involves cryptographic operations
    Cryptographic,
    /// Network communication
    NetworkCommunication,
    /// File system access
    FileSystemAccess,
}

/// Arena-specific metadata for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaNodeMetadata {
    /// When the node was allocated
    pub allocated_at: SystemTime,
    /// Number of times the node has been accessed
    pub access_count: u64,
    /// Last access time
    pub last_accessed: SystemTime,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Allocation generation (for GC integration)
    pub generation: u32,
}

/// AI-readable metadata for the entire arena
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIArenaMetadata {
    /// Total nodes in arena
    pub total_nodes: usize,
    /// Semantic domains represented
    pub semantic_domains: Vec<String>,
    /// Business concepts present
    pub business_concepts: Vec<String>,
    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Security analysis
    pub security_analysis: SecurityAnalysis,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Complexity metrics for the arena
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity estimate
    pub cyclomatic_complexity: f64,
    /// Cognitive complexity estimate  
    pub cognitive_complexity: f64,
    /// Nesting depth
    pub max_nesting_depth: u32,
    /// Number of decision points
    pub decision_points: u32,
}

/// Performance profile of the arena
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Hot path nodes (frequently accessed)
    pub hot_path_nodes: Vec<AstNodeRef>,
    /// Cold path nodes (rarely accessed)
    pub cold_path_nodes: Vec<AstNodeRef>,
    /// Memory usage distribution
    pub memory_distribution: MemoryDistribution,
    /// Access patterns
    pub access_patterns: Vec<AccessPattern>,
}

/// Memory usage distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDistribution {
    /// Small nodes (< 100 bytes)
    pub small_nodes: usize,
    /// Medium nodes (100-1000 bytes)
    pub medium_nodes: usize,
    /// Large nodes (> 1000 bytes)
    pub large_nodes: usize,
    /// Total memory usage
    pub total_memory: usize,
}

/// Access pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    /// Pattern type
    pub pattern_type: AccessPatternType,
    /// Nodes following this pattern
    pub nodes: Vec<AstNodeRef>,
    /// Frequency of this pattern
    pub frequency: f64,
}

/// Types of access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessPatternType {
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Temporal locality
    TemporalLocality,
    /// Spatial locality
    SpatialLocality,
    /// Write-heavy
    WriteHeavy,
    /// Read-heavy
    ReadHeavy,
}

/// Security analysis of the arena
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    /// Security-sensitive nodes
    pub sensitive_nodes: Vec<AstNodeRef>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Security implications summary
    pub security_implications: Vec<SecurityImplication>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Affected nodes
    pub affected_nodes: Vec<AstNodeRef>,
    /// Estimated benefit (0.0 to 1.0)
    pub estimated_benefit: f64,
    /// Description of the opportunity
    pub description: String,
}

/// Types of optimization opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Memory layout optimization
    MemoryLayout,
    /// Cache optimization
    CacheOptimization,
    /// Deduplication opportunity
    Deduplication,
    /// Compression opportunity
    Compression,
    /// Access pattern optimization
    AccessPattern,
    /// Garbage collection optimization
    GarbageCollection,
}

impl Default for SemanticMetadata {
    fn default() -> Self {
        Self {
            business_domain: None,
            importance_score: 0.5,
            related_concepts: Vec::new(),
            relationships: Vec::new(),
            performance_hints: Vec::new(),
            security_implications: Vec::new(),
        }
    }
}

impl ArenaNodeMetadata {
    /// Create new arena metadata for a node
    pub fn new(memory_usage: usize) -> Self {
        let now = SystemTime::now();
        Self {
            allocated_at: now,
            access_count: 0,
            last_accessed: now,
            memory_usage,
            generation: 0,
        }
    }

    /// Record an access to this node
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now();
    }

    /// Get the age of this node
    pub fn age(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.allocated_at)
            .unwrap_or(Duration::ZERO)
    }

    /// Get the time since last access
    pub fn time_since_last_access(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.last_accessed)
            .unwrap_or(Duration::ZERO)
    }
}

impl<T> SemanticArenaNode<T> {
    /// Create a new semantic arena node
    pub fn new(
        node: AstNode<T>,
        memory_usage: usize,
        semantic_metadata: Option<SemanticMetadata>,
    ) -> Self {
        Self {
            node,
            semantic_metadata: semantic_metadata.unwrap_or_default(),
            arena_metadata: ArenaNodeMetadata::new(memory_usage),
        }
    }

    /// Add a semantic relationship to another node
    pub fn add_relationship(&mut self, relationship: SemanticRelationship) {
        self.semantic_metadata.relationships.push(relationship);
    }

    /// Add a performance hint
    pub fn add_performance_hint(&mut self, hint: PerformanceHint) {
        self.semantic_metadata.performance_hints.push(hint);
    }

    /// Add a security implication
    pub fn add_security_implication(&mut self, implication: SecurityImplication) {
        self.semantic_metadata.security_implications.push(implication);
    }

    /// Record access to this node
    pub fn record_access(&mut self) {
        self.arena_metadata.record_access();
    }

    /// Check if this node is frequently accessed
    pub fn is_hot_path(&self) -> bool {
        self.arena_metadata.access_count > 10 && 
        self.arena_metadata.time_since_last_access() < Duration::from_secs(60)
    }

    /// Check if this node is rarely accessed
    pub fn is_cold_path(&self) -> bool {
        self.arena_metadata.access_count < 3 || 
        self.arena_metadata.time_since_last_access() > Duration::from_secs(3600)
    }

    /// Get AI-readable metadata for this node
    pub fn ai_metadata(&self) -> NodeAIMetadata {
        NodeAIMetadata {
            node_id: self.node.id,
            business_domain: self.semantic_metadata.business_domain.clone(),
            importance_score: self.semantic_metadata.importance_score,
            access_frequency: self.calculate_access_frequency(),
            memory_usage: self.arena_metadata.memory_usage,
            age: self.arena_metadata.age(),
            relationships: self.semantic_metadata.relationships.clone(),
            performance_hints: self.semantic_metadata.performance_hints.clone(),
            security_implications: self.semantic_metadata.security_implications.clone(),
        }
    }

    /// Calculate access frequency
    pub fn calculate_access_frequency(&self) -> f64 {
        let age_seconds = self.arena_metadata.age().as_secs_f64();
        if age_seconds > 0.0 {
            self.arena_metadata.access_count as f64 / age_seconds
        } else {
            0.0
        }
    }
}

/// AI-readable metadata for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAIMetadata {
    /// Node identifier
    pub node_id: NodeId,
    /// Business domain
    pub business_domain: Option<String>,
    /// Semantic importance
    pub importance_score: f64,
    /// Access frequency (accesses per second)
    pub access_frequency: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Age of the node
    pub age: Duration,
    /// Semantic relationships
    pub relationships: Vec<SemanticRelationship>,
    /// Performance hints
    pub performance_hints: Vec<PerformanceHint>,
    /// Security implications
    pub security_implications: Vec<SecurityImplication>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, LiteralExpr, LiteralValue};
    use prism_common::span::{Span, Position};

    #[test]
    fn test_semantic_metadata_creation() {
        let metadata = SemanticMetadata::default();
        assert_eq!(metadata.importance_score, 0.5);
        assert!(metadata.relationships.is_empty());
        assert!(metadata.performance_hints.is_empty());
    }

    #[test]
    fn test_arena_node_metadata() {
        let mut metadata = ArenaNodeMetadata::new(100);
        assert_eq!(metadata.access_count, 0);
        assert_eq!(metadata.memory_usage, 100);
        
        metadata.record_access();
        assert_eq!(metadata.access_count, 1);
    }

    #[test]
    fn test_semantic_arena_node() {
        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        });
        let node = AstNode::new(expr, span, NodeId::new(1));
        
        let mut semantic_node = SemanticArenaNode::new(node, 100, None);
        assert!(!semantic_node.is_hot_path());
        assert!(semantic_node.is_cold_path());
        
        // Simulate frequent access
        for _ in 0..15 {
            semantic_node.record_access();
        }
        
        assert!(semantic_node.is_hot_path());
        assert!(!semantic_node.is_cold_path());
    }

    #[test]
    fn test_semantic_relationship() {
        let source_id = SourceId::new(1);
        let target_ref = AstNodeRef::new(42, source_id);
        
        let relationship = SemanticRelationship {
            relationship_type: RelationshipType::DependsOn,
            target_node: target_ref,
            strength: 0.8,
            description: "Test dependency".to_string(),
        };
        
        assert!(matches!(relationship.relationship_type, RelationshipType::DependsOn));
        assert_eq!(relationship.strength, 0.8);
    }
} 