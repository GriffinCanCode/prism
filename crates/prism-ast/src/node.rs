//! Core AST node infrastructure with semantic metadata support

use crate::metadata::{AiContext, NodeMetadata};
use prism_common::{span::Span, HasNodeId, HasSpan, NodeId, SourceId};
use std::fmt;

/// A complete AST node with semantic metadata
///
/// This is the fundamental building block of the Prism AST. Every node carries:
/// - Rich semantic metadata for AI comprehension
/// - Source location information for error reporting
/// - Unique identifier for cross-referencing
/// - The actual AST node content
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AstNode<T> {
    /// The actual AST node content
    pub kind: T,
    /// Source location of this node
    pub span: Span,
    /// Unique identifier for this node
    pub id: NodeId,
    /// AI-readable semantic metadata
    pub metadata: NodeMetadata,
}

impl<T> AstNode<T> {
    /// Create a new AST node
    pub fn new(kind: T, span: Span, id: NodeId) -> Self {
        Self {
            kind,
            span,
            id,
            metadata: NodeMetadata::default(),
        }
    }

    /// Create a new AST node with metadata
    pub fn with_metadata(kind: T, span: Span, id: NodeId, metadata: NodeMetadata) -> Self {
        Self {
            kind,
            span,
            id,
            metadata,
        }
    }

    /// Add AI context to this node
    pub fn with_ai_context(mut self, context: AiContext) -> Self {
        self.metadata.ai_context = Some(context);
        self
    }

    /// Add a semantic annotation to this node
    pub fn with_semantic_annotation(mut self, annotation: String) -> Self {
        self.metadata.semantic_annotations.push(annotation);
        self
    }

    /// Add a business rule to this node
    pub fn with_business_rule(mut self, rule: String) -> Self {
        self.metadata.business_rules.push(rule);
        self
    }

    /// Map the node content while preserving metadata
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> AstNode<U> {
        AstNode {
            kind: f(self.kind),
            span: self.span,
            id: self.id,
            metadata: self.metadata,
        }
    }

    /// Map the node content by reference while preserving metadata
    pub fn map_ref<U>(&self, f: impl FnOnce(&T) -> &U) -> AstNode<&U> {
        AstNode {
            kind: f(&self.kind),
            span: self.span,
            id: self.id,
            metadata: self.metadata.clone(),
        }
    }

    /// Get a reference to the node content
    pub fn as_ref(&self) -> &T {
        &self.kind
    }

    /// Get a mutable reference to the node content
    pub fn as_mut(&mut self) -> &mut T {
        &mut self.kind
    }

    /// Check if this node has AI context
    pub fn has_ai_context(&self) -> bool {
        self.metadata.ai_context.is_some()
    }

    /// Get the AI context if present
    pub fn ai_context(&self) -> Option<&AiContext> {
        self.metadata.ai_context.as_ref()
    }

    /// Check if this node has semantic annotations
    pub fn has_semantic_annotations(&self) -> bool {
        !self.metadata.semantic_annotations.is_empty()
    }

    /// Get all semantic annotations
    pub fn semantic_annotations(&self) -> &[String] {
        &self.metadata.semantic_annotations
    }

    /// Check if this node has business rules
    pub fn has_business_rules(&self) -> bool {
        !self.metadata.business_rules.is_empty()
    }

    /// Get all business rules
    pub fn business_rules(&self) -> &[String] {
        &self.metadata.business_rules
    }

    /// Check if this node is marked as AI-generated
    pub fn is_ai_generated(&self) -> bool {
        self.metadata.is_ai_generated
    }

    /// Mark this node as AI-generated
    pub fn mark_ai_generated(mut self) -> Self {
        self.metadata.is_ai_generated = true;
        self
    }

    /// Get the semantic importance score of this node
    pub fn semantic_importance(&self) -> f64 {
        self.metadata.semantic_importance
    }

    /// Set the semantic importance score
    pub fn with_semantic_importance(mut self, importance: f64) -> Self {
        self.metadata.semantic_importance = importance;
        self
    }

    /// Check if this node represents a security-sensitive operation
    pub fn is_security_sensitive(&self) -> bool {
        self.metadata.security_sensitive
    }

    /// Mark this node as security-sensitive
    pub fn mark_security_sensitive(mut self) -> Self {
        self.metadata.security_sensitive = true;
        self
    }

    /// Get the performance characteristics of this node
    pub fn performance_characteristics(&self) -> &[String] {
        &self.metadata.performance_characteristics
    }

    /// Add a performance characteristic
    pub fn with_performance_characteristic(mut self, characteristic: String) -> Self {
        self.metadata.performance_characteristics.push(characteristic);
        self
    }
}

impl<T> HasSpan for AstNode<T> {
    fn span(&self) -> Span {
        self.span
    }
}

impl<T> HasNodeId for AstNode<T> {
    fn node_id(&self) -> NodeId {
        self.id
    }
}

impl<T: fmt::Display> fmt::Display for AstNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// A reference to an AST node within an arena
///
/// This is used for memory-efficient AST storage where nodes are allocated
/// in an arena and referenced by index rather than owning pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AstNodeRef {
    /// Index into the arena
    pub index: u32,
    /// Source ID for validation
    pub source_id: SourceId,
}

impl AstNodeRef {
    /// Create a new AST node reference
    pub fn new(index: u32, source_id: SourceId) -> Self {
        Self { index, source_id }
    }

    /// Get the index
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get the source ID
    pub fn source_id(&self) -> SourceId {
        self.source_id
    }

    /// Get the node ID (alias for index)
    pub fn id(&self) -> u32 {
        self.index
    }
}

impl fmt::Display for AstNodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.source_id, self.index)
    }
}

/// Trait for AST nodes that can be visited
pub trait AstNodeKind {
    /// Get the name of this AST node kind
    fn node_kind_name(&self) -> &'static str;

    /// Get the children of this node as node references
    fn children(&self) -> Vec<AstNodeRef>;

    /// Get the semantic domain this node belongs to
    fn semantic_domain(&self) -> Option<&str> {
        None
    }

    /// Get the AI comprehension hints for this node
    fn ai_comprehension_hints(&self) -> Vec<String> {
        Vec::new()
    }

    /// Check if this node represents a side-effectful operation
    fn is_side_effectful(&self) -> bool {
        false
    }

    /// Get the computational complexity of this node
    fn computational_complexity(&self) -> ComplexityClass {
        ComplexityClass::Constant
    }
}

/// Computational complexity classification for AST nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComplexityClass {
    /// O(1) - Constant time
    Constant,
    /// O(log n) - Logarithmic time
    Logarithmic,
    /// O(n) - Linear time
    Linear,
    /// O(n log n) - Linearithmic time
    Linearithmic,
    /// O(n²) - Quadratic time
    Quadratic,
    /// O(n³) - Cubic time
    Cubic,
    /// O(2ⁿ) - Exponential time
    Exponential,
    /// Unknown or variable complexity
    Unknown,
}

impl fmt::Display for ComplexityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant => write!(f, "O(1)"),
            Self::Logarithmic => write!(f, "O(log n)"),
            Self::Linear => write!(f, "O(n)"),
            Self::Linearithmic => write!(f, "O(n log n)"),
            Self::Quadratic => write!(f, "O(n²)"),
            Self::Cubic => write!(f, "O(n³)"),
            Self::Exponential => write!(f, "O(2ⁿ)"),
            Self::Unknown => write!(f, "O(?)"),
        }
    }
}

/// A lightweight AST node identifier for cross-referencing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeRef {
    /// The node ID
    pub id: NodeId,
    /// The source ID for validation
    pub source_id: SourceId,
}

impl NodeRef {
    /// Create a new node reference
    pub fn new(id: NodeId, source_id: SourceId) -> Self {
        Self { id, source_id }
    }

    /// Create a node reference from an AST node
    pub fn from_node<T>(node: &AstNode<T>) -> Self {
        Self::new(node.id, node.span.source_id)
    }
}

impl fmt::Display for NodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.source_id, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::span::Position;

    #[test]
    fn test_ast_node_creation() {
        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 5, 4),
            SourceId::new(1),
        );
        let node = AstNode::new("test", span, NodeId::new(1));
        
        assert_eq!(node.kind, "test");
        assert_eq!(node.span, span);
        assert_eq!(node.id, NodeId::new(1));
    }

    #[test]
    fn test_ast_node_with_ai_context() {
        let span = Span::dummy();
        let ai_context = AiContext::new()
            .with_purpose("Test purpose")
            .with_domain("Test domain");
        
        let node = AstNode::new("test", span, NodeId::new(1))
            .with_ai_context(ai_context);
        
        assert!(node.has_ai_context());
        assert_eq!(node.ai_context().unwrap().purpose, Some("Test purpose".to_string()));
    }

    #[test]
    fn test_ast_node_semantic_annotations() {
        let span = Span::dummy();
        let node = AstNode::new("test", span, NodeId::new(1))
            .with_semantic_annotation("Important operation".to_string())
            .with_business_rule("Must validate input".to_string());
        
        assert!(node.has_semantic_annotations());
        assert!(node.has_business_rules());
        assert_eq!(node.semantic_annotations().len(), 1);
        assert_eq!(node.business_rules().len(), 1);
    }

    #[test]
    fn test_complexity_class_display() {
        assert_eq!(ComplexityClass::Constant.to_string(), "O(1)");
        assert_eq!(ComplexityClass::Linear.to_string(), "O(n)");
        assert_eq!(ComplexityClass::Quadratic.to_string(), "O(n²)");
    }
} 