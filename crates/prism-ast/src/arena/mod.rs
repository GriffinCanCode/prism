//! AST Arena System - Efficient Memory Management for AST Nodes
//!
//! This module provides a specialized arena-based memory management system for AST nodes,
//! designed to integrate with Prism's semantic type system, capability-based security,
//! and AI-first metadata generation.
//!
//! ## Architecture
//!
//! The arena system follows a modular architecture with clear separation of concerns:
//!
//! - **`core`**: Core arena allocation and node storage
//! - **`semantic`**: Semantic type integration and AI metadata
//! - **`serialization`**: Multi-target serialization (TypeScript, WASM, Native)
//! - **`optimization`**: Performance optimizations and cache management
//! - **`integration`**: Integration with runtime memory management
//!
//! ## Design Principles
//!
//! 1. **Semantic Preservation**: All AST nodes maintain rich semantic metadata
//! 2. **Capability Integration**: Memory operations respect capability boundaries
//! 3. **Multi-Target Support**: Efficient serialization for all compilation targets
//! 4. **AI-First Design**: Comprehensive metadata for AI analysis
//! 5. **Performance Focus**: Cache-friendly layouts and minimal overhead

use crate::{AstNode, AstNodeRef};
use prism_common::{NodeId, SourceId};
use std::marker::PhantomData;
use thiserror::Error;

// Re-export public types from submodules
pub use self::core::{AstArena, ArenaConfig, ArenaStats};
pub use self::semantic::{SemanticArenaNode, SemanticMetadata, AIArenaMetadata};
pub use self::serialization::{SerializationFormat, SerializationTarget, SerializationError};
pub use self::optimization::{CacheStrategy, OptimizationHints, PerformanceMetrics};

// Submodules
mod core;
mod semantic;
mod serialization;
mod optimization;
mod integration;

/// Arena-based node reference with type information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypedNodeRef<T> {
    /// Node reference
    node_ref: AstNodeRef,
    /// Type phantom
    _phantom: PhantomData<T>,
}

impl<T> TypedNodeRef<T> {
    /// Create a new typed node reference
    pub fn new(node_ref: AstNodeRef) -> Self {
        Self {
            node_ref,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying node reference
    pub fn node_ref(&self) -> AstNodeRef {
        self.node_ref
    }
}

/// Errors that can occur in the arena system
#[derive(Debug, Error)]
pub enum ArenaError {
    /// Node not found in arena
    #[error("Node not found: {node_ref:?}")]
    NodeNotFound { node_ref: AstNodeRef },
    
    /// Invalid node type for operation
    #[error("Invalid node type: expected {expected}, got {actual}")]
    InvalidNodeType { expected: String, actual: String },
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializationError),
    
    /// Memory allocation error
    #[error("Memory allocation failed: {reason}")]
    AllocationFailed { reason: String },
    
    /// Capability error
    #[error("Insufficient capability: {required}")]
    InsufficientCapability { required: String },
}

/// Result type for arena operations
pub type ArenaResult<T> = Result<T, ArenaError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, LiteralExpr, LiteralValue};
    use prism_common::span::{Span, Position};

    #[test]
    fn test_typed_node_ref_creation() {
        let source_id = SourceId::new(1);
        let node_ref = AstNodeRef::new(42, source_id);
        let typed_ref = TypedNodeRef::<Expr>::new(node_ref);
        
        assert_eq!(typed_ref.node_ref(), node_ref);
    }

    #[test]
    fn test_arena_error_display() {
        let source_id = SourceId::new(1);
        let node_ref = AstNodeRef::new(42, source_id);
        let error = ArenaError::NodeNotFound { node_ref };
        
        assert!(error.to_string().contains("Node not found"));
    }
} 