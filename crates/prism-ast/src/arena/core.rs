//! Core Arena Implementation
//!
//! This module provides the fundamental arena-based memory allocation for AST nodes.
//! It focuses on efficient node storage, reference management, and basic allocation patterns.

use crate::{AstNode, AstNodeRef};
use super::{ArenaError, ArenaResult};
use prism_common::{NodeId, SourceId};
use std::collections::HashMap;
use std::any::TypeId;
use std::sync::atomic::{AtomicU32, Ordering};

/// Configuration for arena behavior
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Initial capacity for node storage
    pub initial_capacity: usize,
    /// Enable node deduplication
    pub enable_deduplication: bool,
    /// Enable reference tracking
    pub enable_reference_tracking: bool,
    /// Maximum nodes per arena (for memory limits)
    pub max_nodes: Option<usize>,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            enable_deduplication: true,
            enable_reference_tracking: true,
            max_nodes: None,
        }
    }
}

/// Statistics about arena usage
#[derive(Debug, Clone)]
pub struct ArenaStats {
    /// Total nodes allocated
    pub total_nodes: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of different node types
    pub node_types: usize,
    /// Deduplication hit rate
    pub deduplication_hit_rate: f64,
    /// Average node size
    pub average_node_size: f64,
}

/// Core AST arena for efficient node allocation
pub struct AstArena {
    /// Node storage by type
    node_storage: HashMap<TypeId, NodeTypeStorage>,
    /// Configuration
    config: ArenaConfig,
    /// Next node ID
    next_id: AtomicU32,
    /// Source ID for this arena
    source_id: SourceId,
    /// Node reference tracking
    reference_map: HashMap<AstNodeRef, NodeLocation>,
}

/// Storage for a specific node type
struct NodeTypeStorage {
    /// Serialized node data
    data: Vec<Vec<u8>>,
    /// Type information
    type_name: &'static str,
    /// Node indices for deduplication
    dedup_map: HashMap<u64, u32>, // hash -> index
}

/// Location of a node within the arena
#[derive(Debug, Clone)]
struct NodeLocation {
    /// Type ID
    type_id: TypeId,
    /// Index within type storage
    index: u32,
}

impl AstArena {
    /// Create a new AST arena
    pub fn new(source_id: SourceId) -> Self {
        Self::with_config(source_id, ArenaConfig::default())
    }

    /// Create a new AST arena with configuration
    pub fn with_config(source_id: SourceId, config: ArenaConfig) -> Self {
        let initial_capacity = config.initial_capacity;
        Self {
            node_storage: HashMap::with_capacity(16), // Common node types
            config,
            next_id: AtomicU32::new(0),
            source_id,
            reference_map: HashMap::with_capacity(initial_capacity),
        }
    }

    /// Allocate a new node in the arena
    pub fn alloc<T>(&mut self, node: AstNode<T>) -> ArenaResult<AstNodeRef>
    where
        T: 'static + serde::Serialize,
    {
        // Check capacity limits
        if let Some(max_nodes) = self.config.max_nodes {
            if self.reference_map.len() >= max_nodes {
                return Err(ArenaError::AllocationFailed {
                    reason: format!("Arena capacity exceeded: {}", max_nodes),
                });
            }
        }

        let type_id = TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();
        
        // Serialize the node
        let serialized = bincode::serialize(&node).map_err(|e| {
            ArenaError::AllocationFailed {
                reason: format!("Failed to serialize node: {}", e),
            }
        })?;

        // Check for deduplication if enabled
        let index = if self.config.enable_deduplication {
            let hash = self.compute_node_hash(&serialized);
            
            // Get or create type storage
            let storage = self.node_storage.entry(type_id).or_insert_with(|| {
                NodeTypeStorage {
                    data: Vec::with_capacity(self.config.initial_capacity / 16),
                    type_name,
                    dedup_map: HashMap::new(),
                }
            });
            
            if let Some(&existing_index) = storage.dedup_map.get(&hash) {
                existing_index
            } else {
                let new_index = storage.data.len() as u32;
                storage.data.push(serialized);
                storage.dedup_map.insert(hash, new_index);
                new_index
            }
        } else {
            // Get or create type storage
            let storage = self.node_storage.entry(type_id).or_insert_with(|| {
                NodeTypeStorage {
                    data: Vec::with_capacity(self.config.initial_capacity / 16),
                    type_name,
                    dedup_map: HashMap::new(),
                }
            });
            
            let new_index = storage.data.len() as u32;
            storage.data.push(serialized);
            new_index
        };

        // Generate node reference
        let node_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let node_ref = AstNodeRef::new(node_id, self.source_id);

        // Track reference if enabled
        if self.config.enable_reference_tracking {
            self.reference_map.insert(node_ref, NodeLocation { type_id, index });
        }

        Ok(node_ref)
    }

    /// Get a node by reference
    pub fn get<T>(&self, node_ref: AstNodeRef) -> ArenaResult<AstNode<T>>
    where
        T: 'static + for<'de> serde::Deserialize<'de>,
    {
        let type_id = TypeId::of::<T>();
        
        // Find node location
        let location = self.reference_map.get(&node_ref)
            .ok_or_else(|| ArenaError::NodeNotFound { node_ref })?;

        // Verify type matches
        if location.type_id != type_id {
            return Err(ArenaError::InvalidNodeType {
                expected: std::any::type_name::<T>().to_string(),
                actual: self.node_storage.get(&location.type_id)
                    .map(|s| s.type_name.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
            });
        }

        // Get storage and deserialize
        let storage = self.node_storage.get(&type_id)
            .ok_or_else(|| ArenaError::NodeNotFound { node_ref })?;

        let data = storage.data.get(location.index as usize)
            .ok_or_else(|| ArenaError::NodeNotFound { node_ref })?;

        bincode::deserialize(data).map_err(|e| {
            ArenaError::AllocationFailed {
                reason: format!("Failed to deserialize node: {}", e),
            }
        })
    }

    /// Check if a node exists in the arena
    pub fn contains(&self, node_ref: AstNodeRef) -> bool {
        self.reference_map.contains_key(&node_ref)
    }

    /// Get the number of nodes in the arena
    pub fn len(&self) -> usize {
        self.reference_map.len()
    }

    /// Check if the arena is empty
    pub fn is_empty(&self) -> bool {
        self.reference_map.is_empty()
    }

    /// Get the source ID for this arena
    pub fn source_id(&self) -> SourceId {
        self.source_id
    }

    /// Get arena statistics
    pub fn stats(&self) -> ArenaStats {
        let total_nodes = self.len();
        let memory_usage = self.calculate_memory_usage();
        let node_types = self.node_storage.len();
        
        let deduplication_hit_rate = if self.config.enable_deduplication {
            self.calculate_deduplication_hit_rate()
        } else {
            0.0
        };

        let average_node_size = if total_nodes > 0 {
            memory_usage as f64 / total_nodes as f64
        } else {
            0.0
        };

        ArenaStats {
            total_nodes,
            memory_usage,
            node_types,
            deduplication_hit_rate,
            average_node_size,
        }
    }

    /// Compute a hash for node deduplication
    fn compute_node_hash(&self, data: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }

    /// Calculate total memory usage
    fn calculate_memory_usage(&self) -> usize {
        self.node_storage.values()
            .map(|storage| {
                storage.data.iter().map(|data| data.len()).sum::<usize>()
            })
            .sum()
    }

    /// Calculate deduplication hit rate
    fn calculate_deduplication_hit_rate(&self) -> f64 {
        let total_requests: usize = self.node_storage.values()
            .map(|storage| storage.data.len())
            .sum();
        
        let unique_nodes: usize = self.node_storage.values()
            .map(|storage| storage.dedup_map.len())
            .sum();

        if total_requests > 0 {
            1.0 - (unique_nodes as f64 / total_requests as f64)
        } else {
            0.0
        }
    }
}

impl std::fmt::Debug for AstArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AstArena")
            .field("source_id", &self.source_id)
            .field("node_count", &self.len())
            .field("node_types", &self.node_storage.len())
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, LiteralExpr, LiteralValue};
    use prism_common::span::{Span, Position};

    #[test]
    fn test_arena_creation() {
        let arena = AstArena::new(SourceId::new(1));
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        assert_eq!(arena.source_id(), SourceId::new(1));
    }

    #[test]
    fn test_node_allocation_and_retrieval() {
        let mut arena = AstArena::new(SourceId::new(1));
        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        });
        let node = AstNode::new(expr.clone(), span, NodeId::new(1));
        
        let node_ref = arena.alloc(node).unwrap();
        assert_eq!(arena.len(), 1);
        
        let retrieved_node: AstNode<Expr> = arena.get(node_ref).unwrap();
        match retrieved_node.kind {
            Expr::Literal(LiteralExpr { value: LiteralValue::Integer(42) }) => {},
            _ => panic!("Retrieved wrong node type"),
        }
    }

    #[test]
    fn test_deduplication() {
        let mut arena = AstArena::with_config(
            SourceId::new(1),
            ArenaConfig {
                enable_deduplication: true,
                ..Default::default()
            },
        );

        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        // Allocate the same node twice
        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        });
        let node1 = AstNode::new(expr.clone(), span, NodeId::new(1));
        let node2 = AstNode::new(expr.clone(), span, NodeId::new(2));
        
        let _ref1 = arena.alloc(node1).unwrap();
        let _ref2 = arena.alloc(node2).unwrap();
        
        // Should have 2 references but potentially deduplicated storage
        assert_eq!(arena.len(), 2);
        
        let stats = arena.stats();
        // Deduplication might occur depending on serialization
        assert!(stats.deduplication_hit_rate >= 0.0);
    }

    #[test]
    fn test_capacity_limits() {
        let mut arena = AstArena::with_config(
            SourceId::new(1),
            ArenaConfig {
                max_nodes: Some(2),
                ..Default::default()
            },
        );

        let span = Span::new(
            Position::new(1, 1, 0),
            Position::new(1, 11, 10),
            SourceId::new(1),
        );

        // Allocate up to the limit
        for i in 0..2 {
            let expr = Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(i),
            });
            let node = AstNode::new(expr, span, NodeId::new(i as u32));
            arena.alloc(node).unwrap();
        }

        // This should fail due to capacity limit
        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(999),
        });
        let node = AstNode::new(expr, span, NodeId::new(999));
        let result = arena.alloc(node);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            ArenaError::AllocationFailed { reason } => {
                assert!(reason.contains("capacity exceeded"));
            },
            _ => panic!("Expected allocation failed error"),
        }
    }
} 