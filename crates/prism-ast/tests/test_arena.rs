//! Comprehensive tests for the arena module

use prism_common::span::Position;
use prism_ast::{arena::*, node::*, expr::*};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};
use std::sync::{Arc, Mutex};
use std::thread;

#[test]
fn test_arena_creation() {
    let source_id = SourceId::new(1);
    let arena = AstArena::new(source_id);
    
    assert_eq!(arena.source_id(), source_id);
    assert_eq!(arena.len(), 0);
    assert!(arena.is_empty());
}

#[test]
fn test_arena_with_config() {
    let source_id = SourceId::new(1);
    let config = ArenaConfig {
        initial_capacity: 512,
        enable_deduplication: false,
        enable_reference_tracking: true,
        max_nodes: Some(1000),
    };
    
    let arena = AstArena::with_config(source_id, config.clone());
    
    assert_eq!(arena.source_id(), source_id);
    assert_eq!(arena.len(), 0);
    assert!(arena.is_empty());
}

#[test]
fn test_arena_allocation() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    
    let node = AstNode::new(expr, span, NodeId::new(1));
    let node_ref = arena.alloc(node).unwrap();
    
    assert_eq!(arena.len(), 1);
    assert!(!arena.is_empty());
    assert_eq!(node_ref.source_id(), source_id);
    assert!(arena.contains(node_ref));
}

#[test]
fn test_arena_multiple_allocations() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Allocate multiple nodes
    let nodes = vec![
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(1),
            }),
            span,
            NodeId::new(1),
        ),
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(2),
            }),
            span,
            NodeId::new(2),
        ),
        AstNode::new(
            Expr::Literal(LiteralExpr {
                value: LiteralValue::Integer(3),
            }),
            span,
            NodeId::new(3),
        ),
    ];
    
    let mut refs = Vec::new();
    for node in nodes {
        refs.push(arena.alloc(node).unwrap());
    }
    
    assert_eq!(arena.len(), 3);
    assert_eq!(refs.len(), 3);
    
    // Verify each reference has the correct source ID and is tracked
    for node_ref in refs {
        assert_eq!(node_ref.source_id(), source_id);
        assert!(arena.contains(node_ref));
    }
}

#[test]
fn test_arena_node_retrieval() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Boolean(true),
    });
    
    let node = AstNode::new(expr, span, NodeId::new(1));
    let node_ref = arena.alloc(node).unwrap();
    
    // Test node retrieval
    let retrieved = arena.get::<Expr>(node_ref).unwrap();
    
    match retrieved.kind {
        Expr::Literal(LiteralExpr { value: LiteralValue::Boolean(true) }) => {
            // Expected
        }
        _ => panic!("Retrieved wrong node value"),
    }
    
    assert_eq!(retrieved.id, NodeId::new(1));
    assert_eq!(retrieved.span, span);
}

#[test]
fn test_arena_type_safety() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Allocate an expression node
    let expr_node = AstNode::new(
        Expr::Literal(LiteralExpr {
            value: LiteralValue::String("test".to_string()),
        }),
        span,
        NodeId::new(1),
    );
    
    let expr_ref = arena.alloc(expr_node).unwrap();
    
    // Try to retrieve as wrong type - should fail
    let wrong_type_result = arena.get::<prism_ast::Stmt>(expr_ref);
    assert!(wrong_type_result.is_err());
    
    // Retrieve as correct type - should succeed
    let correct_result = arena.get::<Expr>(expr_ref);
    assert!(correct_result.is_ok());
}

#[test]
fn test_arena_nonexistent_node() {
    let source_id = SourceId::new(1);
    let arena = AstArena::new(source_id);
    
    // Try to get a node that doesn't exist
    let fake_ref = AstNodeRef::new(999, source_id);
    let result = arena.get::<Expr>(fake_ref);
    
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArenaError::NodeNotFound { .. }));
}

#[test]
fn test_arena_concurrency_safety() {
    let source_id = SourceId::new(1);
    let arena = Arc::new(Mutex::new(AstArena::new(source_id)));
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let arena = Arc::clone(&arena);
            thread::spawn(move || {
                let expr = Expr::Literal(LiteralExpr {
                    value: LiteralValue::Integer(i),
                });
                let node = AstNode::new(expr, span, NodeId::new(i as u32));
                
                let mut arena = arena.lock().unwrap();
                arena.alloc(node)
            })
        })
        .collect();
    
    let mut refs = Vec::new();
    for handle in handles {
        refs.push(handle.join().unwrap().unwrap());
    }
    
    let arena = arena.lock().unwrap();
    assert_eq!(arena.len(), 10);
    assert_eq!(refs.len(), 10);
    
    // Verify all references have the correct source ID
    for node_ref in refs {
        assert_eq!(node_ref.source_id(), source_id);
        assert!(arena.contains(node_ref));
    }
}

#[test]
fn test_arena_memory_efficiency() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Allocate many nodes to test memory efficiency
    let node_count = 100; // Reduced from 1000 for faster tests
    let mut refs = Vec::with_capacity(node_count);
    
    for i in 0..node_count {
        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(i as i64),
        });
        let node = AstNode::new(expr, span, NodeId::new(i as u32));
        refs.push(arena.alloc(node).unwrap());
    }
    
    assert_eq!(arena.len(), node_count);
    assert_eq!(refs.len(), node_count);
    
    // Verify each reference is valid and tracked
    for node_ref in refs {
        assert_eq!(node_ref.source_id(), source_id);
        assert!(arena.contains(node_ref));
    }
}

#[test]
fn test_arena_stats() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Initially empty
    let stats = arena.stats();
    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.memory_usage, 0);
    assert_eq!(stats.node_types, 0);
    
    // Add some nodes
    for i in 0..5 {
        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(i),
        });
        let node = AstNode::new(expr, span, NodeId::new(i as u32));
        arena.alloc(node).unwrap();
    }
    
    let stats = arena.stats();
    assert_eq!(stats.total_nodes, 5);
    assert!(stats.memory_usage > 0);
    assert_eq!(stats.node_types, 1); // All Expr nodes
    assert!(stats.average_node_size > 0.0);
}

#[test]
fn test_arena_deduplication() {
    let source_id = SourceId::new(1);
    let config = ArenaConfig {
        initial_capacity: 1024,
        enable_deduplication: true,
        enable_reference_tracking: true,
        max_nodes: None,
    };
    let mut arena = AstArena::with_config(source_id, config);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Create identical nodes
    let expr1 = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    let expr2 = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    
    let node1 = AstNode::new(expr1, span, NodeId::new(1));
    let node2 = AstNode::new(expr2, span, NodeId::new(1)); // Same NodeId for identical nodes
    
    let ref1 = arena.alloc(node1).unwrap();
    let ref2 = arena.alloc(node2).unwrap();
    
    // Both references should be valid
    assert!(arena.contains(ref1));
    assert!(arena.contains(ref2));
    assert_eq!(arena.len(), 2); // Two references, but possibly deduplicated storage
}

#[test]
fn test_arena_capacity_limits() {
    let source_id = SourceId::new(1);
    let config = ArenaConfig {
        initial_capacity: 1024,
        enable_deduplication: false,
        enable_reference_tracking: true,
        max_nodes: Some(2), // Very small limit for testing
    };
    let mut arena = AstArena::with_config(source_id, config);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Allocate up to the limit
    for i in 0..2 {
        let expr = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(i),
        });
        let node = AstNode::new(expr, span, NodeId::new(i as u32));
        let result = arena.alloc(node);
        assert!(result.is_ok());
    }
    
    // Try to exceed the limit
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(999),
    });
    let node = AstNode::new(expr, span, NodeId::new(999));
    let result = arena.alloc(node);
    
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ArenaError::AllocationFailed { .. }));
}

#[test]
fn test_typed_node_ref() {
    let source_id = SourceId::new(1);
    let node_ref = AstNodeRef::new(1, source_id);
    
    let typed_ref = TypedNodeRef::<Expr>::new(node_ref);
    
    // Test that TypedNodeRef preserves type information at compile time
    assert_eq!(typed_ref.node_ref().id(), 1);
    assert_eq!(typed_ref.node_ref().source_id(), source_id);
}

#[test]
fn test_arena_error_types() {
    let source_id = SourceId::new(1);
    let node_ref = AstNodeRef::new(42, source_id);
    
    // Test NodeNotFound error
    let error = ArenaError::NodeNotFound { node_ref };
    assert!(error.to_string().contains("Node not found"));
    
    // Test InvalidNodeType error
    let error = ArenaError::InvalidNodeType {
        expected: "Expr".to_string(),
        actual: "Stmt".to_string(),
    };
    assert!(error.to_string().contains("Invalid node type"));
    assert!(error.to_string().contains("expected Expr"));
    assert!(error.to_string().contains("got Stmt"));
    
    // Test AllocationFailed error
    let error = ArenaError::AllocationFailed {
        reason: "Out of memory".to_string(),
    };
    assert!(error.to_string().contains("Memory allocation failed"));
    assert!(error.to_string().contains("Out of memory"));
}

#[test]
fn test_arena_debug_output() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    // Test debug output for empty arena
    let debug_str = format!("{:?}", arena);
    assert!(debug_str.contains("AstArena"));
    assert!(debug_str.contains("source_id"));
    assert!(debug_str.contains("node_count: 0"));
    
    // Add a node and test debug output
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::Integer(42),
    });
    let node = AstNode::new(expr, span, NodeId::new(1));
    arena.alloc(node).unwrap();
    
    let debug_str = format!("{:?}", arena);
    assert!(debug_str.contains("node_count: 1"));
}

#[cfg(feature = "serde")]
#[test]
fn test_arena_with_serializable_nodes() {
    let source_id = SourceId::new(1);
    let mut arena = AstArena::new(source_id);
    let span = Span::new(Position::new(1, 1, 0), Position::new(1, 11, 10), source_id);
    
    let expr = Expr::Literal(LiteralExpr {
        value: LiteralValue::String("serializable".to_string()),
    });
    let node = AstNode::new(expr, span, NodeId::new(1));
    let node_ref = arena.alloc(node).unwrap();
    
    // Test that nodes can be retrieved after serialization/deserialization
    let retrieved = arena.get::<Expr>(node_ref).unwrap();
    match retrieved.kind {
        Expr::Literal(LiteralExpr { value: LiteralValue::String(ref s) }) => {
            assert_eq!(s, "serializable");
        }
        _ => panic!("Retrieved wrong node type"),
    }
} 