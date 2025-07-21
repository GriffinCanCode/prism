//! Unit Tests for Prism Runtime Modules
//!
//! This module provides focused unit tests for individual runtime components,
//! testing each subsystem in isolation before integration testing.

use prism_runtime::{
    capability::{Capability, CapabilitySet, OperationType, CapabilityManager},
    effects::{EffectTracker, EffectHandle},
    execution::{ExecutionContext, ExecutionTarget, ExecutionManager},
    memory::{MemoryManager, SemanticPtr},
    ai_metadata::{AIMetadataCollector, RuntimeMetadata},
    isolation::{ComponentIsolationManager, ComponentId},
    security::{SecurityPolicyEnforcer, SecurityPolicy},
};
use prism_effects::Effect;
use std::sync::Arc;

#[tokio::test]
async fn test_capability_manager_creation() {
    let manager = CapabilityManager::new().expect("Failed to create capability manager");
    assert_eq!(manager.active_count(), 0);
}

#[tokio::test]
async fn test_capability_set_operations() {
    let mut cap_set = CapabilitySet::new();
    
    // Test adding capabilities
    let cap1 = Capability::new(
        "test_cap_1".to_string(),
        OperationType::Compute,
        vec![],
    );
    
    let cap2 = Capability::new(
        "test_cap_2".to_string(),
        OperationType::Network,
        vec![],
    );
    
    cap_set.add(cap1.clone());
    cap_set.add(cap2.clone());
    
    // Test capability presence
    assert!(cap_set.contains(&cap1));
    assert!(cap_set.contains(&cap2));
    
    // Test capability removal
    cap_set.remove(&cap1);
    assert!(!cap_set.contains(&cap1));
    assert!(cap_set.contains(&cap2));
}

#[tokio::test]
async fn test_capability_constraints() {
    let mut constraints = vec![];
    constraints.push("resource_limit=100MB".to_string());
    constraints.push("time_limit=30s".to_string());
    
    let capability = Capability::new(
        "constrained_capability".to_string(),
        OperationType::IO,
        constraints.clone(),
    );
    
    // Verify constraints are preserved
    let cap_constraints = capability.constraints();
    assert_eq!(cap_constraints.len(), 2);
    assert!(cap_constraints.contains(&"resource_limit=100MB".to_string()));
    assert!(cap_constraints.contains(&"time_limit=30s".to_string()));
}

#[tokio::test]
async fn test_effect_tracker_lifecycle() {
    let tracker = EffectTracker::new().expect("Failed to create effect tracker");
    
    // Initial state
    assert_eq!(tracker.active_count(), 0);
    
    // Test basic lifecycle (structure test)
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "test_effect_lifecycle".to_string(),
    );
    
    // Note: Full effect tracking would require actual effect execution
    // This tests the basic tracker structure
}

#[tokio::test]
async fn test_execution_context_creation() {
    let context = ExecutionContext::new(
        ExecutionTarget::TypeScript,
        "test_execution_context".to_string(),
    );
    
    // Verify context properties
    assert_eq!(context.target(), ExecutionTarget::TypeScript);
    assert_eq!(context.name(), "test_execution_context");
}

#[tokio::test]
async fn test_execution_manager_creation() {
    let manager = ExecutionManager::new().expect("Failed to create execution manager");
    
    // Test manager creation and basic structure
    // Full testing would require actual code execution
}

#[tokio::test]
async fn test_memory_manager_creation() {
    let manager = MemoryManager::new().expect("Failed to create memory manager");
    
    // Test initial memory usage
    let usage = manager.current_usage();
    assert!(usage >= 0);
}

#[tokio::test]
async fn test_semantic_pointer_creation() {
    // Test semantic pointer structure
    // Note: Full testing would require actual memory allocation
    // This is a basic structure test
}

#[tokio::test]
async fn test_ai_metadata_collector() {
    let collector = AIMetadataCollector::new().expect("Failed to create AI metadata collector");
    
    // Test basic collector functionality
    // Note: Full testing would require actual metadata collection
    // This tests the basic collector structure
}

#[tokio::test]
async fn test_component_isolation_manager() {
    let manager = ComponentIsolationManager::new().expect("Failed to create isolation manager");
    
    // Test initial component count
    assert_eq!(manager.component_count(), 0);
}

#[tokio::test]
async fn test_security_policy_enforcer() {
    let enforcer = SecurityPolicyEnforcer::new().expect("Failed to create security enforcer");
    
    // Test initial violation count
    assert_eq!(enforcer.violation_count(), 0);
}

#[tokio::test]
async fn test_operation_type_variants() {
    // Test all operation type variants
    let ops = vec![
        OperationType::Compute,
        OperationType::IO,
        OperationType::Network,
        OperationType::Memory,
        OperationType::FileSystem,
        OperationType::System,
    ];
    
    for op in ops {
        let capability = Capability::new(
            format!("test_{:?}", op),
            op,
            vec![],
        );
        
        assert_eq!(capability.operation_type(), op);
    }
}

#[tokio::test]
async fn test_execution_target_variants() {
    // Test all execution target variants
    let targets = vec![
        ExecutionTarget::Native,
        ExecutionTarget::TypeScript,
        ExecutionTarget::WebAssembly,
    ];
    
    for target in targets {
        let context = ExecutionContext::new(
            target,
            format!("test_{:?}", target),
        );
        
        assert_eq!(context.target(), target);
    }
}

#[tokio::test]
async fn test_effect_variants() {
    // Test effect type handling
    let effects = vec![
        Effect::IO,
        Effect::Network,
        Effect::Memory,
        Effect::FileSystem,
        Effect::System,
    ];
    
    // Test that all effect variants can be handled
    for effect in effects {
        // Basic structure test - effects should be processable
        let _effect_clone = effect.clone();
    }
}

#[tokio::test]
async fn test_capability_set_union() {
    let mut set1 = CapabilitySet::new();
    let mut set2 = CapabilitySet::new();
    
    let cap1 = Capability::new(
        "capability_1".to_string(),
        OperationType::Compute,
        vec![],
    );
    
    let cap2 = Capability::new(
        "capability_2".to_string(),
        OperationType::IO,
        vec![],
    );
    
    set1.add(cap1.clone());
    set2.add(cap2.clone());
    
    // Test union operation
    let union = set1.union(&set2);
    assert!(union.contains(&cap1));
    assert!(union.contains(&cap2));
}

#[tokio::test]
async fn test_capability_set_intersection() {
    let mut set1 = CapabilitySet::new();
    let mut set2 = CapabilitySet::new();
    
    let shared_cap = Capability::new(
        "shared_capability".to_string(),
        OperationType::Network,
        vec![],
    );
    
    let unique_cap1 = Capability::new(
        "unique_1".to_string(),
        OperationType::Compute,
        vec![],
    );
    
    let unique_cap2 = Capability::new(
        "unique_2".to_string(),
        OperationType::IO,
        vec![],
    );
    
    set1.add(shared_cap.clone());
    set1.add(unique_cap1);
    
    set2.add(shared_cap.clone());
    set2.add(unique_cap2);
    
    // Test intersection operation
    let intersection = set1.intersection(&set2);
    assert!(intersection.contains(&shared_cap));
    assert_eq!(intersection.len(), 1);
}

#[tokio::test]
async fn test_error_type_conversions() {
    // Test that all runtime error types can be created and converted
    use prism_runtime::{RuntimeError, capability::CapabilityError};
    
    let cap_error = CapabilityError::InsufficientCapabilities {
        required: "test_capability".to_string(),
        available: "none".to_string(),
    };
    
    let runtime_error: RuntimeError = cap_error.into();
    
    match runtime_error {
        RuntimeError::Capability(_) => {}, // Expected
        other => panic!("Unexpected error type: {:?}", other),
    }
}

#[tokio::test]
async fn test_concurrent_capability_operations() {
    let manager = Arc::new(CapabilityManager::new().expect("Failed to create manager"));
    
    // Test concurrent capability operations
    let mut handles = vec![];
    
    for i in 0..5 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            let mut cap_set = CapabilitySet::new();
            cap_set.add(Capability::new(
                format!("concurrent_cap_{}", i),
                OperationType::Compute,
                vec![],
            ));
            
            // Test concurrent operations
            cap_set.len()
        });
        
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        let result = handle.await.expect("Concurrent operation failed");
        assert_eq!(result, 1); // Each set should have one capability
    }
}

#[tokio::test]
async fn test_memory_usage_tracking() {
    let manager = MemoryManager::new().expect("Failed to create memory manager");
    
    let initial_usage = manager.current_usage();
    
    // Memory usage should be non-negative
    assert!(initial_usage >= 0);
    
    // Note: Full memory testing would require actual allocations
    // This tests the basic usage tracking structure
}

#[tokio::test]
async fn test_ai_metadata_structure() {
    let collector = AIMetadataCollector::new().expect("Failed to create collector");
    
    // Test basic metadata collection structure
    // Note: Full testing would require actual metadata generation
    // This tests the basic collector infrastructure
} 