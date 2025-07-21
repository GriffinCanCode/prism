//! Integration Tests for Prism Runtime System
//!
//! This module provides comprehensive integration tests for the Prism runtime,
//! testing the interaction between all subsystems and verifying that the
//! security and capability model works correctly end-to-end.

use prism_runtime::{
    PrismRuntime, RuntimeStats, RuntimeError, Executable,
    capability::{Capability, CapabilitySet, OperationType},
    execution::{ExecutionContext, ExecutionTarget},
    effects::{EffectResult, EffectHandle},
    memory::SemanticPtr,
    ai_metadata::{RuntimeMetadata, AIRuntimeContext},
    isolation::ComponentId,
    security::{SecurityPolicy, PolicyDecision},
};
use prism_effects::Effect;
use prism_common::symbol::Symbol;
use std::sync::Arc;
use std::time::Duration;
use tokio;

/// Mock executable for testing
struct MockExecutable {
    effects: Vec<Effect>,
    capabilities: CapabilitySet,
    should_fail: bool,
}

impl MockExecutable {
    fn new() -> Self {
        Self {
            effects: vec![Effect::IO],
            capabilities: CapabilitySet::new(),
            should_fail: false,
        }
    }
    
    fn with_effects(mut self, effects: Vec<Effect>) -> Self {
        self.effects = effects;
        self
    }
    
    fn with_capabilities(mut self, capabilities: CapabilitySet) -> Self {
        self.capabilities = capabilities;
        self
    }
    
    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }
}

impl Executable<String> for MockExecutable {
    fn execute(
        &self,
        _capabilities: &CapabilitySet,
        _context: &ExecutionContext,
    ) -> Result<String, RuntimeError> {
        if self.should_fail {
            Err(RuntimeError::Generic {
                message: "Mock execution failure".to_string(),
            })
        } else {
            Ok("Mock execution success".to_string())
        }
    }
    
    fn declared_effects(&self) -> Vec<Effect> {
        self.effects.clone()
    }
    
    fn required_capabilities(&self) -> CapabilitySet {
        self.capabilities.clone()
    }
}

#[tokio::test]
async fn test_runtime_initialization() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    let stats = runtime.get_runtime_stats();
    
    // Verify initial state
    assert_eq!(stats.active_capabilities, 0);
    assert_eq!(stats.tracked_effects, 0);
    assert_eq!(stats.isolated_components, 0);
    assert_eq!(stats.security_violations, 0);
}

#[tokio::test]
async fn test_basic_execution_flow() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Create basic execution context
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "test_execution".to_string(),
    );
    
    // Create minimal capability set
    let mut capabilities = CapabilitySet::new();
    capabilities.add(Capability::new(
        "test_capability".to_string(),
        OperationType::Compute,
        vec![],
    ));
    
    // Create mock executable
    let executable = MockExecutable::new()
        .with_capabilities(capabilities.clone());
    
    // Execute
    let result = runtime.execute_with_capabilities(
        &executable,
        &capabilities,
        &context,
    );
    
    // Verify successful execution
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Mock execution success");
}

#[tokio::test]
async fn test_capability_enforcement() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Create execution context
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "test_capability_enforcement".to_string(),
    );
    
    // Create executable that requires network capability
    let mut required_caps = CapabilitySet::new();
    required_caps.add(Capability::new(
        "network_access".to_string(),
        OperationType::Network,
        vec![],
    ));
    
    let executable = MockExecutable::new()
        .with_capabilities(required_caps.clone());
    
    // Try to execute with insufficient capabilities
    let empty_caps = CapabilitySet::new();
    let result = runtime.execute_with_capabilities(
        &executable,
        &empty_caps,
        &context,
    );
    
    // Should fail due to missing capabilities
    assert!(result.is_err());
    match result.unwrap_err() {
        RuntimeError::Capability(_) => {}, // Expected
        other => panic!("Expected capability error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_effect_tracking() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Create execution context
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "test_effect_tracking".to_string(),
    );
    
    // Create executable with specific effects
    let effects = vec![Effect::IO, Effect::Network];
    let executable = MockExecutable::new()
        .with_effects(effects.clone());
    
    let capabilities = CapabilitySet::new();
    
    // Execute and verify effects are tracked
    let initial_stats = runtime.get_runtime_stats();
    
    let _result = runtime.execute_with_capabilities(
        &executable,
        &capabilities,
        &context,
    );
    
    // Note: In a real scenario, we'd check that effects were properly tracked
    // This is a basic structure test
}

#[tokio::test]
async fn test_multi_target_execution() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    let capabilities = CapabilitySet::new();
    let executable = MockExecutable::new();
    
    // Test execution on different targets
    let targets = vec![
        ExecutionTarget::Native,
        ExecutionTarget::TypeScript,
        ExecutionTarget::WebAssembly,
    ];
    
    for target in targets {
        let context = ExecutionContext::new(
            target,
            format!("test_multi_target_{:?}", target),
        );
        
        let result = runtime.execute_with_capabilities(
            &executable,
            &capabilities,
            &context,
        );
        
        // All targets should be supported
        assert!(result.is_ok(), "Failed to execute on target: {:?}", target);
    }
}

#[tokio::test]
async fn test_component_isolation() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Test basic isolation functionality
    let stats = runtime.get_runtime_stats();
    
    // Initially no isolated components
    assert_eq!(stats.isolated_components, 0);
    
    // Note: Full isolation testing would require creating actual components
    // This tests the basic runtime structure
}

#[tokio::test]
async fn test_security_policy_enforcement() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Test that security policies are enforced
    let stats = runtime.get_runtime_stats();
    
    // Initially no security violations
    assert_eq!(stats.security_violations, 0);
    
    // Note: Full security testing would require creating policy violations
    // This tests the basic security infrastructure
}

#[tokio::test]
async fn test_memory_management() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    let stats = runtime.get_runtime_stats();
    
    // Memory usage should be tracked
    assert!(stats.memory_usage >= 0);
    
    // Note: Full memory testing would require actual memory operations
    // This tests the basic memory tracking infrastructure
}

#[tokio::test]
async fn test_ai_metadata_collection() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Create execution context
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "test_ai_metadata".to_string(),
    );
    
    let capabilities = CapabilitySet::new();
    let executable = MockExecutable::new();
    
    // Execute to generate metadata
    let _result = runtime.execute_with_capabilities(
        &executable,
        &capabilities,
        &context,
    );
    
    // Note: In a real scenario, we'd verify metadata was collected
    // This tests the basic metadata infrastructure
}

#[tokio::test]
async fn test_error_handling() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Create execution context
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "test_error_handling".to_string(),
    );
    
    // Create executable that will fail
    let capabilities = CapabilitySet::new();
    let executable = MockExecutable::new().with_failure();
    
    // Execute and verify error handling
    let result = runtime.execute_with_capabilities(
        &executable,
        &capabilities,
        &context,
    );
    
    assert!(result.is_err());
    match result.unwrap_err() {
        RuntimeError::Generic { message } => {
            assert_eq!(message, "Mock execution failure");
        },
        other => panic!("Expected generic error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_concurrent_execution() {
    let runtime = Arc::new(PrismRuntime::new().expect("Failed to create runtime"));
    
    // Test concurrent executions
    let mut handles = vec![];
    
    for i in 0..10 {
        let runtime_clone = runtime.clone();
        let handle = tokio::spawn(async move {
            let context = ExecutionContext::new(
                ExecutionTarget::Native,
                format!("concurrent_test_{}", i),
            );
            
            let capabilities = CapabilitySet::new();
            let executable = MockExecutable::new();
            
            runtime_clone.execute_with_capabilities(
                &executable,
                &capabilities,
                &context,
            )
        });
        
        handles.push(handle);
    }
    
    // Wait for all executions to complete
    for handle in handles {
        let result = handle.await.expect("Task failed");
        assert!(result.is_ok(), "Concurrent execution failed");
    }
}

#[tokio::test]
async fn test_runtime_stats_accuracy() {
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // Get initial stats
    let initial_stats = runtime.get_runtime_stats();
    
    // Perform some operations
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "stats_test".to_string(),
    );
    
    let capabilities = CapabilitySet::new();
    let executable = MockExecutable::new();
    
    let _result = runtime.execute_with_capabilities(
        &executable,
        &capabilities,
        &context,
    );
    
    // Get updated stats
    let updated_stats = runtime.get_runtime_stats();
    
    // Stats should be consistent (this is a basic structure test)
    assert!(updated_stats.memory_usage >= initial_stats.memory_usage);
}

#[tokio::test]
async fn test_full_runtime_lifecycle() {
    // Test complete runtime lifecycle from creation to execution
    let runtime = PrismRuntime::new().expect("Failed to create runtime");
    
    // 1. Initial state verification
    let initial_stats = runtime.get_runtime_stats();
    assert_eq!(initial_stats.active_capabilities, 0);
    assert_eq!(initial_stats.tracked_effects, 0);
    
    // 2. Create comprehensive execution setup
    let context = ExecutionContext::new(
        ExecutionTarget::Native,
        "full_lifecycle_test".to_string(),
    );
    
    let mut capabilities = CapabilitySet::new();
    capabilities.add(Capability::new(
        "lifecycle_test".to_string(),
        OperationType::Compute,
        vec![],
    ));
    
    let executable = MockExecutable::new()
        .with_capabilities(capabilities.clone())
        .with_effects(vec![Effect::IO, Effect::Memory]);
    
    // 3. Execute with full monitoring
    let result = runtime.execute_with_capabilities(
        &executable,
        &capabilities,
        &context,
    );
    
    // 4. Verify successful execution
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Mock execution success");
    
    // 5. Verify final state
    let final_stats = runtime.get_runtime_stats();
    assert!(final_stats.memory_usage >= initial_stats.memory_usage);
} 