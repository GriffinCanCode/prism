//! Integration Tests for VM Effect Enforcement
//!
//! This module provides comprehensive integration tests for the VM effect enforcement
//! system, ensuring that effects are properly validated and executed at runtime
//! across both interpreter and JIT compilation paths.

use prism_vm::{
    PrismVM, VMConfig, ExecutionResult,
    execution::{VMEffectEnforcer, EffectEnforcementConfig, StackValue},
    bytecode::{PrismBytecode, FunctionDefinition, Instruction, instructions::PrismOpcode},
};
use prism_effects::{
    effects::{Effect, EffectDefinition, EffectSystem, EffectCategory},
    execution::ExecutionSystem as EffectExecutionSystem,
    validation::EffectValidator,
    security::SecureExecutionContext,
};
use prism_runtime::authority::capability::{CapabilitySet, Capability};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Helper function to create a test effect system
fn create_test_effect_system() -> (
    Arc<RwLock<EffectSystem>>,
    Arc<RwLock<EffectExecutionSystem>>,
    Arc<EffectValidator>,
) {
    let effect_system = Arc::new(RwLock::new(EffectSystem::new()));
    let effect_executor = Arc::new(RwLock::new(EffectExecutionSystem::new()));
    let effect_validator = Arc::new(EffectValidator::new());
    
    // Register some test effects
    {
        let mut system = effect_system.write().unwrap();
        
        let io_effect = EffectDefinition {
            name: "IOEffect".to_string(),
            description: "File I/O operations".to_string(),
            category: EffectCategory::IO,
            parent_effect: None,
            parameters: vec![],
            security_implications: vec!["File system access".to_string()],
            ai_metadata: prism_effects::effects::definition::EffectAIMetadata {
                business_purpose: "Handle file operations".to_string(),
                stakeholders: vec!["System administrators".to_string()],
                business_rules: vec!["I/O must be within quota".to_string()],
            },
        };
        
        let network_effect = EffectDefinition {
            name: "NetworkEffect".to_string(),
            description: "Network operations".to_string(),
            category: EffectCategory::Network,
            parent_effect: None,
            parameters: vec![],
            security_implications: vec!["Network access".to_string()],
            ai_metadata: prism_effects::effects::definition::EffectAIMetadata {
                business_purpose: "Handle network operations".to_string(),
                stakeholders: vec!["Network administrators".to_string()],
                business_rules: vec!["Network access must be authorized".to_string()],
            },
        };
        
        system.register_effect(io_effect).unwrap();
        system.register_effect(network_effect).unwrap();
    }
    
    (effect_system, effect_executor, effect_validator)
}

/// Helper function to create test capabilities
fn create_test_capabilities() -> CapabilitySet {
    let mut capabilities = CapabilitySet::new();
    
    let io_capability = Capability::new(
        "io.read".to_string(),
        prism_runtime::authority::capability::Operation::Read,
        std::time::SystemTime::now() + Duration::from_secs(3600), // Valid for 1 hour
    );
    
    let network_capability = Capability::new(
        "network.connect".to_string(),
        prism_runtime::authority::capability::Operation::Execute,
        std::time::SystemTime::now() + Duration::from_secs(3600),
    );
    
    capabilities.add_capability(io_capability);
    capabilities.add_capability(network_capability);
    
    capabilities
}

/// Helper function to create test bytecode with effects
fn create_test_bytecode_with_effects() -> PrismBytecode {
    let io_effect = Effect {
        definition: "IOEffect".to_string(),
        span: prism_common::span::Span::new(0, 0, 0, 0),
    };
    
    let network_effect = Effect {
        definition: "NetworkEffect".to_string(),
        span: prism_common::span::Span::new(0, 0, 0, 0),
    };
    
    let function = FunctionDefinition {
        id: 1,
        name: "test_function".to_string(),
        param_count: 0,
        local_count: 1,
        instructions: vec![
            Instruction {
                opcode: PrismOpcode::EFFECT_ENTER(0),
                effects: vec![io_effect.clone()],
                required_capabilities: vec!["io.read".to_string()],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::EFFECT_INVOKE(0),
                effects: vec![io_effect.clone()],
                required_capabilities: vec!["io.read".to_string()],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::EFFECT_EXIT,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::LOAD_CONST(0),
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::RETURN_VALUE,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
        ],
        effects: vec![io_effect, network_effect],
        capabilities: vec!["io.read".to_string(), "network.connect".to_string()],
        metadata: None,
    };
    
    PrismBytecode {
        version: prism_vm::bytecode::BytecodeVersion::V1,
        constants: vec![prism_vm::bytecode::Constant::Integer(42)],
        functions: vec![function],
        metadata: prism_vm::bytecode::ModuleMetadata {
            name: "test_module".to_string(),
            version: "1.0.0".to_string(),
            dependencies: vec![],
            exports: vec!["test_function".to_string()],
        },
    }
}

#[tokio::test]
async fn test_basic_effect_enforcement() {
    // Create test components
    let (effect_system, effect_executor, effect_validator) = create_test_effect_system();
    let capabilities = create_test_capabilities();
    let bytecode = create_test_bytecode_with_effects();
    
    // Create VM with effect enforcement
    let mut vm = PrismVM::new().expect("Failed to create VM");
    vm.setup_effect_enforcement(effect_system, effect_executor, effect_validator)
        .expect("Failed to setup effect enforcement");
    
    // Set capabilities
    vm.set_capabilities(capabilities);
    
    // Load test module
    vm.load_module("test".to_string(), bytecode)
        .expect("Failed to load module");
    
    // Execute function with effects
    let result = vm.execute_function("test", "test_function", vec![])
        .expect("Failed to execute function");
    
    assert!(result.success);
    assert_eq!(result.return_value, Some(StackValue::Integer(42)));
    
    // Check effect statistics
    let effect_stats = vm.get_effect_stats().expect("No effect statistics");
    assert!(effect_stats.contexts_created > 0);
    assert!(effect_stats.effects_entered > 0);
    assert!(effect_stats.effects_executed > 0);
}

#[tokio::test]
async fn test_effect_validation_failure() {
    // Create test components with limited capabilities
    let (effect_system, effect_executor, effect_validator) = create_test_effect_system();
    let mut limited_capabilities = CapabilitySet::new();
    // Only add network capability, not I/O capability
    let network_capability = Capability::new(
        "network.connect".to_string(),
        prism_runtime::authority::capability::Operation::Execute,
        std::time::SystemTime::now() + Duration::from_secs(3600),
    );
    limited_capabilities.add_capability(network_capability);
    
    let bytecode = create_test_bytecode_with_effects();
    
    // Create VM with effect enforcement
    let mut vm = PrismVM::new().expect("Failed to create VM");
    vm.setup_effect_enforcement(effect_system, effect_executor, effect_validator)
        .expect("Failed to setup effect enforcement");
    
    // Set limited capabilities
    vm.set_capabilities(limited_capabilities);
    
    // Load test module
    vm.load_module("test".to_string(), bytecode)
        .expect("Failed to load module");
    
    // Execute function should fail due to missing I/O capability
    let result = vm.execute_function("test", "test_function", vec![]);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_effect_nesting_validation() {
    // Create test components
    let (effect_system, effect_executor, effect_validator) = create_test_effect_system();
    let capabilities = create_test_capabilities();
    
    // Create bytecode with nested effects
    let io_effect = Effect {
        definition: "IOEffect".to_string(),
        span: prism_common::span::Span::new(0, 0, 0, 0),
    };
    
    let network_effect = Effect {
        definition: "NetworkEffect".to_string(),
        span: prism_common::span::Span::new(0, 0, 0, 0),
    };
    
    let function = FunctionDefinition {
        id: 1,
        name: "nested_effects".to_string(),
        param_count: 0,
        local_count: 1,
        instructions: vec![
            // Enter first effect
            Instruction {
                opcode: PrismOpcode::EFFECT_ENTER(0), // IO effect
                effects: vec![io_effect.clone()],
                required_capabilities: vec!["io.read".to_string()],
                metadata: None,
            },
            // Enter nested effect
            Instruction {
                opcode: PrismOpcode::EFFECT_ENTER(1), // Network effect
                effects: vec![network_effect.clone()],
                required_capabilities: vec!["network.connect".to_string()],
                metadata: None,
            },
            // Invoke nested effect
            Instruction {
                opcode: PrismOpcode::EFFECT_INVOKE(1),
                effects: vec![network_effect.clone()],
                required_capabilities: vec!["network.connect".to_string()],
                metadata: None,
            },
            // Exit nested effect
            Instruction {
                opcode: PrismOpcode::EFFECT_EXIT,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            // Invoke first effect
            Instruction {
                opcode: PrismOpcode::EFFECT_INVOKE(0),
                effects: vec![io_effect.clone()],
                required_capabilities: vec!["io.read".to_string()],
                metadata: None,
            },
            // Exit first effect
            Instruction {
                opcode: PrismOpcode::EFFECT_EXIT,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::LOAD_CONST(0),
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::RETURN_VALUE,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
        ],
        effects: vec![io_effect, network_effect],
        capabilities: vec!["io.read".to_string(), "network.connect".to_string()],
        metadata: None,
    };
    
    let bytecode = PrismBytecode {
        version: prism_vm::bytecode::BytecodeVersion::V1,
        constants: vec![prism_vm::bytecode::Constant::Integer(100)],
        functions: vec![function],
        metadata: prism_vm::bytecode::ModuleMetadata {
            name: "nested_test".to_string(),
            version: "1.0.0".to_string(),
            dependencies: vec![],
            exports: vec!["nested_effects".to_string()],
        },
    };
    
    // Create VM with effect enforcement
    let mut vm = PrismVM::new().expect("Failed to create VM");
    vm.setup_effect_enforcement(effect_system, effect_executor, effect_validator)
        .expect("Failed to setup effect enforcement");
    
    // Set capabilities
    vm.set_capabilities(capabilities);
    
    // Load test module
    vm.load_module("nested_test".to_string(), bytecode)
        .expect("Failed to load module");
    
    // Execute function with nested effects
    let result = vm.execute_function("nested_test", "nested_effects", vec![])
        .expect("Failed to execute function");
    
    assert!(result.success);
    assert_eq!(result.return_value, Some(StackValue::Integer(100)));
    
    // Check effect statistics
    let effect_stats = vm.get_effect_stats().expect("No effect statistics");
    assert!(effect_stats.contexts_created > 0);
    assert_eq!(effect_stats.effects_entered, 2); // Two effects entered
    assert_eq!(effect_stats.effects_executed, 2); // Two effects executed
}

#[tokio::test]
async fn test_effect_abort_cleanup() {
    // Create test components
    let (effect_system, effect_executor, effect_validator) = create_test_effect_system();
    let capabilities = create_test_capabilities();
    
    // Create bytecode with effect abort
    let io_effect = Effect {
        definition: "IOEffect".to_string(),
        span: prism_common::span::Span::new(0, 0, 0, 0),
    };
    
    let function = FunctionDefinition {
        id: 1,
        name: "abort_test".to_string(),
        param_count: 0,
        local_count: 1,
        instructions: vec![
            Instruction {
                opcode: PrismOpcode::EFFECT_ENTER(0),
                effects: vec![io_effect.clone()],
                required_capabilities: vec!["io.read".to_string()],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::EFFECT_ABORT,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::LOAD_CONST(0),
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
            Instruction {
                opcode: PrismOpcode::RETURN_VALUE,
                effects: vec![],
                required_capabilities: vec![],
                metadata: None,
            },
        ],
        effects: vec![io_effect],
        capabilities: vec!["io.read".to_string()],
        metadata: None,
    };
    
    let bytecode = PrismBytecode {
        version: prism_vm::bytecode::BytecodeVersion::V1,
        constants: vec![prism_vm::bytecode::Constant::Integer(0)],
        functions: vec![function],
        metadata: prism_vm::bytecode::ModuleMetadata {
            name: "abort_test".to_string(),
            version: "1.0.0".to_string(),
            dependencies: vec![],
            exports: vec!["abort_test".to_string()],
        },
    };
    
    // Create VM with effect enforcement
    let mut vm = PrismVM::new().expect("Failed to create VM");
    vm.setup_effect_enforcement(effect_system, effect_executor, effect_validator)
        .expect("Failed to setup effect enforcement");
    
    // Set capabilities
    vm.set_capabilities(capabilities);
    
    // Load test module
    vm.load_module("abort_test".to_string(), bytecode)
        .expect("Failed to load module");
    
    // Execute function with effect abort
    let result = vm.execute_function("abort_test", "abort_test", vec![])
        .expect("Failed to execute function");
    
    assert!(result.success);
    
    // Check that effects were properly cleaned up
    let effect_stats = vm.get_effect_stats().expect("No effect statistics");
    assert!(effect_stats.contexts_created > 0);
    assert!(effect_stats.effects_entered > 0);
}

#[cfg(feature = "jit")]
#[tokio::test]
async fn test_jit_effect_enforcement() {
    // Create test components
    let (effect_system, effect_executor, effect_validator) = create_test_effect_system();
    let capabilities = create_test_capabilities();
    let bytecode = create_test_bytecode_with_effects();
    
    // Create VM with JIT enabled
    let mut vm_config = VMConfig::default();
    vm_config.enable_jit = true;
    vm_config.jit_threshold = 1; // Compile immediately
    
    let mut vm = PrismVM::with_config(vm_config).expect("Failed to create VM");
    vm.setup_effect_enforcement(effect_system, effect_executor, effect_validator)
        .expect("Failed to setup effect enforcement");
    
    // Set capabilities
    vm.set_capabilities(capabilities);
    
    // Load test module
    vm.load_module("jit_test".to_string(), bytecode)
        .expect("Failed to load module");
    
    // Execute function multiple times to trigger JIT compilation
    for i in 0..5 {
        let result = vm.execute_function("jit_test", "test_function", vec![])
            .expect(&format!("Failed to execute function on iteration {}", i));
        
        assert!(result.success);
        assert_eq!(result.return_value, Some(StackValue::Integer(42)));
    }
    
    // Check that JIT compilation occurred and effects were enforced
    let vm_stats = vm.get_stats();
    let effect_stats = vm.get_effect_stats().expect("No effect statistics");
    
    assert!(effect_stats.contexts_created >= 5); // At least one per execution
    assert!(effect_stats.effects_entered >= 5);
    assert!(effect_stats.effects_executed >= 5);
    
    // Check JIT statistics if available
    #[cfg(feature = "jit")]
    if let Some(jit_stats) = vm_stats.jit_stats {
        assert!(jit_stats.baseline_compilations > 0 || jit_stats.optimizing_compilations > 0);
    }
}

#[tokio::test]
async fn test_effect_enforcement_configuration() {
    // Create test components
    let (effect_system, effect_executor, effect_validator) = create_test_effect_system();
    let capabilities = create_test_capabilities();
    let bytecode = create_test_bytecode_with_effects();
    
    // Create VM with custom effect enforcement configuration
    let mut vm = PrismVM::new().expect("Failed to create VM");
    vm.setup_effect_enforcement(effect_system, effect_executor, effect_validator)
        .expect("Failed to setup effect enforcement");
    
    // Update effect enforcement configuration
    let config = EffectEnforcementConfig {
        strict_validation: true,
        track_execution: true,
        max_nesting_depth: 10,
        execution_timeout: Duration::from_secs(5),
        enable_caching: true,
    };
    
    vm.update_effect_config(config).expect("Failed to update effect config");
    
    // Set capabilities
    vm.set_capabilities(capabilities);
    
    // Load test module
    vm.load_module("config_test".to_string(), bytecode)
        .expect("Failed to load module");
    
    // Execute function
    let result = vm.execute_function("config_test", "test_function", vec![])
        .expect("Failed to execute function");
    
    assert!(result.success);
    
    // Verify configuration was applied
    let effect_stats = vm.get_effect_stats().expect("No effect statistics");
    assert!(effect_stats.contexts_created > 0);
    assert!(effect_stats.validation_time > Duration::from_nanos(0));
} 