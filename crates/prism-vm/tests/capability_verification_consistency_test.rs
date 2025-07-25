//! Capability Verification Consistency Tests
//!
//! This test suite verifies that capability verification is consistent between
//! compile-time analysis and runtime execution, ensuring no security gaps exist
//! between the two verification phases.

use prism_vm::{
    PrismVM, VMConfig,
    bytecode::{PrismBytecode, FunctionDefinition, Instruction, instructions::PrismOpcode},
    execution::{StackValue, ExecutionResult},
    capabilities::{CapabilitySet, Capability},
    PrismVMError,
};
use prism_effects::effects::definition::Effect;
use prism_common::span::Span;
use std::collections::HashMap;

/// Test that function-level capability requirements are verified consistently
#[test]
fn test_function_capability_verification_consistency() {
    let mut vm = create_test_vm();
    
    // Create a function that requires FileSystem capability
    let function = create_test_function_with_capabilities(vec!["FileSystem".to_string()]);
    let bytecode = create_test_bytecode(function);
    
    // Test 1: VM without required capability should fail
    vm.load_bytecode("test_module", bytecode.clone()).unwrap();
    let result = vm.execute_function("test_module", "test_function", vec![]);
    
    assert!(matches!(result, Err(PrismVMError::CapabilityViolation { .. })));
    
    // Test 2: VM with required capability should succeed
    let mut vm_with_caps = create_test_vm_with_capabilities(vec!["FileSystem".to_string()]);
    vm_with_caps.load_bytecode("test_module", bytecode).unwrap();
    let result = vm_with_caps.execute_function("test_module", "test_function", vec![]);
    
    assert!(result.is_ok());
}

/// Test that instruction-level capability requirements are verified consistently
#[test]
fn test_instruction_capability_verification_consistency() {
    let mut vm = create_test_vm();
    
    // Create a function with an instruction that requires Network capability
    let function = create_test_function_with_instruction_capabilities();
    let bytecode = create_test_bytecode(function);
    
    // Test 1: VM without required capability should fail at instruction level
    vm.load_bytecode("test_module", bytecode.clone()).unwrap();
    let result = vm.execute_function("test_module", "test_function", vec![]);
    
    assert!(matches!(result, Err(PrismVMError::CapabilityViolation { .. })));
    
    // Test 2: VM with required capability should succeed
    let mut vm_with_caps = create_test_vm_with_capabilities(vec!["Network".to_string()]);
    vm_with_caps.load_bytecode("test_module", bytecode).unwrap();
    let result = vm_with_caps.execute_function("test_module", "test_function", vec![]);
    
    assert!(result.is_ok());
}

/// Test that effect-related capability requirements are verified consistently
#[test]
fn test_effect_capability_verification_consistency() {
    let mut vm = create_test_vm();
    
    // Create a function with effects that require specific capabilities
    let function = create_test_function_with_effects();
    let bytecode = create_test_bytecode(function);
    
    // Test 1: VM without required capability should fail
    vm.load_bytecode("test_module", bytecode.clone()).unwrap();
    let result = vm.execute_function("test_module", "test_function", vec![]);
    
    assert!(matches!(result, Err(PrismVMError::CapabilityViolation { .. })));
    
    // Test 2: VM with required capability should succeed
    let mut vm_with_caps = create_test_vm_with_capabilities(vec!["FileSystem".to_string()]);
    vm_with_caps.load_bytecode("test_module", bytecode).unwrap();
    let result = vm_with_caps.execute_function("test_module", "test_function", vec![]);
    
    assert!(result.is_ok());
}

/// Test that JIT compilation maintains capability verification consistency
#[test]
fn test_jit_capability_verification_consistency() {
    let mut vm_config = VMConfig::default();
    vm_config.enable_jit = true;
    let mut vm = PrismVM::new(vm_config).unwrap();
    
    // Create a function that will be JIT compiled
    let function = create_test_function_with_mixed_capabilities();
    let bytecode = create_test_bytecode(function);
    
    // Test 1: JIT compilation should generate capability checks
    vm.load_bytecode("test_module", bytecode.clone()).unwrap();
    let result = vm.execute_function("test_module", "test_function", vec![]);
    
    assert!(matches!(result, Err(PrismVMError::CapabilityViolation { .. })));
    
    // Test 2: JIT with capabilities should work
    let mut vm_with_caps = create_test_vm_with_capabilities_and_jit(vec![
        "FileSystem".to_string(),
        "Network".to_string(),
    ]);
    vm_with_caps.load_bytecode("test_module", bytecode).unwrap();
    let result = vm_with_caps.execute_function("test_module", "test_function", vec![]);
    
    assert!(result.is_ok());
}

/// Test that capability verification is consistent across multiple execution paths
#[test]
fn test_multi_path_capability_verification_consistency() {
    // Test interpreter path
    let mut interpreter_vm = create_test_vm();
    let function = create_test_function_with_capabilities(vec!["ProcessControl".to_string()]);
    let bytecode = create_test_bytecode(function.clone());
    
    interpreter_vm.load_bytecode("test_module", bytecode).unwrap();
    let interpreter_result = interpreter_vm.execute_function("test_module", "test_function", vec![]);
    
    // Test JIT path
    let mut jit_vm = create_test_vm_with_jit();
    let bytecode = create_test_bytecode(function);
    
    jit_vm.load_bytecode("test_module", bytecode).unwrap();
    let jit_result = jit_vm.execute_function("test_module", "test_function", vec![]);
    
    // Both should fail consistently
    assert!(matches!(interpreter_result, Err(PrismVMError::CapabilityViolation { .. })));
    assert!(matches!(jit_result, Err(PrismVMError::CapabilityViolation { .. })));
}

/// Test that capability verification handles edge cases consistently
#[test]
fn test_edge_case_capability_verification_consistency() {
    let mut vm = create_test_vm();
    
    // Test empty capabilities
    let function = create_test_function_with_capabilities(vec![]);
    let bytecode = create_test_bytecode(function);
    
    vm.load_bytecode("test_module", bytecode).unwrap();
    let result = vm.execute_function("test_module", "test_function", vec![]);
    
    assert!(result.is_ok()); // Should succeed with no capability requirements
    
    // Test invalid capability names
    let function = create_test_function_with_capabilities(vec!["InvalidCapability".to_string()]);
    let bytecode = create_test_bytecode(function);
    
    vm.load_bytecode("test_module", bytecode).unwrap();
    let result = vm.execute_function("test_module", "test_function", vec![]);
    
    assert!(matches!(result, Err(PrismVMError::CapabilityViolation { .. })));
}

// Helper functions for test setup

fn create_test_vm() -> PrismVM {
    let config = VMConfig::default();
    PrismVM::new(config).unwrap()
}

fn create_test_vm_with_capabilities(capabilities: Vec<String>) -> PrismVM {
    let mut vm = create_test_vm();
    let mut capability_set = CapabilitySet::new();
    
    for cap_name in capabilities {
        capability_set.add_capability(Capability::new(cap_name));
    }
    
    vm.set_capabilities(capability_set);
    vm
}

fn create_test_vm_with_jit() -> PrismVM {
    let mut config = VMConfig::default();
    config.enable_jit = true;
    PrismVM::new(config).unwrap()
}

fn create_test_vm_with_capabilities_and_jit(capabilities: Vec<String>) -> PrismVM {
    let mut config = VMConfig::default();
    config.enable_jit = true;
    let mut vm = PrismVM::new(config).unwrap();
    
    let mut capability_set = CapabilitySet::new();
    for cap_name in capabilities {
        capability_set.add_capability(Capability::new(cap_name));
    }
    
    vm.set_capabilities(capability_set);
    vm
}

fn create_test_function_with_capabilities(capabilities: Vec<String>) -> FunctionDefinition {
    FunctionDefinition {
        id: 1,
        name: "test_function".to_string(),
        param_count: 0,
        local_count: 0,
        instructions: vec![
            Instruction {
                opcode: PrismOpcode::NOP,
                operands: vec![],
                required_capabilities: vec![],
                effects: vec![],
            }
        ],
        capabilities,
        effects: vec![],
    }
}

fn create_test_function_with_instruction_capabilities() -> FunctionDefinition {
    FunctionDefinition {
        id: 1,
        name: "test_function".to_string(),
        param_count: 0,
        local_count: 0,
        instructions: vec![
            Instruction {
                opcode: PrismOpcode::NOP,
                operands: vec![],
                required_capabilities: vec!["Network".to_string()],
                effects: vec![],
            }
        ],
        capabilities: vec![],
        effects: vec![],
    }
}

fn create_test_function_with_effects() -> FunctionDefinition {
    let effect = Effect::new("IO.FileSystem.Read".to_string(), Span::default());
    
    FunctionDefinition {
        id: 1,
        name: "test_function".to_string(),
        param_count: 0,
        local_count: 0,
        instructions: vec![
            Instruction {
                opcode: PrismOpcode::NOP,
                operands: vec![],
                required_capabilities: vec![],
                effects: vec![effect.clone()],
            }
        ],
        capabilities: vec![],
        effects: vec![effect],
    }
}

fn create_test_function_with_mixed_capabilities() -> FunctionDefinition {
    let effect = Effect::new("IO.Network.Connect".to_string(), Span::default());
    
    FunctionDefinition {
        id: 1,
        name: "test_function".to_string(),
        param_count: 0,
        local_count: 0,
        instructions: vec![
            Instruction {
                opcode: PrismOpcode::NOP,
                operands: vec![],
                required_capabilities: vec!["FileSystem".to_string()],
                effects: vec![effect.clone()],
            }
        ],
        capabilities: vec!["FileSystem".to_string()],
        effects: vec![effect],
    }
}

fn create_test_bytecode(function: FunctionDefinition) -> PrismBytecode {
    let mut bytecode = PrismBytecode::new();
    bytecode.add_function(function);
    bytecode
} 