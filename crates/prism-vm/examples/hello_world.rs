//! Hello World Example for Prism VM
//!
//! This example demonstrates the complete Prism VM pipeline:
//! 1. Creating bytecode programmatically
//! 2. Loading it into the VM
//! 3. Executing the program
//! 4. Observing the results

use prism_vm::{
    PrismVM, VMConfig, PrismBytecode, FunctionDefinition, Instruction, ConstantPool, 
    ConstantValue, TypeDefinition, TypeKind, PrimitiveType,
    instructions::PrismOpcode, execution::StackValue,
};
use prism_runtime::authority::capability::CapabilitySet;
use std::collections::HashMap;
use tracing::{info, Level};
use tracing_subscriber;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🚀 Starting Prism VM Hello World Example");

    // Create VM with default configuration
    let mut vm = PrismVM::new()?;
    
    // Set up basic capabilities
    let capabilities = CapabilitySet::new();
    vm.set_capabilities(capabilities);

    // Create a simple "Hello, World!" program
    let bytecode = create_hello_world_bytecode()?;
    
    // Load the program into the VM
    vm.load_module("hello_world".to_string(), bytecode)?;
    
    // Execute the main function
    info!("Executing main function...");
    let result = vm.execute_main("hello_world")?;
    
    // Display results
    if result.success {
        info!("✅ Execution completed successfully!");
        if let Some(return_value) = result.return_value {
            info!("Return value: {:?}", return_value);
        }
        
        info!("Execution Statistics:");
        info!("  - Instructions executed: {}", result.stats.instructions_executed);
        info!("  - Execution time: {} μs", result.stats.execution_time_us);
        info!("  - Max stack depth: {}", result.stats.max_stack_depth);
        info!("  - Function calls: {}", result.stats.function_calls);
        info!("  - Memory allocated: {} bytes", result.stats.memory_allocated);
    } else {
        info!("❌ Execution failed");
    }

    // Display VM statistics
    let vm_stats = vm.get_stats();
    info!("VM Statistics:");
    info!("  - Modules loaded: {}", vm_stats.modules_loaded);
    info!("  - Interpreter stats: {:?}", vm_stats.interpreter_stats);

    // Shutdown the VM
    vm.shutdown()?;
    info!("🏁 VM shutdown complete");

    Ok(())
}

/// Create a simple bytecode program that prints "Hello, World!"
fn create_hello_world_bytecode() -> Result<PrismBytecode, Box<dyn std::error::Error>> {
    info!("Creating Hello World bytecode...");
    
    // Create constant pool
    let mut constants = ConstantPool::new();
    
    // Add string constant for "Hello, World!"
    let hello_const_id = constants.add_string("Hello, World!".to_string())?;
    
    // Create type definitions
    let mut types = HashMap::new();
    
    // String type
    let string_type = TypeDefinition {
        id: 0,
        name: "String".to_string(),
        kind: TypeKind::Primitive(PrimitiveType::String),
        domain: None,
        business_rules: Vec::new(),
        validation_predicates: Vec::new(),
    };
    types.insert(0, string_type);
    
    // Unit type (for main function return)
    let unit_type = TypeDefinition {
        id: 1,
        name: "Unit".to_string(),
        kind: TypeKind::Primitive(PrimitiveType::Unit),
        domain: None,
        business_rules: Vec::new(),
        validation_predicates: Vec::new(),
    };
    types.insert(1, unit_type);

    // Create main function
    let main_function = FunctionDefinition {
        id: 0,
        name: "main".to_string(),
        type_id: 1, // Returns Unit
        param_count: 0,
        local_count: 1, // One local variable for the string
        max_stack_depth: 2,
        capabilities: Vec::new(),
        effects: Vec::new(),
        instructions: vec![
            // Load "Hello, World!" string constant
            Instruction::new(PrismOpcode::LOAD_CONST(hello_const_id)),
            
            // Store in local variable 0
            Instruction::new(PrismOpcode::STORE_LOCAL(0)),
            
            // Load it back for "printing" (in a real implementation, this would call a print function)
            Instruction::new(PrismOpcode::LOAD_LOCAL(0)),
            
            // For this example, we'll just leave the string on the stack as our "output"
            // In a real implementation, we'd have a PRINT instruction or call a print function
            
            // Return (the string will be our return value for demonstration)
            Instruction::new(PrismOpcode::RETURN),
        ],
        debug_info: None,
        ai_metadata: None,
    };

    // Create functions map
    let mut functions = HashMap::new();
    functions.insert("main".to_string(), main_function);

    // Create the complete bytecode
    let bytecode = PrismBytecode {
        version: prism_vm::bytecode::BytecodeVersion::V1_0,
        module_name: "hello_world".to_string(),
        constants,
        types,
        functions,
        metadata: prism_vm::bytecode::ModuleMetadata {
            source_file: Some("hello_world.prsm".to_string()),
            compilation_timestamp: std::time::SystemTime::now(),
            compiler_version: "0.1.0".to_string(),
            optimization_level: 0,
            debug_symbols: true,
            dependencies: Vec::new(),
            exports: Vec::new(),
            ai_metadata: None,
        },
    };

    info!("✅ Bytecode created successfully");
    Ok(bytecode)
}

/// Create a more complex example with arithmetic
#[allow(dead_code)]
fn create_arithmetic_bytecode() -> Result<PrismBytecode, Box<dyn std::error::Error>> {
    info!("Creating arithmetic bytecode...");
    
    let mut constants = ConstantPool::new();
    
    // Add numeric constants
    let const_10 = constants.add_integer(10)?;
    let const_20 = constants.add_integer(20)?;
    
    // Create types
    let mut types = HashMap::new();
    
    let int_type = TypeDefinition {
        id: 0,
        name: "Integer".to_string(),
        kind: TypeKind::Primitive(PrimitiveType::Integer),
        domain: None,
        business_rules: Vec::new(),
        validation_predicates: Vec::new(),
    };
    types.insert(0, int_type);

    // Create add function: adds two numbers
    let add_function = FunctionDefinition {
        id: 0,
        name: "main".to_string(),
        type_id: 0, // Returns Integer
        param_count: 0,
        local_count: 0,
        max_stack_depth: 3,
        capabilities: Vec::new(),
        effects: Vec::new(),
        instructions: vec![
            // Load first number (10)
            Instruction::new(PrismOpcode::LOAD_CONST(const_10)),
            
            // Load second number (20)
            Instruction::new(PrismOpcode::LOAD_CONST(const_20)),
            
            // Add them
            Instruction::new(PrismOpcode::ADD),
            
            // Result (30) is now on top of stack
            Instruction::new(PrismOpcode::RETURN),
        ],
        debug_info: None,
        ai_metadata: None,
    };

    let mut functions = HashMap::new();
    functions.insert("main".to_string(), add_function);

    let bytecode = PrismBytecode {
        version: prism_vm::bytecode::BytecodeVersion::V1_0,
        module_name: "arithmetic".to_string(),
        constants,
        types,
        functions,
        metadata: prism_vm::bytecode::ModuleMetadata {
            source_file: Some("arithmetic.prsm".to_string()),
            compilation_timestamp: std::time::SystemTime::now(),
            compiler_version: "0.1.0".to_string(),
            optimization_level: 0,
            debug_symbols: true,
            dependencies: Vec::new(),
            exports: Vec::new(),
            ai_metadata: None,
        },
    };

    info!("✅ Arithmetic bytecode created successfully");
    Ok(bytecode)
} 