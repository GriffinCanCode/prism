//! Bytecode Interpreter
//!
//! This module implements the core bytecode interpreter for the Prism VM,
//! executing instructions while enforcing capabilities and tracking effects.

use crate::{VMResult, PrismVMError, bytecode::*};
use crate::execution::{ExecutionStack, StackValue, StackFrame, ExecutionResult, ExecutionStats, VMEffectEnforcer};
use prism_runtime::authority::capability::CapabilitySet;
use prism_pir::{Effect, Capability};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration};
use tracing::{debug, info, span, Level, trace};

/// Interpreter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpreterConfig {
    /// Maximum stack size
    pub max_stack_size: usize,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable debugging
    pub enable_debugging: bool,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            max_stack_size: 1024 * 1024, // 1MB stack
            enable_profiling: false,
            enable_debugging: true,
            max_execution_time: Some(Duration::from_secs(30)),
        }
    }
}

/// Bytecode interpreter
#[derive(Debug)]
pub struct Interpreter {
    /// Configuration
    config: InterpreterConfig,
    /// Execution stack
    stack: ExecutionStack,
    /// Instruction pointer
    instruction_pointer: u32,
    /// Current function being executed
    current_function: Option<u32>,
    /// Execution statistics
    stats: ExecutionStats,
    /// Available capabilities
    capabilities: CapabilitySet,
    /// Active effects
    active_effects: Vec<Effect>,
    /// Effect enforcer for runtime validation
    effect_enforcer: Option<Arc<RwLock<VMEffectEnforcer>>>,
}

impl Interpreter {
    /// Create a new interpreter with configuration
    pub fn new(config: InterpreterConfig) -> VMResult<Self> {
        let stack = ExecutionStack::with_max_size(config.max_stack_size);
        
        Ok(Self {
            config,
            stack,
            instruction_pointer: 0,
            current_function: None,
            stats: ExecutionStats::default(),
            capabilities: CapabilitySet::new(),
            active_effects: Vec::new(),
            effect_enforcer: None,
        })
    }

    /// Set the effect enforcer for runtime validation
    pub fn set_effect_enforcer(&mut self, enforcer: Arc<RwLock<VMEffectEnforcer>>) {
        self.effect_enforcer = Some(enforcer);
    }

    /// Execute a function with given arguments
    pub fn execute_function(
        &mut self,
        bytecode: &PrismBytecode,
        function: &FunctionDefinition,
        args: Vec<StackValue>,
    ) -> VMResult<ExecutionResult> {
        let _span = span!(Level::INFO, "execute_function", function = %function.name).entered();
        let start_time = Instant::now();

        info!("Executing function: {}", function.name);

        // Validate argument count
        if args.len() != function.param_count as usize {
            return Err(PrismVMError::ExecutionError {
                message: format!(
                    "Argument count mismatch: expected {}, got {}",
                    function.param_count, args.len()
                ),
            });
        }

        // Check capabilities
        for required_cap in &function.capabilities {
            if !self.capabilities.has_capability(required_cap) {
                return Err(PrismVMError::CapabilityViolation {
                    message: format!("Missing required capability: {:?}", required_cap),
                });
            }
        }

        // Create effect context if effect enforcer is available
        if let Some(ref effect_enforcer) = self.effect_enforcer {
            let mut enforcer = effect_enforcer.write().unwrap();
            enforcer.create_context(function.id, function, self.capabilities.clone())?;
        }

        // Create new stack frame
        let frame = StackFrame::new(
            function.name.clone(),
            function.id,
            0, // Return address (not used for main function)
        );

        // Push frame and arguments
        self.stack.push_frame(frame)?;
        
        // Push arguments as local variables
        for (i, arg) in args.into_iter().enumerate() {
            if let Ok(frame) = self.stack.current_frame_mut() {
                frame.add_local(arg);
            }
        }

        // Initialize local variables (beyond parameters)
        for _ in function.param_count..function.local_count {
            if let Ok(frame) = self.stack.current_frame_mut() {
                frame.add_local(StackValue::Null);
            }
        }

        // Set up execution state
        self.instruction_pointer = 0;
        self.current_function = Some(function.id);
        self.active_effects.extend(function.effects.clone());

        // Execute instructions
        let result = self.execute_instructions(bytecode, function);

        // Calculate execution time
        let execution_time = start_time.elapsed();
        self.stats.execution_time_us = execution_time.as_micros() as u64;

        // Clean up effect context
        if let Some(ref effect_enforcer) = self.effect_enforcer {
            let mut enforcer = effect_enforcer.write().unwrap();
            enforcer.destroy_context(function.id)?;
        }

        // Clean up
        self.stack.pop_frame()?;
        self.active_effects.clear();

        match result {
            Ok(return_value) => {
                info!("Function execution completed successfully in {:?}", execution_time);
                Ok(ExecutionResult {
                    return_value,
                    stats: self.stats.clone(),
                    success: true,
                })
            }
            Err(e) => {
                info!("Function execution failed: {}", e);
                Ok(ExecutionResult {
                    return_value: None,
                    stats: self.stats.clone(),
                    success: false,
                })
            }
        }
    }

    /// Execute instructions in a function
    fn execute_instructions(
        &mut self,
        bytecode: &PrismBytecode,
        function: &FunctionDefinition,
    ) -> VMResult<Option<StackValue>> {
        let instructions = &function.instructions;
        
        while (self.instruction_pointer as usize) < instructions.len() {
            let instruction = &instructions[self.instruction_pointer as usize];
            
            // Check execution timeout
            if let Some(max_time) = self.config.max_execution_time {
                if self.stats.execution_time_us > max_time.as_micros() as u64 {
                    return Err(PrismVMError::ExecutionError {
                        message: "Execution timeout exceeded".to_string(),
                    });
                }
            }

            // Execute instruction
            let should_continue = self.execute_instruction(bytecode, instruction)?;
            
            if !should_continue {
                break;
            }

            self.instruction_pointer += 1;
            self.stats.instructions_executed += 1;
        }

        // Return top of stack if available
        if self.stack.size() > 0 {
            Ok(Some(self.stack.pop()?))
        } else {
            Ok(None)
        }
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        bytecode: &PrismBytecode,
        instruction: &Instruction,
    ) -> VMResult<bool> {
        use instructions::PrismOpcode;

        trace!("Executing instruction: {:?}", instruction.opcode);

        // ENHANCED: Runtime capability verification for each instruction
        // This ensures runtime verification matches compile-time analysis
        for required_cap in &instruction.required_capabilities {
            if !self.capabilities.has_capability(required_cap) {
                return Err(PrismVMError::CapabilityViolation {
                    message: format!("Instruction {:?} requires capability: {:?}", instruction.opcode, required_cap),
                });
            }
        }

        // ENHANCED: Verify effect-related capabilities if effects are present
        for effect in &instruction.effects {
            // Check if we have the capability to perform this effect
            if let Some(required_capability) = effect.required_capability() {
                if !self.capabilities.has_capability(&required_capability) {
                    return Err(PrismVMError::CapabilityViolation {
                        message: format!("Effect {:?} requires capability: {:?}", effect, required_capability),
                    });
                }
            }
        }

        // Track effects
        self.active_effects.extend(instruction.effects.clone());

        match instruction.opcode {
            // Stack Operations
            PrismOpcode::NOP => {
                // No operation
            }
            PrismOpcode::DUP => {
                self.stack.dup()?;
            }
            PrismOpcode::POP => {
                self.stack.pop()?;
            }
            PrismOpcode::SWAP => {
                self.stack.swap()?;
            }
            PrismOpcode::ROT3 => {
                self.stack.rot3()?;
            }

            // Constants and Literals
            PrismOpcode::LOAD_CONST(index) => {
                let constant = bytecode.constants.get(index)
                    .ok_or_else(|| PrismVMError::ExecutionError {
                        message: format!("Invalid constant index: {}", index),
                    })?;
                let value = StackValue::from_constant(constant);
                self.stack.push(value)?;
            }
            PrismOpcode::LOAD_NULL => {
                self.stack.push(StackValue::Null)?;
            }
            PrismOpcode::LOAD_TRUE => {
                self.stack.push(StackValue::Boolean(true))?;
            }
            PrismOpcode::LOAD_FALSE => {
                self.stack.push(StackValue::Boolean(false))?;
            }
            PrismOpcode::LOAD_SMALL_INT(value) => {
                self.stack.push(StackValue::Integer(value as i64))?;
            }
            PrismOpcode::LOAD_ZERO => {
                self.stack.push(StackValue::Integer(0))?;
            }
            PrismOpcode::LOAD_ONE => {
                self.stack.push(StackValue::Integer(1))?;
            }

            // Local Variables
            PrismOpcode::LOAD_LOCAL(slot) => {
                let value = self.stack.get_local(slot)?.clone();
                self.stack.push(value)?;
            }
            PrismOpcode::STORE_LOCAL(slot) => {
                let value = self.stack.pop()?;
                self.stack.set_local(slot, value)?;
            }
            PrismOpcode::LOAD_UPVALUE(slot) => {
                let value = self.stack.get_upvalue(slot)?.clone();
                self.stack.push(value)?;
            }
            PrismOpcode::STORE_UPVALUE(slot) => {
                let value = self.stack.pop()?;
                self.stack.set_upvalue(slot, value)?;
            }

            // Arithmetic Operations
            PrismOpcode::ADD => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = self.add_values(a, b)?;
                self.stack.push(result)?;
            }
            PrismOpcode::SUB => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = self.sub_values(a, b)?;
                self.stack.push(result)?;
            }
            PrismOpcode::MUL => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = self.mul_values(a, b)?;
                self.stack.push(result)?;
            }
            PrismOpcode::DIV => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = self.div_values(a, b)?;
                self.stack.push(result)?;
            }
            PrismOpcode::NEG => {
                let a = self.stack.pop()?;
                let result = self.neg_value(a)?;
                self.stack.push(result)?;
            }

            // Comparison Operations
            PrismOpcode::EQ => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = StackValue::Boolean(self.values_equal(&a, &b));
                self.stack.push(result)?;
            }
            PrismOpcode::NE => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = StackValue::Boolean(!self.values_equal(&a, &b));
                self.stack.push(result)?;
            }
            PrismOpcode::LT => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = StackValue::Boolean(self.compare_values(&a, &b)? < 0);
                self.stack.push(result)?;
            }
            PrismOpcode::LE => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = StackValue::Boolean(self.compare_values(&a, &b)? <= 0);
                self.stack.push(result)?;
            }
            PrismOpcode::GT => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = StackValue::Boolean(self.compare_values(&a, &b)? > 0);
                self.stack.push(result)?;
            }
            PrismOpcode::GE => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = StackValue::Boolean(self.compare_values(&a, &b)? >= 0);
                self.stack.push(result)?;
            }

            // Control Flow
            PrismOpcode::JUMP(offset) => {
                self.instruction_pointer = (self.instruction_pointer as i32 + offset as i32) as u32;
                return Ok(true); // Continue execution, but don't increment IP
            }
            PrismOpcode::JUMP_IF_TRUE(offset) => {
                let condition = self.stack.pop()?;
                if condition.is_truthy() {
                    self.instruction_pointer = (self.instruction_pointer as i32 + offset as i32) as u32;
                    return Ok(true);
                }
            }
            PrismOpcode::JUMP_IF_FALSE(offset) => {
                let condition = self.stack.pop()?;
                if !condition.is_truthy() {
                    self.instruction_pointer = (self.instruction_pointer as i32 + offset as i32) as u32;
                    return Ok(true);
                }
            }
            PrismOpcode::RETURN => {
                return Ok(false); // Stop execution
            }
            PrismOpcode::RETURN_VALUE => {
                return Ok(false); // Stop execution, value is on stack
            }

            // Array Operations
            PrismOpcode::NEW_ARRAY(size) => {
                let mut elements = Vec::new();
                for _ in 0..size {
                    elements.push(self.stack.pop()?);
                }
                elements.reverse(); // Restore original order
                self.stack.push(StackValue::Array(elements))?;
            }
            PrismOpcode::GET_INDEX => {
                let index = self.stack.pop()?;
                let array = self.stack.pop()?;
                let result = self.get_array_element(array, index)?;
                self.stack.push(result)?;
            }
            PrismOpcode::SET_INDEX => {
                let value = self.stack.pop()?;
                let index = self.stack.pop()?;
                let mut array = self.stack.pop()?;
                self.set_array_element(&mut array, index, value)?;
                self.stack.push(array)?;
            }
            PrismOpcode::ARRAY_LEN => {
                let array = self.stack.pop()?;
                let length = self.get_array_length(array)?;
                self.stack.push(StackValue::Integer(length))?;
            }

            // String Operations
            PrismOpcode::STR_CONCAT => {
                let b = self.stack.pop()?;
                let a = self.stack.pop()?;
                let result = self.concat_strings(a, b)?;
                self.stack.push(result)?;
            }

            // Type Operations
            PrismOpcode::GET_TYPE => {
                let value = self.stack.pop()?;
                let type_name = StackValue::String(value.type_name().to_string());
                self.stack.push(type_name)?;
            }
            PrismOpcode::IS_NULL => {
                let value = self.stack.pop()?;
                let result = StackValue::Boolean(matches!(value, StackValue::Null));
                self.stack.push(result)?;
            }

            // Effect Operations
            PrismOpcode::EFFECT_ENTER(effect_id) => {
                if let Some(ref effect_enforcer) = self.effect_enforcer {
                    let function_id = self.current_function.ok_or_else(|| PrismVMError::ExecutionError {
                        message: "EFFECT_ENTER called outside function context".to_string(),
                    })?;
                    
                    let mut enforcer = effect_enforcer.write().unwrap();
                    enforcer.enter_effect(function_id, effect_id, &mut self.stack)?;
                } else {
                    return Err(PrismVMError::EffectError {
                        message: "Effect enforcer not available for EFFECT_ENTER".to_string(),
                    });
                }
            }
            PrismOpcode::EFFECT_EXIT => {
                if let Some(ref effect_enforcer) = self.effect_enforcer {
                    let function_id = self.current_function.ok_or_else(|| PrismVMError::ExecutionError {
                        message: "EFFECT_EXIT called outside function context".to_string(),
                    })?;
                    
                    let mut enforcer = effect_enforcer.write().unwrap();
                    enforcer.exit_effect(function_id, &mut self.stack)?;
                } else {
                    return Err(PrismVMError::EffectError {
                        message: "Effect enforcer not available for EFFECT_EXIT".to_string(),
                    });
                }
            }
            PrismOpcode::EFFECT_INVOKE(effect_id) => {
                if let Some(ref effect_enforcer) = self.effect_enforcer {
                    let function_id = self.current_function.ok_or_else(|| PrismVMError::ExecutionError {
                        message: "EFFECT_INVOKE called outside function context".to_string(),
                    })?;
                    
                    let mut enforcer = effect_enforcer.write().unwrap();
                    enforcer.invoke_effect(function_id, effect_id, &mut self.stack)?;
                } else {
                    return Err(PrismVMError::EffectError {
                        message: "Effect enforcer not available for EFFECT_INVOKE".to_string(),
                    });
                }
            }
            PrismOpcode::EFFECT_HANDLE(handler_id) => {
                if let Some(ref effect_enforcer) = self.effect_enforcer {
                    let function_id = self.current_function.ok_or_else(|| PrismVMError::ExecutionError {
                        message: "EFFECT_HANDLE called outside function context".to_string(),
                    })?;
                    
                    let mut enforcer = effect_enforcer.write().unwrap();
                    enforcer.handle_effect(function_id, handler_id, &mut self.stack)?;
                } else {
                    return Err(PrismVMError::EffectError {
                        message: "Effect enforcer not available for EFFECT_HANDLE".to_string(),
                    });
                }
            }
            PrismOpcode::EFFECT_RESUME => {
                // Effect resume would integrate with the effect system's continuation mechanism
                debug!("EFFECT_RESUME instruction executed");
                // TODO: Implement effect resumption logic
            }
            PrismOpcode::EFFECT_ABORT => {
                // Effect abort would clean up the current effect context
                debug!("EFFECT_ABORT instruction executed");
                if let Some(ref effect_enforcer) = self.effect_enforcer {
                    let function_id = self.current_function.ok_or_else(|| PrismVMError::ExecutionError {
                        message: "EFFECT_ABORT called outside function context".to_string(),
                    })?;
                    
                    let mut enforcer = effect_enforcer.write().unwrap();
                    // Force exit all active effects
                    while let Ok(()) = enforcer.exit_effect(function_id, &mut self.stack) {
                        // Continue exiting effects until none are active
                    }
                }
            }

            // Capability Operations
            PrismOpcode::CAP_CHECK(cap_id) => {
                // Check if a specific capability is available
                // This would integrate with the capability system
                debug!("Checking capability {}", cap_id);
                // For now, push true if we have any capabilities
                let has_capability = !self.capabilities.is_empty();
                self.stack.push(StackValue::Boolean(has_capability))?;
            }

            // Debugging Operations
            PrismOpcode::BREAKPOINT => {
                if self.config.enable_debugging {
                    debug!("Breakpoint hit at instruction {}", self.instruction_pointer);
                    // In a real debugger, this would pause execution
                }
            }

            // Placeholder for unimplemented instructions
            _ => {
                return Err(PrismVMError::ExecutionError {
                    message: format!("Unimplemented instruction: {:?}", instruction.opcode),
                });
            }
        }

        Ok(true) // Continue execution
    }

    /// Add two values
    fn add_values(&self, a: StackValue, b: StackValue) -> VMResult<StackValue> {
        match (a, b) {
            (StackValue::Integer(a), StackValue::Integer(b)) => Ok(StackValue::Integer(a + b)),
            (StackValue::Float(a), StackValue::Float(b)) => Ok(StackValue::Float(a + b)),
            (StackValue::Integer(a), StackValue::Float(b)) => Ok(StackValue::Float(a as f64 + b)),
            (StackValue::Float(a), StackValue::Integer(b)) => Ok(StackValue::Float(a + b as f64)),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid types for addition".to_string(),
            }),
        }
    }

    /// Subtract two values
    fn sub_values(&self, a: StackValue, b: StackValue) -> VMResult<StackValue> {
        match (a, b) {
            (StackValue::Integer(a), StackValue::Integer(b)) => Ok(StackValue::Integer(a - b)),
            (StackValue::Float(a), StackValue::Float(b)) => Ok(StackValue::Float(a - b)),
            (StackValue::Integer(a), StackValue::Float(b)) => Ok(StackValue::Float(a as f64 - b)),
            (StackValue::Float(a), StackValue::Integer(b)) => Ok(StackValue::Float(a - b as f64)),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid types for subtraction".to_string(),
            }),
        }
    }

    /// Multiply two values
    fn mul_values(&self, a: StackValue, b: StackValue) -> VMResult<StackValue> {
        match (a, b) {
            (StackValue::Integer(a), StackValue::Integer(b)) => Ok(StackValue::Integer(a * b)),
            (StackValue::Float(a), StackValue::Float(b)) => Ok(StackValue::Float(a * b)),
            (StackValue::Integer(a), StackValue::Float(b)) => Ok(StackValue::Float(a as f64 * b)),
            (StackValue::Float(a), StackValue::Integer(b)) => Ok(StackValue::Float(a * b as f64)),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid types for multiplication".to_string(),
            }),
        }
    }

    /// Divide two values
    fn div_values(&self, a: StackValue, b: StackValue) -> VMResult<StackValue> {
        match (a, b) {
            (StackValue::Integer(a), StackValue::Integer(b)) => {
                if b == 0 {
                    return Err(PrismVMError::ExecutionError {
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(StackValue::Integer(a / b))
            }
            (StackValue::Float(a), StackValue::Float(b)) => {
                if b == 0.0 {
                    return Err(PrismVMError::ExecutionError {
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(StackValue::Float(a / b))
            }
            (StackValue::Integer(a), StackValue::Float(b)) => {
                if b == 0.0 {
                    return Err(PrismVMError::ExecutionError {
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(StackValue::Float(a as f64 / b))
            }
            (StackValue::Float(a), StackValue::Integer(b)) => {
                if b == 0 {
                    return Err(PrismVMError::ExecutionError {
                        message: "Division by zero".to_string(),
                    });
                }
                Ok(StackValue::Float(a / b as f64))
            }
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid types for division".to_string(),
            }),
        }
    }

    /// Negate a value
    fn neg_value(&self, a: StackValue) -> VMResult<StackValue> {
        match a {
            StackValue::Integer(a) => Ok(StackValue::Integer(-a)),
            StackValue::Float(a) => Ok(StackValue::Float(-a)),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid type for negation".to_string(),
            }),
        }
    }

    /// Check if two values are equal
    fn values_equal(&self, a: &StackValue, b: &StackValue) -> bool {
        match (a, b) {
            (StackValue::Null, StackValue::Null) => true,
            (StackValue::Boolean(a), StackValue::Boolean(b)) => a == b,
            (StackValue::Integer(a), StackValue::Integer(b)) => a == b,
            (StackValue::Float(a), StackValue::Float(b)) => a == b,
            (StackValue::String(a), StackValue::String(b)) => a == b,
            _ => false,
        }
    }

    /// Compare two values (-1, 0, 1)
    fn compare_values(&self, a: &StackValue, b: &StackValue) -> VMResult<i32> {
        match (a, b) {
            (StackValue::Integer(a), StackValue::Integer(b)) => Ok(a.cmp(b) as i32),
            (StackValue::Float(a), StackValue::Float(b)) => Ok(a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32),
            (StackValue::String(a), StackValue::String(b)) => Ok(a.cmp(b) as i32),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid types for comparison".to_string(),
            }),
        }
    }

    /// Get array element
    fn get_array_element(&self, array: StackValue, index: StackValue) -> VMResult<StackValue> {
        match (array, index) {
            (StackValue::Array(arr), StackValue::Integer(idx)) => {
                if idx < 0 || idx as usize >= arr.len() {
                    return Err(PrismVMError::ExecutionError {
                        message: "Array index out of bounds".to_string(),
                    });
                }
                Ok(arr[idx as usize].clone())
            }
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid array access".to_string(),
            }),
        }
    }

    /// Set array element
    fn set_array_element(&self, array: &mut StackValue, index: StackValue, value: StackValue) -> VMResult<()> {
        match (array, index) {
            (StackValue::Array(arr), StackValue::Integer(idx)) => {
                if idx < 0 || idx as usize >= arr.len() {
                    return Err(PrismVMError::ExecutionError {
                        message: "Array index out of bounds".to_string(),
                    });
                }
                arr[idx as usize] = value;
                Ok(())
            }
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid array assignment".to_string(),
            }),
        }
    }

    /// Get array length
    fn get_array_length(&self, array: StackValue) -> VMResult<i64> {
        match array {
            StackValue::Array(arr) => Ok(arr.len() as i64),
            StackValue::String(s) => Ok(s.len() as i64),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid type for length operation".to_string(),
            }),
        }
    }

    /// Concatenate strings
    fn concat_strings(&self, a: StackValue, b: StackValue) -> VMResult<StackValue> {
        match (a, b) {
            (StackValue::String(a), StackValue::String(b)) => Ok(StackValue::String(a + &b)),
            _ => Err(PrismVMError::ExecutionError {
                message: "Invalid types for string concatenation".to_string(),
            }),
        }
    }

    /// Set capabilities for execution
    pub fn set_capabilities(&mut self, capabilities: CapabilitySet) {
        self.capabilities = capabilities;
    }

    /// Get current execution statistics
    pub fn get_stats(&self) -> InterpreterStats {
        InterpreterStats {
            instructions_executed: self.stats.instructions_executed,
            execution_time_us: self.stats.execution_time_us,
            stack_depth: self.stack.size(),
            memory_usage: self.stack.size() * std::mem::size_of::<StackValue>(),
        }
    }

    /// Shutdown the interpreter
    pub fn shutdown(self) -> VMResult<()> {
        debug!("Shutting down interpreter");
        Ok(())
    }
}

/// Interpreter statistics
#[derive(Debug, Clone)]
pub struct InterpreterStats {
    /// Instructions executed
    pub instructions_executed: u64,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Current stack depth
    pub stack_depth: usize,
    /// Memory usage in bytes
    pub memory_usage: usize,
} 