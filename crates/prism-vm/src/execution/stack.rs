//! Execution Stack
//!
//! This module implements the execution stack for the Prism VM,
//! including stack frames and value management.

use crate::{VMResult, PrismVMError, bytecode::constants::Constant};
use prism_pir::{Effect, Capability};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Maximum stack size (configurable)
pub const DEFAULT_MAX_STACK_SIZE: usize = 1024 * 1024; // 1MB

/// Execution stack for the VM
#[derive(Debug)]
pub struct ExecutionStack {
    /// Stack frames
    frames: Vec<StackFrame>,
    /// Value stack
    values: Vec<StackValue>,
    /// Maximum stack size
    max_size: usize,
    /// Current stack depth
    depth: usize,
}

/// Stack frame representing a function call
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function name
    pub function_name: String,
    /// Function ID
    pub function_id: u32,
    /// Return address (instruction pointer)
    pub return_address: u32,
    /// Base pointer for locals
    pub base_pointer: usize,
    /// Local variables
    pub locals: SmallVec<[StackValue; 8]>,
    /// Upvalues (for closures)
    pub upvalues: SmallVec<[StackValue; 4]>,
    /// Exception handlers active in this frame
    pub exception_handlers: Vec<ExceptionHandler>,
    /// Effects active in this frame
    pub active_effects: Vec<Effect>,
    /// Capabilities available in this frame
    pub capabilities: Vec<Capability>,
}

/// Exception handler information
#[derive(Debug, Clone)]
pub struct ExceptionHandler {
    /// Start instruction offset
    pub start_offset: u32,
    /// End instruction offset
    pub end_offset: u32,
    /// Handler instruction offset
    pub handler_offset: u32,
    /// Exception type (None for catch-all)
    pub exception_type: Option<String>,
}

/// Stack value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StackValue {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Byte array
    Bytes(Vec<u8>),
    /// Array of values
    Array(Vec<StackValue>),
    /// Object with fields
    Object(HashMap<String, StackValue>),
    /// Function reference
    Function {
        /// Function ID
        id: u32,
        /// Captured upvalues (for closures)
        upvalues: Vec<StackValue>,
    },
    /// Type reference
    Type(u32),
    /// Capability token
    Capability(String),
    /// Effect handle
    Effect(String),
}

impl ExecutionStack {
    /// Create a new execution stack
    pub fn new() -> Self {
        Self::with_max_size(DEFAULT_MAX_STACK_SIZE)
    }

    /// Create a new execution stack with maximum size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            frames: Vec::new(),
            values: Vec::new(),
            max_size,
            depth: 0,
        }
    }

    /// Push a value onto the stack
    pub fn push(&mut self, value: StackValue) -> VMResult<()> {
        if self.values.len() >= self.max_size {
            return Err(PrismVMError::RuntimeError {
                message: "Stack overflow".to_string(),
            });
        }
        
        self.values.push(value);
        self.depth = self.depth.max(self.values.len());
        Ok(())
    }

    /// Pop a value from the stack
    pub fn pop(&mut self) -> VMResult<StackValue> {
        self.values.pop().ok_or_else(|| PrismVMError::RuntimeError {
            message: "Stack underflow".to_string(),
        })
    }

    /// Peek at the top value without removing it
    pub fn peek(&self) -> VMResult<&StackValue> {
        self.values.last().ok_or_else(|| PrismVMError::RuntimeError {
            message: "Stack is empty".to_string(),
        })
    }

    /// Peek at a value at depth n from the top
    pub fn peek_at(&self, depth: usize) -> VMResult<&StackValue> {
        if depth >= self.values.len() {
            return Err(PrismVMError::RuntimeError {
                message: format!("Invalid stack depth: {}", depth),
            });
        }
        let index = self.values.len() - 1 - depth;
        Ok(&self.values[index])
    }

    /// Duplicate the top value
    pub fn dup(&mut self) -> VMResult<()> {
        let value = self.peek()?.clone();
        self.push(value)
    }

    /// Swap the top two values
    pub fn swap(&mut self) -> VMResult<()> {
        if self.values.len() < 2 {
            return Err(PrismVMError::RuntimeError {
                message: "Not enough values to swap".to_string(),
            });
        }
        let len = self.values.len();
        self.values.swap(len - 1, len - 2);
        Ok(())
    }

    /// Rotate top three values (top -> third)
    pub fn rot3(&mut self) -> VMResult<()> {
        if self.values.len() < 3 {
            return Err(PrismVMError::RuntimeError {
                message: "Not enough values to rotate".to_string(),
            });
        }
        let len = self.values.len();
        // Move top to third position: [a, b, c] -> [c, a, b]
        let top = self.values.remove(len - 1);
        self.values.insert(len - 3, top);
        Ok(())
    }

    /// Push a new stack frame
    pub fn push_frame(&mut self, frame: StackFrame) -> VMResult<()> {
        self.frames.push(frame);
        Ok(())
    }

    /// Pop the current stack frame
    pub fn pop_frame(&mut self) -> VMResult<StackFrame> {
        self.frames.pop().ok_or_else(|| PrismVMError::RuntimeError {
            message: "No frame to pop".to_string(),
        })
    }

    /// Get the current stack frame
    pub fn current_frame(&self) -> VMResult<&StackFrame> {
        self.frames.last().ok_or_else(|| PrismVMError::RuntimeError {
            message: "No current frame".to_string(),
        })
    }

    /// Get the current stack frame mutably
    pub fn current_frame_mut(&mut self) -> VMResult<&mut StackFrame> {
        self.frames.last_mut().ok_or_else(|| PrismVMError::RuntimeError {
            message: "No current frame".to_string(),
        })
    }

    /// Get local variable from current frame
    pub fn get_local(&self, slot: u8) -> VMResult<&StackValue> {
        let frame = self.frames.last()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for local variable access".to_string(),
            })?;
        
        frame.locals.get(slot as usize)
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: format!("Invalid local variable slot: {}", slot),
            })
    }

    /// Set local variable in current frame
    pub fn set_local(&mut self, slot: u8, value: StackValue) -> VMResult<()> {
        let frame = self.frames.last_mut()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for local variable access".to_string(),
            })?;
        
        // Extend locals if necessary
        while frame.locals.len() <= slot as usize {
            frame.locals.push(StackValue::Null);
        }
        
        frame.locals[slot as usize] = value;
        Ok(())
    }

    /// Get upvalue from current frame
    pub fn get_upvalue(&self, slot: u8) -> VMResult<&StackValue> {
        let frame = self.frames.last()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for upvalue access".to_string(),
            })?;
        
        frame.upvalues.get(slot as usize)
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: format!("Invalid upvalue slot: {}", slot),
            })
    }

    /// Set upvalue in current frame
    pub fn set_upvalue(&mut self, slot: u8, value: StackValue) -> VMResult<()> {
        let frame = self.frames.last_mut()
            .ok_or_else(|| PrismVMError::RuntimeError {
                message: "No active frame for upvalue access".to_string(),
            })?;
        
        // Extend upvalues if necessary
        while frame.upvalues.len() <= slot as usize {
            frame.upvalues.push(StackValue::Null);
        }
        
        frame.upvalues[slot as usize] = value;
        Ok(())
    }

    /// Get the current stack size
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Get the maximum stack depth reached
    pub fn max_depth(&self) -> usize {
        self.depth
    }

    /// Get the number of frames
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Clear the stack
    pub fn clear(&mut self) {
        self.values.clear();
        self.frames.clear();
        self.depth = 0;
    }

    /// Check if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get stack statistics
    pub fn statistics(&self) -> StackStatistics {
        StackStatistics {
            current_size: self.values.len(),
            max_depth: self.depth,
            frame_count: self.frames.len(),
            max_size: self.max_size,
        }
    }
}

impl Default for ExecutionStack {
    fn default() -> Self {
        Self::new()
    }
}

impl StackFrame {
    /// Create a new stack frame
    pub fn new(function_name: String, function_id: u32, return_address: u32) -> Self {
        Self {
            function_name,
            function_id,
            return_address,
            base_pointer: 0,
            locals: SmallVec::new(),
            upvalues: SmallVec::new(),
            exception_handlers: Vec::new(),
            active_effects: Vec::new(),
            capabilities: Vec::new(),
        }
    }

    /// Add a local variable
    pub fn add_local(&mut self, value: StackValue) {
        self.locals.push(value);
    }

    /// Add an upvalue
    pub fn add_upvalue(&mut self, value: StackValue) {
        self.upvalues.push(value);
    }

    /// Add an exception handler
    pub fn add_exception_handler(&mut self, handler: ExceptionHandler) {
        self.exception_handlers.push(handler);
    }

    /// Add an active effect
    pub fn add_effect(&mut self, effect: Effect) {
        self.active_effects.push(effect);
    }

    /// Add a capability
    pub fn add_capability(&mut self, capability: Capability) {
        self.capabilities.push(capability);
    }
}

impl StackValue {
    /// Check if this value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            StackValue::Null => false,
            StackValue::Boolean(b) => *b,
            StackValue::Integer(i) => *i != 0,
            StackValue::Float(f) => *f != 0.0 && !f.is_nan(),
            StackValue::String(s) => !s.is_empty(),
            StackValue::Bytes(b) => !b.is_empty(),
            StackValue::Array(a) => !a.is_empty(),
            StackValue::Object(o) => !o.is_empty(),
            StackValue::Function { .. } => true,
            StackValue::Type(_) => true,
            StackValue::Capability(_) => true,
            StackValue::Effect(_) => true,
        }
    }

    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            StackValue::Null => "null",
            StackValue::Boolean(_) => "boolean",
            StackValue::Integer(_) => "integer",
            StackValue::Float(_) => "float",
            StackValue::String(_) => "string",
            StackValue::Bytes(_) => "bytes",
            StackValue::Array(_) => "array",
            StackValue::Object(_) => "object",
            StackValue::Function { .. } => "function",
            StackValue::Type(_) => "type",
            StackValue::Capability(_) => "capability",
            StackValue::Effect(_) => "effect",
        }
    }

    /// Convert from a constant
    pub fn from_constant(constant: &Constant) -> Self {
        match constant {
            Constant::Null => StackValue::Null,
            Constant::Boolean(b) => StackValue::Boolean(*b),
            Constant::Integer(i) => StackValue::Integer(*i),
            Constant::Float(f) => StackValue::Float(*f),
            Constant::String(s) => StackValue::String(s.clone()),
            Constant::Bytes(b) => StackValue::Bytes(b.clone()),
            Constant::Type(id) => StackValue::Type(*id),
            Constant::Function(id) => StackValue::Function {
                id: *id,
                upvalues: Vec::new(),
            },
            Constant::Composite(_) => {
                // For composite constants, we'd need to recursively convert
                // For now, just return null
                StackValue::Null
            }
        }
    }
}

/// Stack statistics
#[derive(Debug, Clone)]
pub struct StackStatistics {
    /// Current stack size
    pub current_size: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Number of frames
    pub frame_count: usize,
    /// Maximum stack size
    pub max_size: usize,
} 