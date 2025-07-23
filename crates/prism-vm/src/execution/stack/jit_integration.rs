//! JIT Stack Integration
//!
//! This module provides integration between the stack management system and the JIT compiler,
//! enabling efficient stack frame management for compiled code while maintaining semantic
//! preservation and capability-based security.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};

/// JIT stack manager for compiled code integration
#[derive(Debug)]
pub struct JITStackManager {
    /// Compiled frame mappings
    frame_mappings: Arc<RwLock<HashMap<u32, CompiledFrameMapping>>>,
    
    /// Stack reconstruction info for deoptimization
    reconstruction_info: Arc<RwLock<HashMap<u32, StackReconstructionInfo>>>,
    
    /// JIT integration statistics
    stats: Arc<RwLock<JITStackStats>>,
    
    /// Configuration
    config: JITStackConfig,
}

/// JIT stack configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JITStackConfig {
    /// Enable stack frame pooling for compiled code
    pub enable_frame_pooling: bool,
    
    /// Enable deoptimization support
    pub enable_deoptimization: bool,
    
    /// Maximum compiled frames to track
    pub max_compiled_frames: usize,
    
    /// Enable stack reconstruction caching
    pub enable_reconstruction_cache: bool,
}

impl Default for JITStackConfig {
    fn default() -> Self {
        Self {
            enable_frame_pooling: true,
            enable_deoptimization: true,
            max_compiled_frames: 10000,
            enable_reconstruction_cache: true,
        }
    }
}

/// Mapping between interpreted and compiled stack frames
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledFrameMapping {
    /// Function ID
    pub function_id: u32,
    
    /// Interpreted frame information
    pub interpreted_frame: FrameInfo,
    
    /// Compiled frame information
    pub compiled_frame: CompiledFrameInfo,
    
    /// Variable mappings
    pub variable_mappings: Vec<VariableMapping>,
    
    /// When this mapping was created
    pub created_at: Instant,
}

/// Frame information for interpreted execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameInfo {
    /// Function name
    pub function_name: String,
    
    /// Local variable count
    pub local_count: u8,
    
    /// Stack offset
    pub stack_offset: usize,
    
    /// Capabilities required
    pub capabilities: Vec<String>,
}

/// Frame information for compiled execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledFrameInfo {
    /// Entry point address
    pub entry_point: u64, // Using u64 instead of raw pointer for serialization
    
    /// Frame size in bytes
    pub frame_size: usize,
    
    /// Register usage map
    pub register_usage: HashMap<String, RegisterInfo>,
    
    /// Stack layout
    pub stack_layout: StackLayout,
}

/// Register information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterInfo {
    /// Register name
    pub name: String,
    
    /// Purpose (local variable, temporary, etc.)
    pub purpose: RegisterPurpose,
    
    /// Data type
    pub data_type: String,
}

/// Register purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegisterPurpose {
    /// Local variable storage
    LocalVariable { slot: u8 },
    
    /// Temporary computation value
    Temporary { id: u32 },
    
    /// Function parameter
    Parameter { index: u8 },
    
    /// Return value
    ReturnValue,
    
    /// Capability token
    Capability { name: String },
}

/// Stack layout for compiled frames
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackLayout {
    /// Total frame size
    pub frame_size: usize,
    
    /// Local variable offsets
    pub local_offsets: HashMap<u8, i32>,
    
    /// Spill slot offsets
    pub spill_offsets: HashMap<u32, i32>,
    
    /// Alignment requirements
    pub alignment: usize,
}

/// Variable mapping between interpreted and compiled code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMapping {
    /// Interpreted variable slot
    pub interpreted_slot: u8,
    
    /// Compiled variable location
    pub compiled_location: CompiledLocation,
    
    /// Variable type
    pub variable_type: String,
    
    /// Access frequency (for optimization)
    pub access_frequency: u32,
}

/// Location of variable in compiled code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompiledLocation {
    /// In a register
    Register { name: String },
    
    /// On stack at offset
    Stack { offset: i32 },
    
    /// In memory at address
    Memory { offset: usize },
    
    /// Constant value
    Constant { value: String },
}

/// Stack reconstruction information for deoptimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackReconstructionInfo {
    /// Function ID
    pub function_id: u32,
    
    /// Bytecode instruction pointer
    pub bytecode_ip: u32,
    
    /// Interpreter stack state
    pub stack_state: InterpreterStackState,
    
    /// Local variables to restore
    pub locals_to_restore: Vec<LocalRestore>,
    
    /// Stack values to restore
    pub stack_values_to_restore: Vec<StackValueRestore>,
}

/// Interpreter stack state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpreterStackState {
    /// Stack depth
    pub depth: usize,
    
    /// Frame count
    pub frame_count: usize,
    
    /// Current instruction pointer
    pub instruction_pointer: u32,
}

/// Local variable restoration info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalRestore {
    /// Variable slot
    pub slot: u8,
    
    /// Variable value
    pub value: StackValueRestore,
    
    /// Variable type
    pub var_type: String,
}

/// Stack value restoration info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackValueRestore {
    /// Stack position
    pub position: usize,
    
    /// Value type
    pub value_type: String,
    
    /// Serialized value
    pub serialized_value: Vec<u8>,
}

/// JIT stack statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JITStackStats {
    /// Compiled frames created
    pub compiled_frames_created: u64,
    
    /// Deoptimizations performed
    pub deoptimizations: u64,
    
    /// Stack reconstructions
    pub stack_reconstructions: u64,
    
    /// Average reconstruction time
    pub avg_reconstruction_time_us: f64,
    
    /// Frame mapping cache hits
    pub mapping_cache_hits: u64,
    
    /// Frame mapping cache misses
    pub mapping_cache_misses: u64,
    
    /// Memory usage for JIT stacks
    pub jit_stack_memory_bytes: usize,
}

impl Default for JITStackStats {
    fn default() -> Self {
        Self {
            compiled_frames_created: 0,
            deoptimizations: 0,
            stack_reconstructions: 0,
            avg_reconstruction_time_us: 0.0,
            mapping_cache_hits: 0,
            mapping_cache_misses: 0,
            jit_stack_memory_bytes: 0,
        }
    }
}

impl JITStackManager {
    /// Create a new JIT stack manager
    pub fn new() -> VMResult<Self> {
        let _span = span!(Level::INFO, "jit_stack_init").entered();
        info!("Initializing JIT stack management");

        Ok(Self {
            frame_mappings: Arc::new(RwLock::new(HashMap::new())),
            reconstruction_info: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(JITStackStats::default())),
            config: JITStackConfig::default(),
        })
    }

    /// Create a mapping for a compiled frame
    pub fn create_frame_mapping(
        &self,
        function_id: u32,
        function_name: String,
        compiled_info: CompiledFrameInfo,
        local_count: u8,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "create_frame_mapping", 
            function_id = function_id,
            function = %function_name
        ).entered();

        let interpreted_frame = FrameInfo {
            function_name: function_name.clone(),
            local_count,
            stack_offset: 0, // Will be set during execution
            capabilities: Vec::new(), // Will be populated from function metadata
        };

        // Generate variable mappings
        let variable_mappings = self.generate_variable_mappings(local_count, &compiled_info);

        let mapping = CompiledFrameMapping {
            function_id,
            interpreted_frame,
            compiled_frame: compiled_info,
            variable_mappings,
            created_at: Instant::now(),
        };

        // Store the mapping
        {
            let mut mappings = self.frame_mappings.write().unwrap();
            mappings.insert(function_id, mapping);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.compiled_frames_created += 1;
        }

        debug!("Created frame mapping for function: {}", function_name);
        Ok(())
    }

    /// Generate variable mappings between interpreted and compiled code
    fn generate_variable_mappings(
        &self,
        local_count: u8,
        compiled_info: &CompiledFrameInfo,
    ) -> Vec<VariableMapping> {
        let mut mappings = Vec::new();

        for slot in 0..local_count {
            // Determine the best location for this variable
            let compiled_location = if let Some(offset) = compiled_info.stack_layout.local_offsets.get(&slot) {
                CompiledLocation::Stack { offset: *offset }
            } else {
                // Default to stack location
                CompiledLocation::Stack { offset: slot as i32 * 8 }
            };

            mappings.push(VariableMapping {
                interpreted_slot: slot,
                compiled_location,
                variable_type: "unknown".to_string(), // Would be filled from type information
                access_frequency: 0, // Would be updated during profiling
            });
        }

        mappings
    }

    /// Prepare stack reconstruction information for deoptimization
    pub fn prepare_reconstruction_info(
        &self,
        function_id: u32,
        bytecode_ip: u32,
        stack: &ExecutionStack,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "prepare_reconstruction", 
            function_id = function_id,
            ip = bytecode_ip
        ).entered();

        let stack_state = InterpreterStackState {
            depth: stack.size(),
            frame_count: stack.frame_count(),
            instruction_pointer: bytecode_ip,
        };

        // Capture current local variables
        let locals_to_restore = self.capture_locals(stack)?;
        
        // Capture current stack values
        let stack_values_to_restore = self.capture_stack_values(stack)?;

        let reconstruction_info = StackReconstructionInfo {
            function_id,
            bytecode_ip,
            stack_state,
            locals_to_restore,
            stack_values_to_restore,
        };

        // Store reconstruction info
        {
            let mut info = self.reconstruction_info.write().unwrap();
            info.insert(function_id, reconstruction_info);
        }

        debug!("Prepared reconstruction info for function: {}", function_id);
        Ok(())
    }

    /// Capture local variables for reconstruction
    fn capture_locals(&self, stack: &ExecutionStack) -> VMResult<Vec<LocalRestore>> {
        let mut locals = Vec::new();

        if let Ok(frame) = stack.current_frame() {
            for (slot, value) in frame.locals.iter().enumerate() {
                let value_restore = self.serialize_stack_value(value)?;
                locals.push(LocalRestore {
                    slot: slot as u8,
                    value: value_restore,
                    var_type: value.type_name().to_string(),
                });
            }
        }

        Ok(locals)
    }

    /// Capture stack values for reconstruction
    fn capture_stack_values(&self, _stack: &ExecutionStack) -> VMResult<Vec<StackValueRestore>> {
        // This would capture the current stack values
        // For now, return empty vector
        Ok(Vec::new())
    }

    /// Serialize a stack value for reconstruction
    fn serialize_stack_value(&self, value: &StackValue) -> VMResult<StackValueRestore> {
        // Simple serialization - in practice, this would be more sophisticated
        let serialized = serde_json::to_vec(value).map_err(|e| PrismVMError::RuntimeError {
            message: format!("Failed to serialize stack value: {}", e),
        })?;

        Ok(StackValueRestore {
            position: 0, // Would be filled with actual position
            value_type: value.type_name().to_string(),
            serialized_value: serialized,
        })
    }

    /// Perform deoptimization and reconstruct interpreter stack
    pub fn deoptimize_to_interpreter(
        &self,
        function_id: u32,
        stack: &mut ExecutionStack,
    ) -> VMResult<()> {
        let _span = span!(Level::INFO, "deoptimize", function_id = function_id).entered();
        info!("Performing deoptimization for function: {}", function_id);

        let start_time = Instant::now();

        // Get reconstruction info
        let reconstruction_info = {
            let info = self.reconstruction_info.read().unwrap();
            info.get(&function_id).cloned()
                .ok_or_else(|| PrismVMError::RuntimeError {
                    message: format!("No reconstruction info for function: {}", function_id),
                })?
        };

        // Reconstruct locals
        self.reconstruct_locals(stack, &reconstruction_info.locals_to_restore)?;

        // Reconstruct stack values
        self.reconstruct_stack_values(stack, &reconstruction_info.stack_values_to_restore)?;

        // Update statistics
        let reconstruction_time = start_time.elapsed();
        {
            let mut stats = self.stats.write().unwrap();
            stats.deoptimizations += 1;
            stats.stack_reconstructions += 1;
            
            // Update average reconstruction time
            let new_time_us = reconstruction_time.as_micros() as f64;
            if stats.stack_reconstructions == 1 {
                stats.avg_reconstruction_time_us = new_time_us;
            } else {
                let alpha = 0.1; // Exponential moving average
                stats.avg_reconstruction_time_us = alpha * new_time_us + (1.0 - alpha) * stats.avg_reconstruction_time_us;
            }
        }

        info!("Deoptimization completed in {:?}", reconstruction_time);
        Ok(())
    }

    /// Reconstruct local variables
    fn reconstruct_locals(
        &self,
        stack: &mut ExecutionStack,
        locals: &[LocalRestore],
    ) -> VMResult<()> {
        for local in locals {
            let value = self.deserialize_stack_value(&local.value)?;
            stack.set_local(local.slot, value)?;
        }
        Ok(())
    }

    /// Reconstruct stack values
    fn reconstruct_stack_values(
        &self,
        _stack: &mut ExecutionStack,
        _values: &[StackValueRestore],
    ) -> VMResult<()> {
        // This would reconstruct the stack values
        // For now, just return Ok
        Ok(())
    }

    /// Deserialize a stack value
    fn deserialize_stack_value(&self, value_restore: &StackValueRestore) -> VMResult<StackValue> {
        serde_json::from_slice(&value_restore.serialized_value).map_err(|e| PrismVMError::RuntimeError {
            message: format!("Failed to deserialize stack value: {}", e),
        })
    }

    /// Get frame mapping for a function
    pub fn get_frame_mapping(&self, function_id: u32) -> Option<CompiledFrameMapping> {
        let mappings = self.frame_mappings.read().unwrap();
        let mapping = mappings.get(&function_id).cloned();
        
        // Update cache statistics
        {
            let mut stats = self.stats.write().unwrap();
            if mapping.is_some() {
                stats.mapping_cache_hits += 1;
            } else {
                stats.mapping_cache_misses += 1;
            }
        }
        
        mapping
    }

    /// Get JIT stack statistics
    pub fn stats(&self) -> JITStackStats {
        self.stats.read().unwrap().clone()
    }

    /// Clean up old mappings and reconstruction info
    pub fn cleanup_old_data(&self, max_age: Duration) -> usize {
        let threshold = Instant::now() - max_age;
        let mut cleaned = 0;

        // Clean up frame mappings
        {
            let mut mappings = self.frame_mappings.write().unwrap();
            let initial_count = mappings.len();
            mappings.retain(|_, mapping| mapping.created_at > threshold);
            cleaned += initial_count - mappings.len();
        }

        // Clean up reconstruction info (this would need timestamps too)
        // For now, just clear everything older than threshold
        
        if cleaned > 0 {
            debug!("Cleaned up {} old JIT stack entries", cleaned);
        }

        cleaned
    }
} 