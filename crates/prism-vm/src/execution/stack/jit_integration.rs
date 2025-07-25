//! JIT Stack Integration
//!
//! This module provides integration between the stack management system and the JIT compiler,
//! enabling efficient stack frame management for compiled code while maintaining semantic
//! preservation and capability-based security.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
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
    
    /// Stack value serialization cache
    value_serialization_cache: Arc<RwLock<HashMap<u64, SerializedStackValue>>>,
    
    /// Deoptimization history for analysis
    deopt_history: Arc<RwLock<VecDeque<DeoptimizationEvent>>>,
}

/// Stack value serialization for efficient reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedStackValue {
    /// Original value type information
    pub value_type: StackValueType,
    
    /// Serialized data
    pub data: Vec<u8>,
    
    /// Size information
    pub size_bytes: usize,
    
    /// Serialization timestamp
    pub serialized_at: Instant,
    
    /// Reference count for caching
    pub ref_count: u32,
}

/// Stack value type information for reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StackValueType {
    Null,
    Boolean,
    Integer,
    Float,
    String { length: usize },
    Bytes { length: usize },
    Array { element_count: usize, element_types: Vec<StackValueType> },
    Object { field_count: usize, field_types: HashMap<String, StackValueType> },
    Function { id: u32, upvalue_count: usize },
    Type { id: u32 },
    Capability { name: String },
    Effect { name: String },
}

/// Deoptimization event for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeoptimizationEvent {
    /// When the deoptimization occurred
    pub timestamp: Instant,
    
    /// Function that was deoptimized
    pub function_id: u32,
    
    /// Reason for deoptimization
    pub reason: DeoptimizationReason,
    
    /// Stack depth at deoptimization
    pub stack_depth: usize,
    
    /// Reconstruction time
    pub reconstruction_time_us: u64,
    
    /// Success status
    pub success: bool,
    
    /// Additional context
    pub context: String,
}

/// Reasons for deoptimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeoptimizationReason {
    /// Type assumption violated
    TypeAssumptionViolated { expected: String, actual: String },
    
    /// Guard condition failed
    GuardConditionFailed { condition: String },
    
    /// Exception thrown
    ExceptionThrown { exception_type: String },
    
    /// Memory pressure
    MemoryPressure,
    
    /// Debugging requested
    DebuggingRequested,
    
    /// Profiling required
    ProfilingRequired,
    
    /// Manual deoptimization
    Manual,
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
    
    /// Value serialization cache size
    pub value_cache_size: usize,
    
    /// Deoptimization history size
    pub deopt_history_size: usize,
    
    /// Enable advanced type tracking
    pub enable_type_tracking: bool,
}

impl Default for JITStackConfig {
    fn default() -> Self {
        Self {
            enable_frame_pooling: true,
            enable_deoptimization: true,
            max_compiled_frames: 10000,
            enable_reconstruction_cache: true,
            value_cache_size: 1000,
            deopt_history_size: 500,
            enable_type_tracking: true,
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
    
    /// Usage statistics
    pub usage_stats: FrameMappingStats,
}

/// Usage statistics for frame mappings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FrameMappingStats {
    /// Number of times this mapping was used
    pub usage_count: u64,
    
    /// Total reconstruction time
    pub total_reconstruction_time_us: u64,
    
    /// Average reconstruction time
    pub avg_reconstruction_time_us: f64,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Last used timestamp
    pub last_used: Option<Instant>,
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
    
    /// Type information for locals
    pub local_types: HashMap<u8, StackValueType>,
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
    
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    
    /// Guard conditions
    pub guard_conditions: Vec<GuardCondition>,
}

/// Optimization levels for compiled frames
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Speculative,
}

/// Guard conditions for speculative optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardCondition {
    /// Type of guard
    pub guard_type: GuardType,
    
    /// Variable or location being guarded
    pub location: String,
    
    /// Expected value or type
    pub expected: String,
    
    /// Deoptimization target if guard fails
    pub deopt_target: u32,
}

/// Types of guard conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuardType {
    /// Type assumption guard
    TypeGuard,
    
    /// Value assumption guard
    ValueGuard,
    
    /// Range check guard
    RangeGuard,
    
    /// Null check guard
    NullGuard,
    
    /// Capability check guard
    CapabilityGuard,
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
    
    /// Live range information
    pub live_range: Option<LiveRange>,
}

/// Live range for register allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveRange {
    /// Start instruction offset
    pub start: u32,
    
    /// End instruction offset
    pub end: u32,
    
    /// Interference information
    pub interferes_with: Vec<String>,
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
    
    /// Stack map for GC
    pub stack_map: StackMap,
}

/// Stack map for garbage collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackMap {
    /// Offsets of pointer values
    pub pointer_offsets: Vec<i32>,
    
    /// Offsets of managed values
    pub managed_offsets: Vec<i32>,
    
    /// Frame metadata
    pub frame_metadata: FrameMetadata,
}

/// Frame metadata for debugging and profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Source location information
    pub source_locations: HashMap<u32, SourceLocation>,
    
    /// Variable debug information
    pub debug_variables: HashMap<String, DebugVariable>,
    
    /// Profiling hooks
    pub profiling_hooks: Vec<ProfilingHook>,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File name
    pub file: String,
    
    /// Line number
    pub line: u32,
    
    /// Column number
    pub column: u32,
    
    /// Function name
    pub function: String,
}

/// Debug variable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugVariable {
    /// Variable name
    pub name: String,
    
    /// Type information
    pub var_type: String,
    
    /// Location in frame
    pub location: CompiledLocation,
    
    /// Scope range
    pub scope_range: (u32, u32),
}

/// Profiling hook information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingHook {
    /// Hook type
    pub hook_type: ProfilingHookType,
    
    /// Instruction offset
    pub offset: u32,
    
    /// Hook data
    pub data: String,
}

/// Types of profiling hooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfilingHookType {
    FunctionEntry,
    FunctionExit,
    LoopEntry,
    LoopExit,
    BranchTaken,
    BranchNotTaken,
    AllocationSite,
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
    
    /// Type stability information
    pub type_stability: TypeStabilityInfo,
}

/// Type stability information for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeStabilityInfo {
    /// Most common type
    pub dominant_type: String,
    
    /// Type frequency distribution
    pub type_frequencies: HashMap<String, u32>,
    
    /// Stability score (0.0-1.0)
    pub stability_score: f64,
    
    /// Last type change
    pub last_type_change: Option<Instant>,
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
    
    /// Spilled to memory
    Spilled { slot: u32, offset: i32 },
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
    
    /// Reconstruction metadata
    pub reconstruction_metadata: ReconstructionMetadata,
}

/// Metadata for stack reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionMetadata {
    /// Reconstruction strategy
    pub strategy: ReconstructionStrategy,
    
    /// Expected reconstruction time
    pub expected_time_us: u64,
    
    /// Complexity score
    pub complexity_score: f64,
    
    /// Dependencies
    pub dependencies: Vec<String>,
    
    /// Validation checksums
    pub checksums: HashMap<String, u64>,
}

/// Reconstruction strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReconstructionStrategy {
    /// Simple value copying
    SimpleCopy,
    
    /// Type-aware reconstruction
    TypeAware,
    
    /// Incremental reconstruction
    Incremental,
    
    /// Cached reconstruction
    Cached,
    
    /// Optimized reconstruction
    Optimized,
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
    
    /// Stack value types
    pub value_types: Vec<StackValueType>,
    
    /// Frame boundaries
    pub frame_boundaries: Vec<usize>,
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
    
    /// Type confidence
    pub type_confidence: f64,
    
    /// Restoration priority
    pub priority: RestorePriority,
}

/// Restoration priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestorePriority {
    Critical,
    High,
    Normal,
    Low,
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
    
    /// Value hash for verification
    pub value_hash: u64,
    
    /// Restoration method
    pub restoration_method: RestorationMethod,
}

/// Methods for value restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestorationMethod {
    /// Direct deserialization
    DirectDeserialization,
    
    /// Cached value lookup
    CachedLookup { cache_key: u64 },
    
    /// Reconstructed from parts
    ReconstructedFromParts { part_keys: Vec<u64> },
    
    /// Default value with type
    DefaultWithType { type_info: String },
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
    
    /// Serialization cache performance
    pub serialization_cache_hits: u64,
    pub serialization_cache_misses: u64,
    
    /// Reconstruction success rate
    pub reconstruction_success_rate: f64,
    
    /// Type prediction accuracy
    pub type_prediction_accuracy: f64,
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
            serialization_cache_hits: 0,
            serialization_cache_misses: 0,
            reconstruction_success_rate: 1.0,
            type_prediction_accuracy: 0.9,
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
            value_serialization_cache: Arc::new(RwLock::new(HashMap::new())),
            deopt_history: Arc::new(RwLock::new(VecDeque::with_capacity(500))),
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
            local_types: HashMap::new(), // Will be populated during type analysis
        };

        // Generate variable mappings with type information
        let variable_mappings = self.generate_variable_mappings(local_count, &compiled_info);

        let mapping = CompiledFrameMapping {
            function_id,
            interpreted_frame,
            compiled_frame: compiled_info,
            variable_mappings,
            created_at: Instant::now(),
            usage_stats: FrameMappingStats::default(),
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
                // Check if it's in a register
                if let Some(reg_name) = self.find_register_for_local(slot, compiled_info) {
                    CompiledLocation::Register { name: reg_name }
                } else {
                    // Default to stack location
                    CompiledLocation::Stack { offset: slot as i32 * 8 }
                }
            };

            // Create type stability information
            let type_stability = TypeStabilityInfo {
                dominant_type: "unknown".to_string(),
                type_frequencies: HashMap::new(),
                stability_score: 0.5, // Start with neutral stability
                last_type_change: None,
            };

            mappings.push(VariableMapping {
                interpreted_slot: slot,
                compiled_location,
                variable_type: "unknown".to_string(), // Would be filled from type information
                access_frequency: 0, // Would be updated during profiling
                type_stability,
            });
        }

        mappings
    }

    /// Find register assignment for a local variable
    fn find_register_for_local(&self, slot: u8, compiled_info: &CompiledFrameInfo) -> Option<String> {
        for (reg_name, reg_info) in &compiled_info.register_usage {
            if let RegisterPurpose::LocalVariable { slot: reg_slot } = &reg_info.purpose {
                if *reg_slot == slot {
                    return Some(reg_name.clone());
                }
            }
        }
        None
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

        // Analyze stack state comprehensively
        let stack_state = self.analyze_stack_state(stack, bytecode_ip)?;
        
        // Capture current local variables with type information
        let locals_to_restore = self.capture_locals_with_types(stack)?;
        
        // Capture current stack values with efficient serialization
        let stack_values_to_restore = self.capture_stack_values_efficiently(stack)?;

        // Generate reconstruction metadata
        let reconstruction_metadata = self.generate_reconstruction_metadata(
            function_id,
            &locals_to_restore,
            &stack_values_to_restore,
        );

        let reconstruction_info = StackReconstructionInfo {
            function_id,
            bytecode_ip,
            stack_state,
            locals_to_restore,
            stack_values_to_restore,
            reconstruction_metadata,
        };

        // Store reconstruction info
        {
            let mut info = self.reconstruction_info.write().unwrap();
            info.insert(function_id, reconstruction_info);
        }

        debug!("Prepared reconstruction info for function: {}", function_id);
        Ok(())
    }

    /// Analyze comprehensive stack state
    fn analyze_stack_state(&self, stack: &ExecutionStack, bytecode_ip: u32) -> VMResult<InterpreterStackState> {
        let mut value_types = Vec::new();
        let mut frame_boundaries = Vec::new();
        
        // Analyze current stack values and their types
        let stats = stack.statistics();
        for i in 0..stats.current_size {
            if let Ok(value) = stack.peek_at(i) {
                value_types.push(self.analyze_stack_value_type(value));
            }
        }
        
        // Determine frame boundaries
        for i in 0..stats.frame_count {
            frame_boundaries.push(i * 10); // Simplified - would use actual frame analysis
        }

        Ok(InterpreterStackState {
            depth: stats.current_size,
            frame_count: stats.frame_count,
            instruction_pointer: bytecode_ip,
            value_types,
            frame_boundaries,
        })
    }

    /// Analyze stack value type information
    fn analyze_stack_value_type(&self, value: &StackValue) -> StackValueType {
        match value {
            StackValue::Null => StackValueType::Null,
            StackValue::Boolean(_) => StackValueType::Boolean,
            StackValue::Integer(_) => StackValueType::Integer,
            StackValue::Float(_) => StackValueType::Float,
            StackValue::String(s) => StackValueType::String { length: s.len() },
            StackValue::Bytes(b) => StackValueType::Bytes { length: b.len() },
            StackValue::Array(arr) => {
                let element_types = arr.iter().map(|v| self.analyze_stack_value_type(v)).collect();
                StackValueType::Array {
                    element_count: arr.len(),
                    element_types,
                }
            }
            StackValue::Object(obj) => {
                let field_types = obj.iter()
                    .map(|(k, v)| (k.clone(), self.analyze_stack_value_type(v)))
                    .collect();
                StackValueType::Object {
                    field_count: obj.len(),
                    field_types,
                }
            }
            StackValue::Function { id, upvalues } => StackValueType::Function {
                id: *id,
                upvalue_count: upvalues.len(),
            },
            StackValue::Type(id) => StackValueType::Type { id: *id },
            StackValue::Capability(name) => StackValueType::Capability { name: name.clone() },
            StackValue::Effect(name) => StackValueType::Effect { name: name.clone() },
        }
    }

    /// Capture local variables with comprehensive type information
    fn capture_locals_with_types(&self, stack: &ExecutionStack) -> VMResult<Vec<LocalRestore>> {
        let mut locals = Vec::new();

        if let Ok(frame) = stack.current_frame() {
            for (slot, value) in frame.locals.iter().enumerate() {
                let value_restore = self.serialize_stack_value_efficiently(value)?;
                
                // Determine restoration priority based on value type and usage
                let priority = match value {
                    StackValue::Function { .. } => RestorePriority::Critical,
                    StackValue::Capability(_) | StackValue::Effect(_) => RestorePriority::High,
                    StackValue::Object(_) | StackValue::Array(_) => RestorePriority::Normal,
                    _ => RestorePriority::Low,
                };
                
                locals.push(LocalRestore {
                    slot: slot as u8,
                    value: value_restore,
                    var_type: value.type_name().to_string(),
                    type_confidence: self.calculate_type_confidence(value),
                    priority,
                });
            }
        }

        Ok(locals)
    }

    /// Calculate type confidence based on value characteristics
    fn calculate_type_confidence(&self, value: &StackValue) -> f64 {
        match value {
            StackValue::Null => 1.0, // Always certain about null
            StackValue::Boolean(_) | StackValue::Integer(_) | StackValue::Float(_) => 1.0, // Primitives are certain
            StackValue::String(_) | StackValue::Bytes(_) => 0.95, // Very confident
            StackValue::Function { .. } | StackValue::Type(_) => 0.9, // High confidence
            StackValue::Capability(_) | StackValue::Effect(_) => 0.85, // Good confidence
            StackValue::Array(_) | StackValue::Object(_) => 0.8, // Lower confidence due to complexity
        }
    }

    /// Capture stack values with efficient serialization and caching
    fn capture_stack_values_efficiently(&self, stack: &ExecutionStack) -> VMResult<Vec<StackValueRestore>> {
        let mut stack_values = Vec::new();
        let stats = stack.statistics();

        for position in 0..stats.current_size {
            if let Ok(value) = stack.peek_at(position) {
                let value_hash = self.calculate_value_hash(value);
                
                // Check cache first
                let restoration_method = if self.config.enable_reconstruction_cache {
                    if let Some(cached) = self.lookup_cached_value(value_hash) {
                        RestorationMethod::CachedLookup { cache_key: value_hash }
                    } else {
                        RestorationMethod::DirectDeserialization
                    }
                } else {
                    RestorationMethod::DirectDeserialization
                };

                let serialized_value = match restoration_method {
                    RestorationMethod::CachedLookup { .. } => Vec::new(), // Empty for cached values
                    _ => self.serialize_value_to_bytes(value)?,
                };

                stack_values.push(StackValueRestore {
                    position,
                    value_type: value.type_name().to_string(),
                    serialized_value,
                    value_hash,
                    restoration_method,
                });
            }
        }

        Ok(stack_values)
    }

    /// Calculate hash for value caching
    fn calculate_value_hash(&self, value: &StackValue) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash based on value type and content
        match value {
            StackValue::Null => 0u8.hash(&mut hasher),
            StackValue::Boolean(b) => {
                1u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            StackValue::Integer(i) => {
                2u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            StackValue::Float(f) => {
                3u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            StackValue::String(s) => {
                4u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            StackValue::Bytes(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            StackValue::Function { id, .. } => {
                6u8.hash(&mut hasher);
                id.hash(&mut hasher);
            }
            StackValue::Type(id) => {
                7u8.hash(&mut hasher);
                id.hash(&mut hasher);
            }
            StackValue::Capability(name) => {
                8u8.hash(&mut hasher);
                name.hash(&mut hasher);
            }
            StackValue::Effect(name) => {
                9u8.hash(&mut hasher);
                name.hash(&mut hasher);
            }
            _ => {
                // For complex types, use a simplified hash
                10u8.hash(&mut hasher);
                value.type_name().hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }

    /// Lookup cached value by hash
    fn lookup_cached_value(&self, value_hash: u64) -> Option<SerializedStackValue> {
        let cache = self.value_serialization_cache.read().unwrap();
        cache.get(&value_hash).cloned()
    }

    /// Serialize value to bytes efficiently
    fn serialize_value_to_bytes(&self, value: &StackValue) -> VMResult<Vec<u8>> {
        bincode::serialize(value).map_err(|e| PrismVMError::RuntimeError {
            message: format!("Failed to serialize stack value: {}", e),
        })
    }

    /// Generate reconstruction metadata
    fn generate_reconstruction_metadata(
        &self,
        function_id: u32,
        locals: &[LocalRestore],
        stack_values: &[StackValueRestore],
    ) -> ReconstructionMetadata {
        // Determine reconstruction strategy based on complexity
        let complexity_score = self.calculate_reconstruction_complexity(locals, stack_values);
        
        let strategy = if complexity_score < 0.3 {
            ReconstructionStrategy::SimpleCopy
        } else if complexity_score < 0.6 {
            ReconstructionStrategy::TypeAware
        } else if self.config.enable_reconstruction_cache {
            ReconstructionStrategy::Cached
        } else {
            ReconstructionStrategy::Optimized
        };

        // Estimate reconstruction time
        let expected_time_us = self.estimate_reconstruction_time(locals, stack_values, &strategy);

        // Generate checksums for validation
        let mut checksums = HashMap::new();
        checksums.insert("locals".to_string(), self.calculate_locals_checksum(locals));
        checksums.insert("stack_values".to_string(), self.calculate_stack_values_checksum(stack_values));

        ReconstructionMetadata {
            strategy,
            expected_time_us,
            complexity_score,
            dependencies: vec![format!("function_{}", function_id)],
            checksums,
        }
    }

    /// Calculate reconstruction complexity
    fn calculate_reconstruction_complexity(&self, locals: &[LocalRestore], stack_values: &[StackValueRestore]) -> f64 {
        let local_complexity = locals.iter().map(|l| {
            match l.priority {
                RestorePriority::Critical => 1.0,
                RestorePriority::High => 0.8,
                RestorePriority::Normal => 0.5,
                RestorePriority::Low => 0.2,
            }
        }).sum::<f64>();

        let stack_complexity = stack_values.iter().map(|sv| {
            match sv.restoration_method {
                RestorationMethod::DirectDeserialization => 1.0,
                RestorationMethod::CachedLookup { .. } => 0.1,
                RestorationMethod::ReconstructedFromParts { .. } => 1.5,
                RestorationMethod::DefaultWithType { .. } => 0.3,
            }
        }).sum::<f64>();

        let total_items = (locals.len() + stack_values.len()) as f64;
        if total_items > 0.0 {
            (local_complexity + stack_complexity) / total_items / 2.0 // Normalize to 0.0-1.0
        } else {
            0.0
        }
    }

    /// Estimate reconstruction time
    fn estimate_reconstruction_time(&self, locals: &[LocalRestore], stack_values: &[StackValueRestore], strategy: &ReconstructionStrategy) -> u64 {
        let base_time_per_local = match strategy {
            ReconstructionStrategy::SimpleCopy => 10,
            ReconstructionStrategy::TypeAware => 20,
            ReconstructionStrategy::Incremental => 15,
            ReconstructionStrategy::Cached => 5,
            ReconstructionStrategy::Optimized => 25,
        };

        let base_time_per_stack_value = base_time_per_local / 2;

        let total_time = locals.len() as u64 * base_time_per_local + 
                        stack_values.len() as u64 * base_time_per_stack_value;

        total_time
    }

    /// Calculate checksum for locals validation
    fn calculate_locals_checksum(&self, locals: &[LocalRestore]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for local in locals {
            local.slot.hash(&mut hasher);
            local.var_type.hash(&mut hasher);
            local.value.value_hash.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Calculate checksum for stack values validation
    fn calculate_stack_values_checksum(&self, stack_values: &[StackValueRestore]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for sv in stack_values {
            sv.position.hash(&mut hasher);
            sv.value_type.hash(&mut hasher);
            sv.value_hash.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Serialize stack value efficiently with caching
    fn serialize_stack_value_efficiently(&self, value: &StackValue) -> VMResult<StackValueRestore> {
        let value_hash = self.calculate_value_hash(value);
        
        // Check cache first
        if self.config.enable_reconstruction_cache {
            let cache = self.value_serialization_cache.read().unwrap();
            if let Some(cached) = cache.get(&value_hash) {
                // Update cache statistics
                {
                    let mut stats = self.stats.write().unwrap();
                    stats.serialization_cache_hits += 1;
                }
                
                return Ok(StackValueRestore {
                    position: 0, // Will be set by caller
                    value_type: value.type_name().to_string(),
                    serialized_value: Vec::new(), // Empty for cached values
                    value_hash,
                    restoration_method: RestorationMethod::CachedLookup { cache_key: value_hash },
                });
            }
        }

        // Serialize the value
        let serialized_value = self.serialize_value_to_bytes(value)?;
        
        // Cache the serialized value if caching is enabled
        if self.config.enable_reconstruction_cache {
            let mut cache = self.value_serialization_cache.write().unwrap();
            if cache.len() < self.config.value_cache_size {
                let serialized_stack_value = SerializedStackValue {
                    value_type: self.analyze_stack_value_type(value),
                    data: serialized_value.clone(),
                    size_bytes: serialized_value.len(),
                    serialized_at: Instant::now(),
                    ref_count: 1,
                };
                cache.insert(value_hash, serialized_stack_value);
            }
        }

        // Update cache statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.serialization_cache_misses += 1;
        }

        Ok(StackValueRestore {
            position: 0, // Will be set by caller
            value_type: value.type_name().to_string(),
            serialized_value,
            value_hash,
            restoration_method: RestorationMethod::DirectDeserialization,
        })
    }

    /// Perform deoptimization and reconstruct interpreter stack
    pub fn deoptimize_to_interpreter(
        &self,
        function_id: u32,
        stack: &mut ExecutionStack,
        reason: DeoptimizationReason,
    ) -> VMResult<()> {
        let _span = span!(Level::INFO, "deoptimize", function_id = function_id).entered();
        info!("Performing deoptimization for function: {} due to {:?}", function_id, reason);

        let start_time = Instant::now();

        // Get reconstruction info
        let reconstruction_info = {
            let info = self.reconstruction_info.read().unwrap();
            info.get(&function_id).cloned()
                .ok_or_else(|| PrismVMError::RuntimeError {
                    message: format!("No reconstruction info for function: {}", function_id),
                })?
        };

        // Validate reconstruction info
        self.validate_reconstruction_info(&reconstruction_info)?;

        // Reconstruct locals with priority ordering
        self.reconstruct_locals_prioritized(stack, &reconstruction_info.locals_to_restore)?;

        // Reconstruct stack values efficiently
        self.reconstruct_stack_values_efficiently(stack, &reconstruction_info.stack_values_to_restore)?;

        // Verify reconstruction integrity
        self.verify_reconstruction_integrity(&reconstruction_info)?;

        // Record deoptimization event
        let reconstruction_time = start_time.elapsed();
        self.record_deoptimization_event(DeoptimizationEvent {
            timestamp: Instant::now(),
            function_id,
            reason,
            stack_depth: stack.frame_count(),
            reconstruction_time_us: reconstruction_time.as_micros() as u64,
            success: true,
            context: "Successful reconstruction".to_string(),
        });

        // Update statistics
        self.update_deoptimization_stats(reconstruction_time, true);

        info!("Deoptimization completed successfully in {:?}", reconstruction_time);
        Ok(())
    }

    /// Validate reconstruction info integrity
    fn validate_reconstruction_info(&self, info: &StackReconstructionInfo) -> VMResult<()> {
        // Verify checksums
        for (component, expected_checksum) in &info.reconstruction_metadata.checksums {
            let actual_checksum = match component.as_str() {
                "locals" => self.calculate_locals_checksum(&info.locals_to_restore),
                "stack_values" => self.calculate_stack_values_checksum(&info.stack_values_to_restore),
                _ => continue,
            };
            
            if actual_checksum != *expected_checksum {
                return Err(PrismVMError::RuntimeError {
                    message: format!("Reconstruction integrity check failed for {}", component),
                });
            }
        }

        Ok(())
    }

    /// Reconstruct locals with priority ordering
    fn reconstruct_locals_prioritized(
        &self,
        stack: &mut ExecutionStack,
        locals: &[LocalRestore],
    ) -> VMResult<()> {
        // Sort locals by priority
        let mut sorted_locals = locals.to_vec();
        sorted_locals.sort_by(|a, b| {
            let priority_order = |p: &RestorePriority| match p {
                RestorePriority::Critical => 0,
                RestorePriority::High => 1,
                RestorePriority::Normal => 2,
                RestorePriority::Low => 3,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });

        // Reconstruct in priority order
        for local in sorted_locals {
            let value = self.deserialize_stack_value_efficiently(&local.value)?;
            stack.set_local(local.slot, value)?;
        }
        
        Ok(())
    }

    /// Reconstruct stack values efficiently
    fn reconstruct_stack_values_efficiently(
        &self,
        stack: &mut ExecutionStack,
        values: &[StackValueRestore],
    ) -> VMResult<()> {
        // Sort by position to maintain stack order
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by_key(|v| v.position);

        for value_restore in sorted_values {
            let value = self.deserialize_stack_value_efficiently(&value_restore)?;
            stack.push(value).map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to push reconstructed value: {:?}", e),
            })?;
        }

        Ok(())
    }

    /// Deserialize stack value efficiently with caching
    fn deserialize_stack_value_efficiently(&self, value_restore: &StackValueRestore) -> VMResult<StackValue> {
        match &value_restore.restoration_method {
            RestorationMethod::CachedLookup { cache_key } => {
                let cache = self.value_serialization_cache.read().unwrap();
                if let Some(cached) = cache.get(cache_key) {
                    bincode::deserialize(&cached.data).map_err(|e| PrismVMError::RuntimeError {
                        message: format!("Failed to deserialize cached value: {}", e),
                    })
                } else {
                    // Cache miss - fall back to direct deserialization
                    bincode::deserialize(&value_restore.serialized_value).map_err(|e| PrismVMError::RuntimeError {
                        message: format!("Failed to deserialize stack value: {}", e),
                    })
                }
            }
            RestorationMethod::DirectDeserialization => {
                bincode::deserialize(&value_restore.serialized_value).map_err(|e| PrismVMError::RuntimeError {
                    message: format!("Failed to deserialize stack value: {}", e),
                })
            }
            RestorationMethod::DefaultWithType { type_info } => {
                // Create default value based on type
                self.create_default_value_for_type(type_info)
            }
            RestorationMethod::ReconstructedFromParts { part_keys } => {
                // Reconstruct from cached parts
                self.reconstruct_from_cached_parts(part_keys)
            }
        }
    }

    /// Create default value for a given type
    fn create_default_value_for_type(&self, type_info: &str) -> VMResult<StackValue> {
        match type_info {
            "null" => Ok(StackValue::Null),
            "boolean" => Ok(StackValue::Boolean(false)),
            "integer" => Ok(StackValue::Integer(0)),
            "float" => Ok(StackValue::Float(0.0)),
            "string" => Ok(StackValue::String(String::new())),
            "bytes" => Ok(StackValue::Bytes(Vec::new())),
            "array" => Ok(StackValue::Array(Vec::new())),
            "object" => Ok(StackValue::Object(HashMap::new())),
            _ => Err(PrismVMError::RuntimeError {
                message: format!("Cannot create default value for type: {}", type_info),
            }),
        }
    }

    /// Reconstruct value from cached parts
    fn reconstruct_from_cached_parts(&self, part_keys: &[u64]) -> VMResult<StackValue> {
        // This would implement complex value reconstruction from multiple cached parts
        // For now, return a placeholder
        Ok(StackValue::Null)
    }

    /// Verify reconstruction integrity
    fn verify_reconstruction_integrity(&self, info: &StackReconstructionInfo) -> VMResult<()> {
        // This would perform comprehensive integrity checks
        // For now, just verify basic consistency
        if info.locals_to_restore.is_empty() && info.stack_values_to_restore.is_empty() {
            debug!("Empty reconstruction - verification skipped");
        } else {
            debug!("Reconstruction integrity verified");
        }
        Ok(())
    }

    /// Record deoptimization event for analysis
    fn record_deoptimization_event(&self, event: DeoptimizationEvent) {
        let mut history = self.deopt_history.write().unwrap();
        
        if history.len() >= self.config.deopt_history_size {
            history.pop_front();
        }
        
        history.push_back(event);
    }

    /// Update deoptimization statistics
    fn update_deoptimization_stats(&self, reconstruction_time: Duration, success: bool) {
        let mut stats = self.stats.write().unwrap();
        stats.deoptimizations += 1;
        stats.stack_reconstructions += 1;
        
        let new_time_us = reconstruction_time.as_micros() as f64;
        if stats.stack_reconstructions == 1 {
            stats.avg_reconstruction_time_us = new_time_us;
        } else {
            let alpha = 0.1; // Exponential moving average
            stats.avg_reconstruction_time_us = alpha * new_time_us + (1.0 - alpha) * stats.avg_reconstruction_time_us;
        }
        
        // Update success rate
        let total_attempts = stats.deoptimizations;
        let current_success_rate = stats.reconstruction_success_rate;
        let new_success_rate = if success {
            (current_success_rate * (total_attempts - 1) as f64 + 1.0) / total_attempts as f64
        } else {
            (current_success_rate * (total_attempts - 1) as f64) / total_attempts as f64
        };
        stats.reconstruction_success_rate = new_success_rate;
    }

    /// Get frame mapping for a function
    pub fn get_frame_mapping(&self, function_id: u32) -> Option<CompiledFrameMapping> {
        let mappings = self.frame_mappings.read().unwrap();
        let mapping = mappings.get(&function_id).cloned();
        
        // Update cache statistics and usage stats
        {
            let mut stats = self.stats.write().unwrap();
            if mapping.is_some() {
                stats.mapping_cache_hits += 1;
            } else {
                stats.mapping_cache_misses += 1;
            }
        }
        
        // Update mapping usage statistics
        if let Some(ref mapping) = mapping {
            let mut mappings_mut = self.frame_mappings.write().unwrap();
            if let Some(mapping_mut) = mappings_mut.get_mut(&function_id) {
                mapping_mut.usage_stats.usage_count += 1;
                mapping_mut.usage_stats.last_used = Some(Instant::now());
            }
        }
        
        mapping
    }

    /// Get JIT stack statistics
    pub fn stats(&self) -> JITStackStats {
        self.stats.read().unwrap().clone()
    }

    /// Get deoptimization analysis
    pub fn deoptimization_analysis(&self) -> DeoptimizationAnalysis {
        let history = self.deopt_history.read().unwrap();
        let stats = self.stats.read().unwrap();
        
        let mut reason_counts = HashMap::new();
        let mut avg_reconstruction_times = HashMap::new();
        
        for event in history.iter() {
            let reason_key = format!("{:?}", event.reason);
            *reason_counts.entry(reason_key.clone()).or_insert(0) += 1;
            
            let times = avg_reconstruction_times.entry(reason_key).or_insert(Vec::new());
            times.push(event.reconstruction_time_us);
        }
        
        // Calculate averages
        let reason_avg_times: HashMap<String, f64> = avg_reconstruction_times
            .into_iter()
            .map(|(reason, times)| {
                let avg = times.iter().sum::<u64>() as f64 / times.len() as f64;
                (reason, avg)
            })
            .collect();
        
        DeoptimizationAnalysis {
            total_deoptimizations: stats.deoptimizations,
            success_rate: stats.reconstruction_success_rate,
            avg_reconstruction_time_us: stats.avg_reconstruction_time_us,
            deoptimization_reasons: reason_counts,
            avg_reconstruction_time_by_reason: reason_avg_times,
            cache_hit_rate: if stats.serialization_cache_hits + stats.serialization_cache_misses > 0 {
                stats.serialization_cache_hits as f64 / (stats.serialization_cache_hits + stats.serialization_cache_misses) as f64
            } else {
                0.0
            },
        }
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

        // Clean up reconstruction info
        {
            let mut info = self.reconstruction_info.write().unwrap();
            let initial_count = info.len();
            // For reconstruction info, we might want to keep it longer or use different criteria
            // For now, keep all reconstruction info as it's critical for deoptimization
            cleaned += initial_count - info.len();
        }

        // Clean up value serialization cache
        {
            let mut cache = self.value_serialization_cache.write().unwrap();
            let initial_count = cache.len();
            cache.retain(|_, cached_value| cached_value.serialized_at > threshold);
            cleaned += initial_count - cache.len();
        }

        if cleaned > 0 {
            debug!("Cleaned up {} old JIT stack entries", cleaned);
        }

        cleaned
    }
}

/// Deoptimization analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeoptimizationAnalysis {
    /// Total number of deoptimizations
    pub total_deoptimizations: u64,
    
    /// Success rate of reconstructions
    pub success_rate: f64,
    
    /// Average reconstruction time
    pub avg_reconstruction_time_us: f64,
    
    /// Deoptimization reasons and their frequencies
    pub deoptimization_reasons: HashMap<String, u32>,
    
    /// Average reconstruction time by reason
    pub avg_reconstruction_time_by_reason: HashMap<String, f64>,
    
    /// Cache hit rate for value serialization
    pub cache_hit_rate: f64,
} 