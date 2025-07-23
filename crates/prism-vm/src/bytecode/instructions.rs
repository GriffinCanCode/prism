//! Prism VM Instruction Set
//!
//! This module defines the complete instruction set for the Prism Virtual Machine,
//! including stack operations, arithmetic, control flow, object operations,
//! effect handling, and capability enforcement.

use crate::VMResult;
use prism_pir::{Effect, Capability};
use serde::{Deserialize, Serialize};

/// Prism VM opcodes - Complete instruction set
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PrismOpcode {
    // Stack Operations (0x00-0x0F)
    /// No operation
    NOP = 0x00,
    /// Duplicate top stack value
    DUP = 0x01,
    /// Remove top stack value
    POP = 0x02,
    /// Swap top two values
    SWAP = 0x03,
    /// Rotate top three values (top -> third)
    ROT3 = 0x04,
    /// Duplicate value at depth n
    DUP_N(u8) = 0x05,
    /// Remove n values from stack
    POP_N(u8) = 0x06,
    
    // Constants and Literals (0x10-0x1F)
    /// Load constant from pool
    LOAD_CONST(u16) = 0x10,
    /// Load null value
    LOAD_NULL = 0x11,
    /// Load boolean true
    LOAD_TRUE = 0x12,
    /// Load boolean false
    LOAD_FALSE = 0x13,
    /// Load small integer (-128 to 127)
    LOAD_SMALL_INT(i8) = 0x14,
    /// Load zero
    LOAD_ZERO = 0x15,
    /// Load one
    LOAD_ONE = 0x16,
    
    // Local Variables (0x20-0x2F)
    /// Load local variable
    LOAD_LOCAL(u8) = 0x20,
    /// Store to local variable
    STORE_LOCAL(u8) = 0x21,
    /// Load closure upvalue
    LOAD_UPVALUE(u8) = 0x22,
    /// Store closure upvalue
    STORE_UPVALUE(u8) = 0x23,
    /// Load local variable (extended addressing)
    LOAD_LOCAL_EXT(u16) = 0x24,
    /// Store to local variable (extended addressing)
    STORE_LOCAL_EXT(u16) = 0x25,
    
    // Global Variables (0x30-0x3F)
    /// Load global variable
    LOAD_GLOBAL(u16) = 0x30,
    /// Store global variable
    STORE_GLOBAL(u16) = 0x31,
    /// Load global by name hash
    LOAD_GLOBAL_HASH(u32) = 0x32,
    /// Store global by name hash
    STORE_GLOBAL_HASH(u32) = 0x33,
    
    // Arithmetic Operations (0x40-0x4F)
    /// Addition
    ADD = 0x40,
    /// Subtraction
    SUB = 0x41,
    /// Multiplication
    MUL = 0x42,
    /// Division
    DIV = 0x43,
    /// Modulo
    MOD = 0x44,
    /// Negation
    NEG = 0x45,
    /// Absolute value
    ABS = 0x46,
    /// Power
    POW = 0x47,
    /// Square root
    SQRT = 0x48,
    
    // Bitwise Operations (0x50-0x5F)
    /// Bitwise AND
    BIT_AND = 0x50,
    /// Bitwise OR
    BIT_OR = 0x51,
    /// Bitwise XOR
    BIT_XOR = 0x52,
    /// Bitwise NOT
    BIT_NOT = 0x53,
    /// Left shift
    SHL = 0x54,
    /// Right shift
    SHR = 0x55,
    /// Arithmetic right shift
    SAR = 0x56,
    
    // Comparison Operations (0x60-0x6F)
    /// Equality
    EQ = 0x60,
    /// Inequality
    NE = 0x61,
    /// Less than
    LT = 0x62,
    /// Less than or equal
    LE = 0x63,
    /// Greater than
    GT = 0x64,
    /// Greater than or equal
    GE = 0x65,
    /// Compare (returns -1, 0, or 1)
    CMP = 0x66,
    /// Semantic equality (Prism-specific)
    SEMANTIC_EQ = 0x67,
    
    // Logical Operations (0x70-0x7F)
    /// Logical AND
    AND = 0x70,
    /// Logical OR
    OR = 0x71,
    /// Logical NOT
    NOT = 0x72,
    /// Logical XOR
    XOR = 0x73,
    
    // Control Flow (0x80-0x8F)
    /// Unconditional jump
    JUMP(i16) = 0x80,
    /// Jump if true
    JUMP_IF_TRUE(i16) = 0x81,
    /// Jump if false
    JUMP_IF_FALSE(i16) = 0x82,
    /// Jump if null
    JUMP_IF_NULL(i16) = 0x83,
    /// Jump if not null
    JUMP_IF_NOT_NULL(i16) = 0x84,
    /// Function call
    CALL(u8) = 0x85,
    /// Tail call
    TAIL_CALL(u8) = 0x86,
    /// Return from function
    RETURN = 0x87,
    /// Return with value
    RETURN_VALUE = 0x88,
    /// Dynamic call (function on stack)
    CALL_DYNAMIC(u8) = 0x89,
    
    // Object Operations (0x90-0x9F)
    /// Create new object
    NEW_OBJECT(u16) = 0x90,
    /// Get object field
    GET_FIELD(u16) = 0x91,
    /// Set object field
    SET_FIELD(u16) = 0x92,
    /// Get object method
    GET_METHOD(u16) = 0x93,
    /// Get field by name hash
    GET_FIELD_HASH(u32) = 0x94,
    /// Set field by name hash
    SET_FIELD_HASH(u32) = 0x95,
    /// Check if object has field
    HAS_FIELD(u16) = 0x96,
    /// Delete object field
    DELETE_FIELD(u16) = 0x97,
    
    // Array Operations (0xA0-0xAF)
    /// Create new array
    NEW_ARRAY(u16) = 0xA0,
    /// Get array element
    GET_INDEX = 0xA1,
    /// Set array element
    SET_INDEX = 0xA2,
    /// Get array length
    ARRAY_LEN = 0xA3,
    /// Array push
    ARRAY_PUSH = 0xA4,
    /// Array pop
    ARRAY_POP = 0xA5,
    /// Array slice
    ARRAY_SLICE = 0xA6,
    /// Array concat
    ARRAY_CONCAT = 0xA7,
    
    // String Operations (0xB0-0xBF)
    /// String concatenation
    STR_CONCAT = 0xB0,
    /// String length
    STR_LEN = 0xB1,
    /// String substring
    STR_SUBSTR = 0xB2,
    /// String find
    STR_FIND = 0xB3,
    /// String replace
    STR_REPLACE = 0xB4,
    /// String to upper
    STR_UPPER = 0xB5,
    /// String to lower
    STR_LOWER = 0xB6,
    /// String trim
    STR_TRIM = 0xB7,
    
    // Type Operations (0xC0-0xCF)
    /// Runtime type check
    TYPE_CHECK(u16) = 0xC0,
    /// Type casting
    TYPE_CAST(u16) = 0xC1,
    /// Instance check
    INSTANCE_OF(u16) = 0xC2,
    /// Get type of value
    GET_TYPE = 0xC3,
    /// Type name
    TYPE_NAME = 0xC4,
    /// Is null check
    IS_NULL = 0xC5,
    /// Is number check
    IS_NUMBER = 0xC6,
    /// Is string check
    IS_STRING = 0xC7,
    
    // Effect Operations (0xD0-0xDF)
    /// Enter effect context
    EFFECT_ENTER(u16) = 0xD0,
    /// Exit effect context
    EFFECT_EXIT = 0xD1,
    /// Invoke effectful operation
    EFFECT_INVOKE(u16) = 0xD2,
    /// Handle effect
    EFFECT_HANDLE(u16) = 0xD3,
    /// Resume effect handler
    EFFECT_RESUME = 0xD4,
    /// Abort effect
    EFFECT_ABORT = 0xD5,
    
    // Capability Operations (0xE0-0xEF)
    /// Check capability
    CAP_CHECK(u16) = 0xE0,
    /// Delegate capability
    CAP_DELEGATE(u16) = 0xE1,
    /// Revoke capability
    CAP_REVOKE(u16) = 0xE2,
    /// Acquire capability
    CAP_ACQUIRE(u16) = 0xE3,
    /// Release capability
    CAP_RELEASE(u16) = 0xE4,
    /// List capabilities
    CAP_LIST = 0xE5,
    
    // Concurrency Operations (0xF0-0xFF)
    /// Spawn new actor
    SPAWN_ACTOR(u16) = 0xF0,
    /// Send message to actor
    SEND_MESSAGE = 0xF1,
    /// Receive message
    RECEIVE_MESSAGE = 0xF2,
    /// Await async operation
    AWAIT = 0xF3,
    /// Yield control
    YIELD = 0xF4,
    /// Create future
    CREATE_FUTURE = 0xF5,
    /// Resolve future
    RESOLVE_FUTURE = 0xF6,
    /// Reject future
    REJECT_FUTURE = 0xF7,
    
    // Pattern Matching (0x100-0x10F) - Extended opcodes
    /// Pattern match
    MATCH(u16) = 0x100,
    /// Match guard
    MATCH_GUARD = 0x101,
    /// Bind pattern variable
    BIND_PATTERN(u8) = 0x102,
    /// Destructure tuple
    DESTRUCTURE_TUPLE(u8) = 0x103,
    /// Destructure array
    DESTRUCTURE_ARRAY(u8) = 0x104,
    /// Destructure object
    DESTRUCTURE_OBJECT(u16) = 0x105,
    
    // Advanced Operations (0x110-0x11F)
    /// Create closure
    CLOSURE(u16) = 0x110,
    /// Partial application
    PARTIAL(u8) = 0x111,
    /// Compose functions
    COMPOSE = 0x112,
    /// Pipe operation
    PIPE = 0x113,
    /// Memoize function
    MEMOIZE = 0x114,
    
    // Debugging Operations (0x120-0x12F)
    /// Debugger breakpoint
    BREAKPOINT = 0x120,
    /// Execution trace
    TRACE(u16) = 0x121,
    /// Start profiling
    PROFILE_START(u16) = 0x122,
    /// End profiling
    PROFILE_END = 0x123,
    /// Log message
    LOG(u16) = 0x124,
    /// Assert condition
    ASSERT = 0x125,
    
    // Memory Management (0x130-0x13F)
    /// Garbage collection hint
    GC_HINT = 0x130,
    /// Reference count increment
    REF_INC = 0x131,
    /// Reference count decrement
    REF_DEC = 0x132,
    /// Weak reference
    WEAK_REF = 0x133,
    /// Strong reference
    STRONG_REF = 0x134,
    
    // I/O Operations (0x140-0x14F)
    /// Read from stream
    IO_READ(u16) = 0x140,
    /// Write to stream
    IO_WRITE(u16) = 0x141,
    /// Flush stream
    IO_FLUSH(u16) = 0x142,
    /// Close stream
    IO_CLOSE(u16) = 0x143,
    /// Open file
    IO_OPEN(u16) = 0x144,
    
    // Exception Handling (0x150-0x15F)
    /// Throw exception
    THROW = 0x150,
    /// Try block start
    TRY_START(u16) = 0x151,
    /// Try block end
    TRY_END = 0x152,
    /// Catch block
    CATCH(u16) = 0x153,
    /// Finally block
    FINALLY = 0x154,
    /// Rethrow exception
    RETHROW = 0x155,
}

/// Complete instruction with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    /// The opcode
    pub opcode: PrismOpcode,
    /// Source location for debugging
    pub source_location: Option<SourceLocation>,
    /// Required capabilities for this instruction
    pub required_capabilities: Vec<Capability>,
    /// Effects produced by this instruction
    pub effects: Vec<Effect>,
    /// Instruction metadata
    pub metadata: Option<InstructionMetadata>,
}

/// Source location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File ID
    pub file_id: u32,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u32,
    /// Length in characters
    pub length: u32,
}

/// Instruction metadata for optimization and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionMetadata {
    /// Execution frequency hint
    pub frequency_hint: Option<FrequencyHint>,
    /// Stack effect (net change in stack depth)
    pub stack_effect: i16,
    /// Whether this instruction can throw
    pub can_throw: bool,
    /// Whether this instruction has side effects
    pub has_side_effects: bool,
    /// Performance cost estimate
    pub cost_estimate: Option<u32>,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Frequency hint for instruction execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrequencyHint {
    /// Executed very rarely
    Cold,
    /// Normal execution frequency
    Normal,
    /// Executed frequently
    Hot,
    /// Critical path instruction
    Critical,
}

/// Optimization hints for the JIT compiler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationHint {
    /// Inline this operation
    Inline,
    /// Don't inline this operation
    NoInline,
    /// Likely branch direction
    LikelyBranch(bool),
    /// Loop invariant
    LoopInvariant,
    /// Pure function (no side effects)
    Pure,
    /// Memoizable operation
    Memoizable,
}

impl Instruction {
    /// Create a new instruction with just the opcode
    pub fn new(opcode: PrismOpcode) -> Self {
        Self {
            opcode,
            source_location: None,
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            metadata: None,
        }
    }

    /// Create a new instruction with source location
    pub fn with_location(opcode: PrismOpcode, location: SourceLocation) -> Self {
        Self {
            opcode,
            source_location: Some(location),
            required_capabilities: Vec::new(),
            effects: Vec::new(),
            metadata: None,
        }
    }

    /// Add capability requirement
    pub fn require_capability(mut self, capability: Capability) -> Self {
        self.required_capabilities.push(capability);
        self
    }

    /// Add effect
    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: InstructionMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get the instruction size in bytes
    pub fn size(&self) -> u32 {
        match self.opcode {
            // Single byte instructions
            PrismOpcode::NOP | PrismOpcode::DUP | PrismOpcode::POP | PrismOpcode::SWAP |
            PrismOpcode::LOAD_NULL | PrismOpcode::LOAD_TRUE | PrismOpcode::LOAD_FALSE |
            PrismOpcode::LOAD_ZERO | PrismOpcode::LOAD_ONE |
            PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL | PrismOpcode::DIV |
            PrismOpcode::MOD | PrismOpcode::NEG | PrismOpcode::ABS |
            PrismOpcode::BIT_AND | PrismOpcode::BIT_OR | PrismOpcode::BIT_XOR | PrismOpcode::BIT_NOT |
            PrismOpcode::SHL | PrismOpcode::SHR | PrismOpcode::SAR |
            PrismOpcode::EQ | PrismOpcode::NE | PrismOpcode::LT | PrismOpcode::LE |
            PrismOpcode::GT | PrismOpcode::GE | PrismOpcode::CMP | PrismOpcode::SEMANTIC_EQ |
            PrismOpcode::AND | PrismOpcode::OR | PrismOpcode::NOT | PrismOpcode::XOR |
            PrismOpcode::RETURN | PrismOpcode::RETURN_VALUE |
            PrismOpcode::GET_INDEX | PrismOpcode::SET_INDEX | PrismOpcode::ARRAY_LEN |
            PrismOpcode::ARRAY_PUSH | PrismOpcode::ARRAY_POP | PrismOpcode::ARRAY_SLICE | PrismOpcode::ARRAY_CONCAT |
            PrismOpcode::STR_CONCAT | PrismOpcode::STR_LEN | PrismOpcode::STR_SUBSTR |
            PrismOpcode::STR_FIND | PrismOpcode::STR_REPLACE | PrismOpcode::STR_UPPER |
            PrismOpcode::STR_LOWER | PrismOpcode::STR_TRIM |
            PrismOpcode::GET_TYPE | PrismOpcode::TYPE_NAME | PrismOpcode::IS_NULL |
            PrismOpcode::IS_NUMBER | PrismOpcode::IS_STRING |
            PrismOpcode::EFFECT_EXIT | PrismOpcode::EFFECT_RESUME | PrismOpcode::EFFECT_ABORT |
            PrismOpcode::CAP_LIST |
            PrismOpcode::SEND_MESSAGE | PrismOpcode::RECEIVE_MESSAGE | PrismOpcode::AWAIT |
            PrismOpcode::YIELD | PrismOpcode::CREATE_FUTURE | PrismOpcode::RESOLVE_FUTURE | PrismOpcode::REJECT_FUTURE |
            PrismOpcode::MATCH_GUARD | PrismOpcode::COMPOSE | PrismOpcode::PIPE |
            PrismOpcode::BREAKPOINT | PrismOpcode::PROFILE_END | PrismOpcode::ASSERT |
            PrismOpcode::GC_HINT | PrismOpcode::REF_INC | PrismOpcode::REF_DEC |
            PrismOpcode::WEAK_REF | PrismOpcode::STRONG_REF |
            PrismOpcode::THROW | PrismOpcode::TRY_END | PrismOpcode::FINALLY | PrismOpcode::RETHROW => 1,

            // Two byte instructions (opcode + u8)
            PrismOpcode::DUP_N(_) | PrismOpcode::POP_N(_) | PrismOpcode::LOAD_SMALL_INT(_) |
            PrismOpcode::LOAD_LOCAL(_) | PrismOpcode::STORE_LOCAL(_) | PrismOpcode::LOAD_UPVALUE(_) | PrismOpcode::STORE_UPVALUE(_) |
            PrismOpcode::CALL(_) | PrismOpcode::TAIL_CALL(_) | PrismOpcode::CALL_DYNAMIC(_) |
            PrismOpcode::BIND_PATTERN(_) | PrismOpcode::DESTRUCTURE_TUPLE(_) | PrismOpcode::DESTRUCTURE_ARRAY(_) |
            PrismOpcode::PARTIAL(_) => 2,

            // Three byte instructions (opcode + u16)
            PrismOpcode::LOAD_CONST(_) | PrismOpcode::LOAD_LOCAL_EXT(_) | PrismOpcode::STORE_LOCAL_EXT(_) |
            PrismOpcode::LOAD_GLOBAL(_) | PrismOpcode::STORE_GLOBAL(_) |
            PrismOpcode::JUMP(_) | PrismOpcode::JUMP_IF_TRUE(_) | PrismOpcode::JUMP_IF_FALSE(_) |
            PrismOpcode::JUMP_IF_NULL(_) | PrismOpcode::JUMP_IF_NOT_NULL(_) |
            PrismOpcode::NEW_OBJECT(_) | PrismOpcode::GET_FIELD(_) | PrismOpcode::SET_FIELD(_) |
            PrismOpcode::GET_METHOD(_) | PrismOpcode::HAS_FIELD(_) | PrismOpcode::DELETE_FIELD(_) |
            PrismOpcode::NEW_ARRAY(_) |
            PrismOpcode::TYPE_CHECK(_) | PrismOpcode::TYPE_CAST(_) | PrismOpcode::INSTANCE_OF(_) |
            PrismOpcode::EFFECT_ENTER(_) | PrismOpcode::EFFECT_INVOKE(_) | PrismOpcode::EFFECT_HANDLE(_) |
            PrismOpcode::CAP_CHECK(_) | PrismOpcode::CAP_DELEGATE(_) | PrismOpcode::CAP_REVOKE(_) |
            PrismOpcode::CAP_ACQUIRE(_) | PrismOpcode::CAP_RELEASE(_) |
            PrismOpcode::SPAWN_ACTOR(_) |
            PrismOpcode::MATCH(_) | PrismOpcode::DESTRUCTURE_OBJECT(_) |
            PrismOpcode::CLOSURE(_) |
            PrismOpcode::TRACE(_) | PrismOpcode::PROFILE_START(_) | PrismOpcode::LOG(_) |
            PrismOpcode::IO_READ(_) | PrismOpcode::IO_WRITE(_) | PrismOpcode::IO_FLUSH(_) |
            PrismOpcode::IO_CLOSE(_) | PrismOpcode::IO_OPEN(_) |
            PrismOpcode::TRY_START(_) | PrismOpcode::CATCH(_) => 3,

            // Five byte instructions (opcode + u32)
            PrismOpcode::LOAD_GLOBAL_HASH(_) | PrismOpcode::STORE_GLOBAL_HASH(_) |
            PrismOpcode::GET_FIELD_HASH(_) | PrismOpcode::SET_FIELD_HASH(_) => 5,

            // Extended opcodes (varies)
            _ => 3, // Default to 3 bytes for extended opcodes
        }
    }

    /// Check if this instruction can throw an exception
    pub fn can_throw(&self) -> bool {
        self.metadata.as_ref().map_or(false, |m| m.can_throw) ||
        matches!(self.opcode,
            PrismOpcode::DIV | PrismOpcode::MOD |
            PrismOpcode::GET_FIELD(_) | PrismOpcode::SET_FIELD(_) |
            PrismOpcode::GET_INDEX | PrismOpcode::SET_INDEX |
            PrismOpcode::TYPE_CAST(_) |
            PrismOpcode::CALL(_) | PrismOpcode::CALL_DYNAMIC(_) |
            PrismOpcode::THROW | PrismOpcode::RETHROW |
            PrismOpcode::ASSERT
        )
    }

    /// Check if this instruction has side effects
    pub fn has_side_effects(&self) -> bool {
        self.metadata.as_ref().map_or(false, |m| m.has_side_effects) ||
        !self.effects.is_empty() ||
        matches!(self.opcode,
            PrismOpcode::STORE_LOCAL(_) | PrismOpcode::STORE_LOCAL_EXT(_) |
            PrismOpcode::STORE_GLOBAL(_) | PrismOpcode::STORE_GLOBAL_HASH(_) |
            PrismOpcode::SET_FIELD(_) | PrismOpcode::SET_FIELD_HASH(_) |
            PrismOpcode::SET_INDEX | PrismOpcode::DELETE_FIELD(_) |
            PrismOpcode::ARRAY_PUSH | PrismOpcode::ARRAY_POP |
            PrismOpcode::CALL(_) | PrismOpcode::CALL_DYNAMIC(_) | PrismOpcode::TAIL_CALL(_) |
            PrismOpcode::EFFECT_INVOKE(_) | PrismOpcode::EFFECT_HANDLE(_) |
            PrismOpcode::CAP_DELEGATE(_) | PrismOpcode::CAP_REVOKE(_) |
            PrismOpcode::SPAWN_ACTOR(_) | PrismOpcode::SEND_MESSAGE |
            PrismOpcode::IO_READ(_) | PrismOpcode::IO_WRITE(_) | PrismOpcode::IO_FLUSH(_) |
            PrismOpcode::IO_CLOSE(_) | PrismOpcode::IO_OPEN(_) |
            PrismOpcode::THROW | PrismOpcode::RETHROW |
            PrismOpcode::LOG(_) | PrismOpcode::PROFILE_START(_) | PrismOpcode::PROFILE_END |
            PrismOpcode::GC_HINT
        )
    }

    /// Get the stack effect of this instruction (net change in stack depth)
    pub fn stack_effect(&self) -> i16 {
        if let Some(metadata) = &self.metadata {
            return metadata.stack_effect;
        }

        // Default stack effects for common instructions
        match self.opcode {
            // Stack operations
            PrismOpcode::NOP => 0,
            PrismOpcode::DUP => 1,
            PrismOpcode::POP => -1,
            PrismOpcode::SWAP => 0,
            PrismOpcode::ROT3 => 0,
            PrismOpcode::DUP_N(_) => 1,
            PrismOpcode::POP_N(n) => -(n as i16),

            // Load operations (push to stack)
            PrismOpcode::LOAD_CONST(_) | PrismOpcode::LOAD_NULL | PrismOpcode::LOAD_TRUE |
            PrismOpcode::LOAD_FALSE | PrismOpcode::LOAD_SMALL_INT(_) | PrismOpcode::LOAD_ZERO |
            PrismOpcode::LOAD_ONE | PrismOpcode::LOAD_LOCAL(_) | PrismOpcode::LOAD_LOCAL_EXT(_) |
            PrismOpcode::LOAD_UPVALUE(_) | PrismOpcode::LOAD_GLOBAL(_) | PrismOpcode::LOAD_GLOBAL_HASH(_) => 1,

            // Store operations (pop from stack)
            PrismOpcode::STORE_LOCAL(_) | PrismOpcode::STORE_LOCAL_EXT(_) | PrismOpcode::STORE_UPVALUE(_) |
            PrismOpcode::STORE_GLOBAL(_) | PrismOpcode::STORE_GLOBAL_HASH(_) => -1,

            // Binary operations (pop 2, push 1)
            PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL | PrismOpcode::DIV |
            PrismOpcode::MOD | PrismOpcode::POW | PrismOpcode::BIT_AND | PrismOpcode::BIT_OR |
            PrismOpcode::BIT_XOR | PrismOpcode::SHL | PrismOpcode::SHR | PrismOpcode::SAR |
            PrismOpcode::EQ | PrismOpcode::NE | PrismOpcode::LT | PrismOpcode::LE |
            PrismOpcode::GT | PrismOpcode::GE | PrismOpcode::CMP | PrismOpcode::SEMANTIC_EQ |
            PrismOpcode::AND | PrismOpcode::OR | PrismOpcode::XOR => -1,

            // Unary operations (pop 1, push 1)
            PrismOpcode::NEG | PrismOpcode::ABS | PrismOpcode::SQRT | PrismOpcode::BIT_NOT |
            PrismOpcode::NOT | PrismOpcode::GET_TYPE | PrismOpcode::TYPE_NAME |
            PrismOpcode::IS_NULL | PrismOpcode::IS_NUMBER | PrismOpcode::IS_STRING => 0,

            // Control flow
            PrismOpcode::JUMP(_) => 0,
            PrismOpcode::JUMP_IF_TRUE(_) | PrismOpcode::JUMP_IF_FALSE(_) |
            PrismOpcode::JUMP_IF_NULL(_) | PrismOpcode::JUMP_IF_NOT_NULL(_) => -1,
            PrismOpcode::CALL(argc) => -(argc as i16), // Pop args, push result
            PrismOpcode::TAIL_CALL(_) => 0, // Doesn't affect current frame stack
            PrismOpcode::RETURN => -1,
            PrismOpcode::RETURN_VALUE => 0,

            // Object operations
            PrismOpcode::NEW_OBJECT(_) => 1,
            PrismOpcode::GET_FIELD(_) | PrismOpcode::GET_FIELD_HASH(_) => 0, // Pop object, push field
            PrismOpcode::SET_FIELD(_) | PrismOpcode::SET_FIELD_HASH(_) => -2, // Pop object and value
            PrismOpcode::GET_METHOD(_) => 0, // Pop object, push method
            PrismOpcode::HAS_FIELD(_) => 0, // Pop object, push boolean
            PrismOpcode::DELETE_FIELD(_) => -1, // Pop object

            // Array operations
            PrismOpcode::NEW_ARRAY(size) => -(size as i16) + 1, // Pop elements, push array
            PrismOpcode::GET_INDEX => -1, // Pop array and index, push element
            PrismOpcode::SET_INDEX => -3, // Pop array, index, and value
            PrismOpcode::ARRAY_LEN => 0, // Pop array, push length
            PrismOpcode::ARRAY_PUSH => -2, // Pop array and element
            PrismOpcode::ARRAY_POP => 0, // Pop array, push element
            PrismOpcode::ARRAY_SLICE => -2, // Pop array, start, end, push slice
            PrismOpcode::ARRAY_CONCAT => -1, // Pop two arrays, push concatenated

            // String operations
            PrismOpcode::STR_CONCAT => -1, // Pop two strings, push concatenated
            PrismOpcode::STR_LEN => 0, // Pop string, push length
            PrismOpcode::STR_SUBSTR => -2, // Pop string, start, length, push substring
            PrismOpcode::STR_FIND => -1, // Pop string and pattern, push index
            PrismOpcode::STR_REPLACE => -2, // Pop string, pattern, replacement, push result
            PrismOpcode::STR_UPPER | PrismOpcode::STR_LOWER | PrismOpcode::STR_TRIM => 0,

            // Default: assume no stack effect
            _ => 0,
        }
    }
}

impl PrismOpcode {
    /// Get the opcode value as a byte
    pub fn as_byte(self) -> u8 {
        // This is a simplified implementation
        // In a real implementation, we'd need to handle the parameter encoding
        match self {
            PrismOpcode::NOP => 0x00,
            PrismOpcode::DUP => 0x01,
            PrismOpcode::POP => 0x02,
            PrismOpcode::SWAP => 0x03,
            PrismOpcode::ROT3 => 0x04,
            // ... continue for all opcodes
            _ => 0xFF, // Placeholder for extended opcodes
        }
    }

    /// Check if this is an extended opcode (requires more than 1 byte)
    pub fn is_extended(self) -> bool {
        matches!(self,
            PrismOpcode::MATCH(_) | PrismOpcode::MATCH_GUARD | PrismOpcode::BIND_PATTERN(_) |
            PrismOpcode::DESTRUCTURE_TUPLE(_) | PrismOpcode::DESTRUCTURE_ARRAY(_) | PrismOpcode::DESTRUCTURE_OBJECT(_) |
            PrismOpcode::CLOSURE(_) | PrismOpcode::PARTIAL(_) | PrismOpcode::COMPOSE | PrismOpcode::PIPE | PrismOpcode::MEMOIZE
        )
    }

    /// Get a human-readable name for the opcode
    pub fn name(self) -> &'static str {
        match self {
            PrismOpcode::NOP => "nop",
            PrismOpcode::DUP => "dup",
            PrismOpcode::POP => "pop",
            PrismOpcode::SWAP => "swap",
            PrismOpcode::ROT3 => "rot3",
            PrismOpcode::DUP_N(_) => "dup_n",
            PrismOpcode::POP_N(_) => "pop_n",
            PrismOpcode::LOAD_CONST(_) => "load_const",
            PrismOpcode::LOAD_NULL => "load_null",
            PrismOpcode::LOAD_TRUE => "load_true",
            PrismOpcode::LOAD_FALSE => "load_false",
            PrismOpcode::LOAD_SMALL_INT(_) => "load_small_int",
            PrismOpcode::LOAD_ZERO => "load_zero",
            PrismOpcode::LOAD_ONE => "load_one",
            PrismOpcode::LOAD_LOCAL(_) => "load_local",
            PrismOpcode::STORE_LOCAL(_) => "store_local",
            PrismOpcode::LOAD_UPVALUE(_) => "load_upvalue",
            PrismOpcode::STORE_UPVALUE(_) => "store_upvalue",
            PrismOpcode::LOAD_LOCAL_EXT(_) => "load_local_ext",
            PrismOpcode::STORE_LOCAL_EXT(_) => "store_local_ext",
            PrismOpcode::LOAD_GLOBAL(_) => "load_global",
            PrismOpcode::STORE_GLOBAL(_) => "store_global",
            PrismOpcode::LOAD_GLOBAL_HASH(_) => "load_global_hash",
            PrismOpcode::STORE_GLOBAL_HASH(_) => "store_global_hash",
            PrismOpcode::ADD => "add",
            PrismOpcode::SUB => "sub",
            PrismOpcode::MUL => "mul",
            PrismOpcode::DIV => "div",
            PrismOpcode::MOD => "mod",
            PrismOpcode::NEG => "neg",
            PrismOpcode::ABS => "abs",
            PrismOpcode::POW => "pow",
            PrismOpcode::SQRT => "sqrt",
            PrismOpcode::BIT_AND => "bit_and",
            PrismOpcode::BIT_OR => "bit_or",
            PrismOpcode::BIT_XOR => "bit_xor",
            PrismOpcode::BIT_NOT => "bit_not",
            PrismOpcode::SHL => "shl",
            PrismOpcode::SHR => "shr",
            PrismOpcode::SAR => "sar",
            PrismOpcode::EQ => "eq",
            PrismOpcode::NE => "ne",
            PrismOpcode::LT => "lt",
            PrismOpcode::LE => "le",
            PrismOpcode::GT => "gt",
            PrismOpcode::GE => "ge",
            PrismOpcode::CMP => "cmp",
            PrismOpcode::SEMANTIC_EQ => "semantic_eq",
            PrismOpcode::AND => "and",
            PrismOpcode::OR => "or",
            PrismOpcode::NOT => "not",
            PrismOpcode::XOR => "xor",
            PrismOpcode::JUMP(_) => "jump",
            PrismOpcode::JUMP_IF_TRUE(_) => "jump_if_true",
            PrismOpcode::JUMP_IF_FALSE(_) => "jump_if_false",
            PrismOpcode::JUMP_IF_NULL(_) => "jump_if_null",
            PrismOpcode::JUMP_IF_NOT_NULL(_) => "jump_if_not_null",
            PrismOpcode::CALL(_) => "call",
            PrismOpcode::TAIL_CALL(_) => "tail_call",
            PrismOpcode::RETURN => "return",
            PrismOpcode::RETURN_VALUE => "return_value",
            PrismOpcode::CALL_DYNAMIC(_) => "call_dynamic",
            _ => "unknown", // Placeholder for remaining opcodes
        }
    }
} 