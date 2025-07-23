//! Bytecode Format and Instruction Set
//!
//! This module defines the Prism VM bytecode format, instruction set, and
//! serialization/deserialization functionality.
//!
//! ## Design Principles
//!
//! 1. **Stack-Based**: All operations work on a stack-based execution model
//! 2. **Semantic Preservation**: Bytecode maintains semantic information from PIR
//! 3. **Capability Aware**: Instructions include capability requirements
//! 4. **Effect Tracking**: Effects are explicitly represented in bytecode
//! 5. **Debugging Support**: Rich metadata for debugging and profiling

pub mod instructions;
pub mod constants;
pub mod serialization;

// Re-export main types
pub use instructions::{PrismOpcode, Instruction, InstructionMetadata};
pub use constants::{ConstantPool, Constant, ConstantType};
pub use serialization::{BytecodeSerializer, BytecodeDeserializer};

use crate::{VMResult, VMVersion};
use prism_pir::{Effect, Capability};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Magic number for Prism bytecode files ("PRIS" in ASCII)
pub const PRISM_BYTECODE_MAGIC: u32 = 0x50524953;

/// Current bytecode format version
pub const BYTECODE_FORMAT_VERSION: u16 = 1;

/// Complete Prism bytecode representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismBytecode {
    /// Magic number for identification
    pub magic: u32,
    
    /// Bytecode format version
    pub format_version: u16,
    
    /// VM version that generated this bytecode
    pub vm_version: VMVersion,
    
    /// Unique bytecode identifier
    pub id: Uuid,
    
    /// Constant pool
    pub constants: ConstantPool,
    
    /// Type definitions with semantic information
    pub types: Vec<TypeDefinition>,
    
    /// Function definitions
    pub functions: Vec<FunctionDefinition>,
    
    /// Global variables
    pub globals: Vec<GlobalDefinition>,
    
    /// Module metadata
    pub metadata: ModuleMetadata,
    
    /// Debug information (optional)
    pub debug_info: Option<DebugInfo>,
}

/// Bytecode version information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BytecodeVersion {
    /// Format version
    pub format: u16,
    /// VM version
    pub vm: VMVersion,
}

/// Type definition in bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    /// Type ID
    pub id: u32,
    /// Type name
    pub name: String,
    /// Type kind
    pub kind: TypeKind,
    /// Semantic domain
    pub domain: Option<String>,
    /// Business rules (as bytecode)
    pub business_rules: Vec<Instruction>,
    /// Validation predicates (as bytecode)
    pub validation_predicates: Vec<Instruction>,
}

/// Type kind enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeKind {
    /// Primitive type
    Primitive(PrimitiveType),
    /// Composite type (struct, enum, etc.)
    Composite(CompositeType),
    /// Function type
    Function(FunctionType),
    /// Effect type
    Effect(EffectType),
}

/// Primitive type kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimitiveType {
    /// Integer with signedness and width
    Integer { signed: bool, width: u8 },
    /// Floating point with width
    Float { width: u8 },
    /// Boolean value
    Boolean,
    /// String value
    String,
    /// Unit type (void)
    Unit,
}

/// Composite type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeType {
    /// Composite kind
    pub kind: CompositeKind,
    /// Field definitions
    pub fields: Vec<FieldDefinition>,
    /// Method definitions
    pub methods: Vec<u32>, // Function IDs
}

/// Composite type kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositeKind {
    /// Struct type
    Struct,
    /// Enum type
    Enum,
    /// Union type
    Union,
    /// Tuple type
    Tuple,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,
    /// Field type ID
    pub type_id: u32,
    /// Field offset (for structs)
    pub offset: Option<u32>,
    /// Business meaning
    pub business_meaning: Option<String>,
}

/// Function type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionType {
    /// Parameter types
    pub parameters: Vec<u32>, // Type IDs
    /// Return type
    pub return_type: u32, // Type ID
    /// Effect signature
    pub effects: Vec<Effect>,
    /// Required capabilities
    pub capabilities: Vec<Capability>,
}

/// Effect type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectType {
    /// Effect name
    pub name: String,
    /// Effect operations
    pub operations: Vec<EffectOperation>,
}

/// Effect operation definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectOperation {
    /// Operation name
    pub name: String,
    /// Operation type ID
    pub type_id: u32,
    /// Side effects
    pub side_effects: Vec<String>,
}

/// Function definition in bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function ID
    pub id: u32,
    /// Function name
    pub name: String,
    /// Function type ID
    pub type_id: u32,
    /// Parameter count
    pub param_count: u8,
    /// Local variable count
    pub local_count: u8,
    /// Maximum stack depth required
    pub max_stack_depth: u16,
    /// Required capabilities
    pub capabilities: Vec<Capability>,
    /// Declared effects
    pub effects: Vec<Effect>,
    /// Bytecode instructions
    pub instructions: Vec<Instruction>,
    /// Exception handlers
    pub exception_handlers: Vec<ExceptionHandler>,
    /// Debug information
    pub debug_info: Option<FunctionDebugInfo>,
    /// Business responsibility
    pub responsibility: Option<String>,
    /// Performance characteristics
    pub performance_characteristics: Vec<String>,
}

/// Global variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalDefinition {
    /// Global ID
    pub id: u32,
    /// Global name
    pub name: String,
    /// Type ID
    pub type_id: u32,
    /// Initial value (constant pool index)
    pub initial_value: Option<u32>,
    /// Whether this global is mutable
    pub mutable: bool,
    /// Business meaning
    pub business_meaning: Option<String>,
}

/// Exception handler definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExceptionHandler {
    /// Start instruction offset
    pub start_offset: u32,
    /// End instruction offset
    pub end_offset: u32,
    /// Handler instruction offset
    pub handler_offset: u32,
    /// Exception type ID (None for catch-all)
    pub exception_type: Option<u32>,
}

/// Module metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    /// Module name
    pub name: String,
    /// Module version
    pub version: String,
    /// Compilation timestamp
    pub compiled_at: chrono::DateTime<chrono::Utc>,
    /// Source file checksums
    pub source_checksums: HashMap<String, String>,
    /// Compilation flags
    pub compilation_flags: HashMap<String, String>,
    /// Dependencies
    pub dependencies: Vec<DependencyInfo>,
    /// AI metadata
    pub ai_metadata: Option<AIMetadata>,
}

/// Dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// Dependency name
    pub name: String,
    /// Dependency version
    pub version: String,
    /// Dependency checksum
    pub checksum: String,
}

/// AI metadata for bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Intent descriptions
    pub intents: HashMap<String, String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common patterns
    pub patterns: Vec<String>,
    /// Performance hints
    pub performance_hints: Vec<String>,
}

/// Debug information for the entire bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    /// Source file mapping
    pub source_files: Vec<SourceFile>,
    /// Line number mapping
    pub line_mappings: Vec<LineMapping>,
    /// Variable debug info
    pub variables: Vec<VariableDebugInfo>,
}

/// Source file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    /// File ID
    pub id: u32,
    /// File path
    pub path: String,
    /// File content hash
    pub hash: String,
}

/// Line number mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineMapping {
    /// Instruction offset
    pub instruction_offset: u32,
    /// Source file ID
    pub file_id: u32,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: u32,
}

/// Function debug information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDebugInfo {
    /// Source location
    pub source_location: SourceLocation,
    /// Local variable debug info
    pub locals: Vec<LocalVariableDebugInfo>,
    /// Instruction source mappings
    pub instruction_mappings: Vec<InstructionSourceMapping>,
}

/// Variable debug information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDebugInfo {
    /// Variable name
    pub name: String,
    /// Type ID
    pub type_id: u32,
    /// Scope start offset
    pub scope_start: u32,
    /// Scope end offset
    pub scope_end: u32,
}

/// Local variable debug information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVariableDebugInfo {
    /// Variable name
    pub name: String,
    /// Local slot index
    pub slot: u8,
    /// Type ID
    pub type_id: u32,
    /// Live range start
    pub live_start: u32,
    /// Live range end
    pub live_end: u32,
}

/// Source location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File ID
    pub file_id: u32,
    /// Start line
    pub start_line: u32,
    /// Start column
    pub start_column: u32,
    /// End line
    pub end_line: u32,
    /// End column
    pub end_column: u32,
}

/// Instruction source mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionSourceMapping {
    /// Instruction offset
    pub instruction_offset: u32,
    /// Source location
    pub source_location: SourceLocation,
}

impl PrismBytecode {
    /// Create a new bytecode instance
    pub fn new(name: String) -> Self {
        Self {
            magic: PRISM_BYTECODE_MAGIC,
            format_version: BYTECODE_FORMAT_VERSION,
            vm_version: VMVersion::CURRENT,
            id: Uuid::new_v4(),
            constants: ConstantPool::new(),
            types: Vec::new(),
            functions: Vec::new(),
            globals: Vec::new(),
            metadata: ModuleMetadata {
                name,
                version: "0.1.0".to_string(),
                compiled_at: chrono::Utc::now(),
                source_checksums: HashMap::new(),
                compilation_flags: HashMap::new(),
                dependencies: Vec::new(),
                ai_metadata: None,
            },
            debug_info: None,
        }
    }

    /// Validate the bytecode format
    pub fn validate(&self) -> VMResult<()> {
        // Check magic number
        if self.magic != PRISM_BYTECODE_MAGIC {
            return Err(crate::PrismVMError::InvalidBytecode {
                message: format!("Invalid magic number: 0x{:08X}", self.magic),
            });
        }

        // Check format version
        if self.format_version > BYTECODE_FORMAT_VERSION {
            return Err(crate::PrismVMError::InvalidBytecode {
                message: format!("Unsupported format version: {}", self.format_version),
            });
        }

        // Check VM version compatibility
        if !VMVersion::CURRENT.is_compatible_with(&self.vm_version) {
            return Err(crate::PrismVMError::InvalidBytecode {
                message: format!("Incompatible VM version: {}", self.vm_version),
            });
        }

        // Validate constant pool
        self.constants.validate()?;

        // Validate functions
        for function in &self.functions {
            self.validate_function(function)?;
        }

        Ok(())
    }

    /// Validate a function definition
    fn validate_function(&self, function: &FunctionDefinition) -> VMResult<()> {
        // Check that all referenced type IDs exist
        if function.type_id as usize >= self.types.len() {
            return Err(crate::PrismVMError::InvalidBytecode {
                message: format!("Function {} references invalid type ID: {}", function.name, function.type_id),
            });
        }

        // Validate instructions
        for (i, instruction) in function.instructions.iter().enumerate() {
            if let Err(e) = self.validate_instruction(instruction, function) {
                return Err(crate::PrismVMError::InvalidBytecode {
                    message: format!("Invalid instruction at offset {} in function {}: {}", i, function.name, e),
                });
            }
        }

        Ok(())
    }

    /// Validate an instruction
    fn validate_instruction(&self, instruction: &Instruction, function: &FunctionDefinition) -> VMResult<()> {
        use instructions::PrismOpcode;

        match instruction.opcode {
            PrismOpcode::LOAD_CONST(index) => {
                if index as usize >= self.constants.constants.len() {
                    return Err(crate::PrismVMError::InvalidBytecode {
                        message: format!("LOAD_CONST references invalid constant index: {}", index),
                    });
                }
            }
            PrismOpcode::LOAD_LOCAL(slot) | PrismOpcode::STORE_LOCAL(slot) => {
                if slot >= function.local_count {
                    return Err(crate::PrismVMError::InvalidBytecode {
                        message: format!("Local slot {} out of range (max: {})", slot, function.local_count),
                    });
                }
            }
            PrismOpcode::LOAD_GLOBAL(id) | PrismOpcode::STORE_GLOBAL(id) => {
                if !self.globals.iter().any(|g| g.id == id) {
                    return Err(crate::PrismVMError::InvalidBytecode {
                        message: format!("Global ID {} not found", id),
                    });
                }
            }
            _ => {} // Other instructions don't need validation here
        }

        Ok(())
    }

    /// Get function by name
    pub fn get_function(&self, name: &str) -> Option<&FunctionDefinition> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Get function by ID
    pub fn get_function_by_id(&self, id: u32) -> Option<&FunctionDefinition> {
        self.functions.iter().find(|f| f.id == id)
    }

    /// Get type by ID
    pub fn get_type(&self, id: u32) -> Option<&TypeDefinition> {
        self.types.iter().find(|t| t.id == id)
    }

    /// Get global by ID
    pub fn get_global(&self, id: u32) -> Option<&GlobalDefinition> {
        self.globals.iter().find(|g| g.id == id)
    }
} 