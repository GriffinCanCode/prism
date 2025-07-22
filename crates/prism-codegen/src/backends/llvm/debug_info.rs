//! LLVM Debug Information Generation
//!
//! This module handles debug information generation for LLVM IR,
//! including DWARF metadata, source location tracking, and debugging support.

use super::{LLVMResult, LLVMError};
use super::types::{LLVMTargetArch, LLVMOptimizationLevel};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Debug information configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLVMDebugConfig {
    /// Enable debug information generation
    pub enable_debug_info: bool,
    /// Debug information level
    pub debug_level: DebugLevel,
    /// DWARF version
    pub dwarf_version: u32,
    /// Source file path
    pub source_file: Option<String>,
    /// Source directory
    pub source_directory: Option<String>,
    /// Producer information
    pub producer: String,
    /// Include inline information
    pub include_inline_info: bool,
    /// Include variable locations
    pub include_variable_locations: bool,
    /// Include type information
    pub include_type_info: bool,
    /// Include function information
    pub include_function_info: bool,
    /// Optimization level for debug info
    pub optimization_level: LLVMOptimizationLevel,
    /// Target architecture
    pub target_arch: LLVMTargetArch,
}

impl Default for LLVMDebugConfig {
    fn default() -> Self {
        Self {
            enable_debug_info: true,
            debug_level: DebugLevel::Full,
            dwarf_version: 4,
            source_file: None,
            source_directory: None,
            producer: "Prism LLVM Backend".to_string(),
            include_inline_info: true,
            include_variable_locations: true,
            include_type_info: true,
            include_function_info: true,
            optimization_level: LLVMOptimizationLevel::None,
            target_arch: LLVMTargetArch::default(),
        }
    }
}

/// Debug information levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugLevel {
    /// No debug information
    None,
    /// Line numbers only
    LineNumbers,
    /// Basic debug information
    Basic,
    /// Full debug information
    Full,
    /// Extended debug information with optimization tracking
    Extended,
}

/// LLVM debug information generator
pub struct LLVMDebugInfo {
    /// Debug configuration
    config: LLVMDebugConfig,
    /// Debug metadata counter
    metadata_counter: u32,
    /// Compile unit metadata
    compile_unit: Option<DebugMetadata>,
    /// File metadata cache
    file_metadata: HashMap<String, DebugMetadata>,
    /// Type metadata cache
    type_metadata: HashMap<String, DebugMetadata>,
    /// Function metadata cache
    function_metadata: HashMap<String, DebugMetadata>,
    /// Variable metadata cache
    variable_metadata: HashMap<String, DebugMetadata>,
    /// Scope metadata stack
    scope_stack: Vec<DebugMetadata>,
}

/// Debug metadata representation
#[derive(Debug, Clone)]
pub struct DebugMetadata {
    /// Metadata ID
    pub id: u32,
    /// Metadata type
    pub metadata_type: DebugMetadataType,
    /// Metadata content
    pub content: String,
    /// Associated source location
    pub location: Option<SourceLocation>,
}

/// Debug metadata types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebugMetadataType {
    CompileUnit,
    File,
    Function,
    Variable,
    Type,
    Scope,
    Location,
    Expression,
}

/// Source location information
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based)
    pub column: u32,
    /// Scope metadata ID
    pub scope: Option<u32>,
}

/// Debug variable information
#[derive(Debug, Clone)]
pub struct DebugVariable {
    /// Variable name
    pub name: String,
    /// Variable type
    pub var_type: String,
    /// Source location
    pub location: SourceLocation,
    /// LLVM value
    pub llvm_value: String,
    /// Variable scope
    pub scope: u32,
    /// Whether variable is parameter
    pub is_parameter: bool,
    /// Parameter index (if parameter)
    pub parameter_index: Option<u32>,
}

/// Debug function information
#[derive(Debug, Clone)]
pub struct DebugFunction {
    /// Function name
    pub name: String,
    /// Mangled name
    pub mangled_name: Option<String>,
    /// Return type
    pub return_type: String,
    /// Parameter types
    pub parameter_types: Vec<String>,
    /// Source location
    pub location: SourceLocation,
    /// Function scope
    pub scope: u32,
    /// Whether function is local
    pub is_local: bool,
    /// Whether function is definition
    pub is_definition: bool,
}

/// Debug type information
#[derive(Debug, Clone)]
pub struct DebugType {
    /// Type name
    pub name: String,
    /// Type size in bits
    pub size_bits: u64,
    /// Type alignment in bits
    pub align_bits: u64,
    /// Type encoding
    pub encoding: TypeEncoding,
    /// Source location
    pub location: Option<SourceLocation>,
    /// Composite type members (if applicable)
    pub members: Vec<DebugTypeMember>,
}

/// Type encoding
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeEncoding {
    Void,
    Boolean,
    SignedInt,
    UnsignedInt,
    Float,
    Pointer,
    Array,
    Struct,
    Union,
    Enum,
    Function,
    Custom(String),
}

/// Debug type member (for composite types)
#[derive(Debug, Clone)]
pub struct DebugTypeMember {
    /// Member name
    pub name: String,
    /// Member type
    pub member_type: String,
    /// Member offset in bits
    pub offset_bits: u64,
    /// Member size in bits
    pub size_bits: u64,
    /// Source location
    pub location: Option<SourceLocation>,
}

impl LLVMDebugInfo {
    /// Create new debug information generator
    pub fn new(config: LLVMDebugConfig) -> Self {
        Self {
            config,
            metadata_counter: 0,
            compile_unit: None,
            file_metadata: HashMap::new(),
            type_metadata: HashMap::new(),
            function_metadata: HashMap::new(),
            variable_metadata: HashMap::new(),
            scope_stack: Vec::new(),
        }
    }

    /// Initialize debug information
    pub fn initialize(&mut self) -> LLVMResult<Vec<String>> {
        if !self.config.enable_debug_info {
            return Ok(Vec::new());
        }

        let mut metadata = Vec::new();

        // Create compile unit
        let compile_unit = self.create_compile_unit()?;
        metadata.push(compile_unit.content.clone());
        self.compile_unit = Some(compile_unit);

        // Add module flags
        metadata.push(format!(
            "!llvm.module.flags = !{{!{}, !{}, !{}}}",
            self.next_metadata_id(),
            self.next_metadata_id(),
            self.next_metadata_id()
        ));

        metadata.push(format!(
            "!{} = !{{i32 2, !\"Dwarf Version\", i32 {}}}",
            self.metadata_counter - 2,
            self.config.dwarf_version
        ));

        metadata.push(format!(
            "!{} = !{{i32 2, !\"Debug Info Version\", i32 3}}",
            self.metadata_counter - 1
        ));

        metadata.push(format!(
            "!{} = !{{i32 1, !\"wchar_size\", i32 4}}",
            self.metadata_counter
        ));

        // Add debug info identifier
        if let Some(ref compile_unit) = self.compile_unit {
            metadata.push(format!("!llvm.dbg.cu = !{{{}}}", compile_unit.id));
        }

        Ok(metadata)
    }

    /// Create compile unit metadata
    fn create_compile_unit(&mut self) -> LLVMResult<DebugMetadata> {
        let id = self.next_metadata_id();
        
        let source_file = self.config.source_file.as_deref().unwrap_or("unknown");
        let source_dir = self.config.source_directory.as_deref().unwrap_or(".");

        // Create file metadata first
        let file_metadata = self.create_file_metadata(source_file, source_dir)?;

        let content = format!(
            "!{} = distinct !DICompileUnit(language: DW_LANG_C, file: !{}, producer: \"{}\", isOptimized: {}, runtimeVersion: 0, emissionKind: FullDebug, enums: !{}, retainedTypes: !{}, globals: !{}, imports: !{})",
            id,
            file_metadata.id,
            self.config.producer,
            self.config.optimization_level != LLVMOptimizationLevel::None,
            self.next_metadata_id(), // enums
            self.next_metadata_id(), // retained types
            self.next_metadata_id(), // globals
            self.next_metadata_id()  // imports
        );

        // Create empty arrays for collections
        let arrays = vec![
            format!("!{} = !{{}}", self.metadata_counter - 3), // enums
            format!("!{} = !{{}}", self.metadata_counter - 2), // retained types
            format!("!{} = !{{}}", self.metadata_counter - 1), // globals
            format!("!{} = !{{}}", self.metadata_counter),     // imports
        ];

        let mut full_content = vec![content];
        full_content.extend(arrays);

        Ok(DebugMetadata {
            id,
            metadata_type: DebugMetadataType::CompileUnit,
            content: full_content.join("\n"),
            location: None,
        })
    }

    /// Create file metadata
    pub fn create_file_metadata(&mut self, filename: &str, directory: &str) -> LLVMResult<DebugMetadata> {
        let cache_key = format!("{}:{}", directory, filename);
        
        if let Some(cached) = self.file_metadata.get(&cache_key) {
            return Ok(cached.clone());
        }

        let id = self.next_metadata_id();
        let content = format!(
            "!{} = !DIFile(filename: \"{}\", directory: \"{}\")",
            id, filename, directory
        );

        let metadata = DebugMetadata {
            id,
            metadata_type: DebugMetadataType::File,
            content,
            location: None,
        };

        self.file_metadata.insert(cache_key, metadata.clone());
        Ok(metadata)
    }

    /// Create function debug metadata
    pub fn create_function_metadata(&mut self, function: &DebugFunction) -> LLVMResult<DebugMetadata> {
        if let Some(cached) = self.function_metadata.get(&function.name) {
            return Ok(cached.clone());
        }

        let id = self.next_metadata_id();
        
        // Create file metadata for function location
        let file_metadata = self.create_file_metadata(&function.location.file, ".")?;
        
        // Create function type metadata
        let function_type_id = self.create_function_type_metadata(
            &function.return_type,
            &function.parameter_types
        )?;

        let scope_id = if let Some(ref compile_unit) = self.compile_unit {
            compile_unit.id
        } else {
            return Err(LLVMError::MissingCompileUnit);
        };

        let content = format!(
            "!{} = distinct !DISubprogram(name: \"{}\", linkageName: \"{}\", scope: !{}, file: !{}, line: {}, type: !{}, scopeLine: {}, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !{}, retainedNodes: !{})",
            id,
            function.name,
            function.mangled_name.as_deref().unwrap_or(&function.name),
            scope_id,
            file_metadata.id,
            function.location.line,
            function_type_id,
            function.location.line,
            scope_id,
            self.next_metadata_id() // retained nodes
        );

        // Create empty retained nodes array
        let retained_nodes = format!("!{} = !{{}}", self.metadata_counter);
        let full_content = format!("{}\n{}", content, retained_nodes);

        let metadata = DebugMetadata {
            id,
            metadata_type: DebugMetadataType::Function,
            content: full_content,
            location: Some(function.location.clone()),
        };

        self.function_metadata.insert(function.name.clone(), metadata.clone());
        Ok(metadata)
    }

    /// Create function type metadata
    fn create_function_type_metadata(&mut self, return_type: &str, param_types: &[String]) -> LLVMResult<u32> {
        let id = self.next_metadata_id();
        
        // Create return type metadata
        let return_type_id = self.create_basic_type_metadata(return_type)?;
        
        // Create parameter type metadata
        let mut param_type_ids = vec![return_type_id.to_string()];
        for param_type in param_types {
            let param_type_id = self.create_basic_type_metadata(param_type)?;
            param_type_ids.push(param_type_id.to_string());
        }

        let types_array_id = self.next_metadata_id();
        let types_array = format!(
            "!{} = !{{{}}}",
            types_array_id,
            param_type_ids.join(", !")
        );

        let function_type = format!(
            "!{} = !DISubroutineType(types: !{})",
            id,
            types_array_id
        );

        // We need to store these for later output
        // This is a simplified version - in practice, we'd need better metadata management

        Ok(id)
    }

    /// Create basic type metadata
    fn create_basic_type_metadata(&mut self, type_name: &str) -> LLVMResult<u32> {
        if let Some(cached) = self.type_metadata.get(type_name) {
            return Ok(cached.id);
        }

        let id = self.next_metadata_id();
        let (size, encoding) = self.get_type_info(type_name);

        let content = format!(
            "!{} = !DIBasicType(name: \"{}\", size: {}, encoding: {})",
            id, type_name, size, encoding
        );

        let metadata = DebugMetadata {
            id,
            metadata_type: DebugMetadataType::Type,
            content,
            location: None,
        };

        self.type_metadata.insert(type_name.to_string(), metadata);
        Ok(id)
    }

    /// Create variable debug metadata
    pub fn create_variable_metadata(&mut self, variable: &DebugVariable) -> LLVMResult<DebugMetadata> {
        let cache_key = format!("{}:{}", variable.scope, variable.name);
        
        if let Some(cached) = self.variable_metadata.get(&cache_key) {
            return Ok(cached.clone());
        }

        let id = self.next_metadata_id();
        
        // Create type metadata for variable
        let type_id = self.create_basic_type_metadata(&variable.var_type)?;
        
        // Create file metadata
        let file_metadata = self.create_file_metadata(&variable.location.file, ".")?;

        let tag = if variable.is_parameter {
            "DW_TAG_arg_variable"
        } else {
            "DW_TAG_auto_variable"
        };

        let mut content = format!(
            "!{} = !DILocalVariable(name: \"{}\", scope: !{}, file: !{}, line: {}, type: !{}, flags: DIFlagZero",
            id,
            variable.name,
            variable.scope,
            file_metadata.id,
            variable.location.line,
            type_id
        );

        if let Some(param_index) = variable.parameter_index {
            content.push_str(&format!(", arg: {}", param_index));
        }

        content.push(')');

        let metadata = DebugMetadata {
            id,
            metadata_type: DebugMetadataType::Variable,
            content,
            location: Some(variable.location.clone()),
        };

        self.variable_metadata.insert(cache_key, metadata.clone());
        Ok(metadata)
    }

    /// Create location metadata
    pub fn create_location_metadata(&mut self, location: &SourceLocation) -> LLVMResult<String> {
        if !self.config.enable_debug_info {
            return Ok(String::new());
        }

        let scope_id = location.scope.unwrap_or_else(|| {
            if let Some(ref compile_unit) = self.compile_unit {
                compile_unit.id
            } else {
                0
            }
        });

        Ok(format!(
            "!DILocation(line: {}, column: {}, scope: !{})",
            location.line,
            location.column,
            scope_id
        ))
    }

    /// Generate debug intrinsic call
    pub fn generate_debug_declare(&self, variable: &DebugVariable, metadata_id: u32) -> String {
        if !self.config.enable_debug_info {
            return String::new();
        }

        format!(
            "call void @llvm.dbg.declare(metadata {} {}, metadata !{}, metadata !DIExpression()), !dbg !{}",
            variable.var_type,
            variable.llvm_value,
            metadata_id,
            metadata_id
        )
    }

    /// Generate debug value intrinsic call
    pub fn generate_debug_value(&self, variable: &DebugVariable, metadata_id: u32, value: &str) -> String {
        if !self.config.enable_debug_info {
            return String::new();
        }

        format!(
            "call void @llvm.dbg.value(metadata {} {}, metadata !{}, metadata !DIExpression()), !dbg !{}",
            variable.var_type,
            value,
            metadata_id,
            metadata_id
        )
    }

    /// Generate instruction with debug location
    pub fn add_debug_location(&self, instruction: &str, location: &SourceLocation) -> String {
        if !self.config.enable_debug_info {
            return instruction.to_string();
        }

        let scope_id = location.scope.unwrap_or_else(|| {
            if let Some(ref compile_unit) = self.compile_unit {
                compile_unit.id
            } else {
                0
            }
        });

        format!(
            "{}, !dbg !DILocation(line: {}, column: {}, scope: !{})",
            instruction,
            location.line,
            location.column,
            scope_id
        )
    }

    /// Get all generated metadata
    pub fn get_all_metadata(&self) -> Vec<String> {
        let mut metadata = Vec::new();

        // Add compile unit
        if let Some(ref compile_unit) = self.compile_unit {
            metadata.push(compile_unit.content.clone());
        }

        // Add file metadata
        for file_meta in self.file_metadata.values() {
            metadata.push(file_meta.content.clone());
        }

        // Add type metadata
        for type_meta in self.type_metadata.values() {
            metadata.push(type_meta.content.clone());
        }

        // Add function metadata
        for func_meta in self.function_metadata.values() {
            metadata.push(func_meta.content.clone());
        }

        // Add variable metadata
        for var_meta in self.variable_metadata.values() {
            metadata.push(var_meta.content.clone());
        }

        metadata
    }

    /// Helper methods
    fn next_metadata_id(&mut self) -> u32 {
        let id = self.metadata_counter;
        self.metadata_counter += 1;
        id
    }

    fn get_type_info(&self, type_name: &str) -> (u32, &'static str) {
        match type_name {
            "void" => (0, "DW_ATE_void"),
            "i1" => (1, "DW_ATE_boolean"),
            "i8" => (8, "DW_ATE_signed"),
            "i16" => (16, "DW_ATE_signed"),
            "i32" => (32, "DW_ATE_signed"),
            "i64" => (64, "DW_ATE_signed"),
            "f32" => (32, "DW_ATE_float"),
            "f64" => (64, "DW_ATE_float"),
            _ if type_name.ends_with('*') => (64, "DW_ATE_address"), // Assuming 64-bit pointers
            _ => (0, "DW_ATE_void"), // Unknown type
        }
    }

    /// Get debug configuration
    pub fn get_config(&self) -> &LLVMDebugConfig {
        &self.config
    }

    /// Check if debug info is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enable_debug_info
    }

    /// Get current metadata counter
    pub fn get_metadata_counter(&self) -> u32 {
        self.metadata_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_info_creation() {
        let config = LLVMDebugConfig::default();
        let debug_info = LLVMDebugInfo::new(config);
        
        assert!(debug_info.config.enable_debug_info);
        assert_eq!(debug_info.config.debug_level, DebugLevel::Full);
        assert_eq!(debug_info.metadata_counter, 0);
    }

    #[test]
    fn test_debug_info_initialization() {
        let config = LLVMDebugConfig {
            source_file: Some("test.prism".to_string()),
            source_directory: Some("/tmp".to_string()),
            ..Default::default()
        };
        
        let mut debug_info = LLVMDebugInfo::new(config);
        let metadata = debug_info.initialize().unwrap();
        
        assert!(!metadata.is_empty());
        assert!(debug_info.compile_unit.is_some());
    }

    #[test]
    fn test_file_metadata_creation() {
        let config = LLVMDebugConfig::default();
        let mut debug_info = LLVMDebugInfo::new(config);
        
        let file_metadata = debug_info.create_file_metadata("test.prism", "/tmp").unwrap();
        
        assert_eq!(file_metadata.metadata_type, DebugMetadataType::File);
        assert!(file_metadata.content.contains("test.prism"));
        assert!(file_metadata.content.contains("/tmp"));
    }

    #[test]
    fn test_function_metadata_creation() {
        let config = LLVMDebugConfig::default();
        let mut debug_info = LLVMDebugInfo::new(config);
        
        // Initialize first to create compile unit
        let _ = debug_info.initialize().unwrap();
        
        let function = DebugFunction {
            name: "test_function".to_string(),
            mangled_name: None,
            return_type: "i32".to_string(),
            parameter_types: vec!["i32".to_string(), "i32".to_string()],
            location: SourceLocation {
                file: "test.prism".to_string(),
                line: 10,
                column: 1,
                scope: None,
            },
            scope: 0,
            is_local: false,
            is_definition: true,
        };
        
        let func_metadata = debug_info.create_function_metadata(&function).unwrap();
        
        assert_eq!(func_metadata.metadata_type, DebugMetadataType::Function);
        assert!(func_metadata.content.contains("test_function"));
    }

    #[test]
    fn test_variable_metadata_creation() {
        let config = LLVMDebugConfig::default();
        let mut debug_info = LLVMDebugInfo::new(config);
        
        let variable = DebugVariable {
            name: "test_var".to_string(),
            var_type: "i32".to_string(),
            location: SourceLocation {
                file: "test.prism".to_string(),
                line: 5,
                column: 10,
                scope: None,
            },
            llvm_value: "%test_var".to_string(),
            scope: 1,
            is_parameter: false,
            parameter_index: None,
        };
        
        let var_metadata = debug_info.create_variable_metadata(&variable).unwrap();
        
        assert_eq!(var_metadata.metadata_type, DebugMetadataType::Variable);
        assert!(var_metadata.content.contains("test_var"));
    }

    #[test]
    fn test_debug_intrinsic_generation() {
        let config = LLVMDebugConfig::default();
        let debug_info = LLVMDebugInfo::new(config);
        
        let variable = DebugVariable {
            name: "test_var".to_string(),
            var_type: "i32".to_string(),
            location: SourceLocation {
                file: "test.prism".to_string(),
                line: 5,
                column: 10,
                scope: None,
            },
            llvm_value: "%test_var".to_string(),
            scope: 1,
            is_parameter: false,
            parameter_index: None,
        };
        
        let declare_call = debug_info.generate_debug_declare(&variable, 42);
        assert!(declare_call.contains("llvm.dbg.declare"));
        assert!(declare_call.contains("%test_var"));
        
        let value_call = debug_info.generate_debug_value(&variable, 42, "%new_value");
        assert!(value_call.contains("llvm.dbg.value"));
        assert!(value_call.contains("%new_value"));
    }

    #[test]
    fn test_debug_location_addition() {
        let config = LLVMDebugConfig::default();
        let debug_info = LLVMDebugInfo::new(config);
        
        let location = SourceLocation {
            file: "test.prism".to_string(),
            line: 10,
            column: 5,
            scope: Some(1),
        };
        
        let instruction = "  %result = add i32 %a, %b";
        let with_debug = debug_info.add_debug_location(instruction, &location);
        
        assert!(with_debug.contains("!dbg !DILocation"));
        assert!(with_debug.contains("line: 10"));
        assert!(with_debug.contains("column: 5"));
    }

    #[test]
    fn test_type_info_extraction() {
        let config = LLVMDebugConfig::default();
        let debug_info = LLVMDebugInfo::new(config);
        
        let (size, encoding) = debug_info.get_type_info("i32");
        assert_eq!(size, 32);
        assert_eq!(encoding, "DW_ATE_signed");
        
        let (size, encoding) = debug_info.get_type_info("f64");
        assert_eq!(size, 64);
        assert_eq!(encoding, "DW_ATE_float");
        
        let (size, encoding) = debug_info.get_type_info("i32*");
        assert_eq!(size, 64); // Assuming 64-bit pointers
        assert_eq!(encoding, "DW_ATE_address");
    }

    #[test]
    fn test_disabled_debug_info() {
        let config = LLVMDebugConfig {
            enable_debug_info: false,
            ..Default::default()
        };
        
        let mut debug_info = LLVMDebugInfo::new(config);
        let metadata = debug_info.initialize().unwrap();
        
        assert!(metadata.is_empty());
        assert!(!debug_info.is_enabled());
    }
} 