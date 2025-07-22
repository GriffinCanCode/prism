//! LLVM Type System
//!
//! This module handles LLVM type definitions, PIR type conversion,
//! and type-related utilities for the LLVM backend with semantic preservation.

use super::{LLVMResult, LLVMError};
use crate::backends::{
    PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRCompositeKind,
    PIRSemanticType, SecurityClassification, PIRFunctionType,
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

/// LLVM optimization levels aligned with LLVM standards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLVMOptimizationLevel {
    /// No optimization (-O0) - preserves all debug info and semantic metadata
    None,
    /// Basic optimization (-O1) - minimal optimizations with semantic preservation
    Basic,
    /// Aggressive optimization (-O2) - standard production optimization
    Aggressive,
    /// Maximum optimization (-O3) - aggressive optimization with inlining
    Maximum,
}

impl From<u8> for LLVMOptimizationLevel {
    fn from(level: u8) -> Self {
        match level {
            0 => Self::None,
            1 => Self::Basic,
            2 => Self::Aggressive,
            _ => Self::Maximum,
        }
    }
}

/// LLVM target architectures with comprehensive support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLVMTargetArch {
    /// x86-64 (AMD64) architecture
    X86_64,
    /// ARM64/AArch64 architecture
    AArch64,
    /// RISC-V 64-bit architecture
    RISCV64,
    /// WebAssembly (for comparison/hybrid compilation)
    WebAssembly,
    /// ARM 32-bit architecture
    ARM,
    /// PowerPC 64-bit architecture
    PowerPC64,
    /// MIPS 64-bit architecture
    MIPS64,
}

impl Default for LLVMTargetArch {
    fn default() -> Self {
        Self::X86_64 // Default to x86-64 for maximum compatibility
    }
}

impl fmt::Display for LLVMTargetArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64-unknown-linux-gnu"),
            Self::AArch64 => write!(f, "aarch64-unknown-linux-gnu"),
            Self::RISCV64 => write!(f, "riscv64-unknown-linux-gnu"),
            Self::WebAssembly => write!(f, "wasm32-unknown-wasi"),
            Self::ARM => write!(f, "arm-unknown-linux-gnueabihf"),
            Self::PowerPC64 => write!(f, "powerpc64-unknown-linux-gnu"),
            Self::MIPS64 => write!(f, "mips64-unknown-linux-gnu"),
        }
    }
}

/// LLVM types mapping PIR types to LLVM IR types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LLVMType {
    /// 1-bit integer (boolean)
    I1,
    /// 8-bit integer
    I8,
    /// 16-bit integer
    I16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 128-bit integer
    I128,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// Pointer to another type
    Ptr(Box<LLVMType>),
    /// Array type [N x Type]
    Array(Box<LLVMType>, usize),
    /// Vector type <N x Type>
    Vector(Box<LLVMType>, usize),
    /// Structure type { Type, Type, ... }
    Struct(Vec<LLVMType>),
    /// Function type RetType (ParamTypes...)
    Function(Box<LLVMType>, Vec<LLVMType>),
    /// Void type
    Void,
    /// Opaque pointer (LLVM 15+ style)
    OpaquePtr,
}

impl fmt::Display for LLVMType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I1 => write!(f, "i1"),
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::I128 => write!(f, "i128"),
            Self::F32 => write!(f, "float"),
            Self::F64 => write!(f, "double"),
            Self::Ptr(inner) => write!(f, "{}*", inner),
            Self::Array(inner, size) => write!(f, "[{} x {}]", size, inner),
            Self::Vector(inner, size) => write!(f, "<{} x {}>", size, inner),
            Self::Struct(fields) => {
                write!(f, "{{ ")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", field)?;
                }
                write!(f, " }}")
            }
            Self::Function(ret, params) => {
                write!(f, "{} (", ret)?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", param)?;
                }
                write!(f, ")")
            }
            Self::Void => write!(f, "void"),
            Self::OpaquePtr => write!(f, "ptr"),
        }
    }
}

/// LLVM type information with semantic context
#[derive(Debug, Clone)]
pub struct LLVMTypeInfo {
    /// LLVM type representation
    pub llvm_type: LLVMType,
    /// Original semantic type information
    pub semantic_info: Option<PIRSemanticType>,
    /// Generated validation functions
    pub validators: Vec<String>,
    /// Business rules associated with this type
    pub business_rules: Vec<String>,
    /// Security classification for access control
    pub security_level: SecurityClassification,
    /// Alignment requirements
    pub alignment: Option<u32>,
    /// Size in bytes (if known at compile time)
    pub size_bytes: Option<u64>,
}

/// LLVM function signature with semantic information
#[derive(Debug, Clone)]
pub struct LLVMFunctionSignature {
    /// Parameter types with semantic names
    pub params: Vec<(String, LLVMType)>,
    /// Return type
    pub return_type: LLVMType,
    /// Required capabilities for this function
    pub capabilities: Vec<String>,
    /// Effects produced by this function
    pub effects: Vec<String>,
    /// Calling convention
    pub calling_convention: LLVMCallingConvention,
    /// Function attributes
    pub attributes: Vec<LLVMFunctionAttribute>,
}

/// LLVM calling conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LLVMCallingConvention {
    /// C calling convention (default)
    C,
    /// Fast calling convention
    Fast,
    /// Cold calling convention (rarely called)
    Cold,
    /// Preserve most registers
    PreserveMost,
    /// Preserve all registers
    PreserveAll,
    /// Tail call optimized
    Tail,
}

impl Default for LLVMCallingConvention {
    fn default() -> Self {
        Self::C
    }
}

/// LLVM function attributes for optimization and behavior
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LLVMFunctionAttribute {
    /// Function does not access memory
    ReadNone,
    /// Function only reads memory
    ReadOnly,
    /// Function does not throw exceptions
    NoUnwind,
    /// Function should always be inlined
    AlwaysInline,
    /// Function should never be inlined
    NoInline,
    /// Function is optimized for size
    OptSize,
    /// Function is optimized for minimum size
    MinSize,
    /// Function returns without unwinding
    NoReturn,
    /// Function has no side effects
    Speculatable,
    /// Custom attribute with name and value
    Custom(String, Option<String>),
}

/// LLVM type converter for PIR types with semantic preservation
pub struct LLVMTypeConverter {
    /// Target architecture affecting type choices
    target_arch: LLVMTargetArch,
    /// Type cache for performance
    type_cache: HashMap<String, LLVMTypeInfo>,
    /// Pointer size for the target
    pointer_size: u32,
    /// Data layout information
    data_layout: String,
}

impl LLVMTypeConverter {
    /// Create a new type converter for the specified target
    pub fn new(target_arch: LLVMTargetArch) -> Self {
        let pointer_size = match target_arch {
            LLVMTargetArch::X86_64 | LLVMTargetArch::AArch64 | 
            LLVMTargetArch::RISCV64 | LLVMTargetArch::PowerPC64 | 
            LLVMTargetArch::MIPS64 => 64,
            LLVMTargetArch::ARM | LLVMTargetArch::WebAssembly => 32,
        };

        let data_layout = Self::generate_data_layout(target_arch);

        Self {
            target_arch,
            type_cache: HashMap::new(),
            pointer_size,
            data_layout,
        }
    }

    /// Generate data layout string for the target architecture
    fn generate_data_layout(target_arch: LLVMTargetArch) -> String {
        match target_arch {
            LLVMTargetArch::X86_64 => {
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128".to_string()
            }
            LLVMTargetArch::AArch64 => {
                "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string()
            }
            LLVMTargetArch::RISCV64 => {
                "e-m:e-p:64:64-i64:64-i128:128-n64-S128".to_string()
            }
            LLVMTargetArch::ARM => {
                "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string()
            }
            LLVMTargetArch::WebAssembly => {
                "e-m:e-p:32:32-i64:64-n32:64-S128".to_string()
            }
            LLVMTargetArch::PowerPC64 => {
                "e-m:e-i64:64-n32:64-S128-v256:256:256-v512:512:512".to_string()
            }
            LLVMTargetArch::MIPS64 => {
                "e-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".to_string()
            }
        }
    }

    /// Convert PIR type to LLVM type with semantic preservation
    pub fn convert_pir_type(&mut self, pir_type: &PIRTypeInfo) -> LLVMResult<LLVMTypeInfo> {
        let llvm_type = self.convert_pir_type_to_llvm(pir_type)?;
        
        Ok(LLVMTypeInfo {
            llvm_type,
            semantic_info: None, // Will be populated by caller if available
            validators: Vec::new(),
            business_rules: Vec::new(),
            security_level: SecurityClassification::Public, // Default, will be overridden
            alignment: self.get_type_alignment(&llvm_type),
            size_bytes: self.get_type_size(&llvm_type),
        })
    }

    /// Convert PIR semantic type to LLVM type with full semantic context
    pub fn convert_semantic_type(&mut self, semantic_type: &PIRSemanticType) -> LLVMResult<LLVMTypeInfo> {
        let mut type_info = self.convert_pir_type(&semantic_type.base_type)?;
        
        // Preserve semantic information
        type_info.semantic_info = Some(semantic_type.clone());
        type_info.security_level = semantic_type.security_classification.clone();
        
        // Generate business rule function names
        type_info.business_rules = semantic_type.business_rules.iter()
            .map(|rule| format!("validate_{}_{}", semantic_type.name.to_lowercase(), rule.name.to_lowercase()))
            .collect();
        
        // Generate validator function names
        type_info.validators = semantic_type.validation_predicates.iter()
            .map(|pred| format!("validate_{}_{}", semantic_type.name.to_lowercase(), pred.name.to_lowercase()))
            .collect();
        
        Ok(type_info)
    }

    /// Convert PIR type to LLVM type (internal implementation)
    fn convert_pir_type_to_llvm(&self, pir_type: &PIRTypeInfo) -> LLVMResult<LLVMType> {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => {
                match prim {
                    PIRPrimitiveType::Integer { signed: _, width } => {
                        match width {
                            1 => Ok(LLVMType::I1),
                            8 => Ok(LLVMType::I8),
                            16 => Ok(LLVMType::I16),
                            32 => Ok(LLVMType::I32),
                            64 => Ok(LLVMType::I64),
                            128 => Ok(LLVMType::I128),
                            _ => Err(LLVMError::TypeConversion {
                                message: format!("Unsupported integer width: {}", width),
                            }),
                        }
                    }
                    PIRPrimitiveType::Float { width } => {
                        match width {
                            32 => Ok(LLVMType::F32),
                            64 => Ok(LLVMType::F64),
                            _ => Err(LLVMError::TypeConversion {
                                message: format!("Unsupported float width: {}", width),
                            }),
                        }
                    }
                    PIRPrimitiveType::Boolean => Ok(LLVMType::I1),
                    PIRPrimitiveType::String => Ok(LLVMType::Ptr(Box::new(LLVMType::I8))),
                    PIRPrimitiveType::Unit => Ok(LLVMType::Void),
                }
            }
            PIRTypeInfo::Composite(comp) => {
                match comp.kind {
                    PIRCompositeKind::Struct => {
                        let field_types = comp.fields.iter()
                            .map(|f| self.convert_pir_type_to_llvm(&f.field_type))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(LLVMType::Struct(field_types))
                    }
                    PIRCompositeKind::Enum => {
                        // Enums are represented as integers with the appropriate size
                        Ok(LLVMType::I32) // Default to i32, could be optimized based on variant count
                    }
                    PIRCompositeKind::Union => {
                        // Unions are represented as the largest member type
                        let field_types = comp.fields.iter()
                            .map(|f| self.convert_pir_type_to_llvm(&f.field_type))
                            .collect::<Result<Vec<_>, _>>()?;
                        
                        // For simplicity, represent as a struct (LLVM doesn't have native unions)
                        Ok(LLVMType::Struct(field_types))
                    }
                    PIRCompositeKind::Tuple => {
                        let field_types = comp.fields.iter()
                            .map(|f| self.convert_pir_type_to_llvm(&f.field_type))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(LLVMType::Struct(field_types))
                    }
                }
            }
            PIRTypeInfo::Function(func) => {
                let return_type = Box::new(self.convert_pir_type_to_llvm(&func.return_type)?);
                let param_types = func.parameters.iter()
                    .map(|p| self.convert_pir_type_to_llvm(&p.param_type))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LLVMType::Function(return_type, param_types))
            }
            PIRTypeInfo::Generic(_) => {
                // Generics should be resolved before reaching this point
                Err(LLVMError::TypeConversion {
                    message: "Generic types must be resolved before LLVM conversion".to_string(),
                })
            }
            PIRTypeInfo::Effect(_) => {
                // Effects are represented as opaque pointers to effect handles
                Ok(LLVMType::OpaquePtr)
            }
        }
    }

    /// Get the alignment requirement for an LLVM type
    fn get_type_alignment(&self, llvm_type: &LLVMType) -> Option<u32> {
        match llvm_type {
            LLVMType::I1 => Some(1),
            LLVMType::I8 => Some(1),
            LLVMType::I16 => Some(2),
            LLVMType::I32 => Some(4),
            LLVMType::I64 => Some(8),
            LLVMType::I128 => Some(16),
            LLVMType::F32 => Some(4),
            LLVMType::F64 => Some(8),
            LLVMType::Ptr(_) | LLVMType::OpaquePtr => Some(self.pointer_size / 8),
            LLVMType::Array(inner, _) => self.get_type_alignment(inner),
            LLVMType::Vector(inner, _) => self.get_type_alignment(inner),
            LLVMType::Struct(fields) => {
                fields.iter()
                    .filter_map(|f| self.get_type_alignment(f))
                    .max()
            }
            LLVMType::Function(_, _) => Some(self.pointer_size / 8),
            LLVMType::Void => None,
        }
    }

    /// Get the size in bytes for an LLVM type (if known at compile time)
    fn get_type_size(&self, llvm_type: &LLVMType) -> Option<u64> {
        match llvm_type {
            LLVMType::I1 => Some(1), // Usually packed, but conservatively 1 byte
            LLVMType::I8 => Some(1),
            LLVMType::I16 => Some(2),
            LLVMType::I32 => Some(4),
            LLVMType::I64 => Some(8),
            LLVMType::I128 => Some(16),
            LLVMType::F32 => Some(4),
            LLVMType::F64 => Some(8),
            LLVMType::Ptr(_) | LLVMType::OpaquePtr => Some(self.pointer_size as u64 / 8),
            LLVMType::Array(inner, count) => {
                self.get_type_size(inner).map(|inner_size| inner_size * (*count as u64))
            }
            LLVMType::Vector(inner, count) => {
                self.get_type_size(inner).map(|inner_size| inner_size * (*count as u64))
            }
            LLVMType::Struct(fields) => {
                let mut total_size = 0u64;
                let mut current_offset = 0u64;
                
                for field in fields {
                    if let (Some(field_size), Some(field_align)) = 
                        (self.get_type_size(field), self.get_type_alignment(field)) {
                        // Align to field boundary
                        current_offset = (current_offset + field_align as u64 - 1) 
                            & !(field_align as u64 - 1);
                        current_offset += field_size;
                        total_size = current_offset;
                    } else {
                        return None; // Cannot determine size
                    }
                }
                
                Some(total_size)
            }
            LLVMType::Function(_, _) => Some(self.pointer_size as u64 / 8),
            LLVMType::Void => Some(0),
        }
    }

    /// Get the data layout string for this target
    pub fn get_data_layout(&self) -> &str {
        &self.data_layout
    }

    /// Get the target triple string
    pub fn get_target_triple(&self) -> String {
        self.target_arch.to_string()
    }

    /// Get pointer size in bits
    pub fn get_pointer_size_bits(&self) -> u32 {
        self.pointer_size
    }

    /// Clear the type cache (useful for testing or memory management)
    pub fn clear_cache(&mut self) {
        self.type_cache.clear();
    }
}

impl Clone for LLVMTypeConverter {
    fn clone(&self) -> Self {
        Self {
            target_arch: self.target_arch,
            type_cache: HashMap::new(), // Don't clone cache for performance
            pointer_size: self.pointer_size,
            data_layout: self.data_layout.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_converter_creation() {
        let converter = LLVMTypeConverter::new(LLVMTargetArch::X86_64);
        assert_eq!(converter.get_pointer_size_bits(), 64);
        assert!(converter.get_data_layout().contains("64:64"));
    }

    #[test]
    fn test_primitive_type_conversion() {
        let mut converter = LLVMTypeConverter::new(LLVMTargetArch::X86_64);
        
        let int32_type = PIRTypeInfo::Primitive(PIRPrimitiveType::Integer { signed: true, width: 32 });
        let result = converter.convert_pir_type(&int32_type).unwrap();
        assert_eq!(result.llvm_type, LLVMType::I32);
        assert_eq!(result.alignment, Some(4));
        assert_eq!(result.size_bytes, Some(4));
        
        let bool_type = PIRTypeInfo::Primitive(PIRPrimitiveType::Boolean);
        let result = converter.convert_pir_type(&bool_type).unwrap();
        assert_eq!(result.llvm_type, LLVMType::I1);
        
        let string_type = PIRTypeInfo::Primitive(PIRPrimitiveType::String);
        let result = converter.convert_pir_type(&string_type).unwrap();
        assert_eq!(result.llvm_type, LLVMType::Ptr(Box::new(LLVMType::I8)));
    }

    #[test]
    fn test_target_arch_display() {
        assert_eq!(LLVMTargetArch::X86_64.to_string(), "x86_64-unknown-linux-gnu");
        assert_eq!(LLVMTargetArch::AArch64.to_string(), "aarch64-unknown-linux-gnu");
        assert_eq!(LLVMTargetArch::RISCV64.to_string(), "riscv64-unknown-linux-gnu");
    }

    #[test]
    fn test_llvm_type_display() {
        assert_eq!(LLVMType::I32.to_string(), "i32");
        assert_eq!(LLVMType::F64.to_string(), "double");
        assert_eq!(LLVMType::Ptr(Box::new(LLVMType::I8)).to_string(), "i8*");
        assert_eq!(LLVMType::Array(Box::new(LLVMType::I32), 10).to_string(), "[10 x i32]");
        
        let struct_type = LLVMType::Struct(vec![LLVMType::I32, LLVMType::F64]);
        assert_eq!(struct_type.to_string(), "{ i32, double }");
    }

    #[test]
    fn test_optimization_level_conversion() {
        assert_eq!(LLVMOptimizationLevel::from(0), LLVMOptimizationLevel::None);
        assert_eq!(LLVMOptimizationLevel::from(1), LLVMOptimizationLevel::Basic);
        assert_eq!(LLVMOptimizationLevel::from(2), LLVMOptimizationLevel::Aggressive);
        assert_eq!(LLVMOptimizationLevel::from(3), LLVMOptimizationLevel::Maximum);
        assert_eq!(LLVMOptimizationLevel::from(99), LLVMOptimizationLevel::Maximum);
    }
} 