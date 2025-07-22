//! WebAssembly Type System
//!
//! This module handles WebAssembly type definitions, PIR type conversion,
//! and type-related utilities for the WebAssembly backend.

use super::{WasmResult, WasmError};
use crate::backends::{
    PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRCompositeKind,
    PIRSemanticType, SecurityClassification,
};
use serde::{Serialize, Deserialize};
use std::fmt;

/// WebAssembly optimization levels aligned with industry standards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmOptimizationLevel {
    /// No optimization (-O0) - preserves all debug info
    None,
    /// Size optimization (-Os) - minimize binary size
    Size,
    /// Speed optimization (-O2) - optimize for performance
    Speed,
    /// Maximum optimization (-O3) - aggressive optimization
    Maximum,
}

impl From<u8> for WasmOptimizationLevel {
    fn from(level: u8) -> Self {
        match level {
            0 => Self::None,
            1 => Self::Size,
            2 => Self::Speed,
            _ => Self::Maximum,
        }
    }
}

/// WebAssembly runtime targets with specific capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmRuntimeTarget {
    /// Browser WebAssembly - limited system access
    Browser,
    /// WASI (WebAssembly System Interface) - full system interface
    WASI,
    /// Wasmtime runtime - high-performance server execution
    Wasmtime,
    /// Wasmer runtime - universal WebAssembly runtime
    Wasmer,
    /// Node.js WASM - JavaScript integration
    NodeJS,
}

impl Default for WasmRuntimeTarget {
    fn default() -> Self {
        Self::WASI // Default to WASI for maximum capability and security
    }
}

/// WebAssembly feature configuration following WebAssembly standards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmFeatures {
    /// Enable multi-value returns (WebAssembly 1.1)
    pub multi_value: bool,
    /// Enable bulk memory operations (WebAssembly 1.1)
    pub bulk_memory: bool,
    /// Enable SIMD instructions (WebAssembly 2.0)
    pub simd: bool,
    /// Enable threads (WebAssembly 2.0)
    pub threads: bool,
    /// Enable tail calls (WebAssembly 2.0)
    pub tail_calls: bool,
    /// Enable reference types (WebAssembly 2.0)
    pub reference_types: bool,
    /// Enable exception handling (WebAssembly 2.0)
    pub exception_handling: bool,
}

impl Default for WasmFeatures {
    fn default() -> Self {
        Self {
            multi_value: true,
            bulk_memory: true,
            simd: false, // Conservative default - requires explicit opt-in
            threads: false, // Security-sensitive - requires explicit opt-in
            tail_calls: true,
            reference_types: true,
            exception_handling: false, // Experimental feature
        }
    }
}

/// WebAssembly types mapping PIR types to WASM types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WasmType {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// External reference (WebAssembly 2.0)
    ExternRef,
    /// Function reference (WebAssembly 2.0)
    FuncRef,
    /// 128-bit vector (SIMD)
    V128,
}

impl fmt::Display for WasmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::ExternRef => write!(f, "externref"),
            Self::FuncRef => write!(f, "funcref"),
            Self::V128 => write!(f, "v128"),
        }
    }
}

/// WebAssembly function signature with semantic information
#[derive(Debug, Clone)]
pub struct WasmFunctionSignature {
    /// Parameter types with semantic names
    pub params: Vec<(String, WasmType)>,
    /// Return types (multi-value support)
    pub returns: Vec<WasmType>,
    /// Required capabilities for this function
    pub capabilities: Vec<String>,
    /// Effects produced by this function
    pub effects: Vec<String>,
}

/// WebAssembly type information with semantic context
#[derive(Debug, Clone)]
pub struct WasmTypeInfo {
    /// WASM type representation
    pub wasm_type: WasmType,
    /// Original semantic type information
    pub semantic_info: Option<PIRSemanticType>,
    /// Generated validation functions
    pub validators: Vec<String>,
    /// Business rules associated with this type
    pub business_rules: Vec<String>,
}

/// WebAssembly import declaration for runtime functions
#[derive(Debug, Clone)]
pub struct WasmImport {
    /// Module name (e.g., "wasi_snapshot_preview1")
    pub module: String,
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: WasmFunctionSignature,
    /// Human-readable description
    pub description: String,
}

/// WebAssembly type converter for PIR types
pub struct WasmTypeConverter {
    /// Feature configuration affecting type choices
    features: WasmFeatures,
    /// Runtime target affecting available types
    runtime_target: WasmRuntimeTarget,
}

impl WasmTypeConverter {
    /// Create a new type converter with configuration
    pub fn new(features: WasmFeatures, runtime_target: WasmRuntimeTarget) -> Self {
        Self {
            features,
            runtime_target,
        }
    }

    /// Convert PIR type to WebAssembly type with semantic preservation
    pub fn convert_pir_type_to_wasm(&self, pir_type: &PIRTypeInfo) -> WasmResult<WasmType> {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => self.convert_primitive_type(prim),
            PIRTypeInfo::Composite(comp) => self.convert_composite_type(comp),
            PIRTypeInfo::Function(_) => Ok(WasmType::FuncRef),
            PIRTypeInfo::Generic(_) => {
                // Generic types need runtime representation
                if self.features.reference_types {
                    Ok(WasmType::ExternRef)
                } else {
                    Ok(WasmType::I32) // Fallback to pointer
                }
            }
            PIRTypeInfo::Effect(_) => Ok(WasmType::I32), // Effect handles as i32
        }
    }

    /// Convert primitive PIR type to WASM type
    fn convert_primitive_type(&self, prim: &PIRPrimitiveType) -> WasmResult<WasmType> {
        match prim {
            PIRPrimitiveType::Integer { signed: _, width } => {
                match width {
                    1..=32 => Ok(WasmType::I32),
                    33..=64 => Ok(WasmType::I64),
                    _ => {
                        // Very large integers need special handling
                        if *width <= 128 && self.features.simd {
                            Ok(WasmType::V128) // Use SIMD for very large integers
                        } else {
                            Ok(WasmType::I64) // Default to i64 for large integers
                        }
                    }
                }
            }
            PIRPrimitiveType::Float { width } => {
                match width {
                    32 => Ok(WasmType::F32),
                    64 => Ok(WasmType::F64),
                    _ => {
                        return Err(WasmError::TypeConversion {
                            message: format!("Unsupported float width: {}", width),
                        });
                    }
                }
            }
            PIRPrimitiveType::Boolean => Ok(WasmType::I32), // Boolean as i32
            PIRPrimitiveType::String => Ok(WasmType::I32), // String as pointer
            PIRPrimitiveType::Unit => Ok(WasmType::I32), // Unit as i32 (could be optimized away)
        }
    }

    /// Convert composite PIR type to WASM type
    fn convert_composite_type(&self, comp: &PIRCompositeType) -> WasmResult<WasmType> {
        match comp.kind {
            PIRCompositeKind::Struct => Ok(WasmType::I32), // Struct as pointer
            PIRCompositeKind::Enum => {
                // Enums can be optimized based on variant count
                if comp.fields.len() <= 256 {
                    Ok(WasmType::I32) // Small enum as i32
                } else {
                    Ok(WasmType::I32) // Large enum still as pointer
                }
            }
            PIRCompositeKind::Union => Ok(WasmType::I32), // Union as pointer
            PIRCompositeKind::Tuple => {
                // Tuples with few elements might be passed by value
                if comp.fields.len() <= 2 {
                    Ok(WasmType::I32) // Small tuple as i32 or multiple values
                } else {
                    Ok(WasmType::I32) // Large tuple as pointer
                }
            }
        }
    }

    /// Get the type name for validation function generation
    pub fn get_type_name(&self, type_info: &PIRTypeInfo) -> String {
        match type_info {
            PIRTypeInfo::Primitive(prim) => {
                match prim {
                    PIRPrimitiveType::Integer { signed, width } => {
                        format!("{}int{}", if *signed { "i" } else { "u" }, width)
                    }
                    PIRPrimitiveType::Float { width } => format!("f{}", width),
                    PIRPrimitiveType::Boolean => "bool".to_string(),
                    PIRPrimitiveType::String => "string".to_string(),
                    PIRPrimitiveType::Unit => "unit".to_string(),
                }
            }
            PIRTypeInfo::Composite(comp) => {
                format!("{:?}_composite", comp.kind).to_lowercase()
            }
            PIRTypeInfo::Function(_) => "function".to_string(),
            PIRTypeInfo::Generic(gen) => gen.name.clone(),
            PIRTypeInfo::Effect(eff) => eff.name.clone(),
        }
    }

    /// Generate WASM type signature for function
    pub fn generate_function_signature(
        &self,
        params: &[(String, PIRTypeInfo)],
        return_type: &PIRTypeInfo,
    ) -> WasmResult<WasmFunctionSignature> {
        let mut wasm_params = Vec::new();
        
        for (name, param_type) in params {
            let wasm_type = self.convert_pir_type_to_wasm(param_type)?;
            wasm_params.push((name.clone(), wasm_type));
        }
        
        let return_wasm_type = self.convert_pir_type_to_wasm(return_type)?;
        let returns = if matches!(return_type, PIRTypeInfo::Primitive(PIRPrimitiveType::Unit)) {
            vec![] // Unit type has no return value
        } else {
            vec![return_wasm_type]
        };
        
        Ok(WasmFunctionSignature {
            params: wasm_params,
            returns,
            capabilities: Vec::new(), // Will be filled by caller
            effects: Vec::new(),      // Will be filled by caller
        })
    }

    /// Check if a type requires runtime validation
    pub fn requires_runtime_validation(&self, semantic_type: &PIRSemanticType) -> bool {
        // Types with business rules or security constraints need runtime validation
        !semantic_type.business_rules.is_empty() ||
        !semantic_type.validation_predicates.is_empty() ||
        matches!(semantic_type.security_classification, 
                SecurityClassification::Confidential | 
                SecurityClassification::Restricted | 
                SecurityClassification::TopSecret)
    }

    /// Generate validation function name for a semantic type
    pub fn generate_validation_function_name(&self, type_name: &str) -> String {
        format!("validate_{}", type_name.to_lowercase())
    }

    /// Get WASM memory representation size for a type
    pub fn get_memory_size(&self, wasm_type: WasmType) -> u32 {
        match wasm_type {
            WasmType::I32 => 4,
            WasmType::I64 => 8,
            WasmType::F32 => 4,
            WasmType::F64 => 8,
            WasmType::ExternRef => 4, // Pointer size
            WasmType::FuncRef => 4,   // Pointer size
            WasmType::V128 => 16,     // 128-bit vector
        }
    }

    /// Check if type supports the current feature set
    pub fn is_type_supported(&self, wasm_type: WasmType) -> bool {
        match wasm_type {
            WasmType::I32 | WasmType::I64 | WasmType::F32 | WasmType::F64 => true,
            WasmType::ExternRef | WasmType::FuncRef => self.features.reference_types,
            WasmType::V128 => self.features.simd,
        }
    }
}

impl Default for WasmTypeConverter {
    fn default() -> Self {
        Self::new(WasmFeatures::default(), WasmRuntimeTarget::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::PIRPrimitiveType;

    #[test]
    fn test_primitive_type_conversion() {
        let converter = WasmTypeConverter::default();
        
        let int32_type = PIRTypeInfo::Primitive(PIRPrimitiveType::Integer { signed: true, width: 32 });
        assert_eq!(converter.convert_pir_type_to_wasm(&int32_type).unwrap(), WasmType::I32);
        
        let float64_type = PIRTypeInfo::Primitive(PIRPrimitiveType::Float { width: 64 });
        assert_eq!(converter.convert_pir_type_to_wasm(&float64_type).unwrap(), WasmType::F64);
        
        let bool_type = PIRTypeInfo::Primitive(PIRPrimitiveType::Boolean);
        assert_eq!(converter.convert_pir_type_to_wasm(&bool_type).unwrap(), WasmType::I32);
    }

    #[test]
    fn test_type_name_generation() {
        let converter = WasmTypeConverter::default();
        
        let int_type = PIRTypeInfo::Primitive(PIRPrimitiveType::Integer { signed: true, width: 32 });
        assert_eq!(converter.get_type_name(&int_type), "iint32");
        
        let string_type = PIRTypeInfo::Primitive(PIRPrimitiveType::String);
        assert_eq!(converter.get_type_name(&string_type), "string");
    }

    #[test]
    fn test_memory_sizes() {
        let converter = WasmTypeConverter::default();
        
        assert_eq!(converter.get_memory_size(WasmType::I32), 4);
        assert_eq!(converter.get_memory_size(WasmType::I64), 8);
        assert_eq!(converter.get_memory_size(WasmType::F64), 8);
        assert_eq!(converter.get_memory_size(WasmType::V128), 16);
    }

    #[test]
    fn test_feature_support() {
        let features = WasmFeatures {
            reference_types: false,
            simd: false,
            ..WasmFeatures::default()
        };
        let converter = WasmTypeConverter::new(features, WasmRuntimeTarget::default());
        
        assert!(converter.is_type_supported(WasmType::I32));
        assert!(converter.is_type_supported(WasmType::F64));
        assert!(!converter.is_type_supported(WasmType::ExternRef));
        assert!(!converter.is_type_supported(WasmType::V128));
    }
} 