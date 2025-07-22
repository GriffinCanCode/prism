//! Python Type System with 2025 Features
//!
//! This module handles Python type definitions, PIR type conversion,
//! and type-related utilities following Python 3.12+ best practices.

use super::{PythonResult, PythonError};
use crate::backends::{
    PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRCompositeKind,
    PIRSemanticType, SecurityClassification,
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

/// Python target versions with different feature support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PythonTarget {
    /// Python 3.10 - Structural pattern matching, union types
    Python310,
    /// Python 3.11 - Exception groups, better error messages
    Python311,
    /// Python 3.12 - PEP 695 generic syntax, better typing
    Python312,
    /// Python 3.13+ - Latest features
    Python313Plus,
}

impl Default for PythonTarget {
    fn default() -> Self {
        Self::Python312 // Default to Python 3.12 for modern features
    }
}

impl fmt::Display for PythonTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Python310 => write!(f, "3.10"),
            Self::Python311 => write!(f, "3.11"),
            Self::Python312 => write!(f, "3.12"),
            Self::Python313Plus => write!(f, "3.13+"),
        }
    }
}

/// Python language features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonFeatures {
    /// Use type hints throughout generated code
    pub type_hints: bool,
    /// Generate dataclasses for semantic types
    pub dataclasses: bool,
    /// Use Pydantic models for validation
    pub pydantic_models: bool,
    /// Use async/await for effect handling
    pub async_await: bool,
    /// Use structural pattern matching (3.10+)
    pub pattern_matching: bool,
    /// Use PEP 695 generic syntax (3.12+)
    pub generic_syntax: bool,
    /// Use protocols for structural typing
    pub protocols: bool,
    /// Generate runtime type checking
    pub runtime_type_checking: bool,
    /// Use union types with | syntax
    pub union_types: bool,
    /// Generate docstrings with type information
    pub typed_docstrings: bool,
}

impl Default for PythonFeatures {
    fn default() -> Self {
        Self {
            type_hints: true,
            dataclasses: true,
            pydantic_models: true,
            async_await: true,
            pattern_matching: true,
            generic_syntax: true,
            protocols: true,
            runtime_type_checking: true,
            union_types: true,
            typed_docstrings: true,
        }
    }
}

/// Python type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonTypeConfig {
    /// Target Python version
    pub target_version: PythonTarget,
    /// Enabled features
    pub features: PythonFeatures,
    /// Use strict type checking
    pub strict_typing: bool,
    /// Generate type aliases for complex types
    pub type_aliases: bool,
    /// Use NewType for semantic types
    pub newtype_pattern: bool,
    /// Generate generic type parameters
    pub generic_parameters: bool,
}

impl Default for PythonTypeConfig {
    fn default() -> Self {
        Self {
            target_version: PythonTarget::default(),
            features: PythonFeatures::default(),
            strict_typing: true,
            type_aliases: true,
            newtype_pattern: true,
            generic_parameters: true,
        }
    }
}

/// Python types mapping PIR types to Python type annotations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PythonType {
    /// Basic types
    Int,
    Float,
    Bool,
    Str,
    None,
    
    /// Collection types
    List(Box<PythonType>),
    Dict(Box<PythonType>, Box<PythonType>),
    Set(Box<PythonType>),
    Tuple(Vec<PythonType>),
    
    /// Advanced types (3.10+)
    Union(Vec<PythonType>),
    Optional(Box<PythonType>),
    
    /// Generic types
    Generic(String, Vec<PythonType>),
    
    /// Callable types
    Callable {
        params: Vec<PythonType>,
        return_type: Box<PythonType>,
    },
    
    /// Protocol types
    Protocol(String),
    
    /// Custom types
    Custom(String),
    
    /// Semantic types (NewType wrapper)
    Semantic {
        name: String,
        base_type: Box<PythonType>,
    },
    
    /// Any type
    Any,
}

impl fmt::Display for PythonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Float => write!(f, "float"),
            Self::Bool => write!(f, "bool"),
            Self::Str => write!(f, "str"),
            Self::None => write!(f, "None"),
            Self::List(inner) => write!(f, "list[{}]", inner),
            Self::Dict(key, value) => write!(f, "dict[{}, {}]", key, value),
            Self::Set(inner) => write!(f, "set[{}]", inner),
            Self::Tuple(types) => {
                write!(f, "tuple[")?;
                for (i, typ) in types.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", typ)?;
                }
                write!(f, "]")
            }
            Self::Union(types) => {
                for (i, typ) in types.iter().enumerate() {
                    if i > 0 { write!(f, " | ")?; }
                    write!(f, "{}", typ)?;
                }
                Ok(())
            }
            Self::Optional(inner) => write!(f, "{} | None", inner),
            Self::Generic(name, params) => {
                write!(f, "{}", name)?;
                if !params.is_empty() {
                    write!(f, "[")?;
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", param)?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
            Self::Callable { params, return_type } => {
                write!(f, "Callable[[")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", param)?;
                }
                write!(f, "], {}]", return_type)
            }
            Self::Protocol(name) => write!(f, "{}", name),
            Self::Custom(name) => write!(f, "{}", name),
            Self::Semantic { name, .. } => write!(f, "{}", name),
            Self::Any => write!(f, "Any"),
        }
    }
}

/// Python type information with semantic context
#[derive(Debug, Clone)]
pub struct PythonTypeInfo {
    /// Python type representation
    pub python_type: PythonType,
    /// Original semantic type information
    pub semantic_info: Option<PIRSemanticType>,
    /// Generated validation functions
    pub validators: Vec<String>,
    /// Business rules associated with this type
    pub business_rules: Vec<String>,
    /// Import requirements
    pub imports: Vec<String>,
}

/// Python type converter for PIR types
pub struct PythonTypeConverter {
    /// Configuration
    config: PythonTypeConfig,
    /// Type cache for reuse
    type_cache: HashMap<String, PythonType>,
    /// Semantic type registry
    semantic_types: HashMap<String, PythonTypeInfo>,
}

impl PythonTypeConverter {
    /// Create a new type converter with configuration
    pub fn new(config: PythonTypeConfig) -> Self {
        Self {
            config,
            type_cache: HashMap::new(),
            semantic_types: HashMap::new(),
        }
    }

    /// Convert PIR type to Python type with semantic preservation
    pub fn convert_pir_type_to_python(&mut self, pir_type: &PIRTypeInfo) -> PythonResult<PythonType> {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => Ok(self.convert_primitive_type(prim)?),
            PIRTypeInfo::Composite(comp) => Ok(self.convert_composite_type(comp)?),
            PIRTypeInfo::Function(func) => Ok(self.convert_function_type(func)?),
            PIRTypeInfo::Generic(generic) => Ok(self.convert_generic_type(generic)?),
            PIRTypeInfo::Effect(effect) => Ok(self.convert_effect_type(effect)?),
        }
    }

    /// Convert PIR semantic type to Python semantic type
    pub fn convert_semantic_type_to_python(&mut self, semantic_type: &PIRSemanticType) -> PythonResult<PythonType> {
        let base_type = self.convert_pir_type_to_python(&semantic_type.base_type)?;
        
        if self.config.newtype_pattern {
            let semantic_python_type = PythonType::Semantic {
                name: semantic_type.name.clone(),
                base_type: Box::new(base_type.clone()),
            };
            
            // Store semantic type info
            let type_info = PythonTypeInfo {
                python_type: semantic_python_type.clone(),
                semantic_info: Some(semantic_type.clone()),
                validators: self.generate_validators_for_semantic_type(semantic_type),
                business_rules: semantic_type.business_rules.iter()
                    .map(|rule| rule.name.clone())
                    .collect(),
                imports: self.get_required_imports_for_semantic_type(semantic_type),
            };
            
            self.semantic_types.insert(semantic_type.name.clone(), type_info);
            Ok(semantic_python_type)
        } else {
            Ok(base_type)
        }
    }

    /// Convert primitive PIR type to Python type
    fn convert_primitive_type(&self, prim: &PIRPrimitiveType) -> PythonResult<PythonType> {
        match prim {
            PIRPrimitiveType::Integer { signed: _, width } => {
                match width {
                    1..=64 => Ok(PythonType::Int),
                    _ => {
                        // Very large integers - Python handles arbitrary precision
                        Ok(PythonType::Int)
                    }
                }
            }
            PIRPrimitiveType::Float { width } => {
                match width {
                    32 | 64 => Ok(PythonType::Float),
                    _ => Err(PythonError::TypeConversion {
                        message: format!("Unsupported float width: {}", width),
                    }),
                }
            }
            PIRPrimitiveType::Boolean => Ok(PythonType::Bool),
            PIRPrimitiveType::String => Ok(PythonType::Str),
            PIRPrimitiveType::Unit => Ok(PythonType::None),
        }
    }

    /// Convert composite PIR type to Python type
    fn convert_composite_type(&self, comp: &PIRCompositeType) -> PythonResult<PythonType> {
        match comp.kind {
            PIRCompositeKind::Struct => {
                // Generate a custom class type
                Ok(PythonType::Custom(comp.name.clone()))
            }
            PIRCompositeKind::Enum => {
                // Use Enum or Literal types
                Ok(PythonType::Custom(format!("{}Enum", comp.name)))
            }
            PIRCompositeKind::Union => {
                // Convert to Python Union type
                let field_types: Result<Vec<_>, _> = comp.fields.iter()
                    .map(|field| self.convert_pir_type_to_python(&field.field_type))
                    .collect();
                
                match field_types {
                    Ok(types) => Ok(PythonType::Union(types)),
                    Err(e) => Err(e),
                }
            }
            PIRCompositeKind::Array => {
                if let Some(element_field) = comp.fields.first() {
                    let element_type = self.convert_pir_type_to_python(&element_field.field_type)?;
                    Ok(PythonType::List(Box::new(element_type)))
                } else {
                    Ok(PythonType::List(Box::new(PythonType::Any)))
                }
            }
            PIRCompositeKind::Map => {
                if comp.fields.len() >= 2 {
                    let key_type = self.convert_pir_type_to_python(&comp.fields[0].field_type)?;
                    let value_type = self.convert_pir_type_to_python(&comp.fields[1].field_type)?;
                    Ok(PythonType::Dict(Box::new(key_type), Box::new(value_type)))
                } else {
                    Ok(PythonType::Dict(Box::new(PythonType::Any), Box::new(PythonType::Any)))
                }
            }
        }
    }

    /// Convert function type to Python callable
    fn convert_function_type(&self, func: &crate::backends::PIRFunctionType) -> PythonResult<PythonType> {
        let param_types: Result<Vec<_>, _> = func.parameters.iter()
            .map(|param| self.convert_pir_type_to_python(&param.param_type))
            .collect();
        
        let return_type = self.convert_pir_type_to_python(&func.return_type)?;
        
        match param_types {
            Ok(params) => Ok(PythonType::Callable {
                params,
                return_type: Box::new(return_type),
            }),
            Err(e) => Err(e),
        }
    }

    /// Convert generic type to Python generic
    fn convert_generic_type(&self, generic: &crate::backends::PIRGenericType) -> PythonResult<PythonType> {
        let type_params: Result<Vec<_>, _> = generic.type_parameters.iter()
            .map(|param| self.convert_pir_type_to_python(&param.constraint))
            .collect();
        
        match type_params {
            Ok(params) => Ok(PythonType::Generic(generic.name.clone(), params)),
            Err(e) => Err(e),
        }
    }

    /// Convert effect type to Python type
    fn convert_effect_type(&self, effect: &crate::backends::Effect) -> PythonResult<PythonType> {
        // Effects are represented as callable types that return Awaitable
        Ok(PythonType::Generic(
            "Awaitable".to_string(),
            vec![PythonType::Any],
        ))
    }

    /// Generate validators for semantic type
    fn generate_validators_for_semantic_type(&self, semantic_type: &PIRSemanticType) -> Vec<String> {
        let mut validators = Vec::new();
        
        // Generate business rule validators
        for rule in &semantic_type.business_rules {
            validators.push(format!("validate_{}", rule.name.to_lowercase()));
        }
        
        // Generate predicate validators
        for predicate in &semantic_type.validation_predicates {
            validators.push(format!("validate_{}", predicate.name.to_lowercase()));
        }
        
        validators
    }

    /// Get required imports for semantic type
    fn get_required_imports_for_semantic_type(&self, semantic_type: &PIRSemanticType) -> Vec<String> {
        let mut imports = Vec::new();
        
        // Always need typing for NewType
        if self.config.newtype_pattern {
            imports.push("from typing import NewType".to_string());
        }
        
        // Add Pydantic if using models
        if self.config.features.pydantic_models {
            imports.push("from pydantic import BaseModel, validator".to_string());
        }
        
        // Add dataclass if using dataclasses
        if self.config.features.dataclasses {
            imports.push("from dataclasses import dataclass".to_string());
        }
        
        // Add async imports for effects
        if !semantic_type.effects.is_empty() && self.config.features.async_await {
            imports.push("from typing import Awaitable".to_string());
            imports.push("import asyncio".to_string());
        }
        
        imports
    }

    /// Generate Python type annotation string
    pub fn generate_type_annotation(&self, python_type: &PythonType) -> String {
        python_type.to_string()
    }

    /// Check if type requires runtime validation
    pub fn requires_runtime_validation(&self, semantic_type: &PIRSemanticType) -> bool {
        !semantic_type.business_rules.is_empty() ||
        !semantic_type.validation_predicates.is_empty() ||
        matches!(semantic_type.security_classification, 
                SecurityClassification::Confidential | 
                SecurityClassification::Restricted | 
                SecurityClassification::TopSecret)
    }

    /// Get semantic type info
    pub fn get_semantic_type_info(&self, type_name: &str) -> Option<&PythonTypeInfo> {
        self.semantic_types.get(type_name)
    }

    /// Generate import statements for a set of types
    pub fn generate_imports(&self, types: &[PythonType]) -> Vec<String> {
        let mut imports = std::collections::HashSet::new();
        
        for python_type in types {
            match python_type {
                PythonType::List(_) | PythonType::Dict(_, _) | PythonType::Set(_) | PythonType::Tuple(_) => {
                    if self.config.target_version == PythonTarget::Python310 {
                        imports.insert("from typing import List, Dict, Set, Tuple".to_string());
                    }
                }
                PythonType::Union(_) => {
                    if self.config.target_version == PythonTarget::Python310 {
                        imports.insert("from typing import Union".to_string());
                    }
                }
                PythonType::Optional(_) => {
                    if self.config.target_version == PythonTarget::Python310 {
                        imports.insert("from typing import Optional".to_string());
                    }
                }
                PythonType::Generic(_, _) => {
                    imports.insert("from typing import Generic, TypeVar".to_string());
                }
                PythonType::Callable { .. } => {
                    imports.insert("from typing import Callable".to_string());
                }
                PythonType::Protocol(_) => {
                    imports.insert("from typing import Protocol".to_string());
                }
                PythonType::Any => {
                    imports.insert("from typing import Any".to_string());
                }
                PythonType::Semantic { .. } => {
                    imports.insert("from typing import NewType".to_string());
                }
                _ => {}
            }
        }
        
        imports.into_iter().collect()
    }
}

impl Default for PythonTypeConverter {
    fn default() -> Self {
        Self::new(PythonTypeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_target_display() {
        assert_eq!(PythonTarget::Python310.to_string(), "3.10");
        assert_eq!(PythonTarget::Python312.to_string(), "3.12");
    }

    #[test]
    fn test_python_type_display() {
        assert_eq!(PythonType::Int.to_string(), "int");
        assert_eq!(PythonType::List(Box::new(PythonType::Str)).to_string(), "list[str]");
        assert_eq!(
            PythonType::Dict(Box::new(PythonType::Str), Box::new(PythonType::Int)).to_string(),
            "dict[str, int]"
        );
        assert_eq!(
            PythonType::Union(vec![PythonType::Str, PythonType::Int]).to_string(),
            "str | int"
        );
    }

    #[test]
    fn test_primitive_type_conversion() {
        let converter = PythonTypeConverter::default();
        
        let int_type = PIRPrimitiveType::Integer { signed: true, width: 32 };
        let result = converter.convert_primitive_type(&int_type).unwrap();
        assert_eq!(result, PythonType::Int);
        
        let bool_type = PIRPrimitiveType::Boolean;
        let result = converter.convert_primitive_type(&bool_type).unwrap();
        assert_eq!(result, PythonType::Bool);
        
        let string_type = PIRPrimitiveType::String;
        let result = converter.convert_primitive_type(&string_type).unwrap();
        assert_eq!(result, PythonType::Str);
    }

    #[test]
    fn test_import_generation() {
        let converter = PythonTypeConverter::default();
        let types = vec![
            PythonType::List(Box::new(PythonType::Str)),
            PythonType::Union(vec![PythonType::Int, PythonType::Str]),
            PythonType::Any,
        ];
        
        let imports = converter.generate_imports(&types);
        assert!(!imports.is_empty());
    }

    #[test]
    fn test_python_features_default() {
        let features = PythonFeatures::default();
        assert!(features.type_hints);
        assert!(features.dataclasses);
        assert!(features.pydantic_models);
        assert!(features.async_await);
        assert!(features.pattern_matching);
    }
} 