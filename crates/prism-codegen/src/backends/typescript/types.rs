//! TypeScript Type System with 2025 Features
//!
//! This module handles TypeScript type generation with modern 2025 features including:
//! - Branded types for semantic safety
//! - Template literal types for dynamic string types
//! - Enhanced control flow analysis support
//! - Utility types for advanced type manipulation
//! - Modern ESM type imports/exports
//! - Discriminated unions for state management

use super::{TypeScriptResult, TypeScriptError, TypeScriptTarget, TypeScriptFeatures};
use crate::backends::{
    PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRCompositeKind,
    PIRSemanticType, SecurityClassification,
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// TypeScript type representation with 2025 features
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeScriptType {
    /// Primitive types
    String,
    Number,
    Boolean,
    Undefined,
    Null,
    Unknown,
    Never,
    Void,
    
    /// Modern TypeScript types
    BigInt,
    Symbol,
    
    /// Template literal types
    TemplateLiteral {
        parts: Vec<String>,
        placeholders: Vec<TypeScriptType>,
    },
    
    /// Branded types for semantic safety
    Branded {
        base_type: Box<TypeScriptType>,
        brand: String,
    },
    
    /// Literal types
    StringLiteral(String),
    NumberLiteral(f64),
    BooleanLiteral(bool),
    
    /// Union types
    Union(Vec<TypeScriptType>),
    
    /// Intersection types
    Intersection(Vec<TypeScriptType>),
    
    /// Object types
    Object {
        properties: HashMap<String, ObjectProperty>,
        index_signature: Option<IndexSignature>,
    },
    
    /// Array types
    Array(Box<TypeScriptType>),
    
    /// Tuple types
    Tuple(Vec<TypeScriptType>),
    
    /// Function types
    Function {
        parameters: Vec<Parameter>,
        return_type: Box<TypeScriptType>,
        is_async: bool,
    },
    
    /// Generic types
    Generic {
        name: String,
        type_parameters: Vec<TypeScriptType>,
    },
    
    /// Conditional types
    Conditional {
        check_type: Box<TypeScriptType>,
        extends_type: Box<TypeScriptType>,
        true_type: Box<TypeScriptType>,
        false_type: Box<TypeScriptType>,
    },
    
    /// Mapped types
    Mapped {
        key_type: Box<TypeScriptType>,
        value_type: Box<TypeScriptType>,
        modifiers: MappedTypeModifiers,
    },
    
    /// Utility types
    Utility {
        kind: UtilityTypeKind,
        type_args: Vec<TypeScriptType>,
    },
    
    /// Type reference
    Reference(String),
}

/// Object property definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectProperty {
    pub property_type: TypeScriptType,
    pub optional: bool,
    pub readonly: bool,
}

/// Index signature for objects
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexSignature {
    pub key_type: TypeScriptType,
    pub value_type: TypeScriptType,
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub parameter_type: TypeScriptType,
    pub optional: bool,
    pub default_value: Option<String>,
}

/// Mapped type modifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MappedTypeModifiers {
    pub readonly: Option<bool>,
    pub optional: Option<bool>,
}

/// Utility type kinds
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UtilityTypeKind {
    Partial,
    Required,
    Readonly,
    Pick,
    Omit,
    Exclude,
    Extract,
    NonNullable,
    ReturnType,
    InstanceType,
    Parameters,
    ConstructorParameters,
    Record,
    Awaited,
    Capitalize,
    Uncapitalize,
    Uppercase,
    Lowercase,
    /// Custom utility type
    Custom(String),
}

/// TypeScript type converter with 2025 features
#[derive(Debug, Clone)]
pub struct TypeScriptTypeConverter {
    /// TypeScript features to use
    features: TypeScriptFeatures,
    /// Target environment
    target: TypeScriptTarget,
    /// Brand registry for semantic types
    brand_registry: HashMap<String, String>,
    /// Template literal cache
    template_cache: HashMap<String, TypeScriptType>,
}

impl TypeScriptTypeConverter {
    /// Create a new TypeScript type converter
    pub fn new(features: TypeScriptFeatures, target: TypeScriptTarget) -> Self {
        Self {
            features,
            target,
            brand_registry: HashMap::new(),
            template_cache: HashMap::new(),
        }
    }

    /// Convert PIR type to TypeScript type with 2025 features
    pub fn convert_pir_type_to_typescript(&mut self, pir_type: &PIRTypeInfo) -> TypeScriptResult<TypeScriptType> {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => self.convert_primitive_type(prim),
            PIRTypeInfo::Composite(comp) => self.convert_composite_type(comp),
            PIRTypeInfo::Function(func) => self.convert_function_type(func),
            PIRTypeInfo::Generic(generic) => self.convert_generic_type(generic),
            PIRTypeInfo::Effect(effect) => self.convert_effect_type(effect),
        }
    }

    /// Convert PIR semantic type to branded TypeScript type
    pub fn convert_semantic_type_to_branded(&mut self, semantic_type: &PIRSemanticType) -> TypeScriptResult<TypeScriptType> {
        let base_type = self.convert_pir_type_to_typescript(&semantic_type.base_type)?;
        
        if self.features.branded_types {
            let brand = self.generate_semantic_brand(&semantic_type.name, &semantic_type.domain);
            self.brand_registry.insert(semantic_type.name.clone(), brand.clone());
            
            Ok(TypeScriptType::Branded {
                base_type: Box::new(base_type),
                brand,
            })
        } else {
            Ok(base_type)
        }
    }

    /// Generate template literal type for capability names
    pub fn generate_capability_template_literal(&mut self, domain: &str) -> TypeScriptResult<TypeScriptType> {
        if !self.features.template_literal_types {
            return Ok(TypeScriptType::String);
        }

        let template = format!("capability:{}", domain.to_lowercase());
        if let Some(cached) = self.template_cache.get(&template) {
            return Ok(cached.clone());
        }

        let template_type = TypeScriptType::TemplateLiteral {
            parts: vec!["capability:".to_string(), "".to_string()],
            placeholders: vec![TypeScriptType::Reference(format!("{}Domain", domain))],
        };

        self.template_cache.insert(template, template_type.clone());
        Ok(template_type)
    }

    /// Generate utility type for advanced type manipulation
    pub fn generate_utility_type(&self, kind: UtilityTypeKind, type_args: Vec<TypeScriptType>) -> TypeScriptType {
        TypeScriptType::Utility { kind, type_args }
    }

    /// Generate discriminated union for state management
    pub fn generate_discriminated_union(&self, variants: Vec<(String, TypeScriptType)>) -> TypeScriptType {
        let union_types = variants.into_iter()
            .map(|(discriminant, payload_type)| {
                let mut properties = HashMap::new();
                properties.insert("type".to_string(), ObjectProperty {
                    property_type: TypeScriptType::StringLiteral(discriminant),
                    optional: false,
                    readonly: true,
                });
                
                if !matches!(payload_type, TypeScriptType::Void) {
                    properties.insert("payload".to_string(), ObjectProperty {
                        property_type: payload_type,
                        optional: false,
                        readonly: true,
                    });
                }
                
                TypeScriptType::Object {
                    properties,
                    index_signature: None,
                }
            })
            .collect();

        TypeScriptType::Union(union_types)
    }

    /// Convert primitive type with modern features
    fn convert_primitive_type(&self, prim: &PIRPrimitiveType) -> TypeScriptResult<TypeScriptType> {
        match prim {
            PIRPrimitiveType::Integer { signed: _, width: _ } => {
                // Use bigint for large integers if supported
                if self.features.enhanced_control_flow && self.supports_bigint() {
                    Ok(TypeScriptType::Number) // Could be BigInt for large numbers
                } else {
                    Ok(TypeScriptType::Number)
                }
            }
            PIRPrimitiveType::Float { width: _ } => Ok(TypeScriptType::Number),
            PIRPrimitiveType::Boolean => Ok(TypeScriptType::Boolean),
            PIRPrimitiveType::String => Ok(TypeScriptType::String),
            PIRPrimitiveType::Unit => Ok(TypeScriptType::Void),
        }
    }

    /// Convert composite type with advanced patterns
    fn convert_composite_type(&self, comp: &PIRCompositeType) -> TypeScriptResult<TypeScriptType> {
        match comp.kind {
            PIRCompositeKind::Struct => {
                let mut properties = HashMap::new();
                for field in &comp.fields {
                    let field_type = self.convert_field_type(&field.field_type)?;
                    properties.insert(field.name.clone(), ObjectProperty {
                        property_type: field_type,
                        optional: false, // Could be derived from field metadata
                        readonly: false, // Could be derived from field metadata
                    });
                }
                
                Ok(TypeScriptType::Object {
                    properties,
                    index_signature: None,
                })
            }
            PIRCompositeKind::Enum => {
                // Generate string union type for enums
                let variants = comp.fields.iter()
                    .map(|field| TypeScriptType::StringLiteral(field.name.clone()))
                    .collect();
                Ok(TypeScriptType::Union(variants))
            }
            PIRCompositeKind::Union => {
                let variants = comp.fields.iter()
                    .map(|field| self.convert_field_type(&field.field_type))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TypeScriptType::Union(variants))
            }
            PIRCompositeKind::Tuple => {
                let elements = comp.fields.iter()
                    .map(|field| self.convert_field_type(&field.field_type))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TypeScriptType::Tuple(elements))
            }
        }
    }

    /// Convert function type with async support
    fn convert_function_type(&self, func: &crate::backends::PIRFunctionType) -> TypeScriptResult<TypeScriptType> {
        let parameters = func.parameters.iter()
            .map(|param| {
                let param_type = self.convert_field_type(&param.param_type)?;
                Ok(Parameter {
                    name: param.name.clone(),
                    parameter_type: param_type,
                    optional: param.default_value.is_some(),
                    default_value: None, // Could extract from param metadata
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let return_type = Box::new(self.convert_field_type(&func.return_type)?);
        let is_async = !func.effects.effects.is_empty(); // Functions with effects are async

        Ok(TypeScriptType::Function {
            parameters,
            return_type,
            is_async,
        })
    }

    /// Convert generic type
    fn convert_generic_type(&self, generic: &crate::backends::PIRGenericType) -> TypeScriptResult<TypeScriptType> {
        let type_parameters = generic.type_parameters.iter()
            .map(|param| self.convert_field_type(param))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(TypeScriptType::Generic {
            name: generic.name.clone(),
            type_parameters,
        })
    }

    /// Convert effect type to capability handle
    fn convert_effect_type(&self, effect: &crate::backends::PIREffectType) -> TypeScriptResult<TypeScriptType> {
        if self.features.template_literal_types {
            Ok(TypeScriptType::TemplateLiteral {
                parts: vec!["effect:".to_string(), "".to_string()],
                placeholders: vec![TypeScriptType::StringLiteral(effect.name.clone())],
            })
        } else {
            Ok(TypeScriptType::Reference("EffectHandle".to_string()))
        }
    }

    /// Helper to convert field types
    fn convert_field_type(&self, field_type: &PIRTypeInfo) -> TypeScriptResult<TypeScriptType> {
        // This is a simplified conversion - would need to handle the full PIRTypeInfo
        match field_type {
            PIRTypeInfo::Primitive(prim) => self.convert_primitive_type(prim),
            _ => Ok(TypeScriptType::Unknown), // Simplified for now
        }
    }

    /// Generate semantic brand for branded types
    fn generate_semantic_brand(&self, type_name: &str, domain: &str) -> String {
        format!("__{}_{}_brand", domain.to_lowercase(), type_name.to_lowercase())
    }

    /// Check if target supports BigInt
    fn supports_bigint(&self) -> bool {
        match self.target {
            TypeScriptTarget::ES2022 | TypeScriptTarget::ES2023 | TypeScriptTarget::ESNext => true,
            TypeScriptTarget::Node18 | TypeScriptTarget::Node20 => true,
            TypeScriptTarget::Deno | TypeScriptTarget::Bun => true,
            TypeScriptTarget::Browser => false, // Conservative for wide compatibility
        }
    }
}

impl std::fmt::Display for TypeScriptType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Number => write!(f, "number"),
            Self::Boolean => write!(f, "boolean"),
            Self::Undefined => write!(f, "undefined"),
            Self::Null => write!(f, "null"),
            Self::Unknown => write!(f, "unknown"),
            Self::Never => write!(f, "never"),
            Self::Void => write!(f, "void"),
            Self::BigInt => write!(f, "bigint"),
            Self::Symbol => write!(f, "symbol"),
            
            Self::TemplateLiteral { parts, placeholders } => {
                write!(f, "`")?;
                for (i, part) in parts.iter().enumerate() {
                    write!(f, "{}", part)?;
                    if i < placeholders.len() {
                        write!(f, "${{{}}}", placeholders[i])?;
                    }
                }
                write!(f, "`")
            }
            
            Self::Branded { base_type, brand } => {
                write!(f, "{} & {{ readonly [{}]: '{}' }}", base_type, "__brand", brand)
            }
            
            Self::StringLiteral(s) => write!(f, "'{}'", s.replace('\'', "\\'")),
            Self::NumberLiteral(n) => write!(f, "{}", n),
            Self::BooleanLiteral(b) => write!(f, "{}", b),
            
            Self::Union(types) => {
                let type_strings: Vec<String> = types.iter().map(|t| t.to_string()).collect();
                write!(f, "{}", type_strings.join(" | "))
            }
            
            Self::Intersection(types) => {
                let type_strings: Vec<String> = types.iter().map(|t| t.to_string()).collect();
                write!(f, "{}", type_strings.join(" & "))
            }
            
            Self::Object { properties, index_signature } => {
                write!(f, "{{ ")?;
                for (key, prop) in properties {
                    if prop.readonly {
                        write!(f, "readonly ")?;
                    }
                    write!(f, "{}", key)?;
                    if prop.optional {
                        write!(f, "?")?;
                    }
                    write!(f, ": {}; ", prop.property_type)?;
                }
                if let Some(sig) = index_signature {
                    write!(f, "[key: {}]: {}; ", sig.key_type, sig.value_type)?;
                }
                write!(f, "}}")
            }
            
            Self::Array(element_type) => write!(f, "{}[]", element_type),
            
            Self::Tuple(elements) => {
                let element_strings: Vec<String> = elements.iter().map(|t| t.to_string()).collect();
                write!(f, "[{}]", element_strings.join(", "))
            }
            
            Self::Function { parameters, return_type, is_async } => {
                let param_strings: Vec<String> = parameters.iter()
                    .map(|p| format!("{}{}: {}", 
                        p.name, 
                        if p.optional { "?" } else { "" }, 
                        p.parameter_type))
                    .collect();
                
                if *is_async {
                    write!(f, "({}) => Promise<{}>", param_strings.join(", "), return_type)
                } else {
                    write!(f, "({}) => {}", param_strings.join(", "), return_type)
                }
            }
            
            Self::Generic { name, type_parameters } => {
                if type_parameters.is_empty() {
                    write!(f, "{}", name)
                } else {
                    let param_strings: Vec<String> = type_parameters.iter().map(|t| t.to_string()).collect();
                    write!(f, "{}<{}>", name, param_strings.join(", "))
                }
            }
            
            Self::Conditional { check_type, extends_type, true_type, false_type } => {
                write!(f, "{} extends {} ? {} : {}", check_type, extends_type, true_type, false_type)
            }
            
            Self::Mapped { key_type, value_type, modifiers } => {
                write!(f, "{{ ")?;
                if let Some(readonly) = modifiers.readonly {
                    if readonly {
                        write!(f, "readonly ")?;
                    } else {
                        write!(f, "-readonly ")?;
                    }
                }
                write!(f, "[K in {}]", key_type)?;
                if let Some(optional) = modifiers.optional {
                    if optional {
                        write!(f, "?")?;
                    } else {
                        write!(f, "-?")?;
                    }
                }
                write!(f, ": {} }}", value_type)
            }
            
            Self::Utility { kind, type_args } => {
                let type_arg_strings: Vec<String> = type_args.iter().map(|t| t.to_string()).collect();
                match kind {
                    UtilityTypeKind::Custom(name) => {
                        write!(f, "{}<{}>", name, type_arg_strings.join(", "))
                    }
                    _ => {
                        write!(f, "{:?}<{}>", kind, type_arg_strings.join(", "))
                    }
                }
            }
            
            Self::Reference(name) => write!(f, "{}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branded_type_generation() {
        let features = TypeScriptFeatures::default();
        let target = TypeScriptTarget::ES2023;
        let mut converter = TypeScriptTypeConverter::new(features, target);

        let branded_type = TypeScriptType::Branded {
            base_type: Box::new(TypeScriptType::String),
            brand: "__email_communication_brand".to_string(),
        };

        let result = branded_type.to_string();
        assert!(result.contains("string & { readonly [__brand]"));
    }

    #[test]
    fn test_template_literal_type() {
        let template_type = TypeScriptType::TemplateLiteral {
            parts: vec!["capability:".to_string(), "".to_string()],
            placeholders: vec![TypeScriptType::StringLiteral("database".to_string())],
        };

        let result = template_type.to_string();
        assert!(result.contains("`capability:${'database'}`"));
    }

    #[test]
    fn test_discriminated_union() {
        let features = TypeScriptFeatures::default();
        let target = TypeScriptTarget::ES2023;
        let converter = TypeScriptTypeConverter::new(features, target);

        let union = converter.generate_discriminated_union(vec![
            ("loading".to_string(), TypeScriptType::Void),
            ("success".to_string(), TypeScriptType::String),
            ("error".to_string(), TypeScriptType::String),
        ]);

        let result = union.to_string();
        assert!(result.contains("type: 'loading'"));
        assert!(result.contains("type: 'success'"));
        assert!(result.contains("type: 'error'"));
    }
}