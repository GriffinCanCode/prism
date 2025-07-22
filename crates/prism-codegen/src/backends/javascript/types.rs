//! JavaScript Type System and Conversion
//!
//! This module handles JavaScript type representation, conversion from PIR types,
//! and runtime type validation using modern JavaScript patterns.

use super::{JavaScriptResult, JavaScriptError, JavaScriptTarget, JavaScriptFeatures};
use crate::backends::{PIRTypeInfo, PIRPrimitiveType, PIRCompositeType, PIRSemanticType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JavaScript type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JavaScriptType {
    /// Primitive types
    Number,
    BigInt,
    String,
    Boolean,
    Symbol,
    Undefined,
    Null,
    /// Object types
    Object,
    Array(Box<JavaScriptType>),
    Function {
        params: Vec<JavaScriptType>,
        return_type: Box<JavaScriptType>,
    },
    /// Modern JavaScript types
    Promise(Box<JavaScriptType>),
    Map(Box<JavaScriptType>, Box<JavaScriptType>),
    Set(Box<JavaScriptType>),
    WeakMap(Box<JavaScriptType>, Box<JavaScriptType>),
    WeakSet(Box<JavaScriptType>),
    /// Branded/Semantic types
    Branded {
        base: Box<JavaScriptType>,
        brand: String,
    },
    /// Union types
    Union(Vec<JavaScriptType>),
    /// Generic types
    Generic(String),
    /// Any type (fallback)
    Any,
}

impl JavaScriptType {
    /// Get the runtime type check expression
    pub fn runtime_check_expression(&self, value_expr: &str) -> String {
        match self {
            Self::Number => format!("typeof {} === 'number'", value_expr),
            Self::BigInt => format!("typeof {} === 'bigint'", value_expr),
            Self::String => format!("typeof {} === 'string'", value_expr),
            Self::Boolean => format!("typeof {} === 'boolean'", value_expr),
            Self::Symbol => format!("typeof {} === 'symbol'", value_expr),
            Self::Undefined => format!("{} === undefined", value_expr),
            Self::Null => format!("{} === null", value_expr),
            Self::Object => format!("typeof {} === 'object' && {} !== null", value_expr, value_expr),
            Self::Array(element_type) => format!(
                "Array.isArray({}) && {}.every(item => {})",
                value_expr,
                value_expr,
                element_type.runtime_check_expression("item")
            ),
            Self::Function { .. } => format!("typeof {} === 'function'", value_expr),
            Self::Promise(_) => format!("{} instanceof Promise", value_expr),
            Self::Map(_, _) => format!("{} instanceof Map", value_expr),
            Self::Set(_) => format!("{} instanceof Set", value_expr),
            Self::WeakMap(_, _) => format!("{} instanceof WeakMap", value_expr),
            Self::WeakSet(_) => format!("{} instanceof WeakSet", value_expr),
            Self::Branded { base, brand } => format!(
                "{} && {}.constructor.name === '{}'",
                base.runtime_check_expression(value_expr),
                value_expr,
                brand
            ),
            Self::Union(types) => {
                let checks: Vec<String> = types.iter()
                    .map(|t| format!("({})", t.runtime_check_expression(value_expr)))
                    .collect();
                checks.join(" || ")
            },
            Self::Generic(_) => "true".to_string(), // Generic types accept anything
            Self::Any => "true".to_string(),
        }
    }

    /// Get the TypeScript-style type annotation
    pub fn typescript_annotation(&self) -> String {
        match self {
            Self::Number => "number".to_string(),
            Self::BigInt => "bigint".to_string(),
            Self::String => "string".to_string(),
            Self::Boolean => "boolean".to_string(),
            Self::Symbol => "symbol".to_string(),
            Self::Undefined => "undefined".to_string(),
            Self::Null => "null".to_string(),
            Self::Object => "object".to_string(),
            Self::Array(element_type) => format!("{}[]", element_type.typescript_annotation()),
            Self::Function { params, return_type } => {
                let param_types: Vec<String> = params.iter()
                    .enumerate()
                    .map(|(i, t)| format!("arg{}: {}", i, t.typescript_annotation()))
                    .collect();
                format!("({}) => {}", param_types.join(", "), return_type.typescript_annotation())
            },
            Self::Promise(inner) => format!("Promise<{}>", inner.typescript_annotation()),
            Self::Map(key, value) => format!("Map<{}, {}>", key.typescript_annotation(), value.typescript_annotation()),
            Self::Set(element) => format!("Set<{}>", element.typescript_annotation()),
            Self::WeakMap(key, value) => format!("WeakMap<{}, {}>", key.typescript_annotation(), value.typescript_annotation()),
            Self::WeakSet(element) => format!("WeakSet<{}>", element.typescript_annotation()),
            Self::Branded { base, brand } => format!("{} & {{ readonly __brand: '{}' }}", base.typescript_annotation(), brand),
            Self::Union(types) => {
                let type_annotations: Vec<String> = types.iter()
                    .map(|t| t.typescript_annotation())
                    .collect();
                type_annotations.join(" | ")
            },
            Self::Generic(name) => name.clone(),
            Self::Any => "any".to_string(),
        }
    }
}

/// Type configuration for JavaScript backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConfig {
    /// Enable runtime type validation
    pub enable_runtime_validation: bool,
    /// Use BigInt for large integers
    pub use_bigint_for_large_integers: bool,
    /// Enable branded types for semantic safety
    pub enable_branded_types: bool,
    /// Generate TypeScript-compatible type annotations
    pub generate_typescript_annotations: bool,
    /// Use WeakMap/WeakSet for memory efficiency
    pub use_weak_collections: bool,
    /// Enable Symbol-based private properties
    pub use_symbol_private_properties: bool,
}

impl Default for TypeConfig {
    fn default() -> Self {
        Self {
            enable_runtime_validation: true,
            use_bigint_for_large_integers: true,
            enable_branded_types: true,
            generate_typescript_annotations: true,
            use_weak_collections: true,
            use_symbol_private_properties: true,
        }
    }
}

/// JavaScript type converter
#[derive(Debug, Clone)]
pub struct JavaScriptTypeConverter {
    config: TypeConfig,
    target: JavaScriptTarget,
    features: JavaScriptFeatures,
    /// Cache for converted types
    type_cache: HashMap<String, JavaScriptType>,
}

impl JavaScriptTypeConverter {
    /// Create a new type converter
    pub fn new(config: TypeConfig) -> Self {
        Self {
            config,
            target: JavaScriptTarget::default(),
            features: JavaScriptFeatures::default(),
            type_cache: HashMap::new(),
        }
    }

    /// Convert PIR type to JavaScript type
    pub fn convert_pir_type(&mut self, pir_type: &PIRTypeInfo) -> JavaScriptResult<JavaScriptType> {
        match pir_type {
            PIRTypeInfo::Primitive(prim) => self.convert_primitive_type(prim),
            PIRTypeInfo::Composite(comp) => self.convert_composite_type(comp),
            PIRTypeInfo::Function(func) => self.convert_function_type(func),
            PIRTypeInfo::Generic(generic) => Ok(JavaScriptType::Generic(generic.name.clone())),
            PIRTypeInfo::Effect(_) => Ok(JavaScriptType::Object), // Effects are objects
        }
    }

    /// Convert PIR semantic type to JavaScript branded type
    pub fn convert_semantic_type(&mut self, semantic_type: &PIRSemanticType) -> JavaScriptResult<JavaScriptType> {
        let base_type = self.convert_pir_type(&semantic_type.base_type)?;
        
        if self.config.enable_branded_types {
            Ok(JavaScriptType::Branded {
                base: Box::new(base_type),
                brand: semantic_type.name.clone(),
            })
        } else {
            Ok(base_type)
        }
    }

    /// Convert primitive type
    fn convert_primitive_type(&self, prim: &PIRPrimitiveType) -> JavaScriptResult<JavaScriptType> {
        match prim {
            PIRPrimitiveType::Integer { signed: _, width } => {
                if *width > 32 && self.config.use_bigint_for_large_integers && self.features.bigint {
                    Ok(JavaScriptType::BigInt)
                } else {
                    Ok(JavaScriptType::Number)
                }
            },
            PIRPrimitiveType::Float { width: _ } => Ok(JavaScriptType::Number),
            PIRPrimitiveType::Boolean => Ok(JavaScriptType::Boolean),
            PIRPrimitiveType::String => Ok(JavaScriptType::String),
            PIRPrimitiveType::Unit => Ok(JavaScriptType::Undefined),
        }
    }

    /// Convert composite type
    fn convert_composite_type(&mut self, comp: &PIRCompositeType) -> JavaScriptResult<JavaScriptType> {
        match &comp.kind {
            crate::backends::PIRCompositeKind::Array => {
                if let Some(element_type) = comp.fields.first() {
                    let js_element_type = self.convert_pir_type(&element_type.field_type)?;
                    Ok(JavaScriptType::Array(Box::new(js_element_type)))
                } else {
                    Ok(JavaScriptType::Array(Box::new(JavaScriptType::Any)))
                }
            },
            crate::backends::PIRCompositeKind::Record => Ok(JavaScriptType::Object),
            crate::backends::PIRCompositeKind::Tuple => Ok(JavaScriptType::Array(Box::new(JavaScriptType::Any))),
            crate::backends::PIRCompositeKind::Union => {
                let mut union_types = Vec::new();
                for field in &comp.fields {
                    let js_type = self.convert_pir_type(&field.field_type)?;
                    union_types.push(js_type);
                }
                Ok(JavaScriptType::Union(union_types))
            },
        }
    }

    /// Convert function type
    fn convert_function_type(&mut self, func: &crate::backends::PIRFunction) -> JavaScriptResult<JavaScriptType> {
        let mut param_types = Vec::new();
        for param in &func.signature.parameters {
            let js_type = self.convert_pir_type(&param.param_type)?;
            param_types.push(js_type);
        }

        let return_type = if let Some(ret_type) = &func.signature.return_type {
            let js_ret_type = self.convert_pir_type(ret_type)?;
            if self.features.async_await && !func.signature.effects.effects.is_empty() {
                JavaScriptType::Promise(Box::new(js_ret_type))
            } else {
                js_ret_type
            }
        } else {
            JavaScriptType::Undefined
        };

        Ok(JavaScriptType::Function {
            params: param_types,
            return_type: Box::new(return_type),
        })
    }

    /// Generate runtime type validation code
    pub fn generate_type_validation(&self, js_type: &JavaScriptType, value_name: &str) -> String {
        if !self.config.enable_runtime_validation {
            return String::new();
        }

        format!(
            r#"
// Runtime type validation for {}
if (!({}) {{
    throw new TypeError(`Expected {}, got ${{typeof {}}}: ${{JSON.stringify({})}}`);
}}
"#,
            value_name,
            js_type.runtime_check_expression(value_name),
            js_type.typescript_annotation(),
            value_name,
            value_name
        )
    }

    /// Generate branded type constructor
    pub fn generate_branded_type_constructor(&self, brand: &str, base_type: &JavaScriptType) -> JavaScriptResult<String> {
        if !self.config.enable_branded_types {
            return Err(JavaScriptError::TypeConversion {
                from: "semantic".to_string(),
                to: "branded".to_string(),
                reason: "Branded types are disabled".to_string(),
            });
        }

        let validation_code = self.generate_type_validation(base_type, "value");
        let symbol_private = if self.config.use_symbol_private_properties && self.features.symbols {
            format!("const _{}_brand = Symbol('{}');", brand.to_lowercase(), brand)
        } else {
            String::new()
        };

        Ok(format!(
            r#"
{}

/**
 * Branded type: {}
 * Base type: {}
 */
export class {} {{
    constructor(value) {{{}
        this.value = value;
        {}
        Object.freeze(this);
    }}

    valueOf() {{
        return this.value;
    }}

    toString() {{
        return String(this.value);
    }}

    toJSON() {{
        return this.value;
    }}

    static is(value) {{
        return value instanceof {};
    }}

    static from(value) {{
        return new {}(value);
    }}

    static validate(value) {{
        try {{
            new {}(value);
            return true;
        }} catch (error) {{
            return false;
        }}
    }}
}}
"#,
            symbol_private,
            brand,
            base_type.typescript_annotation(),
            brand,
            validation_code,
            if self.config.use_symbol_private_properties && self.features.symbols {
                format!("this[_{}_brand] = true;", brand.to_lowercase())
            } else {
                format!("this.__brand = '{}';", brand)
            },
            brand,
            brand,
            brand
        ))
    }

    /// Generate type guard functions
    pub fn generate_type_guards(&self, types: &[(String, JavaScriptType)]) -> String {
        let mut output = String::new();
        
        output.push_str("// === TYPE GUARDS ===\n\n");

        for (name, js_type) in types {
            let guard_name = format!("is{}", self.to_pascal_case(name));
            output.push_str(&format!(
                r#"/**
 * Type guard for {}
 * @param {{any}} value - Value to check
 * @returns {{boolean}} True if value is of type {}
 */
export function {}(value) {{
    return {};
}}

"#,
                name,
                js_type.typescript_annotation(),
                guard_name,
                js_type.runtime_check_expression("value")
            ));
        }

        output
    }

    /// Generate type assertion functions
    pub fn generate_type_assertions(&self, types: &[(String, JavaScriptType)]) -> String {
        let mut output = String::new();
        
        output.push_str("// === TYPE ASSERTIONS ===\n\n");

        for (name, js_type) in types {
            let assert_name = format!("assert{}", self.to_pascal_case(name));
            let guard_name = format!("is{}", self.to_pascal_case(name));
            
            output.push_str(&format!(
                r#"/**
 * Type assertion for {}
 * @param {{any}} value - Value to assert
 * @param {{string}} [message] - Custom error message
 * @returns {{{}}} The value if valid
 * @throws {{TypeError}} If value is not of expected type
 */
export function {}(value, message) {{
    if (!{}(value)) {{
        throw new TypeError(message || `Expected {}, got ${{typeof value}}: ${{JSON.stringify(value)}}`);
    }}
    return value;
}}

"#,
                name,
                js_type.typescript_annotation(),
                assert_name,
                guard_name,
                js_type.typescript_annotation()
            ));
        }

        output
    }

    /// Convert string to PascalCase
    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                }
            })
            .collect()
    }
}

/// Type features for JavaScript backend
pub use super::JavaScriptFeatures;
pub use super::JavaScriptTarget; 