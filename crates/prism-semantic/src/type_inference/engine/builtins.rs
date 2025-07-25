//! Built-in Type Manager
//!
//! This module manages the initialization and management of built-in types and functions.
//! It provides the foundational types that are available in all Prism programs.
//!
//! **Single Responsibility**: Manage built-in types and functions
//! **What it does**: Initialize built-ins, provide type lookups, manage primitive types
//! **What it doesn't do**: Perform inference, handle user-defined types, manage environments

use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        TypeInferenceResult, InferredType, InferenceSource,
        environment::{TypeEnvironment, TypeBinding},
        constraints::ConstraintSet,
    },
};
use prism_common::Span;
use std::collections::HashMap;

/// Manager for built-in types and functions
#[derive(Debug)]
pub struct BuiltinTypeManager {
    /// Built-in types
    builtin_types: HashMap<String, SemanticType>,
    /// Built-in functions
    builtin_functions: HashMap<String, SemanticType>,
    /// Built-in operators
    builtin_operators: HashMap<String, SemanticType>,
    /// Built-in constants
    builtin_constants: HashMap<String, SemanticType>,
}

impl BuiltinTypeManager {
    /// Create a new builtin type manager
    pub fn new() -> SemanticResult<Self> {
        let mut manager = Self {
            builtin_types: HashMap::new(),
            builtin_functions: HashMap::new(),
            builtin_operators: HashMap::new(),
            builtin_constants: HashMap::new(),
        };
        
        manager.initialize_builtin_types()?;
        manager.initialize_builtin_functions()?;
        manager.initialize_builtin_operators()?;
        manager.initialize_builtin_constants()?;
        
        Ok(manager)
    }

    /// Initialize built-in types and functions in the inference result
    pub fn initialize_builtins(&self, result: &mut TypeInferenceResult) -> SemanticResult<()> {
        // Add built-in types to the environment
        for (name, semantic_type) in &self.builtin_types {
            let binding = TypeBinding::new(
                name.clone(),
                InferredType {
                    type_info: semantic_type.clone(),
                    confidence: 1.0,
                    inference_source: InferenceSource::Explicit,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: Span::dummy(),
                },
                false, // built-ins are not mutable
                0, // global scope
            );
            result.global_env.add_binding(binding);
        }

        // Add built-in functions to the environment
        for (name, func_type) in &self.builtin_functions {
            let binding = TypeBinding::new(
                name.clone(),
                InferredType {
                    type_info: func_type.clone(),
                    confidence: 1.0,
                    inference_source: InferenceSource::Explicit,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: Span::dummy(),
                },
                false, // built-ins are not mutable
                0, // global scope
            );
            result.global_env.add_binding(binding);
        }

        // Add built-in operators to the environment
        for (name, op_type) in &self.builtin_operators {
            let binding = TypeBinding::new(
                name.clone(),
                InferredType {
                    type_info: op_type.clone(),
                    confidence: 1.0,
                    inference_source: InferenceSource::Explicit,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: Span::dummy(),
                },
                false, // built-ins are not mutable
                0, // global scope
            );
            result.global_env.add_binding(binding);
        }

        // Add built-in constants to the environment
        for (name, const_type) in &self.builtin_constants {
            let binding = TypeBinding::new(
                name.clone(),
                InferredType {
                    type_info: const_type.clone(),
                    confidence: 1.0,
                    inference_source: InferenceSource::Explicit,
                    constraints: Vec::new(),
                    ai_metadata: None,
                    span: Span::dummy(),
                },
                false, // built-ins are not mutable
                0, // global scope
            );
            result.global_env.add_binding(binding);
        }

        Ok(())
    }

    /// Get a built-in type by name
    pub fn get_builtin_type(&self, name: &str) -> Option<&SemanticType> {
        self.builtin_types.get(name)
    }

    /// Get a built-in function by name
    pub fn get_builtin_function(&self, name: &str) -> Option<&SemanticType> {
        self.builtin_functions.get(name)
    }

    /// Get a built-in operator by name
    pub fn get_builtin_operator(&self, name: &str) -> Option<&SemanticType> {
        self.builtin_operators.get(name)
    }

    /// Check if a name is a built-in
    pub fn is_builtin(&self, name: &str) -> bool {
        self.builtin_types.contains_key(name) ||
        self.builtin_functions.contains_key(name) ||
        self.builtin_operators.contains_key(name) ||
        self.builtin_constants.contains_key(name)
    }

    /// Reset the builtin manager (no-op since built-ins are immutable)
    pub fn reset(&mut self) {
        // Built-ins don't change, so no reset needed
    }

    /// Initialize built-in types
    fn initialize_builtin_types(&mut self) -> SemanticResult<()> {
        // Primitive types
        self.builtin_types.insert(
            "Int".to_string(),
            self.create_primitive_type("Int", "i64"),
        );
        self.builtin_types.insert(
            "Integer".to_string(),
            self.create_primitive_type("Integer", "i64"),
        );
        self.builtin_types.insert(
            "Float".to_string(),
            self.create_primitive_type("Float", "f64"),
        );
        self.builtin_types.insert(
            "String".to_string(),
            self.create_primitive_type("String", "string"),
        );
        self.builtin_types.insert(
            "Bool".to_string(),
            self.create_primitive_type("Bool", "bool"),
        );
        self.builtin_types.insert(
            "Boolean".to_string(),
            self.create_primitive_type("Boolean", "bool"),
        );
        self.builtin_types.insert(
            "Char".to_string(),
            self.create_primitive_type("Char", "char"),
        );
        self.builtin_types.insert(
            "Unit".to_string(),
            self.create_primitive_type("Unit", "unit"),
        );

        // Collection types (generic)
        self.builtin_types.insert(
            "Array".to_string(),
            self.create_generic_type("Array", vec!["T"]),
        );
        self.builtin_types.insert(
            "List".to_string(),
            self.create_generic_type("List", vec!["T"]),
        );
        self.builtin_types.insert(
            "Map".to_string(),
            self.create_generic_type("Map", vec!["K", "V"]),
        );
        self.builtin_types.insert(
            "Set".to_string(),
            self.create_generic_type("Set", vec!["T"]),
        );

        // Option and Result types
        self.builtin_types.insert(
            "Option".to_string(),
            self.create_generic_type("Option", vec!["T"]),
        );
        self.builtin_types.insert(
            "Result".to_string(),
            self.create_generic_type("Result", vec!["T", "E"]),
        );

        Ok(())
    }

    /// Initialize built-in functions
    fn initialize_builtin_functions(&mut self) -> SemanticResult<()> {
        let int_type = self.create_primitive_type("Int", "i64");
        let float_type = self.create_primitive_type("Float", "f64");
        let string_type = self.create_primitive_type("String", "string");
        let bool_type = self.create_primitive_type("Bool", "bool");
        let unit_type = self.create_primitive_type("Unit", "unit");

        // Math functions
        self.builtin_functions.insert(
            "abs".to_string(),
            SemanticType::Function {
                params: vec![int_type.clone()],
                return_type: Box::new(int_type.clone()),
                effects: Vec::new(),
            },
        );

        self.builtin_functions.insert(
            "sqrt".to_string(),
            SemanticType::Function {
                params: vec![float_type.clone()],
                return_type: Box::new(float_type.clone()),
                effects: Vec::new(),
            },
        );

        // String functions
        self.builtin_functions.insert(
            "length".to_string(),
            SemanticType::Function {
                params: vec![string_type.clone()],
                return_type: Box::new(int_type.clone()),
                effects: Vec::new(),
            },
        );

        // I/O functions (with effects)
        self.builtin_functions.insert(
            "print".to_string(),
            SemanticType::Function {
                params: vec![string_type.clone()],
                return_type: Box::new(unit_type.clone()),
                effects: vec!["IO".to_string()],
            },
        );

        self.builtin_functions.insert(
            "println".to_string(),
            SemanticType::Function {
                params: vec![string_type.clone()],
                return_type: Box::new(unit_type.clone()),
                effects: vec!["IO".to_string()],
            },
        );

        // Type conversion functions
        self.builtin_functions.insert(
            "toString".to_string(),
            SemanticType::Function {
                params: vec![int_type.clone()],
                return_type: Box::new(string_type.clone()),
                effects: Vec::new(),
            },
        );

        self.builtin_functions.insert(
            "toInt".to_string(),
            SemanticType::Function {
                params: vec![string_type.clone()],
                return_type: Box::new(self.create_option_type(int_type.clone())),
                effects: Vec::new(),
            },
        );

        Ok(())
    }

    /// Initialize built-in operators
    fn initialize_builtin_operators(&mut self) -> SemanticResult<()> {
        let int_type = self.create_primitive_type("Int", "i64");
        let float_type = self.create_primitive_type("Float", "f64");
        let bool_type = self.create_primitive_type("Bool", "bool");
        let string_type = self.create_primitive_type("String", "string");

        // Arithmetic operators
        self.add_binary_operator("+", &int_type, &int_type, &int_type);
        self.add_binary_operator("-", &int_type, &int_type, &int_type);
        self.add_binary_operator("*", &int_type, &int_type, &int_type);
        self.add_binary_operator("/", &int_type, &int_type, &int_type);
        self.add_binary_operator("%", &int_type, &int_type, &int_type);

        // Float arithmetic
        self.add_binary_operator("+", &float_type, &float_type, &float_type);
        self.add_binary_operator("-", &float_type, &float_type, &float_type);
        self.add_binary_operator("*", &float_type, &float_type, &float_type);
        self.add_binary_operator("/", &float_type, &float_type, &float_type);

        // String concatenation
        self.add_binary_operator("++", &string_type, &string_type, &string_type);

        // Comparison operators
        self.add_binary_operator("==", &int_type, &int_type, &bool_type);
        self.add_binary_operator("!=", &int_type, &int_type, &bool_type);
        self.add_binary_operator("<", &int_type, &int_type, &bool_type);
        self.add_binary_operator(">", &int_type, &int_type, &bool_type);
        self.add_binary_operator("<=", &int_type, &int_type, &bool_type);
        self.add_binary_operator(">=", &int_type, &int_type, &bool_type);

        // Logical operators
        self.add_binary_operator("&&", &bool_type, &bool_type, &bool_type);
        self.add_binary_operator("||", &bool_type, &bool_type, &bool_type);

        // Unary operators
        self.add_unary_operator("!", &bool_type, &bool_type);
        self.add_unary_operator("-", &int_type, &int_type);
        self.add_unary_operator("-", &float_type, &float_type);

        Ok(())
    }

    /// Initialize built-in constants
    fn initialize_builtin_constants(&mut self) -> SemanticResult<()> {
        let int_type = self.create_primitive_type("Int", "i64");
        let float_type = self.create_primitive_type("Float", "f64");
        let bool_type = self.create_primitive_type("Bool", "bool");

        // Mathematical constants
        self.builtin_constants.insert("PI".to_string(), float_type.clone());
        self.builtin_constants.insert("E".to_string(), float_type.clone());

        // Boolean constants
        self.builtin_constants.insert("true".to_string(), bool_type.clone());
        self.builtin_constants.insert("false".to_string(), bool_type.clone());

        // Numeric limits
        self.builtin_constants.insert("MAX_INT".to_string(), int_type.clone());
        self.builtin_constants.insert("MIN_INT".to_string(), int_type.clone());

        Ok(())
    }

    /// Create a primitive semantic type
    fn create_primitive_type(&self, name: &str, base: &str) -> SemanticType {
        let primitive_type = match name {
            "Int" | "Integer" => prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(64)),
            "Float" => prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64),
            "String" => prism_ast::PrimitiveType::String,
            "Bool" | "Boolean" => prism_ast::PrimitiveType::Boolean,
            "Char" => prism_ast::PrimitiveType::Char,
            "Unit" => prism_ast::PrimitiveType::Unit,
            _ => prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(64)), // Default fallback
        };

        SemanticType::primitive(name, primitive_type, Span::dummy())
    }

    /// Create a generic semantic type
    fn create_generic_type(&self, name: &str, params: Vec<&str>) -> SemanticType {
        let type_params = params
            .into_iter()
            .map(|param| SemanticType::Variable(param.to_string()))
            .collect();

        SemanticType::Generic {
            name: name.to_string(),
            parameters: type_params,
        }
    }

    /// Create an Option type
    fn create_option_type(&self, inner_type: SemanticType) -> SemanticType {
        SemanticType::Generic {
            name: "Option".to_string(),
            parameters: vec![inner_type],
        }
    }

    /// Add a binary operator
    fn add_binary_operator(&mut self, op: &str, left: &SemanticType, right: &SemanticType, result: &SemanticType) {
        let func_type = SemanticType::Function {
            params: vec![left.clone(), right.clone()],
            return_type: Box::new(result.clone()),
            effects: Vec::new(),
        };
        self.builtin_operators.insert(op.to_string(), func_type);
    }

    /// Add a unary operator
    fn add_unary_operator(&mut self, op: &str, operand: &SemanticType, result: &SemanticType) {
        let func_type = SemanticType::Function {
            params: vec![operand.clone()],
            return_type: Box::new(result.clone()),
            effects: Vec::new(),
        };
        self.builtin_operators.insert(op.to_string(), func_type);
    }
}

impl Default for BuiltinTypeManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default BuiltinTypeManager")
    }
} 