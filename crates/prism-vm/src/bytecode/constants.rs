//! Constant Pool Management
//!
//! This module manages the constant pool for Prism bytecode, which stores
//! literal values, strings, and other constants used by the bytecode.

use crate::{VMResult, PrismVMError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use indexmap::IndexMap;

/// Constant pool containing all constants used in bytecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantPool {
    /// Constants stored by index
    pub constants: Vec<Constant>,
    /// String interning map for deduplication
    string_map: HashMap<String, u32>,
    /// Integer interning map for deduplication
    integer_map: HashMap<i64, u32>,
    /// Float interning map for deduplication
    float_map: HashMap<OrderedFloat, u32>,
}

/// Wrapper for f64 that implements Eq and Hash for use in HashMap
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct OrderedFloat(u64); // Store as bits for deterministic comparison

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat(f.to_bits())
    }
}

impl From<OrderedFloat> for f64 {
    fn from(of: OrderedFloat) -> Self {
        f64::from_bits(of.0)
    }
}

/// Constant value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constant {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Byte array
    Bytes(Vec<u8>),
    /// Type reference
    Type(u32), // Type ID
    /// Function reference
    Function(u32), // Function ID
    /// Composite constant (array, object, etc.)
    Composite(CompositeConstant),
}

/// Composite constant types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositeConstant {
    /// Array of constants
    Array(Vec<u32>), // Constant indices
    /// Object with field mappings
    Object(IndexMap<String, u32>), // Field name -> constant index
    /// Tuple of constants
    Tuple(Vec<u32>), // Constant indices
}

/// Constant type enumeration for type checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstantType {
    /// Null type
    Null,
    /// Boolean type
    Boolean,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// String type
    String,
    /// Bytes type
    Bytes,
    /// Type reference
    Type,
    /// Function reference
    Function,
    /// Array type
    Array,
    /// Object type
    Object,
    /// Tuple type
    Tuple,
}

impl ConstantPool {
    /// Create a new empty constant pool
    pub fn new() -> Self {
        Self {
            constants: Vec::new(),
            string_map: HashMap::new(),
            integer_map: HashMap::new(),
            float_map: HashMap::new(),
        }
    }

    /// Add a constant and return its index
    pub fn add_constant(&mut self, constant: Constant) -> u32 {
        // Check for existing constant to avoid duplication
        if let Some(existing_index) = self.find_existing(&constant) {
            return existing_index;
        }

        let index = self.constants.len() as u32;
        
        // Update interning maps
        match &constant {
            Constant::String(s) => {
                self.string_map.insert(s.clone(), index);
            }
            Constant::Integer(i) => {
                self.integer_map.insert(*i, index);
            }
            Constant::Float(f) => {
                self.float_map.insert((*f).into(), index);
            }
            _ => {} // Other types don't need interning
        }

        self.constants.push(constant);
        index
    }

    /// Find existing constant by value
    fn find_existing(&self, constant: &Constant) -> Option<u32> {
        match constant {
            Constant::String(s) => self.string_map.get(s).copied(),
            Constant::Integer(i) => self.integer_map.get(i).copied(),
            Constant::Float(f) => self.float_map.get(&(*f).into()).copied(),
            _ => {
                // For other types, do linear search (less common)
                self.constants.iter().position(|c| self.constants_equal(c, constant))
                    .map(|i| i as u32)
            }
        }
    }

    /// Check if two constants are equal
    fn constants_equal(&self, a: &Constant, b: &Constant) -> bool {
        match (a, b) {
            (Constant::Null, Constant::Null) => true,
            (Constant::Boolean(a), Constant::Boolean(b)) => a == b,
            (Constant::Integer(a), Constant::Integer(b)) => a == b,
            (Constant::Float(a), Constant::Float(b)) => a.to_bits() == b.to_bits(),
            (Constant::String(a), Constant::String(b)) => a == b,
            (Constant::Bytes(a), Constant::Bytes(b)) => a == b,
            (Constant::Type(a), Constant::Type(b)) => a == b,
            (Constant::Function(a), Constant::Function(b)) => a == b,
            (Constant::Composite(a), Constant::Composite(b)) => self.composite_constants_equal(a, b),
            _ => false,
        }
    }

    /// Check if two composite constants are equal
    fn composite_constants_equal(&self, a: &CompositeConstant, b: &CompositeConstant) -> bool {
        match (a, b) {
            (CompositeConstant::Array(a), CompositeConstant::Array(b)) => a == b,
            (CompositeConstant::Object(a), CompositeConstant::Object(b)) => a == b,
            (CompositeConstant::Tuple(a), CompositeConstant::Tuple(b)) => a == b,
            _ => false,
        }
    }

    /// Add a null constant
    pub fn add_null(&mut self) -> u32 {
        self.add_constant(Constant::Null)
    }

    /// Add a boolean constant
    pub fn add_boolean(&mut self, value: bool) -> u32 {
        self.add_constant(Constant::Boolean(value))
    }

    /// Add an integer constant
    pub fn add_integer(&mut self, value: i64) -> u32 {
        self.add_constant(Constant::Integer(value))
    }

    /// Add a float constant
    pub fn add_float(&mut self, value: f64) -> u32 {
        self.add_constant(Constant::Float(value))
    }

    /// Add a string constant
    pub fn add_string(&mut self, value: String) -> u32 {
        self.add_constant(Constant::String(value))
    }

    /// Add a byte array constant
    pub fn add_bytes(&mut self, value: Vec<u8>) -> u32 {
        self.add_constant(Constant::Bytes(value))
    }

    /// Add a type reference constant
    pub fn add_type_ref(&mut self, type_id: u32) -> u32 {
        self.add_constant(Constant::Type(type_id))
    }

    /// Add a function reference constant
    pub fn add_function_ref(&mut self, function_id: u32) -> u32 {
        self.add_constant(Constant::Function(function_id))
    }

    /// Add an array constant
    pub fn add_array(&mut self, elements: Vec<u32>) -> u32 {
        self.add_constant(Constant::Composite(CompositeConstant::Array(elements)))
    }

    /// Add an object constant
    pub fn add_object(&mut self, fields: IndexMap<String, u32>) -> u32 {
        self.add_constant(Constant::Composite(CompositeConstant::Object(fields)))
    }

    /// Add a tuple constant
    pub fn add_tuple(&mut self, elements: Vec<u32>) -> u32 {
        self.add_constant(Constant::Composite(CompositeConstant::Tuple(elements)))
    }

    /// Get a constant by index
    pub fn get(&self, index: u32) -> Option<&Constant> {
        self.constants.get(index as usize)
    }

    /// Get the number of constants
    pub fn len(&self) -> usize {
        self.constants.len()
    }

    /// Check if the constant pool is empty
    pub fn is_empty(&self) -> bool {
        self.constants.is_empty()
    }

    /// Get the type of a constant
    pub fn get_type(&self, index: u32) -> Option<ConstantType> {
        self.get(index).map(|c| c.get_type())
    }

    /// Validate the constant pool
    pub fn validate(&self) -> VMResult<()> {
        for (index, constant) in self.constants.iter().enumerate() {
            self.validate_constant(index as u32, constant)?;
        }
        Ok(())
    }

    /// Validate a single constant
    fn validate_constant(&self, index: u32, constant: &Constant) -> VMResult<()> {
        match constant {
            Constant::Composite(composite) => {
                self.validate_composite_constant(index, composite)?;
            }
            Constant::Float(f) => {
                if f.is_nan() || f.is_infinite() {
                    return Err(PrismVMError::InvalidBytecode {
                        message: format!("Invalid float constant at index {}: {}", index, f),
                    });
                }
            }
            _ => {} // Other constants are always valid
        }
        Ok(())
    }

    /// Validate a composite constant
    fn validate_composite_constant(&self, _index: u32, composite: &CompositeConstant) -> VMResult<()> {
        match composite {
            CompositeConstant::Array(elements) | CompositeConstant::Tuple(elements) => {
                for &element_index in elements {
                    if element_index as usize >= self.constants.len() {
                        return Err(PrismVMError::InvalidBytecode {
                            message: format!("Composite constant references invalid index: {}", element_index),
                        });
                    }
                }
            }
            CompositeConstant::Object(fields) => {
                for &field_index in fields.values() {
                    if field_index as usize >= self.constants.len() {
                        return Err(PrismVMError::InvalidBytecode {
                            message: format!("Object constant field references invalid index: {}", field_index),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Get statistics about the constant pool
    pub fn statistics(&self) -> ConstantPoolStatistics {
        let mut stats = ConstantPoolStatistics::default();
        
        for constant in &self.constants {
            match constant {
                Constant::Null => stats.null_count += 1,
                Constant::Boolean(_) => stats.boolean_count += 1,
                Constant::Integer(_) => stats.integer_count += 1,
                Constant::Float(_) => stats.float_count += 1,
                Constant::String(s) => {
                    stats.string_count += 1;
                    stats.total_string_bytes += s.len();
                }
                Constant::Bytes(b) => {
                    stats.bytes_count += 1;
                    stats.total_bytes_size += b.len();
                }
                Constant::Type(_) => stats.type_ref_count += 1,
                Constant::Function(_) => stats.function_ref_count += 1,
                Constant::Composite(composite) => {
                    match composite {
                        CompositeConstant::Array(_) => stats.array_count += 1,
                        CompositeConstant::Object(_) => stats.object_count += 1,
                        CompositeConstant::Tuple(_) => stats.tuple_count += 1,
                    }
                }
            }
        }

        stats.total_constants = self.constants.len();
        stats
    }

    /// Optimize the constant pool by removing unused constants
    pub fn optimize(&mut self, used_indices: &[bool]) -> HashMap<u32, u32> {
        let mut index_mapping = HashMap::new();
        let mut new_constants = Vec::new();
        let mut new_string_map = HashMap::new();
        let mut new_integer_map = HashMap::new();
        let mut new_float_map = HashMap::new();

        for (old_index, constant) in self.constants.iter().enumerate() {
            if old_index < used_indices.len() && used_indices[old_index] {
                let new_index = new_constants.len() as u32;
                index_mapping.insert(old_index as u32, new_index);
                
                // Update interning maps
                match constant {
                    Constant::String(s) => {
                        new_string_map.insert(s.clone(), new_index);
                    }
                    Constant::Integer(i) => {
                        new_integer_map.insert(*i, new_index);
                    }
                    Constant::Float(f) => {
                        new_float_map.insert((*f).into(), new_index);
                    }
                    _ => {}
                }
                
                new_constants.push(constant.clone());
            }
        }

        self.constants = new_constants;
        self.string_map = new_string_map;
        self.integer_map = new_integer_map;
        self.float_map = new_float_map;

        index_mapping
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Constant {
    /// Get the type of this constant
    pub fn get_type(&self) -> ConstantType {
        match self {
            Constant::Null => ConstantType::Null,
            Constant::Boolean(_) => ConstantType::Boolean,
            Constant::Integer(_) => ConstantType::Integer,
            Constant::Float(_) => ConstantType::Float,
            Constant::String(_) => ConstantType::String,
            Constant::Bytes(_) => ConstantType::Bytes,
            Constant::Type(_) => ConstantType::Type,
            Constant::Function(_) => ConstantType::Function,
            Constant::Composite(composite) => match composite {
                CompositeConstant::Array(_) => ConstantType::Array,
                CompositeConstant::Object(_) => ConstantType::Object,
                CompositeConstant::Tuple(_) => ConstantType::Tuple,
            },
        }
    }

    /// Check if this constant is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Constant::Null => false,
            Constant::Boolean(b) => *b,
            Constant::Integer(i) => *i != 0,
            Constant::Float(f) => *f != 0.0 && !f.is_nan(),
            Constant::String(s) => !s.is_empty(),
            Constant::Bytes(b) => !b.is_empty(),
            Constant::Type(_) | Constant::Function(_) => true,
            Constant::Composite(composite) => match composite {
                CompositeConstant::Array(a) => !a.is_empty(),
                CompositeConstant::Object(o) => !o.is_empty(),
                CompositeConstant::Tuple(t) => !t.is_empty(),
            },
        }
    }

    /// Get the size in bytes of this constant
    pub fn size_bytes(&self) -> usize {
        match self {
            Constant::Null => 0,
            Constant::Boolean(_) => 1,
            Constant::Integer(_) => 8,
            Constant::Float(_) => 8,
            Constant::String(s) => s.len(),
            Constant::Bytes(b) => b.len(),
            Constant::Type(_) | Constant::Function(_) => 4,
            Constant::Composite(composite) => match composite {
                CompositeConstant::Array(a) => a.len() * 4,
                CompositeConstant::Object(o) => o.len() * 8, // Approximate
                CompositeConstant::Tuple(t) => t.len() * 4,
            },
        }
    }
}

/// Statistics about the constant pool
#[derive(Debug, Default)]
pub struct ConstantPoolStatistics {
    /// Total number of constants
    pub total_constants: usize,
    /// Number of null constants
    pub null_count: usize,
    /// Number of boolean constants
    pub boolean_count: usize,
    /// Number of integer constants
    pub integer_count: usize,
    /// Number of float constants
    pub float_count: usize,
    /// Number of string constants
    pub string_count: usize,
    /// Total bytes used by strings
    pub total_string_bytes: usize,
    /// Number of byte array constants
    pub bytes_count: usize,
    /// Total size of byte arrays
    pub total_bytes_size: usize,
    /// Number of type reference constants
    pub type_ref_count: usize,
    /// Number of function reference constants
    pub function_ref_count: usize,
    /// Number of array constants
    pub array_count: usize,
    /// Number of object constants
    pub object_count: usize,
    /// Number of tuple constants
    pub tuple_count: usize,
} 