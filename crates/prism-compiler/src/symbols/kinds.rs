//! Symbol Classification System
//!
//! This module defines the comprehensive symbol classification system for the Prism compiler,
//! following PLT-004 specifications. It provides detailed categorization of symbols with
//! semantic meaning and AI-comprehensible metadata.
//!
//! ## Conceptual Responsibility
//! 
//! This module handles ONE thing: "Symbol Classification and Categorization"
//! - Symbol type definitions and variants
//! - Classification metadata and context
//! - Conversion utilities for semantic integration
//! 
//! It does NOT handle:
//! - Symbol storage (delegated to table.rs)
//! - Symbol metadata (delegated to metadata.rs) 
//! - Symbol resolution (delegated to resolution subsystem)

use prism_common::{span::Span, NodeId};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive symbol classification following PLT-004 specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolKind {
    /// Module symbol with semantic sections and capabilities
    Module {
        /// Module sections (PLD-002 integration)
        sections: Vec<ModuleSection>,
        /// Required capabilities for this module
        capabilities: Vec<String>,
        /// Effect boundaries
        effects: Vec<String>,
        /// Module cohesion metrics
        cohesion_info: Option<CohesionInfo>,
    },
    
    /// Function symbol with comprehensive signature information
    Function {
        /// Function signature information
        signature: FunctionSignature,
        /// Function classification
        function_type: FunctionType,
        /// Contract specifications
        contracts: Option<ContractSpecification>,
        /// Performance characteristics
        performance_info: Option<PerformanceInfo>,
    },
    
    /// Type definition symbol with semantic meaning
    Type {
        /// Type category classification
        type_category: TypeCategory,
        /// Type constraints and validation rules
        constraints: Vec<TypeConstraint>,
        /// Semantic type information
        semantic_info: Option<SemanticTypeInfo>,
        /// Business rules associated with this type
        business_rules: Vec<BusinessRule>,
    },
    
    /// Variable symbol with mutability and type information
    Variable {
        /// Variable information
        variable_info: VariableInfo,
        /// Initialization information
        initialization: Option<InitializationInfo>,
        /// Usage patterns
        usage_patterns: Vec<UsagePattern>,
    },
    
    /// Constant symbol with value and type information
    Constant {
        /// Constant information
        constant_info: ConstantInfo,
        /// Compile-time value if available
        compile_time_value: Option<CompileTimeValue>,
    },
    
    /// Function parameter symbol
    Parameter {
        /// Parameter information
        parameter_info: ParameterInfo,
        /// Default value information
        default_value: Option<DefaultValueInfo>,
    },
    
    /// Import symbol with source and alias information
    Import {
        /// Import information
        import_info: ImportInfo,
        /// Resolution information
        resolution: Option<ImportResolution>,
    },
    
    /// Export symbol with visibility and alias information
    Export {
        /// Export information
        export_info: ExportInfo,
        /// Re-export information
        reexport_info: Option<ReexportInfo>,
    },
    
    /// Capability symbol for security system integration
    Capability {
        /// Capability type and requirements
        capability_info: CapabilityInfo,
        /// Security level and audit requirements
        security_requirements: Vec<SecurityRequirement>,
    },
    
    /// Effect symbol for effect system integration
    Effect {
        /// Effect information
        effect_info: EffectInfo,
        /// Composition rules for effect combinations
        composition_rules: Vec<CompositionRule>,
    },
    
    /// Section symbol for module organization (PLD-002)
    Section {
        /// Section type and purpose
        section_info: SectionInfo,
        /// Parent module reference
        parent_module: String,
    },
}

/// Module section information (PLD-002 integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSection {
    /// Section type
    pub section_type: SectionType,
    /// Section name
    pub name: String,
    /// Section purpose and responsibility
    pub purpose: Option<String>,
    /// Symbols contained in this section
    pub symbol_count: usize,
}

/// Module section types following PLD-002
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    /// Configuration and settings
    Config,
    /// Type definitions
    Types,
    /// Error definitions and handling
    Errors,
    /// Internal implementation details
    Internal,
    /// Public interface definitions
    Interface,
    /// Event definitions and handling
    Events,
    /// Lifecycle management
    Lifecycle,
    /// Test definitions
    Tests,
    /// Usage examples
    Examples,
    /// Performance-critical code
    Performance,
    /// Custom section type
    Custom(String),
}

/// Function signature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    /// Function parameters
    pub parameters: Vec<ParameterInfo>,
    /// Return type reference
    pub return_type: Option<String>,
    /// Whether function is async
    pub is_async: bool,
    /// Generic parameters
    pub generic_parameters: Vec<GenericParameter>,
    /// Where clauses and constraints
    pub where_clauses: Vec<WhereClause>,
}

/// Function type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionType {
    /// Regular function
    Regular,
    /// Method (associated with a type)
    Method { receiver_type: String },
    /// Static method
    StaticMethod { associated_type: String },
    /// Constructor
    Constructor { constructed_type: String },
    /// Destructor
    Destructor { destructed_type: String },
    /// Operator overload
    Operator { operator: String },
    /// Lambda/closure
    Lambda { captures: Vec<String> },
    /// Generator function
    Generator { yield_type: String },
    /// Async generator
    AsyncGenerator { yield_type: String },
}

/// Type category classification with enhanced semantics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeCategory {
    /// Primitive types (built-in language types)
    Primitive {
        /// Primitive type name
        primitive_type: PrimitiveType,
        /// Size in bits
        size_bits: Option<u32>,
    },
    /// Semantic types with business meaning (PLD-001)
    Semantic {
        /// Base type this semantic type extends
        base_type: String,
        /// Business domain
        domain: String,
        /// Validation rules
        validation_rules: Vec<String>,
    },
    /// Composite types (structs, enums, unions)
    Composite {
        /// Composition type
        composition_type: CompositionType,
        /// Field information
        fields: Vec<FieldInfo>,
        /// Type relationships
        relationships: Vec<TypeRelationship>,
    },
    /// Function types
    Function {
        /// Function signature
        signature: FunctionSignature,
    },
    /// Generic/parameterized types
    Generic {
        /// Generic parameters
        parameters: Vec<GenericParameter>,
        /// Constraints on parameters
        constraints: Vec<GenericConstraint>,
    },
    /// Effect types (PLD-003)
    Effect {
        /// Effects this type can produce
        effects: Vec<String>,
        /// Capabilities required
        capabilities: Vec<String>,
    },
}

/// Primitive type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimitiveType {
    /// Signed integer
    SignedInt(u8),
    /// Unsigned integer
    UnsignedInt(u8),
    /// Floating point
    Float(u8),
    /// Boolean
    Bool,
    /// String
    String,
    /// Character
    Char,
    /// Unit/void type
    Unit,
    /// Never type (for functions that don't return)
    Never,
}

/// Variable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableInfo {
    /// Whether the variable is mutable
    pub is_mutable: bool,
    /// Type hint or annotation
    pub type_hint: Option<String>,
    /// Whether variable is captured in closures
    pub is_captured: bool,
    /// Lifetime information
    pub lifetime: Option<String>,
    /// Storage class (stack, heap, static, etc.)
    pub storage_class: StorageClass,
}

/// Storage class for variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageClass {
    /// Stack-allocated
    Stack,
    /// Heap-allocated
    Heap,
    /// Static storage
    Static,
    /// Thread-local storage
    ThreadLocal,
    /// Register hint
    Register,
}

/// Function information extracted from existing code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    /// Function signature
    pub signature: FunctionSignature,
    /// Function type
    pub function_type: FunctionType,
    /// Visibility
    pub visibility: crate::symbols::data::SymbolVisibility,
    /// Location in source
    pub location: Span,
    /// AI-readable description
    pub ai_description: Option<String>,
}

/// Type information extracted from existing code  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type category
    pub type_category: TypeCategory,
    /// Type constraints
    pub constraints: Vec<TypeConstraint>,
    /// Visibility
    pub visibility: crate::symbols::data::SymbolVisibility,
    /// Location in source
    pub location: Span,
    /// AI-readable description
    pub ai_description: Option<String>,
}

/// Module information extracted from existing code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module sections
    pub sections: Vec<ModuleSection>,
    /// Module capabilities
    pub capabilities: Vec<String>,
    /// Module effects
    pub effects: Vec<String>,
    /// Visibility
    pub visibility: crate::symbols::data::SymbolVisibility,
    /// Location in source
    pub location: Span,
    /// AI-readable description
    pub ai_description: Option<String>,
}

/// Parameter kind classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterKind {
    /// Regular parameter
    Regular,
    /// Variadic parameter (...args)
    Variadic,
    /// Named parameter (keyword argument)
    Named,
    /// Self parameter (for methods)
    SelfParam { 
        /// Whether self is mutable
        is_mutable: bool,
        /// Whether self is owned, borrowed, or reference
        ownership: SelfOwnership,
    },
}

/// Self parameter ownership
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelfOwnership {
    /// Owned self (self)
    Owned,
    /// Borrowed self (&self)
    Borrowed,
    /// Mutable borrowed self (&mut self)
    MutableBorrowed,
}

/// Parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter kind
    pub kind: ParameterKind,
    /// Type annotation
    pub type_annotation: Option<String>,
    /// Default value
    pub default_value: Option<String>,
    /// Parameter attributes
    pub attributes: Vec<String>,
}

// Additional supporting types for comprehensive symbol classification...

/// Type constraint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint expression
    pub expression: String,
    /// Constraint category
    pub category: ConstraintCategory,
}

/// Constraint categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintCategory {
    /// Size constraint
    Size,
    /// Range constraint
    Range,
    /// Format constraint
    Format,
    /// Business rule constraint
    BusinessRule,
    /// Security constraint
    Security,
}

/// Semantic type information for business integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTypeInfo {
    /// Business domain this type belongs to
    pub domain: String,
    /// Business rules that apply to this type
    pub business_rules: Vec<BusinessRule>,
    /// Validation rules
    pub validation_rules: Vec<String>,
    /// AI context for this semantic type
    pub ai_context: Option<String>,
}

/// Business rule information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule enforcement level
    pub enforcement: EnforcementLevel,
    /// Rule category
    pub category: String,
}

/// Rule enforcement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced
    Required,
    /// Should be enforced
    Recommended,
    /// Optional enforcement
    Optional,
}

// Placeholder types for comprehensive system (to be implemented)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionInfo {
    pub score: f64,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSpecification {
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub invariants: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInfo {
    pub complexity: String,
    pub memory_usage: Option<String>,
    pub optimization_hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantInfo {
    pub value_type: Option<String>,
    pub is_compile_time: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializationInfo {
    pub is_initialized: bool,
    pub initialization_location: Option<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    pub pattern_type: String,
    pub frequency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileTimeValue {
    pub value: String,
    pub value_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultValueInfo {
    pub value: String,
    pub is_computed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    pub source_module: String,
    pub alias: Option<String>,
    pub imported_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResolution {
    pub resolved_path: String,
    pub resolution_time: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    pub export_name: Option<String>,
    pub is_default: bool,
    pub visibility: crate::symbols::data::SymbolVisibility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReexportInfo {
    pub original_module: String,
    pub original_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityInfo {
    pub capability_type: String,
    pub required_permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRequirement {
    pub requirement_type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectInfo {
    pub effect_name: String,
    pub effect_category: String,
    pub parameters: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRule {
    pub rule_name: String,
    pub rule_expression: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionInfo {
    pub section_type: SectionType,
    pub purpose: String,
    pub symbol_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionType {
    pub name: String,
    pub kind: String, // "struct", "enum", "union", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    pub name: String,
    pub field_type: String,
    pub visibility: crate::symbols::data::SymbolVisibility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationship {
    pub relationship_type: String,
    pub related_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericParameter {
    pub name: String,
    pub bounds: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericConstraint {
    pub parameter: String,
    pub constraint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhereClause {
    pub clause: String,
    pub constraint_type: String,
}

impl SymbolKind {
    /// Get a human-readable description of this symbol kind
    pub fn description(&self) -> String {
        match self {
            SymbolKind::Module { sections, .. } => {
                format!("Module with {} sections", sections.len())
            }
            SymbolKind::Function { signature, function_type, .. } => {
                format!("{:?} function with {} parameters", 
                    function_type, signature.parameters.len())
            }
            SymbolKind::Type { type_category, .. } => {
                format!("Type ({:?})", type_category)
            }
            SymbolKind::Variable { variable_info, .. } => {
                format!("{} variable", 
                    if variable_info.is_mutable { "Mutable" } else { "Immutable" })
            }
            SymbolKind::Constant { .. } => "Constant".to_string(),
            SymbolKind::Parameter { parameter_info, .. } => {
                format!("{:?} parameter", parameter_info.kind)
            }
            SymbolKind::Import { import_info, .. } => {
                format!("Import from {}", import_info.source_module)
            }
            SymbolKind::Export { export_info, .. } => {
                format!("Export {}", 
                    export_info.export_name.as_deref().unwrap_or("default"))
            }
            SymbolKind::Capability { capability_info, .. } => {
                format!("Capability ({})", capability_info.capability_type)
            }
            SymbolKind::Effect { effect_info, .. } => {
                format!("Effect ({})", effect_info.effect_name)
            }
            SymbolKind::Section { section_info, .. } => {
                format!("Section ({:?})", section_info.section_type)
            }
        }
    }
    
    /// Check if this symbol kind represents a callable entity
    pub fn is_callable(&self) -> bool {
        matches!(self, SymbolKind::Function { .. })
    }
    
    /// Check if this symbol kind represents a type definition
    pub fn is_type_definition(&self) -> bool {
        matches!(self, SymbolKind::Type { .. })
    }
    
    /// Check if this symbol kind represents a value
    pub fn is_value(&self) -> bool {
        matches!(self, SymbolKind::Variable { .. } | SymbolKind::Constant { .. } | SymbolKind::Parameter { .. })
    }
    
    /// Get the AI-comprehensible category for this symbol kind
    pub fn ai_category(&self) -> &'static str {
        match self {
            SymbolKind::Module { .. } => "module",
            SymbolKind::Function { .. } => "function",
            SymbolKind::Type { .. } => "type",
            SymbolKind::Variable { .. } => "variable",
            SymbolKind::Constant { .. } => "constant",
            SymbolKind::Parameter { .. } => "parameter",
            SymbolKind::Import { .. } => "import",
            SymbolKind::Export { .. } => "export",
            SymbolKind::Capability { .. } => "capability",
            SymbolKind::Effect { .. } => "effect",
            SymbolKind::Section { .. } => "section",
        }
    }
}

impl Default for FunctionSignature {
    fn default() -> Self {
        Self {
            parameters: Vec::new(),
            return_type: None,
            is_async: false,
            generic_parameters: Vec::new(),
            where_clauses: Vec::new(),
        }
    }
}

impl Default for VariableInfo {
    fn default() -> Self {
        Self {
            is_mutable: false,
            type_hint: None,
            is_captured: false,
            lifetime: None,
            storage_class: StorageClass::Stack,
        }
    }
} 