//! Semantic Representation - Core PIR Types
//!
//! This module defines the core semantic representation types that preserve
//! all semantic information from the AST while adding business context.

use crate::{PIRError, PIRResult};
use prism_common::symbol::Symbol;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The complete Prism Intermediate Representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismIR {
    /// PIR modules containing business capabilities
    pub modules: Vec<PIRModule>,
    /// Global type registry with semantic information
    pub type_registry: SemanticTypeRegistry,
    /// Effect graph for capability analysis
    pub effect_graph: EffectGraph,
    /// Cohesion metrics for optimization guidance
    pub cohesion_metrics: CohesionMetrics,
    /// AI metadata for external tools
    pub ai_metadata: crate::ai_integration::AIMetadata,
    /// PIR metadata and versioning
    pub metadata: PIRMetadata,
}

/// PIR module representing a cohesive business capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRModule {
    /// Module name
    pub name: String,
    /// Business capability domain
    pub capability: String,
    /// Module sections organized by conceptual purpose
    pub sections: Vec<PIRSection>,
    /// Module dependencies
    pub dependencies: Vec<ModuleDependency>,
    /// Business context information
    pub business_context: crate::business::BusinessContext,
    /// Domain-specific rules
    pub domain_rules: Vec<DomainRule>,
    /// Effects provided or required by this module
    pub effects: Vec<Effect>,
    /// Capabilities provided or required
    pub capabilities: Vec<Capability>,
    /// Performance characteristics
    pub performance_profile: crate::quality::PerformanceProfile,
    /// Conceptual cohesion score (0.0 to 1.0)
    pub cohesion_score: f64,
}

/// PIR section organized by conceptual purpose within a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRSection {
    /// Type definitions with semantic meaning
    Types(TypeSection),
    /// Function definitions with business logic
    Functions(FunctionSection),
    /// Constants with domain significance
    Constants(ConstantSection),
    /// Interface definitions for contracts
    Interface(InterfaceSection),
    /// Implementation details
    Implementation(ImplementationSection),
}

/// Type section containing semantically rich types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSection {
    /// Semantic types defined in this section
    pub types: Vec<PIRSemanticType>,
}

/// Function section with effect-aware functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSection {
    /// Functions with full semantic and business information
    pub functions: Vec<PIRFunction>,
}

/// Constant section with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantSection {
    /// Constants with semantic significance
    pub constants: Vec<PIRConstant>,
}

/// Interface section defining contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceSection {
    /// Interface definitions
    pub interfaces: Vec<PIRInterface>,
}

/// Implementation section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationSection {
    /// Implementation items
    pub items: Vec<PIRImplementationItem>,
}

/// PIR semantic type with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRSemanticType {
    /// Type name
    pub name: String,
    /// Base type information
    pub base_type: PIRTypeInfo,
    /// Semantic domain this type belongs to
    pub domain: String,
    /// Business rules associated with this type
    pub business_rules: Vec<crate::business::BusinessRule>,
    /// Validation predicates
    pub validation_predicates: Vec<ValidationPredicate>,
    /// Type constraints
    pub constraints: Vec<PIRTypeConstraint>,
    /// AI context for understanding
    pub ai_context: PIRTypeAIContext,
    /// Security classification
    pub security_classification: SecurityClassification,
}

/// PIR type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRTypeInfo {
    /// Primitive types
    Primitive(PIRPrimitiveType),
    /// Composite types (structs, enums, etc.)
    Composite(PIRCompositeType),
    /// Function types with effects
    Function(PIRFunctionType),
    /// Generic types with bounds
    Generic(PIRGenericType),
    /// Effect types for capability system
    Effect(PIREffectType),
}

/// PIR primitive types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRPrimitiveType {
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

/// PIR composite type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRCompositeType {
    /// Kind of composite
    pub kind: PIRCompositeKind,
    /// Fields in the composite
    pub fields: Vec<PIRField>,
    /// Methods associated with the type
    pub methods: Vec<PIRMethod>,
}

/// Kind of composite type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRCompositeKind {
    /// Struct type
    Struct,
    /// Enum type
    Enum,
    /// Union type
    Union,
    /// Tuple type
    Tuple,
}

/// PIR field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: PIRTypeInfo,
    /// Visibility level
    pub visibility: PIRVisibility,
    /// Business meaning of this field
    pub business_meaning: Option<String>,
    /// Validation rules for this field
    pub validation_rules: Vec<String>,
}

/// PIR method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRMethod {
    /// Method name
    pub name: String,
    /// Method signature
    pub signature: PIRFunctionType,
    /// Method implementation (if available)
    pub implementation: Option<PIRExpression>,
    /// Business purpose of this method
    pub business_purpose: Option<String>,
}

/// PIR function type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRFunctionType {
    /// Function parameters
    pub parameters: Vec<PIRParameter>,
    /// Return type
    pub return_type: Box<PIRTypeInfo>,
    /// Effect signature
    pub effects: EffectSignature,
    /// Performance contracts
    pub contracts: PIRPerformanceContract,
}

/// PIR parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: PIRTypeInfo,
    /// Default value (if any)
    pub default_value: Option<PIRExpression>,
    /// Business meaning of this parameter
    pub business_meaning: Option<String>,
}

/// PIR generic type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRGenericType {
    /// Type parameter name
    pub name: String,
    /// Type bounds
    pub bounds: Vec<PIRTypeConstraint>,
    /// Default type (if any)
    pub default_type: Option<Box<PIRTypeInfo>>,
}

/// PIR effect type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIREffectType {
    /// Effect name
    pub name: String,
    /// Effect operations
    pub operations: Vec<PIREffectOperation>,
    /// Required capabilities
    pub capabilities: Vec<String>,
}

/// PIR effect operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIREffectOperation {
    /// Operation name
    pub name: String,
    /// Operation signature
    pub signature: PIRFunctionType,
    /// Side effects of this operation
    pub side_effects: Vec<String>,
}

/// PIR type constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRTypeConstraint {
    /// Range constraint for numeric types
    Range {
        /// Minimum value
        min: Option<PIRExpression>,
        /// Maximum value
        max: Option<PIRExpression>,
    },
    /// Pattern constraint for string types
    Pattern {
        /// Regular expression pattern
        pattern: String,
    },
    /// Business rule constraint
    BusinessRule {
        /// Business rule to enforce
        rule: crate::business::BusinessRule,
    },
    /// Custom constraint with predicate
    Custom {
        /// Custom predicate expression
        predicate: PIRExpression,
    },
}

/// PIR function with full semantic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRFunction {
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: PIRFunctionType,
    /// Function body
    pub body: PIRExpression,
    /// Business responsibility of this function
    pub responsibility: Option<String>,
    /// Algorithm description
    pub algorithm: Option<String>,
    /// Complexity analysis
    pub complexity: Option<PIRComplexityAnalysis>,
    /// Capabilities required to execute
    pub capabilities_required: Vec<Capability>,
    /// Performance characteristics
    pub performance_characteristics: Vec<String>,
    /// AI hints for understanding
    pub ai_hints: Vec<String>,
}

/// PIR constant with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRConstant {
    /// Constant name
    pub name: String,
    /// Constant type
    pub const_type: PIRTypeInfo,
    /// Constant value
    pub value: PIRExpression,
    /// Business meaning of this constant
    pub business_meaning: Option<String>,
}

/// PIR interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRInterface {
    /// Interface name
    pub name: String,
    /// Interface methods
    pub methods: Vec<PIRMethod>,
    /// Required capabilities
    pub capabilities: Vec<Capability>,
}

/// PIR implementation item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRImplementationItem {
    /// Function implementation
    Function(PIRFunction),
    /// Type implementation
    Type(PIRTypeImplementation),
}

/// PIR type implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRTypeImplementation {
    /// Type being implemented
    pub target_type: String,
    /// Interface being implemented
    pub interface: String,
    /// Method implementations
    pub methods: Vec<PIRMethod>,
}

/// PIR expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRExpression {
    /// Literal value
    Literal(PIRLiteral),
    /// Variable reference
    Variable(String),
    /// Function call with effects
    Call {
        /// Function being called
        function: Box<PIRExpression>,
        /// Arguments to the function
        arguments: Vec<PIRExpression>,
        /// Effects of this call
        effects: Vec<Effect>,
    },
    /// Binary operation
    Binary {
        /// Left operand
        left: Box<PIRExpression>,
        /// Binary operator
        operator: PIRBinaryOp,
        /// Right operand
        right: Box<PIRExpression>,
    },
    /// Unary operation
    Unary {
        /// Unary operator
        operator: PIRUnaryOp,
        /// Operand
        operand: Box<PIRExpression>,
    },
    /// Block expression
    Block {
        /// Statements in the block
        statements: Vec<PIRStatement>,
        /// Result expression (if any)
        result: Option<Box<PIRExpression>>,
    },
    /// Conditional expression
    If {
        /// Condition
        condition: Box<PIRExpression>,
        /// Then branch
        then_branch: Box<PIRExpression>,
        /// Else branch (if any)
        else_branch: Option<Box<PIRExpression>>,
    },
    /// Pattern matching expression
    Match {
        /// Expression being matched
        scrutinee: Box<PIRExpression>,
        /// Match arms
        arms: Vec<PIRMatchArm>,
    },
    /// Type assertion
    TypeAssertion {
        /// Expression to assert
        expression: Box<PIRExpression>,
        /// Target type
        target_type: PIRTypeInfo,
    },
}

/// PIR literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRLiteral {
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// Boolean literal
    Boolean(bool),
    /// String literal
    String(String),
    /// Unit literal
    Unit,
}

/// PIR binary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRBinaryOp {
    /// Addition
    Add,
    /// Subtraction
    Subtract,
    /// Multiplication
    Multiply,
    /// Division
    Divide,
    /// Modulo
    Modulo,
    /// Equality
    Equal,
    /// Inequality
    NotEqual,
    /// Less than
    Less,
    /// Less than or equal
    LessEqual,
    /// Greater than
    Greater,
    /// Greater than or equal
    GreaterEqual,
    /// Logical and
    And,
    /// Logical or
    Or,
    /// Prism-specific semantic equality
    SemanticEqual,
}

/// PIR unary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRUnaryOp {
    /// Logical not
    Not,
    /// Arithmetic negation
    Negate,
}

/// PIR statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRStatement {
    /// Expression statement
    Expression(PIRExpression),
    /// Variable binding
    Let {
        /// Variable name
        name: String,
        /// Type annotation (if any)
        type_annotation: Option<PIRTypeInfo>,
        /// Initial value
        value: PIRExpression,
    },
    /// Assignment
    Assignment {
        /// Assignment target
        target: PIRExpression,
        /// Value to assign
        value: PIRExpression,
    },
    /// Return statement
    Return(Option<PIRExpression>),
}

/// PIR match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRMatchArm {
    /// Pattern to match
    pub pattern: PIRPattern,
    /// Guard condition (if any)
    pub guard: Option<PIRExpression>,
    /// Arm body
    pub body: PIRExpression,
}

/// PIR pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRPattern {
    /// Wildcard pattern (matches anything)
    Wildcard,
    /// Variable pattern (binds to variable)
    Variable(String),
    /// Literal pattern (matches literal)
    Literal(PIRLiteral),
    /// Constructor pattern
    Constructor {
        /// Constructor name
        name: String,
        /// Field patterns
        fields: Vec<PIRPattern>,
    },
}

/// PIR visibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRVisibility {
    /// Public visibility
    Public,
    /// Private visibility
    Private,
    /// Internal visibility (module-level)
    Internal,
}

/// PIR type AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRTypeAIContext {
    /// Intent description
    pub intent: Option<String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common mistakes to avoid
    pub common_mistakes: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
}

/// PIR complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRComplexityAnalysis {
    /// Time complexity
    pub time_complexity: String,
    /// Space complexity
    pub space_complexity: String,
    /// Best case scenario
    pub best_case: Option<String>,
    /// Average case scenario
    pub average_case: Option<String>,
    /// Worst case scenario
    pub worst_case: Option<String>,
}

/// PIR performance contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRPerformanceContract {
    /// Preconditions
    pub preconditions: Vec<PIRCondition>,
    /// Postconditions
    pub postconditions: Vec<PIRCondition>,
    /// Performance guarantees
    pub performance_guarantees: Vec<PIRPerformanceGuarantee>,
}

/// PIR condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRCondition {
    /// Condition name
    pub name: String,
    /// Condition expression
    pub expression: PIRExpression,
    /// Error message if condition fails
    pub error_message: String,
}

/// PIR performance guarantee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRPerformanceGuarantee {
    /// Type of performance guarantee
    pub guarantee_type: PIRPerformanceType,
    /// Bound expression
    pub bound: PIRExpression,
    /// Description of the guarantee
    pub description: String,
}

/// PIR performance types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIRPerformanceType {
    /// Time complexity bound
    TimeComplexity,
    /// Space complexity bound
    SpaceComplexity,
    /// Maximum execution time
    MaxExecutionTime,
    /// Maximum memory usage
    MaxMemoryUsage,
}

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityClassification {
    /// Public information
    Public,
    /// Internal use only
    Internal,
    /// Confidential information
    Confidential,
    /// Restricted access
    Restricted,
    /// Top secret
    TopSecret,
}

/// Module dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDependency {
    /// Module name
    pub name: String,
    /// Type of dependency
    pub dependency_type: DependencyType,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
}

/// Dependency types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Direct dependency
    Direct,
    /// Transitive dependency
    Transitive,
    /// Optional dependency
    Optional,
    /// Development-only dependency
    Development,
}

/// Domain rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule expression
    pub expression: PIRExpression,
    /// Enforcement level
    pub enforcement: EnforcementLevel,
}

/// Enforcement level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced
    Required,
    /// Should be enforced
    Recommended,
    /// May be enforced
    Optional,
}

/// Effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    /// Effect name
    pub name: String,
    /// Effect type
    pub effect_type: String,
    /// Effect description
    pub description: Option<String>,
}

/// Capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Capability name
    pub name: String,
    /// Capability description
    pub description: Option<String>,
    /// Required permissions
    pub permissions: Vec<String>,
}

/// Effect signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSignature {
    /// Input effects
    pub input_effects: Vec<Effect>,
    /// Output effects
    pub output_effects: Vec<Effect>,
    /// Effect dependencies
    pub effect_dependencies: Vec<EffectDependency>,
}

/// Effect dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDependency {
    /// Source effect
    pub source: String,
    /// Target effect
    pub target: String,
    /// Dependency type
    pub dependency_type: EffectDependencyType,
}

/// Effect dependency type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectDependencyType {
    /// Requires the other effect
    Requires,
    /// Provides the other effect
    Provides,
    /// Conflicts with the other effect
    Conflicts,
    /// Enhances the other effect
    Enhances,
}

/// Validation predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPredicate {
    /// Predicate name
    pub name: String,
    /// Predicate expression
    pub expression: String,
    /// Error message if validation fails
    pub error_message: String,
}

/// Semantic type registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTypeRegistry {
    /// Registered types
    pub types: HashMap<String, PIRSemanticType>,
    /// Type relationships
    pub relationships: HashMap<String, Vec<TypeRelationship>>,
    /// Global type constraints
    pub global_constraints: Vec<PIRTypeConstraint>,
}

/// Type relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Related type name
    pub related_type: String,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
}

/// Relationship type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Inheritance relationship
    Inherits,
    /// Implementation relationship
    Implements,
    /// Containment relationship
    Contains,
    /// Usage relationship
    Uses,
    /// Similarity relationship
    Similar,
}

/// Effect graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectGraph {
    /// Effect nodes
    pub nodes: HashMap<String, EffectNode>,
    /// Effect edges
    pub edges: Vec<EffectEdge>,
}

/// Effect node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectNode {
    /// Effect name
    pub name: String,
    /// Effect type
    pub effect_type: String,
    /// Required capabilities
    pub capabilities: Vec<String>,
    /// Side effects
    pub side_effects: Vec<String>,
}

/// Effect edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectEdge {
    /// Source effect
    pub source: String,
    /// Target effect
    pub target: String,
    /// Edge type
    pub edge_type: EffectEdgeType,
}

/// Effect edge type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectEdgeType {
    /// Requires relationship
    Requires,
    /// Provides relationship
    Provides,
    /// Conflicts relationship
    Conflicts,
    /// Enhances relationship
    Enhances,
}

/// Cohesion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionMetrics {
    /// Overall cohesion score
    pub overall_score: f64,
    /// Module cohesion scores
    pub module_scores: HashMap<String, f64>,
    /// Coupling metrics
    pub coupling_metrics: CouplingMetrics,
}

/// Coupling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    /// Afferent coupling (incoming dependencies)
    pub afferent: HashMap<String, u32>,
    /// Efferent coupling (outgoing dependencies)
    pub efferent: HashMap<String, u32>,
    /// Instability measure
    pub instability: HashMap<String, f64>,
}

/// PIR metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIRMetadata {
    /// PIR version
    pub version: String,
    /// Creation timestamp
    pub created_at: String,
    /// Source AST hash
    pub source_hash: u64,
    /// Optimization level applied
    pub optimization_level: u8,
    /// Target platforms
    pub target_platforms: Vec<String>,
}

impl PrismIR {
    /// Create a new empty PIR
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            type_registry: SemanticTypeRegistry {
                types: HashMap::new(),
                relationships: HashMap::new(),
                global_constraints: Vec::new(),
            },
            effect_graph: EffectGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
            },
            cohesion_metrics: CohesionMetrics {
                overall_score: 0.0,
                module_scores: HashMap::new(),
                coupling_metrics: CouplingMetrics {
                    afferent: HashMap::new(),
                    efferent: HashMap::new(),
                    instability: HashMap::new(),
                },
            },
            ai_metadata: crate::ai_integration::AIMetadata::default(),
            metadata: PIRMetadata {
                version: crate::PIRVersion::CURRENT.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                source_hash: 0,
                optimization_level: 0,
                target_platforms: Vec::new(),
            },
        }
    }

    /// Get a module by name
    pub fn get_module(&self, name: &str) -> Option<&PIRModule> {
        self.modules.iter().find(|module| module.name == name)
    }

    /// Get a mutable reference to a module by name
    pub fn get_module_mut(&mut self, name: &str) -> Option<&mut PIRModule> {
        self.modules.iter_mut().find(|module| module.name == name)
    }

    /// Add a module to the PIR
    pub fn add_module(&mut self, module: PIRModule) {
        self.modules.push(module);
    }

    /// Get the overall cohesion score
    pub fn cohesion_score(&self) -> f64 {
        self.cohesion_metrics.overall_score
    }
}

impl Default for PrismIR {
    fn default() -> Self {
        Self::new()
    }
} 