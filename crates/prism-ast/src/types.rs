//! Type system AST nodes for the Prism programming language
//!
//! This module implements Prism's semantic type system from PLD-001.

use crate::{AstNode, Expr};
use prism_common::symbol::Symbol;
use std::collections::HashMap;

/// Type AST node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Type {
    /// Primitive types
    Primitive(PrimitiveType),
    /// Named types
    Named(NamedType),
    /// Generic types
    Generic(GenericType),
    /// Function types
    Function(FunctionType),
    /// Tuple types
    Tuple(TupleType),
    /// Array types
    Array(ArrayType),
    /// Semantic types (Prism-specific)
    Semantic(SemanticType),
    /// Dependent types
    Dependent(DependentType),
    /// Effect types
    Effect(EffectType),
    /// Union types
    Union(Box<UnionType>),
    /// Intersection types
    Intersection(IntersectionType),
    /// Type-level computation result
    Computed(ComputedType),
    /// Error type (for recovery)
    Error(ErrorType),
}

/// Primitive type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PrimitiveType {
    /// Boolean type
    Boolean,
    /// Integer types
    Integer(IntegerType),
    /// Float types
    Float(FloatType),
    /// String type
    String,
    /// Character type
    Char,
    /// Unit type
    Unit,
    /// Never type
    Never,
}

/// Integer type variants
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IntegerType {
    /// Signed integers
    Signed(u8),  // bit width
    /// Unsigned integers
    Unsigned(u8),  // bit width
    /// Natural numbers (non-negative)
    Natural,
    /// Arbitrary precision integers
    BigInt,
}

/// Float type variants
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FloatType {
    /// IEEE 754 single precision
    F32,
    /// IEEE 754 double precision
    F64,
    /// Arbitrary precision decimal
    Decimal,
}

/// Named type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NamedType {
    /// Type name
    pub name: Symbol,
    /// Type arguments
    pub type_arguments: Vec<AstNode<Type>>,
}

/// Generic type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericType {
    /// Type parameters
    pub parameters: Vec<TypeParameter>,
    /// Base type
    pub base_type: Box<AstNode<Type>>,
}

/// Type parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypeParameter {
    /// Parameter name
    pub name: Symbol,
    /// Type bounds
    pub bounds: Vec<AstNode<Type>>,
    /// Default type
    pub default: Option<AstNode<Type>>,
}

/// Function type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionType {
    /// Parameter types
    pub parameters: Vec<AstNode<Type>>,
    /// Return type
    pub return_type: Box<AstNode<Type>>,
    /// Effect signature
    pub effects: Vec<Effect>,
}

/// Tuple type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TupleType {
    /// Element types
    pub elements: Vec<AstNode<Type>>,
}

/// Array type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayType {
    /// Element type
    pub element_type: Box<AstNode<Type>>,
    /// Array size (if known)
    pub size: Option<Box<AstNode<Expr>>>,
}

/// Semantic type (Prism's semantic type system)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemanticType {
    /// Base type
    pub base_type: Box<AstNode<Type>>,
    /// Type constraints
    pub constraints: Vec<TypeConstraint>,
    /// Semantic metadata
    pub metadata: SemanticTypeMetadata,
    /// Business domain
    pub business_domain: Option<String>,
    /// Static assertions
    pub static_assertions: Vec<StaticAssertion>,
    /// Type-level computations
    pub computations: Vec<TypeLevelComputation>,
}

/// Type constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TypeConstraint {
    /// Range constraint
    Range(RangeConstraint),
    /// Pattern constraint
    Pattern(PatternConstraint),
    /// Length constraint
    Length(LengthConstraint),
    /// Format constraint
    Format(FormatConstraint),
    /// Custom constraint
    Custom(CustomConstraint),
    /// Business rule constraint
    BusinessRule(BusinessRuleConstraint),
    /// Compile-time constraint
    CompileTime(CompileTimeConstraint),
    /// Dependent constraint
    Dependent(DependentConstraint),
}

/// Range constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RangeConstraint {
    /// Minimum value
    pub min: Option<AstNode<Expr>>,
    /// Maximum value
    pub max: Option<AstNode<Expr>>,
    /// Whether bounds are inclusive
    pub inclusive: bool,
}

/// Pattern constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PatternConstraint {
    /// Regular expression pattern
    pub pattern: String,
    /// Pattern flags
    pub flags: Vec<String>,
}

/// Length constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LengthConstraint {
    /// Minimum length
    pub min_length: Option<usize>,
    /// Maximum length
    pub max_length: Option<usize>,
}

/// Format constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormatConstraint {
    /// Format specification
    pub format: String,
    /// Format parameters
    pub parameters: HashMap<String, String>,
}

/// Custom constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint expression
    pub expression: AstNode<Expr>,
}

/// Business rule constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusinessRuleConstraint {
    /// Rule description
    pub description: String,
    /// Rule expression
    pub expression: AstNode<Expr>,
    /// Rule priority
    pub priority: u8,
}

/// Compile-time constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompileTimeConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint predicate (must be evaluable at compile time)
    pub predicate: AstNode<Expr>,
    /// Error message
    pub error_message: String,
    /// Constraint priority
    pub priority: ConstraintPriority,
    /// Whether this is a static assertion
    pub is_static_assertion: bool,
}

/// Dependent constraint that depends on other types/values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DependentConstraint {
    /// Constraint expression
    pub expression: AstNode<Expr>,
    /// Dependencies (types or values this constraint depends on)
    pub dependencies: Vec<DependentParameter>,
    /// Constraint evaluation strategy
    pub evaluation_strategy: DependentEvaluationStrategy,
}

/// Constraint priority for ordering evaluation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConstraintPriority {
    /// Low priority (performance hints)
    Low,
    /// Medium priority (business rules)
    Medium,
    /// High priority (safety constraints)
    High,
    /// Critical priority (security, correctness)
    Critical,
}

/// Dependent evaluation strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DependentEvaluationStrategy {
    /// Eager evaluation (compute at type definition time)
    Eager,
    /// Lazy evaluation (compute when needed)
    Lazy,
    /// Cached evaluation (compute once, cache result)
    Cached,
}

/// Semantic type metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemanticTypeMetadata {
    /// Business rules
    pub business_rules: Vec<String>,
    /// Examples
    pub examples: Vec<String>,
    /// Validation rules
    pub validation_rules: Vec<String>,
    /// AI context
    pub ai_context: Option<String>,
    /// Security classification
    pub security_classification: SecurityClassification,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
}

/// Security classification
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SecurityClassification {
    /// Public data
    Public,
    /// Internal data
    Internal,
    /// Confidential data
    Confidential,
    /// Restricted data
    Restricted,
    /// Top secret data
    TopSecret,
}

/// Dependent type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DependentType {
    /// Type parameters
    pub parameters: Vec<DependentParameter>,
    /// Dependent type expression
    pub type_expression: AstNode<Expr>,
    /// Parameter constraints
    pub parameter_constraints: Vec<ParameterConstraint>,
    /// Type-level computation rules
    pub computation_rules: Vec<ComputationRule>,
    /// Validation predicates
    pub validation_predicates: Vec<ValidationPredicate>,
    /// Refinement conditions
    pub refinement_conditions: Vec<RefinementCondition>,
}

/// Dependent type parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DependentParameter {
    /// Parameter name
    pub name: Symbol,
    /// Parameter type
    pub parameter_type: AstNode<Type>,
    /// Parameter bounds
    pub bounds: Vec<ParameterBound>,
    /// Default value
    pub default_value: Option<AstNode<Expr>>,
    /// Is compile-time constant
    pub is_compile_time_constant: bool,
    /// Parameter role in dependent type
    pub role: ParameterRole,
}

/// Parameter constraint for dependent types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterConstraint {
    /// Constraint expression
    pub expression: AstNode<Expr>,
    /// Constraint description
    pub description: String,
    /// Constraint severity
    pub severity: ConstraintSeverity,
}

/// Parameter bound
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterBound {
    /// Lower bound
    Lower(AstNode<Expr>),
    /// Upper bound
    Upper(AstNode<Expr>),
    /// Equality bound
    Equal(AstNode<Expr>),
    /// Type bound
    Type(AstNode<Type>),
    /// Trait bound
    Trait(Symbol),
}

/// Parameter role in dependent type
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterRole {
    /// Size parameter (e.g., array length)
    Size,
    /// Index parameter (e.g., array index)
    Index,
    /// Capacity parameter (e.g., buffer capacity)
    Capacity,
    /// Precision parameter (e.g., decimal precision)
    Precision,
    /// Scale parameter (e.g., decimal scale)
    Scale,
    /// Custom parameter role
    Custom(String),
}

/// Type-level computation rule
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputationRule {
    /// Rule name
    pub name: String,
    /// Input parameters
    pub inputs: Vec<Symbol>,
    /// Output expression
    pub output: AstNode<Expr>,
    /// Rule conditions
    pub conditions: Vec<AstNode<Expr>>,
}

/// Validation predicate for dependent types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationPredicate {
    /// Predicate expression
    pub expression: AstNode<Expr>,
    /// Error message if validation fails
    pub error_message: String,
    /// Validation level
    pub level: ValidationLevel,
}

/// Refinement condition for dependent types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RefinementCondition {
    /// Condition expression
    pub condition: AstNode<Expr>,
    /// Refined type
    pub refined_type: AstNode<Type>,
    /// Condition description
    pub description: String,
}

/// Constraint severity
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConstraintSeverity {
    /// Error - must be satisfied
    Error,
    /// Warning - should be satisfied
    Warning,
    /// Info - informational constraint
    Info,
}

/// Validation level
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ValidationLevel {
    /// Compile-time validation
    CompileTime,
    /// Runtime validation
    Runtime,
    /// Both compile-time and runtime
    Both,
}

/// Effect type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectType {
    /// Base type
    pub base_type: Box<AstNode<Type>>,
    /// Effects
    pub effects: Vec<Effect>,
    /// Effect composition rules
    pub composition_rules: Vec<EffectCompositionRule>,
    /// Capability requirements
    pub capability_requirements: Vec<CapabilityRequirement>,
    /// Effect metadata
    pub metadata: EffectMetadata,
}

/// Effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Effect {
    /// IO effect
    IO(IOEffect),
    /// State effect
    State(StateEffect),
    /// Exception effect
    Exception(ExceptionEffect),
    /// Async effect
    Async(AsyncEffect),
    /// Database effect
    Database(DatabaseEffect),
    /// Network effect
    Network(NetworkEffect),
    /// File system effect
    FileSystem(FileSystemEffect),
    /// Memory effect
    Memory(MemoryEffect),
    /// Computation effect
    Computation(ComputationEffect),
    /// Security effect
    Security(SecurityEffect),
    /// Custom effect
    Custom(CustomEffect),
}

/// IO effect with enhanced capabilities
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IOEffect {
    /// IO operations
    pub operations: Vec<IOOperation>,
    /// IO resources
    pub resources: Vec<IOResource>,
    /// IO constraints
    pub constraints: Vec<IOConstraint>,
}

/// IO operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IOOperation {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Append operation
    Append,
    /// Create operation
    Create,
    /// Delete operation
    Delete,
    /// List operation
    List,
    /// Seek operation
    Seek,
}

/// IO resource
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IOResource {
    /// Resource type
    pub resource_type: IOResourceType,
    /// Resource identifier
    pub identifier: String,
    /// Access pattern
    pub access_pattern: AccessPattern,
}

/// IO resource type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IOResourceType {
    /// File resource
    File,
    /// Directory resource
    Directory,
    /// Network socket
    Socket,
    /// Database connection
    Database,
    /// Standard input/output
    StandardIO,
    /// Custom resource
    Custom(String),
}

/// IO constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IOConstraint {
    /// Constraint type
    pub constraint_type: IOConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// IO constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IOConstraintType {
    /// Maximum file size
    MaxFileSize,
    /// Maximum connections
    MaxConnections,
    /// Timeout constraint
    Timeout,
    /// Rate limit
    RateLimit,
    /// Custom constraint
    Custom(String),
}

/// Async effect with enhanced capabilities
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AsyncEffect {
    /// Async operations
    pub operations: Vec<AsyncOperation>,
    /// Concurrency constraints
    pub concurrency_constraints: Vec<ConcurrencyConstraint>,
    /// Scheduling requirements
    pub scheduling_requirements: Vec<SchedulingRequirement>,
}

/// Async operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AsyncOperation {
    /// Await operation
    Await,
    /// Spawn operation
    Spawn,
    /// Join operation
    Join,
    /// Select operation
    Select,
    /// Timeout operation
    Timeout,
}

/// Concurrency constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConcurrencyConstraint {
    /// Constraint type
    pub constraint_type: ConcurrencyConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// Concurrency constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConcurrencyConstraintType {
    /// Maximum concurrent tasks
    MaxConcurrentTasks,
    /// Maximum memory usage
    MaxMemoryUsage,
    /// Maximum CPU usage
    MaxCpuUsage,
    /// Custom constraint
    Custom(String),
}

/// Scheduling requirement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SchedulingRequirement {
    /// Requirement type
    pub requirement_type: SchedulingRequirementType,
    /// Requirement value
    pub value: AstNode<Expr>,
}

/// Scheduling requirement type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SchedulingRequirementType {
    /// Priority requirement
    Priority,
    /// Deadline requirement
    Deadline,
    /// Affinity requirement
    Affinity,
    /// Custom requirement
    Custom(String),
}

/// Memory effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryEffect {
    /// Memory operations
    pub operations: Vec<MemoryOperation>,
    /// Memory constraints
    pub constraints: Vec<MemoryConstraint>,
}

/// Memory operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryOperation {
    /// Allocate operation
    Allocate,
    /// Deallocate operation
    Deallocate,
    /// Reallocate operation
    Reallocate,
    /// Copy operation
    Copy,
    /// Move operation
    Move,
}

/// Memory constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryConstraint {
    /// Constraint type
    pub constraint_type: MemoryConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// Memory constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryConstraintType {
    /// Maximum allocation size
    MaxAllocationSize,
    /// Maximum total memory
    MaxTotalMemory,
    /// Alignment requirement
    Alignment,
    /// Custom constraint
    Custom(String),
}

/// Computation effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputationEffect {
    /// Computation complexity
    pub complexity: ComputationComplexity,
    /// Resource requirements
    pub resource_requirements: Vec<ResourceRequirement>,
}

/// Computation complexity
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputationComplexity {
    /// Time complexity
    pub time_complexity: ComplexityBound,
    /// Space complexity
    pub space_complexity: ComplexityBound,
}

/// Complexity bound
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComplexityBound {
    /// Constant time/space
    Constant,
    /// Logarithmic
    Logarithmic,
    /// Linear
    Linear,
    /// Quadratic
    Quadratic,
    /// Exponential
    Exponential,
    /// Custom bound
    Custom(AstNode<Expr>),
}

/// Resource requirement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceRequirement {
    /// Resource type
    pub resource_type: ResourceType,
    /// Required amount
    pub amount: AstNode<Expr>,
}

/// Resource type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceType {
    /// CPU time
    CpuTime,
    /// Memory
    Memory,
    /// Network bandwidth
    NetworkBandwidth,
    /// Storage
    Storage,
    /// Custom resource
    Custom(String),
}

/// Security effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecurityEffect {
    /// Security operations
    pub operations: Vec<SecurityOperation>,
    /// Security constraints
    pub constraints: Vec<SecurityConstraint>,
}

/// Security operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SecurityOperation {
    /// Authentication
    Authentication,
    /// Authorization
    Authorization,
    /// Encryption
    Encryption,
    /// Decryption
    Decryption,
    /// Signing
    Signing,
    /// Verification
    Verification,
}

/// Security constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecurityConstraint {
    /// Constraint type
    pub constraint_type: SecurityConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// Security constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SecurityConstraintType {
    /// Minimum security level
    MinSecurityLevel,
    /// Required permissions
    RequiredPermissions,
    /// Encryption algorithm
    EncryptionAlgorithm,
    /// Custom constraint
    Custom(String),
}

/// Effect composition rule
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectCompositionRule {
    /// Rule name
    pub name: String,
    /// Input effects
    pub input_effects: Vec<Effect>,
    /// Output effect
    pub output_effect: Effect,
    /// Composition conditions
    pub conditions: Vec<AstNode<Expr>>,
}

/// Capability requirement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CapabilityRequirement {
    /// Capability name
    pub name: Symbol,
    /// Required permissions
    pub permissions: Vec<Permission>,
    /// Capability constraints
    pub constraints: Vec<CapabilityConstraint>,
}

/// Permission
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Permission {
    /// Permission name
    pub name: Symbol,
    /// Permission scope
    pub scope: PermissionScope,
    /// Permission level
    pub level: PermissionLevel,
}

/// Permission scope
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PermissionScope {
    /// Global scope
    Global,
    /// Module scope
    Module(Symbol),
    /// Function scope
    Function(Symbol),
    /// Type scope
    Type(Symbol),
    /// Custom scope
    Custom(String),
}

/// Permission level
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PermissionLevel {
    /// Read permission
    Read,
    /// Write permission
    Write,
    /// Read and write permission
    ReadWrite,
    /// Execute permission
    Execute,
    /// Admin permission
    Admin,
    /// Custom permission level
    Custom(String),
}

/// Capability constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CapabilityConstraint {
    /// Constraint expression
    pub expression: AstNode<Expr>,
    /// Constraint description
    pub description: String,
}

/// Effect metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectMetadata {
    /// Effect name
    pub name: Option<Symbol>,
    /// Effect description
    pub description: Option<String>,
    /// Effect category
    pub category: EffectCategory,
    /// AI context
    pub ai_context: Option<String>,
}

/// Effect category
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EffectCategory {
    /// Pure computation
    Pure,
    /// Side effects
    SideEffects,
    /// Resource management
    ResourceManagement,
    /// Security operations
    Security,
    /// Custom category
    Custom(String),
}

/// Union type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionType {
    /// Union members
    pub members: Vec<AstNode<Type>>,
    /// Union discriminant (for tagged unions)
    pub discriminant: Option<Box<UnionDiscriminant>>,
    /// Union constraints
    pub constraints: Vec<UnionConstraint>,
    /// Common operations supported by all members
    pub common_operations: Vec<OperationSignature>,
    /// Union metadata
    pub metadata: UnionMetadata,
}

/// Intersection type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntersectionType {
    /// Intersection members
    pub members: Vec<AstNode<Type>>,
    /// Intersection constraints
    pub constraints: Vec<IntersectionConstraint>,
    /// Merged operations from all members
    pub merged_operations: Vec<OperationSignature>,
    /// Intersection metadata
    pub metadata: IntersectionMetadata,
}

/// Union discriminant for tagged unions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionDiscriminant {
    /// Discriminant field name
    pub field_name: Symbol,
    /// Discriminant type
    pub discriminant_type: Box<AstNode<Type>>,
    /// Tag mapping
    pub tag_mapping: HashMap<String, usize>,
}

/// Union constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnionConstraint {
    /// Mutual exclusion constraint
    MutualExclusion(Vec<usize>),
    /// Minimum members constraint
    MinMembers(usize),
    /// Maximum members constraint
    MaxMembers(usize),
    /// Type compatibility constraint
    TypeCompatibility(CompatibilityRule),
    /// Custom union constraint
    Custom(String, AstNode<Expr>),
}

/// Intersection constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IntersectionConstraint {
    /// Compatibility constraint
    Compatibility(Vec<usize>),
    /// Conflict resolution constraint
    ConflictResolution(ConflictResolutionRule),
    /// Property merging constraint
    PropertyMerging(PropertyMergingRule),
    /// Custom intersection constraint
    Custom(String, AstNode<Expr>),
}

/// Compatibility rule for union types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompatibilityRule {
    /// Rule name
    pub name: String,
    /// Compatible type pairs
    pub compatible_pairs: Vec<(usize, usize)>,
    /// Compatibility predicate
    pub predicate: AstNode<Expr>,
}

/// Conflict resolution rule for intersection types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConflictResolutionRule {
    /// Rule name
    pub name: String,
    /// Conflict detection predicate
    pub conflict_predicate: AstNode<Expr>,
    /// Resolution strategy
    pub resolution_strategy: ResolutionStrategy,
}

/// Property merging rule for intersection types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PropertyMergingRule {
    /// Rule name
    pub name: String,
    /// Property selector
    pub property_selector: AstNode<Expr>,
    /// Merging strategy
    pub merging_strategy: MergingStrategy,
}

/// Resolution strategy for conflicts
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolutionStrategy {
    /// Use first occurrence
    First,
    /// Use last occurrence
    Last,
    /// Use most specific type
    MostSpecific,
    /// Use least specific type
    LeastSpecific,
    /// Custom resolution function
    Custom(AstNode<Expr>),
}

/// Merging strategy for properties
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MergingStrategy {
    /// Union merge
    Union,
    /// Intersection merge
    Intersection,
    /// Override merge
    Override,
    /// Combine merge
    Combine(AstNode<Expr>),
}

/// Operation signature for type operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OperationSignature {
    /// Operation name
    pub name: Symbol,
    /// Input types
    pub inputs: Vec<AstNode<Type>>,
    /// Output type
    pub output: AstNode<Type>,
    /// Operation effects
    pub effects: Vec<Effect>,
    /// Operation constraints
    pub constraints: Vec<OperationConstraint>,
}

/// Operation constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OperationConstraint {
    /// Constraint expression
    pub expression: AstNode<Expr>,
    /// Constraint description
    pub description: String,
}

/// Union metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnionMetadata {
    /// Union name
    pub name: Option<Symbol>,
    /// Union description
    pub description: Option<String>,
    /// Union semantics
    pub semantics: UnionSemantics,
    /// AI context
    pub ai_context: Option<String>,
}

/// Intersection metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntersectionMetadata {
    /// Intersection name
    pub name: Option<Symbol>,
    /// Intersection description
    pub description: Option<String>,
    /// Intersection semantics
    pub semantics: IntersectionSemantics,
    /// AI context
    pub ai_context: Option<String>,
}

impl Default for UnionMetadata {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            semantics: UnionSemantics::Disjoint,
            ai_context: None,
        }
    }
}

/// Union semantics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnionSemantics {
    /// Disjoint union (tagged)
    Disjoint,
    /// Overlapping union
    Overlapping,
    /// Exclusive union
    Exclusive,
    /// Inclusive union
    Inclusive,
}

/// Intersection semantics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IntersectionSemantics {
    /// Structural intersection
    Structural,
    /// Nominal intersection
    Nominal,
    /// Behavioral intersection
    Behavioral,
    /// Hybrid intersection
    Hybrid,
}

/// Error type (for recovery)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErrorType {
    /// Error message
    pub message: String,
}

/// Type-level computation result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputedType {
    /// The computed type expression
    pub computation: TypeLevelComputation,
    /// The resolved type (if computed)
    pub resolved_type: Option<Box<AstNode<Type>>>,
    /// Computation metadata
    pub metadata: ComputationMetadata,
}

/// Type-level computation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TypeLevelComputation {
    /// Type function application
    FunctionApplication(TypeFunctionApplication),
    /// Static assertion
    StaticAssertion(StaticAssertion),
    /// Conditional type
    Conditional(ConditionalType),
    /// Type arithmetic
    Arithmetic(TypeArithmetic),
    /// Constraint validation
    ConstraintValidation(ConstraintValidation),
}

/// Type function application
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypeFunctionApplication {
    /// Function name
    pub function_name: Symbol,
    /// Type arguments
    pub type_args: Vec<AstNode<Type>>,
    /// Value arguments
    pub value_args: Vec<AstNode<Expr>>,
    /// Expected return type
    pub return_type: Box<AstNode<Type>>,
}

/// Static assertion at type level
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StaticAssertion {
    /// Assertion condition
    pub condition: AstNode<Expr>,
    /// Error message if assertion fails
    pub error_message: String,
    /// Assertion context
    pub context: StaticAssertionContext,
}

/// Static assertion context
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StaticAssertionContext {
    /// Type constraint validation
    TypeConstraint,
    /// Business rule validation
    BusinessRule,
    /// Security requirement
    SecurityRequirement,
    /// Performance constraint
    PerformanceConstraint,
}

/// Conditional type (compile-time if-then-else for types)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditionalType {
    /// Condition to evaluate
    pub condition: AstNode<Expr>,
    /// Type if condition is true
    pub then_type: Box<AstNode<Type>>,
    /// Type if condition is false
    pub else_type: Box<AstNode<Type>>,
}

/// Type arithmetic for dependent types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypeArithmetic {
    /// Arithmetic operation
    pub operation: TypeArithmeticOp,
    /// Left operand
    pub left: Box<AstNode<Type>>,
    /// Right operand
    pub right: Box<AstNode<Type>>,
}

/// Type arithmetic operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TypeArithmeticOp {
    /// Type union
    Union,
    /// Type intersection
    Intersection,
    /// Type subtraction
    Subtraction,
    /// Size addition (for arrays/vectors)
    SizeAdd,
    /// Size multiplication
    SizeMultiply,
}

/// Constraint validation at type level
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintValidation {
    /// Type to validate
    pub target_type: Box<AstNode<Type>>,
    /// Constraints to check
    pub constraints: Vec<TypeConstraint>,
    /// Validation strategy
    pub strategy: ValidationStrategy,
}

/// Validation strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ValidationStrategy {
    /// Validate all constraints (AND)
    All,
    /// Validate any constraint (OR)
    Any,
    /// Custom validation logic
    Custom(AstNode<Expr>),
}

/// Computation metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputationMetadata {
    /// Whether computation was successful
    pub computed: bool,
    /// Computation time (if available)
    pub computation_time_us: Option<u64>,
    /// Dependencies used in computation
    pub dependencies: Vec<Symbol>,
    /// Errors during computation
    pub errors: Vec<String>,
    /// Warnings during computation
    pub warnings: Vec<String>,
}

/// Type declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypeDecl {
    /// Type name
    pub name: Symbol,
    /// Type parameters
    pub type_parameters: Vec<TypeParameter>,
    /// Type kind
    pub kind: TypeKind,
    /// Visibility
    pub visibility: Visibility,
}

/// Type kind
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TypeKind {
    /// Type alias
    Alias(AstNode<Type>),
    /// Semantic type
    Semantic(SemanticType),
    /// Enum type
    Enum(EnumType),
    /// Struct type
    Struct(StructType),
    /// Trait type
    Trait(TraitType),
}

/// Enum type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnumType {
    /// Enum variants
    pub variants: Vec<EnumVariant>,
}

/// Enum variant
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnumVariant {
    /// Variant name
    pub name: Symbol,
    /// Variant fields
    pub fields: Vec<AstNode<Type>>,
}

/// Struct type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructType {
    /// Struct fields
    pub fields: Vec<StructField>,
}

/// Struct field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructField {
    /// Field name
    pub name: Symbol,
    /// Field type
    pub field_type: AstNode<Type>,
    /// Field visibility
    pub visibility: Visibility,
}

/// Trait type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraitType {
    /// Trait methods
    pub methods: Vec<TraitMethod>,
}

/// Trait method
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TraitMethod {
    /// Method name
    pub name: Symbol,
    /// Method type
    pub method_type: FunctionType,
}

/// Visibility
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Visibility {
    /// Public visibility
    Public,
    /// Private visibility
    Private,
    /// Internal visibility
    Internal,
}

/// State effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StateEffect {
    /// State type
    pub state_type: AstNode<Type>,
    /// Access pattern
    pub access_pattern: AccessPattern,
    /// State constraints
    pub constraints: Vec<StateConstraint>,
}

/// Access pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AccessPattern {
    /// Read-only access
    Read,
    /// Write-only access
    Write,
    /// Read-write access
    ReadWrite,
}

/// State constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StateConstraint {
    /// Constraint type
    pub constraint_type: StateConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// State constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StateConstraintType {
    /// Immutable state
    Immutable,
    /// Thread-safe state
    ThreadSafe,
    /// Persistent state
    Persistent,
    /// Custom constraint
    Custom(String),
}

/// Exception effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExceptionEffect {
    /// Exception types that can be thrown
    pub exception_types: Vec<AstNode<Type>>,
    /// Exception handling strategy
    pub handling_strategy: ExceptionHandlingStrategy,
}

/// Exception handling strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExceptionHandlingStrategy {
    /// Propagate exceptions
    Propagate,
    /// Handle exceptions locally
    LocalHandle,
    /// Transform exceptions
    Transform,
    /// Custom strategy
    Custom(String),
}

/// Database effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DatabaseEffect {
    /// Database operations
    pub operations: Vec<DatabaseOperation>,
    /// Transaction requirements
    pub transaction_required: bool,
    /// Database constraints
    pub constraints: Vec<DatabaseConstraint>,
}

/// Database operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DatabaseOperation {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Delete operation
    Delete,
    /// Schema operation
    Schema,
    /// Index operation
    Index,
    /// Migration operation
    Migration,
}

/// Database constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DatabaseConstraint {
    /// Constraint type
    pub constraint_type: DatabaseConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// Database constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DatabaseConstraintType {
    /// Maximum connections
    MaxConnections,
    /// Query timeout
    QueryTimeout,
    /// Transaction isolation level
    IsolationLevel,
    /// Custom constraint
    Custom(String),
}

/// Network effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NetworkEffect {
    /// Network protocols
    pub protocols: Vec<String>,
    /// Network endpoints
    pub endpoints: Vec<String>,
    /// Network constraints
    pub constraints: Vec<NetworkConstraint>,
}

/// Network constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NetworkConstraint {
    /// Constraint type
    pub constraint_type: NetworkConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// Network constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NetworkConstraintType {
    /// Maximum bandwidth
    MaxBandwidth,
    /// Connection timeout
    ConnectionTimeout,
    /// Retry count
    RetryCount,
    /// Custom constraint
    Custom(String),
}

/// File system effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FileSystemEffect {
    /// File operations
    pub operations: Vec<FileOperation>,
    /// File paths
    pub paths: Vec<String>,
    /// File constraints
    pub constraints: Vec<FileConstraint>,
}

/// File operation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FileOperation {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Delete operation
    Delete,
    /// Create operation
    Create,
    /// Move operation
    Move,
    /// Copy operation
    Copy,
}

/// File constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FileConstraint {
    /// Constraint type
    pub constraint_type: FileConstraintType,
    /// Constraint value
    pub value: AstNode<Expr>,
}

/// File constraint type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FileConstraintType {
    /// Maximum file size
    MaxFileSize,
    /// File permissions
    FilePermissions,
    /// File type
    FileType,
    /// Custom constraint
    Custom(String),
}

/// Custom effect
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomEffect {
    /// Effect name
    pub name: String,
    /// Effect parameters
    pub parameters: HashMap<String, AstNode<Expr>>,
    /// Effect constraints
    pub constraints: Vec<CustomEffectConstraint>,
}

/// Custom effect constraint
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomEffectConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint expression
    pub expression: AstNode<Expr>,
    /// Constraint description
    pub description: String,
}

impl Default for SecurityClassification {
    fn default() -> Self {
        Self::Public
    }
}

impl Default for SemanticTypeMetadata {
    fn default() -> Self {
        Self {
            business_rules: Vec::new(),
            examples: Vec::new(),
            validation_rules: Vec::new(),
            ai_context: None,
            security_classification: SecurityClassification::default(),
            compliance_requirements: Vec::new(),
        }
    }
} 