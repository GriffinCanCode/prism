//! Semantic database for AI-readable compilation output
//!
//! This module implements the comprehensive semantic database that preserves and
//! exposes semantic information throughout compilation for AI consumption.

use crate::error::{CompilerError, CompilerResult};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_ast::{Program, AstNode, Expr, Stmt, Type};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Semantic database for AI integration
#[derive(Debug)]
pub struct SemanticDatabase {
    /// Symbol information
    symbols: Arc<RwLock<HashMap<Symbol, SymbolInfo>>>,
    /// Type relationships
    type_relationships: Arc<RwLock<HashMap<NodeId, TypeRelationships>>>,
    /// Effect signatures
    effect_signatures: Arc<RwLock<HashMap<NodeId, EffectSignature>>>,
    /// Call graph
    call_graph: Arc<RwLock<CallGraph>>,
    /// Data flow graph
    data_flow_graph: Arc<RwLock<DataFlowGraph>>,
    /// AI metadata
    ai_metadata: Arc<RwLock<HashMap<NodeId, AIMetadata>>>,
    /// Configuration
    config: SemanticConfig,
}

/// Configuration for semantic analysis
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    /// Enable call graph analysis
    pub enable_call_graph: bool,
    /// Enable data flow analysis
    pub enable_data_flow: bool,
    /// Enable effect tracking
    pub enable_effect_tracking: bool,
}

/// Symbol information for AI understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol identifier
    pub id: Symbol,
    /// Symbol name
    pub name: String,
    /// Type information
    pub type_info: TypeInfo,
    /// Source location
    pub source_location: Span,
    /// Visibility
    pub visibility: Visibility,
    /// Semantic annotations
    pub semantic_annotations: Vec<String>,
    /// Business context
    pub business_context: Option<BusinessContext>,
    /// AI hints
    pub ai_hints: Vec<String>,
}

/// Type information with semantic meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type ID
    pub type_id: NodeId,
    /// Type kind
    pub type_kind: TypeKind,
    /// Type parameters
    pub type_parameters: Vec<TypeParameter>,
    /// Type constraints
    pub constraints: Vec<TypeConstraint>,
    /// Semantic meaning
    pub semantic_meaning: SemanticMeaning,
    /// AI-readable description
    pub ai_description: Option<String>,
}

/// Type kind classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeKind {
    /// Primitive type (i32, f64, bool, etc.)
    Primitive(PrimitiveType),
    /// Semantic type with business meaning
    Semantic(SemanticType),
    /// Composite type (struct, enum, union)
    Composite(CompositeType),
    /// Function type
    Function(FunctionType),
    /// Generic type parameter
    Generic(GenericType),
    /// Effect type
    Effect(EffectType),
}

/// Primitive type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimitiveType {
    /// Signed integer
    SignedInt(u8), // bit width
    /// Unsigned integer
    UnsignedInt(u8), // bit width
    /// Floating point
    Float(u8), // bit width
    /// Boolean
    Bool,
    /// String
    String,
    /// Unit type
    Unit,
}

/// Semantic type with business meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticType {
    /// Base type
    pub base_type: Box<TypeKind>,
    /// Semantic domain
    pub domain: String,
    /// Business rules
    pub business_rules: Vec<BusinessRule>,
    /// Validation predicates
    pub validation_predicates: Vec<ValidationPredicate>,
    /// AI context
    pub ai_context: AITypeContext,
}

/// Composite type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeType {
    /// Composite kind
    pub kind: CompositeKind,
    /// Fields or variants
    pub fields: Vec<FieldInfo>,
    /// Methods
    pub methods: Vec<MethodInfo>,
}

/// Composite type kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositeKind {
    /// Struct
    Struct,
    /// Enum
    Enum,
    /// Union
    Union,
    /// Tuple
    Tuple,
}

/// Field information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: TypeInfo,
    /// Visibility
    pub visibility: Visibility,
    /// Documentation
    pub documentation: Option<String>,
    /// AI hints
    pub ai_hints: Vec<String>,
}

/// Method information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodInfo {
    /// Method name
    pub name: String,
    /// Method signature
    pub signature: FunctionType,
    /// Implementation location
    pub implementation: Option<Span>,
    /// Documentation
    pub documentation: Option<String>,
}

/// Function type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionType {
    /// Parameter types
    pub parameters: Vec<ParameterInfo>,
    /// Return type
    pub return_type: Box<TypeInfo>,
    /// Effect signature
    pub effects: EffectSignature,
    /// Contracts
    pub contracts: ContractSpecification,
}

/// Parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: TypeInfo,
    /// Default value
    pub default_value: Option<String>,
    /// AI description
    pub ai_description: Option<String>,
}

/// Generic type parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericType {
    /// Type parameter name
    pub name: String,
    /// Bounds
    pub bounds: Vec<TypeConstraint>,
    /// Default type
    pub default_type: Option<Box<TypeInfo>>,
}

/// Effect type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectType {
    /// Effect name
    pub name: String,
    /// Effect operations
    pub operations: Vec<EffectOperation>,
    /// Capabilities required
    pub capabilities: Vec<String>,
}

/// Effect operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectOperation {
    /// Operation name
    pub name: String,
    /// Operation signature
    pub signature: FunctionType,
    /// Side effects
    pub side_effects: Vec<String>,
}

/// Type parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeParameter {
    /// Parameter name
    pub name: String,
    /// Parameter bounds
    pub bounds: Vec<TypeConstraint>,
    /// Variance
    pub variance: Variance,
}

/// Type variance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Variance {
    /// Covariant
    Covariant,
    /// Contravariant
    Contravariant,
    /// Invariant
    Invariant,
}

/// Type constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// Constraint kind
    pub kind: ConstraintKind,
    /// Target type
    pub target_type: Option<TypeInfo>,
    /// Predicate
    pub predicate: Option<String>,
}

/// Type constraint kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Trait bound
    TraitBound(String),
    /// Lifetime bound
    LifetimeBound(String),
    /// Size constraint
    SizeConstraint(SizeConstraint),
    /// Value constraint
    ValueConstraint(ValueConstraint),
    /// Business rule constraint
    BusinessRuleConstraint(BusinessRule),
}

/// Size constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SizeConstraint {
    /// Fixed size
    Fixed(usize),
    /// Maximum size
    Maximum(usize),
    /// Minimum size
    Minimum(usize),
    /// Range
    Range(usize, usize),
}

/// Value constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueConstraint {
    /// Range constraint
    Range(String, String), // min, max as expressions
    /// Pattern constraint
    Pattern(String),
    /// Enum constraint
    Enum(Vec<String>),
    /// Custom predicate
    Custom(String),
}

/// Business rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule predicate
    pub predicate: String,
    /// Error message
    pub error_message: String,
    /// AI explanation
    pub ai_explanation: Option<String>,
}

/// Validation predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPredicate {
    /// Predicate name
    pub name: String,
    /// Predicate expression
    pub expression: String,
    /// Error message
    pub error_message: String,
}

/// Semantic meaning of a type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMeaning {
    /// Domain
    pub domain: String,
    /// Purpose
    pub purpose: String,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// Business entities
    pub business_entities: Vec<String>,
}

/// AI type context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AITypeContext {
    /// Intent description
    pub intent: Option<String>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common mistakes
    pub common_mistakes: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
    /// Performance characteristics
    pub performance_characteristics: Vec<String>,
}

/// Symbol visibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    /// Public
    Public,
    /// Private
    Private,
    /// Protected
    Protected,
    /// Internal
    Internal,
}

/// Business context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Business domain
    pub domain: String,
    /// Business entities
    pub entities: Vec<String>,
    /// Business relationships
    pub relationships: Vec<String>,
    /// Business constraints
    pub constraints: Vec<String>,
}

/// Type relationships for AI understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRelationships {
    /// Inheritance relationships
    pub inheritance: Vec<InheritanceRelation>,
    /// Composition relationships
    pub composition: Vec<CompositionRelation>,
    /// Dependency relationships
    pub dependencies: Vec<DependencyRelation>,
    /// Usage relationships
    pub usages: Vec<UsageRelation>,
}

/// Inheritance relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceRelation {
    /// Parent type
    pub parent: NodeId,
    /// Child type
    pub child: NodeId,
    /// Relationship kind
    pub kind: InheritanceKind,
}

/// Inheritance kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InheritanceKind {
    /// Interface implementation
    Interface,
    /// Class inheritance
    Class,
    /// Trait implementation
    Trait,
}

/// Composition relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRelation {
    /// Container type
    pub container: NodeId,
    /// Component type
    pub component: NodeId,
    /// Relationship strength
    pub strength: CompositionStrength,
}

/// Composition strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionStrength {
    /// Strong composition (ownership)
    Strong,
    /// Weak composition (reference)
    Weak,
    /// Aggregation
    Aggregation,
}

/// Dependency relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyRelation {
    /// Dependent type
    pub dependent: NodeId,
    /// Dependency type
    pub dependency: NodeId,
    /// Dependency kind
    pub kind: DependencyKind,
}

/// Dependency kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyKind {
    /// Uses
    Uses,
    /// Imports
    Imports,
    /// Extends
    Extends,
    /// Implements
    Implements,
}

/// Usage relation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRelation {
    /// User type
    pub user: NodeId,
    /// Used type
    pub used: NodeId,
    /// Usage context
    pub context: UsageContext,
}

/// Usage context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageContext {
    /// Parameter
    Parameter,
    /// Return type
    ReturnType,
    /// Field type
    FieldType,
    /// Local variable
    LocalVariable,
    /// Generic argument
    GenericArgument,
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
    /// Capability requirements
    pub capability_requirements: Vec<Capability>,
}

/// Effect information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    /// Effect name
    pub name: String,
    /// Effect type
    pub effect_type: String,
    /// Effect parameters
    pub parameters: Vec<String>,
    /// AI description
    pub ai_description: Option<String>,
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
    /// Causal dependency
    Causal,
    /// Temporal dependency
    Temporal,
    /// Resource dependency
    Resource,
}

/// Capability requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Capability name
    pub name: String,
    /// Capability type
    pub capability_type: String,
    /// Required permissions
    pub permissions: Vec<String>,
    /// AI description
    pub ai_description: Option<String>,
}

/// Call graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraph {
    /// Nodes (functions)
    pub nodes: HashMap<NodeId, CallGraphNode>,
    /// Edges (calls)
    pub edges: Vec<CallGraphEdge>,
}

/// Call graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphNode {
    /// Function ID
    pub function_id: NodeId,
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: FunctionType,
    /// Call count
    pub call_count: u64,
}

/// Call graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphEdge {
    /// Caller function
    pub caller: NodeId,
    /// Callee function
    pub callee: NodeId,
    /// Call sites
    pub call_sites: Vec<Span>,
    /// Call frequency
    pub frequency: u64,
}

/// Data flow graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowGraph {
    /// Nodes (variables, expressions)
    pub nodes: HashMap<NodeId, DataFlowNode>,
    /// Edges (data dependencies)
    pub edges: Vec<DataFlowEdge>,
}

/// Data flow node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowNode {
    /// Node ID
    pub node_id: NodeId,
    /// Node type
    pub node_type: DataFlowNodeType,
    /// Type information
    pub type_info: TypeInfo,
    /// Source location
    pub location: Span,
}

/// Data flow node type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFlowNodeType {
    /// Variable definition
    Variable,
    /// Function parameter
    Parameter,
    /// Function call
    FunctionCall,
    /// Field access
    FieldAccess,
    /// Array access
    ArrayAccess,
    /// Literal
    Literal,
}

/// Data flow edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowEdge {
    /// Source node
    pub source: NodeId,
    /// Target node
    pub target: NodeId,
    /// Edge type
    pub edge_type: DataFlowEdgeType,
}

/// Data flow edge type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFlowEdgeType {
    /// Definition-use
    DefUse,
    /// Control dependency
    Control,
    /// Data dependency
    Data,
}

/// Contract specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSpecification {
    /// Preconditions
    pub preconditions: Vec<Condition>,
    /// Postconditions
    pub postconditions: Vec<Condition>,
    /// Invariants
    pub invariants: Vec<Condition>,
}

/// Contract condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Condition name
    pub name: String,
    /// Condition expression
    pub expression: String,
    /// Error message
    pub error_message: String,
    /// AI explanation
    pub ai_explanation: Option<String>,
}

/// AI metadata for compilation artifacts - Enhanced for PLD-010 compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Module context with rich semantic information
    pub module_context: Option<ModuleAIContext>,
    /// Function contexts with algorithm descriptions and complexity analysis
    pub function_contexts: HashMap<NodeId, FunctionAIContext>,
    /// Type contexts with business rules and validation predicates
    pub type_contexts: HashMap<NodeId, TypeAIContext>,
    /// Semantic relationships between code elements
    pub relationships: SemanticRelationships,
    /// Business context and domain knowledge
    pub business_context: Option<BusinessContext>,
    /// Performance characteristics and optimization hints
    pub performance_metadata: PerformanceMetadata,
    /// Security implications and classifications
    pub security_metadata: SecurityMetadata,
    /// Compliance requirements and regulatory information
    pub compliance_metadata: ComplianceMetadata,
    /// Cross-target consistency information
    pub consistency_metadata: ConsistencyMetadata,
    /// AI comprehension aids
    pub comprehension_aids: ComprehensionAids,
}

/// Module AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleAIContext {
    /// Module purpose
    pub purpose: String,
    /// Capabilities provided
    pub capabilities: Vec<String>,
    /// Responsibilities
    pub responsibilities: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Business domain
    pub business_domain: String,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Security considerations
    pub security_considerations: Vec<String>,
}

/// Function AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAIContext {
    /// Function purpose
    pub purpose: String,
    /// Algorithm description
    pub algorithm: Option<String>,
    /// Complexity analysis
    pub complexity: Option<ComplexityAnalysis>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Common mistakes
    pub common_mistakes: Vec<String>,
    /// Performance characteristics
    pub performance_characteristics: Vec<String>,
}

/// Type AI context (already defined above as AITypeContext)
pub type TypeAIContext = AITypeContext;

/// Complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysis {
    /// Time complexity
    pub time_complexity: String,
    /// Space complexity
    pub space_complexity: String,
    /// Best case
    pub best_case: Option<String>,
    /// Average case
    pub average_case: Option<String>,
    /// Worst case
    pub worst_case: Option<String>,
}

/// Semantic relationships between code elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelationships {
    /// Type relationships
    pub type_relationships: HashMap<NodeId, TypeRelationships>,
    /// Function relationships
    pub function_relationships: HashMap<NodeId, FunctionRelationships>,
    /// Module relationships
    pub module_relationships: HashMap<NodeId, ModuleRelationships>,
}

/// Function relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRelationships {
    /// Calls made by this function
    pub calls: Vec<NodeId>,
    /// Functions that call this function
    pub callers: Vec<NodeId>,
    /// Functions with similar signatures
    pub similar_functions: Vec<NodeId>,
    /// Overloaded functions
    pub overloads: Vec<NodeId>,
}

/// Module relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRelationships {
    /// Modules imported by this module
    pub imports: Vec<NodeId>,
    /// Modules that import this module
    pub importers: Vec<NodeId>,
    /// Related modules
    pub related_modules: Vec<NodeId>,
}

/// Semantic information for a program element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInfo {
    /// Symbol table
    pub symbols: HashMap<Symbol, SymbolInfo>,
    /// Type information
    pub types: HashMap<NodeId, TypeInfo>,
    /// Effect signatures
    pub effects: HashMap<NodeId, EffectSignature>,
    /// Contracts
    pub contracts: HashMap<NodeId, ContractSpecification>,
    /// AI metadata
    pub ai_metadata: AIMetadata,
}

impl SemanticDatabase {
    /// Create a new semantic database
    pub fn new(config: &crate::CompilerConfig) -> CompilerResult<Self> {
        let semantic_config = SemanticConfig {
            enable_ai_metadata: config.ai_metadata,
            enable_call_graph: true,
            enable_data_flow: true,
            enable_effect_tracking: true,
        };

        Ok(Self {
            symbols: Arc::new(RwLock::new(HashMap::new())),
            type_relationships: Arc::new(RwLock::new(HashMap::new())),
            effect_signatures: Arc::new(RwLock::new(HashMap::new())),
            call_graph: Arc::new(RwLock::new(CallGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
            })),
            data_flow_graph: Arc::new(RwLock::new(DataFlowGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
            })),
            ai_metadata: Arc::new(RwLock::new(HashMap::new())),
            config: semantic_config,
        })
    }

    /// Add symbol information
    pub fn add_symbol(&self, symbol: Symbol, info: SymbolInfo) -> CompilerResult<()> {
        let mut symbols = self.symbols.write().unwrap();
        symbols.insert(symbol, info);
        Ok(())
    }

    /// Get symbol information
    pub fn get_symbol(&self, symbol: &Symbol) -> Option<SymbolInfo> {
        let symbols = self.symbols.read().unwrap();
        symbols.get(symbol).cloned()
    }

    /// Add type relationships
    pub fn add_type_relationships(&self, node_id: NodeId, relationships: TypeRelationships) -> CompilerResult<()> {
        let mut type_relationships = self.type_relationships.write().unwrap();
        type_relationships.insert(node_id, relationships);
        Ok(())
    }

    /// Get type relationships
    pub fn get_type_relationships(&self, node_id: &NodeId) -> Option<TypeRelationships> {
        let type_relationships = self.type_relationships.read().unwrap();
        type_relationships.get(node_id).cloned()
    }

    /// Add effect signature
    pub fn add_effect_signature(&self, node_id: NodeId, signature: EffectSignature) -> CompilerResult<()> {
        let mut effect_signatures = self.effect_signatures.write().unwrap();
        effect_signatures.insert(node_id, signature);
        Ok(())
    }

    /// Get effect signature
    pub fn get_effect_signature(&self, node_id: &NodeId) -> Option<EffectSignature> {
        let effect_signatures = self.effect_signatures.read().unwrap();
        effect_signatures.get(node_id).cloned()
    }

    /// Update call graph
    pub fn update_call_graph(&self, caller: NodeId, callee: NodeId, call_site: Span) -> CompilerResult<()> {
        let mut call_graph = self.call_graph.write().unwrap();
        
        // Add edge or update existing edge
        if let Some(edge) = call_graph.edges.iter_mut().find(|e| e.caller == caller && e.callee == callee) {
            edge.call_sites.push(call_site);
            edge.frequency += 1;
        } else {
            call_graph.edges.push(CallGraphEdge {
                caller,
                callee,
                call_sites: vec![call_site],
                frequency: 1,
            });
        }
        
        Ok(())
    }

    /// Get call graph
    pub fn get_call_graph(&self) -> CallGraph {
        let call_graph = self.call_graph.read().unwrap();
        call_graph.clone()
    }

    /// Add AI metadata
    pub fn add_ai_metadata(&self, node_id: NodeId, metadata: AIMetadata) -> CompilerResult<()> {
        if self.config.enable_ai_metadata {
            let mut ai_metadata = self.ai_metadata.write().unwrap();
            ai_metadata.insert(node_id, metadata);
        }
        Ok(())
    }

    /// Get AI metadata
    pub fn get_ai_metadata(&self, node_id: &NodeId) -> Option<AIMetadata> {
        let ai_metadata = self.ai_metadata.read().unwrap();
        ai_metadata.get(node_id).cloned()
    }

    /// Export AI context for external tools
    pub fn export_ai_context(&self, location: Span) -> CompilerResult<AIReadableContext> {
        // Implementation will gather all relevant semantic information
        // for the given location and format it for AI consumption
        todo!("Implement AI context export")
    }

    /// Generate comprehensive semantic analysis
    pub fn analyze_program(&self, program: &Program) -> CompilerResult<SemanticInfo> {
        // Implementation will analyze the entire program and generate
        // comprehensive semantic information
        todo!("Implement program analysis")
    }
}

/// AI-readable context for external tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIReadableContext {
    /// Local scope information
    pub local_scope: HashMap<String, TypeInfo>,
    /// Available functions
    pub available_functions: Vec<FunctionSignature>,
    /// Imported modules
    pub imported_modules: Vec<ModuleInfo>,
    /// Current effects
    pub current_effects: Vec<Effect>,
    /// Business context
    pub business_context: Option<BusinessContext>,
    /// Performance constraints
    pub performance_constraints: Vec<String>,
}

/// Function signature for AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    /// Function name
    pub name: String,
    /// Parameters
    pub parameters: Vec<ParameterInfo>,
    /// Return type
    pub return_type: TypeInfo,
    /// Effects
    pub effects: Vec<Effect>,
    /// Documentation
    pub documentation: Option<String>,
}

/// Module information for AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    /// Module path
    pub path: String,
    /// Exported symbols
    pub exports: Vec<String>,
    /// Module purpose
    pub purpose: Option<String>,
}

/// Performance metadata for AI optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetadata {
    /// Performance characteristics by code region
    pub characteristics: HashMap<NodeId, PerformanceCharacteristic>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Resource usage patterns
    pub resource_patterns: Vec<ResourcePattern>,
    /// Parallelization opportunities
    pub parallelization_hints: Vec<ParallelizationHint>,
}

/// Performance characteristic for a code region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristic {
    /// Location in code
    pub location: Span,
    /// Characteristic type
    pub characteristic_type: PerformanceCharacteristicType,
    /// Description
    pub description: String,
    /// Severity/Impact
    pub impact: ImpactLevel,
    /// Suggested optimizations
    pub optimizations: Vec<String>,
}

/// Performance characteristic type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceCharacteristicType {
    /// CPU intensive operation
    CPUIntensive,
    /// Memory intensive operation
    MemoryIntensive,
    /// I/O bound operation
    IOBound,
    /// Network bound operation
    NetworkBound,
    /// Cache-friendly access pattern
    CacheFriendly,
    /// Cache-unfriendly access pattern
    CacheUnfriendly,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Location in code
    pub location: Span,
    /// Optimization type
    pub optimization_type: OptimizationType,
    /// Description
    pub description: String,
    /// Expected benefit
    pub expected_benefit: String,
    /// Implementation complexity
    pub complexity: OptimizationComplexity,
}

/// Optimization type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Loop optimization
    LoopOptimization,
    /// Memory layout optimization
    MemoryLayout,
    /// Algorithm optimization
    Algorithm,
    /// Data structure optimization
    DataStructure,
    /// Vectorization
    Vectorization,
    /// Parallelization
    Parallelization,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Location in code
    pub location: Span,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Description
    pub description: String,
    /// Severity
    pub severity: SeverityLevel,
    /// Mitigation strategies
    pub mitigations: Vec<String>,
}

/// Bottleneck type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU bottleneck
    CPU,
    /// Memory bottleneck
    Memory,
    /// I/O bottleneck
    IO,
    /// Network bottleneck
    Network,
    /// Synchronization bottleneck
    Synchronization,
}

/// Resource usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePattern {
    /// Pattern type
    pub pattern_type: ResourcePatternType,
    /// Description
    pub description: String,
    /// Affected locations
    pub locations: Vec<Span>,
    /// Resource efficiency
    pub efficiency: f64,
}

/// Resource pattern type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourcePatternType {
    /// Sequential access
    SequentialAccess,
    /// Random access
    RandomAccess,
    /// Burst allocation
    BurstAllocation,
    /// Gradual allocation
    GradualAllocation,
    /// Resource pooling
    ResourcePooling,
}

/// Parallelization hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationHint {
    /// Location in code
    pub location: Span,
    /// Parallelization type
    pub parallelization_type: ParallelizationType,
    /// Description
    pub description: String,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Expected speedup
    pub expected_speedup: Option<f64>,
}

/// Parallelization type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelizationType {
    /// Data parallelism
    DataParallel,
    /// Task parallelism
    TaskParallel,
    /// Pipeline parallelism
    PipelineParallel,
    /// SIMD vectorization
    SIMD,
}

/// Security metadata for AI security analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetadata {
    /// Security implications by code region
    pub implications: HashMap<NodeId, SecurityImplication>,
    /// Vulnerability patterns
    pub vulnerabilities: Vec<VulnerabilityPattern>,
    /// Security best practices
    pub best_practices: Vec<SecurityBestPractice>,
    /// Threat model information
    pub threat_model: Option<ThreatModel>,
}

/// Security implication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplication {
    /// Location in code
    pub location: Span,
    /// Security category
    pub category: SecurityCategory,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Description
    pub description: String,
    /// Mitigation strategies
    pub mitigations: Vec<String>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
}

/// Vulnerability pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityPattern {
    /// Pattern name
    pub name: String,
    /// Vulnerability type
    pub vulnerability_type: VulnerabilityType,
    /// Affected locations
    pub locations: Vec<Span>,
    /// Severity
    pub severity: SeverityLevel,
    /// Remediation steps
    pub remediation: Vec<String>,
}

/// Vulnerability type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityType {
    /// Buffer overflow
    BufferOverflow,
    /// SQL injection
    SQLInjection,
    /// Cross-site scripting
    XSS,
    /// Information disclosure
    InformationDisclosure,
    /// Privilege escalation
    PrivilegeEscalation,
    /// Denial of service
    DoS,
    /// Race condition
    RaceCondition,
}

/// Security best practice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityBestPractice {
    /// Practice name
    pub name: String,
    /// Description
    pub description: String,
    /// Applicable locations
    pub locations: Vec<Span>,
    /// Implementation guidance
    pub guidance: Vec<String>,
}

/// Threat model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatModel {
    /// Assets being protected
    pub assets: Vec<String>,
    /// Threat actors
    pub threat_actors: Vec<ThreatActor>,
    /// Attack vectors
    pub attack_vectors: Vec<AttackVector>,
    /// Security controls
    pub security_controls: Vec<SecurityControl>,
}

/// Threat actor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatActor {
    /// Actor name
    pub name: String,
    /// Capabilities
    pub capabilities: Vec<String>,
    /// Motivations
    pub motivations: Vec<String>,
}

/// Attack vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackVector {
    /// Vector name
    pub name: String,
    /// Description
    pub description: String,
    /// Likelihood
    pub likelihood: LikelihoodLevel,
    /// Impact
    pub impact: ImpactLevel,
}

/// Security control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityControl {
    /// Control name
    pub name: String,
    /// Control type
    pub control_type: SecurityControlType,
    /// Effectiveness
    pub effectiveness: f64,
    /// Implementation status
    pub implementation_status: ImplementationStatus,
}

/// Security control type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityControlType {
    /// Preventive control
    Preventive,
    /// Detective control
    Detective,
    /// Corrective control
    Corrective,
    /// Compensating control
    Compensating,
}

/// Implementation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationStatus {
    /// Not implemented
    NotImplemented,
    /// Partially implemented
    PartiallyImplemented,
    /// Fully implemented
    FullyImplemented,
    /// Under review
    UnderReview,
}

/// Compliance metadata for regulatory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetadata {
    /// Compliance requirements by code region
    pub requirements: HashMap<NodeId, ComplianceRequirement>,
    /// Regulatory frameworks
    pub frameworks: Vec<RegulatoryFramework>,
    /// Audit trails
    pub audit_trails: Vec<AuditTrail>,
    /// Compliance status
    pub compliance_status: ComplianceStatus,
}

/// Compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    /// Requirement ID
    pub id: String,
    /// Framework name
    pub framework: String,
    /// Description
    pub description: String,
    /// Affected locations
    pub locations: Vec<Span>,
    /// Implementation evidence
    pub evidence: Vec<String>,
    /// Status
    pub status: ComplianceStatus,
}

/// Regulatory framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryFramework {
    /// Framework name
    pub name: String,
    /// Version
    pub version: String,
    /// Applicable requirements
    pub requirements: Vec<String>,
    /// Jurisdiction
    pub jurisdiction: String,
}

/// Audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    /// Event ID
    pub event_id: String,
    /// Timestamp
    pub timestamp: String,
    /// Event type
    pub event_type: String,
    /// Description
    pub description: String,
    /// Actor
    pub actor: String,
    /// Affected resources
    pub resources: Vec<String>,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Compliant
    Compliant,
    /// Non-compliant
    NonCompliant,
    /// Partially compliant
    PartiallyCompliant,
    /// Under review
    UnderReview,
    /// Not applicable
    NotApplicable,
}

/// Cross-target consistency metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetadata {
    /// Semantic consistency across targets
    pub semantic_consistency: SemanticConsistency,
    /// Type preservation information
    pub type_preservation: TypePreservation,
    /// Effect consistency
    pub effect_consistency: EffectConsistency,
    /// Performance consistency
    pub performance_consistency: PerformanceConsistency,
}

/// Semantic consistency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConsistency {
    /// Business rules preserved across targets
    pub business_rules_preserved: bool,
    /// Validation predicates preserved
    pub validation_preserved: bool,
    /// Semantic meaning preserved
    pub meaning_preserved: bool,
    /// Inconsistencies found
    pub inconsistencies: Vec<SemanticInconsistency>,
}

/// Semantic inconsistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInconsistency {
    /// Inconsistency type
    pub inconsistency_type: InconsistencyType,
    /// Description
    pub description: String,
    /// Affected targets
    pub affected_targets: Vec<String>,
    /// Severity
    pub severity: SeverityLevel,
    /// Resolution suggestions
    pub resolutions: Vec<String>,
}

/// Inconsistency type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InconsistencyType {
    /// Type mismatch
    TypeMismatch,
    /// Business rule violation
    BusinessRuleViolation,
    /// Effect mismatch
    EffectMismatch,
    /// Performance characteristic mismatch
    PerformanceMismatch,
}

/// Type preservation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypePreservation {
    /// Semantic types preserved
    pub semantic_types_preserved: bool,
    /// Business rules preserved
    pub business_rules_preserved: bool,
    /// Validation predicates preserved
    pub validation_predicates_preserved: bool,
    /// Type mappings across targets
    pub type_mappings: HashMap<String, TargetTypeMapping>,
}

/// Target type mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetTypeMapping {
    /// Source type
    pub source_type: String,
    /// Target mappings
    pub target_mappings: HashMap<String, String>,
    /// Preservation level
    pub preservation_level: PreservationLevel,
}

/// Preservation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreservationLevel {
    /// Full preservation
    Full,
    /// Partial preservation
    Partial,
    /// Minimal preservation
    Minimal,
    /// No preservation
    None,
}

/// Effect consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectConsistency {
    /// Effects preserved across targets
    pub effects_preserved: bool,
    /// Capability requirements preserved
    pub capabilities_preserved: bool,
    /// Side effects tracked
    pub side_effects_tracked: bool,
    /// Effect violations
    pub violations: Vec<EffectViolation>,
}

/// Effect violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectViolation {
    /// Violation type
    pub violation_type: EffectViolationType,
    /// Description
    pub description: String,
    /// Location
    pub location: Span,
    /// Affected targets
    pub affected_targets: Vec<String>,
}

/// Effect violation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectViolationType {
    /// Missing capability
    MissingCapability,
    /// Unauthorized effect
    UnauthorizedEffect,
    /// Effect conflict
    EffectConflict,
    /// Side effect not tracked
    UntrackedSideEffect,
}

/// Performance consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConsistency {
    /// Performance characteristics preserved
    pub characteristics_preserved: bool,
    /// Complexity guarantees preserved
    pub complexity_preserved: bool,
    /// Resource usage patterns consistent
    pub resource_patterns_consistent: bool,
    /// Performance variations
    pub variations: Vec<PerformanceVariation>,
}

/// Performance variation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceVariation {
    /// Variation type
    pub variation_type: PerformanceVariationType,
    /// Description
    pub description: String,
    /// Affected targets
    pub affected_targets: Vec<String>,
    /// Impact assessment
    pub impact: ImpactLevel,
}

/// Performance variation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceVariationType {
    /// Time complexity change
    TimeComplexityChange,
    /// Space complexity change
    SpaceComplexityChange,
    /// Resource usage change
    ResourceUsageChange,
    /// Parallelization difference
    ParallelizationDifference,
}

/// AI comprehension aids
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensionAids {
    /// Code explanations
    pub explanations: HashMap<NodeId, CodeExplanation>,
    /// Concept mappings
    pub concept_mappings: Vec<ConceptMapping>,
    /// Learning resources
    pub learning_resources: Vec<LearningResource>,
    /// Related examples
    pub examples: Vec<CodeExample>,
}

/// Code explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExplanation {
    /// Location in code
    pub location: Span,
    /// Plain language explanation
    pub explanation: String,
    /// Key concepts
    pub concepts: Vec<String>,
    /// Why this approach was chosen
    pub rationale: Option<String>,
    /// Alternative approaches
    pub alternatives: Vec<String>,
}

/// Concept mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMapping {
    /// Source concept
    pub source_concept: String,
    /// Target concept
    pub target_concept: String,
    /// Relationship type
    pub relationship: ConceptRelationship,
    /// Explanation
    pub explanation: String,
}

/// Concept relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptRelationship {
    /// Is a type of
    IsA,
    /// Part of
    PartOf,
    /// Similar to
    SimilarTo,
    /// Opposite of
    OppositeOf,
    /// Implements
    Implements,
    /// Uses
    Uses,
}

/// Learning resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResource {
    /// Resource title
    pub title: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// URL or reference
    pub reference: String,
    /// Description
    pub description: String,
    /// Relevant concepts
    pub concepts: Vec<String>,
}

/// Resource type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    /// Documentation
    Documentation,
    /// Tutorial
    Tutorial,
    /// Example code
    ExampleCode,
    /// Research paper
    ResearchPaper,
    /// Blog post
    BlogPost,
    /// Video
    Video,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Example title
    pub title: String,
    /// Code snippet
    pub code: String,
    /// Language
    pub language: String,
    /// Explanation
    pub explanation: String,
    /// Related concepts
    pub concepts: Vec<String>,
}

/// Impact level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Critical impact
    Critical,
    /// High impact
    High,
    /// Medium impact
    Medium,
    /// Low impact
    Low,
    /// Negligible impact
    Negligible,
}

/// Severity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    /// Critical severity
    Critical,
    /// High severity
    High,
    /// Medium severity
    Medium,
    /// Low severity
    Low,
    /// Informational
    Informational,
}

/// Likelihood level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LikelihoodLevel {
    /// Very high likelihood
    VeryHigh,
    /// High likelihood
    High,
    /// Medium likelihood
    Medium,
    /// Low likelihood
    Low,
    /// Very low likelihood
    VeryLow,
}

/// Optimization complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationComplexity {
    /// Simple optimization
    Simple,
    /// Moderate complexity
    Moderate,
    /// Complex optimization
    Complex,
    /// Very complex optimization
    VeryComplex,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_call_graph: true,
            enable_data_flow: true,
            enable_effect_tracking: true,
        }
    }
}

impl Default for AIMetadata {
    fn default() -> Self {
        Self {
            module_context: None,
            function_contexts: HashMap::new(),
            type_contexts: HashMap::new(),
            relationships: SemanticRelationships {
                type_relationships: HashMap::new(),
                function_relationships: HashMap::new(),
                module_relationships: HashMap::new(),
            },
            business_context: None,
            performance_metadata: PerformanceMetadata {
                characteristics: HashMap::new(),
                optimization_opportunities: Vec::new(),
                bottlenecks: Vec::new(),
                resource_patterns: Vec::new(),
                parallelization_hints: Vec::new(),
            },
            security_metadata: SecurityMetadata {
                implications: HashMap::new(),
                vulnerabilities: Vec::new(),
                best_practices: Vec::new(),
                threat_model: None,
            },
            compliance_metadata: ComplianceMetadata {
                requirements: HashMap::new(),
                frameworks: Vec::new(),
                audit_trails: Vec::new(),
                compliance_status: ComplianceStatus::NotApplicable,
            },
            consistency_metadata: ConsistencyMetadata {
                semantic_consistency: SemanticConsistency {
                    business_rules_preserved: true,
                    validation_preserved: true,
                    meaning_preserved: true,
                    inconsistencies: Vec::new(),
                },
                type_preservation: TypePreservation {
                    semantic_types_preserved: true,
                    business_rules_preserved: true,
                    validation_predicates_preserved: true,
                    type_mappings: HashMap::new(),
                },
                effect_consistency: EffectConsistency {
                    effects_preserved: true,
                    capabilities_preserved: true,
                    side_effects_tracked: true,
                    violations: Vec::new(),
                },
                performance_consistency: PerformanceConsistency {
                    characteristics_preserved: true,
                    complexity_preserved: true,
                    resource_patterns_consistent: true,
                    variations: Vec::new(),
                },
            },
            comprehension_aids: ComprehensionAids {
                explanations: HashMap::new(),
                concept_mappings: Vec::new(),
                learning_resources: Vec::new(),
                examples: Vec::new(),
            },
        }
    }
} 