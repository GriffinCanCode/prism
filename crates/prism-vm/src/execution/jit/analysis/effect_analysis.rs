//! Effect Analysis for Safe Optimizations
//!
//! This module provides comprehensive effect system analysis to ensure optimizations maintain
//! program correctness and don't violate safety guarantees. It integrates with the prism-effects
//! system to provide rich effect information for optimization decisions.
//!
//! ## Research Foundation
//!
//! Based on research from:
//! - Effect systems in modern functional languages (Koka, Eff, Frank)
//! - Capability-based security models for safe optimization
//! - Memory safety analysis through effect tracking
//! - Region-based memory management and effect inference
//! - LLVM's MemorySSA and alias analysis for safe transformations
//!
//! ## Key Features
//!
//! - **Effect Inference**: Automatically infer effects from bytecode instructions
//! - **Effect Flow Analysis**: Track how effects propagate through control flow
//! - **Safety Validation**: Ensure optimizations don't violate effect constraints
//! - **Memory Safety**: Detect potential memory safety violations
//! - **Capability Integration**: Leverage capability system for security analysis
//! - **Optimization Opportunities**: Identify safe optimization opportunities

use crate::{VMResult, PrismVMError, bytecode::{FunctionDefinition, Instruction, PrismOpcode}};
use super::{AnalysisConfig, control_flow::{ControlFlowGraph, BasicBlock}};
use prism_effects::{
    effects::{Effect, EffectDefinition, EffectCategory, EffectRegistry, EffectSystem},
    security::{SecuritySystem, Capability, SecurityLevel},
    execution::{EffectHandler, EffectResult},
};
use prism_common::span::Span;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::sync::Arc;

/// Effect analyzer with prism-effects integration
#[derive(Debug)]
pub struct EffectAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    
    /// Effect system for effect definitions and inference
    effect_system: EffectSystem,
    
    /// Security system for capability analysis
    security_system: SecuritySystem,
    
    /// Effect inference engine
    inference_engine: EffectInferenceEngine,
    
    /// Effect flow analyzer
    flow_analyzer: EffectFlowAnalyzer,
    
    /// Safety validator
    safety_validator: SafetyValidator,
    
    /// Optimization opportunity detector
    optimization_detector: OptimizationOpportunityDetector,
}

/// Comprehensive effect analysis results
#[derive(Debug, Clone)]
pub struct EffectAnalysis {
    /// Function being analyzed
    pub function_id: u32,
    
    /// Inferred effects for each instruction
    pub instruction_effects: HashMap<u32, Vec<InferredEffect>>,
    
    /// Effect flow through the function
    pub effect_flow: EffectFlow,
    
    /// Effect constraints for optimization safety
    pub effect_constraints: Vec<EffectConstraint>,
    
    /// Safety analysis results
    pub safety_analysis: SafetyAnalysis,
    
    /// Memory effect analysis
    pub memory_analysis: MemoryEffectAnalysis,
    
    /// Capability requirements and flow
    pub capability_analysis: CapabilityAnalysis,
    
    /// Optimization opportunities based on effects
    pub optimization_opportunities: Vec<EffectOptimizationOpportunity>,
    
    /// Effect summary for interprocedural analysis
    pub effect_summary: EffectSummary,
}

/// Effect flow analysis tracking effect propagation
#[derive(Debug, Clone, Default)]
pub struct EffectFlow {
    /// Effects present at each program point
    pub effects_at_point: HashMap<u32, EffectSet>,
    
    /// Effects that enter each basic block
    pub block_entry_effects: HashMap<u32, EffectSet>,
    
    /// Effects that exit each basic block
    pub block_exit_effects: HashMap<u32, EffectSet>,
    
    /// Side effect locations in the function
    pub side_effect_locations: Vec<u32>,
    
    /// Pure regions (no side effects)
    pub pure_regions: Vec<PureRegion>,
    
    /// Effect dependencies between instructions
    pub effect_dependencies: HashMap<u32, Vec<u32>>,
    
    /// Memory aliasing information
    pub memory_aliases: HashMap<u32, AliasSet>,
}

/// Set of effects with efficient operations
#[derive(Debug, Clone, Default)]
pub struct EffectSet {
    /// Effects in this set
    pub effects: HashSet<InferredEffect>,
    
    /// Summary categories for quick checks
    pub categories: HashSet<EffectCategory>,
    
    /// Whether this set contains impure effects
    pub has_side_effects: bool,
    
    /// Whether this set contains memory effects
    pub has_memory_effects: bool,
    
    /// Whether this set contains I/O effects
    pub has_io_effects: bool,
}

/// Pure region with no side effects
#[derive(Debug, Clone)]
pub struct PureRegion {
    /// Start instruction offset
    pub start: u32,
    
    /// End instruction offset
    pub end: u32,
    
    /// Basic blocks in this region
    pub blocks: HashSet<u32>,
    
    /// Purity level
    pub purity_level: PurityLevel,
}

/// Levels of purity for optimization decisions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PurityLevel {
    /// Completely pure (no effects)
    Pure,
    
    /// Locally pure (only local effects)
    LocallyPure,
    
    /// Observationally pure (effects don't escape)
    ObservationallyPure,
    
    /// Has side effects
    Impure,
}

/// Memory alias set for tracking memory dependencies
#[derive(Debug, Clone, Default)]
pub struct AliasSet {
    /// Memory locations that may alias
    pub may_alias: HashSet<MemoryLocation>,
    
    /// Memory locations that must alias
    pub must_alias: HashSet<MemoryLocation>,
    
    /// Memory locations that cannot alias
    pub no_alias: HashSet<MemoryLocation>,
}

/// Memory location abstraction
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    /// Local variable slot
    Local(u8),
    
    /// Global variable
    Global(u32),
    
    /// Object field
    Field { object: Box<MemoryLocation>, field: String },
    
    /// Array element
    Index { array: Box<MemoryLocation>, index: Option<i64> },
    
    /// Abstract heap location
    Heap(u32),
    
    /// Unknown location
    Unknown,
}

/// Inferred effect with confidence and source information
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InferredEffect {
    /// The effect definition
    pub effect: Effect,
    
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    
    /// Source of inference
    pub source: EffectInferenceSource,
    
    /// Memory locations affected
    pub memory_locations: Vec<MemoryLocation>,
    
    /// Capability requirements
    pub required_capabilities: Vec<Capability>,
    
    /// Effect parameters
    pub parameters: HashMap<String, EffectParameter>,
}

/// Source of effect inference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectInferenceSource {
    /// Explicitly declared in bytecode
    Explicit,
    
    /// Inferred from instruction semantics
    InstructionSemantics,
    
    /// Inferred from function calls
    FunctionCall,
    
    /// Inferred from memory operations
    MemoryOperation,
    
    /// Inferred from capability usage
    CapabilityUsage,
    
    /// Propagated from called functions
    Interprocedural,
}

/// Effect parameter for parameterized effects
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectParameter {
    /// Memory location parameter
    MemoryLocation(MemoryLocation),
    
    /// Integer parameter
    Integer(i64),
    
    /// String parameter
    String(String),
    
    /// Type parameter
    Type(u32),
}

/// Effect constraints for safe optimization
#[derive(Debug, Clone)]
pub struct EffectConstraint {
    /// Instruction location
    pub location: u32,
    
    /// Type of constraint
    pub constraint_type: EffectConstraintType,
    
    /// Required effects that must be preserved
    pub required_effects: EffectSet,
    
    /// Forbidden effects that must not be introduced
    pub forbidden_effects: EffectSet,
    
    /// Memory ordering constraints
    pub memory_ordering: MemoryOrdering,
    
    /// Capability constraints
    pub capability_constraints: Vec<CapabilityConstraint>,
}

/// Types of effect constraints
#[derive(Debug, Clone)]
pub enum EffectConstraintType {
    /// Effects must be preserved exactly
    PreserveEffects,
    
    /// Effects must not be reordered
    NoReordering,
    
    /// Memory operations must maintain ordering
    MemoryOrdering,
    
    /// Capability checks must be preserved
    CapabilityCheck,
    
    /// Exception handling must be preserved
    ExceptionHandling,
    
    /// Observable behavior must be maintained
    ObservableBehavior,
}

/// Memory ordering requirements
#[derive(Debug, Clone)]
pub enum MemoryOrdering {
    /// No specific ordering required
    None,
    
    /// Acquire ordering (loads)
    Acquire,
    
    /// Release ordering (stores)
    Release,
    
    /// Sequential consistency
    SeqCst,
    
    /// Custom ordering constraints
    Custom(Vec<OrderingConstraint>),
}

/// Specific ordering constraint between operations
#[derive(Debug, Clone)]
pub struct OrderingConstraint {
    /// First operation
    pub before: u32,
    
    /// Second operation
    pub after: u32,
    
    /// Type of ordering
    pub ordering_type: OrderingType,
}

/// Type of memory ordering constraint
#[derive(Debug, Clone)]
pub enum OrderingType {
    /// Operations must execute in program order
    ProgramOrder,
    
    /// Memory operations must be ordered
    MemoryOrder,
    
    /// Control dependencies must be preserved
    ControlDependency,
    
    /// Data dependencies must be preserved
    DataDependency,
}

/// Capability constraint for security
#[derive(Debug, Clone)]
pub struct CapabilityConstraint {
    /// Required capability
    pub capability: Capability,
    
    /// Constraint type
    pub constraint_type: CapabilityConstraintType,
    
    /// Location where constraint applies
    pub location: u32,
}

/// Types of capability constraints
#[derive(Debug, Clone)]
pub enum CapabilityConstraintType {
    /// Capability must be available
    Required,
    
    /// Capability must not be available
    Forbidden,
    
    /// Capability must be delegated
    Delegated,
    
    /// Capability must be revoked
    Revoked,
}

/// Safety analysis results
#[derive(Debug, Clone, Default)]
pub struct SafetyAnalysis {
    /// Memory safety violations
    pub memory_safety_violations: Vec<MemorySafetyViolation>,
    
    /// Potential data races
    pub data_race_potential: Vec<DataRaceLocation>,
    
    /// Unsafe optimizations that should be avoided
    pub unsafe_optimizations: Vec<UnsafeOptimization>,
    
    /// Security vulnerabilities
    pub security_vulnerabilities: Vec<SecurityVulnerability>,
    
    /// Effect safety violations
    pub effect_violations: Vec<EffectSafetyViolation>,
}

/// Memory safety violation
#[derive(Debug, Clone)]
pub struct MemorySafetyViolation {
    /// Location of violation
    pub location: u32,
    
    /// Type of violation
    pub violation_type: MemorySafetyViolationType,
    
    /// Severity level
    pub severity: SafetyViolationSeverity,
    
    /// Description of the violation
    pub description: String,
    
    /// Memory locations involved
    pub memory_locations: Vec<MemoryLocation>,
}

/// Types of memory safety violations
#[derive(Debug, Clone)]
pub enum MemorySafetyViolationType {
    /// Use after free
    UseAfterFree,
    
    /// Double free
    DoubleFree,
    
    /// Buffer overflow/underflow
    BufferOverflow,
    
    /// Null pointer dereference
    NullPointerDereference,
    
    /// Uninitialized memory access
    UninitializedAccess,
    
    /// Memory leak
    MemoryLeak,
    
    /// Invalid pointer arithmetic
    InvalidPointerArithmetic,
}

/// Data race location
#[derive(Debug, Clone)]
pub struct DataRaceLocation {
    /// First access location
    pub location1: u32,
    
    /// Second access location
    pub location2: u32,
    
    /// Memory location accessed
    pub memory_location: MemoryLocation,
    
    /// Types of accesses
    pub access_types: (MemoryAccessType, MemoryAccessType),
    
    /// Confidence in data race detection
    pub confidence: f64,
}

/// Type of memory access
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryAccessType {
    /// Read access
    Read,
    
    /// Write access
    Write,
    
    /// Read-modify-write access
    ReadWrite,
}

/// Unsafe optimization to avoid
#[derive(Debug, Clone)]
pub struct UnsafeOptimization {
    /// Location where optimization would be unsafe
    pub location: u32,
    
    /// Type of unsafe optimization
    pub optimization_type: UnsafeOptimizationType,
    
    /// Reason why it's unsafe
    pub reason: String,
    
    /// Effects that would be violated
    pub violated_effects: Vec<InferredEffect>,
}

/// Types of unsafe optimizations
#[derive(Debug, Clone)]
pub enum UnsafeOptimizationType {
    /// Dead code elimination of effectful code
    DeadCodeElimination,
    
    /// Reordering of dependent operations
    InstructionReordering,
    
    /// Loop optimization that changes semantics
    LoopOptimization,
    
    /// Inlining that violates effects
    Inlining,
    
    /// Constant propagation through effects
    ConstantPropagation,
    
    /// Common subexpression elimination of effectful code
    CommonSubexpressionElimination,
}

/// Security vulnerability
#[derive(Debug, Clone)]
pub struct SecurityVulnerability {
    /// Location of vulnerability
    pub location: u32,
    
    /// Type of vulnerability
    pub vulnerability_type: SecurityVulnerabilityType,
    
    /// Severity level
    pub severity: SafetyViolationSeverity,
    
    /// Description
    pub description: String,
    
    /// Required capabilities to exploit
    pub required_capabilities: Vec<Capability>,
}

/// Types of security vulnerabilities
#[derive(Debug, Clone)]
pub enum SecurityVulnerabilityType {
    /// Capability bypass
    CapabilityBypass,
    
    /// Information leak
    InformationLeak,
    
    /// Privilege escalation
    PrivilegeEscalation,
    
    /// Injection attack
    InjectionAttack,
    
    /// Side channel
    SideChannel,
}

/// Effect safety violation
#[derive(Debug, Clone)]
pub struct EffectSafetyViolation {
    /// Location of violation
    pub location: u32,
    
    /// Expected effects
    pub expected_effects: EffectSet,
    
    /// Actual effects
    pub actual_effects: EffectSet,
    
    /// Violation description
    pub description: String,
}

/// Severity levels for safety violations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SafetyViolationSeverity {
    /// Informational
    Info,
    
    /// Low severity
    Low,
    
    /// Medium severity
    Medium,
    
    /// High severity
    High,
    
    /// Critical severity
    Critical,
}

/// Memory effect analysis
#[derive(Debug, Clone, Default)]
pub struct MemoryEffectAnalysis {
    /// Memory regions accessed
    pub memory_regions: HashMap<u32, MemoryRegion>,
    
    /// Memory dependencies between instructions
    pub memory_dependencies: Vec<MemoryDependency>,
    
    /// Alias analysis results
    pub alias_analysis: HashMap<u32, AliasSet>,
    
    /// Memory ordering constraints
    pub ordering_constraints: Vec<OrderingConstraint>,
    
    /// Escape analysis results
    pub escape_analysis: EscapeAnalysis,
}

/// Memory region information
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Region identifier
    pub id: u32,
    
    /// Memory locations in this region
    pub locations: Vec<MemoryLocation>,
    
    /// Access pattern
    pub access_pattern: AccessPattern,
    
    /// Whether region escapes function
    pub escapes: bool,
    
    /// Thread safety level
    pub thread_safety: ThreadSafetyLevel,
}

/// Memory access pattern
#[derive(Debug, Clone)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,
    
    /// Random access
    Random,
    
    /// Write-once, read-many
    WriteOnceReadMany,
    
    /// Read-only
    ReadOnly,
    
    /// Write-only
    WriteOnly,
}

/// Thread safety level
#[derive(Debug, Clone)]
pub enum ThreadSafetyLevel {
    /// Thread-safe
    ThreadSafe,
    
    /// Not thread-safe
    NotThreadSafe,
    
    /// Requires synchronization
    RequiresSynchronization,
    
    /// Unknown
    Unknown,
}

/// Memory dependency between instructions
#[derive(Debug, Clone)]
pub struct MemoryDependency {
    /// Source instruction
    pub source: u32,
    
    /// Target instruction
    pub target: u32,
    
    /// Type of dependency
    pub dependency_type: MemoryDependencyType,
    
    /// Memory location
    pub memory_location: MemoryLocation,
}

/// Types of memory dependencies
#[derive(Debug, Clone)]
pub enum MemoryDependencyType {
    /// Read-after-write dependency
    ReadAfterWrite,
    
    /// Write-after-read dependency
    WriteAfterRead,
    
    /// Write-after-write dependency
    WriteAfterWrite,
    
    /// Address dependency
    AddressDependency,
    
    /// Control dependency
    ControlDependency,
}

/// Escape analysis results
#[derive(Debug, Clone, Default)]
pub struct EscapeAnalysis {
    /// Objects that escape the function
    pub escaping_objects: HashSet<u32>,
    
    /// Objects that don't escape
    pub non_escaping_objects: HashSet<u32>,
    
    /// Escape paths for each object
    pub escape_paths: HashMap<u32, Vec<EscapePath>>,
}

/// Path through which an object escapes
#[derive(Debug, Clone)]
pub struct EscapePath {
    /// Instructions in the escape path
    pub path: Vec<u32>,
    
    /// Type of escape
    pub escape_type: EscapeType,
}

/// Type of object escape
#[derive(Debug, Clone)]
pub enum EscapeType {
    /// Return from function
    Return,
    
    /// Assignment to global
    Global,
    
    /// Assignment to heap
    Heap,
    
    /// Passed to function call
    FunctionCall,
    
    /// Stored in escaping object
    StoredInEscaping,
}

/// Capability analysis results
#[derive(Debug, Clone, Default)]
pub struct CapabilityAnalysis {
    /// Required capabilities at each instruction
    pub required_capabilities: HashMap<u32, Vec<Capability>>,
    
    /// Capability flow through the function
    pub capability_flow: HashMap<u32, CapabilitySet>,
    
    /// Security boundaries
    pub security_boundaries: Vec<SecurityBoundary>,
    
    /// Capability violations
    pub violations: Vec<CapabilityViolation>,
}

/// Set of capabilities
#[derive(Debug, Clone, Default)]
pub struct CapabilitySet {
    /// Available capabilities
    pub available: HashSet<Capability>,
    
    /// Required capabilities
    pub required: HashSet<Capability>,
    
    /// Delegated capabilities
    pub delegated: HashSet<Capability>,
}

/// Security boundary in the code
#[derive(Debug, Clone)]
pub struct SecurityBoundary {
    /// Start of boundary
    pub start: u32,
    
    /// End of boundary
    pub end: u32,
    
    /// Type of boundary
    pub boundary_type: SecurityBoundaryType,
    
    /// Required capabilities to cross
    pub required_capabilities: Vec<Capability>,
}

/// Types of security boundaries
#[derive(Debug, Clone)]
pub enum SecurityBoundaryType {
    /// Trust boundary
    Trust,
    
    /// Privilege boundary
    Privilege,
    
    /// Isolation boundary
    Isolation,
    
    /// Effect boundary
    Effect,
}

/// Capability violation
#[derive(Debug, Clone)]
pub struct CapabilityViolation {
    /// Location of violation
    pub location: u32,
    
    /// Required capability
    pub required_capability: Capability,
    
    /// Available capabilities
    pub available_capabilities: Vec<Capability>,
    
    /// Violation description
    pub description: String,
}

/// Optimization opportunity based on effects
#[derive(Debug, Clone)]
pub struct EffectOptimizationOpportunity {
    /// Location of opportunity
    pub location: u32,
    
    /// Type of optimization
    pub optimization_type: EffectOptimizationType,
    
    /// Expected benefit
    pub expected_benefit: OptimizationBenefit,
    
    /// Safety constraints
    pub safety_constraints: Vec<EffectConstraint>,
    
    /// Required conditions
    pub required_conditions: Vec<OptimizationCondition>,
}

/// Types of effect-based optimizations
#[derive(Debug, Clone)]
pub enum EffectOptimizationType {
    /// Pure function optimization
    PureFunctionOptimization,
    
    /// Effect-preserving reordering
    EffectPreservingReordering,
    
    /// Memory optimization based on escape analysis
    MemoryOptimization,
    
    /// Capability-based optimization
    CapabilityOptimization,
    
    /// Effect elimination
    EffectElimination,
    
    /// Effect batching
    EffectBatching,
}

/// Expected optimization benefit
#[derive(Debug, Clone)]
pub struct OptimizationBenefit {
    /// Performance improvement estimate
    pub performance_gain: f64,
    
    /// Memory usage improvement
    pub memory_savings: i64,
    
    /// Code size impact
    pub code_size_impact: i32,
    
    /// Confidence in benefit estimate
    pub confidence: f64,
}

/// Condition required for optimization
#[derive(Debug, Clone)]
pub enum OptimizationCondition {
    /// No side effects in region
    NoSideEffects(u32, u32),
    
    /// Memory locations don't alias
    NoAlias(MemoryLocation, MemoryLocation),
    
    /// Capability available
    CapabilityAvailable(Capability),
    
    /// Effect constraint satisfied
    EffectConstraint(EffectConstraint),
    
    /// Control flow condition
    ControlFlow(ControlFlowCondition),
}

/// Control flow condition
#[derive(Debug, Clone)]
pub enum ControlFlowCondition {
    /// Block is always executed
    AlwaysExecuted(u32),
    
    /// Block is never executed
    NeverExecuted(u32),
    
    /// Blocks are mutually exclusive
    MutuallyExclusive(u32, u32),
    
    /// Loop bound is known
    LoopBound(u32, u64),
}

/// Effect summary for interprocedural analysis
#[derive(Debug, Clone, Default)]
pub struct EffectSummary {
    /// Effects this function may have
    pub may_effects: EffectSet,
    
    /// Effects this function must have
    pub must_effects: EffectSet,
    
    /// Memory locations this function may access
    pub may_access: Vec<MemoryLocation>,
    
    /// Memory locations this function must access
    pub must_access: Vec<MemoryLocation>,
    
    /// Required capabilities
    pub required_capabilities: Vec<Capability>,
    
    /// Capability effects
    pub capability_effects: Vec<CapabilityEffect>,
    
    /// Exception effects
    pub exception_effects: Vec<ExceptionEffect>,
}

/// Capability effect
#[derive(Debug, Clone)]
pub struct CapabilityEffect {
    /// Capability involved
    pub capability: Capability,
    
    /// Type of effect
    pub effect_type: CapabilityEffectType,
    
    /// Conditions under which effect occurs
    pub conditions: Vec<String>,
}

/// Types of capability effects
#[derive(Debug, Clone)]
pub enum CapabilityEffectType {
    /// Requires capability
    Requires,
    
    /// Delegates capability
    Delegates,
    
    /// Revokes capability
    Revokes,
    
    /// Checks capability
    Checks,
}

/// Exception effect
#[derive(Debug, Clone)]
pub struct ExceptionEffect {
    /// Exception type
    pub exception_type: Option<u32>,
    
    /// Conditions under which exception is thrown
    pub conditions: Vec<String>,
    
    /// Instructions that may throw
    pub throwing_instructions: Vec<u32>,
}

/// Effect inference engine
#[derive(Debug)]
pub struct EffectInferenceEngine {
    /// Effect registry for lookups
    effect_registry: Arc<EffectRegistry>,
    
    /// Instruction effect mappings
    instruction_effects: HashMap<PrismOpcode, Vec<EffectTemplate>>,
    
    /// Function effect cache
    function_effect_cache: HashMap<u32, EffectSummary>,
}

/// Template for inferring effects from instructions
#[derive(Debug, Clone)]
pub struct EffectTemplate {
    /// Effect definition
    pub effect: EffectDefinition,
    
    /// Conditions for this effect to apply
    pub conditions: Vec<EffectCondition>,
    
    /// Parameter mappings
    pub parameter_mappings: HashMap<String, ParameterMapping>,
}

/// Condition for effect template application
#[derive(Debug, Clone)]
pub enum EffectCondition {
    /// Instruction has specific operand
    HasOperand(String),
    
    /// Operand has specific value
    OperandValue(String, EffectParameter),
    
    /// Context condition
    Context(String),
    
    /// Capability condition
    Capability(Capability),
}

/// Parameter mapping for effect templates
#[derive(Debug, Clone)]
pub enum ParameterMapping {
    /// Map from instruction operand
    Operand(String),
    
    /// Constant value
    Constant(EffectParameter),
    
    /// Derived from context
    Context(String),
}

/// Effect flow analyzer
#[derive(Debug)]
pub struct EffectFlowAnalyzer {
    /// Control flow graph
    cfg: Option<Arc<ControlFlowGraph>>,
    
    /// Dataflow analysis state
    dataflow_state: DataflowState,
}

/// Dataflow analysis state
#[derive(Debug, Default)]
pub struct DataflowState {
    /// Effects flowing into each block
    pub block_in: HashMap<u32, EffectSet>,
    
    /// Effects flowing out of each block
    pub block_out: HashMap<u32, EffectSet>,
    
    /// Effects generated by each block
    pub block_gen: HashMap<u32, EffectSet>,
    
    /// Effects killed by each block
    pub block_kill: HashMap<u32, EffectSet>,
}

/// Safety validator
#[derive(Debug)]
pub struct SafetyValidator {
    /// Security system for capability checks
    security_system: Arc<SecuritySystem>,
    
    /// Known safe patterns
    safe_patterns: Vec<SafetyPattern>,
    
    /// Known unsafe patterns
    unsafe_patterns: Vec<UnsafetyPattern>,
}

/// Pattern that is known to be safe
#[derive(Debug, Clone)]
pub struct SafetyPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern description
    pub description: String,
    
    /// Conditions for safety
    pub conditions: Vec<SafetyCondition>,
}

/// Pattern that is known to be unsafe
#[derive(Debug, Clone)]
pub struct UnsafetyPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern description
    pub description: String,
    
    /// Conditions that make it unsafe
    pub conditions: Vec<UnsafetyCondition>,
    
    /// Mitigation strategies
    pub mitigations: Vec<String>,
}

/// Condition for safety
#[derive(Debug, Clone)]
pub enum SafetyCondition {
    /// No effects of specific type
    NoEffects(EffectCategory),
    
    /// Memory regions don't alias
    NoAlias(MemoryLocation, MemoryLocation),
    
    /// Capability is available
    HasCapability(Capability),
    
    /// Control flow constraint
    ControlFlow(String),
}

/// Condition that makes something unsafe
#[derive(Debug, Clone)]
pub enum UnsafetyCondition {
    /// Conflicting effects
    ConflictingEffects(EffectSet, EffectSet),
    
    /// Memory aliasing
    MemoryAliasing(MemoryLocation, MemoryLocation),
    
    /// Missing capability
    MissingCapability(Capability),
    
    /// Race condition potential
    RaceCondition(u32, u32),
}

/// Optimization opportunity detector
#[derive(Debug)]
pub struct OptimizationOpportunityDetector {
    /// Optimization patterns
    patterns: Vec<OptimizationPattern>,
    
    /// Cost model for benefit estimation
    cost_model: CostModel,
}

/// Pattern for optimization opportunities
#[derive(Debug, Clone)]
pub struct OptimizationPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern description
    pub description: String,
    
    /// Conditions for applying optimization
    pub conditions: Vec<OptimizationCondition>,
    
    /// Expected benefit
    pub benefit: OptimizationBenefit,
    
    /// Safety requirements
    pub safety_requirements: Vec<SafetyCondition>,
}

/// Cost model for optimization benefit estimation
#[derive(Debug)]
pub struct CostModel {
    /// Instruction costs
    pub instruction_costs: HashMap<PrismOpcode, u32>,
    
    /// Effect costs
    pub effect_costs: HashMap<EffectCategory, u32>,
    
    /// Memory access costs
    pub memory_costs: HashMap<MemoryAccessType, u32>,
}

impl EffectAnalyzer {
    /// Create a new effect analyzer
    pub fn new(config: &AnalysisConfig) -> VMResult<Self> {
        let effect_system = EffectSystem::new();
        let security_system = SecuritySystem::new();
        
        Ok(Self {
            config: config.clone(),
            effect_system,
            security_system,
            inference_engine: EffectInferenceEngine::new()?,
            flow_analyzer: EffectFlowAnalyzer::new(),
            safety_validator: SafetyValidator::new()?,
            optimization_detector: OptimizationOpportunityDetector::new()?,
        })
    }

    /// Perform comprehensive effect analysis on a function
    pub fn analyze(&mut self, function: &FunctionDefinition) -> VMResult<EffectAnalysis> {
        // Step 1: Infer effects for each instruction
        let instruction_effects = self.infer_instruction_effects(function)?;
        
        // Step 2: Analyze effect flow through the function
        let effect_flow = self.analyze_effect_flow(function, &instruction_effects)?;
        
        // Step 3: Generate effect constraints
        let effect_constraints = self.generate_effect_constraints(function, &effect_flow)?;
        
        // Step 4: Perform safety analysis
        let safety_analysis = self.perform_safety_analysis(function, &effect_flow)?;
        
        // Step 5: Analyze memory effects
        let memory_analysis = self.analyze_memory_effects(function, &instruction_effects)?;
        
        // Step 6: Analyze capabilities
        let capability_analysis = self.analyze_capabilities(function, &instruction_effects)?;
        
        // Step 7: Detect optimization opportunities
        let optimization_opportunities = self.detect_optimization_opportunities(
            function, &effect_flow, &safety_analysis
        )?;
        
        // Step 8: Generate effect summary
        let effect_summary = self.generate_effect_summary(&effect_flow, &memory_analysis)?;

        Ok(EffectAnalysis {
            function_id: function.id,
            instruction_effects,
            effect_flow,
            effect_constraints,
            safety_analysis,
            memory_analysis,
            capability_analysis,
            optimization_opportunities,
            effect_summary,
        })
    }

    /// Infer effects for each instruction in the function
    fn infer_instruction_effects(&mut self, function: &FunctionDefinition) -> VMResult<HashMap<u32, Vec<InferredEffect>>> {
        let mut instruction_effects = HashMap::new();
        
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let effects = self.inference_engine.infer_effects(instruction, offset as u32)?;
            if !effects.is_empty() {
                instruction_effects.insert(offset as u32, effects);
            }
        }
        
        Ok(instruction_effects)
    }

    /// Analyze how effects flow through the function
    fn analyze_effect_flow(
        &mut self, 
        function: &FunctionDefinition, 
        instruction_effects: &HashMap<u32, Vec<InferredEffect>>
    ) -> VMResult<EffectFlow> {
        self.flow_analyzer.analyze(function, instruction_effects)
    }

    /// Generate effect constraints for optimization safety
    fn generate_effect_constraints(
        &self, 
        function: &FunctionDefinition, 
        effect_flow: &EffectFlow
    ) -> VMResult<Vec<EffectConstraint>> {
        let mut constraints = Vec::new();
        
        // Generate constraints based on effect flow
        for (location, effects) in &effect_flow.effects_at_point {
            if effects.has_side_effects {
                constraints.push(EffectConstraint {
                    location: *location,
                    constraint_type: EffectConstraintType::PreserveEffects,
                    required_effects: effects.clone(),
                    forbidden_effects: EffectSet::default(),
                    memory_ordering: MemoryOrdering::None,
                    capability_constraints: Vec::new(),
                });
            }
        }
        
        // Add memory ordering constraints
        for dependency in &effect_flow.effect_dependencies {
            let location = *dependency.0;
            for &dependent_location in dependency.1 {
                constraints.push(EffectConstraint {
                    location,
                    constraint_type: EffectConstraintType::NoReordering,
                    required_effects: EffectSet::default(),
                    forbidden_effects: EffectSet::default(),
                    memory_ordering: MemoryOrdering::Custom(vec![
                        OrderingConstraint {
                            before: location,
                            after: dependent_location,
                            ordering_type: OrderingType::ProgramOrder,
                        }
                    ]),
                    capability_constraints: Vec::new(),
                });
            }
        }
        
        Ok(constraints)
    }

    /// Perform safety analysis
    fn perform_safety_analysis(
        &mut self, 
        function: &FunctionDefinition, 
        effect_flow: &EffectFlow
    ) -> VMResult<SafetyAnalysis> {
        self.safety_validator.analyze(function, effect_flow)
    }

    /// Analyze memory effects
    fn analyze_memory_effects(
        &self, 
        function: &FunctionDefinition, 
        instruction_effects: &HashMap<u32, Vec<InferredEffect>>
    ) -> VMResult<MemoryEffectAnalysis> {
        let mut memory_analysis = MemoryEffectAnalysis::default();
        
        // Analyze memory regions and dependencies
        for (location, effects) in instruction_effects {
            for effect in effects {
                if effect.effect.definition == "Memory.Read" || effect.effect.definition == "Memory.Write" {
                    // Analyze memory locations
                    for memory_location in &effect.memory_locations {
                        self.analyze_memory_location(*location, memory_location, &mut memory_analysis)?;
                    }
                }
            }
        }
        
        // Perform escape analysis
        memory_analysis.escape_analysis = self.perform_escape_analysis(function, instruction_effects)?;
        
        Ok(memory_analysis)
    }

    /// Analyze memory location access
    fn analyze_memory_location(
        &self,
        location: u32,
        memory_location: &MemoryLocation,
        analysis: &mut MemoryEffectAnalysis,
    ) -> VMResult<()> {
        // Create or update memory region
        let region_id = self.get_or_create_memory_region(memory_location, analysis)?;
        
        // Update access pattern
        if let Some(region) = analysis.memory_regions.get_mut(&region_id) {
            // Analyze access pattern based on location sequence
            // This is a simplified implementation
            region.access_pattern = AccessPattern::Random; // Default
        }
        
        Ok(())
    }

    /// Get or create memory region for a location
    fn get_or_create_memory_region(
        &self,
        memory_location: &MemoryLocation,
        analysis: &mut MemoryEffectAnalysis,
    ) -> VMResult<u32> {
        // Find existing region or create new one
        for (id, region) in &analysis.memory_regions {
            if region.locations.contains(memory_location) {
                return Ok(*id);
            }
        }
        
        // Create new region
        let region_id = analysis.memory_regions.len() as u32;
        let region = MemoryRegion {
            id: region_id,
            locations: vec![memory_location.clone()],
            access_pattern: AccessPattern::Sequential,
            escapes: false,
            thread_safety: ThreadSafetyLevel::Unknown,
        };
        
        analysis.memory_regions.insert(region_id, region);
        Ok(region_id)
    }

    /// Perform escape analysis
    fn perform_escape_analysis(
        &self,
        function: &FunctionDefinition,
        instruction_effects: &HashMap<u32, Vec<InferredEffect>>,
    ) -> VMResult<EscapeAnalysis> {
        let mut escape_analysis = EscapeAnalysis::default();
        
        // Analyze each instruction for escape potential
        for (location, instruction) in function.instructions.iter().enumerate() {
            match instruction.opcode {
                PrismOpcode::RETURN | PrismOpcode::RETURN_VALUE => {
                    // Values returned escape the function
                    // This is a simplified analysis
                    escape_analysis.escaping_objects.insert(location as u32);
                }
                PrismOpcode::STORE_GLOBAL(_) => {
                    // Values stored to globals escape
                    escape_analysis.escaping_objects.insert(location as u32);
                }
                PrismOpcode::CALL(_) | PrismOpcode::CALL_DYNAMIC(_) => {
                    // Values passed to functions may escape
                    // This requires interprocedural analysis
                    escape_analysis.escaping_objects.insert(location as u32);
                }
                _ => {
                    // Local operations don't escape
                    escape_analysis.non_escaping_objects.insert(location as u32);
                }
            }
        }
        
        Ok(escape_analysis)
    }

    /// Analyze capabilities
    fn analyze_capabilities(
        &self, 
        function: &FunctionDefinition, 
        instruction_effects: &HashMap<u32, Vec<InferredEffect>>
    ) -> VMResult<CapabilityAnalysis> {
        let mut capability_analysis = CapabilityAnalysis::default();
        
        // Analyze capability requirements for each instruction
        for (location, instruction) in function.instructions.iter().enumerate() {
            let required_caps = self.get_required_capabilities(instruction)?;
            if !required_caps.is_empty() {
                capability_analysis.required_capabilities.insert(location as u32, required_caps);
            }
        }
        
        // Analyze capability flow
        for (location, effects) in instruction_effects {
            for effect in effects {
                for capability in &effect.required_capabilities {
                    capability_analysis.capability_flow
                        .entry(*location)
                        .or_default()
                        .required
                        .insert(capability.clone());
                }
            }
        }
        
        Ok(capability_analysis)
    }

    /// Get required capabilities for an instruction
    fn get_required_capabilities(&self, instruction: &Instruction) -> VMResult<Vec<Capability>> {
        // Return capabilities from instruction metadata
        Ok(instruction.required_capabilities.clone())
    }

    /// Detect optimization opportunities
    fn detect_optimization_opportunities(
        &mut self,
        function: &FunctionDefinition,
        effect_flow: &EffectFlow,
        safety_analysis: &SafetyAnalysis,
    ) -> VMResult<Vec<EffectOptimizationOpportunity>> {
        self.optimization_detector.detect_opportunities(function, effect_flow, safety_analysis)
    }

    /// Generate effect summary for interprocedural analysis
    fn generate_effect_summary(
        &self,
        effect_flow: &EffectFlow,
        memory_analysis: &MemoryEffectAnalysis,
    ) -> VMResult<EffectSummary> {
        let mut summary = EffectSummary::default();
        
        // Collect all effects that may occur
        for effects in effect_flow.effects_at_point.values() {
            summary.may_effects.effects.extend(effects.effects.clone());
            summary.may_effects.categories.extend(effects.categories.clone());
            summary.may_effects.has_side_effects |= effects.has_side_effects;
            summary.may_effects.has_memory_effects |= effects.has_memory_effects;
            summary.may_effects.has_io_effects |= effects.has_io_effects;
        }
        
        // Collect memory locations that may be accessed
        for region in memory_analysis.memory_regions.values() {
            summary.may_access.extend(region.locations.clone());
        }
        
        Ok(summary)
    }
}

impl EffectInferenceEngine {
    /// Create a new effect inference engine
    pub fn new() -> VMResult<Self> {
        let effect_registry = Arc::new(EffectRegistry::new());
        let mut engine = Self {
            effect_registry,
            instruction_effects: HashMap::new(),
            function_effect_cache: HashMap::new(),
        };
        
        engine.initialize_instruction_effects()?;
        Ok(engine)
    }

    /// Initialize effect mappings for instructions
    fn initialize_instruction_effects(&mut self) -> VMResult<()> {
        // Memory effects
        self.add_memory_effects()?;
        
        // I/O effects
        self.add_io_effects()?;
        
        // Control flow effects
        self.add_control_effects()?;
        
        // Capability effects
        self.add_capability_effects()?;
        
        Ok(())
    }

    /// Add memory effect mappings
    fn add_memory_effects(&mut self) -> VMResult<()> {
        // Load operations
        for opcode in [PrismOpcode::LOAD_LOCAL(0), PrismOpcode::LOAD_GLOBAL(0)] {
            let template = EffectTemplate {
                effect: EffectDefinition::new(
                    "Memory.Read".to_string(),
                    "Read from memory location".to_string(),
                    EffectCategory::Memory,
                ),
                conditions: Vec::new(),
                parameter_mappings: HashMap::new(),
            };
            self.instruction_effects.entry(opcode).or_default().push(template);
        }
        
        // Store operations
        for opcode in [PrismOpcode::STORE_LOCAL(0), PrismOpcode::STORE_GLOBAL(0)] {
            let template = EffectTemplate {
                effect: EffectDefinition::new(
                    "Memory.Write".to_string(),
                    "Write to memory location".to_string(),
                    EffectCategory::Memory,
                ),
                conditions: Vec::new(),
                parameter_mappings: HashMap::new(),
            };
            self.instruction_effects.entry(opcode).or_default().push(template);
        }
        
        Ok(())
    }

    /// Add I/O effect mappings
    fn add_io_effects(&mut self) -> VMResult<()> {
        for opcode in [PrismOpcode::IO_READ(0), PrismOpcode::IO_WRITE(0)] {
            let template = EffectTemplate {
                effect: EffectDefinition::new(
                    "IO.Operation".to_string(),
                    "I/O operation".to_string(),
                    EffectCategory::IO,
                ),
                conditions: Vec::new(),
                parameter_mappings: HashMap::new(),
            };
            self.instruction_effects.entry(opcode).or_default().push(template);
        }
        
        Ok(())
    }

    /// Add control flow effect mappings
    fn add_control_effects(&mut self) -> VMResult<()> {
        for opcode in [PrismOpcode::CALL(0), PrismOpcode::RETURN] {
            let template = EffectTemplate {
                effect: EffectDefinition::new(
                    "Control.Call".to_string(),
                    "Control flow change".to_string(),
                    EffectCategory::System,
                ),
                conditions: Vec::new(),
                parameter_mappings: HashMap::new(),
            };
            self.instruction_effects.entry(opcode).or_default().push(template);
        }
        
        Ok(())
    }

    /// Add capability effect mappings
    fn add_capability_effects(&mut self) -> VMResult<()> {
        for opcode in [PrismOpcode::CAP_CHECK(0), PrismOpcode::CAP_DELEGATE(0)] {
            let template = EffectTemplate {
                effect: EffectDefinition::new(
                    "Capability.Check".to_string(),
                    "Capability operation".to_string(),
                    EffectCategory::Security,
                ),
                conditions: Vec::new(),
                parameter_mappings: HashMap::new(),
            };
            self.instruction_effects.entry(opcode).or_default().push(template);
        }
        
        Ok(())
    }

    /// Infer effects for a specific instruction
    pub fn infer_effects(&self, instruction: &Instruction, location: u32) -> VMResult<Vec<InferredEffect>> {
        let mut effects = Vec::new();
        
        // Add explicitly declared effects
        for effect in &instruction.effects {
            effects.push(InferredEffect {
                effect: effect.clone(),
                confidence: 1.0,
                source: EffectInferenceSource::Explicit,
                memory_locations: Vec::new(),
                required_capabilities: instruction.required_capabilities.clone(),
                parameters: HashMap::new(),
            });
        }
        
        // Infer effects from instruction semantics
        if let Some(templates) = self.instruction_effects.get(&instruction.opcode) {
            for template in templates {
                if self.template_applies(template, instruction)? {
                    effects.push(InferredEffect {
                        effect: Effect::new(template.effect.name.clone(), Span::default()),
                        confidence: 0.9,
                        source: EffectInferenceSource::InstructionSemantics,
                        memory_locations: self.infer_memory_locations(instruction)?,
                        required_capabilities: instruction.required_capabilities.clone(),
                        parameters: self.map_parameters(template, instruction)?,
                    });
                }
            }
        }
        
        Ok(effects)
    }

    /// Check if an effect template applies to an instruction
    fn template_applies(&self, template: &EffectTemplate, instruction: &Instruction) -> VMResult<bool> {
        for condition in &template.conditions {
            if !self.check_condition(condition, instruction)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Check a specific condition
    fn check_condition(&self, condition: &EffectCondition, instruction: &Instruction) -> VMResult<bool> {
        match condition {
            EffectCondition::HasOperand(_operand) => {
                // Check if instruction has the specified operand
                // This is a simplified implementation
                Ok(true)
            }
            EffectCondition::OperandValue(_operand, _value) => {
                // Check if operand has specific value
                Ok(true)
            }
            EffectCondition::Context(_context) => {
                // Check context condition
                Ok(true)
            }
            EffectCondition::Capability(capability) => {
                // Check if instruction requires capability
                Ok(instruction.required_capabilities.contains(capability))
            }
        }
    }

    /// Infer memory locations affected by instruction
    fn infer_memory_locations(&self, instruction: &Instruction) -> VMResult<Vec<MemoryLocation>> {
        let mut locations = Vec::new();
        
        match instruction.opcode {
            PrismOpcode::LOAD_LOCAL(slot) | PrismOpcode::STORE_LOCAL(slot) => {
                locations.push(MemoryLocation::Local(slot));
            }
            PrismOpcode::LOAD_GLOBAL(id) | PrismOpcode::STORE_GLOBAL(id) => {
                locations.push(MemoryLocation::Global(id));
            }
            PrismOpcode::GET_FIELD(_) | PrismOpcode::SET_FIELD(_) => {
                // Would need more context to determine exact field
                locations.push(MemoryLocation::Unknown);
            }
            _ => {
                // No specific memory location
            }
        }
        
        Ok(locations)
    }

    /// Map template parameters to instruction
    fn map_parameters(&self, template: &EffectTemplate, instruction: &Instruction) -> VMResult<HashMap<String, EffectParameter>> {
        let mut parameters = HashMap::new();
        
        for (param_name, mapping) in &template.parameter_mappings {
            let value = match mapping {
                ParameterMapping::Operand(operand_name) => {
                    // Extract operand value - simplified implementation
                    match operand_name.as_str() {
                        "slot" => {
                            if let PrismOpcode::LOAD_LOCAL(slot) = instruction.opcode {
                                EffectParameter::Integer(slot as i64)
                            } else {
                                continue;
                            }
                        }
                        _ => continue,
                    }
                }
                ParameterMapping::Constant(value) => value.clone(),
                ParameterMapping::Context(_context) => {
                    // Extract from context - simplified
                    EffectParameter::String("unknown".to_string())
                }
            };
            
            parameters.insert(param_name.clone(), value);
        }
        
        Ok(parameters)
    }
}

impl EffectFlowAnalyzer {
    /// Create a new effect flow analyzer
    pub fn new() -> Self {
        Self {
            cfg: None,
            dataflow_state: DataflowState::default(),
        }
    }

    /// Analyze effect flow through a function
    pub fn analyze(
        &mut self,
        function: &FunctionDefinition,
        instruction_effects: &HashMap<u32, Vec<InferredEffect>>,
    ) -> VMResult<EffectFlow> {
        let mut effect_flow = EffectFlow::default();
        
        // Simple forward flow analysis
        for (location, effects) in instruction_effects {
            let mut effect_set = EffectSet::default();
            
            for effect in effects {
                effect_set.effects.insert(effect.clone());
                effect_set.categories.insert(effect.effect.definition.parse().unwrap_or(EffectCategory::System));
                
                // Update flags based on effect category
                match effect.effect.definition.as_str() {
                    s if s.starts_with("Memory.") => effect_set.has_memory_effects = true,
                    s if s.starts_with("IO.") => effect_set.has_io_effects = true,
                    _ => {}
                }
                
                if !matches!(effect.effect.definition.as_str(), "Pure" | "Memory.Read") {
                    effect_set.has_side_effects = true;
                }
            }
            
            effect_flow.effects_at_point.insert(*location, effect_set);
            
            if effect_flow.effects_at_point[location].has_side_effects {
                effect_flow.side_effect_locations.push(*location);
            }
        }
        
        // Identify pure regions
        effect_flow.pure_regions = self.identify_pure_regions(function, &effect_flow.side_effect_locations)?;
        
        // Build effect dependencies
        effect_flow.effect_dependencies = self.build_effect_dependencies(function, instruction_effects)?;
        
        Ok(effect_flow)
    }

    /// Identify pure regions in the function
    fn identify_pure_regions(
        &self,
        function: &FunctionDefinition,
        side_effect_locations: &[u32],
    ) -> VMResult<Vec<PureRegion>> {
        let mut pure_regions = Vec::new();
        let mut current_start = 0;
        
        for &side_effect_location in side_effect_locations {
            if side_effect_location > current_start {
                pure_regions.push(PureRegion {
                    start: current_start,
                    end: side_effect_location - 1,
                    blocks: HashSet::new(), // Would need CFG to populate
                    purity_level: PurityLevel::Pure,
                });
            }
            current_start = side_effect_location + 1;
        }
        
        // Add final pure region if exists
        if current_start < function.instructions.len() as u32 {
            pure_regions.push(PureRegion {
                start: current_start,
                end: function.instructions.len() as u32 - 1,
                blocks: HashSet::new(),
                purity_level: PurityLevel::Pure,
            });
        }
        
        Ok(pure_regions)
    }

    /// Build effect dependencies between instructions
    fn build_effect_dependencies(
        &self,
        function: &FunctionDefinition,
        instruction_effects: &HashMap<u32, Vec<InferredEffect>>,
    ) -> VMResult<HashMap<u32, Vec<u32>>> {
        let mut dependencies = HashMap::new();
        
        // Simple dependency analysis based on memory locations
        for (location1, effects1) in instruction_effects {
            for (location2, effects2) in instruction_effects {
                if location1 >= location2 {
                    continue;
                }
                
                // Check for memory dependencies
                if self.have_memory_dependency(effects1, effects2) {
                    dependencies.entry(*location1).or_insert_with(Vec::new).push(*location2);
                }
            }
        }
        
        Ok(dependencies)
    }

    /// Check if two effect sets have memory dependencies
    fn have_memory_dependency(&self, effects1: &[InferredEffect], effects2: &[InferredEffect]) -> bool {
        for effect1 in effects1 {
            for effect2 in effects2 {
                // Check if they access the same memory location
                for loc1 in &effect1.memory_locations {
                    for loc2 in &effect2.memory_locations {
                        if self.memory_locations_may_alias(loc1, loc2) {
                            // Check for read-write or write-write dependency
                            let is_write1 = effect1.effect.definition.contains("Write");
                            let is_write2 = effect2.effect.definition.contains("Write");
                            
                            if is_write1 || is_write2 {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Check if two memory locations may alias
    fn memory_locations_may_alias(&self, loc1: &MemoryLocation, loc2: &MemoryLocation) -> bool {
        match (loc1, loc2) {
            (MemoryLocation::Local(slot1), MemoryLocation::Local(slot2)) => slot1 == slot2,
            (MemoryLocation::Global(id1), MemoryLocation::Global(id2)) => id1 == id2,
            (MemoryLocation::Unknown, _) | (_, MemoryLocation::Unknown) => true,
            _ => false, // Conservative: different types don't alias
        }
    }
}

impl SafetyValidator {
    /// Create a new safety validator
    pub fn new() -> VMResult<Self> {
        Ok(Self {
            security_system: Arc::new(SecuritySystem::new()),
            safe_patterns: Vec::new(),
            unsafe_patterns: Vec::new(),
        })
    }

    /// Perform safety analysis
    pub fn analyze(
        &mut self,
        function: &FunctionDefinition,
        effect_flow: &EffectFlow,
    ) -> VMResult<SafetyAnalysis> {
        let mut safety_analysis = SafetyAnalysis::default();
        
        // Check for memory safety violations
        safety_analysis.memory_safety_violations = self.check_memory_safety(function, effect_flow)?;
        
        // Check for data races
        safety_analysis.data_race_potential = self.check_data_races(function, effect_flow)?;
        
        // Check for unsafe optimizations
        safety_analysis.unsafe_optimizations = self.check_unsafe_optimizations(function, effect_flow)?;
        
        Ok(safety_analysis)
    }

    /// Check for memory safety violations
    fn check_memory_safety(
        &self,
        function: &FunctionDefinition,
        effect_flow: &EffectFlow,
    ) -> VMResult<Vec<MemorySafetyViolation>> {
        let mut violations = Vec::new();
        
        // Check for potential null pointer dereferences
        for (location, instruction) in function.instructions.iter().enumerate() {
            match instruction.opcode {
                PrismOpcode::GET_FIELD(_) | PrismOpcode::SET_FIELD(_) |
                PrismOpcode::GET_INDEX | PrismOpcode::SET_INDEX => {
                    // These could potentially dereference null
                    violations.push(MemorySafetyViolation {
                        location: location as u32,
                        violation_type: MemorySafetyViolationType::NullPointerDereference,
                        severity: SafetyViolationSeverity::Medium,
                        description: "Potential null pointer dereference".to_string(),
                        memory_locations: vec![MemoryLocation::Unknown],
                    });
                }
                _ => {}
            }
        }
        
        Ok(violations)
    }

    /// Check for potential data races
    fn check_data_races(
        &self,
        function: &FunctionDefinition,
        effect_flow: &EffectFlow,
    ) -> VMResult<Vec<DataRaceLocation>> {
        let mut data_races = Vec::new();
        
        // This is a simplified analysis - real implementation would need
        // interprocedural analysis and thread modeling
        for (loc1, effects1) in &effect_flow.effects_at_point {
            for (loc2, effects2) in &effect_flow.effects_at_point {
                if loc1 >= loc2 {
                    continue;
                }
                
                // Check if both access memory and at least one is a write
                if effects1.has_memory_effects && effects2.has_memory_effects {
                    if effects1.has_side_effects || effects2.has_side_effects {
                        data_races.push(DataRaceLocation {
                            location1: *loc1,
                            location2: *loc2,
                            memory_location: MemoryLocation::Unknown,
                            access_types: (MemoryAccessType::Write, MemoryAccessType::Read),
                            confidence: 0.3, // Low confidence without proper analysis
                        });
                    }
                }
            }
        }
        
        Ok(data_races)
    }

    /// Check for unsafe optimizations
    fn check_unsafe_optimizations(
        &self,
        function: &FunctionDefinition,
        effect_flow: &EffectFlow,
    ) -> VMResult<Vec<UnsafeOptimization>> {
        let mut unsafe_opts = Vec::new();
        
        // Check for dead code elimination of effectful code
        for location in &effect_flow.side_effect_locations {
            unsafe_opts.push(UnsafeOptimization {
                location: *location,
                optimization_type: UnsafeOptimizationType::DeadCodeElimination,
                reason: "Cannot eliminate code with side effects".to_string(),
                violated_effects: Vec::new(), // Would populate with actual effects
            });
        }
        
        Ok(unsafe_opts)
    }
}

impl OptimizationOpportunityDetector {
    /// Create a new optimization opportunity detector
    pub fn new() -> VMResult<Self> {
        Ok(Self {
            patterns: Vec::new(),
            cost_model: CostModel {
                instruction_costs: HashMap::new(),
                effect_costs: HashMap::new(),
                memory_costs: HashMap::new(),
            },
        })
    }

    /// Detect optimization opportunities
    pub fn detect_opportunities(
        &mut self,
        function: &FunctionDefinition,
        effect_flow: &EffectFlow,
        safety_analysis: &SafetyAnalysis,
    ) -> VMResult<Vec<EffectOptimizationOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Detect pure function optimization opportunities
        for pure_region in &effect_flow.pure_regions {
            if pure_region.end - pure_region.start > 5 {
                opportunities.push(EffectOptimizationOpportunity {
                    location: pure_region.start,
                    optimization_type: EffectOptimizationType::PureFunctionOptimization,
                    expected_benefit: OptimizationBenefit {
                        performance_gain: 1.2,
                        memory_savings: 0,
                        code_size_impact: 0,
                        confidence: 0.8,
                    },
                    safety_constraints: Vec::new(),
                    required_conditions: vec![
                        OptimizationCondition::NoSideEffects(pure_region.start, pure_region.end)
                    ],
                });
            }
        }
        
        // Detect memory optimization opportunities
        if safety_analysis.memory_safety_violations.is_empty() {
            opportunities.push(EffectOptimizationOpportunity {
                location: 0,
                optimization_type: EffectOptimizationType::MemoryOptimization,
                expected_benefit: OptimizationBenefit {
                    performance_gain: 1.1,
                    memory_savings: 1024,
                    code_size_impact: -100,
                    confidence: 0.6,
                },
                safety_constraints: Vec::new(),
                required_conditions: Vec::new(),
            });
        }
        
        Ok(opportunities)
    }
}

// Implement EffectSet operations
impl EffectSet {
    /// Create a new empty effect set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an effect to the set
    pub fn add_effect(&mut self, effect: InferredEffect) {
        // Update categories
        if let Ok(category) = effect.effect.definition.parse::<EffectCategory>() {
            self.categories.insert(category);
        }
        
        // Update flags
        match effect.effect.definition.as_str() {
            s if s.starts_with("Memory.") => self.has_memory_effects = true,
            s if s.starts_with("IO.") => self.has_io_effects = true,
            _ => {}
        }
        
        if !matches!(effect.effect.definition.as_str(), "Pure") {
            self.has_side_effects = true;
        }
        
        self.effects.insert(effect);
    }

    /// Check if set contains effects of a specific category
    pub fn has_category(&self, category: &EffectCategory) -> bool {
        self.categories.contains(category)
    }

    /// Merge another effect set into this one
    pub fn merge(&mut self, other: &EffectSet) {
        self.effects.extend(other.effects.clone());
        self.categories.extend(other.categories.clone());
        self.has_side_effects |= other.has_side_effects;
        self.has_memory_effects |= other.has_memory_effects;
        self.has_io_effects |= other.has_io_effects;
    }
}

impl std::str::FromStr for EffectCategory {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Pure" => Ok(EffectCategory::Pure),
            "IO" => Ok(EffectCategory::IO),
            "Memory" => Ok(EffectCategory::Memory),
            "System" => Ok(EffectCategory::System),
            "Security" => Ok(EffectCategory::Security),
            _ => Err(()),
        }
    }
} 