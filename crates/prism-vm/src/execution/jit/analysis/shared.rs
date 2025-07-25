//! Shared Analysis Types and Interfaces
//!
//! This module provides common data structures, traits, and interfaces used across
//! all analysis modules to ensure consistency and proper interoperability.

use crate::{VMResult, PrismVMError, bytecode::{FunctionDefinition, Instruction}};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeSet, BTreeMap};
use std::fmt;
use std::time::Duration;

/// Common analysis trait that all analyzers must implement
pub trait Analysis: Send + Sync {
    /// Configuration type for this analysis
    type Config;
    /// Result type produced by this analysis
    type Result: Clone + fmt::Debug;
    /// Dependencies required by this analysis
    type Dependencies;

    /// Create a new analyzer with the given configuration
    fn new(config: &Self::Config) -> VMResult<Self> where Self: Sized;

    /// Perform the analysis on a function with provided dependencies
    fn analyze(&mut self, function: &FunctionDefinition, deps: Self::Dependencies) -> VMResult<Self::Result>;

    /// Get the analysis kind identifier
    fn analysis_kind() -> AnalysisKind where Self: Sized;

    /// Get the list of analysis dependencies
    fn dependencies() -> Vec<AnalysisKind> where Self: Sized;

    /// Validate that dependencies are satisfied
    fn validate_dependencies(deps: &Self::Dependencies) -> VMResult<()>;
}

/// Enumeration of all analysis types in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AnalysisKind {
    ControlFlow,
    DataFlow,
    Loop,
    Type,
    Effect,
    Hotness,
    Capability,
    CapabilityAwareInlining,
}

impl fmt::Display for AnalysisKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisKind::ControlFlow => write!(f, "control_flow"),
            AnalysisKind::DataFlow => write!(f, "data_flow"),
            AnalysisKind::Loop => write!(f, "loop"),
            AnalysisKind::Type => write!(f, "type"),
            AnalysisKind::Effect => write!(f, "effect"),
            AnalysisKind::Hotness => write!(f, "hotness"),
            AnalysisKind::Capability => write!(f, "capability"),
            AnalysisKind::CapabilityAwareInlining => write!(f, "capability_aware_inlining"),
        }
    }
}

/// Unified variable representation used across all analyses
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Variable {
    /// Variable name or identifier
    pub name: String,
    /// Variable type
    pub var_type: VariableType,
    /// Scope information
    pub scope: VariableScope,
}

/// Variable type classification
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VariableType {
    /// Local variable with slot index
    Local { index: u8 },
    /// Global variable
    Global { name: String },
    /// Temporary variable
    Temporary { id: u32 },
    /// Stack slot
    Stack { offset: i32 },
    /// Register
    Register { reg: u8 },
}

/// Variable scope information
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VariableScope {
    /// Function-local scope
    Function,
    /// Block-local scope
    Block { block_id: u32 },
    /// Global scope
    Global,
}

/// Unified memory access pattern representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryAccessPattern {
    /// Base address expression
    pub base: MemoryBase,
    /// Access stride in bytes
    pub stride: i64,
    /// Pattern classification
    pub pattern_type: AccessPatternType,
    /// Pattern regularity metrics
    pub regularity: PatternRegularity,
    /// Temporal locality score (0.0 to 1.0)
    pub temporal_locality: f64,
    /// Spatial locality score (0.0 to 1.0)
    pub spatial_locality: f64,
}

/// Memory base address representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryBase {
    /// Variable reference
    Variable(Variable),
    /// Constant address
    Constant(u64),
    /// Complex expression
    Expression(String),
}

/// Memory access pattern types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessPatternType {
    /// Sequential access (stride = element_size)
    Sequential,
    /// Strided access with known stride
    Strided { stride: i64 },
    /// Random access pattern
    Random,
    /// Indirect access (pointer chasing)
    Indirect,
    /// Array access with index pattern
    ArrayAccess { index_pattern: IndexPattern },
}

/// Index pattern for array accesses
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexPattern {
    /// Linear index (i, i+1, i+2, ...)
    Linear { step: i64 },
    /// Nested loop index (i*N + j)
    Nested { outer_step: i64, inner_step: i64 },
    /// Random index
    Random,
    /// Sparse index access
    Sparse { density: f64 },
}

/// Pattern regularity metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternRegularity {
    /// How regular the pattern is (0.0 to 1.0)
    pub regularity_score: f64,
    /// Predictability of next access (0.0 to 1.0)
    pub predictability: f64,
    /// Variance in access pattern
    pub variance: f64,
}

/// Unified induction variable representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InductionVariable {
    /// Variable being analyzed
    pub variable: Variable,
    /// Loop ID this IV belongs to
    pub loop_id: u32,
    /// Induction variable classification
    pub iv_type: InductionVariableType,
    /// Initial value (if statically determinable)
    pub initial_value: Option<i64>,
    /// Step value per iteration
    pub step: i64,
    /// Final value (if statically determinable)
    pub final_value: Option<i64>,
    /// Whether this IV controls loop termination
    pub is_loop_control: bool,
    /// Variables dependent on this IV
    pub dependent_variables: Vec<DependentVariable>,
}

/// Induction variable type classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InductionVariableType {
    /// Basic induction variable (i = i + c)
    Basic,
    /// Derived induction variable (j = a * i + b)
    Derived {
        /// Base induction variable
        base: Variable,
        /// Multiplier coefficient
        multiplier: i64,
        /// Constant offset
        offset: i64,
    },
}

/// Variable dependent on an induction variable
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependentVariable {
    /// Dependent variable
    pub variable: Variable,
    /// Dependency relationship
    pub relationship: DependencyRelationship,
}

/// Dependency relationship types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DependencyRelationship {
    /// Linear dependency (var = a * iv + b)
    Linear { multiplier: i64, offset: i64 },
    /// Polynomial dependency
    Polynomial { coefficients: Vec<i64> },
    /// Complex dependency (non-analyzable)
    Complex,
}

/// Unified optimization opportunity representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Unique opportunity identifier
    pub id: String,
    /// Optimization classification
    pub kind: OptimizationKind,
    /// Location in code
    pub location: CodeLocation,
    /// Estimated benefit (0.0 to 1.0)
    pub estimated_benefit: f64,
    /// Implementation cost (0.0 to 1.0)
    pub implementation_cost: f64,
    /// Confidence in this opportunity (0.0 to 1.0)
    pub confidence: f64,
    /// Prerequisites for applying this optimization
    pub prerequisites: Vec<String>,
    /// Potential negative side effects
    pub side_effects: Vec<String>,
    /// Detailed analysis of this opportunity
    pub analysis: OptimizationAnalysis,
    /// Source analysis that detected this opportunity
    pub source_analysis: AnalysisKind,
}

/// Code location representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeLocation {
    /// Single instruction
    Instruction { offset: u32 },
    /// Range of instructions
    InstructionRange { start: u32, end: u32 },
    /// Basic block
    BasicBlock { block_id: u32 },
    /// Loop
    Loop { loop_id: u32 },
    /// Function call site
    CallSite { instruction: u32, target: Option<u32> },
    /// Memory access location
    MemoryAccess { instruction: u32 },
}

/// Optimization kind enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationKind {
    /// Dead code elimination
    DeadCodeElimination {
        dead_instructions: Vec<u32>,
        unreachable_blocks: Vec<u32>,
        dead_variables: Vec<Variable>,
    },
    /// Constant folding and propagation
    ConstantFolding {
        expressions: Vec<FoldableExpression>,
        propagation_chains: Vec<PropagationChain>,
    },
    /// Common subexpression elimination
    CommonSubexpressionElimination {
        subexpressions: Vec<CommonSubexpression>,
    },
    /// Strength reduction
    StrengthReduction {
        operations: Vec<StrengthReductionOp>,
        induction_variables: Vec<InductionVariable>,
    },
    /// Loop optimizations
    LoopOptimization {
        loop_id: u32,
        techniques: Vec<LoopOptimizationTechnique>,
    },
    /// Function inlining
    Inlining {
        call_site: u32,
        target_function: u32,
        strategy: InliningStrategy,
    },
    /// Vectorization
    Vectorization {
        target: VectorizationTarget,
        vector_width: u32,
        strategy: VectorizationStrategy,
    },
    /// Branch optimization
    BranchOptimization {
        branch_location: u32,
        optimization_type: BranchOptimizationType,
    },
    /// Memory optimization
    MemoryOptimization {
        operations: Vec<MemoryOperation>,
        strategy: MemoryOptimizationStrategy,
    },
    /// Type specialization
    TypeSpecialization {
        location: u32,
        specialization_type: SpecializationType,
    },
    /// Hotness-based optimization
    HotnessOptimization {
        hot_spot_id: u32,
        optimization_type: HotnessOptimizationType,
    },
}

/// Detailed optimization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAnalysis {
    /// Performance impact estimation
    pub performance_impact: PerformanceImpact,
    /// Resource usage impact
    pub resource_impact: ResourceImpact,
    /// Dependencies for this optimization
    pub dependencies: Vec<OptimizationDependency>,
    /// Risk assessment
    pub risks: Vec<OptimizationRisk>,
    /// Profitability analysis
    pub profitability: ProfitabilityAnalysis,
}

/// Performance impact metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Expected speedup factor
    pub speedup: f64,
    /// Instruction count reduction
    pub instruction_reduction: i32,
    /// Memory access reduction
    pub memory_access_reduction: i32,
    /// Branch misprediction improvement
    pub branch_improvement: f64,
    /// Cache performance improvement
    pub cache_improvement: f64,
}

/// Resource usage impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceImpact {
    /// Code size change in bytes
    pub code_size_delta: i32,
    /// Register pressure change
    pub register_pressure_delta: i32,
    /// Memory usage change
    pub memory_usage_delta: i32,
    /// Compilation time cost in milliseconds
    pub compilation_time_cost: f64,
}

/// Optimization dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationDependency {
    /// Type of dependency
    pub dependency_type: DependencyType,
    /// Human-readable description
    pub description: String,
    /// Whether this dependency is currently satisfied
    pub satisfied: bool,
}

/// Dependency types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Requires another optimization to be applied first
    RequiresOptimization(String),
    /// Requires specific analysis results
    RequiresAnalysis(AnalysisKind),
    /// Requires runtime conditions
    RequiresRuntimeCondition(String),
    /// Conflicts with another optimization
    ConflictsWith(String),
}

/// Optimization risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRisk {
    /// Type of risk
    pub risk_type: RiskType,
    /// Risk description
    pub description: String,
    /// Risk probability (0.0 to 1.0)
    pub probability: f64,
    /// Risk severity (0.0 to 1.0)
    pub severity: f64,
}

/// Risk types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskType {
    /// May change program semantics
    SemanticChange,
    /// May introduce correctness issues
    CorrectnessRisk,
    /// May degrade performance in some cases
    PerformanceRegression,
    /// May increase resource usage
    ResourceIncrease,
    /// May complicate debugging
    DebuggingComplexity,
}

/// Profitability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitabilityAnalysis {
    /// Expected performance gain in cycles
    pub performance_gain: f64,
    /// Code size impact in bytes
    pub code_size_impact: i32,
    /// Compilation time cost in milliseconds
    pub compilation_time_cost: f64,
    /// Risk level (0.0 to 1.0)
    pub risk_level: f64,
    /// Overall profitability score
    pub profitability_score: f64,
}

// Supporting types for optimization kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldableExpression {
    pub location: u32,
    pub operator: String,
    pub operands: Vec<ConstantValue>,
    pub result: ConstantValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationChain {
    pub source: ConstantValue,
    pub uses: Vec<u32>,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonSubexpression {
    pub expression: String,
    pub locations: Vec<u32>,
    pub benefit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengthReductionOp {
    pub original_op: String,
    pub reduced_op: String,
    pub location: u32,
    pub transformation: MathematicalTransformation,
    pub benefit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathematicalTransformation {
    PowerOfTwoMultiplication { power: u32 },
    PowerOfTwoDivision { power: u32 },
    MultiplicationToAddition { constant: i64, chain: Vec<String> },
    ExponentiationToMultiplication { exponent: u32 },
    ModuloPowerOfTwo { power: u32 },
    AlgebraicSimplification { transformation: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopOptimizationTechnique {
    InvariantCodeMotion { expressions: Vec<String> },
    Unrolling { factor: u32, allow_partial: bool },
    Vectorization { vector_width: u32, operations: Vec<String> },
    Fusion { target_loop: u32 },
    Distribution { split_points: Vec<u32> },
    StrengthReduction { operations: Vec<String> },
    Interchange { loops: Vec<u32> },
    Tiling { tile_sizes: Vec<u32> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InliningStrategy {
    Full,
    Partial { conditions: Vec<String> },
    Speculative { probability: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorizationTarget {
    Loop { loop_id: u32 },
    StraightLine { start: u32, end: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorizationStrategy {
    Auto,
    Explicit { width: u32 },
    Predicated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BranchOptimizationType {
    Elimination,
    PredictionImprovement,
    Fusion,
    Reordering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOperation {
    pub op_type: MemoryOpType,
    pub location: u32,
    pub address: String,
    pub access_pattern: MemoryAccessPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOpType {
    Load,
    Store,
    Prefetch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimizationStrategy {
    Coalescing,
    Prefetching,
    CacheFriendlyReordering,
    AlignmentOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializationType {
    MonomorphicCall { function_id: u32, specialized_types: Vec<String> },
    IntegerSpecialization { variable: Variable, range: IntegerRange },
    ArraySpecialization { array_type: String, element_type: String },
    FieldSpecialization { object_type: String, field: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotnessOptimizationType {
    AggressiveInlining { call_sites: Vec<u32>, depth: u32 },
    SpeculativeOptimization { operations: Vec<u32>, strategy: String },
    TierUpCompilation { target_tier: String },
    CodeLayoutOptimization { hot_regions: Vec<u32>, cold_regions: Vec<u32> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegerRange {
    pub min: i64,
    pub max: i64,
    pub is_exact: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstantValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Null,
}

/// Analysis metadata for tracking analysis execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis execution time
    pub analysis_time: Duration,
    /// Analysis passes completed
    pub passes_run: Vec<String>,
    /// Warnings generated during analysis
    pub warnings: Vec<String>,
    /// Analysis confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Error types for analysis operations
#[derive(Debug, Clone)]
pub enum AnalysisError {
    /// Missing required dependency
    MissingDependency(AnalysisKind),
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Analysis timeout
    Timeout(Duration),
    /// Internal analysis error
    InternalError(String),
    /// Dependency cycle detected
    DependencyCycle(Vec<AnalysisKind>),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisError::MissingDependency(kind) => {
                write!(f, "Missing required dependency: {}", kind)
            }
            AnalysisError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            AnalysisError::Timeout(duration) => {
                write!(f, "Analysis timeout after {:?}", duration)
            }
            AnalysisError::InternalError(msg) => {
                write!(f, "Internal analysis error: {}", msg)
            }
            AnalysisError::DependencyCycle(cycle) => {
                write!(f, "Dependency cycle detected: {:?}", cycle)
            }
        }
    }
}

impl std::error::Error for AnalysisError {}

/// Helper functions for common operations
impl Variable {
    /// Create a new local variable
    pub fn local(name: impl Into<String>, index: u8) -> Self {
        Self {
            name: name.into(),
            var_type: VariableType::Local { index },
            scope: VariableScope::Function,
        }
    }

    /// Create a new global variable
    pub fn global(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            var_type: VariableType::Global { name },
            scope: VariableScope::Global,
        }
    }

    /// Create a new temporary variable
    pub fn temporary(id: u32) -> Self {
        Self {
            name: format!("temp_{}", id),
            var_type: VariableType::Temporary { id },
            scope: VariableScope::Function,
        }
    }
}

impl OptimizationOpportunity {
    /// Calculate the net benefit of this optimization
    pub fn net_benefit(&self) -> f64 {
        (self.estimated_benefit - self.implementation_cost).max(0.0) * self.confidence
    }

    /// Check if this optimization is profitable
    pub fn is_profitable(&self) -> bool {
        self.analysis.profitability.profitability_score > 0.0 && 
        self.estimated_benefit > self.implementation_cost
    }

    /// Get the risk-adjusted benefit
    pub fn risk_adjusted_benefit(&self) -> f64 {
        let risk_factor = 1.0 - self.analysis.profitability.risk_level;
        self.net_benefit() * risk_factor
    }
}

impl MemoryAccessPattern {
    /// Create a sequential access pattern
    pub fn sequential(base: Variable) -> Self {
        Self {
            base: MemoryBase::Variable(base),
            stride: 1,
            pattern_type: AccessPatternType::Sequential,
            regularity: PatternRegularity {
                regularity_score: 1.0,
                predictability: 1.0,
                variance: 0.0,
            },
            temporal_locality: 0.8,
            spatial_locality: 1.0,
        }
    }

    /// Create a strided access pattern
    pub fn strided(base: Variable, stride: i64) -> Self {
        Self {
            base: MemoryBase::Variable(base),
            stride,
            pattern_type: AccessPatternType::Strided { stride },
            regularity: PatternRegularity {
                regularity_score: 0.9,
                predictability: 0.9,
                variance: 0.1,
            },
            temporal_locality: 0.6,
            spatial_locality: 0.8,
        }
    }

    /// Create a random access pattern
    pub fn random(base: Variable) -> Self {
        Self {
            base: MemoryBase::Variable(base),
            stride: 0,
            pattern_type: AccessPatternType::Random,
            regularity: PatternRegularity {
                regularity_score: 0.0,
                predictability: 0.0,
                variance: 1.0,
            },
            temporal_locality: 0.1,
            spatial_locality: 0.1,
        }
    }
}

impl InductionVariable {
    /// Create a basic induction variable
    pub fn basic(variable: Variable, loop_id: u32, step: i64) -> Self {
        Self {
            variable,
            loop_id,
            iv_type: InductionVariableType::Basic,
            initial_value: None,
            step,
            final_value: None,
            is_loop_control: false,
            dependent_variables: Vec::new(),
        }
    }

    /// Create a derived induction variable
    pub fn derived(
        variable: Variable, 
        loop_id: u32, 
        base: Variable, 
        multiplier: i64, 
        offset: i64
    ) -> Self {
        Self {
            variable,
            loop_id,
            iv_type: InductionVariableType::Derived { base, multiplier, offset },
            initial_value: None,
            step: multiplier,
            final_value: None,
            is_loop_control: false,
            dependent_variables: Vec::new(),
        }
    }
} 