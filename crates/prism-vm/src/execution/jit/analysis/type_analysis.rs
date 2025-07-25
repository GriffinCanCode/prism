//! Type Analysis for JIT Optimization
//!
//! This module provides comprehensive type inference and analysis capabilities for the JIT compiler,
//! enabling advanced type-based optimizations and safety checks. It implements a hybrid approach
//! combining constraint-based type inference with control flow analysis and value range propagation.
//!
//! ## Research Inspirations & Credits
//!
//! - **Constraint-based Type Inference**: Based on Hindley-Milner algorithm with unification (Damas & Milner, 1982)
//! - **Control Flow Analysis**: Inspired by V8 TurboFan's type refinement through control flow (Google V8 Team)
//! - **Value Range Analysis**: Adapted from LLVM's ScalarEvolution and GCC's value range propagation
//! - **Tiered Compilation**: Following HotSpot JVM's approach to hot spot detection (Oracle/Sun)
//! - **Polymorphic Inline Caching**: Based on Smalltalk-80 and Self virtual machine techniques
//!
//! ## Concepts
//!
//! Our implementation combines:
//! - **Unified Analysis Framework**: Single pass integrating type inference, range analysis, and specialization detection
//! - **Confidence-Weighted Types**: Probabilistic type inference with confidence scoring for speculative optimization
//! - **Cross-Analysis Integration**: Type flow analysis directly informs value range propagation and vice versa
//! - **Bytecode-Aware Constraints**: Constraint generation specifically tailored to stack-based VM instructions
//! - **Multi-Tier Specialization**: Hierarchical specialization opportunities from arithmetic to polymorphic dispatch
//!
//! ## Key Features
//!
//! - Constraint-based type inference with unification
//! - Control flow sensitive type propagation
//! - Value range analysis for integer types
//! - Polymorphic type handling
//! - Type specialization opportunities detection
//! - Memory safety analysis through type tracking

use crate::{VMResult, PrismVMError, bytecode::{FunctionDefinition, Instruction, PrismOpcode}};
use super::{AnalysisConfig, control_flow::{ControlFlowGraph, BasicBlock}};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::fmt;

/// Type analyzer implementing constraint-based type inference
#[derive(Debug)]
pub struct TypeAnalyzer {
    config: AnalysisConfig,
    /// Type variable generator
    type_var_counter: u32,
    /// Constraint solver state
    solver_state: ConstraintSolverState,
}

/// Constraint solver state
#[derive(Debug, Default)]
struct ConstraintSolverState {
    /// Current substitution mapping
    substitutions: HashMap<TypeVariable, InferredType>,
    /// Unification stack for constraint solving
    unification_stack: VecDeque<(InferredType, InferredType)>,
    /// Occurs check cache
    occurs_cache: HashMap<(TypeVariable, InferredType), bool>,
}

/// Comprehensive type analysis results
#[derive(Debug, Clone)]
pub struct TypeAnalysis {
    pub function_id: u32,
    pub type_environment: TypeEnvironment,
    pub type_constraints: Vec<TypeConstraint>,
    pub type_inference: TypeInference,
    pub value_ranges: ValueRangeAnalysis,
    pub specialization_opportunities: Vec<SpecializationOpportunity>,
    pub type_flow: TypeFlowAnalysis,
    pub polymorphic_sites: Vec<PolymorphicSite>,
}

/// Enhanced type environment with scope tracking
#[derive(Debug, Clone, Default)]
pub struct TypeEnvironment {
    /// Variable types with scope information
    pub variable_types: HashMap<String, ScopedType>,
    /// Function signatures with polymorphic support
    pub function_signatures: HashMap<u32, PolymorphicSignature>,
    /// Type aliases and definitions
    pub type_definitions: HashMap<String, TypeDefinition>,
    /// Global type assumptions
    pub global_assumptions: HashMap<String, InferredType>,
}

/// Type with scope information
#[derive(Debug, Clone)]
pub struct ScopedType {
    pub inferred_type: InferredType,
    pub scope: TypeScope,
    pub confidence: f64,
    pub definition_location: u32,
    pub last_assignment: Option<u32>,
}

/// Type scope information
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeScope {
    Global,
    Function,
    Block { block_id: u32 },
    Loop { loop_id: u32 },
}

/// Polymorphic function signature
#[derive(Debug, Clone)]
pub struct PolymorphicSignature {
    pub type_parameters: Vec<TypeVariable>,
    pub parameter_types: Vec<InferredType>,
    pub return_type: InferredType,
    pub constraints: Vec<TypeConstraint>,
    pub effect_signature: Vec<String>,
}

/// Type definition for user-defined types
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub name: String,
    pub definition: TypeDefinitionKind,
    pub type_parameters: Vec<TypeVariable>,
}

/// Kind of type definition
#[derive(Debug, Clone)]
pub enum TypeDefinitionKind {
    Struct { fields: Vec<(String, InferredType)> },
    Enum { variants: Vec<(String, Vec<InferredType>)> },
    Alias { target: InferredType },
}

/// Type constraints for constraint-based inference
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub constraint_id: u32,
    pub location: u32,
    pub constraint_type: ConstraintType,
    pub origin: ConstraintOrigin,
    pub priority: ConstraintPriority,
}

/// Enhanced constraint types
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Type equality constraint
    Equality { left: InferredType, right: InferredType },
    /// Subtype constraint
    Subtype { subtype: InferredType, supertype: InferredType },
    /// Type class constraint
    TypeClass { type_var: TypeVariable, class: String },
    /// Field access constraint
    FieldAccess { object: InferredType, field: String, result: InferredType },
    /// Function call constraint
    FunctionCall { function: InferredType, args: Vec<InferredType>, result: InferredType },
    /// Array access constraint
    ArrayAccess { array: InferredType, index: InferredType, element: InferredType },
    /// Arithmetic constraint
    Arithmetic { op: ArithmeticOp, left: InferredType, right: InferredType, result: InferredType },
    /// Conditional constraint (for control flow)
    Conditional { condition: InferredType, then_type: InferredType, else_type: InferredType, result: InferredType },
}

/// Arithmetic operations for type constraints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmeticOp {
    Add, Sub, Mul, Div, Mod, BitAnd, BitOr, BitXor, Shl, Shr,
}

/// Constraint origin for debugging
#[derive(Debug, Clone)]
pub enum ConstraintOrigin {
    Instruction { offset: u32, opcode: String },
    FunctionCall { function_id: u32 },
    FieldAccess { object: String, field: String },
    TypeAnnotation,
    Default,
}

/// Constraint priority for solving order
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Enhanced type inference results
#[derive(Debug, Clone, Default)]
pub struct TypeInference {
    /// Inferred types for all program points
    pub inferred_types: HashMap<String, InferredType>,
    /// Confidence scores for each inference
    pub confidence_scores: HashMap<String, f64>,
    /// Type variable bindings
    pub type_bindings: HashMap<TypeVariable, InferredType>,
    /// Unresolved type variables
    pub unresolved_variables: HashSet<TypeVariable>,
    /// Type errors and warnings
    pub type_errors: Vec<TypeError>,
    /// Inference statistics
    pub statistics: InferenceStatistics,
}

/// Type error information
#[derive(Debug, Clone)]
pub struct TypeError {
    pub location: u32,
    pub error_type: TypeErrorKind,
    pub message: String,
    pub severity: ErrorSeverity,
}

/// Type error kinds
#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    TypeMismatch { expected: InferredType, actual: InferredType },
    UnresolvedType { type_var: TypeVariable },
    RecursiveType { type_var: TypeVariable },
    FieldNotFound { object_type: InferredType, field: String },
    InvalidOperation { op: String, operand_types: Vec<InferredType> },
    ArityMismatch { expected: usize, actual: usize },
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Fatal,
}

/// Inference statistics
#[derive(Debug, Clone, Default)]
pub struct InferenceStatistics {
    pub constraints_generated: usize,
    pub constraints_solved: usize,
    pub unification_steps: usize,
    pub type_variables_created: usize,
    pub polymorphic_instantiations: usize,
}

/// Value range analysis for integer types
#[derive(Debug, Clone, Default)]
pub struct ValueRangeAnalysis {
    /// Integer value ranges at each program point
    pub integer_ranges: HashMap<String, IntegerRange>,
    /// Constant values detected
    pub constants: HashMap<String, ConstantValue>,
    /// Overflow detection results
    pub overflow_sites: Vec<OverflowSite>,
}

/// Integer value range
#[derive(Debug, Clone)]
pub struct IntegerRange {
    pub min: i64,
    pub max: i64,
    pub is_exact: bool,
    pub confidence: f64,
}

/// Constant value information
#[derive(Debug, Clone)]
pub enum ConstantValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Null,
}

/// Overflow detection site
#[derive(Debug, Clone)]
pub struct OverflowSite {
    pub location: u32,
    pub operation: ArithmeticOp,
    pub operand_ranges: Vec<IntegerRange>,
    pub overflow_probability: f64,
}

/// Type flow analysis for control flow sensitive typing
#[derive(Debug, Clone, Default)]
pub struct TypeFlowAnalysis {
    /// Type states at each basic block entry
    pub block_entry_types: HashMap<u32, TypeState>,
    /// Type states at each basic block exit
    pub block_exit_types: HashMap<u32, TypeState>,
    /// Type refinements from conditionals
    pub type_refinements: HashMap<u32, TypeRefinement>,
    /// Phi node type merging information
    pub phi_types: HashMap<u32, PhiTypeInfo>,
}

/// Type state at a program point
#[derive(Debug, Clone, Default)]
pub struct TypeState {
    pub variable_types: HashMap<String, InferredType>,
    pub stack_types: Vec<InferredType>,
    pub type_assumptions: HashMap<String, TypeAssumption>,
}

/// Type refinement from conditional
#[derive(Debug, Clone)]
pub struct TypeRefinement {
    pub variable: String,
    pub original_type: InferredType,
    pub refined_type: InferredType,
    pub condition: RefinementCondition,
}

/// Refinement condition
#[derive(Debug, Clone)]
pub enum RefinementCondition {
    TypeTest { test_type: InferredType },
    NullCheck,
    RangeCheck { min: i64, max: i64 },
    Comparison { op: ComparisonOp, value: ConstantValue },
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal, NotEqual, LessThan, LessEqual, GreaterThan, GreaterEqual,
}

/// Type assumption
#[derive(Debug, Clone)]
pub struct TypeAssumption {
    pub assumed_type: InferredType,
    pub confidence: f64,
    pub source: AssumptionSource,
}

/// Source of type assumption
#[derive(Debug, Clone)]
pub enum AssumptionSource {
    TypeTest,
    PreviousAssignment,
    FunctionParameter,
    ControlFlow,
}

/// Phi node type information
#[derive(Debug, Clone)]
pub struct PhiTypeInfo {
    pub incoming_types: Vec<InferredType>,
    pub merged_type: InferredType,
    pub merge_strategy: MergeStrategy,
}

/// Type merge strategy
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    Union,
    Intersection,
    LeastUpperBound,
    WidenToAny,
}

/// Type specialization opportunity
#[derive(Debug, Clone)]
pub struct SpecializationOpportunity {
    pub location: u32,
    pub opportunity_type: SpecializationType,
    pub potential_benefit: f64,
    pub required_conditions: Vec<String>,
}

/// Types of specialization opportunities
#[derive(Debug, Clone)]
pub enum SpecializationType {
    MonomorphicCall { function_id: u32, specialized_types: Vec<InferredType> },
    IntegerSpecialization { variable: String, range: IntegerRange },
    ArraySpecialization { array_type: InferredType, element_type: InferredType },
    FieldSpecialization { object_type: InferredType, field: String },
}

/// Polymorphic call site information
#[derive(Debug, Clone)]
pub struct PolymorphicSite {
    pub location: u32,
    pub possible_types: Vec<InferredType>,
    pub call_frequency: HashMap<InferredType, f64>,
    pub specialization_benefit: f64,
}

/// Enhanced inferred type system with polymorphism and constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferredType {
    /// Primitive integer type with bit width
    Integer { bits: u8, signed: bool },
    /// Floating point type with precision
    Float { bits: u8 },
    /// Boolean type
    Boolean,
    /// String type
    String,
    /// Array type with element type
    Array { element: Box<InferredType>, size: Option<u64> },
    /// Tuple type
    Tuple { elements: Vec<InferredType> },
    /// Function type
    Function { params: Vec<InferredType>, return_type: Box<InferredType> },
    /// Object/struct type
    Object { name: String, fields: HashMap<String, InferredType> },
    /// Union type (for polymorphism)
    Union { types: Vec<InferredType> },
    /// Type variable (for constraint solving)
    Variable(TypeVariable),
    /// Null/void type
    Null,
    /// Any type (top type)
    Any,
    /// Never type (bottom type)
    Never,
    /// Reference type
    Reference { target: Box<InferredType>, mutable: bool },
    /// Generic type with parameters
    Generic { base: String, parameters: Vec<InferredType> },
}

/// Type variable for constraint-based inference
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeVariable {
    pub id: u32,
    pub name: Option<String>,
    pub kind: TypeVariableKind,
}

/// Kind of type variable
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeVariableKind {
    /// Regular type variable
    Type,
    /// Integer type variable
    Integer,
    /// Numeric type variable (int or float)
    Numeric,
    /// Collection type variable
    Collection,
}

/// Function signature with polymorphic support
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub parameters: Vec<InferredType>,
    pub return_type: InferredType,
    pub type_parameters: Vec<TypeVariable>,
    pub constraints: Vec<TypeConstraint>,
}

// Display implementations for better debugging
impl fmt::Display for InferredType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferredType::Integer { bits, signed } => {
                write!(f, "{}{}", if *signed { "i" } else { "u" }, bits)
            }
            InferredType::Float { bits } => write!(f, "f{}", bits),
            InferredType::Boolean => write!(f, "bool"),
            InferredType::String => write!(f, "string"),
            InferredType::Array { element, size } => {
                if let Some(s) = size {
                    write!(f, "[{}; {}]", element, s)
                } else {
                    write!(f, "[{}]", element)
                }
            }
            InferredType::Tuple { elements } => {
                write!(f, "({})", elements.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "))
            }
            InferredType::Function { params, return_type } => {
                write!(f, "({}) -> {}", 
                    params.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", "),
                    return_type)
            }
            InferredType::Object { name, .. } => write!(f, "{}", name),
            InferredType::Union { types } => {
                write!(f, "{}", types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" | "))
            }
            InferredType::Variable(tv) => write!(f, "'{}", tv.id),
            InferredType::Null => write!(f, "null"),
            InferredType::Any => write!(f, "any"),
            InferredType::Never => write!(f, "never"),
            InferredType::Reference { target, mutable } => {
                write!(f, "&{}{}", if *mutable { "mut " } else { "" }, target)
            }
            InferredType::Generic { base, parameters } => {
                write!(f, "{}<{}>", base, 
                    parameters.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", "))
            }
        }
    }
}

impl fmt::Display for TypeVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            write!(f, "{}_{}", name, self.id)
        } else {
            write!(f, "t{}", self.id)
        }
    }
}

impl TypeAnalyzer {
    pub fn new(config: &AnalysisConfig) -> VMResult<Self> {
        Ok(Self {
            config: config.clone(),
            type_var_counter: 0,
            solver_state: ConstraintSolverState::default(),
        })
    }

    /// Perform comprehensive type analysis on a function
    pub fn analyze(&mut self, function: &FunctionDefinition) -> VMResult<TypeAnalysis> {
        // Reset state for new function
        self.reset_analysis_state();
        
        // Step 1: Initialize type environment
        let mut type_environment = self.initialize_type_environment(function)?;
        
        // Step 2: Generate initial type constraints
        let mut constraints = self.generate_constraints(function)?;
        
        // Step 3: Solve constraints using unification
        let type_inference = self.solve_constraints(&mut constraints)?;
        
        // Step 4: Perform value range analysis
        let value_ranges = self.analyze_value_ranges(function, &type_inference)?;
        
        // Step 5: Analyze type flow through control flow
        let type_flow = self.analyze_type_flow(function, &type_inference)?;
        
        // Step 6: Detect specialization opportunities
        let specialization_opportunities = self.detect_specialization_opportunities(
            function, &type_inference, &value_ranges
        )?;
        
        // Step 7: Identify polymorphic sites
        let polymorphic_sites = self.identify_polymorphic_sites(function, &type_inference)?;
        
        // Step 8: Update type environment with inferred types
        self.update_type_environment(&mut type_environment, &type_inference)?;

        Ok(TypeAnalysis {
            function_id: function.id,
            type_environment,
            type_constraints: constraints,
            type_inference,
            value_ranges,
            specialization_opportunities,
            type_flow,
            polymorphic_sites,
        })
    }

    /// Reset analysis state for new function
    fn reset_analysis_state(&mut self) {
        self.type_var_counter = 0;
        self.solver_state = ConstraintSolverState::default();
    }

    /// Generate a fresh type variable
    fn fresh_type_variable(&mut self, kind: TypeVariableKind, name: Option<String>) -> TypeVariable {
        let id = self.type_var_counter;
        self.type_var_counter += 1;
        TypeVariable { id, name, kind }
    }

    /// Initialize type environment with function parameters and known types
    fn initialize_type_environment(&mut self, function: &FunctionDefinition) -> VMResult<TypeEnvironment> {
        let mut env = TypeEnvironment::default();
        
        // Add function parameters with fresh type variables
        for (i, _param) in function.instructions.iter().enumerate().take(function.param_count as usize) {
            let param_name = format!("param_{}", i);
            let type_var = self.fresh_type_variable(TypeVariableKind::Type, Some(param_name.clone()));
            env.variable_types.insert(param_name, ScopedType {
                inferred_type: InferredType::Variable(type_var),
                scope: TypeScope::Function,
                confidence: 0.5,
                definition_location: 0,
                last_assignment: None,
            });
        }
        
        // Add local variables
        for i in 0..function.local_count {
            let local_name = format!("local_{}", i);
            let type_var = self.fresh_type_variable(TypeVariableKind::Type, Some(local_name.clone()));
            env.variable_types.insert(local_name, ScopedType {
                inferred_type: InferredType::Variable(type_var),
                scope: TypeScope::Function,
                confidence: 0.3,
                definition_location: 0,
                last_assignment: None,
            });
        }
        
        Ok(env)
    }

    /// Generate type constraints from function instructions
    fn generate_constraints(&mut self, function: &FunctionDefinition) -> VMResult<Vec<TypeConstraint>> {
        let mut constraints = Vec::new();
        let mut constraint_id = 0;
        
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let location = offset as u32;
            let instruction_constraints = self.generate_instruction_constraints(
                instruction, location, &mut constraint_id
            )?;
            constraints.extend(instruction_constraints);
        }
        
        Ok(constraints)
    }

    /// Generate constraints for a single instruction
    fn generate_instruction_constraints(
        &mut self,
        instruction: &Instruction,
        location: u32,
        constraint_id: &mut u32,
    ) -> VMResult<Vec<TypeConstraint>> {
        let mut constraints = Vec::new();
        
        match instruction.opcode {
            // Constant loading
            PrismOpcode::LOAD_CONST(idx) => {
                // Would need access to constant pool to determine exact type
                let result_var = self.fresh_type_variable(TypeVariableKind::Type, None);
                // Add constraint based on constant type (simplified)
                constraints.push(self.create_constraint(
                    *constraint_id,
                    location,
                    ConstraintType::Equality {
                        left: InferredType::Variable(result_var),
                        right: InferredType::Any, // Would be specific based on constant
                    },
                    ConstraintOrigin::Instruction {
                        offset: location,
                        opcode: "LOAD_CONST".to_string(),
                    },
                    ConstraintPriority::High,
                ));
                *constraint_id += 1;
            }
            
            // Arithmetic operations
            PrismOpcode::ADD => {
                let left_var = self.fresh_type_variable(TypeVariableKind::Numeric, None);
                let right_var = self.fresh_type_variable(TypeVariableKind::Numeric, None);
                let result_var = self.fresh_type_variable(TypeVariableKind::Numeric, None);
                
                constraints.push(self.create_constraint(
                    *constraint_id,
                    location,
                    ConstraintType::Arithmetic {
                        op: ArithmeticOp::Add,
                        left: InferredType::Variable(left_var),
                        right: InferredType::Variable(right_var),
                        result: InferredType::Variable(result_var),
                    },
                    ConstraintOrigin::Instruction {
                        offset: location,
                        opcode: "ADD".to_string(),
                    },
                    ConstraintPriority::Normal,
                ));
                *constraint_id += 1;
            }
            
            // Type tests
            PrismOpcode::IS_NULL => {
                let operand_var = self.fresh_type_variable(TypeVariableKind::Type, None);
                constraints.push(self.create_constraint(
                    *constraint_id,
                    location,
                    ConstraintType::Equality {
                        left: InferredType::Boolean,
                        right: InferredType::Boolean, // Result is always boolean
                    },
                    ConstraintOrigin::Instruction {
                        offset: location,
                        opcode: "IS_NULL".to_string(),
                    },
                    ConstraintPriority::High,
                ));
                *constraint_id += 1;
            }
            
            // Function calls
            PrismOpcode::CALL(argc) => {
                let mut param_vars = Vec::new();
                for i in 0..argc {
                    param_vars.push(self.fresh_type_variable(
                        TypeVariableKind::Type, 
                        Some(format!("call_arg_{}", i))
                    ));
                }
                let function_var = self.fresh_type_variable(TypeVariableKind::Type, None);
                let result_var = self.fresh_type_variable(TypeVariableKind::Type, None);
                
                constraints.push(self.create_constraint(
                    *constraint_id,
                    location,
                    ConstraintType::FunctionCall {
                        function: InferredType::Variable(function_var),
                        args: param_vars.into_iter().map(InferredType::Variable).collect(),
                        result: InferredType::Variable(result_var),
                    },
                    ConstraintOrigin::Instruction {
                        offset: location,
                        opcode: format!("CALL({})", argc),
                    },
                    ConstraintPriority::Normal,
                ));
                *constraint_id += 1;
            }
            
            // Array operations
            PrismOpcode::GET_INDEX => {
                let array_var = self.fresh_type_variable(TypeVariableKind::Collection, None);
                let index_var = self.fresh_type_variable(TypeVariableKind::Integer, None);
                let element_var = self.fresh_type_variable(TypeVariableKind::Type, None);
                
                constraints.push(self.create_constraint(
                    *constraint_id,
                    location,
                    ConstraintType::ArrayAccess {
                        array: InferredType::Variable(array_var),
                        index: InferredType::Variable(index_var),
                        element: InferredType::Variable(element_var),
                    },
                    ConstraintOrigin::Instruction {
                        offset: location,
                        opcode: "GET_INDEX".to_string(),
                    },
                    ConstraintPriority::Normal,
                ));
                *constraint_id += 1;
            }
            
            // Control flow
            PrismOpcode::JUMP_IF_TRUE(_) | PrismOpcode::JUMP_IF_FALSE(_) => {
                let condition_var = self.fresh_type_variable(TypeVariableKind::Type, None);
                constraints.push(self.create_constraint(
                    *constraint_id,
                    location,
                    ConstraintType::Equality {
                        left: InferredType::Variable(condition_var),
                        right: InferredType::Boolean,
                    },
                    ConstraintOrigin::Instruction {
                        offset: location,
                        opcode: "CONDITIONAL_JUMP".to_string(),
                    },
                    ConstraintPriority::High,
                ));
                *constraint_id += 1;
            }
            
            _ => {
                // For other instructions, generate basic constraints
                // This would be expanded with specific logic for each opcode
            }
        }
        
        Ok(constraints)
    }

    /// Create a type constraint
    fn create_constraint(
        &self,
        id: u32,
        location: u32,
        constraint_type: ConstraintType,
        origin: ConstraintOrigin,
        priority: ConstraintPriority,
    ) -> TypeConstraint {
        TypeConstraint {
            constraint_id: id,
            location,
            constraint_type,
            origin,
            priority,
        }
    }

    /// Solve type constraints using unification algorithm
    fn solve_constraints(&mut self, constraints: &mut [TypeConstraint]) -> VMResult<TypeInference> {
        let mut inference = TypeInference::default();
        
        // Sort constraints by priority
        constraints.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        // Process each constraint
        for constraint in constraints.iter() {
            match self.solve_constraint(constraint) {
                Ok(_) => {
                    inference.statistics.constraints_solved += 1;
                }
                Err(e) => {
                    inference.type_errors.push(TypeError {
                        location: constraint.location,
                        error_type: TypeErrorKind::UnresolvedType { 
                            type_var: TypeVariable { id: 0, name: None, kind: TypeVariableKind::Type }
                        },
                        message: format!("Failed to solve constraint: {}", e),
                        severity: ErrorSeverity::Error,
                    });
                }
            }
        }
        
        // Extract final type bindings
        inference.type_bindings = self.solver_state.substitutions.clone();
        
        // Apply substitutions to get final inferred types
        self.apply_substitutions(&mut inference)?;
        
        Ok(inference)
    }

    /// Solve a single constraint
    fn solve_constraint(&mut self, constraint: &TypeConstraint) -> VMResult<()> {
        match &constraint.constraint_type {
            ConstraintType::Equality { left, right } => {
                self.unify(left.clone(), right.clone())?;
            }
            ConstraintType::Subtype { subtype, supertype } => {
                // Implement subtype constraint solving
                self.check_subtype(subtype, supertype)?;
            }
            ConstraintType::Arithmetic { op, left, right, result } => {
                // Solve arithmetic constraints
                self.solve_arithmetic_constraint(*op, left, right, result)?;
            }
            ConstraintType::FunctionCall { function, args, result } => {
                // Solve function call constraints
                self.solve_function_call_constraint(function, args, result)?;
            }
            ConstraintType::ArrayAccess { array, index, element } => {
                // Solve array access constraints
                self.solve_array_access_constraint(array, index, element)?;
            }
            _ => {
                // Handle other constraint types
            }
        }
        Ok(())
    }

    /// Unification algorithm for type constraint solving
    fn unify(&mut self, type1: InferredType, type2: InferredType) -> VMResult<()> {
        self.solver_state.unification_stack.push_back((type1, type2));
        
        while let Some((t1, t2)) = self.solver_state.unification_stack.pop_front() {
            if t1 == t2 {
                continue;
            }
            
            match (t1, t2) {
                (InferredType::Variable(var), ty) | (ty, InferredType::Variable(var)) => {
                    if self.occurs_check(&var, &ty)? {
                        return Err(PrismVMError::TypeInferenceError {
                            message: format!("Occurs check failed: {} occurs in {}", var, ty),
                        });
                    }
                    self.solver_state.substitutions.insert(var, ty);
                }
                (InferredType::Array { element: e1, size: s1 }, 
                 InferredType::Array { element: e2, size: s2 }) => {
                    if s1 != s2 {
                        return Err(PrismVMError::TypeInferenceError {
                            message: "Array size mismatch".to_string(),
                        });
                    }
                    self.solver_state.unification_stack.push_back((*e1, *e2));
                }
                (InferredType::Function { params: p1, return_type: r1 },
                 InferredType::Function { params: p2, return_type: r2 }) => {
                    if p1.len() != p2.len() {
                        return Err(PrismVMError::TypeInferenceError {
                            message: "Function arity mismatch".to_string(),
                        });
                    }
                    for (param1, param2) in p1.into_iter().zip(p2.into_iter()) {
                        self.solver_state.unification_stack.push_back((param1, param2));
                    }
                    self.solver_state.unification_stack.push_back((*r1, *r2));
                }
                _ => {
                    return Err(PrismVMError::TypeInferenceError {
                        message: format!("Cannot unify types: {} and {}", t1, t2),
                    });
                }
            }
        }
        
        Ok(())
    }

    /// Occurs check to prevent infinite types
    fn occurs_check(&mut self, var: &TypeVariable, ty: &InferredType) -> VMResult<bool> {
        if let Some(&result) = self.solver_state.occurs_cache.get(&(var.clone(), ty.clone())) {
            return Ok(result);
        }
        
        let result = match ty {
            InferredType::Variable(v) if v == var => true,
            InferredType::Variable(v) => {
                if let Some(substituted) = self.solver_state.substitutions.get(v) {
                    self.occurs_check(var, substituted)?
                } else {
                    false
                }
            }
            InferredType::Array { element, .. } => self.occurs_check(var, element)?,
            InferredType::Function { params, return_type } => {
                params.iter().any(|p| self.occurs_check(var, p).unwrap_or(false)) ||
                self.occurs_check(var, return_type)?
            }
            InferredType::Tuple { elements } => {
                elements.iter().any(|e| self.occurs_check(var, e).unwrap_or(false))
            }
            _ => false,
        };
        
        self.solver_state.occurs_cache.insert((var.clone(), ty.clone()), result);
        Ok(result)
    }

    /// Check subtype relationship
    fn check_subtype(&self, subtype: &InferredType, supertype: &InferredType) -> VMResult<()> {
        match (subtype, supertype) {
            (_, InferredType::Any) => Ok(()),
            (InferredType::Never, _) => Ok(()),
            (InferredType::Integer { bits: b1, signed: s1 }, 
             InferredType::Integer { bits: b2, signed: s2 }) => {
                if b1 <= b2 && s1 == s2 {
                    Ok(())
                } else {
                    Err(PrismVMError::TypeInferenceError {
                        message: format!("Integer subtype check failed: {} <: {}", subtype, supertype),
                    })
                }
            }
            _ => {
                if subtype == supertype {
                    Ok(())
                } else {
                    Err(PrismVMError::TypeInferenceError {
                        message: format!("Subtype check failed: {} <: {}", subtype, supertype),
                    })
                }
            }
        }
    }

    /// Solve arithmetic constraint
    fn solve_arithmetic_constraint(
        &mut self,
        op: ArithmeticOp,
        left: &InferredType,
        right: &InferredType,
        result: &InferredType,
    ) -> VMResult<()> {
        match op {
            ArithmeticOp::Add | ArithmeticOp::Sub | ArithmeticOp::Mul => {
                // For basic arithmetic, operands should be numeric and result should match
                let numeric_constraint = InferredType::Union {
                    types: vec![
                        InferredType::Integer { bits: 32, signed: true },
                        InferredType::Integer { bits: 64, signed: true },
                        InferredType::Float { bits: 64 },
                    ],
                };
                
                self.unify(left.clone(), numeric_constraint.clone())?;
                self.unify(right.clone(), numeric_constraint.clone())?;
                self.unify(result.clone(), numeric_constraint)?;
            }
            ArithmeticOp::Div => {
                // Division might promote integers to floats
                let numeric_constraint = InferredType::Union {
                    types: vec![
                        InferredType::Integer { bits: 32, signed: true },
                        InferredType::Float { bits: 64 },
                    ],
                };
                
                self.unify(left.clone(), numeric_constraint.clone())?;
                self.unify(right.clone(), numeric_constraint)?;
                self.unify(result.clone(), InferredType::Float { bits: 64 })?;
            }
            ArithmeticOp::BitAnd | ArithmeticOp::BitOr | ArithmeticOp::BitXor => {
                // Bitwise operations require integer types
                let integer_constraint = InferredType::Union {
                    types: vec![
                        InferredType::Integer { bits: 32, signed: true },
                        InferredType::Integer { bits: 64, signed: true },
                    ],
                };
                
                self.unify(left.clone(), integer_constraint.clone())?;
                self.unify(right.clone(), integer_constraint.clone())?;
                self.unify(result.clone(), integer_constraint)?;
            }
            _ => {
                // Handle other arithmetic operations
            }
        }
        Ok(())
    }

    /// Solve function call constraint
    fn solve_function_call_constraint(
        &mut self,
        function: &InferredType,
        args: &[InferredType],
        result: &InferredType,
    ) -> VMResult<()> {
        // Create function type constraint
        let function_type = InferredType::Function {
            params: args.to_vec(),
            return_type: Box::new(result.clone()),
        };
        
        self.unify(function.clone(), function_type)?;
        Ok(())
    }

    /// Solve array access constraint
    fn solve_array_access_constraint(
        &mut self,
        array: &InferredType,
        index: &InferredType,
        element: &InferredType,
    ) -> VMResult<()> {
        // Index should be integer
        self.unify(index.clone(), InferredType::Integer { bits: 32, signed: true })?;
        
        // Array should be array type with matching element type
        let array_type = InferredType::Array {
            element: Box::new(element.clone()),
            size: None,
        };
        
        self.unify(array.clone(), array_type)?;
        Ok(())
    }

    /// Apply substitutions to get final inferred types
    fn apply_substitutions(&mut self, inference: &mut TypeInference) -> VMResult<()> {
        // Apply substitutions to resolve type variables
        for (var, ty) in &self.solver_state.substitutions.clone() {
            let resolved_type = self.resolve_type(ty.clone())?;
            inference.inferred_types.insert(var.to_string(), resolved_type);
            inference.confidence_scores.insert(var.to_string(), 0.8);
        }
        
        Ok(())
    }

    /// Resolve a type by applying all substitutions
    fn resolve_type(&self, ty: InferredType) -> VMResult<InferredType> {
        match ty {
            InferredType::Variable(var) => {
                if let Some(substituted) = self.solver_state.substitutions.get(&var) {
                    self.resolve_type(substituted.clone())
                } else {
                    Ok(InferredType::Variable(var))
                }
            }
            InferredType::Array { element, size } => {
                Ok(InferredType::Array {
                    element: Box::new(self.resolve_type(*element)?),
                    size,
                })
            }
            InferredType::Function { params, return_type } => {
                let resolved_params = params.into_iter()
                    .map(|p| self.resolve_type(p))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(InferredType::Function {
                    params: resolved_params,
                    return_type: Box::new(self.resolve_type(*return_type)?),
                })
            }
            _ => Ok(ty),
        }
    }

    /// Analyze value ranges for integer types
    fn analyze_value_ranges(
        &mut self,
        function: &FunctionDefinition,
        type_inference: &TypeInference,
    ) -> VMResult<ValueRangeAnalysis> {
        let mut analysis = ValueRangeAnalysis::default();
        
        // Initialize ranges for parameters and locals
        for (i, _) in function.instructions.iter().enumerate().take(function.param_count as usize) {
            let param_name = format!("param_{}", i);
            // Parameters start with unknown ranges
            analysis.integer_ranges.insert(param_name, IntegerRange {
                min: i64::MIN,
                max: i64::MAX,
                is_exact: false,
                confidence: 0.3,
            });
        }
        
        // Analyze each instruction for value range propagation
        let mut current_ranges = analysis.integer_ranges.clone();
        
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let location = offset as u32;
            self.analyze_instruction_ranges(
                instruction, 
                location, 
                &mut current_ranges, 
                &mut analysis,
                type_inference
            )?;
        }
        
        // Detect potential overflow sites
        self.detect_overflow_sites(function, &current_ranges, &mut analysis)?;
        
        // Extract constants
        self.extract_constants(&current_ranges, &mut analysis);
        
        Ok(analysis)
    }

    /// Analyze ranges for a single instruction
    fn analyze_instruction_ranges(
        &mut self,
        instruction: &Instruction,
        location: u32,
        current_ranges: &mut HashMap<String, IntegerRange>,
        analysis: &mut ValueRangeAnalysis,
        type_inference: &TypeInference,
    ) -> VMResult<()> {
        match instruction.opcode {
            // Constant loading - creates exact ranges
            PrismOpcode::LOAD_CONST(idx) => {
                // Would need access to constant pool to get exact value
                // For now, create a placeholder for demonstration
                let stack_var = "stack_top".to_string();
                current_ranges.insert(stack_var.clone(), IntegerRange {
                    min: 42, // Would be actual constant value
                    max: 42,
                    is_exact: true,
                    confidence: 1.0,
                });
                analysis.constants.insert(stack_var, ConstantValue::Integer(42));
            }
            
            // Small integer constants
            PrismOpcode::LOAD_SMALL_INT(val) => {
                let stack_var = "stack_top".to_string();
                let int_val = val as i64;
                current_ranges.insert(stack_var.clone(), IntegerRange {
                    min: int_val,
                    max: int_val,
                    is_exact: true,
                    confidence: 1.0,
                });
                analysis.constants.insert(stack_var, ConstantValue::Integer(int_val));
            }
            
            // Arithmetic operations
            PrismOpcode::ADD => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.add_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                    
                    // Check for potential overflow
                    if self.could_overflow_add(&left_range, &right_range) {
                        analysis.overflow_sites.push(OverflowSite {
                            location,
                            operation: ArithmeticOp::Add,
                            operand_ranges: vec![left_range, right_range],
                            overflow_probability: self.calculate_overflow_probability(&left_range, &right_range, ArithmeticOp::Add),
                        });
                    }
                }
            }
            
            PrismOpcode::SUB => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.sub_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                    
                    if self.could_overflow_sub(&left_range, &right_range) {
                        analysis.overflow_sites.push(OverflowSite {
                            location,
                            operation: ArithmeticOp::Sub,
                            operand_ranges: vec![left_range, right_range],
                            overflow_probability: self.calculate_overflow_probability(&left_range, &right_range, ArithmeticOp::Sub),
                        });
                    }
                }
            }
            
            PrismOpcode::MUL => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.mul_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                    
                    if self.could_overflow_mul(&left_range, &right_range) {
                        analysis.overflow_sites.push(OverflowSite {
                            location,
                            operation: ArithmeticOp::Mul,
                            operand_ranges: vec![left_range, right_range],
                            overflow_probability: self.calculate_overflow_probability(&left_range, &right_range, ArithmeticOp::Mul),
                        });
                    }
                }
            }
            
            PrismOpcode::DIV => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    // Check for division by zero
                    if right_range.min <= 0 && right_range.max >= 0 {
                        analysis.overflow_sites.push(OverflowSite {
                            location,
                            operation: ArithmeticOp::Div,
                            operand_ranges: vec![left_range.clone(), right_range.clone()],
                            overflow_probability: if right_range.min == 0 && right_range.max == 0 { 1.0 } else { 0.5 },
                        });
                    }
                    
                    let result_range = self.div_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                }
            }
            
            // Local variable operations
            PrismOpcode::LOAD_LOCAL(slot) => {
                let local_name = format!("local_{}", slot);
                if let Some(range) = current_ranges.get(&local_name).cloned() {
                    current_ranges.insert("stack_top".to_string(), range);
                }
            }
            
            PrismOpcode::STORE_LOCAL(slot) => {
                let local_name = format!("local_{}", slot);
                if let Some(range) = current_ranges.get("stack_top").cloned() {
                    current_ranges.insert(local_name, range);
                }
            }
            
            // Comparison operations - produce boolean results but can refine ranges
            PrismOpcode::LT | PrismOpcode::LE | PrismOpcode::GT | PrismOpcode::GE | PrismOpcode::EQ | PrismOpcode::NE => {
                // Result is always boolean, but we could use this for range refinement in control flow
                current_ranges.insert("stack_top".to_string(), IntegerRange {
                    min: 0,
                    max: 1,
                    is_exact: false,
                    confidence: 1.0,
                });
            }
            
            // Bitwise operations
            PrismOpcode::BIT_AND => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.bitand_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                }
            }
            
            PrismOpcode::BIT_OR => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.bitor_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                }
            }
            
            // Shifts
            PrismOpcode::SHL => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.shl_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                    
                    // Left shift can easily overflow
                    if right_range.max > 31 || (left_range.max > 0 && right_range.max > 0) {
                        analysis.overflow_sites.push(OverflowSite {
                            location,
                            operation: ArithmeticOp::Shl,
                            operand_ranges: vec![left_range, right_range],
                            overflow_probability: 0.8,
                        });
                    }
                }
            }
            
            PrismOpcode::SHR => {
                if let (Some(left_range), Some(right_range)) = (
                    current_ranges.get("stack_left").cloned(),
                    current_ranges.get("stack_right").cloned(),
                ) {
                    let result_range = self.shr_ranges(&left_range, &right_range);
                    current_ranges.insert("stack_top".to_string(), result_range);
                }
            }
            
            _ => {
                // For other instructions, invalidate stack top range
                current_ranges.insert("stack_top".to_string(), IntegerRange {
                    min: i64::MIN,
                    max: i64::MAX,
                    is_exact: false,
                    confidence: 0.1,
                });
            }
        }
        
        // Update analysis with current ranges
        for (var, range) in current_ranges.iter() {
            analysis.integer_ranges.insert(var.clone(), range.clone());
        }
        
        Ok(())
    }

    /// Add two integer ranges
    fn add_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        let min = left.min.saturating_add(right.min);
        let max = left.max.saturating_add(right.max);
        let is_exact = left.is_exact && right.is_exact;
        let confidence = (left.confidence * right.confidence).min(0.9);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Subtract two integer ranges
    fn sub_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        let min = left.min.saturating_sub(right.max);
        let max = left.max.saturating_sub(right.min);
        let is_exact = left.is_exact && right.is_exact;
        let confidence = (left.confidence * right.confidence).min(0.9);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Multiply two integer ranges
    fn mul_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        let products = [
            left.min.saturating_mul(right.min),
            left.min.saturating_mul(right.max),
            left.max.saturating_mul(right.min),
            left.max.saturating_mul(right.max),
        ];
        
        let min = products.iter().min().copied().unwrap_or(i64::MIN);
        let max = products.iter().max().copied().unwrap_or(i64::MAX);
        let is_exact = left.is_exact && right.is_exact;
        let confidence = (left.confidence * right.confidence * 0.8).min(0.9);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Divide two integer ranges
    fn div_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        if right.min <= 0 && right.max >= 0 {
            // Division by zero possible, return full range
            return IntegerRange {
                min: i64::MIN,
                max: i64::MAX,
                is_exact: false,
                confidence: 0.1,
            };
        }
        
        let quotients = [
            if right.min != 0 { Some(left.min / right.min) } else { None },
            if right.max != 0 { Some(left.min / right.max) } else { None },
            if right.min != 0 { Some(left.max / right.min) } else { None },
            if right.max != 0 { Some(left.max / right.max) } else { None },
        ];
        
        let valid_quotients: Vec<i64> = quotients.into_iter().flatten().collect();
        
        if valid_quotients.is_empty() {
            return IntegerRange {
                min: i64::MIN,
                max: i64::MAX,
                is_exact: false,
                confidence: 0.1,
            };
        }
        
        let min = valid_quotients.iter().min().copied().unwrap_or(i64::MIN);
        let max = valid_quotients.iter().max().copied().unwrap_or(i64::MAX);
        let is_exact = left.is_exact && right.is_exact && valid_quotients.len() == 1;
        let confidence = (left.confidence * right.confidence * 0.7).min(0.9);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Bitwise AND of two ranges
    fn bitand_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        // Conservative approximation: result is between 0 and min(left.max, right.max)
        let min = 0.max(left.min.min(right.min));
        let max = left.max.min(right.max);
        let is_exact = left.is_exact && right.is_exact && left.min == left.max && right.min == right.max;
        let confidence = (left.confidence * right.confidence * 0.6).min(0.8);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Bitwise OR of two ranges
    fn bitor_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        // Conservative approximation: result is between max(left.min, right.min) and left.max | right.max
        let min = left.min.max(right.min);
        let max = if left.max >= 0 && right.max >= 0 {
            // Approximate upper bound for positive numbers
            let max_bits = (left.max.max(right.max) as f64).log2().ceil() as u32;
            ((1i64 << max_bits) - 1).min(i64::MAX)
        } else {
            i64::MAX
        };
        let is_exact = left.is_exact && right.is_exact && left.min == left.max && right.min == right.max;
        let confidence = (left.confidence * right.confidence * 0.6).min(0.8);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Left shift ranges
    fn shl_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        if right.min < 0 || right.max > 63 {
            // Invalid shift amount
            return IntegerRange {
                min: i64::MIN,
                max: i64::MAX,
                is_exact: false,
                confidence: 0.1,
            };
        }
        
        let min = if left.min >= 0 {
            left.min.saturating_shl(right.min as u32)
        } else {
            left.min.saturating_shl(right.max as u32)
        };
        
        let max = if left.max >= 0 {
            left.max.saturating_shl(right.max as u32)
        } else {
            left.max.saturating_shl(right.min as u32)
        };
        
        let is_exact = left.is_exact && right.is_exact;
        let confidence = (left.confidence * right.confidence * 0.7).min(0.8);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Right shift ranges
    fn shr_ranges(&self, left: &IntegerRange, right: &IntegerRange) -> IntegerRange {
        if right.min < 0 || right.max > 63 {
            // Invalid shift amount
            return IntegerRange {
                min: i64::MIN,
                max: i64::MAX,
                is_exact: false,
                confidence: 0.1,
            };
        }
        
        let min = left.min.saturating_shr(right.max as u32);
        let max = left.max.saturating_shr(right.min as u32);
        let is_exact = left.is_exact && right.is_exact;
        let confidence = (left.confidence * right.confidence * 0.8).min(0.9);
        
        IntegerRange { min, max, is_exact, confidence }
    }

    /// Check if addition could overflow
    fn could_overflow_add(&self, left: &IntegerRange, right: &IntegerRange) -> bool {
        // Check for i32 overflow (common case)
        let i32_max = i32::MAX as i64;
        let i32_min = i32::MIN as i64;
        
        left.max.saturating_add(right.max) > i32_max ||
        left.min.saturating_add(right.min) < i32_min
    }

    /// Check if subtraction could overflow
    fn could_overflow_sub(&self, left: &IntegerRange, right: &IntegerRange) -> bool {
        let i32_max = i32::MAX as i64;
        let i32_min = i32::MIN as i64;
        
        left.max.saturating_sub(right.min) > i32_max ||
        left.min.saturating_sub(right.max) < i32_min
    }

    /// Check if multiplication could overflow
    fn could_overflow_mul(&self, left: &IntegerRange, right: &IntegerRange) -> bool {
        let i32_max = i32::MAX as i64;
        let i32_min = i32::MIN as i64;
        
        let products = [
            left.min.saturating_mul(right.min),
            left.min.saturating_mul(right.max),
            left.max.saturating_mul(right.min),
            left.max.saturating_mul(right.max),
        ];
        
        products.iter().any(|&p| p > i32_max || p < i32_min)
    }

    /// Calculate overflow probability
    fn calculate_overflow_probability(&self, left: &IntegerRange, right: &IntegerRange, op: ArithmeticOp) -> f64 {
        match op {
            ArithmeticOp::Add => {
                let range_size = (left.max - left.min + 1) * (right.max - right.min + 1);
                if range_size <= 0 { return 0.0; }
                
                // Simple heuristic: larger ranges and values closer to limits increase probability
                let left_risk = (left.max.abs() as f64) / (i32::MAX as f64);
                let right_risk = (right.max.abs() as f64) / (i32::MAX as f64);
                (left_risk * right_risk).min(1.0)
            }
            ArithmeticOp::Mul => {
                // Multiplication overflow is more likely with larger values
                let left_risk = (left.max.abs() as f64).sqrt() / ((i32::MAX as f64).sqrt());
                let right_risk = (right.max.abs() as f64).sqrt() / ((i32::MAX as f64).sqrt());
                (left_risk * right_risk * 2.0).min(1.0)
            }
            ArithmeticOp::Sub => {
                // Subtraction overflow when subtracting large negative from large positive
                if left.max > 0 && right.min < 0 {
                    let risk = (left.max as f64 - right.min as f64) / (i32::MAX as f64 * 2.0);
                    risk.min(1.0)
                } else {
                    0.1
                }
            }
            ArithmeticOp::Div => {
                // Division by zero risk
                if right.min <= 0 && right.max >= 0 {
                    let zero_range = if right.min == 0 && right.max == 0 { 1.0 } else {
                        1.0 / (right.max - right.min + 1) as f64
                    };
                    zero_range
                } else {
                    0.0
                }
            }
            _ => 0.1, // Default low probability
        }
    }

    /// Detect overflow sites in the function
    fn detect_overflow_sites(
        &mut self,
        function: &FunctionDefinition,
        ranges: &HashMap<String, IntegerRange>,
        analysis: &mut ValueRangeAnalysis,
    ) -> VMResult<()> {
        // Additional overflow detection logic
        // This could include more sophisticated analysis like loop iteration counts
        
        for (offset, instruction) in function.instructions.iter().enumerate() {
            match instruction.opcode {
                // Array indexing bounds checks
                PrismOpcode::GET_INDEX | PrismOpcode::SET_INDEX => {
                    if let Some(index_range) = ranges.get("stack_index") {
                        if index_range.min < 0 {
                            analysis.overflow_sites.push(OverflowSite {
                                location: offset as u32,
                                operation: ArithmeticOp::Add, // Representing bounds check
                                operand_ranges: vec![index_range.clone()],
                                overflow_probability: if index_range.max < 0 { 1.0 } else { 0.5 },
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        
        Ok(())
    }

    /// Extract constant values from ranges
    fn extract_constants(&self, ranges: &HashMap<String, IntegerRange>, analysis: &mut ValueRangeAnalysis) {
        for (var_name, range) in ranges {
            if range.is_exact && range.min == range.max {
                analysis.constants.insert(var_name.clone(), ConstantValue::Integer(range.min));
            }
        }
    }

    /// Analyze type flow through control flow
    fn analyze_type_flow(
        &mut self,
        function: &FunctionDefinition,
        type_inference: &TypeInference,
    ) -> VMResult<TypeFlowAnalysis> {
        let mut analysis = TypeFlowAnalysis::default();
        
        // We would need the CFG here, but for now we'll create a simplified version
        // In a full implementation, this would integrate with the control flow analyzer
        
        // Initialize entry state
        let mut entry_state = TypeState::default();
        
        // Add parameter types to entry state
        for (i, _) in function.instructions.iter().enumerate().take(function.param_count as usize) {
            let param_name = format!("param_{}", i);
            if let Some(inferred_type) = type_inference.inferred_types.get(&param_name) {
                entry_state.variable_types.insert(param_name.clone(), inferred_type.clone());
                entry_state.type_assumptions.insert(param_name, TypeAssumption {
                    assumed_type: inferred_type.clone(),
                    confidence: 0.8,
                    source: AssumptionSource::FunctionParameter,
                });
            }
        }
        
        // Add local variable types
        for i in 0..function.local_count {
            let local_name = format!("local_{}", i);
            entry_state.variable_types.insert(local_name, InferredType::Any);
        }
        
        analysis.block_entry_types.insert(0, entry_state.clone());
        
        // Simulate control flow analysis by processing instructions sequentially
        let mut current_state = entry_state;
        let mut current_block = 0u32;
        
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let location = offset as u32;
            
            // Update type state based on instruction
            self.update_type_state_for_instruction(
                instruction,
                location,
                &mut current_state,
                &mut analysis,
            )?;
            
            // Handle control flow instructions
            match instruction.opcode {
                PrismOpcode::JUMP(_) => {
                    // Unconditional jump - save exit state
                    analysis.block_exit_types.insert(current_block, current_state.clone());
                    current_block += 1;
                    analysis.block_entry_types.insert(current_block, current_state.clone());
                }
                PrismOpcode::JUMP_IF_TRUE(_) | PrismOpcode::JUMP_IF_FALSE(_) => {
                    // Conditional branch - create refined states
                    let (true_state, false_state) = self.refine_state_for_condition(
                        &current_state,
                        &instruction.opcode,
                        location,
                    )?;
                    
                    // Save both branch states
                    analysis.block_exit_types.insert(current_block, current_state.clone());
                    current_block += 1;
                    analysis.block_entry_types.insert(current_block, true_state);
                    current_block += 1;
                    analysis.block_entry_types.insert(current_block, false_state);
                }
                PrismOpcode::RETURN | PrismOpcode::RETURN_VALUE => {
                    // Function exit
                    analysis.block_exit_types.insert(current_block, current_state.clone());
                }
                _ => {
                    // Regular instruction - continue in same block
                }
            }
        }
        
        // Save final state
        if !analysis.block_exit_types.contains_key(&current_block) {
            analysis.block_exit_types.insert(current_block, current_state);
        }
        
        Ok(analysis)
    }

    /// Update type state for a single instruction
    fn update_type_state_for_instruction(
        &mut self,
        instruction: &Instruction,
        location: u32,
        state: &mut TypeState,
        analysis: &mut TypeFlowAnalysis,
    ) -> VMResult<()> {
        match instruction.opcode {
            // Type tests create refinements
            PrismOpcode::IS_NULL => {
                // Pop operand, push boolean result
                if let Some(operand_type) = state.stack_types.pop() {
                    // Create type refinement if we know the operand
                    if let Some(var_name) = self.find_variable_for_stack_top(state) {
                        let refinement = TypeRefinement {
                            variable: var_name.clone(),
                            original_type: operand_type.clone(),
                            refined_type: if operand_type == InferredType::Null {
                                InferredType::Null
                            } else {
                                // Type is not null in false branch
                                self.remove_null_from_type(operand_type)
                            },
                            condition: RefinementCondition::NullCheck,
                        };
                        analysis.type_refinements.insert(location, refinement);
                    }
                }
                state.stack_types.push(InferredType::Boolean);
            }
            
            PrismOpcode::IS_NUMBER => {
                if let Some(operand_type) = state.stack_types.pop() {
                    if let Some(var_name) = self.find_variable_for_stack_top(state) {
                        let numeric_type = InferredType::Union {
                            types: vec![
                                InferredType::Integer { bits: 32, signed: true },
                                InferredType::Float { bits: 64 },
                            ],
                        };
                        let refinement = TypeRefinement {
                            variable: var_name,
                            original_type: operand_type,
                            refined_type: numeric_type.clone(),
                            condition: RefinementCondition::TypeTest { test_type: numeric_type },
                        };
                        analysis.type_refinements.insert(location, refinement);
                    }
                }
                state.stack_types.push(InferredType::Boolean);
            }
            
            // Arithmetic operations
            PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL | PrismOpcode::DIV => {
                if state.stack_types.len() >= 2 {
                    let right = state.stack_types.pop().unwrap();
                    let left = state.stack_types.pop().unwrap();
                    
                    // Result type depends on operand types
                    let result_type = self.infer_arithmetic_result_type(&left, &right, &instruction.opcode);
                    state.stack_types.push(result_type);
                }
            }
            
            // Comparison operations
            PrismOpcode::LT | PrismOpcode::LE | PrismOpcode::GT | PrismOpcode::GE | 
            PrismOpcode::EQ | PrismOpcode::NE => {
                if state.stack_types.len() >= 2 {
                    state.stack_types.pop(); // right operand
                    state.stack_types.pop(); // left operand
                    state.stack_types.push(InferredType::Boolean);
                }
            }
            
            // Local variable operations
            PrismOpcode::LOAD_LOCAL(slot) => {
                let local_name = format!("local_{}", slot);
                if let Some(local_type) = state.variable_types.get(&local_name).cloned() {
                    state.stack_types.push(local_type);
                } else {
                    state.stack_types.push(InferredType::Any);
                }
            }
            
            PrismOpcode::STORE_LOCAL(slot) => {
                let local_name = format!("local_{}", slot);
                if let Some(value_type) = state.stack_types.pop() {
                    // Update variable type
                    state.variable_types.insert(local_name.clone(), value_type.clone());
                    
                    // Update type assumption
                    state.type_assumptions.insert(local_name, TypeAssumption {
                        assumed_type: value_type,
                        confidence: 0.9,
                        source: AssumptionSource::PreviousAssignment,
                    });
                }
            }
            
            // Constants
            PrismOpcode::LOAD_CONST(_) => {
                // Would need constant pool access for exact type
                state.stack_types.push(InferredType::Any);
            }
            
            PrismOpcode::LOAD_SMALL_INT(_) => {
                state.stack_types.push(InferredType::Integer { bits: 32, signed: true });
            }
            
            PrismOpcode::LOAD_TRUE | PrismOpcode::LOAD_FALSE => {
                state.stack_types.push(InferredType::Boolean);
            }
            
            PrismOpcode::LOAD_NULL => {
                state.stack_types.push(InferredType::Null);
            }
            
            // Stack operations
            PrismOpcode::DUP => {
                if let Some(top_type) = state.stack_types.last().cloned() {
                    state.stack_types.push(top_type);
                }
            }
            
            PrismOpcode::POP => {
                state.stack_types.pop();
            }
            
            PrismOpcode::SWAP => {
                if state.stack_types.len() >= 2 {
                    let len = state.stack_types.len();
                    state.stack_types.swap(len - 1, len - 2);
                }
            }
            
            _ => {
                // For other instructions, make conservative assumptions
            }
        }
        
        Ok(())
    }

    /// Refine type state based on conditional branch
    fn refine_state_for_condition(
        &mut self,
        base_state: &TypeState,
        opcode: &PrismOpcode,
        location: u32,
    ) -> VMResult<(TypeState, TypeState)> {
        let mut true_state = base_state.clone();
        let mut false_state = base_state.clone();
        
        // Get the condition from stack top
        if let Some(condition_type) = base_state.stack_types.last() {
            match opcode {
                PrismOpcode::JUMP_IF_TRUE(_) => {
                    // In true branch, condition is truthy
                    if let Some(var_name) = self.find_variable_for_stack_top(&true_state) {
                        // Refine type based on truthiness
                        let refined_type = self.refine_type_for_truthiness(condition_type, true);
                        true_state.variable_types.insert(var_name.clone(), refined_type.clone());
                        true_state.type_assumptions.insert(var_name, TypeAssumption {
                            assumed_type: refined_type,
                            confidence: 0.8,
                            source: AssumptionSource::ControlFlow,
                        });
                    }
                    
                    // In false branch, condition is falsy
                    if let Some(var_name) = self.find_variable_for_stack_top(&false_state) {
                        let refined_type = self.refine_type_for_truthiness(condition_type, false);
                        false_state.variable_types.insert(var_name.clone(), refined_type.clone());
                        false_state.type_assumptions.insert(var_name, TypeAssumption {
                            assumed_type: refined_type,
                            confidence: 0.8,
                            source: AssumptionSource::ControlFlow,
                        });
                    }
                }
                PrismOpcode::JUMP_IF_FALSE(_) => {
                    // Opposite of JUMP_IF_TRUE
                    if let Some(var_name) = self.find_variable_for_stack_top(&true_state) {
                        let refined_type = self.refine_type_for_truthiness(condition_type, false);
                        true_state.variable_types.insert(var_name.clone(), refined_type.clone());
                        true_state.type_assumptions.insert(var_name, TypeAssumption {
                            assumed_type: refined_type,
                            confidence: 0.8,
                            source: AssumptionSource::ControlFlow,
                        });
                    }
                    
                    if let Some(var_name) = self.find_variable_for_stack_top(&false_state) {
                        let refined_type = self.refine_type_for_truthiness(condition_type, true);
                        false_state.variable_types.insert(var_name.clone(), refined_type.clone());
                        false_state.type_assumptions.insert(var_name, TypeAssumption {
                            assumed_type: refined_type,
                            confidence: 0.8,
                            source: AssumptionSource::ControlFlow,
                        });
                    }
                }
                _ => {}
            }
        }
        
        Ok((true_state, false_state))
    }

    /// Find variable name corresponding to stack top (simplified)
    fn find_variable_for_stack_top(&self, _state: &TypeState) -> Option<String> {
        // This is a simplified implementation
        // In reality, we'd need to track which variables correspond to stack positions
        Some("stack_top".to_string())
    }

    /// Remove null from a union type
    fn remove_null_from_type(&self, ty: InferredType) -> InferredType {
        match ty {
            InferredType::Union { types } => {
                let non_null_types: Vec<_> = types.into_iter()
                    .filter(|t| *t != InferredType::Null)
                    .collect();
                
                if non_null_types.is_empty() {
                    InferredType::Never
                } else if non_null_types.len() == 1 {
                    non_null_types.into_iter().next().unwrap()
                } else {
                    InferredType::Union { types: non_null_types }
                }
            }
            InferredType::Null => InferredType::Never,
            other => other,
        }
    }

    /// Refine type based on truthiness
    fn refine_type_for_truthiness(&self, ty: &InferredType, is_truthy: bool) -> InferredType {
        match (ty, is_truthy) {
            (InferredType::Boolean, true) => InferredType::Boolean, // Could be more specific
            (InferredType::Boolean, false) => InferredType::Boolean,
            (InferredType::Integer { bits, signed }, true) => {
                // Non-zero integer
                InferredType::Integer { bits: *bits, signed: *signed }
            }
            (InferredType::Integer { bits, signed }, false) => {
                // Zero integer - could create exact range
                InferredType::Integer { bits: *bits, signed: *signed }
            }
            (InferredType::Union { types }, is_truthy) => {
                // Refine each type in union
                let refined_types: Vec<_> = types.iter()
                    .map(|t| self.refine_type_for_truthiness(t, is_truthy))
                    .filter(|t| *t != InferredType::Never)
                    .collect();
                
                if refined_types.is_empty() {
                    InferredType::Never
                } else if refined_types.len() == 1 {
                    refined_types.into_iter().next().unwrap()
                } else {
                    InferredType::Union { types: refined_types }
                }
            }
            (other, _) => other.clone(),
        }
    }

    /// Infer result type of arithmetic operation
    fn infer_arithmetic_result_type(&self, left: &InferredType, right: &InferredType, opcode: &PrismOpcode) -> InferredType {
        match (left, right, opcode) {
            // Integer + Integer = Integer
            (InferredType::Integer { bits: b1, signed: s1 }, 
             InferredType::Integer { bits: b2, signed: s2 }, 
             PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL) => {
                // Result uses larger of the two bit widths
                let result_bits = (*b1).max(*b2);
                let result_signed = *s1 || *s2; // Signed if either operand is signed
                InferredType::Integer { bits: result_bits, signed: result_signed }
            }
            
            // Division always produces float (in many languages)
            (InferredType::Integer { .. }, InferredType::Integer { .. }, PrismOpcode::DIV) => {
                InferredType::Float { bits: 64 }
            }
            
            // Float operations
            (InferredType::Float { bits: b1 }, InferredType::Float { bits: b2 }, _) => {
                InferredType::Float { bits: (*b1).max(*b2) }
            }
            
            // Mixed integer/float
            (InferredType::Integer { .. }, InferredType::Float { bits }, _) |
            (InferredType::Float { bits }, InferredType::Integer { .. }, _) => {
                InferredType::Float { bits: *bits }
            }
            
            // Union types - try to narrow down
            (InferredType::Union { types: t1 }, InferredType::Union { types: t2 }, op) => {
                let mut result_types = Vec::new();
                for left_ty in t1 {
                    for right_ty in t2 {
                        let result = self.infer_arithmetic_result_type(left_ty, right_ty, op);
                        if !result_types.contains(&result) {
                            result_types.push(result);
                        }
                    }
                }
                
                if result_types.len() == 1 {
                    result_types.into_iter().next().unwrap()
                } else {
                    InferredType::Union { types: result_types }
                }
            }
            
            // Default case
            _ => InferredType::Any,
        }
    }

    /// Detect specialization opportunities
    fn detect_specialization_opportunities(
        &mut self,
        function: &FunctionDefinition,
        type_inference: &TypeInference,
        value_ranges: &ValueRangeAnalysis,
    ) -> VMResult<Vec<SpecializationOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Analyze each instruction for specialization opportunities
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let location = offset as u32;
            
            // Check for different types of specialization opportunities
            self.detect_arithmetic_specialization(
                instruction, location, type_inference, value_ranges, &mut opportunities
            )?;
            
            self.detect_call_specialization(
                instruction, location, type_inference, &mut opportunities
            )?;
            
            self.detect_array_specialization(
                instruction, location, type_inference, &mut opportunities
            )?;
            
            self.detect_field_specialization(
                instruction, location, type_inference, &mut opportunities
            )?;
        }
        
        // Sort opportunities by potential benefit
        opportunities.sort_by(|a, b| b.potential_benefit.partial_cmp(&a.potential_benefit).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(opportunities)
    }

    /// Detect arithmetic specialization opportunities
    fn detect_arithmetic_specialization(
        &mut self,
        instruction: &Instruction,
        location: u32,
        type_inference: &TypeInference,
        value_ranges: &ValueRangeAnalysis,
        opportunities: &mut Vec<SpecializationOpportunity>,
    ) -> VMResult<()> {
        match instruction.opcode {
            PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL | PrismOpcode::DIV => {
                // Check if operands have specific integer ranges that allow optimization
                if let (Some(left_range), Some(right_range)) = (
                    value_ranges.integer_ranges.get("stack_left"),
                    value_ranges.integer_ranges.get("stack_right"),
                ) {
                    // Small integer optimization
                    if self.is_small_integer_range(left_range) && self.is_small_integer_range(right_range) {
                        opportunities.push(SpecializationOpportunity {
                            location,
                            opportunity_type: SpecializationType::IntegerSpecialization {
                                variable: "arithmetic_operands".to_string(),
                                range: IntegerRange {
                                    min: left_range.min.min(right_range.min),
                                    max: left_range.max.max(right_range.max),
                                    is_exact: left_range.is_exact && right_range.is_exact,
                                    confidence: (left_range.confidence * right_range.confidence).min(0.9),
                                },
                            },
                            potential_benefit: self.calculate_integer_specialization_benefit(left_range, right_range),
                            required_conditions: vec![
                                "Operands must be small integers".to_string(),
                                "No overflow possible".to_string(),
                            ],
                        });
                    }
                    
                    // Power-of-two optimization for multiplication/division
                    if matches!(instruction.opcode, PrismOpcode::MUL | PrismOpcode::DIV) {
                        if self.is_power_of_two_range(right_range) {
                            opportunities.push(SpecializationOpportunity {
                                location,
                                opportunity_type: SpecializationType::IntegerSpecialization {
                                    variable: "power_of_two_operand".to_string(),
                                    range: right_range.clone(),
                                },
                                potential_benefit: 0.8, // High benefit for shift operations
                                required_conditions: vec![
                                    "Right operand is power of two".to_string(),
                                    "Can replace with shift operation".to_string(),
                                ],
                            });
                        }
                    }
                }
                
                // Constant folding opportunities
                if let Some(constant_value) = value_ranges.constants.get("stack_left") {
                    if let ConstantValue::Integer(val) = constant_value {
                        match instruction.opcode {
                            PrismOpcode::ADD if *val == 0 => {
                                opportunities.push(SpecializationOpportunity {
                                    location,
                                    opportunity_type: SpecializationType::IntegerSpecialization {
                                        variable: "identity_add".to_string(),
                                        range: IntegerRange { min: 0, max: 0, is_exact: true, confidence: 1.0 },
                                    },
                                    potential_benefit: 0.9, // Very high benefit - eliminate operation
                                    required_conditions: vec!["Left operand is zero".to_string()],
                                });
                            }
                            PrismOpcode::MUL if *val == 1 => {
                                opportunities.push(SpecializationOpportunity {
                                    location,
                                    opportunity_type: SpecializationType::IntegerSpecialization {
                                        variable: "identity_mul".to_string(),
                                        range: IntegerRange { min: 1, max: 1, is_exact: true, confidence: 1.0 },
                                    },
                                    potential_benefit: 0.9,
                                    required_conditions: vec!["Left operand is one".to_string()],
                                });
                            }
                            PrismOpcode::MUL if *val == 0 => {
                                opportunities.push(SpecializationOpportunity {
                                    location,
                                    opportunity_type: SpecializationType::IntegerSpecialization {
                                        variable: "zero_mul".to_string(),
                                        range: IntegerRange { min: 0, max: 0, is_exact: true, confidence: 1.0 },
                                    },
                                    potential_benefit: 0.95, // Replace with constant zero
                                    required_conditions: vec!["Left operand is zero".to_string()],
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }

    /// Detect function call specialization opportunities
    fn detect_call_specialization(
        &mut self,
        instruction: &Instruction,
        location: u32,
        type_inference: &TypeInference,
        opportunities: &mut Vec<SpecializationOpportunity>,
    ) -> VMResult<()> {
        match instruction.opcode {
            PrismOpcode::CALL(argc) | PrismOpcode::TAIL_CALL(argc) => {
                // Check if we can determine the exact function being called
                if let Some(function_type) = type_inference.inferred_types.get("stack_function") {
                    match function_type {
                        InferredType::Function { params, return_type } => {
                            // Check if all parameter types are concrete (not polymorphic)
                            let all_concrete = params.iter().all(|p| self.is_concrete_type(p));
                            
                            if all_concrete {
                                let specialized_types = params.clone();
                                opportunities.push(SpecializationOpportunity {
                                    location,
                                    opportunity_type: SpecializationType::MonomorphicCall {
                                        function_id: 0, // Would need actual function resolution
                                        specialized_types,
                                    },
                                    potential_benefit: self.calculate_monomorphic_call_benefit(params, return_type),
                                    required_conditions: vec![
                                        "Function signature is fully concrete".to_string(),
                                        "No polymorphic parameters".to_string(),
                                    ],
                                });
                            }
                        }
                        InferredType::Union { types } => {
                            // Multiple possible functions - check if we can specialize for common cases
                            let function_types: Vec<_> = types.iter()
                                .filter_map(|t| match t {
                                    InferredType::Function { params, return_type } => Some((params, return_type)),
                                    _ => None,
                                })
                                .collect();
                            
                            if function_types.len() <= 3 { // Reasonable number for specialization
                                for (params, return_type) in function_types {
                                    opportunities.push(SpecializationOpportunity {
                                        location,
                                        opportunity_type: SpecializationType::MonomorphicCall {
                                            function_id: 0,
                                            specialized_types: params.clone(),
                                        },
                                        potential_benefit: self.calculate_monomorphic_call_benefit(params, return_type) * 0.7, // Lower benefit due to multiple versions
                                        required_conditions: vec![
                                            format!("One of {} possible function types", function_types.len()),
                                        ],
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }

    /// Detect array specialization opportunities
    fn detect_array_specialization(
        &mut self,
        instruction: &Instruction,
        location: u32,
        type_inference: &TypeInference,
        opportunities: &mut Vec<SpecializationOpportunity>,
    ) -> VMResult<()> {
        match instruction.opcode {
            PrismOpcode::GET_INDEX | PrismOpcode::SET_INDEX => {
                // Check if we can specialize based on array element type
                if let Some(array_type) = type_inference.inferred_types.get("stack_array") {
                    match array_type {
                        InferredType::Array { element, size } => {
                            if self.is_concrete_type(element) {
                                opportunities.push(SpecializationOpportunity {
                                    location,
                                    opportunity_type: SpecializationType::ArraySpecialization {
                                        array_type: array_type.clone(),
                                        element_type: (**element).clone(),
                                    },
                                    potential_benefit: self.calculate_array_specialization_benefit(element, size),
                                    required_conditions: vec![
                                        "Array element type is concrete".to_string(),
                                        "Can use typed array operations".to_string(),
                                    ],
                                });
                            }
                        }
                        _ => {}
                    }
                }
                
                // Check for bounds checking elimination
                if let Some(index_type) = type_inference.inferred_types.get("stack_index") {
                    match index_type {
                        InferredType::Integer { .. } => {
                            // Could eliminate bounds check if we know array size and index range
                            opportunities.push(SpecializationOpportunity {
                                location,
                                opportunity_type: SpecializationType::IntegerSpecialization {
                                    variable: "array_index".to_string(),
                                    range: IntegerRange {
                                        min: 0,
                                        max: i32::MAX as i64,
                                        is_exact: false,
                                        confidence: 0.6,
                                    },
                                },
                                potential_benefit: 0.4, // Moderate benefit from bounds check elimination
                                required_conditions: vec![
                                    "Index is non-negative integer".to_string(),
                                    "Array bounds are known".to_string(),
                                ],
                            });
                        }
                        _ => {}
                    }
                }
            }
            
            PrismOpcode::NEW_ARRAY(size) => {
                // Specialized array creation for known sizes
                if *size <= 16 { // Small arrays
                    opportunities.push(SpecializationOpportunity {
                        location,
                        opportunity_type: SpecializationType::ArraySpecialization {
                            array_type: InferredType::Array {
                                element: Box::new(InferredType::Any),
                                size: Some(*size as u64),
                            },
                            element_type: InferredType::Any,
                        },
                        potential_benefit: 0.6,
                        required_conditions: vec![
                            format!("Small array of size {}", size),
                            "Can use stack allocation".to_string(),
                        ],
                    });
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }

    /// Detect field access specialization opportunities
    fn detect_field_specialization(
        &mut self,
        instruction: &Instruction,
        location: u32,
        type_inference: &TypeInference,
        opportunities: &mut Vec<SpecializationOpportunity>,
    ) -> VMResult<()> {
        match instruction.opcode {
            PrismOpcode::GET_FIELD(field_id) | PrismOpcode::SET_FIELD(field_id) => {
                if let Some(object_type) = type_inference.inferred_types.get("stack_object") {
                    match object_type {
                        InferredType::Object { name, fields } => {
                            // Check if field access can be optimized
                            let field_name = format!("field_{}", field_id); // Simplified
                            
                            if let Some(field_type) = fields.get(&field_name) {
                                if self.is_concrete_type(field_type) {
                                    opportunities.push(SpecializationOpportunity {
                                        location,
                                        opportunity_type: SpecializationType::FieldSpecialization {
                                            object_type: object_type.clone(),
                                            field: field_name,
                                        },
                                        potential_benefit: self.calculate_field_specialization_benefit(field_type),
                                        required_conditions: vec![
                                            "Object type is concrete".to_string(),
                                            "Field type is known".to_string(),
                                            "Can use direct field access".to_string(),
                                        ],
                                    });
                                }
                            }
                        }
                        InferredType::Union { types } => {
                            // Multiple possible object types
                            let object_types: Vec<_> = types.iter()
                                .filter_map(|t| match t {
                                    InferredType::Object { name, fields } => Some((name, fields)),
                                    _ => None,
                                })
                                .collect();
                            
                            if object_types.len() <= 4 { // Reasonable for inline caching
                                for (name, fields) in object_types {
                                    let field_name = format!("field_{}", field_id);
                                    if let Some(field_type) = fields.get(&field_name) {
                                        opportunities.push(SpecializationOpportunity {
                                            location,
                                            opportunity_type: SpecializationType::FieldSpecialization {
                                                object_type: InferredType::Object {
                                                    name: name.clone(),
                                                    fields: fields.clone(),
                                                },
                                                field: field_name,
                                            },
                                            potential_benefit: self.calculate_field_specialization_benefit(field_type) * 0.6,
                                            required_conditions: vec![
                                                format!("Object type is {}", name),
                                                "Inline cache can handle polymorphism".to_string(),
                                            ],
                                        });
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }

    /// Check if a type is concrete (not polymorphic)
    fn is_concrete_type(&self, ty: &InferredType) -> bool {
        match ty {
            InferredType::Variable(_) | InferredType::Any => false,
            InferredType::Union { types } => types.len() == 1 && self.is_concrete_type(&types[0]),
            InferredType::Array { element, .. } => self.is_concrete_type(element),
            InferredType::Function { params, return_type } => {
                params.iter().all(|p| self.is_concrete_type(p)) && self.is_concrete_type(return_type)
            }
            InferredType::Object { fields, .. } => {
                fields.values().all(|f| self.is_concrete_type(f))
            }
            InferredType::Tuple { elements } => {
                elements.iter().all(|e| self.is_concrete_type(e))
            }
            _ => true, // Primitive types are concrete
        }
    }

    /// Check if integer range is small enough for specialized operations
    fn is_small_integer_range(&self, range: &IntegerRange) -> bool {
        let range_size = range.max - range.min + 1;
        range_size <= 256 && range.min >= -128 && range.max <= 127
    }

    /// Check if range represents power-of-two values
    fn is_power_of_two_range(&self, range: &IntegerRange) -> bool {
        if !range.is_exact || range.min != range.max {
            return false;
        }
        
        let val = range.min;
        val > 0 && (val & (val - 1)) == 0
    }

    /// Calculate benefit of integer specialization
    fn calculate_integer_specialization_benefit(&self, left: &IntegerRange, right: &IntegerRange) -> f64 {
        let mut benefit = 0.5; // Base benefit
        
        // Higher benefit for exact ranges
        if left.is_exact && right.is_exact {
            benefit += 0.3;
        }
        
        // Higher benefit for small ranges
        if self.is_small_integer_range(left) && self.is_small_integer_range(right) {
            benefit += 0.2;
        }
        
        // Consider confidence
        benefit *= (left.confidence * right.confidence).min(1.0);
        
        benefit.min(1.0)
    }

    /// Calculate benefit of monomorphic call specialization
    fn calculate_monomorphic_call_benefit(&self, params: &[InferredType], return_type: &InferredType) -> f64 {
        let mut benefit = 0.6; // Base benefit for eliminating dynamic dispatch
        
        // Higher benefit for more concrete types
        let concrete_ratio = params.iter()
            .chain(std::iter::once(return_type))
            .map(|t| if self.is_concrete_type(t) { 1.0 } else { 0.0 })
            .sum::<f64>() / (params.len() + 1) as f64;
        
        benefit += concrete_ratio * 0.3;
        
        // Higher benefit for functions with many parameters (more type checks eliminated)
        if params.len() > 2 {
            benefit += 0.1;
        }
        
        benefit.min(1.0)
    }

    /// Calculate benefit of array specialization
    fn calculate_array_specialization_benefit(&self, element_type: &InferredType, size: &Option<u64>) -> f64 {
        let mut benefit = 0.4; // Base benefit
        
        // Higher benefit for concrete element types
        if self.is_concrete_type(element_type) {
            benefit += 0.3;
        }
        
        // Higher benefit for known-size arrays
        if let Some(s) = size {
            if *s <= 1024 { // Reasonable size for specialization
                benefit += 0.2;
            }
        }
        
        // Higher benefit for primitive element types
        match element_type {
            InferredType::Integer { .. } | InferredType::Float { .. } | InferredType::Boolean => {
                benefit += 0.2;
            }
            _ => {}
        }
        
        benefit.min(1.0)
    }

    /// Calculate benefit of field specialization
    fn calculate_field_specialization_benefit(&self, field_type: &InferredType) -> f64 {
        let mut benefit = 0.3; // Base benefit for eliminating field lookup
        
        // Higher benefit for concrete field types
        if self.is_concrete_type(field_type) {
            benefit += 0.4;
        }
        
        // Higher benefit for primitive field types (can be stored inline)
        match field_type {
            InferredType::Integer { .. } | InferredType::Float { .. } | InferredType::Boolean => {
                benefit += 0.2;
            }
            _ => {}
        }
        
        benefit.min(1.0)
    }

    /// Identify polymorphic call sites
    fn identify_polymorphic_sites(
        &mut self,
        function: &FunctionDefinition,
        type_inference: &TypeInference,
    ) -> VMResult<Vec<PolymorphicSite>> {
        let mut sites = Vec::new();
        
        // Analyze each instruction for polymorphic behavior
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let location = offset as u32;
            
            match instruction.opcode {
                // Function calls are prime candidates for polymorphism
                PrismOpcode::CALL(_) | PrismOpcode::TAIL_CALL(_) | PrismOpcode::CALL_DYNAMIC(_) => {
                    if let Some(site) = self.analyze_polymorphic_call_site(location, instruction, type_inference)? {
                        sites.push(site);
                    }
                }
                
                // Field access can be polymorphic based on object type
                PrismOpcode::GET_FIELD(_) | PrismOpcode::SET_FIELD(_) | 
                PrismOpcode::GET_FIELD_HASH(_) | PrismOpcode::SET_FIELD_HASH(_) => {
                    if let Some(site) = self.analyze_polymorphic_field_access(location, instruction, type_inference)? {
                        sites.push(site);
                    }
                }
                
                // Method calls (if we had them) would also be polymorphic
                PrismOpcode::GET_METHOD(_) => {
                    if let Some(site) = self.analyze_polymorphic_method_access(location, instruction, type_inference)? {
                        sites.push(site);
                    }
                }
                
                // Array access can benefit from type-specific optimizations
                PrismOpcode::GET_INDEX | PrismOpcode::SET_INDEX => {
                    if let Some(site) = self.analyze_polymorphic_array_access(location, instruction, type_inference)? {
                        sites.push(site);
                    }
                }
                
                // Arithmetic operations can be polymorphic (int vs float)
                PrismOpcode::ADD | PrismOpcode::SUB | PrismOpcode::MUL | PrismOpcode::DIV => {
                    if let Some(site) = self.analyze_polymorphic_arithmetic(location, instruction, type_inference)? {
                        sites.push(site);
                    }
                }
                
                _ => {}
            }
        }
        
        // Sort by specialization benefit (highest first)
        sites.sort_by(|a, b| b.specialization_benefit.partial_cmp(&a.specialization_benefit).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(sites)
    }

    /// Analyze polymorphic function call sites
    fn analyze_polymorphic_call_site(
        &mut self,
        location: u32,
        instruction: &Instruction,
        type_inference: &TypeInference,
    ) -> VMResult<Option<PolymorphicSite>> {
        // Look for function type in inference results
        if let Some(function_type) = type_inference.inferred_types.get("stack_function") {
            match function_type {
                InferredType::Union { types } => {
                    // Multiple possible function types - this is polymorphic
                    let function_types: Vec<InferredType> = types.iter()
                        .filter_map(|t| match t {
                            InferredType::Function { .. } => Some(t.clone()),
                            _ => None,
                        })
                        .collect();
                    
                    if function_types.len() > 1 && function_types.len() <= 8 {
                        // Reasonable polymorphism level for inline caching
                        let mut call_frequency = HashMap::new();
                        
                        // Simulate frequency distribution (in real implementation, would use profiling data)
                        let base_frequency = 1.0 / function_types.len() as f64;
                        for (i, func_type) in function_types.iter().enumerate() {
                            // First type is most common, others decrease
                            let frequency = base_frequency * (1.0 - i as f64 * 0.1).max(0.1);
                            call_frequency.insert(func_type.clone(), frequency);
                        }
                        
                        let specialization_benefit = self.calculate_call_site_specialization_benefit(&function_types);
                        
                        return Ok(Some(PolymorphicSite {
                            location,
                            possible_types: function_types,
                            call_frequency,
                            specialization_benefit,
                        }));
                    }
                }
                InferredType::Any => {
                    // Completely unknown function - very polymorphic
                    return Ok(Some(PolymorphicSite {
                        location,
                        possible_types: vec![InferredType::Any],
                        call_frequency: [(InferredType::Any, 1.0)].into_iter().collect(),
                        specialization_benefit: 0.3, // Lower benefit due to uncertainty
                    }));
                }
                _ => {
                    // Monomorphic call - not polymorphic
                }
            }
        }
        
        Ok(None)
    }

    /// Analyze polymorphic field access
    fn analyze_polymorphic_field_access(
        &mut self,
        location: u32,
        instruction: &Instruction,
        type_inference: &TypeInference,
    ) -> VMResult<Option<PolymorphicSite>> {
        if let Some(object_type) = type_inference.inferred_types.get("stack_object") {
            match object_type {
                InferredType::Union { types } => {
                    let object_types: Vec<InferredType> = types.iter()
                        .filter_map(|t| match t {
                            InferredType::Object { .. } => Some(t.clone()),
                            _ => None,
                        })
                        .collect();
                    
                    if object_types.len() > 1 && object_types.len() <= 6 {
                        // Good candidate for inline caching
                        let mut call_frequency = HashMap::new();
                        
                        // Simulate frequency based on type complexity
                        for (i, obj_type) in object_types.iter().enumerate() {
                            let frequency = match obj_type {
                                InferredType::Object { fields, .. } => {
                                    // More common types have fewer fields (heuristic)
                                    let complexity = fields.len() as f64;
                                    (1.0 / (1.0 + complexity * 0.1)) / object_types.len() as f64
                                }
                                _ => 1.0 / object_types.len() as f64,
                            };
                            call_frequency.insert(obj_type.clone(), frequency);
                        }
                        
                        let specialization_benefit = self.calculate_field_access_specialization_benefit(&object_types);
                        
                        return Ok(Some(PolymorphicSite {
                            location,
                            possible_types: object_types,
                            call_frequency,
                            specialization_benefit,
                        }));
                    }
                }
                InferredType::Any => {
                    // Unknown object type
                    return Ok(Some(PolymorphicSite {
                        location,
                        possible_types: vec![InferredType::Any],
                        call_frequency: [(InferredType::Any, 1.0)].into_iter().collect(),
                        specialization_benefit: 0.4,
                    }));
                }
                _ => {}
            }
        }
        
        Ok(None)
    }

    /// Analyze polymorphic method access
    fn analyze_polymorphic_method_access(
        &mut self,
        location: u32,
        instruction: &Instruction,
        type_inference: &TypeInference,
    ) -> VMResult<Option<PolymorphicSite>> {
        // Similar to field access but for methods
        if let Some(object_type) = type_inference.inferred_types.get("stack_object") {
            match object_type {
                InferredType::Union { types } => {
                    let method_types: Vec<InferredType> = types.iter()
                        .filter_map(|t| match t {
                            InferredType::Object { .. } => {
                                // Would need to resolve method from object type
                                Some(InferredType::Function {
                                    params: vec![t.clone()], // 'this' parameter
                                    return_type: Box::new(InferredType::Any),
                                })
                            }
                            _ => None,
                        })
                        .collect();
                    
                    if method_types.len() > 1 {
                        let mut call_frequency = HashMap::new();
                        let base_frequency = 1.0 / method_types.len() as f64;
                        
                        for method_type in &method_types {
                            call_frequency.insert(method_type.clone(), base_frequency);
                        }
                        
                        let specialization_benefit = self.calculate_method_specialization_benefit(&method_types);
                        
                        return Ok(Some(PolymorphicSite {
                            location,
                            possible_types: method_types,
                            call_frequency,
                            specialization_benefit,
                        }));
                    }
                }
                _ => {}
            }
        }
        
        Ok(None)
    }

    /// Analyze polymorphic array access
    fn analyze_polymorphic_array_access(
        &mut self,
        location: u32,
        instruction: &Instruction,
        type_inference: &TypeInference,
    ) -> VMResult<Option<PolymorphicSite>> {
        if let Some(array_type) = type_inference.inferred_types.get("stack_array") {
            match array_type {
                InferredType::Union { types } => {
                    let array_types: Vec<InferredType> = types.iter()
                        .filter_map(|t| match t {
                            InferredType::Array { .. } => Some(t.clone()),
                            _ => None,
                        })
                        .collect();
                    
                    if array_types.len() > 1 {
                        // Different array element types
                        let mut call_frequency = HashMap::new();
                        
                        for array_ty in &array_types {
                            let frequency = match array_ty {
                                InferredType::Array { element, .. } => {
                                    // Primitive arrays are more common
                                    match **element {
                                        InferredType::Integer { .. } => 0.4,
                                        InferredType::Float { .. } => 0.3,
                                        InferredType::Boolean => 0.2,
                                        _ => 0.1,
                                    }
                                }
                                _ => 0.1,
                            };
                            call_frequency.insert(array_ty.clone(), frequency);
                        }
                        
                        let specialization_benefit = self.calculate_array_access_specialization_benefit(&array_types);
                        
                        return Ok(Some(PolymorphicSite {
                            location,
                            possible_types: array_types,
                            call_frequency,
                            specialization_benefit,
                        }));
                    }
                }
                _ => {}
            }
        }
        
        Ok(None)
    }

    /// Analyze polymorphic arithmetic operations
    fn analyze_polymorphic_arithmetic(
        &mut self,
        location: u32,
        instruction: &Instruction,
        type_inference: &TypeInference,
    ) -> VMResult<Option<PolymorphicSite>> {
        // Check if operands have multiple possible types
        let left_types = self.get_possible_types_for_operand("stack_left", type_inference);
        let right_types = self.get_possible_types_for_operand("stack_right", type_inference);
        
        if left_types.len() > 1 || right_types.len() > 1 {
            // Polymorphic arithmetic - could benefit from type-specific code paths
            let mut possible_combinations = Vec::new();
            let mut call_frequency = HashMap::new();
            
            for left_ty in &left_types {
                for right_ty in &right_types {
                    // Create combination type for this arithmetic operation
                    let combo_type = InferredType::Tuple {
                        elements: vec![left_ty.clone(), right_ty.clone()],
                    };
                    
                    // Calculate frequency based on type commonality
                    let frequency = self.calculate_type_combination_frequency(left_ty, right_ty);
                    
                    possible_combinations.push(combo_type.clone());
                    call_frequency.insert(combo_type, frequency);
                }
            }
            
            if possible_combinations.len() > 1 && possible_combinations.len() <= 6 {
                let specialization_benefit = self.calculate_arithmetic_specialization_benefit(&left_types, &right_types);
                
                return Ok(Some(PolymorphicSite {
                    location,
                    possible_types: possible_combinations,
                    call_frequency,
                    specialization_benefit,
                }));
            }
        }
        
        Ok(None)
    }

    /// Get possible types for an operand
    fn get_possible_types_for_operand(&self, operand_name: &str, type_inference: &TypeInference) -> Vec<InferredType> {
        if let Some(operand_type) = type_inference.inferred_types.get(operand_name) {
            match operand_type {
                InferredType::Union { types } => types.clone(),
                other => vec![other.clone()],
            }
        } else {
            vec![InferredType::Any]
        }
    }

    /// Calculate frequency of type combination
    fn calculate_type_combination_frequency(&self, left: &InferredType, right: &InferredType) -> f64 {
        let left_commonality = self.get_type_commonality(left);
        let right_commonality = self.get_type_commonality(right);
        
        // Frequency is product of individual type commonalities
        left_commonality * right_commonality
    }

    /// Get commonality score for a type (how often it appears in practice)
    fn get_type_commonality(&self, ty: &InferredType) -> f64 {
        match ty {
            InferredType::Integer { bits: 32, signed: true } => 0.5, // Most common integer type
            InferredType::Integer { .. } => 0.3,
            InferredType::Float { bits: 64 } => 0.4, // Common float type
            InferredType::Float { .. } => 0.2,
            InferredType::Boolean => 0.3,
            InferredType::String => 0.4,
            InferredType::Array { element, .. } => {
                // Array commonality depends on element type
                self.get_type_commonality(element) * 0.7
            }
            InferredType::Object { .. } => 0.2, // Objects are less common in arithmetic
            InferredType::Any => 0.1, // Unknown types are least common
            _ => 0.1,
        }
    }

    /// Calculate specialization benefit for call sites
    fn calculate_call_site_specialization_benefit(&self, function_types: &[InferredType]) -> f64 {
        let mut benefit = 0.5; // Base benefit for eliminating dynamic dispatch
        
        // Higher benefit for fewer types (easier to specialize)
        benefit += (1.0 / function_types.len() as f64) * 0.3;
        
        // Higher benefit if all types are concrete
        let concrete_ratio = function_types.iter()
            .map(|t| if self.is_concrete_type(t) { 1.0 } else { 0.0 })
            .sum::<f64>() / function_types.len() as f64;
        benefit += concrete_ratio * 0.2;
        
        benefit.min(1.0)
    }

    /// Calculate specialization benefit for field access
    fn calculate_field_access_specialization_benefit(&self, object_types: &[InferredType]) -> f64 {
        let mut benefit = 0.4; // Base benefit for eliminating field lookup
        
        // Higher benefit for fewer object types
        benefit += (1.0 / object_types.len() as f64) * 0.2;
        
        // Higher benefit if objects have similar structure
        let structural_similarity = self.calculate_structural_similarity(object_types);
        benefit += structural_similarity * 0.3;
        
        benefit.min(1.0)
    }

    /// Calculate specialization benefit for method access
    fn calculate_method_specialization_benefit(&self, method_types: &[InferredType]) -> f64 {
        let mut benefit = 0.6; // Higher base benefit than field access
        
        // Method dispatch is more expensive than field access
        benefit += (1.0 / method_types.len() as f64) * 0.3;
        
        benefit.min(1.0)
    }

    /// Calculate specialization benefit for array access
    fn calculate_array_access_specialization_benefit(&self, array_types: &[InferredType]) -> f64 {
        let mut benefit = 0.3; // Base benefit
        
        // Higher benefit for primitive element types
        let primitive_ratio = array_types.iter()
            .map(|t| match t {
                InferredType::Array { element, .. } => {
                    match **element {
                        InferredType::Integer { .. } | InferredType::Float { .. } | InferredType::Boolean => 1.0,
                        _ => 0.0,
                    }
                }
                _ => 0.0,
            })
            .sum::<f64>() / array_types.len() as f64;
        
        benefit += primitive_ratio * 0.4;
        
        benefit.min(1.0)
    }

    /// Calculate specialization benefit for arithmetic operations
    fn calculate_arithmetic_specialization_benefit(&self, left_types: &[InferredType], right_types: &[InferredType]) -> f64 {
        let mut benefit = 0.4; // Base benefit
        
        // Higher benefit for numeric types
        let numeric_benefit = self.calculate_numeric_type_benefit(left_types) + 
                             self.calculate_numeric_type_benefit(right_types);
        benefit += numeric_benefit * 0.3;
        
        // Lower benefit for many type combinations
        let combinations = left_types.len() * right_types.len();
        if combinations <= 4 {
            benefit += 0.2;
        }
        
        benefit.min(1.0)
    }

    /// Calculate benefit for numeric types in arithmetic
    fn calculate_numeric_type_benefit(&self, types: &[InferredType]) -> f64 {
        types.iter()
            .map(|t| match t {
                InferredType::Integer { .. } => 0.8,
                InferredType::Float { .. } => 0.7,
                InferredType::Union { types } => {
                    // Check if union contains only numeric types
                    let all_numeric = types.iter().all(|t| matches!(t, 
                        InferredType::Integer { .. } | InferredType::Float { .. }
                    ));
                    if all_numeric { 0.6 } else { 0.2 }
                }
                _ => 0.1,
            })
            .sum::<f64>() / types.len() as f64
    }

    /// Calculate structural similarity between object types
    fn calculate_structural_similarity(&self, object_types: &[InferredType]) -> f64 {
        if object_types.len() < 2 {
            return 1.0;
        }
        
        // Extract field sets from each object type
        let field_sets: Vec<_> = object_types.iter()
            .filter_map(|t| match t {
                InferredType::Object { fields, .. } => Some(fields.keys().collect::<std::collections::HashSet<_>>()),
                _ => None,
            })
            .collect();
        
        if field_sets.is_empty() {
            return 0.0;
        }
        
        // Calculate Jaccard similarity between field sets
        let mut total_similarity = 0.0;
        let mut comparisons = 0;
        
        for i in 0..field_sets.len() {
            for j in (i + 1)..field_sets.len() {
                let intersection = field_sets[i].intersection(&field_sets[j]).count();
                let union = field_sets[i].union(&field_sets[j]).count();
                
                if union > 0 {
                    total_similarity += intersection as f64 / union as f64;
                    comparisons += 1;
                }
            }
        }
        
        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    /// Update type environment with inferred types
    fn update_type_environment(
        &mut self,
        env: &mut TypeEnvironment,
        type_inference: &TypeInference,
    ) -> VMResult<()> {
        // Update environment with final inferred types
        for (var_name, inferred_type) in &type_inference.inferred_types {
            if let Some(scoped_type) = env.variable_types.get_mut(var_name) {
                scoped_type.inferred_type = inferred_type.clone();
                scoped_type.confidence = type_inference.confidence_scores
                    .get(var_name)
                    .copied()
                    .unwrap_or(0.5);
            }
        }
        
        Ok(())
    }
} 