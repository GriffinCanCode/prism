//! Constraint Generation and Solving
//!
//! This module implements constraint generation and solving for type inference.
//! It generates type constraints during AST traversal and solves them using
//! unification to find a consistent type assignment.

use super::{TypeVar, InferredType, InferenceSource};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        unification::{Unifier, Substitution, UnificationResult},
        errors::{TypeError, TypeErrorKind},
    }
};
use prism_common::Span;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// A type constraint representing an equality between two types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// Left-hand side of the constraint
    pub lhs: ConstraintType,
    /// Right-hand side of the constraint
    pub rhs: ConstraintType,
    /// Source location where this constraint was generated
    pub origin: Span,
    /// Why this constraint was generated (for error reporting)
    pub reason: ConstraintReason,
    /// Priority of this constraint (higher = solve first)
    pub priority: u32,
}

/// Types that can appear in constraints
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// A type variable
    Variable(TypeVar),
    /// A concrete semantic type
    Concrete(SemanticType),
    /// A function type with parameter and return constraints
    Function {
        params: Vec<ConstraintType>,
        return_type: Box<ConstraintType>,
    },
    /// A list type with element constraint
    List(Box<ConstraintType>),
    /// A record type with field constraints
    Record(HashMap<String, ConstraintType>),
    /// A union type with alternative constraints
    Union(Vec<ConstraintType>),
    /// An intersection type (for advanced type systems)
    Intersection(Vec<ConstraintType>),
    /// A semantic type constraint
    Semantic(SemanticType),
}

impl std::hash::Hash for ConstraintType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ConstraintType::Variable(var) => {
                0u8.hash(state);
                var.hash(state);
            }
            ConstraintType::Concrete(semantic_type) => {
                1u8.hash(state);
                // For now, just hash the debug representation of SemanticType
                format!("{:?}", semantic_type).hash(state);
            }
            ConstraintType::Function { params, return_type } => {
                2u8.hash(state);
                params.hash(state);
                return_type.hash(state);
            }
            ConstraintType::List(element_type) => {
                3u8.hash(state);
                element_type.hash(state);
            }
            ConstraintType::Record(fields) => {
                4u8.hash(state);
                // Convert HashMap to a sorted Vec for consistent hashing
                let mut sorted_fields: Vec<_> = fields.iter().collect();
                sorted_fields.sort_by_key(|(k, _)| *k);
                sorted_fields.hash(state);
            }
            ConstraintType::Union(types) => {
                5u8.hash(state);
                types.hash(state);
            }
            ConstraintType::Intersection(types) => {
                6u8.hash(state);
                types.hash(state);
            }
            ConstraintType::Semantic(semantic_type) => {
                7u8.hash(state);
                // For now, just hash the debug representation of SemanticType
                format!("{:?}", semantic_type).hash(state);
            }
        }
    }
}

/// Reasons why constraints are generated (for error reporting)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintReason {
    /// Function call parameter type constraint
    FunctionCall,
    /// Variable assignment constraint
    Assignment,
    /// Return type constraint
    Return,
    /// Type annotation constraint
    TypeAnnotation,
    /// Binary operation constraint
    BinaryOperation { operator: String },
    /// Unary operation constraint
    UnaryOperation { operator: String },
    /// Array element constraint
    ArrayElement,
    /// Record field constraint
    RecordField,
    /// Pattern matching constraint
    PatternMatch,
    /// Generic instantiation constraint
    GenericInstantiation,
    /// Subtyping constraint
    Subtyping,
    /// Equality constraint
    Equality,
    /// User-defined constraint
    UserDefined(String),
    /// Array element unification
    ArrayElementUnification,
    /// Index type check
    IndexTypeCheck,
    /// Branch unification
    BranchUnification,
    /// Conditional check
    ConditionalCheck,
    /// Guard check
    GuardCheck,
    /// Match arm unification
    MatchArmUnification,
    /// Pattern type check
    PatternTypeCheck,
    /// Pattern unification
    PatternUnification,
    /// Effect consistency
    EffectConsistency,
    /// Literal type
    LiteralType,
    /// Function application constraint
    FunctionApplication,
    /// Iterable type check
    IterableCheck,
    /// Awaitable type check
    AwaitableCheck,
    /// Range unification
    RangeUnification,
    /// Variable usage constraint
    VariableUsage { variable_name: String },
    /// Return type constraint
    ReturnType,
    /// Field access constraint
    FieldAccess { field_name: String },
    /// List element constraint
    ListElement,
    /// Conditional constraint
    Conditional,
    /// Operator type constraint
    OperatorType { operator: String, expected: String },
}

/// A set of type constraints with efficient operations
#[derive(Debug, Clone, Default)]
pub struct ConstraintSet {
    /// The actual constraints
    constraints: Vec<TypeConstraint>,
    /// Index by type variables for efficient lookup
    var_index: HashMap<u32, HashSet<usize>>,
    /// Solved constraints (no longer need to be processed)
    solved: HashSet<usize>,
}

/// Constraint solver that finds type assignments satisfying all constraints
#[derive(Debug)]
pub struct ConstraintSolver {
    /// The unification algorithm
    unifier: Unifier,
    /// Statistics about constraint solving
    stats: SolverStatistics,
}

/// Statistics about constraint solving performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStatistics {
    /// Number of constraints processed
    pub constraints_processed: usize,
    /// Number of unification steps
    pub unification_steps: usize,
    /// Number of constraint propagations
    pub propagations: usize,
    /// Time spent solving (microseconds)
    pub solving_time_us: u64,
    /// Peak memory usage during solving
    pub peak_memory_bytes: usize,
}

impl TypeConstraint {
    /// Create a new type constraint
    pub fn new(
        lhs: ConstraintType,
        rhs: ConstraintType,
        origin: Span,
        reason: ConstraintReason,
    ) -> Self {
        Self {
            lhs,
            rhs,
            origin,
            reason,
            priority: 0,
        }
    }

    /// Create a constraint with priority
    pub fn with_priority(
        lhs: ConstraintType,
        rhs: ConstraintType,
        origin: Span,
        reason: ConstraintReason,
        priority: u32,
    ) -> Self {
        Self {
            lhs,
            rhs,
            origin,
            reason,
            priority,
        }
    }

    /// Get all type variables mentioned in this constraint
    pub fn get_variables(&self) -> HashSet<TypeVar> {
        let mut vars = HashSet::new();
        self.lhs.collect_variables(&mut vars);
        self.rhs.collect_variables(&mut vars);
        vars
    }

    /// Check if this constraint is trivial (always satisfied)
    pub fn is_trivial(&self) -> bool {
        self.lhs == self.rhs
    }

    /// Check if this constraint involves a specific type variable
    pub fn involves_variable(&self, var: &TypeVar) -> bool {
        self.lhs.contains_variable(var) || self.rhs.contains_variable(var)
    }

    /// Apply a substitution to this constraint
    pub fn apply_substitution(&mut self, substitution: &Substitution) -> SemanticResult<()> {
        self.lhs = self.lhs.apply_substitution(substitution)?;
        self.rhs = self.rhs.apply_substitution(substitution)?;
        Ok(())
    }
}

impl ConstraintType {
    /// Collect all type variables in this constraint type
    pub fn collect_variables(&self, vars: &mut HashSet<TypeVar>) {
        match self {
            ConstraintType::Variable(var) => {
                vars.insert(var.clone());
            }
            ConstraintType::Concrete(_) => {
                // Concrete types don't contain variables
            }
            ConstraintType::Function { params, return_type } => {
                for param in params {
                    param.collect_variables(vars);
                }
                return_type.collect_variables(vars);
            }
            ConstraintType::List(element_type) => {
                element_type.collect_variables(vars);
            }
            ConstraintType::Record(fields) => {
                for (_, field_type) in fields {
                    field_type.collect_variables(vars);
                }
            }
            ConstraintType::Union(types) | ConstraintType::Intersection(types) => {
                for t in types {
                    t.collect_variables(vars);
                }
            }
            ConstraintType::Semantic(_) => {
                // Semantic types don't contain type variables directly
            }
        }
    }

    /// Check if this constraint type contains a specific variable
    pub fn contains_variable(&self, var: &TypeVar) -> bool {
        match self {
            ConstraintType::Variable(v) => v == var,
            ConstraintType::Concrete(_) => false,
            ConstraintType::Function { params, return_type } => {
                params.iter().any(|p| p.contains_variable(var)) || return_type.contains_variable(var)
            }
            ConstraintType::List(element_type) => element_type.contains_variable(var),
            ConstraintType::Record(fields) => {
                fields.values().any(|field_type| field_type.contains_variable(var))
            }
            ConstraintType::Union(types) | ConstraintType::Intersection(types) => {
                types.iter().any(|t| t.contains_variable(var))
            }
            ConstraintType::Semantic(_) => false,
        }
    }

    /// Convert to a semantic type if possible
    pub fn to_semantic_type(&self) -> Option<SemanticType> {
        match self {
            ConstraintType::Variable(var) => Some(SemanticType::Variable(var.id.to_string())),
            ConstraintType::Concrete(semantic_type) => Some(semantic_type.clone()),
            ConstraintType::Function { params, return_type } => {
                let param_types: Option<Vec<SemanticType>> = params
                    .iter()
                    .map(|p| p.to_semantic_type())
                    .collect();
                let return_semantic = return_type.to_semantic_type()?;
                
                Some(SemanticType::Function {
                    params: param_types?,
                    return_type: Box::new(return_semantic),
                    effects: Vec::new(),
                })
            }
            ConstraintType::List(element_type) => {
                let element_semantic = element_type.to_semantic_type()?;
                Some(SemanticType::List(Box::new(element_semantic)))
            }
            ConstraintType::Record(fields) => {
                let mut semantic_fields = HashMap::new();
                for (name, field_type) in fields {
                    semantic_fields.insert(name.clone(), field_type.to_semantic_type()?);
                }
                Some(SemanticType::Record(semantic_fields))
            }
            ConstraintType::Union(types) => {
                let semantic_types: Option<Vec<SemanticType>> = types
                    .iter()
                    .map(|t| t.to_semantic_type())
                    .collect();
                Some(SemanticType::Union(semantic_types?))
            }
            ConstraintType::Intersection(_) => {
                // Intersection types don't have a direct semantic type representation
                None
            }
            ConstraintType::Semantic(semantic_type) => Some(semantic_type.clone()),
        }
    }

    /// Apply a substitution to this constraint type (immutable version)
    pub fn apply_substitution(&self, substitution: &Substitution) -> SemanticResult<ConstraintType> {
        match self {
            ConstraintType::Variable(var) => {
                if let Some(substitute) = substitution.get_constraint_type(&var.id) {
                    Ok(substitute.clone())
                } else {
                    Ok(self.clone())
                }
            }
            ConstraintType::Concrete(_) => Ok(self.clone()),
            ConstraintType::Function { params, return_type } => {
                let new_params = params.iter()
                    .map(|p| p.apply_substitution(substitution))
                    .collect::<SemanticResult<Vec<_>>>()?;
                let new_return = return_type.apply_substitution(substitution)?;
                Ok(ConstraintType::Function {
                    params: new_params,
                    return_type: Box::new(new_return),
                })
            }
            ConstraintType::List(element_type) => {
                let new_element = element_type.apply_substitution(substitution)?;
                Ok(ConstraintType::List(Box::new(new_element)))
            }
            ConstraintType::Record(fields) => {
                let new_fields = fields.iter()
                    .map(|(name, field_type)| {
                        field_type.apply_substitution(substitution)
                            .map(|t| (name.clone(), t))
                    })
                    .collect::<SemanticResult<HashMap<_, _>>>()?;
                Ok(ConstraintType::Record(new_fields))
            }
            ConstraintType::Union(types) => {
                let new_types = types.iter()
                    .map(|t| t.apply_substitution(substitution))
                    .collect::<SemanticResult<Vec<_>>>()?;
                Ok(ConstraintType::Union(new_types))
            }
            ConstraintType::Intersection(types) => {
                let new_types = types.iter()
                    .map(|t| t.apply_substitution(substitution))
                    .collect::<SemanticResult<Vec<_>>>()?;
                Ok(ConstraintType::Intersection(new_types))
            }
            ConstraintType::Semantic(_semantic_type) => {
                // For semantic types, just return as-is for now
                Ok(self.clone())
            }
        }
    }
}

impl ConstraintSet {
    /// Create a new empty constraint set
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            var_index: HashMap::new(),
            solved: HashSet::new(),
        }
    }

    /// Add a constraint to the set
    pub fn add(&mut self, constraint: TypeConstraint) {
        let index = self.constraints.len();
        
        // Update variable index
        for var in constraint.get_variables() {
            self.var_index
                .entry(var.id)
                .or_insert_with(HashSet::new)
                .insert(index);
        }
        
        self.constraints.push(constraint);
    }

    /// Add multiple constraints
    pub fn extend(&mut self, other: ConstraintSet) {
        for constraint in other.constraints {
            self.add(constraint);
        }
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[TypeConstraint] {
        &self.constraints
    }
    
    /// Add a constraint to the set
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.constraints.push(constraint);
    }
    
    /// Merge another constraint set into this one
    pub fn merge(&mut self, other: ConstraintSet) {
        for constraint in other.constraints {
            self.add_constraint(constraint);
        }
    }

    /// Get unsolved constraints
    pub fn unsolved_constraints(&self) -> impl Iterator<Item = (usize, &TypeConstraint)> {
        self.constraints
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.solved.contains(i))
    }

    /// Mark a constraint as solved
    pub fn mark_solved(&mut self, index: usize) {
        self.solved.insert(index);
    }

    /// Get constraints involving a specific variable
    pub fn constraints_for_variable(&self, var_id: u32) -> Vec<(usize, &TypeConstraint)> {
        if let Some(indices) = self.var_index.get(&var_id) {
            indices
                .iter()
                .filter(|&&i| !self.solved.contains(&i))
                .map(|&i| (i, &self.constraints[i]))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get the number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if the constraint set is empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get the number of unsolved constraints
    pub fn unsolved_count(&self) -> usize {
        self.constraints.len() - self.solved.len()
    }

    /// Apply a substitution to all constraints
    pub fn apply_substitution(&mut self, substitution: &Substitution) -> SemanticResult<()> {
        for constraint in &mut self.constraints {
            constraint.lhs.apply_substitution(substitution)?;
            constraint.rhs.apply_substitution(substitution)?;
        }
        Ok(())
    }

    /// Sort constraints by priority (higher priority first)
    pub fn sort_by_priority(&mut self) {
        let mut indexed_constraints: Vec<(usize, TypeConstraint)> = self.constraints
            .drain(..)
            .enumerate()
            .collect();
        
        indexed_constraints.sort_by(|(_, a), (_, b)| b.priority.cmp(&a.priority));
        
        // Rebuild the constraint set with new indices
        self.constraints = indexed_constraints.into_iter().map(|(_, c)| c).collect();
        self.rebuild_index();
    }

    /// Rebuild the variable index after structural changes
    fn rebuild_index(&mut self) {
        self.var_index.clear();
        self.solved.clear();
        
        for (index, constraint) in self.constraints.iter().enumerate() {
            for var in constraint.get_variables() {
                self.var_index
                    .entry(var.id)
                    .or_insert_with(HashSet::new)
                    .insert(index);
            }
        }
    }
}

impl ConstraintSolver {
    /// Create a new constraint solver
    pub fn new() -> Self {
        Self {
            unifier: Unifier::new(),
            stats: SolverStatistics::default(),
        }
    }

    /// Add a single constraint to solve later
    pub fn add_constraint(&mut self, _constraint: TypeConstraint) {
        // For now, we'll just store constraints and solve them all at once
        // In a more sophisticated implementation, we might do incremental solving
    }

    /// Solve a set of constraints
    pub fn solve(&mut self, constraints: &ConstraintSet) -> SemanticResult<Substitution> {
        let start_time = std::time::Instant::now();
        
        // Initialize statistics
        self.stats = SolverStatistics::default();
        self.stats.constraints_processed = constraints.len();
        
        // Create a working copy of constraints
        let mut working_constraints = constraints.clone();
        working_constraints.sort_by_priority();
        
        // Initialize substitution
        let mut substitution = Substitution::empty();
        
        // Main solving loop
        loop {
            let mut progress = false;
            
            // Collect unsolved constraints first to avoid borrowing issues
            let unsolved_constraints: Vec<_> = working_constraints.unsolved_constraints().collect();
            let mut constraints_to_mark = Vec::new();
            
            for (index, constraint) in unsolved_constraints {
                if self.try_solve_constraint(&constraint, &mut substitution)? {
                    constraints_to_mark.push(index);
                    progress = true;
                }
            }
            
            // Mark constraints as solved
            for index in constraints_to_mark {
                working_constraints.mark_solved(index);
            }
            
            // Apply substitution if progress was made
            if progress {
                working_constraints.apply_substitution(&substitution)?;
            }
            
            if !progress {
                break;
            }
        }
        
        // Check if all constraints were solved
        if working_constraints.unsolved_count() > 0 {
            return self.report_unsolved_constraints(&working_constraints);
        }
        
        // Record timing
        self.stats.solving_time_us = start_time.elapsed().as_micros() as u64;
        
        Ok(substitution)
    }

    /// Try to solve a single constraint
    fn try_solve_constraint(
        &mut self,
        constraint: &TypeConstraint,
        substitution: &mut Substitution,
    ) -> SemanticResult<bool> {
        // Skip trivial constraints
        if constraint.is_trivial() {
            return Ok(true);
        }
        
        // Try unification
        match self.unifier.unify(&constraint.lhs, &constraint.rhs) {
            Ok(UnificationResult::Success(new_substitution)) => {
                substitution.compose(new_substitution);
                Ok(true)
            }
            Ok(UnificationResult::Deferred) => {
                // Can't solve this constraint yet, try later
                Ok(false)
            }
            Err(unification_error) => {
                // Convert unification error to type error
                Err(SemanticError::TypeInferenceError {
                    message: format!(
                        "Type constraint violation: {} (reason: {:?})",
                        unification_error,
                        constraint.reason
                    ),
                })
            }
        }
    }

    /// Report unsolved constraints as errors
    fn report_unsolved_constraints(
        &self,
        constraints: &ConstraintSet,
    ) -> SemanticResult<Substitution> {
        let unsolved: Vec<_> = constraints.unsolved_constraints().collect();
        
        if unsolved.is_empty() {
            return Ok(Substitution::empty());
        }
        
        let first_unsolved = &unsolved[0].1;
        Err(SemanticError::TypeInferenceError {
            message: format!(
                "Could not solve type constraint: {:?} = {:?} (reason: {:?})",
                first_unsolved.lhs,
                first_unsolved.rhs,
                first_unsolved.reason
            ),
        })
    }

    /// Get solving statistics
    pub fn get_statistics(&self) -> &SolverStatistics {
        &self.stats
    }

    /// Reset solver state
    pub fn reset(&mut self) {
        self.unifier = Unifier::new();
        self.stats = SolverStatistics::default();
    }
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TypeConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} = {:?}", self.lhs, self.rhs)
    }
}

impl fmt::Display for ConstraintReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintReason::FunctionCall => write!(f, "function call"),
            ConstraintReason::Assignment => write!(f, "assignment"),
            ConstraintReason::Return => write!(f, "return type"),
            ConstraintReason::TypeAnnotation => write!(f, "type annotation"),
            ConstraintReason::BinaryOperation { operator } => {
                write!(f, "binary operator '{}'", operator)
            }
            ConstraintReason::UnaryOperation { operator } => {
                write!(f, "unary operator '{}'", operator)
            }
            ConstraintReason::ArrayElement => write!(f, "array element"),
            ConstraintReason::RecordField => write!(f, "record field"),
            ConstraintReason::PatternMatch => write!(f, "pattern match"),
            ConstraintReason::GenericInstantiation => write!(f, "generic instantiation"),
            ConstraintReason::Subtyping => write!(f, "subtyping"),
            ConstraintReason::Equality => write!(f, "equality"),
            ConstraintReason::UserDefined(msg) => write!(f, "user defined: {}", msg),
            ConstraintReason::ArrayElementUnification => write!(f, "array element unification"),
            ConstraintReason::IndexTypeCheck => write!(f, "index type check"),
            ConstraintReason::BranchUnification => write!(f, "branch unification"),
            ConstraintReason::ConditionalCheck => write!(f, "conditional check"),
            ConstraintReason::GuardCheck => write!(f, "guard check"),
            ConstraintReason::MatchArmUnification => write!(f, "match arm unification"),
            ConstraintReason::PatternTypeCheck => write!(f, "pattern type check"),
            ConstraintReason::PatternUnification => write!(f, "pattern unification"),
            ConstraintReason::EffectConsistency => write!(f, "effect consistency"),
            ConstraintReason::LiteralType => write!(f, "literal type"),
            ConstraintReason::FunctionApplication => write!(f, "function application"),
            ConstraintReason::IterableCheck => write!(f, "iterable check"),
            ConstraintReason::AwaitableCheck => write!(f, "awaitable check"),
            ConstraintReason::RangeUnification => write!(f, "range unification"),
            ConstraintReason::VariableUsage { variable_name } => {
                write!(f, "variable '{}' usage", variable_name)
            }
            ConstraintReason::ReturnType => write!(f, "return type"),
            ConstraintReason::FieldAccess { field_name } => {
                write!(f, "field '{}' access", field_name)
            }
            ConstraintReason::ListElement => write!(f, "list element"),
            ConstraintReason::Conditional => write!(f, "conditional expression"),
            ConstraintReason::OperatorType { operator, expected } => {
                write!(f, "operator {} requires type {}", operator, expected)
            }
        }
    }
} 