//! Type Environment and Scoping
//!
//! This module implements type environments for managing variable bindings
//! and scoping during type inference. It supports lexical scoping, let-polymorphism,
//! and semantic type information.

use super::{InferredType, TypeVar, InferenceSource, constraints::ConstraintSet};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::unification::Substitution,
};
use prism_common::{Span, NodeId};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// A type binding in the environment
#[derive(Debug, Clone)]
pub struct TypeBinding {
    /// Variable name
    pub name: String,
    /// Inferred type information
    pub type_info: InferredType,
    /// Whether this binding is mutable
    pub is_mutable: bool,
    /// Scope level where this binding was introduced
    pub scope_level: usize,
    /// Node ID where this binding was created
    pub definition_node: Option<NodeId>,
    /// Whether this binding is polymorphic (can be instantiated)
    pub is_polymorphic: bool,
    /// Type scheme for polymorphic bindings
    pub type_scheme: Option<TypeScheme>,
}

/// A type scheme for let-polymorphism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeScheme {
    /// Quantified type variables
    pub quantified_vars: Vec<TypeVar>,
    /// The body type
    pub body_type: SemanticType,
    /// Constraints on the quantified variables
    pub constraints: Vec<TypeConstraint>,
}

/// Constraint on type variables in schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// The constrained type variable
    pub type_var: TypeVar,
    /// The constraint kind
    pub constraint: ConstraintKind,
}

/// Different kinds of type constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Type must implement a trait/interface
    Trait(String),
    /// Type must be a subtype of another type
    Subtype(SemanticType),
    /// Type must be numeric
    Numeric,
    /// Type must be comparable
    Comparable,
    /// Type must be serializable
    Serializable,
}

/// A scope in the type environment
#[derive(Debug, Clone)]
pub struct Scope {
    /// Bindings in this scope
    bindings: HashMap<String, TypeBinding>,
    /// Parent scope (for lexical scoping)
    parent: Option<Box<Scope>>,
    /// Scope level (0 = global, higher = more nested)
    level: usize,
    /// Scope kind
    kind: ScopeKind,
    /// Node ID where this scope was created
    scope_node: Option<NodeId>,
}

/// Different kinds of scopes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Global/module scope
    Global,
    /// Function scope
    Function,
    /// Block scope
    Block,
    /// Let expression scope
    Let,
    /// Match arm scope
    Match,
    /// Loop scope
    Loop,
}

/// Type environment managing scopes and bindings
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    /// Current scope stack
    scope_stack: Vec<Scope>,
    /// Current scope level
    current_level: usize,
    /// Global bindings (built-ins, imports, etc.)
    global_bindings: HashMap<String, TypeBinding>,
    /// Statistics about the environment
    stats: EnvironmentStatistics,
}

/// Statistics about type environment usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentStatistics {
    /// Number of scopes created
    pub scopes_created: usize,
    /// Number of bindings created
    pub bindings_created: usize,
    /// Number of variable lookups
    pub variable_lookups: usize,
    /// Number of successful lookups
    pub successful_lookups: usize,
    /// Maximum scope depth reached
    pub max_scope_depth: usize,
    /// Number of polymorphic instantiations
    pub polymorphic_instantiations: usize,
}

impl TypeBinding {
    /// Create a new type binding
    pub fn new(
        name: String,
        type_info: InferredType,
        is_mutable: bool,
        scope_level: usize,
    ) -> Self {
        Self {
            name,
            type_info,
            is_mutable,
            scope_level,
            definition_node: None,
            is_polymorphic: false,
            type_scheme: None,
        }
    }

    /// Create a polymorphic binding with a type scheme
    pub fn polymorphic(
        name: String,
        type_scheme: TypeScheme,
        scope_level: usize,
    ) -> Self {
        // Create a placeholder type info for the binding
        let type_info = InferredType {
            type_info: type_scheme.body_type.clone(),
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: Vec::new(),
            ai_metadata: None,
            span: Span::dummy(),
        };

        Self {
            name,
            type_info,
            is_mutable: false,
            scope_level,
            definition_node: None,
            is_polymorphic: true,
            type_scheme: Some(type_scheme),
        }
    }

    /// Instantiate this binding if it's polymorphic
    pub fn instantiate(&self, type_var_gen: &mut super::TypeVarGenerator) -> SemanticResult<InferredType> {
        if let Some(scheme) = &self.type_scheme {
            scheme.instantiate(type_var_gen)
        } else {
            Ok(self.type_info.clone())
        }
    }

    /// Apply a substitution to this binding
    pub fn apply_substitution(&mut self, substitution: &Substitution) -> SemanticResult<()> {
        self.type_info.type_info = substitution.apply_to_semantic_type(&self.type_info.type_info)?;
        
        if let Some(scheme) = &mut self.type_scheme {
            scheme.apply_substitution(substitution)?;
        }
        
        Ok(())
    }
}

impl TypeScheme {
    /// Create a new type scheme
    pub fn new(
        quantified_vars: Vec<TypeVar>,
        body_type: SemanticType,
        constraints: Vec<TypeConstraint>,
    ) -> Self {
        Self {
            quantified_vars,
            body_type,
            constraints,
        }
    }

    /// Create a monomorphic type scheme (no quantified variables)
    pub fn monomorphic(body_type: SemanticType) -> Self {
        Self {
            quantified_vars: Vec::new(),
            body_type,
            constraints: Vec::new(),
        }
    }

    /// Instantiate this type scheme with fresh type variables
    pub fn instantiate(&self, type_var_gen: &mut super::TypeVarGenerator) -> SemanticResult<InferredType> {
        if self.quantified_vars.is_empty() {
            // Monomorphic case
            return Ok(InferredType {
                type_info: self.body_type.clone(),
                confidence: 1.0,
                inference_source: InferenceSource::Explicit,
                constraints: Vec::new(),
                ai_metadata: None,
                span: Span::dummy(),
            });
        }

        // Create fresh type variables for each quantified variable
        let mut substitution = Substitution::empty();
        for quantified_var in &self.quantified_vars {
            let fresh_var = type_var_gen.fresh(quantified_var.origin);
            let fresh_semantic = SemanticType::Variable(fresh_var.id.to_string());
            substitution.add_semantic_mapping(quantified_var.id, fresh_semantic);
        }

        // Apply substitution to body type
        let instantiated_type = substitution.apply_to_semantic_type(&self.body_type)?;

        // Instantiate constraints (convert from environment::TypeConstraint to constraints::TypeConstraint)
        let instantiated_constraints = self.constraints
            .iter()
            .map(|constraint| {
                // Convert environment constraint to inference constraint
                super::constraints::TypeConstraint {
                    lhs: super::constraints::ConstraintType::Variable(constraint.type_var.clone()),
                    rhs: super::constraints::ConstraintType::Concrete(instantiated_type.clone()),
                    origin: Span::dummy(),
                    reason: super::constraints::ConstraintReason::TypeAnnotation,
                    priority: 50,
                }
            })
            .collect();

        Ok(InferredType {
            type_info: instantiated_type,
            confidence: 1.0,
            inference_source: InferenceSource::Explicit,
            constraints: instantiated_constraints,
            ai_metadata: None,
            span: Span::dummy(),
        })
    }

    /// Apply a substitution to this type scheme
    pub fn apply_substitution(&mut self, substitution: &Substitution) -> SemanticResult<()> {
        // Only apply to free variables (not quantified ones)
        let quantified_ids: HashSet<u32> = self.quantified_vars.iter().map(|v| v.id).collect();
        
        // Create a restricted substitution that doesn't affect quantified variables
        let mut restricted_substitution = Substitution::empty();
        for var_id in substitution.domain() {
            if !quantified_ids.contains(var_id) {
                if let Some(constraint_type) = substitution.get_type_var_substitution(*var_id) {
                    restricted_substitution.add_mapping(*var_id, constraint_type.clone());
                }
                if let Some(semantic_type) = substitution.get_semantic_substitution(*var_id) {
                    restricted_substitution.add_semantic_mapping(*var_id, semantic_type.clone());
                }
            }
        }

        self.body_type = restricted_substitution.apply_to_semantic_type(&self.body_type)?;
        Ok(())
    }

    /// Get free type variables in this scheme
    pub fn free_variables(&self) -> HashSet<TypeVar> {
        let mut free_vars = HashSet::new();
        self.collect_variables_in_semantic_type(&self.body_type, &mut free_vars);
        
        // Remove quantified variables
        let quantified_set: HashSet<_> = self.quantified_vars.iter().collect();
        free_vars.retain(|var| !quantified_set.contains(var));
        
        free_vars
    }

    fn collect_variables_in_semantic_type(&self, semantic_type: &SemanticType, vars: &mut HashSet<TypeVar>) {
        match semantic_type {
            SemanticType::Variable(var_name) => {
                if let Ok(var_id) = var_name.parse::<u32>() {
                    // Find the TypeVar with this ID (this is a simplification)
                    if let Some(type_var) = self.quantified_vars.iter().find(|v| v.id == var_id) {
                        vars.insert(type_var.clone());
                    }
                }
            }
            SemanticType::Function { params, return_type, .. } => {
                for param in params {
                    self.collect_variables_in_semantic_type(param, vars);
                }
                self.collect_variables_in_semantic_type(return_type, vars);
            }
            SemanticType::List(element_type) => {
                self.collect_variables_in_semantic_type(element_type, vars);
            }
            SemanticType::Record(fields) => {
                for (_, field_type) in fields {
                    self.collect_variables_in_semantic_type(field_type, vars);
                }
            }
            SemanticType::Union(types) => {
                for t in types {
                    self.collect_variables_in_semantic_type(t, vars);
                }
            }
            SemanticType::Primitive(_) | SemanticType::Generic { .. } | SemanticType::Complex { .. } => {
                // No variables in primitive, generic, or complex types
            }
        }
    }
}

impl Scope {
    /// Create a new scope
    pub fn new(kind: ScopeKind, level: usize, parent: Option<Box<Scope>>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent,
            level,
            kind,
            scope_node: None,
        }
    }

    /// Add a binding to this scope
    pub fn add_binding(&mut self, binding: TypeBinding) {
        self.bindings.insert(binding.name.clone(), binding);
    }

    /// Look up a binding in this scope (not parent scopes)
    pub fn lookup_local(&self, name: &str) -> Option<&TypeBinding> {
        self.bindings.get(name)
    }

    /// Look up a binding in this scope or parent scopes
    pub fn lookup(&self, name: &str) -> Option<&TypeBinding> {
        if let Some(binding) = self.bindings.get(name) {
            Some(binding)
        } else if let Some(parent) = &self.parent {
            parent.lookup(name)
        } else {
            None
        }
    }

    /// Check if this scope contains a binding
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Get all bindings in this scope
    pub fn bindings(&self) -> &HashMap<String, TypeBinding> {
        &self.bindings
    }

    /// Apply a substitution to all bindings in this scope
    pub fn apply_substitution(&mut self, substitution: &Substitution) -> SemanticResult<()> {
        for (_, binding) in &mut self.bindings {
            binding.apply_substitution(substitution)?;
        }
        
        if let Some(parent) = &mut self.parent {
            parent.apply_substitution(substitution)?;
        }
        
        Ok(())
    }
}

impl TypeEnvironment {
    /// Create a new type environment
    pub fn new() -> Self {
        let global_scope = Scope::new(ScopeKind::Global, 0, None);
        
        Self {
            scope_stack: vec![global_scope],
            current_level: 0,
            global_bindings: HashMap::new(),
            stats: EnvironmentStatistics::default(),
        }
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self, kind: ScopeKind) {
        self.current_level += 1;
        let parent = self.scope_stack.pop();
        let new_scope = Scope::new(kind, self.current_level, parent.map(Box::new));
        self.scope_stack.push(new_scope);
        
        self.stats.scopes_created += 1;
        self.stats.max_scope_depth = self.stats.max_scope_depth.max(self.current_level);
    }

    /// Exit the current scope
    pub fn exit_scope(&mut self) -> SemanticResult<()> {
        if self.scope_stack.len() <= 1 {
            return Err(SemanticError::TypeInferenceError {
                message: "Cannot exit global scope".to_string(),
            });
        }

        if let Some(current_scope) = self.scope_stack.pop() {
            if let Some(parent) = current_scope.parent {
                self.scope_stack.push(*parent);
                self.current_level -= 1;
            }
        }

        Ok(())
    }

    /// Add a binding to the current scope
    pub fn add_binding(&mut self, binding: TypeBinding) {
        if let Some(current_scope) = self.scope_stack.last_mut() {
            current_scope.add_binding(binding);
            self.stats.bindings_created += 1;
        }
    }

    /// Add a global binding
    pub fn add_global_binding(&mut self, binding: TypeBinding) {
        self.global_bindings.insert(binding.name.clone(), binding);
        self.stats.bindings_created += 1;
    }

    /// Look up a variable in the environment
    pub fn lookup(&mut self, name: &str) -> Option<&TypeBinding> {
        self.stats.variable_lookups += 1;

        // First check current scope stack
        for scope in self.scope_stack.iter().rev() {
            if let Some(binding) = scope.lookup_local(name) {
                return Some(binding);
            }
        }

        // Then check global bindings
        if let Some(binding) = self.global_bindings.get(name) {
            return Some(binding);
        }

        None
    }

    /// Look up and instantiate a polymorphic binding
    pub fn lookup_and_instantiate(
        &mut self,
        name: &str,
        type_var_gen: &mut super::TypeVarGenerator,
    ) -> SemanticResult<Option<InferredType>> {
        // Look up the binding without incrementing stats yet
        let binding_found = {
            // First check current scope stack
            for scope in self.scope_stack.iter().rev() {
                if let Some(binding) = scope.lookup_local(name) {
                    return Ok(Some({
                        self.stats.variable_lookups += 1;
                        self.stats.successful_lookups += 1;
                        if binding.is_polymorphic {
                            self.stats.polymorphic_instantiations += 1;
                        }
                        binding.instantiate(type_var_gen)?
                    }));
                }
            }

            // Then check global bindings
            if let Some(binding) = self.global_bindings.get(name) {
                self.stats.variable_lookups += 1;
                self.stats.successful_lookups += 1;
                if binding.is_polymorphic {
                    self.stats.polymorphic_instantiations += 1;
                }
                return Ok(Some(binding.instantiate(type_var_gen)?));
            }
            
            self.stats.variable_lookups += 1;
            Ok(None)
        };
        
        binding_found
    }

    /// Get the current scope level
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    /// Get the current scope kind
    pub fn current_scope_kind(&self) -> Option<ScopeKind> {
        self.scope_stack.last().map(|scope| scope.kind)
    }

    /// Check if we're in a specific kind of scope
    pub fn in_scope(&self, kind: ScopeKind) -> bool {
        self.scope_stack.iter().any(|scope| scope.kind == kind)
    }

    /// Apply a substitution to the entire environment
    pub fn apply_substitution(&mut self, substitution: &Substitution) -> SemanticResult<()> {
        // Apply to scope stack
        for scope in &mut self.scope_stack {
            scope.apply_substitution(substitution)?;
        }

        // Apply to global bindings
        for (_, binding) in &mut self.global_bindings {
            binding.apply_substitution(substitution)?;
        }

        Ok(())
    }

    /// Generalize a type by quantifying over free variables
    pub fn generalize(
        &mut self,
        semantic_type: &SemanticType,
        span: Span,
    ) -> TypeScheme {
        // Find free variables in the type that are not bound in the environment
        let free_vars = self.find_free_variables(semantic_type);
        
        // Create type variables for generalization
        let quantified_vars: Vec<TypeVar> = free_vars
            .into_iter()
            .map(|var_name| {
                let var_id = var_name.parse::<u32>().unwrap_or(0);
                TypeVar {
                    id: var_id,
                    name: Some(var_name),
                    origin: span,
                }
            })
            .collect();

        TypeScheme::new(quantified_vars, semantic_type.clone(), Vec::new())
    }

    /// Find free variables in a semantic type
    fn find_free_variables(&mut self, semantic_type: &SemanticType) -> Vec<String> {
        let mut free_vars = Vec::new();
        self.collect_free_variables(semantic_type, &mut free_vars);
        free_vars.sort();
        free_vars.dedup();
        free_vars
    }

    /// Collect free variables recursively
    fn collect_free_variables(&mut self, semantic_type: &SemanticType, vars: &mut Vec<String>) {
        match semantic_type {
            SemanticType::Variable(var_name) => {
                // Check if this variable is bound in the environment
                if self.lookup(var_name).is_none() {
                    vars.push(var_name.clone());
                }
            }
            SemanticType::Function { params, return_type, .. } => {
                for param in params {
                    self.collect_free_variables(param, vars);
                }
                self.collect_free_variables(return_type, vars);
            }
            SemanticType::List(element_type) => {
                self.collect_free_variables(element_type, vars);
            }
            SemanticType::Record(fields) => {
                for (_, field_type) in fields {
                    self.collect_free_variables(field_type, vars);
                }
            }
            SemanticType::Union(types) => {
                for t in types {
                    self.collect_free_variables(t, vars);
                }
            }
            SemanticType::Primitive(_) | SemanticType::Generic { .. } | SemanticType::Complex { .. } => {
                // No free variables in primitive, generic, or complex types
            }
        }
    }

    /// Merge another environment into this one
    pub fn merge(&mut self, other: TypeEnvironment) {
        // Merge global bindings
        self.global_bindings.extend(other.global_bindings);
        
        // Merge statistics
        self.stats.scopes_created += other.stats.scopes_created;
        self.stats.bindings_created += other.stats.bindings_created;
        self.stats.variable_lookups += other.stats.variable_lookups;
        self.stats.successful_lookups += other.stats.successful_lookups;
        self.stats.max_scope_depth = self.stats.max_scope_depth.max(other.stats.max_scope_depth);
        self.stats.polymorphic_instantiations += other.stats.polymorphic_instantiations;
    }

    /// Get environment statistics
    pub fn get_statistics(&self) -> &EnvironmentStatistics {
        &self.stats
    }

    /// Reset environment statistics
    pub fn reset_statistics(&mut self) {
        self.stats = EnvironmentStatistics::default();
    }

    /// Get all bindings in the current scope
    pub fn current_scope_bindings(&self) -> Option<&HashMap<String, TypeBinding>> {
        self.scope_stack.last().map(|scope| &scope.bindings)
    }

    /// Get all global bindings
    pub fn global_bindings(&self) -> &HashMap<String, TypeBinding> {
        &self.global_bindings
    }
}

impl Default for TypeEnvironment {
    fn default() -> Self {
        Self::new()
    }
} 