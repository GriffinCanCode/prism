//! Unification Algorithm
//!
//! This module implements the unification algorithm used to solve type constraints.
//! It includes the classic Robinson unification algorithm with extensions for
//! semantic types and occurs check.

use super::{
    TypeVar,
    constraints::{ConstraintType, TypeConstraint},
};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
};
use prism_common::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;

/// Result of a unification attempt
#[derive(Debug, Clone)]
pub enum UnificationResult {
    /// Unification succeeded with a substitution
    Success(Substitution),
    /// Unification was deferred (couldn't be resolved yet)
    Deferred,
}

/// A substitution mapping type variables to types
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    /// Mapping from type variable IDs to constraint types
    type_var_map: HashMap<u32, ConstraintType>,
    /// Mapping from type variable IDs to semantic types (for final resolution)
    semantic_map: HashMap<u32, SemanticType>,
    /// Domain of this substitution (variables it affects)
    domain: std::collections::HashSet<u32>,
}

/// Unification algorithm implementation
#[derive(Debug)]
pub struct Unifier {
    /// Statistics about unification operations
    stats: UnificationStatistics,
    /// Maximum depth for occurs check to prevent infinite recursion
    max_occurs_depth: usize,
}

/// Statistics about unification performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnificationStatistics {
    /// Number of unification attempts
    pub unification_attempts: usize,
    /// Number of successful unifications
    pub successful_unifications: usize,
    /// Number of failed unifications
    pub failed_unifications: usize,
    /// Number of deferred unifications
    pub deferred_unifications: usize,
    /// Number of occurs check failures
    pub occurs_check_failures: usize,
    /// Total time spent in unification (microseconds)
    pub total_time_us: u64,
}

/// Error that can occur during unification
#[derive(Debug, Clone)]
pub enum UnificationError {
    /// Types cannot be unified (fundamental mismatch)
    TypeMismatch {
        left: ConstraintType,
        right: ConstraintType,
        reason: String,
    },
    /// Occurs check failed (would create infinite type)
    OccursCheck {
        variable: TypeVar,
        type_expr: ConstraintType,
    },
    /// Arity mismatch (different number of arguments)
    ArityMismatch {
        left_arity: usize,
        right_arity: usize,
        context: String,
    },
    /// Field mismatch in record types
    FieldMismatch {
        missing_fields: Vec<String>,
        extra_fields: Vec<String>,
    },
    /// Internal unification error
    InternalError {
        message: String,
    },
}

impl Substitution {
    /// Create an empty substitution
    pub fn empty() -> Self {
        Self {
            type_var_map: HashMap::new(),
            semantic_map: HashMap::new(),
            domain: std::collections::HashSet::new(),
        }
    }

    /// Create a substitution with a single mapping
    pub fn single(var_id: u32, constraint_type: ConstraintType) -> Self {
        let mut map = HashMap::new();
        let mut domain = std::collections::HashSet::new();
        
        map.insert(var_id, constraint_type);
        domain.insert(var_id);
        
        Self {
            type_var_map: map,
            semantic_map: HashMap::new(),
            domain,
        }
    }

    /// Get the substitution for a type variable
    pub fn get_type_var_substitution(&self, var_id: u32) -> Option<&ConstraintType> {
        self.type_var_map.get(&var_id)
    }

    /// Get the semantic type substitution for a type variable
    pub fn get_semantic_substitution(&self, var_id: u32) -> Option<&SemanticType> {
        self.semantic_map.get(&var_id)
    }

    /// Add a mapping to this substitution
    pub fn add_mapping(&mut self, var_id: u32, constraint_type: ConstraintType) {
        self.type_var_map.insert(var_id, constraint_type);
        self.domain.insert(var_id);
    }

    /// Add a semantic mapping
    pub fn add_semantic_mapping(&mut self, var_id: u32, semantic_type: SemanticType) {
        self.semantic_map.insert(var_id, semantic_type);
        self.domain.insert(var_id);
    }

    /// Compose this substitution with another (self ∘ other)
    pub fn compose(&mut self, other: Substitution) {
        // Apply other to the range of self
        for (var_id, constraint_type) in &mut self.type_var_map {
            if let Err(_) = constraint_type.apply_substitution(&other) {
                // If application fails, we might need to handle this more gracefully
                continue;
            }
        }
        
        // Add mappings from other that aren't in self
        for (var_id, constraint_type) in other.type_var_map {
            if !self.type_var_map.contains_key(&var_id) {
                self.type_var_map.insert(var_id, constraint_type);
                self.domain.insert(var_id);
            }
        }
        
        // Same for semantic mappings
        for (var_id, semantic_type) in other.semantic_map {
            if !self.semantic_map.contains_key(&var_id) {
                self.semantic_map.insert(var_id, semantic_type);
                self.domain.insert(var_id);
            }
        }
        
        // Merge domains
        self.domain.extend(other.domain);
    }

    /// Apply this substitution to a constraint type
    pub fn apply_to_constraint_type(&self, constraint_type: &ConstraintType) -> SemanticResult<ConstraintType> {
        let mut result = constraint_type.clone();
        result.apply_substitution(self)?;
        Ok(result)
    }

    /// Apply this substitution to a semantic type
    pub fn apply_to_semantic_type(&self, semantic_type: &SemanticType) -> SemanticResult<SemanticType> {
        match semantic_type {
            SemanticType::Variable(var_name) => {
                // Try to parse the variable name as a numeric ID
                if let Ok(var_id) = var_name.parse::<u32>() {
                    if let Some(semantic_sub) = self.semantic_map.get(&var_id) {
                        Ok(semantic_sub.clone())
                    } else if let Some(constraint_sub) = self.type_var_map.get(&var_id) {
                        // Convert constraint type to semantic type if possible
                        constraint_sub.to_semantic_type()
                            .ok_or_else(|| SemanticError::TypeInferenceError {
                                message: format!("Cannot convert constraint type to semantic type: {:?}", constraint_sub),
                            })
                    } else {
                        Ok(semantic_type.clone())
                    }
                } else {
                    Ok(semantic_type.clone())
                }
            }
            SemanticType::Function { params, return_type, effects } => {
                let new_params: Result<Vec<_>, _> = params
                    .iter()
                    .map(|p| self.apply_to_semantic_type(p))
                    .collect();
                let new_return = self.apply_to_semantic_type(return_type)?;
                
                Ok(SemanticType::Function {
                    params: new_params?,
                    return_type: Box::new(new_return),
                    effects: effects.clone(),
                })
            }
            SemanticType::List(element_type) => {
                let new_element = self.apply_to_semantic_type(element_type)?;
                Ok(SemanticType::List(Box::new(new_element)))
            }
            SemanticType::Record(fields) => {
                let mut new_fields = HashMap::new();
                for (name, field_type) in fields {
                    new_fields.insert(name.clone(), self.apply_to_semantic_type(field_type)?);
                }
                Ok(SemanticType::Record(new_fields))
            }
            SemanticType::Union(types) => {
                let new_types: Result<Vec<_>, _> = types
                    .iter()
                    .map(|t| self.apply_to_semantic_type(t))
                    .collect();
                Ok(SemanticType::Union(new_types?))
            }
            // For primitive, generic, and complex types, no substitution needed
            SemanticType::Primitive(_) | SemanticType::Generic { .. } | SemanticType::Complex { .. } => {
                Ok(semantic_type.clone())
            }
        }
    }

    /// Check if this substitution is empty
    pub fn is_empty(&self) -> bool {
        self.type_var_map.is_empty() && self.semantic_map.is_empty()
    }

    /// Get the domain of this substitution
    pub fn domain(&self) -> &std::collections::HashSet<u32> {
        &self.domain
    }

    /// Get constraint type for a variable ID
    pub fn get_constraint_type(&self, var_id: &u32) -> Option<&ConstraintType> {
        self.type_var_map.get(var_id)
    }

    /// Get semantic type for a variable ID
    pub fn get_semantic_type(&self, var_id: &u32) -> Option<&SemanticType> {
        self.semantic_map.get(var_id)
    }

    /// Get the size of this substitution
    pub fn size(&self) -> usize {
        self.type_var_map.len() + self.semantic_map.len()
    }
}

impl Unifier {
    /// Create a new unifier
    pub fn new() -> Self {
        Self {
            stats: UnificationStatistics::default(),
            max_occurs_depth: 100,
        }
    }

    /// Create a unifier with custom configuration
    pub fn with_config(max_occurs_depth: usize) -> Self {
        Self {
            stats: UnificationStatistics::default(),
            max_occurs_depth,
        }
    }

    /// Unify two constraint types
    pub fn unify(
        &mut self,
        left: &ConstraintType,
        right: &ConstraintType,
    ) -> Result<UnificationResult, UnificationError> {
        let start_time = std::time::Instant::now();
        self.stats.unification_attempts += 1;
        
        let result = self.unify_internal(left, right, 0);
        
        // Update statistics
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        match &result {
            Ok(UnificationResult::Success(_)) => self.stats.successful_unifications += 1,
            Ok(UnificationResult::Deferred) => self.stats.deferred_unifications += 1,
            Err(_) => self.stats.failed_unifications += 1,
        }
        
        result
    }

    /// Internal unification with depth tracking
    fn unify_internal(
        &mut self,
        left: &ConstraintType,
        right: &ConstraintType,
        depth: usize,
    ) -> Result<UnificationResult, UnificationError> {
        // Prevent infinite recursion
        if depth > self.max_occurs_depth {
            return Ok(UnificationResult::Deferred);
        }
        
        match (left, right) {
            // Variable cases
            (ConstraintType::Variable(var), other) | (other, ConstraintType::Variable(var)) => {
                self.unify_variable(var, other)
            }
            
            // Concrete types
            (ConstraintType::Concrete(left_type), ConstraintType::Concrete(right_type)) => {
                self.unify_concrete_types(left_type, right_type)
            }
            
            // Function types
            (
                ConstraintType::Function { params: left_params, return_type: left_return },
                ConstraintType::Function { params: right_params, return_type: right_return },
            ) => {
                self.unify_function_types(left_params, left_return, right_params, right_return, depth)
            }
            
            // List types
            (ConstraintType::List(left_elem), ConstraintType::List(right_elem)) => {
                self.unify_internal(left_elem, right_elem, depth + 1)
            }
            
            // Record types
            (ConstraintType::Record(left_fields), ConstraintType::Record(right_fields)) => {
                self.unify_record_types(left_fields, right_fields, depth)
            }
            
            // Union types
            (ConstraintType::Union(left_types), ConstraintType::Union(right_types)) => {
                self.unify_union_types(left_types, right_types, depth)
            }
            
            // Intersection types
            (ConstraintType::Intersection(left_types), ConstraintType::Intersection(right_types)) => {
                self.unify_intersection_types(left_types, right_types, depth)
            }
            
            // Mixed cases that might be compatible
            (ConstraintType::Concrete(concrete), ConstraintType::List(elem)) |
            (ConstraintType::List(elem), ConstraintType::Concrete(concrete)) => {
                self.unify_concrete_with_structured(concrete, &ConstraintType::List(elem.clone()))
            }
            
            // Incompatible types
            _ => Err(UnificationError::TypeMismatch {
                left: left.clone(),
                right: right.clone(),
                reason: "Incompatible type constructors".to_string(),
            }),
        }
    }

    /// Unify a variable with another type
    fn unify_variable(
        &mut self,
        var: &TypeVar,
        other: &ConstraintType,
    ) -> Result<UnificationResult, UnificationError> {
        // Occurs check
        if self.occurs_check(var, other) {
            self.stats.occurs_check_failures += 1;
            return Err(UnificationError::OccursCheck {
                variable: var.clone(),
                type_expr: other.clone(),
            });
        }
        
        // Create substitution
        let substitution = Substitution::single(var.id, other.clone());
        Ok(UnificationResult::Success(substitution))
    }

    /// Unify two concrete semantic types
    fn unify_concrete_types(
        &mut self,
        left: &SemanticType,
        right: &SemanticType,
    ) -> Result<UnificationResult, UnificationError> {
        match (left, right) {
            // Identical primitive types
            (SemanticType::Primitive(left_prim), SemanticType::Primitive(right_prim)) => {
                if left_prim == right_prim {
                    Ok(UnificationResult::Success(Substitution::empty()))
                } else {
                    Err(UnificationError::TypeMismatch {
                        left: ConstraintType::Concrete(left.clone()),
                        right: ConstraintType::Concrete(right.clone()),
                        reason: "Different primitive types".to_string(),
                    })
                }
            }
            
            // Function types
            (
                SemanticType::Function { params: left_params, return_type: left_return, .. },
                SemanticType::Function { params: right_params, return_type: right_return, .. },
            ) => {
                if left_params.len() != right_params.len() {
                    return Err(UnificationError::ArityMismatch {
                        left_arity: left_params.len(),
                        right_arity: right_params.len(),
                        context: "function parameters".to_string(),
                    });
                }
                
                let mut combined_substitution = Substitution::empty();
                
                // Unify parameters
                for (left_param, right_param) in left_params.iter().zip(right_params.iter()) {
                    let left_constraint = ConstraintType::Concrete(left_param.clone());
                    let right_constraint = ConstraintType::Concrete(right_param.clone());
                    
                    match self.unify_internal(&left_constraint, &right_constraint, 0)? {
                        UnificationResult::Success(sub) => combined_substitution.compose(sub),
                        UnificationResult::Deferred => return Ok(UnificationResult::Deferred),
                    }
                }
                
                // Unify return types
                let left_return_constraint = ConstraintType::Concrete((**left_return).clone());
                let right_return_constraint = ConstraintType::Concrete((**right_return).clone());
                
                match self.unify_internal(&left_return_constraint, &right_return_constraint, 0)? {
                    UnificationResult::Success(sub) => combined_substitution.compose(sub),
                    UnificationResult::Deferred => return Ok(UnificationResult::Deferred),
                }
                
                Ok(UnificationResult::Success(combined_substitution))
            }
            
            // List types
            (SemanticType::List(left_elem), SemanticType::List(right_elem)) => {
                let left_constraint = ConstraintType::Concrete((**left_elem).clone());
                let right_constraint = ConstraintType::Concrete((**right_elem).clone());
                self.unify_internal(&left_constraint, &right_constraint, 0)
            }
            
            // Record types
            (SemanticType::Record(left_fields), SemanticType::Record(right_fields)) => {
                let mut left_constraint_fields = HashMap::new();
                let mut right_constraint_fields = HashMap::new();
                
                for (name, field_type) in left_fields {
                    left_constraint_fields.insert(name.clone(), ConstraintType::Concrete(field_type.clone()));
                }
                
                for (name, field_type) in right_fields {
                    right_constraint_fields.insert(name.clone(), ConstraintType::Concrete(field_type.clone()));
                }
                
                self.unify_record_types(&left_constraint_fields, &right_constraint_fields, 0)
            }
            
            // Variables in concrete types
            (SemanticType::Variable(left_var), SemanticType::Variable(right_var)) => {
                if left_var == right_var {
                    Ok(UnificationResult::Success(Substitution::empty()))
                } else {
                    // Two different variables - can't unify without more information
                    Ok(UnificationResult::Deferred)
                }
            }
            
            // Mixed cases
            _ => Err(UnificationError::TypeMismatch {
                left: ConstraintType::Concrete(left.clone()),
                right: ConstraintType::Concrete(right.clone()),
                reason: "Incompatible semantic types".to_string(),
            }),
        }
    }

    /// Unify function types
    fn unify_function_types(
        &mut self,
        left_params: &[ConstraintType],
        left_return: &ConstraintType,
        right_params: &[ConstraintType],
        right_return: &ConstraintType,
        depth: usize,
    ) -> Result<UnificationResult, UnificationError> {
        if left_params.len() != right_params.len() {
            return Err(UnificationError::ArityMismatch {
                left_arity: left_params.len(),
                right_arity: right_params.len(),
                context: "function parameters".to_string(),
            });
        }
        
        let mut combined_substitution = Substitution::empty();
        
        // Unify parameters (contravariant)
        for (left_param, right_param) in left_params.iter().zip(right_params.iter()) {
            match self.unify_internal(right_param, left_param, depth + 1)? {
                UnificationResult::Success(sub) => combined_substitution.compose(sub),
                UnificationResult::Deferred => return Ok(UnificationResult::Deferred),
            }
        }
        
        // Unify return types (covariant)
        match self.unify_internal(left_return, right_return, depth + 1)? {
            UnificationResult::Success(sub) => combined_substitution.compose(sub),
            UnificationResult::Deferred => return Ok(UnificationResult::Deferred),
        }
        
        Ok(UnificationResult::Success(combined_substitution))
    }

    /// Unify record types
    fn unify_record_types(
        &mut self,
        left_fields: &HashMap<String, ConstraintType>,
        right_fields: &HashMap<String, ConstraintType>,
        depth: usize,
    ) -> Result<UnificationResult, UnificationError> {
        // Check for missing/extra fields
        let left_field_names: std::collections::HashSet<_> = left_fields.keys().collect();
        let right_field_names: std::collections::HashSet<_> = right_fields.keys().collect();
        
        let missing_in_right: Vec<String> = left_field_names
            .difference(&right_field_names)
            .map(|s| s.to_string())
            .collect();
        let missing_in_left: Vec<String> = right_field_names
            .difference(&left_field_names)
            .map(|s| s.to_string())
            .collect();
        
        if !missing_in_right.is_empty() || !missing_in_left.is_empty() {
            return Err(UnificationError::FieldMismatch {
                missing_fields: missing_in_right,
                extra_fields: missing_in_left,
            });
        }
        
        // Unify common fields
        let mut combined_substitution = Substitution::empty();
        
        for (field_name, left_field_type) in left_fields {
            if let Some(right_field_type) = right_fields.get(field_name) {
                match self.unify_internal(left_field_type, right_field_type, depth + 1)? {
                    UnificationResult::Success(sub) => combined_substitution.compose(sub),
                    UnificationResult::Deferred => return Ok(UnificationResult::Deferred),
                }
            }
        }
        
        Ok(UnificationResult::Success(combined_substitution))
    }

    /// Unify union types
    fn unify_union_types(
        &mut self,
        left_types: &[ConstraintType],
        right_types: &[ConstraintType],
        _depth: usize,
    ) -> Result<UnificationResult, UnificationError> {
        // For now, we require exact matches for union types
        // A more sophisticated implementation would handle subtyping
        if left_types.len() != right_types.len() {
            return Err(UnificationError::TypeMismatch {
                left: ConstraintType::Union(left_types.to_vec()),
                right: ConstraintType::Union(right_types.to_vec()),
                reason: "Different number of union alternatives".to_string(),
            });
        }
        
        // This is a simplified implementation
        // In practice, we'd need to handle permutations of union members
        Ok(UnificationResult::Deferred)
    }

    /// Unify intersection types
    fn unify_intersection_types(
        &mut self,
        _left_types: &[ConstraintType],
        _right_types: &[ConstraintType],
        _depth: usize,
    ) -> Result<UnificationResult, UnificationError> {
        // Intersection types are complex and not fully implemented
        Ok(UnificationResult::Deferred)
    }

    /// Unify concrete type with structured type
    fn unify_concrete_with_structured(
        &mut self,
        _concrete: &SemanticType,
        _structured: &ConstraintType,
    ) -> Result<UnificationResult, UnificationError> {
        // This would handle cases like unifying a concrete list type with a constraint list type
        Ok(UnificationResult::Deferred)
    }

    /// Occurs check to prevent infinite types
    fn occurs_check(&self, var: &TypeVar, constraint_type: &ConstraintType) -> bool {
        self.occurs_check_internal(var, constraint_type, 0)
    }

    /// Internal occurs check with depth limit
    fn occurs_check_internal(&self, var: &TypeVar, constraint_type: &ConstraintType, depth: usize) -> bool {
        if depth > self.max_occurs_depth {
            return true; // Conservative: assume occurs check fails
        }
        
        match constraint_type {
            ConstraintType::Variable(other_var) => var.id == other_var.id,
            ConstraintType::Concrete(_) => false,
            ConstraintType::Function { params, return_type } => {
                params.iter().any(|p| self.occurs_check_internal(var, p, depth + 1)) ||
                self.occurs_check_internal(var, return_type, depth + 1)
            }
            ConstraintType::List(element_type) => {
                self.occurs_check_internal(var, element_type, depth + 1)
            }
            ConstraintType::Record(fields) => {
                fields.values().any(|field_type| self.occurs_check_internal(var, field_type, depth + 1))
            }
            ConstraintType::Union(types) | ConstraintType::Intersection(types) => {
                types.iter().any(|t| self.occurs_check_internal(var, t, depth + 1))
            }
        }
    }

    /// Get unification statistics
    pub fn get_statistics(&self) -> &UnificationStatistics {
        &self.stats
    }

    /// Reset unifier state
    pub fn reset(&mut self) {
        self.stats = UnificationStatistics::default();
    }
}

impl Default for Unifier {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for UnificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnificationError::TypeMismatch { left, right, reason } => {
                write!(f, "Cannot unify {:?} with {:?}: {}", left, right, reason)
            }
            UnificationError::OccursCheck { variable, type_expr } => {
                write!(f, "Occurs check failed: variable {:?} occurs in {:?}", variable, type_expr)
            }
            UnificationError::ArityMismatch { left_arity, right_arity, context } => {
                write!(f, "Arity mismatch in {}: {} vs {}", context, left_arity, right_arity)
            }
            UnificationError::FieldMismatch { missing_fields, extra_fields } => {
                write!(f, "Field mismatch: missing {:?}, extra {:?}", missing_fields, extra_fields)
            }
            UnificationError::InternalError { message } => {
                write!(f, "Internal unification error: {}", message)
            }
        }
    }
}

impl std::error::Error for UnificationError {} 