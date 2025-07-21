//! Type Inference Subsystem
//!
//! This module implements a complete type inference engine based on the Hindley-Milner algorithm
//! with modern extensions for semantic types, constraint solving, and AI integration.
//!
//! ## Architecture
//!
//! The type inference system is organized into several key components:
//!
//! - `engine/` - Core type inference engine and orchestration
//! - `constraints/` - Constraint generation and solving
//! - `unification/` - Unification algorithm for type constraints
//! - `environment/` - Type environments and scoping
//! - `semantic/` - Semantic type inference extensions
//! - `errors/` - Type error reporting and diagnostics
//! - `metadata/` - AI metadata generation for inference
//!
//! ## Design Principles
//!
//! 1. **Separation of Concerns**: Each module handles one aspect of type inference
//! 2. **Extensibility**: Easy to add new type constructs and inference rules
//! 3. **Performance**: Efficient algorithms with good complexity characteristics
//! 4. **Error Quality**: Rich error messages with source locations and suggestions
//! 5. **AI Integration**: Generate comprehensive metadata for AI understanding

pub mod engine;
pub mod constraints;
pub mod unification;
pub mod environment;
pub mod semantic;
pub mod errors;
pub mod metadata;

// Re-export main types
pub use engine::{TypeInferenceEngine, InferenceConfig, InferenceResult};
pub use constraints::{ConstraintSolver, TypeConstraint, ConstraintSet};
pub use unification::{Unifier, UnificationResult, Substitution};
pub use environment::{TypeEnvironment, TypeBinding, Scope};
pub use semantic::{SemanticTypeInference, SemanticConstraint};
pub use errors::{TypeError, TypeErrorKind, TypeDiagnostic};
pub use metadata::{InferenceMetadata, TypeInferenceAI};

use crate::SemanticResult;
use prism_common::{NodeId, Span};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Core type variable representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeVar {
    /// Unique identifier for this type variable
    pub id: u32,
    /// Optional name for debugging
    pub name: Option<String>,
    /// Source location where this type variable was created
    pub origin: Span,
}

/// Type variable generator for creating fresh type variables
#[derive(Debug, Clone)]
pub struct TypeVarGenerator {
    next_id: u32,
}

impl TypeVarGenerator {
    /// Create a new type variable generator
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    /// Generate a fresh type variable
    pub fn fresh(&mut self, origin: Span) -> TypeVar {
        let id = self.next_id;
        self.next_id += 1;
        TypeVar {
            id,
            name: None,
            origin,
        }
    }

    /// Generate a fresh type variable with a name
    pub fn fresh_named(&mut self, name: String, origin: Span) -> TypeVar {
        let id = self.next_id;
        self.next_id += 1;
        TypeVar {
            id,
            name: Some(name),
            origin,
        }
    }
}

impl Default for TypeVarGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Inferred type information with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredType {
    /// The actual type that was inferred
    pub type_info: crate::types::SemanticType,
    /// Confidence level in the inference (0.0 to 1.0)
    pub confidence: f64,
    /// How this type was inferred
    pub inference_source: InferenceSource,
    /// Constraints that led to this inference
    pub constraints: Vec<TypeConstraint>,
    /// AI-generated metadata about this type
    pub ai_metadata: Option<InferenceMetadata>,
    /// Source location of the expression that has this type
    pub span: Span,
}

/// Source of a type inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceSource {
    /// Type was explicitly annotated
    Explicit,
    /// Type inferred from literal value
    Literal,
    /// Type inferred from variable usage
    Variable,
    /// Type inferred from function application
    Application,
    /// Type inferred from function definition
    Function,
    /// Type inferred from pattern matching
    Pattern,
    /// Type inferred from operator usage
    Operator,
    /// Type inferred from structural analysis
    Structural,
    /// Type inferred from semantic context
    Semantic,
    /// Type inferred with AI assistance
    AIAssisted,
    /// Default/fallback type
    Default,
}

/// Result of type inference for a program or expression
#[derive(Debug, Clone)]
pub struct TypeInferenceResult {
    /// Types inferred for each AST node
    pub node_types: HashMap<NodeId, InferredType>,
    /// Global type environment after inference
    pub global_env: TypeEnvironment,
    /// Constraints generated during inference
    pub constraints: ConstraintSet,
    /// Substitution found by constraint solving
    pub substitution: Substitution,
    /// Any type errors encountered
    pub errors: Vec<TypeError>,
    /// AI metadata for the entire inference process
    pub ai_metadata: Option<metadata::GlobalInferenceMetadata>,
}

impl TypeInferenceResult {
    /// Create an empty type inference result
    pub fn empty() -> Self {
        Self {
            node_types: HashMap::new(),
            global_env: TypeEnvironment::new(),
            constraints: ConstraintSet::new(),
            substitution: Substitution::empty(),
            errors: Vec::new(),
            ai_metadata: None,
        }
    }

    /// Check if inference was successful (no errors)
    pub fn is_successful(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the inferred type for a specific node
    pub fn get_type(&self, node_id: NodeId) -> Option<&InferredType> {
        self.node_types.get(&node_id)
    }

    /// Add an inferred type for a node
    pub fn add_type(&mut self, node_id: NodeId, inferred_type: InferredType) {
        self.node_types.insert(node_id, inferred_type);
    }

    /// Add a type error
    pub fn add_error(&mut self, error: TypeError) {
        self.errors.push(error);
    }

    /// Merge another inference result into this one
    pub fn merge(&mut self, other: TypeInferenceResult) {
        self.node_types.extend(other.node_types);
        self.constraints.extend(other.constraints);
        self.substitution.compose(other.substitution);
        self.errors.extend(other.errors);
        
        // Merge environments
        self.global_env.merge(other.global_env);
    }
}

/// Statistics about type inference performance and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStatistics {
    /// Number of nodes processed
    pub nodes_processed: usize,
    /// Number of type variables generated
    pub type_vars_generated: usize,
    /// Number of constraints generated
    pub constraints_generated: usize,
    /// Number of unification steps
    pub unification_steps: usize,
    /// Time spent in inference (milliseconds)
    pub inference_time_ms: u64,
    /// Time spent in constraint solving (milliseconds)
    pub constraint_solving_time_ms: u64,
    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,
}

/// Memory usage statistics for type inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Number of type environments created
    pub environments_created: usize,
    /// Number of substitutions created
    pub substitutions_created: usize,
}

impl Default for InferenceStatistics {
    fn default() -> Self {
        Self {
            nodes_processed: 0,
            type_vars_generated: 0,
            constraints_generated: 0,
            unification_steps: 0,
            inference_time_ms: 0,
            constraint_solving_time_ms: 0,
            memory_stats: MemoryStatistics {
                peak_memory_bytes: 0,
                environments_created: 0,
                substitutions_created: 0,
            },
        }
    }
} 