//! Canonical form representation for Prism syntax.
//!
//! This module defines the canonical semantic representation that all syntax styles
//! are normalized to, maintaining conceptual cohesion around "canonical structure
//! definition and semantic representation".

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use rustc_hash::FxHashMap;

/// The canonical form of Prism syntax after normalization.
/// 
/// This represents the unified semantic structure that all syntax styles
/// are converted to. It preserves all semantic meaning while providing
/// a consistent format for downstream processing and AI analysis.
/// 
/// # Conceptual Cohesion
/// 
/// The CanonicalForm maintains conceptual cohesion by focusing solely on
/// "semantic structure representation". It contains the essential semantic
/// elements without syntax-specific formatting details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalForm {
    /// Top-level nodes in the canonical representation
    pub nodes: Vec<CanonicalNode>,
    
    /// Preserved metadata from original syntax
    pub metadata: CanonicalMetadata,
    
    /// AI-specific metadata for comprehension
    pub ai_metadata: AIMetadata,
    
    /// Semantic version of the canonical format
    pub semantic_version: String,
    
    /// Hash for semantic equivalence checking
    pub semantic_hash: u64,
}

/// A node in the canonical syntax tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalNode {
    /// Module declaration with sections
    Module {
        /// Module name
        name: String,
        /// Module sections (types, interface, etc.)
        sections: Vec<CanonicalSection>,
        /// Module annotations
        annotations: Vec<Annotation>,
        /// Source span information
        span: CanonicalSpan,
        /// Semantic metadata
        semantic_metadata: NodeSemanticMetadata,
    },
    
    /// Function declaration
    Function {
        /// Function name
        name: String,
        /// Parameters
        parameters: Vec<Parameter>,
        /// Return type
        return_type: Option<CanonicalType>,
        /// Function body
        body: Option<CanonicalExpression>,
        /// Function annotations
        annotations: Vec<Annotation>,
        /// Source span information
        span: CanonicalSpan,
        /// Semantic metadata
        semantic_metadata: NodeSemanticMetadata,
    },
    
    /// Type declaration
    Type {
        /// Type name
        name: String,
        /// Type definition
        definition: CanonicalType,
        /// Type constraints
        constraints: Vec<TypeConstraint>,
        /// Type annotations
        annotations: Vec<Annotation>,
        /// Source span information
        span: CanonicalSpan,
        /// Semantic metadata
        semantic_metadata: NodeSemanticMetadata,
    },
    
    /// Statement
    Statement {
        /// Statement content
        statement: CanonicalStatement,
        /// Source span information
        span: CanonicalSpan,
        /// Semantic metadata
        semantic_metadata: NodeSemanticMetadata,
    },
}

/// A section within a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSection {
    /// Section type (types, interface, internal, etc.)
    pub section_type: SectionType,
    
    /// Items within this section
    pub items: Vec<CanonicalNode>,
    
    /// Section-specific metadata
    pub metadata: SectionMetadata,
}

/// Types of module sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    /// Configuration section
    Config,
    /// Type definitions
    Types,
    /// Error definitions
    Errors,
    /// Internal/private items
    Internal,
    /// Public interface
    Interface,
    /// Event definitions
    Events,
    /// Lifecycle hooks
    Lifecycle,
    /// Tests
    Tests,
    /// Examples
    Examples,
    /// Performance-related items
    Performance,
    /// Custom section
    Custom(String),
}

/// Canonical representation of types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalType {
    /// Primitive type
    Primitive(PrimitiveType),
    
    /// Named type
    Named(String),
    
    /// Generic type with parameters
    Generic {
        base: Box<CanonicalType>,
        parameters: Vec<CanonicalType>,
    },
    
    /// Function type
    Function {
        parameters: Vec<CanonicalType>,
        return_type: Box<CanonicalType>,
    },
    
    /// Tuple type
    Tuple(Vec<CanonicalType>),
    
    /// Record/struct type
    Record(Vec<RecordField>),
    
    /// Union type
    Union(Vec<CanonicalType>),
    
    /// Semantic type with constraints
    Semantic {
        base: Box<CanonicalType>,
        constraints: Vec<SemanticConstraint>,
        business_rules: Vec<String>,
    },
}

/// Primitive types in Prism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimitiveType {
    /// Boolean
    Boolean,
    /// Integer
    Integer,
    /// Float
    Float,
    /// String
    String,
    /// Unit type
    Unit,
}

/// Field in a record type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: CanonicalType,
    /// Whether field is optional
    pub optional: bool,
}

/// Canonical representation of expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalExpression {
    /// Literal value
    Literal(LiteralValue),
    
    /// Variable reference
    Variable(String),
    
    /// Function call
    Call {
        function: Box<CanonicalExpression>,
        arguments: Vec<CanonicalExpression>,
    },
    
    /// Binary operation
    Binary {
        left: Box<CanonicalExpression>,
        operator: BinaryOperator,
        right: Box<CanonicalExpression>,
    },
    
    /// Conditional expression
    Conditional {
        condition: Box<CanonicalExpression>,
        then_branch: Box<CanonicalExpression>,
        else_branch: Option<Box<CanonicalExpression>>,
    },
    
    /// Block expression
    Block(Vec<CanonicalStatement>),
}

/// Canonical representation of statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalStatement {
    /// Expression statement
    Expression(CanonicalExpression),
    
    /// Variable declaration
    Declaration {
        name: String,
        type_annotation: Option<CanonicalType>,
        initializer: Option<CanonicalExpression>,
        mutable: bool,
    },
    
    /// Assignment
    Assignment {
        target: String,
        value: CanonicalExpression,
    },
    
    /// Return statement
    Return(Option<CanonicalExpression>),
    
    /// Control flow
    ControlFlow(ControlFlowStatement),
}

/// Control flow statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFlowStatement {
    /// If statement
    If {
        condition: CanonicalExpression,
        then_block: Vec<CanonicalStatement>,
        else_block: Option<Vec<CanonicalStatement>>,
    },
    
    /// While loop
    While {
        condition: CanonicalExpression,
        body: Vec<CanonicalStatement>,
    },
    
    /// For loop
    For {
        variable: String,
        iterable: CanonicalExpression,
        body: Vec<CanonicalStatement>,
    },
    
    /// Match expression
    Match {
        expression: CanonicalExpression,
        arms: Vec<MatchArm>,
    },
}

/// Match arm in pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchArm {
    /// Pattern to match
    pub pattern: Pattern,
    /// Guard condition
    pub guard: Option<CanonicalExpression>,
    /// Body to execute
    pub body: Vec<CanonicalStatement>,
}

/// Pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    /// Wildcard pattern
    Wildcard,
    /// Literal pattern
    Literal(LiteralValue),
    /// Variable binding
    Variable(String),
    /// Tuple pattern
    Tuple(Vec<Pattern>),
    /// Record pattern
    Record(Vec<(String, Pattern)>),
}

/// Literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiteralValue {
    /// Boolean literal
    Boolean(bool),
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Unit literal
    Unit,
}

/// Binary operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo,
    
    // Comparison
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    
    // Logical
    And, Or,
    
    // Semantic operators
    SemanticEqual,      // Semantic equality
    TypeCompatible,     // Type compatibility
    ConceptualMatch,    // Conceptual similarity
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: CanonicalType,
    /// Default value
    pub default: Option<CanonicalExpression>,
}

/// Annotation (e.g., @responsibility, @aiContext)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation name
    pub name: String,
    /// Annotation value
    pub value: AnnotationValue,
}

/// Annotation value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationValue {
    /// String value
    String(String),
    /// Structured value
    Structured(FxHashMap<String, AnnotationValue>),
    /// List value
    List(Vec<AnnotationValue>),
}

/// Type constraint for semantic types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstraint {
    /// Constraint name
    pub name: String,
    /// Constraint value
    pub value: ConstraintValue,
}

/// Semantic constraint for types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConstraint {
    /// Constraint type
    pub constraint_type: String,
    /// Constraint parameters
    pub parameters: FxHashMap<String, String>,
}

/// Constraint value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintValue {
    /// String constraint
    String(String),
    /// Numeric constraint
    Number(f64),
    /// Boolean constraint
    Boolean(bool),
    /// Pattern constraint
    Pattern(String),
}

/// Span information in canonical form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalSpan {
    /// Start position
    pub start: Position,
    /// End position
    pub end: Position,
    /// Source file identifier
    pub source_id: u32,
}

/// Position in source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
}

/// Overall metadata for canonical form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMetadata {
    /// Original syntax style
    pub original_style: crate::detection::SyntaxStyle,
    /// Normalization timestamp
    pub normalized_at: String,
    /// Preserved formatting hints
    pub formatting_hints: FxHashMap<String, String>,
    /// Source file information
    pub source_info: SourceInfo,
}

/// AI-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadata {
    /// Business context
    pub business_context: Option<String>,
    /// Domain concepts identified
    pub domain_concepts: Vec<String>,
    /// Key relationships
    pub relationships: Vec<String>,
    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,
}

/// Semantic metadata for individual nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSemanticMetadata {
    /// Responsibility of this node
    pub responsibility: Option<String>,
    /// Business rules associated
    pub business_rules: Vec<String>,
    /// AI comprehension hints
    pub ai_hints: Vec<String>,
    /// Documentation quality score
    pub documentation_score: f64,
}

/// Section-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionMetadata {
    /// Section purpose
    pub purpose: Option<String>,
    /// Cohesion score for this section
    pub cohesion_score: f64,
    /// Dependencies within section
    pub dependencies: Vec<String>,
}

/// Source file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// File path
    pub file_path: Option<String>,
    /// File size in bytes
    pub file_size: u64,
    /// File hash for integrity
    pub file_hash: String,
}

/// Complexity metrics for AI analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic: f64,
    /// Cognitive complexity
    pub cognitive: f64,
    /// Nesting depth
    pub nesting_depth: usize,
    /// Number of dependencies
    pub dependencies: usize,
}

/// Canonical structure for organizing related nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalStructure {
    /// Structure type
    pub structure_type: StructureType,
    /// Contained nodes
    pub nodes: Vec<CanonicalNode>,
    /// Structure metadata
    pub metadata: StructureMetadata,
}

/// Types of canonical structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructureType {
    /// Module structure
    Module,
    /// Package structure
    Package,
    /// Library structure
    Library,
    /// Application structure
    Application,
}

/// Metadata for structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureMetadata {
    /// Structure name
    pub name: String,
    /// Version information
    pub version: Option<String>,
    /// Author information
    pub author: Option<String>,
    /// Description
    pub description: Option<String>,
}

impl CanonicalForm {
    /// Create a placeholder canonical form for development
    pub fn placeholder() -> Self {
        Self {
            nodes: Vec::new(),
            metadata: CanonicalMetadata::default(),
            ai_metadata: AIMetadata::default(),
            semantic_version: "0.1.0".to_string(),
            semantic_hash: 0,
        }
    }
    
    /// Calculate semantic hash for equivalence checking
    pub fn semantic_hash(&self) -> u64 {
        // TODO: Implement actual semantic hashing
        self.semantic_hash
    }
    
    /// Update the semantic hash based on content
    pub fn update_semantic_hash(&mut self) {
        // TODO: Implement semantic hash calculation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.nodes).hash(&mut hasher);
        self.semantic_hash = hasher.finish();
    }
}

impl Default for CanonicalMetadata {
    fn default() -> Self {
        Self {
            original_style: crate::detection::SyntaxStyle::Canonical,
            normalized_at: chrono::Utc::now().to_rfc3339(),
            formatting_hints: FxHashMap::default(),
            source_info: SourceInfo::default(),
        }
    }
}

impl Default for AIMetadata {
    fn default() -> Self {
        Self {
            business_context: None,
            domain_concepts: Vec::new(),
            relationships: Vec::new(),
            complexity_metrics: ComplexityMetrics::default(),
        }
    }
}

impl Default for NodeSemanticMetadata {
    fn default() -> Self {
        Self {
            responsibility: None,
            business_rules: Vec::new(),
            ai_hints: Vec::new(),
            documentation_score: 0.0,
        }
    }
}

impl Default for SourceInfo {
    fn default() -> Self {
        Self {
            file_path: None,
            file_size: 0,
            file_hash: String::new(),
        }
    }
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            cyclomatic: 0.0,
            cognitive: 0.0,
            nesting_depth: 0,
            dependencies: 0,
        }
    }
} 