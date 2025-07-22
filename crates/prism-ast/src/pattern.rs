//! Pattern matching AST nodes for the Prism programming language

use crate::{AstNode, AstNodeKind, Expr, node::ComplexityClass, AstNodeRef};
use prism_common::symbol::Symbol;
use std::fmt;

/// Pattern AST node for pattern matching
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Pattern {
    /// Wildcard pattern (_)
    Wildcard,
    /// Identifier pattern (binding)
    Identifier(IdentifierPattern),
    /// Literal pattern
    Literal(LiteralPattern),
    /// Tuple pattern
    Tuple(TuplePattern),
    /// Array pattern
    Array(ArrayPattern),
    /// Object pattern
    Object(ObjectPattern),
    /// Or pattern (pattern1 | pattern2)
    Or(OrPattern),
    /// Rest pattern (...rest)
    Rest(RestPattern),
    /// Guard pattern (pattern if condition)
    Guard(GuardPattern),
    /// Type pattern (pattern: Type)
    Type(TypePattern),
    /// Range pattern (1..10)
    Range(RangePattern),
    /// Constructor pattern (Some(x))
    Constructor(ConstructorPattern),
    /// Slice pattern ([a, b, ..rest])
    Slice(SlicePattern),
    /// Record pattern ({ x, y })
    Record(RecordPattern),
    /// Error pattern (for recovery)
    Error(ErrorPattern),
}

/// Identifier pattern with optional type annotation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IdentifierPattern {
    /// Identifier name
    pub name: Symbol,
    /// Optional type annotation
    pub type_annotation: Option<AstNode<crate::Type>>,
    /// Whether this is a mutable binding
    pub is_mutable: bool,
}

/// Literal pattern for matching literal values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LiteralPattern {
    /// The literal value to match
    pub value: crate::LiteralValue,
}

/// Tuple pattern for matching tuples
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TuplePattern {
    /// Tuple element patterns
    pub elements: Vec<AstNode<Pattern>>,
}

/// Array pattern for matching arrays
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayPattern {
    /// Array element patterns
    pub elements: Vec<AstNode<Pattern>>,
    /// Whether this is an exact match (no extra elements)
    pub exact: bool,
}

/// Object pattern for matching objects/structs
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectPattern {
    /// Object field patterns
    pub fields: Vec<ObjectPatternField>,
    /// Whether this is an exact match (no extra fields)
    pub exact: bool,
}

/// Object pattern field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectPatternField {
    /// Field key
    pub key: ObjectPatternKey,
    /// Field pattern
    pub pattern: AstNode<Pattern>,
    /// Whether this field is optional
    pub optional: bool,
    /// Default value if field is missing
    pub default: Option<AstNode<Expr>>,
}

/// Object pattern field key
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ObjectPatternKey {
    /// Identifier key
    Identifier(Symbol),
    /// String literal key
    String(String),
    /// Computed key expression
    Computed(AstNode<Expr>),
}

/// Or pattern for alternative matches
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OrPattern {
    /// Alternative patterns
    pub alternatives: Vec<AstNode<Pattern>>,
}

/// Rest pattern for capturing remaining elements
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RestPattern {
    /// Optional name to bind the rest to
    pub name: Option<Symbol>,
    /// Type annotation for the rest
    pub type_annotation: Option<AstNode<crate::Type>>,
}

/// Guard pattern with condition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GuardPattern {
    /// Base pattern
    pub pattern: Box<AstNode<Pattern>>,
    /// Guard condition
    pub condition: AstNode<Expr>,
}

/// Type pattern for type checking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypePattern {
    /// Base pattern
    pub pattern: Box<AstNode<Pattern>>,
    /// Type to check against
    pub type_annotation: AstNode<crate::Type>,
}

/// Range pattern for matching ranges
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RangePattern {
    /// Start of range
    pub start: AstNode<Expr>,
    /// End of range
    pub end: AstNode<Expr>,
    /// Whether the range is inclusive
    pub inclusive: bool,
}

/// Constructor pattern for matching constructed values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstructorPattern {
    /// Constructor name/path
    pub constructor: Symbol,
    /// Constructor arguments
    pub arguments: Vec<AstNode<Pattern>>,
    /// Type arguments for generic constructors
    pub type_arguments: Option<Vec<AstNode<crate::Type>>>,
}

/// Slice pattern for matching array slices
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlicePattern {
    /// Prefix patterns
    pub prefix: Vec<AstNode<Pattern>>,
    /// Middle pattern (rest)
    pub middle: Option<Box<AstNode<Pattern>>>,
    /// Suffix patterns
    pub suffix: Vec<AstNode<Pattern>>,
}

/// Record pattern for matching records/structs
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecordPattern {
    /// Record type name
    pub type_name: Option<Symbol>,
    /// Field patterns
    pub fields: Vec<RecordPatternField>,
    /// Whether to allow extra fields
    pub extensible: bool,
}

/// Record pattern field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecordPatternField {
    /// Field name
    pub name: Symbol,
    /// Field pattern (None means shorthand like {x} instead of {x: x})
    pub pattern: Option<AstNode<Pattern>>,
}

/// Error pattern for recovery
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErrorPattern {
    /// Error message
    pub message: String,
    /// Recovery context
    pub context: String,
}

impl AstNodeKind for Pattern {
    fn node_kind_name(&self) -> &'static str {
        match self {
            Pattern::Wildcard => "wildcard_pattern",
            Pattern::Identifier(_) => "identifier_pattern",
            Pattern::Literal(_) => "literal_pattern",
            Pattern::Tuple(_) => "tuple_pattern",
            Pattern::Array(_) => "array_pattern",
            Pattern::Object(_) => "object_pattern",
            Pattern::Or(_) => "or_pattern",
            Pattern::Rest(_) => "rest_pattern",
            Pattern::Guard(_) => "guard_pattern",
            Pattern::Type(_) => "type_pattern",
            Pattern::Range(_) => "range_pattern",
            Pattern::Constructor(_) => "constructor_pattern",
            Pattern::Slice(_) => "slice_pattern",
            Pattern::Record(_) => "record_pattern",
            Pattern::Error(_) => "error_pattern",
        }
    }

    fn children(&self) -> Vec<AstNodeRef> {
        // Return empty vector for now - in a full implementation,
        // this would collect all child node references
        Vec::new()
    }

    fn semantic_domain(&self) -> Option<&str> {
        match self {
            Pattern::Wildcard => Some("pattern_matching"),
            Pattern::Identifier(_) => Some("variable_binding"),
            Pattern::Literal(_) => Some("literal_matching"),
            Pattern::Tuple(_) => Some("structural_matching"),
            Pattern::Array(_) => Some("collection_matching"),
            Pattern::Object(_) => Some("structural_matching"),
            Pattern::Or(_) => Some("alternative_matching"),
            Pattern::Rest(_) => Some("collection_matching"),
            Pattern::Guard(_) => Some("conditional_matching"),
            Pattern::Type(_) => Some("type_matching"),
            Pattern::Range(_) => Some("range_matching"),
            Pattern::Constructor(_) => Some("algebraic_matching"),
            Pattern::Slice(_) => Some("collection_matching"),
            Pattern::Record(_) => Some("structural_matching"),
            Pattern::Error(_) => Some("error_recovery"),
        }
    }

    fn ai_comprehension_hints(&self) -> Vec<String> {
        match self {
            Pattern::Wildcard => vec!["matches_any_value".to_string()],
            Pattern::Identifier(id) => vec![
                format!("binds_to_variable_{}", id.name),
                if id.is_mutable { "mutable_binding".to_string() } else { "immutable_binding".to_string() }
            ],
            Pattern::Literal(lit) => vec![format!("matches_literal_{:?}", lit.value)],
            Pattern::Tuple(tuple) => vec![
                format!("matches_tuple_with_{}_elements", tuple.elements.len()),
                "structural_destructuring".to_string()
            ],
            Pattern::Array(array) => vec![
                format!("matches_array_with_{}_elements", array.elements.len()),
                if array.exact { "exact_length_match".to_string() } else { "flexible_length_match".to_string() }
            ],
            Pattern::Object(obj) => vec![
                format!("matches_object_with_{}_fields", obj.fields.len()),
                if obj.exact { "exact_field_match".to_string() } else { "partial_field_match".to_string() }
            ],
            Pattern::Or(or) => vec![
                format!("matches_one_of_{}_alternatives", or.alternatives.len()),
                "alternative_matching".to_string()
            ],
            Pattern::Rest(_) => vec!["captures_remaining_elements".to_string()],
            Pattern::Guard(_) => vec!["conditional_pattern_match".to_string()],
            Pattern::Type(_) => vec!["type_constrained_match".to_string()],
            Pattern::Range(_) => vec!["range_value_match".to_string()],
            Pattern::Constructor(ctor) => vec![
                format!("matches_constructor_{}", ctor.constructor),
                "algebraic_data_type_match".to_string()
            ],
            Pattern::Slice(_) => vec!["slice_pattern_match".to_string()],
            Pattern::Record(_) => vec!["record_destructuring".to_string()],
            Pattern::Error(_) => vec!["error_recovery_pattern".to_string()],
        }
    }

    fn is_side_effectful(&self) -> bool {
        match self {
            Pattern::Guard(guard) => {
                // Guard conditions might have side effects
                true
            }
            _ => false, // Most patterns are pure
        }
    }

    fn computational_complexity(&self) -> ComplexityClass {
        match self {
            Pattern::Wildcard | Pattern::Identifier(_) | Pattern::Literal(_) => ComplexityClass::Constant,
            Pattern::Tuple(tuple) => {
                if tuple.elements.len() < 10 {
                    ComplexityClass::Constant
                } else {
                    ComplexityClass::Linear
                }
            }
            Pattern::Array(array) => {
                if array.elements.len() < 10 {
                    ComplexityClass::Constant
                } else {
                    ComplexityClass::Linear
                }
            }
            Pattern::Object(obj) => {
                if obj.fields.len() < 10 {
                    ComplexityClass::Constant
                } else {
                    ComplexityClass::Linear
                }
            }
            Pattern::Or(or) => {
                if or.alternatives.len() < 5 {
                    ComplexityClass::Constant
                } else {
                    ComplexityClass::Linear
                }
            }
            Pattern::Guard(_) => ComplexityClass::Linear, // Depends on guard condition
            Pattern::Range(_) => ComplexityClass::Constant,
            Pattern::Constructor(_) => ComplexityClass::Constant,
            Pattern::Slice(_) => ComplexityClass::Linear,
            Pattern::Record(_) => ComplexityClass::Linear,
            _ => ComplexityClass::Constant,
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Identifier(id) => write!(f, "{}", id.name),
            Pattern::Literal(lit) => write!(f, "{}", lit.value),
            Pattern::Tuple(tuple) => {
                write!(f, "(")?;
                for (i, elem) in tuple.elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem.kind)?;
                }
                write!(f, ")")
            }
            Pattern::Array(array) => {
                write!(f, "[")?;
                for (i, elem) in array.elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem.kind)?;
                }
                write!(f, "]")
            }
            Pattern::Object(obj) => {
                write!(f, "{{")?;
                for (i, field) in obj.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match &field.key {
                        ObjectPatternKey::Identifier(id) => write!(f, "{}", id)?,
                        ObjectPatternKey::String(s) => write!(f, "\"{}\"", s)?,
                        ObjectPatternKey::Computed(_) => write!(f, "[computed]")?,
                    }
                    write!(f, ": {}", field.pattern.kind)?;
                }
                write!(f, "}}")
            }
            Pattern::Or(or) => {
                for (i, alt) in or.alternatives.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", alt.kind)?;
                }
                Ok(())
            }
            Pattern::Rest(rest) => {
                write!(f, "...")?;
                if let Some(name) = &rest.name {
                    write!(f, "{}", name)?;
                }
                Ok(())
            }
            Pattern::Guard(guard) => {
                write!(f, "{} if {:?}", guard.pattern.kind, guard.condition.kind)
            }
            Pattern::Type(type_pat) => {
                write!(f, "{}: {:?}", type_pat.pattern.kind, type_pat.type_annotation.kind)
            }
            Pattern::Range(range) => {
                write!(f, "{:?}..{}{:?}", 
                    range.start.kind, 
                    if range.inclusive { "=" } else { "" }, 
                    range.end.kind
                )
            }
            Pattern::Constructor(ctor) => {
                write!(f, "{}(", ctor.constructor)?;
                for (i, arg) in ctor.arguments.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg.kind)?;
                }
                write!(f, ")")
            }
            Pattern::Slice(slice) => {
                write!(f, "[")?;
                for (i, pat) in slice.prefix.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", pat.kind)?;
                }
                if let Some(middle) = &slice.middle {
                    if !slice.prefix.is_empty() {
                        write!(f, ", ")?;
                    }
                    write!(f, "..{}", middle.kind)?;
                }
                if !slice.suffix.is_empty() {
                    if !slice.prefix.is_empty() || slice.middle.is_some() {
                        write!(f, ", ")?;
                    }
                    for (i, pat) in slice.suffix.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", pat.kind)?;
                    }
                }
                write!(f, "]")
            }
            Pattern::Record(record) => {
                if let Some(type_name) = &record.type_name {
                    write!(f, "{} ", type_name)?;
                }
                write!(f, "{{")?;
                for (i, field) in record.fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", field.name)?;
                    if let Some(pattern) = &field.pattern {
                        write!(f, ": {}", pattern.kind)?;
                    }
                }
                write!(f, "}}")
            }
            Pattern::Error(err) => write!(f, "error({})", err.message),
        }
    }
}

/// Helper trait for pattern analysis
pub trait PatternAnalysis {
    /// Check if pattern is exhaustive for a given type
    fn is_exhaustive(&self, type_info: Option<&crate::Type>) -> bool;
    
    /// Get all variables bound by this pattern
    fn bound_variables(&self) -> Vec<Symbol>;
    
    /// Check if pattern is irrefutable (always matches)
    fn is_irrefutable(&self) -> bool;
    
    /// Get the complexity of matching this pattern
    fn match_complexity(&self) -> ComplexityClass;
}

impl PatternAnalysis for Pattern {
    fn is_exhaustive(&self, _type_info: Option<&crate::Type>) -> bool {
        match self {
            Pattern::Wildcard => true,
            Pattern::Identifier(_) => true,
            Pattern::Rest(_) => true,
            Pattern::Or(or) => {
                // Would need type information to determine exhaustiveness
                or.alternatives.len() > 10 // Heuristic
            }
            _ => false, // Most patterns are not exhaustive by themselves
        }
    }
    
    fn bound_variables(&self) -> Vec<Symbol> {
        match self {
            Pattern::Wildcard => Vec::new(),
            Pattern::Identifier(id) => vec![id.name],
            Pattern::Literal(_) => Vec::new(),
            Pattern::Tuple(tuple) => {
                tuple.elements.iter()
                    .flat_map(|elem| elem.kind.bound_variables())
                    .collect()
            }
            Pattern::Array(array) => {
                array.elements.iter()
                    .flat_map(|elem| elem.kind.bound_variables())
                    .collect()
            }
            Pattern::Object(obj) => {
                obj.fields.iter()
                    .flat_map(|field| field.pattern.kind.bound_variables())
                    .collect()
            }
            Pattern::Or(or) => {
                // For or patterns, only variables bound in all alternatives are available
                if let Some(first) = or.alternatives.first() {
                    let first_vars = first.kind.bound_variables();
                    first_vars.into_iter()
                        .filter(|var| {
                            or.alternatives.iter().skip(1).all(|alt| {
                                alt.kind.bound_variables().contains(var)
                            })
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            }
            Pattern::Rest(rest) => {
                rest.name.map(|name| vec![name]).unwrap_or_default()
            }
            Pattern::Guard(guard) => guard.pattern.kind.bound_variables(),
            Pattern::Type(type_pat) => type_pat.pattern.kind.bound_variables(),
            Pattern::Range(_) => Vec::new(),
            Pattern::Constructor(ctor) => {
                ctor.arguments.iter()
                    .flat_map(|arg| arg.kind.bound_variables())
                    .collect()
            }
            Pattern::Slice(slice) => {
                let mut vars = Vec::new();
                vars.extend(slice.prefix.iter().flat_map(|p| p.kind.bound_variables()));
                if let Some(middle) = &slice.middle {
                    vars.extend(middle.kind.bound_variables());
                }
                vars.extend(slice.suffix.iter().flat_map(|p| p.kind.bound_variables()));
                vars
            }
            Pattern::Record(record) => {
                record.fields.iter()
                    .flat_map(|field| {
                        if let Some(pattern) = &field.pattern {
                            pattern.kind.bound_variables()
                        } else {
                            vec![field.name] // Shorthand binding
                        }
                    })
                    .collect()
            }
            Pattern::Error(_) => Vec::new(),
        }
    }
    
    fn is_irrefutable(&self) -> bool {
        match self {
            Pattern::Wildcard => true,
            Pattern::Identifier(_) => true,
            Pattern::Rest(_) => true,
            Pattern::Tuple(tuple) => {
                tuple.elements.iter().all(|elem| elem.kind.is_irrefutable())
            }
            Pattern::Array(array) => {
                !array.exact && array.elements.iter().all(|elem| elem.kind.is_irrefutable())
            }
            Pattern::Object(obj) => {
                !obj.exact && obj.fields.iter().all(|field| {
                    field.optional || field.pattern.kind.is_irrefutable()
                })
            }
            Pattern::Guard(_) => false, // Guards can fail
            Pattern::Type(type_pat) => type_pat.pattern.kind.is_irrefutable(),
            _ => false, // Most other patterns can fail to match
        }
    }
    
    fn match_complexity(&self) -> ComplexityClass {
        self.computational_complexity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::{span::Span, NodeId};

    #[test]
    fn test_pattern_display() {
        let wildcard = Pattern::Wildcard;
        assert_eq!(wildcard.to_string(), "_");
        
        let id_pattern = Pattern::Identifier(IdentifierPattern {
            name: Symbol::intern("x"),
            type_annotation: None,
            is_mutable: false,
        });
        assert_eq!(id_pattern.to_string(), "x");
    }
    
    #[test]
    fn test_pattern_bound_variables() {
        let tuple_pattern = Pattern::Tuple(TuplePattern {
            elements: vec![
                AstNode::new(
                    Pattern::Identifier(IdentifierPattern {
                        name: Symbol::intern("x"),
                        type_annotation: None,
                        is_mutable: false,
                    }),
                    Span::dummy(),
                    NodeId::new(1),
                ),
                AstNode::new(
                    Pattern::Identifier(IdentifierPattern {
                        name: Symbol::intern("y"),
                        type_annotation: None,
                        is_mutable: false,
                    }),
                    Span::dummy(),
                    NodeId::new(2),
                ),
            ],
        });
        
        let vars = tuple_pattern.bound_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&Symbol::intern("x")));
        assert!(vars.contains(&Symbol::intern("y")));
    }
    
    #[test]
    fn test_pattern_irrefutable() {
        assert!(Pattern::Wildcard.is_irrefutable());
        assert!(Pattern::Identifier(IdentifierPattern {
            name: Symbol::intern("x"),
            type_annotation: None,
            is_mutable: false,
        }).is_irrefutable());
        assert!(!Pattern::Literal(LiteralPattern {
            value: crate::LiteralValue::Integer(42),
        }).is_irrefutable());
    }
    
    #[test]
    fn test_pattern_node_kind() {
        let pattern = Pattern::Wildcard;
        assert_eq!(pattern.node_kind_name(), "wildcard_pattern");
        assert_eq!(pattern.semantic_domain(), Some("pattern_matching"));
        assert!(!pattern.is_side_effectful());
        assert_eq!(pattern.computational_complexity(), ComplexityClass::Constant);
    }
} 