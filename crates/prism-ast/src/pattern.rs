//! Pattern matching AST nodes for the Prism programming language

use crate::{AstNode, Expr};
use prism_common::symbol::Symbol;

/// Pattern AST node for pattern matching
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Pattern {
    /// Wildcard pattern (_)
    Wildcard,
    /// Identifier pattern (binding)
    Identifier(Symbol),
    /// Literal pattern
    Literal(crate::LiteralValue),
    /// Tuple pattern
    Tuple(Vec<AstNode<Pattern>>),
    /// Array pattern
    Array(Vec<AstNode<Pattern>>),
    /// Object pattern
    Object(Vec<ObjectPatternField>),
    /// Or pattern (pattern1 | pattern2)
    Or(Vec<AstNode<Pattern>>),
    /// Rest pattern (...rest)
    Rest(Option<Symbol>),
}

/// Object pattern field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectPatternField {
    /// Field key
    pub key: Symbol,
    /// Field pattern
    pub pattern: AstNode<Pattern>,
    /// Whether this field is optional
    pub optional: bool,
} 