//! Expression AST nodes for the Prism programming language
//!
//! This module defines all expression types with rich semantic metadata for AI comprehension.

use crate::{node::AstNodeKind, AstNode, AstNodeRef, ComplexityClass};
use prism_common::{span::Span, symbol::Symbol, NodeId, SourceId};
use std::fmt;

/// Expression AST node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expr {
    /// Literal values
    Literal(LiteralExpr),
    /// Variable references
    Variable(VariableExpr),
    /// Binary operations
    Binary(BinaryExpr),
    /// Unary operations
    Unary(UnaryExpr),
    /// Function calls
    Call(CallExpr),
    /// Member access (obj.field)
    Member(MemberExpr),
    /// Index access (obj[index])
    Index(IndexExpr),
    /// Array literals
    Array(ArrayExpr),
    /// Object literals
    Object(ObjectExpr),
    /// Lambda expressions
    Lambda(LambdaExpr),
    /// Match expressions
    Match(MatchExpr),
    /// If expressions
    If(IfExpr),
    /// While expressions
    While(WhileExpr),
    /// For expressions
    For(ForExpr),
    /// Try expressions
    Try(TryExpr),
    /// Type assertions
    TypeAssertion(TypeAssertionExpr),
    /// Await expressions
    Await(AwaitExpr),
    /// Yield expressions
    Yield(YieldExpr),
    /// Range expressions (1..10)
    Range(RangeExpr),
    /// Tuple expressions
    Tuple(TupleExpr),
    /// Block expressions
    Block(BlockExpr),
    /// Return expressions
    Return(ReturnExpr),
    /// Break expressions
    Break(BreakExpr),
    /// Continue expressions
    Continue(ContinueExpr),
    /// Throw expressions
    Throw(ThrowExpr),
    /// Error expression (for recovery)
    Error(ErrorExpr),
}

/// Literal expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LiteralExpr {
    /// The literal value
    pub value: LiteralValue,
}

/// Literal values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LiteralValue {
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Boolean(bool),
    /// Null literal
    Null,
    /// Money literal (e.g., 10.50.USD)
    Money { amount: f64, currency: String },
    /// Duration literal (e.g., 5.minutes)
    Duration { value: f64, unit: String },
    /// Regular expression literal
    Regex(String),
}

/// Variable expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VariableExpr {
    /// Variable name
    pub name: Symbol,
}

/// Binary expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryExpr {
    /// Left operand
    pub left: Box<AstNode<Expr>>,
    /// Binary operator
    pub operator: BinaryOperator,
    /// Right operand
    pub right: Box<AstNode<Expr>>,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    
    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    
    // Logical
    And,
    Or,
    
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
    
    // Assignment
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    
    // Semantic operators (Prism-specific)
    SemanticEqual,      // ===
    TypeCompatible,     // ~=
    ConceptualMatch,    // ≈
    
    // Range operators
    Range,              // ..
    RangeInclusive,     // ..=
}

/// Unary expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnaryExpr {
    /// Unary operator
    pub operator: UnaryOperator,
    /// Operand
    pub operand: Box<AstNode<Expr>>,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnaryOperator {
    /// Logical not
    Not,
    /// Arithmetic negation
    Negate,
    /// Bitwise not
    BitNot,
    /// Reference/address-of
    Reference,
    /// Dereference
    Dereference,
}

/// Function call expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CallExpr {
    /// Function being called
    pub callee: Box<AstNode<Expr>>,
    /// Function arguments
    pub arguments: Vec<AstNode<Expr>>,
    /// Type arguments (for generics)
    pub type_arguments: Option<Vec<AstNode<crate::Type>>>,
    /// Call style for AI comprehension
    pub call_style: CallStyle,
}

/// Call style for AI comprehension
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CallStyle {
    /// Regular function call
    Function,
    /// Method call
    Method,
    /// Constructor call
    Constructor,
    /// Operator call (desugared)
    Operator,
    /// Capability invocation
    Capability,
}

/// Member access expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemberExpr {
    /// Object being accessed
    pub object: Box<AstNode<Expr>>,
    /// Member name
    pub member: Symbol,
    /// Whether this is a safe navigation (?.)
    pub safe_navigation: bool,
}

/// Index access expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexExpr {
    /// Object being indexed
    pub object: Box<AstNode<Expr>>,
    /// Index expression
    pub index: Box<AstNode<Expr>>,
}

/// Array literal expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayExpr {
    /// Array elements
    pub elements: Vec<AstNode<Expr>>,
}

/// Object literal expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectExpr {
    /// Object fields
    pub fields: Vec<ObjectField>,
}

/// Object field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectField {
    /// Field key
    pub key: ObjectKey,
    /// Field value
    pub value: AstNode<Expr>,
}

/// Object field key
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ObjectKey {
    /// Identifier key
    Identifier(Symbol),
    /// String key
    String(String),
    /// Computed key
    Computed(AstNode<Expr>),
}

/// Lambda expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LambdaExpr {
    /// Lambda parameters
    pub parameters: Vec<Parameter>,
    /// Return type annotation
    pub return_type: Option<Box<AstNode<crate::Type>>>,
    /// Lambda body
    pub body: Box<AstNode<Expr>>,
    /// Whether this is an async lambda
    pub is_async: bool,
}

/// Function parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameter {
    /// Parameter name
    pub name: Symbol,
    /// Parameter type
    pub type_annotation: Option<AstNode<crate::Type>>,
    /// Default value
    pub default_value: Option<AstNode<Expr>>,
    /// Whether this parameter is mutable
    pub is_mutable: bool,
}

/// Match expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchExpr {
    /// Expression being matched
    pub scrutinee: Box<AstNode<Expr>>,
    /// Match arms
    pub arms: Vec<MatchArm>,
}

/// Match arm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchArm {
    /// Pattern to match
    pub pattern: AstNode<crate::Pattern>,
    /// Guard condition
    pub guard: Option<AstNode<Expr>>,
    /// Arm body
    pub body: AstNode<Expr>,
}

/// If expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IfExpr {
    /// Condition
    pub condition: Box<AstNode<Expr>>,
    /// Then branch
    pub then_branch: Box<AstNode<Expr>>,
    /// Else branch
    pub else_branch: Option<Box<AstNode<Expr>>>,
}

/// While expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WhileExpr {
    /// Loop condition
    pub condition: Box<AstNode<Expr>>,
    /// Loop body
    pub body: Box<AstNode<Expr>>,
}

/// For expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForExpr {
    /// Loop variable
    pub variable: Symbol,
    /// Iterable expression
    pub iterable: Box<AstNode<Expr>>,
    /// Loop body
    pub body: Box<AstNode<Expr>>,
}

/// Try expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TryExpr {
    /// Try block
    pub try_block: Box<AstNode<Expr>>,
    /// Catch clauses
    pub catch_clauses: Vec<CatchClause>,
    /// Finally block
    pub finally_block: Option<Box<AstNode<Expr>>>,
}

/// Catch clause
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CatchClause {
    /// Exception pattern
    pub pattern: Option<AstNode<crate::Pattern>>,
    /// Catch body
    pub body: AstNode<Expr>,
}

/// Type assertion expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TypeAssertionExpr {
    /// Expression being asserted
    pub expression: Box<AstNode<Expr>>,
    /// Target type
    pub target_type: Box<AstNode<crate::Type>>,
}

/// Await expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AwaitExpr {
    /// Expression being awaited
    pub expression: Box<AstNode<Expr>>,
}

/// Yield expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct YieldExpr {
    /// Value being yielded
    pub value: Option<Box<AstNode<Expr>>>,
}

/// Range expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RangeExpr {
    /// Start of range
    pub start: Box<AstNode<Expr>>,
    /// End of range
    pub end: Box<AstNode<Expr>>,
    /// Whether the range is inclusive
    pub inclusive: bool,
}

/// Tuple expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TupleExpr {
    /// Tuple elements
    pub elements: Vec<AstNode<Expr>>,
}

/// Block expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockExpr {
    /// Block statements
    pub statements: Vec<AstNode<crate::Stmt>>,
    /// Final expression (optional)
    pub final_expr: Option<Box<AstNode<Expr>>>,
}

/// Return expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReturnExpr {
    /// Value being returned
    pub value: Option<Box<AstNode<Expr>>>,
}

/// Break expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BreakExpr {
    /// Value to break with
    pub value: Option<Box<AstNode<Expr>>>,
}

/// Continue expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContinueExpr {
    /// Optional continue value
    pub value: Option<Box<AstNode<Expr>>>,
}

/// Throw expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThrowExpr {
    /// Exception being thrown
    pub exception: Box<AstNode<Expr>>,
}

/// Error expression (for recovery)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErrorExpr {
    /// Error message
    pub message: String,
}

// Implement AstNodeKind for all expression types
impl AstNodeKind for Expr {
    fn node_kind_name(&self) -> &'static str {
        match self {
            Self::Literal(_) => "Literal",
            Self::Variable(_) => "Variable",
            Self::Binary(_) => "Binary",
            Self::Unary(_) => "Unary",
            Self::Call(_) => "Call",
            Self::Member(_) => "Member",
            Self::Index(_) => "Index",
            Self::Array(_) => "Array",
            Self::Object(_) => "Object",
            Self::Lambda(_) => "Lambda",
            Self::Match(_) => "Match",
            Self::If(_) => "If",
            Self::While(_) => "While",
            Self::For(_) => "For",
            Self::Try(_) => "Try",
            Self::TypeAssertion(_) => "TypeAssertion",
            Self::Await(_) => "Await",
            Self::Yield(_) => "Yield",
            Self::Range(_) => "Range",
            Self::Tuple(_) => "Tuple",
            Self::Block(_) => "Block",
            Self::Return(_) => "Return",
            Self::Break(_) => "Break",
            Self::Continue(_) => "Continue",
            Self::Throw(_) => "Throw",
            Self::Error(_) => "Error",
        }
    }

    fn children(&self) -> Vec<AstNodeRef> {
        // This would be implemented with actual node references
        // For now, returning empty vec as placeholder
        Vec::new()
    }

    fn semantic_domain(&self) -> Option<&str> {
        match self {
            Self::Literal(_) => Some("Data"),
            Self::Variable(_) => Some("Data"),
            Self::Binary(_) => Some("Computation"),
            Self::Unary(_) => Some("Computation"),
            Self::Call(_) => Some("Control Flow"),
            Self::Member(_) => Some("Data Access"),
            Self::Index(_) => Some("Data Access"),
            Self::Array(_) => Some("Data Structure"),
            Self::Object(_) => Some("Data Structure"),
            Self::Lambda(_) => Some("Function"),
            Self::Match(_) => Some("Control Flow"),
            Self::If(_) => Some("Control Flow"),
            Self::While(_) => Some("Control Flow"),
            Self::For(_) => Some("Control Flow"),
            Self::Try(_) => Some("Error Handling"),
            Self::TypeAssertion(_) => Some("Type System"),
            Self::Await(_) => Some("Async"),
            Self::Yield(_) => Some("Generator"),
            Self::Range(_) => Some("Data Structure"),
            Self::Tuple(_) => Some("Data Structure"),
            Self::Block(_) => Some("Control Flow"),
            Self::Return(_) => Some("Control Flow"),
            Self::Break(_) => Some("Control Flow"),
            Self::Continue(_) => Some("Control Flow"),
            Self::Throw(_) => Some("Error Handling"),
            Self::Error(_) => Some("Error Handling"),
        }
    }

    fn is_side_effectful(&self) -> bool {
        match self {
            Self::Call(_) => true,
            Self::Await(_) => true,
            Self::Yield(_) => true,
            Self::Return(_) => true,
            Self::Break(_) => true,
            Self::Continue(_) => true,
            Self::Throw(_) => true,
            Self::Binary(BinaryExpr { operator, .. }) => matches!(
                operator,
                BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubtractAssign
                    | BinaryOperator::MultiplyAssign
                    | BinaryOperator::DivideAssign
            ),
            Self::Error(_) => true,
            _ => false,
        }
    }

    fn computational_complexity(&self) -> ComplexityClass {
        match self {
            Self::Literal(_) | Self::Variable(_) => ComplexityClass::Constant,
            Self::Binary(_) | Self::Unary(_) => ComplexityClass::Constant,
            Self::Call(_) => ComplexityClass::Unknown,
            Self::Member(_) | Self::Index(_) => ComplexityClass::Constant,
            Self::Array(ArrayExpr { elements, .. }) => {
                if elements.len() > 100 {
                    ComplexityClass::Linear
                } else {
                    ComplexityClass::Constant
                }
            }
            Self::Object(ObjectExpr { fields, .. }) => {
                if fields.len() > 100 {
                    ComplexityClass::Linear
                } else {
                    ComplexityClass::Constant
                }
            }
            Self::Match(_) => ComplexityClass::Linear,
            Self::While(_) | Self::For(_) => ComplexityClass::Unknown,
            Self::Error(_) => ComplexityClass::Unknown,
            _ => ComplexityClass::Constant,
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Subtract => write!(f, "-"),
            Self::Multiply => write!(f, "*"),
            Self::Divide => write!(f, "/"),
            Self::Modulo => write!(f, "%"),
            Self::Power => write!(f, "**"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::Less => write!(f, "<"),
            Self::LessEqual => write!(f, "<="),
            Self::Greater => write!(f, ">"),
            Self::GreaterEqual => write!(f, ">="),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::BitAnd => write!(f, "&"),
            Self::BitOr => write!(f, "|"),
            Self::BitXor => write!(f, "^"),
            Self::LeftShift => write!(f, "<<"),
            Self::RightShift => write!(f, ">>"),
            Self::Assign => write!(f, "="),
            Self::AddAssign => write!(f, "+="),
            Self::SubtractAssign => write!(f, "-="),
            Self::MultiplyAssign => write!(f, "*="),
            Self::DivideAssign => write!(f, "/="),
            Self::SemanticEqual => write!(f, "==="),
            Self::TypeCompatible => write!(f, "~="),
            Self::ConceptualMatch => write!(f, "≈"),
            Self::Range => write!(f, ".."),
            Self::RangeInclusive => write!(f, "..="),
        }
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Not => write!(f, "not"),
            Self::Negate => write!(f, "-"),
            Self::BitNot => write!(f, "~"),
            Self::Reference => write!(f, "&"),
            Self::Dereference => write!(f, "*"),
        }
    }
}

impl fmt::Display for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(n) => write!(f, "{}", n),
            Self::Float(n) => write!(f, "{}", n),
            Self::String(s) => write!(f, "\"{}\"", s),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::Null => write!(f, "null"),
            Self::Money { amount, currency } => write!(f, "{}.{}", amount, currency),
            Self::Duration { value, unit } => write!(f, "{}.{}", value, unit),
            Self::Regex(pattern) => write!(f, "r\"{}\"", pattern),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::span::Span;

    #[test]
    fn test_binary_operator_display() {
        assert_eq!(BinaryOperator::Add.to_string(), "+");
        assert_eq!(BinaryOperator::And.to_string(), "and");
        assert_eq!(BinaryOperator::SemanticEqual.to_string(), "===");
    }

    #[test]
    fn test_unary_operator_display() {
        assert_eq!(UnaryOperator::Not.to_string(), "not");
        assert_eq!(UnaryOperator::Negate.to_string(), "-");
    }

    #[test]
    fn test_literal_value_display() {
        assert_eq!(LiteralValue::Integer(42).to_string(), "42");
        assert_eq!(LiteralValue::String("hello".to_string()).to_string(), "\"hello\"");
        assert_eq!(LiteralValue::Boolean(true).to_string(), "true");
        assert_eq!(
            LiteralValue::Money {
                amount: 10.50,
                currency: "USD".to_string()
            }
            .to_string(),
            "10.5.USD"
        );
    }

    #[test]
    fn test_expr_node_kind() {
        let literal = Expr::Literal(LiteralExpr {
            value: LiteralValue::Integer(42),
        });
        
        assert_eq!(literal.node_kind_name(), "Literal");
        assert_eq!(literal.semantic_domain(), Some("Data"));
        assert!(!literal.is_side_effectful());
        assert_eq!(literal.computational_complexity(), ComplexityClass::Constant);
    }

    #[test]
    fn test_call_expr_side_effects() {
        let call = Expr::Call(CallExpr {
            callee: Box::new(AstNode::new(
                Expr::Variable(VariableExpr {
                    name: Symbol::intern("test"),
                }),
                Span::dummy(),
                NodeId::new(1),
            )),
            arguments: vec![],
            type_arguments: None,
            call_style: CallStyle::Function,
        });
        
        assert!(call.is_side_effectful());
        assert_eq!(call.computational_complexity(), ComplexityClass::Unknown);
    }
} 