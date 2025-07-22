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
    /// Actor creation expressions
    Actor(ActorExpr),
    /// Spawn expressions (spawn async tasks/actors)
    Spawn(SpawnExpr),
    /// Channel expressions (channel creation and operations)
    Channel(ChannelExpr),
    /// Select expressions (select over channels)
    Select(SelectExpr),
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
    /// Formatted string (f-string) - Python specific
    FormattedString(FormattedStringExpr),
    /// List comprehension - Python specific
    ListComprehension(ListComprehensionExpr),
    /// Set comprehension - Python specific
    SetComprehension(SetComprehensionExpr),
    /// Dictionary comprehension - Python specific
    DictComprehension(DictComprehensionExpr),
    /// Generator expression - Python specific
    GeneratorExpression(GeneratorExpressionExpr),
    /// Named expression (walrus operator) - Python specific
    NamedExpression(NamedExpressionExpr),
    /// Starred expression (*args) - Python specific
    StarredExpression(StarredExpressionExpr),
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
    FloorDivide,        // // (Python integer division)
    MatrixMultiply,     // @ (Python matrix multiplication)
    
    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    // Aliases for compatibility
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    
    // Logical
    And,
    Or,
    // Aliases for compatibility
    LogicalAnd,
    LogicalOr,
    
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    // Aliases for compatibility
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    
    // Assignment
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    WalrusAssign,       // := (Python walrus operator)
    
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
    /// Logical not (alias)
    LogicalNot,
    /// Arithmetic negation
    Negate,
    /// Bitwise not
    BitNot,
    /// Bitwise not (alias)
    BitwiseNot,
    /// Reference/address-of
    Reference,
    /// Dereference
    Dereference,
    /// Pre-increment
    PreIncrement,
    /// Post-increment
    PostIncrement,
    /// Pre-decrement
    PreDecrement,
    /// Post-decrement
    PostDecrement,
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

/// Actor creation expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActorExpr {
    /// Actor implementation
    pub actor_impl: Box<AstNode<Expr>>,
    /// Initial capabilities for the actor
    pub capabilities: Vec<AstNode<Expr>>,
    /// Actor configuration
    pub config: Option<Box<AstNode<Expr>>>,
}

/// Spawn expression for creating async tasks or actors
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpawnExpr {
    /// Expression to spawn (async block, actor, etc.)
    pub expression: Box<AstNode<Expr>>,
    /// Spawn mode (task, actor, etc.)
    pub spawn_mode: SpawnMode,
    /// Capabilities for the spawned entity
    pub capabilities: Vec<AstNode<Expr>>,
    /// Optional priority
    pub priority: Option<Box<AstNode<Expr>>>,
}

/// Spawn modes for different concurrency patterns
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpawnMode {
    /// Spawn an async task
    Task,
    /// Spawn an actor
    Actor,
    /// Spawn in a structured scope
    Scoped,
}

/// Channel expression for channel operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelExpr {
    /// Channel operation type
    pub operation: ChannelOperation,
    /// Channel reference (for send/receive)
    pub channel: Option<Box<AstNode<Expr>>>,
    /// Value (for send operations)
    pub value: Option<Box<AstNode<Expr>>>,
    /// Channel type (for creation)
    pub channel_type: Option<Box<AstNode<crate::Type>>>,
    /// Buffer size (for buffered channels)
    pub buffer_size: Option<Box<AstNode<Expr>>>,
}

/// Channel operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelOperation {
    /// Create a new channel
    Create,
    /// Send a value to a channel
    Send,
    /// Receive a value from a channel
    Receive,
    /// Try to send (non-blocking)
    TrySend,
    /// Try to receive (non-blocking)
    TryReceive,
    /// Close a channel
    Close,
}

/// Select expression for selecting over multiple channels
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SelectExpr {
    /// Select arms
    pub arms: Vec<SelectArm>,
    /// Optional default arm (for non-blocking select)
    pub default_arm: Option<SelectArm>,
}

/// Select arm for channel operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SelectArm {
    /// Channel operation pattern
    pub pattern: ChannelPattern,
    /// Guard condition (optional)
    pub guard: Option<Box<AstNode<Expr>>>,
    /// Body expression
    pub body: Box<AstNode<Expr>>,
}

/// Channel pattern for select arms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelPattern {
    /// Receive pattern: channel => value
    Receive {
        channel: Box<AstNode<Expr>>,
        binding: Option<Symbol>,
    },
    /// Send pattern: channel <= value
    Send {
        channel: Box<AstNode<Expr>>,
        value: Box<AstNode<Expr>>,
    },
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

/// Error expression for recovery
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErrorExpr {
    /// Error message
    pub message: String,
}

/// Formatted string expression (Python f-strings)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormattedStringExpr {
    /// String parts (literal text and expressions)
    pub parts: Vec<FStringPart>,
}

/// F-string part
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FStringPart {
    /// Literal text
    Literal(String),
    /// Expression with optional formatting
    Expression {
        expression: AstNode<Expr>,
        conversion: Option<FStringConversion>,
        format_spec: Option<String>,
        debug: bool, // = after expression for debug mode
    },
}

/// F-string conversion types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FStringConversion {
    Str,   // !s
    Repr,  // !r
    Ascii, // !a
}

/// List comprehension expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ListComprehensionExpr {
    /// Element expression
    pub element: Box<AstNode<Expr>>,
    /// Comprehension generators
    pub generators: Vec<ComprehensionGenerator>,
}

/// Set comprehension expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SetComprehensionExpr {
    /// Element expression
    pub element: Box<AstNode<Expr>>,
    /// Comprehension generators
    pub generators: Vec<ComprehensionGenerator>,
}

/// Dictionary comprehension expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DictComprehensionExpr {
    /// Key expression
    pub key: Box<AstNode<Expr>>,
    /// Value expression
    pub value: Box<AstNode<Expr>>,
    /// Comprehension generators
    pub generators: Vec<ComprehensionGenerator>,
}

/// Generator expression
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GeneratorExpressionExpr {
    /// Element expression
    pub element: Box<AstNode<Expr>>,
    /// Comprehension generators
    pub generators: Vec<ComprehensionGenerator>,
}

/// Comprehension generator (for x in iterable if condition)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComprehensionGenerator {
    /// Target variable
    pub target: Symbol,
    /// Iterable expression
    pub iter: AstNode<Expr>,
    /// Conditional filters
    pub ifs: Vec<AstNode<Expr>>,
    /// Whether this is an async generator
    pub is_async: bool,
}

/// Named expression (walrus operator :=)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NamedExpressionExpr {
    /// Target variable name
    pub target: Symbol,
    /// Value expression
    pub value: Box<AstNode<Expr>>,
}

/// Starred expression (*args)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StarredExpressionExpr {
    /// Expression being starred
    pub value: Box<AstNode<Expr>>,
    /// Context (load, store, or del)
    pub context: ExpressionContext,
}

/// Expression context for starred expressions
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExpressionContext {
    /// Load context (reading)
    Load,
    /// Store context (assignment)
    Store,
    /// Delete context (deletion)
    Del,
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
            Self::Yield(_) => "YieldExpr",
            Self::Actor(_) => "ActorExpr",
            Self::Spawn(_) => "SpawnExpr", 
            Self::Channel(_) => "ChannelExpr",
            Self::Select(_) => "SelectExpr",
            Self::Range(_) => "RangeExpr",
            Self::Tuple(_) => "Tuple",
            Self::Block(_) => "Block",
            Self::Return(_) => "Return",
            Self::Break(_) => "Break",
            Self::Continue(_) => "Continue",
            Self::Throw(_) => "Throw",
            Self::Error(_) => "Error",
            Self::FormattedString(_) => "FormattedString",
            Self::ListComprehension(_) => "ListComprehension",
            Self::SetComprehension(_) => "SetComprehension",
            Self::DictComprehension(_) => "DictComprehension",
            Self::GeneratorExpression(_) => "GeneratorExpression",
            Self::NamedExpression(_) => "NamedExpression",
            Self::StarredExpression(_) => "StarredExpression",
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
            Self::Actor(_) => Some("Concurrency"),
            Self::Spawn(_) => Some("Concurrency"),
            Self::Channel(_) => Some("Concurrency"),
            Self::Select(_) => Some("Concurrency"),
            Self::Range(_) => Some("Data Structure"),
            Self::Tuple(_) => Some("Data Structure"),
            Self::Block(_) => Some("Control Flow"),
            Self::Return(_) => Some("Control Flow"),
            Self::Break(_) => Some("Control Flow"),
            Self::Continue(_) => Some("Control Flow"),
            Self::Throw(_) => Some("Error Handling"),
            Self::Error(_) => Some("Error Handling"),
            Self::FormattedString(_) => Some("Data"),
            Self::ListComprehension(_) => Some("Data Structure"),
            Self::SetComprehension(_) => Some("Data Structure"),
            Self::DictComprehension(_) => Some("Data Structure"),
            Self::GeneratorExpression(_) => Some("Generator"),
            Self::NamedExpression(_) => Some("Data"),
            Self::StarredExpression(_) => Some("Data"),
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
            Self::NamedExpression(_) => true, // Walrus operator creates binding
            Self::Binary(BinaryExpr { operator, .. }) => matches!(
                operator,
                BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubtractAssign
                    | BinaryOperator::MultiplyAssign
                    | BinaryOperator::DivideAssign
                    | BinaryOperator::WalrusAssign // Python walrus operator
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
            // Python-specific expressions
            Self::FormattedString(_) => ComplexityClass::Linear, // f-strings can be complex
            Self::ListComprehension(_) | Self::SetComprehension(_) | 
            Self::DictComprehension(_) | Self::GeneratorExpression(_) => ComplexityClass::Linear, // Comprehensions are linear
            Self::NamedExpression(_) => ComplexityClass::Constant, // Walrus operator is constant
            Self::StarredExpression(_) => ComplexityClass::Constant, // Star unpacking is constant
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
            Self::FloorDivide => write!(f, "//"),
            Self::MatrixMultiply => write!(f, "@"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::Less => write!(f, "<"),
            Self::LessEqual => write!(f, "<="),
            Self::Greater => write!(f, ">"),
            Self::GreaterEqual => write!(f, ">="),
            Self::LessThan => write!(f, "<"),
            Self::LessThanOrEqual => write!(f, "<="),
            Self::GreaterThan => write!(f, ">"),
            Self::GreaterThanOrEqual => write!(f, ">="),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::LogicalAnd => write!(f, "&&"),
            Self::LogicalOr => write!(f, "||"),
            Self::BitAnd => write!(f, "&"),
            Self::BitOr => write!(f, "|"),
            Self::BitXor => write!(f, "^"),
            Self::BitwiseAnd => write!(f, "&"),
            Self::BitwiseOr => write!(f, "|"),
            Self::BitwiseXor => write!(f, "^"),
            Self::LeftShift => write!(f, "<<"),
            Self::RightShift => write!(f, ">>"),
            Self::Assign => write!(f, "="),
            Self::AddAssign => write!(f, "+="),
            Self::SubtractAssign => write!(f, "-="),
            Self::MultiplyAssign => write!(f, "*="),
            Self::DivideAssign => write!(f, "/="),
            Self::WalrusAssign => write!(f, ":="),
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
            Self::LogicalNot => write!(f, "!"),
            Self::Negate => write!(f, "-"),
            Self::BitNot => write!(f, "~"),
            Self::BitwiseNot => write!(f, "~"),
            Self::Reference => write!(f, "&"),
            Self::Dereference => write!(f, "*"),
            Self::PreIncrement => write!(f, "++"),
            Self::PostIncrement => write!(f, "++"),
            Self::PreDecrement => write!(f, "--"),
            Self::PostDecrement => write!(f, "--"),
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(lit) => write!(f, "{}", lit.value),
            Expr::Variable(var) => write!(f, "{}", var.name),
            Expr::Binary(binary) => write!(f, "({} {} {})", binary.left.kind, binary.operator, binary.right.kind),
            Expr::Unary(unary) => write!(f, "({}{})", unary.operator, unary.operand.kind),
            Expr::Call(call) => {
                write!(f, "{}(", call.callee.kind)?;
                for (i, arg) in call.arguments.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg.kind)?;
                }
                write!(f, ")")
            }
            Expr::Member(member) => {
                write!(f, "{}.{}", member.object.kind, member.member)
            }
            Expr::Index(index) => {
                write!(f, "{}[{}]", index.object.kind, index.index.kind)
            }
            Expr::Array(array) => {
                write!(f, "[")?;
                for (i, elem) in array.elements.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", elem.kind)?;
                }
                write!(f, "]")
            }
            Expr::Object(obj) => {
                write!(f, "{{")?;
                for (i, field) in obj.fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    match &field.key {
                        ObjectKey::Identifier(id) => write!(f, "{}", id)?,
                        ObjectKey::String(s) => write!(f, "\"{}\"", s)?,
                        ObjectKey::Computed(_) => write!(f, "[computed]")?,
                    }
                    write!(f, ": {}", field.value.kind)?;
                }
                write!(f, "}}")
            }
            Expr::If(if_expr) => {
                write!(f, "if {} then {}", if_expr.condition.kind, if_expr.then_branch.kind)?;
                if let Some(else_branch) = &if_expr.else_branch {
                    write!(f, " else {}", else_branch.kind)?;
                }
                Ok(())
            }
            Expr::Block(block) => {
                write!(f, "{{ ... }}")  // Simplified for now
            }
            _ => write!(f, "<expr>"),  // Fallback for other expression types
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