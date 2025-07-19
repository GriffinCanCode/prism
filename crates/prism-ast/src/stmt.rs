//! Statement AST nodes for the Prism programming language

use crate::{AstNode, Expr, Pattern, Type, TypeDecl, Visibility};
use prism_common::symbol::Symbol;

/// Statement AST node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Stmt {
    /// Expression statement
    Expression(ExpressionStmt),
    /// Variable declaration
    Variable(VariableDecl),
    /// Function declaration
    Function(FunctionDecl),
    /// Type declaration
    Type(TypeDecl),
    /// Module declaration
    Module(ModuleDecl),
    /// Section declaration
    Section(SectionDecl),
    /// Import statement
    Import(ImportDecl),
    /// Export statement
    Export(ExportDecl),
    /// Constant declaration
    Const(ConstDecl),
    /// If statement
    If(IfStmt),
    /// While statement
    While(WhileStmt),
    /// For statement
    For(ForStmt),
    /// Match statement
    Match(MatchStmt),
    /// Return statement
    Return(ReturnStmt),
    /// Break statement
    Break(BreakStmt),
    /// Continue statement
    Continue(ContinueStmt),
    /// Throw statement
    Throw(ThrowStmt),
    /// Try statement
    Try(TryStmt),
    /// Block statement
    Block(BlockStmt),
    /// Error statement (for recovery)
    Error(ErrorStmt),
}

/// Expression statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExpressionStmt {
    /// Expression
    pub expression: AstNode<Expr>,
}

/// Variable declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VariableDecl {
    /// Variable name
    pub name: Symbol,
    /// Variable type
    pub type_annotation: Option<AstNode<Type>>,
    /// Initial value
    pub initializer: Option<AstNode<Expr>>,
    /// Whether the variable is mutable
    pub is_mutable: bool,
    /// Visibility
    pub visibility: Visibility,
}

/// Function declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FunctionDecl {
    /// Function name
    pub name: Symbol,
    /// Function parameters
    pub parameters: Vec<Parameter>,
    /// Return type
    pub return_type: Option<AstNode<Type>>,
    /// Function body
    pub body: Option<Box<AstNode<Stmt>>>,
    /// Visibility
    pub visibility: Visibility,
    /// Function attributes
    pub attributes: Vec<Attribute>,
    /// Contracts (pre/post conditions)
    pub contracts: Option<Contracts>,
    /// Whether the function is async
    pub is_async: bool,
}

/// Function parameter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Parameter {
    /// Parameter name
    pub name: Symbol,
    /// Parameter type
    pub type_annotation: Option<AstNode<Type>>,
    /// Default value
    pub default_value: Option<AstNode<Expr>>,
    /// Whether the parameter is mutable
    pub is_mutable: bool,
}

/// Function contracts
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contracts {
    /// Preconditions
    pub requires: Vec<AstNode<Expr>>,
    /// Postconditions
    pub ensures: Vec<AstNode<Expr>>,
    /// Invariants
    pub invariants: Vec<AstNode<Expr>>,
}

/// Module declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModuleDecl {
    /// Module name
    pub name: Symbol,
    /// Module capability
    pub capability: Option<String>,
    /// Module description
    pub description: Option<String>,
    /// Module dependencies
    pub dependencies: Vec<String>,
    /// Module stability level
    pub stability: StabilityLevel,
    /// Module version
    pub version: Option<String>,
    /// Module sections
    pub sections: Vec<AstNode<SectionDecl>>,
    /// AI context
    pub ai_context: Option<String>,
    /// Module visibility
    pub visibility: Visibility,
}

/// Module stability level
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StabilityLevel {
    /// Experimental
    Experimental,
    /// Stable
    Stable,
    /// Deprecated
    Deprecated,
}

/// Section declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SectionDecl {
    /// Section kind
    pub kind: SectionKind,
    /// Section items
    pub items: Vec<AstNode<Stmt>>,
    /// Section visibility
    pub visibility: Visibility,
}

/// Section kind
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SectionKind {
    /// Configuration section
    Config,
    /// Types section
    Types,
    /// Errors section
    Errors,
    /// Internal section
    Internal,
    /// Interface section
    Interface,
    /// Events section
    Events,
    /// Lifecycle section
    Lifecycle,
    /// Tests section
    Tests,
    /// Examples section
    Examples,
    /// Custom section
    Custom(String),
}

/// Import declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImportDecl {
    /// Import path
    pub path: String,
    /// Import items
    pub items: ImportItems,
    /// Import alias
    pub alias: Option<Symbol>,
}

/// Import items
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImportItems {
    /// Import all items
    All,
    /// Import specific items
    Specific(Vec<ImportItem>),
}

/// Import item
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImportItem {
    /// Item name
    pub name: Symbol,
    /// Item alias
    pub alias: Option<Symbol>,
}

/// Export declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExportDecl {
    /// Export items
    pub items: Vec<ExportItem>,
}

/// Export item
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExportItem {
    /// Item name
    pub name: Symbol,
    /// Item alias
    pub alias: Option<Symbol>,
}

/// Constant declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstDecl {
    /// Constant name
    pub name: Symbol,
    /// Constant type
    pub type_annotation: Option<AstNode<Type>>,
    /// Constant value
    pub value: AstNode<Expr>,
    /// Visibility
    pub visibility: Visibility,
}

/// If statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IfStmt {
    /// Condition
    pub condition: AstNode<Expr>,
    /// Then branch
    pub then_branch: Box<AstNode<Stmt>>,
    /// Else branch
    pub else_branch: Option<Box<AstNode<Stmt>>>,
}

/// While statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WhileStmt {
    /// Condition
    pub condition: AstNode<Expr>,
    /// Body
    pub body: Box<AstNode<Stmt>>,
}

/// For statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForStmt {
    /// Iterator variable
    pub variable: Symbol,
    /// Iterable expression
    pub iterable: AstNode<Expr>,
    /// Body
    pub body: Box<AstNode<Stmt>>,
}

/// Match statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchStmt {
    /// Expression to match
    pub expression: AstNode<Expr>,
    /// Match arms
    pub arms: Vec<MatchArm>,
}

/// Match arm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatchArm {
    /// Pattern
    pub pattern: AstNode<Pattern>,
    /// Guard condition
    pub guard: Option<AstNode<Expr>>,
    /// Body
    pub body: Box<AstNode<Stmt>>,
}

/// Return statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReturnStmt {
    /// Return value
    pub value: Option<AstNode<Expr>>,
}

/// Break statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BreakStmt {
    /// Break value
    pub value: Option<AstNode<Expr>>,
}

/// Continue statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContinueStmt {
    /// Continue value
    pub value: Option<AstNode<Expr>>,
}

/// Throw statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThrowStmt {
    /// Exception being thrown
    pub exception: AstNode<Expr>,
}

/// Try statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TryStmt {
    /// Try block
    pub try_block: Box<AstNode<Stmt>>,
    /// Catch clauses
    pub catch_clauses: Vec<CatchClause>,
    /// Finally block
    pub finally_block: Option<Box<AstNode<Stmt>>>,
}

/// Catch clause
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CatchClause {
    /// Error variable
    pub variable: Option<Symbol>,
    /// Error type
    pub error_type: Option<AstNode<Type>>,
    /// Body
    pub body: Box<AstNode<Stmt>>,
}

/// Block statement
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockStmt {
    /// Statements in the block
    pub statements: Vec<AstNode<Stmt>>,
}

/// Attribute
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Attribute {
    /// Attribute name
    pub name: String,
    /// Attribute arguments
    pub arguments: Vec<AttributeArgument>,
}

/// Attribute argument
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeArgument {
    /// Literal argument
    Literal(crate::LiteralValue),
    /// Named argument
    Named { name: String, value: crate::LiteralValue },
}

/// Error statement for recovery
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ErrorStmt {
    /// Error message
    pub message: String,
    /// Recovery context
    pub context: String,
}

impl Default for StabilityLevel {
    fn default() -> Self {
        Self::Experimental
    }
}

impl Default for Visibility {
    fn default() -> Self {
        Self::Private
    }
} 