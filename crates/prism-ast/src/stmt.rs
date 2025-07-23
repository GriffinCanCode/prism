//! Statement AST nodes for the Prism programming language

use crate::{AstNode, Expr, Pattern, Type, TypeDecl, Visibility, metadata::AiContext};
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
    /// Actor declaration
    Actor(ActorDecl),
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

/// Actor declaration for PLD-005 concurrency model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActorDecl {
    /// Actor name
    pub name: Symbol,
    /// Actor state fields
    pub state_fields: Vec<Parameter>,
    /// Message type this actor can receive
    pub message_type: Option<AstNode<Type>>,
    /// Message handlers
    pub message_handlers: Vec<MessageHandler>,
    /// Actor capabilities required
    pub capabilities: Vec<AstNode<Expr>>,
    /// Actor effects produced
    pub effects: Vec<AstNode<Expr>>,
    /// Supervision strategy
    pub supervision_strategy: Option<SupervisionStrategy>,
    /// Actor lifecycle hooks
    pub lifecycle_hooks: Vec<LifecycleHook>,
    /// Actor visibility
    pub visibility: Visibility,
    /// Actor attributes
    pub attributes: Vec<Attribute>,
    /// AI context for actor behavior
    pub ai_context: Option<ActorAiContext>,
}

/// Message handler within an actor
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MessageHandler {
    /// Message pattern to match
    pub message_pattern: AstNode<crate::Pattern>,
    /// Handler body
    pub body: Box<AstNode<Stmt>>,
    /// Handler attributes
    pub attributes: Vec<Attribute>,
    /// Effects this handler may produce
    pub effects: Vec<AstNode<Expr>>,
}

/// Supervision strategy for actors
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SupervisionStrategy {
    /// Restart only the failed actor
    OneForOne,
    /// Restart all sibling actors
    OneForAll,
    /// Restart actors in dependency order
    RestForOne,
    /// Custom supervision strategy
    Custom(String),
}

/// Actor lifecycle hooks
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LifecycleHook {
    /// Hook type (start, stop, restart, etc.)
    pub hook_type: LifecycleHookType,
    /// Hook body
    pub body: Box<AstNode<Stmt>>,
}

/// Types of actor lifecycle hooks
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LifecycleHookType {
    /// Called when actor starts
    OnStart,
    /// Called when actor stops
    OnStop,
    /// Called when actor restarts
    OnRestart,
    /// Called on child failure
    OnChildFailure,
}

/// AI context for actor comprehension
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActorAiContext {
    /// Business purpose of this actor
    pub purpose: String,
    /// State management strategy
    pub state_management: String,
    /// Concurrency safety guarantees
    pub concurrency_safety: String,
    /// Performance characteristics
    pub performance_characteristics: String,
    /// Typical message patterns
    pub message_patterns: Vec<String>,
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

/// **ENHANCED**: Smart Module Declaration (PLD-002 compliant)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModuleDecl {
    /// Module name
    pub name: Symbol,
    /// Module capability description
    pub capability: Option<String>,
    /// Module description
    pub description: Option<String>,
    /// Module dependencies with semantic information
    pub dependencies: Vec<String>,
    /// Module stability level
    pub stability: StabilityLevel,
    /// Module version
    pub version: Option<String>,
    /// Module sections (enhanced for PLD-002)
    pub sections: Vec<AstNode<SectionDecl>>,
    /// AI context block for comprehension
    pub ai_context: Option<AiContext>,
    /// Module visibility
    pub visibility: Visibility,
    /// Module annotations/attributes
    pub attributes: Vec<Attribute>,
    /// Sub-modules for large capabilities
    pub submodules: Vec<AstNode<ModuleDecl>>,
    /// Module traits implemented
    pub implemented_traits: Vec<Symbol>,
    /// Lifecycle hooks
    pub lifecycle_hooks: Vec<ModuleLifecycleHook>,
    /// Module events
    pub events: Vec<EventDecl>,
    /// Cohesion metadata (calculated by compiler)
    pub cohesion_metadata: Option<CohesionMetadata>,
    /// Dependency injection configuration
    pub injection_config: Option<InjectionConfig>,
    /// Module composition traits
    pub composition_traits: Vec<CompositionTrait>,
}

/// **NEW**: Module dependency with semantic information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModuleDependency {
    /// Dependency module path
    pub path: String,
    /// Optional alias for the dependency
    pub alias: Option<Symbol>,
    /// Specific items imported from the dependency
    pub items: DependencyItems,
    /// Version requirement
    pub version: Option<String>,
    /// Whether this is a development-only dependency
    pub is_dev_only: bool,
    /// Dependency injection binding
    pub injection_binding: Option<InjectionBinding>,
}

/// **NEW**: Dependency items specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DependencyItems {
    /// Import all items
    All,
    /// Import specific items
    Specific(Vec<DependencyItem>),
}

/// **NEW**: Specific dependency item
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DependencyItem {
    /// Item name
    pub name: Symbol,
    /// Optional alias
    pub alias: Option<Symbol>,
}

/// **NEW**: Dependency injection binding
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InjectionBinding {
    /// Binding name
    pub name: Symbol,
    /// Binding type
    pub binding_type: AstNode<Type>,
    /// Whether this is a singleton
    pub is_singleton: bool,
}

/// **NEW**: Dependency injection configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InjectionConfig {
    /// Injection bindings
    pub bindings: Vec<InjectionBinding>,
    /// Injection scope
    pub scope: InjectionScope,
}

/// **NEW**: Injection scope
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InjectionScope {
    /// Module-scoped injection
    Module,
    /// Function-scoped injection
    Function,
    /// Request-scoped injection
    Request,
    /// Singleton injection
    Singleton,
}

/// **NEW**: Module composition trait
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompositionTrait {
    /// Trait name
    pub name: Symbol,
    /// Trait implementation
    pub implementation: Vec<AstNode<Stmt>>,
}

/// **NEW**: AI Context block for module comprehension
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModuleAiContext {
    /// Primary purpose of this module
    pub purpose: String,
    /// Compliance requirements (e.g., PCI-DSS, GDPR)
    pub compliance: Vec<String>,
    /// Critical execution paths
    pub critical_paths: Vec<CriticalPath>,
    /// Error handling strategy
    pub error_handling: Option<String>,
    /// Performance characteristics
    pub performance_notes: Vec<String>,
    /// Security implications
    pub security_notes: Vec<String>,
    /// AI hints for understanding
    pub ai_hints: Vec<AiHint>,
    /// Business context information
    pub business_context: Vec<String>,
    /// Architecture patterns used
    pub architecture_patterns: Vec<String>,
}

/// **NEW**: Critical execution path description
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CriticalPath {
    /// Path name
    pub name: String,
    /// Path description
    pub description: String,
    /// Requirements for this path
    pub requirements: Vec<String>,
    /// Performance SLA
    pub sla: Option<PerformanceSla>,
}

/// **NEW**: Performance SLA
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerformanceSla {
    /// Maximum response time
    pub max_response_time: String,
    /// Maximum throughput
    pub max_throughput: Option<String>,
    /// Availability requirement
    pub availability: Option<String>,
}

/// **NEW**: AI hint for module understanding
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AiHint {
    /// Hint category
    pub category: AiHintCategory,
    /// Hint content
    pub content: String,
}

/// **NEW**: AI hint categories
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AiHintCategory {
    /// Performance-related hint
    Performance,
    /// Security-related hint
    Security,
    /// Testing-related hint
    Testing,
    /// Business logic hint
    Business,
    /// Architecture hint
    Architecture,
    /// Maintenance hint
    Maintenance,
    /// Debugging hint
    Debugging,
}

/// **NEW**: Module lifecycle hook
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModuleLifecycleHook {
    /// Hook event
    pub event: LifecycleEvent,
    /// Hook body
    pub body: Box<AstNode<Stmt>>,
    /// Hook priority (for ordering)
    pub priority: Option<i32>,
}

/// **NEW**: Module lifecycle events
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LifecycleEvent {
    /// Module load event
    Load,
    /// Module unload event
    Unload,
    /// Module initialization
    Initialize,
    /// Module shutdown
    Shutdown,
    /// Module hot reload
    HotReload,
    /// Module dependency resolved
    DependencyResolved,
}

/// **NEW**: Event declaration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EventDecl {
    /// Event name
    pub name: Symbol,
    /// Event parameters
    pub parameters: Vec<Parameter>,
    /// Event description
    pub description: Option<String>,
    /// Event priority
    pub priority: Option<EventPriority>,
}

/// **NEW**: Event priority levels
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventPriority {
    /// Low priority event
    Low,
    /// Normal priority event
    Normal,
    /// High priority event
    High,
    /// Critical priority event
    Critical,
}

/// **NEW**: Cohesion metadata (calculated by compiler)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CohesionMetadata {
    /// Overall cohesion score (0-100)
    pub overall_score: f64,
    /// Type cohesion score
    pub type_cohesion: f64,
    /// Data flow cohesion score
    pub data_flow_cohesion: f64,
    /// Semantic cohesion score
    pub semantic_cohesion: f64,
    /// Dependency cohesion score
    pub dependency_cohesion: f64,
    /// Cohesion strengths
    pub strengths: Vec<String>,
    /// Cohesion improvement suggestions
    pub suggestions: Vec<String>,
    /// Cohesion analysis timestamp
    pub analyzed_at: Option<String>,
    /// Cohesion trend (improving, declining, stable)
    pub trend: Option<CohesionTrend>,
}

/// **NEW**: Cohesion trend analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CohesionTrend {
    /// Cohesion is improving
    Improving,
    /// Cohesion is declining
    Declining,
    /// Cohesion is stable
    Stable,
    /// Not enough data for trend analysis
    Unknown,
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
    /// Beta
    Beta,
    /// Alpha
    Alpha,
}

/// **ENHANCED**: Section declaration with PLD-002 features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SectionDecl {
    /// Section kind
    pub kind: SectionKind,
    /// Section name (for custom sections)
    pub name: Option<String>,
    /// Section purpose/description
    pub purpose: Option<String>,
    /// Section items
    pub items: Vec<AstNode<Stmt>>,
    /// Section visibility
    pub visibility: Visibility,
    /// Section requirements (for capability-gated sections)
    pub requirements: Vec<SectionRequirement>,
    /// Section attributes
    pub attributes: Vec<Attribute>,
    /// Section-specific AI context
    pub ai_context: Option<AiContext>,
    /// Section performance characteristics
    pub performance_hints: Vec<String>,
    /// Section security implications
    pub security_notes: Vec<String>,
}

/// **ENHANCED**: Section kind with all PLD-002 section types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SectionKind {
    /// Configuration section - module constants and settings
    Config,
    /// Types section - type definitions
    Types,
    /// Errors section - error type definitions
    Errors,
    /// Internal section - private implementation details
    Internal,
    /// Interface section - public API
    Interface,
    /// Performance section - capability-gated optimizations
    Performance,
    /// Events section - event definitions
    Events,
    /// Lifecycle section - module lifecycle hooks
    Lifecycle,
    /// Tests section - inline test cases
    Tests,
    /// Examples section - usage examples
    Examples,
    /// State machine section - state transitions
    StateMachine,
    /// Operations section - business operations
    Operations,
    /// Validation section - validation rules
    Validation,
    /// Migration section - data migration logic
    Migration,
    /// Documentation section - inline documentation
    Documentation,
    /// Custom section with name
    Custom(String),
}

/// **NEW**: Section requirement for capability-gated sections
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SectionRequirement {
    /// Requirement type
    pub kind: RequirementKind,
    /// Requirement value
    pub value: String,
    /// Requirement description
    pub description: Option<String>,
}

/// **NEW**: Section requirement kinds
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RequirementKind {
    /// Capability requirement
    Capability,
    /// Permission requirement
    Permission,
    /// Security level requirement
    SecurityLevel,
    /// Performance requirement
    Performance,
    /// Version requirement
    Version,
    /// Custom requirement
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

/// **NEW**: Export items specification
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExportItems {
    /// Export all items
    All,
    /// Export specific items
    Specific(Vec<ExportItem>),
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
    /// Constant attributes
    pub attributes: Vec<Attribute>,
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

/// Attribute attached to declarations (e.g., @deprecated, @capability)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Attribute {
    /// Attribute name
    pub name: Symbol,
    /// Attribute value (if any)
    pub value: Option<AttributeValue>,
    /// Attribute arguments
    pub arguments: Vec<AttributeArgument>,
}

/// Attribute value
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Array of values
    Array(Vec<AttributeValue>),
    /// Object/map of values
    Object(Vec<(String, AttributeValue)>),
}

/// Attribute argument
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeArgument {
    /// Named argument
    Named { name: Symbol, value: AttributeValue },
    /// Literal argument
    Literal(AttributeValue),
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