//! Effect Inference Engine
//!
//! This module implements effect inference for type inference, integrating with the
//! prism-effects system to analyze and track computational effects during type inference.
//!
//! **Single Responsibility**: Effect analysis during type inference
//! **What it does**: Infer effects from expressions, track effect flow, integrate with effect system
//! **What it doesn't do**: Manage effect definitions, enforce capabilities, handle effect execution

use super::{InferenceEngine, EffectAware};
use crate::{
    SemanticResult, SemanticError,
    types::SemanticType,
    type_inference::{
        InferredType, InferenceSource,
        constraints::{ConstraintSet, TypeConstraint, ConstraintType, ConstraintReason},
        environment::TypeEnvironment,
    },
};
use prism_ast::{Expr, Stmt, Pattern, FunctionDecl, CallExpr, BinaryOperator, UnaryOperator};
use prism_common::{NodeId, Span};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Information about control flow for effect analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowInfo {
    /// Entry points to the flow
    pub entry_points: Vec<String>,
    /// Exit points from the flow
    pub exit_points: Vec<String>,
    /// Branching information
    pub branches: Vec<String>,
}

/// Engine for inferring effects during type inference
#[derive(Debug)]
pub struct EffectInferenceEngine {
    /// Configuration for effect inference
    config: EffectInferenceConfig,
    /// Effect registry for lookups (simplified interface to avoid circular deps)
    effect_registry: EffectRegistryInterface,
    /// Effect flow analyzer
    flow_analyzer: EffectFlowAnalyzer,
    /// Effect constraint generator
    constraint_generator: EffectConstraintGenerator,
    /// Effect composition analyzer
    composition_analyzer: EffectCompositionAnalyzer,
    /// Built-in effect mappings
    builtin_effects: BuiltinEffectMappings,
    /// Current effect context
    current_context: EffectInferenceContext,
}

/// Configuration for effect inference
#[derive(Debug, Clone)]
pub struct EffectInferenceConfig {
    /// Enable detailed effect analysis
    pub enable_detailed_analysis: bool,
    /// Enable effect flow tracking
    pub enable_flow_tracking: bool,
    /// Enable effect composition analysis
    pub enable_composition_analysis: bool,
    /// Enable constraint generation for effects
    pub enable_constraint_generation: bool,
    /// Maximum depth for effect analysis
    pub max_analysis_depth: usize,
    /// Enable integration with prism-effects
    pub enable_effects_integration: bool,
}

/// Simplified interface to effect registry to avoid circular dependencies
#[derive(Debug)]
struct EffectRegistryInterface {
    /// Known effect definitions
    known_effects: HashMap<String, EffectDefinition>,
    /// Effect categories
    effect_categories: HashMap<String, EffectCategory>,
    /// Built-in effect mappings
    builtin_mappings: HashMap<String, Vec<String>>,
}

/// Simplified effect definition for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EffectDefinition {
    /// Effect name
    name: String,
    /// Effect category
    category: EffectCategory,
    /// Effect parameters
    parameters: Vec<EffectParameter>,
    /// Capability requirements
    capabilities: Vec<String>,
    /// Effect description
    description: String,
}

/// Effect category for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
enum EffectCategory {
    /// I/O effects
    IO,
    /// State mutation effects
    State,
    /// Memory allocation effects
    Memory,
    /// Exception effects
    Exception,
    /// Concurrency effects
    Concurrency,
    /// Security effects
    Security,
    /// Network effects
    Network,
    /// File system effects
    FileSystem,
    /// Database effects
    Database,
    /// User-defined effects
    UserDefined(String),
}

/// Effect parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EffectParameter {
    /// Parameter name
    name: String,
    /// Parameter type
    param_type: String,
    /// Parameter description
    description: String,
}

/// Effect flow analyzer for tracking effect propagation
#[derive(Debug)]
struct EffectFlowAnalyzer {
    /// Flow graph
    flow_graph: EffectFlowGraph,
    /// Flow constraints
    flow_constraints: Vec<EffectFlowConstraint>,
    /// Current flow context
    current_context: FlowAnalysisContext,
}

/// Effect flow graph
#[derive(Debug)]
struct EffectFlowGraph {
    /// Nodes in the flow graph
    nodes: HashMap<NodeId, EffectFlowNode>,
    /// Edges representing effect flow
    edges: Vec<EffectFlowEdge>,
    /// Entry points
    entry_points: Vec<NodeId>,
    /// Exit points
    exit_points: Vec<NodeId>,
}

/// Node in the effect flow graph
#[derive(Debug, Clone)]
struct EffectFlowNode {
    /// Node identifier
    node_id: NodeId,
    /// Effects produced by this node
    produced_effects: Vec<InferredEffect>,
    /// Effects consumed by this node
    consumed_effects: Vec<InferredEffect>,
    /// Node type (expression, statement, etc.)
    node_type: String,
    /// Source location
    span: Span,
}

/// Edge in the effect flow graph
#[derive(Debug, Clone)]
struct EffectFlowEdge {
    /// Source node
    source: NodeId,
    /// Target node
    target: NodeId,
    /// Effects flowing along this edge
    flowing_effects: Vec<InferredEffect>,
    /// Edge type
    edge_type: EffectFlowEdgeType,
}

/// Types of effect flow edges
#[derive(Debug, Clone)]
enum EffectFlowEdgeType {
    /// Sequential flow
    Sequential,
    /// Conditional flow
    Conditional,
    /// Exception flow
    Exception,
    /// Function call flow
    FunctionCall,
    /// Return flow
    Return,
    /// Loop flow
    Loop,
}

/// Effect flow constraint
#[derive(Debug, Clone)]
struct EffectFlowConstraint {
    /// Constraint type
    constraint_type: EffectFlowConstraintType,
    /// Source node
    source: NodeId,
    /// Target node
    target: Option<NodeId>,
    /// Constraint description
    description: String,
    /// Constraint priority
    priority: u32,
}

/// Types of effect flow constraints
#[derive(Debug, Clone)]
enum EffectFlowConstraintType {
    /// Effects must flow from source to target
    MustFlow,
    /// Effects cannot flow from source to target
    CannotFlow,
    /// Effects must be contained within a scope
    MustContain,
    /// Effects must be handled
    MustHandle,
    /// Effects must have specific capabilities
    RequiresCapability,
}

/// Context for flow analysis
#[derive(Debug)]
struct FlowAnalysisContext {
    /// Current scope depth
    scope_depth: usize,
    /// Active effect handlers
    active_handlers: Vec<String>,
    /// Current capability context
    capability_context: Vec<String>,
    /// Exception handling context
    exception_context: Option<String>,
}

/// Effect constraint generator
#[derive(Debug)]
struct EffectConstraintGenerator {
    /// Generated constraints
    generated_constraints: Vec<EffectTypeConstraint>,
    /// Constraint templates
    constraint_templates: HashMap<String, EffectConstraintTemplate>,
}

/// Effect-related type constraint
#[derive(Debug, Clone)]
struct EffectTypeConstraint {
    /// Base type constraint
    base_constraint: TypeConstraint,
    /// Effect requirements
    effect_requirements: Vec<EffectRequirement>,
    /// Capability requirements
    capability_requirements: Vec<String>,
    /// Constraint source
    source: EffectConstraintSource,
}

/// Effect requirement
#[derive(Debug, Clone)]
struct EffectRequirement {
    /// Required effect
    effect: String,
    /// Requirement type
    requirement_type: EffectRequirementType,
    /// Conditions for requirement
    conditions: Vec<String>,
}

/// Types of effect requirements
#[derive(Debug, Clone)]
enum EffectRequirementType {
    /// Effect must be present
    MustHave,
    /// Effect must not be present
    MustNotHave,
    /// Effect may be present
    MayHave,
    /// Effect must be handled
    MustHandle,
}

/// Source of effect constraint
#[derive(Debug, Clone)]
enum EffectConstraintSource {
    /// Function signature
    FunctionSignature,
    /// Expression analysis
    Expression,
    /// Statement analysis
    Statement,
    /// Pattern analysis
    Pattern,
    /// Built-in operation
    BuiltinOperation,
}

/// Template for generating effect constraints
#[derive(Debug, Clone)]
struct EffectConstraintTemplate {
    /// Template name
    name: String,
    /// Effect patterns
    effect_patterns: Vec<String>,
    /// Constraint generation rules
    generation_rules: Vec<ConstraintGenerationRule>,
}

/// Rule for generating constraints
#[derive(Debug, Clone)]
struct ConstraintGenerationRule {
    /// Rule condition
    condition: String,
    /// Generated effect requirements
    effect_requirements: Vec<EffectRequirement>,
    /// Generated capability requirements
    capability_requirements: Vec<String>,
}

/// Effect composition analyzer
#[derive(Debug)]
struct EffectCompositionAnalyzer {
    /// Composition rules
    composition_rules: Vec<EffectCompositionRule>,
    /// Composition cache
    composition_cache: HashMap<Vec<String>, CompositionResult>,
}

/// Effect composition rule
#[derive(Debug, Clone)]
struct EffectCompositionRule {
    /// Rule name
    name: String,
    /// Input effect patterns
    input_patterns: Vec<String>,
    /// Output effect
    output_effect: String,
    /// Composition operator
    operator: CompositionOperator,
    /// Rule conditions
    conditions: Vec<String>,
}

/// Composition operators
#[derive(Debug, Clone)]
enum CompositionOperator {
    /// Sequential composition
    Sequential,
    /// Parallel composition
    Parallel,
    /// Alternative composition
    Alternative,
    /// Exception composition
    Exception,
    /// Loop composition
    Loop,
}

/// Result of effect composition
#[derive(Debug, Clone)]
struct CompositionResult {
    /// Composed effect
    composed_effect: String,
    /// Composition confidence
    confidence: f64,
    /// Composition metadata
    metadata: CompositionMetadata,
}

/// Metadata about effect composition
#[derive(Debug, Clone)]
struct CompositionMetadata {
    /// Rules applied
    rules_applied: Vec<String>,
    /// Composition complexity
    complexity: f64,
    /// Potential optimizations
    optimizations: Vec<String>,
}

/// Built-in effect mappings
#[derive(Debug)]
struct BuiltinEffectMappings {
    /// Function to effects mapping
    function_effects: HashMap<String, Vec<String>>,
    /// Operator to effects mapping
    operator_effects: HashMap<String, Vec<String>>,
    /// Statement to effects mapping
    statement_effects: HashMap<String, Vec<String>>,
}

/// Current effect inference context
#[derive(Debug)]
struct EffectInferenceContext {
    /// Current function context
    function_context: Option<FunctionEffectContext>,
    /// Current scope effects
    scope_effects: Vec<String>,
    /// Active effect handlers
    active_handlers: Vec<String>,
    /// Current capability context
    capabilities: Vec<String>,
}

/// Function-specific effect context
#[derive(Debug)]
struct FunctionEffectContext {
    /// Function name
    function_name: String,
    /// Declared effects
    declared_effects: Vec<String>,
    /// Inferred effects
    inferred_effects: Vec<String>,
    /// Effect parameters
    effect_parameters: Vec<String>,
}

/// Result of effect inference for an expression
#[derive(Debug, Clone)]
pub struct EffectInferenceResult {
    /// Inferred effects
    pub effects: Vec<InferredEffect>,
    /// Effect constraints
    pub constraints: Vec<EffectTypeConstraint>,
    /// Effect flow information
    pub flow_info: EffectFlowInfo,
    /// Composition results
    pub composition_results: Vec<CompositionResult>,
}

/// Information about inferred effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredEffect {
    /// Effect name
    pub name: String,
    /// Effect category
    pub category: String,
    /// Effect confidence
    pub confidence: f64,
    /// Effect source
    pub source: EffectInferenceSource,
    /// Effect parameters
    pub parameters: HashMap<String, String>,
    /// Capability requirements
    pub capabilities: Vec<String>,
    /// Effect span
    pub span: Span,
}

/// Source of effect inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectInferenceSource {
    /// Explicit effect annotation
    Explicit,
    /// Function call analysis
    FunctionCall,
    /// Built-in operation
    BuiltinOperation,
    /// Expression analysis
    Expression,
    /// Statement analysis
    Statement,
    /// Effect composition
    Composition,
    /// AI-assisted inference
    AIAssisted,
}

/// Effect flow information
#[derive(Debug, Clone)]
pub struct EffectFlowInfo {
    /// Entry effects
    pub entry_effects: Vec<String>,
    /// Exit effects
    pub exit_effects: Vec<String>,
    /// Flow constraints
    pub flow_constraints: Vec<String>,
    /// Exception flows
    pub exception_flows: Vec<String>,
}

impl EffectInferenceEngine {
    /// Create a new effect inference engine
    pub fn new() -> SemanticResult<Self> {
        Ok(Self {
            config: EffectInferenceConfig::default(),
            effect_registry: EffectRegistryInterface::new(),
            flow_analyzer: EffectFlowAnalyzer::new(),
            constraint_generator: EffectConstraintGenerator::new(),
            composition_analyzer: EffectCompositionAnalyzer::new(),
            builtin_effects: BuiltinEffectMappings::new(),
            current_context: EffectInferenceContext::new(),
        })
    }

    /// Create effect inference engine with custom configuration
    pub fn with_config(config: EffectInferenceConfig) -> SemanticResult<Self> {
        Ok(Self {
            config,
            effect_registry: EffectRegistryInterface::new(),
            flow_analyzer: EffectFlowAnalyzer::new(),
            constraint_generator: EffectConstraintGenerator::new(),
            composition_analyzer: EffectCompositionAnalyzer::new(),
            builtin_effects: BuiltinEffectMappings::new(),
            current_context: EffectInferenceContext::new(),
        })
    }

    /// Infer effects from an expression
    pub fn infer_expression_effects(&mut self, expr: &Expr, span: Span) -> SemanticResult<EffectInferenceResult> {
        match expr {
            Expr::Literal(_) => self.infer_literal_effects(span),
            Expr::Variable(_) => self.infer_variable_effects(span),
            Expr::Binary(binary_expr) => self.infer_binary_effects(&binary_expr.left.kind, &binary_expr.operator, &binary_expr.right.kind, span),
            Expr::Unary(unary_expr) => self.infer_unary_effects(&unary_expr.operator, &unary_expr.operand.kind, span),
            Expr::Call(call_expr) => {
                let arg_exprs: Vec<&Expr> = call_expr.arguments.iter().map(|arg| &arg.kind).collect();
                self.infer_call_effects(&call_expr.callee.kind, &arg_exprs, span)
            }
            Expr::Member(member_expr) => {
                self.infer_member_effects(member_expr, span)
            }
            Expr::Index(index_expr) => {
                self.infer_index_effects(index_expr, span)
            }
            // Simplified handling for complex expressions
            _ => self.infer_default_effects(span),
        }
    }

    /// Infer effects from a statement
    pub fn infer_statement_effects(&mut self, stmt: &Stmt, span: Span) -> SemanticResult<EffectInferenceResult> {
        match stmt {
            Stmt::Expression(expr_stmt) => self.infer_expression_effects(&expr_stmt.expression.kind, expr_stmt.expression.span),
            // Simplified handling for other statements
            _ => self.infer_default_effects(span),
        }
    }

    /// Infer effects from a function declaration
    pub fn infer_function_declaration_effects(&mut self, func_decl: &FunctionDecl) -> SemanticResult<EffectInferenceResult> {
        if !self.config.enable_detailed_analysis {
            return Ok(EffectInferenceResult::empty());
        }

        // Enter function context (FunctionDecl doesn't have effects field)
        self.enter_function_context(&func_decl.name.to_string(), &[]);

        // Analyze function body if it exists
        let body_effects = if let Some(ref body) = func_decl.body {
            self.infer_statement_effects(&body.kind, body.span)?
        } else {
            EffectInferenceResult::empty()
        };

        // For now, assume no declared effects since FunctionDecl doesn't have effects field
        let declared_effects: Vec<String> = Vec::new();

        let consistency_check = self.check_effect_consistency(&body_effects.effects, &declared_effects)?;

        // Exit function context
        self.exit_function_context();

        let mut result = body_effects;
        result.constraints.extend(consistency_check);

        Ok(result)
    }

    /// Reset the effect inference engine
    pub fn reset(&mut self) {
        self.flow_analyzer.reset();
        self.constraint_generator.reset();
        self.composition_analyzer.reset();
        self.current_context = EffectInferenceContext::new();
    }

    // Private helper methods

    fn infer_call_effects(&mut self, function: &Expr, arguments: &[&Expr], span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Try to determine the function being called
        if let Expr::Variable(var_expr) = function {
            let function_name = var_expr.name.resolve().unwrap_or_else(|| "unknown".to_string());
            
            // Check built-in effects
            if let Some(builtin_effects) = self.builtin_effects.function_effects.get(&function_name) {
                for effect_name in builtin_effects {
                    effects.push(InferredEffect {
                        name: effect_name.clone(),
                        category: self.categorize_effect(effect_name),
                        confidence: 0.9,
                        source: EffectInferenceSource::BuiltinOperation,
                        parameters: HashMap::new(),
                        capabilities: self.get_effect_capabilities(effect_name),
                        span,
                    });
                }
            }
        }

        // Analyze argument effects
        for arg in arguments {
            let arg_effects = self.infer_expression_effects(arg, span)?;
            effects.extend(arg_effects.effects);
            constraints.extend(arg_effects.constraints);
        }

        // Generate flow information
        let flow_info = if self.config.enable_flow_tracking {
            self.generate_flow_info(&effects)
        } else {
            EffectFlowInfo::empty()
        };

        // Perform composition analysis
        let composition_results = if self.config.enable_composition_analysis {
            self.analyze_effect_composition(&effects)?
        } else {
            Vec::new()
        };

        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info,
            composition_results,
        })
    }

    fn infer_binary_effects(&mut self, left: &Expr, op: &BinaryOperator, right: &Expr, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Analyze operand effects
        let left_effects = self.infer_expression_effects(left, span)?;
        let right_effects = self.infer_expression_effects(right, span)?;

        effects.extend(left_effects.effects);
        effects.extend(right_effects.effects);
        constraints.extend(left_effects.constraints);
        constraints.extend(right_effects.constraints);

        // Check for operator-specific effects
        let operator_name = format!("{:?}", op);
        if let Some(operator_effects) = self.builtin_effects.operator_effects.get(&operator_name) {
            for effect_name in operator_effects {
                effects.push(InferredEffect {
                    name: effect_name.clone(),
                    category: self.categorize_effect(effect_name),
                    confidence: 0.8,
                    source: EffectInferenceSource::BuiltinOperation,
                    parameters: HashMap::new(),
                    capabilities: self.get_effect_capabilities(effect_name),
                    span,
                });
            }
        }

        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info: EffectFlowInfo::empty(),
            composition_results: Vec::new(),
        })
    }

    fn infer_unary_effects(&mut self, op: &UnaryOperator, operand: &Expr, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Analyze operand effects
        let operand_effects = self.infer_expression_effects(operand, span)?;
        effects.extend(operand_effects.effects);
        constraints.extend(operand_effects.constraints);

        // Check for operator-specific effects
        let operator_name = format!("{:?}", op);
        if let Some(operator_effects) = self.builtin_effects.operator_effects.get(&operator_name) {
            for effect_name in operator_effects {
                effects.push(InferredEffect {
                    name: effect_name.clone(),
                    category: self.categorize_effect(effect_name),
                    confidence: 0.8,
                    source: EffectInferenceSource::BuiltinOperation,
                    parameters: HashMap::new(),
                    capabilities: self.get_effect_capabilities(effect_name),
                    span,
                });
            }
        }

        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info: EffectFlowInfo::empty(),
            composition_results: Vec::new(),
        })
    }

    fn categorize_effect(&self, effect_name: &str) -> String {
        match effect_name {
            "IO" | "print" | "println" | "read" | "write" => "IO".to_string(),
            "Memory" | "alloc" | "free" => "Memory".to_string(),
            "Exception" | "throw" | "error" => "Exception".to_string(),
            "State" | "mutate" | "modify" => "State".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    fn get_effect_capabilities(&self, effect_name: &str) -> Vec<String> {
        match effect_name {
            "IO" => vec!["io_capability".to_string()],
            "Memory" => vec!["memory_capability".to_string()],
            "Exception" => vec!["exception_capability".to_string()],
            _ => Vec::new(),
        }
    }

    fn generate_flow_info(&self, effects: &[InferredEffect]) -> EffectFlowInfo {
        EffectFlowInfo {
            entry_effects: effects.iter().map(|e| e.name.clone()).collect(),
            exit_effects: effects.iter().map(|e| e.name.clone()).collect(),
            flow_constraints: Vec::new(),
            exception_flows: effects.iter()
                .filter(|e| e.category == "Exception")
                .map(|e| e.name.clone())
                .collect(),
        }
    }

    fn analyze_effect_composition(&mut self, effects: &[InferredEffect]) -> SemanticResult<Vec<CompositionResult>> {
        let mut results = Vec::new();

        if effects.len() < 2 {
            return Ok(results);
        }

        // Simple composition analysis - would be more sophisticated in practice
        let effect_names: Vec<String> = effects.iter().map(|e| e.name.clone()).collect();
        
        if let Some(cached_result) = self.composition_analyzer.composition_cache.get(&effect_names) {
            results.push(cached_result.clone());
        } else {
            // Perform composition analysis
            let composed_effect = self.compose_effects(&effect_names)?;
            let result = CompositionResult {
                composed_effect: composed_effect.clone(),
                confidence: 0.8,
                metadata: CompositionMetadata {
                    rules_applied: vec!["sequential_composition".to_string()],
                    complexity: effect_names.len() as f64,
                    optimizations: Vec::new(),
                },
            };
            
            self.composition_analyzer.composition_cache.insert(effect_names, result.clone());
            results.push(result);
        }

        Ok(results)
    }

    fn compose_effects(&self, effect_names: &[String]) -> SemanticResult<String> {
        if effect_names.is_empty() {
            return Ok("Pure".to_string());
        }

        if effect_names.len() == 1 {
            return Ok(effect_names[0].clone());
        }

        // Simple composition - would use sophisticated rules in practice
        if effect_names.contains(&"Exception".to_string()) {
            Ok("Exception".to_string())
        } else if effect_names.contains(&"IO".to_string()) {
            Ok("IO".to_string())
        } else if effect_names.contains(&"State".to_string()) {
            Ok("State".to_string())
        } else {
            Ok(format!("Composite({})", effect_names.join(", ")))
        }
    }

    fn enter_function_context(&mut self, function_name: &str, declared_effects: &[prism_ast::Effect]) {
        let declared_effect_names = declared_effects.iter()
            .map(|effect| format!("{:?}", effect))
            .collect();

        self.current_context.function_context = Some(FunctionEffectContext {
            function_name: function_name.to_string(),
            declared_effects: declared_effect_names,
            inferred_effects: Vec::new(),
            effect_parameters: Vec::new(),
        });
    }

    fn exit_function_context(&mut self) {
        self.current_context.function_context = None;
    }

    fn check_effect_consistency(
        &self,
        inferred_effects: &[InferredEffect],
        declared_effects: &[String],
    ) -> SemanticResult<Vec<EffectTypeConstraint>> {
        let mut constraints = Vec::new();

        // Check that all declared effects are present in inferred effects
        for declared_effect in declared_effects {
            let found = inferred_effects.iter()
                .any(|inferred| inferred.name == *declared_effect);

            if !found {
                // Generate constraint for missing effect
                constraints.push(EffectTypeConstraint {
                    base_constraint: TypeConstraint {
                        lhs: ConstraintType::Concrete(SemanticType::Complex {
                            name: "EffectSignature".to_string(),
                            base_type: crate::types::BaseType::Effect(crate::types::EffectType {
                                name: "EffectSignature".to_string(),
                                parameters: Vec::new(),
                                metadata: crate::types::EffectMetadata {
                                    description: "Effect signature type".to_string(),
                                    side_effects: Vec::new(),
                                    resources: Vec::new(),
                                },
                            }),
                            constraints: Vec::new(),
                            business_rules: Vec::new(),
                            metadata: crate::types::SemanticTypeMetadata::default(),
                            ai_context: None,
                            verification_properties: Vec::new(),
                            location: Span::dummy(),
                        }),
                        rhs: ConstraintType::Concrete(SemanticType::Complex {
                            name: "RequiredEffect".to_string(),
                            base_type: crate::types::BaseType::Effect(crate::types::EffectType {
                                name: "RequiredEffect".to_string(),
                                parameters: Vec::new(),
                                metadata: crate::types::EffectMetadata {
                                    description: "Required effect type".to_string(),
                                    side_effects: Vec::new(),
                                    resources: Vec::new(),
                                },
                            }),
                            constraints: Vec::new(),
                            business_rules: Vec::new(),
                            metadata: crate::types::SemanticTypeMetadata::default(),
                            ai_context: None,
                            verification_properties: Vec::new(),
                            location: Span::dummy(),
                        }),
                        origin: Span::dummy(),
                        reason: ConstraintReason::EffectConsistency,
                        priority: 100,
                    },
                    effect_requirements: vec![EffectRequirement {
                        effect: declared_effect.clone(),
                        requirement_type: EffectRequirementType::MustHave,
                        conditions: Vec::new(),
                    }],
                    capability_requirements: Vec::new(),
                    source: EffectConstraintSource::FunctionSignature,
                });
            }
        }

        Ok(constraints)
    }

    // Placeholder methods for other expression types
    fn infer_member_effects(&mut self, member_expr: &prism_ast::MemberExpr, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer effects from the object being accessed
        let object_effects = self.infer_expression_effects(&member_expr.object.kind, member_expr.object.span)?;
        effects.extend(object_effects.effects);
        constraints.extend(object_effects.constraints);

        // Field access itself has a FieldAccess effect
        effects.push(InferredEffect {
            name: "FieldAccess".to_string(),
            category: "Memory".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Expression,
            parameters: {
                let mut params = HashMap::new();
                let field_name = member_expr.member.resolve().unwrap_or_else(|| "unknown".to_string());
                params.insert("field".to_string(), field_name);
                params
            },
            capabilities: vec!["memory_access".to_string()],
            span,
        });

        let flow_info = self.generate_flow_info(&effects);
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info,
            composition_results: Vec::new(),
        })
    }

    fn infer_index_effects(&mut self, index_expr: &prism_ast::IndexExpr, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer effects from the object being indexed
        let object_effects = self.infer_expression_effects(&index_expr.object.kind, index_expr.object.span)?;
        effects.extend(object_effects.effects);
        constraints.extend(object_effects.constraints);

        // Infer effects from the index expression
        let index_effects = self.infer_expression_effects(&index_expr.index.kind, index_expr.index.span)?;
        effects.extend(index_effects.effects);
        constraints.extend(index_effects.constraints);

        // Array/index access has specific effects
        effects.push(InferredEffect {
            name: "IndexAccess".to_string(),
            category: "Memory".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Expression,
            parameters: HashMap::new(),
            capabilities: vec!["memory_access".to_string()],
            span,
        });

        let flow_info = self.generate_flow_info(&effects);
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info,
            composition_results: Vec::new(),
        })
    }

    fn infer_let_effects(&mut self, value: &prism_ast::AstNode<Expr>, body: &prism_ast::AstNode<Expr>, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer effects from the value expression
        let value_effects = self.infer_expression_effects(&value.kind, value.span)?;
        effects.extend(value_effects.effects);
        constraints.extend(value_effects.constraints);

        // Infer effects from the body expression
        let body_effects = self.infer_expression_effects(&body.kind, body.span)?;
        effects.extend(body_effects.effects);
        constraints.extend(body_effects.constraints);

        // Let binding itself may have variable binding effects
        effects.push(InferredEffect {
            name: "VariableBinding".to_string(),
            category: "State".to_string(),
            confidence: 0.8,
            source: EffectInferenceSource::Expression,
            parameters: HashMap::new(),
            capabilities: vec!["local_state".to_string()],
            span,
        });

        let flow_info = self.generate_flow_info(&effects);
        let composition_results = self.analyze_effect_composition(&effects)?;
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info,
            composition_results,
        })
    }

    fn infer_if_effects(&mut self, condition: &prism_ast::AstNode<Expr>, then_branch: &prism_ast::AstNode<Expr>, else_branch: Option<&prism_ast::AstNode<Expr>>, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer effects from condition
        let condition_effects = self.infer_expression_effects(&condition.kind, condition.span)?;
        let entry_effects: Vec<String> = condition_effects.effects.iter().map(|e| e.name.clone()).collect();
        effects.extend(condition_effects.effects);
        constraints.extend(condition_effects.constraints);

        // Infer effects from then branch
        let then_effects = self.infer_expression_effects(&then_branch.kind, then_branch.span)?;
        effects.extend(then_effects.effects);
        constraints.extend(then_effects.constraints);

        // Infer effects from else branch if present
        if let Some(else_expr) = else_branch {
            let else_effects = self.infer_expression_effects(&else_expr.kind, else_expr.span)?;
            effects.extend(else_effects.effects);
            constraints.extend(else_effects.constraints);
        }

        // Conditional execution has control flow effects
        effects.push(InferredEffect {
            name: "ConditionalExecution".to_string(),
            category: "Control".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Expression,
            parameters: HashMap::new(),
            capabilities: vec!["control_flow".to_string()],
            span,
        });

        Ok(EffectInferenceResult {
            effects: effects.clone(),
            constraints,
            flow_info: EffectFlowInfo {
                entry_effects,
                exit_effects: effects.iter().map(|e| e.name.clone()).collect(),
                flow_constraints: vec!["conditional_flow".to_string()],
                exception_flows: Vec::new(),
            },
            composition_results: self.analyze_effect_composition(&effects)?,
        })
    }

    fn infer_match_effects(&mut self, expr: &prism_ast::AstNode<Expr>, arms: &[prism_ast::expr::MatchArm], span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer effects from the matched expression
        let expr_effects = self.infer_expression_effects(&expr.kind, expr.span)?;
        let expr_effects_clone = expr_effects.effects.clone();
        effects.extend(expr_effects.effects);
        constraints.extend(expr_effects.constraints);

        // Infer effects from each match arm
        for arm in arms {
            // Pattern matching effects (simplified)
            effects.push(InferredEffect {
                name: "PatternMatch".to_string(),
                category: "Control".to_string(),
                confidence: 0.8,
                source: EffectInferenceSource::Expression,
                parameters: HashMap::new(),
                capabilities: vec!["pattern_matching".to_string()],
                span: arm.pattern.span,
            });

            // Guard effects if present
            if let Some(guard) = &arm.guard {
                let guard_effects = self.infer_expression_effects(&guard.kind, guard.span)?;
                effects.extend(guard_effects.effects);
                constraints.extend(guard_effects.constraints);
            }

            // Body effects
            let body_effects = self.infer_expression_effects(&arm.body.kind, arm.body.span)?;
            effects.extend(body_effects.effects);
            constraints.extend(body_effects.constraints);
        }

        let effects_clone_for_exit = effects.clone();
        let effects_clone_for_composition = effects.clone();
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info: EffectFlowInfo {
                entry_effects: expr_effects_clone.iter().map(|e| e.name.clone()).collect(),
                exit_effects: effects_clone_for_exit.iter().map(|e| e.name.clone()).collect(),
                flow_constraints: vec!["match_flow".to_string()],
                exception_flows: Vec::new(),
            },
            composition_results: self.analyze_effect_composition(&effects_clone_for_composition)?,
        })
    }

    fn infer_function_effects(&mut self, func_expr: &prism_ast::LambdaExpr, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Function definition has closure creation effects
        effects.push(InferredEffect {
            name: "ClosureCreation".to_string(),
            category: "Memory".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Expression,
            parameters: HashMap::new(),
            capabilities: vec!["memory_allocation".to_string()],
            span,
        });

        // Analyze function body effects (but don't include them in the closure creation)
        let _body_effects = self.infer_expression_effects(&func_expr.body.kind, func_expr.body.span)?;
        // Body effects would be analyzed when the function is called, not when defined

        let effects_clone = effects.clone();
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info: self.generate_flow_info(&effects_clone),
            composition_results: Vec::new(),
        })
    }

    fn infer_let_statement_effects(&mut self, let_stmt: &prism_ast::VariableDecl, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer effects from the initializer expression if present
        if let Some(ref initializer) = let_stmt.initializer {
            let value_effects = self.infer_expression_effects(&initializer.kind, initializer.span)?;
            effects.extend(value_effects.effects);
            constraints.extend(value_effects.constraints);
        }

        // Variable binding effects
        effects.push(InferredEffect {
            name: "VariableDeclaration".to_string(),
            category: "State".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Statement,
            parameters: {
                let mut params = HashMap::new();
                let var_name = let_stmt.name.resolve().unwrap_or_else(|| "unknown".to_string());
                params.insert("variable".to_string(), var_name);
                params.insert("mutable".to_string(), let_stmt.is_mutable.to_string());
                params
            },
            capabilities: vec!["local_state".to_string()],
            span,
        });

        if let_stmt.is_mutable {
            effects.push(InferredEffect {
                name: "MutableBinding".to_string(),
                category: "State".to_string(),
                confidence: 0.9,
                source: EffectInferenceSource::Statement,
                parameters: HashMap::new(),
                capabilities: vec!["mutable_state".to_string()],
                span,
            });
        }

        let effects_clone = effects.clone();
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info: self.generate_flow_info(&effects_clone),
            composition_results: Vec::new(),
        })
    }

    fn infer_function_statement_effects(&mut self, func_stmt: &prism_ast::FunctionDecl, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Function declaration effects
        effects.push(InferredEffect {
            name: "FunctionDeclaration".to_string(),
            category: "Definition".to_string(),
            confidence: 1.0,
            source: EffectInferenceSource::Statement,
            parameters: {
                let mut params = HashMap::new();
                let func_name = func_stmt.name.resolve().unwrap_or_else(|| "anonymous".to_string());
                params.insert("function".to_string(), func_name);
                params
            },
            capabilities: vec!["function_definition".to_string()],
            span,
        });

        // Function declarations in AST don't have explicit effects field

        // Analyze function body effects
        let body_effects = if let Some(ref body) = func_stmt.body {
            self.infer_statement_effects(&body.kind, body.span)?
        } else {
            EffectInferenceResult {
                effects: Vec::new(),
                constraints: Vec::new(),
                flow_info: EffectFlowInfo::empty(),
                composition_results: Vec::new(),
            }
        };
        // Function declarations don't have declared effects in the AST, so skip consistency check

        let effects_clone = effects.clone();
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info: self.generate_flow_info(&effects_clone),
            composition_results: Vec::new(),
        })
    }

    fn infer_return_effects(&mut self, return_stmt: &prism_ast::ReturnStmt, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Return statement has control flow effects
        effects.push(InferredEffect {
            name: "Return".to_string(),
            category: "Control".to_string(),
            confidence: 1.0,
            source: EffectInferenceSource::Statement,
            parameters: HashMap::new(),
            capabilities: vec!["control_flow".to_string()],
            span,
        });

        // Infer effects from return value if present
        if let Some(value) = &return_stmt.value {
            let value_effects = self.infer_expression_effects(&value.kind, value.span)?;
            effects.extend(value_effects.effects);
            constraints.extend(value_effects.constraints);
        }

        Ok(EffectInferenceResult {
            effects: effects.clone(),
            constraints,
            flow_info: EffectFlowInfo {
                entry_effects: effects.iter().map(|e| e.name.clone()).collect(),
                exit_effects: vec!["Return".to_string()],
                flow_constraints: vec!["early_return".to_string()],
                exception_flows: Vec::new(),
            },
            composition_results: Vec::new(),
        })
    }

    fn infer_if_statement_effects(&mut self, if_stmt: &prism_ast::IfStmt, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer condition effects
        let condition_effects = self.infer_expression_effects(&if_stmt.condition.kind, if_stmt.condition.span)?;
        effects.extend(condition_effects.effects);
        constraints.extend(condition_effects.constraints);

        // Infer then branch effects
        let then_effects = self.infer_statement_effects(&if_stmt.then_branch.kind, if_stmt.then_branch.span)?;
        effects.extend(then_effects.effects);
        constraints.extend(then_effects.constraints);

        // Infer else branch effects if present
        if let Some(else_branch) = &if_stmt.else_branch {
            let else_effects = self.infer_statement_effects(&else_branch.kind, else_branch.span)?;
            effects.extend(else_effects.effects);
            constraints.extend(else_effects.constraints);
        }

        // Control flow effects
        effects.push(InferredEffect {
            name: "ConditionalStatement".to_string(),
            category: "Control".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Statement,
            parameters: HashMap::new(),
            capabilities: vec!["control_flow".to_string()],
            span,
        });

        let flow_info = self.generate_flow_info(&effects);
        let composition_results = self.analyze_effect_composition(&effects)?;
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info,
            composition_results,
        })
    }

    fn infer_while_effects(&mut self, while_stmt: &prism_ast::WhileStmt, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer condition effects
        let condition_effects = self.infer_expression_effects(&while_stmt.condition.kind, while_stmt.condition.span)?;
        let entry_effects: Vec<String> = condition_effects.effects.iter().map(|e| e.name.clone()).collect();
        effects.extend(condition_effects.effects);
        constraints.extend(condition_effects.constraints);

        // Infer body effects
        let body_effects = self.infer_statement_effects(&while_stmt.body.kind, while_stmt.body.span)?;
        effects.extend(body_effects.effects);
        constraints.extend(body_effects.constraints);

        // Loop effects
        effects.push(InferredEffect {
            name: "Loop".to_string(),
            category: "Control".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Statement,
            parameters: {
                let mut params = HashMap::new();
                params.insert("loop_type".to_string(), "while".to_string());
                params
            },
            capabilities: vec!["control_flow".to_string(), "iteration".to_string()],
            span,
        });

        Ok(EffectInferenceResult {
            effects: effects.clone(),
            constraints,
            flow_info: EffectFlowInfo {
                entry_effects,
                exit_effects: effects.iter().map(|e| e.name.clone()).collect(),
                flow_constraints: vec!["loop_flow".to_string()],
                exception_flows: Vec::new(),
            },
            composition_results: self.analyze_effect_composition(&effects)?,
        })
    }

    fn infer_for_effects(&mut self, for_stmt: &prism_ast::ForStmt, span: Span) -> SemanticResult<EffectInferenceResult> {
        let mut effects = Vec::new();
        let mut constraints = Vec::new();

        // Infer iterable effects
        let iterable_effects = self.infer_expression_effects(&for_stmt.iterable.kind, for_stmt.iterable.span)?;
        effects.extend(iterable_effects.effects);
        constraints.extend(iterable_effects.constraints);

        // Infer body effects
        let body_effects = self.infer_statement_effects(&for_stmt.body.kind, for_stmt.body.span)?;
        effects.extend(body_effects.effects);
        constraints.extend(body_effects.constraints);

        // Iterator creation and loop effects
        effects.push(InferredEffect {
            name: "Iterator".to_string(),
            category: "Memory".to_string(),
            confidence: 0.8,
            source: EffectInferenceSource::Statement,
            parameters: HashMap::new(),
            capabilities: vec!["memory_access".to_string()],
            span,
        });

        effects.push(InferredEffect {
            name: "ForLoop".to_string(),
            category: "Control".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Statement,
            parameters: {
                let mut params = HashMap::new();
                params.insert("loop_type".to_string(), "for".to_string());
                let var_name = for_stmt.variable.resolve().unwrap_or_else(|| "it".to_string());
                params.insert("iterator_var".to_string(), var_name);
                params
            },
            capabilities: vec!["control_flow".to_string(), "iteration".to_string()],
            span,
        });

        let flow_info = self.generate_flow_info(&effects);
        let composition_results = self.analyze_effect_composition(&effects)?;
        Ok(EffectInferenceResult {
            effects,
            constraints,
            flow_info,
            composition_results,
        })
    }

    fn infer_literal_effects(&mut self, span: Span) -> SemanticResult<EffectInferenceResult> {
        Ok(EffectInferenceResult {
            effects: Vec::new(),
            constraints: Vec::new(),
            flow_info: EffectFlowInfo {
                entry_effects: Vec::new(),
                exit_effects: Vec::new(),
                flow_constraints: Vec::new(),
                exception_flows: Vec::new(),
            },
            composition_results: Vec::new(),
        })
    }

    fn infer_variable_effects(&mut self, span: Span) -> SemanticResult<EffectInferenceResult> {
        let effects = vec![InferredEffect {
            name: "VariableAccess".to_string(),
            category: "Memory".to_string(),
            confidence: 0.9,
            source: EffectInferenceSource::Expression,
            parameters: HashMap::new(),
            capabilities: vec!["memory_access".to_string()],
            span,
        }];

        let flow_info = self.generate_flow_info(&effects);
        Ok(EffectInferenceResult {
            effects,
            constraints: Vec::new(),
            flow_info,
            composition_results: Vec::new(),
        })
    }

    fn infer_default_effects(&mut self, span: Span) -> SemanticResult<EffectInferenceResult> {
        Ok(EffectInferenceResult {
            effects: Vec::new(),
            constraints: Vec::new(),
            flow_info: EffectFlowInfo {
                entry_effects: Vec::new(),
                exit_effects: Vec::new(),
                flow_constraints: Vec::new(),
                exception_flows: Vec::new(),
            },
            composition_results: Vec::new(),
        })
    }
}

impl InferenceEngine for EffectInferenceEngine {
    type Input = (Expr, Span);
    type Output = EffectInferenceResult;
    
    fn infer(&mut self, input: Self::Input) -> SemanticResult<Self::Output> {
        let (expr, span) = input;
        self.infer_expression_effects(&expr, span)
    }
    
    fn engine_name(&self) -> &'static str {
        "EffectInferenceEngine"
    }
    
    fn reset(&mut self) {
        self.reset();
    }
}

impl EffectAware for EffectInferenceEngine {
    fn infer_effects(&self, input: &Self::Input) -> SemanticResult<Vec<String>> {
        let (expr, _span) = input;
        
        // Basic effect inference - would be more sophisticated in practice
        match expr {
            Expr::Call(_) => Ok(vec!["FunctionCall".to_string()]),
            Expr::Member(_) => Ok(vec!["FieldAccess".to_string()]),
            Expr::Index(_) => Ok(vec!["ArrayAccess".to_string()]),
            _ => Ok(Vec::new()),
        }
    }
}

// Implementation of component types

impl EffectRegistryInterface {
    fn new() -> Self {
        let mut known_effects = HashMap::new();
        let mut effect_categories = HashMap::new();
        let mut builtin_mappings = HashMap::new();

        // Initialize built-in effects
        known_effects.insert("IO".to_string(), EffectDefinition {
            name: "IO".to_string(),
            category: EffectCategory::IO,
            parameters: Vec::new(),
            capabilities: vec!["io_capability".to_string()],
            description: "Input/Output operations".to_string(),
        });

        effect_categories.insert("IO".to_string(), EffectCategory::IO);
        effect_categories.insert("State".to_string(), EffectCategory::State);
        effect_categories.insert("Memory".to_string(), EffectCategory::Memory);

        builtin_mappings.insert("print".to_string(), vec!["IO".to_string()]);
        builtin_mappings.insert("println".to_string(), vec!["IO".to_string()]);

        Self {
            known_effects,
            effect_categories,
            builtin_mappings,
        }
    }
}

impl EffectFlowAnalyzer {
    fn new() -> Self {
        Self {
            flow_graph: EffectFlowGraph::new(),
            flow_constraints: Vec::new(),
            current_context: FlowAnalysisContext::new(),
        }
    }

    fn reset(&mut self) {
        self.flow_graph = EffectFlowGraph::new();
        self.flow_constraints.clear();
        self.current_context = FlowAnalysisContext::new();
    }
}

impl EffectFlowGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
        }
    }
}

impl FlowAnalysisContext {
    fn new() -> Self {
        Self {
            scope_depth: 0,
            active_handlers: Vec::new(),
            capability_context: Vec::new(),
            exception_context: None,
        }
    }
}

impl EffectConstraintGenerator {
    fn new() -> Self {
        Self {
            generated_constraints: Vec::new(),
            constraint_templates: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.generated_constraints.clear();
    }
}

impl EffectCompositionAnalyzer {
    fn new() -> Self {
        Self {
            composition_rules: Vec::new(),
            composition_cache: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.composition_cache.clear();
    }
}

impl BuiltinEffectMappings {
    fn new() -> Self {
        let mut function_effects = HashMap::new();
        let mut operator_effects = HashMap::new();
        let mut statement_effects = HashMap::new();

        // Initialize built-in function effects
        function_effects.insert("print".to_string(), vec!["IO".to_string()]);
        function_effects.insert("println".to_string(), vec!["IO".to_string()]);
        function_effects.insert("read".to_string(), vec!["IO".to_string()]);
        function_effects.insert("write".to_string(), vec!["IO".to_string()]);
        function_effects.insert("alloc".to_string(), vec!["Memory".to_string()]);
        function_effects.insert("free".to_string(), vec!["Memory".to_string()]);
        function_effects.insert("throw".to_string(), vec!["Exception".to_string()]);

        // Initialize operator effects (most operators are pure)
        operator_effects.insert("Assign".to_string(), vec!["State".to_string()]);

        // Initialize statement effects
        statement_effects.insert("Assignment".to_string(), vec!["State".to_string()]);

        Self {
            function_effects,
            operator_effects,
            statement_effects,
        }
    }
}

impl EffectInferenceContext {
    fn new() -> Self {
        Self {
            function_context: None,
            scope_effects: Vec::new(),
            active_handlers: Vec::new(),
            capabilities: Vec::new(),
        }
    }
}

impl EffectInferenceResult {
    fn empty() -> Self {
        Self {
            effects: Vec::new(),
            constraints: Vec::new(),
            flow_info: EffectFlowInfo::empty(),
            composition_results: Vec::new(),
        }
    }
}

impl EffectFlowInfo {
    fn empty() -> Self {
        Self {
            entry_effects: Vec::new(),
            exit_effects: Vec::new(),
            flow_constraints: Vec::new(),
            exception_flows: Vec::new(),
        }
    }
}

// Default implementations

impl Default for EffectInferenceConfig {
    fn default() -> Self {
        Self {
            enable_detailed_analysis: true,
            enable_flow_tracking: true,
            enable_composition_analysis: true,
            enable_constraint_generation: true,
            max_analysis_depth: 100,
            enable_effects_integration: true,
        }
    }
}

impl Default for EffectInferenceEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default EffectInferenceEngine")
    }
} 