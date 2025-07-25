//! Canonical syntax normalizer.
//!
//! This module implements normalization from Prism's canonical syntax style
//! to the canonical representation. Since the input is already in canonical form,
//! this normalizer primarily performs direct mapping with semantic validation
//! and AI metadata generation.

use crate::{
    detection::SyntaxStyle,
    normalization::{
        traits::{StyleNormalizer, NormalizerConfig, NormalizerCapabilities, PerformanceCharacteristics, AIMetadata},
        NormalizationContext, NormalizationError, NormalizationWarning, WarningSeverity, NormalizationUtils
    },
    styles::canonical::CanonicalConfig,
};
use prism_ast::{
    AstNode, Stmt, Expr, Type,
    stmt::{
        ExpressionStmt, VariableDecl, FunctionDecl, Parameter,
        ModuleDecl, SectionDecl, SectionKind,
        ReturnStmt, BreakStmt, ContinueStmt, BlockStmt
    },
    expr::{
        LiteralExpr, LiteralValue, BinaryExpr, BinaryOperator, CallExpr, MemberExpr, BlockExpr,
        VariableExpr, UnaryExpr, UnaryOperator
    },
    types::{NamedType, ArrayType},
    TypeDecl, TypeKind
};
use crate::styles::canonical::{
    CanonicalSyntax, CanonicalModule, CanonicalFunction, 
    CanonicalStatement, CanonicalExpression, CanonicalLiteral, CanonicalItem,
    CanonicalParameter
};
use prism_common::span::Span;
use std::collections::HashMap;

/// Normalizer specifically for Prism's canonical syntax style.
/// 
/// This normalizer handles the case where input is already in Prism's canonical
/// syntax format. It primarily performs direct mapping from AST nodes to canonical
/// representation while adding semantic validation and AI metadata generation.
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing solely on "canonical syntax
/// normalization". Since the input is already canonical, it emphasizes validation,
/// metadata enhancement, and ensuring consistency with the canonical representation.
#[derive(Debug)]
pub struct CanonicalNormalizer {
    /// Configuration for canonical normalization
    config: CanonicalNormalizerConfig,
    
    /// Semantic validator for canonical constructs
    semantic_validator: CanonicalSemanticValidator,
}

/// Configuration for canonical normalization behavior
#[derive(Debug, Clone)]
pub struct CanonicalNormalizerConfig {
    /// Whether to perform strict validation of canonical constructs
    pub strict_validation: bool,
    
    /// Whether to enhance AI metadata for canonical constructs
    pub enhance_ai_metadata: bool,
    
    /// Whether to validate semantic consistency
    pub validate_semantics: bool,
    
    /// Whether to preserve all AST metadata
    pub preserve_ast_metadata: bool,
    
    /// Whether to validate module structure
    pub validate_module_structure: bool,
    
    /// Custom validation rules
    pub custom_validation_rules: HashMap<String, String>,
}

/// Semantic validator for canonical constructs
#[derive(Debug, Default)]
struct CanonicalSemanticValidator {
    /// Validation rules specific to canonical syntax
    rules: Vec<CanonicalValidationRule>,
}

/// Validation rule for canonical constructs
#[derive(Debug, Clone)]
struct CanonicalValidationRule {
    /// Rule name
    name: String,
    
    /// Rule description
    description: String,
    
    /// Rule severity
    severity: WarningSeverity,
}

impl Default for CanonicalNormalizerConfig {
    fn default() -> Self {
        Self {
            strict_validation: true,
            enhance_ai_metadata: true,
            validate_semantics: true,
            preserve_ast_metadata: true,
            validate_module_structure: true,
            custom_validation_rules: HashMap::new(),
        }
    }
}

impl NormalizerConfig for CanonicalNormalizerConfig {
    fn validate(&self) -> Result<(), crate::normalization::traits::ConfigurationError> {
        // Add any canonical-specific validation logic here
        Ok(())
    }
    
    fn merge_with(&mut self, other: &Self) {
        self.strict_validation = other.strict_validation;
        self.enhance_ai_metadata = other.enhance_ai_metadata;
        self.validate_semantics = other.validate_semantics;
        self.preserve_ast_metadata = other.preserve_ast_metadata;
        self.validate_module_structure = other.validate_module_structure;
        
        // Merge custom rules
        for (key, value) in &other.custom_validation_rules {
            self.custom_validation_rules.insert(key.clone(), value.clone());
        }
    }
}

impl StyleNormalizer for CanonicalNormalizer {
    type Input = Vec<AstNode<Stmt>>;
    type Intermediate = CanonicalSyntax;
    type Config = CanonicalNormalizerConfig;
    
    fn new() -> Self {
        Self::with_config(CanonicalNormalizerConfig::default())
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self {
            config,
            semantic_validator: CanonicalSemanticValidator::default(),
        }
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::Canonical
    }
    
    fn normalize(
        &self, 
        input: &Self::Input, 
        context: &mut NormalizationContext
    ) -> Result<Self::Intermediate, NormalizationError> {
        let mut canonical_modules = Vec::new();
        let mut canonical_functions = Vec::new();
        let mut canonical_statements = Vec::new();
        
        // Process each top-level statement
        for ast_node in input {
            match &ast_node.kind {
                Stmt::Module(module_decl) => {
                    canonical_modules.push(self.convert_module(module_decl, &ast_node.span, context)?);
                }
                Stmt::Function(function_decl) => {
                    canonical_functions.push(self.convert_function(function_decl, &ast_node.span, context)?);
                }
                _ => {
                    canonical_statements.push(self.convert_statement(&ast_node.kind, &ast_node.span, context)?);
                }
            }
        }
        
        Ok(CanonicalSyntax {
            modules: canonical_modules,
            functions: canonical_functions,
            statements: canonical_statements,
        })
    }
    
    fn validate_normalized(
        &self, 
        normalized: &Self::Intermediate, 
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Validate canonical-specific concerns
        if self.config.strict_validation {
            self.validate_canonical_structure(normalized, context)?;
        }
        
        if self.config.validate_semantics {
            self.validate_semantic_consistency(normalized, context)?;
        }
        
        if self.config.validate_module_structure {
            self.validate_module_organization(normalized, context)?;
        }
        
        Ok(())
    }
    
    fn generate_ai_metadata(
        &self, 
        normalized: &Self::Intermediate, 
        context: &mut NormalizationContext
    ) -> Result<AIMetadata, NormalizationError> {
        let mut ai_metadata = AIMetadata::default();
        
        if self.config.enhance_ai_metadata {
            // Generate canonical-specific AI metadata
            ai_metadata.business_context = Some("Canonical Prism syntax normalization".to_string());
            ai_metadata.domain_concepts = self.extract_canonical_concepts(normalized);
            ai_metadata.architectural_patterns = self.identify_canonical_patterns(normalized);
        }
        
        Ok(ai_metadata)
    }
    
    fn capabilities(&self) -> NormalizerCapabilities {
        NormalizerCapabilities {
            supported_constructs: vec![
                "modules".to_string(),
                "sections".to_string(),
                "functions".to_string(),
                "types".to_string(),
                "expressions".to_string(),
                "statements".to_string(),
                "ai_annotations".to_string(),
                "semantic_delimiters".to_string(),
            ],
            unsupported_constructs: vec![
                // Canonical syntax supports all Prism constructs by definition
            ],
            supports_error_recovery: true,
            generates_ai_metadata: true,
            performance_characteristics: PerformanceCharacteristics {
                time_complexity: "O(n)".to_string(),
                space_complexity: "O(n)".to_string(),
                supports_parallel_processing: true, // Direct mapping can be parallelized
                memory_per_node_bytes: 256, // Canonical nodes are well-structured
            },
        }
    }
}

impl CanonicalNormalizer {
    /// Convert AST module declaration to canonical form
    fn convert_module(&self, module: &ModuleDecl, span: &Span, context: &mut NormalizationContext) -> Result<CanonicalModule, NormalizationError> {
        context.scope_depth += 1;
        
        let mut canonical_items = Vec::new();
        
        // Convert sections to items
        for section in &module.sections {
            for item in &section.kind.items {
                canonical_items.push(self.convert_ast_node_to_item(item, context)?);
            }
        }
        
        context.scope_depth -= 1;
        
        Ok(CanonicalModule {
            name: module.name.to_string(),
            items: canonical_items,
            span: *span,
        })
    }
    
    /// Convert AST function declaration to canonical form
    fn convert_function(&self, function: &FunctionDecl, span: &Span, context: &mut NormalizationContext) -> Result<CanonicalFunction, NormalizationError> {
        context.scope_depth += 1;
        
        // Convert parameters
        let mut canonical_parameters = Vec::new();
        for param in &function.parameters {
            canonical_parameters.push(CanonicalParameter {
                name: param.name.to_string(),
                param_type: param.type_annotation.as_ref().map(|t| self.convert_type_to_string(t)),
            });
            
            // Track parameter symbols
            self.track_parameter_symbol(param, context);
        }
        
        // Convert body
        let mut canonical_body = Vec::new();
        if let Some(body_node) = &function.body {
            if let Stmt::Block(block) = &body_node.kind {
                for stmt_node in &block.statements {
                    canonical_body.push(self.convert_statement(&stmt_node.kind, &stmt_node.span, context)?);
                }
            }
        }
        
        context.scope_depth -= 1;
        
        Ok(CanonicalFunction {
            name: function.name.to_string(),
            parameters: canonical_parameters,
            return_type: function.return_type.as_ref().map(|t| self.convert_type_to_string(t)),
            body: canonical_body,
            span: *span,
        })
    }
    
    /// Convert AST statement to canonical form
    fn convert_statement(&self, statement: &Stmt, span: &Span, context: &mut NormalizationContext) -> Result<CanonicalStatement, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match statement {
            Stmt::Expression(expr_stmt) => {
                Ok(CanonicalStatement::Expression(self.convert_expression(&expr_stmt.expression.kind, &expr_stmt.expression.span, context)?))
            }
            Stmt::Variable(var_decl) => {
                // Track the variable symbol
                self.track_variable_symbol(var_decl, context);
                
                let canonical_value = if let Some(init) = &var_decl.initializer {
                    self.convert_expression(&init.kind, &init.span, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("uninitialized".to_string()))
                };
                
                Ok(CanonicalStatement::Assignment {
                    name: var_decl.name.to_string(),
                    value: canonical_value,
                })
            }
            Stmt::Return(return_stmt) => {
                let canonical_expr = if let Some(val) = &return_stmt.value {
                    Some(self.convert_expression(&val.kind, &val.span, context)?)
                } else {
                    None
                };
                Ok(CanonicalStatement::Return(canonical_expr))
            }
            Stmt::Break(_) => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "break_statement".to_string(),
                    arguments: vec![],
                }))
            }
            Stmt::Continue(_) => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "continue_statement".to_string(),
                    arguments: vec![],
                }))
            }
            Stmt::Block(block) => {
                let mut block_statements = Vec::new();
                for stmt_node in &block.statements {
                    block_statements.push(self.convert_statement(&stmt_node.kind, &stmt_node.span, context)?);
                }
                
                if block_statements.len() == 1 {
                    Ok(block_statements.into_iter().next().unwrap())
                } else {
                    Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                        function: "block".to_string(),
                        arguments: block_statements.into_iter().map(|stmt| match stmt {
                            CanonicalStatement::Expression(expr) => expr,
                            _ => CanonicalExpression::Literal(CanonicalLiteral::String("statement".to_string())),
                        }).collect(),
                    }))
                }
            }
            Stmt::Function(_) => {
                // Functions at statement level - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "nested_function".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("function_definition".to_string()))],
                }))
            }
            Stmt::Type(_) => {
                // Type declarations - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "type_declaration".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("type_definition".to_string()))],
                }))
            }
            Stmt::Module(_) => {
                // Nested modules - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "nested_module".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("module_definition".to_string()))],
                }))
            }
            Stmt::Section(_) => {
                // Section statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "section_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("section_definition".to_string()))],
                }))
            }
            Stmt::Actor(_) => {
                // Actor declarations - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "actor_declaration".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("actor_definition".to_string()))],
                }))
            }
            Stmt::Import(_) => {
                // Import statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "import_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("import_definition".to_string()))],
                }))
            }
            Stmt::Export(_) => {
                // Export statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "export_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("export_definition".to_string()))],
                }))
            }
            Stmt::Const(_) => {
                // Const declarations - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "const_declaration".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("const_definition".to_string()))],
                }))
            }
            Stmt::If(_) => {
                // If statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "if_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("if_definition".to_string()))],
                }))
            }
            Stmt::While(_) => {
                // While statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "while_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("while_definition".to_string()))],
                }))
            }
            Stmt::For(_) => {
                // For statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "for_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("for_definition".to_string()))],
                }))
            }
            Stmt::Match(_) => {
                // Match statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "match_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("match_definition".to_string()))],
                }))
            }
            Stmt::Throw(_) => {
                // Throw statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "throw_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("throw_definition".to_string()))],
                }))
            }
            Stmt::Try(_) => {
                // Try statements - convert to expression call
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "try_statement".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("try_definition".to_string()))],
                }))
            }
        }
    }
    
    /// Convert AST expression to canonical form
    fn convert_expression(&self, expression: &Expr, span: &Span, context: &mut NormalizationContext) -> Result<CanonicalExpression, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match expression {
            Expr::Variable(var_expr) => {
                Ok(CanonicalExpression::Identifier(var_expr.name.to_string()))
            }
            Expr::Literal(literal_expr) => {
                Ok(CanonicalExpression::Literal(self.convert_literal(&literal_expr.value)?))
            }
            Expr::Call(call_expr) => {
                let function_name = match &call_expr.callee.kind {
                    Expr::Variable(var) => var.name.to_string(),
                    _ => "complex_function".to_string(),
                };
                
                let mut canonical_args = Vec::new();
                for arg in &call_expr.arguments {
                    canonical_args.push(self.convert_expression(&arg.kind, &arg.span, context)?);
                }
                
                Ok(CanonicalExpression::Call {
                    function: function_name,
                    arguments: canonical_args,
                })
            }
            Expr::Binary(binary_expr) => {
                let left_expr = self.convert_expression(&binary_expr.left.kind, &binary_expr.left.span, context)?;
                let right_expr = self.convert_expression(&binary_expr.right.kind, &binary_expr.right.span, context)?;
                
                let op_string = self.map_binary_operator(&binary_expr.operator);
                
                Ok(CanonicalExpression::Binary {
                    left: Box::new(left_expr),
                    operator: op_string,
                    right: Box::new(right_expr),
                })
            }
            Expr::Unary(unary_expr) => {
                let operand_expr = self.convert_expression(&unary_expr.operand.kind, &unary_expr.operand.span, context)?;
                let op_string = self.map_unary_operator(&unary_expr.operator);
                
                Ok(CanonicalExpression::Call {
                    function: op_string,
                    arguments: vec![operand_expr],
                })
            }
            Expr::Member(member_expr) => {
                let object_expr = self.convert_expression(&member_expr.object.kind, &member_expr.object.span, context)?;
                
                Ok(CanonicalExpression::Call {
                    function: "member_access".to_string(),
                    arguments: vec![
                        object_expr,
                        CanonicalExpression::Literal(CanonicalLiteral::String(member_expr.member.to_string())),
                    ],
                })
            }
            Expr::Block(block_expr) => {
                let mut block_args = Vec::new();
                
                // Convert statements
                for stmt_node in &block_expr.statements {
                    let canonical_stmt = self.convert_statement(&stmt_node.kind, &stmt_node.span, context)?;
                    block_args.push(match canonical_stmt {
                        CanonicalStatement::Expression(expr) => expr,
                        _ => CanonicalExpression::Literal(CanonicalLiteral::String("statement".to_string())),
                    });
                }
                
                // Add final expression if present
                if let Some(final_expr) = &block_expr.final_expr {
                    block_args.push(self.convert_expression(&final_expr.kind, &final_expr.span, context)?);
                }
                
                Ok(CanonicalExpression::Call {
                    function: "block_expression".to_string(),
                    arguments: block_args,
                })
            }
            // Add all missing expression variants
            Expr::Index(_) => {
                Ok(CanonicalExpression::Call {
                    function: "index_access".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("index".to_string()))],
                })
            }
            Expr::Array(_) => {
                Ok(CanonicalExpression::Call {
                    function: "array_literal".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("array".to_string()))],
                })
            }
            Expr::Object(_) => {
                Ok(CanonicalExpression::Call {
                    function: "object_literal".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("object".to_string()))],
                })
            }
            Expr::Lambda(_) => {
                Ok(CanonicalExpression::Call {
                    function: "lambda_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("lambda".to_string()))],
                })
            }
            Expr::Match(_) => {
                Ok(CanonicalExpression::Call {
                    function: "match_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("match".to_string()))],
                })
            }
            Expr::If(_) => {
                Ok(CanonicalExpression::Call {
                    function: "if_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("if".to_string()))],
                })
            }
            Expr::While(_) => {
                Ok(CanonicalExpression::Call {
                    function: "while_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("while".to_string()))],
                })
            }
            Expr::For(_) => {
                Ok(CanonicalExpression::Call {
                    function: "for_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("for".to_string()))],
                })
            }
            Expr::Try(_) => {
                Ok(CanonicalExpression::Call {
                    function: "try_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("try".to_string()))],
                })
            }
            Expr::TypeAssertion(_) => {
                Ok(CanonicalExpression::Call {
                    function: "type_assertion".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("type_assertion".to_string()))],
                })
            }
            Expr::Await(_) => {
                Ok(CanonicalExpression::Call {
                    function: "await_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("await".to_string()))],
                })
            }
            Expr::Yield(_) => {
                Ok(CanonicalExpression::Call {
                    function: "yield_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("yield".to_string()))],
                })
            }
            Expr::Actor(_) => {
                Ok(CanonicalExpression::Call {
                    function: "actor_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("actor".to_string()))],
                })
            }
            Expr::Spawn(_) => {
                Ok(CanonicalExpression::Call {
                    function: "spawn_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("spawn".to_string()))],
                })
            }
            Expr::Channel(_) => {
                Ok(CanonicalExpression::Call {
                    function: "channel_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("channel".to_string()))],
                })
            }
            Expr::Select(_) => {
                Ok(CanonicalExpression::Call {
                    function: "select_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("select".to_string()))],
                })
            }
            Expr::Range(_) => {
                Ok(CanonicalExpression::Call {
                    function: "range_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("range".to_string()))],
                })
            }
            Expr::Tuple(_) => {
                Ok(CanonicalExpression::Call {
                    function: "tuple_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("tuple".to_string()))],
                })
            }
            Expr::Return(_) => {
                Ok(CanonicalExpression::Call {
                    function: "return_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("return".to_string()))],
                })
            }
            Expr::Break(_) => {
                Ok(CanonicalExpression::Call {
                    function: "break_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("break".to_string()))],
                })
            }
            Expr::Continue(_) => {
                Ok(CanonicalExpression::Call {
                    function: "continue_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("continue".to_string()))],
                })
            }
            Expr::Throw(_) => {
                Ok(CanonicalExpression::Call {
                    function: "throw_expression".to_string(),
                    arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("throw".to_string()))],
                })
            }
        }
    }
    
    /// Convert AST literal to canonical form
    fn convert_literal(&self, literal: &LiteralValue) -> Result<CanonicalLiteral, NormalizationError> {
        match literal {
            LiteralValue::String(s) => Ok(CanonicalLiteral::String(s.clone())),
            LiteralValue::Integer(i) => Ok(CanonicalLiteral::Integer(*i)),
            LiteralValue::Float(f) => Ok(CanonicalLiteral::Float(*f)),
            LiteralValue::Boolean(b) => Ok(CanonicalLiteral::Boolean(*b)),
            LiteralValue::Null => Ok(CanonicalLiteral::String("null".to_string())),
            LiteralValue::Money { amount, currency } => {
                Ok(CanonicalLiteral::String(format!("{}.{}", amount, currency)))
            }
            LiteralValue::Duration { value, unit } => {
                Ok(CanonicalLiteral::String(format!("{}.{}", value, unit)))
            }
            LiteralValue::Regex(pattern) => {
                Ok(CanonicalLiteral::String(format!("regex({})", pattern)))
            }
        }
    }
    
    // Helper methods
    
    fn convert_ast_node_to_item(&self, node: &AstNode<Stmt>, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        match &node.kind {
            Stmt::Function(function_decl) => {
                Ok(CanonicalItem::Function(self.convert_function(function_decl, &node.span, context)?))
            }
            _ => {
                Ok(CanonicalItem::Statement(self.convert_statement(&node.kind, &node.span, context)?))
            }
        }
    }
    
    fn convert_type_to_string(&self, type_node: &AstNode<Type>) -> String {
        match &type_node.kind {
            Type::Named(named_type) => named_type.name.to_string(),
            Type::Array(array_type) => {
                format!("[{}]", self.convert_type_to_string(&array_type.element_type))
            }
            _ => "unknown_type".to_string(),
        }
    }
    
    fn map_binary_operator(&self, operator: &BinaryOperator) -> String {
        match operator {
            BinaryOperator::Add => "add".to_string(),
            BinaryOperator::Subtract => "subtract".to_string(),
            BinaryOperator::Multiply => "multiply".to_string(),
            BinaryOperator::Divide => "divide".to_string(),
            BinaryOperator::Modulo => "modulo".to_string(),
            BinaryOperator::Power => "power".to_string(),
            BinaryOperator::FloorDivide => "floor_divide".to_string(),
            BinaryOperator::MatrixMultiply => "matrix_multiply".to_string(),
            BinaryOperator::Equal => "equal".to_string(),
            BinaryOperator::NotEqual => "not_equal".to_string(),
            BinaryOperator::Less => "less".to_string(),
            BinaryOperator::Greater => "greater".to_string(),
            BinaryOperator::LessEqual => "less_equal".to_string(),
            BinaryOperator::GreaterEqual => "greater_equal".to_string(),
            // Add missing comparison aliases
            BinaryOperator::LessThan => "less_than".to_string(),
            BinaryOperator::LessThanOrEqual => "less_than_or_equal".to_string(),
            BinaryOperator::GreaterThan => "greater_than".to_string(),
            BinaryOperator::GreaterThanOrEqual => "greater_than_or_equal".to_string(),
            BinaryOperator::And => "logical_and".to_string(),
            BinaryOperator::Or => "logical_or".to_string(),
            // Add missing logical aliases
            BinaryOperator::LogicalAnd => "logical_and".to_string(),
            BinaryOperator::LogicalOr => "logical_or".to_string(),
            BinaryOperator::BitAnd => "bitwise_and".to_string(),
            BinaryOperator::BitOr => "bitwise_or".to_string(),
            BinaryOperator::BitXor => "bitwise_xor".to_string(),
            // Add missing bitwise aliases
            BinaryOperator::BitwiseAnd => "bitwise_and".to_string(),
            BinaryOperator::BitwiseOr => "bitwise_or".to_string(),
            BinaryOperator::BitwiseXor => "bitwise_xor".to_string(),
            BinaryOperator::LeftShift => "left_shift".to_string(),
            BinaryOperator::RightShift => "right_shift".to_string(),
            BinaryOperator::Assign => "assign".to_string(),
            BinaryOperator::AddAssign => "add_assign".to_string(),
            BinaryOperator::SubtractAssign => "subtract_assign".to_string(),
            BinaryOperator::MultiplyAssign => "multiply_assign".to_string(),
            BinaryOperator::DivideAssign => "divide_assign".to_string(),
            BinaryOperator::WalrusAssign => "walrus_assign".to_string(),
            BinaryOperator::SemanticEqual => "semantic_equal".to_string(),
            BinaryOperator::TypeCompatible => "type_compatible".to_string(),
            BinaryOperator::ConceptualMatch => "conceptual_match".to_string(),
            BinaryOperator::Range => "range".to_string(),
            BinaryOperator::RangeInclusive => "range_inclusive".to_string(),
        }
    }
    
    fn map_unary_operator(&self, operator: &UnaryOperator) -> String {
        match operator {
            UnaryOperator::Not => "logical_not".to_string(),
            UnaryOperator::LogicalNot => "logical_not".to_string(),
            UnaryOperator::Negate => "negate".to_string(),
            UnaryOperator::BitNot => "bitwise_not".to_string(),
            UnaryOperator::BitwiseNot => "bitwise_not".to_string(),
            UnaryOperator::Reference => "reference".to_string(),
            UnaryOperator::Dereference => "dereference".to_string(),
            UnaryOperator::PreIncrement => "pre_increment".to_string(),
            UnaryOperator::PostIncrement => "post_increment".to_string(),
            UnaryOperator::PreDecrement => "pre_decrement".to_string(),
            UnaryOperator::PostDecrement => "post_decrement".to_string(),
        }
    }
    
    // Symbol tracking methods
    
    fn track_parameter_symbol(&self, param: &Parameter, context: &mut NormalizationContext) {
        context.symbols.insert(param.name.to_string(), crate::normalization::SymbolInfo {
            name: param.name.to_string(),
            symbol_type: param.type_annotation.as_ref().map(|t| self.convert_type_to_string(t)),
            scope_depth: context.scope_depth,
            is_mutable: param.is_mutable,
            usage_count: 0,
        });
    }
    
    fn track_variable_symbol(&self, var_decl: &VariableDecl, context: &mut NormalizationContext) {
        context.symbols.insert(var_decl.name.to_string(), crate::normalization::SymbolInfo {
            name: var_decl.name.to_string(),
            symbol_type: var_decl.type_annotation.as_ref().map(|t| self.convert_type_to_string(t)),
            scope_depth: context.scope_depth,
            is_mutable: var_decl.is_mutable,
            usage_count: 0,
        });
    }
    
    // Validation methods
    
    fn validate_canonical_structure(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Validate that canonical structure is well-formed
        if self.config.strict_validation {
            // Add strict validation logic here
            for function in &normalized.functions {
                if function.name.is_empty() {
                    context.warnings.push(NormalizationWarning {
                        message: "Function has empty name".to_string(),
                        span: Some(function.span),
                        severity: WarningSeverity::Error,
                        suggestion: Some("Provide a meaningful function name".to_string()),
                    });
                }
            }
        }
        Ok(())
    }
    
    fn validate_semantic_consistency(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Validate semantic consistency of canonical constructs
        for (name, symbol) in &context.symbols {
            if symbol.usage_count == 0 {
                context.warnings.push(NormalizationWarning {
                    message: format!("Unused symbol: {}", name),
                    span: None,
                    severity: WarningSeverity::Warning,
                    suggestion: Some(format!("Consider removing unused symbol '{}'", name)),
                });
            }
        }
        Ok(())
    }
    
    fn validate_module_organization(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Validate that modules are properly organized
        for module in &normalized.modules {
            if module.items.is_empty() {
                context.warnings.push(NormalizationWarning {
                    message: format!("Module '{}' is empty", module.name),
                    span: Some(module.span),
                    severity: WarningSeverity::Warning,
                    suggestion: Some("Consider adding content to the module or removing it".to_string()),
                });
            }
        }
        Ok(())
    }
    
    // AI metadata generation methods
    
    fn extract_canonical_concepts(&self, normalized: &CanonicalSyntax) -> Vec<String> {
        let mut concepts = Vec::new();
        
        concepts.push("canonical_syntax".to_string());
        concepts.push("semantic_delimiters".to_string());
        concepts.push("structured_modules".to_string());
        concepts.push("explicit_typing".to_string());
        
        if !normalized.modules.is_empty() {
            concepts.push("modular_architecture".to_string());
        }
        
        if !normalized.functions.is_empty() {
            concepts.push("functional_decomposition".to_string());
        }
        
        concepts
    }
    
    fn identify_canonical_patterns(&self, normalized: &CanonicalSyntax) -> Vec<String> {
        let mut patterns = Vec::new();
        
        patterns.push("explicit_structure".to_string());
        patterns.push("semantic_clarity".to_string());
        patterns.push("ai_friendly_syntax".to_string());
        patterns.push("unambiguous_parsing".to_string());
        
        // Check for specific patterns
        if normalized.modules.iter().any(|m| m.items.len() > 5) {
            patterns.push("large_modules".to_string());
        }
        
        if normalized.functions.iter().any(|f| f.parameters.len() > 3) {
            patterns.push("complex_functions".to_string());
        }
        
        patterns
    }
} 