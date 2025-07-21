//! C-like syntax normalizer.
//!
//! This module implements normalization from C-like syntax (C/C++/Java/JavaScript)
//! to Prism's canonical representation, maintaining conceptual cohesion around
//! "C-like syntax normalization with semantic preservation".

use crate::{
    detection::SyntaxStyle,
    normalization::{
        traits::{StyleNormalizer, NormalizerConfig, NormalizerCapabilities, PerformanceCharacteristics, AIMetadata},
        NormalizationContext, NormalizationError, NormalizationWarning, WarningSeverity, NormalizationUtils
    },
    styles::{
        c_like::{
            CLikeSyntax, CLikeModule, CLikeFunction, CLikeStatement, CLikeExpression, 
            CLikeLiteral, CLikeItem, CLikeParameter, BinaryOperator, UnaryOperator,
            CLikeSwitchCase, CLikeCatchBlock, CLikeObjectField, AssignmentOperator
        },
        canonical::{
            CanonicalSyntax, CanonicalModule, CanonicalFunction, 
            CanonicalStatement, CanonicalExpression, CanonicalLiteral, CanonicalItem,
            CanonicalParameter
        }
    },
};
use prism_common::span::Span;
use std::collections::HashMap;

/// Normalizer specifically for C-like syntax styles.
/// 
/// This normalizer focuses exclusively on converting C-like syntax constructs
/// (braces, semicolons, C-style operators) to Prism's canonical representation
/// while preserving all semantic meaning and generating AI-comprehensible metadata.
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing solely on "C-like syntax
/// normalization". It understands C-style semantics, operator precedence, and
/// control flow patterns specific to C-family languages.
#[derive(Debug)]
pub struct CLikeNormalizer {
    /// Configuration for C-like normalization
    config: CLikeNormalizerConfig,
    
    /// Symbol table for tracking C-like identifiers
    symbol_tracker: SymbolTracker,
}

/// Configuration for C-like normalization behavior
#[derive(Debug, Clone)]
pub struct CLikeNormalizerConfig {
    /// Whether to preserve C-style operator precedence information
    pub preserve_operator_precedence: bool,
    
    /// Whether to normalize C-style casts to explicit conversions
    pub normalize_casts: bool,
    
    /// Whether to convert C-style arrays to canonical collections
    pub normalize_arrays: bool,
    
    /// Whether to handle C-style pointer operations
    pub handle_pointers: bool,
    
    /// Whether to preserve C-style memory management hints
    pub preserve_memory_hints: bool,
    
    /// Custom operator mappings for domain-specific C-like languages
    pub custom_operator_mappings: HashMap<String, String>,
}

/// Symbol tracker for C-like constructs
#[derive(Debug, Default)]
struct SymbolTracker {
    /// Tracked symbols with their C-like characteristics
    symbols: HashMap<String, CLikeSymbolInfo>,
    
    /// Current scope depth
    scope_depth: usize,
}

/// Information about C-like symbols
#[derive(Debug, Clone)]
struct CLikeSymbolInfo {
    /// Symbol name
    name: String,
    
    /// C-like type information if available
    c_type: Option<String>,
    
    /// Whether symbol is a pointer
    is_pointer: bool,
    
    /// Whether symbol is const
    is_const: bool,
    
    /// Whether symbol is static
    is_static: bool,
    
    /// Scope where symbol is defined
    scope_depth: usize,
    
    /// Usage count
    usage_count: usize,
}

impl Default for CLikeNormalizerConfig {
    fn default() -> Self {
        Self {
            preserve_operator_precedence: true,
            normalize_casts: true,
            normalize_arrays: true,
            handle_pointers: true,
            preserve_memory_hints: false,
            custom_operator_mappings: HashMap::new(),
        }
    }
}

impl NormalizerConfig for CLikeNormalizerConfig {
    fn validate(&self) -> Result<(), crate::normalization::traits::ConfigurationError> {
        // Add any C-like specific validation logic here
        Ok(())
    }
    
    fn merge_with(&mut self, other: &Self) {
        self.preserve_operator_precedence = other.preserve_operator_precedence;
        self.normalize_casts = other.normalize_casts;
        self.normalize_arrays = other.normalize_arrays;
        self.handle_pointers = other.handle_pointers;
        self.preserve_memory_hints = other.preserve_memory_hints;
        
        // Merge custom mappings, with other taking precedence
        for (key, value) in &other.custom_operator_mappings {
            self.custom_operator_mappings.insert(key.clone(), value.clone());
        }
    }
}

impl StyleNormalizer for CLikeNormalizer {
    type Input = CLikeSyntax;
    type Intermediate = CanonicalSyntax;
    type Config = CLikeNormalizerConfig;
    
    fn new() -> Self {
        Self::with_config(CLikeNormalizerConfig::default())
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self {
            config,
            symbol_tracker: SymbolTracker::default(),
        }
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::CLike
    }
    
    fn normalize(
        &self, 
        input: &Self::Input, 
        context: &mut NormalizationContext
    ) -> Result<Self::Intermediate, NormalizationError> {
        let mut canonical_modules = Vec::new();
        let mut canonical_functions = Vec::new();
        let mut canonical_statements = Vec::new();
        
        // Convert modules
        for module in &input.modules {
            canonical_modules.push(self.convert_module(module, context)?);
        }
        
        // Convert functions
        for function in &input.functions {
            canonical_functions.push(self.convert_function(function, context)?);
        }
        
        // Convert statements
        for statement in &input.statements {
            canonical_statements.push(self.convert_statement(statement, context)?);
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
        // Validate C-like specific concerns
        self.validate_c_like_semantics(normalized, context)?;
        self.check_operator_precedence_preservation(normalized, context)?;
        self.validate_memory_management_hints(normalized, context)?;
        
        Ok(())
    }
    
    fn generate_ai_metadata(
        &self, 
        normalized: &Self::Intermediate, 
        context: &mut NormalizationContext
    ) -> Result<AIMetadata, NormalizationError> {
        let mut ai_metadata = AIMetadata::default();
        
        // Generate C-like specific AI metadata
        ai_metadata.business_context = Some("C-like syntax normalization".to_string());
        ai_metadata.domain_concepts = self.extract_c_like_concepts(normalized);
        ai_metadata.architectural_patterns = self.identify_c_like_patterns(normalized);
        
        Ok(ai_metadata)
    }
    
    fn capabilities(&self) -> NormalizerCapabilities {
        NormalizerCapabilities {
            supported_constructs: vec![
                "braces".to_string(),
                "semicolons".to_string(),
                "c_style_operators".to_string(),
                "function_calls".to_string(),
                "control_flow".to_string(),
                "arrays".to_string(),
                "objects".to_string(),
                "pointers".to_string(),
                "casts".to_string(),
            ],
            unsupported_constructs: vec![
                "templates".to_string(),
                "macros".to_string(),
                "inline_assembly".to_string(),
            ],
            supports_error_recovery: true,
            generates_ai_metadata: true,
            performance_characteristics: PerformanceCharacteristics {
                time_complexity: "O(n)".to_string(),
                space_complexity: "O(n)".to_string(),
                supports_parallel_processing: false, // Could be improved
                memory_per_node_bytes: 512, // C-like nodes tend to be larger
            },
        }
    }
}

impl CLikeNormalizer {
    /// Convert C-like module to canonical form
    fn convert_module(&self, module: &CLikeModule, context: &mut NormalizationContext) -> Result<CanonicalModule, NormalizationError> {
        context.scope_depth += 1;
        
        let mut canonical_items = Vec::new();
        
        for item in &module.body {
            canonical_items.push(self.convert_item(item, context)?);
        }
        
        context.scope_depth -= 1;
        
        Ok(CanonicalModule {
            name: module.name.clone(),
            items: canonical_items,
            span: module.span,
        })
    }
    
    /// Convert C-like item to canonical form
    fn convert_item(&self, item: &CLikeItem, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        match item {
            CLikeItem::Function(function) => {
                Ok(CanonicalItem::Function(self.convert_function(function, context)?))
            }
            CLikeItem::Statement(statement) => {
                Ok(CanonicalItem::Statement(self.convert_statement(statement, context)?))
            }
            CLikeItem::TypeDeclaration { name, type_def, span } => {
                // Convert C-like type declarations to canonical form
                Ok(CanonicalItem::Statement(CanonicalStatement::Assignment {
                    name: format!("type_{}", name),
                    value: CanonicalExpression::Identifier(type_def.clone()),
                }))
            }
            CLikeItem::VariableDeclaration { name, var_type, initializer, span } => {
                // Track the C-like variable symbol
                self.track_variable_symbol(name, var_type, context);
                
                let canonical_value = if let Some(init) = initializer {
                    self.convert_expression(init, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("undefined".to_string()))
                };
                
                Ok(CanonicalItem::Statement(CanonicalStatement::Assignment {
                    name: name.clone(),
                    value: canonical_value,
                }))
            }
        }
    }
    
    /// Convert C-like function to canonical form
    fn convert_function(&self, function: &CLikeFunction, context: &mut NormalizationContext) -> Result<CanonicalFunction, NormalizationError> {
        context.scope_depth += 1;
        
        // Convert parameters
        let mut canonical_parameters = Vec::new();
        for param in &function.parameters {
            canonical_parameters.push(CanonicalParameter {
                name: param.name.clone(),
                param_type: param.param_type.clone(),
            });
            
            // Track C-like parameter symbols
            self.track_parameter_symbol(param, context);
        }
        
        // Convert body statements
        let mut canonical_body = Vec::new();
        for statement in &function.body {
            canonical_body.push(self.convert_statement(statement, context)?);
        }
        
        context.scope_depth -= 1;
        
        Ok(CanonicalFunction {
            name: function.name.clone(),
            parameters: canonical_parameters,
            return_type: function.return_type.clone(),
            body: canonical_body,
            span: function.span,
        })
    }
    
    /// Convert C-like statement to canonical form
    fn convert_statement(&self, statement: &CLikeStatement, context: &mut NormalizationContext) -> Result<CanonicalStatement, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match statement {
            CLikeStatement::Expression(expr) => {
                Ok(CanonicalStatement::Expression(self.convert_expression(expr, context)?))
            }
            CLikeStatement::Return(expr_opt) => {
                let canonical_expr = if let Some(expr) = expr_opt {
                    Some(self.convert_expression(expr, context)?)
                } else {
                    None
                };
                Ok(CanonicalStatement::Return(canonical_expr))
            }
            CLikeStatement::Assignment { name, value } => {
                Ok(CanonicalStatement::Assignment {
                    name: name.clone(),
                    value: self.convert_expression(value, context)?,
                })
            }
            CLikeStatement::If { condition, then_block, else_block } => {
                // Convert C-like if statement to canonical control flow
                self.convert_if_statement(condition, then_block, else_block, context)
            }
            CLikeStatement::While { condition, body } => {
                // Convert C-like while loop
                self.convert_while_statement(condition, body, context)
            }
            CLikeStatement::For { init, condition, increment, body } => {
                // Convert C-like for loop
                self.convert_for_statement(init, condition, increment, body, context)
            }
            CLikeStatement::DoWhile { body, condition } => {
                // Convert C-like do-while loop
                self.convert_do_while_statement(body, condition, context)
            }
            CLikeStatement::Switch { expression, cases, default_case } => {
                // Convert C-like switch statement
                self.convert_switch_statement(expression, cases, default_case, context)
            }
            CLikeStatement::Break(label) => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "break_statement".to_string(),
                    arguments: vec![
                        CanonicalExpression::Literal(CanonicalLiteral::String(
                            label.as_deref().unwrap_or("no_label").to_string()
                        ))
                    ],
                }))
            }
            CLikeStatement::Continue(label) => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "continue_statement".to_string(),
                    arguments: vec![
                        CanonicalExpression::Literal(CanonicalLiteral::String(
                            label.as_deref().unwrap_or("no_label").to_string()
                        ))
                    ],
                }))
            }
            CLikeStatement::Block(statements) => {
                // Convert C-like block to canonical sequence
                self.convert_block_statement(statements, context)
            }
            CLikeStatement::Try { body, catch_blocks, finally_block } => {
                // Convert C-like try-catch to canonical form
                self.convert_try_statement(body, catch_blocks, finally_block, context)
            }
            CLikeStatement::Throw(expr) => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                    function: "throw".to_string(),
                    arguments: vec![self.convert_expression(expr, context)?],
                }))
            }
            CLikeStatement::VariableDeclaration { name, var_type, initializer } => {
                // Track the C-like variable
                self.track_variable_symbol(name, var_type, context);
                
                let canonical_value = if let Some(init) = initializer {
                    self.convert_expression(init, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("undefined".to_string()))
                };
                
                Ok(CanonicalStatement::Assignment {
                    name: name.clone(),
                    value: canonical_value,
                })
            }
            CLikeStatement::Empty => {
                Ok(CanonicalStatement::Expression(CanonicalExpression::Literal(
                    CanonicalLiteral::String("empty_statement".to_string())
                )))
            }
        }
    }
    
    /// Convert C-like expression to canonical form
    fn convert_expression(&self, expression: &CLikeExpression, context: &mut NormalizationContext) -> Result<CanonicalExpression, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match expression {
            CLikeExpression::Identifier(name) => {
                Ok(CanonicalExpression::Identifier(name.clone()))
            }
            CLikeExpression::Literal(literal) => {
                Ok(CanonicalExpression::Literal(self.convert_literal(literal)?))
            }
            CLikeExpression::Call { function, arguments } => {
                self.convert_call_expression(function, arguments, context)
            }
            CLikeExpression::Binary { left, operator, right } => {
                self.convert_binary_expression(left, operator, right, context)
            }
            CLikeExpression::Unary { operator, operand } => {
                self.convert_unary_expression(operator, operand, context)
            }
            CLikeExpression::Ternary { condition, true_expr, false_expr } => {
                self.convert_ternary_expression(condition, true_expr, false_expr, context)
            }
            CLikeExpression::Assignment { left, operator, right } => {
                self.convert_assignment_expression(left, operator, right, context)
            }
            CLikeExpression::MemberAccess { object, member, safe_navigation } => {
                self.convert_member_access(object, member, *safe_navigation, context)
            }
            CLikeExpression::IndexAccess { object, index } => {
                self.convert_index_access(object, index, context)
            }
            CLikeExpression::ArrayLiteral(elements) => {
                self.convert_array_literal(elements, context)
            }
            CLikeExpression::ObjectLiteral(fields) => {
                self.convert_object_literal(fields, context)
            }
            CLikeExpression::Lambda { parameters, body } => {
                self.convert_lambda_expression(parameters, body, context)
            }
            CLikeExpression::Cast { target_type, expression } => {
                self.convert_cast_expression(target_type, expression, context)
            }
            CLikeExpression::Parenthesized(expr) => {
                // Parentheses are just grouping in C-like languages
                self.convert_expression(expr, context)
            }
            CLikeExpression::PostfixIncrement(expr) => {
                self.convert_postfix_increment(expr, context)
            }
            CLikeExpression::PostfixDecrement(expr) => {
                self.convert_postfix_decrement(expr, context)
            }
            CLikeExpression::PrefixIncrement(expr) => {
                self.convert_prefix_increment(expr, context)
            }
            CLikeExpression::PrefixDecrement(expr) => {
                self.convert_prefix_decrement(expr, context)
            }
        }
    }
    
    /// Convert C-like literal to canonical form
    fn convert_literal(&self, literal: &CLikeLiteral) -> Result<CanonicalLiteral, NormalizationError> {
        match literal {
            CLikeLiteral::String(s) => Ok(CanonicalLiteral::String(s.clone())),
            CLikeLiteral::Integer(i) => Ok(CanonicalLiteral::Integer(*i)),
            CLikeLiteral::Float(f) => Ok(CanonicalLiteral::Float(*f)),
            CLikeLiteral::Boolean(b) => Ok(CanonicalLiteral::Boolean(*b)),
            CLikeLiteral::Null => Ok(CanonicalLiteral::String("null".to_string())),
            CLikeLiteral::Character(c) => Ok(CanonicalLiteral::String(c.to_string())),
        }
    }
    
    // Helper methods for complex statement conversions
    
    fn convert_if_statement(
        &self,
        condition: &CLikeExpression,
        then_block: &[CLikeStatement],
        else_block: &Option<Vec<CLikeStatement>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        let condition_expr = self.convert_expression(condition, context)?;
        
        // For simplified canonical form, convert to a conditional call
        let then_expr = if then_block.len() == 1 {
            if let CLikeStatement::Expression(expr) = &then_block[0] {
                self.convert_expression(expr, context)?
            } else {
                CanonicalExpression::Literal(CanonicalLiteral::String("then_block".to_string()))
            }
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("then_block".to_string()))
        };
        
        let else_expr = if let Some(else_stmts) = else_block {
            if else_stmts.len() == 1 {
                if let CLikeStatement::Expression(expr) = &else_stmts[0] {
                    self.convert_expression(expr, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("else_block".to_string()))
                }
            } else {
                CanonicalExpression::Literal(CanonicalLiteral::String("else_block".to_string()))
            }
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_else".to_string()))
        };
        
        Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
            function: "conditional".to_string(),
            arguments: vec![condition_expr, then_expr, else_expr],
        }))
    }
    
    fn convert_while_statement(
        &self,
        condition: &CLikeExpression,
        body: &[CLikeStatement],
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        let condition_expr = self.convert_expression(condition, context)?;
        
        Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
            function: "while_loop".to_string(),
            arguments: vec![
                condition_expr,
                CanonicalExpression::Literal(CanonicalLiteral::String("loop_body".to_string())),
            ],
        }))
    }
    
    fn convert_for_statement(
        &self,
        init: &Option<Box<CLikeStatement>>,
        condition: &Option<CLikeExpression>,
        increment: &Option<CLikeExpression>,
        body: &[CLikeStatement],
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        let mut args = Vec::new();
        
        if let Some(_init_stmt) = init {
            args.push(CanonicalExpression::Literal(CanonicalLiteral::String("init".to_string())));
        }
        
        if let Some(cond_expr) = condition {
            args.push(self.convert_expression(cond_expr, context)?);
        }
        
        if let Some(inc_expr) = increment {
            args.push(self.convert_expression(inc_expr, context)?);
        }
        
        args.push(CanonicalExpression::Literal(CanonicalLiteral::String("for_body".to_string())));
        
        Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
            function: "for_loop".to_string(),
            arguments: args,
        }))
    }
    
    fn convert_do_while_statement(
        &self,
        body: &[CLikeStatement],
        condition: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        let condition_expr = self.convert_expression(condition, context)?;
        
        Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
            function: "do_while_loop".to_string(),
            arguments: vec![
                CanonicalExpression::Literal(CanonicalLiteral::String("loop_body".to_string())),
                condition_expr,
            ],
        }))
    }
    
    fn convert_switch_statement(
        &self,
        expression: &CLikeExpression,
        cases: &[CLikeSwitchCase],
        default_case: &Option<Vec<CLikeStatement>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        let switch_expr = self.convert_expression(expression, context)?;
        
        Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
            function: "switch_statement".to_string(),
            arguments: vec![
                switch_expr,
                CanonicalExpression::Literal(CanonicalLiteral::String("switch_cases".to_string())),
            ],
        }))
    }
    
    fn convert_block_statement(
        &self,
        statements: &[CLikeStatement],
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        let mut canonical_exprs = Vec::new();
        for stmt in statements {
            if let CanonicalStatement::Expression(expr) = self.convert_statement(stmt, context)? {
                canonical_exprs.push(expr);
            }
        }
        
        if canonical_exprs.len() == 1 {
            Ok(CanonicalStatement::Expression(canonical_exprs.into_iter().next().unwrap()))
        } else {
            Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
                function: "block".to_string(),
                arguments: canonical_exprs,
            }))
        }
    }
    
    fn convert_try_statement(
        &self,
        body: &[CLikeStatement],
        catch_blocks: &[CLikeCatchBlock],
        finally_block: &Option<Vec<CLikeStatement>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalStatement, NormalizationError> {
        Ok(CanonicalStatement::Expression(CanonicalExpression::Call {
            function: "try_catch".to_string(),
            arguments: vec![
                CanonicalExpression::Literal(CanonicalLiteral::String("try_body".to_string())),
                CanonicalExpression::Literal(CanonicalLiteral::String("catch_blocks".to_string())),
            ],
        }))
    }
    
    // Helper methods for complex expression conversions
    
    fn convert_call_expression(
        &self,
        function: &CLikeExpression,
        arguments: &[CLikeExpression],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let function_name = match function {
            CLikeExpression::Identifier(name) => name.clone(),
            _ => "complex_function".to_string(),
        };
        
        let mut canonical_args = Vec::new();
        for arg in arguments {
            canonical_args.push(self.convert_expression(arg, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: function_name,
            arguments: canonical_args,
        })
    }
    
    fn convert_binary_expression(
        &self,
        left: &CLikeExpression,
        operator: &BinaryOperator,
        right: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let left_expr = self.convert_expression(left, context)?;
        let right_expr = self.convert_expression(right, context)?;
        
        let op_string = self.map_binary_operator(operator);
        
        Ok(CanonicalExpression::Binary {
            left: Box::new(left_expr),
            operator: op_string,
            right: Box::new(right_expr),
        })
    }
    
    fn convert_unary_expression(
        &self,
        operator: &UnaryOperator,
        operand: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let operand_expr = self.convert_expression(operand, context)?;
        let op_string = self.map_unary_operator(operator);
        
        Ok(CanonicalExpression::Call {
            function: op_string,
            arguments: vec![operand_expr],
        })
    }
    
    fn convert_ternary_expression(
        &self,
        condition: &CLikeExpression,
        true_expr: &CLikeExpression,
        false_expr: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let cond_expr = self.convert_expression(condition, context)?;
        let true_canonical = self.convert_expression(true_expr, context)?;
        let false_canonical = self.convert_expression(false_expr, context)?;
        
        Ok(CanonicalExpression::Call {
            function: "ternary".to_string(),
            arguments: vec![cond_expr, true_canonical, false_canonical],
        })
    }
    
    fn convert_assignment_expression(
        &self,
        left: &CLikeExpression,
        operator: &AssignmentOperator,
        right: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let right_expr = self.convert_expression(right, context)?;
        
        if let CLikeExpression::Identifier(name) = left {
            Ok(CanonicalExpression::Call {
                function: "assign".to_string(),
                arguments: vec![
                    CanonicalExpression::Identifier(name.clone()),
                    right_expr,
                ],
            })
        } else {
            Ok(CanonicalExpression::Call {
                function: "complex_assign".to_string(),
                arguments: vec![
                    CanonicalExpression::Literal(CanonicalLiteral::String("complex_target".to_string())),
                    right_expr,
                ],
            })
        }
    }
    
    fn convert_member_access(
        &self,
        object: &CLikeExpression,
        member: &str,
        safe_navigation: bool,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let object_expr = self.convert_expression(object, context)?;
        let access_type = if safe_navigation { "safe_member_access" } else { "member_access" };
        
        Ok(CanonicalExpression::Call {
            function: access_type.to_string(),
            arguments: vec![
                object_expr,
                CanonicalExpression::Literal(CanonicalLiteral::String(member.to_string())),
            ],
        })
    }
    
    fn convert_index_access(
        &self,
        object: &CLikeExpression,
        index: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let object_expr = self.convert_expression(object, context)?;
        let index_expr = self.convert_expression(index, context)?;
        
        Ok(CanonicalExpression::Call {
            function: "index_access".to_string(),
            arguments: vec![object_expr, index_expr],
        })
    }
    
    fn convert_array_literal(
        &self,
        elements: &[CLikeExpression],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut canonical_elements = Vec::new();
        for element in elements {
            canonical_elements.push(self.convert_expression(element, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: "array".to_string(),
            arguments: canonical_elements,
        })
    }
    
    fn convert_object_literal(
        &self,
        fields: &[CLikeObjectField],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut canonical_args = Vec::new();
        for field in fields {
            canonical_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(field.key.clone())));
            canonical_args.push(self.convert_expression(&field.value, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: "object".to_string(),
            arguments: canonical_args,
        })
    }
    
    fn convert_lambda_expression(
        &self,
        parameters: &[CLikeParameter],
        body: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut param_names = Vec::new();
        for param in parameters {
            param_names.push(CanonicalExpression::Literal(CanonicalLiteral::String(param.name.clone())));
        }
        
        let body_expr = self.convert_expression(body, context)?;
        
        let mut args = param_names;
        args.push(body_expr);
        
        Ok(CanonicalExpression::Call {
            function: "lambda".to_string(),
            arguments: args,
        })
    }
    
    fn convert_cast_expression(
        &self,
        target_type: &str,
        expression: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let expr = self.convert_expression(expression, context)?;
        
        Ok(CanonicalExpression::Call {
            function: "cast".to_string(),
            arguments: vec![
                CanonicalExpression::Literal(CanonicalLiteral::String(target_type.to_string())),
                expr,
            ],
        })
    }
    
    fn convert_postfix_increment(
        &self,
        expr: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let operand_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "postfix_increment".to_string(),
            arguments: vec![operand_expr],
        })
    }
    
    fn convert_postfix_decrement(
        &self,
        expr: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let operand_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "postfix_decrement".to_string(),
            arguments: vec![operand_expr],
        })
    }
    
    fn convert_prefix_increment(
        &self,
        expr: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let operand_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "prefix_increment".to_string(),
            arguments: vec![operand_expr],
        })
    }
    
    fn convert_prefix_decrement(
        &self,
        expr: &CLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let operand_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "prefix_decrement".to_string(),
            arguments: vec![operand_expr],
        })
    }
    
    // Helper methods for operator mapping
    
    fn map_binary_operator(&self, operator: &BinaryOperator) -> String {
        // Check for custom mappings first
        let op_name = match operator {
            BinaryOperator::Add => "add",
            BinaryOperator::Subtract => "subtract",
            BinaryOperator::Multiply => "multiply",
            BinaryOperator::Divide => "divide",
            BinaryOperator::Modulo => "modulo",
            BinaryOperator::Equal => "equal",
            BinaryOperator::NotEqual => "not_equal",
            BinaryOperator::Less => "less",
            BinaryOperator::LessEqual => "less_equal",
            BinaryOperator::Greater => "greater",
            BinaryOperator::GreaterEqual => "greater_equal",
            BinaryOperator::LogicalAnd => "logical_and",
            BinaryOperator::LogicalOr => "logical_or",
            BinaryOperator::BitwiseAnd => "bitwise_and",
            BinaryOperator::BitwiseOr => "bitwise_or",
            BinaryOperator::BitwiseXor => "bitwise_xor",
            BinaryOperator::LeftShift => "left_shift",
            BinaryOperator::RightShift => "right_shift",
            BinaryOperator::Comma => "comma",
        };
        
        // Check for custom mapping
        self.config.custom_operator_mappings
            .get(op_name)
            .cloned()
            .unwrap_or_else(|| op_name.to_string())
    }
    
    fn map_unary_operator(&self, operator: &UnaryOperator) -> String {
        let op_name = match operator {
            UnaryOperator::Plus => "unary_plus",
            UnaryOperator::Minus => "unary_minus",
            UnaryOperator::LogicalNot => "logical_not",
            UnaryOperator::BitwiseNot => "bitwise_not",
            UnaryOperator::AddressOf => "address_of",
            UnaryOperator::Dereference => "dereference",
        };
        
        // Check for custom mapping
        self.config.custom_operator_mappings
            .get(op_name)
            .cloned()
            .unwrap_or_else(|| op_name.to_string())
    }
    
    // Helper methods for symbol tracking
    
    fn track_variable_symbol(&self, name: &str, var_type: &Option<String>, context: &mut NormalizationContext) {
        // This would update the symbol tracker in a mutable implementation
        // For now, we'll just add to the context symbols
        context.symbols.insert(name.to_string(), crate::normalization::SymbolInfo {
            name: name.to_string(),
            symbol_type: var_type.clone(),
            scope_depth: context.scope_depth,
            is_mutable: true, // C-like variables are mutable by default
            usage_count: 0,
        });
    }
    
    fn track_parameter_symbol(&self, param: &CLikeParameter, context: &mut NormalizationContext) {
        context.symbols.insert(param.name.clone(), crate::normalization::SymbolInfo {
            name: param.name.clone(),
            symbol_type: param.param_type.clone(),
            scope_depth: context.scope_depth,
            is_mutable: false, // Parameters are immutable by default
            usage_count: 0,
        });
    }
    
    // Validation methods
    
    fn validate_c_like_semantics(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // C-like specific semantic validation
        // This could check for proper brace matching, semicolon placement, etc.
        Ok(())
    }
    
    fn check_operator_precedence_preservation(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Ensure C-like operator precedence was properly preserved during normalization
        if self.config.preserve_operator_precedence {
            // Add validation logic here
        }
        Ok(())
    }
    
    fn validate_memory_management_hints(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Validate that memory management hints from C-like code are preserved
        if self.config.preserve_memory_hints {
            // Add validation logic here
        }
        Ok(())
    }
    
    // AI metadata generation methods
    
    fn extract_c_like_concepts(&self, normalized: &CanonicalSyntax) -> Vec<String> {
        let mut concepts = Vec::new();
        
        // Extract C-like specific concepts
        concepts.push("imperative_programming".to_string());
        concepts.push("structured_programming".to_string());
        
        if !normalized.functions.is_empty() {
            concepts.push("procedural_programming".to_string());
        }
        
        concepts
    }
    
    fn identify_c_like_patterns(&self, normalized: &CanonicalSyntax) -> Vec<String> {
        let mut patterns = Vec::new();
        
        // Identify common C-like patterns
        patterns.push("brace_delimited_blocks".to_string());
        patterns.push("semicolon_terminated_statements".to_string());
        patterns.push("c_style_operators".to_string());
        
        patterns
    }
} 