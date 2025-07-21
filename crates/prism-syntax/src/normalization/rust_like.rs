//! Rust-like syntax normalizer.
//!
//! This module implements normalization from Rust-like syntax (Rust/Swift/Go)
//! to Prism's canonical representation, maintaining conceptual cohesion around
//! "Rust-like syntax normalization with expression-oriented semantic preservation".

use crate::{
    detection::SyntaxStyle,
    normalization::{
        traits::{StyleNormalizer, NormalizerConfig, NormalizerCapabilities, PerformanceCharacteristics, AIMetadata},
        NormalizationContext, NormalizationError, NormalizationWarning, WarningSeverity, NormalizationUtils
    },
    styles::{
        rust_like::{
            RustLikeSyntax, RustLikeModule, RustLikeFunction, RustLikeStatement, RustLikeExpression, 
            RustLikeLiteral, RustLikeItem, RustLikeParameter, RustLikeType, RustLikePattern,
            RustLikeMatchArm, RustLikeStruct, RustLikeEnum, RustLikeImpl, RustLikeField,
            RustLikeVariant, RustLikeVariantData, RustLikeImplItem, Visibility, RustLikeLetCondition
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

/// Normalizer specifically for Rust-like syntax styles.
/// 
/// This normalizer focuses exclusively on converting Rust-like syntax constructs
/// (expression-oriented programming, pattern matching, ownership annotations) to Prism's 
/// canonical representation while preserving all semantic meaning and generating 
/// AI-comprehensible metadata.
/// 
/// # Conceptual Cohesion
/// 
/// This struct maintains conceptual cohesion by focusing solely on "Rust-like syntax
/// normalization". It understands Rust-style semantics, expression-oriented programming,
/// pattern matching, and ownership concepts specific to Rust-family languages.
#[derive(Debug)]
pub struct RustLikeNormalizer {
    /// Configuration for Rust-like normalization
    config: RustLikeNormalizerConfig,
    
    /// Pattern matching tracker
    pattern_tracker: PatternTracker,
}

/// Configuration for Rust-like normalization behavior
#[derive(Debug, Clone)]
pub struct RustLikeNormalizerConfig {
    /// Whether to preserve ownership annotations (& and &mut)
    pub preserve_ownership_annotations: bool,
    
    /// Whether to normalize match expressions to canonical form
    pub normalize_match_expressions: bool,
    
    /// Whether to preserve lifetime annotations
    pub preserve_lifetimes: bool,
    
    /// Whether to convert Rust-style error handling (Result, Option) to canonical form
    pub normalize_error_handling: bool,
    
    /// Whether to preserve trait bounds and generic constraints
    pub preserve_trait_bounds: bool,
    
    /// Whether to handle unsafe blocks specially
    pub handle_unsafe_blocks: bool,
    
    /// Custom Rust-like construct mappings
    pub custom_construct_mappings: HashMap<String, String>,
}

/// Pattern matching tracker for Rust-like constructs
#[derive(Debug, Default)]
struct PatternTracker {
    /// Current pattern matching depth
    pattern_depth: usize,
    
    /// Patterns encountered during normalization
    patterns_seen: Vec<String>,
}

/// Information about Rust-like symbols
#[derive(Debug, Clone)]
struct RustLikeSymbolInfo {
    /// Symbol name
    name: String,
    
    /// Rust-like type information if available
    rust_type: Option<RustLikeType>,
    
    /// Whether symbol is mutable
    is_mutable: bool,
    
    /// Whether symbol is a reference
    is_reference: bool,
    
    /// Whether symbol is owned or borrowed
    ownership: OwnershipKind,
    
    /// Scope where symbol is defined
    scope_depth: usize,
    
    /// Usage count
    usage_count: usize,
}

/// Ownership kinds in Rust-like languages
#[derive(Debug, Clone, PartialEq)]
enum OwnershipKind {
    /// Owned value
    Owned,
    
    /// Immutable borrow
    BorrowedImmutable,
    
    /// Mutable borrow
    BorrowedMutable,
    
    /// Unknown or not applicable
    Unknown,
}

impl Default for RustLikeNormalizerConfig {
    fn default() -> Self {
        Self {
            preserve_ownership_annotations: true,
            normalize_match_expressions: true,
            preserve_lifetimes: false, // Simplified for now
            normalize_error_handling: true,
            preserve_trait_bounds: true,
            handle_unsafe_blocks: true,
            custom_construct_mappings: HashMap::new(),
        }
    }
}

impl NormalizerConfig for RustLikeNormalizerConfig {
    fn validate(&self) -> Result<(), crate::normalization::traits::ConfigurationError> {
        // Add any Rust-like specific validation logic here
        Ok(())
    }
    
    fn merge_with(&mut self, other: &Self) {
        self.preserve_ownership_annotations = other.preserve_ownership_annotations;
        self.normalize_match_expressions = other.normalize_match_expressions;
        self.preserve_lifetimes = other.preserve_lifetimes;
        self.normalize_error_handling = other.normalize_error_handling;
        self.preserve_trait_bounds = other.preserve_trait_bounds;
        self.handle_unsafe_blocks = other.handle_unsafe_blocks;
        
        // Merge custom mappings
        for (key, value) in &other.custom_construct_mappings {
            self.custom_construct_mappings.insert(key.clone(), value.clone());
        }
    }
}

impl StyleNormalizer for RustLikeNormalizer {
    type Input = RustLikeSyntax;
    type Intermediate = CanonicalSyntax;
    type Config = RustLikeNormalizerConfig;
    
    fn new() -> Self {
        Self::with_config(RustLikeNormalizerConfig::default())
    }
    
    fn with_config(config: Self::Config) -> Self {
        Self {
            config,
            pattern_tracker: PatternTracker::default(),
        }
    }
    
    fn syntax_style(&self) -> SyntaxStyle {
        SyntaxStyle::RustLike
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
        // Validate Rust-like specific concerns
        self.validate_expression_orientation(normalized, context)?;
        self.check_pattern_matching_completeness(normalized, context)?;
        self.validate_ownership_preservation(normalized, context)?;
        
        Ok(())
    }
    
    fn generate_ai_metadata(
        &self, 
        normalized: &Self::Intermediate, 
        context: &mut NormalizationContext
    ) -> Result<AIMetadata, NormalizationError> {
        let mut ai_metadata = AIMetadata::default();
        
        // Generate Rust-like specific AI metadata
        ai_metadata.business_context = Some("Rust-like syntax normalization".to_string());
        ai_metadata.domain_concepts = self.extract_rust_concepts(normalized);
        ai_metadata.architectural_patterns = self.identify_rust_patterns(normalized);
        
        Ok(ai_metadata)
    }
    
    fn capabilities(&self) -> NormalizerCapabilities {
        NormalizerCapabilities {
            supported_constructs: vec![
                "expression_oriented".to_string(),
                "pattern_matching".to_string(),
                "ownership_annotations".to_string(),
                "match_expressions".to_string(),
                "result_option_types".to_string(),
                "trait_bounds".to_string(),
                "generic_types".to_string(),
                "unsafe_blocks".to_string(),
                "borrowing".to_string(),
            ],
            unsupported_constructs: vec![
                "lifetime_annotations".to_string(),
                "procedural_macros".to_string(),
                "async_await".to_string(),
            ],
            supports_error_recovery: true,
            generates_ai_metadata: true,
            performance_characteristics: PerformanceCharacteristics {
                time_complexity: "O(n)".to_string(),
                space_complexity: "O(n)".to_string(),
                supports_parallel_processing: false, // Pattern matching is sequential
                memory_per_node_bytes: 640, // Rust-like nodes tend to be complex
            },
        }
    }
}

impl RustLikeNormalizer {
    /// Convert Rust-like module to canonical form
    fn convert_module(&self, module: &RustLikeModule, context: &mut NormalizationContext) -> Result<CanonicalModule, NormalizationError> {
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
    
    /// Convert Rust-like item to canonical form
    fn convert_item(&self, item: &RustLikeItem, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        match item {
            RustLikeItem::Function(function) => {
                Ok(CanonicalItem::Function(self.convert_function(function, context)?))
            }
            RustLikeItem::Statement(statement) => {
                Ok(CanonicalItem::Statement(self.convert_statement(statement, context)?))
            }
            RustLikeItem::Struct(struct_def) => {
                self.convert_struct_item(struct_def, context)
            }
            RustLikeItem::Enum(enum_def) => {
                self.convert_enum_item(enum_def, context)
            }
            RustLikeItem::Impl(impl_block) => {
                self.convert_impl_item(impl_block, context)
            }
        }
    }
    
    /// Convert Rust-like function to canonical form
    fn convert_function(&self, function: &RustLikeFunction, context: &mut NormalizationContext) -> Result<CanonicalFunction, NormalizationError> {
        context.scope_depth += 1;
        
        // Convert parameters
        let mut canonical_parameters = Vec::new();
        for param in &function.parameters {
            canonical_parameters.push(CanonicalParameter {
                name: param.name.clone(),
                param_type: Some(self.convert_type_to_string(&param.param_type)),
            });
            
            // Track Rust-like parameter symbols
            self.track_parameter_symbol(param, context);
        }
        
        // Convert body (Rust functions have expression bodies)
        let canonical_body = match &function.body {
            RustLikeExpression::Block(statements, final_expr) => {
                let mut canonical_statements = Vec::new();
                
                // Convert statements
                for stmt in statements {
                    canonical_statements.push(self.convert_statement(stmt, context)?);
                }
                
                // Add final expression as return if present
                if let Some(expr) = final_expr {
                    canonical_statements.push(CanonicalStatement::Return(
                        Some(self.convert_expression(expr, context)?)
                    ));
                }
                
                canonical_statements
            }
            _ => {
                // Single expression function
                vec![CanonicalStatement::Return(
                    Some(self.convert_expression(&function.body, context)?)
                )]
            }
        };
        
        context.scope_depth -= 1;
        
        Ok(CanonicalFunction {
            name: function.name.clone(),
            parameters: canonical_parameters,
            return_type: function.return_type.as_ref().map(|t| self.convert_type_to_string(t)),
            body: canonical_body,
            span: function.span,
        })
    }
    
    /// Convert Rust-like statement to canonical form
    fn convert_statement(&self, statement: &RustLikeStatement, context: &mut NormalizationContext) -> Result<CanonicalStatement, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match statement {
            RustLikeStatement::Let { pattern, value, is_mutable } => {
                // Convert let binding to canonical assignment
                let pattern_name = self.extract_pattern_name(pattern);
                
                // Track the symbol with Rust-like characteristics
                self.track_let_binding(&pattern_name, pattern, *is_mutable, context);
                
                let canonical_value = if let Some(val) = value {
                    self.convert_expression(val, context)?
                } else {
                    CanonicalExpression::Literal(CanonicalLiteral::String("uninitialized".to_string()))
                };
                
                Ok(CanonicalStatement::Assignment {
                    name: pattern_name,
                    value: canonical_value,
                })
            }
            RustLikeStatement::Expression(expr) => {
                Ok(CanonicalStatement::Expression(self.convert_expression(expr, context)?))
            }
        }
    }
    
    /// Convert Rust-like expression to canonical form
    fn convert_expression(&self, expression: &RustLikeExpression, context: &mut NormalizationContext) -> Result<CanonicalExpression, NormalizationError> {
        context.metrics.nodes_processed += 1;
        
        match expression {
            RustLikeExpression::Identifier(name) => {
                Ok(CanonicalExpression::Identifier(name.clone()))
            }
            RustLikeExpression::Literal(literal) => {
                Ok(CanonicalExpression::Literal(self.convert_literal(literal)?))
            }
            RustLikeExpression::Block(statements, final_expr) => {
                self.convert_block_expression(statements, final_expr, context)
            }
            RustLikeExpression::If { condition, then_branch, else_branch } => {
                self.convert_if_expression(condition, then_branch, else_branch, context)
            }
            RustLikeExpression::IfLet { pattern, value, then_branch, else_branch } => {
                self.convert_if_let_expression(pattern, value, then_branch, else_branch, context)
            }
            RustLikeExpression::LetChain { conditions, body, else_branch } => {
                self.convert_let_chain_expression(conditions, body, else_branch, context)
            }
            RustLikeExpression::Match { expr, arms } => {
                if self.config.normalize_match_expressions {
                    self.convert_match_expression_to_canonical(expr, arms, context)
                } else {
                    self.convert_match_expression(expr, arms, context)
                }
            }
            RustLikeExpression::Call { function, arguments } => {
                self.convert_call_expression(function, arguments, context)
            }
            RustLikeExpression::Binary { left, operator, right } => {
                self.convert_binary_expression(left, operator, right, context)
            }
            RustLikeExpression::Unary { operator, operand } => {
                self.convert_unary_expression(operator, operand, context)
            }
            RustLikeExpression::Reference { is_mutable, expr } => {
                self.convert_reference_expression(*is_mutable, expr, context)
            }
            RustLikeExpression::Dereference(expr) => {
                self.convert_dereference_expression(expr, context)
            }
            RustLikeExpression::TryOperator(expr) => {
                self.convert_try_operator_expression(expr, context)
            }
            RustLikeExpression::FieldAccess { object, field } => {
                self.convert_field_access_expression(object, field, context)
            }
            RustLikeExpression::MethodCall { receiver, method, arguments } => {
                self.convert_method_call_expression(receiver, method, arguments, context)
            }
            RustLikeExpression::Array(elements) => {
                self.convert_array_expression(elements, context)
            }
            RustLikeExpression::Tuple(elements) => {
                self.convert_tuple_expression(elements, context)
            }
            RustLikeExpression::Struct { name, fields } => {
                self.convert_struct_expression(name, fields, context)
            }
            RustLikeExpression::Async(expr) => {
                self.convert_async_expression(expr, context)
            }
            RustLikeExpression::Await(expr) => {
                self.convert_await_expression(expr, context)
            }
            RustLikeExpression::Range { start, end, inclusive } => {
                self.convert_range_expression(start, end, *inclusive, context)
            }
            RustLikeExpression::Closure { parameters, body, is_async, is_move } => {
                self.convert_closure_expression(parameters, body, *is_async, *is_move, context)
            }
            RustLikeExpression::Loop { body, label } => {
                self.convert_loop_expression(body, label, context)
            }
            RustLikeExpression::While { condition, body, label } => {
                self.convert_while_expression(condition, body, label, context)
            }
            RustLikeExpression::For { pattern, iterable, body, label } => {
                self.convert_for_expression(pattern, iterable, body, label, context)
            }
            RustLikeExpression::Break { label, value } => {
                self.convert_break_expression(label, value, context)
            }
            RustLikeExpression::Continue { label } => {
                self.convert_continue_expression(label, context)
            }
            RustLikeExpression::Return(value) => {
                self.convert_return_expression(value, context)
            }
            RustLikeExpression::Unsafe(expr) => {
                self.convert_unsafe_expression(expr, context)
            }
            RustLikeExpression::Const(expr) => {
                self.convert_const_expression(expr, context)
            }
            RustLikeExpression::RawPointer { is_mutable, expr } => {
                self.convert_raw_pointer_expression(*is_mutable, expr, context)
            }
            RustLikeExpression::Cast { expr, target_type } => {
                self.convert_cast_expression(expr, target_type, context)
            }
        }
    }
    
    /// Convert Rust-like literal to canonical form
    fn convert_literal(&self, literal: &RustLikeLiteral) -> Result<CanonicalLiteral, NormalizationError> {
        match literal {
            RustLikeLiteral::String(s) => Ok(CanonicalLiteral::String(s.clone())),
            RustLikeLiteral::Integer(i) => Ok(CanonicalLiteral::Integer(*i)),
            RustLikeLiteral::Float(f) => Ok(CanonicalLiteral::Float(*f)),
            RustLikeLiteral::Boolean(b) => Ok(CanonicalLiteral::Boolean(*b)),
            RustLikeLiteral::Unit => Ok(CanonicalLiteral::String("unit".to_string())),
        }
    }
    
    // Helper methods for complex constructs
    
    fn convert_struct_item(&self, struct_def: &RustLikeStruct, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        let mut field_args = Vec::new();
        
        for field in &struct_def.fields {
            field_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(field.name.clone())));
            field_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(
                self.convert_type_to_string(&field.field_type)
            )));
        }
        
        Ok(CanonicalItem::Statement(CanonicalStatement::Expression(
            CanonicalExpression::Call {
                function: "struct_definition".to_string(),
                arguments: vec![
                    CanonicalExpression::Literal(CanonicalLiteral::String(struct_def.name.clone())),
                    CanonicalExpression::Call {
                        function: "fields".to_string(),
                        arguments: field_args,
                    },
                ],
            }
        )))
    }
    
    fn convert_enum_item(&self, enum_def: &RustLikeEnum, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        let mut variant_args = Vec::new();
        
        for variant in &enum_def.variants {
            variant_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(variant.name.clone())));
            
            // Add variant data if present
            if let Some(data) = &variant.data {
                match data {
                    RustLikeVariantData::Tuple(types) => {
                        variant_args.push(CanonicalExpression::Call {
                            function: "tuple_variant".to_string(),
                            arguments: types.iter()
                                .map(|t| CanonicalExpression::Literal(CanonicalLiteral::String(
                                    self.convert_type_to_string(t)
                                )))
                                .collect(),
                        });
                    }
                    RustLikeVariantData::Struct(fields) => {
                        let mut field_args = Vec::new();
                        for field in fields {
                            field_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(field.name.clone())));
                            field_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(
                                self.convert_type_to_string(&field.field_type)
                            )));
                        }
                        variant_args.push(CanonicalExpression::Call {
                            function: "struct_variant".to_string(),
                            arguments: field_args,
                        });
                    }
                }
            }
        }
        
        Ok(CanonicalItem::Statement(CanonicalStatement::Expression(
            CanonicalExpression::Call {
                function: "enum_definition".to_string(),
                arguments: vec![
                    CanonicalExpression::Literal(CanonicalLiteral::String(enum_def.name.clone())),
                    CanonicalExpression::Call {
                        function: "variants".to_string(),
                        arguments: variant_args,
                    },
                ],
            }
        )))
    }
    
    fn convert_impl_item(&self, impl_block: &RustLikeImpl, context: &mut NormalizationContext) -> Result<CanonicalItem, NormalizationError> {
        let target_type = self.convert_type_to_string(&impl_block.target);
        
        let mut impl_args = vec![
            CanonicalExpression::Literal(CanonicalLiteral::String(target_type))
        ];
        
        if let Some(trait_name) = &impl_block.trait_name {
            impl_args.push(CanonicalExpression::Literal(CanonicalLiteral::String(trait_name.clone())));
        }
        
        // Convert impl items (functions)
        let mut item_exprs = Vec::new();
        for item in &impl_block.items {
            match item {
                RustLikeImplItem::Function(func) => {
                    let canonical_func = self.convert_function(func, context)?;
                    item_exprs.push(CanonicalExpression::Call {
                        function: "impl_function".to_string(),
                        arguments: vec![
                            CanonicalExpression::Literal(CanonicalLiteral::String(canonical_func.name)),
                        ],
                    });
                }
            }
        }
        
        impl_args.push(CanonicalExpression::Call {
            function: "impl_items".to_string(),
            arguments: item_exprs,
        });
        
        Ok(CanonicalItem::Statement(CanonicalStatement::Expression(
            CanonicalExpression::Call {
                function: "impl_block".to_string(),
                arguments: impl_args,
            }
        )))
    }
    
    /// Convert block expressions to canonical form
    fn convert_block_expression(
        &self,
        statements: &[RustLikeStatement],
        final_expr: &Option<Box<RustLikeExpression>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut args = Vec::new();
        
        // Convert statements
        for stmt in statements {
            let canonical_stmt = self.convert_statement(stmt, context)?;
            match canonical_stmt {
                CanonicalStatement::Expression(expr) => args.push(expr),
                CanonicalStatement::Return(Some(expr)) => {
                    args.push(CanonicalExpression::Call {
                        function: "return".to_string(),
                        arguments: vec![expr],
                    });
                }
                CanonicalStatement::Return(None) => {
                    args.push(CanonicalExpression::Call {
                        function: "return".to_string(),
                        arguments: vec![CanonicalExpression::Literal(CanonicalLiteral::String("unit".to_string()))],
                    });
                }
                CanonicalStatement::Assignment { name, value } => {
                    args.push(CanonicalExpression::Call {
                        function: "assign".to_string(),
                        arguments: vec![
                            CanonicalExpression::Identifier(name),
                            value,
                        ],
                    });
                }
            }
        }
        
        // Add final expression if present
        if let Some(final_expr) = final_expr {
            args.push(self.convert_expression(final_expr, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: "block".to_string(),
            arguments: args,
        })
    }

    /// Convert let chain expressions to canonical form (Rust 2024 feature)
    fn convert_let_chain_expression(
        &self,
        conditions: &[RustLikeLetCondition],
        body: &RustLikeExpression,
        else_branch: &Option<Box<RustLikeExpression>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut condition_args = Vec::new();
        
        for condition in conditions {
            match condition {
                RustLikeLetCondition::Let { pattern, value } => {
                    let pattern_str = self.convert_pattern_to_string(pattern);
                    let value_expr = self.convert_expression(value, context)?;
                    condition_args.push(CanonicalExpression::Call {
                        function: "let_condition".to_string(),
                        arguments: vec![
                            CanonicalExpression::Literal(CanonicalLiteral::String(pattern_str)),
                            value_expr,
                        ],
                    });
                }
                RustLikeLetCondition::Expression(expr) => {
                    condition_args.push(self.convert_expression(expr, context)?);
                }
            }
        }
        
        let body_expr = self.convert_expression(body, context)?;
        let else_expr = if let Some(else_branch) = else_branch {
            self.convert_expression(else_branch, context)?
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_else".to_string()))
        };
        
        let mut args = vec![CanonicalExpression::Call {
            function: "conditions".to_string(),
            arguments: condition_args,
        }];
        args.push(body_expr);
        args.push(else_expr);
        
        Ok(CanonicalExpression::Call {
            function: "let_chain".to_string(),
            arguments: args,
        })
    }

    /// Convert if expressions to canonical form
    fn convert_if_expression(
        &self,
        condition: &RustLikeExpression,
        then_branch: &RustLikeExpression,
        else_branch: &Option<Box<RustLikeExpression>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let condition_expr = self.convert_expression(condition, context)?;
        let then_expr = self.convert_expression(then_branch, context)?;
        let else_expr = if let Some(else_branch) = else_branch {
            self.convert_expression(else_branch, context)?
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_else".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "if".to_string(),
            arguments: vec![condition_expr, then_expr, else_expr],
        })
    }

    /// Convert match expressions to canonical form
    fn convert_match_expression(
        &self,
        expr: &RustLikeExpression,
        arms: &[RustLikeMatchArm],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let match_expr = self.convert_expression(expr, context)?;
        let mut arm_args = Vec::new();
        
        for arm in arms {
            let pattern_str = self.convert_pattern_to_string(&arm.pattern);
            let body_expr = self.convert_expression(&arm.body, context)?;
            let guard_expr = if let Some(guard) = &arm.guard {
                self.convert_expression(guard, context)?
            } else {
                CanonicalExpression::Literal(CanonicalLiteral::String("no_guard".to_string()))
            };
            
            arm_args.push(CanonicalExpression::Call {
                function: "match_arm".to_string(),
                arguments: vec![
                    CanonicalExpression::Literal(CanonicalLiteral::String(pattern_str)),
                    guard_expr,
                    body_expr,
                ],
            });
        }
        
        Ok(CanonicalExpression::Call {
            function: "match".to_string(),
            arguments: vec![
                match_expr,
                CanonicalExpression::Call {
                    function: "arms".to_string(),
                    arguments: arm_args,
                },
            ],
        })
    }

    /// Convert match expressions to canonical form (normalized version)
    fn convert_match_expression_to_canonical(
        &self,
        expr: &RustLikeExpression,
        arms: &[RustLikeMatchArm],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        // For normalized version, convert match to a series of if-else expressions
        let match_expr = self.convert_expression(expr, context)?;
        
        if arms.is_empty() {
            return Ok(CanonicalExpression::Call {
                function: "empty_match".to_string(),
                arguments: vec![match_expr],
            });
        }
        
        // Convert to nested if-else structure
        let mut result = None;
        for arm in arms.iter().rev() {
            let pattern_str = self.convert_pattern_to_string(&arm.pattern);
            let body_expr = self.convert_expression(&arm.body, context)?;
            
            let condition = CanonicalExpression::Call {
                function: "pattern_match".to_string(),
                arguments: vec![
                    match_expr.clone(),
                    CanonicalExpression::Literal(CanonicalLiteral::String(pattern_str)),
                ],
            };
            
            let current_if = if let Some(else_branch) = result {
                CanonicalExpression::Call {
                    function: "if".to_string(),
                    arguments: vec![condition, body_expr, else_branch],
                }
            } else {
                CanonicalExpression::Call {
                    function: "if".to_string(),
                    arguments: vec![condition, body_expr],
                }
            };
            
            result = Some(current_if);
        }
        
        Ok(result.unwrap())
    }

    /// Convert call expressions to canonical form
    fn convert_call_expression(
        &self,
        function: &RustLikeExpression,
        arguments: &[RustLikeExpression],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let function_name = match function {
            RustLikeExpression::Identifier(name) => name.clone(),
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

    /// Convert binary expressions to canonical form
    fn convert_binary_expression(
        &self,
        left: &RustLikeExpression,
        operator: &str,
        right: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let left_expr = self.convert_expression(left, context)?;
        let right_expr = self.convert_expression(right, context)?;
        let mapped_op = self.map_rust_binary_operator(operator);
        
        Ok(CanonicalExpression::Binary {
            left: Box::new(left_expr),
            operator: mapped_op,
            right: Box::new(right_expr),
        })
    }

    /// Convert unary expressions to canonical form
    fn convert_unary_expression(
        &self,
        operator: &str,
        operand: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let operand_expr = self.convert_expression(operand, context)?;
        let mapped_op = self.map_rust_unary_operator(operator);
        
        Ok(CanonicalExpression::Call {
            function: mapped_op,
            arguments: vec![operand_expr],
        })
    }

    /// Convert reference expressions to canonical form
    fn convert_reference_expression(
        &self,
        is_mutable: bool,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        let ref_type = if is_mutable { "mutable_ref" } else { "immutable_ref" };
        
        if self.config.preserve_ownership_annotations {
            Ok(CanonicalExpression::Call {
                function: ref_type.to_string(),
                arguments: vec![inner_expr],
            })
        } else {
            Ok(inner_expr) // Strip ownership annotations if not preserving
        }
    }

    /// Convert dereference expressions to canonical form
    fn convert_dereference_expression(
        &self,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        
        if self.config.preserve_ownership_annotations {
            Ok(CanonicalExpression::Call {
                function: "dereference".to_string(),
                arguments: vec![inner_expr],
            })
        } else {
            Ok(inner_expr) // Strip dereference if not preserving ownership
        }
    }

    /// Convert try operator expressions to canonical form
    fn convert_try_operator_expression(
        &self,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        
        if self.config.normalize_error_handling {
            Ok(CanonicalExpression::Call {
                function: "try_operator".to_string(),
                arguments: vec![inner_expr],
            })
        } else {
            Ok(CanonicalExpression::Call {
                function: "unwrap_or_propagate".to_string(),
                arguments: vec![inner_expr],
            })
        }
    }

    /// Convert field access expressions to canonical form
    fn convert_field_access_expression(
        &self,
        object: &RustLikeExpression,
        field: &str,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let object_expr = self.convert_expression(object, context)?;
        
        Ok(CanonicalExpression::Call {
            function: "field_access".to_string(),
            arguments: vec![
                object_expr,
                CanonicalExpression::Literal(CanonicalLiteral::String(field.to_string())),
            ],
        })
    }

    /// Convert method call expressions to canonical form
    fn convert_method_call_expression(
        &self,
        receiver: &RustLikeExpression,
        method: &str,
        arguments: &[RustLikeExpression],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let receiver_expr = self.convert_expression(receiver, context)?;
        let mut args = vec![
            receiver_expr,
            CanonicalExpression::Literal(CanonicalLiteral::String(method.to_string())),
        ];
        
        for arg in arguments {
            args.push(self.convert_expression(arg, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: "method_call".to_string(),
            arguments: args,
        })
    }

    /// Convert array expressions to canonical form
    fn convert_array_expression(
        &self,
        elements: &[RustLikeExpression],
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

    /// Convert tuple expressions to canonical form
    fn convert_tuple_expression(
        &self,
        elements: &[RustLikeExpression],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut canonical_elements = Vec::new();
        for element in elements {
            canonical_elements.push(self.convert_expression(element, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: "tuple".to_string(),
            arguments: canonical_elements,
        })
    }

    /// Convert struct expressions to canonical form
    fn convert_struct_expression(
        &self,
        name: &str,
        fields: &[(String, RustLikeExpression)],
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut args = vec![CanonicalExpression::Literal(CanonicalLiteral::String(name.to_string()))];
        
        for (field_name, field_value) in fields {
            args.push(CanonicalExpression::Literal(CanonicalLiteral::String(field_name.clone())));
            args.push(self.convert_expression(field_value, context)?);
        }
        
        Ok(CanonicalExpression::Call {
            function: "struct_literal".to_string(),
            arguments: args,
        })
    }

    /// Convert literal expressions to canonical form
    fn convert_literal(&self, literal: &RustLikeLiteral) -> Result<CanonicalLiteral, NormalizationError> {
        match literal {
            RustLikeLiteral::String(s) => Ok(CanonicalLiteral::String(s.clone())),
            RustLikeLiteral::Integer(i) => Ok(CanonicalLiteral::Integer(*i)),
            RustLikeLiteral::Float(f) => Ok(CanonicalLiteral::Float(*f)),
            RustLikeLiteral::Boolean(b) => Ok(CanonicalLiteral::Boolean(*b)),
            RustLikeLiteral::Unit => Ok(CanonicalLiteral::String("()".to_string())),
        }
    }

    /// Map Rust binary operators to canonical form
    fn map_rust_binary_operator(&self, op: &str) -> String {
        let op_name = match op {
            "+" => "add",
            "-" => "subtract",
            "*" => "multiply",
            "/" => "divide",
            "%" => "modulo",
            "==" => "equal",
            "!=" => "not_equal",
            "<" => "less",
            "<=" => "less_equal",
            ">" => "greater",
            ">=" => "greater_equal",
            "&&" => "logical_and",
            "||" => "logical_or",
            "&" => "bitwise_and",
            "|" => "bitwise_or",
            "^" => "bitwise_xor",
            "<<" => "left_shift",
            ">>" => "right_shift",
            _ => op,
        };
        
        self.config.custom_construct_mappings
            .get(op_name)
            .cloned()
            .unwrap_or_else(|| op_name.to_string())
    }
    
    fn map_rust_unary_operator(&self, op: &str) -> String {
        let op_name = match op {
            "+" => "unary_plus",
            "-" => "unary_minus",
            "!" => "logical_not",
            "~" => "bitwise_not",
            _ => op,
        };
        
        self.config.custom_construct_mappings
            .get(op_name)
            .cloned()
            .unwrap_or_else(|| op_name.to_string())
    }
    
    // Symbol tracking methods
    
    fn track_parameter_symbol(&self, param: &RustLikeParameter, context: &mut NormalizationContext) {
        context.symbols.insert(param.name.clone(), crate::normalization::SymbolInfo {
            name: param.name.clone(),
            symbol_type: Some(self.convert_type_to_string(&param.param_type)),
            scope_depth: context.scope_depth,
            is_mutable: param.is_mutable,
            usage_count: 0,
        });
    }
    
    fn track_let_binding(&self, name: &str, pattern: &RustLikePattern, is_mutable: bool, context: &mut NormalizationContext) {
        context.symbols.insert(name.to_string(), crate::normalization::SymbolInfo {
            name: name.to_string(),
            symbol_type: None, // Type inference would be needed
            scope_depth: context.scope_depth,
            is_mutable,
            usage_count: 0,
        });
    }
    
    // Validation methods
    
    fn validate_expression_orientation(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Validate that Rust-like expression orientation was preserved
        Ok(())
    }
    
    fn check_pattern_matching_completeness(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Check that pattern matching was properly converted
        Ok(())
    }
    
    fn validate_ownership_preservation(
        &self,
        normalized: &CanonicalSyntax,
        context: &mut NormalizationContext
    ) -> Result<(), NormalizationError> {
        // Ensure ownership annotations were preserved if configured
        if self.config.preserve_ownership_annotations {
            // Add validation logic here
        }
        Ok(())
    }
    
    // AI metadata generation methods
    
    fn extract_rust_concepts(&self, normalized: &CanonicalSyntax) -> Vec<String> {
        let mut concepts = Vec::new();
        
        concepts.push("expression_oriented_programming".to_string());
        concepts.push("pattern_matching".to_string());
        concepts.push("ownership_system".to_string());
        concepts.push("zero_cost_abstractions".to_string());
        
        if !normalized.functions.is_empty() {
            concepts.push("functional_programming".to_string());
        }
        
        concepts
    }
    
    fn identify_rust_patterns(&self, normalized: &CanonicalSyntax) -> Vec<String> {
        let mut patterns = Vec::new();
        
        patterns.push("borrowing_and_ownership".to_string());
        patterns.push("algebraic_data_types".to_string());
        patterns.push("trait_system".to_string());
        patterns.push("memory_safety".to_string());
        
        patterns
    }

    /// Convert a pattern to a string representation
    fn convert_pattern_to_string(&self, pattern: &RustLikePattern) -> String {
        match pattern {
            RustLikePattern::Identifier(name) => name.clone(),
            RustLikePattern::Literal(literal) => format!("{:?}", literal),
            RustLikePattern::Wildcard => "_".to_string(),
            RustLikePattern::Tuple(patterns) => {
                let pattern_strs: Vec<String> = patterns.iter()
                    .map(|p| self.convert_pattern_to_string(p))
                    .collect();
                format!("({})", pattern_strs.join(", "))
            }
            RustLikePattern::Struct { name, fields } => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|(field_name, field_pattern)| {
                        format!("{}: {}", field_name, self.convert_pattern_to_string(field_pattern))
                    })
                    .collect();
                format!("{} {{ {} }}", name, field_strs.join(", "))
            }
        }
    }

    /// Convert a Rust type to string representation
    fn convert_type_to_string(&self, rust_type: &RustLikeType) -> String {
        match rust_type {
            RustLikeType::Identifier(name) => name.clone(),
            RustLikeType::Reference { is_mutable, inner } => {
                let mutability = if *is_mutable { "&mut " } else { "&" };
                format!("{}{}", mutability, self.convert_type_to_string(inner))
            }
            RustLikeType::Tuple(types) => {
                let type_strs: Vec<String> = types.iter()
                    .map(|t| self.convert_type_to_string(t))
                    .collect();
                format!("({})", type_strs.join(", "))
            }
            RustLikeType::Array { element_type, size } => {
                if let Some(size) = size {
                    format!("[{}; {}]", self.convert_type_to_string(element_type), size)
                } else {
                    format!("[{}]", self.convert_type_to_string(element_type))
                }
            }
            RustLikeType::Function { parameters, return_type } => {
                let param_strs: Vec<String> = parameters.iter()
                    .map(|t| self.convert_type_to_string(t))
                    .collect();
                format!("fn({}) -> {}", param_strs.join(", "), self.convert_type_to_string(return_type))
            }
            RustLikeType::Generic { base, args } => {
                let arg_strs: Vec<String> = args.iter()
                    .map(|t| self.convert_type_to_string(t))
                    .collect();
                format!("{}<{}>", base, arg_strs.join(", "))
            }
            RustLikeType::ImplTrait { traits } => {
                format!("impl {}", traits.join(" + "))
            }
            RustLikeType::DynTrait { traits } => {
                format!("dyn {}", traits.join(" + "))
            }
            RustLikeType::RawPointer { is_mutable, inner } => {
                let mutability = if *is_mutable { "*mut " } else { "*const " };
                format!("{}{}", mutability, self.convert_type_to_string(inner))
            }
            RustLikeType::Associated { base, associated } => {
                format!("{}::{}", self.convert_type_to_string(base), associated)
            }
            RustLikeType::HigherRanked { lifetimes, inner } => {
                format!("for<{}> {}", lifetimes.join(", "), self.convert_type_to_string(inner))
            }
            RustLikeType::Never => "!".to_string(),
            RustLikeType::Unit => "()".to_string(),
            RustLikeType::Slice(inner) => {
                format!("[{}]", self.convert_type_to_string(inner))
            }
            RustLikeType::Path { segments, generics } => {
                let path_str = segments.join("::");
                if generics.is_empty() {
                    path_str
                } else {
                    let generic_strs: Vec<String> = generics.iter()
                        .map(|t| self.convert_type_to_string(t))
                        .collect();
                    format!("{}<{}>", path_str, generic_strs.join(", "))
                }
            }
        }
    }

    /// Convert async expressions to canonical form
    fn convert_async_expression(
        &self,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "async_block".to_string(),
            arguments: vec![inner_expr],
        })
    }

    /// Convert await expressions to canonical form
    fn convert_await_expression(
        &self,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "await".to_string(),
            arguments: vec![inner_expr],
        })
    }

    /// Convert range expressions to canonical form
    fn convert_range_expression(
        &self,
        start: &Option<Box<RustLikeExpression>>,
        end: &Option<Box<RustLikeExpression>>,
        inclusive: bool,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut args = Vec::new();
        
        if let Some(start_expr) = start {
            args.push(self.convert_expression(start_expr, context)?);
        } else {
            args.push(CanonicalExpression::Literal(CanonicalLiteral::String("no_start".to_string())));
        }
        
        if let Some(end_expr) = end {
            args.push(self.convert_expression(end_expr, context)?);
        } else {
            args.push(CanonicalExpression::Literal(CanonicalLiteral::String("no_end".to_string())));
        }
        
        args.push(CanonicalExpression::Literal(CanonicalLiteral::Boolean(inclusive)));
        
        Ok(CanonicalExpression::Call {
            function: "range".to_string(),
            arguments: args,
        })
    }

    /// Convert closure expressions to canonical form
    fn convert_closure_expression(
        &self,
        parameters: &[RustLikeParameter],
        body: &RustLikeExpression,
        is_async: bool,
        is_move: bool,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let mut args = Vec::new();
        
        // Add parameters
        for param in parameters {
            args.push(CanonicalExpression::Literal(CanonicalLiteral::String(param.name.clone())));
        }
        
        // Add body
        args.push(self.convert_expression(body, context)?);
        
        // Add flags
        args.push(CanonicalExpression::Literal(CanonicalLiteral::Boolean(is_async)));
        args.push(CanonicalExpression::Literal(CanonicalLiteral::Boolean(is_move)));
        
        Ok(CanonicalExpression::Call {
            function: "closure".to_string(),
            arguments: args,
        })
    }

    /// Convert loop expressions to canonical form
    fn convert_loop_expression(
        &self,
        body: &RustLikeExpression,
        label: &Option<String>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let body_expr = self.convert_expression(body, context)?;
        let label_expr = if let Some(label) = label {
            CanonicalExpression::Literal(CanonicalLiteral::String(label.clone()))
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "loop".to_string(),
            arguments: vec![body_expr, label_expr],
        })
    }

    /// Convert while expressions to canonical form
    fn convert_while_expression(
        &self,
        condition: &RustLikeExpression,
        body: &RustLikeExpression,
        label: &Option<String>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let condition_expr = self.convert_expression(condition, context)?;
        let body_expr = self.convert_expression(body, context)?;
        let label_expr = if let Some(label) = label {
            CanonicalExpression::Literal(CanonicalLiteral::String(label.clone()))
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "while".to_string(),
            arguments: vec![condition_expr, body_expr, label_expr],
        })
    }

    /// Convert for expressions to canonical form
    fn convert_for_expression(
        &self,
        pattern: &RustLikePattern,
        iterable: &RustLikeExpression,
        body: &RustLikeExpression,
        label: &Option<String>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let pattern_str = self.convert_pattern_to_string(pattern);
        let iterable_expr = self.convert_expression(iterable, context)?;
        let body_expr = self.convert_expression(body, context)?;
        let label_expr = if let Some(label) = label {
            CanonicalExpression::Literal(CanonicalLiteral::String(label.clone()))
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "for_loop".to_string(),
            arguments: vec![
                CanonicalExpression::Literal(CanonicalLiteral::String(pattern_str)),
                iterable_expr,
                body_expr,
                label_expr,
            ],
        })
    }

    /// Convert break expressions to canonical form
    fn convert_break_expression(
        &self,
        label: &Option<String>,
        value: &Option<Box<RustLikeExpression>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let label_expr = if let Some(label) = label {
            CanonicalExpression::Literal(CanonicalLiteral::String(label.clone()))
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
        };
        
        let value_expr = if let Some(value) = value {
            self.convert_expression(value, context)?
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_value".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "break".to_string(),
            arguments: vec![label_expr, value_expr],
        })
    }

    /// Convert continue expressions to canonical form
    fn convert_continue_expression(
        &self,
        label: &Option<String>,
        _context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let label_expr = if let Some(label) = label {
            CanonicalExpression::Literal(CanonicalLiteral::String(label.clone()))
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("no_label".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "continue".to_string(),
            arguments: vec![label_expr],
        })
    }

    /// Convert return expressions to canonical form
    fn convert_return_expression(
        &self,
        value: &Option<Box<RustLikeExpression>>,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let value_expr = if let Some(value) = value {
            self.convert_expression(value, context)?
        } else {
            CanonicalExpression::Literal(CanonicalLiteral::String("unit".to_string()))
        };
        
        Ok(CanonicalExpression::Call {
            function: "return".to_string(),
            arguments: vec![value_expr],
        })
    }

    /// Convert unsafe expressions to canonical form
    fn convert_unsafe_expression(
        &self,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        if self.config.handle_unsafe_blocks {
            let inner_expr = self.convert_expression(expr, context)?;
            Ok(CanonicalExpression::Call {
                function: "unsafe_block".to_string(),
                arguments: vec![inner_expr],
            })
        } else {
            // Just convert the inner expression
            self.convert_expression(expr, context)
        }
    }

    /// Convert const expressions to canonical form
    fn convert_const_expression(
        &self,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        Ok(CanonicalExpression::Call {
            function: "const_block".to_string(),
            arguments: vec![inner_expr],
        })
    }

    /// Convert raw pointer expressions to canonical form
    fn convert_raw_pointer_expression(
        &self,
        is_mutable: bool,
        expr: &RustLikeExpression,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let inner_expr = self.convert_expression(expr, context)?;
        let mutability = if is_mutable { "mutable" } else { "immutable" };
        
        Ok(CanonicalExpression::Call {
            function: "raw_pointer".to_string(),
            arguments: vec![
                CanonicalExpression::Literal(CanonicalLiteral::String(mutability.to_string())),
                inner_expr,
            ],
        })
    }

    /// Convert cast expressions to canonical form
    fn convert_cast_expression(
        &self,
        expr: &RustLikeExpression,
        target_type: &RustLikeType,
        context: &mut NormalizationContext
    ) -> Result<CanonicalExpression, NormalizationError> {
        let expr_canonical = self.convert_expression(expr, context)?;
        let type_str = self.convert_type_to_string(target_type);
        
        Ok(CanonicalExpression::Call {
            function: "cast".to_string(),
            arguments: vec![
                expr_canonical,
                CanonicalExpression::Literal(CanonicalLiteral::String(type_str)),
            ],
        })
    }
} 