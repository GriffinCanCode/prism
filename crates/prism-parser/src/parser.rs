//! Main parser implementation for the Prism programming language
//!
//! This module implements the hybrid parsing approach with recursive descent
//! for complex constructs and Pratt parsing for expressions.

use crate::{
    error::{ErrorContext, ParseError, ParseErrorKind, ParseResult},
    precedence::{associativity, infix_precedence, prefix_precedence, Precedence},
    constraint_validation::{ConstraintValidator, ValidationConfig, ValidationStrictness},
    ParseConfig,
};
use prism_ast::{
    AstArena, AstNode, Expr, Item, NodeId, Program, ProgramMetadata, Stmt, Type,
};
use prism_common::{span::Span, SourceId};
use prism_lexer::{Token, TokenKind};
use std::collections::VecDeque;

/// The main parser for Prism source code
pub struct Parser {
    /// Token stream to parse
    tokens: Vec<Token>,
    /// Current position in token stream
    current: usize,
    /// Memory arena for AST nodes
    arena: AstArena,
    /// Parse errors encountered
    errors: Vec<ParseError>,
    /// Parser configuration
    config: ParseConfig,
    /// Source ID for span creation
    source_id: SourceId,
    /// Recovery mode flag
    recovery_mode: bool,
    /// Next node ID
    next_id: u32,
    /// Constraint validator for semantic types
    constraint_validator: ConstraintValidator,
}

impl Parser {
    /// Create a new parser with default configuration
    pub fn new(tokens: Vec<Token>) -> Self {
        Self::with_config(tokens, ParseConfig::default())
    }

    /// Create a new parser with custom configuration
    pub fn with_config(tokens: Vec<Token>, config: ParseConfig) -> Self {
        let source_id = if tokens.is_empty() {
            SourceId::new(0)
        } else {
            tokens[0].span.source_id
        };

        Self {
            tokens,
            current: 0,
            arena: AstArena::new(source_id),
            errors: Vec::new(),
            config,
            source_id,
            recovery_mode: false,
            next_id: 0,
            constraint_validator: ConstraintValidator::new(ValidationConfig::default()),
        }
    }

    // Public accessor methods for recovery module
    pub fn get_errors(&self) -> &[ParseError] {
        &self.errors
    }

    pub fn get_current_position(&self) -> usize {
        self.current
    }

    pub fn get_tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn insert_token(&mut self, position: usize, token: Token) {
        self.tokens.insert(position, token);
    }

    pub fn set_recovery_mode(&mut self, recovery_mode: bool) {
        self.recovery_mode = recovery_mode;
    }

    pub fn create_error_node<T>(&mut self, kind: T, span: Span) -> AstNode<T> {
        self.create_node(kind, span)
    }

    /// Parse a complete program
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let mut items = Vec::new();

        while !self.is_at_end() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(error) => {
                    self.errors.push(error);
                    if self.errors.len() >= self.config.max_errors {
                        break;
                    }
                    self.synchronize();
                }
            }
        }

        let metadata = ProgramMetadata {
            primary_capability: if self.config.extract_ai_context {
                Some("Prism program with AI-first design".to_string())
            } else {
                None
            },
            capabilities: Vec::new(),
            dependencies: Vec::new(),
            security_implications: Vec::new(),
            performance_notes: Vec::new(),
            ai_insights: if self.config.extract_ai_context {
                Some(format!("Program with {} items, complexity score: {}", 
                    items.len(), 
                    self.calculate_complexity_score(&items)))
            } else {
                None
            },
        };

        if self.errors.is_empty() {
            Ok(Program {
                items,
                source_id: self.source_id,
                metadata,
            })
        } else {
            Err(self.errors[0].clone())
        }
    }

    /// Parse a single expression
    pub fn parse_expression(&mut self) -> ParseResult<AstNode<Expr>> {
        self.parse_precedence(Precedence::Assignment)
    }

    /// Parse a single statement
    pub fn parse_statement(&mut self) -> ParseResult<AstNode<Stmt>> {
        self.parse_stmt()
    }

    /// Parse a type annotation
    pub fn parse_type(&mut self) -> ParseResult<AstNode<Type>> {
        self.parse_type_annotation()
    }

    /// Create a new AST node with arena allocation
    fn create_node<T>(&mut self, kind: T, span: Span) -> AstNode<T> {
        let id = self.next_node_id();
        AstNode::new(kind, span, id)
    }

    /// Get the next node ID
    fn next_node_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        NodeId::new(id)
    }

    // Token stream management

    /// Check if we're at the end of the token stream
    pub fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || self.peek().kind == TokenKind::Eof
    }

    /// Get the current token without consuming it
    pub fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token {
            kind: TokenKind::Eof,
            span: Span::dummy(),
            semantic_context: None,
        })
    }

    /// Get the previous token
    pub fn previous(&self) -> &Token {
        if self.current == 0 {
            &Token {
                kind: TokenKind::Eof,
                span: Span::dummy(),
                semantic_context: None,
            }
        } else {
            &self.tokens[self.current - 1]
        }
    }

    /// Advance to the next token and return the current one
    pub fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    /// Check if the current token matches the given kind
    pub fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(&kind)
        }
    }

    /// Consume a token if it matches the expected kind
    pub fn consume(&mut self, expected: TokenKind, message: &str) -> ParseResult<&Token> {
        if self.check(expected.clone()) {
            Ok(self.advance())
        } else {
            let found = self.peek().kind.clone();
            let span = self.peek().span;
            Err(ParseError::unexpected_token(
                vec![expected],
                found,
                span,
            ).with_context(ErrorContext::from_tokens(&self.tokens, self.current, 3)))
        }
    }

    /// Get the current span for error reporting
    pub fn current_span(&self) -> Span {
        self.peek().span
    }

    // Methods for recovery module

    /// Get errors (for recovery)
    pub fn get_errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Get current position (for recovery)
    pub fn get_current_position(&self) -> usize {
        self.current
    }

    /// Get tokens (for recovery)
    pub fn get_tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Insert a token at current position (for recovery)
    pub fn insert_token(&mut self, token: Token) {
        self.tokens.insert(self.current, token);
    }

    /// Set recovery mode (for recovery)
    pub fn set_recovery_mode(&mut self, mode: bool) {
        self.recovery_mode = mode;
    }

    /// Create an error node (for recovery)
    pub fn create_error_node<T>(&mut self, kind: T, span: Span) -> AstNode<T> {
        self.create_node(kind, span)
    }

    /// Check if we're at a block end (for recovery)
    pub fn check_block_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::RightBrace | TokenKind::Eof)
    }

    // Error recovery

    /// Synchronize after an error by finding the next statement boundary
    fn synchronize(&mut self) {
        self.recovery_mode = true;
        self.advance(); // Skip the problematic token

        while !self.is_at_end() {
            // Stop at statement boundaries
            if self.previous().kind == TokenKind::Semicolon {
                self.recovery_mode = false;
                return;
            }

            // Stop at keywords that start new statements
            match self.peek().kind {
                TokenKind::Module
                | TokenKind::Function
                | TokenKind::Type
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return => {
                    self.recovery_mode = false;
                    return;
                }
                _ => {}
            }

            self.advance();
        }

        self.recovery_mode = false;
    }

    // Parsing methods (stubs for now)

    /// Parse a top-level item
    fn parse_item(&mut self) -> ParseResult<AstNode<Item>> {
        let start_span = self.current_span();
        
        match self.peek().kind {
            TokenKind::Module => {
                let module = self.parse_module()?;
                let mut item_node = self.create_node(Item::Module(module.kind), module.span);
                
                // Add module-specific AI context
                if self.config.extract_ai_context {
                    let ai_context = prism_ast::AiContext::new()
                        .with_purpose("Define module boundary and capabilities".to_string())
                        .with_domain("Module System".to_string());
                    item_node = item_node.with_ai_context(ai_context);
                }
                
                Ok(item_node)
            }
            TokenKind::Function | TokenKind::Fn => {
                let function = self.parse_function()?;
                let mut item_node = self.create_node(Item::Function(function.kind), function.span);
                
                // Add function-specific AI context
                if self.config.extract_ai_context {
                    let ai_context = prism_ast::AiContext::new()
                        .with_purpose("Define function with semantic contracts".to_string())
                        .with_domain("Function Definition".to_string());
                    item_node = item_node.with_ai_context(ai_context);
                }
                
                Ok(item_node)
            }
            TokenKind::Type => {
                let type_decl = self.parse_type_definition()?;
                let mut item_node = self.create_node(Item::Type(type_decl.kind), type_decl.span);
                
                // Add type-specific AI context
                if self.config.extract_ai_context {
                    let ai_context = prism_ast::AiContext::new()
                        .with_purpose("Define semantic type with business constraints".to_string())
                        .with_domain("Type System".to_string());
                    item_node = item_node.with_ai_context(ai_context);
                }
                
                Ok(item_node)
            }
            _ => {
                let stmt = self.parse_stmt()?;
                let span = stmt.span;
                let item = self.create_node(Item::Statement(stmt.kind), span);
                Ok(item)
            }
        }
    }

    /// Parse module declaration with semantic context
    fn parse_module(&mut self) -> ParseResult<AstNode<prism_ast::ModuleDecl>> {
        let start_span = self.current_span();
        self.consume(TokenKind::Module, "Expected 'module'")?;
        
        let name = self.consume_identifier("Expected module name")?;
        
        // Parse optional capability annotation
        let capability = if self.check(TokenKind::LeftBrace) {
            Some("default_capability".to_string())
        } else {
            None
        };
        
        // Create module declaration
        let module_decl = prism_ast::ModuleDecl {
            name: prism_common::symbol::Symbol::intern(&name),
            capability,
            description: Some(format!("Module: {}", name)),
            dependencies: Vec::new(),
            stability: prism_ast::StabilityLevel::Experimental,
            version: Some("1.0.0".to_string()),
            sections: Vec::new(),
            ai_context: Some(format!("Module '{}' provides business functionality", name)),
            visibility: prism_ast::Visibility::Public,
        };
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(module_decl, full_span))
    }

    /// Parse function declaration with semantic context
    fn parse_function(&mut self) -> ParseResult<AstNode<prism_ast::FunctionDecl>> {
        let start_span = self.current_span();
        
        // Consume function keyword
        if self.check(TokenKind::Function) {
            self.advance();
        } else {
            self.consume(TokenKind::Fn, "Expected 'fn'")?;
        }
        
        let name = self.consume_identifier("Expected function name")?;
        
        // Parse parameters (simplified)
        let parameters = if self.check(TokenKind::LeftParen) {
            self.advance(); // consume '('
            let mut params = Vec::new();
            
            while !self.check(TokenKind::RightParen) && !self.is_at_end() {
                let param_name = self.consume_identifier("Expected parameter name")?;
                
                // Optional type annotation
                let type_annotation = if self.check(TokenKind::Colon) {
                    self.advance();
                    Some(self.parse_type_annotation()?)
                } else {
                    None
                };
                
                params.push(prism_ast::Parameter {
                    name: prism_common::symbol::Symbol::intern(&param_name),
                    type_annotation,
                    default_value: None,
                    is_mutable: false,
                });
                
                if self.check(TokenKind::Comma) {
                    self.advance();
                }
            }
            
            self.consume(TokenKind::RightParen, "Expected ')'")?;
            params
        } else {
            Vec::new()
        };
        
        // Parse return type
        let return_type = if self.check(TokenKind::Arrow) {
            self.advance();
            Some(Box::new(self.parse_type_annotation()?))
        } else {
            None
        };
        
        // Create function declaration
        let function_decl = prism_ast::FunctionDecl {
            name: prism_common::symbol::Symbol::intern(&name),
            parameters,
            return_type,
            body: None, // For now, we'll skip body parsing
            visibility: prism_ast::Visibility::Public,
            attributes: Vec::new(),
            contracts: None,
            is_async: false,
        };
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(function_decl, full_span))
    }

    /// Parse type definition with semantic context
    fn parse_type_definition(&mut self) -> ParseResult<AstNode<prism_ast::TypeDecl>> {
        let start_span = self.current_span();
        self.consume(TokenKind::Type, "Expected 'type'")?;
        
        let name = self.consume_identifier("Expected type name")?;
        
        // Parse optional generic parameters
        let type_parameters = if self.check(TokenKind::Less) {
            self.parse_type_parameters()?
        } else {
            Vec::new()
        };
        
        // Parse type definition
        self.consume(TokenKind::Assign, "Expected '=' after type name")?;
        let type_def = self.parse_type_annotation()?;
        
        // Determine type kind
        let kind = match &type_def.kind {
            prism_ast::Type::Semantic(semantic_type) => {
                prism_ast::TypeKind::Semantic(semantic_type.clone())
            }
            _ => prism_ast::TypeKind::Alias(type_def)
        };
        
        // Create type declaration
        let type_decl = prism_ast::TypeDecl {
            name: prism_common::symbol::Symbol::intern(&name),
            type_parameters: type_parameters.into_iter().map(|name| {
                prism_ast::TypeParameter {
                    name: prism_common::symbol::Symbol::intern(&name),
                    bounds: Vec::new(),
                    default: None,
                }
            }).collect(),
            kind,
            visibility: prism_ast::Visibility::Public,
        };
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(type_decl, full_span))
    }

    /// Parse type parameters (simplified)
    fn parse_type_parameters(&mut self) -> ParseResult<Vec<String>> {
        self.consume(TokenKind::Less, "Expected '<'")?;
        
        let mut params = Vec::new();
        
        while !self.check(TokenKind::Greater) && !self.is_at_end() {
            let param = self.consume_identifier("Expected type parameter name")?;
            params.push(param);
            
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        
        self.consume(TokenKind::Greater, "Expected '>'")?;
        
        Ok(params)
    }

    /// Parse a statement
    fn parse_stmt(&mut self) -> ParseResult<AstNode<Stmt>> {
        let span = self.current_span();
        
        // Create a placeholder expression statement
        let expr_stmt = prism_ast::ExpressionStmt {
            expression: self.create_node(
                prism_ast::Expr::Literal(prism_ast::LiteralExpr {
                    value: prism_ast::LiteralValue::Null,
                }),
                span,
            ),
        };
        
        Ok(self.create_node(expr_stmt, span))
    }

    /// Parse a type annotation
    fn parse_type_annotation(&mut self) -> ParseResult<AstNode<Type>> {
        let span = self.current_span();
        
        match self.peek().kind {
            TokenKind::Identifier(ref name) => {
                let name_str = name.clone();
                self.advance();
                
                // Check for generic arguments
                let type_arguments = if self.check(TokenKind::Less) {
                    self.parse_generic_arguments()?
                } else {
                    Vec::new()
                };
                
                // Check for semantic constraints with "where" keyword
                if self.check(TokenKind::Where) {
                    self.parse_semantic_type_with_constraints(name_str, type_arguments, span)
                } else {
                    // Simple named type
                    let named_type = prism_ast::NamedType {
                        name: prism_common::symbol::Symbol::intern(&name_str),
                        type_arguments,
                    };
                    Ok(self.create_node(Type::Named(named_type), span))
                }
            }
            TokenKind::LeftParen => {
                // Tuple type
                self.parse_tuple_type()
            }
            TokenKind::LeftBracket => {
                // Array type
                self.parse_array_type()
            }
            _ => {
                // Create a placeholder type for error recovery
                let named_type = prism_ast::NamedType {
                    name: prism_common::symbol::Symbol::intern("unknown"),
                    type_arguments: Vec::new(),
                };
                Ok(self.create_node(Type::Named(named_type), span))
            }
        }
    }

    /// Parse semantic type with constraints
    fn parse_semantic_type_with_constraints(
        &mut self,
        base_name: String,
        type_arguments: Vec<AstNode<Type>>,
        start_span: Span,
    ) -> ParseResult<AstNode<Type>> {
        // Parse base type
        let base_type = Box::new(self.create_node(
            Type::Named(prism_ast::NamedType {
                name: prism_common::symbol::Symbol::intern(&base_name),
                type_arguments,
            }),
            start_span,
        ));

        // Consume "where" keyword
        self.consume(TokenKind::Where, "Expected 'where' after type name")?;

        // Parse constraint block
        self.consume(TokenKind::LeftBrace, "Expected '{' after 'where'")?;

        let mut constraints = Vec::new();
        let mut business_rules = Vec::new();
        let mut validation_rules = Vec::new();
        let mut ai_context = None;
        let mut security_classification = prism_ast::SecurityClassification::Public;
        let mut compliance_requirements = Vec::new();

        while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
            if self.check(TokenKind::Identifier("".to_string())) {
                let constraint_name = self.advance().kind.clone();
                
                if let TokenKind::Identifier(name) = constraint_name {
                    self.consume(TokenKind::Colon, "Expected ':' after constraint name")?;
                    
                    match name.as_str() {
                        "min_value" | "max_value" => {
                            let value = self.parse_constraint_value()?;
                            constraints.push(self.create_range_constraint(&name, value)?);
                        }
                        "min_length" | "max_length" => {
                            let value = self.parse_constraint_value()?;
                            constraints.push(self.create_length_constraint(&name, value)?);
                        }
                        "pattern" => {
                            let pattern = self.parse_string_literal()?;
                            constraints.push(prism_ast::TypeConstraint::Pattern(
                                prism_ast::PatternConstraint {
                                    pattern,
                                    flags: Vec::new(),
                                }
                            ));
                        }
                        "format" => {
                            let format = self.parse_string_literal()?;
                            constraints.push(prism_ast::TypeConstraint::Format(
                                prism_ast::FormatConstraint {
                                    format,
                                    parameters: std::collections::HashMap::new(),
                                }
                            ));
                        }
                        "precision" | "currency" | "non_negative" | "immutable" | "validated" => {
                            let value = self.parse_constraint_value()?;
                            constraints.push(prism_ast::TypeConstraint::Custom(
                                prism_ast::CustomConstraint {
                                    name: name.clone(),
                                    expression: self.create_literal_expr(value)?,
                                }
                            ));
                        }
                        "business_rule" => {
                            let rule = self.parse_business_rule()?;
                            business_rules.push(rule.description.clone());
                            constraints.push(prism_ast::TypeConstraint::BusinessRule(rule));
                        }
                        "security_classification" => {
                            security_classification = self.parse_security_classification()?;
                        }
                        "compliance" => {
                            compliance_requirements = self.parse_compliance_requirements()?;
                        }
                        "ai_context" => {
                            ai_context = Some(self.parse_string_literal()?);
                        }
                        _ => {
                            // Custom constraint
                            let value = self.parse_constraint_value()?;
                            constraints.push(prism_ast::TypeConstraint::Custom(
                                prism_ast::CustomConstraint {
                                    name: name.clone(),
                                    expression: self.create_literal_expr(value)?,
                                }
                            ));
                        }
                    }
                }
            }

            // Optional comma
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }

        self.consume(TokenKind::RightBrace, "Expected '}' after constraints")?;

        // Create semantic type metadata
        let metadata = prism_ast::SemanticTypeMetadata {
            business_rules,
            examples: Vec::new(),
            validation_rules,
            ai_context,
            security_classification,
            compliance_requirements,
        };

        // Create semantic type
        let semantic_type = prism_ast::SemanticType {
            base_type,
            constraints,
            metadata,
        };

        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        // Validate the semantic type if validation is enabled
        if self.config.semantic_metadata {
            let validation_result = self.constraint_validator.validate_semantic_type(
                &semantic_type,
                full_span,
            );
            
            // Convert validation errors to parse errors
            for validation_error in validation_result.errors {
                let parse_error = ParseError::semantic_error(
                    "Semantic type validation failed".to_string(),
                    validation_error.to_string(),
                    full_span,
                );
                self.errors.push(parse_error);
            }
            
            // Add AI insights as metadata if available
            if !validation_result.ai_summary.ai_insights.is_empty() {
                // Create enhanced metadata with validation insights
                let enhanced_metadata = prism_ast::SemanticTypeMetadata {
                    business_rules: semantic_type.metadata.business_rules.clone(),
                    examples: semantic_type.metadata.examples.clone(),
                    validation_rules: semantic_type.metadata.validation_rules.clone(),
                    ai_context: semantic_type.metadata.ai_context.clone(),
                    security_classification: semantic_type.metadata.security_classification.clone(),
                    compliance_requirements: semantic_type.metadata.compliance_requirements.clone(),
                };
                
                let validated_semantic_type = prism_ast::SemanticType {
                    base_type: semantic_type.base_type,
                    constraints: semantic_type.constraints,
                    metadata: enhanced_metadata,
                };
                
                let mut node = self.create_node(Type::Semantic(validated_semantic_type), full_span);
                
                // Add validation insights as semantic annotations
                for insight in validation_result.ai_summary.ai_insights {
                    node = node.with_semantic_annotation(insight);
                }
                
                return Ok(node);
            }
        }
        
        Ok(self.create_node(Type::Semantic(semantic_type), full_span))
    }

    /// Parse constraint value (number, string, boolean, or expression)
    fn parse_constraint_value(&mut self) -> ParseResult<ConstraintValue> {
        match &self.peek().kind {
            TokenKind::IntegerLiteral(value) => {
                let val = *value;
                self.advance();
                Ok(ConstraintValue::Integer(val))
            }
            TokenKind::FloatLiteral(value) => {
                let val = *value;
                self.advance();
                Ok(ConstraintValue::Float(val))
            }
            TokenKind::StringLiteral(value) => {
                let val = value.clone();
                self.advance();
                Ok(ConstraintValue::String(val))
            }
            TokenKind::True => {
                self.advance();
                Ok(ConstraintValue::Boolean(true))
            }
            TokenKind::False => {
                self.advance();
                Ok(ConstraintValue::Boolean(false))
            }
            _ => {
                // Parse as expression
                let expr = self.parse_expression()?;
                Ok(ConstraintValue::Expression(expr))
            }
        }
    }

    /// Create range constraint from name and value
    fn create_range_constraint(&self, name: &str, value: ConstraintValue) -> ParseResult<prism_ast::TypeConstraint> {
        let expr = self.constraint_value_to_expr(value)?;
        
        let constraint = match name {
            "min_value" => prism_ast::RangeConstraint {
                min: Some(expr),
                max: None,
                inclusive: true,
            },
            "max_value" => prism_ast::RangeConstraint {
                min: None,
                max: Some(expr),
                inclusive: true,
            },
            _ => return Err(ParseError::invalid_syntax(
                "range_constraint".to_string(),
                format!("Invalid range constraint: {}", name),
                self.current_span(),
            )),
        };

        Ok(prism_ast::TypeConstraint::Range(constraint))
    }

    /// Create length constraint from name and value
    fn create_length_constraint(&self, name: &str, value: ConstraintValue) -> ParseResult<prism_ast::TypeConstraint> {
        let length_value = match value {
            ConstraintValue::Integer(val) => val as usize,
            _ => return Err(ParseError::invalid_syntax(
                "length_constraint".to_string(),
                "Length constraint must be an integer".to_string(),
                self.current_span(),
            )),
        };

        let constraint = match name {
            "min_length" => prism_ast::LengthConstraint {
                min_length: Some(length_value),
                max_length: None,
            },
            "max_length" => prism_ast::LengthConstraint {
                min_length: None,
                max_length: Some(length_value),
            },
            _ => return Err(ParseError::invalid_syntax(
                "length_constraint".to_string(),
                format!("Invalid length constraint: {}", name),
                self.current_span(),
            )),
        };

        Ok(prism_ast::TypeConstraint::Length(constraint))
    }

    /// Parse business rule constraint
    fn parse_business_rule(&mut self) -> ParseResult<prism_ast::BusinessRuleConstraint> {
        let description = self.parse_string_literal()?;
        
        // Optional expression
        let expression = if self.check(TokenKind::Comma) {
            self.advance();
            self.parse_expression()?
        } else {
            // Create a placeholder expression
            self.create_literal_expr(ConstraintValue::Boolean(true))?
        };

        Ok(prism_ast::BusinessRuleConstraint {
            description,
            expression,
            priority: 1, // Default priority
        })
    }

    /// Parse security classification
    fn parse_security_classification(&mut self) -> ParseResult<prism_ast::SecurityClassification> {
        let classification = self.parse_string_literal()?;
        
        match classification.to_lowercase().as_str() {
            "public" => Ok(prism_ast::SecurityClassification::Public),
            "internal" => Ok(prism_ast::SecurityClassification::Internal),
            "confidential" => Ok(prism_ast::SecurityClassification::Confidential),
            "restricted" => Ok(prism_ast::SecurityClassification::Restricted),
            "top_secret" => Ok(prism_ast::SecurityClassification::TopSecret),
            _ => Err(ParseError::invalid_syntax(
                "security_classification".to_string(),
                format!("Invalid security classification: {}", classification),
                self.current_span(),
            )),
        }
    }

    /// Parse compliance requirements
    fn parse_compliance_requirements(&mut self) -> ParseResult<Vec<String>> {
        self.consume(TokenKind::LeftBracket, "Expected '[' for compliance requirements")?;
        
        let mut requirements = Vec::new();
        
        while !self.check(TokenKind::RightBracket) && !self.is_at_end() {
            let requirement = self.parse_string_literal()?;
            requirements.push(requirement);
            
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        
        self.consume(TokenKind::RightBracket, "Expected ']' after compliance requirements")?;
        
        Ok(requirements)
    }

    /// Convert constraint value to expression
    fn constraint_value_to_expr(&self, value: ConstraintValue) -> ParseResult<AstNode<prism_ast::Expr>> {
        match value {
            ConstraintValue::Integer(val) => {
                Ok(self.create_literal_expr(ConstraintValue::Integer(val))?)
            }
            ConstraintValue::Float(val) => {
                Ok(self.create_literal_expr(ConstraintValue::Float(val))?)
            }
            ConstraintValue::String(val) => {
                Ok(self.create_literal_expr(ConstraintValue::String(val))?)
            }
            ConstraintValue::Boolean(val) => {
                Ok(self.create_literal_expr(ConstraintValue::Boolean(val))?)
            }
            ConstraintValue::Expression(expr) => Ok(expr),
        }
    }

    /// Create literal expression from constraint value
    fn create_literal_expr(&self, value: ConstraintValue) -> ParseResult<AstNode<prism_ast::Expr>> {
        let span = self.current_span();
        
        let literal_value = match value {
            ConstraintValue::Integer(val) => prism_ast::LiteralValue::Integer(val),
            ConstraintValue::Float(val) => prism_ast::LiteralValue::Float(val),
            ConstraintValue::String(val) => prism_ast::LiteralValue::String(val),
            ConstraintValue::Boolean(val) => prism_ast::LiteralValue::Boolean(val),
            ConstraintValue::Expression(_) => {
                return Err(ParseError::invalid_syntax(
                    "literal_expression".to_string(),
                    "Cannot create literal from expression".to_string(),
                    span,
                ));
            }
        };

        Ok(self.create_node(
            prism_ast::Expr::Literal(prism_ast::LiteralExpr {
                value: literal_value,
            }),
            span,
        ))
    }

    /// Parse tuple type
    fn parse_tuple_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.current_span();
        self.consume(TokenKind::LeftParen, "Expected '('")?;
        
        let mut elements = Vec::new();
        
        while !self.check(TokenKind::RightParen) && !self.is_at_end() {
            let element = self.parse_type_annotation()?;
            elements.push(element);
            
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')'")?;
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(
            Type::Tuple(prism_ast::TupleType { elements }),
            full_span,
        ))
    }

    /// Parse array type
    fn parse_array_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.current_span();
        self.consume(TokenKind::LeftBracket, "Expected '['")?;
        
        let element_type = Box::new(self.parse_type_annotation()?);
        
        // Optional size specification
        let size = if self.check(TokenKind::Semicolon) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        
        self.consume(TokenKind::RightBracket, "Expected ']'")?;
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(
            Type::Array(prism_ast::ArrayType {
                element_type,
                size,
            }),
            full_span,
        ))
    }

    /// Parse generic arguments (e.g., <T, U>)
    fn parse_generic_arguments(&mut self) -> ParseResult<Vec<AstNode<Type>>> {
        self.consume(TokenKind::Less, "Expected '<' for generic arguments")?;
        
        let mut arguments = Vec::new();
        
        while !self.check(TokenKind::Greater) && !self.is_at_end() {
            let type_arg = self.parse_type_annotation()?;
            arguments.push(type_arg);
            
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        
        self.consume(TokenKind::Greater, "Expected '>' for generic arguments")?;
        
        Ok(arguments)
    }

    /// Parse string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match &self.peek().kind {
            TokenKind::StringLiteral(value) => {
                let val = value.clone();
                self.advance();
                Ok(val)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::StringLiteral(String::new())],
                self.peek().kind.clone(),
                self.current_span(),
            )),
        }
    }

    /// Parse number literal
    fn parse_number_literal(&mut self) -> ParseResult<f64> {
        match &self.peek().kind {
            TokenKind::IntegerLiteral(value) => {
                let val = *value as f64;
                self.advance();
                Ok(val)
            }
            TokenKind::FloatLiteral(value) => {
                let val = *value;
                self.advance();
                Ok(val)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::IntegerLiteral(0), TokenKind::FloatLiteral(0.0)],
                self.peek().kind.clone(),
                self.current_span(),
            )),
        }
    }

    /// Check if current token matches identifier pattern
    fn check_identifier(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
    }

    /// Consume identifier token
    fn consume_identifier(&mut self, message: &str) -> ParseResult<String> {
        match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let val = name.clone();
                self.advance();
                Ok(val)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::Identifier(String::new())],
                self.peek().kind.clone(),
                self.current_span(),
            )),
        }
    }

    /// Enhanced create_node that integrates semantic context
    fn create_node<T>(&mut self, kind: T, span: Span) -> AstNode<T> {
        let id = self.next_node_id();
        let mut node = AstNode::new(kind, span, id);
        
        // Integrate semantic context if available
        if self.config.extract_ai_context {
            node = self.integrate_semantic_context(node);
        }
        
        node
    }

    /// Update parse_item to use semantic context integration
    fn parse_item(&mut self) -> ParseResult<AstNode<Item>> {
        let start_span = self.current_span();
        
        match self.peek().kind {
            TokenKind::Module => {
                let module = self.parse_module()?;
                let mut item_node = self.create_node(Item::Module(module.kind), module.span);
                
                // Add module-specific AI context
                if self.config.extract_ai_context {
                    let ai_context = prism_ast::AiContext::new()
                        .with_purpose("Define module boundary and capabilities".to_string())
                        .with_domain("Module System".to_string());
                    item_node = item_node.with_ai_context(ai_context);
                }
                
                Ok(item_node)
            }
            TokenKind::Function | TokenKind::Fn => {
                let function = self.parse_function()?;
                let mut item_node = self.create_node(Item::Function(function.kind), function.span);
                
                // Add function-specific AI context
                if self.config.extract_ai_context {
                    let ai_context = prism_ast::AiContext::new()
                        .with_purpose("Define function with semantic contracts".to_string())
                        .with_domain("Function Definition".to_string());
                    item_node = item_node.with_ai_context(ai_context);
                }
                
                Ok(item_node)
            }
            TokenKind::Type => {
                let type_decl = self.parse_type_definition()?;
                let mut item_node = self.create_node(Item::Type(type_decl.kind), type_decl.span);
                
                // Add type-specific AI context
                if self.config.extract_ai_context {
                    let ai_context = prism_ast::AiContext::new()
                        .with_purpose("Define semantic type with business constraints".to_string())
                        .with_domain("Type System".to_string());
                    item_node = item_node.with_ai_context(ai_context);
                }
                
                Ok(item_node)
            }
            _ => {
                let stmt = self.parse_stmt()?;
                let span = stmt.span;
                let item = self.create_node(Item::Statement(stmt.kind), span);
                Ok(item)
            }
        }
    }

    /// Parse module declaration with semantic context
    fn parse_module(&mut self) -> ParseResult<AstNode<prism_ast::ModuleDecl>> {
        let start_span = self.current_span();
        self.consume(TokenKind::Module, "Expected 'module'")?;
        
        let name = self.consume_identifier("Expected module name")?;
        
        // Parse optional capability annotation
        let capability = if self.check(TokenKind::LeftBrace) {
            Some("default_capability".to_string())
        } else {
            None
        };
        
        // Create module declaration
        let module_decl = prism_ast::ModuleDecl {
            name: prism_common::symbol::Symbol::intern(&name),
            capability,
            description: Some(format!("Module: {}", name)),
            dependencies: Vec::new(),
            stability: prism_ast::StabilityLevel::Experimental,
            version: Some("1.0.0".to_string()),
            sections: Vec::new(),
            ai_context: Some(format!("Module '{}' provides business functionality", name)),
            visibility: prism_ast::Visibility::Public,
        };
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(module_decl, full_span))
    }

    /// Parse function declaration with semantic context
    fn parse_function(&mut self) -> ParseResult<AstNode<prism_ast::FunctionDecl>> {
        let start_span = self.current_span();
        
        // Consume function keyword
        if self.check(TokenKind::Function) {
            self.advance();
        } else {
            self.consume(TokenKind::Fn, "Expected 'fn'")?;
        }
        
        let name = self.consume_identifier("Expected function name")?;
        
        // Parse parameters (simplified)
        let parameters = if self.check(TokenKind::LeftParen) {
            self.advance(); // consume '('
            let mut params = Vec::new();
            
            while !self.check(TokenKind::RightParen) && !self.is_at_end() {
                let param_name = self.consume_identifier("Expected parameter name")?;
                
                // Optional type annotation
                let type_annotation = if self.check(TokenKind::Colon) {
                    self.advance();
                    Some(self.parse_type_annotation()?)
                } else {
                    None
                };
                
                params.push(prism_ast::Parameter {
                    name: prism_common::symbol::Symbol::intern(&param_name),
                    type_annotation,
                    default_value: None,
                    is_mutable: false,
                });
                
                if self.check(TokenKind::Comma) {
                    self.advance();
                }
            }
            
            self.consume(TokenKind::RightParen, "Expected ')'")?;
            params
        } else {
            Vec::new()
        };
        
        // Parse return type
        let return_type = if self.check(TokenKind::Arrow) {
            self.advance();
            Some(Box::new(self.parse_type_annotation()?))
        } else {
            None
        };
        
        // Create function declaration
        let function_decl = prism_ast::FunctionDecl {
            name: prism_common::symbol::Symbol::intern(&name),
            parameters,
            return_type,
            body: None, // For now, we'll skip body parsing
            visibility: prism_ast::Visibility::Public,
            attributes: Vec::new(),
            contracts: None,
            is_async: false,
        };
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(function_decl, full_span))
    }

    /// Parse type definition with semantic context
    fn parse_type_definition(&mut self) -> ParseResult<AstNode<prism_ast::TypeDecl>> {
        let start_span = self.current_span();
        self.consume(TokenKind::Type, "Expected 'type'")?;
        
        let name = self.consume_identifier("Expected type name")?;
        
        // Parse optional generic parameters
        let type_parameters = if self.check(TokenKind::Less) {
            self.parse_type_parameters()?
        } else {
            Vec::new()
        };
        
        // Parse type definition
        self.consume(TokenKind::Assign, "Expected '=' after type name")?;
        let type_def = self.parse_type_annotation()?;
        
        // Determine type kind
        let kind = match &type_def.kind {
            prism_ast::Type::Semantic(semantic_type) => {
                prism_ast::TypeKind::Semantic(semantic_type.clone())
            }
            _ => prism_ast::TypeKind::Alias(type_def)
        };
        
        // Create type declaration
        let type_decl = prism_ast::TypeDecl {
            name: prism_common::symbol::Symbol::intern(&name),
            type_parameters: type_parameters.into_iter().map(|name| {
                prism_ast::TypeParameter {
                    name: prism_common::symbol::Symbol::intern(&name),
                    bounds: Vec::new(),
                    default: None,
                }
            }).collect(),
            kind,
            visibility: prism_ast::Visibility::Public,
        };
        
        let end_span = self.current_span();
        let full_span = Span::new(start_span.start, end_span.end, self.source_id);
        
        Ok(self.create_node(type_decl, full_span))
    }

    /// Parse type parameters (simplified)
    fn parse_type_parameters(&mut self) -> ParseResult<Vec<String>> {
        self.consume(TokenKind::Less, "Expected '<'")?;
        
        let mut params = Vec::new();
        
        while !self.check(TokenKind::Greater) && !self.is_at_end() {
            let param = self.consume_identifier("Expected type parameter name")?;
            params.push(param);
            
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        
        self.consume(TokenKind::Greater, "Expected '>'")?;
        
        Ok(params)
    }

    /// Integrate semantic context from lexer into AST node
    fn integrate_semantic_context<T>(&mut self, mut node: AstNode<T>) -> AstNode<T> {
        if self.config.extract_ai_context {
            // Get semantic context from the current token if available
            if let Some(semantic_context) = &self.peek().semantic_context {
                // Convert lexer semantic context to AST AI context
                let ai_context = prism_ast::AiContext::new()
                    .with_purpose(semantic_context.purpose.clone().unwrap_or_default())
                    .with_domain(semantic_context.domain.clone().unwrap_or_default());
                
                // Add AI hints as semantic annotations
                for hint in &semantic_context.ai_hints {
                    node = node.with_semantic_annotation(hint.clone());
                }
                
                // Add security implications
                for implication in &semantic_context.security_implications {
                    node.metadata.security_implications.push(implication.clone());
                }
                
                // Set AI context
                node = node.with_ai_context(ai_context);
            }
        }
        
        node
    }

    // Expression parsing with Pratt parser

    /// Parse an expression with the given minimum precedence
    fn parse_precedence(&mut self, min_precedence: Precedence) -> ParseResult<AstNode<Expr>> {
        let mut left = self.parse_prefix()?;

        while !self.is_at_end() {
            let token = &self.peek().kind;
            
            if let Some(precedence) = infix_precedence(token) {
                if precedence < min_precedence {
                    break;
                }
                
                let assoc = associativity(token);
                let next_min_precedence = match assoc {
                    crate::precedence::Associativity::Left => {
                        // For left-associative operators, use higher precedence
                        match precedence {
                            Precedence::None => Precedence::Assignment,
                            Precedence::Assignment => Precedence::Or,
                            Precedence::Or => Precedence::And,
                            Precedence::And => Precedence::BitOr,
                            Precedence::BitOr => Precedence::BitXor,
                            Precedence::BitXor => Precedence::BitAnd,
                            Precedence::BitAnd => Precedence::Equality,
                            Precedence::Equality => Precedence::Comparison,
                            Precedence::Comparison => Precedence::Shift,
                            Precedence::Shift => Precedence::Term,
                            Precedence::Term => Precedence::Factor,
                            Precedence::Factor => Precedence::Unary,
                            Precedence::Unary => Precedence::Power,
                            Precedence::Power => Precedence::Call,
                            Precedence::Call => Precedence::Primary,
                            Precedence::Primary => Precedence::Primary,
                        }
                    }
                    crate::precedence::Associativity::Right => precedence,
                    crate::precedence::Associativity::None => {
                        // For non-associative operators, use higher precedence
                        match precedence {
                            Precedence::None => Precedence::Assignment,
                            Precedence::Assignment => Precedence::Or,
                            Precedence::Or => Precedence::And,
                            Precedence::And => Precedence::BitOr,
                            Precedence::BitOr => Precedence::BitXor,
                            Precedence::BitXor => Precedence::BitAnd,
                            Precedence::BitAnd => Precedence::Equality,
                            Precedence::Equality => Precedence::Comparison,
                            Precedence::Comparison => Precedence::Shift,
                            Precedence::Shift => Precedence::Term,
                            Precedence::Term => Precedence::Factor,
                            Precedence::Factor => Precedence::Unary,
                            Precedence::Unary => Precedence::Power,
                            Precedence::Power => Precedence::Call,
                            Precedence::Call => Precedence::Primary,
                            Precedence::Primary => Precedence::Primary,
                        }
                    }
                };
                
                left = self.parse_infix(left, next_min_precedence)?;
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Parse a prefix expression
    fn parse_prefix(&mut self) -> ParseResult<AstNode<Expr>> {
        let token = self.advance().clone();
        let span = token.span;

        match token.kind {
            TokenKind::IntegerLiteral(value) => {
                Ok(self.create_node(
                    Expr::Literal(prism_ast::LiteralExpr {
                        value: prism_ast::LiteralValue::Integer(value),
                    }),
                    span,
                ))
            }
            TokenKind::FloatLiteral(value) => {
                Ok(self.create_node(
                    Expr::Literal(prism_ast::LiteralExpr {
                        value: prism_ast::LiteralValue::Float(value),
                    }),
                    span,
                ))
            }
            TokenKind::StringLiteral(value) => {
                Ok(self.create_node(
                    Expr::Literal(prism_ast::LiteralExpr {
                        value: prism_ast::LiteralValue::String(value),
                    }),
                    span,
                ))
            }
            TokenKind::True => {
                Ok(self.create_node(
                    Expr::Literal(prism_ast::LiteralExpr {
                        value: prism_ast::LiteralValue::Boolean(true),
                    }),
                    span,
                ))
            }
            TokenKind::False => {
                Ok(self.create_node(
                    Expr::Literal(prism_ast::LiteralExpr {
                        value: prism_ast::LiteralValue::Boolean(false),
                    }),
                    span,
                ))
            }
            TokenKind::Null => {
                Ok(self.create_node(
                    Expr::Literal(prism_ast::LiteralExpr {
                        value: prism_ast::LiteralValue::Null,
                    }),
                    span,
                ))
            }
            TokenKind::Identifier(name) => {
                Ok(self.create_node(
                    Expr::Variable(prism_ast::VariableExpr {
                        name: name.to_string(),
                    }),
                    span,
                ))
            }
            TokenKind::LeftParen => {
                let expr = self.parse_expression()?;
                self.consume(TokenKind::RightParen, "Expected ')' after expression")?;
                Ok(expr)
            }
            _ => {
                Err(ParseError::unexpected_token(
                    vec![TokenKind::IntegerLiteral(0)], // Placeholder
                    token.kind,
                    span,
                ))
            }
        }
    }

    /// Parse an infix expression
    fn parse_infix(&mut self, left: AstNode<Expr>, min_precedence: Precedence) -> ParseResult<AstNode<Expr>> {
        let operator_token = self.advance().clone();
        let right = self.parse_precedence(min_precedence)?;
        
        let span = Span::new(
            left.span.start,
            right.span.end,
            self.source_id,
        );

        let operator = match operator_token.kind {
            TokenKind::Plus => prism_ast::BinaryOperator::Add,
            TokenKind::Minus => prism_ast::BinaryOperator::Subtract,
            TokenKind::Star => prism_ast::BinaryOperator::Multiply,
            TokenKind::Slash => prism_ast::BinaryOperator::Divide,
            TokenKind::Percent => prism_ast::BinaryOperator::Modulo,
            TokenKind::Equal => prism_ast::BinaryOperator::Equal,
            TokenKind::NotEqual => prism_ast::BinaryOperator::NotEqual,
            TokenKind::Less => prism_ast::BinaryOperator::Less,
            TokenKind::Greater => prism_ast::BinaryOperator::Greater,
            TokenKind::LessEqual => prism_ast::BinaryOperator::LessEqual,
            TokenKind::GreaterEqual => prism_ast::BinaryOperator::GreaterEqual,
            TokenKind::AndAnd => prism_ast::BinaryOperator::And,
            TokenKind::OrOr => prism_ast::BinaryOperator::Or,
            _ => {
                return Err(ParseError::unexpected_token(
                    vec![TokenKind::Plus],
                    operator_token.kind,
                    operator_token.span,
                ));
            }
        };

        Ok(self.create_node(
            Expr::Binary(prism_ast::BinaryExpr {
                left: Box::new(left),
                operator,
                right: Box::new(right),
            }),
            span,
        ))
    }

    // Helper methods

    /// Calculate complexity score for a program
    fn calculate_complexity_score(&self, items: &[AstNode<Item>]) -> f64 {
        // Simple complexity calculation based on number of items
        items.len() as f64 * 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::{span::Position, SourceId};
    use prism_lexer::{Lexer, LexerConfig, SemanticLexer};

    fn create_test_tokens(source: &str) -> Vec<Token> {
        let source_id = SourceId::new(1);
        let mut symbol_table = prism_common::symbol::SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        result.tokens
    }

    #[test]
    fn test_parse_simple_semantic_type() {
        let source = r#"Money where { precision: 2, currency: "USD", non_negative: true }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                assert_eq!(semantic_type.constraints.len(), 3);
                
                // Check that constraints were parsed correctly
                let has_precision = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Custom(custom) if custom.name == "precision")
                });
                assert!(has_precision);
                
                let has_currency = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Custom(custom) if custom.name == "currency")
                });
                assert!(has_currency);
                
                let has_non_negative = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Custom(custom) if custom.name == "non_negative")
                });
                assert!(has_non_negative);
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_semantic_type_with_range_constraints() {
        let source = r#"Age where { min_value: 0, max_value: 150 }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                assert_eq!(semantic_type.constraints.len(), 2);
                
                // Check that range constraints were parsed correctly
                let has_min_value = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Range(range) if range.min.is_some())
                });
                assert!(has_min_value);
                
                let has_max_value = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Range(range) if range.max.is_some())
                });
                assert!(has_max_value);
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_semantic_type_with_pattern_constraint() {
        let source = r#"Email where { pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$" }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                assert_eq!(semantic_type.constraints.len(), 1);
                
                // Check that pattern constraint was parsed correctly
                let has_pattern = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Pattern(pattern) 
                        if pattern.pattern.contains("@"))
                });
                assert!(has_pattern);
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_semantic_type_with_length_constraints() {
        let source = r#"Username where { min_length: 3, max_length: 20 }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                assert_eq!(semantic_type.constraints.len(), 2);
                
                // Check that length constraints were parsed correctly
                let has_min_length = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Length(length) if length.min_length.is_some())
                });
                assert!(has_min_length);
                
                let has_max_length = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::Length(length) if length.max_length.is_some())
                });
                assert!(has_max_length);
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_semantic_type_with_security_classification() {
        let source = r#"PatientId where { security_classification: "confidential", compliance: ["HIPAA", "GDPR"] }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                // Check security classification
                assert_eq!(semantic_type.metadata.security_classification, 
                          prism_ast::SecurityClassification::Confidential);
                
                // Check compliance requirements
                assert_eq!(semantic_type.metadata.compliance_requirements.len(), 2);
                assert!(semantic_type.metadata.compliance_requirements.contains(&"HIPAA".to_string()));
                assert!(semantic_type.metadata.compliance_requirements.contains(&"GDPR".to_string()));
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_semantic_type_with_business_rule() {
        let source = r#"AccountBalance where { business_rule: "Balance cannot be negative except for overdraft accounts" }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                assert_eq!(semantic_type.constraints.len(), 1);
                
                // Check that business rule constraint was parsed correctly
                let has_business_rule = semantic_type.constraints.iter().any(|c| {
                    matches!(c, prism_ast::TypeConstraint::BusinessRule(rule) 
                        if rule.description.contains("Balance cannot be negative"))
                });
                assert!(has_business_rule);
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_complex_semantic_type() {
        let source = r#"
        CreditCard where {
            pattern: "^[0-9]{13,19}$",
            min_length: 13,
            max_length: 19,
            security_classification: "restricted",
            compliance: ["PCI-DSS"],
            business_rule: "Must pass Luhn algorithm validation",
            ai_context: "Credit card number with security validation"
        }
        "#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        assert!(result.is_ok());
        let type_node = result.unwrap();
        
        match &type_node.kind {
            Type::Semantic(semantic_type) => {
                // Should have multiple constraints
                assert!(semantic_type.constraints.len() >= 4);
                
                // Check security classification
                assert_eq!(semantic_type.metadata.security_classification, 
                          prism_ast::SecurityClassification::Restricted);
                
                // Check compliance requirements
                assert!(semantic_type.metadata.compliance_requirements.contains(&"PCI-DSS".to_string()));
                
                // Check AI context
                assert!(semantic_type.metadata.ai_context.is_some());
                assert!(semantic_type.metadata.ai_context.as_ref().unwrap().contains("Credit card"));
            }
            _ => panic!("Expected semantic type, got {:?}", type_node.kind),
        }
    }

    #[test]
    fn test_parse_function_with_semantic_types() {
        let source = r#"
        function transfer_funds(
            from: AccountId where { format: "ACC-{uuid}" },
            to: AccountId where { format: "ACC-{uuid}" },
            amount: Money where { precision: 2, currency: "USD", min_value: 0.01 }
        ) -> Result<Transaction, TransferError>
        "#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_function();
        
        assert!(result.is_ok());
        let function_node = result.unwrap();
        
        // Check that function was parsed with semantic type parameters
        assert_eq!(function_node.kind.parameters.len(), 3);
        
        // Check that parameters have semantic types
        for param in &function_node.kind.parameters {
            assert!(param.type_annotation.is_some());
        }
    }

    #[test]
    fn test_parse_type_declaration_with_semantic_type() {
        let source = r#"
        type UserId = UUID where {
            format: "USR-{8}-{4}-{4}-{4}-{12}",
            immutable: true,
            security_classification: "internal"
        }
        "#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_definition();
        
        assert!(result.is_ok());
        let type_decl_node = result.unwrap();
        
        // Check that type declaration was parsed correctly
        assert_eq!(type_decl_node.kind.name.to_string(), "UserId");
        
        // Check that it has semantic type kind
        match &type_decl_node.kind.kind {
            prism_ast::TypeKind::Semantic(semantic_type) => {
                assert!(!semantic_type.constraints.is_empty());
                assert_eq!(semantic_type.metadata.security_classification, 
                          prism_ast::SecurityClassification::Internal);
            }
            _ => panic!("Expected semantic type kind"),
        }
    }

    #[test]
    fn test_constraint_validation_integration() {
        let source = r#"InvalidRange where { min_value: 100, max_value: 0 }"#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_type_annotation();
        
        // Should parse successfully but have validation errors
        assert!(result.is_ok());
        
        // Check that validation errors were recorded
        assert!(!parser.errors.is_empty());
        
        // Check that the error is about range validation
        let has_range_error = parser.errors.iter().any(|error| {
            error.message.contains("validation failed")
        });
        assert!(has_range_error);
    }

    #[test]
    fn test_ai_context_integration() {
        let source = r#"
        module PaymentProcessor {
            type Money = Decimal where {
                precision: 2,
                currency: "USD",
                min_value: 0.00
            }
        }
        "#;
        let tokens = create_test_tokens(source);
        let mut parser = Parser::with_config(tokens, ParseConfig {
            extract_ai_context: true,
            semantic_metadata: true,
            ..Default::default()
        });
        
        let result = parser.parse_program();
        
        assert!(result.is_ok());
        let program = result.unwrap();
        
        // Check that AI context was extracted
        assert!(program.metadata.ai_insights.is_some());
        
        // Check that semantic annotations were added
        if let Some(item) = program.items.first() {
            assert!(item.has_semantic_annotations());
        }
    }
}

/// Helper enum for constraint values
#[derive(Debug, Clone)]
enum ConstraintValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Expression(AstNode<prism_ast::Expr>),
} 