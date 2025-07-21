//! Type Parsing
//!
//! This module embodies the single concept of "Type Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing type annotations with semantic constraints.
//!
//! **Conceptual Responsibility**: Parse type expressions and semantic constraints
//! **What it does**: primitive types, compound types, semantic constraints, business rules
//! **What it doesn't do**: statement parsing, expression parsing, token navigation

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager, parsing_coordinator::ParsingCoordinator},
    analysis::constraint_validation::{ConstraintValidator, ValidationConfig},
};
use prism_ast::{AstNode, Type, TypeKind, TypeConstraint, Attribute, AttributeArgument, SemanticConstraint, ErrorType};
use prism_lexer::{Token, TokenKind};
use prism_common::{span::Span, NodeId};
use std::collections::HashMap;

/// Type parser - handles all type expressions and semantic constraints
/// 
/// This struct embodies the single concept of parsing types.
/// It understands Prism's semantic type system including
/// business rules, constraints, and AI-comprehensible metadata.
pub struct TypeParser<'a> {
    /// Reference to the token stream manager (no ownership)
    token_stream: &'a mut TokenStreamManager,
    /// Reference to coordinator for node creation and error handling
    coordinator: &'a mut ParsingCoordinator,
    /// Constraint validator for semantic validation
    constraint_validator: ConstraintValidator,
}

impl<'a> TypeParser<'a> {
    /// Create a new type parser
    pub fn new(
        token_stream: &'a mut TokenStreamManager,
        coordinator: &'a mut ParsingCoordinator,
    ) -> Self {
        Self {
            token_stream,
            coordinator,
            constraint_validator: ConstraintValidator::new(ValidationConfig::default()),
        }
    }
    
    /// Helper function to combine spans safely
    fn combine_spans(&self, start: Span, end: Span) -> Span {
        start.combine(&end).unwrap_or(start)
    }

    /// Parse a type expression
    pub fn parse_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.token_stream.current_span();
        
        // Parse base type
        let mut base_type = self.parse_primary_type()?;
        
        // Handle type modifiers and compound types
        loop {
            match self.token_stream.current_kind() {
                // Array type: Type[]
                TokenKind::LeftBracket => {
                    self.token_stream.advance();
                    self.token_stream.expect(TokenKind::RightBracket)?;
                    
                    let end_span = self.token_stream.previous_span();
                    let span = self.combine_spans(start_span, end_span);
                    
                    base_type = self.coordinator.create_node(
                        Type::Array(prism_ast::ArrayType {
                            element_type: Box::new(base_type),
                            size: None,
                        }),
                        span,
                    );
                }
                
                // Optional type: Type?
                TokenKind::Question => {
                    self.token_stream.advance();
                    
                    let end_span = self.token_stream.previous_span();
                    let span = Span::combine(start_span, end_span);
                    
                    base_type = self.coordinator.create_type_node(
                        TypeKind::Optional(base_type),
                        span,
                    );
                }
                
                // Union type: Type | Type
                TokenKind::Pipe => {
                    let mut types = vec![base_type];
                    
                    while self.token_stream.consume(TokenKind::Pipe) {
                        types.push(self.parse_base_type()?);
                    }
                    
                    let end_span = self.token_stream.current_span();
                    let span = Span::combine(start_span, end_span);
                    
                    base_type = self.coordinator.create_type_node(
                        TypeKind::Union(types),
                        span,
                    );
                }
                
                _ => break,
            }
        }
        
        // Parse semantic constraints if present
        if self.token_stream.check(TokenKind::Where) {
            base_type = self.parse_constrained_type(base_type, start_span)?;
        }
        
        Ok(base_type)
    }

    /// Parse a primary type
    fn parse_primary_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            // Optional type (?)
            TokenKind::Question => {
                self.token_stream.advance();
                let base_type = self.parse_primary_type()?;
                let end_span = self.token_stream.current_span();
                let span = start_span.combine(&end_span).unwrap_or(start_span);
                
                // For now, represent optional as a union with null
                let null_type = AstNode::new(
                    Type::Primitive(prism_ast::PrimitiveType::Unit),
                    span,
                    prism_common::NodeId::new(0),
                );
                
                let union_type = prism_ast::UnionType {
                    members: vec![base_type, null_type],
                    discriminant: None,
                    constraints: Vec::new(),
                    common_operations: Vec::new(),
                    metadata: prism_ast::UnionMetadata::default(),
                };
                
                Ok(self.coordinator.create_node(
                    Type::Union(Box::new(union_type)),
                    span,
                ))
            }
            
            // Union type (|)
            TokenKind::Pipe => {
                self.token_stream.advance();
                let mut types = Vec::new();
                
                // Parse first type
                types.push(self.parse_primary_type()?);
                
                // Parse additional types
                while self.token_stream.check(TokenKind::Pipe) {
                    self.token_stream.advance();
                    types.push(self.parse_primary_type()?);
                }
                
                let end_span = self.token_stream.current_span();
                let span = start_span.combine(&end_span).unwrap_or(start_span);
                
                let union_type = prism_ast::UnionType {
                    members: types,
                    discriminant: None,
                    constraints: Vec::new(),
                    common_operations: Vec::new(),
                    metadata: prism_ast::UnionMetadata::default(),
                };
                
                Ok(self.coordinator.create_node(
                    Type::Union(Box::new(union_type)),
                    span,
                ))
            }
            
            // Primitive types
            TokenKind::Identifier(type_name) => {
                match type_name.as_str() {
                    "bool" | "boolean" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::Boolean),
                            span,
                        ))
                    }
                    "i32" | "int" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32))),
                            span,
                        ))
                    }
                    "u64" | "uint" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Unsigned(64))),
                            span,
                        ))
                    }
                    "f64" | "float" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64)),
                            span,
                        ))
                    }
                    "string" | "str" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::String),
                            span,
                        ))
                    }
                    "char" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::Char),
                            span,
                        ))
                    }
                    "()" | "unit" => {
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_node(
                            Type::Primitive(prism_ast::PrimitiveType::Unit),
                            span,
                        ))
                    }
                    _ => {
                        // Named type
                        let type_name_symbol = prism_common::symbol::Symbol::intern(type_name);
                        self.token_stream.advance();
                        let span = self.token_stream.previous_span();
                        
                        let named_type = prism_ast::NamedType {
                            name: type_name_symbol,
                            type_arguments: Vec::new(), // TODO: Parse type arguments
                        };
                        
                        Ok(self.coordinator.create_node(
                            Type::Named(named_type),
                            span,
                        ))
                    }
                }
            }
            
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::Identifier("type".to_string())],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse a generic type
    fn parse_generic_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.token_stream.current_span();
        
        // Parse base type
        let base_type = self.parse_primary_type()?;
        
        // Parse generic parameters if present
        if self.token_stream.check(TokenKind::Less) {
            self.token_stream.advance(); // consume '<'
            
            let mut type_arguments = Vec::new();
            
            // Parse first type argument
            type_arguments.push(self.parse_type()?);
            
            // Parse additional type arguments
            while self.token_stream.check(TokenKind::Comma) {
                self.token_stream.advance(); // consume ','
                type_arguments.push(self.parse_type()?);
            }
            
            self.token_stream.expect(TokenKind::Greater)?; // consume '>'
            
            let end_span = self.token_stream.current_span();
            let span = start_span.combine(&end_span).unwrap_or(start_span);
            
            // Update the base type to include type arguments
            if let Type::Named(mut named_type) = base_type.kind {
                named_type.type_arguments = type_arguments;
                Ok(self.coordinator.create_node(
                    Type::Named(named_type),
                    span,
                ))
            } else {
                // For non-named types, create a generic type
                let generic_type = prism_ast::GenericType {
                    parameters: Vec::new(), // TODO: Parse type parameters properly
                    base_type: Box::new(base_type),
                };
                
                Ok(self.coordinator.create_node(
                    Type::Generic(generic_type),
                    span,
                ))
            }
        } else {
            Ok(base_type)
        }
    }

    /// Parse a tuple type
    fn parse_tuple_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.token_stream.current_span();
        
        self.token_stream.expect(TokenKind::LeftParen)?; // consume '('
        
        let mut element_types = Vec::new();
        
        if !self.token_stream.check(TokenKind::RightParen) {
            // Parse first element
            element_types.push(self.parse_type()?);
            
            // Parse additional elements
            while self.token_stream.check(TokenKind::Comma) {
                self.token_stream.advance(); // consume ','
                if self.token_stream.check(TokenKind::RightParen) {
                    break; // Trailing comma
                }
                element_types.push(self.parse_type()?);
            }
        }
        
        self.token_stream.expect(TokenKind::RightParen)?; // consume ')'
        
        let end_span = self.token_stream.current_span();
        let span = start_span.combine(&end_span).unwrap_or(start_span);
        
        let tuple_type = prism_ast::TupleType {
            elements: element_types,
        };
        
        Ok(self.coordinator.create_node(
            Type::Tuple(tuple_type),
            span,
        ))
    }

    /// Parse a function type
    fn parse_function_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.token_stream.current_span();
        
        // Parse parameter types
        let mut parameters = Vec::new();
        
        if self.token_stream.check(TokenKind::LeftParen) {
            self.token_stream.advance(); // consume '('
            
            if !self.token_stream.check(TokenKind::RightParen) {
                // Parse first parameter
                parameters.push(self.parse_type()?);
                
                // Parse additional parameters
                while self.token_stream.check(TokenKind::Comma) {
                    self.token_stream.advance(); // consume ','
                    if self.token_stream.check(TokenKind::RightParen) {
                        break; // Trailing comma
                    }
                    parameters.push(self.parse_type()?);
                }
            }
            
            self.token_stream.expect(TokenKind::RightParen)?; // consume ')'
        }
        
        // Parse arrow
        self.token_stream.expect(TokenKind::Arrow)?; // consume '->'
        
        // Parse return type
        let return_type = Box::new(self.parse_type()?);
        
        let end_span = self.token_stream.current_span();
        let span = start_span.combine(&end_span).unwrap_or(start_span);
        
        let function_type = prism_ast::FunctionType {
            parameters,
            return_type,
            effects: Vec::new(), // TODO: Parse effects
        };
        
        Ok(self.coordinator.create_node(
            Type::Function(function_type),
            span,
        ))
    }

    /// Parse a semantic type with constraints
    fn parse_semantic_type(&mut self) -> ParseResult<AstNode<Type>> {
        let start_span = self.token_stream.current_span();
        
        // Parse base type
        let base_type = Box::new(self.parse_primary_type()?);
        
        // Parse constraints if present
        let mut constraints = Vec::new();
        
        if self.token_stream.check(TokenKind::Where) {
            self.token_stream.advance(); // consume 'where'
            
            // Parse constraint list
            loop {
                if let Some(constraint) = self.parse_type_constraint()? {
                    constraints.push(constraint);
                }
                
                if self.token_stream.check(TokenKind::Comma) {
                    self.token_stream.advance(); // consume ','
                } else {
                    break;
                }
            }
        }
        
        let end_span = self.token_stream.current_span();
        let span = start_span.combine(&end_span).unwrap_or(start_span);
        
        if !constraints.is_empty() {
            let semantic_type = prism_ast::SemanticType {
                base_type,
                constraints,
                metadata: prism_ast::SemanticTypeMetadata::default(),
            };
            
            Ok(self.coordinator.create_node(
                Type::Semantic(semantic_type),
                span,
            ))
        } else {
            Ok(*base_type)
        }
    }

    /// Parse a type constraint
    fn parse_type_constraint(&mut self) -> ParseResult<Option<TypeConstraint>> {
        match self.token_stream.current_kind() {
            TokenKind::Identifier(constraint_name) => {
                match constraint_name.as_str() {
                    "range" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::LeftParen)?;
                        
                        // Parse range bounds (simplified for now)
                        let _min = self.parse_constraint_value()?;
                        self.token_stream.expect(TokenKind::Comma)?;
                        let _max = self.parse_constraint_value()?;
                        
                        self.token_stream.expect(TokenKind::RightParen)?;
                        
                        // Create a simple range constraint
                        let range_constraint = prism_ast::RangeConstraint {
                            min: None, // TODO: Parse actual expressions
                            max: None, // TODO: Parse actual expressions
                            inclusive: true,
                        };
                        
                        Ok(Some(TypeConstraint::Range(range_constraint)))
                    }
                    "length" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::LeftParen)?;
                        
                        // Parse length constraints (simplified)
                        let _length = self.parse_constraint_value()?;
                        
                        self.token_stream.expect(TokenKind::RightParen)?;
                        
                        let length_constraint = prism_ast::LengthConstraint {
                            min_length: None, // TODO: Parse actual values
                            max_length: None, // TODO: Parse actual values
                        };
                        
                        Ok(Some(TypeConstraint::Length(length_constraint)))
                    }
                    "pattern" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::LeftParen)?;
                        
                        let pattern = self.parse_string_literal()?;
                        
                        self.token_stream.expect(TokenKind::RightParen)?;
                        
                        let pattern_constraint = prism_ast::PatternConstraint {
                            pattern,
                            flags: Vec::new(),
                        };
                        
                        Ok(Some(TypeConstraint::Pattern(pattern_constraint)))
                    }
                    "custom" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::LeftParen)?;
                        
                        let rule_text = self.parse_string_literal()?;
                        
                        self.token_stream.expect(TokenKind::RightParen)?;
                        
                        let custom_constraint = prism_ast::CustomConstraint {
                            expression: self.coordinator.create_node(
                        prism_ast::Expr::Literal(prism_ast::LiteralValue::String(rule_text)),
                        span
                    ),
                            parameters: std::collections::HashMap::new(),
                        };
                        
                        Ok(Some(TypeConstraint::Custom(custom_constraint)))
                    }
                    "business_rule" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::LeftParen)?;
                        
                        let rule_text = self.parse_string_literal()?;
                        
                        self.token_stream.expect(TokenKind::RightParen)?;
                        
                        let business_rule_constraint = prism_ast::BusinessRuleConstraint {
                            rule_type: prism_ast::BusinessRuleType::Custom,
                            description: rule_text,
                            parameters: std::collections::HashMap::new(),
                        };
                        
                        Ok(Some(TypeConstraint::BusinessRule(business_rule_constraint)))
                    }
                    _ => Ok(None)
                }
            }
            _ => Ok(None)
        }
    }

    /// Parse a constraint value
    fn parse_constraint_value(&mut self) -> ParseResult<String> {
        match self.token_stream.current_kind() {
            TokenKind::IntegerLiteral(value) => {
                let value_str = value.to_string();
                self.token_stream.advance();
                Ok(value_str)
            }
            TokenKind::StringLiteral(value) => {
                let value_str = value.clone();
                self.token_stream.advance();
                Ok(value_str)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::IntegerLiteral(0), TokenKind::StringLiteral("".to_string())],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Parse a string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match self.token_stream.current_kind() {
            TokenKind::StringLiteral(value) => {
                let value_str = value.clone();
                self.token_stream.advance();
                Ok(value_str)
            }
            _ => Err(ParseError::unexpected_token(
                vec![TokenKind::StringLiteral("".to_string())],
                self.token_stream.current_kind().clone(),
                self.token_stream.current_span(),
            )),
        }
    }

    /// Validate a constraint (placeholder for now)
    fn validate_constraint(&mut self, constraint: &TypeConstraint, span: Span) -> Result<(), String> {
        // For now, all constraints are valid
        // TODO: Implement proper constraint validation
        Ok(())
    }

    /// Validate semantic constraints for a type
    pub fn validate_constraints(&self, type_node: NodeId, constraints: &[SemanticConstraint]) -> ParseResult<()> {
        // Use the constraint validator to check semantic validity
        for constraint in constraints {
            if let Err(error) = self.constraint_validator.validate_constraint(constraint) {
                return Err(ParseError::semantic_error(error, "Constraint validation failed".to_string(), span));
            }
        }
        Ok(())
    }

    /// Get AI-comprehensible metadata for a type
    pub fn extract_type_metadata(&self, type_node: NodeId) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Extract semantic meaning from type structure
        // This would be implemented based on the actual type analysis
        metadata.insert("concept".to_string(), "Type Definition".to_string());
        metadata.insert("purpose".to_string(), "Express semantic constraints".to_string());
        metadata.insert("ai_hint".to_string(), "Types carry business meaning".to_string());
        
        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test utilities would be defined here or in a test module

    #[test]
    fn test_primitive_types() {
        let sources = vec!["i32", "f64", "string", "bool"];
        
        for source in sources {
            let mut parser = create_test_parser(source);
            let result = parser.type_parser.parse_type();
            assert!(result.is_ok(), "Failed to parse type: {}", source);
        }
    }

    #[test]
    fn test_compound_types() {
        let sources = vec![
            "Vec<i32>",
            "(i32, string)",
            "Option<User>",
            "i32[]",
        ];
        
        for source in sources {
            let mut parser = create_test_parser(source);
            let result = parser.type_parser.parse_type();
            assert!(result.is_ok(), "Failed to parse type: {}", source);
        }
    }

    #[test]
    fn test_function_types() {
        let source = "fn(i32, string) -> bool";
        let mut parser = create_test_parser(source);
        let result = parser.type_parser.parse_type();
        assert!(result.is_ok());
    }

    #[test]
    fn test_constrained_types() {
        let source = "i32 where min_value = 0, max_value = 100";
        let mut parser = create_test_parser(source);
        let result = parser.type_parser.parse_type();
        assert!(result.is_ok());
    }

    #[test]
    fn test_semantic_constraints() {
        let source = "string where min_length = 3, pattern = \"[A-Z]+\"";
        let mut parser = create_test_parser(source);
        let result = parser.type_parser.parse_type();
        assert!(result.is_ok());
    }

    #[test]
    fn test_business_rules() {
        let source = "Money where currency = \"USD\", business_rule = \"Must be positive\"";
        let mut parser = create_test_parser(source);
        let result = parser.type_parser.parse_type();
        assert!(result.is_ok());
    }
} 