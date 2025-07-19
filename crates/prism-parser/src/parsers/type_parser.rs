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
use prism_ast::{AstNode, Type, TypeKind};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
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
        let mut base_type = self.parse_base_type()?;
        
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

    /// Parse a base type (primitive, named, or compound)
    fn parse_base_type(&mut self) -> ParseResult<NodeId> {
        let start_span = self.token_stream.current_span();
        
        match self.token_stream.current_kind() {
            // Primitive types
            TokenKind::Identifier(name) => {
                let type_name = name.clone();
                self.token_stream.advance();
                
                match type_name.as_str() {
                    // Numeric types
                    "i8" | "i16" | "i32" | "i64" | "i128" |
                    "u8" | "u16" | "u32" | "u64" | "u128" |
                    "f32" | "f64" => {
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_type_node(
                            TypeKind::Primitive(type_name),
                            span,
                        ))
                    }
                    
                    // String types
                    "string" | "str" => {
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_type_node(
                            TypeKind::Primitive(type_name),
                            span,
                        ))
                    }
                    
                    // Boolean type
                    "bool" => {
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_type_node(
                            TypeKind::Primitive(type_name),
                            span,
                        ))
                    }
                    
                    // Unit type
                    "void" | "unit" => {
                        let span = self.token_stream.previous_span();
                        Ok(self.coordinator.create_type_node(
                            TypeKind::Unit,
                            span,
                        ))
                    }
                    
                    // Named type (user-defined)
                    _ => {
                        // Check for generic parameters
                        if self.token_stream.check(TokenKind::Less) {
                            self.parse_generic_type(type_name, start_span)
                        } else {
                            let span = self.token_stream.previous_span();
                            Ok(self.coordinator.create_type_node(
                                TypeKind::Named(type_name),
                                span,
                            ))
                        }
                    }
                }
            }
            
            // Tuple type: (Type, Type, ...)
            TokenKind::LeftParen => {
                self.parse_tuple_type(start_span)
            }
            
            // Function type: (Type, Type) -> Type
            TokenKind::Fn => {
                self.parse_function_type(start_span)
            }
            
            _ => Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "type".to_string(),
            )),
        }
    }

    /// Parse a generic type: Name<Type, Type, ...>
    fn parse_generic_type(&mut self, base_name: String, start_span: Span) -> ParseResult<NodeId> {
        self.token_stream.expect(TokenKind::Less)?;
        
        let mut type_args = Vec::new();
        
        while !self.token_stream.check(TokenKind::Greater) && !self.token_stream.is_at_end() {
            type_args.push(self.parse_type()?);
            
            if !self.token_stream.check(TokenKind::Greater) {
                self.token_stream.expect(TokenKind::Comma)?;
            }
        }
        
        self.token_stream.expect(TokenKind::Greater)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_type_node(
            TypeKind::Generic {
                base: base_name,
                args: type_args,
            },
            span,
        ))
    }

    /// Parse a tuple type: (Type, Type, ...)
    fn parse_tuple_type(&mut self, start_span: Span) -> ParseResult<NodeId> {
        self.token_stream.expect(TokenKind::LeftParen)?;
        
        let mut element_types = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
            element_types.push(self.parse_type()?);
            
            if !self.token_stream.check(TokenKind::RightParen) {
                self.token_stream.expect(TokenKind::Comma)?;
            }
        }
        
        self.token_stream.expect(TokenKind::RightParen)?;
        
        let end_span = self.token_stream.previous_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_type_node(
            TypeKind::Tuple(element_types),
            span,
        ))
    }

    /// Parse a function type: fn(Type, Type) -> Type
    fn parse_function_type(&mut self, start_span: Span) -> ParseResult<NodeId> {
        self.token_stream.expect(TokenKind::Fn)?;
        
        // Parse parameter types
        self.token_stream.expect(TokenKind::LeftParen)?;
        let mut param_types = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
            param_types.push(self.parse_type()?);
            
            if !self.token_stream.check(TokenKind::RightParen) {
                self.token_stream.expect(TokenKind::Comma)?;
            }
        }
        
        self.token_stream.expect(TokenKind::RightParen)?;
        
        // Parse return type
        self.token_stream.expect(TokenKind::Arrow)?;
        let return_type = self.parse_type()?;
        
        let end_span = self.token_stream.current_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_type_node(
            TypeKind::Function {
                params: param_types,
                return_type: Box::new(return_type),
            },
            span,
        ))
    }

    /// Parse a constrained type with semantic constraints
    fn parse_constrained_type(&mut self, base_type: NodeId, start_span: Span) -> ParseResult<NodeId> {
        self.token_stream.expect(TokenKind::Where)?;
        
        let mut constraints = Vec::new();
        let mut business_rules = Vec::new();
        
        // Parse constraint clauses
        loop {
            if let Some(constraint) = self.parse_semantic_constraint()? {
                constraints.push(constraint);
            }
            
            if let Some(rule) = self.parse_business_rule()? {
                business_rules.push(rule);
            }
            
            if !self.token_stream.consume(TokenKind::Comma) {
                break;
            }
        }
        
        let end_span = self.token_stream.current_span();
        let span = Span::combine(start_span, end_span);
        
        Ok(self.coordinator.create_type_node(
            TypeKind::Constrained {
                base_type,
                constraints,
                business_rules,
            },
            span,
        ))
    }

    /// Parse a semantic constraint
    fn parse_semantic_constraint(&mut self) -> ParseResult<Option<SemanticConstraint>> {
        match self.token_stream.current_kind() {
            TokenKind::Identifier(name) => {
                let constraint_name = name.clone();
                self.token_stream.advance();
                
                match constraint_name.as_str() {
                    // Range constraints
                    "min_value" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let value = self.parse_constraint_value()?;
                        Ok(Some(SemanticConstraint::MinValue(value)))
                    }
                    
                    "max_value" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let value = self.parse_constraint_value()?;
                        Ok(Some(SemanticConstraint::MaxValue(value)))
                    }
                    
                    // Length constraints
                    "min_length" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let value = self.parse_constraint_value()?;
                        Ok(Some(SemanticConstraint::MinLength(value as usize)))
                    }
                    
                    "max_length" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let value = self.parse_constraint_value()?;
                        Ok(Some(SemanticConstraint::MaxLength(value as usize)))
                    }
                    
                    // Pattern constraints
                    "pattern" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let pattern = self.parse_string_literal()?;
                        Ok(Some(SemanticConstraint::Pattern(pattern)))
                    }
                    
                    "format" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let format = self.parse_string_literal()?;
                        Ok(Some(SemanticConstraint::Format(format)))
                    }
                    
                    // Precision constraints
                    "precision" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let value = self.parse_constraint_value()?;
                        Ok(Some(SemanticConstraint::Precision(value as u32)))
                    }
                    
                    // Currency constraints
                    "currency" => {
                        self.token_stream.expect(TokenKind::Assign)?;
                        let currency = self.parse_string_literal()?;
                        Ok(Some(SemanticConstraint::Currency(currency)))
                    }
                    
                    // Boolean constraints
                    "non_negative" => {
                        Ok(Some(SemanticConstraint::NonNegative))
                    }
                    
                    "immutable" => {
                        Ok(Some(SemanticConstraint::Immutable))
                    }
                    
                    "validated" => {
                        Ok(Some(SemanticConstraint::Validated))
                    }
                    
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    /// Parse a business rule
    fn parse_business_rule(&mut self) -> ParseResult<Option<BusinessRule>> {
        match self.token_stream.current_kind() {
            TokenKind::Identifier(name) => {
                let rule_name = name.clone();
                
                match rule_name.as_str() {
                    "business_rule" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::Assign)?;
                        let rule_text = self.parse_string_literal()?;
                        Ok(Some(BusinessRule::Custom(rule_text)))
                    }
                    
                    "security_classification" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::Assign)?;
                        let classification = self.parse_string_literal()?;
                        Ok(Some(BusinessRule::SecurityClassification(classification)))
                    }
                    
                    "compliance" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::Assign)?;
                        let standard = self.parse_string_literal()?;
                        Ok(Some(BusinessRule::Compliance(standard)))
                    }
                    
                    "ai_context" => {
                        self.token_stream.advance();
                        self.token_stream.expect(TokenKind::Assign)?;
                        let context = self.parse_string_literal()?;
                        Ok(Some(BusinessRule::AIContext(context)))
                    }
                    
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    /// Parse a constraint value (number)
    fn parse_constraint_value(&mut self) -> ParseResult<f64> {
        match self.token_stream.current_kind() {
            TokenKind::IntegerLiteral(value) => {
                let result = *value as f64;
                self.token_stream.advance();
                Ok(result)
            }
            TokenKind::FloatLiteral(value) => {
                let result = *value;
                self.token_stream.advance();
                Ok(result)
            }
            _ => Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "numeric value".to_string(),
            )),
        }
    }

    /// Parse a string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match self.token_stream.current_kind() {
            TokenKind::StringLiteral(value) => {
                let result = value.clone();
                self.token_stream.advance();
                Ok(result)
            }
            _ => Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "string literal".to_string(),
            )),
        }
    }

    /// Validate semantic constraints for a type
    pub fn validate_constraints(&self, type_node: NodeId, constraints: &[SemanticConstraint]) -> ParseResult<()> {
        // Use the constraint validator to check semantic validity
        for constraint in constraints {
            if let Err(error) = self.constraint_validator.validate_constraint(constraint) {
                return Err(ParseError::semantic_error(error));
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