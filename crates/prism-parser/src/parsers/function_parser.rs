//! Function Parsing
//!
//! This module embodies the single concept of "Function Definition Parsing".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: parsing function definitions with all their components.
//!
//! **Conceptual Responsibility**: Parse function definitions with signatures and bodies
//! **What it does**: function keywords, parameters, return types, constraints, effects, bodies
//! **What it doesn't do**: expression parsing, type parsing, statement parsing, token navigation

use crate::{
    core::{error::{ParseError, ParseResult}, token_stream_manager::TokenStreamManager, parsing_coordinator::ParsingCoordinator},
    parsers::{expression_parser::ExpressionParser, type_parser::TypeParser, statement_parser::StatementParser},
};
use prism_ast::{AstNode, FunctionDecl, Contracts, Visibility, Attribute, AttributeArgument, Parameter};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;
use std::collections::HashMap;

/// Function parser - handles function definitions with all their components
/// 
/// This struct embodies the single concept of parsing function definitions.
/// It coordinates with other parsers for sub-components (types, expressions, statements)
/// but owns the logic for function structure and semantics.
pub struct FunctionParser<'a> {
    /// Reference to the token stream manager (no ownership)
    token_stream: &'a mut TokenStreamManager,
    /// Reference to expression parser for default values and constraints
    expr_parser: &'a mut ExpressionParser<'a>,
    /// Reference to type parser for parameter and return types
    type_parser: &'a mut TypeParser<'a>,
    /// Reference to statement parser for function bodies
    stmt_parser: &'a mut StatementParser<'a>,
    /// Reference to coordinator for node creation and error handling
    coordinator: &'a mut ParsingCoordinator,
}

impl<'a> FunctionParser<'a> {
    /// Create a new function parser
    pub fn new(
        token_stream: &'a mut TokenStreamManager,
        expr_parser: &'a mut ExpressionParser<'a>,
        type_parser: &'a mut TypeParser<'a>,
        stmt_parser: &'a mut StatementParser<'a>,
        coordinator: &'a mut ParsingCoordinator,
    ) -> Self {
        Self {
            token_stream,
            expr_parser,
            type_parser,
            stmt_parser,
            coordinator,
        }
    }

    /// Parse a function definition
    pub fn parse_function(&mut self) -> ParseResult<AstNode<prism_ast::Stmt>> {
        let start_span = self.token_stream.current_span();
        
        // Parse optional visibility modifier
        let visibility = self.parse_visibility()?;
        
        // Parse optional attributes
        let attributes = self.parse_attributes()?;
        
        // Parse optional async modifier
        let is_async = self.token_stream.check(TokenKind::Async);
        if is_async {
            self.token_stream.advance(); // consume 'async'
        }
        
        // Parse function keyword (either 'function' or 'fn')
        let is_canonical_style = match self.token_stream.current_kind() {
            TokenKind::Function => {
                self.token_stream.advance();
                true
            }
            TokenKind::Fn => {
                self.token_stream.advance();
                false
            }
            _ => return Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "function or fn".to_string(),
            )),
        };
        
        // Parse function name
        let name = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Parse optional generic parameters
        let generic_params = self.parse_generic_parameters()?;
        
        // Parse parameters
        let parameters = self.parse_parameters()?;
        
        // Parse optional return type
        let return_type = if self.token_stream.consume(TokenKind::Arrow) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Parse optional constraints (requires/ensures)
        let contracts = self.parse_contracts()?;
        
        // Parse function body
        let body = if self.token_stream.check(TokenKind::LeftBrace) {
            Some(Box::new(self.stmt_parser.parse_block_statement()?))
        } else {
            // Abstract function (interface/trait)
            self.token_stream.expect(TokenKind::Semicolon)?;
            None
        };
        
        let end_span = self.token_stream.current_span();
        let span = Span::combine(start_span, end_span);
        
        let function_stmt = prism_ast::Stmt::Function(FunctionDecl {
            name,
            parameters,
            return_type,
            body,
            visibility,
            attributes,
            contracts,
            is_async,
        });
        
        let node = self.coordinator.create_node(function_stmt, span);
        Ok(node)
    }
    
    /// Parse generic parameters: `<T, U: Trait, V>`
    fn parse_generic_parameters(&mut self) -> ParseResult<Vec<String>> {
        if !self.token_stream.consume(TokenKind::Less) {
            return Ok(Vec::new());
        }
        
        let mut params = Vec::new();
        
        while !self.token_stream.check(TokenKind::Greater) && !self.token_stream.is_at_end() {
            let param_name = self.token_stream.expect_identifier()?;
            
            // TODO: Parse generic constraints (T: Trait)
            // For now, just collect the parameter name
            params.push(param_name);
            
            if !self.token_stream.check(TokenKind::Greater) {
                self.token_stream.expect(TokenKind::Comma)?;
            }
        }
        
        self.token_stream.expect(TokenKind::Greater)?;
        Ok(params)
    }
    
    /// Parse function parameters: `(name: Type, name2: Type = default)`
    fn parse_parameters(&mut self) -> ParseResult<Vec<Parameter>> {
        self.token_stream.expect(TokenKind::LeftParen)?;
        
        let mut parameters = Vec::new();
        
        while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
            parameters.push(self.parse_parameter()?);
            
            if !self.token_stream.check(TokenKind::RightParen) {
                self.token_stream.expect(TokenKind::Comma)?;
            }
        }
        
        self.token_stream.expect(TokenKind::RightParen)?;
        Ok(parameters)
    }
    
    /// Parse a single parameter: `name: Type = default_value`
    fn parse_parameter(&mut self) -> ParseResult<Parameter> {
        let param_name = prism_common::symbol::Symbol::intern(&self.token_stream.expect_identifier()?);
        
        // Optional type annotation
        let type_annotation = if self.token_stream.consume(TokenKind::Colon) {
            Some(self.type_parser.parse_type()?)
        } else {
            None
        };
        
        // Optional default value
        let default_value = if self.token_stream.consume(TokenKind::Assign) {
            Some(ExpressionParser::parse_expression(self.coordinator, crate::core::precedence::Precedence::Assignment)?)
        } else {
            None
        };
        
        Ok(Parameter {
            name: param_name,
            type_annotation,
            default_value,
            is_mutable: false, // TODO: Parse mutability
        })
    }
    
    /// Parse contracts: `requires condition ensures condition`
    fn parse_contracts(&mut self) -> ParseResult<Option<Contracts>> {
        let mut requires = Vec::new();
        let mut ensures = Vec::new();
        
        // Parse requires clauses
        while self.token_stream.consume(TokenKind::Requires) {
            let condition = ExpressionParser::parse_expression(self.coordinator, crate::core::precedence::Precedence::Assignment)?;
            requires.push(condition);
        }
        
        // Parse ensures clauses
        while self.token_stream.consume(TokenKind::Ensures) {
            let condition = ExpressionParser::parse_expression(self.coordinator, crate::core::precedence::Precedence::Assignment)?;
            ensures.push(condition);
        }
        
        if requires.is_empty() && ensures.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Contracts {
                requires,
                ensures,
                invariants: Vec::new(), // TODO: Parse invariants
            }))
        }
    }
    

    
    /// Validate function definition semantics
    pub fn validate_function_semantics(&self, function_node: NodeId) -> ParseResult<()> {
        // Validate that function follows Prism conventions:
        // 1. Single responsibility principle (inferred from name)
        // 2. Proper parameter naming
        // 3. Return type consistency
        // 4. Effect declarations match body
        
        // Placeholder implementation
        Ok(())
    }
    
    /// Extract AI-comprehensible metadata from function
    pub fn extract_function_metadata(&self, function_node: NodeId) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Extract semantic meaning from function structure
        metadata.insert("concept".to_string(), "Function Definition".to_string());
        metadata.insert("purpose".to_string(), "Define reusable computation with clear contracts".to_string());
        metadata.insert("ai_hint".to_string(), "Functions should have single responsibility".to_string());
        metadata.insert("responsibility_principle".to_string(), "One clear purpose per function".to_string());
        
        metadata
    }
    
    /// Parse function attributes like @responsibility, @aiContext, etc.
    fn parse_attributes(&mut self) -> ParseResult<Vec<Attribute>> {
        let mut attributes = Vec::new();
        
        while self.token_stream.check(TokenKind::At) {
            self.token_stream.advance(); // consume '@'
            
            let attr_name = self.token_stream.expect_identifier()?;
            let mut arguments = Vec::new();
            
            // Check for attribute arguments
            if self.token_stream.check(TokenKind::LeftParen) {
                self.token_stream.advance(); // consume '('
                
                while !self.token_stream.check(TokenKind::RightParen) && !self.token_stream.is_at_end() {
                    let arg = self.parse_attribute_argument()?;
                    arguments.push(arg);
                    
                    if !self.token_stream.check(TokenKind::RightParen) {
                        self.token_stream.expect(TokenKind::Comma)?;
                    }
                }
                
                self.token_stream.expect(TokenKind::RightParen)?;
            }
            
            attributes.push(Attribute {
                name: attr_name,
                arguments,
            });
        }
        
        Ok(attributes)
    }
    
    /// Parse a single attribute argument
    fn parse_attribute_argument(&mut self) -> ParseResult<AttributeArgument> {
        // Check for named argument (name = value)
        if self.token_stream.check_identifier() {
            let lookahead_pos = self.token_stream.position();
            let name = self.token_stream.expect_identifier()?;
            
            if self.token_stream.check(TokenKind::Assign) {
                self.token_stream.advance(); // consume '='
                let value = self.parse_literal_value()?;
                return Ok(AttributeArgument::Named { name, value });
            } else {
                // Not a named argument, backtrack and parse as literal
                self.token_stream.set_position(lookahead_pos);
            }
        }
        
        // Parse as literal argument
        let value = self.parse_literal_value()?;
        Ok(AttributeArgument::Literal(value))
    }
    
    /// Parse a literal value for attributes
    fn parse_literal_value(&mut self) -> ParseResult<prism_ast::LiteralValue> {
        match self.token_stream.current_kind() {
            TokenKind::StringLiteral(value) => {
                let result = prism_ast::LiteralValue::String(value.clone());
                self.token_stream.advance();
                Ok(result)
            }
            TokenKind::IntegerLiteral(value) => {
                let result = prism_ast::LiteralValue::Integer(*value);
                self.token_stream.advance();
                Ok(result)
            }
            TokenKind::FloatLiteral(value) => {
                let result = prism_ast::LiteralValue::Float(*value);
                self.token_stream.advance();
                Ok(result)
            }
            TokenKind::True => {
                self.token_stream.advance();
                Ok(prism_ast::LiteralValue::Boolean(true))
            }
            TokenKind::False => {
                self.token_stream.advance();
                Ok(prism_ast::LiteralValue::Boolean(false))
            }
            TokenKind::Null => {
                self.token_stream.advance();
                Ok(prism_ast::LiteralValue::Null)
            }
            _ => Err(ParseError::unexpected_token(
                self.token_stream.current_token().clone(),
                "literal value".to_string(),
            )),
        }
    }
    
    /// Parse optional visibility modifier
    fn parse_visibility(&mut self) -> ParseResult<Visibility> {
        match self.token_stream.current_kind() {
            TokenKind::Public | TokenKind::Pub => {
                self.token_stream.advance();
                Ok(Visibility::Public)
            }
            TokenKind::Private => {
                self.token_stream.advance();
                Ok(Visibility::Private)
            }
            TokenKind::Internal => {
                self.token_stream.advance();
                Ok(Visibility::Internal)
            }
            _ => {
                // Default visibility is private for functions
                Ok(Visibility::Private)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test utilities would be defined here or in a test module

    #[test]
    fn test_simple_function() {
        let source = "function add(a: i32, b: i32) -> i32 { return a + b }";
        let mut parser = create_test_parser(source);
        let result = parser.function_parser.parse_function();
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_with_effects() {
        let source = r#"
            function processPayment(amount: Money<USD>) -> Result<Transaction, PaymentError>
                effects [Database.Write, Audit.Log]
            {
                // Implementation
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.function_parser.parse_function();
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_with_constraints() {
        let source = r#"
            function transfer(amount: Money<USD>) -> Result<(), TransferError>
                requires amount > 0.USD
                ensures balance_unchanged_on_error()
            {
                // Implementation
            }
        "#;
        
        let mut parser = create_test_parser(source);
        let result = parser.function_parser.parse_function();
        assert!(result.is_ok());
    }

    #[test]
    fn test_generic_function() {
        let source = "function map<T, U>(items: List<T>, f: fn(T) -> U) -> List<U> { /* impl */ }";
        let mut parser = create_test_parser(source);
        let result = parser.function_parser.parse_function();
        assert!(result.is_ok());
    }

    #[test]
    fn test_rust_style_function() {
        let source = "fn authenticate(user: User) -> Result<Session, AuthError> { /* impl */ }";
        let mut parser = create_test_parser(source);
        let result = parser.function_parser.parse_function();
        assert!(result.is_ok());
    }
} 