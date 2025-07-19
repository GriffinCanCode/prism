//! Error recovery for the Prism parser
//!
//! This module provides sophisticated error recovery mechanisms that maintain
//! AST structure while recovering from syntax errors, with semantic awareness
//! for better error reporting and recovery strategies.

use crate::{
    core::error::{ErrorContext, ParseError, ParseErrorKind, ParseResult, ErrorSeverity},
    parser::Parser,
};
use prism_ast::{AstNode, Expr, Item, Stmt, Type, ErrorStmt, ErrorExpr, ErrorType};
use prism_lexer::{Token, TokenKind};
use std::collections::HashSet;

/// Recovery strategies for different error types
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Skip to next statement boundary
    SkipToStatement,
    /// Skip to next expression boundary
    SkipToExpression,
    /// Insert missing delimiter
    InsertDelimiter(TokenKind),
    /// Replace with error node
    ErrorNode,
    /// Continue with assumption
    ContinueWithAssumption(TokenKind),
}

/// Context for error recovery decisions
#[derive(Debug, Clone)]
pub struct RecoveryContext {
    /// Current parsing context
    pub context: ParsingContext,
    /// Expected tokens at error point
    pub expected: Vec<TokenKind>,
    /// Found token at error point
    pub found: TokenKind,
    /// Number of previous errors
    pub error_count: usize,
    /// Available recovery strategies
    pub available_strategies: Vec<RecoveryStrategy>,
}

/// Current parsing context for recovery decisions
#[derive(Debug, Clone, PartialEq)]
pub enum ParsingContext {
    TopLevel,
    Module,
    Section,
    Function,
    Type,
    Expression,
    Statement,
    Parameter,
    Argument,
}

impl Parser {
    /// Attempt to recover from a parse error
    pub fn recover_from_error(&mut self, error: ParseError) -> ParseResult<()> {
        let context = self.build_recovery_context(&error);
        let strategy = self.select_recovery_strategy(&context);
        
        match strategy {
            RecoveryStrategy::SkipToStatement => {
                self.skip_to_statement_boundary();
                Ok(())
            }
            RecoveryStrategy::SkipToExpression => {
                self.skip_to_expression_boundary();
                Ok(())
            }
            RecoveryStrategy::InsertDelimiter(delimiter) => {
                self.insert_virtual_delimiter(delimiter);
                Ok(())
            }
            RecoveryStrategy::ErrorNode => {
                // Error node will be created by caller
                Ok(())
            }
            RecoveryStrategy::ContinueWithAssumption(assumed_token) => {
                self.assume_token(assumed_token);
                Ok(())
            }
        }
    }
    
    /// Build recovery context from error
    fn build_recovery_context(&self, error: &ParseError) -> RecoveryContext {
        let context = self.current_parsing_context();
        let (expected, found) = match &error.kind {
            ParseErrorKind::UnexpectedToken { expected, found } => {
                (expected.clone(), found.clone())
            }
            ParseErrorKind::UnexpectedEof { expected } => {
                (expected.clone(), TokenKind::Eof)
            }
            _ => (Vec::new(), TokenKind::Error),
        };
        
        let available_strategies = self.get_available_strategies(&context, &expected, &found);
        
        RecoveryContext {
            context,
            expected,
            found,
            error_count: self.get_errors().len(),
            available_strategies,
        }
    }
    
    /// Determine current parsing context
    fn current_parsing_context(&self) -> ParsingContext {
        // This is a simplified implementation
        // In practice, we'd maintain a context stack
        if self.is_in_function() {
            ParsingContext::Function
        } else if self.is_in_type() {
            ParsingContext::Type
        } else if self.is_in_expression() {
            ParsingContext::Expression
        } else if self.is_in_module() {
            ParsingContext::Module
        } else {
            ParsingContext::TopLevel
        }
    }
    
    /// Get available recovery strategies for context
    fn get_available_strategies(
        &self,
        context: &ParsingContext,
        expected: &[TokenKind],
        found: &TokenKind,
    ) -> Vec<RecoveryStrategy> {
        let mut strategies = Vec::new();
        
        match context {
            ParsingContext::TopLevel | ParsingContext::Module => {
                strategies.push(RecoveryStrategy::SkipToStatement);
                if self.can_insert_delimiter(expected) {
                    strategies.push(RecoveryStrategy::InsertDelimiter(TokenKind::Semicolon));
                }
            }
            ParsingContext::Function => {
                strategies.push(RecoveryStrategy::SkipToStatement);
                if expected.contains(&TokenKind::LeftBrace) {
                    strategies.push(RecoveryStrategy::InsertDelimiter(TokenKind::LeftBrace));
                }
                if expected.contains(&TokenKind::RightBrace) {
                    strategies.push(RecoveryStrategy::InsertDelimiter(TokenKind::RightBrace));
                }
            }
            ParsingContext::Expression => {
                strategies.push(RecoveryStrategy::SkipToExpression);
                strategies.push(RecoveryStrategy::ErrorNode);
                if expected.contains(&TokenKind::RightParen) {
                    strategies.push(RecoveryStrategy::InsertDelimiter(TokenKind::RightParen));
                }
            }
            ParsingContext::Type => {
                strategies.push(RecoveryStrategy::ErrorNode);
                if self.can_assume_identifier(expected, found) {
                    strategies.push(RecoveryStrategy::ContinueWithAssumption(
                        TokenKind::Identifier(prism_common::symbol::Symbol::intern("placeholder"))
                    ));
                }
            }
            _ => {
                strategies.push(RecoveryStrategy::SkipToStatement);
                strategies.push(RecoveryStrategy::ErrorNode);
            }
        }
        
        strategies
    }
    
    /// Select the best recovery strategy
    fn select_recovery_strategy(&self, context: &RecoveryContext) -> RecoveryStrategy {
        // Priority-based selection
        for strategy in &context.available_strategies {
            if self.is_strategy_viable(strategy, context) {
                return strategy.clone();
            }
        }
        
        // Fallback to error node
        RecoveryStrategy::ErrorNode
    }
    
    /// Check if a recovery strategy is viable
    fn is_strategy_viable(&self, strategy: &RecoveryStrategy, context: &RecoveryContext) -> bool {
        match strategy {
            RecoveryStrategy::SkipToStatement => {
                self.can_find_statement_boundary()
            }
            RecoveryStrategy::SkipToExpression => {
                self.can_find_expression_boundary()
            }
            RecoveryStrategy::InsertDelimiter(delimiter) => {
                self.is_delimiter_insertion_safe(delimiter)
            }
            RecoveryStrategy::ErrorNode => {
                true // Always viable
            }
            RecoveryStrategy::ContinueWithAssumption(token) => {
                self.is_assumption_safe(token, context)
            }
        }
    }
    
    /// Skip to the next statement boundary
    fn skip_to_statement_boundary(&mut self) {
        let mut depth = 0;
        
        while !self.is_at_end() {
            match self.peek().kind {
                TokenKind::LeftBrace => depth += 1,
                TokenKind::RightBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                TokenKind::Semicolon if depth == 0 => {
                    self.advance();
                    break;
                }
                // Statement-starting keywords
                TokenKind::Module
                | TokenKind::Function
                | TokenKind::Type
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return if depth == 0 => {
                    break;
                }
                _ => {}
            }
            
            self.advance();
        }
    }
    
    /// Skip to the next expression boundary
    fn skip_to_expression_boundary(&mut self) {
        let mut paren_depth = 0;
        let mut bracket_depth = 0;
        
        while !self.is_at_end() {
            match self.peek().kind {
                TokenKind::LeftParen => paren_depth += 1,
                TokenKind::RightParen => {
                    if paren_depth == 0 {
                        break;
                    }
                    paren_depth -= 1;
                }
                TokenKind::LeftBracket => bracket_depth += 1,
                TokenKind::RightBracket => {
                    if bracket_depth == 0 {
                        break;
                    }
                    bracket_depth -= 1;
                }
                TokenKind::Comma | TokenKind::Semicolon 
                    if paren_depth == 0 && bracket_depth == 0 => {
                    break;
                }
                _ => {}
            }
            
            self.advance();
        }
    }
    
    /// Insert a virtual delimiter for recovery
    fn insert_virtual_delimiter(&mut self, delimiter: TokenKind) {
        // Create a virtual token for the missing delimiter
        let span = self.current_span();
        let virtual_token = Token {
            kind: delimiter,
            span,
            semantic_context: None,
        };
        
        // Insert into token stream
        let current_pos = self.get_current_position();
        self.insert_token(current_pos, virtual_token);
    }
    
    /// Assume a token exists for recovery
    fn assume_token(&mut self, assumed_token: TokenKind) {
        // Similar to insert_virtual_delimiter but for assumptions
        let span = self.current_span();
        let virtual_token = Token {
            kind: assumed_token,
            span,
            semantic_context: None,
        };
        
        let current_pos = self.get_current_position();
        self.insert_token(current_pos, virtual_token);
    }
    
    /// Create an error statement for recovery
    pub fn create_error_statement(&mut self, error: &ParseError) -> AstNode<Stmt> {
        let span = error.span;
        let error_stmt = ErrorStmt {
            message: error.message.clone(),
            error_kind: error.kind.clone(),
            recovery_suggestions: error.suggestions.clone(),
            context: error.context.clone(),
        };
        
        self.create_error_node(Stmt::Error(error_stmt), span)
    }
    
    /// Create an error expression for recovery
    pub fn create_error_expression(&mut self, error: &ParseError) -> AstNode<Expr> {
        let span = error.span;
        let error_expr = ErrorExpr {
            message: error.message.clone(),
            error_kind: error.kind.clone(),
            recovery_suggestions: error.suggestions.clone(),
            context: error.context.clone(),
        };
        
        self.create_error_node(Expr::Error(error_expr), span)
    }
    
    /// Create an error type for recovery
    pub fn create_error_type(&mut self, error: &ParseError) -> AstNode<Type> {
        let span = error.span;
        let error_type = ErrorType {
            message: error.message.clone(),
            error_kind: error.kind.clone(),
            recovery_suggestions: error.suggestions.clone(),
            context: error.context.clone(),
        };
        
        self.create_error_node(Type::Error(error_type), span)
    }
    
    /// Create an error item for recovery
    pub fn create_error_item(&mut self, error: &ParseError) -> AstNode<Item> {
        let span = error.span;
        let error_stmt = ErrorStmt {
            message: error.message.clone(),
            error_kind: error.kind.clone(),
            recovery_suggestions: error.suggestions.clone(),
            context: error.context.clone(),
        };
        
        self.create_error_node(Item::Statement(Stmt::Error(error_stmt)), span)
    }
    
    // Helper methods for context detection
    
    fn is_in_function(&self) -> bool {
        // Check if we're currently parsing inside a function
        // This would be tracked by a context stack in practice
        false
    }
    
    fn is_in_type(&self) -> bool {
        // Check if we're currently parsing inside a type
        false
    }
    
    fn is_in_expression(&self) -> bool {
        // Check if we're currently parsing inside an expression
        false
    }
    
    fn is_in_module(&self) -> bool {
        // Check if we're currently parsing inside a module
        false
    }
    
    // Helper methods for recovery strategy validation
    
    fn can_find_statement_boundary(&self) -> bool {
        // Look ahead to see if we can find a statement boundary
        let mut pos = self.get_current_position();
        let mut depth = 0;
        let tokens = self.get_tokens();
        
        while pos < tokens.len() {
            match tokens[pos].kind {
                TokenKind::LeftBrace => depth += 1,
                TokenKind::RightBrace => depth -= 1,
                TokenKind::Semicolon if depth == 0 => return true,
                TokenKind::Module
                | TokenKind::Function
                | TokenKind::Type
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var if depth == 0 => return true,
                TokenKind::Eof => return false,
                _ => {}
            }
            pos += 1;
        }
        
        false
    }
    
    fn can_find_expression_boundary(&self) -> bool {
        // Look ahead to see if we can find an expression boundary
        let mut pos = self.get_current_position();
        let mut paren_depth = 0;
        let tokens = self.get_tokens();
        
        while pos < tokens.len() {
            match tokens[pos].kind {
                TokenKind::LeftParen => paren_depth += 1,
                TokenKind::RightParen => {
                    if paren_depth == 0 {
                        return true;
                    }
                    paren_depth -= 1;
                }
                TokenKind::Comma | TokenKind::Semicolon if paren_depth == 0 => return true,
                TokenKind::Eof => return false,
                _ => {}
            }
            pos += 1;
        }
        
        false
    }
    
    fn can_insert_delimiter(&self, expected: &[TokenKind]) -> bool {
        // Check if inserting a delimiter would be safe
        expected.iter().any(|token| matches!(
            token,
            TokenKind::LeftBrace
            | TokenKind::RightBrace
            | TokenKind::LeftParen
            | TokenKind::RightParen
            | TokenKind::LeftBracket
            | TokenKind::RightBracket
            | TokenKind::Semicolon
            | TokenKind::Comma
        ))
    }
    
    fn is_delimiter_insertion_safe(&self, delimiter: &TokenKind) -> bool {
        // Check if inserting this delimiter won't cause more problems
        match delimiter {
            TokenKind::RightBrace => {
                // Safe if we're in a block
                self.is_in_block()
            }
            TokenKind::RightParen => {
                // Safe if we're in parentheses
                self.is_in_parentheses()
            }
            TokenKind::Semicolon => {
                // Usually safe
                true
            }
            _ => false,
        }
    }
    
    fn can_assume_identifier(&self, expected: &[TokenKind], found: &TokenKind) -> bool {
        // Check if we can safely assume an identifier
        expected.iter().any(|token| matches!(token, TokenKind::Identifier(_)))
            && !matches!(found, TokenKind::Eof)
    }
    
    fn is_assumption_safe(&self, token: &TokenKind, context: &RecoveryContext) -> bool {
        // Check if assuming this token is safe
        match token {
            TokenKind::Identifier(_) => {
                // Safe in most contexts
                true
            }
            _ => false,
        }
    }
    
    fn is_in_block(&self) -> bool {
        // Check if we're currently inside a block
        // This would track brace depth in practice
        false
    }
    
    fn is_in_parentheses(&self) -> bool {
        // Check if we're currently inside parentheses
        // This would track paren depth in practice
        false
    }
    
    /// Generate helpful error suggestions
    pub fn generate_error_suggestions(&self, error: &ParseError) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        match &error.kind {
            ParseErrorKind::UnexpectedToken { expected, found } => {
                if expected.len() == 1 {
                    suggestions.push(format!("Try using '{:?}' instead of '{:?}'", expected[0], found));
                } else {
                    suggestions.push(format!("Expected one of: {:?}", expected));
                }
                
                // Context-specific suggestions
                match found {
                    TokenKind::Identifier(name) if expected.contains(&TokenKind::Function) => {
                        suggestions.push(format!("Did you mean to declare a function? Try 'function {}'", name));
                    }
                    TokenKind::Identifier(name) if expected.contains(&TokenKind::Type) => {
                        suggestions.push(format!("Did you mean to declare a type? Try 'type {}'", name));
                    }
                    _ => {}
                }
            }
            ParseErrorKind::UnexpectedEof { expected } => {
                suggestions.push("Unexpected end of file".to_string());
                if expected.contains(&TokenKind::RightBrace) {
                    suggestions.push("Missing closing brace '}'".to_string());
                }
                if expected.contains(&TokenKind::RightParen) {
                    suggestions.push("Missing closing parenthesis ')'".to_string());
                }
            }
            ParseErrorKind::InvalidSyntax { construct, .. } => {
                suggestions.push(format!("Invalid {} syntax", construct));
                match construct.as_str() {
                    "function" => {
                        suggestions.push("Function syntax: function name(params) -> ReturnType { body }".to_string());
                    }
                    "type" => {
                        suggestions.push("Type syntax: type Name = TypeDefinition".to_string());
                    }
                    "module" => {
                        suggestions.push("Module syntax: module Name { sections }".to_string());
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        
        suggestions
    }
    
    /// Enhanced synchronization with semantic awareness
    pub fn synchronize_with_semantic_awareness(&mut self) {
        self.set_recovery_mode(true);
        
        // Skip the problematic token
        if !self.is_at_end() {
            self.advance();
        }
        
        while !self.is_at_end() {
            // Check for statement boundaries
            if self.previous().kind == TokenKind::Semicolon {
                self.set_recovery_mode(false);
                return;
            }
            
            // Check for block boundaries
            match self.peek().kind {
                TokenKind::RightBrace => {
                    // Don't consume the closing brace
                    self.set_recovery_mode(false);
                    return;
                }
                
                // Statement-starting keywords
                TokenKind::Module
                | TokenKind::Section
                | TokenKind::Function
                | TokenKind::Fn
                | TokenKind::Type
                | TokenKind::Let
                | TokenKind::Const
                | TokenKind::Var
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Match
                | TokenKind::Return
                | TokenKind::Break
                | TokenKind::Continue => {
                    self.set_recovery_mode(false);
                    return;
                }
                
                _ => {}
            }
            
            self.advance();
        }
        
        self.set_recovery_mode(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;
    use prism_lexer::{SemanticLexer, Lexer};
    use prism_common::SourceId;
    
    #[test]
    fn test_error_recovery_missing_semicolon() {
        let source = "let x = 5\nlet y = 10;";
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        // This should recover from the missing semicolon
        let result = parser.parse_program();
        assert!(result.is_err()); // Should have errors but still parse
        
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("Expected"));
    }
    
    #[test]
    fn test_error_recovery_missing_brace() {
        let source = r#"
            module Test {
                function test() {
                    return 42;
                // Missing closing brace
            }
        "#;
        
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_program();
        assert!(result.is_err());
        
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
    }
    
    #[test]
    fn test_error_suggestions() {
        let source = "functio test() {}"; // Typo in "function"
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let result = parser.parse_program();
        assert!(result.is_err());
        
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
        
        let suggestions = parser.generate_error_suggestions(&errors[0]);
        assert!(!suggestions.is_empty());
    }
} 