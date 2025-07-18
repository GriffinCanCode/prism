//! Parser combinators for the Prism programming language
//!
//! This module provides reusable parser combinators for common patterns
//! in Prism syntax, making the parser more modular and maintainable.

use crate::{
    error::{ParseError, ParseErrorKind, ParseResult},
    Parser,
};
use prism_ast::{AstNode, Annotation, Parameter, TypeConstraint, Attribute};
use prism_lexer::{Token, TokenKind};
use std::collections::HashMap;

/// A combinator result that can be chained
pub type CombinatorResult<T> = ParseResult<T>;

/// Trait for parser combinators
pub trait Combinator<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<T>;
}

/// Combinator for parsing comma-separated lists
pub struct CommaSeparated<T> {
    item_parser: fn(&mut Parser) -> CombinatorResult<T>,
    allow_trailing_comma: bool,
    min_items: usize,
    max_items: Option<usize>,
}

impl<T> CommaSeparated<T> {
    pub fn new(item_parser: fn(&mut Parser) -> CombinatorResult<T>) -> Self {
        Self {
            item_parser,
            allow_trailing_comma: true,
            min_items: 0,
            max_items: None,
        }
    }
    
    pub fn min_items(mut self, min: usize) -> Self {
        self.min_items = min;
        self
    }
    
    pub fn max_items(mut self, max: usize) -> Self {
        self.max_items = Some(max);
        self
    }
    
    pub fn no_trailing_comma(mut self) -> Self {
        self.allow_trailing_comma = false;
        self
    }
}

impl<T> Combinator<Vec<T>> for CommaSeparated<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<Vec<T>> {
        let mut items = Vec::new();
        
        // Parse first item if present
        if let Ok(first_item) = (self.item_parser)(parser) {
            items.push(first_item);
            
            // Parse remaining items
            while parser.check(TokenKind::Comma) {
                parser.advance(); // consume comma
                
                // Check for trailing comma
                if self.allow_trailing_comma && parser.check_list_end() {
                    break;
                }
                
                // Check max items
                if let Some(max) = self.max_items {
                    if items.len() >= max {
                        return Err(ParseError::invalid_syntax(
                            "list".to_string(),
                            format!("Too many items (max: {})", max),
                            parser.current_span(),
                        ));
                    }
                }
                
                let item = (self.item_parser)(parser)?;
                items.push(item);
            }
        }
        
        // Check minimum items
        if items.len() < self.min_items {
            return Err(ParseError::invalid_syntax(
                "list".to_string(),
                format!("Not enough items (min: {}, found: {})", self.min_items, items.len()),
                parser.current_span(),
            ));
        }
        
        Ok(items)
    }
}

/// Combinator for parsing optional elements
pub struct Optional<T> {
    item_parser: fn(&mut Parser) -> CombinatorResult<T>,
}

impl<T> Optional<T> {
    pub fn new(item_parser: fn(&mut Parser) -> CombinatorResult<T>) -> Self {
        Self { item_parser }
    }
}

impl<T> Combinator<Option<T>> for Optional<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<Option<T>> {
        match (self.item_parser)(parser) {
            Ok(item) => Ok(Some(item)),
            Err(_) => Ok(None),
        }
    }
}

/// Combinator for parsing delimited lists
pub struct Delimited<T> {
    open: TokenKind,
    close: TokenKind,
    item_parser: fn(&mut Parser) -> CombinatorResult<T>,
    separator: Option<TokenKind>,
    allow_empty: bool,
}

impl<T> Delimited<T> {
    pub fn new(
        open: TokenKind,
        close: TokenKind,
        item_parser: fn(&mut Parser) -> CombinatorResult<T>,
    ) -> Self {
        Self {
            open,
            close,
            item_parser,
            separator: Some(TokenKind::Comma),
            allow_empty: true,
        }
    }
    
    pub fn with_separator(mut self, separator: TokenKind) -> Self {
        self.separator = Some(separator);
        self
    }
    
    pub fn no_separator(mut self) -> Self {
        self.separator = None;
        self
    }
    
    pub fn require_items(mut self) -> Self {
        self.allow_empty = false;
        self
    }
}

impl<T> Combinator<Vec<T>> for Delimited<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<Vec<T>> {
        parser.consume(self.open.clone(), &format!("Expected '{:?}'", self.open))?;
        
        let mut items = Vec::new();
        
        // Check for empty list
        if parser.check(self.close.clone()) {
            parser.advance(); // consume close
            if !self.allow_empty {
                return Err(ParseError::invalid_syntax(
                    "delimited_list".to_string(),
                    "Empty list not allowed".to_string(),
                    parser.current_span(),
                ));
            }
            return Ok(items);
        }
        
        // Parse items
        loop {
            let item = (self.item_parser)(parser)?;
            items.push(item);
            
            if parser.check(self.close.clone()) {
                parser.advance(); // consume close
                break;
            }
            
            if let Some(separator) = &self.separator {
                parser.consume(separator.clone(), &format!("Expected '{:?}' or '{:?}'", separator, self.close))?;
                
                // Check for trailing separator
                if parser.check(self.close.clone()) {
                    parser.advance(); // consume close
                    break;
                }
            }
        }
        
        Ok(items)
    }
}

/// Combinator for parsing sequences
pub struct Sequence<T> {
    parsers: Vec<fn(&mut Parser) -> CombinatorResult<T>>,
}

impl<T> Sequence<T> {
    pub fn new(parsers: Vec<fn(&mut Parser) -> CombinatorResult<T>>) -> Self {
        Self { parsers }
    }
}

impl<T> Combinator<Vec<T>> for Sequence<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<Vec<T>> {
        let mut results = Vec::new();
        
        for parser_fn in &self.parsers {
            let result = parser_fn(parser)?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Combinator for parsing alternatives
pub struct Alternative<T> {
    parsers: Vec<fn(&mut Parser) -> CombinatorResult<T>>,
}

impl<T> Alternative<T> {
    pub fn new(parsers: Vec<fn(&mut Parser) -> CombinatorResult<T>>) -> Self {
        Self { parsers }
    }
}

impl<T> Combinator<T> for Alternative<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<T> {
        let mut errors = Vec::new();
        
        for parser_fn in &self.parsers {
            match parser_fn(parser) {
                Ok(result) => return Ok(result),
                Err(error) => errors.push(error),
            }
        }
        
        // Return the first error if all alternatives fail
        Err(errors.into_iter().next().unwrap_or_else(|| {
            ParseError::invalid_syntax(
                "alternative".to_string(),
                "No alternatives matched".to_string(),
                parser.current_span(),
            )
        }))
    }
}

/// Combinator for parsing with lookahead
pub struct Lookahead<T> {
    predicate: fn(&Parser) -> bool,
    item_parser: fn(&mut Parser) -> CombinatorResult<T>,
}

impl<T> Lookahead<T> {
    pub fn new(
        predicate: fn(&Parser) -> bool,
        item_parser: fn(&mut Parser) -> CombinatorResult<T>,
    ) -> Self {
        Self {
            predicate,
            item_parser,
        }
    }
}

impl<T> Combinator<Option<T>> for Lookahead<T> {
    fn parse(&mut self, parser: &mut Parser) -> CombinatorResult<Option<T>> {
        if (self.predicate)(parser) {
            Ok(Some((self.item_parser)(parser)?))
        } else {
            Ok(None)
        }
    }
}

// Specific combinators for Prism constructs

impl Parser {
    /// Parse type parameters: `<T, U, V>`
    pub fn parse_type_parameters(&mut self) -> CombinatorResult<Vec<String>> {
        if !self.check(TokenKind::Less) {
            return Ok(Vec::new());
        }
        
        let mut delimited = Delimited::new(
            TokenKind::Less,
            TokenKind::Greater,
            |parser| {
                let name = parser.consume_identifier("Expected type parameter name")?;
                Ok(name)
            },
        );
        
        delimited.parse(self)
    }
    
    /// Parse function parameters: `(name: Type, name2: Type)`
    pub fn parse_parameters(&mut self) -> CombinatorResult<Vec<Parameter>> {
        let mut delimited = Delimited::new(
            TokenKind::LeftParen,
            TokenKind::RightParen,
            |parser| parser.parse_parameter(),
        );
        
        delimited.parse(self)
    }
    
    /// Parse a single parameter: `name: Type`
    pub fn parse_parameter(&mut self) -> CombinatorResult<Parameter> {
        let name = self.consume_identifier("Expected parameter name")?;
        self.consume(TokenKind::Colon, "Expected ':' after parameter name")?;
        let param_type = self.parse_type()?;
        
        Ok(Parameter {
            name,
            param_type,
            default_value: None,
            attributes: Vec::new(),
        })
    }
    
    /// Parse annotations: `@annotation value`
    pub fn parse_annotations(&mut self) -> CombinatorResult<Vec<Annotation>> {
        let mut annotations = Vec::new();
        
        while self.check(TokenKind::At) {
            let annotation = self.parse_annotation()?;
            annotations.push(annotation);
        }
        
        Ok(annotations)
    }
    
    /// Parse a single annotation: `@name value`
    pub fn parse_annotation(&mut self) -> CombinatorResult<Annotation> {
        self.consume(TokenKind::At, "Expected '@'")?;
        let name = self.consume_identifier("Expected annotation name")?;
        
        let value = if self.check(TokenKind::StringLiteral("".to_string())) {
            Some(self.consume_string("Expected annotation value")?)
        } else {
            None
        };
        
        Ok(Annotation {
            name,
            value,
            arguments: Vec::new(),
        })
    }
    
    /// Parse attributes: `[attribute1, attribute2]`
    pub fn parse_attributes(&mut self) -> CombinatorResult<Vec<Attribute>> {
        if !self.check(TokenKind::LeftBracket) {
            return Ok(Vec::new());
        }
        
        let mut delimited = Delimited::new(
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            |parser| parser.parse_attribute(),
        );
        
        delimited.parse(self)
    }
    
    /// Parse a single attribute
    pub fn parse_attribute(&mut self) -> CombinatorResult<Attribute> {
        let name = self.consume_identifier("Expected attribute name")?;
        
        let arguments = if self.check(TokenKind::LeftParen) {
            let mut delimited = Delimited::new(
                TokenKind::LeftParen,
                TokenKind::RightParen,
                |parser| parser.parse_expression(),
            );
            delimited.parse(self)?
        } else {
            Vec::new()
        };
        
        Ok(Attribute {
            name,
            arguments,
        })
    }
    
    /// Parse type constraints: `{ min: 0, max: 100 }`
    pub fn parse_type_constraints(&mut self) -> CombinatorResult<Vec<TypeConstraint>> {
        if !self.check(TokenKind::LeftBrace) {
            return Ok(Vec::new());
        }
        
        let mut delimited = Delimited::new(
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            |parser| parser.parse_type_constraint(),
        );
        
        delimited.parse(self)
    }
    
    /// Parse a single type constraint
    pub fn parse_type_constraint(&mut self) -> CombinatorResult<TypeConstraint> {
        let name = self.consume_identifier("Expected constraint name")?;
        self.consume(TokenKind::Colon, "Expected ':' after constraint name")?;
        
        // For now, just parse a simple value
        let value = if self.check(TokenKind::StringLiteral("".to_string())) {
            self.consume_string("Expected constraint value")?
        } else if let Ok(num) = self.consume_number("Expected constraint value") {
            num.to_string()
        } else {
            return Err(ParseError::invalid_syntax(
                "constraint".to_string(),
                "Expected constraint value".to_string(),
                self.current_span(),
            ));
        };
        
        Ok(TypeConstraint::Custom(prism_ast::CustomConstraint {
            name: name.clone(),
            expression: self.create_node(
                prism_ast::Expr::Literal(prism_ast::LiteralExpr {
                    value: prism_ast::LiteralValue::String(value),
                }),
                self.current_span(),
            ),
        }))
    }
    
    /// Parse argument lists: `(arg1, arg2, arg3)`
    pub fn parse_arguments(&mut self) -> CombinatorResult<Vec<AstNode<prism_ast::Expr>>> {
        let mut delimited = Delimited::new(
            TokenKind::LeftParen,
            TokenKind::RightParen,
            |parser| parser.parse_expression(),
        );
        
        delimited.parse(self)
    }
    
    /// Parse generic arguments: `<Type1, Type2>`
    pub fn parse_generic_arguments(&mut self) -> CombinatorResult<Vec<AstNode<prism_ast::Type>>> {
        if !self.check(TokenKind::Less) {
            return Ok(Vec::new());
        }
        
        let mut delimited = Delimited::new(
            TokenKind::Less,
            TokenKind::Greater,
            |parser| parser.parse_type(),
        );
        
        delimited.parse(self)
    }
    
    /// Parse a list of statements separated by semicolons or newlines
    pub fn parse_statement_list(&mut self) -> CombinatorResult<Vec<AstNode<prism_ast::Stmt>>> {
        let mut statements = Vec::new();
        
        while !self.is_at_end() && !self.check_block_end() {
            let stmt = self.parse_statement()?;
            statements.push(stmt);
            
            // Optional semicolon
            if self.check(TokenKind::Semicolon) {
                self.advance();
            }
        }
        
        Ok(statements)
    }
    
    // Helper methods
    
    fn consume_identifier(&mut self, message: &str) -> CombinatorResult<String> {
        if let TokenKind::Identifier(name) = &self.peek().kind {
            self.advance();
            Ok(name.to_string())
        } else {
            self.error(
                vec![TokenKind::Identifier(prism_common::symbol::Symbol::intern(""))],
                self.peek().kind.clone(),
                self.current_span(),
            )
        }
    }

    /// Parse a string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        if let TokenKind::StringLiteral(value) = &self.peek().kind {
            let value = value.clone();
            self.advance();
            Ok(value)
        } else {
            self.error(
                vec![TokenKind::StringLiteral("".to_string())],
                self.peek().kind.clone(),
                self.current_span(),
            )
        }
    }

    /// Parse a literal value
    fn parse_literal_value(&mut self) -> ParseResult<f64> {
        match &self.peek().kind {
            TokenKind::IntegerLiteral(value) => {
                let value = *value as f64;
                self.advance();
                Ok(value)
            }
            TokenKind::FloatLiteral(value) => {
                let value = *value;
                self.advance();
                Ok(value)
            }
            _ => self.error(
                vec![TokenKind::IntegerLiteral(0), TokenKind::FloatLiteral(0.0)],
                self.peek().kind.clone(),
                self.current_span(),
            ),
        }
    }
    
    fn check_list_end(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::RightParen
                | TokenKind::RightBracket
                | TokenKind::RightBrace
                | TokenKind::Greater
                | TokenKind::Eof
        )
    }
    
    fn check_block_end(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::RightBrace | TokenKind::Eof
        )
    }

    /// Create an error result
    fn error<T>(&self, expected: Vec<TokenKind>, found: TokenKind, span: Span) -> ParseResult<T> {
        Err(ParseError::unexpected_token(expected, found, span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parser;
    use prism_lexer::{SemanticLexer, Lexer};
    use prism_common::SourceId;
    
    #[test]
    fn test_parse_type_parameters() {
        let source = "<T, U, V>";
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let params = parser.parse_type_parameters().unwrap();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0], "T");
        assert_eq!(params[1], "U");
        assert_eq!(params[2], "V");
    }
    
    #[test]
    fn test_parse_annotations() {
        let source = r#"@author "John Doe" @version"#;
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let annotations = parser.parse_annotations().unwrap();
        assert_eq!(annotations.len(), 2);
        assert_eq!(annotations[0].name, "author");
        assert_eq!(annotations[0].value, Some("John Doe".to_string()));
        assert_eq!(annotations[1].name, "version");
        assert_eq!(annotations[1].value, None);
    }
    
    #[test]
    fn test_comma_separated_combinator() {
        let source = "a, b, c";
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let mut combinator = CommaSeparated::new(|parser| {
            parser.consume_identifier("Expected identifier")
        });
        
        let result = combinator.parse(&mut parser).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "a");
        assert_eq!(result[1], "b");
        assert_eq!(result[2], "c");
    }
    
    #[test]
    fn test_delimited_combinator() {
        let source = "(a, b, c)";
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let mut combinator = Delimited::new(
            TokenKind::LeftParen,
            TokenKind::RightParen,
            |parser| parser.consume_identifier("Expected identifier"),
        );
        
        let result = combinator.parse(&mut parser).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "a");
        assert_eq!(result[1], "b");
        assert_eq!(result[2], "c");
    }
    
    #[test]
    fn test_optional_combinator() {
        let source = "test";
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let mut combinator = Optional::new(|parser| {
            parser.consume_identifier("Expected identifier")
        });
        
        let result = combinator.parse(&mut parser).unwrap();
        assert_eq!(result, Some("test".to_string()));
    }
    
    #[test]
    fn test_optional_combinator_none() {
        let source = "123";
        let mut lexer = SemanticLexer::new(source, SourceId::new(0));
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        
        let mut combinator = Optional::new(|parser| {
            parser.consume_string("Expected string")
        });
        
        let result = combinator.parse(&mut parser).unwrap();
        assert_eq!(result, None);
    }
} 