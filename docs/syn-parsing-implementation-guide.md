# Syn-Inspired Parsing Implementation Guide for Prism

## Overview

This guide provides practical steps to adopt syn's parsing patterns in the Prism parser while maintaining separation of concerns (SoC). The goal is to improve parsing maintainability and composability without over-engineering.

## Key Benefits of Syn's Approach

1. **Parse Trait System**: Type-driven parsing where each type knows how to parse itself
2. **Composable Parsers**: Small, focused parsers that combine naturally
3. **Better Error Handling**: More precise error locations and recovery
4. **Reduced Boilerplate**: Less repetitive parsing code

## Implementation Strategy

### Phase 1: Foundation (Week 1)

#### 1.1 Add Dependencies

Update `crates/prism-parser/Cargo.toml`:

```toml
[dependencies]
# Keep existing dependencies
prism-common = { workspace = true }
prism-ast = { workspace = true }
prism-lexer = { workspace = true }

# Add syn-inspired parsing support
syn = { version = "2.0", features = ["parsing", "proc-macro"] }
quote = { version = "1.0" }

# Enhanced error handling
thiserror = { workspace = true }
```

#### 1.2 Create Parse Trait Infrastructure

Create `crates/prism-parser/src/core/parse_trait.rs`:

```rust
//! Parse trait infrastructure inspired by syn
//!
//! This module provides the core parsing traits that enable type-driven parsing

use crate::core::{ParseError, ParseResult, TokenStreamManager};
use prism_lexer::{Token, TokenKind};
use prism_common::span::Span;

/// Input stream for parsing - similar to syn's ParseStream
pub struct ParseStream<'a> {
    tokens: &'a mut TokenStreamManager,
}

impl<'a> ParseStream<'a> {
    pub fn new(tokens: &'a mut TokenStreamManager) -> Self {
        Self { tokens }
    }

    /// Parse a type T from the stream
    pub fn parse<T: Parse>(&mut self) -> ParseResult<T> {
        T::parse(self)
    }

    /// Check if the next token matches the given kind
    pub fn peek<T: Peek>(&self, token: T) -> bool {
        token.peek(self.tokens.peek())
    }

    /// Consume and return the next token if it matches
    pub fn parse_token<T: ParseToken>(&mut self) -> ParseResult<T> {
        T::parse_token(self)
    }

    /// Get current span
    pub fn span(&self) -> Span {
        self.tokens.current_span()
    }

    /// Check if at end of stream
    pub fn is_empty(&self) -> bool {
        self.tokens.is_at_end()
    }

    /// Fork the stream for lookahead
    pub fn fork(&self) -> ParseStream<'_> {
        // Implementation would create a copy for lookahead
        todo!("Implement forking for lookahead")
    }
}

/// Core parsing trait - similar to syn's Parse
pub trait Parse: Sized {
    fn parse(input: &mut ParseStream) -> ParseResult<Self>;
}

/// Trait for types that can be peeked at
pub trait Peek {
    fn peek(&self, token: &Token) -> bool;
}

/// Trait for parsing specific tokens
pub trait ParseToken: Sized {
    fn parse_token(input: &mut ParseStream) -> ParseResult<Self>;
}

// Implement Peek for TokenKind
impl Peek for TokenKind {
    fn peek(&self, token: &Token) -> bool {
        std::mem::discriminant(&token.kind) == std::mem::discriminant(self)
    }
}
```

#### 1.3 Update Core Module Structure

Update `crates/prism-parser/src/core/mod.rs`:

```rust
//! Core Parsing Infrastructure
//!
//! Enhanced with syn-inspired parsing traits and utilities

pub mod token_stream_manager;
pub mod parsing_coordinator;
pub mod error;
pub mod precedence;
pub mod recovery;
pub mod parse_trait;  // NEW

// Re-export enhanced types
pub use token_stream_manager::TokenStreamManager;
pub use parsing_coordinator::ParsingCoordinator;
pub use error::{ParseError, ParseErrorKind, ParseResult, ErrorContext};
pub use precedence::{Precedence, Associativity, infix_precedence, prefix_precedence, associativity};
pub use recovery::{RecoveryStrategy, RecoveryContext, ParsingContext};
pub use parse_trait::{Parse, ParseStream, Peek, ParseToken};  // NEW
```

### Phase 2: Implement Parse Trait for AST Types (Week 2)

#### 2.1 Basic AST Parse Implementations

Create `crates/prism-ast/src/parse_impls.rs`:

```rust
//! Parse implementations for AST types
//!
//! This module implements the Parse trait for core AST types

use crate::{AstNode, Expr, LiteralExpr, LiteralValue, VariableExpr, BinaryExpr, BinaryOperator};
use prism_parser::core::{Parse, ParseStream, ParseResult, Peek};
use prism_lexer::TokenKind;
use prism_common::{symbol::Symbol, NodeId};

impl Parse for LiteralExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.tokens.peek();
        
        let value = match &token.kind {
            TokenKind::IntegerLiteral(n) => {
                input.tokens.advance();
                LiteralValue::Integer(*n)
            }
            TokenKind::FloatLiteral(f) => {
                input.tokens.advance();
                LiteralValue::Float(*f)
            }
            TokenKind::StringLiteral(s) => {
                input.tokens.advance();
                LiteralValue::String(s.clone())
            }
            TokenKind::True => {
                input.tokens.advance();
                LiteralValue::Boolean(true)
            }
            TokenKind::False => {
                input.tokens.advance();
                LiteralValue::Boolean(false)
            }
            TokenKind::Null => {
                input.tokens.advance();
                LiteralValue::Null
            }
            _ => return Err(ParseError::expected_literal(input.span())),
        };

        Ok(LiteralExpr { value })
    }
}

impl Parse for VariableExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let token = input.tokens.peek();
        
        match &token.kind {
            TokenKind::Identifier(name) => {
                input.tokens.advance();
                Ok(VariableExpr {
                    name: Symbol::intern(name),
                })
            }
            _ => Err(ParseError::expected_identifier(input.span())),
        }
    }
}

// Helper for parsing expressions with precedence
impl Expr {
    pub fn parse_with_precedence(
        input: &mut ParseStream, 
        min_precedence: u8
    ) -> ParseResult<AstNode<Expr>> {
        // Implementation of Pratt parser using Parse trait
        let mut left = Self::parse_primary(input)?;
        
        while !input.is_empty() {
            let token = input.tokens.peek();
            if let Some(precedence) = get_infix_precedence(&token.kind) {
                if precedence < min_precedence {
                    break;
                }
                
                left = Self::parse_infix(input, left, precedence)?;
            } else {
                break;
            }
        }
        
        Ok(left)
    }

    fn parse_primary(input: &mut ParseStream) -> ParseResult<AstNode<Expr>> {
        let span = input.span();
        let id = NodeId::new(0); // Would use proper ID generation
        
        // Try different primary expression types
        if input.peek(TokenKind::IntegerLiteral(0)) || 
           input.peek(TokenKind::StringLiteral(String::new())) ||
           input.peek(TokenKind::True) || input.peek(TokenKind::False) ||
           input.peek(TokenKind::Null) {
            let literal = input.parse::<LiteralExpr>()?;
            Ok(AstNode::new(Expr::Literal(literal), span, id))
        } else if input.peek(TokenKind::Identifier(String::new())) {
            let var = input.parse::<VariableExpr>()?;
            Ok(AstNode::new(Expr::Variable(var), span, id))
        } else {
            Err(ParseError::expected_expression(span))
        }
    }

    fn parse_infix(
        input: &mut ParseStream,
        left: AstNode<Expr>,
        precedence: u8,
    ) -> ParseResult<AstNode<Expr>> {
        let op_token = input.tokens.advance();
        let operator = match op_token.kind {
            TokenKind::Plus => BinaryOperator::Add,
            TokenKind::Minus => BinaryOperator::Subtract,
            TokenKind::Star => BinaryOperator::Multiply,
            TokenKind::Slash => BinaryOperator::Divide,
            // ... other operators
            _ => return Err(ParseError::unexpected_token(op_token.span)),
        };

        let right = Self::parse_with_precedence(input, precedence + 1)?;
        
        let span = left.span.merge(right.span);
        let id = NodeId::new(0);
        
        Ok(AstNode::new(
            Expr::Binary(BinaryExpr {
                left: Box::new(left),
                operator,
                right: Box::new(right),
            }),
            span,
            id,
        ))
    }
}

fn get_infix_precedence(token: &TokenKind) -> Option<u8> {
    match token {
        TokenKind::Plus | TokenKind::Minus => Some(1),
        TokenKind::Star | TokenKind::Slash => Some(2),
        _ => None,
    }
}
```

#### 2.2 Update AST Module

Update `crates/prism-ast/src/lib.rs`:

```rust
//! Abstract Syntax Tree definitions for the Prism programming language
//!
//! Enhanced with syn-inspired parsing capabilities

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod arena;
pub mod expr;
pub mod metadata;
pub mod node;
pub mod pattern;
pub mod stmt;
pub mod types;
pub mod visitor;
pub mod type_inference;
pub mod transformations;
pub mod parse_impls;  // NEW

// Re-export main types
pub use arena::AstArena;
pub use expr::*;
pub use metadata::*;
pub use node::*;
pub use pattern::*;
pub use stmt::*;
pub use types::*;
pub use visitor::*;
pub use type_inference::*;
pub use transformations::*;

// Re-export parsing functionality
#[cfg(feature = "parsing")]
pub use parse_impls::*;

use prism_common::SourceId;
use std::collections::HashMap;

// ... rest of existing code
```

### Phase 3: Enhanced Combinators (Week 2-3)

#### 3.1 Create Syn-Style Combinators

Create `crates/prism-parser/src/combinators/mod.rs`:

```rust
//! Syn-inspired parser combinators
//!
//! This module provides composable parsing utilities

pub mod delimited;
pub mod punctuated;
pub mod optional;
pub mod alternative;

use crate::core::{Parse, ParseStream, ParseResult};

/// Parse a delimited sequence: prefix, content, suffix
pub fn delimited<P, C, S>(
    input: &mut ParseStream,
    _prefix: P,
    _suffix: S,
) -> ParseResult<C>
where
    P: Parse,
    C: Parse,
    S: Parse,
{
    let _prefix = input.parse::<P>()?;
    let content = input.parse::<C>()?;
    let _suffix = input.parse::<S>()?;
    Ok(content)
}

/// Parse an optional item
pub fn optional<T: Parse>(input: &mut ParseStream) -> ParseResult<Option<T>> {
    match input.parse::<T>() {
        Ok(item) => Ok(Some(item)),
        Err(_) => Ok(None), // In a real implementation, we'd check if it's a recoverable error
    }
}

/// Parse one of several alternatives
pub fn alternative<T>(
    input: &mut ParseStream,
    parsers: &[fn(&mut ParseStream) -> ParseResult<T>],
) -> ParseResult<T> {
    let mut errors = Vec::new();
    
    for parser in parsers {
        match parser(input) {
            Ok(result) => return Ok(result),
            Err(err) => errors.push(err),
        }
    }
    
    // Return the first error if all alternatives failed
    Err(errors.into_iter().next().unwrap_or_else(|| {
        ParseError::no_alternatives(input.span())
    }))
}
```

#### 3.2 Punctuated Sequences (Like syn::punctuated)

Create `crates/prism-parser/src/combinators/punctuated.rs`:

```rust
//! Punctuated sequence parsing - inspired by syn::punctuated
//!
//! Handles comma-separated lists and similar patterns

use crate::core::{Parse, ParseStream, ParseResult, ParseToken};
use std::vec::Vec;

/// A punctuated sequence of items
pub struct Punctuated<T, P> {
    items: Vec<T>,
    punctuation: Vec<P>,
}

impl<T, P> Punctuated<T, P> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            punctuation: Vec::new(),
        }
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn push_punct(&mut self, punct: P) {
        self.punctuation.push(punct);
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }

    pub fn into_items(self) -> Vec<T> {
        self.items
    }

    /// Parse a punctuated sequence with no trailing punctuation
    pub fn parse_separated(input: &mut ParseStream) -> ParseResult<Self>
    where
        T: Parse,
        P: ParseToken,
    {
        let mut punctuated = Self::new();
        
        // Parse first item if present
        if !input.is_empty() {
            punctuated.push(input.parse()?);
            
            // Parse remaining items with punctuation
            while !input.is_empty() {
                // Try to parse punctuation
                if let Ok(punct) = input.parse_token::<P>() {
                    punctuated.push_punct(punct);
                    
                    // Must have another item after punctuation
                    punctuated.push(input.parse()?);
                } else {
                    break;
                }
            }
        }
        
        Ok(punctuated)
    }

    /// Parse a punctuated sequence allowing trailing punctuation
    pub fn parse_terminated(input: &mut ParseStream) -> ParseResult<Self>
    where
        T: Parse,
        P: ParseToken,
    {
        let mut punctuated = Self::new();
        
        while !input.is_empty() {
            punctuated.push(input.parse()?);
            
            // Try to parse punctuation
            if let Ok(punct) = input.parse_token::<P>() {
                punctuated.push_punct(punct);
                
                // Check if there's another item
                if input.is_empty() {
                    break; // Trailing punctuation is OK
                }
            } else {
                break; // No more punctuation
            }
        }
        
        Ok(punctuated)
    }
}

impl<T, P> Default for Punctuated<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Parse for common punctuated patterns
impl<T, P> Parse for Punctuated<T, P>
where
    T: Parse,
    P: ParseToken,
{
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        Self::parse_separated(input)
    }
}
```

### Phase 4: Refactor Expression Parser (Week 3)

#### 4.1 Enhanced Expression Parser

Update `crates/prism-parser/src/parsers/expression_parser.rs`:

```rust
//! Expression Parsing with Syn-Inspired Patterns
//!
//! This module uses the Parse trait system for more maintainable expression parsing

use crate::core::{Parse, ParseStream, ParseResult, Precedence};
use crate::combinators::{delimited, optional, alternative};
use prism_ast::{AstNode, Expr, LiteralExpr, VariableExpr, BinaryExpr, CallExpr};
use prism_lexer::TokenKind;
use prism_common::{symbol::Symbol, NodeId};

/// Enhanced expression parser using Parse traits
pub struct ExpressionParser;

impl ExpressionParser {
    /// Parse an expression using the Parse trait system
    pub fn parse_expression(input: &mut ParseStream) -> ParseResult<AstNode<Expr>> {
        Expr::parse_with_precedence(input, 0)
    }

    /// Parse a primary expression (literals, identifiers, parenthesized expressions)
    pub fn parse_primary(input: &mut ParseStream) -> ParseResult<AstNode<Expr>> {
        let span = input.span();
        let id = NodeId::new(0);

        // Use alternative combinator to try different primary types
        alternative(input, &[
            // Parenthesized expression
            |input| {
                let expr = delimited(
                    input,
                    TokenKind::LeftParen,
                    TokenKind::RightParen,
                )?;
                Ok(AstNode::new(expr, span, id))
            },
            // Literal expression
            |input| {
                let literal = input.parse::<LiteralExpr>()?;
                Ok(AstNode::new(Expr::Literal(literal), span, id))
            },
            // Variable expression
            |input| {
                let var = input.parse::<VariableExpr>()?;
                Ok(AstNode::new(Expr::Variable(var), span, id))
            },
            // Function call - would be more complex in real implementation
            |input| {
                let call = input.parse::<CallExpr>()?;
                Ok(AstNode::new(Expr::Call(call), span, id))
            },
        ])
    }
}

// Implement Parse for complex expressions
impl Parse for CallExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Parse function name
        let callee = ExpressionParser::parse_primary(input)?;
        
        // Parse arguments in parentheses
        if input.peek(TokenKind::LeftParen) {
            let args = delimited(
                input,
                TokenKind::LeftParen,
                TokenKind::RightParen,
            )?;
            
            Ok(CallExpr {
                callee: Box::new(callee),
                arguments: args, // Would parse as Punctuated<Expr, Comma>
            })
        } else {
            Err(ParseError::expected_token(TokenKind::LeftParen, input.span()))
        }
    }
}
```

### Phase 5: Integration and Testing (Week 4)

#### 5.1 Update Main Parser

Update `crates/prism-parser/src/parser.rs`:

```rust
//! Main Parser Coordinator with Syn-Inspired Enhancements
//!
//! Enhanced with Parse trait system for better maintainability

use crate::{
    core::{Parse, ParseStream, ParseResult, TokenStreamManager, ParsingCoordinator},
    parsers::ExpressionParser,
};
use prism_ast::{AstNode, Program, Item};
use prism_lexer::Token;

/// Enhanced parser with syn-inspired patterns
pub struct Parser {
    /// Token stream manager
    token_stream: TokenStreamManager,
    /// Parsing coordinator
    coordinator: ParsingCoordinator,
    /// Configuration
    config: ParseConfig,
}

impl Parser {
    /// Parse using the new Parse trait system
    pub fn parse_with_traits<T: Parse>(&mut self) -> ParseResult<T> {
        let mut stream = ParseStream::new(&mut self.token_stream);
        T::parse(&mut stream)
    }

    /// Enhanced program parsing
    pub fn parse_program_enhanced(&mut self) -> ParseResult<Program> {
        let mut stream = ParseStream::new(&mut self.token_stream);
        
        let mut items = Vec::new();
        while !stream.is_empty() {
            let item = stream.parse::<Item>()?;
            items.push(item);
        }

        Ok(Program::new(items, self.coordinator.source_id()))
    }
}

// Implement Parse for Program
impl Parse for Program {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let mut items = Vec::new();
        let source_id = prism_common::SourceId::new(0); // Would get from context
        
        while !input.is_empty() {
            items.push(input.parse::<AstNode<Item>>()?);
        }
        
        Ok(Program::new(items, source_id))
    }
}
```

#### 5.2 Enhanced Error Handling

Update `crates/prism-parser/src/core/error.rs`:

```rust
//! Enhanced error handling with syn-inspired patterns

use prism_common::span::Span;
use prism_lexer::TokenKind;
use thiserror::Error;

/// Enhanced parse error with better diagnostics
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("Expected {expected}, found {found}")]
    UnexpectedToken {
        expected: Vec<TokenKind>,
        found: TokenKind,
        span: Span,
    },

    #[error("Expected literal value")]
    ExpectedLiteral { span: Span },

    #[error("Expected identifier")]
    ExpectedIdentifier { span: Span },

    #[error("Expected expression")]
    ExpectedExpression { span: Span },

    #[error("No valid alternatives found")]
    NoAlternatives { span: Span },

    #[error("Invalid syntax: {message}")]
    InvalidSyntax { message: String, span: Span },
}

impl ParseError {
    pub fn expected_literal(span: Span) -> Self {
        Self::ExpectedLiteral { span }
    }

    pub fn expected_identifier(span: Span) -> Self {
        Self::ExpectedIdentifier { span }
    }

    pub fn expected_expression(span: Span) -> Self {
        Self::ExpectedExpression { span }
    }

    pub fn no_alternatives(span: Span) -> Self {
        Self::NoAlternatives { span }
    }

    pub fn expected_token(expected: TokenKind, span: Span) -> Self {
        Self::UnexpectedToken {
            expected: vec![expected],
            found: TokenKind::Eof, // Would get actual token
            span,
        }
    }

    pub fn unexpected_token(span: Span) -> Self {
        Self::InvalidSyntax {
            message: "Unexpected token".to_string(),
            span,
        }
    }
}
```

### Phase 6: Gradual Migration Plan

#### 6.1 Package Modifications Required

**Packages that need modification to maintain SoC:**

1. **`prism-parser`** (Primary changes)
   - Add Parse trait infrastructure
   - Enhance combinators
   - Refactor expression parsing
   - Improve error handling

2. **`prism-ast`** (Secondary changes)
   - Add Parse implementations for AST types
   - Feature-gate parsing functionality
   - Maintain existing visitor patterns

3. **`prism-lexer`** (Minimal changes)
   - Add ParseToken implementations for tokens
   - Enhance token type information

4. **`prism-common`** (No changes)
   - Keep as foundational utilities

#### 6.2 Migration Strategy

**Week 1: Foundation**
- Implement Parse trait infrastructure
- Add basic combinators
- Create error handling improvements

**Week 2: Core Types**
- Implement Parse for basic AST types
- Add punctuated sequence support
- Enhance expression parsing

**Week 3: Advanced Features**
- Implement complex expression parsing
- Add statement parsing with Parse traits
- Create comprehensive combinators

**Week 4: Integration**
- Integrate new system with existing code
- Add comprehensive tests
- Performance optimization

**Week 5: Migration**
- Gradually migrate existing parsers
- Maintain backward compatibility
- Complete documentation

### Phase 7: Testing Strategy

#### 7.1 Compatibility Tests

Create `crates/prism-parser/tests/syn_compatibility.rs`:

```rust
//! Tests to ensure syn-inspired parsing maintains compatibility

use prism_parser::{Parser, ParseConfig};
use prism_lexer::Lexer;

#[test]
fn test_expression_parsing_compatibility() {
    let source = "1 + 2 * 3";
    let tokens = Lexer::new(source).collect_tokens().unwrap();
    let mut parser = Parser::new(tokens);
    
    // Test both old and new parsing methods produce same result
    let old_result = parser.parse_expression_old();
    let new_result = parser.parse_with_traits::<Expr>();
    
    // Compare AST structure (would implement PartialEq)
    assert_eq!(old_result.unwrap(), new_result.unwrap());
}

#[test]
fn test_performance_regression() {
    let source = include_str!("../test_data/large_file.prism");
    let tokens = Lexer::new(source).collect_tokens().unwrap();
    
    let start = std::time::Instant::now();
    let mut parser = Parser::new(tokens);
    let _result = parser.parse_program_enhanced().unwrap();
    let duration = start.elapsed();
    
    // Ensure performance is acceptable (< 10% regression)
    assert!(duration.as_millis() < 1000, "Parsing took too long: {:?}", duration);
}
```

## Key Implementation Principles

### 1. Maintain Separation of Concerns
- Keep parsing logic in `prism-parser`
- AST types remain in `prism-ast`
- Parse implementations are feature-gated
- No circular dependencies

### 2. Avoid Over-Engineering
- Don't implement every syn feature
- Focus on what improves your specific use case
- Keep existing visitor patterns
- Maintain backward compatibility during transition

### 3. Gradual Migration
- Implement new system alongside old
- Migrate one parser at a time
- Extensive testing at each step
- Feature flags for easy rollback

### 4. Performance Considerations
- Profile before and after changes
- Use zero-cost abstractions where possible
- Maintain or improve parsing speed
- Monitor memory usage

## Benefits After Implementation

1. **Maintainability**: Easier to add new syntax
2. **Composability**: Reusable parsing components
3. **Error Quality**: Better error messages and locations
4. **Testing**: Easier to unit test individual parsers
5. **Extensibility**: Simple to add new language features

## Conclusion

This implementation guide provides a practical path to adopt syn's parsing patterns while maintaining your current architecture. The key is gradual migration with extensive testing to ensure no regressions in functionality or performance.

The syn-inspired approach will make your parser more maintainable and extensible, which is valuable for a language that will evolve over time. Focus on the core benefits (Parse trait, combinators, better errors) rather than implementing every syn feature. 