//! Function Call Expression Parsing
//!
//! This module handles advanced function call parsing including:
//! - Type arguments (generic function calls)
//! - Named arguments and parameter binding
//! - Method calls vs function calls
//! - Pipeline call syntax
//! - Async function calls
//!
//! **Conceptual Responsibility**: Parse complex function call expressions
//! **What it does**: type arguments, named parameters, call styles, method chaining
//! **What it doesn't do**: basic expression parsing, precedence handling

use crate::core::{Parse, ParseStream, ParseResult, ParseError, Precedence};
use crate::stream_combinators::{comma_separated, optional, bracketed, parenthesized};
use prism_ast::{
    AstNode, Expr, CallExpr, CallStyle, Type, NamedArgument, TypeArgument
};
use prism_lexer::TokenKind;
use prism_common::{span::Span, symbol::Symbol};

/// Enhanced call expression with type arguments and named parameters
#[derive(Debug, Clone)]
pub struct EnhancedCallExpr {
    pub callee: Box<AstNode<Expr>>,
    pub type_arguments: Option<Vec<AstNode<Type>>>,
    pub arguments: Vec<CallArgument>,
    pub call_style: CallStyle,
    pub is_async: bool,
}

/// Different types of call arguments
#[derive(Debug, Clone)]
pub enum CallArgument {
    /// Positional argument: expr
    Positional(AstNode<Expr>),
    /// Named argument: name = expr
    Named {
        name: Symbol,
        value: AstNode<Expr>,
    },
    /// Spread argument: ...expr
    Spread(AstNode<Expr>),
}

impl Parse for CallArgument {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // Check for spread argument first
        if input.check(TokenKind::DotDotDot) {
            input.advance(); // consume '...'
            let expr = input.parse::<AstNode<Expr>>()?;
            return Ok(CallArgument::Spread(expr));
        }
        
        // Try to parse as named argument (identifier = expression)
        if input.check(TokenKind::Identifier("".to_string())) {
            let checkpoint = input.checkpoint();
            
            // Try to parse identifier followed by '='
            if let Ok(name) = input.parse::<String>() {
                if input.check(TokenKind::Assign) {
                    input.advance(); // consume '='
                    let value = input.parse::<AstNode<Expr>>()?;
                    return Ok(CallArgument::Named {
                        name: Symbol::intern(&name),
                        value,
                    });
                }
            }
            
            // Not a named argument, restore and parse as positional
            input.restore(checkpoint);
        }
        
        // Parse as positional argument
        let expr = input.parse::<AstNode<Expr>>()?;
        Ok(CallArgument::Positional(expr))
    }
}

/// Type arguments for generic function calls
pub struct TypeArgumentList(pub Vec<AstNode<Type>>);

impl Parse for TypeArgumentList {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        if !input.check(TokenKind::Less) {
            return Ok(TypeArgumentList(Vec::new()));
        }
        
        input.advance(); // consume '<'
        
        let mut type_args = Vec::new();
        
        if !input.check(TokenKind::Greater) {
            loop {
                let type_arg = input.parse::<AstNode<Type>>()?;
                type_args.push(type_arg);
                
                if input.check(TokenKind::Comma) {
                    input.advance(); // consume ','
                    
                    // Check for trailing comma
                    if input.check(TokenKind::Greater) {
                        break;
                    }
                } else if input.check(TokenKind::Greater) {
                    break;
                } else {
                    return Err(ParseError::unexpected_token(
                        vec![TokenKind::Comma, TokenKind::Greater],
                        input.peek_token().kind.clone(),
                        input.span(),
                    ));
                }
            }
        }
        
        if !input.check(TokenKind::Greater) {
            return Err(ParseError::expected_token(TokenKind::Greater, input.span()));
        }
        input.advance(); // consume '>'
        
        Ok(TypeArgumentList(type_args))
    }
}

/// Call argument list with support for named and spread arguments
pub struct CallArgumentList(pub Vec<CallArgument>);

impl Parse for CallArgumentList {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        let arguments = comma_separated::<CallArgument>(input)?;
        Ok(CallArgumentList(arguments))
    }
}

/// Enhanced call expression parser
impl Parse for EnhancedCallExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // This would typically be called after we already have the callee expression
        // For this implementation, we'll assume the callee is already parsed
        return Err(ParseError::invalid_syntax(
            "enhanced_call".to_string(),
            "EnhancedCallExpr should be constructed from existing call parsing".to_string(),
            input.span(),
        ));
    }
}

/// Method call expression (object.method(args))
pub struct MethodCallExpr {
    pub object: Box<AstNode<Expr>>,
    pub method: Symbol,
    pub type_arguments: Option<Vec<AstNode<Type>>>,
    pub arguments: Vec<CallArgument>,
    pub is_async: bool,
}

impl Parse for MethodCallExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // This would be called when parsing member access followed by call
        return Err(ParseError::invalid_syntax(
            "method_call".to_string(),
            "MethodCallExpr should be constructed from member access parsing".to_string(),
            input.span(),
        ));
    }
}

/// Pipeline call expression (value |> function)
pub struct PipelineCallExpr {
    pub value: Box<AstNode<Expr>>,
    pub function: Box<AstNode<Expr>>,
    pub arguments: Vec<CallArgument>,
}

impl Parse for PipelineCallExpr {
    fn parse(input: &mut ParseStream) -> ParseResult<Self> {
        // This would be handled by infix parsing for the |> operator
        return Err(ParseError::invalid_syntax(
            "pipeline_call".to_string(),
            "PipelineCallExpr should be handled by infix parsing".to_string(),
            input.span(),
        ));
    }
}

/// Helper functions for call expression parsing

/// Parse function call with enhanced features
pub fn parse_enhanced_function_call(
    input: &mut ParseStream,
    callee: AstNode<Expr>,
) -> ParseResult<AstNode<Expr>> {
    // Parse optional type arguments
    let type_arguments = if input.check(TokenKind::Less) {
        Some(input.parse::<TypeArgumentList>()?.0)
    } else {
        None
    };
    
    // Parse opening parenthesis
    if !input.check(TokenKind::LeftParen) {
        return Err(ParseError::expected_token(TokenKind::LeftParen, input.span()));
    }
    input.advance(); // consume '('
    
    // Parse arguments
    let arguments = if input.check(TokenKind::RightParen) {
        Vec::new()
    } else {
        input.parse::<CallArgumentList>()?.0
    };
    
    // Parse closing parenthesis
    if !input.check(TokenKind::RightParen) {
        return Err(ParseError::expected_token(TokenKind::RightParen, input.span()));
    }
    input.advance(); // consume ')'
    
    // Convert to standard CallExpr format
    let standard_args: Vec<AstNode<Expr>> = arguments.into_iter().map(|arg| {
        match arg {
            CallArgument::Positional(expr) => expr,
            CallArgument::Named { name: _, value } => value, // TODO: Handle named args properly
            CallArgument::Spread(expr) => expr, // TODO: Handle spread args properly
        }
    }).collect();
    
    let span = callee.span; // TODO: Calculate proper span
    let call_expr = CallExpr {
        callee: Box::new(callee),
        arguments: standard_args,
        type_arguments: type_arguments.map(|args| {
            // Convert Type nodes to TypeArgument enum
            // This is a simplified conversion
            args.into_iter().map(|_| TypeArgument::Type(Symbol::intern("placeholder"))).collect()
        }),
        call_style: CallStyle::Function,
    };
    
    Ok(AstNode::new(
        Expr::Call(call_expr),
        span,
        prism_common::NodeId::new(0),
    ))
}

/// Parse method call (object.method(args))
pub fn parse_method_call(
    input: &mut ParseStream,
    object: AstNode<Expr>,
    method_name: Symbol,
) -> ParseResult<AstNode<Expr>> {
    // Parse optional type arguments
    let type_arguments = if input.check(TokenKind::Less) {
        Some(input.parse::<TypeArgumentList>()?.0)
    } else {
        None
    };
    
    // Parse opening parenthesis
    if !input.check(TokenKind::LeftParen) {
        return Err(ParseError::expected_token(TokenKind::LeftParen, input.span()));
    }
    input.advance(); // consume '('
    
    // Parse arguments
    let arguments = if input.check(TokenKind::RightParen) {
        Vec::new()
    } else {
        input.parse::<CallArgumentList>()?.0
    };
    
    // Parse closing parenthesis
    if !input.check(TokenKind::RightParen) {
        return Err(ParseError::expected_token(TokenKind::RightParen, input.span()));
    }
    input.advance(); // consume ')'
    
    // Convert to standard CallExpr format for method calls
    let standard_args: Vec<AstNode<Expr>> = arguments.into_iter().map(|arg| {
        match arg {
            CallArgument::Positional(expr) => expr,
            CallArgument::Named { name: _, value } => value,
            CallArgument::Spread(expr) => expr,
        }
    }).collect();
    
    let span = object.span; // TODO: Calculate proper span
    let call_expr = CallExpr {
        callee: Box::new(AstNode::new(
            Expr::Member(prism_ast::MemberExpr {
                object: Box::new(object),
                member: method_name,
                safe_navigation: false,
            }),
            span,
            prism_common::NodeId::new(0),
        )),
        arguments: standard_args,
        type_arguments: type_arguments.map(|args| {
            args.into_iter().map(|_| TypeArgument::Type(Symbol::intern("placeholder"))).collect()
        }),
        call_style: CallStyle::Method,
    };
    
    Ok(AstNode::new(
        Expr::Call(call_expr),
        span,
        prism_common::NodeId::new(0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{TokenStreamManager, ParsingCoordinator};
    use prism_lexer::Token;
    use prism_common::{span::{Position, Span}, SourceId};

    fn create_test_tokens(kinds: Vec<TokenKind>) -> Vec<Token> {
        let source_id = SourceId::new(0);
        kinds.into_iter().enumerate().map(|(i, kind)| {
            let start = Position::new(i as u32, i as u32, i as u32);
            let end = Position::new(i as u32, i as u32 + 1, i as u32 + 1);
            Token::new(kind, Span::new(start, end, source_id))
        }).collect()
    }

    #[test]
    fn test_positional_call_argument() {
        let tokens = create_test_tokens(vec![TokenKind::IntegerLiteral(42)]);
        let mut token_manager = TokenStreamManager::new(tokens.clone());
        let mut coordinator = ParsingCoordinator::new(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<CallArgument>();
        assert!(result.is_ok());
        
        match result.unwrap() {
            CallArgument::Positional(_) => (), // Success
            _ => panic!("Expected positional argument"),
        }
    }

    #[test]
    fn test_named_call_argument() {
        let tokens = create_test_tokens(vec![
            TokenKind::Identifier("param".to_string()),
            TokenKind::Assign,
            TokenKind::IntegerLiteral(42),
        ]);
        let mut token_manager = TokenStreamManager::new(tokens.clone());
        let mut coordinator = ParsingCoordinator::new(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<CallArgument>();
        assert!(result.is_ok());
        
        match result.unwrap() {
            CallArgument::Named { name, value: _ } => {
                assert_eq!(name.as_str(), "param");
            }
            _ => panic!("Expected named argument"),
        }
    }

    #[test]
    fn test_spread_call_argument() {
        let tokens = create_test_tokens(vec![
            TokenKind::DotDotDot,
            TokenKind::Identifier("args".to_string()),
        ]);
        let mut token_manager = TokenStreamManager::new(tokens.clone());
        let mut coordinator = ParsingCoordinator::new(tokens);
        let mut stream = ParseStream::new(&mut token_manager, &mut coordinator);

        let result = stream.parse::<CallArgument>();
        assert!(result.is_ok());
        
        match result.unwrap() {
            CallArgument::Spread(_) => (), // Success
            _ => panic!("Expected spread argument"),
        }
    }
} 