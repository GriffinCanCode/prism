//! Stream-based parser combinators
//!
//! This module provides composable parsing utilities that work with ParseStream,
//! complementing the existing analysis/combinators.rs which works with the Parser API.
//! 
//! ## Separation of Concerns
//! - `analysis/combinators.rs`: Parser-based combinators for analysis and validation
//! - `stream_combinators/mod.rs`: ParseStream-based combinators for type-driven parsing

pub mod delimited;
pub mod punctuated;
pub mod optional;
pub mod alternative;

use crate::core::{Parse, ParseStream, ParseResult, ParseToken, Peek, ParseError};
use prism_lexer::TokenKind;

/// Parse a delimited sequence: prefix, content, suffix
/// 
/// Example: parenthesized expressions, bracketed lists, etc.
pub fn delimited<P, C, S>(
    input: &mut ParseStream,
    prefix: P,
    suffix: S,
) -> ParseResult<C>
where
    P: ParseToken,
    C: Parse,
    S: ParseToken,
{
    let _prefix = input.parse_token::<P>()?;
    let content = input.parse::<C>()?;
    let _suffix = input.parse_token::<S>()?;
    Ok(content)
}

/// Parse an optional item with safe backtracking
/// 
/// Returns Some(item) if the item can be parsed, None otherwise.
/// Uses checkpoint mechanism to prevent input consumption on failure.
pub fn optional<T: Parse>(input: &mut ParseStream) -> ParseResult<Option<T>> {
    match input.try_parse::<T>() {
        Ok(item) => Ok(Some(item)),
        Err(_) => Ok(None),
    }
}

/// Parse one of several alternatives with safe backtracking
/// 
/// Tries each parser function in order until one succeeds.
/// Uses checkpoint mechanism to prevent input consumption on failure.
pub fn alternative<T>(
    input: &mut ParseStream,
    parsers: &[fn(&mut ParseStream) -> ParseResult<T>],
) -> ParseResult<T> {
    let mut errors = Vec::new();
    
    for parser in parsers {
        let checkpoint = input.checkpoint();
        match parser(input) {
            Ok(result) => return Ok(result),
            Err(err) => {
                input.restore(checkpoint);
                errors.push(err);
            }
        }
    }
    
    // All alternatives failed - return the first error
    Err(errors.into_iter().next().unwrap_or_else(|| {
        ParseError::invalid_syntax(
            "alternative".to_string(),
            "No alternatives matched".to_string(),
            input.span(),
        )
    }))
}

/// Parse a sequence of items separated by a delimiter with safe backtracking
/// 
/// This version properly handles trailing delimiters and uses checkpoints.
pub fn separated_list<T, D>(
    input: &mut ParseStream,
    delimiter: D,
) -> ParseResult<Vec<T>>
where
    T: Parse,
    D: Peek + ParseToken + Clone,
{
    let mut items = Vec::new();
    
    // Try to parse first item
    if let Some(first_item) = input.speculative_parse::<T>() {
        items.push(first_item);
        
        // Parse remaining items with delimiters
        while input.peek(delimiter.clone()) {
            let _delimiter = input.parse_token::<D>()?;
            
            // Try to parse next item - if it fails, we have a trailing delimiter
            if let Some(next_item) = input.speculative_parse::<T>() {
                items.push(next_item);
            } else {
                break; // Trailing delimiter is OK
            }
        }
    }
    
    Ok(items)
}

/// Parse zero or more items with safe backtracking
pub fn many<T: Parse>(input: &mut ParseStream) -> ParseResult<Vec<T>> {
    let mut items = Vec::new();
    
    while !input.is_empty() {
        if let Some(item) = input.speculative_parse::<T>() {
            items.push(item);
        } else {
            break; // Can't parse more items
        }
    }
    
    Ok(items)
}

/// Parse one or more items with safe backtracking
pub fn many1<T: Parse>(input: &mut ParseStream) -> ParseResult<Vec<T>> {
    let mut items = vec![input.parse()?]; // Parse at least one
    
    while !input.is_empty() {
        if let Some(item) = input.speculative_parse::<T>() {
            items.push(item);
        } else {
            break; // Can't parse more items
        }
    }
    
    Ok(items)
}

/// Parse with lookahead - check if something can be parsed without consuming input
pub fn lookahead<T: Parse>(input: &ParseStream) -> bool {
    let checkpoint = input.checkpoint();
    let mut temp_stream = ParseStream::new(input.tokens, input.coordinator);
    temp_stream.restore(checkpoint);
    
    temp_stream.speculative_parse::<T>().is_some()
}

/// Parse a comma-separated list (convenience function)
pub fn comma_separated<T: Parse>(input: &mut ParseStream) -> ParseResult<Vec<T>> {
    separated_list::<T, TokenKind>(input, TokenKind::Comma)
}

/// Parse a semicolon-separated list (convenience function)  
pub fn semicolon_separated<T: Parse>(input: &mut ParseStream) -> ParseResult<Vec<T>> {
    separated_list::<T, TokenKind>(input, TokenKind::Semicolon)
}

/// Parse items enclosed in parentheses (convenience function)
pub fn parenthesized<T: Parse>(input: &mut ParseStream) -> ParseResult<T> {
    delimited::<TokenKind, T, TokenKind>(input, TokenKind::LeftParen, TokenKind::RightParen)
}

/// Parse items enclosed in brackets (convenience function)
pub fn bracketed<T: Parse>(input: &mut ParseStream) -> ParseResult<T> {
    delimited::<TokenKind, T, TokenKind>(input, TokenKind::LeftBracket, TokenKind::RightBracket)
}

/// Parse items enclosed in braces (convenience function)
pub fn braced<T: Parse>(input: &mut ParseStream) -> ParseResult<T> {
    delimited::<TokenKind, T, TokenKind>(input, TokenKind::LeftBrace, TokenKind::RightBrace)
} 