//! Lexer implementation for the Prism programming language
//!
//! This module provides the main lexer that converts source code into tokens
//! with rich semantic information for AI comprehension.

use crate::syntax::{SyntaxDetector, StyleRules};
use crate::token::{SemanticContext, SyntaxStyle, Token, TokenKind};
use logos::Logos;
use prism_common::{
    diagnostics::DiagnosticBag,
    span::{Position, Span},
    symbol::SymbolTable,
    SourceId,
};
use std::collections::VecDeque;
use thiserror::Error;

/// Lexer errors
#[derive(Error, Debug, Clone)]
pub enum LexerError {
    #[error("Invalid character '{0}' at position {1}")]
    InvalidCharacter(char, Position),
    #[error("Unterminated string literal at position {0}")]
    UnterminatedString(Position),
    #[error("Invalid number literal at position {0}")]
    InvalidNumber(Position),
    #[error("Invalid escape sequence in string at position {0}")]
    InvalidEscape(Position),
    #[error("Unexpected end of file")]
    UnexpectedEof,
    #[error("Semantic analysis failed: {0}")]
    SemanticAnalysisFailed(String),
}

/// Configuration for the lexer
#[derive(Debug, Clone)]
pub struct LexerConfig {
    /// Preferred syntax style (None for auto-detection)
    pub syntax_style: Option<SyntaxStyle>,
    /// Enable semantic metadata extraction
    pub semantic_metadata: bool,
    /// Enable aggressive error recovery
    pub aggressive_recovery: bool,
    /// Maximum number of errors before stopping
    pub max_errors: usize,
}

impl Default for LexerConfig {
    fn default() -> Self {
        Self {
            syntax_style: None,
            semantic_metadata: true,
            aggressive_recovery: true,
            max_errors: 100,
        }
    }
}

/// Result of lexical analysis
#[derive(Debug, Clone)]
pub struct LexerResult {
    /// Generated tokens
    pub tokens: Vec<Token>,
    /// Diagnostics (errors and warnings)
    pub diagnostics: DiagnosticBag,
    /// Detected syntax style
    pub syntax_style: SyntaxStyle,
    /// Semantic summary for AI comprehension
    pub semantic_summary: Option<SemanticSummary>,
}

/// Semantic summary of the tokenized code
#[derive(Debug, Clone)]
pub struct SemanticSummary {
    /// Identified modules
    pub modules: Vec<String>,
    /// Identified functions
    pub functions: Vec<String>,
    /// Identified types
    pub types: Vec<String>,
    /// Identified capabilities
    pub capabilities: Vec<String>,
    /// Overall semantic score
    pub semantic_score: f64,
}

/// The main lexer for Prism source code
pub struct Lexer<'source> {
    /// Source code being lexed
    source: &'source str,
    /// Current position in the source
    position: Position,
    /// Source file identifier
    source_id: SourceId,
    /// Symbol table for string interning
    symbol_table: &'source mut SymbolTable,
    /// Diagnostic collector
    diagnostics: DiagnosticBag,
    /// Character iterator
    chars: std::iter::Peekable<std::str::CharIndices<'source>>,
    /// Current character
    current_char: Option<char>,
    /// Detected syntax style
    syntax_style: SyntaxStyle,
    /// Style-specific rules
    style_rules: StyleRules,
    /// Configuration
    config: LexerConfig,
}

impl<'source> Lexer<'source> {
    /// Create a new lexer
    pub fn new(
        source: &'source str,
        source_id: SourceId,
        symbol_table: &'source mut SymbolTable,
        config: LexerConfig,
    ) -> Self {
        let mut chars = source.char_indices().peekable();
        let current_char = chars.next().map(|(_, c)| c);
        
        // Detect syntax style if not specified
        let syntax_style = config.syntax_style.clone()
            .unwrap_or_else(|| SyntaxDetector::detect_syntax(source).detected_style);
        
        let style_rules = StyleRules::for_style(&syntax_style);
        
        Self {
            source,
            position: Position::new(1, 1),
            source_id,
            symbol_table,
            diagnostics: DiagnosticBag::new(),
            chars,
            current_char,
            syntax_style,
            style_rules,
            config,
        }
    }

    /// Tokenize the source code
    pub fn tokenize(mut self) -> LexerResult {
        let mut tokens = Vec::new();
        
        while let Some(token) = self.next_token() {
            match token {
                Ok(token) => {
                    if !matches!(token.kind, TokenKind::Eof) {
                        tokens.push(token);
                    } else {
                        tokens.push(token);
                        break;
                    }
                }
                Err(error) => {
                    self.diagnostics.error(error.to_string());
                    if self.diagnostics.error_count() >= self.config.max_errors {
                        break;
                    }
                    // Try to recover by skipping the current character
                    self.advance();
                }
            }
        }
        
        let semantic_summary = if self.config.semantic_metadata {
            Some(self.generate_semantic_summary(&tokens))
        } else {
            None
        };
        
        LexerResult {
            tokens,
            diagnostics: self.diagnostics,
            syntax_style: self.syntax_style,
            semantic_summary,
        }
    }

    /// Get the next token
    fn next_token(&mut self) -> Option<Result<Token, LexerError>> {
        self.skip_whitespace_and_comments();
        
        let start_pos = self.position;
        
        match self.current_char {
            None => Some(Ok(Token::new(
                TokenKind::Eof,
                Span::new(self.source_id, start_pos, self.position),
                self.syntax_style.clone(),
            ))),
            Some(ch) => {
                let result = match ch {
                    // String literals
                    '"' => self.read_string(),
                    '\'' => self.read_char_or_string(),
                    
                    // Numbers
                    '0'..='9' => self.read_number(),
                    
                    // Identifiers and keywords
                    'a'..='z' | 'A'..='Z' | '_' => self.read_identifier_or_keyword(),
                    
                    // Operators and punctuation
                    '+' => self.read_plus(),
                    '-' => self.read_minus(),
                    '*' => self.read_star(),
                    '/' => self.read_slash(),
                    '%' => Some(self.single_char_token(TokenKind::Percent)),
                    '=' => self.read_equals(),
                    '!' => self.read_bang(),
                    '<' => self.read_less(),
                    '>' => self.read_greater(),
                    '&' => self.read_ampersand(),
                    '|' => self.read_pipe(),
                    '^' => Some(self.single_char_token(TokenKind::Caret)),
                    '~' => Some(self.single_char_token(TokenKind::Tilde)),
                    
                    // Delimiters
                    '(' => Some(self.single_char_token(TokenKind::LeftParen)),
                    ')' => Some(self.single_char_token(TokenKind::RightParen)),
                    '[' => Some(self.single_char_token(TokenKind::LeftBracket)),
                    ']' => Some(self.single_char_token(TokenKind::RightBracket)),
                    '{' => Some(self.single_char_token(TokenKind::LeftBrace)),
                    '}' => Some(self.single_char_token(TokenKind::RightBrace)),
                    
                    // Punctuation
                    ',' => Some(self.single_char_token(TokenKind::Comma)),
                    ';' => Some(self.single_char_token(TokenKind::Semicolon)),
                    ':' => self.read_colon(),
                    '.' => self.read_dot(),
                    '?' => self.read_question(),
                    '@' => self.read_at(),
                    
                    // Newlines (significant in some syntax styles)
                    '\n' => {
                        if self.style_rules.indentation_semantic {
                            Some(self.single_char_token(TokenKind::Newline))
                        } else {
                            self.advance();
                            self.next_token()
                        }
                    }
                    
                    // Invalid characters
                    _ => Some(Err(LexerError::InvalidCharacter(ch, self.position))),
                };
                
                result
            }
        }
    }

    /// Skip whitespace and comments
    fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.current_char {
            match ch {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                '\n' if !self.style_rules.indentation_semantic => {
                    self.advance();
                }
                '/' if self.peek_char() == Some('/') => {
                    self.skip_line_comment();
                }
                '/' if self.peek_char() == Some('*') => {
                    self.skip_block_comment();
                }
                _ => break,
            }
        }
    }

    /// Skip a line comment
    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.current_char {
            self.advance();
            if ch == '\n' {
                break;
            }
        }
    }

    /// Skip a block comment
    fn skip_block_comment(&mut self) {
        self.advance(); // Skip '/'
        self.advance(); // Skip '*'
        
        while let Some(ch) = self.current_char {
            if ch == '*' && self.peek_char() == Some('/') {
                self.advance(); // Skip '*'
                self.advance(); // Skip '/'
                break;
            }
            self.advance();
        }
    }

    /// Read a string literal
    fn read_string(&mut self) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        self.advance(); // Skip opening quote
        
        let mut value = String::new();
        
        while let Some(ch) = self.current_char {
            match ch {
                '"' => {
                    self.advance(); // Skip closing quote
                    return Some(Ok(Token::new(
                        TokenKind::StringLiteral(value),
                        Span::new(self.source_id, start_pos, self.position),
                        self.syntax_style.clone(),
                    )));
                }
                '\\' => {
                    self.advance();
                    match self.current_char {
                        Some('n') => value.push('\n'),
                        Some('t') => value.push('\t'),
                        Some('r') => value.push('\r'),
                        Some('\\') => value.push('\\'),
                        Some('"') => value.push('"'),
                        Some(ch) => return Some(Err(LexerError::InvalidEscape(self.position))),
                        None => return Some(Err(LexerError::UnexpectedEof)),
                    }
                    self.advance();
                }
                '\n' => {
                    return Some(Err(LexerError::UnterminatedString(self.position)));
                }
                _ => {
                    value.push(ch);
                    self.advance();
                }
            }
        }
        
        Some(Err(LexerError::UnterminatedString(start_pos)))
    }

    /// Read a character or string literal (single quotes)
    fn read_char_or_string(&mut self) -> Option<Result<Token, LexerError>> {
        // For now, treat single quotes as string literals
        // This could be enhanced to distinguish between char and string literals
        let start_pos = self.position;
        self.advance(); // Skip opening quote
        
        let mut value = String::new();
        
        while let Some(ch) = self.current_char {
            match ch {
                '\'' => {
                    self.advance(); // Skip closing quote
                    return Some(Ok(Token::new(
                        TokenKind::StringLiteral(value),
                        Span::new(self.source_id, start_pos, self.position),
                        self.syntax_style.clone(),
                    )));
                }
                '\\' => {
                    self.advance();
                    match self.current_char {
                        Some('n') => value.push('\n'),
                        Some('t') => value.push('\t'),
                        Some('r') => value.push('\r'),
                        Some('\\') => value.push('\\'),
                        Some('\'') => value.push('\''),
                        Some(_) => return Some(Err(LexerError::InvalidEscape(self.position))),
                        None => return Some(Err(LexerError::UnexpectedEof)),
                    }
                    self.advance();
                }
                '\n' => {
                    return Some(Err(LexerError::UnterminatedString(self.position)));
                }
                _ => {
                    value.push(ch);
                    self.advance();
                }
            }
        }
        
        Some(Err(LexerError::UnterminatedString(start_pos)))
    }

    /// Read a number literal
    fn read_number(&mut self) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        let mut value = String::new();
        let mut is_float = false;
        
        // Read integer part
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check for decimal point
        if self.current_char == Some('.') && self.peek_char().map_or(false, |c| c.is_ascii_digit()) {
            is_float = true;
            value.push('.');
            self.advance();
            
            // Read fractional part
            while let Some(ch) = self.current_char {
                if ch.is_ascii_digit() {
                    value.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        // Parse the number
        let token_kind = if is_float {
            match value.parse::<f64>() {
                Ok(f) => TokenKind::FloatLiteral(f),
                Err(_) => return Some(Err(LexerError::InvalidNumber(start_pos))),
            }
        } else {
            match value.parse::<i64>() {
                Ok(i) => TokenKind::IntegerLiteral(i),
                Err(_) => return Some(Err(LexerError::InvalidNumber(start_pos))),
            }
        };
        
        Some(Ok(Token::new(
            token_kind,
            Span::new(self.source_id, start_pos, self.position),
            self.syntax_style.clone(),
        )))
    }

    /// Read an identifier or keyword
    fn read_identifier_or_keyword(&mut self) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        let mut value = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check if it's a keyword
        let token_kind = match value.as_str() {
            "module" => TokenKind::Module,
            "mod" => TokenKind::Module,
            "section" => TokenKind::Section,
            "capability" => TokenKind::Capability,
            "cap" => TokenKind::Capability,
            "function" => TokenKind::Function,
            "fn" => TokenKind::Fn,
            "type" => TokenKind::Type,
            "interface" => TokenKind::Interface,
            "trait" => TokenKind::Trait,
            "let" => TokenKind::Let,
            "const" => TokenKind::Const,
            "var" => TokenKind::Var,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "loop" => TokenKind::Loop,
            "match" => TokenKind::Match,
            "case" => TokenKind::Case,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "yield" => TokenKind::Yield,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "where" => TokenKind::Where,
            "with" => TokenKind::With,
            "requires" => TokenKind::Requires,
            "ensures" => TokenKind::Ensures,
            "invariant" => TokenKind::Invariant,
            "effects" => TokenKind::Effects,
            "secure" => TokenKind::Secure,
            "unsafe" => TokenKind::Unsafe,
            "async" => TokenKind::Async,
            "await" => TokenKind::Await,
            "try" => TokenKind::Try,
            "catch" => TokenKind::Catch,
            "throw" => TokenKind::Throw,
            "error" => TokenKind::Error,
            "result" => TokenKind::Result,
            "import" => TokenKind::Import,
            "export" => TokenKind::Export,
            "use" => TokenKind::Use,
            "from" => TokenKind::From,
            "public" => TokenKind::Public,
            "pub" => TokenKind::Pub,
            "private" => TokenKind::Private,
            "internal" => TokenKind::Internal,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            "in" => TokenKind::In,
            "as" => TokenKind::As,
            "is" => TokenKind::Is,
            // Semantic type constraint keywords
            "min_value" => TokenKind::Identifier(value),
            "max_value" => TokenKind::Identifier(value),
            "min_length" => TokenKind::Identifier(value),
            "max_length" => TokenKind::Identifier(value),
            "pattern" => TokenKind::Identifier(value),
            "format" => TokenKind::Identifier(value),
            "precision" => TokenKind::Identifier(value),
            "currency" => TokenKind::Identifier(value),
            "non_negative" => TokenKind::Identifier(value),
            "immutable" => TokenKind::Identifier(value),
            "validated" => TokenKind::Identifier(value),
            "business_rule" => TokenKind::Identifier(value),
            "security_classification" => TokenKind::Identifier(value),
            "compliance" => TokenKind::Identifier(value),
            "ai_context" => TokenKind::Identifier(value),
            _ => TokenKind::Identifier(value),
        };
        
        Some(Ok(Token::new(
            token_kind,
            Span::new(self.source_id, start_pos, self.position),
            self.syntax_style.clone(),
        )))
    }

    /// Read plus or plus-assign
    fn read_plus(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '+'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::PlusAssign)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Plus)))
        }
    }

    /// Read minus or arrow or minus-assign
    fn read_minus(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '-'
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::MinusAssign)))
            }
            Some('>') => {
                self.advance(); // Skip '-'
                self.advance(); // Skip '>'
                Some(Ok(self.make_token(TokenKind::Arrow)))
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Minus))),
        }
    }

    /// Read star or star-assign or power
    fn read_star(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '*'
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::StarAssign)))
            }
            Some('*') => {
                self.advance(); // Skip '*'
                self.advance(); // Skip '*'
                Some(Ok(self.make_token(TokenKind::Power)))
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Star))),
        }
    }

    /// Read slash or slash-assign
    fn read_slash(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '/'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::SlashAssign)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Slash)))
        }
    }

    /// Read equals or double equals
    fn read_equals(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '='
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::Equal)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Assign)))
        }
    }

    /// Read bang or not-equals
    fn read_bang(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '!'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::NotEqual)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Bang)))
        }
    }

    /// Read less-than or less-equal
    fn read_less(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '<'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::LessEqual)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Less)))
        }
    }

    /// Read greater-than or greater-equal
    fn read_greater(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '>'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::GreaterEqual)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Greater)))
        }
    }

    /// Read ampersand or logical and
    fn read_ampersand(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('&') {
            self.advance(); // Skip '&'
            self.advance(); // Skip '&'
            Some(Ok(self.make_token(TokenKind::AndAnd)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Ampersand)))
        }
    }

    /// Read pipe or logical or
    fn read_pipe(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('|') {
            self.advance(); // Skip '|'
            self.advance(); // Skip '|'
            Some(Ok(self.make_token(TokenKind::OrOr)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Pipe)))
        }
    }

    /// Read colon or double colon
    fn read_colon(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some(':') {
            self.advance(); // Skip ':'
            self.advance(); // Skip ':'
            Some(Ok(self.make_token(TokenKind::DoubleColon)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Colon)))
        }
    }

    /// Read dot
    fn read_dot(&mut self) -> Option<Result<Token, LexerError>> {
        Some(Ok(self.single_char_token(TokenKind::Dot)))
    }

    /// Read question mark
    fn read_question(&mut self) -> Option<Result<Token, LexerError>> {
        Some(Ok(self.single_char_token(TokenKind::Question)))
    }

    /// Read at symbol
    fn read_at(&mut self) -> Option<Result<Token, LexerError>> {
        Some(Ok(self.single_char_token(TokenKind::At)))
    }

    /// Create a single character token
    fn single_char_token(&mut self, kind: TokenKind) -> Token {
        let start_pos = self.position;
        self.advance();
        Token::new(
            kind,
            Span::new(self.source_id, start_pos, self.position),
            self.syntax_style.clone(),
        )
    }

    /// Make a token at the current position
    fn make_token(&self, kind: TokenKind) -> Token {
        Token::new(
            kind,
            Span::new(self.source_id, self.position, self.position),
            self.syntax_style.clone(),
        )
    }

    /// Advance to the next character
    fn advance(&mut self) {
        if let Some(ch) = self.current_char {
            if ch == '\n' {
                self.position.line += 1;
                self.position.column = 1;
            } else {
                self.position.column += 1;
            }
        }
        
        self.current_char = self.chars.next().map(|(_, c)| c);
    }

    /// Peek at the next character without advancing
    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    /// Generate semantic summary
    fn generate_semantic_summary(&self, tokens: &[Token]) -> SemanticSummary {
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut capabilities = Vec::new();
        
        for token in tokens {
            match &token.kind {
                TokenKind::Module => {
                    // Look for the next identifier as the module name
                    // This is a simplified approach
                }
                TokenKind::Function | TokenKind::Fn => {
                    // Look for the next identifier as the function name
                }
                TokenKind::Type => {
                    // Look for the next identifier as the type name
                }
                TokenKind::Capability => {
                    // Look for the next identifier as the capability name
                }
                _ => {}
            }
        }
        
        SemanticSummary {
            modules,
            functions,
            types,
            capabilities,
            semantic_score: 0.8, // Placeholder
        }
    }
}

/// Semantic lexer that enriches tokens with AI-comprehensible metadata
pub struct SemanticLexer<'source> {
    /// Base lexer
    lexer: Lexer<'source>,
}

impl<'source> SemanticLexer<'source> {
    /// Create a new semantic lexer
    pub fn new(
        source: &'source str,
        source_id: SourceId,
        symbol_table: &'source mut SymbolTable,
        config: LexerConfig,
    ) -> Self {
        Self {
            lexer: Lexer::new(source, source_id, symbol_table, config),
        }
    }

    /// Tokenize with semantic enrichment
    pub fn tokenize_with_semantics(self) -> LexerResult {
        let mut result = self.lexer.tokenize();
        
        // Enrich tokens with semantic context
        for token in &mut result.tokens {
            if let Some(context) = self.infer_semantic_context(token) {
                token.semantic_context = Some(context);
            }
        }
        
        result
    }

    /// Infer semantic context for a token
    fn infer_semantic_context(&self, token: &Token) -> Option<SemanticContext> {
        match &token.kind {
            TokenKind::Module => {
                let mut context = SemanticContext::with_purpose("Define module boundary and capabilities");
                context.domain = Some("Module System".to_string());
                context.add_concept("Conceptual Cohesion");
                context.add_concept("Capability Isolation");
                context.add_ai_hint("Modules represent single business capabilities");
                Some(context)
            }
            TokenKind::Function | TokenKind::Fn => {
                let mut context = SemanticContext::with_purpose("Define function with semantic contracts");
                context.domain = Some("Function Definition".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Effect System");
                context.add_ai_hint("Functions should have single responsibility");
                Some(context)
            }
            TokenKind::Type => {
                let mut context = SemanticContext::with_purpose("Define semantic type with constraints");
                context.domain = Some("Type System".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Business Rules");
                context.add_ai_hint("Types should express business meaning");
                Some(context)
            }
            TokenKind::Effects => {
                let mut context = SemanticContext::with_purpose("Declare computational effects");
                context.domain = Some("Effect System".to_string());
                context.add_concept("Capability-Based Security");
                context.add_security_implication("Effects must be explicitly declared");
                Some(context)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::symbol::SymbolTable;

    #[test]
    fn test_basic_tokenization() {
        let source = "module Test { function foo() { return 42; } }";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert!(!result.tokens.is_empty());
        assert_eq!(result.tokens[0].kind, TokenKind::Module);
    }

    #[test]
    fn test_string_literals() {
        let source = r#""hello world""#;
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.tokens[0].kind, TokenKind::StringLiteral("hello world".to_string()));
    }

    #[test]
    fn test_number_literals() {
        let source = "42 3.14";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = Lexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize();
        
        assert_eq!(result.tokens[0].kind, TokenKind::IntegerLiteral(42));
        assert_eq!(result.tokens[1].kind, TokenKind::FloatLiteral(3.14));
    }

    #[test]
    fn test_semantic_enrichment() {
        let source = "module UserAuth { function authenticate() {} }";
        let source_id = SourceId::new(1);
        let mut symbol_table = SymbolTable::new();
        let config = LexerConfig::default();
        
        let lexer = SemanticLexer::new(source, source_id, &mut symbol_table, config);
        let result = lexer.tokenize_with_semantics();
        
        // Find the module token
        let module_token = result.tokens.iter()
            .find(|t| matches!(t.kind, TokenKind::Module))
            .unwrap();
            
        assert!(module_token.semantic_context.is_some());
        let context = module_token.semantic_context.as_ref().unwrap();
        assert!(context.purpose.is_some());
        assert_eq!(context.domain, Some("Module System".to_string()));
    }
} 