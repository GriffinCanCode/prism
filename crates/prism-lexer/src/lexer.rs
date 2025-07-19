//! Lexer implementation for the Prism programming language
//!
//! This module provides the main lexer that converts source code into tokens.
//! The lexer focuses purely on CHARACTER-TO-TOKEN conversion and does NOT:
//! - Detect syntax styles (belongs in parser)
//! - Generate semantic summaries (belongs in parser)
//! - Analyze token relationships (belongs in parser)

use crate::token::{Token, TokenKind};
use crate::recovery::{ErrorRecovery, ErrorPattern, SyntaxStyle};
use prism_common::{
    diagnostics::DiagnosticBag,
    span::{Position, Span},
    symbol::SymbolTable,
    SourceId,
};
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
}

impl From<LexerError> for ErrorPattern {
    fn from(error: LexerError) -> Self {
        match error {
            LexerError::InvalidCharacter(ch, _) => ErrorPattern::InvalidCharacter(ch),
            LexerError::UnterminatedString(_) => ErrorPattern::UnterminatedString,
            LexerError::InvalidNumber(_) => ErrorPattern::InvalidNumber,
            LexerError::InvalidEscape(_) => ErrorPattern::InvalidEscape,
            LexerError::UnexpectedEof => ErrorPattern::UnexpectedEof,
        }
    }
}

/// Configuration for the lexer
#[derive(Debug, Clone)]
pub struct LexerConfig {
    /// Enable aggressive error recovery
    pub aggressive_recovery: bool,
    /// Maximum number of errors before stopping
    pub max_errors: usize,
    /// Whether to preserve whitespace tokens
    pub preserve_whitespace: bool,
    /// Whether to preserve comment tokens
    pub preserve_comments: bool,
}

impl Default for LexerConfig {
    fn default() -> Self {
        Self {
            aggressive_recovery: true,
            max_errors: 100,
            preserve_whitespace: false,
            preserve_comments: false,
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
}

/// The main lexer for Prism source code
/// 
/// This lexer has a SINGLE responsibility: convert characters to tokens.
/// It does NOT analyze syntax styles, generate semantic summaries, or
/// understand token relationships - those are parser responsibilities.
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
    /// Configuration
    config: LexerConfig,
    /// Error recovery handler
    error_recovery: ErrorRecovery,
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
        
        Self {
            source,
            position: Position::new(1, 1, 0),
            source_id,
            symbol_table,
            diagnostics: DiagnosticBag::new(),
            chars,
            current_char,
            config,
            error_recovery: ErrorRecovery::new(),
        }
    }

    /// Tokenize the source code
    pub fn tokenize(mut self) -> LexerResult {
        let mut tokens = Vec::new();
        
        while let Some(token) = self.next_token() {
            match token {
                Ok(token) => {
                    // Always preserve EOF token
                    if matches!(token.kind, TokenKind::Eof) {
                        tokens.push(token);
                        break;
                    }
                    
                    // Filter whitespace/comments based on config
                    match &token.kind {
                        TokenKind::Whitespace if !self.config.preserve_whitespace => continue,
                        TokenKind::LineComment(_) | TokenKind::BlockComment(_) 
                            if !self.config.preserve_comments => continue,
                        _ => tokens.push(token),
                    }
                }
                Err(error) => {
                    self.diagnostics.error(error.to_string(), Span::from_position(self.position, self.source_id));
                    if self.diagnostics.error_count() >= self.config.max_errors {
                        break;
                    }
                    
                    // Try error recovery
                    if self.config.aggressive_recovery {
                        if let Some(recovery_result) = self.error_recovery.recover_from_error(
                            error.into(),
                            self.source_id,
                            self.position,
                            SyntaxStyle::Canonical,
                        ) {
                            for token in recovery_result.tokens {
                                tokens.push(token);
                            }
                        }
                    }
                    
                    // Skip the problematic character
                    self.advance();
                }
            }
        }
        
        LexerResult {
            tokens,
            diagnostics: self.diagnostics,
        }
    }

    /// Get the next token
    fn next_token(&mut self) -> Option<Result<Token, LexerError>> {
        self.skip_whitespace_and_comments();
        
        let start_pos = self.position;
        
        match self.current_char {
            None => Some(Ok(Token::new(
                TokenKind::Eof,
                Span::new(start_pos, self.position, self.source_id),
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
                    '%' => Some(Ok(self.single_char_token(TokenKind::Percent))),
                    '=' => self.read_equals(),
                    '!' => self.read_bang(),
                    '<' => self.read_less(),
                    '>' => self.read_greater(),
                    '&' => self.read_ampersand(),
                    '|' => self.read_pipe(),
                    '^' => Some(Ok(self.single_char_token(TokenKind::Caret))),
                    '~' => Some(Ok(self.single_char_token(TokenKind::Tilde))),
                    
                    // Delimiters
                    '(' => Some(Ok(self.single_char_token(TokenKind::LeftParen))),
                    ')' => Some(Ok(self.single_char_token(TokenKind::RightParen))),
                    '[' => Some(Ok(self.single_char_token(TokenKind::LeftBracket))),
                    ']' => Some(Ok(self.single_char_token(TokenKind::RightBracket))),
                    '{' => Some(Ok(self.single_char_token(TokenKind::LeftBrace))),
                    '}' => Some(Ok(self.single_char_token(TokenKind::RightBrace))),
                    
                    // Punctuation
                    ',' => Some(Ok(self.single_char_token(TokenKind::Comma))),
                    ';' => Some(Ok(self.single_char_token(TokenKind::Semicolon))),
                    ':' => self.read_colon(),
                    '.' => self.read_dot(),
                    '?' => Some(Ok(self.single_char_token(TokenKind::Question))),
                    '@' => Some(Ok(self.single_char_token(TokenKind::At))),
                    
                    // Whitespace (preserve if configured)
                    ' ' | '\t' | '\r' => {
                        if self.config.preserve_whitespace {
                            Some(Ok(self.read_whitespace()))
                        } else {
                            self.advance();
                            self.next_token()
                        }
                    }
                    
                    // Newlines (always significant for some parsing)
                    '\n' => Some(Ok(self.single_char_token(TokenKind::Newline))),
                    
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
                ' ' | '\t' | '\r' if !self.config.preserve_whitespace => {
                    self.advance();
                }
                '/' if self.peek_char() == Some('/') && !self.config.preserve_comments => {
                    self.skip_line_comment();
                }
                '/' if self.peek_char() == Some('*') && !self.config.preserve_comments => {
                    self.skip_block_comment();
                }
                _ => break,
            }
        }
    }

    /// Skip a line comment
    fn skip_line_comment(&mut self) {
        let mut content = String::new();
        
        self.advance(); // Skip first '/'
        self.advance(); // Skip second '/'
        
        while let Some(ch) = self.current_char {
            if ch == '\n' {
                break;
            }
            content.push(ch);
            self.advance();
        }
        
        if self.config.preserve_comments {
            // This would need to be handled differently if we want to preserve comments
            // For now, we skip them
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

    /// Read whitespace token
    fn read_whitespace(&mut self) -> Token {
        let start_pos = self.position;
        let mut content = String::new();
        
        while let Some(ch) = self.current_char {
            match ch {
                ' ' | '\t' | '\r' => {
                    content.push(ch);
                    self.advance();
                }
                _ => break,
            }
        }
        
        Token::new(
            TokenKind::Whitespace,
            Span::new(start_pos, self.position, self.source_id),
        )
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

    /// Read minus or minus-assign
    fn read_minus(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '-'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::MinusAssign)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Minus)))
        }
    }

    /// Read star, star-assign, or power
    fn read_star(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '*'
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::StarAssign)))
            }
            Some('*') => {
                self.advance(); // Skip first '*'
                self.advance(); // Skip second '*'
                Some(Ok(self.make_token(TokenKind::Power)))
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Star))),
        }
    }

    /// Read slash, slash-assign, or comments
    fn read_slash(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '/'
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::SlashAssign)))
            }
            Some('/') => {
                // Line comment - handle based on config
                if self.config.preserve_comments {
                    self.read_line_comment()
                } else {
                    self.skip_line_comment();
                    self.next_token()
                }
            }
            Some('*') => {
                // Block comment - handle based on config
                if self.config.preserve_comments {
                    self.read_block_comment()
                } else {
                    self.skip_block_comment();
                    self.next_token()
                }
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Slash))),
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

    /// Read less, less-equal, or left-shift
    fn read_less(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '<'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::LessEqual)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Less)))
        }
    }

    /// Read greater, greater-equal, or right-shift
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

    /// Read line comment as token
    fn read_line_comment(&mut self) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        let mut content = String::new();
        
        self.advance(); // Skip first '/'
        self.advance(); // Skip second '/'
        
        while let Some(ch) = self.current_char {
            if ch == '\n' {
                break;
            }
            content.push(ch);
            self.advance();
        }
        
        Some(Ok(Token::new(
            TokenKind::LineComment(content),
            Span::new(start_pos, self.position, self.source_id),
        )))
    }

    /// Read block comment as token
    fn read_block_comment(&mut self) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        let mut content = String::new();
        
        self.advance(); // Skip '/'
        self.advance(); // Skip '*'
        
        while let Some(ch) = self.current_char {
            if ch == '*' && self.peek_char() == Some('/') {
                self.advance(); // Skip '*'
                self.advance(); // Skip '/'
                break;
            }
            content.push(ch);
            self.advance();
        }
        
        Some(Ok(Token::new(
            TokenKind::BlockComment(content),
            Span::new(start_pos, self.position, self.source_id),
        )))
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
                        Span::new(start_pos, self.position, self.source_id),
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

    /// Read a character or string literal
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
                        Span::new(start_pos, self.position, self.source_id),
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
            Span::new(start_pos, self.position, self.source_id),
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
        
        // Check if it's a keyword - this is the ONLY analysis we do
        let token_kind = match value.as_str() {
            // Core language keywords
            "module" => TokenKind::Module,
            "section" => TokenKind::Section,
            "capability" => TokenKind::Capability,
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
            // Everything else is an identifier
            _ => TokenKind::Identifier(value),
        };
        
        Some(Ok(Token::new(
            token_kind,
            Span::new(start_pos, self.position, self.source_id),
        )))
    }

    /// Create a single character token
    fn single_char_token(&mut self, kind: TokenKind) -> Token {
        let start_pos = self.position;
        self.advance();
        Token::new(
            kind,
            Span::new(start_pos, self.position, self.source_id),
        )
    }

    /// Make a token at the current position
    fn make_token(&self, kind: TokenKind) -> Token {
        Token::new(
            kind,
            Span::new(self.position, self.position, self.source_id),
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
} 