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
    #[error("Invalid regex literal at position {0}")]
    InvalidRegex(Position),
    #[error("Invalid money literal at position {0}")]
    InvalidMoney(Position),
    #[error("Invalid duration literal at position {0}")]
    InvalidDuration(Position),
    #[error("Indentation error at position {0}")]
    IndentationError(Position),
}

impl From<LexerError> for ErrorPattern {
    fn from(error: LexerError) -> Self {
        match error {
            LexerError::InvalidCharacter(ch, _) => ErrorPattern::InvalidCharacter(ch),
            LexerError::UnterminatedString(_) => ErrorPattern::UnterminatedString,
            LexerError::InvalidNumber(_) => ErrorPattern::InvalidNumber,
            LexerError::InvalidEscape(_) => ErrorPattern::InvalidEscape,
            LexerError::UnexpectedEof => ErrorPattern::UnexpectedEof,
            LexerError::InvalidRegex(_) => ErrorPattern::InvalidRegex,
            LexerError::InvalidMoney(_) => ErrorPattern::InvalidMoney,
            LexerError::InvalidDuration(_) => ErrorPattern::InvalidDuration,
            LexerError::IndentationError(_) => ErrorPattern::IndentationError,
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
    /// Indentation stack for Python-like syntax
    indent_stack: Vec<usize>,
    /// Whether we're at the beginning of a line
    at_line_start: bool,
    /// Pending dedent tokens to emit
    pending_dedents: Vec<usize>,
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
            indent_stack: Vec::new(),
            at_line_start: true,
            pending_dedents: Vec::new(),
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
        // Handle pending dedent tokens first
        if let Some(dedent_level) = self.pending_dedents.pop() {
            let span = Span::new(self.position, self.position, self.source_id);
            return Some(Ok(Token::new(TokenKind::Dedent(dedent_level), span)));
        }

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
                    '%' => self.read_percent(),
                    '=' => self.read_equals(),
                    '!' => self.read_bang(),
                    '<' => self.read_less(),
                    '>' => self.read_greater(),
                    '&' => self.read_ampersand(),
                    '|' => self.read_pipe(),
                    '^' => Some(Ok(self.single_char_token(TokenKind::Caret))),
                    '~' => self.read_tilde(),
                    
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
                    '?' => self.read_question(),
                    '@' => Some(Ok(self.single_char_token(TokenKind::At))),
                    '#' => Some(Ok(self.single_char_token(TokenKind::Hash))),
                    '$' => self.read_dollar(),
                    '≈' => Some(Ok(self.single_char_token(TokenKind::ConceptuallySimilar))),
                    
                    // Currency symbols (money literals)
                    '€' | '£' | '¥' | '₹' | '₽' | '₩' | '₪' | '₨' | '₡' | '₦' | '₵' | '₴' | '₸' | '₿' => {
                        if self.peek_char().map_or(false, |c| c.is_ascii_digit()) {
                            self.read_money_literal(ch)
                        } else {
                            Some(Err(LexerError::InvalidCharacter(ch, self.position)))
                        }
                    }
                    
                    // Whitespace (preserve if configured)
                    ' ' | '\t' | '\r' => {
                        if self.config.preserve_whitespace {
                            Some(Ok(self.read_whitespace()))
                        } else {
                            self.advance();
                            self.next_token()
                        }
                    }
                    
                    // Newlines (handle indentation tracking)
                    '\n' => {
                        self.at_line_start = true;
                        let newline_token = Some(Ok(self.single_char_token(TokenKind::Newline)));
                        // After newline, check for indentation changes on next call
                        self.process_indentation_after_newline();
                        newline_token
                    }
                    
                    // Invalid characters
                    _ => Some(Err(LexerError::InvalidCharacter(ch, self.position))),
                };
                
                // Reset line start flag after processing any non-whitespace token
                if !matches!(ch, ' ' | '\t' | '\r' | '\n') {
                    self.at_line_start = false;
                }
                
                result
            }
        }
    }

    /// Process indentation changes after a newline
    fn process_indentation_after_newline(&mut self) {
        // This will be called on the next token request to handle indentation
        // For now, we'll implement a simple version
    }

    /// Handle indentation at the start of a line
    fn handle_line_start_indentation(&mut self) -> Option<Result<Token, LexerError>> {
        if !self.at_line_start {
            return None;
        }

        let start_pos = self.position;
        let mut indent_level = 0;
        
        // Count spaces and tabs for indentation
        while let Some(ch) = self.current_char {
            match ch {
                ' ' => {
                    indent_level += 1;
                    self.advance();
                }
                '\t' => {
                    indent_level += 8; // Tab = 8 spaces
                    self.advance();
                }
                '\n' | '\r' => {
                    // Empty line, ignore indentation
                    return None;
                }
                _ => break,
            }
        }
        
        self.at_line_start = false;
        
        // Compare with current indentation level
        let current_level = self.indent_stack.last().copied().unwrap_or(0);
        
        if indent_level > current_level {
            // Increased indentation
            self.indent_stack.push(indent_level);
            Some(Ok(Token::new(
                TokenKind::Indent(indent_level),
                Span::new(start_pos, self.position, self.source_id),
            )))
        } else if indent_level < current_level {
            // Decreased indentation - may need multiple dedents
            let mut dedent_count = 0;
            while let Some(&level) = self.indent_stack.last() {
                if level <= indent_level {
                    break;
                }
                self.indent_stack.pop();
                self.pending_dedents.push(level);
                dedent_count += 1;
            }
            
            if dedent_count > 0 {
                // Return the first dedent, others will be returned on subsequent calls
                let dedent_level = self.pending_dedents.pop().unwrap();
                Some(Ok(Token::new(
                    TokenKind::Dedent(dedent_level),
                    Span::new(start_pos, self.position, self.source_id),
                )))
            } else {
                None
            }
        } else {
            // Same indentation level, no token needed
            None
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
            // Check for regex literal: /pattern/flags
            Some(ch) if ch != '/' && ch != '*' && ch != '=' => {
                // This could be a regex literal
                if self.could_be_regex_literal() {
                    self.read_regex_literal()
                } else {
                    Some(Ok(self.single_char_token(TokenKind::Slash)))
                }
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Slash))),
        }
    }

    /// Check if the current position could start a regex literal
    /// This is a simple heuristic - in a real implementation, we'd need more context
    fn could_be_regex_literal(&self) -> bool {
        // Simple heuristic: if we're after certain tokens, this might be a regex
        // In practice, this would need more sophisticated context analysis
        true // For now, always try regex parsing
    }

    /// Read a regex literal: /pattern/flags
    fn read_regex_literal(&mut self) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        self.advance(); // Skip opening '/'
        
        let mut pattern = String::new();
        let mut escaped = false;
        
        // Read the pattern part
        while let Some(ch) = self.current_char {
            if escaped {
                pattern.push('\\');
                pattern.push(ch);
                escaped = false;
                self.advance();
            } else if ch == '\\' {
                escaped = true;
                self.advance();
            } else if ch == '/' {
                // End of pattern
                self.advance(); // Skip closing '/'
                break;
            } else if ch == '\n' {
                return Some(Err(LexerError::UnterminatedString(self.position)));
            } else {
                pattern.push(ch);
                self.advance();
            }
        }
        
        // Read optional flags (i, g, m, s, etc.)
        let mut flags = String::new();
        while let Some(ch) = self.current_char {
            if ch.is_alphabetic() {
                flags.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Combine pattern and flags
        let regex_value = if flags.is_empty() {
            pattern
        } else {
            format!("{}|{}", pattern, flags)
        };
        
        Some(Ok(Token::new(
            TokenKind::RegexLiteral(regex_value),
            Span::new(start_pos, self.position, self.source_id),
        )))
    }

    /// Read equals or double equals
    fn read_equals(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '='
                if self.peek_char() == Some('=') {
                    self.advance(); // Skip second '='
                    self.advance(); // Skip third '='
                    Some(Ok(self.make_token(TokenKind::SemanticEqual)))
                } else {
                    self.advance(); // Skip second '='
                    Some(Ok(self.make_token(TokenKind::Equal)))
                }
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Assign))),
        }
    }

    /// Read bang or not-equals
    fn read_bang(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '!'
            if self.peek_char() == Some('=') {
                self.advance(); // Skip '='
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::SemanticNotEqual)))
            } else {
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::NotEqual)))
            }
        } else {
            Some(Ok(self.single_char_token(TokenKind::Bang)))
        }
    }

    /// Read less, less-equal, or left-shift
    fn read_less(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '<'
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::LessEqual)))
            }
            Some('<') => {
                self.advance(); // Skip '<'
                self.advance(); // Skip '<'
                Some(Ok(self.make_token(TokenKind::LeftShift)))
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Less))),
        }
    }

    /// Read greater, greater-equal, or right-shift
    fn read_greater(&mut self) -> Option<Result<Token, LexerError>> {
        match self.peek_char() {
            Some('=') => {
                self.advance(); // Skip '>'
                self.advance(); // Skip '='
                Some(Ok(self.make_token(TokenKind::GreaterEqual)))
            }
            Some('>') => {
                self.advance(); // Skip '>'
                self.advance(); // Skip '>'
                Some(Ok(self.make_token(TokenKind::RightShift)))
            }
            _ => Some(Ok(self.single_char_token(TokenKind::Greater))),
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

    /// Read dot, range, or spread operators
    fn read_dot(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('.') {
            self.advance(); // Skip first '.'
            if self.peek_char() == Some('.') {
                self.advance(); // Skip second '.'
                self.advance(); // Skip third '.'
                Some(Ok(self.make_token(TokenKind::DotDotDot)))
            } else {
                self.advance(); // Skip second '.'
                Some(Ok(self.make_token(TokenKind::DotDot)))
            }
        } else {
            Some(Ok(self.single_char_token(TokenKind::Dot)))
        }
    }

    /// Read question or null coalescing operator
    fn read_question(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('?') {
            self.advance(); // Skip '?'
            self.advance(); // Skip '?'
            Some(Ok(self.make_token(TokenKind::DoubleQuestion)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Question)))
        }
    }

    /// Read at symbol
    fn read_at(&mut self) -> Option<Result<Token, LexerError>> {
        Some(Ok(self.single_char_token(TokenKind::At)))
    }

    /// Read tilde or type compatible operator
    fn read_tilde(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '~'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::TypeCompatible)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Tilde)))
        }
    }

    /// Read percent or percent-assign
    fn read_percent(&mut self) -> Option<Result<Token, LexerError>> {
        if self.peek_char() == Some('=') {
            self.advance(); // Skip '%'
            self.advance(); // Skip '='
            Some(Ok(self.make_token(TokenKind::PercentAssign)))
        } else {
            Some(Ok(self.single_char_token(TokenKind::Percent)))
        }
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

    /// Read a number literal (enhanced to detect money and duration literals)
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
        
        // Check for duration suffixes (s, m, h, d, ms, etc.)
        if let Some(duration_literal) = self.try_read_duration_suffix(&value) {
            return Some(Ok(Token::new(
                TokenKind::DurationLiteral(duration_literal),
                Span::new(start_pos, self.position, self.source_id),
            )));
        }
        
        // Parse the number normally
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

    /// Try to read duration suffix and return complete duration literal
    fn try_read_duration_suffix(&mut self, number: &str) -> Option<String> {
        let checkpoint = self.position;
        let current_char_checkpoint = self.current_char;
        
        // Try to read duration units
        let mut unit = String::new();
        
        // Read alphabetic characters for the unit
        while let Some(ch) = self.current_char {
            if ch.is_alphabetic() {
                unit.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check if it's a valid duration unit
        if self.is_valid_duration_unit(&unit) {
            Some(format!("{}{}", number, unit))
        } else {
            // Restore position if not a valid duration
            self.position = checkpoint;
            self.current_char = current_char_checkpoint;
            None
        }
    }

    /// Check if a string is a valid duration unit
    fn is_valid_duration_unit(&self, unit: &str) -> bool {
        matches!(
            unit,
            "ns" | "us" | "μs" | "ms" | "s" | "m" | "h" | "d" | "w" | "y" |
            "nanosecond" | "nanoseconds" |
            "microsecond" | "microseconds" |
            "millisecond" | "milliseconds" |
            "second" | "seconds" |
            "minute" | "minutes" |
            "hour" | "hours" |
            "day" | "days" |
            "week" | "weeks" |
            "year" | "years"
        )
    }

    /// Read dollar sign - check for money literal
    fn read_dollar(&mut self) -> Option<Result<Token, LexerError>> {
        // Check if this starts a money literal
        if self.peek_char().map_or(false, |c| c.is_ascii_digit()) {
            self.read_money_literal('$')
        } else {
            Some(Ok(self.single_char_token(TokenKind::Dollar)))
        }
    }

    /// Read a money literal starting with a currency symbol
    fn read_money_literal(&mut self, currency_symbol: char) -> Option<Result<Token, LexerError>> {
        let start_pos = self.position;
        let mut value = String::new();
        
        // Add currency symbol
        value.push(currency_symbol);
        self.advance(); // Skip currency symbol
        
        // Read the numeric part
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() || ch == '.' || ch == ',' {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Validate that we have a valid money format
        if self.is_valid_money_format(&value) {
            Some(Ok(Token::new(
                TokenKind::MoneyLiteral(value),
                Span::new(start_pos, self.position, self.source_id),
            )))
        } else {
            Some(Err(LexerError::InvalidMoney(start_pos)))
        }
    }

    /// Check if a string represents a valid money format
    fn is_valid_money_format(&self, value: &str) -> bool {
        if value.len() < 2 {
            return false;
        }
        
        // Should start with currency symbol
        let first_char = value.chars().next().unwrap();
        if !self.is_currency_symbol(first_char) {
            return false;
        }
        
        // Rest should be a valid number (allowing commas for thousands)
        let number_part = &value[1..];
        let cleaned = number_part.replace(',', "");
        
        // Should be a valid float
        cleaned.parse::<f64>().is_ok()
    }

    /// Check if a character is a currency symbol
    fn is_currency_symbol(&self, ch: char) -> bool {
        matches!(ch, '$' | '€' | '£' | '¥' | '₹' | '₽' | '₩' | '₪' | '₨' | '₡' | '₦' | '₵' | '₴' | '₸' | '₿')
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
            "switch" => TokenKind::Switch,
            "case" => TokenKind::Case,
            "default" => TokenKind::Default,
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
            "finally" => TokenKind::Finally,
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
            "elif" => TokenKind::ElseIf,
            "when" => TokenKind::When,
            "sync" => TokenKind::Sync,
            "protected" => TokenKind::Protected,
            "nil" => TokenKind::Nil,
            "undefined" => TokenKind::Undefined,
            "typeof" => TokenKind::Typeof,
            "sizeof" => TokenKind::Sizeof,
            "performance" => TokenKind::Performance,
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