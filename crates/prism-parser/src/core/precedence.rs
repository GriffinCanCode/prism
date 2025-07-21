//! Operator precedence and associativity for expression parsing
//!
//! This module implements a Pratt parser approach to handle operator
//! precedence and associativity correctly in Prism expressions.

use prism_lexer::TokenKind;

/// Precedence levels for operators (higher numbers = higher precedence)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precedence {
    /// No precedence (used for error recovery)
    None = 0,
    /// Assignment operators (=, +=, -=, etc.)
    Assignment = 1,
    /// Logical OR (||)
    Or = 2,
    /// Logical AND (&&)
    And = 3,
    /// Bitwise OR (|)
    BitOr = 4,
    /// Bitwise XOR (^)
    BitXor = 5,
    /// Bitwise AND (&)
    BitAnd = 6,
    /// Equality (==, !=, ===, ~=, ≈)
    Equality = 7,
    /// Comparison (<, >, <=, >=)
    Comparison = 8,
    /// Bitwise shift (<<, >>)
    Shift = 9,
    /// Addition and subtraction (+, -)
    Term = 10,
    /// Multiplication, division, modulo (*, /, %)
    Factor = 11,
    /// Unary operators (!, -, +)
    Unary = 12,
    /// Exponentiation (**)
    Power = 13,
    /// Function calls, member access, indexing
    Call = 14,
    /// Primary expressions (literals, identifiers, parentheses)
    Primary = 15,
}

/// Associativity of operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Associativity {
    /// Left-associative (a + b + c = (a + b) + c)
    Left,
    /// Right-associative (a = b = c = a = (b = c))
    Right,
    /// Non-associative (comparisons cannot be chained)
    None,
}

/// Get the precedence of a token when used as an infix operator
pub fn infix_precedence(token: &TokenKind) -> Option<Precedence> {
    match token {
        // Assignment operators
        TokenKind::Assign
        | TokenKind::PlusAssign
        | TokenKind::MinusAssign
        | TokenKind::StarAssign
        | TokenKind::SlashAssign
        | TokenKind::WalrusOperator => Some(Precedence::Assignment),
        
        // Logical operators
        TokenKind::OrOr => Some(Precedence::Or),
        TokenKind::AndAnd => Some(Precedence::And),

        // Bitwise operators
        TokenKind::Pipe => Some(Precedence::BitOr),
        TokenKind::Caret => Some(Precedence::BitXor),
        TokenKind::Ampersand => Some(Precedence::BitAnd),

        // Equality operators
        TokenKind::Equal
        | TokenKind::NotEqual => Some(Precedence::Equality),

        // Comparison operators
        TokenKind::Less
        | TokenKind::LessEqual
        | TokenKind::Greater
        | TokenKind::GreaterEqual => Some(Precedence::Comparison),

        // Shift operators
        // Shift operators not yet defined in lexer
        // TokenKind::LeftShift | TokenKind::RightShift => Some(Precedence::Shift),

        // Arithmetic operators
        TokenKind::Plus | TokenKind::Minus => Some(Precedence::Term),
        TokenKind::Star | TokenKind::Slash | TokenKind::Percent | TokenKind::IntegerDivision | TokenKind::At => Some(Precedence::Factor),
        TokenKind::Power => Some(Precedence::Power),

        // Call and access operators
        TokenKind::LeftParen | TokenKind::Dot | TokenKind::LeftBracket => Some(Precedence::Call),

        // Range operators
        // Range operator not yet defined in lexer
        // TokenKind::DotDot => Some(Precedence::Comparison),

        // Type operators
        TokenKind::As => Some(Precedence::Comparison),

        _ => None,
    }
}

/// Get the precedence of a token when used as a prefix operator
pub fn prefix_precedence(token: &TokenKind) -> Option<Precedence> {
    match token {
        TokenKind::Bang | TokenKind::Minus | TokenKind::Plus => {
            Some(Precedence::Unary)
        }
        _ => None,
    }
}

/// Get the associativity of an operator
pub fn associativity(token: &TokenKind) -> Associativity {
    match token {
        // Right-associative operators
        TokenKind::Assign
        | TokenKind::PlusAssign
        | TokenKind::MinusAssign
        | TokenKind::StarAssign
        | TokenKind::SlashAssign
        | TokenKind::WalrusOperator  // Python walrus operator is right-associative
        | TokenKind::Power => Associativity::Right,

        // Non-associative operators
        TokenKind::Equal
        | TokenKind::NotEqual
        | TokenKind::Less
        | TokenKind::LessEqual
        | TokenKind::Greater
        | TokenKind::GreaterEqual
        | TokenKind::As => Associativity::None,

        // Left-associative operators (default)
        _ => Associativity::Left,
    }
}

/// Check if a token can start an expression
pub fn can_start_expression(token: &TokenKind) -> bool {
    matches!(
        token,
        // Literals
        TokenKind::IntegerLiteral(_)
            | TokenKind::FloatLiteral(_)
            | TokenKind::StringLiteral(_)
            | TokenKind::FStringStart(_)  // Python f-strings
            // Boolean literal handled by True/False tokens
            // | TokenKind::BooleanLiteral(_)
            | TokenKind::True
            | TokenKind::False
            | TokenKind::Null
            // Identifiers
            | TokenKind::Identifier(_)
            // Prefix operators
            | TokenKind::Bang
            | TokenKind::Minus
            | TokenKind::Plus
            // Grouping
            | TokenKind::LeftParen
            | TokenKind::LeftBracket
            | TokenKind::LeftBrace
            // Keywords that can start expressions
            | TokenKind::If
            | TokenKind::Match
            | TokenKind::Try
            | TokenKind::Async
            | TokenKind::Function
            | TokenKind::Fn
            // Special literals would go here when implemented
    )
}

/// Check if a token is a binary operator
pub fn is_binary_operator(token: &TokenKind) -> bool {
    infix_precedence(token).is_some()
}

/// Check if a token is a unary operator
pub fn is_unary_operator(token: &TokenKind) -> bool {
    prefix_precedence(token).is_some()
}

/// Get the minimum precedence for parsing
pub fn min_precedence() -> Precedence {
    Precedence::Assignment
}

/// Get the maximum precedence for parsing
pub fn max_precedence() -> Precedence {
    Precedence::Primary
}

/// Calculate the next precedence level for right-associative operators
pub fn next_precedence(current: Precedence, assoc: Associativity) -> Precedence {
    match assoc {
        Associativity::Left => {
            // For left-associative, we need higher precedence
            match current {
                Precedence::None => Precedence::Assignment,
                Precedence::Assignment => Precedence::Or,
                Precedence::Or => Precedence::And,
                Precedence::And => Precedence::BitOr,
                Precedence::BitOr => Precedence::BitXor,
                Precedence::BitXor => Precedence::BitAnd,
                Precedence::BitAnd => Precedence::Equality,
                Precedence::Equality => Precedence::Comparison,
                Precedence::Comparison => Precedence::Shift,
                Precedence::Shift => Precedence::Term,
                Precedence::Term => Precedence::Factor,
                Precedence::Factor => Precedence::Unary,
                Precedence::Unary => Precedence::Power,
                Precedence::Power => Precedence::Call,
                Precedence::Call => Precedence::Primary,
                Precedence::Primary => Precedence::Primary,
            }
        }
        Associativity::Right => {
            // For right-associative, we use the same precedence
            current
        }
        Associativity::None => {
            // For non-associative, we use higher precedence
            match current {
                Precedence::None => Precedence::Assignment,
                Precedence::Assignment => Precedence::Or,
                Precedence::Or => Precedence::And,
                Precedence::And => Precedence::BitOr,
                Precedence::BitOr => Precedence::BitXor,
                Precedence::BitXor => Precedence::BitAnd,
                Precedence::BitAnd => Precedence::Equality,
                Precedence::Equality => Precedence::Comparison,
                Precedence::Comparison => Precedence::Shift,
                Precedence::Shift => Precedence::Term,
                Precedence::Term => Precedence::Factor,
                Precedence::Factor => Precedence::Unary,
                Precedence::Unary => Precedence::Power,
                Precedence::Power => Precedence::Call,
                Precedence::Call => Precedence::Primary,
                Precedence::Primary => Precedence::Primary,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precedence_ordering() {
        assert!(Precedence::Assignment < Precedence::Or);
        assert!(Precedence::Or < Precedence::And);
        assert!(Precedence::And < Precedence::Equality);
        assert!(Precedence::Equality < Precedence::Comparison);
        assert!(Precedence::Comparison < Precedence::Term);
        assert!(Precedence::Term < Precedence::Factor);
        assert!(Precedence::Factor < Precedence::Unary);
        assert!(Precedence::Unary < Precedence::Power);
        assert!(Precedence::Power < Precedence::Call);
        assert!(Precedence::Call < Precedence::Primary);
    }

    #[test]
    fn test_arithmetic_precedence() {
        assert_eq!(infix_precedence(&TokenKind::Plus), Some(Precedence::Term));
        assert_eq!(infix_precedence(&TokenKind::Star), Some(Precedence::Factor));
        assert_eq!(infix_precedence(&TokenKind::Power), Some(Precedence::Power));
        
        assert!(Precedence::Factor > Precedence::Term);
        assert!(Precedence::Power > Precedence::Factor);
    }

    #[test]
    fn test_associativity() {
        assert_eq!(associativity(&TokenKind::Plus), Associativity::Left);
        assert_eq!(associativity(&TokenKind::Assign), Associativity::Right);
        assert_eq!(associativity(&TokenKind::Equal), Associativity::None);
    }

    #[test]
    fn test_expression_starters() {
        use prism_common::symbol::Symbol;
        
        assert!(can_start_expression(&TokenKind::IntegerLiteral(42)));
        assert!(can_start_expression(&TokenKind::Identifier("test".to_string())));
        assert!(can_start_expression(&TokenKind::LeftParen));
        assert!(can_start_expression(&TokenKind::Minus));
        assert!(!can_start_expression(&TokenKind::Plus));
        assert!(!can_start_expression(&TokenKind::Semicolon));
    }
} 