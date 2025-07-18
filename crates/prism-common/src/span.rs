//! Source location and span utilities

use std::fmt;
use crate::SourceId;

/// A position in source code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Position {
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based)  
    pub column: u32,
    /// Byte offset from start of file (0-based)
    pub offset: u32,
}

impl Position {
    /// Create a new position
    pub fn new(line: u32, column: u32, offset: u32) -> Self {
        Self { line, column, offset }
    }

    /// Create a position at the start of a file
    pub fn start() -> Self {
        Self::new(1, 1, 0)
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Position {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.offset.cmp(&other.offset)
    }
}

/// A span of source code from start to end position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Span {
    /// Start position (inclusive)
    pub start: Position,
    /// End position (exclusive)
    pub end: Position,
    /// Source file identifier
    pub source_id: SourceId,
}

impl Span {
    /// Create a new span
    pub fn new(start: Position, end: Position, source_id: SourceId) -> Self {
        Self { start, end, source_id }
    }

    /// Create a span from a single position
    pub fn from_position(pos: Position, source_id: SourceId) -> Self {
        Self::new(pos, pos, source_id)
    }

    /// Create a dummy span for generated code
    pub fn dummy() -> Self {
        Self::new(Position::start(), Position::start(), SourceId::new(0))
    }

    /// Check if this span contains another span
    pub fn contains(&self, other: &Span) -> bool {
        self.source_id == other.source_id
            && self.start <= other.start
            && other.end <= self.end
    }

    /// Combine two spans into one that covers both
    pub fn combine(&self, other: &Span) -> Option<Span> {
        if self.source_id != other.source_id {
            return None;
        }

        let start = std::cmp::min(self.start, other.start);
        let end = std::cmp::max(self.end, other.end);
        Some(Span::new(start, end, self.source_id))
    }

    /// Get the length of this span in bytes
    pub fn len(&self) -> u32 {
        self.end.offset.saturating_sub(self.start.offset)
    }

    /// Check if this span is empty
    pub fn is_empty(&self) -> bool {
        self.start.offset >= self.end.offset
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start.line == self.end.line {
            write!(f, "{}:{}:{}-{}", self.source_id, self.start.line, self.start.column, self.end.column)
        } else {
            write!(f, "{}:{}:{}-{}:{}", self.source_id, self.start.line, self.start.column, self.end.line, self.end.column)
        }
    }
}

/// Trait for types that can be spanned
pub trait Spanned {
    /// Get the span of this item
    fn span(&self) -> Span;
}

/// A spanned value - combines a value with its source location
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpannedValue<T> {
    /// The actual value
    pub value: T,
    /// Source location of this value
    pub span: Span,
}

impl<T> SpannedValue<T> {
    /// Create a new spanned value
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    /// Map the value while preserving the span
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> SpannedValue<U> {
        SpannedValue::new(f(self.value), self.span)
    }

    /// Get a reference to the value
    pub fn as_ref(&self) -> &T {
        &self.value
    }

    /// Get the span
    pub fn get_span(&self) -> Span {
        self.span
    }
}

impl<T> Spanned for SpannedValue<T> {
    fn span(&self) -> Span {
        self.span
    }
}

impl<T: fmt::Display> fmt::Display for SpannedValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
} 