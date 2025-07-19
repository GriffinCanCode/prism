//! Symbol table and string interning utilities

use std::fmt;
use std::sync::{Mutex, OnceLock};
use rustc_hash::FxHashMap;
use string_interner::{DefaultSymbol, StringInterner, backend::StringBackend, Symbol as SymbolTrait};

/// A symbol representing an interned string
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(DefaultSymbol);

/// Global symbol table
static GLOBAL_SYMBOL_TABLE: OnceLock<Mutex<SymbolTable>> = OnceLock::new();

impl Symbol {
    /// Create a new symbol from a raw value (used internally)
    pub(crate) fn new(sym: DefaultSymbol) -> Self {
        Self(sym)
    }

    /// Get the raw symbol value
    pub fn raw(self) -> DefaultSymbol {
        self.0
    }

    /// Intern a string using the global symbol table
    pub fn intern(s: &str) -> Self {
        let table = GLOBAL_SYMBOL_TABLE.get_or_init(|| Mutex::new(SymbolTable::new()));
        let mut table = table.lock().unwrap();
        table.intern(s)
    }

    /// Resolve a symbol to its string representation
    pub fn resolve(self) -> Option<String> {
        let table = GLOBAL_SYMBOL_TABLE.get_or_init(|| Mutex::new(SymbolTable::new()));
        let table = table.lock().unwrap();
        table.resolve(self).map(|s| s.to_string())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sym:{:?}", self.0)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Symbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize the symbol as its string representation
        match self.resolve() {
            Some(string) => serializer.serialize_str(&string),
            None => serializer.serialize_str(&format!("unknown_symbol_{}", self.0.to_usize())),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Symbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        Ok(Symbol::intern(&string))
    }
}

/// A symbol table for interning strings and managing identifiers
#[derive(Debug)]
pub struct SymbolTable {
    interner: StringInterner<StringBackend<DefaultSymbol>>,
    keywords: FxHashMap<Symbol, Keyword>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolTable {
    /// Create a new symbol table
    pub fn new() -> Self {
        let mut table = Self {
            interner: StringInterner::default(),
            keywords: FxHashMap::default(),
        };
        
        // Intern all keywords
        table.intern_keywords();
        table
    }

    /// Intern a string and return its symbol
    pub fn intern(&mut self, string: &str) -> Symbol {
        Symbol::new(self.interner.get_or_intern(string))
    }

    /// Get the string for a symbol
    pub fn resolve(&self, symbol: Symbol) -> Option<&str> {
        self.interner.resolve(symbol.0)
    }

    /// Check if a symbol is a keyword
    pub fn is_keyword(&self, symbol: Symbol) -> bool {
        self.keywords.contains_key(&symbol)
    }

    /// Get the keyword type for a symbol, if it is a keyword
    pub fn keyword(&self, symbol: Symbol) -> Option<Keyword> {
        self.keywords.get(&symbol).copied()
    }

    /// Intern all keywords
    fn intern_keywords(&mut self) {
        let keywords = [
            ("module", Keyword::Module),
            ("section", Keyword::Section),
            ("capability", Keyword::Capability),
            ("function", Keyword::Function),
            ("fn", Keyword::Fn),
            ("type", Keyword::Type),
            ("let", Keyword::Let),
            ("const", Keyword::Const),
            ("var", Keyword::Var),
            ("if", Keyword::If),
            ("else", Keyword::Else),
            ("while", Keyword::While),
            ("for", Keyword::For),
            ("match", Keyword::Match),
            ("case", Keyword::Case),
            ("return", Keyword::Return),
            ("break", Keyword::Break),
            ("continue", Keyword::Continue),
            ("true", Keyword::True),
            ("false", Keyword::False),
            ("null", Keyword::Null),
            ("and", Keyword::And),
            ("or", Keyword::Or),
            ("not", Keyword::Not),
            ("in", Keyword::In),
            ("as", Keyword::As),
            ("where", Keyword::Where),
            ("with", Keyword::With),
            ("requires", Keyword::Requires),
            ("ensures", Keyword::Ensures),
            ("invariant", Keyword::Invariant),
            ("effects", Keyword::Effects),
            ("async", Keyword::Async),
            ("await", Keyword::Await),
            ("try", Keyword::Try),
            ("catch", Keyword::Catch),
            ("throw", Keyword::Throw),
            ("import", Keyword::Import),
            ("export", Keyword::Export),
            ("public", Keyword::Public),
            ("private", Keyword::Private),
            ("internal", Keyword::Internal),
            ("config", Keyword::Config),
            ("types", Keyword::Types),
            ("errors", Keyword::Errors),
            ("interface", Keyword::Interface),
            ("events", Keyword::Events),
            ("lifecycle", Keyword::Lifecycle),
            ("tests", Keyword::Tests),
            ("examples", Keyword::Examples),
        ];

        for (keyword_str, keyword) in keywords {
            let symbol = self.intern(keyword_str);
            self.keywords.insert(symbol, keyword);
        }
    }
}

/// Prism language keywords
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Keyword {
    // Structure keywords
    Module,
    Section,
    Capability,
    Function,
    Fn,
    Type,
    
    // Variable keywords
    Let,
    Const,
    Var,
    
    // Control flow
    If,
    Else,
    While,
    For,
    Match,
    Case,
    Return,
    Break,
    Continue,
    
    // Literals
    True,
    False,
    Null,
    
    // Logical operators (word forms)
    And,
    Or,
    Not,
    
    // Misc
    In,
    As,
    Where,
    With,
    
    // Semantic types and contracts
    Requires,
    Ensures,
    Invariant,
    Effects,
    
    // Async/await
    Async,
    Await,
    
    // Error handling
    Try,
    Catch,
    Throw,
    
    // Module system
    Import,
    Export,
    
    // Visibility
    Public,
    Private,
    Internal,
    
    // Section types
    Config,
    Types,
    Errors,
    Interface,
    Events,
    Lifecycle,
    Tests,
    Examples,
}

impl Keyword {
    /// Get the string representation of a keyword
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Section => "section",
            Self::Capability => "capability",
            Self::Function => "function",
            Self::Fn => "fn",
            Self::Type => "type",
            Self::Let => "let",
            Self::Const => "const",
            Self::Var => "var",
            Self::If => "if",
            Self::Else => "else",
            Self::While => "while",
            Self::For => "for",
            Self::Match => "match",
            Self::Case => "case",
            Self::Return => "return",
            Self::Break => "break",
            Self::Continue => "continue",
            Self::True => "true",
            Self::False => "false",
            Self::Null => "null",
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
            Self::In => "in",
            Self::As => "as",
            Self::Where => "where",
            Self::With => "with",
            Self::Requires => "requires",
            Self::Ensures => "ensures",
            Self::Invariant => "invariant",
            Self::Effects => "effects",
            Self::Async => "async",
            Self::Await => "await",
            Self::Try => "try",
            Self::Catch => "catch",
            Self::Throw => "throw",
            Self::Import => "import",
            Self::Export => "export",
            Self::Public => "public",
            Self::Private => "private",
            Self::Internal => "internal",
            Self::Config => "config",
            Self::Types => "types",
            Self::Errors => "errors",
            Self::Interface => "interface",
            Self::Events => "events",
            Self::Lifecycle => "lifecycle",
            Self::Tests => "tests",
            Self::Examples => "examples",
        }
    }
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
} 