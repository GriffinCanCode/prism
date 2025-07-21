//! Scope Data Structures
//!
//! This module defines the core data structures for representing scopes,
//! including the ScopeId type and ScopeData structure that holds all
//! information about a particular scope.
//!
//! **Conceptual Responsibility**: Scope data representation and storage
//! **What it does**: Define scope data structures, scope identification, data access
//! **What it doesn't do**: Scope hierarchy management, symbol resolution, visibility rules

use crate::scope::{ScopeKind, ScopeMetadata, EffectBoundary, CapabilityBoundary};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Unique identifier for scopes within the compiler
/// 
/// Uses a simple u32 for efficiency and easy debugging.
/// Scopes are numbered sequentially starting from 1 (0 reserved for "no scope").
pub type ScopeId = u32;

/// Reserved scope ID for "no scope" or invalid scope references
pub const NO_SCOPE: ScopeId = 0;

/// Comprehensive scope data structure
/// 
/// Contains all information about a scope including its kind, location,
/// contained symbols, imports/exports, and metadata for AI comprehension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeData {
    /// Unique scope identifier
    pub id: ScopeId,
    
    /// Scope classification and semantic meaning
    pub kind: ScopeKind,
    
    /// Source location where this scope is defined
    pub location: Span,
    
    /// Associated AST node (if any)
    pub ast_node: Option<NodeId>,
    
    /// Symbols defined directly in this scope
    /// 
    /// Note: This contains references to symbols, not the symbols themselves.
    /// Actual symbol data is stored in the symbol table.
    pub symbols: HashSet<Symbol>,
    
    /// Imported symbols (name -> original symbol)
    /// 
    /// Maps the local name to the imported symbol reference.
    /// Example: `import { foo as bar } from "module"` creates entry "bar" -> Symbol("foo")
    pub imports: HashMap<String, ImportedSymbol>,
    
    /// Exported symbols (name -> symbol)
    /// 
    /// Maps the export name to the symbol being exported.
    /// Example: `export { foo as bar }` creates entry "bar" -> Symbol("foo")
    pub exports: HashMap<String, ExportedSymbol>,
    
    /// Scope metadata for AI comprehension and tooling
    pub metadata: ScopeMetadata,
    
    /// Effect boundaries for this scope (PLD-003 integration)
    pub effect_boundaries: Vec<EffectBoundary>,
    
    /// Capability boundaries for this scope (PLD-003 integration)
    pub capability_boundaries: Vec<CapabilityBoundary>,
}

/// Information about an imported symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedSymbol {
    /// The original symbol being imported
    pub symbol: Symbol,
    
    /// The source module/scope where this symbol originates
    pub source_module: String,
    
    /// Optional alias for the imported symbol
    pub alias: Option<String>,
    
    /// Location of the import statement
    pub import_location: Span,
    
    /// Whether this is a re-export
    pub is_reexport: bool,
}

/// Information about an exported symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedSymbol {
    /// The symbol being exported
    pub symbol: Symbol,
    
    /// Optional export name (different from symbol name)
    pub export_name: Option<String>,
    
    /// Location of the export statement
    pub export_location: Span,
    
    /// Export visibility level
    pub visibility: ExportVisibility,
}

/// Visibility level for exported symbols
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportVisibility {
    /// Exported to all importers
    Public,
    /// Exported only to specific modules
    Restricted { allowed_modules: Vec<String> },
    /// Re-exported from another module
    ReExport { original_module: String },
}

impl ScopeData {
    /// Create a new scope with the given ID and kind
    pub fn new(id: ScopeId, kind: ScopeKind, location: Span) -> Self {
        Self {
            id,
            kind,
            location,
            ast_node: None,
            symbols: HashSet::new(),
            imports: HashMap::new(),
            exports: HashMap::new(),
            metadata: ScopeMetadata::default(),
            effect_boundaries: Vec::new(),
            capability_boundaries: Vec::new(),
        }
    }
    
    /// Check if this scope contains a specific symbol
    pub fn contains_symbol(&self, symbol: &Symbol) -> bool {
        self.symbols.contains(symbol)
    }
    
    /// Get the number of symbols directly defined in this scope
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }
    
    /// Get the number of imports in this scope
    pub fn import_count(&self) -> usize {
        self.imports.len()
    }
    
    /// Get the number of exports from this scope
    pub fn export_count(&self) -> usize {
        self.exports.len()
    }
    
    /// Check if a name is imported in this scope
    pub fn has_import(&self, name: &str) -> bool {
        self.imports.contains_key(name)
    }
    
    /// Check if a name is exported from this scope
    pub fn has_export(&self, name: &str) -> bool {
        self.exports.contains_key(name)
    }
    
    /// Get an imported symbol by name
    pub fn get_import(&self, name: &str) -> Option<&ImportedSymbol> {
        self.imports.get(name)
    }
    
    /// Get an exported symbol by name
    pub fn get_export(&self, name: &str) -> Option<&ExportedSymbol> {
        self.exports.get(name)
    }
    
    /// Add a symbol to this scope
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.symbols.insert(symbol);
    }
    
    /// Remove a symbol from this scope
    pub fn remove_symbol(&mut self, symbol: &Symbol) -> bool {
        self.symbols.remove(symbol)
    }
    
    /// Add an import to this scope
    pub fn add_import(&mut self, name: String, imported: ImportedSymbol) {
        self.imports.insert(name, imported);
    }
    
    /// Add an export to this scope
    pub fn add_export(&mut self, name: String, exported: ExportedSymbol) {
        self.exports.insert(name, exported);
    }
    
    /// Get all symbol references in this scope (direct + imported)
    pub fn all_accessible_symbols(&self) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        
        // Add direct symbols
        symbols.extend(self.symbols.iter().cloned());
        
        // Add imported symbols
        for imported in self.imports.values() {
            symbols.push(imported.symbol);
        }
        
        symbols
    }
    
    /// Check if this scope is a module scope
    pub fn is_module(&self) -> bool {
        matches!(self.kind, ScopeKind::Module { .. })
    }
    
    /// Check if this scope is a function scope
    pub fn is_function(&self) -> bool {
        matches!(self.kind, ScopeKind::Function { .. })
    }
    
    /// Check if this scope is a block scope
    pub fn is_block(&self) -> bool {
        matches!(self.kind, ScopeKind::Block { .. })
    }
    
    /// Get a human-readable description of this scope
    pub fn description(&self) -> String {
        match &self.kind {
            ScopeKind::Global => "Global scope".to_string(),
            ScopeKind::Module { module_name, .. } => format!("Module '{}'", module_name),
            ScopeKind::Function { function_name, .. } => format!("Function '{}'", function_name),
            ScopeKind::Block { block_type } => format!("Block ({:?})", block_type),
            ScopeKind::Type { type_name, .. } => format!("Type '{}'", type_name),
            ScopeKind::Section { section_type, .. } => format!("Section ({:?})", section_type),
            ScopeKind::ControlFlow { control_type } => format!("Control flow ({:?})", control_type),
            ScopeKind::Lambda { .. } => "Lambda scope".to_string(),
        }
    }
}

impl ImportedSymbol {
    /// Create a new imported symbol
    pub fn new(
        symbol: Symbol,
        source_module: String,
        import_location: Span,
    ) -> Self {
        Self {
            symbol,
            source_module,
            alias: None,
            import_location,
            is_reexport: false,
        }
    }
    
    /// Create a new imported symbol with an alias
    pub fn with_alias(
        symbol: Symbol,
        source_module: String,
        alias: String,
        import_location: Span,
    ) -> Self {
        Self {
            symbol,
            source_module,
            alias: Some(alias),
            import_location,
            is_reexport: false,
        }
    }
    
    /// Get the effective name of this import (alias or original name)
    pub fn effective_name(&self) -> String {
        self.alias.clone().unwrap_or_else(|| {
            // This is a simplified approach - in reality we'd need to resolve
            // the symbol to get its original name
            format!("symbol_{}", self.symbol.raw().to_usize())
        })
    }
}

impl ExportedSymbol {
    /// Create a new exported symbol
    pub fn new(
        symbol: Symbol,
        export_location: Span,
        visibility: ExportVisibility,
    ) -> Self {
        Self {
            symbol,
            export_name: None,
            export_location,
            visibility,
        }
    }
    
    /// Create a new exported symbol with a custom export name
    pub fn with_name(
        symbol: Symbol,
        export_name: String,
        export_location: Span,
        visibility: ExportVisibility,
    ) -> Self {
        Self {
            symbol,
            export_name: Some(export_name),
            export_location,
            visibility,
        }
    }
    
    /// Get the effective export name
    pub fn effective_name(&self) -> String {
        self.export_name.clone().unwrap_or_else(|| {
            // This is a simplified approach - in reality we'd need to resolve
            // the symbol to get its original name
            format!("symbol_{}", self.symbol.raw().to_usize())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::ScopeKind;
    
    #[test]
    fn test_scope_data_creation() {
        let scope = ScopeData::new(
            1,
            ScopeKind::Global,
            Span::new(0, 10),
        );
        
        assert_eq!(scope.id, 1);
        assert!(matches!(scope.kind, ScopeKind::Global));
        assert_eq!(scope.symbol_count(), 0);
        assert_eq!(scope.import_count(), 0);
        assert_eq!(scope.export_count(), 0);
    }
    
    #[test]
    fn test_symbol_operations() {
        let mut scope = ScopeData::new(
            1,
            ScopeKind::Global,
            Span::new(0, 10),
        );
        
        let symbol = Symbol::intern("test");
        
        assert!(!scope.contains_symbol(&symbol));
        
        scope.add_symbol(symbol);
        assert!(scope.contains_symbol(&symbol));
        assert_eq!(scope.symbol_count(), 1);
        
        let removed = scope.remove_symbol(&symbol);
        assert!(removed);
        assert!(!scope.contains_symbol(&symbol));
        assert_eq!(scope.symbol_count(), 0);
    }
    
    #[test]
    fn test_import_operations() {
        let mut scope = ScopeData::new(
            1,
            ScopeKind::Global,
            Span::new(0, 10),
        );
        
        let symbol = Symbol::intern("imported_func");
        let imported = ImportedSymbol::new(
            symbol,
            "other_module".to_string(),
            Span::new(5, 15),
        );
        
        scope.add_import("func".to_string(), imported);
        
        assert!(scope.has_import("func"));
        assert_eq!(scope.import_count(), 1);
        
        let import = scope.get_import("func").unwrap();
        assert_eq!(import.symbol, symbol);
        assert_eq!(import.source_module, "other_module");
    }
} 