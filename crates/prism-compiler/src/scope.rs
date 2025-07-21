//! Scope Management - Hierarchical Scope Tree Management
//!
//! This module embodies the single concept of "Scope Management".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: managing hierarchical scopes and their relationships.
//!
//! **Conceptual Responsibility**: Scope hierarchy and relationship management
//! **What it does**: scope creation, hierarchy tracking, scope queries, parent-child relationships
//! **What it doesn't do**: symbol resolution, symbol storage, semantic analysis (delegates to specialized modules)

use crate::error::{CompilerError, CompilerResult};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Type alias for scope identifiers
pub type ScopeId = u32;

/// Hierarchical scope tree manager
/// 
/// Manages the tree structure of scopes and their relationships,
/// integrating with existing infrastructure while maintaining conceptual cohesion
#[derive(Debug)]
pub struct ScopeTree {
    /// Scope data storage
    scopes: Arc<RwLock<HashMap<ScopeId, ScopeData>>>,
    /// Parent-child relationships
    parent_map: Arc<RwLock<HashMap<ScopeId, ScopeId>>>,
    /// Children relationships  
    children_map: Arc<RwLock<HashMap<ScopeId, Vec<ScopeId>>>>,
    /// Root scope identifier
    root_scope: Option<ScopeId>,
    /// Next scope ID counter
    next_id: Arc<RwLock<ScopeId>>,
    /// Configuration
    config: ScopeTreeConfig,
}

/// Configuration for scope tree behavior
#[derive(Debug, Clone)]
pub struct ScopeTreeConfig {
    /// Enable scope caching
    pub enable_caching: bool,
    /// Enable scope metadata
    pub enable_metadata: bool,
    /// Maximum scope depth (for cycle detection)
    pub max_scope_depth: usize,
}

impl Default for ScopeTreeConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_metadata: true,
            max_scope_depth: 100,
        }
    }
}

/// Comprehensive scope data with semantic context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeData {
    /// Unique scope identifier
    pub id: ScopeId,
    /// Scope kind and semantic meaning
    pub kind: ScopeKind,
    /// Source location of scope definition
    pub location: Span,
    /// Associated AST node
    pub ast_node: Option<NodeId>,
    /// Symbols defined directly in this scope (references to symbol table)
    pub symbols: HashSet<Symbol>,
    /// Imported symbols (name -> original symbol)
    pub imports: HashMap<String, Symbol>,
    /// Exported symbols (name -> symbol)
    pub exports: HashMap<String, Symbol>,
    /// Scope metadata
    pub metadata: ScopeMetadata,
}

/// Scope classification with semantic meaning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScopeKind {
    /// Global/root scope
    Global,
    /// Module scope
    Module {
        module_name: String,
        sections: Vec<String>,
        capabilities: Vec<String>,
    },
    /// Section within a module (PLD-002 integration)
    Section {
        section_type: SectionType,
        parent_module: String,
    },
    /// Function scope
    Function {
        function_name: String,
        parameters: Vec<String>,
        is_async: bool,
    },
    /// Block scope (within functions, control structures)
    Block {
        block_type: BlockType,
    },
    /// Type definition scope
    Type {
        type_name: String,
        type_category: String,
    },
    /// Control flow scope (if, while, for, match)
    ControlFlow {
        control_type: ControlFlowType,
    },
    /// Lambda/closure scope
    Lambda {
        captures: Vec<String>,
    },
}

/// Module section types (PLD-002 integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    Config,
    Types,
    Errors,
    Internal,
    Interface,
    Events,
    Lifecycle,
    Tests,
    Examples,
    Performance,
    Custom(String),
}

/// Block type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockType {
    Regular,
    Unsafe { justification: String },
    Effect { effects: Vec<String> },
    Performance { capabilities: Vec<String> },
}

/// Control flow type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFlowType {
    If,
    While,
    For,
    Match,
    Try,
    Async,
}

/// Scope metadata for additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeMetadata {
    /// AI-readable scope context
    pub ai_context: Option<String>,
    /// Business responsibility of this scope
    pub responsibility: Option<String>,
    /// Documentation for this scope
    pub documentation: Option<String>,
    /// Performance characteristics
    pub performance_notes: Vec<String>,
    /// Security implications
    pub security_notes: Vec<String>,
}

impl Default for ScopeMetadata {
    fn default() -> Self {
        Self {
            ai_context: None,
            responsibility: None,
            documentation: None,
            performance_notes: Vec::new(),
            security_notes: Vec::new(),
        }
    }
}

impl ScopeTree {
    /// Create a new scope tree
    pub fn new() -> CompilerResult<Self> {
        Ok(Self {
            scopes: Arc::new(RwLock::new(HashMap::new())),
            parent_map: Arc::new(RwLock::new(HashMap::new())),
            children_map: Arc::new(RwLock::new(HashMap::new())),
            root_scope: None,
            next_id: Arc::new(RwLock::new(1)),
            config: ScopeTreeConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ScopeTreeConfig) -> CompilerResult<Self> {
        Ok(Self {
            scopes: Arc::new(RwLock::new(HashMap::new())),
            parent_map: Arc::new(RwLock::new(HashMap::new())),
            children_map: Arc::new(RwLock::new(HashMap::new())),
            root_scope: None,
            next_id: Arc::new(RwLock::new(1)),
            config,
        })
    }

    /// Create a new scope with optional parent
    pub fn create_scope(&mut self, kind: ScopeKind, location: Span, parent: Option<ScopeId>) -> CompilerResult<ScopeId> {
        let id = {
            let mut next_id = self.next_id.write().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let scope_data = ScopeData {
            id,
            kind,
            location,
            ast_node: None,
            symbols: HashSet::new(),
            imports: HashMap::new(),
            exports: HashMap::new(),
            metadata: ScopeMetadata::default(),
        };

        // Store scope data
        {
            let mut scopes = self.scopes.write().unwrap();
            scopes.insert(id, scope_data);
        }

        // Handle parent-child relationships
        if let Some(parent_id) = parent {
            // Validate parent exists
            {
                let scopes = self.scopes.read().unwrap();
                if !scopes.contains_key(&parent_id) {
                    return Err(CompilerError::InvalidInput {
                        message: format!("Parent scope {} does not exist", parent_id),
                    });
                }
            }

            // Set up parent-child relationship
            {
                let mut parent_map = self.parent_map.write().unwrap();
                parent_map.insert(id, parent_id);
            }

            {
                let mut children_map = self.children_map.write().unwrap();
                children_map.entry(parent_id).or_insert_with(Vec::new).push(id);
            }

            // Check for excessive depth
            if self.get_scope_depth(id)? > self.config.max_scope_depth {
                return Err(CompilerError::InvalidInput {
                    message: format!("Scope depth exceeds maximum of {}", self.config.max_scope_depth),
                });
            }
        } else {
            // This is a root scope
            if self.root_scope.is_none() {
                self.root_scope = Some(id);
            }
        }

        Ok(id)
    }

    /// Get scope data by ID
    pub fn get_scope(&self, scope_id: ScopeId) -> Option<ScopeData> {
        let scopes = self.scopes.read().unwrap();
        scopes.get(&scope_id).cloned()
    }

    /// Get parent scope ID
    pub fn get_parent(&self, scope_id: ScopeId) -> Option<ScopeId> {
        let parent_map = self.parent_map.read().unwrap();
        parent_map.get(&scope_id).copied()
    }

    /// Get child scope IDs
    pub fn get_children(&self, scope_id: ScopeId) -> Vec<ScopeId> {
        let children_map = self.children_map.read().unwrap();
        children_map.get(&scope_id).cloned().unwrap_or_default()
    }

    /// Get scope chain from scope to root
    pub fn get_scope_chain(&self, scope_id: ScopeId) -> Vec<ScopeId> {
        let mut chain = Vec::new();
        let mut current = Some(scope_id);

        while let Some(id) = current {
            chain.push(id);
            current = self.get_parent(id);
        }

        chain
    }

    /// Get scope depth (distance from root)
    pub fn get_scope_depth(&self, scope_id: ScopeId) -> CompilerResult<usize> {
        let chain = self.get_scope_chain(scope_id);
        Ok(chain.len().saturating_sub(1))
    }

    /// Check if scope is ancestor of another scope
    pub fn is_ancestor(&self, ancestor_id: ScopeId, descendant_id: ScopeId) -> bool {
        let chain = self.get_scope_chain(descendant_id);
        chain.contains(&ancestor_id)
    }

    /// Find common ancestor of two scopes
    pub fn find_common_ancestor(&self, scope1: ScopeId, scope2: ScopeId) -> Option<ScopeId> {
        let chain1 = self.get_scope_chain(scope1);
        let chain2 = self.get_scope_chain(scope2);

        // Find first common scope (working backwards from root)
        for scope1_id in chain1.iter().rev() {
            if chain2.contains(scope1_id) {
                return Some(*scope1_id);
            }
        }

        None
    }

    /// Add symbol to scope
    pub fn add_symbol_to_scope(&self, scope_id: ScopeId, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            scope_data.symbols.insert(symbol);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }

    /// Add import to scope
    pub fn add_import_to_scope(&self, scope_id: ScopeId, name: String, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            scope_data.imports.insert(name, symbol);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }

    /// Add export to scope
    pub fn add_export_to_scope(&self, scope_id: ScopeId, name: String, symbol: Symbol) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            scope_data.exports.insert(name, symbol);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }

    /// Update scope metadata
    pub fn update_scope_metadata(&self, scope_id: ScopeId, updater: impl FnOnce(&mut ScopeMetadata)) -> CompilerResult<()> {
        let mut scopes = self.scopes.write().unwrap();
        if let Some(scope_data) = scopes.get_mut(&scope_id) {
            updater(&mut scope_data.metadata);
            Ok(())
        } else {
            Err(CompilerError::InvalidInput {
                message: format!("Scope {} does not exist", scope_id),
            })
        }
    }

    /// Get scopes by kind
    pub fn get_scopes_by_kind(&self, kind_filter: impl Fn(&ScopeKind) -> bool) -> Vec<ScopeData> {
        let scopes = self.scopes.read().unwrap();
        scopes.values()
            .filter(|scope| kind_filter(&scope.kind))
            .cloned()
            .collect()
    }

    /// Get root scope
    pub fn get_root_scope(&self) -> Option<ScopeId> {
        self.root_scope
    }

    /// Get all scopes
    pub fn get_all_scopes(&self) -> Vec<ScopeData> {
        let scopes = self.scopes.read().unwrap();
        scopes.values().cloned().collect()
    }

    /// Get scope tree statistics
    pub fn stats(&self) -> ScopeTreeStats {
        let scopes = self.scopes.read().unwrap();
        let children_map = self.children_map.read().unwrap();
        
        let mut kind_counts = HashMap::new();
        let mut max_depth = 0;
        let mut total_symbols = 0;

        for scope in scopes.values() {
            let kind_name = match &scope.kind {
                ScopeKind::Global => "Global",
                ScopeKind::Module { .. } => "Module",
                ScopeKind::Section { .. } => "Section",
                ScopeKind::Function { .. } => "Function",
                ScopeKind::Block { .. } => "Block",
                ScopeKind::Type { .. } => "Type",
                ScopeKind::ControlFlow { .. } => "ControlFlow",
                ScopeKind::Lambda { .. } => "Lambda",
            };
            *kind_counts.entry(kind_name.to_string()).or_insert(0) += 1;

            total_symbols += scope.symbols.len();

            // Calculate depth for this scope
            let depth = self.get_scope_depth(scope.id).unwrap_or(0);
            max_depth = max_depth.max(depth);
        }

        ScopeTreeStats {
            total_scopes: scopes.len(),
            scopes_by_kind: kind_counts,
            max_depth,
            total_symbols,
            average_symbols_per_scope: if scopes.is_empty() { 0.0 } else { total_symbols as f64 / scopes.len() as f64 },
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ScopeTreeConfig {
        &self.config
    }
}

impl Default for ScopeTree {
    fn default() -> Self {
        Self::new().expect("Failed to create default ScopeTree")
    }
}

/// Scope tree statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeTreeStats {
    /// Total number of scopes
    pub total_scopes: usize,
    /// Count of scopes by kind
    pub scopes_by_kind: HashMap<String, usize>,
    /// Maximum scope depth
    pub max_depth: usize,
    /// Total symbols across all scopes
    pub total_symbols: usize,
    /// Average symbols per scope
    pub average_symbols_per_scope: f64,
} 