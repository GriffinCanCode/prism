//! Symbol Table - Core Symbol Storage and Management
//!
//! This module embodies the single concept of "Symbol Storage and Management".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: managing the storage, retrieval, and metadata of symbols.
//!
//! **Conceptual Responsibility**: Symbol storage and metadata management
//! **What it does**: symbol registration, storage, metadata attachment, interning integration
//! **What it doesn't do**: symbol resolution, scope management, semantic analysis (delegates to specialized modules)

use crate::error::{CompilerError, CompilerResult};
use crate::semantic::{SemanticDatabase, SymbolInfo, TypeInfo};
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_ast::{AstNode, Type, Effect};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

// Arena allocation support following existing patterns
use typed_arena::Arena;

/// Core symbol table for storing and managing symbols with semantic metadata
/// 
/// This integrates with existing infrastructure:
/// - Uses prism_common::symbol::Symbol for interning
/// - Leverages prism_compiler::semantic::SemanticDatabase for metadata
/// - Integrates with prism_ast types and effects
#[derive(Debug)]
pub struct SymbolTable {
    /// Symbol storage with comprehensive metadata
    symbols: Arc<RwLock<HashMap<Symbol, SymbolData>>>,
    /// Integration with semantic database
    semantic_db: Arc<SemanticDatabase>,
    /// Symbol table configuration
    config: SymbolTableConfig,
}

/// Configuration for symbol table behavior
#[derive(Debug, Clone)]
pub struct SymbolTableConfig {
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    /// Enable semantic type integration
    pub enable_semantic_types: bool,
    /// Enable effect tracking
    pub enable_effect_tracking: bool,
    /// Enable documentation validation
    pub enable_doc_validation: bool,
    /// Enable business context extraction
    pub enable_business_context: bool,
}

impl Default for SymbolTableConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_semantic_types: true,
            enable_effect_tracking: true,
            enable_doc_validation: true,
            enable_business_context: true,
        }
    }
}

/// Comprehensive symbol data with semantic information
/// 
/// Builds on existing prism_compiler::semantic::SymbolInfo but adds
/// symbol table specific metadata and integrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolData {
    /// Symbol identifier (leverages existing Symbol type)
    pub symbol: Symbol,
    /// Symbol name for display
    pub name: String,
    /// Symbol classification
    pub kind: SymbolKind,
    /// Source location
    pub location: Span,
    /// Visibility level
    pub visibility: SymbolVisibility,
    /// AST node reference
    pub ast_node: Option<NodeId>,
    /// Semantic type reference (integrates with semantic database)
    pub semantic_type: Option<String>, // Key into semantic database
    /// Effect information (integrates with effect system)
    pub effects: Vec<SymbolEffect>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Documentation metadata
    pub documentation: Option<String>,
    /// Business responsibility
    pub responsibility: Option<String>,
    /// AI-readable context
    pub ai_context: Option<String>,
}

/// Symbol classification following PLT-004 specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolKind {
    /// Module symbol
    Module {
        sections: Vec<String>,
        capabilities: Vec<String>,
    },
    /// Function symbol  
    Function {
        parameters: Vec<String>,
        return_type: Option<String>,
        is_async: bool,
    },
    /// Type definition symbol
    Type {
        type_category: TypeCategory,
        constraints: Vec<String>,
    },
    /// Variable symbol
    Variable {
        is_mutable: bool,
        type_hint: Option<String>,
    },
    /// Constant symbol
    Constant {
        value_type: Option<String>,
    },
    /// Parameter symbol
    Parameter {
        parameter_kind: ParameterKind,
        default_value: Option<String>,
    },
    /// Import symbol
    Import {
        source_module: String,
        alias: Option<String>,
    },
    /// Export symbol
    Export {
        export_name: Option<String>,
    },
}

/// Type category classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeCategory {
    Primitive,
    Semantic,
    Composite,
    Function,
    Effect,
}

/// Parameter kind classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterKind {
    Regular,
    Variadic,
    Named,
    Self_,
}

/// Symbol visibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolVisibility {
    Public,
    Module,
    Private,
    Internal,
}

/// Effect information for symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEffect {
    /// Effect name
    pub name: String,
    /// Effect category
    pub category: String,
    /// Effect parameters
    pub parameters: Vec<String>,
}

impl SymbolTable {
    /// Create a new symbol table with semantic database integration
    pub fn new(semantic_db: Arc<SemanticDatabase>) -> CompilerResult<Self> {
        Ok(Self {
            symbols: Arc::new(RwLock::new(HashMap::new())),
            semantic_db,
            config: SymbolTableConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(semantic_db: Arc<SemanticDatabase>, config: SymbolTableConfig) -> CompilerResult<Self> {
        Ok(Self {
            symbols: Arc::new(RwLock::new(HashMap::new())),
            semantic_db,
            config,
        })
    }

    /// Register a new symbol with comprehensive metadata
    pub fn register_symbol(&self, symbol_data: SymbolData) -> CompilerResult<()> {
        let symbol = symbol_data.symbol;
        
        // Store in symbol table
        {
            let mut symbols = self.symbols.write().unwrap();
            symbols.insert(symbol, symbol_data.clone());
        }

        // Integrate with semantic database if enabled
        if self.config.enable_semantic_types {
            let symbol_info = self.convert_to_symbol_info(&symbol_data)?;
            self.semantic_db.add_symbol(symbol, symbol_info)?;
        }

        Ok(())
    }

    /// Get symbol data by symbol
    pub fn get_symbol(&self, symbol: &Symbol) -> Option<SymbolData> {
        let symbols = self.symbols.read().unwrap();
        symbols.get(symbol).cloned()
    }

    /// Get all symbols matching a predicate
    pub fn find_symbols<F>(&self, predicate: F) -> Vec<SymbolData>
    where
        F: Fn(&SymbolData) -> bool,
    {
        let symbols = self.symbols.read().unwrap();
        symbols.values().filter(|data| predicate(data)).cloned().collect()
    }

    /// Update symbol metadata
    pub fn update_symbol(&self, symbol: &Symbol, updater: impl FnOnce(&mut SymbolData)) -> CompilerResult<()> {
        let mut symbols = self.symbols.write().unwrap();
        if let Some(symbol_data) = symbols.get_mut(symbol) {
            updater(symbol_data);
            Ok(())
        } else {
            Err(CompilerError::SymbolNotFound { 
                symbol: symbol.resolve().unwrap_or_else(|| format!("{:?}", symbol))
            })
        }
    }

    /// Get symbols by kind
    pub fn get_symbols_by_kind(&self, kind_filter: impl Fn(&SymbolKind) -> bool) -> Vec<SymbolData> {
        self.find_symbols(|data| kind_filter(&data.kind))
    }

    /// Get symbols by visibility
    pub fn get_symbols_by_visibility(&self, visibility: SymbolVisibility) -> Vec<SymbolData> {
        self.find_symbols(|data| std::mem::discriminant(&data.visibility) == std::mem::discriminant(&visibility))
    }

    /// Convert SymbolData to SemanticDatabase SymbolInfo for integration
    fn convert_to_symbol_info(&self, symbol_data: &SymbolData) -> CompilerResult<SymbolInfo> {
        use crate::semantic::{SymbolInfo, TypeInfo, TypeKind, Visibility, BusinessContext};

        let visibility = match symbol_data.visibility {
            SymbolVisibility::Public => Visibility::Public,
            SymbolVisibility::Module => Visibility::Internal,
            SymbolVisibility::Private => Visibility::Private,
            SymbolVisibility::Internal => Visibility::Internal,
        };

        let type_info = TypeInfo {
            type_id: symbol_data.ast_node.unwrap_or(NodeId(0)),
            type_kind: self.convert_type_category(&symbol_data.kind),
            type_parameters: Vec::new(),
            constraints: Vec::new(),
            semantic_meaning: crate::semantic::SemanticMeaning {
                domain: symbol_data.responsibility.clone(),
                business_rules: Vec::new(),
                constraints: Vec::new(),
            },
            ai_description: symbol_data.ai_context.clone(),
        };

        let business_context = symbol_data.responsibility.as_ref().map(|resp| BusinessContext {
            domain: resp.clone(),
            stakeholders: Vec::new(),
            business_rules: Vec::new(),
            compliance_requirements: Vec::new(),
        });

        Ok(SymbolInfo {
            id: symbol_data.symbol,
            name: symbol_data.name.clone(),
            type_info,
            source_location: symbol_data.location,
            visibility,
            semantic_annotations: Vec::new(),
            business_context,
            ai_hints: symbol_data.ai_context.as_ref().map(|ctx| vec![ctx.clone()]).unwrap_or_default(),
        })
    }

    /// Convert SymbolKind to TypeKind for semantic database integration
    fn convert_type_category(&self, kind: &SymbolKind) -> TypeKind {
        use crate::semantic::{TypeKind, PrimitiveType, SemanticType, CompositeType, FunctionType, GenericType, EffectType};

        match kind {
            SymbolKind::Function { .. } => TypeKind::Function(FunctionType {
                parameters: Vec::new(),
                return_type: None,
                effects: Vec::new(),
                contracts: Vec::new(),
            }),
            SymbolKind::Type { type_category, .. } => match type_category {
                TypeCategory::Primitive => TypeKind::Primitive(PrimitiveType::String),
                TypeCategory::Semantic => TypeKind::Semantic(SemanticType {
                    name: "SemanticType".to_string(),
                    base_type: None,
                    constraints: Vec::new(),
                    business_rules: Vec::new(),
                    validation_rules: Vec::new(),
                    ai_context: None,
                }),
                TypeCategory::Composite => TypeKind::Composite(CompositeType {
                    name: "CompositeType".to_string(),
                    fields: Vec::new(),
                    composition_type: crate::semantic::CompositionType::Struct,
                }),
                TypeCategory::Function => TypeKind::Function(FunctionType {
                    parameters: Vec::new(),
                    return_type: None,
                    effects: Vec::new(),
                    contracts: Vec::new(),
                }),
                TypeCategory::Effect => TypeKind::Effect(EffectType {
                    name: "EffectType".to_string(),
                    effects: Vec::new(),
                    capabilities: Vec::new(),
                }),
            },
            _ => TypeKind::Primitive(PrimitiveType::String),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &SymbolTableConfig {
        &self.config
    }

    /// Get statistics about the symbol table
    pub fn stats(&self) -> SymbolTableStats {
        let symbols = self.symbols.read().unwrap();
        
        let mut kind_counts = HashMap::new();
        let mut visibility_counts = HashMap::new();
        
        for symbol_data in symbols.values() {
            let kind_name = match &symbol_data.kind {
                SymbolKind::Module { .. } => "Module",
                SymbolKind::Function { .. } => "Function", 
                SymbolKind::Type { .. } => "Type",
                SymbolKind::Variable { .. } => "Variable",
                SymbolKind::Constant { .. } => "Constant",
                SymbolKind::Parameter { .. } => "Parameter",
                SymbolKind::Import { .. } => "Import",
                SymbolKind::Export { .. } => "Export",
            };
            *kind_counts.entry(kind_name.to_string()).or_insert(0) += 1;

            let visibility_name = match symbol_data.visibility {
                SymbolVisibility::Public => "Public",
                SymbolVisibility::Module => "Module",
                SymbolVisibility::Private => "Private", 
                SymbolVisibility::Internal => "Internal",
            };
            *visibility_counts.entry(visibility_name.to_string()).or_insert(0) += 1;
        }

        SymbolTableStats {
            total_symbols: symbols.len(),
            symbols_by_kind: kind_counts,
            symbols_by_visibility: visibility_counts,
        }
    }
}

/// Symbol table statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolTableStats {
    /// Total number of symbols
    pub total_symbols: usize,
    /// Count of symbols by kind
    pub symbols_by_kind: HashMap<String, usize>,
    /// Count of symbols by visibility
    pub symbols_by_visibility: HashMap<String, usize>,
} 