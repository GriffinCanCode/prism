//! Core Symbol Table Implementation
//!
//! This module provides the core symbol table implementation for storing and
//! managing symbols with comprehensive metadata and semantic integration.
//! It follows PLT-004 specifications for AI-first symbol management.
//!
//! ## Conceptual Responsibility
//! 
//! This module handles ONE thing: "Symbol Storage and Retrieval"
//! - Symbol registration and storage
//! - Symbol lookup and querying
//! - Integration with semantic database
//! - Symbol table statistics and monitoring
//! 
//! It does NOT handle:
//! - Symbol classification (delegated to kinds.rs)
//! - Symbol metadata generation (delegated to metadata.rs)
//! - Symbol resolution (delegated to resolution subsystem)

use crate::error::{CompilerError, CompilerResult};
use crate::semantic::{SemanticDatabase, SymbolInfo, TypeInfo};
use crate::symbols::data::{SymbolData, SymbolId, SymbolVisibility};
use crate::symbols::kinds::SymbolKind;
use crate::symbols::metadata::SymbolMetadata;
use prism_common::symbol::Symbol;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Core symbol table for storing and managing symbols with semantic metadata
/// 
/// This integrates with existing infrastructure while providing a clean,
/// modular interface for symbol management. It maintains both internal
/// symbol tracking and integration with the semantic database.
#[derive(Debug)]
pub struct SymbolTable {
    /// Symbol storage indexed by Symbol (interned string)
    symbols: Arc<RwLock<HashMap<Symbol, SymbolData>>>,
    
    /// Symbol lookup by SymbolId for fast internal access
    symbol_by_id: Arc<RwLock<HashMap<SymbolId, Symbol>>>,
    
    /// Integration with semantic database
    semantic_db: Arc<SemanticDatabase>,
    
    /// Symbol table configuration
    config: SymbolTableConfig,
}

/// Configuration for symbol table behavior
#[derive(Debug, Clone)]
pub struct SymbolTableConfig {
    /// Enable AI metadata generation during symbol registration
    pub enable_ai_metadata: bool,
    
    /// Enable semantic type integration with semantic database
    pub enable_semantic_types: bool,
    
    /// Enable effect tracking and validation
    pub enable_effect_tracking: bool,
    
    /// Enable documentation validation and generation
    pub enable_doc_validation: bool,
    
    /// Enable business context extraction and analysis
    pub enable_business_context: bool,
    
    /// Enable performance optimization hints
    pub enable_performance_optimization: bool,
    
    /// Enable automatic symbol relationship discovery
    pub enable_relationship_discovery: bool,
    
    /// Maximum number of symbols before triggering cleanup
    pub max_symbols: Option<usize>,
    
    /// Enable symbol usage tracking
    pub enable_usage_tracking: bool,
}

impl Default for SymbolTableConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_semantic_types: true,
            enable_effect_tracking: true,
            enable_doc_validation: true,
            enable_business_context: true,
            enable_performance_optimization: true,
            enable_relationship_discovery: true,
            max_symbols: Some(100_000),
            enable_usage_tracking: true,
        }
    }
}

impl SymbolTable {
    /// Create a new symbol table with semantic database integration
    pub fn new(semantic_db: Arc<SemanticDatabase>) -> CompilerResult<Self> {
        Ok(Self {
            symbols: Arc::new(RwLock::new(HashMap::new())),
            symbol_by_id: Arc::new(RwLock::new(HashMap::new())),
            semantic_db,
            config: SymbolTableConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(
        semantic_db: Arc<SemanticDatabase>, 
        config: SymbolTableConfig
    ) -> CompilerResult<Self> {
        Ok(Self {
            symbols: Arc::new(RwLock::new(HashMap::new())),
            symbol_by_id: Arc::new(RwLock::new(HashMap::new())),
            semantic_db,
            config,
        })
    }

    /// Register a new symbol with comprehensive metadata
    pub fn register_symbol(&self, symbol_data: SymbolData) -> CompilerResult<()> {
        let symbol = symbol_data.symbol;
        let symbol_id = symbol_data.id;
        
        // Check for symbol limit if configured
        if let Some(max_symbols) = self.config.max_symbols {
            let current_count = {
                let symbols = self.symbols.read().unwrap();
                symbols.len()
            };
            
            if current_count >= max_symbols {
                return Err(CompilerError::InvalidInput {
                    message: format!("Symbol table limit reached: {}", max_symbols),
                });
            }
        }
        
        // Store in symbol table
        {
            let mut symbols = self.symbols.write().unwrap();
            let mut symbol_by_id = self.symbol_by_id.write().unwrap();
            
            // Check for duplicate registration
            if symbols.contains_key(&symbol) {
                return Err(CompilerError::DuplicateSymbol {
                    symbol: symbol_data.name.clone(),
                    location: symbol_data.location,
                });
            }
            
            symbols.insert(symbol, symbol_data.clone());
            symbol_by_id.insert(symbol_id, symbol);
        }

        // Integrate with semantic database if enabled
        if self.config.enable_semantic_types {
            let symbol_info = self.convert_to_symbol_info(&symbol_data)?;
            self.semantic_db.add_symbol(symbol, symbol_info)?;
        }

        // Perform additional processing based on configuration
        if self.config.enable_relationship_discovery {
            self.discover_relationships(&symbol_data)?;
        }

        Ok(())
    }

    /// Get symbol data by symbol
    pub fn get_symbol(&self, symbol: &Symbol) -> Option<SymbolData> {
        let symbols = self.symbols.read().unwrap();
        symbols.get(symbol).cloned()
    }

    /// Get symbol data by symbol ID
    pub fn get_symbol_by_id(&self, symbol_id: SymbolId) -> Option<SymbolData> {
        let symbol_by_id = self.symbol_by_id.read().unwrap();
        let symbol = symbol_by_id.get(&symbol_id)?;
        self.get_symbol(symbol)
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
    pub fn update_symbol<F>(&self, symbol: &Symbol, updater: F) -> CompilerResult<()>
    where
        F: FnOnce(&mut SymbolData),
    {
        let mut symbols = self.symbols.write().unwrap();
        if let Some(symbol_data) = symbols.get_mut(symbol) {
            updater(symbol_data);
            
            // Update last modified timestamp
            symbol_data.metadata.last_updated = Some(std::time::SystemTime::now());
            
            // Recalculate confidence if metadata was updated
            symbol_data.metadata.calculate_confidence();
            
            Ok(())
        } else {
            Err(CompilerError::SymbolNotFound { 
                symbol: symbol.resolve().unwrap_or_else(|| format!("{:?}", symbol))
            })
        }
    }

    /// Get symbols by kind using a filter function
    pub fn get_symbols_by_kind<F>(&self, kind_filter: F) -> Vec<SymbolData>
    where
        F: Fn(&SymbolKind) -> bool,
    {
        self.find_symbols(|data| kind_filter(&data.kind))
    }

    /// Get symbols by visibility
    pub fn get_symbols_by_visibility(&self, visibility: SymbolVisibility) -> Vec<SymbolData> {
        self.find_symbols(|data| data.visibility == visibility)
    }

    /// Get symbols with specific effects
    pub fn get_symbols_with_effects(&self, effect_names: &[String]) -> Vec<SymbolData> {
        self.find_symbols(|data| {
            effect_names.iter().any(|effect_name| {
                data.effects.iter().any(|effect| effect.name == *effect_name)
            })
        })
    }

    /// Get symbols requiring specific capabilities
    pub fn get_symbols_requiring_capabilities(&self, capabilities: &[String]) -> Vec<SymbolData> {
        self.find_symbols(|data| {
            capabilities.iter().any(|capability| {
                data.required_capabilities.contains(capability)
            })
        })
    }

    /// Get symbols with AI metadata
    pub fn get_symbols_with_ai_metadata(&self) -> Vec<SymbolData> {
        self.find_symbols(|data| data.metadata.ai_context.is_some())
    }

    /// Record symbol usage for tracking and optimization
    pub fn record_symbol_usage(
        &self, 
        symbol: &Symbol, 
        usage_pattern: crate::symbols::data::UsagePatternType,
        context: String
    ) -> CompilerResult<()> {
        if !self.config.enable_usage_tracking {
            return Ok(());
        }

        self.update_symbol(symbol, |symbol_data| {
            symbol_data.record_usage(usage_pattern, context);
        })
    }

    /// Remove a symbol from the table
    pub fn remove_symbol(&self, symbol: &Symbol) -> CompilerResult<SymbolData> {
        let mut symbols = self.symbols.write().unwrap();
        let mut symbol_by_id = self.symbol_by_id.write().unwrap();
        
        if let Some(symbol_data) = symbols.remove(symbol) {
            symbol_by_id.remove(&symbol_data.id);
            Ok(symbol_data)
        } else {
            Err(CompilerError::SymbolNotFound {
                symbol: symbol.resolve().unwrap_or_else(|| format!("{:?}", symbol))
            })
        }
    }

    /// Clear all symbols from the table
    pub fn clear(&self) {
        let mut symbols = self.symbols.write().unwrap();
        let mut symbol_by_id = self.symbol_by_id.write().unwrap();
        
        symbols.clear();
        symbol_by_id.clear();
    }

    /// Get symbol count
    pub fn symbol_count(&self) -> usize {
        let symbols = self.symbols.read().unwrap();
        symbols.len()
    }

    /// Check if a symbol exists
    pub fn contains_symbol(&self, symbol: &Symbol) -> bool {
        let symbols = self.symbols.read().unwrap();
        symbols.contains_key(symbol)
    }

    /// Get all symbols as a vector
    pub fn all_symbols(&self) -> Vec<SymbolData> {
        let symbols = self.symbols.read().unwrap();
        symbols.values().cloned().collect()
    }

    /// Convert SymbolData to SemanticDatabase SymbolInfo for integration
    fn convert_to_symbol_info(&self, symbol_data: &SymbolData) -> CompilerResult<SymbolInfo> {
        use crate::semantic::{SymbolInfo, TypeInfo, TypeKind, Visibility, BusinessContext};

        let visibility = match symbol_data.visibility {
            SymbolVisibility::Public => Visibility::Public,
            SymbolVisibility::Module => Visibility::Internal,
            SymbolVisibility::Private => Visibility::Private,
            SymbolVisibility::Internal => Visibility::Internal,
            SymbolVisibility::Protected => Visibility::Internal,
            SymbolVisibility::Package => Visibility::Internal,
        };

        let type_info = TypeInfo {
            type_id: symbol_data.ast_node.unwrap_or(prism_common::NodeId::new(0)),
            type_kind: self.convert_symbol_kind_to_type_kind(&symbol_data.kind)?,
            type_parameters: Vec::new(),
            constraints: Vec::new(),
            semantic_meaning: crate::semantic::SemanticMeaning {
                domain: symbol_data.metadata.business_context
                    .as_ref()
                    .map(|bc| bc.domain.clone()),
                business_rules: Vec::new(),
                constraints: Vec::new(),
            },
            ai_description: symbol_data.metadata.ai_context
                .as_ref()
                .map(|ai| ai.purpose.clone()),
        };

        let business_context = symbol_data.metadata.business_context.as_ref().map(|bc| {
            BusinessContext {
                domain: bc.domain.clone(),
                stakeholders: bc.stakeholders.clone(),
                business_rules: Vec::new(), // Would convert from BusinessRule to semantic format
                compliance_requirements: Vec::new(), // Would convert compliance requirements
            }
        });

        Ok(SymbolInfo {
            id: symbol_data.symbol,
            name: symbol_data.name.clone(),
            type_info,
            source_location: symbol_data.location,
            visibility,
            semantic_annotations: Vec::new(), // Would convert semantic annotations
            business_context,
            ai_hints: symbol_data.metadata.ai_context
                .as_ref()
                .map(|ctx| vec![ctx.description.clone()])
                .unwrap_or_default(),
        })
    }

    /// Convert SymbolKind to semantic database TypeKind
    fn convert_symbol_kind_to_type_kind(&self, kind: &SymbolKind) -> CompilerResult<TypeKind> {
        use crate::semantic::{TypeKind, PrimitiveType, SemanticType, CompositeType, FunctionType, GenericType, EffectType};

        let type_kind = match kind {
            SymbolKind::Function { signature, .. } => TypeKind::Function(FunctionType {
                parameters: Vec::new(), // Would convert from signature.parameters
                return_type: None, // Would convert from signature.return_type
                effects: Vec::new(),
                contracts: Vec::new(),
            }),
            SymbolKind::Type { type_category, .. } => {
                match type_category {
                    crate::symbols::kinds::TypeCategory::Primitive { primitive_type, .. } => {
                        let prim_type = match primitive_type {
                            crate::symbols::kinds::PrimitiveType::String => PrimitiveType::String,
                            crate::symbols::kinds::PrimitiveType::Bool => PrimitiveType::Bool,
                            crate::symbols::kinds::PrimitiveType::SignedInt(_) => PrimitiveType::SignedInt(32),
                            crate::symbols::kinds::PrimitiveType::UnsignedInt(_) => PrimitiveType::UnsignedInt(32),
                            crate::symbols::kinds::PrimitiveType::Float(_) => PrimitiveType::Float(64),
                            crate::symbols::kinds::PrimitiveType::Char => PrimitiveType::String,
                            crate::symbols::kinds::PrimitiveType::Unit => PrimitiveType::Unit,
                            crate::symbols::kinds::PrimitiveType::Never => PrimitiveType::Unit,
                        };
                        TypeKind::Primitive(prim_type)
                    },
                    crate::symbols::kinds::TypeCategory::Semantic { base_type, domain, .. } => {
                        TypeKind::Semantic(SemanticType {
                            name: domain.clone(),
                            base_type: Some(base_type.clone()),
                            constraints: Vec::new(),
                            business_rules: Vec::new(),
                            validation_rules: Vec::new(),
                            ai_context: None,
                        })
                    },
                    crate::symbols::kinds::TypeCategory::Composite { composition_type, .. } => {
                        TypeKind::Composite(CompositeType {
                            name: composition_type.name.clone(),
                            fields: Vec::new(),
                            composition_type: crate::semantic::CompositionType::Struct,
                        })
                    },
                    crate::symbols::kinds::TypeCategory::Function { signature } => {
                        TypeKind::Function(FunctionType {
                            parameters: Vec::new(),
                            return_type: None,
                            effects: Vec::new(),
                            contracts: Vec::new(),
                        })
                    },
                    crate::symbols::kinds::TypeCategory::Generic { parameters, .. } => {
                        TypeKind::Generic(GenericType {
                            name: "GenericType".to_string(),
                            parameters: Vec::new(),
                            constraints: Vec::new(),
                        })
                    },
                    crate::symbols::kinds::TypeCategory::Effect { effects, capabilities } => {
                        TypeKind::Effect(EffectType {
                            name: "EffectType".to_string(),
                            effects: effects.clone(),
                            capabilities: capabilities.clone(),
                        })
                    },
                }
            },
            _ => TypeKind::Primitive(PrimitiveType::String), // Default fallback
        };

        Ok(type_kind)
    }

    /// Discover symbol relationships (placeholder for future implementation)
    fn discover_relationships(&self, _symbol_data: &SymbolData) -> CompilerResult<()> {
        // This would implement automatic relationship discovery
        // based on symbol usage patterns, type relationships, etc.
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &SymbolTableConfig {
        &self.config
    }

    /// Get symbol table statistics
    pub fn stats(&self) -> SymbolTableStats {
        let symbols = self.symbols.read().unwrap();
        
        let mut kind_counts = HashMap::new();
        let mut visibility_counts = HashMap::new();
        let mut effect_counts = HashMap::new();
        let mut capability_counts = HashMap::new();
        
        for symbol_data in symbols.values() {
            // Count by kind
            let kind_name = symbol_data.kind.ai_category();
            *kind_counts.entry(kind_name.to_string()).or_insert(0) += 1;

            // Count by visibility
            let visibility_name = symbol_data.visibility.description();
            *visibility_counts.entry(visibility_name.to_string()).or_insert(0) += 1;

            // Count effects
            for effect in &symbol_data.effects {
                *effect_counts.entry(effect.name.clone()).or_insert(0) += 1;
            }

            // Count capabilities
            for capability in &symbol_data.required_capabilities {
                *capability_counts.entry(capability.clone()).or_insert(0) += 1;
            }
        }

        let symbols_with_ai_metadata = symbols.values()
            .filter(|s| s.metadata.ai_context.is_some())
            .count();

        let symbols_with_effects = symbols.values()
            .filter(|s| !s.effects.is_empty())
            .count();

        let symbols_with_capabilities = symbols.values()
            .filter(|s| !s.required_capabilities.is_empty())
            .count();

        SymbolTableStats {
            total_symbols: symbols.len(),
            symbols_by_kind: kind_counts,
            symbols_by_visibility: visibility_counts,
            effect_usage: effect_counts,
            capability_usage: capability_counts,
            symbols_with_ai_metadata,
            symbols_with_effects,
            symbols_with_capabilities,
            average_metadata_confidence: symbols.values()
                .map(|s| s.metadata.confidence)
                .sum::<f64>() / symbols.len() as f64,
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
    
    /// Effect usage statistics
    pub effect_usage: HashMap<String, usize>,
    
    /// Capability usage statistics
    pub capability_usage: HashMap<String, usize>,
    
    /// Number of symbols with AI metadata
    pub symbols_with_ai_metadata: usize,
    
    /// Number of symbols with effects
    pub symbols_with_effects: usize,
    
    /// Number of symbols with capability requirements
    pub symbols_with_capabilities: usize,
    
    /// Average metadata confidence score
    pub average_metadata_confidence: f64,
}

impl SymbolTableStats {
    /// Get the most commonly used symbol kind
    pub fn most_common_kind(&self) -> Option<(&String, &usize)> {
        self.symbols_by_kind.iter().max_by_key(|(_, count)| *count)
    }
    
    /// Get the most commonly used effect
    pub fn most_common_effect(&self) -> Option<(&String, &usize)> {
        self.effect_usage.iter().max_by_key(|(_, count)| *count)
    }
    
    /// Calculate AI metadata coverage percentage
    pub fn ai_metadata_coverage(&self) -> f64 {
        if self.total_symbols == 0 {
            0.0
        } else {
            (self.symbols_with_ai_metadata as f64 / self.total_symbols as f64) * 100.0
        }
    }
    
    /// Calculate effect usage percentage
    pub fn effect_usage_percentage(&self) -> f64 {
        if self.total_symbols == 0 {
            0.0
        } else {
            (self.symbols_with_effects as f64 / self.total_symbols as f64) * 100.0
        }
    }
}

// Implement traits for the SymbolTable to match the interface defined in mod.rs
impl crate::symbols::SymbolManager for SymbolTable {
    fn register_symbol(&mut self, symbol_data: SymbolData) -> CompilerResult<()> {
        // For mutable interface, we delegate to the immutable version
        // This is safe because the internal implementation uses Arc<RwLock<>>
        SymbolTable::register_symbol(self, symbol_data)
    }
    
    fn get_symbol(&self, symbol: &Symbol) -> Option<&SymbolData> {
        // This would require returning a reference, which is challenging with RwLock
        // For now, we'll need to adjust the trait or use a different approach
        // This is a design decision that would need to be resolved
        unimplemented!("get_symbol with reference return requires design adjustment")
    }
    
    fn get_symbol_mut(&mut self, symbol: &Symbol) -> Option<&mut SymbolData> {
        // Similar issue with mutable references
        unimplemented!("get_symbol_mut with reference return requires design adjustment")
    }
    
    fn find_symbols<F>(&self, predicate: F) -> Vec<SymbolData>
    where
        F: Fn(&SymbolData) -> bool,
    {
        SymbolTable::find_symbols(self, predicate)
    }
    
    fn update_symbol<F>(&mut self, symbol: &Symbol, updater: F) -> CompilerResult<()>
    where
        F: FnOnce(&mut SymbolData),
    {
        SymbolTable::update_symbol(self, symbol, updater)
    }
}

impl crate::symbols::SymbolQuery for SymbolTable {
    fn find_by_kind(&self, kind_filter: impl Fn(&SymbolKind) -> bool) -> Vec<SymbolData> {
        self.get_symbols_by_kind(kind_filter)
    }
    
    fn find_by_visibility(&self, visibility: SymbolVisibility) -> Vec<SymbolData> {
        self.get_symbols_by_visibility(visibility)
    }
    
    fn find_with_effects(&self, effects: &[String]) -> Vec<SymbolData> {
        self.get_symbols_with_effects(effects)
    }
    
    fn find_with_ai_metadata(&self) -> Vec<SymbolData> {
        self.get_symbols_with_ai_metadata()
    }
} 