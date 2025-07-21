//! Semantic Database - Information Storage and Querying
//!
//! This module embodies the single concept of "Semantic Information Storage".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: storing and querying semantic information for AI consumption.
//!
//! **Conceptual Responsibility**: Semantic information storage and retrieval
//! **What it does**: semantic info storage, querying, AI context export
//! **What it doesn't do**: semantic analysis, type inference, validation

use crate::{SemanticResult, SemanticError, SemanticConfig};
use crate::analyzer::{SymbolInfo, TypeInfo, AnalysisMetadata};
use crate::types::{SemanticType, BusinessRule};
use crate::inference::InferredType;
use crate::validation::ValidationResult;
use crate::patterns::SemanticPattern;
use crate::context::AIMetadata;
use prism_common::{NodeId, span::Span, symbol::Symbol};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Semantic database for storing and querying semantic information
#[derive(Debug)]
pub struct SemanticDatabase {
    /// Configuration
    config: DatabaseConfig,
    /// Symbol information storage
    symbols: Arc<RwLock<HashMap<Symbol, SymbolInfo>>>,
    /// Type information storage
    types: Arc<RwLock<HashMap<NodeId, TypeInfo>>>,
    /// Semantic types storage
    semantic_types: Arc<RwLock<HashMap<Symbol, SemanticType>>>,
    /// Business rules storage
    business_rules: Arc<RwLock<HashMap<String, BusinessRule>>>,
    /// AI metadata storage
    ai_metadata: Arc<RwLock<HashMap<NodeId, AIMetadata>>>,
    /// Location-based index
    location_index: Arc<RwLock<HashMap<Span, Vec<NodeId>>>>,
    /// Semantic information cache
    semantic_info_cache: Arc<RwLock<HashMap<String, SemanticInfo>>>,
}

/// Configuration for semantic database
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Enable caching
    pub enable_caching: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable location indexing
    pub enable_location_index: bool,
    /// Enable AI metadata storage
    pub enable_ai_metadata: bool,
}

/// Comprehensive semantic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInfo {
    /// Symbol table
    pub symbols: HashMap<Symbol, SymbolInfo>,
    /// Type information
    pub types: HashMap<NodeId, TypeInfo>,
    /// Inferred types (if available)
    pub inferred_types: Option<HashMap<NodeId, InferredType>>,
    /// Validation results (if available)
    pub validation_result: Option<ValidationResult>,
    /// Recognized patterns
    pub patterns: Vec<SemanticPattern>,
    /// AI metadata (if available)
    pub ai_metadata: Option<AIMetadata>,
    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
}

impl SemanticDatabase {
    /// Create a new semantic database
    pub fn new(config: &SemanticConfig) -> SemanticResult<Self> {
        let db_config = DatabaseConfig {
            enable_caching: config.enable_incremental,
            max_cache_size: 10_000,
            enable_location_index: true,
            enable_ai_metadata: config.enable_ai_metadata,
        };

        Ok(Self {
            config: db_config,
            symbols: Arc::new(RwLock::new(HashMap::new())),
            types: Arc::new(RwLock::new(HashMap::new())),
            semantic_types: Arc::new(RwLock::new(HashMap::new())),
            business_rules: Arc::new(RwLock::new(HashMap::new())),
            ai_metadata: Arc::new(RwLock::new(HashMap::new())),
            location_index: Arc::new(RwLock::new(HashMap::new())),
            semantic_info_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Store comprehensive semantic information
    pub fn store_semantic_info(&self, info: &SemanticInfo) -> SemanticResult<()> {
        // Store symbols
        {
            let mut symbols = self.symbols.write().unwrap();
            for (symbol, symbol_info) in &info.symbols {
                symbols.insert(*symbol, symbol_info.clone());
            }
        }

        // Store types
        {
            let mut types = self.types.write().unwrap();
            for (node_id, type_info) in &info.types {
                types.insert(*node_id, type_info.clone());
            }
        }

        // Store AI metadata if available
        if let Some(ai_metadata) = &info.ai_metadata {
            if self.config.enable_ai_metadata {
                let mut metadata = self.ai_metadata.write().unwrap();
                // Store metadata for all relevant nodes
                // This is a simplified approach - in practice, we'd have more granular storage
                for (node_id, _) in &info.types {
                    metadata.insert(*node_id, ai_metadata.clone());
                }
            }
        }

        // Update location index if enabled
        if self.config.enable_location_index {
            self.update_location_index(&info.symbols, &info.types)?;
        }

        // Cache the semantic info if caching is enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(&info.analysis_metadata);
            let mut cache = self.semantic_info_cache.write().unwrap();
            
            // Check cache size limit
            if cache.len() >= self.config.max_cache_size {
                // Simple LRU eviction - remove oldest entry
                if let Some(oldest_key) = cache.keys().next().cloned() {
                    cache.remove(&oldest_key);
                }
            }
            
            cache.insert(cache_key, info.clone());
        }

        Ok(())
    }

    /// Get symbol information
    pub fn get_symbol(&self, symbol: &Symbol) -> Option<SymbolInfo> {
        let symbols = self.symbols.read().unwrap();
        symbols.get(symbol).cloned()
    }

    /// Get type information
    pub fn get_type(&self, node_id: &NodeId) -> Option<TypeInfo> {
        let types = self.types.read().unwrap();
        types.get(node_id).cloned()
    }

    /// Get semantic type
    pub fn get_semantic_type(&self, symbol: &Symbol) -> Option<SemanticType> {
        let semantic_types = self.semantic_types.read().unwrap();
        semantic_types.get(symbol).cloned()
    }

    /// Get business rule
    pub fn get_business_rule(&self, rule_id: &str) -> Option<BusinessRule> {
        let business_rules = self.business_rules.read().unwrap();
        business_rules.get(rule_id).cloned()
    }

    /// Get AI metadata for a node
    pub fn get_ai_metadata(&self, node_id: &NodeId) -> Option<AIMetadata> {
        let ai_metadata = self.ai_metadata.read().unwrap();
        ai_metadata.get(node_id).cloned()
    }

    /// Get semantic information at a specific location
    pub fn get_semantic_info_at(&self, location: Span) -> SemanticResult<Option<SemanticInfo>> {
        if !self.config.enable_location_index {
            return Ok(None);
        }

        let location_index = self.location_index.read().unwrap();
        
        // Find nodes at this location
        let nodes = location_index.get(&location);
        if nodes.is_none() {
            return Ok(None);
        }

        let nodes = nodes.unwrap();
        if nodes.is_empty() {
            return Ok(None);
        }

        // Collect semantic information for these nodes
        let mut symbols = HashMap::new();
        let mut types = HashMap::new();
        let mut ai_metadata = None;

        {
            let symbol_storage = self.symbols.read().unwrap();
            let type_storage = self.types.read().unwrap();
            let ai_storage = self.ai_metadata.read().unwrap();

            for node_id in nodes {
                // Find symbols at this location
                for (symbol, symbol_info) in symbol_storage.iter() {
                    if symbol_info.location.overlaps(&location) {
                        symbols.insert(*symbol, symbol_info.clone());
                    }
                }

                // Get type information
                if let Some(type_info) = type_storage.get(node_id) {
                    types.insert(*node_id, type_info.clone());
                }

                // Get AI metadata (use the first one found)
                if ai_metadata.is_none() {
                    ai_metadata = ai_storage.get(node_id).cloned();
                }
            }
        }

        if symbols.is_empty() && types.is_empty() {
            return Ok(None);
        }

        let symbol_count = symbols.len();
        let type_count = types.len();
        
        Ok(Some(SemanticInfo {
            symbols,
            types,
            inferred_types: None,
            validation_result: None,
            patterns: Vec::new(),
            ai_metadata,
            analysis_metadata: AnalysisMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                duration_ms: 0,
                symbols_analyzed: symbol_count,
                types_analyzed: type_count,
                warnings: Vec::new(),
            },
        }))
    }

    /// Store a semantic type
    pub fn store_semantic_type(&self, symbol: Symbol, semantic_type: SemanticType) -> SemanticResult<()> {
        let mut semantic_types = self.semantic_types.write().unwrap();
        semantic_types.insert(symbol, semantic_type);
        Ok(())
    }

    /// Store a business rule
    pub fn store_business_rule(&self, rule: BusinessRule) -> SemanticResult<()> {
        let mut business_rules = self.business_rules.write().unwrap();
        business_rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Store AI metadata
    pub fn store_ai_metadata(&self, node_id: NodeId, metadata: AIMetadata) -> SemanticResult<()> {
        if !self.config.enable_ai_metadata {
            return Ok(());
        }

        let mut ai_metadata = self.ai_metadata.write().unwrap();
        ai_metadata.insert(node_id, metadata);
        Ok(())
    }

    /// Query symbols by pattern
    pub fn query_symbols(&self, pattern: &str) -> Vec<SymbolInfo> {
        let symbols = self.symbols.read().unwrap();
        symbols
            .values()
            .filter(|symbol_info| symbol_info.name.contains(pattern))
            .cloned()
            .collect()
    }

    /// Query types by domain
    pub fn query_types_by_domain(&self, domain: &str) -> Vec<TypeInfo> {
        let types = self.types.read().unwrap();
        types
            .values()
            .filter(|type_info| {
                type_info.domain.as_ref().map_or(false, |d| d.contains(domain))
            })
            .cloned()
            .collect()
    }

    /// Query business rules by domain
    pub fn query_business_rules_by_domain(&self, domain: &str) -> Vec<BusinessRule> {
        let business_rules = self.business_rules.read().unwrap();
        business_rules
            .values()
            .filter(|rule| rule.domain.contains(domain))
            .cloned()
            .collect()
    }

    /// Get all symbols
    pub fn get_all_symbols(&self) -> Vec<SymbolInfo> {
        let symbols = self.symbols.read().unwrap();
        symbols.values().cloned().collect()
    }

    /// Get all types
    pub fn get_all_types(&self) -> Vec<TypeInfo> {
        let types = self.types.read().unwrap();
        types.values().cloned().collect()
    }

    /// Get all semantic types
    pub fn get_all_semantic_types(&self) -> Vec<SemanticType> {
        let semantic_types = self.semantic_types.read().unwrap();
        semantic_types.values().cloned().collect()
    }

    /// Get all business rules
    pub fn get_all_business_rules(&self) -> Vec<BusinessRule> {
        let business_rules = self.business_rules.read().unwrap();
        business_rules.values().cloned().collect()
    }

    /// Export all semantic information for AI consumption
    pub fn export_all_semantic_info(&self) -> SemanticResult<SemanticInfo> {
        let symbols = {
            let symbol_storage = self.symbols.read().unwrap();
            symbol_storage.clone()
        };

        let types = {
            let type_storage = self.types.read().unwrap();
            type_storage.clone()
        };

        // Create aggregate AI metadata
        let ai_metadata = if self.config.enable_ai_metadata {
            let ai_storage = self.ai_metadata.read().unwrap();
            ai_storage.values().next().cloned() // Get any available metadata as example
        } else {
            None
        };

        let symbol_count = symbols.len();
        let type_count = types.len();
        
        Ok(SemanticInfo {
            symbols,
            types,
            inferred_types: None,
            validation_result: None,
            patterns: Vec::new(),
            ai_metadata,
            analysis_metadata: AnalysisMetadata {
                timestamp: chrono::Utc::now().to_rfc3339(),
                duration_ms: 0,
                symbols_analyzed: symbol_count,
                types_analyzed: type_count,
                warnings: Vec::new(),
            },
        })
    }

    /// Clear all stored information
    pub fn clear(&self) -> SemanticResult<()> {
        {
            let mut symbols = self.symbols.write().unwrap();
            symbols.clear();
        }
        {
            let mut types = self.types.write().unwrap();
            types.clear();
        }
        {
            let mut semantic_types = self.semantic_types.write().unwrap();
            semantic_types.clear();
        }
        {
            let mut business_rules = self.business_rules.write().unwrap();
            business_rules.clear();
        }
        {
            let mut ai_metadata = self.ai_metadata.write().unwrap();
            ai_metadata.clear();
        }
        {
            let mut location_index = self.location_index.write().unwrap();
            location_index.clear();
        }
        {
            let mut cache = self.semantic_info_cache.write().unwrap();
            cache.clear();
        }

        Ok(())
    }

    /// Get database statistics
    pub fn get_statistics(&self) -> DatabaseStatistics {
        let symbol_count = {
            let symbols = self.symbols.read().unwrap();
            symbols.len()
        };

        let type_count = {
            let types = self.types.read().unwrap();
            types.len()
        };

        let semantic_type_count = {
            let semantic_types = self.semantic_types.read().unwrap();
            semantic_types.len()
        };

        let business_rule_count = {
            let business_rules = self.business_rules.read().unwrap();
            business_rules.len()
        };

        let ai_metadata_count = {
            let ai_metadata = self.ai_metadata.read().unwrap();
            ai_metadata.len()
        };

        let cache_size = {
            let cache = self.semantic_info_cache.read().unwrap();
            cache.len()
        };

        DatabaseStatistics {
            symbol_count,
            type_count,
            semantic_type_count,
            business_rule_count,
            ai_metadata_count,
            cache_size,
            cache_hit_ratio: 0.0, // Would be calculated from actual metrics
        }
    }

    // Helper methods

    /// Update the location index
    fn update_location_index(
        &self,
        symbols: &HashMap<Symbol, SymbolInfo>,
        types: &HashMap<NodeId, TypeInfo>,
    ) -> SemanticResult<()> {
        let mut location_index = self.location_index.write().unwrap();

        // Index symbols by location
        for symbol_info in symbols.values() {
            location_index
                .entry(symbol_info.location)
                .or_insert_with(Vec::new)
                .push(NodeId::new(0)); // Would use actual node ID
        }

        // Index types by location
        for (node_id, type_info) in types {
            location_index
                .entry(type_info.location)
                .or_insert_with(Vec::new)
                .push(*node_id);
        }

        Ok(())
    }

    /// Generate cache key for semantic info
    fn generate_cache_key(&self, metadata: &AnalysisMetadata) -> String {
        format!("{}_{}_{}_{}", 
            metadata.timestamp,
            metadata.symbols_analyzed,
            metadata.types_analyzed,
            metadata.duration_ms
        )
    }
}

/// Database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStatistics {
    /// Number of symbols stored
    pub symbol_count: usize,
    /// Number of types stored
    pub type_count: usize,
    /// Number of semantic types stored
    pub semantic_type_count: usize,
    /// Number of business rules stored
    pub business_rule_count: usize,
    /// Number of AI metadata entries stored
    pub ai_metadata_count: usize,
    /// Current cache size
    pub cache_size: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 10_000,
            enable_location_index: true,
            enable_ai_metadata: true,
        }
    }
}

// Extend Span with overlap checking
trait SpanExt {
    fn overlaps(&self, other: &Span) -> bool;
}

impl SpanExt for Span {
    fn overlaps(&self, other: &Span) -> bool {
        // Simple overlap check - would need proper implementation
        self.start <= other.end && other.start <= self.end
    }
} 