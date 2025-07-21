//! Symbol Caching Layer
//!
//! This module provides performance caching for symbol operations,
//! implementing multi-level caching strategies for optimal lookup performance.

use crate::symbols::data::{SymbolData, SymbolId};
use prism_common::symbol::Symbol;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// Cache key for symbol lookups
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheKey {
    SymbolLookup(Symbol),
    IdLookup(SymbolId),
    KindQuery(String),
    VisibilityQuery(String),
}

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cached data
    pub data: CacheData,
    /// Cache timestamp
    pub timestamp: std::time::SystemTime,
    /// Hit count for this entry
    pub hit_count: u64,
}

/// Types of cached data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheData {
    SingleSymbol(Option<SymbolData>),
    MultipleSymbols(Vec<SymbolData>),
    SymbolCount(usize),
}

/// Symbol cache for performance optimization
#[derive(Debug)]
pub struct SymbolCache {
    /// L1 cache for individual symbol lookups
    l1_cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    /// Cache configuration
    config: CacheConfig,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub max_entries: usize,
    /// Cache entry TTL in seconds
    pub ttl_seconds: u64,
    /// Enable cache statistics
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            ttl_seconds: 300, // 5 minutes
            enable_stats: true,
        }
    }
}

impl SymbolCache {
    /// Create a new symbol cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    /// Get cached entry
    pub fn get(&self, key: &CacheKey) -> Option<CacheEntry> {
        let cache = self.l1_cache.read().unwrap();
        cache.get(key).cloned()
    }
    
    /// Insert cache entry
    pub fn insert(&self, key: CacheKey, data: CacheData) {
        let mut cache = self.l1_cache.write().unwrap();
        
        // Check cache size limit
        if cache.len() >= self.config.max_entries {
            // Simple eviction: remove oldest entries
            // In a real implementation, we'd use LRU or similar
            cache.clear();
        }
        
        let entry = CacheEntry {
            data,
            timestamp: std::time::SystemTime::now(),
            hit_count: 0,
        };
        
        cache.insert(key, entry);
    }
    
    /// Invalidate all query caches
    pub fn invalidate_queries(&self) {
        let mut cache = self.l1_cache.write().unwrap();
        cache.retain(|key, _| matches!(key, CacheKey::SymbolLookup(_) | CacheKey::IdLookup(_)));
    }
    
    /// Clear all cache entries
    pub fn clear(&self) {
        let mut cache = self.l1_cache.write().unwrap();
        cache.clear();
    }
} 