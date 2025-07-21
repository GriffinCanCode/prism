//! PIR Construction Cache - Semantic-Aware Caching Strategies
//!
//! This module implements PIR-specific caching strategies that understand
//! semantic relationships and business context for optimal cache invalidation.

use crate::{PIRResult, PIRError, semantic::*};
use prism_common::{NodeId, span::Span};
use prism_compiler::query::{CacheKey, InvalidationTrigger};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

/// PIR-specific cache key that includes semantic context
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PIRCacheKey {
    /// Base cache key from compiler
    pub base_key: CacheKey,
    /// Semantic context hash
    pub semantic_hash: Option<u64>,
    /// Business domain context
    pub business_domain: Option<String>,
    /// Effect signature hash
    pub effect_hash: Option<u64>,
}

/// Cache strategy for PIR construction
pub trait PIRCacheStrategy: Send + Sync {
    /// Generate cache key for PIR construction input
    fn generate_key(&self, input: &dyn Hash, context: &PIRCacheContext) -> PIRCacheKey;
    
    /// Check if cached result is still valid
    fn is_valid(&self, key: &PIRCacheKey, context: &PIRCacheContext) -> bool;
    
    /// Determine cache invalidation triggers
    fn invalidation_triggers(&self, key: &PIRCacheKey) -> HashSet<PIRInvalidationTrigger>;
    
    /// Calculate cache priority (higher = more important to keep)
    fn cache_priority(&self, key: &PIRCacheKey, usage_stats: &CacheUsageStats) -> f64;
}

/// Context for PIR caching decisions
#[derive(Debug, Clone)]
pub struct PIRCacheContext {
    /// Current semantic version
    pub semantic_version: String,
    /// Business domain being processed
    pub business_domain: Option<String>,
    /// Active effect context
    pub effect_context: Option<String>,
    /// Optimization level
    pub optimization_level: u8,
    /// Compilation timestamp
    pub timestamp: SystemTime,
}

/// PIR-specific invalidation triggers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PIRInvalidationTrigger {
    /// Base compiler invalidation
    Compiler(InvalidationTrigger),
    /// Semantic model changed
    SemanticModelChanged,
    /// Business domain rules updated
    BusinessDomainUpdated(String),
    /// Effect system capabilities changed
    EffectCapabilitiesChanged,
    /// PIR schema version updated
    PIRSchemaUpdated,
    /// Cohesion analysis parameters changed
    CohesionParametersChanged,
}

/// Cache usage statistics
#[derive(Debug, Clone)]
pub struct CacheUsageStats {
    /// Number of cache hits
    pub hit_count: u64,
    /// Number of cache misses
    pub miss_count: u64,
    /// Last access time
    pub last_accessed: SystemTime,
    /// Average computation time when cache miss
    pub avg_computation_time: Duration,
    /// Memory size of cached result
    pub memory_size_bytes: u64,
}

/// Cached PIR construction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPIRResult {
    /// The cached PIR
    pub pir: PrismIR,
    /// Semantic preservation score at cache time
    pub semantic_score: f64,
    /// Business context coverage at cache time
    pub business_coverage: f64,
    /// Cache timestamp
    pub cached_at: u64, // Unix timestamp
    /// Cache version
    pub cache_version: String,
}

/// PIR cache manager
pub struct PIRCacheManager {
    /// Cache storage
    cache: HashMap<PIRCacheKey, CachedPIRResult>,
    /// Usage statistics
    usage_stats: HashMap<PIRCacheKey, CacheUsageStats>,
    /// Cache strategy
    strategy: Box<dyn PIRCacheStrategy>,
    /// Cache configuration
    config: PIRCacheConfig,
}

/// Configuration for PIR caching
#[derive(Debug, Clone)]
pub struct PIRCacheConfig {
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Maximum cache age in seconds
    pub max_age_seconds: u64,
    /// Enable semantic-aware invalidation
    pub enable_semantic_invalidation: bool,
    /// Enable business-context-aware caching
    pub enable_business_context_caching: bool,
    /// Cache compression level (0-9)
    pub compression_level: u8,
}

impl Default for PIRCacheConfig {
    fn default() -> Self {
        Self {
            max_size_mb: 256,
            max_age_seconds: 3600, // 1 hour
            enable_semantic_invalidation: true,
            enable_business_context_caching: true,
            compression_level: 6,
        }
    }
}

/// Default PIR cache strategy - semantic and business context aware
pub struct DefaultPIRCacheStrategy {
    config: PIRCacheConfig,
}

impl DefaultPIRCacheStrategy {
    pub fn new(config: PIRCacheConfig) -> Self {
        Self { config }
    }
}

impl PIRCacheStrategy for DefaultPIRCacheStrategy {
    fn generate_key(&self, input: &dyn Hash, context: &PIRCacheContext) -> PIRCacheKey {
        // Generate base key
        let mut hasher = rustc_hash::FxHasher::default();
        input.hash(&mut hasher);
        let input_hash = hasher.finish();
        
        let base_key = CacheKey {
            query_type: "pir_construction".to_string(),
            input_hash,
            semantic_hash: None,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_config: Some(format!("opt_{}", context.optimization_level)),
        };
        
        // Add semantic context hash
        let semantic_hash = if self.config.enable_semantic_invalidation {
            let mut hasher = rustc_hash::FxHasher::default();
            context.semantic_version.hash(&mut hasher);
            Some(hasher.finish())
        } else {
            None
        };
        
        // Add effect context hash
        let effect_hash = context.effect_context.as_ref().map(|effect_ctx| {
            let mut hasher = rustc_hash::FxHasher::default();
            effect_ctx.hash(&mut hasher);
            hasher.finish()
        });
        
        PIRCacheKey {
            base_key,
            semantic_hash,
            business_domain: context.business_domain.clone(),
            effect_hash,
        }
    }
    
    fn is_valid(&self, key: &PIRCacheKey, context: &PIRCacheContext) -> bool {
        // Check semantic version compatibility
        if self.config.enable_semantic_invalidation {
            if let Some(cached_semantic_hash) = key.semantic_hash {
                let mut hasher = rustc_hash::FxHasher::default();
                context.semantic_version.hash(&mut hasher);
                let current_semantic_hash = hasher.finish();
                
                if cached_semantic_hash != current_semantic_hash {
                    return false;
                }
            }
        }
        
        // Check business domain compatibility
        if self.config.enable_business_context_caching {
            if key.business_domain != context.business_domain {
                return false;
            }
        }
        
        // Check effect context compatibility
        if let (Some(cached_effect_hash), Some(current_effect_ctx)) = (key.effect_hash, &context.effect_context) {
            let mut hasher = rustc_hash::FxHasher::default();
            current_effect_ctx.hash(&mut hasher);
            let current_effect_hash = hasher.finish();
            
            if cached_effect_hash != current_effect_hash {
                return false;
            }
        }
        
        true
    }
    
    fn invalidation_triggers(&self, key: &PIRCacheKey) -> HashSet<PIRInvalidationTrigger> {
        let mut triggers = HashSet::new();
        
        // Add base compiler triggers
        triggers.insert(PIRInvalidationTrigger::Compiler(InvalidationTrigger::ConfigChanged));
        triggers.insert(PIRInvalidationTrigger::Compiler(InvalidationTrigger::OptimizationLevelChanged));
        
        // Add PIR-specific triggers
        if self.config.enable_semantic_invalidation {
            triggers.insert(PIRInvalidationTrigger::SemanticModelChanged);
        }
        
        if let Some(ref domain) = key.business_domain {
            triggers.insert(PIRInvalidationTrigger::BusinessDomainUpdated(domain.clone()));
        }
        
        if key.effect_hash.is_some() {
            triggers.insert(PIRInvalidationTrigger::EffectCapabilitiesChanged);
        }
        
        triggers.insert(PIRInvalidationTrigger::PIRSchemaUpdated);
        triggers.insert(PIRInvalidationTrigger::CohesionParametersChanged);
        
        triggers
    }
    
    fn cache_priority(&self, key: &PIRCacheKey, usage_stats: &CacheUsageStats) -> f64 {
        let mut priority = 0.0;
        
        // Higher priority for frequently accessed items
        let hit_rate = if usage_stats.hit_count + usage_stats.miss_count > 0 {
            usage_stats.hit_count as f64 / (usage_stats.hit_count + usage_stats.miss_count) as f64
        } else {
            0.0
        };
        priority += hit_rate * 100.0;
        
        // Higher priority for expensive-to-compute items
        let computation_cost = usage_stats.avg_computation_time.as_secs_f64();
        priority += computation_cost * 10.0;
        
        // Lower priority for large memory consumers
        let memory_cost = usage_stats.memory_size_bytes as f64 / (1024.0 * 1024.0); // MB
        priority -= memory_cost * 5.0;
        
        // Higher priority for business-context-aware items
        if self.config.enable_business_context_caching && key.business_domain.is_some() {
            priority += 25.0;
        }
        
        // Higher priority for semantic-aware items
        if self.config.enable_semantic_invalidation && key.semantic_hash.is_some() {
            priority += 20.0;
        }
        
        priority.max(0.0)
    }
}

impl PIRCacheManager {
    /// Create a new PIR cache manager
    pub fn new(strategy: Box<dyn PIRCacheStrategy>, config: PIRCacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            usage_stats: HashMap::new(),
            strategy,
            config,
        }
    }
    
    /// Create with default strategy
    pub fn with_default_strategy(config: PIRCacheConfig) -> Self {
        let strategy = Box::new(DefaultPIRCacheStrategy::new(config.clone()));
        Self::new(strategy, config)
    }
    
    /// Get cached PIR result
    pub fn get(&mut self, key: &PIRCacheKey, context: &PIRCacheContext) -> Option<CachedPIRResult> {
        // Check if cache entry is valid
        if !self.strategy.is_valid(key, context) {
            self.cache.remove(key);
            return None;
        }
        
        // Update usage statistics
        if let Some(stats) = self.usage_stats.get_mut(key) {
            stats.hit_count += 1;
            stats.last_accessed = SystemTime::now();
        }
        
        self.cache.get(key).cloned()
    }
    
    /// Store PIR result in cache
    pub fn put(&mut self, key: PIRCacheKey, result: CachedPIRResult) -> PIRResult<()> {
        // Check cache size limits
        self.enforce_cache_limits()?;
        
        // Store the result
        self.cache.insert(key.clone(), result);
        
        // Initialize usage statistics
        self.usage_stats.insert(key, CacheUsageStats {
            hit_count: 0,
            miss_count: 1,
            last_accessed: SystemTime::now(),
            avg_computation_time: Duration::from_millis(100), // Default estimate
            memory_size_bytes: 1024, // Default estimate
        });
        
        Ok(())
    }
    
    /// Invalidate cache entries based on trigger
    pub fn invalidate(&mut self, trigger: &PIRInvalidationTrigger) {
        let keys_to_remove: Vec<PIRCacheKey> = self.cache.keys()
            .filter(|key| {
                let triggers = self.strategy.invalidation_triggers(key);
                triggers.contains(trigger)
            })
            .cloned()
            .collect();
        
        for key in keys_to_remove {
            self.cache.remove(&key);
            self.usage_stats.remove(&key);
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> PIRCacheStats {
        let total_hits: u64 = self.usage_stats.values().map(|s| s.hit_count).sum();
        let total_misses: u64 = self.usage_stats.values().map(|s| s.miss_count).sum();
        let total_memory: u64 = self.usage_stats.values().map(|s| s.memory_size_bytes).sum();
        
        PIRCacheStats {
            entry_count: self.cache.len(),
            total_hits,
            total_misses,
            hit_rate: if total_hits + total_misses > 0 {
                total_hits as f64 / (total_hits + total_misses) as f64
            } else {
                0.0
            },
            memory_usage_bytes: total_memory,
        }
    }
    
    /// Enforce cache size limits
    fn enforce_cache_limits(&mut self) -> PIRResult<()> {
        let max_size_bytes = self.config.max_size_mb * 1024 * 1024;
        let current_size: u64 = self.usage_stats.values().map(|s| s.memory_size_bytes).sum();
        
        if current_size > max_size_bytes as u64 {
            // Remove lowest priority items
            let mut priorities: Vec<(PIRCacheKey, f64)> = self.cache.keys()
                .map(|key| {
                    let stats = self.usage_stats.get(key).unwrap();
                    let priority = self.strategy.cache_priority(key, stats);
                    (key.clone(), priority)
                })
                .collect();
            
            priorities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Remove lowest priority items until under size limit
            let mut removed_size = 0u64;
            for (key, _) in priorities {
                if current_size - removed_size <= max_size_bytes as u64 {
                    break;
                }
                
                if let Some(stats) = self.usage_stats.remove(&key) {
                    removed_size += stats.memory_size_bytes;
                }
                self.cache.remove(&key);
            }
        }
        
        Ok(())
    }
}

/// PIR cache statistics
#[derive(Debug, Clone)]
pub struct PIRCacheStats {
    /// Number of cache entries
    pub entry_count: usize,
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
}

impl PIRCacheContext {
    /// Create a new PIR cache context
    pub fn new(semantic_version: String) -> Self {
        Self {
            semantic_version,
            business_domain: None,
            effect_context: None,
            optimization_level: 1,
            timestamp: SystemTime::now(),
        }
    }
    
    /// Set business domain context
    pub fn with_business_domain(mut self, domain: String) -> Self {
        self.business_domain = Some(domain);
        self
    }
    
    /// Set effect context
    pub fn with_effect_context(mut self, effect_context: String) -> Self {
        self.effect_context = Some(effect_context);
        self
    }
    
    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level;
        self
    }
} 