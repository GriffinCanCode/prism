//! Core Query Infrastructure
//!
//! This module provides the fundamental query system infrastructure including
//! traits, types, and the query engine that powers the modular query subsystem.

use crate::error::{CompilerError, CompilerResult};
use prism_common::{NodeId, span::Span};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::path::PathBuf;
use async_trait::async_trait;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Serialize, Deserialize};

/// Unique identifier for queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryId(u64);

impl QueryId {
    /// Create a new query ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Cache key for query results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Query type identifier
    pub query_type: String,
    /// Input hash
    pub input_hash: u64,
    /// Semantic context hash
    pub semantic_hash: Option<u64>,
    /// Compiler version
    pub compiler_version: String,
    /// Target configuration
    pub target_config: Option<String>,
}

impl CacheKey {
    /// Create a cache key from input
    pub fn from_input<T: Hash>(query_type: &str, input: &T) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        input.hash(&mut hasher);
        let input_hash = hasher.finish();

        Self {
            query_type: query_type.to_string(),
            input_hash,
            semantic_hash: None,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_config: None,
        }
    }

    /// Add semantic context to cache key
    pub fn with_semantic_context<T: Hash>(mut self, context: &T) -> Self {
        let mut hasher = rustc_hash::FxHasher::default();
        context.hash(&mut hasher);
        self.semantic_hash = Some(hasher.finish());
        self
    }

    /// Add target configuration to cache key
    pub fn with_target_config(mut self, config: &str) -> Self {
        self.target_config = Some(config.to_string());
        self
    }
}

/// Invalidation trigger for cache entries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InvalidationTrigger {
    /// File was modified
    FileChanged(PathBuf),
    /// Dependency was updated
    DependencyUpdated(String),
    /// Configuration changed
    ConfigChanged,
    /// Semantic context changed
    SemanticContextChanged(NodeId),
    /// Manual invalidation
    Manual(String),
    /// Optimization level changed
    OptimizationLevelChanged,
}

/// Query context for execution
#[derive(Debug, Clone)]
pub struct QueryContext {
    /// Current query stack (for dependency tracking)
    pub query_stack: Vec<QueryId>,
    /// Semantic context
    pub semantic_context: Arc<dyn SemanticContext>,
    /// Configuration
    pub config: QueryConfig,
    /// Performance profiler
    pub profiler: Arc<Mutex<QueryProfiler>>,
    /// Compilation context (for real integration)
    pub compilation_context: Option<Arc<crate::context::CompilationContext>>,
    /// Semantic type integration (for real integration)
    pub semantic_type_integration: Option<Arc<crate::semantic::SemanticTypeIntegration>>,
}

impl QueryContext {
    /// Get the compilation context
    pub fn get_compilation_context(&self) -> Option<Arc<crate::context::CompilationContext>> {
        self.compilation_context.clone()
    }
    
    /// Get the semantic type integration
    pub fn get_semantic_type_integration(&self) -> Option<Arc<crate::semantic::SemanticTypeIntegration>> {
        self.semantic_type_integration.clone()
    }
}

/// Semantic context interface
pub trait SemanticContext: Send + Sync {
    /// Get type information
    fn get_type_info(&self, symbol: &str) -> Option<TypeInfo>;
    /// Get effect information
    fn get_effect_info(&self, symbol: &str) -> Option<EffectInfo>;
    /// Get semantic hash for caching
    fn get_semantic_hash(&self) -> u64;
}

/// Type information for semantic context
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type constraints
    pub constraints: Vec<String>,
    /// Semantic metadata
    pub metadata: HashMap<String, String>,
}

/// Effect information for semantic context
#[derive(Debug, Clone)]
pub struct EffectInfo {
    /// Effect type
    pub effect_type: String,
    /// Capabilities required
    pub capabilities: Vec<String>,
    /// Side effects
    pub side_effects: Vec<String>,
}

/// Query configuration
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Enable caching
    pub enable_cache: bool,
    /// Enable dependency tracking
    pub enable_dependency_tracking: bool,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Query timeout
    pub query_timeout: Duration,
}

/// Query profiler for performance analysis
#[derive(Debug)]
pub struct QueryProfiler {
    /// Query execution times
    pub execution_times: HashMap<String, Vec<Duration>>,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, CacheStats>,
    /// Dependency graph
    pub dependency_graph: HashMap<QueryId, HashSet<QueryId>>,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total queries
    pub total_queries: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_queries as f64
        }
    }
}

/// Core query trait that all compiler queries must implement
#[async_trait]
pub trait CompilerQuery<Input, Output>: Send + Sync
where
    Input: Send + Sync + Clone + Hash,
    Output: Send + Sync + Clone,
{
    /// Execute the query
    async fn execute(&self, input: Input, context: QueryContext) -> CompilerResult<Output>;

    /// Generate cache key for the input
    fn cache_key(&self, input: &Input) -> CacheKey;

    /// Get dependencies for this query
    async fn dependencies(&self, input: &Input, context: &QueryContext) -> CompilerResult<HashSet<QueryId>>;

    /// Get invalidation triggers
    fn invalidate_on(&self, input: &Input) -> HashSet<InvalidationTrigger>;

    /// Get query type name for profiling
    fn query_type(&self) -> &'static str;
}

/// Query engine for incremental compilation
#[derive(Debug)]
pub struct QueryEngine {
    /// Query cache
    cache: Arc<RwLock<FxHashMap<CacheKey, CachedResult>>>,
    /// Dependency tracker
    dependency_tracker: Arc<Mutex<DependencyTracker>>,
    /// Invalidation tracker
    invalidation_tracker: Arc<Mutex<InvalidationTracker>>,
    /// Configuration
    config: QueryConfig,
    /// Profiler
    profiler: Arc<Mutex<QueryProfiler>>,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedResult {
    /// Cached value (serialized)
    value: Vec<u8>,
    /// Dependencies
    dependencies: HashSet<QueryId>,
    /// Invalidation triggers
    invalidation_triggers: HashSet<InvalidationTrigger>,
    /// Cache timestamp
    timestamp: Instant,
    /// Access count
    access_count: u64,
}

/// Dependency tracker
#[derive(Debug)]
struct DependencyTracker {
    /// Forward dependencies (query -> dependencies)
    forward_deps: HashMap<QueryId, HashSet<QueryId>>,
    /// Reverse dependencies (dependency -> queries that depend on it)
    reverse_deps: HashMap<QueryId, HashSet<QueryId>>,
}

/// Invalidation tracker
#[derive(Debug)]
struct InvalidationTracker {
    /// Trigger to queries mapping
    trigger_to_queries: HashMap<InvalidationTrigger, HashSet<QueryId>>,
    /// Query to triggers mapping
    query_to_triggers: HashMap<QueryId, HashSet<InvalidationTrigger>>,
}

impl QueryEngine {
    /// Create a new query engine with default configuration
    pub fn new() -> Self {
        let query_config = QueryConfig {
            enable_cache: true,
            enable_dependency_tracking: true,
            enable_profiling: true,
            cache_size_limit: 10_000,
            query_timeout: Duration::from_secs(30),
        };

        Self {
            cache: Arc::new(RwLock::new(FxHashMap::default())),
            dependency_tracker: Arc::new(Mutex::new(DependencyTracker::new())),
            invalidation_tracker: Arc::new(Mutex::new(InvalidationTracker::new())),
            config: query_config,
            profiler: Arc::new(Mutex::new(QueryProfiler::new())),
        }
    }

    /// Create a new query engine with custom configuration
    pub fn with_config(config: QueryConfig) -> CompilerResult<Self> {
        Ok(Self {
            cache: Arc::new(RwLock::new(FxHashMap::default())),
            dependency_tracker: Arc::new(Mutex::new(DependencyTracker::new())),
            invalidation_tracker: Arc::new(Mutex::new(InvalidationTracker::new())),
            config,
            profiler: Arc::new(Mutex::new(QueryProfiler::new())),
        })
    }

    /// Execute a query with caching and dependency tracking
    pub async fn query<Q, I, O>(&self, query: &Q, input: I, context: QueryContext) -> CompilerResult<O>
    where
        Q: CompilerQuery<I, O>,
        I: Send + Sync + Clone + Hash,
        O: Send + Sync + Clone + serde::Serialize + for<'de> serde::Deserialize<'de>,
    {
        let start_time = Instant::now();
        let query_id = QueryId::new();
        let query_type = query.query_type();

        // Update profiler
        if self.config.enable_profiling {
            let mut profiler = self.profiler.lock().unwrap();
            profiler.record_query_start(query_type);
        }

        // Generate cache key
        let cache_key = query.cache_key(&input);

        // Check cache first
        if self.config.enable_cache {
            if let Some(cached_result) = self.get_cached_result::<O>(&cache_key)? {
                let execution_time = start_time.elapsed();
                self.record_cache_hit(query_type, execution_time);
                return Ok(cached_result);
            }
        }

        // Execute query
        let result = query.execute(input.clone(), context.clone()).await?;

        // Track dependencies
        if self.config.enable_dependency_tracking {
            let dependencies = query.dependencies(&input, &context).await?;
            let invalidation_triggers = query.invalidate_on(&input);

            self.update_dependencies(query_id, dependencies)?;
            self.update_invalidation_triggers(query_id, invalidation_triggers)?;
        }

        // Cache result
        if self.config.enable_cache {
            self.cache_result(&cache_key, &result, query_id)?;
        }

        let execution_time = start_time.elapsed();
        self.record_cache_miss(query_type, execution_time);

        Ok(result)
    }

    /// Invalidate queries based on trigger
    pub fn invalidate(&self, trigger: InvalidationTrigger) -> CompilerResult<usize> {
        let mut invalidated_count = 0;

        // Get affected queries
        let affected_queries = {
            let invalidation_tracker = self.invalidation_tracker.lock().unwrap();
            invalidation_tracker.get_affected_queries(&trigger)
        };

        // Remove from cache
        {
            let mut cache = self.cache.write().unwrap();
            for _query_id in &affected_queries {
                // Find cache entries for this query (simplified - in practice we'd need better mapping)
                let keys_to_remove: Vec<_> = cache
                    .keys()
                    .filter(|_key| {
                        // In a real implementation, we'd have a mapping from query_id to cache_key
                        // For now, we'll invalidate more broadly
                        true
                    })
                    .cloned()
                    .collect();

                for key in keys_to_remove {
                    cache.remove(&key);
                    invalidated_count += 1;
                }
            }
        }

        // Update dependency tracker
        if self.config.enable_dependency_tracking {
            let mut dependency_tracker = self.dependency_tracker.lock().unwrap();
            dependency_tracker.invalidate_queries(&affected_queries);
        }

        Ok(invalidated_count)
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, CacheStats> {
        let profiler = self.profiler.lock().unwrap();
        profiler.cache_hit_rates.clone()
    }

    /// Get dependency graph
    pub fn get_dependency_graph(&self) -> HashMap<QueryId, HashSet<QueryId>> {
        let profiler = self.profiler.lock().unwrap();
        profiler.dependency_graph.clone()
    }

    /// Clear all caches
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    // Private helper methods
    fn get_cached_result<O>(&self, cache_key: &CacheKey) -> CompilerResult<Option<O>>
    where
        O: for<'de> serde::Deserialize<'de>,
    {
        let cache = self.cache.read().unwrap();
        
        if let Some(cached_result) = cache.get(cache_key) {
            // Deserialize the cached value
            let result: O = bincode::deserialize(&cached_result.value)
                .map_err(|e| CompilerError::CacheDeserializationError(e.to_string()))?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn cache_result<O>(&self, cache_key: &CacheKey, result: &O, query_id: QueryId) -> CompilerResult<()>
    where
        O: serde::Serialize,
    {
        let serialized = bincode::serialize(result)
            .map_err(|e| CompilerError::CacheSerializationError(e.to_string()))?;

        let cached_result = CachedResult {
            value: serialized,
            dependencies: HashSet::new(), // Will be updated by dependency tracker
            invalidation_triggers: HashSet::new(), // Will be updated by invalidation tracker
            timestamp: Instant::now(),
            access_count: 0,
        };

        let mut cache = self.cache.write().unwrap();
        
        // Check cache size limit
        if cache.len() >= self.config.cache_size_limit {
            self.evict_cache_entries(&mut cache);
        }

        cache.insert(cache_key.clone(), cached_result);
        Ok(())
    }

    fn evict_cache_entries(&self, cache: &mut FxHashMap<CacheKey, CachedResult>) {
        // Simple LRU eviction - remove oldest 10% of entries
        let evict_count = cache.len() / 10;
        
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, result)| result.timestamp);
        
        for (key, _) in entries.into_iter().take(evict_count) {
            cache.remove(key);
        }
    }

    fn update_dependencies(&self, query_id: QueryId, dependencies: HashSet<QueryId>) -> CompilerResult<()> {
        let mut dependency_tracker = self.dependency_tracker.lock().unwrap();
        dependency_tracker.add_dependencies(query_id, dependencies);
        Ok(())
    }

    fn update_invalidation_triggers(&self, query_id: QueryId, triggers: HashSet<InvalidationTrigger>) -> CompilerResult<()> {
        let mut invalidation_tracker = self.invalidation_tracker.lock().unwrap();
        invalidation_tracker.add_triggers(query_id, triggers);
        Ok(())
    }

    fn record_cache_hit(&self, query_type: &str, execution_time: Duration) {
        if let Ok(mut profiler) = self.profiler.lock() {
            profiler.record_cache_hit(query_type, execution_time);
        }
    }

    fn record_cache_miss(&self, query_type: &str, execution_time: Duration) {
        if let Ok(mut profiler) = self.profiler.lock() {
            profiler.record_cache_miss(query_type, execution_time);
        }
    }
}

impl DependencyTracker {
    fn new() -> Self {
        Self {
            forward_deps: HashMap::new(),
            reverse_deps: HashMap::new(),
        }
    }

    fn add_dependencies(&mut self, query_id: QueryId, dependencies: HashSet<QueryId>) {
        // Update forward dependencies
        self.forward_deps.insert(query_id, dependencies.clone());

        // Update reverse dependencies
        for dep_id in dependencies {
            self.reverse_deps
                .entry(dep_id)
                .or_insert_with(HashSet::new)
                .insert(query_id);
        }
    }

    fn invalidate_queries(&mut self, query_ids: &HashSet<QueryId>) {
        for query_id in query_ids {
            self.forward_deps.remove(query_id);
            
            // Remove from reverse dependencies
            for (_, dependents) in self.reverse_deps.iter_mut() {
                dependents.remove(query_id);
            }
        }
    }
}

impl InvalidationTracker {
    fn new() -> Self {
        Self {
            trigger_to_queries: HashMap::new(),
            query_to_triggers: HashMap::new(),
        }
    }

    fn add_triggers(&mut self, query_id: QueryId, triggers: HashSet<InvalidationTrigger>) {
        // Update query to triggers mapping
        self.query_to_triggers.insert(query_id, triggers.clone());

        // Update trigger to queries mapping
        for trigger in triggers {
            self.trigger_to_queries
                .entry(trigger)
                .or_insert_with(HashSet::new)
                .insert(query_id);
        }
    }

    fn get_affected_queries(&self, trigger: &InvalidationTrigger) -> HashSet<QueryId> {
        self.trigger_to_queries
            .get(trigger)
            .cloned()
            .unwrap_or_default()
    }
}

impl QueryProfiler {
    /// Create a new query profiler
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
            cache_hit_rates: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Record query start
    pub fn record_query_start(&mut self, _query_type: &str) {
        // Implementation for recording query start
    }

    /// Record cache hit
    pub fn record_cache_hit(&mut self, query_type: &str, _execution_time: Duration) {
        let stats = self.cache_hit_rates.entry(query_type.to_string()).or_insert_with(|| CacheStats {
            total_queries: 0,
            cache_hits: 0,
            cache_misses: 0,
        });
        stats.total_queries += 1;
        stats.cache_hits += 1;
    }

    /// Record cache miss
    pub fn record_cache_miss(&mut self, query_type: &str, _execution_time: Duration) {
        let stats = self.cache_hit_rates.entry(query_type.to_string()).or_insert_with(|| CacheStats {
            total_queries: 0,
            cache_hits: 0,
            cache_misses: 0,
        });
        stats.total_queries += 1;
        stats.cache_misses += 1;
    }
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            enable_dependency_tracking: true,
            enable_profiling: true,
            cache_size_limit: 10_000,
            query_timeout: Duration::from_secs(30),
        }
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryId({})", self.0)
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Default semantic context implementation
#[derive(Debug, Default)]
pub struct DefaultSemanticContext;

impl SemanticContext for DefaultSemanticContext {
    fn get_type_info(&self, _symbol: &str) -> Option<TypeInfo> {
        None
    }
    
    fn get_effect_info(&self, _symbol: &str) -> Option<EffectInfo> {
        None
    }
    
    fn get_semantic_hash(&self) -> u64 {
        0
    }
} 