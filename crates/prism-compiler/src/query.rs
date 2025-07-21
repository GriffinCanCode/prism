//! Query-based compilation engine for incremental compilation
//!
//! This module implements the core query system that enables fast incremental
//! compilation by computing results on-demand and caching them for reuse.

use crate::error::{CompilerError, CompilerResult};
use prism_common::{NodeId, span::Span};
use prism_ast::Program;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::path::PathBuf;
use async_trait::async_trait;
use rustc_hash::{FxHashMap, FxHashSet};
use prism_ast::{Program, TransformationEngine, TransformationConfig, TransformationResult};

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
    FileChanged(std::path::PathBuf),
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
            for query_id in &affected_queries {
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

/// Parse file query - converts source file to AST
#[derive(Debug, Clone)]
pub struct ParseFileQuery;

#[async_trait]
impl CompilerQuery<PathBuf, Program> for ParseFileQuery {
    async fn execute(&self, file_path: PathBuf, context: QueryContext) -> CompilerResult<Program> {
        // Read source file
        let source = std::fs::read_to_string(&file_path)
            .map_err(|e| CompilerError::FileReadError { 
                path: file_path.clone(), 
                source: e 
            })?;

        // Tokenize
        let mut symbol_table = prism_common::symbol::SymbolTable::new();
        let source_id = prism_common::SourceId::new(1);
        let lexer_config = prism_lexer::LexerConfig::default();
        
        let lexer = prism_lexer::SemanticLexer::new(&source, source_id, &mut symbol_table, lexer_config);
        let lex_result = lexer.tokenize_with_semantics();
        
        if !lex_result.errors.is_empty() {
            return Err(CompilerError::LexError {
                message: format!("Lexing errors: {:?}", lex_result.errors),
                location: lex_result.errors[0].span,
            });
        }

        // Parse
        let parse_config = prism_parser::ParseConfig {
            aggressive_recovery: true,
            extract_ai_context: true,
            max_errors: 100,
            semantic_metadata: true,
        };
        
        let program = prism_parser::parse_program_with_config(lex_result.tokens, parse_config)
            .map_err(|e| CompilerError::ParseError {
                message: e.message,
                location: e.span,
                suggestions: e.suggestions,
            })?;

        Ok(program)
    }

    fn cache_key(&self, input: &PathBuf) -> CacheKey {
        CacheKey::from_input("parse_file", input)
            .with_target_config("prism_v1")
    }

    async fn dependencies(&self, _input: &PathBuf, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Parsing has no dependencies
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, input: &PathBuf) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::FileChanged(input.clone()));
        triggers
    }

    fn query_type(&self) -> &'static str {
        "parse_file"
    }
}

/// Semantic analysis query - analyzes AST for semantic information
#[derive(Debug, Clone)]
pub struct SemanticAnalysisQuery {
    semantic_engine: Arc<prism_semantic::SemanticEngine>,
}

impl SemanticAnalysisQuery {
    pub fn new(semantic_engine: Arc<prism_semantic::SemanticEngine>) -> Self {
        Self { semantic_engine }
    }
}

#[async_trait]
impl CompilerQuery<Program, prism_semantic::SemanticInfo> for SemanticAnalysisQuery {
    async fn execute(&self, program: Program, _context: QueryContext) -> CompilerResult<prism_semantic::SemanticInfo> {
        // Use the centralized semantic engine
        // Note: This would need to be made async-compatible in the actual implementation
        let mut semantic_engine = (*self.semantic_engine).clone(); // This won't work as-is, needs proper async design
        let semantic_info = semantic_engine.analyze_program(&program)
            .map_err(|e| CompilerError::SemanticAnalysisError { message: e.to_string() })?;

        Ok(semantic_info)
    }

    fn cache_key(&self, input: &Program) -> CacheKey {
        let mut hasher = rustc_hash::FxHasher::default();
        // Hash program structure for cache key
        input.items.len().hash(&mut hasher);
        let program_hash = hasher.finish();
        
        CacheKey {
            query_type: "semantic_analysis".to_string(),
            input_hash: program_hash,
            semantic_hash: None,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_config: None,
        }
    }

    async fn dependencies(&self, _input: &Program, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Semantic analysis depends on parsing (handled by orchestrator)
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &Program) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::SemanticContextChanged(NodeId(0)));
        triggers
    }

    fn query_type(&self) -> &'static str {
        "semantic_analysis"
    }
}

impl SemanticAnalysisQuery {
    async fn analyze_program_semantics(&self, program: &Program) -> CompilerResult<crate::semantic::SemanticInfo> {
        use crate::semantic::*;
        use std::collections::HashMap;

        let mut symbols = HashMap::new();
        let mut types = HashMap::new();
        let mut effects = HashMap::new();
        let mut contracts = HashMap::new();

        // Analyze each item in the program
        for item in &program.items {
            match &item.kind {
                prism_ast::Item::Function(func_decl) => {
                    self.analyze_function(func_decl, &mut symbols, &mut types, &mut effects, &mut contracts)?;
                }
                prism_ast::Item::Module(module_decl) => {
                    self.analyze_module(module_decl, &mut symbols, &mut types)?;
                }
                prism_ast::Item::Type(type_decl) => {
                    self.analyze_type_declaration(type_decl, &mut symbols, &mut types)?;
                }
                _ => {
                    // Handle other item types
                }
            }
        }

        // Generate AI metadata
        let ai_metadata = self.generate_ai_metadata(&symbols, &types, &effects)?;

        Ok(SemanticInfo {
            symbols,
            types,
            effects,
            contracts,
            ai_metadata,
        })
    }

    fn analyze_function(
        &self,
        func_decl: &prism_ast::FunctionDecl,
        symbols: &mut HashMap<prism_common::symbol::Symbol, SymbolInfo>,
        types: &mut HashMap<NodeId, TypeInfo>,
        effects: &mut HashMap<NodeId, EffectSignature>,
        contracts: &mut HashMap<NodeId, ContractSpecification>,
    ) -> CompilerResult<()> {
        use crate::semantic::*;

        // Create symbol info for function
        let symbol = func_decl.name;
        let symbol_info = SymbolInfo {
            id: symbol,
            name: symbol.as_str().to_string(),
            type_info: TypeInfo {
                type_id: NodeId(func_decl.id.0),
                type_kind: TypeKind::Function(FunctionType {
                    parameters: func_decl.parameters.iter().map(|param| {
                        ParameterInfo {
                            name: param.name.as_str().to_string(),
                            param_type: TypeInfo {
                                type_id: NodeId(0), // Would be properly computed
                                type_kind: TypeKind::Primitive(PrimitiveType::String), // Placeholder
                                type_parameters: Vec::new(),
                                constraints: Vec::new(),
                                semantic_meaning: SemanticMeaning {
                                    domain: "Function Parameter".to_string(),
                                    purpose: "Function input".to_string(),
                                    related_concepts: Vec::new(),
                                    business_entities: Vec::new(),
                                },
                                ai_description: Some("Function parameter".to_string()),
                            },
                            default_value: param.default_value.as_ref().map(|_| "default".to_string()),
                            attributes: Vec::new(),
                        }
                    }).collect(),
                    return_type: Box::new(TypeInfo {
                        type_id: NodeId(0),
                        type_kind: TypeKind::Primitive(PrimitiveType::Unit),
                        type_parameters: Vec::new(),
                        constraints: Vec::new(),
                        semantic_meaning: SemanticMeaning {
                            domain: "Function Return".to_string(),
                            purpose: "Function output".to_string(),
                            related_concepts: Vec::new(),
                            business_entities: Vec::new(),
                        },
                        ai_description: Some("Function return type".to_string()),
                    }),
                    effects: EffectSignature {
                        function_id: NodeId(func_decl.id.0),
                        input_effects: HashSet::new(),
                        output_effects: HashSet::new(),
                        effect_dependencies: HashSet::new(),
                        capability_requirements: HashSet::new(),
                    },
                    contracts: ContractSpecification {
                        preconditions: Vec::new(),
                        postconditions: Vec::new(),
                        invariants: Vec::new(),
                    },
                }),
                type_parameters: Vec::new(),
                constraints: Vec::new(),
                semantic_meaning: SemanticMeaning {
                    domain: "Function".to_string(),
                    purpose: func_decl.description.clone().unwrap_or_else(|| "Function definition".to_string()),
                    related_concepts: Vec::new(),
                    business_entities: Vec::new(),
                },
                ai_description: func_decl.ai_context.clone(),
            },
            source_location: func_decl.span,
            visibility: match func_decl.visibility {
                prism_ast::Visibility::Public => Visibility::Public,
                prism_ast::Visibility::Private => Visibility::Private,
                prism_ast::Visibility::Internal => Visibility::Internal,
            },
            semantic_annotations: Vec::new(),
            business_context: None,
            ai_hints: Vec::new(),
        };

        symbols.insert(symbol, symbol_info);
        Ok(())
    }

    fn analyze_module(
        &self,
        module_decl: &prism_ast::ModuleDecl,
        symbols: &mut HashMap<prism_common::symbol::Symbol, SymbolInfo>,
        types: &mut HashMap<NodeId, TypeInfo>,
    ) -> CompilerResult<()> {
        use crate::semantic::*;

        let symbol = module_decl.name;
        let symbol_info = SymbolInfo {
            id: symbol,
            name: symbol.as_str().to_string(),
            type_info: TypeInfo {
                type_id: NodeId(0),
                type_kind: TypeKind::Primitive(PrimitiveType::Unit), // Modules don't have runtime types
                type_parameters: Vec::new(),
                constraints: Vec::new(),
                semantic_meaning: SemanticMeaning {
                    domain: "Module System".to_string(),
                    purpose: module_decl.description.clone().unwrap_or_else(|| "Module definition".to_string()),
                    related_concepts: Vec::new(),
                    business_entities: Vec::new(),
                },
                ai_description: module_decl.ai_context.clone(),
            },
            source_location: Span::dummy(), // Would be properly computed
            visibility: match module_decl.visibility {
                prism_ast::Visibility::Public => Visibility::Public,
                prism_ast::Visibility::Private => Visibility::Private,
                prism_ast::Visibility::Internal => Visibility::Internal,
            },
            semantic_annotations: Vec::new(),
            business_context: None,
            ai_hints: Vec::new(),
        };

        symbols.insert(symbol, symbol_info);
        Ok(())
    }

    fn analyze_type_declaration(
        &self,
        _type_decl: &prism_ast::TypeDecl,
        _symbols: &mut HashMap<prism_common::symbol::Symbol, SymbolInfo>,
        _types: &mut HashMap<NodeId, TypeInfo>,
    ) -> CompilerResult<()> {
        // TODO: Implement type declaration analysis
        Ok(())
    }

    fn generate_ai_metadata(
        &self,
        symbols: &HashMap<prism_common::symbol::Symbol, SymbolInfo>,
        types: &HashMap<NodeId, TypeInfo>,
        effects: &HashMap<NodeId, EffectSignature>,
    ) -> CompilerResult<AIMetadata> {
        use crate::semantic::*;

        Ok(AIMetadata {
            module_context: Some(ModuleAIContext {
                purpose: "Prism module with semantic analysis".to_string(),
                capabilities: vec!["Type checking".to_string(), "Effect analysis".to_string()],
                responsibilities: vec!["Semantic validation".to_string()],
                dependencies: Vec::new(),
                business_domain: "Software Development".to_string(),
                compliance_requirements: Vec::new(),
                security_considerations: Vec::new(),
            }),
            function_contexts: HashMap::new(),
            type_contexts: HashMap::new(),
            relationships: SemanticRelationships {
                type_relationships: HashMap::new(),
                function_relationships: HashMap::new(),
                module_relationships: HashMap::new(),
            },
            business_context: None,
            performance_metadata: PerformanceMetadata {
                characteristics: HashMap::new(),
                optimization_opportunities: Vec::new(),
                bottlenecks: Vec::new(),
                resource_patterns: Vec::new(),
                parallelization_hints: Vec::new(),
            },
            security_metadata: SecurityMetadata {
                implications: HashMap::new(),
                vulnerabilities: Vec::new(),
                best_practices: Vec::new(),
                threat_model: None,
            },
            compliance_metadata: ComplianceMetadata {
                requirements: HashMap::new(),
                frameworks: Vec::new(),
                audit_trails: Vec::new(),
                compliance_status: ComplianceStatus::NotApplicable,
            },
            consistency_metadata: ConsistencyMetadata {
                cross_target_consistency: CrossTargetConsistency {
                    type_consistency: TypeConsistency {
                        semantic_types_consistent: true,
                        business_rules_consistent: true,
                        validation_consistent: true,
                        inconsistencies: Vec::new(),
                    },
                    behavior_consistency: BehaviorConsistency {
                        effects_consistent: true,
                        contracts_consistent: true,
                        performance_consistent: true,
                        inconsistencies: Vec::new(),
                    },
                    semantic_consistency: SemanticConsistency {
                        business_rules_preserved: true,
                        validation_preserved: true,
                        meaning_preserved: true,
                        inconsistencies: Vec::new(),
                    },
                    type_preservation: TypePreservation {
                        semantic_types_preserved: true,
                        business_rules_preserved: true,
                        validation_predicates_preserved: true,
                        type_mappings: HashMap::new(),
                    },
                },
            },
            comprehension_aids: ComprehensionAids {
                concept_maps: Vec::new(),
                learning_paths: Vec::new(),
                examples: Vec::new(),
                anti_patterns: Vec::new(),
                best_practices: Vec::new(),
            },
        })
    }
}

/// Code generation query - generates code for specific target
#[derive(Debug, Clone)]
pub struct CodeGenQuery {
    target: crate::context::CompilationTarget,
    codegen: Arc<prism_codegen::MultiTargetCodeGen>,
}

impl CodeGenQuery {
    pub fn new(target: crate::context::CompilationTarget, codegen: Arc<prism_codegen::MultiTargetCodeGen>) -> Self {
        Self { target, codegen }
    }
}

#[async_trait]
impl CompilerQuery<(Program, crate::semantic::SemanticInfo), prism_codegen::CodeArtifact> for CodeGenQuery {
    async fn execute(&self, input: (Program, crate::semantic::SemanticInfo), _context: QueryContext) -> CompilerResult<prism_codegen::CodeArtifact> {
        let (program, _semantic_info) = input;
        
        // Create compilation context
        let compilation_context = crate::context::CompilationContext {
            targets: vec![self.target],
            current_phase: crate::context::CompilationPhase::CodeGeneration,
            diagnostics: crate::context::DiagnosticCollector::new(),
            profiler: crate::context::PerformanceProfiler::new(),
            ai_metadata_collector: crate::context::AIMetadataCollector::new(true),
            project_config: crate::context::ProjectConfig::default(),
        };

        // Generate code
        let config = prism_codegen::CodeGenConfig::default();
        let artifact = self.codegen.generate_target(self.target, &program, &compilation_context, &config)
            .await
            .map_err(|e| CompilerError::CodeGenError {
                target: format!("{:?}", self.target),
                message: e.to_string(),
            })?;

        Ok(artifact)
    }

    fn cache_key(&self, input: &(Program, crate::semantic::SemanticInfo)) -> CacheKey {
        let (program, _) = input;
        let mut hasher = rustc_hash::FxHasher::default();
        program.items.len().hash(&mut hasher);
        let program_hash = hasher.finish();
        
        CacheKey {
            query_type: "code_generation".to_string(),
            input_hash: program_hash,
            semantic_hash: None,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_config: Some(format!("{:?}", self.target)),
        }
    }

    async fn dependencies(&self, _input: &(Program, crate::semantic::SemanticInfo), _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Code generation depends on semantic analysis
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &(Program, crate::semantic::SemanticInfo)) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::ConfigChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "code_generation"
    }
} 

/// Optimization query - applies AST transformations and optimizations
#[derive(Debug, Clone)]
pub struct OptimizationQuery {
    transformation_engine: Arc<Mutex<TransformationEngine>>,
}

impl OptimizationQuery {
    pub fn new() -> Self {
        let config = TransformationConfig {
            enable_dead_code_elimination: true,
            enable_function_inlining: true,
            enable_constant_folding: true,
            enable_type_inference_optimization: true,
            max_inline_size: 50,
            max_inline_depth: 3,
        };
        
        Self {
            transformation_engine: Arc::new(Mutex::new(TransformationEngine::with_config(config))),
        }
    }
    
    pub fn with_config(config: TransformationConfig) -> Self {
        Self {
            transformation_engine: Arc::new(Mutex::new(TransformationEngine::with_config(config))),
        }
    }
}

#[async_trait]
impl CompilerQuery<(Program, crate::semantic::SemanticInfo), (Program, TransformationResult)> for OptimizationQuery {
    async fn execute(&self, input: (Program, crate::semantic::SemanticInfo), _context: QueryContext) -> CompilerResult<(Program, TransformationResult)> {
        let (mut program, semantic_info) = input;
        
        // Apply AST transformations
        let mut engine = self.transformation_engine.lock().unwrap();
        let transformation_result = engine.transform_program(&mut program);
        
        // Log transformation results
        if transformation_result.has_changes() {
            tracing::info!(
                "AST optimizations applied: {} nodes eliminated, {} functions inlined, {} constants folded, {} types simplified",
                transformation_result.nodes_eliminated,
                transformation_result.functions_inlined,
                transformation_result.constants_folded,
                transformation_result.types_simplified
            );
            
            for message in &transformation_result.messages {
                tracing::debug!("Optimization: {}", message);
            }
        }
        
        Ok((program, transformation_result))
    }

    fn cache_key(&self, input: &(Program, crate::semantic::SemanticInfo)) -> CacheKey {
        let mut hasher = rustc_hash::FxHasher::default();
        // Hash program structure for cache key
        input.0.items.len().hash(&mut hasher);
        let program_hash = hasher.finish();
        
        CacheKey {
            query_type: "optimization".to_string(),
            input_hash: program_hash,
            semantic_hash: Some(input.1.ai_metadata.semantic_hash),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_config: None,
        }
    }

    async fn dependencies(&self, _input: &(Program, crate::semantic::SemanticInfo), _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        // Optimization depends on semantic analysis (handled by orchestrator)
        Ok(HashSet::new())
    }

    fn invalidate_on(&self, _input: &(Program, crate::semantic::SemanticInfo)) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        triggers.insert(InvalidationTrigger::SemanticContextChanged(prism_common::NodeId(0)));
        triggers.insert(InvalidationTrigger::OptimizationLevelChanged);
        triggers
    }

    fn query_type(&self) -> &'static str {
        "optimization"
    }
}

/// Invalidation triggers for cache management
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InvalidationTrigger {
    /// Source file changed
    FileChanged(PathBuf),
    /// Semantic context changed
    SemanticContextChanged(prism_common::NodeId),
    /// Compilation target changed
    TargetChanged(String),
    /// Dependencies changed
    DependenciesChanged(PathBuf),
    /// Optimization level changed
    OptimizationLevelChanged,
} 