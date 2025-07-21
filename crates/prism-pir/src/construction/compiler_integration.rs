//! Compiler Integration - PIR Construction Queries
//!
//! This module implements integration between PIR construction and the prism-compiler
//! query system, providing proper dependency management and incremental compilation support.
//!
//! **Conceptual Responsibility**: Compiler query system integration
//! **What it does**: Implements CompilerQuery for PIR construction, manages dependencies, enables caching
//! **What it doesn't do**: AST parsing, PIR validation, semantic analysis (delegates to appropriate systems)

use crate::{PIRResult, PIRError};
use crate::semantic::PrismIR;
use crate::construction::transformation::ASTToPIRTransformer;
use prism_common::{PIRConstructionConfig, PrismError};
use prism_ast::Program;
use std::collections::HashSet;
use std::sync::Arc;

// Note: These would normally come from prism-compiler, but we'll define minimal interfaces
// to avoid circular dependencies while maintaining proper separation of concerns

/// Minimal query interfaces to avoid circular dependencies
/// In a real integration, these would be imported from prism-compiler
pub mod query_interfaces {
    use std::collections::HashSet;
    use std::hash::Hash;
    use async_trait::async_trait;
    use serde::{Serialize, Deserialize};

    /// Query ID for dependency tracking
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct QueryId(pub u64);

    /// Cache key for query results
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct CacheKey {
        pub query_type: String,
        pub input_hash: u64,
    }

    /// Query context for execution
    #[derive(Debug, Clone)]
    pub struct QueryContext {
        pub query_stack: Vec<QueryId>,
        pub enable_cache: bool,
        pub enable_profiling: bool,
    }

    /// Invalidation trigger for cache entries
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum InvalidationTrigger {
        FileChanged(std::path::PathBuf),
        ConfigChanged,
        Manual(String),
    }

    /// Compiler result type
    pub type CompilerResult<T> = Result<T, CompilerError>;

    /// Compiler error type
    #[derive(Debug, thiserror::Error)]
    pub enum CompilerError {
        #[error("PIR construction error: {0}")]
        PIRConstruction(String),
        #[error("Internal error: {0}")]
        Internal(String),
    }

    /// Core query trait that compiler queries must implement
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

    impl From<crate::PIRError> for CompilerError {
        fn from(err: crate::PIRError) -> Self {
            CompilerError::PIRConstruction(err.to_string())
        }
    }
}

use query_interfaces::*;

/// Input for PIR construction query
#[derive(Debug, Clone, Hash)]
pub struct PIRConstructionInput {
    /// Program to transform
    pub program: Program,
    /// Construction configuration
    pub config: PIRConstructionConfig,
    /// Source file path for cache invalidation
    pub source_path: Option<std::path::PathBuf>,
}

/// PIR construction query that integrates with compiler query system
pub struct PIRConstructionQuery {
    /// Transformer instance
    transformer: Arc<std::sync::Mutex<ASTToPIRTransformer>>,
}

impl PIRConstructionQuery {
    /// Create a new PIR construction query
    pub fn new(config: PIRConstructionConfig) -> Self {
        Self {
            transformer: Arc::new(std::sync::Mutex::new(ASTToPIRTransformer::new(config))),
        }
    }

    /// Create with custom transformer
    pub fn with_transformer(transformer: ASTToPIRTransformer) -> Self {
        Self {
            transformer: Arc::new(std::sync::Mutex::new(transformer)),
        }
    }
}

#[async_trait]
impl CompilerQuery<PIRConstructionInput, PrismIR> for PIRConstructionQuery {
    async fn execute(&self, input: PIRConstructionInput, _context: QueryContext) -> CompilerResult<PrismIR> {
        let mut transformer = self.transformer.lock().unwrap();
        
        // Transform program to PIR using the configured transformer
        let pir = transformer.construct_pir_with_config(input.program, &input.config)
            .map_err(CompilerError::from)?;

        Ok(pir)
    }

    fn cache_key(&self, input: &PIRConstructionInput) -> CacheKey {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(input, &mut hasher);
        
        CacheKey {
            query_type: "PIRConstruction".to_string(),
            input_hash: std::hash::Hasher::finish(&hasher),
        }
    }

    async fn dependencies(&self, input: &PIRConstructionInput, _context: &QueryContext) -> CompilerResult<HashSet<QueryId>> {
        let mut deps = HashSet::new();
        
        // PIR construction depends on:
        // - Semantic analysis of the program
        // - Type checking results
        // - Effect analysis (if enabled)
        
        if input.config.enable_ai_metadata {
            // Would depend on semantic analysis query
            deps.insert(QueryId(1)); // Placeholder
        }
        
        if input.config.enable_effect_graph {
            // Would depend on effect analysis query
            deps.insert(QueryId(2)); // Placeholder
        }
        
        if input.config.enable_cohesion_analysis {
            // Would depend on cohesion analysis query
            deps.insert(QueryId(3)); // Placeholder
        }
        
        Ok(deps)
    }

    fn invalidate_on(&self, input: &PIRConstructionInput) -> HashSet<InvalidationTrigger> {
        let mut triggers = HashSet::new();
        
        // Invalidate when source file changes
        if let Some(source_path) = &input.source_path {
            triggers.insert(InvalidationTrigger::FileChanged(source_path.clone()));
        }
        
        // Invalidate when configuration changes
        triggers.insert(InvalidationTrigger::ConfigChanged);
        
        // Invalidate manually when needed
        triggers.insert(InvalidationTrigger::Manual("pir_construction".to_string()));
        
        triggers
    }

    fn query_type(&self) -> &'static str {
        "PIRConstruction"
    }
}

/// PIR construction orchestrator that manages multiple construction queries
pub struct PIRConstructionOrchestrator {
    /// Base configuration
    base_config: PIRConstructionConfig,
    /// Cached queries for different configurations
    queries: std::sync::RwLock<std::collections::HashMap<String, Arc<PIRConstructionQuery>>>,
}

impl PIRConstructionOrchestrator {
    /// Create a new PIR construction orchestrator
    pub fn new(base_config: PIRConstructionConfig) -> Self {
        Self {
            base_config,
            queries: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Get or create a query for the given configuration
    pub fn get_query(&self, config: &PIRConstructionConfig) -> Arc<PIRConstructionQuery> {
        let config_key = self.config_to_key(config);
        
        // Try to get existing query
        {
            let queries = self.queries.read().unwrap();
            if let Some(query) = queries.get(&config_key) {
                return Arc::clone(query);
            }
        }
        
        // Create new query
        let query = Arc::new(PIRConstructionQuery::new(config.clone()));
        
        // Cache the query
        {
            let mut queries = self.queries.write().unwrap();
            queries.insert(config_key, Arc::clone(&query));
        }
        
        query
    }

    /// Execute PIR construction with the given configuration
    pub async fn construct_pir(
        &self,
        program: Program,
        config: Option<PIRConstructionConfig>,
        context: QueryContext,
    ) -> CompilerResult<PrismIR> {
        let config = config.unwrap_or_else(|| self.base_config.clone());
        let query = self.get_query(&config);
        
        let input = PIRConstructionInput {
            program,
            config,
            source_path: None, // Would be provided by caller in real usage
        };
        
        query.execute(input, context).await
    }

    /// Create a configuration key for caching
    fn config_to_key(&self, config: &PIRConstructionConfig) -> String {
        format!(
            "validation:{}_ai:{}_cohesion:{}_effects:{}_business:{}_depth:{}",
            config.enable_validation,
            config.enable_ai_metadata,
            config.enable_cohesion_analysis,
            config.enable_effect_graph,
            config.enable_business_context,
            config.max_construction_depth
        )
    }
}

/// Integration helper for connecting PIR construction with existing compiler infrastructure
pub struct PIRCompilerIntegration;

impl PIRCompilerIntegration {
    /// Create a PIR construction orchestrator integrated with compiler configuration
    pub fn create_orchestrator(compiler_config: &CompilerIntegrationConfig) -> PIRConstructionOrchestrator {
        let pir_config = PIRConstructionConfig {
            enable_validation: compiler_config.enable_validation,
            enable_ai_metadata: compiler_config.enable_ai_export,
            enable_cohesion_analysis: compiler_config.enable_cohesion_analysis,
            enable_effect_graph: compiler_config.enable_effect_analysis,
            enable_business_context: compiler_config.enable_business_context,
            max_construction_depth: compiler_config.max_analysis_depth,
        };

        PIRConstructionOrchestrator::new(pir_config)
    }

    /// Demonstrate proper integration pattern with compiler query engine
    pub fn integration_example() -> String {
        r#"
        // In prism-compiler integration:
        
        use prism_pir::construction::compiler_integration::{PIRConstructionOrchestrator, PIRConstructionInput};
        use prism_compiler::query::{QueryEngine, QueryContext};
        
        // Create PIR construction orchestrator
        let pir_orchestrator = PIRCompilerIntegration::create_orchestrator(&compiler_config);
        
        // Use with compiler query engine
        let query = pir_orchestrator.get_query(&pir_config);
        let input = PIRConstructionInput {
            program: parsed_program,
            config: pir_config,
            source_path: Some(source_file_path),
        };
        
        // Execute through compiler query system
        let pir = query_engine.query(&query, input, query_context).await?;
        
        // PIR construction is now:
        // - Cached by the compiler query system
        // - Invalidated when dependencies change
        // - Integrated with incremental compilation
        // - Profiled and monitored
        "#.to_string()
    }
}

/// Configuration for compiler integration
#[derive(Debug, Clone)]
pub struct CompilerIntegrationConfig {
    /// Enable validation during construction
    pub enable_validation: bool,
    /// Enable AI metadata export
    pub enable_ai_export: bool,
    /// Enable cohesion analysis
    pub enable_cohesion_analysis: bool,
    /// Enable effect analysis
    pub enable_effect_analysis: bool,
    /// Enable business context extraction
    pub enable_business_context: bool,
    /// Maximum analysis depth
    pub max_analysis_depth: usize,
}

impl Default for CompilerIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_ai_export: true,
            enable_cohesion_analysis: true,
            enable_effect_analysis: true,
            enable_business_context: true,
            max_analysis_depth: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pir_construction_query() {
        let config = PIRConstructionConfig::default();
        let query = PIRConstructionQuery::new(config.clone());

        // Create test program
        let program = Program {
            items: Vec::new(),
            metadata: Default::default(),
        };

        let input = PIRConstructionInput {
            program,
            config,
            source_path: None,
        };

        let context = QueryContext {
            query_stack: Vec::new(),
            enable_cache: true,
            enable_profiling: false,
        };

        let result = query.execute(input, context).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_key_generation() {
        let config = PIRConstructionConfig::default();
        let query = PIRConstructionQuery::new(config.clone());

        let program = Program {
            items: Vec::new(),
            metadata: Default::default(),
        };

        let input = PIRConstructionInput {
            program,
            config,
            source_path: None,
        };

        let cache_key = query.cache_key(&input);
        assert_eq!(cache_key.query_type, "PIRConstruction");
        assert_ne!(cache_key.input_hash, 0);
    }

    #[test]
    fn test_orchestrator_query_caching() {
        let config = PIRConstructionConfig::default();
        let orchestrator = PIRConstructionOrchestrator::new(config.clone());

        let query1 = orchestrator.get_query(&config);
        let query2 = orchestrator.get_query(&config);

        // Should return the same cached query
        assert!(Arc::ptr_eq(&query1, &query2));
    }
} 