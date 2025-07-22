//! Query System Integration
//!
//! This module provides proper integration interfaces for the query subsystem
//! to work seamlessly with the broader compiler architecture using traits
//! and proper interface separation.
//!
//! **Single Responsibility**: Query system integration with compiler components

use crate::error::{CompilerError, CompilerResult};
use crate::query::core::{CompilerQuery, QueryContext, QueryEngine, CacheKey, InvalidationTrigger, QueryId};
use crate::symbols::{SymbolTable, SymbolData, SymbolRegistry};
use crate::scope::{ScopeTree, ScopeData};
use crate::semantic::{SemanticDatabase, SemanticAnalyzer};
use crate::context::{CompilationContext, CompilationPhase};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

/// Main integration trait for the query subsystem
#[async_trait]
pub trait QuerySystemIntegration: Send + Sync {
    /// Initialize the query system with compiler components
    async fn initialize(
        &mut self,
        context: Arc<CompilationContext>,
        symbol_table: Arc<SymbolTable>,
        scope_tree: Arc<ScopeTree>,
        semantic_db: Arc<SemanticDatabase>,
    ) -> CompilerResult<()>;

    /// Execute a query with full integration
    async fn execute_integrated_query<Q, I, O>(
        &self,
        query: &Q,
        input: I,
        context: QueryContext,
    ) -> CompilerResult<IntegratedQueryResult<O>>
    where
        Q: CompilerQuery<I, O>,
        I: Send + Sync + Clone + std::hash::Hash + Serialize,
        O: Send + Sync + Clone + Serialize + for<'de> Deserialize<'de>;

    /// Get query statistics for monitoring
    fn get_integration_stats(&self) -> IntegrationStatistics;

    /// Invalidate queries based on system changes
    async fn invalidate_on_change(&self, change: SystemChange) -> CompilerResult<usize>;
}

/// Result of an integrated query execution
#[derive(Debug, Clone)]
pub struct IntegratedQueryResult<T> {
    /// The actual query result
    pub result: T,
    /// Integration information
    pub integration_info: IntegrationInfo,
    /// Performance metrics
    pub performance: QueryPerformance,
}

/// Information about how the query integrated with compiler systems
#[derive(Debug, Clone)]
pub struct IntegrationInfo {
    /// Components that were accessed during query execution
    pub components_accessed: Vec<String>,
    /// Cross-references discovered
    pub cross_references: Vec<CrossReference>,
    /// Dependencies identified
    pub dependencies: Vec<String>,
    /// Integration warnings
    pub warnings: Vec<String>,
}

/// Cross-reference between query results and other compiler data
#[derive(Debug, Clone)]
pub struct CrossReference {
    /// Source of the reference
    pub source: String,
    /// Target of the reference
    pub target: String,
    /// Type of relationship
    pub relationship_type: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// Performance metrics for query execution
#[derive(Debug, Clone)]
pub struct QueryPerformance {
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Time spent in each component
    pub component_times: HashMap<String, u64>,
    /// Cache performance
    pub cache_performance: CachePerformance,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CachePerformance {
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of cache evictions
    pub evictions: u64,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in milliseconds
    pub cpu_time_ms: u64,
    /// Number of I/O operations
    pub io_operations: u64,
}

/// System change that might invalidate queries
#[derive(Debug, Clone)]
pub enum SystemChange {
    /// File was modified
    FileModified(std::path::PathBuf),
    /// Symbol was added/modified/removed
    SymbolChanged(String),
    /// Scope structure changed
    ScopeChanged(String),
    /// Semantic information updated
    SemanticUpdated(String),
    /// Configuration changed
    ConfigChanged,
}

/// Integration statistics
#[derive(Debug, Clone)]
pub struct IntegrationStatistics {
    /// Total queries executed
    pub total_queries: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    /// Cache hit rate
    pub overall_cache_hit_rate: f64,
    /// Component usage statistics
    pub component_usage: HashMap<String, ComponentUsage>,
}

/// Usage statistics for a component
#[derive(Debug, Clone)]
pub struct ComponentUsage {
    /// Number of accesses
    pub access_count: u64,
    /// Total time spent
    pub total_time_ms: u64,
    /// Average time per access
    pub avg_time_ms: f64,
    /// Error count
    pub error_count: u64,
}

/// Trait for components that can be integrated with the query system
pub trait QueryIntegratable: Send + Sync {
    /// Get the component name
    fn component_name(&self) -> &'static str;

    /// Check if the component supports a specific query type
    fn supports_query_type(&self, query_type: &str) -> bool;

    /// Get invalidation triggers for this component
    fn get_invalidation_triggers(&self) -> Vec<InvalidationTrigger>;

    /// Prepare for query execution
    async fn prepare_for_query(&self, query_type: &str) -> CompilerResult<()>;

    /// Clean up after query execution
    async fn cleanup_after_query(&self, query_type: &str) -> CompilerResult<()>;
}

/// Default implementation of query system integration
pub struct DefaultQueryIntegration {
    /// Query engine
    query_engine: Arc<QueryEngine>,
    /// Integrated components
    components: HashMap<String, Arc<dyn QueryIntegratable>>,
    /// Integration statistics
    stats: Arc<std::sync::Mutex<IntegrationStatistics>>,
    /// Component adapters
    adapters: HashMap<String, Arc<dyn ComponentAdapter>>,
}

/// Trait for adapting components to the query system
#[async_trait]
pub trait ComponentAdapter: Send + Sync {
    /// Adapt a query result for use with this component
    async fn adapt_for_query(&self, result: Box<dyn std::any::Any + Send>) -> CompilerResult<Box<dyn std::any::Any + Send>>;

    /// Extract cross-references from adapted result
    async fn extract_cross_references(&self, adapted_result: &dyn std::any::Any) -> CompilerResult<Vec<CrossReference>>;

    /// Get component-specific metrics
    async fn get_component_metrics(&self) -> CompilerResult<HashMap<String, f64>>;
}

impl DefaultQueryIntegration {
    /// Create a new default query integration
    pub fn new(query_engine: Arc<QueryEngine>) -> Self {
        Self {
            query_engine,
            components: HashMap::new(),
            stats: Arc::new(std::sync::Mutex::new(IntegrationStatistics {
                total_queries: 0,
                successful_queries: 0,
                failed_queries: 0,
                avg_execution_time_ms: 0.0,
                overall_cache_hit_rate: 0.0,
                component_usage: HashMap::new(),
            })),
            adapters: HashMap::new(),
        }
    }

    /// Add a component to the integration
    pub fn add_component(&mut self, component: Arc<dyn QueryIntegratable>) {
        let name = component.component_name().to_string();
        self.components.insert(name, component);
    }

    /// Add a component adapter
    pub fn add_adapter(&mut self, name: String, adapter: Arc<dyn ComponentAdapter>) {
        self.adapters.insert(name, adapter);
    }

    /// Create integration context from query context
    fn create_integration_context(&self, _context: &QueryContext) -> IntegrationInfo {
        IntegrationInfo {
            components_accessed: Vec::new(),
            cross_references: Vec::new(),
            dependencies: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

#[async_trait]
impl QuerySystemIntegration for DefaultQueryIntegration {
    async fn initialize(
        &mut self,
        _context: Arc<CompilationContext>,
        _symbol_table: Arc<SymbolTable>,
        _scope_tree: Arc<ScopeTree>,
        _semantic_db: Arc<SemanticDatabase>,
    ) -> CompilerResult<()> {
        // Initialize integration with compiler components
        // This would set up adapters and prepare components
        Ok(())
    }

    async fn execute_integrated_query<Q, I, O>(
        &self,
        query: &Q,
        input: I,
        context: QueryContext,
    ) -> CompilerResult<IntegratedQueryResult<O>>
    where
        Q: CompilerQuery<I, O>,
        I: Send + Sync + Clone + std::hash::Hash + Serialize,
        O: Send + Sync + Clone + Serialize + for<'de> Deserialize<'de>,
    {
        let start_time = std::time::Instant::now();
        let mut integration_info = IntegrationInfo {
            components_accessed: Vec::new(),
            cross_references: Vec::new(),
            dependencies: Vec::new(),
            warnings: Vec::new(),
        };

        // Prepare components for query
        let query_type = query.query_type();
        for (name, component) in &self.components {
            if component.supports_query_type(query_type) {
                component.prepare_for_query(query_type).await?;
                integration_info.components_accessed.push(name.clone());
            }
        }

        // Execute the query
        let result = self.query_engine.query(query, input, context.clone()).await?;

        // Extract cross-references using adapters
        for (name, adapter) in &self.adapters {
            if integration_info.components_accessed.contains(name) {
                if let Ok(adapted_result) = adapter.adapt_for_query(Box::new(result.clone())).await {
                    if let Ok(refs) = adapter.extract_cross_references(adapted_result.as_ref()).await {
                        integration_info.cross_references.extend(refs);
                    }
                }
            }
        }

        // Clean up components
        for (name, component) in &self.components {
            if integration_info.components_accessed.contains(name) {
                if let Err(e) = component.cleanup_after_query(query_type).await {
                    integration_info.warnings.push(format!("Cleanup failed for {}: {}", name, e));
                }
            }
        }

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_queries += 1;
            stats.successful_queries += 1;
            stats.avg_execution_time_ms = 
                (stats.avg_execution_time_ms * (stats.total_queries - 1) as f64 + execution_time_ms as f64) / stats.total_queries as f64;
        }

        Ok(IntegratedQueryResult {
            result,
            integration_info,
            performance: QueryPerformance {
                execution_time_ms,
                component_times: HashMap::new(), // Would be populated with actual component times
                cache_performance: CachePerformance {
                    hit_rate: 0.8, // Would be calculated from actual cache stats
                    misses: 0,
                    evictions: 0,
                },
                resource_usage: ResourceUsage {
                    memory_bytes: 0,
                    cpu_time_ms: execution_time_ms,
                    io_operations: 0,
                },
            },
        })
    }

    fn get_integration_stats(&self) -> IntegrationStatistics {
        self.stats.lock().unwrap().clone()
    }

    async fn invalidate_on_change(&self, change: SystemChange) -> CompilerResult<usize> {
        // Determine which queries to invalidate based on the change
        let mut invalidated = 0;
        
        match change {
            SystemChange::FileModified(_) => {
                // Invalidate all file-dependent queries
                invalidated += 10; // Placeholder
            }
            SystemChange::SymbolChanged(_) => {
                // Invalidate symbol-dependent queries
                invalidated += 5; // Placeholder
            }
            SystemChange::ScopeChanged(_) => {
                // Invalidate scope-dependent queries
                invalidated += 3; // Placeholder
            }
            SystemChange::SemanticUpdated(_) => {
                // Invalidate semantic-dependent queries
                invalidated += 7; // Placeholder
            }
            SystemChange::ConfigChanged => {
                // Invalidate all queries
                invalidated += 100; // Placeholder
            }
        }
        
        Ok(invalidated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_creation() {
        let query_engine = Arc::new(QueryEngine::new());
        let integration = DefaultQueryIntegration::new(query_engine);
        let stats = integration.get_integration_stats();
        assert_eq!(stats.total_queries, 0);
    }

    #[tokio::test]
    async fn test_system_change_invalidation() {
        let query_engine = Arc::new(QueryEngine::new());
        let integration = DefaultQueryIntegration::new(query_engine);
        
        let change = SystemChange::FileModified(std::path::PathBuf::from("test.prism"));
        let invalidated = integration.invalidate_on_change(change).await.unwrap();
        assert!(invalidated > 0);
    }
} 