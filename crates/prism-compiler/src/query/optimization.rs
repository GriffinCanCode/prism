//! Query Optimization - Performance Optimizations and Caching Strategies
//!
//! This module provides specialized optimization techniques for query execution,
//! including advanced caching strategies, query plan optimization, and performance
//! monitoring specifically tailored to compiler queries.

use crate::error::{CompilerError, CompilerResult};
use crate::query::core::{CacheKey, QueryId};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Query optimization configuration
#[derive(Debug, Clone)]
pub struct QueryOptimizationConfig {
    /// Enable query plan optimization
    pub enable_query_planning: bool,
    /// Enable multi-level caching
    pub enable_multilevel_caching: bool,
    /// Enable query result prefetching
    pub enable_prefetching: bool,
    /// Enable query batching
    pub enable_batching: bool,
    /// Maximum cache memory usage in MB
    pub max_cache_memory_mb: u64,
    /// Cache entry TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for QueryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_query_planning: true,
            enable_multilevel_caching: true,
            enable_prefetching: false,
            enable_batching: true,
            max_cache_memory_mb: 256,
            cache_ttl_seconds: 300,
        }
    }
}

/// Query performance optimizer
#[derive(Debug)]
pub struct QueryOptimizer {
    /// Optimization configuration
    config: QueryOptimizationConfig,
    /// Query execution statistics
    stats: QueryExecutionStats,
    /// Cache performance metrics
    cache_metrics: CacheMetrics,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new(config: QueryOptimizationConfig) -> Self {
        Self {
            config,
            stats: QueryExecutionStats::default(),
            cache_metrics: CacheMetrics::default(),
        }
    }

    /// Record query execution
    pub fn record_execution(&mut self, query_type: &str, duration: Duration, cache_hit: bool) {
        self.stats.record_execution(query_type, duration, cache_hit);
        
        if cache_hit {
            self.cache_metrics.record_hit();
        } else {
            self.cache_metrics.record_miss();
        }
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Analyze cache hit rates
        if self.cache_metrics.hit_rate() < 0.5 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::IncreaseCacheSize,
                description: "Low cache hit rate detected. Consider increasing cache size.".to_string(),
                impact: OptimizationImpact::High,
                estimated_improvement: 0.3,
            });
        }
        
        // Analyze slow queries
        if let Some(slowest_query) = self.stats.get_slowest_query_type() {
            if self.stats.get_avg_duration(&slowest_query) > Duration::from_millis(100) {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::OptimizeSlowQuery,
                    description: format!("Query type '{}' is consistently slow", slowest_query),
                    impact: OptimizationImpact::Medium,
                    estimated_improvement: 0.2,
                });
            }
        }
        
        recommendations
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &QueryExecutionStats {
        &self.stats
    }

    /// Get cache metrics
    pub fn get_cache_metrics(&self) -> &CacheMetrics {
        &self.cache_metrics
    }
}

/// Query execution statistics
#[derive(Debug, Default)]
pub struct QueryExecutionStats {
    /// Execution counts by query type
    execution_counts: HashMap<String, u64>,
    /// Total execution times by query type
    total_durations: HashMap<String, Duration>,
    /// Cache hit counts by query type
    cache_hits: HashMap<String, u64>,
    /// Total queries executed
    total_queries: u64,
}

impl QueryExecutionStats {
    /// Record a query execution
    pub fn record_execution(&mut self, query_type: &str, duration: Duration, cache_hit: bool) {
        *self.execution_counts.entry(query_type.to_string()).or_insert(0) += 1;
        *self.total_durations.entry(query_type.to_string()).or_insert(Duration::ZERO) += duration;
        
        if cache_hit {
            *self.cache_hits.entry(query_type.to_string()).or_insert(0) += 1;
        }
        
        self.total_queries += 1;
    }

    /// Get average duration for a query type
    pub fn get_avg_duration(&self, query_type: &str) -> Duration {
        if let (Some(&total_duration), Some(&count)) = (
            self.total_durations.get(query_type),
            self.execution_counts.get(query_type)
        ) {
            if count > 0 {
                return total_duration / count as u32;
            }
        }
        Duration::ZERO
    }

    /// Get the slowest query type
    pub fn get_slowest_query_type(&self) -> Option<String> {
        self.total_durations
            .iter()
            .max_by_key(|(query_type, &total_duration)| {
                let count = self.execution_counts.get(*query_type).unwrap_or(&1);
                total_duration / *count as u32
            })
            .map(|(query_type, _)| query_type.clone())
    }

    /// Get cache hit rate for a query type
    pub fn get_cache_hit_rate(&self, query_type: &str) -> f64 {
        if let (Some(&hits), Some(&total)) = (
            self.cache_hits.get(query_type),
            self.execution_counts.get(query_type)
        ) {
            if total > 0 {
                return hits as f64 / total as f64;
            }
        }
        0.0
    }
}

/// Cache performance metrics
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Cache memory usage estimate
    pub memory_usage_bytes: u64,
    /// Number of cache evictions
    pub evictions: u64,
}

impl CacheMetrics {
    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.total_hits += 1;
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.total_misses += 1;
    }

    /// Calculate cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total > 0 {
            self.total_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Record cache eviction
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Description of the recommendation
    pub description: String,
    /// Impact level of the optimization
    pub impact: OptimizationImpact,
    /// Estimated performance improvement (0.0 to 1.0)
    pub estimated_improvement: f64,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Increase cache size
    IncreaseCacheSize,
    /// Optimize a slow query
    OptimizeSlowQuery,
    /// Enable query batching
    EnableBatching,
    /// Enable prefetching
    EnablePrefetching,
    /// Adjust cache TTL
    AdjustCacheTTL,
    /// Enable query planning
    EnableQueryPlanning,
}

/// Impact levels for optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationImpact {
    /// Low impact optimization
    Low,
    /// Medium impact optimization
    Medium,
    /// High impact optimization
    High,
    /// Critical optimization needed
    Critical,
}

/// Query plan optimizer for complex queries
#[derive(Debug)]
pub struct QueryPlanOptimizer {
    /// Optimization rules
    rules: Vec<OptimizationRule>,
}

impl QueryPlanOptimizer {
    /// Create a new query plan optimizer
    pub fn new() -> Self {
        Self {
            rules: Self::default_optimization_rules(),
        }
    }

    /// Get default optimization rules
    fn default_optimization_rules() -> Vec<OptimizationRule> {
        vec![
            OptimizationRule {
                name: "parallel_independent_queries".to_string(),
                description: "Execute independent queries in parallel".to_string(),
                applies_to: vec!["symbol_resolution".to_string(), "scope_hierarchy".to_string()],
                estimated_speedup: 2.0,
            },
            OptimizationRule {
                name: "cache_intermediate_results".to_string(),
                description: "Cache intermediate results for reuse".to_string(),
                applies_to: vec!["find_symbols_by_kind".to_string()],
                estimated_speedup: 1.5,
            },
        ]
    }
}

/// Optimization rule for query planning
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Query types this rule applies to
    pub applies_to: Vec<String>,
    /// Estimated speedup factor
    pub estimated_speedup: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_optimizer() {
        let config = QueryOptimizationConfig::default();
        let mut optimizer = QueryOptimizer::new(config);
        
        // Record some executions
        optimizer.record_execution("test_query", Duration::from_millis(50), false);
        optimizer.record_execution("test_query", Duration::from_millis(30), true);
        optimizer.record_execution("slow_query", Duration::from_millis(200), false);
        
        // Get recommendations
        let recommendations = optimizer.get_recommendations();
        assert!(!recommendations.is_empty());
        
        // Check stats
        let stats = optimizer.get_stats();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.get_cache_hit_rate("test_query"), 0.5);
    }

    #[test]
    fn test_cache_metrics() {
        let mut metrics = CacheMetrics::default();
        
        metrics.record_hit();
        metrics.record_hit();
        metrics.record_miss();
        
        assert_eq!(metrics.hit_rate(), 2.0 / 3.0);
        assert_eq!(metrics.total_hits, 2);
        assert_eq!(metrics.total_misses, 1);
    }

    #[test]
    fn test_query_plan_optimizer() {
        let optimizer = QueryPlanOptimizer::new();
        assert!(!optimizer.rules.is_empty());
    }
} 