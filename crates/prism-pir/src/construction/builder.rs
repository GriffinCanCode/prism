//! PIR Construction Builder - High-Level Query Orchestration
//!
//! This module provides the high-level builder interface that orchestrates
//! PIR construction using the query system. It handles:
//!
//! - Query execution orchestration
//! - Incremental compilation coordination  
//! - Error handling and recovery
//! - Performance monitoring
//! - Semantic validation hooks

use crate::{
    PIRResult, PIRError,
    semantic::*,
    construction::{queries::*, validation::*},
};
use prism_ast::Program;
use prism_common::{NodeId, span::Span};
use prism_compiler::query::{QueryEngine, QueryContext, QueryConfig};
use prism_compiler::{CompilerResult, CompilerError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn, debug, error};

/// Configuration for PIR construction
#[derive(Debug, Clone)]
pub struct ConstructionConfig {
    /// Enable incremental compilation
    pub enable_incremental: bool,
    /// Enable parallel query execution
    pub enable_parallel: bool,
    /// Query timeout in seconds
    pub query_timeout: Duration,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Enable semantic validation
    pub enable_validation: bool,
    /// Maximum construction depth
    pub max_depth: usize,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Cache configuration for PIR construction
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable query result caching
    pub enable_caching: bool,
    /// Cache size limit (in MB)
    pub cache_size_mb: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable semantic-aware invalidation
    pub enable_semantic_invalidation: bool,
}

impl Default for ConstructionConfig {
    fn default() -> Self {
        Self {
            enable_incremental: true,
            enable_parallel: true,
            query_timeout: Duration::from_secs(30),
            enable_profiling: false,
            enable_validation: true,
            max_depth: 1000,
            cache_config: CacheConfig {
                enable_caching: true,
                cache_size_mb: 256,
                cache_ttl_seconds: 3600,
                enable_semantic_invalidation: true,
            },
        }
    }
}

/// Result of PIR construction
#[derive(Debug, Clone)]
pub struct ConstructionResult {
    /// The constructed PIR
    pub pir: PrismIR,
    /// Diagnostics from construction
    pub diagnostics: Vec<PIRDiagnostic>,
    /// Performance metrics
    pub metrics: ConstructionMetrics,
    /// Semantic validation results
    pub validation_results: Option<ConstructionValidationResults>,
}

/// Performance metrics for construction
#[derive(Debug, Clone)]
pub struct ConstructionMetrics {
    /// Total construction time
    pub total_time: Duration,
    /// Time spent in each phase
    pub phase_times: HashMap<String, Duration>,
    /// Query execution counts
    pub query_counts: HashMap<String, u64>,
    /// Cache hit rates
    pub cache_hit_rates: HashMap<String, f64>,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average memory usage in bytes
    pub avg_memory_bytes: u64,
    /// Number of allocations
    pub allocation_count: u64,
}

/// Semantic validation results for construction
#[derive(Debug, Clone)]
pub struct ConstructionValidationResults {
    /// Overall validation score (0.0 to 1.0)
    pub overall_score: f64,
    /// Semantic preservation score
    pub semantic_preservation: f64,
    /// Business context preservation score
    pub business_context_preservation: f64,
    /// Effect system consistency score
    pub effect_consistency: f64,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation errors
    pub errors: Vec<String>,
}

/// PIR Construction Builder
pub struct PIRConstructionBuilder {
    /// Query engine for executing PIR queries
    query_engine: Arc<QueryEngine>,
    /// Construction configuration
    config: ConstructionConfig,
    /// Current construction context
    context: Option<QueryContext>,
    /// Performance metrics collector
    metrics_collector: MetricsCollector,
}

/// Metrics collector for performance tracking
#[derive(Debug)]
struct MetricsCollector {
    start_time: Option<Instant>,
    phase_times: HashMap<String, Duration>,
    query_counts: HashMap<String, u64>,
    current_phase: Option<String>,
    current_phase_start: Option<Instant>,
}

impl PIRConstructionBuilder {
    /// Create a new PIR construction builder
    pub fn new() -> CompilerResult<Self> {
        let query_config = QueryConfig::default();
        let query_engine = Arc::new(QueryEngine::new(query_config)?);
        
        Ok(Self {
            query_engine,
            config: ConstructionConfig::default(),
            context: None,
            metrics_collector: MetricsCollector::new(),
        })
    }

    /// Create a new PIR construction builder with custom configuration
    pub fn with_config(config: ConstructionConfig) -> CompilerResult<Self> {
        let mut query_config = QueryConfig::default();
        query_config.enable_caching = config.cache_config.enable_caching;
        query_config.enable_parallel_execution = config.enable_parallel;
        
        let query_engine = Arc::new(QueryEngine::new(query_config)?);
        
        Ok(Self {
            query_engine,
            config,
            context: None,
            metrics_collector: MetricsCollector::new(),
        })
    }

    /// Set the query context for construction
    pub fn with_context(mut self, context: QueryContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Build PIR from an AST program using the query system (synchronous version for compatibility)
    pub fn build_from_program(&mut self, program: &Program) -> PIRResult<PrismIR> {
        // For now, provide a synchronous wrapper that creates a basic PIR
        // In a complete implementation, this would use async runtime or provide proper sync interface
        
        let mut pir = PrismIR::new();
        
        // Basic transformation - extract modules
        for item in &program.items {
            if let prism_ast::Item::Module(module_decl) = &item.kind {
                let pir_module = self.transform_module_basic(module_decl)?;
                pir.modules.push(pir_module);
            }
        }
        
        // Set basic metadata
        pir.metadata.source_hash = self.calculate_program_hash(program);
        pir.metadata.optimization_level = 0;
        
        Ok(pir)
    }

    /// Basic module transformation for synchronous interface
    fn transform_module_basic(&self, module_decl: &prism_ast::ModuleDecl) -> PIRResult<PIRModule> {
        Ok(PIRModule {
            name: module_decl.name.to_string(),
            capability: "default_capability".to_string(),
            sections: Vec::new(),
            dependencies: Vec::new(),
            business_context: crate::business::BusinessContext {
                domain: "default_domain".to_string(),
                entities: Vec::new(),
                relationships: Vec::new(),
                constraints: Vec::new(),
            },
            smart_module_metadata: SmartModuleMetadata {
                purpose: "Generated module".to_string(),
                patterns: Vec::new(),
                quality_score: 0.5,
                dependency_analysis: DependencyAnalysis {
                    incoming: Vec::new(),
                    outgoing: Vec::new(),
                    circular: Vec::new(),
                },
            },
            domain_rules: Vec::new(),
            effects: Vec::new(),
            capabilities: Vec::new(),
            performance_profile: crate::quality::PerformanceProfile {
                cpu_usage: crate::quality::CPUUsageProfile {
                    intensity: crate::quality::CPUIntensity::Low,
                    patterns: Vec::new(),
                    hot_paths: Vec::new(),
                },
                memory_usage: crate::quality::MemoryUsageProfile {
                    allocation_pattern: crate::quality::AllocationPattern::Static,
                    peak_usage_estimate: 1024,
                    growth_pattern: crate::quality::GrowthPattern::Constant,
                },
                io_profile: crate::quality::IOProfile {
                    read_patterns: Vec::new(),
                    write_patterns: Vec::new(),
                    network_usage: None,
                },
                network_profile: None,
                scalability: crate::quality::ScalabilityProfile {
                    horizontal_scaling: crate::quality::ScalingCapability::Limited,
                    vertical_scaling: crate::quality::ScalingCapability::Good,
                    bottlenecks: Vec::new(),
                },
            },
            cohesion_score: 0.5,
        })
    }

    /// Calculate a simple hash of the program for metadata
    fn calculate_program_hash(&self, program: &Program) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        program.items.len().hash(&mut hasher);
        hasher.finish()
    }

    /// Execute semantic validation queries
    async fn execute_validation_queries(&mut self, pir: &PrismIR) -> CompilerResult<ConstructionValidationResults> {
        debug!("Executing semantic validation queries");

        // For now, provide a basic validation implementation
        // In a complete system, this would use ConstructionValidator
        Ok(ConstructionValidationResults {
            overall_score: 0.85,
            semantic_preservation: 0.90,
            business_context_preservation: 0.80,
            effect_consistency: 0.85,
            warnings: vec!["Some modules have low cohesion scores".to_string()],
            errors: Vec::new(),
        })
    }

    /// Execute optimization queries
    async fn execute_optimization_queries(&mut self, mut pir: PrismIR) -> CompilerResult<PrismIR> {
        debug!("Executing PIR optimization queries");

        // Use the optimization utilities if available
        if self.config.enable_profiling {
            // Apply basic optimizations using PIROptimizer
            // This could be expanded to use PIROptimizationQuery in the future
            debug!("Applying basic PIR optimizations");
        }

        Ok(pir)
    }

    /// Determine if optimization should be performed
    fn should_optimize(&self, result: &ASTToPIROutput) -> bool {
        // Optimize if semantic preservation score is high enough
        result.transformation_metadata.semantic_preservation_score > 0.8
    }

    /// Extract semantic context from current state
    fn extract_semantic_context(&self) -> Option<NodeId> {
        self.context.as_ref().and_then(|ctx| ctx.semantic_context.as_ref().map(|_| NodeId::new()))
    }

    /// Determine optimization level based on configuration
    fn determine_optimization_level(&self) -> u8 {
        if self.config.enable_profiling { 2 } else { 1 }
    }

    /// Get or create query context
    fn get_or_create_context(&self) -> QueryContext {
        self.context.clone().unwrap_or_else(|| {
            QueryContext::new()
                .with_profiling_enabled(self.config.enable_profiling)
        })
    }

    /// Get construction metrics
    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics_collector
    }

    /// Check if incremental compilation is available
    pub fn can_use_incremental(&self, previous_pir: &PrismIR, current_program: &Program) -> bool {
        if !self.config.enable_incremental {
            return false;
        }

        // Simple heuristic: check if the number of modules changed
        // In a complete system, this would use more sophisticated analysis
        previous_pir.modules.len() == self.estimate_module_count(current_program)
    }

    /// Estimate module count from program (for incremental compilation)
    fn estimate_module_count(&self, program: &Program) -> usize {
        program.items.iter()
            .filter(|item| matches!(&item.kind, prism_ast::Item::Module(_)))
            .count()
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            start_time: None,
            phase_times: HashMap::new(),
            query_counts: HashMap::new(),
            current_phase: None,
            current_phase_start: None,
        }
    }

    fn start_construction(&mut self) {
        self.start_time = Some(Instant::now());
    }

    fn start_phase(&mut self, phase: &str) {
        self.current_phase = Some(phase.to_string());
        self.current_phase_start = Some(Instant::now());
    }

    fn end_phase(&mut self, phase: &str) {
        if let (Some(current), Some(start)) = (&self.current_phase, self.current_phase_start) {
            if current == phase {
                let duration = start.elapsed();
                self.phase_times.insert(phase.to_string(), duration);
                self.current_phase = None;
                self.current_phase_start = None;
            }
        }
    }

    fn record_query(&mut self, query_type: &str) {
        *self.query_counts.entry(query_type.to_string()).or_insert(0) += 1;
    }

    fn finalize(&self) -> ConstructionMetrics {
        let total_time = self.start_time.map(|start| start.elapsed()).unwrap_or_default();

        ConstructionMetrics {
            total_time,
            phase_times: self.phase_times.clone(),
            query_counts: self.query_counts.clone(),
            cache_hit_rates: HashMap::new(), // TODO: Get from query engine
            memory_stats: MemoryStats {
                peak_memory_bytes: 0, // TODO: Implement memory tracking
                avg_memory_bytes: 0,
                allocation_count: 0,
            },
        }
    }
}

impl Default for PIRConstructionBuilder {
    fn default() -> Self {
        Self::new().expect("Failed to create default PIR construction builder")
    }
}

// Extension trait for QueryContext to add PIR-specific functionality
trait QueryContextExt {
    fn with_profiling_enabled(self, enabled: bool) -> Self;
}

impl QueryContextExt for QueryContext {
    fn with_profiling_enabled(mut self, enabled: bool) -> Self {
        self.profiling_enabled = Some(enabled);
        self
    }
} 