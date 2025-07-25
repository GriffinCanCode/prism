//! Compiler Integration for Type Inference
//!
//! This module provides integration with the compiler's performance profiling and query systems.
//! It follows the interfaces established in prism-compiler for proper system coordination.
//!
//! **Single Responsibility**: Compiler system integration
//! **What it does**: Report metrics to compiler, integrate with query system, coordinate with profiling
//! **What it doesn't do**: Perform inference, manage types, handle compilation phases

use crate::type_inference::{InferenceStatistics, MemoryStatistics, constraints::ConstraintSet};
use prism_common::Span;
use serde::{Serialize, Deserialize};
use std::time::Duration;
use std::collections::HashMap;

/// Integration with the compiler's performance profiling system
#[derive(Debug)]
pub struct CompilerIntegration {
    /// Configuration for compiler integration
    config: CompilerIntegrationConfig,
    /// Performance reporter for compiler coordination
    performance_reporter: PerformanceReporter,
    /// Query system coordinator
    query_coordinator: QuerySystemCoordinator,
    /// Event emitter for compiler events
    event_emitter: CompilerEventEmitter,
    /// Metrics aggregator
    metrics_aggregator: MetricsAggregator,
}

/// Configuration for compiler integration
#[derive(Debug, Clone)]
pub struct CompilerIntegrationConfig {
    /// Enable performance reporting to compiler
    pub enable_performance_reporting: bool,
    /// Enable query system integration
    pub enable_query_integration: bool,
    /// Enable compiler event emission
    pub enable_event_emission: bool,
    /// Enable detailed metrics aggregation
    pub enable_detailed_metrics: bool,
    /// Reporting interval in milliseconds
    pub reporting_interval_ms: u64,
    /// Maximum number of cached reports
    pub max_cached_reports: usize,
}

/// Performance reporter that interfaces with compiler's profiling system
#[derive(Debug)]
struct PerformanceReporter {
    /// Cached performance reports
    cached_reports: Vec<CompilerPerformanceReport>,
    /// Report sequence number
    sequence_number: u64,
    /// Last report timestamp
    last_report_time: Option<std::time::Instant>,
}

/// Query system coordinator for integration with prism-compiler query system
#[derive(Debug)]
struct QuerySystemCoordinator {
    /// Active query sessions
    active_sessions: HashMap<String, QuerySession>,
    /// Query performance metrics
    query_metrics: QueryPerformanceMetrics,
    /// Integration statistics
    integration_stats: QueryIntegrationStats,
}

/// Compiler event emitter for structured logging and monitoring
#[derive(Debug)]
struct CompilerEventEmitter {
    /// Event sequence number
    event_sequence: u64,
    /// Event buffer
    event_buffer: Vec<CompilerEvent>,
    /// Target systems for events
    event_targets: Vec<EventTarget>,
}

/// Metrics aggregator for compiler coordination
#[derive(Debug)]
struct MetricsAggregator {
    /// Aggregated metrics by component
    component_metrics: HashMap<String, ComponentMetrics>,
    /// Global aggregation statistics
    global_stats: GlobalAggregationStats,
    /// Aggregation window
    aggregation_window: Duration,
}

/// Performance report structure compatible with compiler's profiling system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerPerformanceReport {
    /// Component name (type_inference)
    pub component: String,
    /// Compilation phase
    pub phase: String,
    /// Report timestamp
    pub timestamp: std::time::SystemTime,
    /// Duration of the operation
    pub duration: Duration,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Cache performance metrics
    pub cache_performance: CachePerformanceMetrics,
    /// AI-specific performance metadata
    pub ai_metadata: AIPerformanceMetadata,
    /// Integration context
    pub integration_context: IntegrationContext,
    /// Sequence number for ordering
    pub sequence_number: u64,
}

/// Cache performance metrics for compiler coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Total number of queries
    pub total_queries: u64,
    /// Cache size in bytes
    pub cache_size: usize,
    /// Cache eviction count
    pub eviction_count: u64,
    /// Average lookup time
    pub avg_lookup_time: Duration,
}

/// AI performance metadata for compiler coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIPerformanceMetadata {
    /// Time spent generating AI metadata
    pub metadata_generation_time: Duration,
    /// Number of AI insights generated
    pub ai_insights_generated: usize,
    /// Number of semantic relationships found
    pub semantic_relationships_found: usize,
    /// AI processing efficiency score
    pub ai_efficiency_score: f64,
    /// AI metadata quality score
    pub ai_quality_score: f64,
}

/// Integration context for compiler coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationContext {
    /// Source file being processed
    pub source_file: Option<String>,
    /// Compilation target
    pub compilation_target: Option<String>,
    /// Integration flags
    pub integration_flags: Vec<String>,
    /// Cross-system references
    pub cross_system_refs: Vec<String>,
}

/// Query session for compiler query system integration
#[derive(Debug, Clone)]
struct QuerySession {
    /// Session ID
    session_id: String,
    /// Session start time
    start_time: std::time::Instant,
    /// Query type
    query_type: String,
    /// Session context
    context: QuerySessionContext,
    /// Performance metrics for this session
    metrics: SessionMetrics,
}

/// Query session context
#[derive(Debug, Clone)]
struct QuerySessionContext {
    /// Inference phase
    inference_phase: String,
    /// Node types being processed
    node_types: Vec<String>,
    /// Constraints being solved
    constraint_count: usize,
    /// Environment scope depth
    scope_depth: usize,
}

/// Performance metrics for query sessions
#[derive(Debug, Clone)]
struct SessionMetrics {
    /// Nodes processed in this session
    nodes_processed: usize,
    /// Constraints generated
    constraints_generated: usize,
    /// Cache hits
    cache_hits: usize,
    /// Cache misses
    cache_misses: usize,
    /// Memory allocated
    memory_allocated: usize,
}

/// Query performance metrics for compiler integration
#[derive(Debug, Clone)]
struct QueryPerformanceMetrics {
    /// Total query sessions
    total_sessions: u64,
    /// Average session duration
    avg_session_duration: Duration,
    /// Query throughput (queries per second)
    query_throughput: f64,
    /// Integration overhead
    integration_overhead: Duration,
}

/// Query integration statistics
#[derive(Debug, Clone)]
struct QueryIntegrationStats {
    /// Successful integrations
    successful_integrations: u64,
    /// Failed integrations
    failed_integrations: u64,
    /// Integration success rate
    success_rate: f64,
    /// Average integration time
    avg_integration_time: Duration,
}

/// Compiler event for structured logging
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompilerEvent {
    /// Event ID
    event_id: String,
    /// Event type
    event_type: CompilerEventType,
    /// Event timestamp
    timestamp: std::time::SystemTime,
    /// Event payload
    payload: CompilerEventPayload,
    /// Event metadata
    metadata: CompilerEventMetadata,
    /// Sequence number
    sequence: u64,
}

/// Types of compiler events
#[derive(Debug, Clone, Serialize, Deserialize)]
enum CompilerEventType {
    /// Performance report event
    PerformanceReport,
    /// Integration status event
    IntegrationStatus,
    /// Error or warning event
    Diagnostic,
    /// Query system event
    QuerySystem,
    /// Cache event
    Cache,
    /// AI metadata event
    AIMetadata,
}

/// Event payload containing the actual data
#[derive(Debug, Clone, Serialize, Deserialize)]
enum CompilerEventPayload {
    /// Performance data
    Performance(CompilerPerformanceReport),
    /// Integration status
    Integration(IntegrationStatusPayload),
    /// Diagnostic information
    Diagnostic(DiagnosticPayload),
    /// Query system data
    Query(QueryEventPayload),
    /// Cache data
    Cache(CacheEventPayload),
    /// AI metadata
    AIMetadata(AIMetadataPayload),
}

/// Event metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompilerEventMetadata {
    /// Source component
    source_component: String,
    /// Target systems
    target_systems: Vec<String>,
    /// Priority level
    priority: EventPriority,
    /// Tags for filtering
    tags: Vec<String>,
}

/// Event priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
enum EventPriority {
    /// Critical events
    Critical,
    /// High priority events
    High,
    /// Normal priority events
    Normal,
    /// Low priority events
    Low,
    /// Debug events
    Debug,
}

/// Event target systems
#[derive(Debug, Clone)]
enum EventTarget {
    /// Compiler's query system
    QuerySystem,
    /// Performance profiler
    PerformanceProfiler,
    /// Structured logging system
    StructuredLogger,
    /// Metrics collector
    MetricsCollector,
    /// External monitoring
    ExternalMonitoring,
}

/// Component metrics for aggregation
#[derive(Debug, Clone)]
struct ComponentMetrics {
    /// Component name
    component_name: String,
    /// Total processing time
    total_processing_time: Duration,
    /// Total nodes processed
    total_nodes_processed: usize,
    /// Total memory used
    total_memory_used: usize,
    /// Error count
    error_count: usize,
    /// Success rate
    success_rate: f64,
}

/// Global aggregation statistics
#[derive(Debug, Clone)]
struct GlobalAggregationStats {
    /// Total components tracked
    total_components: usize,
    /// Overall success rate
    overall_success_rate: f64,
    /// Total processing time across all components
    total_processing_time: Duration,
    /// Peak memory usage
    peak_memory_usage: usize,
    /// Aggregation efficiency
    aggregation_efficiency: f64,
}

// Payload types for different event types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IntegrationStatusPayload {
    status: String,
    component: String,
    details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticPayload {
    level: String,
    message: String,
    source_location: Option<String>,
    suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryEventPayload {
    query_type: String,
    duration: Duration,
    success: bool,
    cache_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEventPayload {
    operation: String,
    hit_rate: f64,
    size: usize,
    evictions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AIMetadataPayload {
    metadata_type: String,
    generation_time: Duration,
    quality_score: f64,
    insights_count: usize,
}

impl CompilerIntegration {
    /// Create a new compiler integration
    pub fn new() -> Self {
        Self::with_config(CompilerIntegrationConfig::default())
    }

    /// Create compiler integration with custom configuration
    pub fn with_config(config: CompilerIntegrationConfig) -> Self {
        Self {
            config: config.clone(),
            performance_reporter: PerformanceReporter::new(config.max_cached_reports),
            query_coordinator: QuerySystemCoordinator::new(),
            event_emitter: CompilerEventEmitter::new(),
            metrics_aggregator: MetricsAggregator::new(Duration::from_secs(60)),
        }
    }

    /// Report program inference performance to the compiler
    pub fn report_program_inference(&mut self, statistics: &InferenceStatistics, elapsed: Duration) {
        if !self.config.enable_performance_reporting {
            return;
        }

        let report = self.create_performance_report(statistics, elapsed, "program_inference");
        self.performance_reporter.submit_report(report.clone());

        // Emit performance event
        if self.config.enable_event_emission {
            self.event_emitter.emit_performance_event(report);
        }

        // Update metrics aggregation
        if self.config.enable_detailed_metrics {
            self.metrics_aggregator.update_component_metrics("type_inference", statistics, elapsed);
        }
    }

    /// Report expression inference performance
    pub fn report_expression_inference(&mut self, node_type: &str, elapsed: Duration) {
        if !self.config.enable_performance_reporting {
            return;
        }

        let statistics = InferenceStatistics {
            nodes_processed: 1,
            type_vars_generated: 0, // Would be tracked during inference
            constraints_generated: 0,
            unification_steps: 0, // Would be tracked during constraint solving
            inference_time_ms: elapsed.as_millis() as u64,
            constraint_solving_time_ms: 0, // Would be tracked separately
            memory_stats: MemoryStatistics {
                peak_memory_bytes: 0,
                environments_created: 0,
                substitutions_created: 0,
            },
        };

        let report = self.create_performance_report(&statistics, elapsed, "expression_inference");
        self.performance_reporter.submit_report(report);
    }

    /// Start a query session for compiler integration
    pub fn start_query_session(&mut self, session_id: String, query_type: String) {
        if !self.config.enable_query_integration {
            return;
        }

        let session = QuerySession {
            session_id: session_id.clone(),
            start_time: std::time::Instant::now(),
            query_type: query_type.clone(),
            context: QuerySessionContext {
                inference_phase: "initialization".to_string(),
                node_types: Vec::new(),
                constraint_count: 0,
                scope_depth: 0,
            },
            metrics: SessionMetrics::default(),
        };

        self.query_coordinator.active_sessions.insert(session_id, session);
    }

    /// End a query session and report metrics
    pub fn end_query_session(&mut self, session_id: &str, success: bool) {
        if !self.config.enable_query_integration {
            return;
        }

        if let Some(session) = self.query_coordinator.active_sessions.remove(session_id) {
            let duration = session.start_time.elapsed();
            
            // Update query metrics
            self.query_coordinator.query_metrics.total_sessions += 1;
            self.query_coordinator.query_metrics.avg_session_duration = 
                (self.query_coordinator.query_metrics.avg_session_duration + duration) / 2;

            // Update integration stats
            if success {
                self.query_coordinator.integration_stats.successful_integrations += 1;
            } else {
                self.query_coordinator.integration_stats.failed_integrations += 1;
            }

            let total_integrations = self.query_coordinator.integration_stats.successful_integrations + 
                                   self.query_coordinator.integration_stats.failed_integrations;
            
            if total_integrations > 0 {
                self.query_coordinator.integration_stats.success_rate = 
                    self.query_coordinator.integration_stats.successful_integrations as f64 / total_integrations as f64;
            }

            // Emit query event
            if self.config.enable_event_emission {
                self.event_emitter.emit_query_event(&session, duration, success);
            }
        }
    }

    /// Report cache performance to compiler
    pub fn report_cache_performance(&mut self, hit_rate: f64, total_queries: u64, cache_size: usize) {
        if !self.config.enable_performance_reporting {
            return;
        }

        let cache_metrics = CachePerformanceMetrics {
            hit_rate,
            total_queries,
            cache_size,
            eviction_count: 0, // Would be tracked separately
            avg_lookup_time: Duration::from_nanos(100), // Estimated
        };

        // Emit cache event
        if self.config.enable_event_emission {
            self.event_emitter.emit_cache_event(cache_metrics);
        }
    }

    /// Get integration statistics for monitoring
    pub fn get_integration_statistics(&self) -> IntegrationStatistics {
        IntegrationStatistics {
            total_reports_sent: self.performance_reporter.sequence_number,
            active_query_sessions: self.query_coordinator.active_sessions.len() as u64,
            query_success_rate: self.query_coordinator.integration_stats.success_rate,
            avg_query_duration: self.query_coordinator.query_metrics.avg_session_duration,
            events_emitted: self.event_emitter.event_sequence,
            integration_overhead: self.calculate_integration_overhead(),
        }
    }

    /// Reset integration state
    pub fn reset(&mut self) {
        self.performance_reporter.reset();
        self.query_coordinator.reset();
        self.event_emitter.reset();
        self.metrics_aggregator.reset();
    }

    // Private helper methods

    fn create_performance_report(
        &mut self,
        statistics: &InferenceStatistics,
        elapsed: Duration,
        phase: &str,
    ) -> CompilerPerformanceReport {
        let sequence_number = self.performance_reporter.get_next_sequence();
        
        CompilerPerformanceReport {
            component: "type_inference".to_string(),
            phase: phase.to_string(),
            timestamp: std::time::SystemTime::now(),
            duration: elapsed,
            memory_usage: self.estimate_memory_usage(statistics),
            cache_performance: self.extract_cache_metrics(statistics),
            ai_metadata: self.extract_ai_metadata(statistics),
            integration_context: self.create_integration_context(),
            sequence_number,
        }
    }

    fn estimate_memory_usage(&self, statistics: &InferenceStatistics) -> usize {
        // Rough estimation based on processed nodes and constraints
        let base_size = std::mem::size_of::<InferenceStatistics>();
        let node_size = statistics.nodes_processed * 256; // Estimated per-node overhead
        let constraint_size = statistics.constraints_generated * 128; // Estimated per-constraint overhead
        let type_var_size = statistics.type_vars_generated * 64; // Estimated per-type-var overhead
        
        base_size + node_size + constraint_size + type_var_size
    }

    fn extract_cache_metrics(&self, statistics: &InferenceStatistics) -> CachePerformanceMetrics {
        CachePerformanceMetrics {
            hit_rate: 0.85, // Would be calculated from actual cache statistics
            total_queries: statistics.unification_steps as u64,
            cache_size: self.estimate_memory_usage(statistics),
            eviction_count: 0,
            avg_lookup_time: Duration::from_nanos(50),
        }
    }

    fn extract_ai_metadata(&self, statistics: &InferenceStatistics) -> AIPerformanceMetadata {
        AIPerformanceMetadata {
            metadata_generation_time: Duration::from_millis(statistics.inference_time_ms / 10),
            ai_insights_generated: statistics.nodes_processed / 5, // Estimated
            semantic_relationships_found: statistics.constraints_generated / 3, // Estimated
            ai_efficiency_score: 0.8,
            ai_quality_score: 0.9,
        }
    }

    fn create_integration_context(&self) -> IntegrationContext {
        IntegrationContext {
            source_file: None, // Would be provided by caller
            compilation_target: None, // Would be provided by caller
            integration_flags: vec!["type_inference".to_string()],
            cross_system_refs: vec!["prism-compiler".to_string(), "prism-effects".to_string()],
        }
    }

    fn calculate_integration_overhead(&self) -> Duration {
        // Estimate overhead from integration activities
        Duration::from_micros(100) // Placeholder
    }
}

/// Integration statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStatistics {
    /// Total performance reports sent
    pub total_reports_sent: u64,
    /// Active query sessions
    pub active_query_sessions: u64,
    /// Query success rate
    pub query_success_rate: f64,
    /// Average query duration
    pub avg_query_duration: Duration,
    /// Total events emitted
    pub events_emitted: u64,
    /// Integration overhead
    pub integration_overhead: Duration,
}

impl PerformanceReporter {
    fn new(max_cached_reports: usize) -> Self {
        Self {
            cached_reports: Vec::with_capacity(max_cached_reports),
            sequence_number: 0,
            last_report_time: None,
        }
    }

    fn submit_report(&mut self, report: CompilerPerformanceReport) {
        self.cached_reports.push(report);
        self.last_report_time = Some(std::time::Instant::now());
        
        // Emit structured log for compiler consumption
        self.emit_structured_log(&self.cached_reports.last().unwrap());
    }

    fn get_next_sequence(&mut self) -> u64 {
        self.sequence_number += 1;
        self.sequence_number
    }

    fn emit_structured_log(&self, report: &CompilerPerformanceReport) {
        // Emit structured log event that the compiler can consume
        tracing::info!(
            target: "prism_compiler::performance_events",
            event_type = "component_performance",
            component = %report.component,
            phase = %report.phase,
            duration_ms = report.duration.as_millis(),
            memory_bytes = report.memory_usage,
            cache_hit_rate = report.cache_performance.hit_rate,
            ai_insights = report.ai_metadata.ai_insights_generated,
            sequence = report.sequence_number,
            "Performance event from type inference engine"
        );
    }

    fn reset(&mut self) {
        self.cached_reports.clear();
        self.last_report_time = None;
        // Don't reset sequence number to maintain ordering across resets
    }
}

impl QuerySystemCoordinator {
    fn new() -> Self {
        Self {
            active_sessions: HashMap::new(),
            query_metrics: QueryPerformanceMetrics {
                total_sessions: 0,
                avg_session_duration: Duration::ZERO,
                query_throughput: 0.0,
                integration_overhead: Duration::ZERO,
            },
            integration_stats: QueryIntegrationStats {
                successful_integrations: 0,
                failed_integrations: 0,
                success_rate: 0.0,
                avg_integration_time: Duration::ZERO,
            },
        }
    }

    fn reset(&mut self) {
        self.active_sessions.clear();
        // Keep metrics for analysis
    }
}

impl CompilerEventEmitter {
    fn new() -> Self {
        Self {
            event_sequence: 0,
            event_buffer: Vec::new(),
            event_targets: vec![
                EventTarget::QuerySystem,
                EventTarget::PerformanceProfiler,
                EventTarget::StructuredLogger,
            ],
        }
    }

    fn emit_performance_event(&mut self, report: CompilerPerformanceReport) {
        let event = self.create_event(
            CompilerEventType::PerformanceReport,
            CompilerEventPayload::Performance(report),
            EventPriority::Normal,
        );
        self.emit_event(event);
    }

    fn emit_query_event(&mut self, session: &QuerySession, duration: Duration, success: bool) {
        let payload = QueryEventPayload {
            query_type: session.query_type.clone(),
            duration,
            success,
            cache_hit: false, // Would be determined from session metrics
        };
        
        let event = self.create_event(
            CompilerEventType::QuerySystem,
            CompilerEventPayload::Query(payload),
            EventPriority::Normal,
        );
        self.emit_event(event);
    }

    fn emit_cache_event(&mut self, cache_metrics: CachePerformanceMetrics) {
        let payload = CacheEventPayload {
            operation: "lookup".to_string(),
            hit_rate: cache_metrics.hit_rate,
            size: cache_metrics.cache_size,
            evictions: cache_metrics.eviction_count,
        };
        
        let event = self.create_event(
            CompilerEventType::Cache,
            CompilerEventPayload::Cache(payload),
            EventPriority::Low,
        );
        self.emit_event(event);
    }

    fn create_event(
        &mut self,
        event_type: CompilerEventType,
        payload: CompilerEventPayload,
        priority: EventPriority,
    ) -> CompilerEvent {
        self.event_sequence += 1;
        
        CompilerEvent {
            event_id: format!("type_inference_{}", self.event_sequence),
            event_type,
            timestamp: std::time::SystemTime::now(),
            payload,
            metadata: CompilerEventMetadata {
                source_component: "type_inference".to_string(),
                target_systems: vec!["prism-compiler".to_string()],
                priority,
                tags: vec!["performance".to_string(), "type_inference".to_string()],
            },
            sequence: self.event_sequence,
        }
    }

    fn emit_event(&mut self, event: CompilerEvent) {
        self.event_buffer.push(event.clone());
        
        // Emit to structured logging for compiler consumption
        tracing::event!(
            target: "prism_compiler::type_inference_events",
            tracing::Level::INFO,
            event_id = %event.event_id,
            event_type = ?event.event_type,
            sequence = event.sequence,
            "Type inference event"
        );
    }

    fn reset(&mut self) {
        self.event_buffer.clear();
        // Keep sequence number for ordering
    }
}

impl MetricsAggregator {
    fn new(aggregation_window: Duration) -> Self {
        Self {
            component_metrics: HashMap::new(),
            global_stats: GlobalAggregationStats {
                total_components: 0,
                overall_success_rate: 0.0,
                total_processing_time: Duration::ZERO,
                peak_memory_usage: 0,
                aggregation_efficiency: 0.0,
            },
            aggregation_window,
        }
    }

    fn update_component_metrics(&mut self, component: &str, statistics: &InferenceStatistics, elapsed: Duration) {
        let metrics = self.component_metrics.entry(component.to_string()).or_insert_with(|| {
            ComponentMetrics {
                component_name: component.to_string(),
                total_processing_time: Duration::ZERO,
                total_nodes_processed: 0,
                total_memory_used: 0,
                error_count: 0,
                success_rate: 1.0,
            }
        });

        metrics.total_processing_time += elapsed;
        metrics.total_nodes_processed += statistics.nodes_processed;
        // metrics.total_memory_used would be updated based on actual memory tracking
        
        // Update global stats
        self.global_stats.total_processing_time += elapsed;
        self.global_stats.total_components = self.component_metrics.len();
    }

    fn reset(&mut self) {
        self.component_metrics.clear();
        self.global_stats = GlobalAggregationStats {
            total_components: 0,
            overall_success_rate: 0.0,
            total_processing_time: Duration::ZERO,
            peak_memory_usage: 0,
            aggregation_efficiency: 0.0,
        };
    }
}

impl Default for CompilerIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_performance_reporting: true,
            enable_query_integration: true,
            enable_event_emission: true,
            enable_detailed_metrics: true,
            reporting_interval_ms: 1000,
            max_cached_reports: 100,
        }
    }
}

impl Default for SessionMetrics {
    fn default() -> Self {
        Self {
            nodes_processed: 0,
            constraints_generated: 0,
            cache_hits: 0,
            cache_misses: 0,
            memory_allocated: 0,
        }
    }
}

impl Default for CompilerIntegration {
    fn default() -> Self {
        Self::new()
    }
} 