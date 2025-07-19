//! Compilation context and target management
//!
//! This module manages the overall compilation state, target configurations,
//! and AI metadata collection during compilation.

use crate::error::{CompilerError, CompilerResult};
use prism_common::{NodeId, span::Span};
use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Compilation context that tracks state during compilation
#[derive(Debug, Clone)]
pub struct CompilationContext {
    /// Target configurations
    pub targets: Vec<CompilationTarget>,
    /// Current compilation phase
    pub current_phase: CompilationPhase,
    /// Diagnostic collector
    pub diagnostics: DiagnosticCollector,
    /// Performance profiler
    pub profiler: PerformanceProfiler,
    /// AI metadata collector
    pub ai_metadata_collector: AIMetadataCollector,
    /// Project configuration
    pub project_config: ProjectConfig,
}

/// Compilation target platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilationTarget {
    /// TypeScript transpilation for rapid prototyping
    TypeScript,
    /// WebAssembly for portable execution
    WebAssembly,
    /// Native code via LLVM for performance
    LLVM,
    /// JavaScript for web deployment
    JavaScript,
}

/// Compilation phases
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationPhase {
    /// Project discovery and setup
    Discovery,
    /// Lexical analysis
    Lexing,
    /// Parsing
    Parsing,
    /// Semantic analysis
    SemanticAnalysis,
    /// Type checking
    TypeChecking,
    /// Effect analysis
    EffectAnalysis,
    /// Optimization
    Optimization,
    /// Code generation
    CodeGeneration,
    /// Linking
    Linking,
    /// Finalization
    Finalization,
}

/// Diagnostic collector for errors, warnings, and hints
#[derive(Debug, Clone)]
pub struct DiagnosticCollector {
    /// Collected diagnostics
    pub diagnostics: Vec<Diagnostic>,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
    /// Hint count
    pub hint_count: usize,
}

/// Diagnostic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    /// Error code
    pub code: Option<String>,
    /// Primary message
    pub message: String,
    /// Source location
    pub location: Span,
    /// Additional labels
    pub labels: Vec<DiagnosticLabel>,
    /// Help text
    pub help: Option<String>,
    /// AI-generated suggestions
    pub ai_suggestions: Vec<AISuggestion>,
}

/// Diagnostic severity level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticLevel {
    /// Error (compilation fails)
    Error,
    /// Warning (compilation succeeds but issue noted)
    Warning,
    /// Hint (suggestion for improvement)
    Hint,
    /// Info (informational message)
    Info,
}

/// Diagnostic label for additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticLabel {
    /// Label text
    pub text: String,
    /// Source location
    pub location: Span,
    /// Label style
    pub style: LabelStyle,
}

/// Label styling
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelStyle {
    /// Primary label (main issue)
    Primary,
    /// Secondary label (related context)
    Secondary,
    /// Note label (additional information)
    Note,
}

/// AI-generated suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AISuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Suggestion text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Code replacement
    pub replacement: Option<String>,
    /// Additional context
    pub context: Option<String>,
}

/// Types of AI suggestions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    /// Fix for an error
    Fix,
    /// Performance improvement
    Performance,
    /// Style improvement
    Style,
    /// Semantic improvement
    Semantic,
    /// Security improvement
    Security,
    /// Accessibility improvement
    Accessibility,
}

/// Performance profiler for compilation metrics
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    /// Phase timings
    pub phase_timings: HashMap<CompilationPhase, std::time::Duration>,
    /// Memory usage tracking
    pub memory_usage: MemoryUsageTracker,
    /// Cache performance
    pub cache_performance: CachePerformanceTracker,
    /// Parallel execution metrics
    pub parallel_metrics: ParallelExecutionMetrics,
}

/// Memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryUsageTracker {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Current memory usage
    pub current_memory: usize,
    /// Memory usage by phase
    pub phase_memory: HashMap<CompilationPhase, usize>,
    /// Memory usage by component
    pub component_memory: HashMap<String, usize>,
}

/// Cache performance tracking
#[derive(Debug, Clone)]
pub struct CachePerformanceTracker {
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Cache hit rate by query type
    pub hit_rates: HashMap<String, f64>,
    /// Cache size
    pub cache_size: usize,
}

/// Parallel execution metrics
#[derive(Debug, Clone)]
pub struct ParallelExecutionMetrics {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Thread utilization
    pub thread_utilization: f64,
    /// Work stealing statistics
    pub work_stealing_stats: WorkStealingStats,
    /// Synchronization overhead
    pub sync_overhead: std::time::Duration,
}

/// Work stealing statistics
#[derive(Debug, Clone)]
pub struct WorkStealingStats {
    /// Tasks stolen
    pub tasks_stolen: u64,
    /// Steal attempts
    pub steal_attempts: u64,
    /// Successful steals
    pub successful_steals: u64,
}

/// AI metadata collector
#[derive(Debug, Clone)]
pub struct AIMetadataCollector {
    /// Enable collection
    pub enabled: bool,
    /// Collected semantic contexts
    pub semantic_contexts: Vec<SemanticContextEntry>,
    /// Business rules discovered
    pub business_rules: Vec<BusinessRuleEntry>,
    /// Performance characteristics
    pub performance_characteristics: Vec<PerformanceCharacteristic>,
    /// Security implications
    pub security_implications: Vec<SecurityImplication>,
}

/// Semantic context entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContextEntry {
    /// Location in source
    pub location: Span,
    /// Context type
    pub context_type: SemanticContextType,
    /// Semantic information
    pub semantic_info: String,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// AI confidence
    pub confidence: f64,
}

/// Types of semantic contexts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticContextType {
    /// Business logic
    BusinessLogic,
    /// Data validation
    DataValidation,
    /// Error handling
    ErrorHandling,
    /// Performance critical
    PerformanceCritical,
    /// Security sensitive
    SecuritySensitive,
    /// User interface
    UserInterface,
}

/// Business rule entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRuleEntry {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Source location
    pub location: Span,
    /// Rule category
    pub category: BusinessRuleCategory,
    /// Enforcement level
    pub enforcement: EnforcementLevel,
}

/// Business rule categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessRuleCategory {
    /// Data validation
    Validation,
    /// Business constraint
    Constraint,
    /// Workflow rule
    Workflow,
    /// Compliance requirement
    Compliance,
    /// Security policy
    Security,
}

/// Rule enforcement levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced (compile error if violated)
    Required,
    /// Should be enforced (warning if violated)
    Recommended,
    /// Optional enforcement (hint if violated)
    Optional,
}

/// Performance characteristic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristic {
    /// Location in source
    pub location: Span,
    /// Characteristic type
    pub characteristic_type: PerformanceCharacteristicType,
    /// Description
    pub description: String,
    /// Complexity analysis
    pub complexity: Option<String>,
    /// Optimization suggestions
    pub optimizations: Vec<String>,
}

/// Types of performance characteristics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceCharacteristicType {
    /// Time complexity
    TimeComplexity,
    /// Space complexity
    SpaceComplexity,
    /// I/O intensive
    IOIntensive,
    /// CPU intensive
    CPUIntensive,
    /// Memory intensive
    MemoryIntensive,
    /// Network intensive
    NetworkIntensive,
}

/// Security implication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplication {
    /// Location in source
    pub location: Span,
    /// Security category
    pub category: SecurityCategory,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Description
    pub description: String,
    /// Mitigation suggestions
    pub mitigations: Vec<String>,
}

/// Security categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityCategory {
    /// Input validation
    InputValidation,
    /// Authentication
    Authentication,
    /// Authorization
    Authorization,
    /// Data encryption
    DataEncryption,
    /// SQL injection
    SQLInjection,
    /// Cross-site scripting
    XSS,
    /// Buffer overflow
    BufferOverflow,
    /// Information disclosure
    InformationDisclosure,
}

/// Risk levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Critical risk
    Critical,
    /// High risk
    High,
    /// Medium risk
    Medium,
    /// Low risk
    Low,
    /// Informational
    Info,
}

/// Project configuration
#[derive(Debug, Clone)]
pub struct ProjectConfig {
    /// Project name
    pub name: String,
    /// Project version
    pub version: String,
    /// Root directory
    pub root_dir: PathBuf,
    /// Source directories
    pub source_dirs: Vec<PathBuf>,
    /// Output directory
    pub output_dir: PathBuf,
    /// Compilation configuration
    pub compilation_config: CompilationConfig,
}

/// Compilation configuration
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Project root directory
    pub project_root: PathBuf,
    /// Target platforms to compile for
    pub targets: Vec<CompilationTarget>,
    /// Optimization level
    pub optimization_level: u8,
    /// Enable language server
    pub enable_language_server: Option<bool>,
    /// Export AI context
    pub export_ai_context: bool,
    /// Enable incremental compilation
    pub incremental: Option<bool>,
    /// Enable AI features
    pub ai_features: Option<bool>,
    /// Debug information
    pub debug_info: Option<bool>,
    /// Enable AST transformations
    pub enable_transformations: Option<bool>,
    /// Transformation configuration
    pub transformation_config: Option<prism_ast::TransformationConfig>,
    /// Additional compiler flags
    pub compiler_flags: HashMap<String, String>,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            project_root: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            targets: vec![CompilationTarget::TypeScript],
            optimization_level: 2,
            enable_language_server: Some(false),
            export_ai_context: false,
            incremental: Some(true),
            ai_features: Some(true),
            debug_info: Some(true),
            enable_transformations: Some(true),
            transformation_config: None, // Uses default config if None
            compiler_flags: HashMap::new(),
        }
    }
}

impl CompilationContext {
    /// Create a new compilation context
    pub fn new(config: &crate::CompilerConfig) -> CompilerResult<Self> {
        Ok(Self {
            targets: config.targets.clone(),
            current_phase: CompilationPhase::Discovery,
            diagnostics: DiagnosticCollector::new(),
            profiler: PerformanceProfiler::new(),
            ai_metadata_collector: AIMetadataCollector::new(config.ai_metadata),
            project_config: ProjectConfig::default(),
        })
    }

    /// Set current compilation phase
    pub fn set_phase(&mut self, phase: CompilationPhase) {
        self.current_phase = phase;
        self.profiler.start_phase(&phase);
    }

    /// Add diagnostic
    pub fn add_diagnostic(&mut self, diagnostic: Diagnostic) {
        match diagnostic.level {
            DiagnosticLevel::Error => self.diagnostics.error_count += 1,
            DiagnosticLevel::Warning => self.diagnostics.warning_count += 1,
            DiagnosticLevel::Hint => self.diagnostics.hint_count += 1,
            DiagnosticLevel::Info => {},
        }
        self.diagnostics.diagnostics.push(diagnostic);
    }

    /// Check if compilation should continue
    pub fn should_continue(&self) -> bool {
        self.diagnostics.error_count == 0
    }

    /// Get compilation statistics
    pub fn get_statistics(&self) -> CompilationStatistics {
        CompilationStatistics {
            total_time: self.profiler.total_time(),
            phase_timings: self.profiler.phase_timings.clone(),
            memory_usage: self.profiler.memory_usage.clone(),
            cache_performance: self.profiler.cache_performance.clone(),
            parallel_metrics: self.profiler.parallel_metrics.clone(),
            diagnostic_counts: DiagnosticCounts {
                errors: self.diagnostics.error_count,
                warnings: self.diagnostics.warning_count,
                hints: self.diagnostics.hint_count,
            },
        }
    }

    /// Export AI metadata
    pub fn export_ai_metadata(&self) -> AIMetadataExport {
        AIMetadataExport {
            semantic_contexts: self.ai_metadata_collector.semantic_contexts.clone(),
            business_rules: self.ai_metadata_collector.business_rules.clone(),
            performance_characteristics: self.ai_metadata_collector.performance_characteristics.clone(),
            security_implications: self.ai_metadata_collector.security_implications.clone(),
        }
    }
}

/// Compilation statistics
#[derive(Debug, Clone)]
pub struct CompilationStatistics {
    /// Total compilation time
    pub total_time: std::time::Duration,
    /// Time per phase
    pub phase_timings: HashMap<CompilationPhase, std::time::Duration>,
    /// Memory usage
    pub memory_usage: MemoryUsageTracker,
    /// Cache performance
    pub cache_performance: CachePerformanceTracker,
    /// Parallel execution metrics
    pub parallel_metrics: ParallelExecutionMetrics,
    /// Diagnostic counts
    pub diagnostic_counts: DiagnosticCounts,
}

/// Diagnostic counts
#[derive(Debug, Clone)]
pub struct DiagnosticCounts {
    /// Number of errors
    pub errors: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Number of hints
    pub hints: usize,
}

/// AI metadata export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadataExport {
    /// Semantic contexts
    pub semantic_contexts: Vec<SemanticContextEntry>,
    /// Business rules
    pub business_rules: Vec<BusinessRuleEntry>,
    /// Performance characteristics
    pub performance_characteristics: Vec<PerformanceCharacteristic>,
    /// Security implications
    pub security_implications: Vec<SecurityImplication>,
}

impl DiagnosticCollector {
    /// Create a new diagnostic collector
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            error_count: 0,
            warning_count: 0,
            hint_count: 0,
        }
    }

    /// Add a diagnostic
    pub fn add(&mut self, diagnostic: Diagnostic) {
        match diagnostic.level {
            DiagnosticLevel::Error => self.error_count += 1,
            DiagnosticLevel::Warning => self.warning_count += 1,
            DiagnosticLevel::Hint => self.hint_count += 1,
            DiagnosticLevel::Info => {},
        }
        self.diagnostics.push(diagnostic);
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get all diagnostics
    pub fn get_diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            phase_timings: HashMap::new(),
            memory_usage: MemoryUsageTracker::new(),
            cache_performance: CachePerformanceTracker::new(),
            parallel_metrics: ParallelExecutionMetrics::new(),
        }
    }

    /// Start timing a phase
    pub fn start_phase(&mut self, phase: &CompilationPhase) {
        // Implementation would start timing the phase
        // For now, we'll just record a placeholder
        self.phase_timings.insert(phase.clone(), std::time::Duration::from_millis(0));
    }

    /// Get total compilation time
    pub fn total_time(&self) -> std::time::Duration {
        self.phase_timings.values().sum()
    }
}

impl MemoryUsageTracker {
    /// Create a new memory usage tracker
    pub fn new() -> Self {
        Self {
            peak_memory: 0,
            current_memory: 0,
            phase_memory: HashMap::new(),
            component_memory: HashMap::new(),
        }
    }
}

impl CachePerformanceTracker {
    /// Create a new cache performance tracker
    pub fn new() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            hit_rates: HashMap::new(),
            cache_size: 0,
        }
    }

    /// Calculate overall hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }
}

impl ParallelExecutionMetrics {
    /// Create a new parallel execution metrics tracker
    pub fn new() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            thread_utilization: 0.0,
            work_stealing_stats: WorkStealingStats {
                tasks_stolen: 0,
                steal_attempts: 0,
                successful_steals: 0,
            },
            sync_overhead: std::time::Duration::from_millis(0),
        }
    }
}

impl AIMetadataCollector {
    /// Create a new AI metadata collector
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            semantic_contexts: Vec::new(),
            business_rules: Vec::new(),
            performance_characteristics: Vec::new(),
            security_implications: Vec::new(),
        }
    }

    /// Add semantic context
    pub fn add_semantic_context(&mut self, entry: SemanticContextEntry) {
        if self.enabled {
            self.semantic_contexts.push(entry);
        }
    }

    /// Add business rule
    pub fn add_business_rule(&mut self, rule: BusinessRuleEntry) {
        if self.enabled {
            self.business_rules.push(rule);
        }
    }

    /// Add performance characteristic
    pub fn add_performance_characteristic(&mut self, characteristic: PerformanceCharacteristic) {
        if self.enabled {
            self.performance_characteristics.push(characteristic);
        }
    }

    /// Add security implication
    pub fn add_security_implication(&mut self, implication: SecurityImplication) {
        if self.enabled {
            self.security_implications.push(implication);
        }
    }
}

impl ProjectConfig {
    /// Create default project configuration
    pub fn default() -> Self {
        Self {
            name: "prism-project".to_string(),
            version: "0.1.0".to_string(),
            root_dir: PathBuf::from("."),
            source_dirs: vec![PathBuf::from("src")],
            output_dir: PathBuf::from("target"),
            compilation_config: CompilationConfig::default(),
        }
    }
}

impl std::fmt::Display for CompilationTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilationTarget::TypeScript => write!(f, "typescript"),
            CompilationTarget::WebAssembly => write!(f, "wasm"),
            CompilationTarget::LLVM => write!(f, "llvm"),
            CompilationTarget::JavaScript => write!(f, "javascript"),
        }
    }
}

impl std::fmt::Display for CompilationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilationPhase::Discovery => write!(f, "discovery"),
            CompilationPhase::Lexing => write!(f, "lexing"),
            CompilationPhase::Parsing => write!(f, "parsing"),
            CompilationPhase::SemanticAnalysis => write!(f, "semantic-analysis"),
            CompilationPhase::TypeChecking => write!(f, "type-checking"),
            CompilationPhase::EffectAnalysis => write!(f, "effect-analysis"),
            CompilationPhase::Optimization => write!(f, "optimization"),
            CompilationPhase::CodeGeneration => write!(f, "code-generation"),
            CompilationPhase::Linking => write!(f, "linking"),
            CompilationPhase::Finalization => write!(f, "finalization"),
        }
    }
} 