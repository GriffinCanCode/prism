//! Compilation Context - Main Compilation State Management
//!
//! This module implements the core compilation context that tracks the overall
//! compilation state, current phase, and coordinates with other context components.
//!
//! **Conceptual Responsibility**: Compilation state and phase management
//! **What it does**: Track compilation phase, coordinate subsystems, manage compilation flow
//! **What it doesn't do**: Collect diagnostics, profile performance, manage project config

use crate::error::{CompilerError, CompilerResult};
use crate::context::{
    DiagnosticCollector, PerformanceProfiler, AIMetadataCollector, ProjectConfig,
    ContextManager, ContextQuery, ContextModifier
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Main compilation context that tracks state during compilation
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
    /// Prism VM for unified debugging and runtime optimization
    PrismVM,
}

/// Compilation phases
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// PIR generation
    PIRGeneration,
    /// Linking
    Linking,
    /// Finalization
    Finalization,
}

/// Compilation statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStatistics {
    /// Total compilation time
    pub total_time: std::time::Duration,
    /// Time per phase
    pub phase_timings: HashMap<CompilationPhase, std::time::Duration>,
    /// Memory usage tracking
    pub memory_usage: crate::context::MemoryUsageTracker,
    /// Cache performance metrics
    pub cache_performance: crate::context::CachePerformanceTracker,
    /// Parallel execution metrics
    pub parallel_metrics: crate::context::ParallelExecutionMetrics,
    /// Diagnostic counts summary
    pub diagnostic_counts: DiagnosticCounts,
}

/// Summary of diagnostic counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticCounts {
    /// Number of errors
    pub errors: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Number of hints
    pub hints: usize,
}

impl CompilationContext {
    /// Create a new compilation context with default settings
    pub fn new(targets: Vec<CompilationTarget>) -> CompilerResult<Self> {
        Ok(Self {
            targets,
            current_phase: CompilationPhase::Discovery,
            diagnostics: DiagnosticCollector::new(),
            profiler: PerformanceProfiler::new(),
            ai_metadata_collector: AIMetadataCollector::new(true),
            project_config: ProjectConfig::default(),
        })
    }

    /// Create a new compilation context from compiler config
    pub fn from_config(config: &crate::context::CompilationConfig) -> CompilerResult<Self> {
        Ok(Self {
            targets: config.targets.clone(),
            current_phase: CompilationPhase::Discovery,
            diagnostics: DiagnosticCollector::new(),
            profiler: PerformanceProfiler::new(),
            ai_metadata_collector: AIMetadataCollector::new(config.ai_features.unwrap_or(true)),
            project_config: ProjectConfig::from_compilation_config(config),
        })
    }

    /// Export AI metadata collected during compilation
    pub fn export_ai_metadata(&self) -> crate::context::AIMetadataExport {
        crate::context::AIMetadataExport {
            semantic_contexts: self.ai_metadata_collector.semantic_contexts.clone(),
            business_rules: self.ai_metadata_collector.business_rules.clone(),
            performance_characteristics: self.ai_metadata_collector.performance_characteristics.clone(),
            security_implications: self.ai_metadata_collector.security_implications.clone(),
        }
    }

    /// Get a summary of compilation progress
    pub fn progress_summary(&self) -> String {
        let phase_name = format!("{:?}", self.current_phase);
        let error_count = self.diagnostics.error_count;
        let warning_count = self.diagnostics.warning_count;
        
        if error_count > 0 {
            format!("{} - {} errors, {} warnings", phase_name, error_count, warning_count)
        } else if warning_count > 0 {
            format!("{} - {} warnings", phase_name, warning_count)
        } else {
            format!("{} - OK", phase_name)
        }
    }

    /// Check if compilation is in an error state
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    /// Get the next expected compilation phase
    pub fn next_phase(&self) -> Option<CompilationPhase> {
        match self.current_phase {
            CompilationPhase::Discovery => Some(CompilationPhase::Lexing),
            CompilationPhase::Lexing => Some(CompilationPhase::Parsing),
            CompilationPhase::Parsing => Some(CompilationPhase::SemanticAnalysis),
            CompilationPhase::SemanticAnalysis => Some(CompilationPhase::TypeChecking),
            CompilationPhase::TypeChecking => Some(CompilationPhase::EffectAnalysis),
            CompilationPhase::EffectAnalysis => Some(CompilationPhase::Optimization),
            CompilationPhase::Optimization => Some(CompilationPhase::PIRGeneration),
            CompilationPhase::PIRGeneration => Some(CompilationPhase::CodeGeneration),
            CompilationPhase::CodeGeneration => Some(CompilationPhase::Linking),
            CompilationPhase::Linking => Some(CompilationPhase::Finalization),
            CompilationPhase::Finalization => None,
        }
    }

    /// Advance to the next compilation phase
    pub fn advance_phase(&mut self) -> CompilerResult<()> {
        if let Some(next_phase) = self.next_phase() {
            self.set_phase(next_phase);
            Ok(())
        } else {
            Err(CompilerError::InvalidOperation {
                message: "Cannot advance past finalization phase".to_string(),
            })
        }
    }
}

impl ContextManager for CompilationContext {
    fn current_phase(&self) -> CompilationPhase {
        self.current_phase.clone()
    }

    fn set_phase(&mut self, phase: CompilationPhase) {
        self.current_phase = phase.clone();
        self.profiler.start_phase(&phase);
    }

    fn add_diagnostic(&mut self, diagnostic: crate::context::Diagnostic) {
        self.diagnostics.add(diagnostic);
    }

    fn should_continue(&self) -> bool {
        !self.has_errors()
    }

    fn get_statistics(&self) -> CompilationStatistics {
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
}

impl ContextQuery for CompilationContext {
    fn get_diagnostics_by_level(&self, level: crate::context::DiagnosticLevel) -> Vec<&crate::context::Diagnostic> {
        self.diagnostics.diagnostics.iter()
            .filter(|d| d.level == level)
            .collect()
    }

    fn get_phase_metrics(&self, phase: CompilationPhase) -> Option<std::time::Duration> {
        self.profiler.phase_timings.get(&phase).cloned()
    }

    fn get_ai_metadata(&self) -> crate::context::AIMetadataExport {
        self.export_ai_metadata()
    }

    fn get_project_info(&self) -> &ProjectConfig {
        &self.project_config
    }
}

impl ContextModifier for CompilationContext {
    fn update_project_config<F>(&mut self, updater: F) -> CompilerResult<()>
    where
        F: FnOnce(&mut ProjectConfig),
    {
        updater(&mut self.project_config);
        Ok(())
    }

    fn set_ai_metadata_enabled(&mut self, enabled: bool) {
        self.ai_metadata_collector.enabled = enabled;
    }

    fn add_performance_note(&mut self, note: String, _phase: CompilationPhase) {
        // Add to profiler's notes (would need to extend PerformanceProfiler)
        // For now, we'll add it as a custom field in project config
        self.project_config.custom_fields.insert(
            format!("performance_note_{}", chrono::Utc::now().timestamp()),
            note,
        );
    }

    fn clear_diagnostics(&mut self) {
        self.diagnostics = DiagnosticCollector::new();
    }
}

impl std::fmt::Display for CompilationTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilationTarget::TypeScript => write!(f, "typescript"),
            CompilationTarget::WebAssembly => write!(f, "wasm"),
            CompilationTarget::LLVM => write!(f, "llvm"),
            CompilationTarget::JavaScript => write!(f, "javascript"),
            CompilationTarget::PrismVM => write!(f, "prism-vm"),
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
            CompilationPhase::PIRGeneration => write!(f, "pir-generation"),
            CompilationPhase::CodeGeneration => write!(f, "code-generation"),
            CompilationPhase::Linking => write!(f, "linking"),
            CompilationPhase::Finalization => write!(f, "finalization"),
        }
    }
}

impl Default for CompilationPhase {
    fn default() -> Self {
        CompilationPhase::Discovery
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_context_creation() {
        let targets = vec![CompilationTarget::TypeScript];
        let context = CompilationContext::new(targets).unwrap();
        
        assert_eq!(context.current_phase(), CompilationPhase::Discovery);
        assert!(context.should_continue());
        assert!(!context.has_errors());
    }

    #[test]
    fn test_phase_advancement() {
        let mut context = CompilationContext::new(vec![CompilationTarget::TypeScript]).unwrap();
        
        assert_eq!(context.current_phase(), CompilationPhase::Discovery);
        
        context.advance_phase().unwrap();
        assert_eq!(context.current_phase(), CompilationPhase::Lexing);
        
        context.advance_phase().unwrap();
        assert_eq!(context.current_phase(), CompilationPhase::Parsing);
    }

    #[test]
    fn test_phase_sequence() {
        let context = CompilationContext::new(vec![CompilationTarget::TypeScript]).unwrap();
        
        assert_eq!(context.next_phase(), Some(CompilationPhase::Lexing));
        
        let mut context = context;
        context.set_phase(CompilationPhase::Finalization);
        assert_eq!(context.next_phase(), None);
    }

    #[test]
    fn test_progress_summary() {
        let context = CompilationContext::new(vec![CompilationTarget::TypeScript]).unwrap();
        let summary = context.progress_summary();
        
        assert!(summary.contains("Discovery"));
        assert!(summary.contains("OK"));
    }
} 