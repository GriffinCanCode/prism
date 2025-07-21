//! Context Management Subsystem
//!
//! This subsystem implements comprehensive context management for the Prism compiler,
//! following PLT-004 specifications. It provides:
//!
//! - **Compilation context tracking** with phase management
//! - **Diagnostic collection** with AI-enhanced error reporting
//! - **Performance profiling** with detailed metrics collection
//! - **AI metadata collection** for external tool integration
//! - **Project configuration** management and validation
//! - **Context building** utilities for proper initialization
//!
//! ## Conceptual Cohesion
//!
//! This subsystem embodies the single concept of "Compilation Context Management".
//! It does NOT handle:
//! - Symbol management (delegated to symbols subsystem)
//! - Scope management (delegated to scope subsystem)
//! - Query execution (delegated to query engine)
//! - Code generation (delegated to codegen backends)
//!
//! ## Architecture
//!
//! ```
//! context/
//! ├── mod.rs           # Public API and re-exports
//! ├── compilation.rs   # CompilationContext - main compilation state
//! ├── diagnostics.rs   # Diagnostic collection and reporting
//! ├── profiling.rs     # Performance profiling and metrics
//! ├── metadata.rs      # AI metadata collection
//! ├── project.rs       # Project configuration management
//! └── builder.rs       # Context construction utilities
//! ```

// Core context management modules
pub mod compilation;
pub mod diagnostics;
pub mod profiling;
pub mod metadata;
pub mod project;
pub mod builder;

// Re-export main types for convenience
pub use compilation::{
    CompilationContext, CompilationPhase, CompilationTarget,
    CompilationStatistics, DiagnosticCounts
};
pub use diagnostics::{
    DiagnosticCollector, Diagnostic, DiagnosticLevel, DiagnosticLabel, LabelStyle,
    AISuggestion, SuggestionType
};
pub use profiling::{
    PerformanceProfiler, MemoryUsageTracker, CachePerformanceTracker,
    ParallelExecutionMetrics, WorkStealingStats
};
pub use metadata::{
    AIMetadataCollector, SemanticContextEntry, SemanticContextType,
    BusinessRuleEntry, BusinessRuleCategory, EnforcementLevel,
    PerformanceCharacteristic, PerformanceCharacteristicType,
    SecurityImplication, SecurityCategory, RiskLevel,
    AIMetadataExport
};
pub use project::{
    ProjectConfig, CompilationConfig
};
pub use builder::{
    ContextBuilder, ContextBuilderConfig, BuildPhase
};

// Type aliases for external use
pub type CompilerConfig = CompilationConfig; // Alias for backward compatibility

/// Main context management interface
pub trait ContextManager {
    /// Get current compilation phase
    fn current_phase(&self) -> CompilationPhase;
    
    /// Set compilation phase
    fn set_phase(&mut self, phase: CompilationPhase);
    
    /// Add diagnostic message
    fn add_diagnostic(&mut self, diagnostic: Diagnostic);
    
    /// Check if compilation should continue
    fn should_continue(&self) -> bool;
    
    /// Get compilation statistics
    fn get_statistics(&self) -> CompilationStatistics;
}

/// Context query interface for retrieving context information
pub trait ContextQuery {
    /// Get diagnostics by level
    fn get_diagnostics_by_level(&self, level: DiagnosticLevel) -> Vec<&Diagnostic>;
    
    /// Get performance metrics for phase
    fn get_phase_metrics(&self, phase: CompilationPhase) -> Option<std::time::Duration>;
    
    /// Get AI metadata for export
    fn get_ai_metadata(&self) -> AIMetadataExport;
    
    /// Get project information
    fn get_project_info(&self) -> &ProjectConfig;
}

/// Context modification interface
pub trait ContextModifier {
    /// Update project configuration
    fn update_project_config<F>(&mut self, updater: F) -> crate::error::CompilerResult<()>
    where
        F: FnOnce(&mut ProjectConfig);
    
    /// Enable/disable AI metadata collection
    fn set_ai_metadata_enabled(&mut self, enabled: bool);
    
    /// Add performance note
    fn add_performance_note(&mut self, note: String, phase: CompilationPhase);
    
    /// Clear diagnostics
    fn clear_diagnostics(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_subsystem_exports() {
        // Ensure all main types are properly exported
        let _phase = CompilationPhase::Discovery;
        let _target = CompilationTarget::TypeScript;
        let _level = DiagnosticLevel::Error;
        let _collector = AIMetadataCollector::new(true);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::context::{CompilationContext, ContextBuilder};
    use prism_common::span::Span;
    
    #[test]
    fn test_context_subsystem_integration() {
        // Test context builder
        let builder = ContextBuilder::new();
        let context = builder
            .with_targets(vec![CompilationTarget::TypeScript])
            .with_ai_metadata_enabled(true)
            .build()
            .unwrap();
        
        // Test context management
        assert_eq!(context.current_phase(), CompilationPhase::Discovery);
        assert!(context.should_continue());
        
        // Test statistics
        let stats = context.get_statistics();
        assert_eq!(stats.diagnostic_counts.errors, 0);
    }
    
    #[test]
    fn test_diagnostic_integration() {
        let mut context = ContextBuilder::new().build().unwrap();
        
        // Add diagnostic
        let diagnostic = Diagnostic {
            level: DiagnosticLevel::Warning,
            code: Some("W001".to_string()),
            message: "Test warning".to_string(),
            location: Span::dummy(),
            labels: vec![],
            help: None,
            ai_suggestions: vec![],
        };
        
        context.add_diagnostic(diagnostic);
        
        // Check statistics
        let stats = context.get_statistics();
        assert_eq!(stats.diagnostic_counts.warnings, 1);
    }
    
    #[test]
    fn test_ai_metadata_integration() {
        let mut context = ContextBuilder::new()
            .with_ai_metadata_enabled(true)
            .build()
            .unwrap();
        
        // Add semantic context
        let semantic_entry = SemanticContextEntry {
            location: Span::dummy(),
            context_type: SemanticContextType::BusinessLogic,
            semantic_info: "Test business logic".to_string(),
            related_concepts: vec!["validation".to_string()],
            confidence: 0.9,
        };
        
        context.ai_metadata_collector.add_semantic_context(semantic_entry);
        
        // Export metadata
        let export = context.export_ai_metadata();
        assert_eq!(export.semantic_contexts.len(), 1);
    }
} 