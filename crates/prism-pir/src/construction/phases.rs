//! PIR Construction Phases - Multi-Stage Transformation Pipeline
//!
//! This module defines the construction phases that transform AST to PIR
//! through a series of well-defined stages, each with specific responsibilities.

use crate::{PIRResult, PIRError, semantic::*};
use prism_ast::Program;
use prism_common::{NodeId, span::Span};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use async_trait::async_trait;

/// Trait for construction phases
#[async_trait]
pub trait ConstructionPhase: Send + Sync {
    /// Phase name for identification
    fn name(&self) -> &'static str;
    
    /// Phase description
    fn description(&self) -> &'static str;
    
    /// Execute the phase
    async fn execute(&self, context: &mut PhaseContext) -> PIRResult<PhaseResult>;
    
    /// Check if this phase can run in parallel with others
    fn can_run_parallel(&self) -> bool {
        false
    }
    
    /// Get dependencies - phases that must complete before this one
    fn dependencies(&self) -> Vec<&'static str> {
        Vec::new()
    }
}

/// Context shared between construction phases
#[derive(Debug)]
pub struct PhaseContext {
    /// Input AST program
    pub program: Program,
    /// Partial PIR being constructed
    pub pir: Option<PrismIR>,
    /// Semantic analysis results
    pub semantic_results: HashMap<NodeId, SemanticInfo>,
    /// Business context extracted so far
    pub business_contexts: HashMap<String, crate::business::BusinessContext>,
    /// Effect analysis results
    pub effect_results: HashMap<NodeId, Vec<String>>,
    /// Validation results from previous phases
    pub validation_results: Vec<ValidationIssue>,
    /// Performance metrics
    pub metrics: PhaseMetrics,
}

/// Result of a construction phase
#[derive(Debug, Clone)]
pub struct PhaseResult {
    /// Success status
    pub success: bool,
    /// Time taken to execute
    pub execution_time: Duration,
    /// Items processed
    pub items_processed: u64,
    /// Warnings generated
    pub warnings: Vec<String>,
    /// Errors encountered
    pub errors: Vec<String>,
}

/// Performance metrics for phases
#[derive(Debug, Default)]
pub struct PhaseMetrics {
    /// Total time spent in all phases
    pub total_time: Duration,
    /// Time per phase
    pub phase_times: HashMap<String, Duration>,
    /// Memory usage per phase
    pub memory_usage: HashMap<String, u64>,
    /// Items processed per phase
    pub items_processed: HashMap<String, u64>,
}

/// Validation issue from construction
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue message
    pub message: String,
    /// Source location
    pub span: Option<Span>,
    /// Phase that detected the issue
    pub phase: String,
}

/// Issue severity levels
#[derive(Debug, Clone)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Semantic information for a node
#[derive(Debug, Clone)]
pub struct SemanticInfo {
    /// Node type information
    pub type_info: Option<String>,
    /// Business context
    pub business_context: Option<String>,
    /// Effect information
    pub effects: Vec<String>,
}

/// Phase 1: Semantic Extraction
pub struct SemanticExtractionPhase;

#[async_trait]
impl ConstructionPhase for SemanticExtractionPhase {
    fn name(&self) -> &'static str {
        "semantic_extraction"
    }
    
    fn description(&self) -> &'static str {
        "Extract semantic information from AST nodes"
    }
    
    async fn execute(&self, context: &mut PhaseContext) -> PIRResult<PhaseResult> {
        let start_time = Instant::now();
        let mut items_processed = 0;
        let mut warnings = Vec::new();
        
        // Extract semantic information from each item in the program
        for item in &context.program.items {
            match self.extract_semantic_info(item) {
                Ok(info) => {
                    context.semantic_results.insert(NodeId::new(), info);
                    items_processed += 1;
                }
                Err(e) => {
                    warnings.push(format!("Failed to extract semantic info: {}", e));
                }
            }
        }
        
        let execution_time = start_time.elapsed();
        context.metrics.phase_times.insert(self.name().to_string(), execution_time);
        context.metrics.items_processed.insert(self.name().to_string(), items_processed);
        
        Ok(PhaseResult {
            success: true,
            execution_time,
            items_processed,
            warnings,
            errors: Vec::new(),
        })
    }
    
    fn can_run_parallel(&self) -> bool {
        true // Can run in parallel with business context extraction
    }
}

impl SemanticExtractionPhase {
    fn extract_semantic_info(&self, item: &prism_ast::AstNode<prism_ast::Item>) -> PIRResult<SemanticInfo> {
        // Simplified semantic extraction
        match &item.kind {
            prism_ast::Item::Function(_) => {
                Ok(SemanticInfo {
                    type_info: Some("function".to_string()),
                    business_context: Some("computation".to_string()),
                    effects: vec!["computation".to_string()],
                })
            }
            prism_ast::Item::Type(_) => {
                Ok(SemanticInfo {
                    type_info: Some("type_definition".to_string()),
                    business_context: Some("data_modeling".to_string()),
                    effects: Vec::new(),
                })
            }
            prism_ast::Item::Module(_) => {
                Ok(SemanticInfo {
                    type_info: Some("module".to_string()),
                    business_context: Some("organization".to_string()),
                    effects: vec!["organization".to_string()],
                })
            }
            _ => {
                Ok(SemanticInfo {
                    type_info: None,
                    business_context: None,
                    effects: Vec::new(),
                })
            }
        }
    }
}

/// Phase 2: Business Context Extraction
pub struct BusinessContextPhase;

#[async_trait]
impl ConstructionPhase for BusinessContextPhase {
    fn name(&self) -> &'static str {
        "business_context"
    }
    
    fn description(&self) -> &'static str {
        "Extract business context and domain knowledge"
    }
    
    async fn execute(&self, context: &mut PhaseContext) -> PIRResult<PhaseResult> {
        let start_time = Instant::now();
        let mut items_processed = 0;
        
        // Extract business contexts from modules
        for item in &context.program.items {
            if let prism_ast::Item::Module(module) = &item.kind {
                let business_context = self.extract_business_context(module)?;
                let module_name = module.name.resolve().unwrap_or_else(|| "unknown".to_string());
                context.business_contexts.insert(module_name, business_context);
                items_processed += 1;
            }
        }
        
        let execution_time = start_time.elapsed();
        context.metrics.phase_times.insert(self.name().to_string(), execution_time);
        context.metrics.items_processed.insert(self.name().to_string(), items_processed);
        
        Ok(PhaseResult {
            success: true,
            execution_time,
            items_processed,
            warnings: Vec::new(),
            errors: Vec::new(),
        })
    }
    
    fn can_run_parallel(&self) -> bool {
        true // Can run in parallel with semantic extraction
    }
}

impl BusinessContextPhase {
    fn extract_business_context(&self, module: &prism_ast::ModuleDecl) -> PIRResult<crate::business::BusinessContext> {
        let domain = module.name.resolve().unwrap_or_else(|| "unknown".to_string());
        Ok(crate::business::BusinessContext::new(domain))
    }
}

/// Phase 3: Effect Analysis
pub struct EffectAnalysisPhase;

#[async_trait]
impl ConstructionPhase for EffectAnalysisPhase {
    fn name(&self) -> &'static str {
        "effect_analysis"
    }
    
    fn description(&self) -> &'static str {
        "Analyze computational effects and capabilities"
    }
    
    async fn execute(&self, context: &mut PhaseContext) -> PIRResult<PhaseResult> {
        let start_time = Instant::now();
        let mut items_processed = 0;
        
        // Analyze effects for each function
        for item in &context.program.items {
            if let prism_ast::Item::Function(func) = &item.kind {
                let effects = self.analyze_function_effects(func)?;
                context.effect_results.insert(NodeId::new(), effects);
                items_processed += 1;
            }
        }
        
        let execution_time = start_time.elapsed();
        context.metrics.phase_times.insert(self.name().to_string(), execution_time);
        
        Ok(PhaseResult {
            success: true,
            execution_time,
            items_processed,
            warnings: Vec::new(),
            errors: Vec::new(),
        })
    }
    
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["semantic_extraction"] // Needs semantic info first
    }
}

impl EffectAnalysisPhase {
    fn analyze_function_effects(&self, _func: &prism_ast::FunctionDecl) -> PIRResult<Vec<String>> {
        // Simplified effect analysis
        Ok(vec!["computation".to_string(), "memory".to_string()])
    }
}

/// Phase 4: PIR Construction Validation
pub struct ValidationPhase;

#[async_trait]
impl ConstructionPhase for ValidationPhase {
    fn name(&self) -> &'static str {
        "validation"
    }
    
    fn description(&self) -> &'static str {
        "Validate PIR construction and semantic preservation"
    }
    
    async fn execute(&self, context: &mut PhaseContext) -> PIRResult<PhaseResult> {
        let start_time = Instant::now();
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // Validate that we have a PIR to validate
        if let Some(ref pir) = context.pir {
            // Check semantic preservation
            for module in &pir.modules {
                if module.sections.is_empty() {
                    warnings.push(format!("Module '{}' has no sections", module.name));
                }
                
                if module.cohesion_score < 0.5 {
                    warnings.push(format!("Module '{}' has low cohesion score", module.name));
                }
            }
        } else {
            errors.push("No PIR available for validation".to_string());
        }
        
        let execution_time = start_time.elapsed();
        context.metrics.phase_times.insert(self.name().to_string(), execution_time);
        
        Ok(PhaseResult {
            success: errors.is_empty(),
            execution_time,
            items_processed: context.pir.as_ref().map(|p| p.modules.len() as u64).unwrap_or(0),
            warnings,
            errors,
        })
    }
    
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["semantic_extraction", "business_context", "effect_analysis"]
    }
}

/// Phase 5: Performance Optimization
pub struct OptimizationPhase;

#[async_trait]
impl ConstructionPhase for OptimizationPhase {
    fn name(&self) -> &'static str {
        "optimization"
    }
    
    fn description(&self) -> &'static str {
        "Optimize PIR for performance and efficiency"
    }
    
    async fn execute(&self, context: &mut PhaseContext) -> PIRResult<PhaseResult> {
        let start_time = Instant::now();
        let mut items_processed = 0;
        
        // Apply optimizations to PIR
        if let Some(ref mut pir) = context.pir {
            for module in &mut pir.modules {
                // Optimize module structure
                items_processed += self.optimize_module(module)?;
            }
        }
        
        let execution_time = start_time.elapsed();
        context.metrics.phase_times.insert(self.name().to_string(), execution_time);
        
        Ok(PhaseResult {
            success: true,
            execution_time,
            items_processed,
            warnings: Vec::new(),
            errors: Vec::new(),
        })
    }
    
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["validation"] // Only optimize after validation passes
    }
}

impl OptimizationPhase {
    fn optimize_module(&self, module: &mut PIRModule) -> PIRResult<u64> {
        // Simplified optimization - improve cohesion score
        if module.cohesion_score < 0.8 {
            module.cohesion_score = (module.cohesion_score + 0.1).min(1.0);
        }
        Ok(1)
    }
}

impl PhaseContext {
    /// Create a new phase context
    pub fn new(program: Program) -> Self {
        Self {
            program,
            pir: None,
            semantic_results: HashMap::new(),
            business_contexts: HashMap::new(),
            effect_results: HashMap::new(),
            validation_results: Vec::new(),
            metrics: PhaseMetrics::default(),
        }
    }
    
    /// Set the PIR being constructed
    pub fn set_pir(&mut self, pir: PrismIR) {
        self.pir = Some(pir);
    }
    
    /// Get the constructed PIR
    pub fn take_pir(&mut self) -> Option<PrismIR> {
        self.pir.take()
    }
} 