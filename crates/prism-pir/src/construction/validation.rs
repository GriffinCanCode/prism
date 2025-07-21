//! PIR Construction Validation - Semantic Preservation Hooks
//!
//! This module provides validation hooks that ensure semantic preservation
//! and business context continuity throughout PIR construction.

use crate::{PIRResult, PIRError, semantic::*};
use prism_common::{NodeId, span::Span};
use std::collections::HashMap;
use async_trait::async_trait;

/// Trait for construction validation hooks
#[async_trait]
pub trait ValidationHook: Send + Sync {
    /// Hook name for identification
    fn name(&self) -> &'static str;
    
    /// Hook description
    fn description(&self) -> &'static str;
    
    /// Validate during construction
    async fn validate(&self, context: &ConstructionConstructionValidationContext) -> PIRResult<ConstructionConstructionValidationResult>;
    
    /// Check if this hook should run for the given context
    fn should_run(&self, context: &ConstructionConstructionValidationContext) -> bool {
        true
    }
    
    /// Get validation priority (higher = run first)
    fn priority(&self) -> u32 {
        100
    }
}

/// Context for construction validation hooks
#[derive(Debug, Clone)]
pub struct ConstructionConstructionValidationContext {
    /// PIR being validated
    pub pir: PrismIR,
    /// Original AST for comparison
    pub original_ast: Option<prism_ast::Program>,
    /// Semantic analysis results
    pub semantic_results: HashMap<NodeId, SemanticValidationInfo>,
    /// Business context mapping
    pub business_contexts: HashMap<String, crate::business::BusinessContext>,
    /// Construction phase
    pub current_phase: String,
    /// Validation configuration
    pub config: ConstructionValidationConfig,
}

/// Result of construction validation (distinct from main PIR validation)
#[derive(Debug, Clone)]
pub struct ConstructionConstructionValidationResult {
    /// Overall validation success
    pub success: bool,
    /// Validation score (0.0 to 1.0)
    pub score: f64,
    /// Specific validation findings
    pub findings: Vec<ConstructionConstructionValidationFinding>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Individual construction validation finding
#[derive(Debug, Clone)]
pub struct ConstructionConstructionValidationFinding {
    /// Finding severity
    pub severity: ConstructionConstructionValidationSeverity,
    /// Finding category
    pub category: ConstructionConstructionValidationCategory,
    /// Finding message
    pub message: String,
    /// Location in PIR
    pub location: Option<PIRLocation>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Construction validation severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructionConstructionValidationSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// Construction validation categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructionConstructionValidationCategory {
    SemanticPreservation,
    BusinessContextContinuity,
    EffectSystemConsistency,
    PerformanceCharacteristics,
    CohesionMetrics,
    AIMetadataQuality,
}

/// Location within PIR for validation findings
#[derive(Debug, Clone)]
pub struct PIRLocation {
    /// Module name
    pub module: String,
    /// Section within module
    pub section: Option<String>,
    /// Item within section
    pub item: Option<String>,
}

/// Semantic validation information
#[derive(Debug, Clone)]
pub struct SemanticValidationInfo {
    /// Original semantic type
    pub original_type: Option<String>,
    /// PIR semantic type
    pub pir_type: Option<String>,
    /// Preservation score
    pub preservation_score: f64,
}

/// Configuration for construction validation
#[derive(Debug, Clone)]
pub struct ConstructionValidationConfig {
    /// Enable semantic preservation validation
    pub enable_semantic_preservation: bool,
    /// Enable business context validation
    pub enable_business_context: bool,
    /// Enable effect system validation
    pub enable_effect_system: bool,
    /// Minimum acceptable scores
    pub min_semantic_score: f64,
    /// Minimum acceptable business context score
    pub min_business_context_score: f64,
    /// Enable strict validation mode
    pub strict_mode: bool,
}

impl Default for ConstructionValidationConfig {
    fn default() -> Self {
        Self {
            enable_semantic_preservation: true,
            enable_business_context: true,
            enable_effect_system: true,
            min_semantic_score: 0.8,
            min_business_context_score: 0.7,
            strict_mode: false,
        }
    }
}

/// Construction validator that orchestrates validation hooks
pub struct ConstructionValidator {
    /// Registered validation hooks
    hooks: Vec<Box<dyn ValidationHook>>,
    /// Validation configuration
    config: ConstructionValidationConfig,
}

impl ConstructionValidator {
    /// Create a new construction validator
    pub fn new(config: ConstructionValidationConfig) -> Self {
        let mut validator = Self {
            hooks: Vec::new(),
            config,
        };
        
        // Register default validation hooks
        validator.register_default_hooks();
        validator
    }
    
    /// Register a validation hook
    pub fn register_hook(&mut self, hook: Box<dyn ValidationHook>) {
        self.hooks.push(hook);
        // Sort hooks by priority (highest first)
        self.hooks.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }
    
    /// Validate PIR construction
    pub async fn validate(&self, context: ConstructionConstructionValidationContext) -> PIRResult<ConstructionConstructionValidationResult> {
        let mut all_findings = Vec::new();
        let mut scores = Vec::new();
        let mut recommendations = Vec::new();
        
        // Run all applicable hooks
        for hook in &self.hooks {
            if hook.should_run(&context) {
                match hook.validate(&context).await {
                    Ok(result) => {
                        scores.push(result.score);
                        all_findings.extend(result.findings);
                        recommendations.extend(result.recommendations);
                    }
                    Err(e) => {
                        all_findings.push(ConstructionConstructionValidationFinding {
                            severity: ConstructionConstructionValidationSeverity::Error,
                            category: ConstructionConstructionValidationCategory::SemanticPreservation,
                            message: format!("Validation hook '{}' failed: {}", hook.name(), e),
                            location: None,
                            suggested_fix: None,
                        });
                    }
                }
            }
        }
        
        // Calculate overall score
        let overall_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        
        // Determine success based on severity of findings and scores
        let has_errors = all_findings.iter().any(|f| f.severity == ConstructionValidationSeverity::Error);
        let meets_score_requirements = overall_score >= self.config.min_semantic_score;
        
        let success = !has_errors && meets_score_requirements;
        
        Ok(ConstructionConstructionValidationResult {
            success,
            score: overall_score,
            findings: all_findings,
            recommendations,
        })
    }
    
    /// Register default validation hooks
    fn register_default_hooks(&mut self) {
        self.register_hook(Box::new(SemanticPreservationHook::new()));
        self.register_hook(Box::new(BusinessContextHook::new()));
        self.register_hook(Box::new(EffectSystemHook::new()));
        self.register_hook(Box::new(CohesionMetricsHook::new()));
        self.register_hook(Box::new(AIMetadataHook::new()));
    }
}

/// Semantic preservation validation hook
pub struct SemanticPreservationHook;

impl SemanticPreservationHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationHook for SemanticPreservationHook {
    fn name(&self) -> &'static str {
        "semantic_preservation"
    }
    
    fn description(&self) -> &'static str {
        "Validates that semantic information is preserved during PIR construction"
    }
    
    async fn validate(&self, context: &ConstructionConstructionValidationContext) -> PIRResult<ConstructionConstructionValidationResult> {
        let mut findings = Vec::new();
        let mut recommendations = Vec::new();
        let mut preservation_scores = Vec::new();
        
        // Validate each module's semantic preservation
        for module in &context.pir.modules {
            let module_score = self.validate_module_preservation(module, context)?;
            preservation_scores.push(module_score);
            
            if module_score < 0.8 {
                findings.push(ConstructionConstructionValidationFinding {
                    severity: if module_score < 0.5 { ConstructionConstructionValidationSeverity::Error } else { ConstructionConstructionValidationSeverity::Warning },
                    category: ConstructionConstructionValidationCategory::SemanticPreservation,
                    message: format!("Module '{}' has low semantic preservation score: {:.2}", module.name, module_score),
                    location: Some(PIRLocation {
                        module: module.name.clone(),
                        section: None,
                        item: None,
                    }),
                    suggested_fix: Some("Review type mappings and ensure all semantic information is captured".to_string()),
                });
                
                recommendations.push(format!("Improve semantic preservation in module '{}'", module.name));
            }
        }
        
        let overall_score = if preservation_scores.is_empty() {
            0.0
        } else {
            preservation_scores.iter().sum::<f64>() / preservation_scores.len() as f64
        };
        
        Ok(ConstructionConstructionValidationResult {
            success: overall_score >= 0.8,
            score: overall_score,
            findings,
            recommendations,
        })
    }
    
    fn should_run(&self, context: &ConstructionConstructionValidationContext) -> bool {
        context.config.enable_semantic_preservation
    }
    
    fn priority(&self) -> u32 {
        200 // High priority
    }
}

impl SemanticPreservationHook {
    fn validate_module_preservation(&self, module: &PIRModule, _context: &ConstructionConstructionValidationContext) -> PIRResult<f64> {
        // Simplified preservation scoring
        let mut score = 1.0;
        
        // Penalize empty modules
        if module.sections.is_empty() {
            score -= 0.3;
        }
        
        // Check for semantic type information
        let has_types = module.sections.iter().any(|section| matches!(section, PIRSection::Types(_)));
        if !has_types {
            score -= 0.2;
        }
        
        // Use cohesion score as a proxy for semantic preservation
        score = score.min(module.cohesion_score);
        
        Ok(score.max(0.0))
    }
}

/// Business context validation hook
pub struct BusinessContextHook;

impl BusinessContextHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationHook for BusinessContextHook {
    fn name(&self) -> &'static str {
        "business_context"
    }
    
    fn description(&self) -> &'static str {
        "Validates that business context is preserved and enhanced during PIR construction"
    }
    
    async fn validate(&self, context: &ConstructionValidationContext) -> PIRResult<ConstructionValidationResult> {
        let mut findings = Vec::new();
        let mut scores = Vec::new();
        
        // Validate business context for each module
        for module in &context.pir.modules {
            let context_score = self.evaluate_business_context(&module.business_context);
            scores.push(context_score);
            
            if context_score < 0.7 {
                findings.push(ConstructionValidationFinding {
                    severity: ConstructionValidationSeverity::Warning,
                    category: ConstructionValidationCategory::BusinessContextContinuity,
                    message: format!("Module '{}' has weak business context: {:.2}", module.name, context_score),
                    location: Some(PIRLocation {
                        module: module.name.clone(),
                        section: None,
                        item: None,
                    }),
                    suggested_fix: Some("Add more business context annotations and domain rules".to_string()),
                });
            }
        }
        
        let overall_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        
        Ok(ConstructionValidationResult {
            success: overall_score >= context.config.min_business_context_score,
            score: overall_score,
            findings,
            recommendations: Vec::new(),
        })
    }
    
    fn should_run(&self, context: &ConstructionValidationContext) -> bool {
        context.config.enable_business_context
    }
    
    fn priority(&self) -> u32 {
        150
    }
}

impl BusinessContextHook {
    fn evaluate_business_context(&self, _business_context: &crate::business::BusinessContext) -> f64 {
        // Simplified business context scoring
        0.8 // Default score for now
    }
}

/// Effect system validation hook
pub struct EffectSystemHook;

impl EffectSystemHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationHook for EffectSystemHook {
    fn name(&self) -> &'static str {
        "effect_system"
    }
    
    fn description(&self) -> &'static str {
        "Validates effect system consistency and capability requirements"
    }
    
    async fn validate(&self, context: &ConstructionValidationContext) -> PIRResult<ConstructionValidationResult> {
        let mut findings = Vec::new();
        
        // Validate effect consistency across modules
        for module in &context.pir.modules {
            if module.effects.is_empty() && !module.capabilities.is_empty() {
                findings.push(ConstructionValidationFinding {
                    severity: ConstructionValidationSeverity::Warning,
                    category: ConstructionValidationCategory::EffectSystemConsistency,
                    message: format!("Module '{}' has capabilities but no effects", module.name),
                    location: Some(PIRLocation {
                        module: module.name.clone(),
                        section: None,
                        item: None,
                    }),
                    suggested_fix: Some("Add effect annotations for capability requirements".to_string()),
                });
            }
        }
        
        Ok(ConstructionValidationResult {
            success: findings.iter().all(|f| f.severity != ConstructionValidationSeverity::Error),
            score: 0.9, // Default good score
            findings,
            recommendations: Vec::new(),
        })
    }
    
    fn should_run(&self, context: &ConstructionValidationContext) -> bool {
        context.config.enable_effect_system
    }
    
    fn priority(&self) -> u32 {
        120
    }
}

/// Cohesion metrics validation hook
pub struct CohesionMetricsHook;

impl CohesionMetricsHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationHook for CohesionMetricsHook {
    fn name(&self) -> &'static str {
        "cohesion_metrics"
    }
    
    fn description(&self) -> &'static str {
        "Validates module cohesion and coupling metrics"
    }
    
    async fn validate(&self, context: &ConstructionValidationContext) -> PIRResult<ConstructionValidationResult> {
        let mut findings = Vec::new();
        let mut scores = Vec::new();
        
        for module in &context.pir.modules {
            scores.push(module.cohesion_score);
            
            if module.cohesion_score < 0.5 {
                findings.push(ConstructionValidationFinding {
                    severity: ConstructionValidationSeverity::Warning,
                    category: ConstructionValidationCategory::CohesionMetrics,
                    message: format!("Module '{}' has low cohesion: {:.2}", module.name, module.cohesion_score),
                    location: Some(PIRLocation {
                        module: module.name.clone(),
                        section: None,
                        item: None,
                    }),
                    suggested_fix: Some("Consider refactoring to improve conceptual cohesion".to_string()),
                });
            }
        }
        
        let overall_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        
        Ok(ConstructionValidationResult {
            success: overall_score >= 0.6,
            score: overall_score,
            findings,
            recommendations: Vec::new(),
        })
    }
    
    fn priority(&self) -> u32 {
        100
    }
}

/// AI metadata quality validation hook
pub struct AIMetadataHook;

impl AIMetadataHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationHook for AIMetadataHook {
    fn name(&self) -> &'static str {
        "ai_metadata"
    }
    
    fn description(&self) -> &'static str {
        "Validates quality and completeness of AI metadata"
    }
    
    async fn validate(&self, context: &ConstructionValidationContext) -> PIRResult<ConstructionValidationResult> {
        let mut findings = Vec::new();
        
        // Check if AI metadata exists
        if context.pir.ai_metadata.is_none() {
            findings.push(ConstructionValidationFinding {
                severity: ConstructionValidationSeverity::Info,
                category: ConstructionValidationCategory::AIMetadataQuality,
                message: "No AI metadata generated".to_string(),
                location: None,
                suggested_fix: Some("Enable AI metadata generation in builder config".to_string()),
            });
        }
        
        Ok(ConstructionValidationResult {
            success: true, // AI metadata is optional
            score: if context.pir.ai_metadata.is_some() { 1.0 } else { 0.5 },
            findings,
            recommendations: Vec::new(),
        })
    }
    
    fn priority(&self) -> u32 {
        50 // Lower priority
    }
} 