//! Effects Module - Computational Effect Management
//!
//! This module provides a comprehensive system for managing computational effects in Prism,
//! everything related to defining, discovering, composing, and managing computational effects.

pub mod definition;
pub mod registry;
pub mod lifecycle;
pub mod composition;
pub mod inference;
pub mod monitoring; // Add monitoring module

use prism_ast::{AstNode, Type};
use crate::EffectSystemError;

// Re-exports for convenience
pub use definition::{Effect, EffectDefinition, EffectCategory, EffectParameter, EffectHierarchy, EffectRegistry};
pub use registry::EffectCompositionRule;
pub use lifecycle::{EffectLifecycleResult, EffectLifecycleMetadata};
pub use composition::{EffectComposition, CompositionOperator};
pub use inference::{EffectInference, InferenceConfig};
pub use monitoring::{EffectMonitor, MonitoringMetrics, PerformanceAnalytics}; // Add monitoring exports

/// Complete effect management system
#[derive(Debug)]
pub struct EffectSystem {
    /// Effect registry (using the definition module's registry)
    pub registry: definition::EffectRegistry,
    /// Effect lifecycle manager
    pub lifecycle: lifecycle::EffectLifecycle,
    /// Effect composition system
    pub composition: composition::EffectComposition,
    /// Effect inference system
    pub inference: inference::EffectInference,
    /// Real-time effect monitoring system
    pub monitor: monitoring::EffectMonitor,
}

impl EffectSystem {
    /// Create new effect system
    pub fn new() -> Self {
        Self {
            registry: definition::EffectRegistry::new(),
            lifecycle: lifecycle::EffectLifecycle::new(),
            composition: composition::EffectComposition::new(),
            inference: inference::EffectInference::new(),
            monitor: monitoring::EffectMonitor::new(),
        }
    }

    /// Register a new effect
    pub fn register_effect(&mut self, effect: definition::EffectDefinition) -> Result<(), EffectSystemError> {
        self.registry.register(effect)
    }

    /// Get effect count
    pub fn get_effect_count(&self) -> usize {
        self.registry.effects.len()
    }

    /// Discover effects in code using the inference engine
    pub fn discover_effects(&mut self, code_unit: &AstNode<Type>) -> Result<Vec<definition::Effect>, EffectSystemError> {
        // Use the inference engine to discover effects
        self.inference.engine.infer_effects(code_unit, &self.registry)
    }

    /// Compose effects using the composition system
    pub fn compose_effects(&self, effects: Vec<definition::Effect>, operator: composition::CompositionOperator) -> Result<definition::Effect, EffectSystemError> {
        if effects.is_empty() {
            return Err(EffectSystemError::EffectValidationFailed {
                reason: "No effects to compose".to_string()
            });
        }

        // Use the composition system to compose effects
        self.composition.compose(effects, operator)
    }

    /// Advanced composition using composition analyzer
    pub fn compose_effects_advanced(&mut self, effects: Vec<definition::Effect>) -> Result<Vec<definition::Effect>, EffectSystemError> {
        // Use the inference engine's composition analyzer for advanced composition
        self.inference.engine.composition_analyzer.compose_effects(&effects, &self.registry)
    }

    /// Discover and compose effects in one operation
    pub fn discover_and_compose(&mut self, code_unit: &AstNode<Type>) -> Result<Vec<definition::Effect>, EffectSystemError> {
        // First discover effects
        let discovered_effects = self.discover_effects(code_unit)?;
        
        // Then compose them using advanced composition
        if discovered_effects.is_empty() {
            return Ok(vec![]);
        }
        
        self.compose_effects_advanced(discovered_effects)
    }

    /// Use the full lifecycle system for complete effect processing
    pub fn process_complete_lifecycle(
        &mut self,
        code_unit: &AstNode<Type>,
        security_context: &crate::security_trust::SecureExecutionContext,
    ) -> Result<lifecycle::EffectLifecycleResult, EffectSystemError> {
        // Create a lifecycle system and use it
        let mut lifecycle_system = lifecycle::EffectLifecycleSystem::new();
        
        lifecycle_system.process_effect_lifecycle(code_unit, security_context)
            .map_err(|e| EffectSystemError::EffectValidationFailed {
                reason: format!("Lifecycle processing failed: {}", e)
            })
    }

    /// Get comprehensive system metrics including monitoring data
    pub fn get_comprehensive_metrics(&self) -> ComprehensiveMetrics {
        ComprehensiveMetrics {
            effect_count: self.registry.effects.len(),
            monitoring_metrics: self.monitor.get_current_metrics(),
            performance_analytics: self.monitor.get_performance_analytics(),
        }
    }
}

impl Default for EffectSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive metrics combining all effect system data
#[derive(Debug)]
pub struct ComprehensiveMetrics {
    /// Number of registered effects
    pub effect_count: usize,
    /// Real-time monitoring metrics
    pub monitoring_metrics: monitoring::MonitoringMetrics,
    /// Performance analytics
    pub performance_analytics: monitoring::PerformanceAnalytics,
}

// Re-export EffectSystemError as EffectError for backward compatibility
pub use crate::EffectSystemError as EffectError; 