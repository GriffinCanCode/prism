//! Effects Module - Computational Effect Management
//!
//! This module provides a comprehensive system for managing computational effects in Prism,
//! everything related to defining, discovering, composing, and managing computational effects.

pub mod definition;
pub mod registry;
pub mod lifecycle;
pub mod composition;
pub mod inference;

use crate::effects::definition::{Effect, EffectDefinition, EffectCategory};
use crate::effects::registry::EffectRegistry;
use crate::effects::inference::EffectInference;
use prism_ast::{AstNode, Type};
use prism_common::span::Span;
use std::collections::HashMap;
use thiserror::Error;

// Re-exports for convenience
pub use definition::{Effect, EffectDefinition, EffectCategory, EffectParameter, EffectRegistry, EffectHierarchy};
pub use registry::{EffectRegistry as EffectRegistryManager};
pub use lifecycle::{EffectLifecycleResult, EffectLifecycleMetadata};
pub use composition::{EffectComposition, EffectCompositionRule, CompositionOperator};
pub use inference::{InferenceConfig};

/// Complete effect management system
#[derive(Debug)]
pub struct EffectSystem {
    /// Effect registry
    pub registry: EffectRegistry,
    /// Effect lifecycle manager
    pub lifecycle: lifecycle::EffectLifecycle,
    /// Effect composition system
    pub composition: EffectComposition,
    /// Effect inference system
    pub inference: EffectInference,
}

impl EffectSystem {
    /// Create new effect system
    pub fn new() -> Self {
        Self {
            registry: EffectRegistry::new(),
            lifecycle: lifecycle::EffectLifecycle::new(),
            composition: EffectComposition::new(),
            inference: inference::EffectInference::new(),
        }
    }

    /// Register a new effect
    pub fn register_effect(&mut self, effect: EffectDefinition) -> Result<(), EffectError> {
        self.registry.register(effect)
            .map_err(|e| EffectError::RegistrationFailed(e.to_string()))
    }

    /// Get effect count
    pub fn get_effect_count(&self) -> usize {
        self.registry.effects.len()
    }

    /// Discover effects in code
    pub fn discover_effects(&mut self, code_unit: &AstNode<Type>) -> Result<Vec<Effect>, EffectError> {
        self.inference.discover_effects(code_unit, &self.registry)
            .map(|result| result.discovered_effects)
    }

    /// Compose effects
    pub fn compose_effects(&self, effects: Vec<Effect>, operator: CompositionOperator) -> Result<Effect, EffectError> {
        self.composition.compose(effects, operator)
    }
}

impl Default for EffectSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors in effect management
#[derive(Debug, Error)]
pub enum EffectError {
    #[error("Effect registration failed: {0}")]
    RegistrationFailed(String),
    
    #[error("Effect not found: {0}")]
    NotFound(String),
    
    #[error("Effect inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Effect composition failed: {0}")]
    CompositionFailed(String),
} 