//! Effect Composition
//!
//! Implementation of effect composition operators: parallel (|), sequential (;), conditional

use super::definition::Effect;
use crate::effects::EffectError;
use prism_ast::{AstNode, Expr};
use std::collections::HashMap;

/// Effect composition system
#[derive(Debug)]
pub struct EffectComposition {
    /// Composition strategies
    pub strategies: Vec<CompositionStrategy>,
    /// Optimization rules
    pub optimization_rules: Vec<OptimizationRule>,
}

impl EffectComposition {
    /// Create new composition system
    pub fn new() -> Self {
        let mut system = Self {
            strategies: Vec::new(),
            optimization_rules: Vec::new(),
        };
        system.initialize_default_strategies();
        system
    }

    /// Compose effects using a specific operator
    pub fn compose(&self, effects: Vec<Effect>, operator: CompositionOperator) -> Result<Effect, EffectError> {
        if effects.is_empty() {
            return Err(EffectError::CompositionFailed("Cannot compose empty effect list".to_string()));
        }

        if effects.len() == 1 {
            return Ok(effects.into_iter().next().unwrap());
        }

        // Create a composed effect based on the operator
        let composed_name = match operator {
            CompositionOperator::Parallel => format!("Parallel({})", 
                effects.iter().map(|e| e.definition.as_str()).collect::<Vec<_>>().join(", ")),
            CompositionOperator::Sequential => format!("Sequential({})", 
                effects.iter().map(|e| e.definition.as_str()).collect::<Vec<_>>().join(", ")),
            CompositionOperator::Conditional(ref condition) => format!("Conditional({}, {})", 
                condition, effects.iter().map(|e| e.definition.as_str()).collect::<Vec<_>>().join(", ")),
        };

        // Use the first effect's span as the composed effect's span
        let span = effects[0].span;
        
        Ok(Effect::new(composed_name, span))
    }

    /// Initialize default composition strategies
    fn initialize_default_strategies(&mut self) {
        // Add default strategies
        self.strategies.push(CompositionStrategy {
            name: "ParallelComposition".to_string(),
            description: "Compose effects that can run in parallel".to_string(),
            operator: CompositionOperator::Parallel,
            applies_fn: Box::new(|_effects| {
                // Simple heuristic: effects can be parallel if they don't conflict
                true // Simplified for now
            }),
            apply_fn: Box::new(|effects| {
                // Simple parallel composition
                Ok(effects)
            }),
        });

        self.strategies.push(CompositionStrategy {
            name: "SequentialComposition".to_string(),
            description: "Compose effects that must run sequentially".to_string(),
            operator: CompositionOperator::Sequential,
            applies_fn: Box::new(|_effects| {
                // All effects can be sequential
                true
            }),
            apply_fn: Box::new(|effects| {
                // Simple sequential composition
                Ok(effects)
            }),
        });
    }
}

impl Default for EffectComposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Composition operators
#[derive(Debug, Clone)]
pub enum CompositionOperator {
    /// Parallel composition (|)
    Parallel,
    /// Sequential composition (;)
    Sequential,
    /// Conditional composition
    Conditional(String),
}

/// A composition strategy
pub struct CompositionStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Composition operator
    pub operator: CompositionOperator,
    /// Function to check if strategy applies
    pub applies_fn: Box<dyn Fn(&[Effect]) -> bool + Send + Sync>,
    /// Function to apply the strategy
    pub apply_fn: Box<dyn Fn(Vec<Effect>) -> Result<Vec<Effect>, EffectError> + Send + Sync>,
}

impl std::fmt::Debug for CompositionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositionStrategy")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("operator", &self.operator)
            .field("applies_fn", &"<function>")
            .field("apply_fn", &"<function>")
            .finish()
    }
}

/// An optimization rule for effect composition
pub struct OptimizationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Function to check if optimization can be applied
    pub can_optimize_fn: Box<dyn Fn(&[Effect]) -> bool + Send + Sync>,
    /// Function to apply the optimization
    pub optimize_fn: Box<dyn Fn(Vec<Effect>) -> Result<Vec<Effect>, EffectError> + Send + Sync>,
}

impl std::fmt::Debug for OptimizationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("can_optimize_fn", &"<function>")
            .field("optimize_fn", &"<function>")
            .finish()
    }
}

/// Rule for composing effects
#[derive(Debug, Clone)]
pub struct EffectCompositionRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Input effects that trigger this rule
    pub input_effects: Vec<String>,
    /// Output effect produced by composition
    pub output_effect: String,
    /// Conditions under which this rule applies
    pub conditions: Vec<AstNode<Expr>>,
    /// AI-readable explanation of the composition
    pub ai_explanation: Option<String>,
}

impl EffectCompositionRule {
    /// Create a new composition rule
    pub fn new(
        name: String,
        description: String,
        input_effects: Vec<String>,
        output_effect: String,
    ) -> Self {
        Self {
            name,
            description,
            input_effects,
            output_effect,
            conditions: Vec::new(),
            ai_explanation: None,
        }
    }

    /// Add a condition to this rule
    pub fn with_condition(mut self, condition: AstNode<Expr>) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Add AI explanation for this rule
    pub fn with_ai_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.ai_explanation = Some(explanation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::span::Span;

    #[test]
    fn test_composition_system_creation() {
        let system = EffectComposition::new();
        assert!(!system.strategies.is_empty());
        assert!(!system.optimization_rules.is_empty());
    }

    #[test]
    fn test_parallel_composition() {
        let operator = CompositionOperator::Parallel;

        let effects = vec![
            Effect::new("IO.FileSystem.Read".to_string(), Span::default()),
            Effect::new("IO.Network.Connect".to_string(), Span::default()),
        ];

        let composed_effect = system.compose(effects, operator).unwrap();
        assert_eq!(composed_effect.definition, "Parallel(IO.FileSystem.Read, IO.Network.Connect)");
    }

    #[test]
    fn test_remove_duplicates() {
        let effects = vec![
            Effect::new("IO.FileSystem.Read".to_string(), Span::default()),
            Effect::new("IO.FileSystem.Read".to_string(), Span::default()),
            Effect::new("IO.Network.Connect".to_string(), Span::default()),
        ];

        let operator = CompositionOperator::Parallel;
        let composed_effect = system.compose(effects, operator).unwrap();
        assert_eq!(composed_effect.definition, "Parallel(IO.FileSystem.Read, IO.Network.Connect)");
    }
} 