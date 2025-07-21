//! Effect Registry
//!
//! Registry and hierarchical organization of effects

use super::definition::{Effect, EffectDefinition, EffectCategory};
use crate::EffectSystemError;
use std::collections::{HashMap, HashSet};

/// Registry of all known effects in the system
#[derive(Debug, Default)]
pub struct EffectRegistry {
    /// Map of effect names to their definitions
    pub effects: HashMap<String, EffectDefinition>,
    /// Hierarchical organization of effects
    pub hierarchy: EffectHierarchy,
    /// Effect composition rules
    pub composition_rules: Vec<EffectCompositionRule>,
}

impl EffectRegistry {
    /// Create a new effect registry with built-in effects
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_effects();
        registry
    }

    /// Register a new effect definition
    pub fn register(&mut self, effect: EffectDefinition) -> Result<(), EffectSystemError> {
        if self.effects.contains_key(&effect.name) {
            return Err(EffectSystemError::EffectAlreadyRegistered {
                name: effect.name.clone()
            });
        }
        
        self.hierarchy.add_effect(&effect);
        self.effects.insert(effect.name.clone(), effect);
        Ok(())
    }

    /// Get an effect definition by name
    pub fn get_effect(&self, name: &str) -> Option<&EffectDefinition> {
        self.effects.get(name)
    }

    /// Check if an effect is a subtype of another
    pub fn is_subeffect(&self, child: &str, parent: &str) -> bool {
        self.hierarchy.is_subeffect(child, parent)
    }

    /// Register all built-in effects from PLD-003
    fn register_builtin_effects(&mut self) {
        // IO Effects
        let io_effects = vec![
            EffectDefinition::new(
                "IO.FileSystem.Read".to_string(),
                "Read files from the file system".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Provides read access to file system resources")
             .with_security_implication("Can access sensitive file contents")
             .with_capability_requirement("FileSystem", vec!["Read".to_string()]),

            EffectDefinition::new(
                "IO.FileSystem.Write".to_string(),
                "Write files to the file system".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Provides write access to file system resources")
             .with_security_implication("Can modify or create files")
             .with_capability_requirement("FileSystem", vec!["Write".to_string()]),

            EffectDefinition::new(
                "IO.Network.Connect".to_string(),
                "Establish network connections".to_string(),
                EffectCategory::Network,
            ).with_ai_context("Enables network communication")
             .with_security_implication("Can send data over network")
             .with_capability_requirement("Network", vec!["Connect".to_string()]),
        ];

        // Database Effects
        let database_effects = vec![
            EffectDefinition::new(
                "Database.Query".to_string(),
                "Execute database queries".to_string(),
                EffectCategory::Database,
            ).with_ai_context("Provides database query capabilities")
             .with_security_implication("Can access database contents")
             .with_capability_requirement("Database", vec!["Query".to_string()]),

            EffectDefinition::new(
                "Database.Transaction".to_string(),
                "Manage database transactions".to_string(),
                EffectCategory::Database,
            ).with_ai_context("Provides transactional database operations")
             .with_security_implication("Can modify database state")
             .with_capability_requirement("Database", vec!["Transaction".to_string()]),
        ];

        // Cryptography Effects
        let crypto_effects = vec![
            EffectDefinition::new(
                "Cryptography.KeyGeneration".to_string(),
                "Generate cryptographic keys".to_string(),
                EffectCategory::Security,
            ).with_ai_context("Generates cryptographic material")
             .with_security_implication("Creates sensitive cryptographic keys")
             .with_capability_requirement("Cryptography", vec!["KeyGeneration".to_string()]),

            EffectDefinition::new(
                "Cryptography.Encryption".to_string(),
                "Encrypt data using cryptographic algorithms".to_string(),
                EffectCategory::Security,
            ).with_ai_context("Provides data encryption capabilities")
             .with_security_implication("Processes potentially sensitive data")
             .with_capability_requirement("Cryptography", vec!["Encryption".to_string()]),
        ];

        // External AI Integration Effects (for server-based AI tools)
        let ai_integration_effects = vec![
            EffectDefinition::new(
                "ExternalAI.DataExport".to_string(),
                "Export data for external AI analysis".to_string(),
                EffectCategory::IO,
            ).with_ai_context("Prepares and exports data for external AI systems to analyze")
             .with_security_implication("May expose sensitive data to external AI services")
             .with_capability_requirement("Network", vec!["Send".to_string()])
             .with_business_rule("Data must be sanitized before export"),

            EffectDefinition::new(
                "ExternalAI.MetadataGeneration".to_string(),
                "Generate AI-comprehensible metadata".to_string(),
                EffectCategory::Pure,
            ).with_ai_context("Creates structured metadata for external AI systems to understand code")
             .with_security_implication("Metadata may reveal code structure and business logic")
             .with_capability_requirement("FileSystem", vec!["Write".to_string()])
             .with_business_rule("Generated metadata must not include sensitive implementation details"),
        ];

        // Register all effects
        for effect in io_effects.into_iter()
            .chain(database_effects)
            .chain(crypto_effects)
            .chain(ai_integration_effects)
        {
            let _ = self.register(effect);
        }
    }
}

/// Hierarchical organization of effects
#[derive(Debug, Default)]
pub struct EffectHierarchy {
    /// Parent-child relationships between effects
    pub relationships: HashMap<String, Vec<String>>,
    /// Root effects (no parents)
    pub roots: HashSet<String>,
}

impl EffectHierarchy {
    /// Add an effect to the hierarchy
    pub fn add_effect(&mut self, effect: &EffectDefinition) {
        if let Some(parent) = &effect.parent_effect {
            self.relationships
                .entry(parent.clone())
                .or_default()
                .push(effect.name.clone());
        } else {
            self.roots.insert(effect.name.clone());
        }
    }

    /// Check if one effect is a subeffect of another
    pub fn is_subeffect(&self, child: &str, parent: &str) -> bool {
        if child == parent {
            return true;
        }

        if let Some(children) = self.relationships.get(parent) {
            for child_effect in children {
                if child_effect == child || self.is_subeffect(child, child_effect) {
                    return true;
                }
            }
        }

        false
    }
}

/// Rule for composing multiple effects (placeholder for now)
#[derive(Debug, Clone)]
pub struct EffectCompositionRule {
    /// Rule name for identification
    pub name: String,
    /// Description of what this rule does
    pub description: String,
    /// Input effects that trigger this rule
    pub input_effects: Vec<String>,
    /// Output effect produced by composition
    pub output_effect: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_registry_creation() {
        let registry = EffectRegistry::new();
        assert!(!registry.effects.is_empty(), "Registry should have built-in effects");
        assert!(registry.effects.contains_key("IO.FileSystem.Read"));
        assert!(registry.effects.contains_key("Database.Query"));
    }

    #[test]
    fn test_effect_hierarchy() {
        let mut hierarchy = EffectHierarchy::default();
        let parent_effect = EffectDefinition::new(
            "Parent".to_string(),
            "Parent effect".to_string(),
            EffectCategory::IO,
        );
        let mut child_effect = EffectDefinition::new(
            "Child".to_string(),
            "Child effect".to_string(),
            EffectCategory::IO,
        );
        child_effect.parent_effect = Some("Parent".to_string());

        hierarchy.add_effect(&parent_effect);
        hierarchy.add_effect(&child_effect);

        assert!(hierarchy.is_subeffect("Child", "Parent"));
        assert!(hierarchy.is_subeffect("Parent", "Parent")); // Self-relationship
        assert!(!hierarchy.is_subeffect("Parent", "Child"));
    }
} 