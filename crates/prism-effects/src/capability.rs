//! Capability-based security model
//!
//! This module implements object capabilities as described in PLD-003,
//! providing unforgeable tokens that grant specific authorities with
//! support for attenuation, composition, and revocation.

use prism_ast::{AstNode, Expr, SecurityClassification};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// Manager for all capabilities in the system
#[derive(Debug, Default)]
pub struct CapabilityManager {
    /// Registered capability definitions
    pub capabilities: HashMap<String, CapabilityDefinition>,
    /// Active capability instances
    pub active_capabilities: HashMap<String, Arc<dyn CapabilityInstance>>,
    /// Capability delegation relationships
    pub delegations: HashMap<String, Vec<String>>,
    /// Revoked capabilities
    pub revoked: HashSet<String>,
}

impl CapabilityManager {
    /// Create a new capability manager
    pub fn new() -> Self {
        let mut manager = Self::default();
        manager.register_builtin_capabilities();
        manager
    }

    /// Register a new capability definition
    pub fn register_capability(&mut self, capability: CapabilityDefinition) -> Result<(), crate::EffectSystemError> {
        if self.capabilities.contains_key(&capability.name) {
            return Err(crate::EffectSystemError::EffectAlreadyRegistered {
                name: capability.name.clone(),
            });
        }

        self.capabilities.insert(capability.name.clone(), capability);
        Ok(())
    }

    /// Get a capability definition by name
    pub fn get_capability(&self, name: &str) -> Option<&CapabilityDefinition> {
        self.capabilities.get(name)
    }

    /// Check if a capability is available and not revoked
    pub fn is_capability_available(&self, name: &str) -> bool {
        !self.revoked.contains(name) && self.active_capabilities.contains_key(name)
    }

    /// Revoke a capability
    pub fn revoke_capability(&mut self, name: &str) -> Result<(), crate::EffectSystemError> {
        if !self.active_capabilities.contains_key(name) {
            return Err(crate::EffectSystemError::CapabilityNotFound {
                name: name.to_string(),
            });
        }

        self.revoked.insert(name.to_string());
        self.active_capabilities.remove(name);
        Ok(())
    }

    /// Attenuate a capability to create a more restricted version
    pub fn attenuate_capability(
        &self,
        capability_name: &str,
        constraints: CapabilityConstraints,
    ) -> Result<AttenuatedCapability, crate::EffectSystemError> {
        let base_capability = self.get_capability(capability_name)
            .ok_or_else(|| crate::EffectSystemError::CapabilityNotFound {
                name: capability_name.to_string(),
            })?;

        Ok(AttenuatedCapability::new(
            base_capability.clone(),
            constraints,
        ))
    }

    /// Register built-in capabilities from PLD-003
    fn register_builtin_capabilities(&mut self) {
        let capabilities = vec![
            // File system capabilities
            CapabilityDefinition::new(
                "FileSystem".to_string(),
                "Access to file system operations".to_string(),
                CapabilityCategory::IO,
            )
            .with_authority("FileSystemAuthority")
            .with_constraint("allowed_paths", "Set of permitted file paths")
            .with_constraint("operations", "Set of allowed operations (Read, Write, Execute)")
            .with_constraint("max_file_size", "Maximum file size limit")
            .with_ai_context("Provides controlled access to file system resources")
            .with_security_property("Cannot access files outside allowed_paths")
            .with_security_property("Cannot perform operations not in operations set"),

            // Network capabilities
            CapabilityDefinition::new(
                "Network".to_string(),
                "Access to network operations".to_string(),
                CapabilityCategory::IO,
            )
            .with_authority("NetworkAuthority")
            .with_constraint("allowed_hosts", "Set of permitted hosts")
            .with_constraint("allowed_ports", "Set of permitted ports")
            .with_constraint("protocols", "Set of allowed protocols")
            .with_constraint("rate_limit", "Maximum requests per second")
            .with_ai_context("Provides controlled network access")
            .with_security_property("Cannot connect to hosts not in allowed_hosts")
            .with_security_property("Rate limited to prevent DoS attacks"),

            // Database capabilities
            CapabilityDefinition::new(
                "Database".to_string(),
                "Access to database operations".to_string(),
                CapabilityCategory::Database,
            )
            .with_authority("DatabaseAuthority")
            .with_constraint("allowed_tables", "Set of accessible tables")
            .with_constraint("operations", "Set of allowed operations")
            .with_constraint("max_connections", "Maximum concurrent connections")
            .with_ai_context("Provides controlled database access")
            .with_security_property("Cannot access tables not in allowed_tables")
            .with_security_property("Operations limited to specified set"),

            // Cryptography capabilities
            CapabilityDefinition::new(
                "Cryptography".to_string(),
                "Access to cryptographic operations".to_string(),
                CapabilityCategory::Security,
            )
            .with_authority("CryptographyAuthority")
            .with_constraint("algorithms", "Set of allowed algorithms")
            .with_constraint("key_sizes", "Permitted key sizes")
            .with_constraint("entropy_sources", "Allowed entropy sources")
            .with_ai_context("Provides cryptographic capabilities")
            .with_security_property("Only approved algorithms allowed")
            .with_security_property("Keys generated with sufficient entropy"),

            // AI capabilities
            CapabilityDefinition::new(
                "AI".to_string(),
                "Access to AI model operations".to_string(),
                CapabilityCategory::AI,
            )
            .with_authority("AIAuthority")
            .with_constraint("approved_models", "Set of approved model IDs")
            .with_constraint("max_tokens", "Maximum tokens per request")
            .with_constraint("content_filters", "Required content filters")
            .with_ai_context("Provides AI model inference capabilities")
            .with_security_property("Only approved models can be used")
            .with_security_property("Content filtering prevents harmful outputs"),
        ];

        for capability in capabilities {
            let _ = self.register_capability(capability);
        }
    }
}

/// Definition of a capability type
#[derive(Debug, Clone)]
pub struct CapabilityDefinition {
    /// Unique name of the capability
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Capability category
    pub category: CapabilityCategory,
    /// Authority that grants this capability
    pub authority: Option<String>,
    /// Constraints that can be applied
    pub constraints: HashMap<String, String>,
    /// AI-comprehensible context
    pub ai_context: Option<String>,
    /// Security properties this capability provides
    pub security_properties: Vec<String>,
    /// Business rules for capability usage
    pub business_rules: Vec<String>,
    /// Examples of capability usage
    pub examples: Vec<String>,
}

impl CapabilityDefinition {
    /// Create a new capability definition
    pub fn new(name: String, description: String, category: CapabilityCategory) -> Self {
        Self {
            name,
            description,
            category,
            authority: None,
            constraints: HashMap::new(),
            ai_context: None,
            security_properties: Vec::new(),
            business_rules: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Set the authority for this capability
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.authority = Some(authority.into());
        self
    }

    /// Add a constraint definition
    pub fn with_constraint(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.constraints.insert(name.into(), description.into());
        self
    }

    /// Add AI context
    pub fn with_ai_context(mut self, context: impl Into<String>) -> Self {
        self.ai_context = Some(context.into());
        self
    }

    /// Add a security property
    pub fn with_security_property(mut self, property: impl Into<String>) -> Self {
        self.security_properties.push(property.into());
        self
    }

    /// Add a business rule
    pub fn with_business_rule(mut self, rule: impl Into<String>) -> Self {
        self.business_rules.push(rule.into());
        self
    }
}

/// Categories of capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CapabilityCategory {
    /// Input/Output capabilities
    IO,
    /// Database capabilities
    Database,
    /// Network capabilities
    Network,
    /// Security capabilities
    Security,
    /// AI capabilities
    AI,
    /// System capabilities
    System,
    /// Memory capabilities
    Memory,
    /// Custom capability category
    Custom(String),
}

impl fmt::Display for CapabilityCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IO => write!(f, "IO"),
            Self::Database => write!(f, "Database"),
            Self::Network => write!(f, "Network"),
            Self::Security => write!(f, "Security"),
            Self::AI => write!(f, "AI"),
            Self::System => write!(f, "System"),
            Self::Memory => write!(f, "Memory"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Trait for capability instances
pub trait CapabilityInstance: std::fmt::Debug + Send + Sync {
    /// Get the capability name
    fn name(&self) -> &str;

    /// Check if this capability allows a specific operation
    fn allows(&self, operation: &str, parameters: &HashMap<String, String>) -> bool;

    /// Get the security classification of this capability
    fn security_classification(&self) -> SecurityClassification;

    /// Check if this capability has been revoked
    fn is_revoked(&self) -> bool;

    /// Get AI-readable information about this capability
    fn ai_info(&self) -> CapabilityAIInfo;
}

/// AI-readable information about a capability
#[derive(Debug, Clone)]
pub struct CapabilityAIInfo {
    /// Purpose of this capability
    pub purpose: String,
    /// What operations it enables
    pub enabled_operations: Vec<String>,
    /// Security guarantees it provides
    pub security_guarantees: Vec<String>,
    /// Potential risks of misuse
    pub risks: Vec<String>,
    /// Best practices for usage
    pub best_practices: Vec<String>,
}

/// Concrete capability instance
#[derive(Debug, Clone)]
pub struct Capability {
    /// The capability definition this is based on
    pub definition: String,
    /// Specific constraints for this instance
    pub constraints: CapabilityConstraints,
    /// Whether this capability is revoked
    pub revoked: bool,
    /// Metadata for this capability instance
    pub metadata: CapabilityMetadata,
}

impl Capability {
    /// Create a new capability instance
    pub fn new(definition: String, constraints: CapabilityConstraints) -> Self {
        Self {
            definition,
            constraints,
            revoked: false,
            metadata: CapabilityMetadata::default(),
        }
    }

    /// Revoke this capability
    pub fn revoke(&mut self) {
        self.revoked = true;
    }

    /// Check if this capability is active
    pub fn is_active(&self) -> bool {
        !self.revoked
    }
}

impl CapabilityInstance for Capability {
    fn name(&self) -> &str {
        &self.definition
    }

    fn allows(&self, operation: &str, parameters: &HashMap<String, String>) -> bool {
        if self.revoked {
            return false;
        }
        
        // Check if operation is allowed based on constraints
        // This is a simplified implementation - would be more sophisticated in practice
        !operation.is_empty() && !parameters.is_empty()
    }

    fn security_classification(&self) -> SecurityClassification {
        self.metadata.security_classification.clone()
    }

    fn is_revoked(&self) -> bool {
        self.revoked
    }

    fn ai_info(&self) -> CapabilityAIInfo {
        CapabilityAIInfo {
            purpose: format!("Capability for {}", self.definition),
            enabled_operations: vec![self.definition.clone()],
            security_guarantees: vec!["Capability-based access control".to_string()],
            risks: vec!["Unauthorized access if capability is compromised".to_string()],
            best_practices: vec!["Use principle of least privilege".to_string()],
        }
    }
}

/// Constraints applied to a capability
#[derive(Debug, Clone, Default)]
pub struct CapabilityConstraints {
    /// String-based constraints
    pub string_constraints: HashMap<String, String>,
    /// List-based constraints
    pub list_constraints: HashMap<String, Vec<String>>,
    /// Numeric constraints
    pub numeric_constraints: HashMap<String, f64>,
    /// Boolean constraints
    pub boolean_constraints: HashMap<String, bool>,
    /// Expression-based constraints
    pub expression_constraints: HashMap<String, AstNode<Expr>>,
}

impl CapabilityConstraints {
    /// Create new empty constraints
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a string constraint
    pub fn with_string(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.string_constraints.insert(name.into(), value.into());
        self
    }

    /// Add a list constraint
    pub fn with_list(mut self, name: impl Into<String>, values: Vec<String>) -> Self {
        self.list_constraints.insert(name.into(), values);
        self
    }

    /// Add a numeric constraint
    pub fn with_numeric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.numeric_constraints.insert(name.into(), value);
        self
    }

    /// Add a boolean constraint
    pub fn with_boolean(mut self, name: impl Into<String>, value: bool) -> Self {
        self.boolean_constraints.insert(name.into(), value);
        self
    }
}

/// Metadata for a capability instance
#[derive(Debug, Clone, Default)]
pub struct CapabilityMetadata {
    /// AI-readable context
    pub ai_context: Option<String>,
    /// Security classification
    pub security_classification: SecurityClassification,
    /// Creation timestamp
    pub created_at: Option<String>,
    /// Last used timestamp
    pub last_used_at: Option<String>,
    /// Usage count
    pub usage_count: u64,
    /// Source of this capability
    pub source: Option<String>,
}

/// Attenuated capability with reduced privileges
#[derive(Debug, Clone)]
pub struct AttenuatedCapability {
    /// Base capability this is attenuated from
    pub base_capability: CapabilityDefinition,
    /// Additional constraints applied
    pub additional_constraints: CapabilityConstraints,
    /// Attenuation metadata
    pub attenuation_metadata: AttenuationMetadata,
}

impl AttenuatedCapability {
    /// Create a new attenuated capability
    pub fn new(base_capability: CapabilityDefinition, constraints: CapabilityConstraints) -> Self {
        Self {
            base_capability,
            additional_constraints: constraints,
            attenuation_metadata: AttenuationMetadata::default(),
        }
    }

    /// Check if this attenuated capability allows an operation
    pub fn allows(&self, operation: &str, parameters: &HashMap<String, String>) -> bool {
        // Check base capability first
        // Then apply additional constraints
        // This is a simplified implementation
        true // TODO: Implement proper constraint checking
    }
}

/// Metadata about capability attenuation
#[derive(Debug, Clone, Default)]
pub struct AttenuationMetadata {
    /// Reason for attenuation
    pub reason: Option<String>,
    /// Who performed the attenuation
    pub attenuated_by: Option<String>,
    /// When attenuation occurred
    pub attenuated_at: Option<String>,
    /// AI-readable explanation
    pub ai_explanation: Option<String>,
}

/// Revocable capability wrapper
#[derive(Debug)]
pub struct RevocableCapability<T: CapabilityInstance> {
    /// Inner capability
    pub inner: Option<T>,
    /// Whether this capability is revoked
    pub revoked: bool,
}

impl<T: CapabilityInstance> RevocableCapability<T> {
    /// Create a new revocable capability
    pub fn new(capability: T) -> Self {
        Self {
            inner: Some(capability),
            revoked: false,
        }
    }

    /// Use the capability if not revoked
    pub fn use_capability<R, F>(&self, operation: F) -> Result<R, crate::EffectSystemError>
    where
        F: FnOnce(&T) -> R,
    {
        if self.revoked {
            return Err(crate::EffectSystemError::CapabilityConstraintViolation {
                constraint: "Capability has been revoked".to_string(),
            });
        }

        match &self.inner {
            Some(capability) => Ok(operation(capability)),
            None => Err(crate::EffectSystemError::CapabilityNotFound {
                name: "RevocableCapability".to_string(),
            }),
        }
    }

    /// Revoke this capability
    pub fn revoke(&mut self) {
        self.revoked = true;
        self.inner = None;
    }

    /// Check if capability is revoked
    pub fn is_revoked(&self) -> bool {
        self.revoked
    }
}

/// Composite capability that combines multiple capabilities
#[derive(Debug, Clone)]
pub struct CompositeCapability {
    /// Name of this composite capability
    pub name: String,
    /// Individual capabilities that make up this composite
    pub capabilities: Vec<Capability>,
    /// Composition rules
    pub composition_rules: Vec<CapabilityCompositionRule>,
    /// AI context for the composite
    pub ai_context: Option<String>,
}

impl CompositeCapability {
    /// Create a new composite capability
    pub fn new(name: String, capabilities: Vec<Capability>) -> Self {
        Self {
            name,
            capabilities,
            composition_rules: Vec::new(),
            ai_context: None,
        }
    }

    /// Add a composition rule
    pub fn with_rule(mut self, rule: CapabilityCompositionRule) -> Self {
        self.composition_rules.push(rule);
        self
    }

    /// Check if all constituent capabilities allow an operation
    pub fn allows(&self, operation: &str, parameters: &HashMap<String, String>) -> bool {
        // Check if any of the constituent capabilities allow the operation
        self.capabilities.iter().any(|cap| {
            // This would need to be implemented based on the specific capability
            !cap.revoked
        })
    }
}

/// Rule for composing capabilities
#[derive(Debug, Clone)]
pub struct CapabilityCompositionRule {
    /// Rule name
    pub name: String,
    /// Description of the rule
    pub description: String,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Operations enabled by this rule
    pub enabled_operations: Vec<String>,
    /// Constraints for the composition
    pub constraints: Vec<AstNode<Expr>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_manager_creation() {
        let manager = CapabilityManager::new();
        assert!(!manager.capabilities.is_empty(), "Manager should have built-in capabilities");
        assert!(manager.capabilities.contains_key("FileSystem"));
        assert!(manager.capabilities.contains_key("Network"));
    }

    #[test]
    fn test_capability_definition_builder() {
        let capability = CapabilityDefinition::new(
            "TestCapability".to_string(),
            "A test capability".to_string(),
            CapabilityCategory::IO,
        )
        .with_authority("TestAuthority")
        .with_constraint("max_size", "Maximum size limit")
        .with_ai_context("This is for testing")
        .with_security_property("Provides secure access");

        assert_eq!(capability.name, "TestCapability");
        assert_eq!(capability.category, CapabilityCategory::IO);
        assert!(capability.authority.is_some());
        assert_eq!(capability.constraints.len(), 1);
        assert_eq!(capability.security_properties.len(), 1);
    }

    #[test]
    fn test_capability_constraints() {
        let constraints = CapabilityConstraints::new()
            .with_string("path", "/tmp")
            .with_list("operations", vec!["Read".to_string(), "Write".to_string()])
            .with_numeric("max_size", 1024.0)
            .with_boolean("read_only", false);

        assert_eq!(constraints.string_constraints.len(), 1);
        assert_eq!(constraints.list_constraints.len(), 1);
        assert_eq!(constraints.numeric_constraints.len(), 1);
        assert_eq!(constraints.boolean_constraints.len(), 1);
    }

    #[test]
    fn test_revocable_capability() {
        let capability = Capability::new(
            "TestCapability".to_string(),
            CapabilityConstraints::new(),
        );
        let mut revocable = RevocableCapability::new(capability);

        assert!(!revocable.is_revoked());
        
        revocable.revoke();
        assert!(revocable.is_revoked());
        assert!(revocable.inner.is_none());
    }
} 