//! Scope Boundaries for Effects and Capabilities
//!
//! This module defines boundaries that scopes establish for effects and
//! capabilities, integrating with PLD-003's effect system and capability-based
//! security model.
//!
//! **Conceptual Responsibility**: Scope boundary definition and enforcement
//! **What it does**: Define effect boundaries, capability boundaries, security boundaries
//! **What it doesn't do**: Effect execution, capability checking, scope hierarchy management

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Effect boundary that a scope establishes
/// 
/// Defines what effects are allowed, required, or prohibited within a scope,
/// integrating with PLD-003's effect system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectBoundary {
    /// Unique identifier for this boundary
    pub id: String,
    
    /// Type of effect boundary
    pub boundary_type: EffectBoundaryType,
    
    /// Effects that are allowed within this boundary
    pub allowed_effects: HashSet<String>,
    
    /// Effects that are required within this boundary
    pub required_effects: HashSet<String>,
    
    /// Effects that are prohibited within this boundary
    pub prohibited_effects: HashSet<String>,
    
    /// Effect composition rules within this boundary
    pub composition_rules: Vec<EffectCompositionRule>,
    
    /// Boundary enforcement mode
    pub enforcement_mode: BoundaryEnforcementMode,
    
    /// Documentation for this boundary
    pub documentation: Option<String>,
    
    /// Boundary metadata for AI comprehension
    pub metadata: BoundaryMetadata,
}

/// Capability boundary that a scope establishes
/// 
/// Defines what capabilities are available, required, or restricted within a scope,
/// integrating with PLD-003's capability-based security.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityBoundary {
    /// Unique identifier for this boundary
    pub id: String,
    
    /// Type of capability boundary
    pub boundary_type: CapabilityBoundaryType,
    
    /// Capabilities that are available within this boundary
    pub available_capabilities: HashSet<String>,
    
    /// Capabilities that are required to enter this boundary
    pub required_capabilities: HashSet<String>,
    
    /// Capabilities that are restricted within this boundary
    pub restricted_capabilities: HashSet<String>,
    
    /// Capability attenuation rules
    pub attenuation_rules: Vec<CapabilityAttenuationRule>,
    
    /// Security classification of this boundary
    pub security_classification: SecurityClassification,
    
    /// Boundary enforcement mode
    pub enforcement_mode: BoundaryEnforcementMode,
    
    /// Documentation for this boundary
    pub documentation: Option<String>,
    
    /// Boundary metadata for AI comprehension
    pub metadata: BoundaryMetadata,
}

/// Security boundary that a scope establishes
/// 
/// Defines security policies, trust levels, and isolation requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityBoundary {
    /// Unique identifier for this boundary
    pub id: String,
    
    /// Security classification level
    pub classification: SecurityClassification,
    
    /// Trust level required to cross this boundary
    pub required_trust_level: TrustLevel,
    
    /// Isolation requirements
    pub isolation_requirements: IsolationRequirements,
    
    /// Security policies enforced at this boundary
    pub security_policies: Vec<SecurityPolicy>,
    
    /// Audit requirements for boundary crossings
    pub audit_requirements: AuditRequirements,
    
    /// Boundary enforcement mode
    pub enforcement_mode: BoundaryEnforcementMode,
    
    /// Documentation for this boundary
    pub documentation: Option<String>,
    
    /// Boundary metadata for AI comprehension
    pub metadata: BoundaryMetadata,
}

/// Types of effect boundaries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectBoundaryType {
    /// Permissive boundary (allows most effects)
    Permissive,
    
    /// Restrictive boundary (allows only specified effects)
    Restrictive,
    
    /// Isolation boundary (isolates effects from outer scope)
    Isolation,
    
    /// Composition boundary (defines effect composition rules)
    Composition,
    
    /// Monitoring boundary (observes but doesn't restrict effects)
    Monitoring,
    
    /// Custom boundary type
    Custom(String),
}

/// Types of capability boundaries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityBoundaryType {
    /// Grant boundary (provides additional capabilities)
    Grant,
    
    /// Restrict boundary (removes or limits capabilities)
    Restrict,
    
    /// Isolate boundary (isolates capabilities from outer scope)
    Isolate,
    
    /// Attenuate boundary (weakens capabilities)
    Attenuate,
    
    /// Validate boundary (checks capability usage)
    Validate,
    
    /// Custom boundary type
    Custom(String),
}

/// Effect composition rule within a boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectCompositionRule {
    /// Name of this composition rule
    pub name: String,
    
    /// Effects involved in this rule
    pub effects: Vec<String>,
    
    /// Composition operation
    pub operation: CompositionOperation,
    
    /// Result of the composition
    pub result: CompositionResult,
    
    /// Conditions under which this rule applies
    pub conditions: Vec<String>,
    
    /// Priority of this rule (higher numbers take precedence)
    pub priority: u32,
}

/// Capability attenuation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityAttenuationRule {
    /// Name of this attenuation rule
    pub name: String,
    
    /// Capability being attenuated
    pub capability: String,
    
    /// Attenuation operation
    pub operation: AttenuationOperation,
    
    /// Parameters for the attenuation
    pub parameters: HashMap<String, String>,
    
    /// Conditions under which this rule applies
    pub conditions: Vec<String>,
}

/// Composition operations for effects
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionOperation {
    /// Sequential composition (A then B)
    Sequential,
    
    /// Parallel composition (A and B simultaneously)
    Parallel,
    
    /// Alternative composition (A or B)
    Alternative,
    
    /// Conditional composition (A if condition else B)
    Conditional,
    
    /// Merge composition (combine A and B)
    Merge,
    
    /// Override composition (B overrides A)
    Override,
}

/// Result of effect composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionResult {
    /// Composition is allowed
    Allow,
    
    /// Composition is denied
    Deny,
    
    /// Composition requires additional capabilities
    RequireCapabilities(Vec<String>),
    
    /// Composition produces a new effect
    ProduceEffect(String),
    
    /// Composition is conditionally allowed
    ConditionalAllow(Vec<String>),
}

/// Attenuation operations for capabilities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttenuationOperation {
    /// Limit the scope of the capability
    LimitScope,
    
    /// Reduce the permissions of the capability
    ReducePermissions,
    
    /// Add time restrictions to the capability
    AddTimeRestriction,
    
    /// Add usage count restrictions
    AddUsageRestriction,
    
    /// Add resource restrictions
    AddResourceRestriction,
    
    /// Custom attenuation operation
    Custom(String),
}

/// Security classification levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecurityClassification {
    /// Public information
    Public,
    
    /// Internal use only
    Internal,
    
    /// Confidential information
    Confidential,
    
    /// Secret information
    Secret,
    
    /// Top secret information
    TopSecret,
    
    /// Custom classification
    Custom(String),
}

/// Trust levels for security boundaries
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustLevel {
    /// No trust required
    None,
    
    /// Basic trust level
    Basic,
    
    /// Verified trust level
    Verified,
    
    /// Authenticated trust level
    Authenticated,
    
    /// Certified trust level
    Certified,
    
    /// Maximum trust level
    Maximum,
}

/// Isolation requirements for security boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationRequirements {
    /// Process isolation required
    pub process_isolation: bool,
    
    /// Memory isolation required
    pub memory_isolation: bool,
    
    /// Network isolation required
    pub network_isolation: bool,
    
    /// File system isolation required
    pub filesystem_isolation: bool,
    
    /// Custom isolation requirements
    pub custom_requirements: HashMap<String, String>,
}

/// Security policy enforced at a boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    
    /// Policy type
    pub policy_type: SecurityPolicyType,
    
    /// Policy rules
    pub rules: Vec<String>,
    
    /// Policy enforcement level
    pub enforcement_level: EnforcementLevel,
    
    /// Policy documentation
    pub documentation: Option<String>,
}

/// Types of security policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityPolicyType {
    /// Access control policy
    AccessControl,
    
    /// Information flow policy
    InformationFlow,
    
    /// Audit policy
    Audit,
    
    /// Compliance policy
    Compliance,
    
    /// Custom policy type
    Custom(String),
}

/// Audit requirements for boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequirements {
    /// Whether boundary crossings should be logged
    pub log_crossings: bool,
    
    /// Whether capability usage should be logged
    pub log_capability_usage: bool,
    
    /// Whether effect execution should be logged
    pub log_effect_execution: bool,
    
    /// Audit retention period in days
    pub retention_period_days: u32,
    
    /// Custom audit requirements
    pub custom_requirements: HashMap<String, String>,
}

/// Boundary enforcement modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryEnforcementMode {
    /// Strict enforcement (violations cause errors)
    Strict,
    
    /// Permissive enforcement (violations cause warnings)
    Permissive,
    
    /// Monitoring only (violations are logged but allowed)
    Monitor,
    
    /// Disabled (no enforcement)
    Disabled,
}

/// Enforcement levels for policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory (recommendations only)
    Advisory,
    
    /// Warning (violations generate warnings)
    Warning,
    
    /// Error (violations cause errors)
    Error,
    
    /// Critical (violations cause immediate termination)
    Critical,
}

/// Metadata for boundaries to enable AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryMetadata {
    /// Purpose of this boundary
    pub purpose: String,
    
    /// Business justification for this boundary
    pub business_justification: Option<String>,
    
    /// Security rationale
    pub security_rationale: Option<String>,
    
    /// Performance implications
    pub performance_implications: Vec<String>,
    
    /// Related boundaries
    pub related_boundaries: Vec<String>,
    
    /// Examples of boundary usage
    pub usage_examples: Vec<String>,
    
    /// Common violations and their resolutions
    pub common_violations: Vec<ViolationExample>,
}

/// Example of a boundary violation and its resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationExample {
    /// Description of the violation
    pub violation: String,
    
    /// Why it's a violation
    pub reason: String,
    
    /// How to resolve the violation
    pub resolution: String,
    
    /// Example code showing the fix
    pub example_fix: Option<String>,
}

impl EffectBoundary {
    /// Create a new effect boundary
    pub fn new(id: String, boundary_type: EffectBoundaryType) -> Self {
        Self {
            id,
            boundary_type,
            allowed_effects: HashSet::new(),
            required_effects: HashSet::new(),
            prohibited_effects: HashSet::new(),
            composition_rules: Vec::new(),
            enforcement_mode: BoundaryEnforcementMode::Strict,
            documentation: None,
            metadata: BoundaryMetadata {
                purpose: "Effect boundary".to_string(),
                business_justification: None,
                security_rationale: None,
                performance_implications: Vec::new(),
                related_boundaries: Vec::new(),
                usage_examples: Vec::new(),
                common_violations: Vec::new(),
            },
        }
    }
    
    /// Check if an effect is allowed within this boundary
    pub fn is_effect_allowed(&self, effect: &str) -> bool {
        match self.boundary_type {
            EffectBoundaryType::Permissive => !self.prohibited_effects.contains(effect),
            EffectBoundaryType::Restrictive => self.allowed_effects.contains(effect),
            EffectBoundaryType::Isolation => self.allowed_effects.contains(effect),
            EffectBoundaryType::Composition => true, // Handled by composition rules
            EffectBoundaryType::Monitoring => true, // Monitoring doesn't restrict
            EffectBoundaryType::Custom(_) => self.allowed_effects.contains(effect),
        }
    }
    
    /// Add an allowed effect to this boundary
    pub fn allow_effect(&mut self, effect: String) {
        self.allowed_effects.insert(effect);
    }
    
    /// Add a prohibited effect to this boundary
    pub fn prohibit_effect(&mut self, effect: String) {
        self.prohibited_effects.insert(effect);
    }
}

impl CapabilityBoundary {
    /// Create a new capability boundary
    pub fn new(id: String, boundary_type: CapabilityBoundaryType) -> Self {
        Self {
            id,
            boundary_type,
            available_capabilities: HashSet::new(),
            required_capabilities: HashSet::new(),
            restricted_capabilities: HashSet::new(),
            attenuation_rules: Vec::new(),
            security_classification: SecurityClassification::Internal,
            enforcement_mode: BoundaryEnforcementMode::Strict,
            documentation: None,
            metadata: BoundaryMetadata {
                purpose: "Capability boundary".to_string(),
                business_justification: None,
                security_rationale: None,
                performance_implications: Vec::new(),
                related_boundaries: Vec::new(),
                usage_examples: Vec::new(),
                common_violations: Vec::new(),
            },
        }
    }
    
    /// Check if a capability is available within this boundary
    pub fn is_capability_available(&self, capability: &str) -> bool {
        match self.boundary_type {
            CapabilityBoundaryType::Grant => self.available_capabilities.contains(capability),
            CapabilityBoundaryType::Restrict => !self.restricted_capabilities.contains(capability),
            CapabilityBoundaryType::Isolate => self.available_capabilities.contains(capability),
            CapabilityBoundaryType::Attenuate => self.available_capabilities.contains(capability),
            CapabilityBoundaryType::Validate => true, // Validation doesn't restrict availability
            CapabilityBoundaryType::Custom(_) => self.available_capabilities.contains(capability),
        }
    }
    
    /// Add an available capability to this boundary
    pub fn grant_capability(&mut self, capability: String) {
        self.available_capabilities.insert(capability);
    }
    
    /// Add a restricted capability to this boundary
    pub fn restrict_capability(&mut self, capability: String) {
        self.restricted_capabilities.insert(capability);
    }
}

impl SecurityBoundary {
    /// Create a new security boundary
    pub fn new(id: String, classification: SecurityClassification) -> Self {
        Self {
            id,
            classification,
            required_trust_level: TrustLevel::Basic,
            isolation_requirements: IsolationRequirements {
                process_isolation: false,
                memory_isolation: false,
                network_isolation: false,
                filesystem_isolation: false,
                custom_requirements: HashMap::new(),
            },
            security_policies: Vec::new(),
            audit_requirements: AuditRequirements {
                log_crossings: true,
                log_capability_usage: false,
                log_effect_execution: false,
                retention_period_days: 30,
                custom_requirements: HashMap::new(),
            },
            enforcement_mode: BoundaryEnforcementMode::Strict,
            documentation: None,
            metadata: BoundaryMetadata {
                purpose: "Security boundary".to_string(),
                business_justification: None,
                security_rationale: None,
                performance_implications: Vec::new(),
                related_boundaries: Vec::new(),
                usage_examples: Vec::new(),
                common_violations: Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_effect_boundary_creation() {
        let mut boundary = EffectBoundary::new(
            "test_boundary".to_string(),
            EffectBoundaryType::Restrictive,
        );
        
        boundary.allow_effect("IO".to_string());
        boundary.prohibit_effect("Network".to_string());
        
        assert!(boundary.is_effect_allowed("IO"));
        assert!(!boundary.is_effect_allowed("Network"));
        assert!(!boundary.is_effect_allowed("Unknown"));
    }
    
    #[test]
    fn test_capability_boundary_creation() {
        let mut boundary = CapabilityBoundary::new(
            "test_boundary".to_string(),
            CapabilityBoundaryType::Grant,
        );
        
        boundary.grant_capability("FileRead".to_string());
        boundary.restrict_capability("NetworkAccess".to_string());
        
        assert!(boundary.is_capability_available("FileRead"));
        // For Grant type, only explicitly granted capabilities are available
        assert!(!boundary.is_capability_available("NetworkAccess"));
    }
    
    #[test]
    fn test_security_boundary_creation() {
        let boundary = SecurityBoundary::new(
            "secure_zone".to_string(),
            SecurityClassification::Confidential,
        );
        
        assert_eq!(boundary.classification, SecurityClassification::Confidential);
        assert_eq!(boundary.required_trust_level, TrustLevel::Basic);
        assert!(boundary.audit_requirements.log_crossings);
    }
    
    #[test]
    fn test_security_classification_ordering() {
        assert!(SecurityClassification::Public < SecurityClassification::Internal);
        assert!(SecurityClassification::Internal < SecurityClassification::Confidential);
        assert!(SecurityClassification::Confidential < SecurityClassification::Secret);
        assert!(SecurityClassification::Secret < SecurityClassification::TopSecret);
    }
    
    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::None < TrustLevel::Basic);
        assert!(TrustLevel::Basic < TrustLevel::Verified);
        assert!(TrustLevel::Verified < TrustLevel::Authenticated);
        assert!(TrustLevel::Authenticated < TrustLevel::Certified);
        assert!(TrustLevel::Certified < TrustLevel::Maximum);
    }
} 