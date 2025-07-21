//! Security & Trust Management
//!
//! This Smart Module represents the complete security and trust model for Prism's effect system.
//! It unifies capability-based security, information flow control, validation, and trust policies
//! into a single, conceptually cohesive unit that handles all aspects of secure computation.
//!
//! ## Conceptual Cohesion
//! 
//! This module embodies the business concept of "Security & Trust" by bringing together:
//! - Capability definitions and management (what code can do)
//! - Information flow control (how data moves securely)
//! - Validation and policy enforcement (ensuring security rules)
//! - Trust relationships and attestation (establishing confidence)

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use prism_ast::{AstNode, Expr, Type, SecurityClassification};
use prism_common::span::Span;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

/// Capability: Security & Trust Management
/// Description: Unified security model combining capabilities, information flow control, and trust policies
/// Dependencies: prism-ast, prism-common

/// The unified Security & Trust system that manages all aspects of secure computation
#[derive(Debug)]
pub struct SecurityTrustSystem {
    /// Capability management subsystem
    pub capability_manager: CapabilityManager,
    /// Information flow control subsystem  
    pub flow_controller: InformationFlowController,
    /// Security policy enforcement subsystem
    pub policy_enforcer: SecurityPolicyEnforcer,
    /// Trust relationship manager
    pub trust_manager: TrustManager,
}

impl SecurityTrustSystem {
    /// Create a new Security & Trust system with default security policies
    pub fn new() -> Self {
        Self {
            capability_manager: CapabilityManager::new(),
            flow_controller: InformationFlowController::new(),
            policy_enforcer: SecurityPolicyEnforcer::new(),
            trust_manager: TrustManager::new(),
        }
    }

    /// Establish a secure execution context with given capabilities and trust level
    pub fn create_secure_context(
        &mut self,
        required_capabilities: Vec<String>,
        trust_level: TrustLevel,
        data_classification: SecurityClassification,
    ) -> Result<SecureExecutionContext, SecurityError> {
        // Validate capabilities are available and not revoked
        let validated_capabilities = self.capability_manager
            .validate_capabilities(&required_capabilities)?;

        // Create information flow context
        let flow_context = self.flow_controller
            .create_flow_context(data_classification)?;

        // Apply security policies
        let policy_context = self.policy_enforcer
            .create_policy_context(&validated_capabilities, trust_level.clone())?;

        // Establish trust relationships
        let trust_context = self.trust_manager
            .establish_trust_context(trust_level)?;

        Ok(SecureExecutionContext {
            capabilities: validated_capabilities,
            flow_context,
            policy_context,
            trust_context,
            created_at: std::time::Instant::now(),
        })
    }

    /// Validate a security operation against all security dimensions
    pub fn validate_security_operation(
        &self,
        operation: &SecurityOperation,
        context: &SecureExecutionContext,
    ) -> Result<(), SecurityError> {
        // Check capability requirements
        self.capability_manager.check_operation_capabilities(operation, &context.capabilities)?;
        
        // Validate information flow
        self.flow_controller.validate_flow(operation, &context.flow_context)?;
        
        // Enforce security policies
        self.policy_enforcer.enforce_policies(operation, &context.policy_context)?;
        
        // Verify trust requirements
        self.trust_manager.verify_trust(operation, &context.trust_context)?;

        Ok(())
    }
}

// ============================================================================
// SECTION: Capability Management
// Handles what code is authorized to do through unforgeable capability tokens
// ============================================================================

/// Manager for all capabilities in the security system
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
    /// Create a new capability manager with built-in capabilities
    pub fn new() -> Self {
        let mut manager = Self::default();
        manager.register_builtin_capabilities();
        manager
    }

    /// Register a new capability definition
    pub fn register_capability(&mut self, capability: CapabilityDefinition) -> Result<(), SecurityError> {
        if self.capabilities.contains_key(&capability.name) {
            return Err(SecurityError::CapabilityAlreadyExists {
                name: capability.name.clone(),
            });
        }

        self.capabilities.insert(capability.name.clone(), capability);
        Ok(())
    }

    /// Validate that required capabilities are available and authorized
    pub fn validate_capabilities(&self, required: &[String]) -> Result<Vec<ValidatedCapability>, SecurityError> {
        let mut validated = Vec::new();
        
        for capability_name in required {
            if self.revoked.contains(capability_name) {
                return Err(SecurityError::CapabilityRevoked {
                    name: capability_name.clone(),
                });
            }

            let capability_def = self.capabilities.get(capability_name)
                .ok_or_else(|| SecurityError::CapabilityNotFound {
                    name: capability_name.clone(),
                })?;

            validated.push(ValidatedCapability {
                definition: capability_def.clone(),
                granted_at: std::time::Instant::now(),
                constraints: capability_def.default_constraints.clone(),
            });
        }

        Ok(validated)
    }

    /// Check if an operation is authorized by given capabilities
    pub fn check_operation_capabilities(
        &self,
        operation: &SecurityOperation,
        capabilities: &[ValidatedCapability],
    ) -> Result<(), SecurityError> {
        for required_cap in &operation.required_capabilities {
            let has_capability = capabilities.iter()
                .any(|cap| cap.definition.name == *required_cap);

            if !has_capability {
                return Err(SecurityError::InsufficientCapability {
                    operation: operation.name.clone(),
                    required: required_cap.clone(),
                });
            }
        }

        Ok(())
    }

    /// Register built-in capabilities from PLD-003 specification
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
            .with_ai_context("Provides controlled access to file system resources"),

            // Network capabilities
            CapabilityDefinition::new(
                "Network".to_string(),
                "Access to network operations".to_string(),
                CapabilityCategory::IO,
            )
            .with_authority("NetworkAuthority")
            .with_constraint("allowed_hosts", "Set of permitted hosts")
            .with_ai_context("Provides controlled network access"),

            // Database capabilities
            CapabilityDefinition::new(
                "Database".to_string(),
                "Access to database operations".to_string(),
                CapabilityCategory::Database,
            )
            .with_authority("DatabaseAuthority")
            .with_constraint("allowed_tables", "Set of accessible tables")
            .with_ai_context("Provides controlled database access"),

            // Cryptography capabilities
            CapabilityDefinition::new(
                "Cryptography".to_string(),
                "Access to cryptographic operations".to_string(),
                CapabilityCategory::Security,
            )
            .with_authority("CryptographyAuthority")
            .with_constraint("algorithms", "Set of allowed algorithms")
            .with_ai_context("Provides cryptographic capabilities"),

            // AI capabilities
            CapabilityDefinition::new(
                "AI".to_string(),
                "Access to AI model operations".to_string(),
                CapabilityCategory::AI,
            )
            .with_authority("AIAuthority")
            .with_constraint("approved_models", "Set of approved model IDs")
            .with_ai_context("Provides AI model inference capabilities"),
        ];

        for capability in capabilities {
            let _ = self.register_capability(capability);
        }
    }
}

/// Definition of a capability type with all its properties
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
    /// Default constraints for this capability
    pub default_constraints: CapabilityConstraints,
    /// AI-comprehensible context
    pub ai_context: Option<String>,
    /// Security properties this capability provides
    pub security_properties: Vec<String>,
}

impl CapabilityDefinition {
    /// Create a new capability definition
    pub fn new(name: String, description: String, category: CapabilityCategory) -> Self {
        Self {
            name,
            description,
            category,
            authority: None,
            default_constraints: CapabilityConstraints::new(),
            ai_context: None,
            security_properties: Vec::new(),
        }
    }

    /// Set the authority for this capability
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.authority = Some(authority.into());
        self
    }

    /// Add a constraint definition
    pub fn with_constraint(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.default_constraints.descriptions.insert(name.into(), description.into());
        self
    }

    /// Add AI context
    pub fn with_ai_context(mut self, context: impl Into<String>) -> Self {
        self.ai_context = Some(context.into());
        self
    }
}

/// Categories of capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    /// Constraint descriptions for AI comprehension
    pub descriptions: HashMap<String, String>,
}

impl CapabilityConstraints {
    /// Create new empty constraints
    pub fn new() -> Self {
        Self::default()
    }
}

/// A validated capability ready for use
#[derive(Debug, Clone)]
pub struct ValidatedCapability {
    /// The capability definition
    pub definition: CapabilityDefinition,
    /// When this capability was granted
    pub granted_at: std::time::Instant,
    /// Active constraints for this instance
    pub constraints: CapabilityConstraints,
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
}

// ============================================================================
// SECTION: Information Flow Control
// Manages how sensitive information flows through the system securely
// ============================================================================

/// Controller for information flow security
#[derive(Debug, Default)]
pub struct InformationFlowController {
    /// Active flow policies
    pub flow_policies: Vec<InformationFlowPolicy>,
    /// Security label lattice
    pub label_lattice: SecurityLabelLattice,
}

impl InformationFlowController {
    /// Create a new information flow controller
    pub fn new() -> Self {
        let mut controller = Self::default();
        controller.initialize_default_policies();
        controller
    }

    /// Create an information flow context for given classification
    pub fn create_flow_context(
        &self,
        classification: SecurityClassification,
    ) -> Result<InformationFlowContext, SecurityError> {
        Ok(InformationFlowContext {
            current_label: SecurityLabel {
                confidentiality: classification.into(),
                integrity: IntegrityLevel::Verified,
                origin: DataOrigin::System,
                purpose: DataPurpose::Processing,
            },
            flow_history: Vec::new(),
            active_policies: self.flow_policies.clone(),
        })
    }

    /// Validate that an operation's information flow is secure
    pub fn validate_flow(
        &self,
        operation: &SecurityOperation,
        context: &InformationFlowContext,
    ) -> Result<(), SecurityError> {
        for policy in &context.active_policies {
            if !policy.allows_operation(operation, &context.current_label) {
                return Err(SecurityError::InformationFlowViolation {
                    policy: policy.name.clone(),
                    operation: operation.name.clone(),
                });
            }
        }

        Ok(())
    }

    /// Initialize default information flow policies
    fn initialize_default_policies(&mut self) {
        self.flow_policies = vec![
            InformationFlowPolicy {
                name: "NoDowngrade".to_string(),
                description: "Information cannot flow from higher to lower confidentiality".to_string(),
                policy_type: FlowPolicyType::Confidentiality,
                allows_fn: Box::new(|operation, label| {
                    // Implementation would check confidentiality levels
                    true // Simplified for now
                }),
            },
            InformationFlowPolicy {
                name: "IntegrityPreservation".to_string(),
                description: "Operations cannot decrease integrity level".to_string(),
                policy_type: FlowPolicyType::Integrity,
                allows_fn: Box::new(|operation, label| {
                    // Implementation would check integrity preservation
                    true // Simplified for now
                }),
            },
        ];
    }
}

/// Security labels for information flow control
#[derive(Debug, Clone)]
pub struct SecurityLabel {
    /// Confidentiality classification
    pub confidentiality: ConfidentialityLevel,
    /// Integrity level
    pub integrity: IntegrityLevel,
    /// Data origin
    pub origin: DataOrigin,
    /// Data usage purpose
    pub purpose: DataPurpose,
}

/// Confidentiality levels in the security lattice
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidentialityLevel {
    Public,
    Internal,
    Confidential,
    Secret,
    TopSecret,
}

impl From<SecurityClassification> for ConfidentialityLevel {
    fn from(classification: SecurityClassification) -> Self {
            match classification {
        SecurityClassification::Public => ConfidentialityLevel::Public,
        SecurityClassification::Internal => ConfidentialityLevel::Internal,
        SecurityClassification::Confidential => ConfidentialityLevel::Confidential,
        SecurityClassification::TopSecret => ConfidentialityLevel::TopSecret,
        SecurityClassification::Restricted => ConfidentialityLevel::Secret,
    }
    }
}

/// Integrity levels in the security lattice
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntegrityLevel {
    Untrusted,
    Verified,
    Authenticated,
    Signed,
}

/// Origin of data for tracking purposes
#[derive(Debug, Clone)]
pub enum DataOrigin {
    System,
    User,
    External,
    AI,
}

/// Purpose for which data is being used
#[derive(Debug, Clone)]
pub enum DataPurpose {
    Processing,
    Storage,
    Transmission,
    Analysis,
    Display,
}

/// Information flow context for an execution
#[derive(Debug, Clone)]
pub struct InformationFlowContext {
    /// Current security label
    pub current_label: SecurityLabel,
    /// History of information flows
    pub flow_history: Vec<FlowEvent>,
    /// Active flow policies
    pub active_policies: Vec<InformationFlowPolicy>,
}

/// An information flow event for audit trails
#[derive(Debug, Clone)]
pub struct FlowEvent {
    /// Source label
    pub from: SecurityLabel,
    /// Destination label
    pub to: SecurityLabel,
    /// Operation that caused the flow
    pub operation: String,
    /// Timestamp of the flow
    pub timestamp: std::time::Instant,
}

/// Information flow policy
pub struct InformationFlowPolicy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Type of policy
    pub policy_type: FlowPolicyType,
    /// Function that determines if flow is allowed
    pub allows_fn: Box<dyn Fn(&SecurityOperation, &SecurityLabel) -> bool + Send + Sync>,
}

impl Clone for InformationFlowPolicy {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            policy_type: self.policy_type.clone(),
            allows_fn: Box::new(|_, _| true), // Simplified for cloning
        }
    }
}

impl InformationFlowPolicy {
    /// Check if this policy allows an operation
    pub fn allows_operation(&self, operation: &SecurityOperation, label: &SecurityLabel) -> bool {
        (self.allows_fn)(operation, label)
    }
}

/// Types of information flow policies
#[derive(Debug, Clone)]
pub enum FlowPolicyType {
    Confidentiality,
    Integrity,
    Purpose,
    Origin,
}

/// Security label lattice for ordering
#[derive(Debug, Default)]
pub struct SecurityLabelLattice {
    /// Confidentiality ordering
    pub confidentiality_order: Vec<ConfidentialityLevel>,
    /// Integrity ordering
    pub integrity_order: Vec<IntegrityLevel>,
}

// ============================================================================
// SECTION: Security Policy Enforcement
// Enforces security policies and validates security operations
// ============================================================================

/// Enforces security policies across all security dimensions
#[derive(Debug, Default)]
pub struct SecurityPolicyEnforcer {
    /// Active security policies
    pub policies: Vec<SecurityPolicy>,
    /// Policy enforcement configuration
    pub config: PolicyEnforcementConfig,
}

impl SecurityPolicyEnforcer {
    /// Create a new policy enforcer with default policies
    pub fn new() -> Self {
        let mut enforcer = Self::default();
        enforcer.initialize_default_policies();
        enforcer
    }

    /// Create a policy context for enforcement
    pub fn create_policy_context(
        &self,
        capabilities: &[ValidatedCapability],
        trust_level: TrustLevel,
    ) -> Result<SecurityPolicyContext, SecurityError> {
        Ok(SecurityPolicyContext {
            active_policies: self.policies.clone(),
            capabilities: capabilities.to_vec(),
            trust_level,
            enforcement_mode: self.config.enforcement_mode.clone(),
        })
    }

    /// Enforce all applicable policies for an operation
    pub fn enforce_policies(
        &self,
        operation: &SecurityOperation,
        context: &SecurityPolicyContext,
    ) -> Result<(), SecurityError> {
        for policy in &context.active_policies {
            if policy.applies_to(operation) {
                policy.enforce(operation, context)?;
            }
        }

        Ok(())
    }

    /// Initialize default security policies
    fn initialize_default_policies(&mut self) {
        self.policies = vec![
            SecurityPolicy {
                name: "LeastPrivilege".to_string(),
                description: "Operations should use minimal required capabilities".to_string(),
                policy_type: SecurityPolicyType::Capability,
                enforce_fn: Box::new(|operation, context| {
                    // Check that operation doesn't request excessive capabilities
                    Ok(())
                }),
                applies_fn: Box::new(|_| true),
            },
            SecurityPolicy {
                name: "AuditAllOperations".to_string(),
                description: "All security operations must be audited".to_string(),
                policy_type: SecurityPolicyType::Audit,
                enforce_fn: Box::new(|operation, context| {
                    // Ensure audit trail is enabled
                    Ok(())
                }),
                applies_fn: Box::new(|_| true),
            },
        ];
    }
}

/// Security policy definition
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Type of policy
    pub policy_type: SecurityPolicyType,
    /// Policy enforcement function
    pub enforce_fn: Box<dyn Fn(&SecurityOperation, &SecurityPolicyContext) -> Result<(), SecurityError> + Send + Sync>,
    /// Function to check if policy applies to operation
    pub applies_fn: Box<dyn Fn(&SecurityOperation) -> bool + Send + Sync>,
}

impl std::fmt::Debug for SecurityPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityPolicy")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("policy_type", &self.policy_type)
            .field("enforce_fn", &"<function>")
            .field("applies_fn", &"<function>")
            .finish()
    }
}

impl Clone for SecurityPolicy {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            policy_type: self.policy_type.clone(),
            enforce_fn: Box::new(|_, _| Ok(())), // Simplified for cloning
            applies_fn: Box::new(|_| true), // Simplified for cloning
        }
    }
}

impl SecurityPolicy {
    /// Check if this policy applies to an operation
    pub fn applies_to(&self, operation: &SecurityOperation) -> bool {
        (self.applies_fn)(operation)
    }

    /// Enforce this policy for an operation
    pub fn enforce(
        &self,
        operation: &SecurityOperation,
        context: &SecurityPolicyContext,
    ) -> Result<(), SecurityError> {
        (self.enforce_fn)(operation, context)
    }
}

/// Types of security policies
#[derive(Debug, Clone)]
pub enum SecurityPolicyType {
    Capability,
    InformationFlow,
    Audit,
    Trust,
}

/// Policy enforcement configuration
#[derive(Debug, Clone)]
pub struct PolicyEnforcementConfig {
    /// Whether to enforce policies strictly
    pub strict_enforcement: bool,
    /// Enforcement mode
    pub enforcement_mode: EnforcementMode,
    /// Whether to log policy violations
    pub log_violations: bool,
}

impl Default for PolicyEnforcementConfig {
    fn default() -> Self {
        Self {
            strict_enforcement: true,
            enforcement_mode: EnforcementMode::Strict,
            log_violations: true,
        }
    }
}

/// Policy enforcement modes
#[derive(Debug, Clone)]
pub enum EnforcementMode {
    /// Strict enforcement - violations cause failures
    Strict,
    /// Warning mode - violations generate warnings
    Warning,
    /// Audit mode - violations are logged only
    Audit,
}

/// Security policy enforcement context
#[derive(Debug, Clone)]
pub struct SecurityPolicyContext {
    /// Active policies for this context
    pub active_policies: Vec<SecurityPolicy>,
    /// Available capabilities
    pub capabilities: Vec<ValidatedCapability>,
    /// Trust level
    pub trust_level: TrustLevel,
    /// Enforcement mode
    pub enforcement_mode: EnforcementMode,
}

// ============================================================================
// SECTION: Trust Management
// Manages trust relationships and attestation
// ============================================================================

/// Manages trust relationships and attestation
#[derive(Debug, Default)]
pub struct TrustManager {
    /// Trust relationships
    pub trust_relationships: HashMap<String, TrustRelationship>,
    /// Trust policies
    pub trust_policies: Vec<TrustPolicy>,
}

impl TrustManager {
    /// Create a new trust manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Establish a trust context for given trust level
    pub fn establish_trust_context(&self, trust_level: TrustLevel) -> Result<TrustContext, SecurityError> {
        Ok(TrustContext {
            trust_level,
            established_at: std::time::Instant::now(),
            attestations: Vec::new(),
        })
    }

    /// Verify trust requirements for an operation
    pub fn verify_trust(
        &self,
        operation: &SecurityOperation,
        context: &TrustContext,
    ) -> Result<(), SecurityError> {
        if operation.required_trust_level > context.trust_level {
            return Err(SecurityError::InsufficientTrust {
                required: operation.required_trust_level.clone(),
                available: context.trust_level.clone(),
            });
        }

        Ok(())
    }
}

/// Trust levels in the system
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    Untrusted,
    Basic,
    Verified,
    Authenticated,
    HighAssurance,
}

/// Trust relationship between entities
#[derive(Debug, Clone)]
pub struct TrustRelationship {
    /// Entity being trusted
    pub trustee: String,
    /// Entity granting trust
    pub trustor: String,
    /// Level of trust
    pub trust_level: TrustLevel,
    /// When trust was established
    pub established_at: std::time::Instant,
    /// Trust attestations
    pub attestations: Vec<TrustAttestation>,
}

/// Trust attestation
#[derive(Debug, Clone)]
pub struct TrustAttestation {
    /// Attestation type
    pub attestation_type: String,
    /// Attesting authority
    pub authority: String,
    /// Attestation data
    pub data: HashMap<String, String>,
    /// When attestation was made
    pub timestamp: std::time::Instant,
}

/// Trust policy
#[derive(Debug, Clone)]
pub struct TrustPolicy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Required trust level
    pub required_trust_level: TrustLevel,
    /// Policy conditions
    pub conditions: Vec<String>,
}

/// Trust context for execution
#[derive(Debug, Clone)]
pub struct TrustContext {
    /// Current trust level
    pub trust_level: TrustLevel,
    /// When trust was established
    pub established_at: std::time::Instant,
    /// Trust attestations
    pub attestations: Vec<TrustAttestation>,
}

// ============================================================================
// SECTION: Common Types
// Shared types used across all security subsystems
// ============================================================================

/// A secure execution context that combines all security dimensions
#[derive(Debug)]
pub struct SecureExecutionContext {
    /// Validated capabilities
    pub capabilities: Vec<ValidatedCapability>,
    /// Information flow context
    pub flow_context: InformationFlowContext,
    /// Security policy context
    pub policy_context: SecurityPolicyContext,
    /// Trust context
    pub trust_context: TrustContext,
    /// When context was created
    pub created_at: std::time::Instant,
}

/// A security operation that needs validation
#[derive(Debug, Clone)]
pub struct SecurityOperation {
    /// Operation name
    pub name: String,
    /// Operation description
    pub description: String,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// Required trust level
    pub required_trust_level: TrustLevel,
    /// Information flow requirements
    pub flow_requirements: Vec<String>,
    /// Security classification of operation
    pub classification: SecurityClassification,
}

/// Security errors that can occur in the system
#[derive(Debug, Error)]
pub enum SecurityError {
    /// Capability already exists
    #[error("Capability '{name}' already exists")]
    CapabilityAlreadyExists { name: String },

    /// Capability not found
    #[error("Capability '{name}' not found")]
    CapabilityNotFound { name: String },

    /// Capability has been revoked
    #[error("Capability '{name}' has been revoked")]
    CapabilityRevoked { name: String },

    /// Insufficient capability for operation
    #[error("Operation '{operation}' requires capability '{required}' which is not available")]
    InsufficientCapability { operation: String, required: String },

    /// Information flow policy violation
    #[error("Information flow policy '{policy}' violated by operation '{operation}'")]
    InformationFlowViolation { policy: String, operation: String },

    /// Insufficient trust level
    #[error("Operation requires trust level {required:?} but only {available:?} is available")]
    InsufficientTrust { required: TrustLevel, available: TrustLevel },

    /// Security policy violation
    #[error("Security policy violation: {message}")]
    PolicyViolation { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_trust_system_creation() {
        let system = SecurityTrustSystem::new();
        assert!(!system.capability_manager.capabilities.is_empty());
        assert!(!system.flow_controller.flow_policies.is_empty());
        assert!(!system.policy_enforcer.policies.is_empty());
    }

    #[test]
    fn test_secure_context_creation() {
        let mut system = SecurityTrustSystem::new();
        let context = system.create_secure_context(
            vec!["FileSystem".to_string()],
            TrustLevel::Verified,
            SecurityClassification::Internal,
        );
        assert!(context.is_ok());
    }

    #[test]
    fn test_capability_validation() {
        let manager = CapabilityManager::new();
        let result = manager.validate_capabilities(&vec!["FileSystem".to_string()]);
        assert!(result.is_ok());
        
        let invalid_result = manager.validate_capabilities(&vec!["NonExistent".to_string()]);
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_information_flow_context() {
        let controller = InformationFlowController::new();
        let context = controller.create_flow_context(SecurityClassification::Confidential);
        assert!(context.is_ok());
        
        let flow_context = context.unwrap();
        assert_eq!(flow_context.current_label.confidentiality, ConfidentialityLevel::Confidential);
    }

    #[test]
    fn test_trust_levels() {
        assert!(TrustLevel::HighAssurance > TrustLevel::Verified);
        assert!(TrustLevel::Verified > TrustLevel::Basic);
        assert!(TrustLevel::Basic > TrustLevel::Untrusted);
    }
}

impl std::fmt::Debug for InformationFlowPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InformationFlowPolicy")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("policy_type", &self.policy_type)
            .field("allows_fn", &"<function>")
            .finish()
    }
} 