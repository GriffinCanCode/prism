//! Effect validation and constraint checking
//!
//! This module provides comprehensive validation of effect usage against
//! capability requirements, security policies, and business rules.

use crate::effects::{Effect};
use crate::capability::Capability;
use crate::capability::CapabilityManager;
use prism_common::span::Span;
use prism_ast::SecurityClassification;
use std::collections::HashMap;

/// Effect validator that ensures secure and correct effect usage
#[derive(Debug)]
pub struct EffectValidator {
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
    /// Security policies
    pub security_policies: Vec<SecurityPolicy>,
    /// Business rule validators
    pub business_validators: Vec<BusinessRuleValidator>,
    /// Validation configuration
    pub config: ValidationConfig,
}

impl EffectValidator {
    /// Create a new effect validator
    pub fn new() -> Self {
        let mut validator = Self {
            validation_rules: Vec::new(),
            security_policies: Vec::new(),
            business_validators: Vec::new(),
            config: ValidationConfig::default(),
        };
        validator.register_default_rules();
        validator
    }

    /// Validate effects against available capabilities
    pub fn validate_effects(
        &self,
        effects: &[Effect],
        available_capabilities: &[Capability],
        capability_manager: &CapabilityManager,
    ) -> Result<(), crate::EffectSystemError> {
        let mut validation_context = ValidationContext::new();

        // Build capability map for quick lookup
        let capability_map: HashMap<String, &Capability> = available_capabilities
            .iter()
            .map(|cap| (cap.definition.clone(), cap))
            .collect();

        // Validate each effect
        for effect in effects {
            self.validate_single_effect(effect, &capability_map, capability_manager, &mut validation_context)?;
        }

        // Apply validation rules
        self.apply_validation_rules(effects, available_capabilities, &mut validation_context)?;

        // Apply security policies
        self.apply_security_policies(effects, available_capabilities, &mut validation_context)?;

        // Apply business rule validations
        self.apply_business_validations(effects, &mut validation_context)?;

        // Check for validation failures
        if !validation_context.violations.is_empty() {
            let violation_messages: Vec<String> = validation_context.violations
                .iter()
                .map(|v| v.message.clone())
                .collect();
            return Err(crate::EffectSystemError::EffectValidationFailed {
                reason: violation_messages.join("; "),
            });
        }

        Ok(())
    }

    /// Validate a single effect
    fn validate_single_effect(
        &self,
        effect: &Effect,
        capability_map: &HashMap<String, &Capability>,
        capability_manager: &CapabilityManager,
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        // For now, we'll skip the effect definition lookup since we don't have direct access
        // In a full implementation, this would be passed as a parameter or accessed through the registry
        // Check if any of the required capabilities are available
        let required_capability_name = self.extract_capability_name(&effect.definition);
        if let Some(required_cap) = required_capability_name {
            if !capability_map.contains_key(&required_cap) {
                context.add_violation(ValidationViolation {
                    violation_type: ViolationType::MissingCapability,
                    message: format!(
                        "Effect '{}' requires capability '{}' which is not available",
                        effect.definition, required_cap
                    ),
                    severity: Severity::Error,
                    effect_name: effect.definition.clone(),
                    span: effect.span,
                });
            } else if let Some(capability) = capability_map.get(&required_cap) {
                if capability.revoked {
                    context.add_violation(ValidationViolation {
                        violation_type: ViolationType::RevokedCapability,
                        message: format!(
                            "Effect '{}' requires capability '{}' which has been revoked",
                            effect.definition, required_cap
                        ),
                        severity: Severity::Error,
                        effect_name: effect.definition.clone(),
                        span: effect.span,
                    });
                }
            }
        }

        Ok(())
    }

    /// Check if a capability has a specific permission
    fn check_permission(&self, capability: &Capability, permission: &str) -> bool {
        // This is a simplified implementation
        // In a full implementation, this would check the capability's constraints
        // to see if the requested permission is allowed
        !capability.revoked
    }

    /// Extract the required capability name from an effect definition
    fn extract_capability_name(&self, effect_definition: &str) -> Option<String> {
        // Extract capability from effect name (simplified mapping)
        if effect_definition.starts_with("IO.FileSystem") {
            Some("FileSystem".to_string())
        } else if effect_definition.starts_with("IO.Network") {
            Some("Network".to_string())
        } else if effect_definition.starts_with("Database") {
            Some("Database".to_string())
        } else if effect_definition.starts_with("Cryptography") {
            Some("Cryptography".to_string())
        } else if effect_definition.starts_with("AI") {
            Some("AI".to_string())
        } else {
            None
        }
    }

    /// Apply validation rules
    fn apply_validation_rules(
        &self,
        effects: &[Effect],
        capabilities: &[Capability],
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        for rule in &self.validation_rules {
            rule.apply(effects, capabilities, context)?;
        }
        Ok(())
    }

    /// Apply security policies
    fn apply_security_policies(
        &self,
        effects: &[Effect],
        capabilities: &[Capability],
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        for policy in &self.security_policies {
            policy.enforce(effects, capabilities, context)?;
        }
        Ok(())
    }

    /// Apply business rule validations
    fn apply_business_validations(
        &self,
        effects: &[Effect],
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        for validator in &self.business_validators {
            validator.validate(effects, context)?;
        }
        Ok(())
    }

    /// Register default validation rules
    fn register_default_rules(&mut self) {
        // Rule: No unsafe operations without explicit approval
        self.validation_rules.push(ValidationRule {
            name: "NoUnsafeWithoutApproval".to_string(),
            description: "Unsafe operations require explicit approval".to_string(),
            rule_type: RuleType::Security,
            condition: Box::new(|effects, _capabilities, _context| {
                effects.iter().any(|e| e.definition.starts_with("Unsafe"))
            }),
            action: Box::new(|effects, _capabilities, context| {
                for effect in effects {
                    if effect.definition.starts_with("Unsafe") {
                        // Check if there's explicit approval in metadata
                        let has_approval = effect.metadata.ai_context
                            .as_ref()
                            .map(|ctx| ctx.contains("approved"))
                            .unwrap_or(false);

                        if !has_approval {
                            context.add_violation(ValidationViolation {
                                violation_type: ViolationType::UnsafeOperationNotApproved,
                                message: format!("Unsafe operation '{}' requires explicit approval", effect.definition),
                                severity: Severity::Error,
                                effect_name: effect.definition.clone(),
                                span: effect.span,
                            });
                        }
                    }
                }
                Ok(())
            }),
        });

        // Rule: AI operations require content filtering
        self.validation_rules.push(ValidationRule {
            name: "AIContentFiltering".to_string(),
            description: "AI operations must have content filtering enabled".to_string(),
            rule_type: RuleType::Safety,
            condition: Box::new(|effects, _capabilities, _context| {
                effects.iter().any(|e| e.definition.starts_with("AI"))
            }),
            action: Box::new(|effects, capabilities, context| {
                for effect in effects {
                    if effect.definition.starts_with("AI") {
                        // Check if AI capability has content filtering enabled
                        let ai_capability = capabilities.iter()
                            .find(|cap| cap.definition == "AI");

                        if let Some(ai_cap) = ai_capability {
                            let has_filtering = ai_cap.constraints.boolean_constraints
                                .get("content_filtering")
                                .unwrap_or(&false);

                            if !has_filtering {
                                context.add_violation(ValidationViolation {
                                    violation_type: ViolationType::MissingContentFiltering,
                                    message: format!("AI operation '{}' requires content filtering to be enabled", effect.definition),
                                    severity: Severity::Warning,
                                    effect_name: effect.definition.clone(),
                                    span: effect.span,
                                });
                            }
                        }
                    }
                }
                Ok(())
            }),
        });

        // Security policy: Information flow control
        self.security_policies.push(SecurityPolicy {
            name: "InformationFlowControl".to_string(),
            description: "Prevent information flow from high to low security levels".to_string(),
            policy_type: PolicyType::InformationFlow,
            enforce_fn: Box::new(|effects, _capabilities, context| {
                // Check for information flow violations
                // This is a simplified implementation
                for effect in effects {
                    if effect.metadata.security_classification == SecurityClassification::TopSecret {
                        // Check if any effects might leak this information
                        let has_network_effect = effects.iter()
                            .any(|e| e.definition.contains("Network"));
                        
                        if has_network_effect {
                            context.add_violation(ValidationViolation {
                                violation_type: ViolationType::InformationFlowViolation,
                                message: "Top secret information cannot be transmitted over network without declassification".to_string(),
                                severity: Severity::Error,
                                effect_name: effect.definition.clone(),
                                span: effect.span,
                            });
                        }
                    }
                }
                Ok(())
            }),
        });

        // Business rule validator: Database transactions must be atomic
        self.business_validators.push(BusinessRuleValidator {
            name: "DatabaseTransactionAtomicity".to_string(),
            description: "Database operations should be wrapped in transactions".to_string(),
            validate_fn: Box::new(|effects, context| {
                let has_db_query = effects.iter().any(|e| e.definition == "Database.Query");
                let has_db_transaction = effects.iter().any(|e| e.definition == "Database.Transaction");

                if has_db_query && !has_db_transaction {
                    // Find the database query effect for proper span reporting
                    if let Some(db_effect) = effects.iter().find(|e| e.definition == "Database.Query") {
                        context.add_violation(ValidationViolation {
                            violation_type: ViolationType::BusinessRuleViolation,
                            message: "Database queries should be wrapped in transactions for consistency".to_string(),
                            severity: Severity::Warning,
                            effect_name: db_effect.definition.clone(),
                            span: db_effect.span,
                        });
                    }
                }
                Ok(())
            }),
        });
    }
}

impl Default for EffectValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for effect validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Whether to enforce security policies strictly
    pub strict_security: bool,
    /// Whether to validate business rules
    pub validate_business_rules: bool,
    /// Maximum allowed security classification
    pub max_security_classification: SecurityClassification,
    /// Whether to allow unsafe operations
    pub allow_unsafe_operations: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_security: true,
            validate_business_rules: true,
            max_security_classification: SecurityClassification::Confidential,
            allow_unsafe_operations: false,
        }
    }
}

/// Context for validation operations
#[derive(Debug)]
pub struct ValidationContext {
    /// Validation violations found
    pub violations: Vec<ValidationViolation>,
    /// Warnings generated during validation
    pub warnings: Vec<ValidationWarning>,
    /// Validation metadata
    pub metadata: ValidationMetadata,
}

impl ValidationContext {
    /// Create a new validation context
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            warnings: Vec::new(),
            metadata: ValidationMetadata::default(),
        }
    }

    /// Add a validation violation
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        self.violations.push(violation);
    }

    /// Add a validation warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.violations.iter().all(|v| v.severity != Severity::Error)
    }
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for validation
#[derive(Debug, Default)]
pub struct ValidationMetadata {
    /// Validation start time
    pub start_time: Option<std::time::Instant>,
    /// Validation duration
    pub duration: Option<std::time::Duration>,
    /// Number of effects validated
    pub effects_count: usize,
    /// Number of capabilities checked
    pub capabilities_count: usize,
}

/// A validation violation
#[derive(Debug, Clone)]
pub struct ValidationViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    /// Human-readable message
    pub message: String,
    /// Severity of the violation
    pub severity: Severity,
    /// Effect that caused the violation
    pub effect_name: String,
    /// Source location
    pub span: Span,
}

/// Types of validation violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationType {
    /// Missing required capability
    MissingCapability,
    /// Capability has been revoked
    RevokedCapability,
    /// Insufficient permission on capability
    InsufficientPermission,
    /// Unsafe operation not approved
    UnsafeOperationNotApproved,
    /// Missing content filtering for AI operations
    MissingContentFiltering,
    /// Information flow policy violation
    InformationFlowViolation,
    /// Business rule violation
    BusinessRuleViolation,
    /// Security policy violation
    SecurityPolicyViolation,
}

/// Severity levels for violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// Error - validation fails
    Error,
    /// Warning - validation passes with warning
    Warning,
    /// Info - informational message
    Info,
}

/// A validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Effect that generated the warning
    pub effect_name: String,
    /// Source location
    pub span: Span,
}

/// Validation rule
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Type of rule
    pub rule_type: RuleType,
    /// Condition for applying the rule
    pub condition: Box<dyn Fn(&[Effect], &[Capability], &ValidationContext) -> bool + Send + Sync>,
    /// Action to take when rule applies
    pub action: Box<dyn Fn(&[Effect], &[Capability], &mut ValidationContext) -> Result<(), crate::EffectSystemError> + Send + Sync>,
}

impl std::fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("rule_type", &self.rule_type)
            .field("condition", &"<function>")
            .field("action", &"<function>")
            .finish()
    }
}

impl ValidationRule {
    /// Apply this validation rule
    pub fn apply(
        &self,
        effects: &[Effect],
        capabilities: &[Capability],
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        if (self.condition)(effects, capabilities, context) {
            (self.action)(effects, capabilities, context)?;
        }
        Ok(())
    }
}

/// Types of validation rules
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleType {
    /// Security-related rule
    Security,
    /// Safety-related rule
    Safety,
    /// Performance-related rule
    Performance,
    /// Business logic rule
    Business,
}

/// Security policy
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy description
    pub description: String,
    /// Type of policy
    pub policy_type: PolicyType,
    /// Policy enforcement function
    pub enforce_fn: Box<dyn Fn(&[Effect], &[Capability], &mut ValidationContext) -> Result<(), crate::EffectSystemError> + Send + Sync>,
}

impl std::fmt::Debug for SecurityPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityPolicy")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("policy_type", &self.policy_type)
            .field("enforce_fn", &"<function>")
            .finish()
    }
}

impl SecurityPolicy {
    /// Enforce this security policy
    pub fn enforce(
        &self,
        effects: &[Effect],
        capabilities: &[Capability],
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        (self.enforce_fn)(effects, capabilities, context)
    }
}

/// Types of security policies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyType {
    /// Access control policy
    AccessControl,
    /// Information flow policy
    InformationFlow,
    /// Audit policy
    Audit,
    /// Capability policy
    Capability,
}

/// Business rule validator
pub struct BusinessRuleValidator {
    /// Validator name
    pub name: String,
    /// Validator description
    pub description: String,
    /// Validation function
    pub validate_fn: Box<dyn Fn(&[Effect], &mut ValidationContext) -> Result<(), crate::EffectSystemError> + Send + Sync>,
}

impl std::fmt::Debug for BusinessRuleValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BusinessRuleValidator")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("validate_fn", &"<function>")
            .finish()
    }
}

impl BusinessRuleValidator {
    /// Validate effects against business rules
    pub fn validate(
        &self,
        effects: &[Effect],
        context: &mut ValidationContext,
    ) -> Result<(), crate::EffectSystemError> {
        (self.validate_fn)(effects, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::CapabilityConstraints;
    use crate::effects::definition::EffectInstanceMetadata;

    #[test]
    fn test_validator_creation() {
        let validator = EffectValidator::new();
        assert!(!validator.validation_rules.is_empty());
        assert!(!validator.security_policies.is_empty());
        assert!(!validator.business_validators.is_empty());
    }

    #[test]
    fn test_validation_context() {
        let mut context = ValidationContext::new();
        assert!(context.is_valid());

        context.add_violation(ValidationViolation {
            violation_type: ViolationType::MissingCapability,
            message: "Test violation".to_string(),
            severity: Severity::Error,
            effect_name: "TestEffect".to_string(),
            span: Span::dummy(),
        });

        assert!(!context.is_valid());
        assert_eq!(context.violations.len(), 1);
    }

    #[test]
    fn test_missing_capability_validation() {
        let validator = EffectValidator::new();
        let capability_manager = CapabilityManager::new();

        let effects = vec![
            Effect::new("IO.FileSystem.Read".to_string(), Span::dummy()),
        ];

        // No capabilities provided
        let capabilities = vec![];

        let result = validator.validate_effects(&effects, &capabilities, &capability_manager);
        assert!(result.is_err());
    }

    #[test]
    fn test_revoked_capability_validation() {
        let validator = EffectValidator::new();
        let capability_manager = CapabilityManager::new();

        let effects = vec![
            Effect::new("IO.FileSystem.Read".to_string(), Span::dummy()),
        ];

        // Provide a revoked capability
        let mut capability = Capability::new(
            "FileSystem".to_string(),
            CapabilityConstraints::new(),
        );
        capability.revoke();
        let capabilities = vec![capability];

        let result = validator.validate_effects(&effects, &capabilities, &capability_manager);
        assert!(result.is_err());
    }
} 