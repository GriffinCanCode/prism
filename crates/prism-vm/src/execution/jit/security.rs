//! JIT Security Integration
//!
//! This module provides security integration for the JIT compiler, ensuring that
//! compiled code maintains the same security guarantees as interpreted code.
//! It integrates with existing security infrastructure rather than duplicating logic.
//!
//! ## Security Responsibilities
//!
//! - **Bytecode Security Validation**: Validate bytecode before JIT compilation
//! - **Capability Integration**: Embed capability checks in generated code
//! - **Effect Tracking**: Monitor effects during compilation and execution
//! - **Runtime Security Enforcement**: Ensure compiled code respects security policies
//! - **Code Generation Validation**: Validate generated machine code for security
//!
//! ## Integration Approach
//!
//! - **Leverages Existing Systems**: Uses prism-effects, prism-runtime security
//! - **No Logic Duplication**: Interfaces with rather than reimplements security
//! - **JIT-Specific Focus**: Handles security concerns unique to compilation
//! - **Performance Optimized**: Minimizes security overhead in generated code

use crate::{VMResult, PrismVMError, bytecode::{PrismBytecode, FunctionDefinition, Instruction}};
use prism_runtime::{
    authority::capability::{CapabilitySet, Capability, Operation},
    security::{SecuritySystem, SecurityConfig, SecurityLevel, PolicyDecision},
    platform::execution::ExecutionContext,
};
use prism_effects::{
    security::{SecuritySystem as EffectSecuritySystem, SecureExecutionContext},
    effects::{Effect, EffectDefinition},
    validation::{EffectValidator, ValidationContext, ValidationViolation},
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, span, Level};

/// JIT security compiler that integrates with existing security infrastructure
#[derive(Debug)]
pub struct SecurityCompiler {
    /// Configuration
    config: SecurityConfig,
    
    /// Integration with runtime security system
    runtime_security: Arc<SecuritySystem>,
    
    /// Integration with effects security system
    effects_security: Arc<RwLock<EffectSecuritySystem>>,
    
    /// Effect validator for compilation-time validation
    effect_validator: Arc<EffectValidator>,
    
    /// Security policy cache for performance
    policy_cache: Arc<RwLock<HashMap<String, PolicyDecision>>>,
    
    /// Compilation security statistics
    stats: Arc<RwLock<SecurityStats>>,
}

/// Security configuration for JIT compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security validation during compilation
    pub enable_validation: bool,
    
    /// Enable capability checking in generated code
    pub enable_capability_checks: bool,
    
    /// Enable effect tracking in generated code
    pub enable_effect_tracking: bool,
    
    /// Enable runtime security policy enforcement
    pub enable_policy_enforcement: bool,
    
    /// Security level for JIT operations
    pub security_level: SecurityLevel,
    
    /// Maximum compilation time before security timeout
    pub max_compilation_time: Duration,
    
    /// Enable security audit logging
    pub enable_audit_logging: bool,
    
    /// Cache security decisions for performance
    pub enable_policy_caching: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            enable_capability_checks: true,
            enable_effect_tracking: true,
            enable_policy_enforcement: true,
            security_level: SecurityLevel::Internal,
            max_compilation_time: Duration::from_secs(30),
            enable_audit_logging: true,
            enable_policy_caching: true,
        }
    }
}

/// Security statistics for JIT compilation
#[derive(Debug, Clone, Default)]
pub struct SecurityStats {
    /// Number of security validations performed
    pub validations_performed: u64,
    /// Number of security violations detected
    pub violations_detected: u64,
    /// Number of capability checks inserted
    pub capability_checks_inserted: u64,
    /// Number of effect tracking points inserted
    pub effect_tracking_points: u64,
    /// Total time spent on security operations
    pub total_security_time: Duration,
    /// Policy cache hit rate
    pub policy_cache_hit_rate: f64,
}

/// Security violation detected during JIT compilation
#[derive(Debug, Clone)]
pub struct SecurityViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    /// Description of the violation
    pub message: String,
    /// Function where violation occurred
    pub function_name: String,
    /// Instruction offset where violation occurred
    pub instruction_offset: Option<u32>,
    /// Severity of the violation
    pub severity: Severity,
}

/// Types of security violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationType {
    /// Capability requirement not met
    CapabilityViolation,
    /// Effect not properly declared
    UndeclaredEffect,
    /// Security policy violation
    PolicyViolation,
    /// Unsafe operation without proper authorization
    UnsafeOperation,
    /// Resource limit exceeded
    ResourceLimitExceeded,
    /// Invalid bytecode structure
    InvalidBytecode,
}

/// Severity levels for security violations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational - log only
    Info,
    /// Warning - allow but log
    Warning,
    /// Error - compilation should fail
    Error,
    /// Critical - immediate termination
    Critical,
}

/// Capability checker for JIT-compiled code
#[derive(Debug)]
pub struct CapabilityChecker {
    /// Required capabilities for compilation
    required_capabilities: HashSet<String>,
    /// Available capabilities
    available_capabilities: Arc<CapabilitySet>,
    /// Capability check insertion points
    check_points: Vec<CheckPoint>,
}

/// Effect tracker for JIT compilation
#[derive(Debug)]
pub struct EffectTracker {
    /// Effects discovered during compilation
    discovered_effects: Vec<Effect>,
    /// Effect tracking insertion points
    tracking_points: Vec<TrackingPoint>,
    /// Effect validation results
    validation_results: Vec<ValidationViolation>,
}

/// Security policy for JIT compilation
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy rules
    pub rules: Vec<SecurityRule>,
    /// Enforcement level
    pub enforcement_level: EnforcementLevel,
}

/// Security rule within a policy
#[derive(Debug, Clone)]
pub struct SecurityRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: String,
    /// Action to take when rule matches
    pub action: SecurityAction,
}

/// Security actions
#[derive(Debug, Clone)]
pub enum SecurityAction {
    /// Allow the operation
    Allow,
    /// Deny the operation
    Deny,
    /// Allow with additional checks
    AllowWithChecks(Vec<String>),
    /// Log and allow
    LogAndAllow,
}

/// Enforcement levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnforcementLevel {
    /// Advisory only - log violations
    Advisory,
    /// Blocking - prevent violations
    Blocking,
    /// Strict - terminate on any violation
    Strict,
}

/// Capability check insertion point
#[derive(Debug, Clone)]
pub struct CheckPoint {
    /// Function where check should be inserted
    pub function_id: u32,
    /// Instruction offset for insertion
    pub instruction_offset: u32,
    /// Required capability
    pub required_capability: String,
    /// Check type
    pub check_type: CheckType,
}

/// Types of capability checks
#[derive(Debug, Clone)]
pub enum CheckType {
    /// Basic capability presence check
    Presence,
    /// Capability with specific parameters
    Parameterized(HashMap<String, String>),
    /// Capability delegation check
    Delegation,
    /// Capability expiration check
    Expiration,
    /// Function entry capability check
    FunctionEntry,
    /// Effect capability check
    EffectCapability,
    /// Instruction effect capability check
    InstructionEffect,
}

/// Effect tracking insertion point
#[derive(Debug, Clone)]
pub struct TrackingPoint {
    /// Function where tracking should be inserted
    pub function_id: u32,
    /// Instruction offset for insertion
    pub instruction_offset: u32,
    /// Effect to track
    pub effect: Effect,
    /// Tracking type
    pub tracking_type: TrackingType,
}

/// Types of effect tracking
#[derive(Debug, Clone)]
pub enum TrackingType {
    /// Track effect start
    Start,
    /// Track effect end
    End,
    /// Track effect parameters
    Parameters,
    /// Track effect result
    Result,
}

impl SecurityCompiler {
    /// Create a new security compiler with default configuration
    pub fn new_default() -> VMResult<Self> {
        Self::new(SecurityConfig::default())
    }

    /// Create a new security compiler with configuration
    pub fn new(config: SecurityConfig) -> VMResult<Self> {
        let _span = span!(Level::INFO, "security_compiler_init").entered();
        info!("Initializing JIT security compiler");

        // Create runtime security system integration
        let runtime_security = Arc::new(
            SecuritySystem::new().map_err(|e| PrismVMError::SecurityError {
                message: format!("Failed to create runtime security system: {}", e),
            })?
        );

        // Create effects security system integration
        let effects_security = Arc::new(RwLock::new(EffectSecuritySystem::new()));

        // Create effect validator
        let effect_validator = Arc::new(EffectValidator::new());

        Ok(Self {
            config,
            runtime_security,
            effects_security,
            effect_validator,
            policy_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SecurityStats::default())),
        })
    }

    /// Validate bytecode security before JIT compilation
    pub fn validate_bytecode_security(
        &self,
        bytecode: &PrismBytecode,
        capabilities: &CapabilitySet,
    ) -> VMResult<Vec<SecurityViolation>> {
        let _span = span!(Level::DEBUG, "validate_bytecode_security").entered();
        let start_time = Instant::now();

        if !self.config.enable_validation {
            return Ok(Vec::new());
        }

        debug!("Validating bytecode security for JIT compilation");

        let mut violations = Vec::new();

        // Basic bytecode validation using existing infrastructure
        if let Err(e) = bytecode.validate() {
            violations.push(SecurityViolation {
                violation_type: ViolationType::InvalidBytecode,
                message: format!("Bytecode validation failed: {}", e),
                function_name: "module".to_string(),
                instruction_offset: None,
                severity: Severity::Error,
            });
        }

        // Validate each function
        for function in &bytecode.functions {
            violations.extend(self.validate_function_security(function, capabilities)?);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.validations_performed += 1;
            stats.violations_detected += violations.len() as u64;
            stats.total_security_time += start_time.elapsed();
        }

        debug!("Security validation completed with {} violations", violations.len());
        Ok(violations)
    }

    /// Validate security for a single function
    fn validate_function_security(
        &self,
        function: &FunctionDefinition,
        capabilities: &CapabilitySet,
    ) -> VMResult<Vec<SecurityViolation>> {
        let mut violations = Vec::new();

        // Check if function requires capabilities that aren't available
        for (offset, instruction) in function.instructions.iter().enumerate() {
            for required_cap in &instruction.required_capabilities {
                if !capabilities.iter().any(|cap| cap.name() == required_cap) {
                    violations.push(SecurityViolation {
                        violation_type: ViolationType::CapabilityViolation,
                        message: format!("Instruction requires unavailable capability: {}", required_cap),
                        function_name: function.name.clone(),
                        instruction_offset: Some(offset as u32),
                        severity: Severity::Error,
                    });
                }
            }

            // Validate effects are properly declared
            for effect in &instruction.effects {
                if !self.is_effect_properly_declared(effect, function) {
                    violations.push(SecurityViolation {
                        violation_type: ViolationType::UndeclaredEffect,
                        message: format!("Effect not properly declared: {:?}", effect),
                        function_name: function.name.clone(),
                        instruction_offset: Some(offset as u32),
                        severity: Severity::Warning,
                    });
                }
            }
        }

        Ok(violations)
    }

    /// Check if an effect is properly declared in the function signature
    fn is_effect_properly_declared(&self, effect: &Effect, function: &FunctionDefinition) -> bool {
        // Use existing effect validation infrastructure
        // This is a simplified check - in practice would use the effect system
        function.declared_effects.iter().any(|declared| {
            declared.name() == effect.name()
        })
    }

    /// Generate capability checks for JIT-compiled code
    pub fn generate_capability_checks(
        &self,
        function: &FunctionDefinition,
        capabilities: &CapabilitySet,
    ) -> VMResult<CapabilityChecker> {
        let _span = span!(Level::DEBUG, "generate_capability_checks", function = %function.name).entered();

        // ENHANCED: Always enable capability checks for security consistency
        // Runtime verification must match compile-time verification completeness
        debug!("Generating capability checks for function: {}", function.name);

        let mut required_capabilities = HashSet::new();
        let mut check_points = Vec::new();

        // ENHANCED: Check function-level capabilities
        for required_cap in &function.capabilities {
            required_capabilities.insert(required_cap.clone());
            
            // Insert capability check at function entry
            check_points.push(CheckPoint {
                function_id: function.id,
                instruction_offset: 0, // Function entry
                required_capability: required_cap.clone(),
                check_type: CheckType::FunctionEntry,
            });
        }

        // ENHANCED: Check effect-related capabilities
        for effect in &function.effects {
            if let Some(required_cap) = effect.required_capability() {
                required_capabilities.insert(required_cap.clone());
                
                // Insert capability check for effect
                check_points.push(CheckPoint {
                    function_id: function.id,
                    instruction_offset: 0, // Function entry for declared effects
                    required_capability: required_cap,
                    check_type: CheckType::EffectCapability,
                });
            }
        }

        // Analyze instructions to determine where capability checks are needed
        for (offset, instruction) in function.instructions.iter().enumerate() {
            for required_cap in &instruction.required_capabilities {
                required_capabilities.insert(required_cap.clone());
                
                // Insert capability check before instruction
                check_points.push(CheckPoint {
                    function_id: function.id,
                    instruction_offset: offset as u32,
                    required_capability: required_cap.clone(),
                    check_type: CheckType::Presence,
                });
            }

            // ENHANCED: Check capabilities for instruction effects
            for effect in &instruction.effects {
                if let Some(required_cap) = effect.required_capability() {
                    required_capabilities.insert(required_cap.clone());
                    
                    // Insert capability check before instruction with effect
                    check_points.push(CheckPoint {
                        function_id: function.id,
                        instruction_offset: offset as u32,
                        required_capability: required_cap,
                        check_type: CheckType::InstructionEffect,
                    });
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.capability_checks_inserted += check_points.len() as u64;
        }

        debug!("Generated {} capability checks for {} required capabilities", 
               check_points.len(), required_capabilities.len());

        Ok(CapabilityChecker {
            required_capabilities,
            available_capabilities: Arc::new(capabilities.clone()),
            check_points,
        })
    }

    /// Generate effect tracking for JIT-compiled code
    pub fn generate_effect_tracking(
        &self,
        function: &FunctionDefinition,
    ) -> VMResult<EffectTracker> {
        let _span = span!(Level::DEBUG, "generate_effect_tracking", function = %function.name).entered();

        if !self.config.enable_effect_tracking {
            return Ok(EffectTracker {
                discovered_effects: Vec::new(),
                tracking_points: Vec::new(),
                validation_results: Vec::new(),
            });
        }

        debug!("Generating effect tracking for function: {}", function.name);

        let mut discovered_effects = Vec::new();
        let mut tracking_points = Vec::new();

        // Analyze instructions to discover effects and generate tracking points
        for (offset, instruction) in function.instructions.iter().enumerate() {
            for effect in &instruction.effects {
                discovered_effects.push(effect.clone());
                
                // Insert tracking point before effect
                tracking_points.push(TrackingPoint {
                    function_id: function.id,
                    instruction_offset: offset as u32,
                    effect: effect.clone(),
                    tracking_type: TrackingType::Start,
                });
            }
        }

        // Validate effects using existing effect validation infrastructure
        let mut validation_context = ValidationContext::new();
        let validation_results = match self.effect_validator.validate_effects(
            &discovered_effects,
            &[], // No capabilities provided for this validation
            &prism_effects::capability::CapabilityManager::new(),
        ) {
            Ok(()) => Vec::new(),
            Err(_) => validation_context.violations,
        };

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.effect_tracking_points += tracking_points.len() as u64;
        }

        debug!("Generated {} effect tracking points", tracking_points.len());

        Ok(EffectTracker {
            discovered_effects,
            tracking_points,
            validation_results,
        })
    }

    /// Validate generated machine code for security
    pub fn validate_generated_code(
        &self,
        machine_code: &[u8],
        function: &FunctionDefinition,
    ) -> VMResult<Vec<SecurityViolation>> {
        let _span = span!(Level::DEBUG, "validate_generated_code", function = %function.name).entered();

        if !self.config.enable_validation {
            return Ok(Vec::new());
        }

        debug!("Validating generated machine code for function: {}", function.name);

        let mut violations = Vec::new();

        // Basic validation - check for obvious security issues
        // In a complete implementation, this would use more sophisticated analysis
        
        // Check code size limits
        if machine_code.len() > 1024 * 1024 { // 1MB limit
            violations.push(SecurityViolation {
                violation_type: ViolationType::ResourceLimitExceeded,
                message: "Generated code exceeds size limit".to_string(),
                function_name: function.name.clone(),
                instruction_offset: None,
                severity: Severity::Error,
            });
        }

        // Check for potentially dangerous instruction patterns
        // This is a simplified check - real implementation would be more sophisticated
        if machine_code.windows(4).any(|window| {
            // Check for system call patterns (simplified)
            window == [0x0f, 0x05, 0x00, 0x00] // syscall instruction pattern
        }) {
            violations.push(SecurityViolation {
                violation_type: ViolationType::UnsafeOperation,
                message: "Generated code contains potentially unsafe system calls".to_string(),
                function_name: function.name.clone(),
                instruction_offset: None,
                severity: Severity::Warning,
            });
        }

        debug!("Machine code validation completed with {} violations", violations.len());
        Ok(violations)
    }

    /// Enforce security policies during compilation
    pub fn enforce_compilation_policies(
        &self,
        function: &FunctionDefinition,
        capabilities: &CapabilitySet,
    ) -> VMResult<PolicyDecision> {
        let _span = span!(Level::DEBUG, "enforce_compilation_policies", function = %function.name).entered();

        if !self.config.enable_policy_enforcement {
            return Ok(PolicyDecision::Allow);
        }

        debug!("Enforcing security policies for function compilation: {}", function.name);

        // Check policy cache first
        let cache_key = format!("{}:{}", function.name, capabilities.active_count());
        if self.config.enable_policy_caching {
            if let Some(cached_decision) = self.policy_cache.read().unwrap().get(&cache_key) {
                let mut stats = self.stats.write().unwrap();
                stats.policy_cache_hit_rate = 
                    (stats.policy_cache_hit_rate * (stats.validations_performed as f64) + 1.0) / 
                    ((stats.validations_performed + 1) as f64);
                return Ok(cached_decision.clone());
            }
        }

        // Create security context for policy enforcement
        let security_context = prism_runtime::security::SecurityContext {
            component_id: prism_runtime::authority::ComponentId::new(),
            security_level: self.config.security_level,
            operation: format!("jit_compile:{}", function.name),
            context_data: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };

        // Use runtime security system for policy enforcement
        // This integrates with existing security infrastructure
        let policy = prism_runtime::security::SecurityPolicy {
            name: "JIT Compilation Policy".to_string(),
            description: "Security policy for JIT compilation operations".to_string(),
            policy_type: prism_runtime::security::PolicyType::AccessControl,
            rules: vec![
                prism_runtime::security::SecurityRule {
                    name: "Capability Validation".to_string(),
                    description: "Ensure required capabilities are available".to_string(),
                    rule_type: prism_runtime::security::RuleType::ComponentValidation,
                    condition: "capabilities_available".to_string(),
                    action: prism_runtime::security::RuleAction::Allow,
                }
            ],
            created_at: std::time::SystemTime::now(),
            modified_at: std::time::SystemTime::now(),
        };

        let decision = self.runtime_security
            .enforce_policy(&policy, &security_context)
            .map_err(|e| PrismVMError::SecurityError {
                message: format!("Policy enforcement failed: {}", e),
            })?;

        // Cache the decision
        if self.config.enable_policy_caching {
            self.policy_cache.write().unwrap().insert(cache_key, decision.clone());
        }

        debug!("Policy enforcement result: {:?}", decision);
        Ok(decision)
    }

    /// Get security statistics
    pub fn get_stats(&self) -> SecurityStats {
        self.stats.read().unwrap().clone()
    }

    /// Create a secure execution context for JIT operations
    pub fn create_secure_context(
        &self,
        capabilities: &CapabilitySet,
    ) -> VMResult<SecureExecutionContext> {
        let _span = span!(Level::DEBUG, "create_secure_context").entered();

        debug!("Creating secure execution context for JIT operations");

        // Convert runtime capabilities to effects capabilities
        let effect_capabilities: Vec<prism_effects::security::Capability> = capabilities
            .iter()
            .map(|cap| prism_effects::security::Capability {
                definition: cap.name().to_string(),
                constraints: prism_effects::capability::CapabilityConstraints::new(),
                revoked: !cap.is_valid(),
                metadata: prism_effects::capability::CapabilityMetadata::default(),
            })
            .collect();

        // Create trust context
        let trust_context = prism_effects::security::trust::TrustContext {
            trust_level: prism_effects::security::trust::TrustLevel::Basic,
            established_at: std::time::Instant::now(),
            attestations: Vec::new(),
        };

        // Create secure execution context using effects security system
        let context = SecureExecutionContext::new(
            "jit_compilation".to_string(),
            prism_effects::security::SecurityLevel::Internal,
            trust_context,
        );

        debug!("Secure execution context created");
        Ok(context)
    }
}

impl Default for SecurityCompiler {
    fn default() -> Self {
        Self::new_default().expect("Failed to create default security compiler")
    }
}

// Implement display for security violation types
impl std::fmt::Display for ViolationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationType::CapabilityViolation => write!(f, "capability_violation"),
            ViolationType::UndeclaredEffect => write!(f, "undeclared_effect"),
            ViolationType::PolicyViolation => write!(f, "policy_violation"),
            ViolationType::UnsafeOperation => write!(f, "unsafe_operation"),
            ViolationType::ResourceLimitExceeded => write!(f, "resource_limit_exceeded"),
            ViolationType::InvalidBytecode => write!(f, "invalid_bytecode"),
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
            Severity::Critical => write!(f, "critical"),
        }
    }
} 