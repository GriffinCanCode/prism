//! Scope Visibility and Access Control
//!
//! This module defines the visibility rules and access control mechanisms
//! for scopes, following PLT-004 specifications and integrating with
//! PLD-003's capability-based security.
//!
//! **Conceptual Responsibility**: Scope visibility and access control
//! **What it does**: Define visibility rules, access levels, permission checking
//! **What it doesn't do**: Scope hierarchy management, symbol resolution, effect tracking

use serde::{Serialize, Deserialize};

/// Scope visibility levels following PLT-004 specification
/// 
/// Inspired by Rust's visibility system and Swift's access levels,
/// enhanced for AI comprehension and capability-based security.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScopeVisibility {
    /// Public to all (equivalent to Rust's `pub`)
    Public,
    
    /// Public within current module (equivalent to Rust's `pub(crate)`)
    Module,
    
    /// Public within current section (PLD-002 integration)
    Section,
    
    /// Private to current scope
    Private,
    
    /// Internal to implementation (not part of public API)
    Internal,
    
    /// Restricted to specific capabilities (PLD-003 integration)
    Capability { 
        /// Required capabilities to access this scope
        required_capabilities: Vec<String>,
        /// Security classification level
        security_level: SecurityLevel,
    },
    
    /// Restricted to specific modules
    RestrictedModule { 
        /// Modules allowed to access this scope
        allowed_modules: Vec<String> 
    },
    
    /// Friend visibility (specific scopes can access)
    Friend { 
        /// Friend scope IDs that can access this scope
        friend_scopes: Vec<crate::scope::ScopeId> 
    },
}

/// Security classification levels for capability-based visibility
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Normal security level
    Normal,
    /// Elevated security level (requires justification)
    Elevated,
    /// Controlled security level (strict capability checking)
    Controlled,
    /// Critical security level (maximum restrictions)
    Critical,
}

/// Access level for different operations within a scope
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// No access allowed
    None,
    
    /// Read-only access
    Read,
    
    /// Read and write access
    ReadWrite,
    
    /// Full access (read, write, execute)
    Full,
    
    /// Administrative access (can modify visibility rules)
    Admin,
}

/// Visibility rule for specific operations or contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibilityRule {
    /// The visibility level
    pub visibility: ScopeVisibility,
    
    /// Context where this rule applies
    pub context: VisibilityContext,
    
    /// Access level granted by this rule
    pub access_level: AccessLevel,
    
    /// Optional conditions for this rule
    pub conditions: Vec<VisibilityCondition>,
    
    /// Documentation explaining this visibility rule
    pub documentation: Option<String>,
}

/// Context where a visibility rule applies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisibilityContext {
    /// Rule applies to symbol access
    SymbolAccess,
    
    /// Rule applies to scope traversal
    ScopeTraversal,
    
    /// Rule applies to import/export operations
    ImportExport,
    
    /// Rule applies to metadata access
    MetadataAccess,
    
    /// Rule applies to effect system operations
    EffectOperations,
    
    /// Rule applies to capability operations
    CapabilityOperations,
    
    /// Custom context
    Custom(String),
}

/// Condition that must be met for a visibility rule to apply
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisibilityCondition {
    /// Must have specific capability
    HasCapability(String),
    
    /// Must be in specific module
    InModule(String),
    
    /// Must be at specific scope depth or less
    MaxDepth(usize),
    
    /// Must be during specific compilation phase
    CompilationPhase(String),
    
    /// Must meet security classification
    SecurityClassification(SecurityLevel),
    
    /// Custom condition with predicate
    Custom {
        name: String,
        description: String,
    },
}

/// Access control result for visibility checks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessResult {
    /// Access is allowed with the given level
    Allowed(AccessLevel),
    
    /// Access is denied with reason
    Denied(AccessDenialReason),
    
    /// Access requires additional capabilities
    RequiresCapabilities(Vec<String>),
    
    /// Access is conditionally allowed (must check conditions)
    Conditional(Vec<VisibilityCondition>),
}

/// Reason why access was denied
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessDenialReason {
    /// Insufficient visibility level
    InsufficientVisibility,
    
    /// Missing required capabilities
    MissingCapabilities(Vec<String>),
    
    /// Security level too low
    InsufficientSecurityLevel,
    
    /// Not in allowed modules
    ModuleRestriction,
    
    /// Scope depth exceeded
    DepthRestriction,
    
    /// Custom denial reason
    Custom(String),
}

impl ScopeVisibility {
    /// Check if this visibility allows access from another scope
    pub fn allows_access_from(&self, from_scope: crate::scope::ScopeId, context: &VisibilityContext) -> AccessResult {
        match self {
            ScopeVisibility::Public => AccessResult::Allowed(AccessLevel::Read),
            ScopeVisibility::Private => AccessResult::Denied(AccessDenialReason::InsufficientVisibility),
            ScopeVisibility::Internal => AccessResult::Denied(AccessDenialReason::InsufficientVisibility),
            
            ScopeVisibility::Module => {
                // This would require module context to determine properly
                // For now, return conditional access
                AccessResult::Conditional(vec![VisibilityCondition::InModule("same_module".to_string())])
            }
            
            ScopeVisibility::Section => {
                // This would require section context to determine properly
                AccessResult::Conditional(vec![VisibilityCondition::Custom {
                    name: "same_section".to_string(),
                    description: "Must be in the same module section".to_string(),
                }])
            }
            
            ScopeVisibility::Capability { required_capabilities, security_level } => {
                if required_capabilities.is_empty() {
                    AccessResult::Allowed(self.access_level_for_security(*security_level))
                } else {
                    AccessResult::RequiresCapabilities(required_capabilities.clone())
                }
            }
            
            ScopeVisibility::RestrictedModule { allowed_modules: _ } => {
                // This would require module context to determine properly
                AccessResult::Conditional(vec![VisibilityCondition::InModule("allowed_module".to_string())])
            }
            
            ScopeVisibility::Friend { friend_scopes } => {
                if friend_scopes.contains(&from_scope) {
                    AccessResult::Allowed(AccessLevel::ReadWrite)
                } else {
                    AccessResult::Denied(AccessDenialReason::InsufficientVisibility)
                }
            }
        }
    }
    
    /// Get the default access level for a security level
    fn access_level_for_security(&self, security_level: SecurityLevel) -> AccessLevel {
        match security_level {
            SecurityLevel::Normal => AccessLevel::ReadWrite,
            SecurityLevel::Elevated => AccessLevel::Read,
            SecurityLevel::Controlled => AccessLevel::Read,
            SecurityLevel::Critical => AccessLevel::None,
        }
    }
    
    /// Check if this visibility is more restrictive than another
    pub fn is_more_restrictive_than(&self, other: &ScopeVisibility) -> bool {
        let self_level = self.restrictiveness_level();
        let other_level = other.restrictiveness_level();
        self_level > other_level
    }
    
    /// Get a numeric level representing how restrictive this visibility is
    fn restrictiveness_level(&self) -> u8 {
        match self {
            ScopeVisibility::Public => 0,
            ScopeVisibility::Module => 1,
            ScopeVisibility::Section => 2,
            ScopeVisibility::RestrictedModule { .. } => 3,
            ScopeVisibility::Friend { .. } => 4,
            ScopeVisibility::Capability { security_level, .. } => {
                match security_level {
                    SecurityLevel::Normal => 5,
                    SecurityLevel::Elevated => 6,
                    SecurityLevel::Controlled => 7,
                    SecurityLevel::Critical => 8,
                }
            }
            ScopeVisibility::Internal => 9,
            ScopeVisibility::Private => 10,
        }
    }
    
    /// Get a human-readable description of this visibility
    pub fn description(&self) -> String {
        match self {
            ScopeVisibility::Public => "Public to all".to_string(),
            ScopeVisibility::Module => "Public within current module".to_string(),
            ScopeVisibility::Section => "Public within current section".to_string(),
            ScopeVisibility::Private => "Private to current scope".to_string(),
            ScopeVisibility::Internal => "Internal implementation detail".to_string(),
            ScopeVisibility::Capability { required_capabilities, security_level } => {
                format!("Requires capabilities: {:?} (Security: {:?})", required_capabilities, security_level)
            }
            ScopeVisibility::RestrictedModule { allowed_modules } => {
                format!("Restricted to modules: {:?}", allowed_modules)
            }
            ScopeVisibility::Friend { friend_scopes } => {
                format!("Friend access to {} scopes", friend_scopes.len())
            }
        }
    }
}

impl VisibilityRule {
    /// Create a new visibility rule
    pub fn new(visibility: ScopeVisibility, context: VisibilityContext, access_level: AccessLevel) -> Self {
        Self {
            visibility,
            context,
            access_level,
            conditions: Vec::new(),
            documentation: None,
        }
    }
    
    /// Add a condition to this visibility rule
    pub fn with_condition(mut self, condition: VisibilityCondition) -> Self {
        self.conditions.push(condition);
        self
    }
    
    /// Add documentation to this visibility rule
    pub fn with_documentation(mut self, doc: String) -> Self {
        self.documentation = Some(doc);
        self
    }
    
    /// Check if this rule applies in the given context
    pub fn applies_to_context(&self, context: &VisibilityContext) -> bool {
        &self.context == context || matches!(self.context, VisibilityContext::Custom(_))
    }
    
    /// Evaluate this rule for a specific access request
    pub fn evaluate(&self, _from_scope: crate::scope::ScopeId, _available_capabilities: &[String]) -> AccessResult {
        // This would contain the full evaluation logic
        // For now, return a simple result based on visibility
        self.visibility.allows_access_from(_from_scope, &self.context)
    }
}

impl Default for ScopeVisibility {
    fn default() -> Self {
        ScopeVisibility::Private
    }
}

impl Default for AccessLevel {
    fn default() -> Self {
        AccessLevel::None
    }
}

/// Helper trait for objects that have visibility rules
pub trait HasVisibility {
    /// Get the visibility for this object
    fn visibility(&self) -> &ScopeVisibility;
    
    /// Set the visibility for this object
    fn set_visibility(&mut self, visibility: ScopeVisibility);
    
    /// Check if this object is accessible from a given scope
    fn is_accessible_from(&self, from_scope: crate::scope::ScopeId, context: &VisibilityContext) -> AccessResult {
        self.visibility().allows_access_from(from_scope, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visibility_restrictiveness() {
        let public = ScopeVisibility::Public;
        let private = ScopeVisibility::Private;
        let module = ScopeVisibility::Module;
        
        assert!(private.is_more_restrictive_than(&public));
        assert!(private.is_more_restrictive_than(&module));
        assert!(module.is_more_restrictive_than(&public));
        assert!(!public.is_more_restrictive_than(&private));
    }
    
    #[test]
    fn test_access_results() {
        let public = ScopeVisibility::Public;
        let private = ScopeVisibility::Private;
        
        let context = VisibilityContext::SymbolAccess;
        
        match public.allows_access_from(1, &context) {
            AccessResult::Allowed(level) => assert_eq!(level, AccessLevel::Read),
            _ => panic!("Public visibility should allow access"),
        }
        
        match private.allows_access_from(1, &context) {
            AccessResult::Denied(reason) => assert_eq!(reason, AccessDenialReason::InsufficientVisibility),
            _ => panic!("Private visibility should deny access"),
        }
    }
    
    #[test]
    fn test_capability_visibility() {
        let cap_vis = ScopeVisibility::Capability {
            required_capabilities: vec!["read".to_string(), "write".to_string()],
            security_level: SecurityLevel::Normal,
        };
        
        let context = VisibilityContext::SymbolAccess;
        
        match cap_vis.allows_access_from(1, &context) {
            AccessResult::RequiresCapabilities(caps) => {
                assert_eq!(caps.len(), 2);
                assert!(caps.contains(&"read".to_string()));
                assert!(caps.contains(&"write".to_string()));
            }
            _ => panic!("Capability visibility should require capabilities"),
        }
    }
    
    #[test]
    fn test_friend_visibility() {
        let friend_vis = ScopeVisibility::Friend {
            friend_scopes: vec![1, 2, 3],
        };
        
        let context = VisibilityContext::SymbolAccess;
        
        // Friend scope should have access
        match friend_vis.allows_access_from(2, &context) {
            AccessResult::Allowed(level) => assert_eq!(level, AccessLevel::ReadWrite),
            _ => panic!("Friend scope should have access"),
        }
        
        // Non-friend scope should be denied
        match friend_vis.allows_access_from(5, &context) {
            AccessResult::Denied(_) => (),
            _ => panic!("Non-friend scope should be denied"),
        }
    }
    
    #[test]
    fn test_visibility_rule() {
        let rule = VisibilityRule::new(
            ScopeVisibility::Public,
            VisibilityContext::SymbolAccess,
            AccessLevel::Read,
        )
        .with_condition(VisibilityCondition::HasCapability("read".to_string()))
        .with_documentation("Public read access with read capability".to_string());
        
        assert!(rule.applies_to_context(&VisibilityContext::SymbolAccess));
        assert!(!rule.applies_to_context(&VisibilityContext::EffectOperations));
        assert!(rule.documentation.is_some());
        assert_eq!(rule.conditions.len(), 1);
    }
} 