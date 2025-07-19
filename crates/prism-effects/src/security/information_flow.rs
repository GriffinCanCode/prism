//! Information Flow Control
//!
//! Implementation of information flow control using security lattices

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use thiserror::Error;

/// Information flow control system
#[derive(Debug)]
pub struct InformationFlowControl {
    /// Security lattice for flow control
    pub lattice: SecurityLattice,
    /// Active information flows
    pub active_flows: Vec<InformationFlow>,
    /// Flow policies
    pub policies: Vec<FlowPolicy>,
    /// Violation counter
    violation_count: usize,
}

impl InformationFlowControl {
    /// Create new information flow control system
    pub fn new() -> Self {
        let mut system = Self {
            lattice: SecurityLattice::new(),
            active_flows: Vec::new(),
            policies: Vec::new(),
            violation_count: 0,
        };
        system.initialize_default_levels();
        system
    }

    /// Initialize default security levels
    fn initialize_default_levels(&mut self) {
        // Create standard security levels
        let public = SecurityLevel::new("Public".to_string(), 0, vec![], vec![]);
        let confidential = SecurityLevel::new("Confidential".to_string(), 1, vec!["Public".to_string()], vec![]);
        let secret = SecurityLevel::new("Secret".to_string(), 2, vec!["Confidential".to_string()], vec![]);
        let top_secret = SecurityLevel::new("TopSecret".to_string(), 3, vec!["Secret".to_string()], vec![]);

        self.lattice.add_level(public);
        self.lattice.add_level(confidential);
        self.lattice.add_level(secret);
        self.lattice.add_level(top_secret);
    }

    /// Validate an information flow
    pub fn validate_flow(
        &mut self,
        flows: &[InformationFlow],
        context_level: &SecurityLevel,
    ) -> Result<FlowValidationResult, FlowError> {
        let mut valid = true;
        let mut violations = Vec::new();

        for flow in flows {
            // Check if flow is allowed by lattice
            let can_flow = self.lattice.can_flow(&flow.source_level, &flow.target_level)?;
            
            if !can_flow {
                valid = false;
                violations.push(FlowViolation {
                    flow: flow.clone(),
                    reason: format!(
                        "Information cannot flow from {} to {}",
                        flow.source_level.name, flow.target_level.name
                    ),
                    severity: ViolationSeverity::High,
                });
                self.violation_count += 1;
            }

            // Check context level compatibility
            if !self.lattice.can_flow(context_level, &flow.source_level)? {
                valid = false;
                violations.push(FlowViolation {
                    flow: flow.clone(),
                    reason: format!(
                        "Context level {} cannot access source level {}",
                        context_level.name, flow.source_level.name
                    ),
                    severity: ViolationSeverity::Critical,
                });
                self.violation_count += 1;
            }

            // Check policies
            for policy in &self.policies {
                if let Some(violation) = policy.check_flow(flow)? {
                    valid = false;
                    violations.push(violation);
                    self.violation_count += 1;
                }
            }
        }

        Ok(FlowValidationResult {
            valid,
            violations,
            flows_checked: flows.len(),
        })
    }

    /// Add a new security level
    pub fn add_security_level(&mut self, level: SecurityLevel) -> Result<(), FlowError> {
        self.lattice.add_level(level);
        Ok(())
    }

    /// Add a flow policy
    pub fn add_policy(&mut self, policy: FlowPolicy) {
        self.policies.push(policy);
    }

    /// Get violation count
    pub fn get_violation_count(&self) -> usize {
        self.violation_count
    }

    /// Compute least upper bound of security levels
    pub fn lub(&self, level1: &SecurityLevel, level2: &SecurityLevel) -> Result<SecurityLevel, FlowError> {
        self.lattice.lub(level1, level2)
    }

    /// Compute greatest lower bound of security levels
    pub fn glb(&self, level1: &SecurityLevel, level2: &SecurityLevel) -> Result<SecurityLevel, FlowError> {
        self.lattice.glb(level1, level2)
    }
}

impl Default for InformationFlowControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Security lattice for information flow control
#[derive(Debug)]
pub struct SecurityLattice {
    /// Security levels in the lattice
    pub levels: HashMap<String, SecurityLevel>,
    /// Computed flow relationships (cached)
    flow_cache: HashMap<(String, String), bool>,
}

impl SecurityLattice {
    /// Create new security lattice
    pub fn new() -> Self {
        Self {
            levels: HashMap::new(),
            flow_cache: HashMap::new(),
        }
    }

    /// Add a security level to the lattice
    pub fn add_level(&mut self, level: SecurityLevel) {
        self.levels.insert(level.name.clone(), level);
        // Clear cache when structure changes
        self.flow_cache.clear();
    }

    /// Check if information can flow from source to target
    pub fn can_flow(&mut self, source: &SecurityLevel, target: &SecurityLevel) -> Result<bool, FlowError> {
        let cache_key = (source.name.clone(), target.name.clone());
        
        if let Some(&cached_result) = self.flow_cache.get(&cache_key) {
            return Ok(cached_result);
        }

        let result = self.compute_flow(source, target)?;
        self.flow_cache.insert(cache_key, result);
        Ok(result)
    }

    /// Compute flow relationship between two levels
    fn compute_flow(&self, source: &SecurityLevel, target: &SecurityLevel) -> Result<bool, FlowError> {
        // Information can flow "up" the lattice (to higher security levels)
        // This implements the standard "no read up, no write down" policy
        
        if source.name == target.name {
            return Ok(true); // Same level
        }

        // Check if target dominates source (source ⊑ target)
        self.dominates(target, source)
    }

    /// Check if level1 dominates level2 (level2 ⊑ level1)
    fn dominates(&self, level1: &SecurityLevel, level2: &SecurityLevel) -> Result<bool, FlowError> {
        if level1.name == level2.name {
            return Ok(true);
        }

        // Use numerical levels as primary comparison
        if level1.numerical_level >= level2.numerical_level {
            // Also check hierarchical relationships
            self.is_ancestor(level1, level2)
        } else {
            Ok(false)
        }
    }

    /// Check if level1 is an ancestor of level2 in the hierarchy
    fn is_ancestor(&self, level1: &SecurityLevel, level2: &SecurityLevel) -> Result<bool, FlowError> {
        // Check if level1 is reachable by following parent relationships from level2
        let mut visited = HashSet::new();
        let mut to_visit = vec![level2.name.clone()];

        while let Some(current_name) = to_visit.pop() {
            if visited.contains(&current_name) {
                continue; // Avoid cycles
            }
            visited.insert(current_name.clone());

            if current_name == level1.name {
                return Ok(true);
            }

            if let Some(current_level) = self.levels.get(&current_name) {
                to_visit.extend(current_level.parents.iter().cloned());
            }
        }

        Ok(false)
    }

    /// Compute least upper bound (join) of two security levels
    pub fn lub(&self, level1: &SecurityLevel, level2: &SecurityLevel) -> Result<SecurityLevel, FlowError> {
        if level1.name == level2.name {
            return Ok(level1.clone());
        }

        // Find the lowest level that dominates both level1 and level2
        let mut candidates = Vec::new();
        
        for (_, level) in &self.levels {
            if self.dominates(level, level1)? && self.dominates(level, level2)? {
                candidates.push(level);
            }
        }

        // Find the minimum among candidates
        candidates.into_iter()
            .min_by_key(|level| level.numerical_level)
            .cloned()
            .ok_or(FlowError::NoLubExists(level1.name.clone(), level2.name.clone()))
    }

    /// Compute greatest lower bound (meet) of two security levels
    pub fn glb(&self, level1: &SecurityLevel, level2: &SecurityLevel) -> Result<SecurityLevel, FlowError> {
        if level1.name == level2.name {
            return Ok(level1.clone());
        }

        // Find the highest level that is dominated by both level1 and level2
        let mut candidates = Vec::new();
        
        for (_, level) in &self.levels {
            if self.dominates(level1, level)? && self.dominates(level2, level)? {
                candidates.push(level);
            }
        }

        // Find the maximum among candidates
        candidates.into_iter()
            .max_by_key(|level| level.numerical_level)
            .cloned()
            .ok_or(FlowError::NoGlbExists(level1.name.clone(), level2.name.clone()))
    }
}

impl Default for SecurityLattice {
    fn default() -> Self {
        Self::new()
    }
}

/// A security level in the lattice
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityLevel {
    /// Name of the security level
    pub name: String,
    /// Numerical level for ordering
    pub numerical_level: u8,
    /// Parent levels in the hierarchy
    pub parents: Vec<String>,
    /// Child levels in the hierarchy
    pub children: Vec<String>,
}

impl SecurityLevel {
    /// Create a new security level
    pub fn new(name: String, numerical_level: u8, parents: Vec<String>, children: Vec<String>) -> Self {
        Self {
            name,
            numerical_level,
            parents,
            children,
        }
    }
}

impl PartialOrd for SecurityLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SecurityLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numerical_level.cmp(&other.numerical_level)
    }
}

/// An information flow between security levels
#[derive(Debug, Clone)]
pub struct InformationFlow {
    /// Source security level
    pub source_level: SecurityLevel,
    /// Target security level
    pub target_level: SecurityLevel,
    /// Type of information being flowed
    pub information_type: String,
    /// Flow metadata
    pub metadata: HashMap<String, String>,
}

impl InformationFlow {
    /// Create a new information flow
    pub fn new(
        source_level: SecurityLevel,
        target_level: SecurityLevel,
        information_type: String,
    ) -> Self {
        Self {
            source_level,
            target_level,
            information_type,
            metadata: HashMap::new(),
        }
    }
}

/// Policy for controlling information flows
#[derive(Debug)]
pub struct FlowPolicy {
    /// Policy name
    pub name: String,
    /// Policy rules
    pub rules: Vec<FlowRule>,
}

impl FlowPolicy {
    /// Check if a flow violates this policy
    pub fn check_flow(&self, flow: &InformationFlow) -> Result<Option<FlowViolation>, FlowError> {
        for rule in &self.rules {
            if let Some(violation) = rule.check(flow)? {
                return Ok(Some(FlowViolation {
                    flow: flow.clone(),
                    reason: violation,
                    severity: ViolationSeverity::Medium,
                }));
            }
        }
        Ok(None)
    }
}

/// A flow rule within a policy
#[derive(Debug)]
pub enum FlowRule {
    /// Deny flows of specific information types
    DenyInformationType(String),
    /// Require specific metadata for flows
    RequireMetadata(String, String),
    /// Deny flows between specific levels
    DenyLevelPair(String, String),
}

impl FlowRule {
    /// Check if a flow violates this rule
    pub fn check(&self, flow: &InformationFlow) -> Result<Option<String>, FlowError> {
        match self {
            FlowRule::DenyInformationType(info_type) => {
                if &flow.information_type == info_type {
                    Ok(Some(format!("Information type '{}' is denied by policy", info_type)))
                } else {
                    Ok(None)
                }
            },
            FlowRule::RequireMetadata(key, expected_value) => {
                match flow.metadata.get(key) {
                    Some(actual_value) if actual_value == expected_value => Ok(None),
                    Some(_) => Ok(Some(format!("Metadata '{}' has incorrect value", key))),
                    None => Ok(Some(format!("Required metadata '{}' is missing", key))),
                }
            },
            FlowRule::DenyLevelPair(source, target) => {
                if &flow.source_level.name == source && &flow.target_level.name == target {
                    Ok(Some(format!("Flow from '{}' to '{}' is denied by policy", source, target)))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// Result of flow validation
#[derive(Debug)]
pub struct FlowValidationResult {
    /// Whether all flows are valid
    pub valid: bool,
    /// Any violations found
    pub violations: Vec<FlowViolation>,
    /// Number of flows checked
    pub flows_checked: usize,
}

/// A flow violation
#[derive(Debug)]
pub struct FlowViolation {
    /// The flow that caused the violation
    pub flow: InformationFlow,
    /// Reason for the violation
    pub reason: String,
    /// Severity of the violation
    pub severity: ViolationSeverity,
}

/// Severity levels for violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Errors in information flow control
#[derive(Debug, Error)]
pub enum FlowError {
    #[error("Security level not found: {0}")]
    LevelNotFound(String),
    
    #[error("No least upper bound exists for levels {0} and {1}")]
    NoLubExists(String, String),
    
    #[error("No greatest lower bound exists for levels {0} and {1}")]
    NoGlbExists(String, String),
    
    #[error("Circular dependency in security lattice")]
    CircularDependency,
    
    #[error("Invalid flow policy: {0}")]
    InvalidPolicy(String),
} 