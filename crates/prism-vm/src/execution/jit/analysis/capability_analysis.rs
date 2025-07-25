//! Capability Analysis for Security-Aware Optimization
//!
//! This module provides capability flow analysis to ensure optimizations
//! respect security boundaries and capability constraints. It integrates with
//! prism-runtime's capability system and prism-effects' security framework
//! to provide security-aware JIT compilation.
//!
//! ## Research Foundation
//!
//! Based on research from modern JIT compilers and security frameworks:
//! - V8's Sandbox and Control Flow Integrity
//! - SpiderMonkey's security boundaries and capability tracking
//! - Academic research on capability-safe compilation
//! - LLVM's security-aware optimization passes
//!
//! ## Integration Points
//!
//! - prism-runtime::authority::capability for capability management
//! - prism-effects::security for security lattice and information flow
//! - Bytecode instruction capability requirements
//! - JIT optimization constraint enforcement

use crate::{VMResult, PrismVMError, bytecode::{FunctionDefinition, Instruction, PrismOpcode}};
use super::{AnalysisConfig, control_flow::ControlFlowGraph};
use prism_runtime::authority::capability::{CapabilitySet, Capability, CapabilityManager, Operation, CapabilityError};
use prism_effects::security::{SecuritySystem, SecurityOperation, SecureExecutionContext, SecurityLevel, InformationFlow};
use prism_pir::{Effect, Capability as PIRCapability};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Capability analyzer for security-aware JIT optimization
#[derive(Debug)]
pub struct CapabilityAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    
    /// Capability manager for validation
    capability_manager: CapabilityManager,
    
    /// Security system for constraint checking
    security_system: SecuritySystem,
    
    /// Cache for capability flow analysis
    flow_cache: HashMap<u32, CapabilityFlow>,
}

/// Comprehensive capability analysis results
#[derive(Debug, Clone)]
pub struct CapabilityAnalysis {
    /// Function being analyzed
    pub function_id: u32,
    
    /// Capability flow through the function
    pub capability_flow: CapabilityFlow,
    
    /// Security constraints that must be maintained
    pub security_constraints: Vec<SecurityConstraint>,
    
    /// Capability requirements at each instruction
    pub instruction_requirements: HashMap<u32, CapabilityRequirement>,
    
    /// Security boundaries within the function
    pub security_boundaries: Vec<SecurityBoundary>,
    
    /// Information flow constraints
    pub information_flows: Vec<InformationFlowConstraint>,
    
    /// Optimization safety analysis
    pub optimization_safety: OptimizationSafety,
    
    /// Capability propagation graph
    pub propagation_graph: CapabilityPropagationGraph,
}

/// Capability flow analysis through function execution
#[derive(Debug, Clone, Default)]
pub struct CapabilityFlow {
    /// Required capabilities at function entry
    pub entry_capabilities: CapabilitySet,
    
    /// Capability requirements at each instruction offset
    pub instruction_capabilities: HashMap<u32, CapabilitySet>,
    
    /// Capability propagation between instructions
    pub capability_propagation: HashMap<u32, Vec<u32>>,
    
    /// Security boundaries that cannot be crossed
    pub security_boundaries: Vec<SecurityBoundary>,
    
    /// Capability delegation points
    pub delegation_points: Vec<CapabilityDelegation>,
    
    /// Capability revocation points
    pub revocation_points: Vec<CapabilityRevocation>,
}

/// Security constraints for JIT optimization
#[derive(Debug, Clone)]
pub struct SecurityConstraint {
    /// Instruction location
    pub location: u32,
    
    /// Type of security constraint
    pub constraint_type: SecurityConstraintType,
    
    /// Required capabilities for this constraint
    pub required_capabilities: CapabilitySet,
    
    /// Security level required
    pub security_level: SecurityLevel,
    
    /// Constraint description
    pub description: String,
    
    /// Severity of violating this constraint
    pub severity: ConstraintSeverity,
}

/// Types of security constraints
#[derive(Debug, Clone)]
pub enum SecurityConstraintType {
    /// Must have specific capability
    RequiresCapability {
        capability: String,
        reason: String,
    },
    
    /// Must not have specific capability
    ForbidsCapability {
        capability: String,
        reason: String,
    },
    
    /// Cannot cross security boundary
    BoundaryCrossing {
        from_level: SecurityLevel,
        to_level: SecurityLevel,
        reason: String,
    },
    
    /// Information flow restriction
    InformationFlow {
        source_level: SecurityLevel,
        target_level: SecurityLevel,
        reason: String,
    },
    
    /// Effect isolation requirement
    EffectIsolation {
        effect: String,
        isolation_level: IsolationLevel,
        reason: String,
    },
    
    /// Capability delegation constraint
    DelegationConstraint {
        delegator: String,
        delegatee: String,
        capabilities: Vec<String>,
        reason: String,
    },
}

/// Severity levels for constraint violations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintSeverity {
    /// Low severity - optimization hint
    Low,
    /// Medium severity - should avoid but not critical
    Medium,
    /// High severity - must not violate
    High,
    /// Critical severity - security vulnerability if violated
    Critical,
}

/// Capability requirement for an instruction
#[derive(Debug, Clone)]
pub struct CapabilityRequirement {
    /// Instruction offset
    pub instruction_offset: u32,
    
    /// Required capabilities before execution
    pub pre_capabilities: CapabilitySet,
    
    /// Required capabilities after execution
    pub post_capabilities: CapabilitySet,
    
    /// Capabilities consumed by this instruction
    pub consumed_capabilities: CapabilitySet,
    
    /// Capabilities produced by this instruction
    pub produced_capabilities: CapabilitySet,
    
    /// Security effects of this instruction
    pub security_effects: Vec<SecurityEffect>,
}

/// Security boundaries within a function
#[derive(Debug, Clone)]
pub struct SecurityBoundary {
    /// Start instruction offset
    pub start_offset: u32,
    
    /// End instruction offset
    pub end_offset: u32,
    
    /// Type of security boundary
    pub boundary_type: SecurityBoundaryType,
    
    /// Capabilities required to cross this boundary
    pub crossing_capabilities: CapabilitySet,
    
    /// Security level inside the boundary
    pub internal_security_level: SecurityLevel,
    
    /// Security level outside the boundary
    pub external_security_level: SecurityLevel,
    
    /// Restrictions on crossing this boundary
    pub crossing_restrictions: Vec<String>,
}

/// Types of security boundaries
#[derive(Debug, Clone)]
pub enum SecurityBoundaryType {
    /// Trust boundary between different trust levels
    TrustBoundary {
        from_trust: TrustLevel,
        to_trust: TrustLevel,
    },
    
    /// Privilege boundary for capability escalation
    PrivilegeBoundary {
        privilege_change: PrivilegeChange,
    },
    
    /// Isolation boundary for effect containment
    IsolationBoundary {
        isolation_type: IsolationType,
    },
    
    /// Capability boundary for capability management
    CapabilityBoundary {
        capability_operation: CapabilityOperation,
    },
}

/// Information flow constraints for optimization
#[derive(Debug, Clone)]
pub struct InformationFlowConstraint {
    /// Source instruction
    pub source_instruction: u32,
    
    /// Target instruction
    pub target_instruction: u32,
    
    /// Information flow being constrained
    pub flow: InformationFlow,
    
    /// Constraint on this flow
    pub constraint: FlowConstraint,
    
    /// Reason for the constraint
    pub reason: String,
}

/// Flow constraint types
#[derive(Debug, Clone)]
pub enum FlowConstraint {
    /// Flow is allowed
    Allowed,
    
    /// Flow is forbidden
    Forbidden,
    
    /// Flow requires specific capabilities
    RequiresCapabilities(CapabilitySet),
    
    /// Flow must be sanitized
    RequiresSanitization(SanitizationType),
}

/// Optimization safety analysis
#[derive(Debug, Clone)]
pub struct OptimizationSafety {
    /// Safe optimizations that preserve security
    pub safe_optimizations: Vec<SafeOptimization>,
    
    /// Unsafe optimizations that could violate security
    pub unsafe_optimizations: Vec<UnsafeOptimization>,
    
    /// Optimization constraints based on capabilities
    pub optimization_constraints: Vec<OptimizationConstraint>,
    
    /// Security-preserving transformation rules
    pub transformation_rules: Vec<TransformationRule>,
}

/// Safe optimization that preserves security properties
#[derive(Debug, Clone)]
pub struct SafeOptimization {
    /// Optimization type
    pub optimization_type: OptimizationType,
    
    /// Instructions that can be safely optimized
    pub applicable_instructions: Vec<u32>,
    
    /// Security properties preserved
    pub preserved_properties: Vec<SecurityProperty>,
    
    /// Conditions under which this optimization is safe
    pub safety_conditions: Vec<SafetyCondition>,
}

/// Unsafe optimization that could violate security
#[derive(Debug, Clone)]
pub struct UnsafeOptimization {
    /// Optimization type
    pub optimization_type: OptimizationType,
    
    /// Instructions that cannot be optimized
    pub forbidden_instructions: Vec<u32>,
    
    /// Security properties that would be violated
    pub violated_properties: Vec<SecurityProperty>,
    
    /// Reason why this optimization is unsafe
    pub violation_reason: String,
}

/// Capability propagation graph
#[derive(Debug, Clone)]
pub struct CapabilityPropagationGraph {
    /// Nodes representing instructions with capability requirements
    pub nodes: HashMap<u32, CapabilityNode>,
    
    /// Edges representing capability flow between instructions
    pub edges: Vec<CapabilityEdge>,
    
    /// Strongly connected components for cycle detection
    pub components: Vec<Vec<u32>>,
    
    /// Topological ordering for analysis
    pub topological_order: Vec<u32>,
}

/// Node in capability propagation graph
#[derive(Debug, Clone)]
pub struct CapabilityNode {
    /// Instruction offset
    pub instruction_offset: u32,
    
    /// Required capabilities
    pub required_capabilities: CapabilitySet,
    
    /// Provided capabilities
    pub provided_capabilities: CapabilitySet,
    
    /// Security level at this point
    pub security_level: SecurityLevel,
    
    /// Effects produced at this instruction
    pub effects: Vec<Effect>,
}

/// Edge in capability propagation graph
#[derive(Debug, Clone)]
pub struct CapabilityEdge {
    /// Source instruction
    pub from: u32,
    
    /// Target instruction
    pub to: u32,
    
    /// Capabilities propagated along this edge
    pub propagated_capabilities: CapabilitySet,
    
    /// Type of capability flow
    pub flow_type: CapabilityFlowType,
    
    /// Conditions for this propagation
    pub conditions: Vec<PropagationCondition>,
}

/// Supporting types for comprehensive analysis
#[derive(Debug, Clone)]
pub enum TrustLevel {
    Untrusted,
    LimitedTrust,
    Trusted,
    HighlyTrusted,
}

#[derive(Debug, Clone)]
pub enum PrivilegeChange {
    Escalation,
    Deescalation,
    Lateral,
}

#[derive(Debug, Clone)]
pub enum IsolationType {
    Memory,
    Execution,
    Data,
    Effect,
}

#[derive(Debug, Clone)]
pub enum CapabilityOperation {
    Grant,
    Revoke,
    Delegate,
    Check,
}

#[derive(Debug, Clone)]
pub enum IsolationLevel {
    None,
    Weak,
    Strong,
    Complete,
}

#[derive(Debug, Clone)]
pub struct CapabilityDelegation {
    pub instruction_offset: u32,
    pub delegator: String,
    pub delegatee: String,
    pub delegated_capabilities: CapabilitySet,
    pub delegation_constraints: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CapabilityRevocation {
    pub instruction_offset: u32,
    pub revoker: String,
    pub revoked_capabilities: CapabilitySet,
    pub revocation_reason: String,
}

#[derive(Debug, Clone)]
pub enum SecurityEffect {
    CapabilityGrant(String),
    CapabilityRevoke(String),
    SecurityLevelChange(SecurityLevel),
    BoundaryTransition(SecurityBoundaryType),
    InformationFlow(InformationFlow),
}

#[derive(Debug, Clone)]
pub enum SanitizationType {
    Encryption,
    Hashing,
    Filtering,
    Validation,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    Inlining,
    ConstantFolding,
    DeadCodeElimination,
    LoopOptimization,
    BoundsCheckElimination,
    RedundancyElimination,
}

#[derive(Debug, Clone)]
pub enum SecurityProperty {
    CapabilityConfinement,
    InformationFlowControl,
    EffectIsolation,
    BoundaryIntegrity,
    TrustPreservation,
}

#[derive(Debug, Clone)]
pub enum SafetyCondition {
    CapabilitiesPresent(CapabilitySet),
    SecurityLevelMaintained(SecurityLevel),
    BoundariesRespected,
    EffectsContained,
}

#[derive(Debug, Clone)]
pub struct OptimizationConstraint {
    pub constraint_type: ConstraintType,
    pub affected_instructions: Vec<u32>,
    pub required_capabilities: CapabilitySet,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    MustPreserveCapability,
    MustNotEscalatePrivilege,
    MustRespectBoundary,
    MustContainEffects,
}

#[derive(Debug, Clone)]
pub struct TransformationRule {
    pub rule_type: TransformationRuleType,
    pub conditions: Vec<String>,
    pub actions: Vec<String>,
    pub security_impact: String,
}

#[derive(Debug, Clone)]
pub enum TransformationRuleType {
    CapabilityPreserving,
    SecurityLevelMaintaining,
    BoundaryRespecting,
    EffectContaining,
}

#[derive(Debug, Clone)]
pub enum CapabilityFlowType {
    Direct,
    Conditional,
    Delegated,
    Inherited,
}

#[derive(Debug, Clone)]
pub enum PropagationCondition {
    AlwaysPropagate,
    ConditionalOnCapability(String),
    ConditionalOnSecurityLevel(SecurityLevel),
    ConditionalOnEffect(String),
}

impl CapabilityAnalyzer {
    /// Create new capability analyzer
    pub fn new(config: &AnalysisConfig) -> VMResult<Self> {
        let capability_manager = CapabilityManager::new()
            .map_err(|e| PrismVMError::AnalysisError(format!("Failed to create capability manager: {}", e)))?;
        
        let security_system = SecuritySystem::new();
        
        Ok(Self {
            config: config.clone(),
            capability_manager,
            security_system,
            flow_cache: HashMap::new(),
        })
    }

    /// Perform comprehensive capability analysis
    pub fn analyze(&mut self, function: &FunctionDefinition) -> VMResult<CapabilityAnalysis> {
        // Step 1: Analyze capability requirements for each instruction
        let instruction_requirements = self.analyze_instruction_requirements(function)?;
        
        // Step 2: Build capability flow graph
        let capability_flow = self.analyze_capability_flow(function, &instruction_requirements)?;
        
        // Step 3: Identify security boundaries
        let security_boundaries = self.identify_security_boundaries(function, &capability_flow)?;
        
        // Step 4: Analyze information flow constraints
        let information_flows = self.analyze_information_flows(function, &capability_flow)?;
        
        // Step 5: Generate security constraints
        let security_constraints = self.generate_security_constraints(
            function, 
            &capability_flow, 
            &security_boundaries
        )?;
        
        // Step 6: Analyze optimization safety
        let optimization_safety = self.analyze_optimization_safety(
            function, 
            &capability_flow, 
            &security_constraints
        )?;
        
        // Step 7: Build capability propagation graph
        let propagation_graph = self.build_propagation_graph(
            function, 
            &instruction_requirements
        )?;
        
        Ok(CapabilityAnalysis {
            function_id: function.id,
            capability_flow,
            security_constraints,
            instruction_requirements,
            security_boundaries,
            information_flows,
            optimization_safety,
            propagation_graph,
        })
    }

    /// Analyze capability requirements for each instruction
    fn analyze_instruction_requirements(
        &self, 
        function: &FunctionDefinition
    ) -> VMResult<HashMap<u32, CapabilityRequirement>> {
        let mut requirements = HashMap::new();
        
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let requirement = self.analyze_single_instruction(instruction, offset as u32)?;
            requirements.insert(offset as u32, requirement);
        }
        
        Ok(requirements)
    }

    /// Analyze capability requirements for a single instruction
    fn analyze_single_instruction(
        &self, 
        instruction: &Instruction, 
        offset: u32
    ) -> VMResult<CapabilityRequirement> {
        let mut pre_capabilities = CapabilitySet::new();
        let mut post_capabilities = CapabilitySet::new();
        let mut consumed_capabilities = CapabilitySet::new();
        let mut produced_capabilities = CapabilitySet::new();
        let mut security_effects = Vec::new();

        // Add capabilities from instruction metadata
        for cap in &instruction.required_capabilities {
            let runtime_cap = self.convert_pir_capability_to_runtime(cap)?;
            pre_capabilities.add(runtime_cap);
        }

        // Analyze opcode-specific capability requirements
        match instruction.opcode {
            PrismOpcode::CAP_CHECK(_) => {
                security_effects.push(SecurityEffect::CapabilityGrant("capability_check".to_string()));
            }
            
            PrismOpcode::CAP_DELEGATE(_) => {
                security_effects.push(SecurityEffect::CapabilityGrant("capability_delegate".to_string()));
            }
            
            PrismOpcode::CAP_REVOKE(_) => {
                security_effects.push(SecurityEffect::CapabilityRevoke("capability_revoke".to_string()));
            }
            
            PrismOpcode::EFFECT_INVOKE(_) => {
                security_effects.push(SecurityEffect::BoundaryTransition(
                    SecurityBoundaryType::IsolationBoundary { 
                        isolation_type: IsolationType::Effect 
                    }
                ));
            }
            
            PrismOpcode::IO_READ(_) | PrismOpcode::IO_WRITE(_) | PrismOpcode::IO_OPEN(_) => {
                // I/O operations require special capabilities
                let io_cap = Capability::new(
                    prism_runtime::authority::capability::Authority::FileSystem(
                        prism_runtime::authority::capability::FileSystemAuthority {
                            operations: std::iter::once(
                                prism_runtime::authority::capability::FileOperation::Read
                            ).collect(),
                            allowed_paths: vec![],
                        }
                    ),
                    prism_runtime::authority::capability::ConstraintSet::new(),
                    std::time::Duration::from_secs(3600),
                    prism_runtime::authority::capability::ComponentId::new(0),
                );
                pre_capabilities.add(io_cap);
            }
            
            _ => {
                // Handle other instructions based on their effects
                if instruction.has_side_effects() {
                    security_effects.push(SecurityEffect::BoundaryTransition(
                        SecurityBoundaryType::IsolationBoundary { 
                            isolation_type: IsolationType::Execution 
                        }
                    ));
                }
            }
        }

        Ok(CapabilityRequirement {
            instruction_offset: offset,
            pre_capabilities,
            post_capabilities,
            consumed_capabilities,
            produced_capabilities,
            security_effects,
        })
    }

    /// Convert PIR capability to runtime capability with comprehensive mapping
    fn convert_pir_capability_to_runtime(&self, pir_cap: &PIRCapability) -> VMResult<Capability> {
        // Determine the appropriate authority type based on PIR capability
        let authority = self.map_pir_capability_to_authority(pir_cap)?;
        
        // Convert PIR constraints to runtime constraints
        let constraints = self.convert_pir_constraints_to_runtime(pir_cap)?;
        
        // Determine validity duration (default to 1 hour, can be overridden by constraints)
        let validity_duration = constraints.time_constraints
            .iter()
            .filter_map(|tc| match tc {
                prism_runtime::authority::capability::TimeConstraint::ValidUntil(until) => {
                    until.duration_since(std::time::SystemTime::now()).ok()
                }
                _ => None,
            })
            .next()
            .unwrap_or(std::time::Duration::from_secs(3600));
        
        let capability = Capability::new(
            authority,
            constraints,
            validity_duration,
            prism_runtime::authority::capability::ComponentId::new(0),
        );
        
        Ok(capability)
    }

    /// Map PIR capability to appropriate runtime authority
    fn map_pir_capability_to_authority(&self, pir_cap: &PIRCapability) -> VMResult<prism_runtime::authority::capability::Authority> {
        use prism_runtime::authority::capability::*;
        
        // Analyze capability name and permissions to determine authority type
        let authority = match pir_cap.name.as_str() {
            // File system capabilities
            name if name.contains("file") || name.contains("io") || name.contains("read") || name.contains("write") => {
                let mut operations = std::collections::HashSet::new();
                let mut allowed_paths = Vec::new();
                
                // Determine specific file operations from permissions
                for permission in &pir_cap.permissions {
                    match permission.as_str() {
                        "read" | "file_read" => { operations.insert(FileOperation::Read); }
                        "write" | "file_write" => { operations.insert(FileOperation::Write); }
                        "execute" | "file_execute" => { operations.insert(FileOperation::Execute); }
                        "delete" | "file_delete" => { operations.insert(FileOperation::Delete); }
                        path if path.starts_with("/") || path.contains("*") => {
                            allowed_paths.push(path.clone());
                        }
                        _ => {
                            // Default to read if unclear
                            operations.insert(FileOperation::Read);
                        }
                    }
                }
                
                // If no operations specified, allow read by default
                if operations.is_empty() {
                    operations.insert(FileOperation::Read);
                }
                
                Authority::FileSystem(FileSystemAuthority {
                    operations,
                    allowed_paths,
                })
            }
            
            // Network capabilities
            name if name.contains("network") || name.contains("http") || name.contains("tcp") || name.contains("udp") => {
                let mut operations = std::collections::HashSet::new();
                let mut allowed_hosts = Vec::new();
                let mut allowed_ports = Vec::new();
                
                for permission in &pir_cap.permissions {
                    match permission.as_str() {
                        "connect" | "network_connect" => { operations.insert(NetworkOperation::Connect); }
                        "listen" | "network_listen" => { operations.insert(NetworkOperation::Listen); }
                        "send" | "network_send" => { operations.insert(NetworkOperation::Send); }
                        "receive" | "network_receive" => { operations.insert(NetworkOperation::Receive); }
                        host if host.contains(".") || host.contains(":") => {
                            allowed_hosts.push(host.clone());
                        }
                        port if port.parse::<u16>().is_ok() => {
                            if let Ok(port_num) = port.parse::<u16>() {
                                allowed_ports.push(port_num);
                            }
                        }
                        _ => {
                            operations.insert(NetworkOperation::Connect);
                        }
                    }
                }
                
                if operations.is_empty() {
                    operations.insert(NetworkOperation::Connect);
                }
                
                Authority::Network(NetworkAuthority {
                    operations,
                    allowed_hosts,
                    allowed_ports,
                })
            }
            
            // Database capabilities
            name if name.contains("database") || name.contains("db") || name.contains("sql") => {
                let mut operations = std::collections::HashSet::new();
                let mut allowed_databases = Vec::new();
                let mut allowed_tables = Vec::new();
                
                for permission in &pir_cap.permissions {
                    match permission.as_str() {
                        "select" | "db_select" => { operations.insert(DatabaseOperation::Select); }
                        "insert" | "db_insert" => { operations.insert(DatabaseOperation::Insert); }
                        "update" | "db_update" => { operations.insert(DatabaseOperation::Update); }
                        "delete" | "db_delete" => { operations.insert(DatabaseOperation::Delete); }
                        db if db.starts_with("db:") => {
                            allowed_databases.push(db[3..].to_string());
                        }
                        table if table.starts_with("table:") => {
                            allowed_tables.push(table[6..].to_string());
                        }
                        _ => {
                            operations.insert(DatabaseOperation::Select);
                        }
                    }
                }
                
                if operations.is_empty() {
                    operations.insert(DatabaseOperation::Select);
                }
                
                Authority::Database(DatabaseAuthority {
                    operations,
                    allowed_databases,
                    allowed_tables,
                })
            }
            
            // Memory capabilities
            name if name.contains("memory") || name.contains("alloc") => {
                let mut operations = std::collections::HashSet::new();
                let mut max_allocation = None;
                let mut allowed_regions = Vec::new();
                
                for permission in &pir_cap.permissions {
                    match permission.as_str() {
                        "allocate" | "memory_allocate" => { operations.insert(MemoryOperation::Allocate); }
                        "deallocate" | "memory_deallocate" => { operations.insert(MemoryOperation::Deallocate); }
                        "read" | "memory_read" => { operations.insert(MemoryOperation::Read); }
                        "write" | "memory_write" => { operations.insert(MemoryOperation::Write); }
                        size if size.starts_with("max:") => {
                            if let Ok(bytes) = size[4..].parse::<usize>() {
                                max_allocation = Some(bytes);
                            }
                        }
                        region if region.starts_with("region:") => {
                            allowed_regions.push(region[7..].to_string());
                        }
                        _ => {
                            operations.insert(MemoryOperation::Read);
                        }
                    }
                }
                
                if operations.is_empty() {
                    operations.insert(MemoryOperation::Read);
                }
                
                Authority::Memory(MemoryAuthority {
                    operations,
                    max_allocation,
                    allowed_regions,
                })
            }
            
            // System capabilities
            name if name.contains("system") || name.contains("process") || name.contains("env") => {
                let mut operations = std::collections::HashSet::new();
                let mut allowed_env_vars = Vec::new();
                
                for permission in &pir_cap.permissions {
                    match permission.as_str() {
                        "process_create" | "spawn" => { operations.insert(SystemOperation::ProcessCreate); }
                        "process_kill" | "kill" => { operations.insert(SystemOperation::ProcessKill); }
                        "env_read" | "environment_read" => { operations.insert(SystemOperation::EnvironmentRead); }
                        "env_write" | "environment_write" => { operations.insert(SystemOperation::EnvironmentWrite); }
                        var if var.starts_with("env:") => {
                            allowed_env_vars.push(var[4..].to_string());
                        }
                        _ => {
                            operations.insert(SystemOperation::EnvironmentRead);
                        }
                    }
                }
                
                if operations.is_empty() {
                    operations.insert(SystemOperation::EnvironmentRead);
                }
                
                Authority::System(SystemAuthority {
                    operations,
                    allowed_env_vars,
                })
            }
            
            // Composite capabilities (multiple authorities)
            name if name.contains("composite") || pir_cap.permissions.len() > 5 => {
                // For complex capabilities, create composite authority
                let mut authorities = Vec::new();
                
                // Recursively create sub-capabilities for different permission groups
                let mut file_permissions = Vec::new();
                let mut network_permissions = Vec::new();
                let mut db_permissions = Vec::new();
                let mut memory_permissions = Vec::new();
                let mut system_permissions = Vec::new();
                
                for permission in &pir_cap.permissions {
                    if permission.contains("file") || permission.contains("read") || permission.contains("write") {
                        file_permissions.push(permission.clone());
                    } else if permission.contains("network") || permission.contains("http") {
                        network_permissions.push(permission.clone());
                    } else if permission.contains("db") || permission.contains("database") {
                        db_permissions.push(permission.clone());
                    } else if permission.contains("memory") || permission.contains("alloc") {
                        memory_permissions.push(permission.clone());
                    } else {
                        system_permissions.push(permission.clone());
                    }
                }
                
                // Create sub-authorities for each category that has permissions
                if !file_permissions.is_empty() {
                    let sub_cap = PIRCapability {
                        name: "file_sub".to_string(),
                        description: None,
                        permissions: file_permissions,
                    };
                    if let Ok(Authority::FileSystem(fs_auth)) = self.map_pir_capability_to_authority(&sub_cap) {
                        authorities.push(Authority::FileSystem(fs_auth));
                    }
                }
                
                if !network_permissions.is_empty() {
                    let sub_cap = PIRCapability {
                        name: "network_sub".to_string(),
                        description: None,
                        permissions: network_permissions,
                    };
                    if let Ok(Authority::Network(net_auth)) = self.map_pir_capability_to_authority(&sub_cap) {
                        authorities.push(Authority::Network(net_auth));
                    }
                }
                
                if !db_permissions.is_empty() {
                    let sub_cap = PIRCapability {
                        name: "database_sub".to_string(),
                        description: None,
                        permissions: db_permissions,
                    };
                    if let Ok(Authority::Database(db_auth)) = self.map_pir_capability_to_authority(&sub_cap) {
                        authorities.push(Authority::Database(db_auth));
                    }
                }
                
                if !memory_permissions.is_empty() {
                    let sub_cap = PIRCapability {
                        name: "memory_sub".to_string(),
                        description: None,
                        permissions: memory_permissions,
                    };
                    if let Ok(Authority::Memory(mem_auth)) = self.map_pir_capability_to_authority(&sub_cap) {
                        authorities.push(Authority::Memory(mem_auth));
                    }
                }
                
                if !system_permissions.is_empty() {
                    let sub_cap = PIRCapability {
                        name: "system_sub".to_string(),
                        description: None,
                        permissions: system_permissions,
                    };
                    if let Ok(Authority::System(sys_auth)) = self.map_pir_capability_to_authority(&sub_cap) {
                        authorities.push(Authority::System(sys_auth));
                    }
                }
                
                Authority::Composite(authorities)
            }
            
            // Default to system authority for unknown capabilities
            _ => {
                Authority::System(SystemAuthority {
                    operations: std::iter::once(SystemOperation::EnvironmentRead).collect(),
                    allowed_env_vars: vec![],
                })
            }
        };
        
        Ok(authority)
    }

    /// Convert PIR capability constraints to runtime constraints
    fn convert_pir_constraints_to_runtime(&self, pir_cap: &PIRCapability) -> VMResult<prism_runtime::authority::capability::ConstraintSet> {
        use prism_runtime::authority::capability::*;
        
        let mut constraints = ConstraintSet::new();
        
        // Analyze capability name and description for constraint hints
        if let Some(description) = &pir_cap.description {
            // Parse time constraints from description
            if description.contains("1h") || description.contains("hour") {
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                ));
            } else if description.contains("1d") || description.contains("day") {
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(86400)
                ));
            } else if description.contains("1m") || description.contains("minute") {
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(60)
                ));
            }
            
            // Parse rate limits from description
            if description.contains("100/s") {
                constraints.rate_limits.push(RateLimit::PerSecond(100));
            } else if description.contains("1000/s") {
                constraints.rate_limits.push(RateLimit::PerSecond(1000));
            } else if description.contains("rate_limit") {
                constraints.rate_limits.push(RateLimit::PerSecond(10)); // Default rate limit
            }
            
            // Parse resource limits from description
            if description.contains("1MB") {
                constraints.resource_limits.push(ResourceLimit::Memory(1024 * 1024));
            } else if description.contains("10MB") {
                constraints.resource_limits.push(ResourceLimit::Memory(10 * 1024 * 1024));
            } else if description.contains("memory_limit") {
                constraints.resource_limits.push(ResourceLimit::Memory(1024 * 1024)); // 1MB default
            }
            
            // Parse context constraints from description
            if description.contains("requires_auth") {
                constraints.context_constraints.push(ContextConstraint::RequiredCapabilities(
                    vec!["authentication".to_string()]
                ));
            }
            
            if description.contains("typescript_only") {
                constraints.context_constraints.push(ContextConstraint::ExecutionTarget(
                    std::iter::once(crate::platform::execution::ExecutionTarget::TypeScript).collect()
                ));
            } else if description.contains("native_only") {
                constraints.context_constraints.push(ContextConstraint::ExecutionTarget(
                    std::iter::once(crate::platform::execution::ExecutionTarget::Native).collect()
                ));
            }
        }
        
        // Add default constraints based on capability type
        match pir_cap.name.as_str() {
            name if name.contains("network") => {
                // Network operations get rate limiting by default
                if constraints.rate_limits.is_empty() {
                    constraints.rate_limits.push(RateLimit::PerSecond(100));
                }
                // Network operations get time limits
                if constraints.time_constraints.is_empty() {
                    constraints.time_constraints.push(TimeConstraint::ValidUntil(
                        std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                    ));
                }
            }
            
            name if name.contains("file") => {
                // File operations get resource limits
                if constraints.resource_limits.is_empty() {
                    constraints.resource_limits.push(ResourceLimit::FileDescriptors(100));
                }
            }
            
            name if name.contains("memory") => {
                // Memory operations get memory limits
                if constraints.resource_limits.is_empty() {
                    constraints.resource_limits.push(ResourceLimit::Memory(10 * 1024 * 1024)); // 10MB
                }
            }
            
            name if name.contains("database") => {
                // Database operations get connection and query limits
                if constraints.rate_limits.is_empty() {
                    constraints.rate_limits.push(RateLimit::PerSecond(50));
                }
                if constraints.resource_limits.is_empty() {
                    constraints.resource_limits.push(ResourceLimit::Memory(5 * 1024 * 1024)); // 5MB
                }
            }
            
            _ => {
                // Default constraints for unknown capabilities
                if constraints.time_constraints.is_empty() {
                    constraints.time_constraints.push(TimeConstraint::ValidUntil(
                        std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                    ));
                }
            }
        }
        
        Ok(constraints)
    }

    /// Convert PIR effect category to runtime capability
    pub fn convert_pir_effect_to_runtime_capability(&self, effect_category: &prism_pir::effects::EffectCategory) -> VMResult<Capability> {
        use prism_runtime::authority::capability::*;
        use prism_pir::effects::EffectCategory;
        
        let authority = match effect_category {
            EffectCategory::IO(io_effect) => {
                use prism_pir::effects::IOEffect;
                match io_effect {
                    IOEffect::FileRead { path, .. } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Read).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    IOEffect::FileWrite { path, .. } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Write).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    IOEffect::FileDelete { path } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Delete).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    IOEffect::DirectoryCreate { path } | IOEffect::DirectoryList { path } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: [FileOperation::Read, FileOperation::Write].into_iter().collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    IOEffect::Console { .. } => {
                        Authority::System(SystemAuthority {
                            operations: std::iter::once(SystemOperation::EnvironmentRead).collect(),
                            allowed_env_vars: vec!["TERM".to_string(), "SHELL".to_string()],
                        })
                    }
                }
            }
            
            EffectCategory::Network(network_effect) => {
                use prism_pir::effects::NetworkEffect;
                match network_effect {
                    NetworkEffect::HttpRequest { url, method: _, headers: _ } => {
                        let host = url.split('/').nth(2).unwrap_or("*").to_string();
                        Authority::Network(NetworkAuthority {
                            operations: [NetworkOperation::Connect, NetworkOperation::Send, NetworkOperation::Receive].into_iter().collect(),
                            allowed_hosts: vec![host],
                            allowed_ports: vec![80, 443], // HTTP/HTTPS
                        })
                    }
                    NetworkEffect::TcpConnect { host, port } => {
                        Authority::Network(NetworkAuthority {
                            operations: [NetworkOperation::Connect, NetworkOperation::Send, NetworkOperation::Receive].into_iter().collect(),
                            allowed_hosts: vec![host.clone()],
                            allowed_ports: vec![*port],
                        })
                    }
                    NetworkEffect::TcpListen { port } => {
                        Authority::Network(NetworkAuthority {
                            operations: std::iter::once(NetworkOperation::Listen).collect(),
                            allowed_hosts: vec!["0.0.0.0".to_string()],
                            allowed_ports: vec![*port],
                        })
                    }
                    NetworkEffect::UdpSend { host, port, .. } | NetworkEffect::UdpReceive { host, port } => {
                        Authority::Network(NetworkAuthority {
                            operations: [NetworkOperation::Send, NetworkOperation::Receive].into_iter().collect(),
                            allowed_hosts: vec![host.clone()],
                            allowed_ports: vec![*port],
                        })
                    }
                    NetworkEffect::WebSocketConnect { url } => {
                        let host = url.split('/').nth(2).unwrap_or("*").to_string();
                        Authority::Network(NetworkAuthority {
                            operations: [NetworkOperation::Connect, NetworkOperation::Send, NetworkOperation::Receive].into_iter().collect(),
                            allowed_hosts: vec![host],
                            allowed_ports: vec![80, 443], // WebSocket over HTTP/HTTPS
                        })
                    }
                }
            }
            
            EffectCategory::Database(db_effect) => {
                use prism_pir::effects::DatabaseEffect;
                match db_effect {
                    DatabaseEffect::Query { database, query: _, parameters: _ } => {
                        Authority::Database(DatabaseAuthority {
                            operations: std::iter::once(DatabaseOperation::Select).collect(),
                            allowed_databases: vec![database.clone()],
                            allowed_tables: vec!["*".to_string()], // Allow all tables for queries
                        })
                    }
                    DatabaseEffect::Insert { database, table, .. } => {
                        Authority::Database(DatabaseAuthority {
                            operations: std::iter::once(DatabaseOperation::Insert).collect(),
                            allowed_databases: vec![database.clone()],
                            allowed_tables: vec![table.clone()],
                        })
                    }
                    DatabaseEffect::Update { database, table, .. } => {
                        Authority::Database(DatabaseAuthority {
                            operations: std::iter::once(DatabaseOperation::Update).collect(),
                            allowed_databases: vec![database.clone()],
                            allowed_tables: vec![table.clone()],
                        })
                    }
                    DatabaseEffect::Delete { database, table, .. } => {
                        Authority::Database(DatabaseAuthority {
                            operations: std::iter::once(DatabaseOperation::Delete).collect(),
                            allowed_databases: vec![database.clone()],
                            allowed_tables: vec![table.clone()],
                        })
                    }
                    DatabaseEffect::Transaction { database, .. } => {
                        Authority::Database(DatabaseAuthority {
                            operations: [
                                DatabaseOperation::Select, 
                                DatabaseOperation::Insert, 
                                DatabaseOperation::Update, 
                                DatabaseOperation::Delete
                            ].into_iter().collect(),
                            allowed_databases: vec![database.clone()],
                            allowed_tables: vec!["*".to_string()],
                        })
                    }
                }
            }
            
            EffectCategory::FileSystem(fs_effect) => {
                use prism_pir::effects::FileSystemEffect;
                match fs_effect {
                    FileSystemEffect::Read { path, .. } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Read).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    FileSystemEffect::Write { path, .. } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Write).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    FileSystemEffect::Delete { path } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Delete).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    FileSystemEffect::CreateDirectory { path } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Write).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                    FileSystemEffect::Move { from, to } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: [FileOperation::Read, FileOperation::Write, FileOperation::Delete].into_iter().collect(),
                            allowed_paths: vec![from.clone(), to.clone()],
                        })
                    }
                    FileSystemEffect::Copy { from, to } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: [FileOperation::Read, FileOperation::Write].into_iter().collect(),
                            allowed_paths: vec![from.clone(), to.clone()],
                        })
                    }
                    FileSystemEffect::Watch { path, .. } => {
                        Authority::FileSystem(FileSystemAuthority {
                            operations: std::iter::once(FileOperation::Read).collect(),
                            allowed_paths: vec![path.clone()],
                        })
                    }
                }
            }
            
            EffectCategory::Memory(memory_effect) => {
                use prism_pir::effects::MemoryEffect;
                match memory_effect {
                    MemoryEffect::Allocate { size, .. } => {
                        Authority::Memory(MemoryAuthority {
                            operations: std::iter::once(MemoryOperation::Allocate).collect(),
                            max_allocation: Some(*size),
                            allowed_regions: vec!["heap".to_string()],
                        })
                    }
                    MemoryEffect::Deallocate { .. } => {
                        Authority::Memory(MemoryAuthority {
                            operations: std::iter::once(MemoryOperation::Deallocate).collect(),
                            max_allocation: None,
                            allowed_regions: vec!["heap".to_string()],
                        })
                    }
                    MemoryEffect::Read { region, .. } => {
                        Authority::Memory(MemoryAuthority {
                            operations: std::iter::once(MemoryOperation::Read).collect(),
                            max_allocation: None,
                            allowed_regions: vec![region.clone()],
                        })
                    }
                    MemoryEffect::Write { region, .. } => {
                        Authority::Memory(MemoryAuthority {
                            operations: std::iter::once(MemoryOperation::Write).collect(),
                            max_allocation: None,
                            allowed_regions: vec![region.clone()],
                        })
                    }
                    MemoryEffect::MemoryMap { size, .. } => {
                        Authority::Memory(MemoryAuthority {
                            operations: [MemoryOperation::Allocate, MemoryOperation::Read, MemoryOperation::Write].into_iter().collect(),
                            max_allocation: Some(*size),
                            allowed_regions: vec!["mmap".to_string()],
                        })
                    }
                }
            }
            
            EffectCategory::System(system_effect) => {
                use prism_pir::effects::SystemEffect;
                match system_effect {
                    SystemEffect::ProcessSpawn { command, .. } => {
                        Authority::System(SystemAuthority {
                            operations: std::iter::once(SystemOperation::ProcessCreate).collect(),
                            allowed_env_vars: vec!["PATH".to_string(), "HOME".to_string()],
                        })
                    }
                    SystemEffect::ProcessKill { .. } => {
                        Authority::System(SystemAuthority {
                            operations: std::iter::once(SystemOperation::ProcessKill).collect(),
                            allowed_env_vars: vec![],
                        })
                    }
                    SystemEffect::EnvironmentRead { variable } => {
                        Authority::System(SystemAuthority {
                            operations: std::iter::once(SystemOperation::EnvironmentRead).collect(),
                            allowed_env_vars: vec![variable.clone()],
                        })
                    }
                    SystemEffect::EnvironmentWrite { variable, .. } => {
                        Authority::System(SystemAuthority {
                            operations: std::iter::once(SystemOperation::EnvironmentWrite).collect(),
                            allowed_env_vars: vec![variable.clone()],
                        })
                    }
                    SystemEffect::SystemCall { call, .. } => {
                        // Map system calls to appropriate operations
                        let operations = match call.as_str() {
                            "fork" | "exec" => std::iter::once(SystemOperation::ProcessCreate).collect(),
                            "kill" => std::iter::once(SystemOperation::ProcessKill).collect(),
                            "getenv" | "setenv" => [SystemOperation::EnvironmentRead, SystemOperation::EnvironmentWrite].into_iter().collect(),
                            _ => std::iter::once(SystemOperation::EnvironmentRead).collect(),
                        };
                        
                        Authority::System(SystemAuthority {
                            operations,
                            allowed_env_vars: vec!["PATH".to_string()],
                        })
                    }
                }
            }
            
            EffectCategory::Cryptography(crypto_effect) => {
                // Cryptography effects map to system authority with specific constraints
                Authority::System(SystemAuthority {
                    operations: std::iter::once(SystemOperation::EnvironmentRead).collect(),
                    allowed_env_vars: vec!["CRYPTO_CONFIG".to_string()],
                })
            }
            
            EffectCategory::Time(time_effect) => {
                // Time effects are generally safe and map to system authority
                Authority::System(SystemAuthority {
                    operations: std::iter::once(SystemOperation::EnvironmentRead).collect(),
                    allowed_env_vars: vec!["TZ".to_string()],
                })
            }
            
            EffectCategory::Security(security_effect) => {
                // Security effects require system authority
                Authority::System(SystemAuthority {
                    operations: [SystemOperation::EnvironmentRead, SystemOperation::EnvironmentWrite].into_iter().collect(),
                    allowed_env_vars: vec!["SECURITY_CONFIG".to_string(), "AUTH_TOKEN".to_string()],
                })
            }
            
            EffectCategory::AI(ai_effect) => {
                // AI effects typically require network access for API calls
                Authority::Network(NetworkAuthority {
                    operations: [NetworkOperation::Connect, NetworkOperation::Send, NetworkOperation::Receive].into_iter().collect(),
                    allowed_hosts: vec!["api.openai.com".to_string(), "*.googleapis.com".to_string()],
                    allowed_ports: vec![443], // HTTPS
                })
            }
        };
        
        // Create constraints based on effect category
        let constraints = self.create_constraints_for_effect_category(effect_category)?;
        
        let capability = Capability::new(
            authority,
            constraints,
            std::time::Duration::from_secs(3600), // 1 hour default
            prism_runtime::authority::capability::ComponentId::new(0),
        );
        
        Ok(capability)
    }

    /// Create appropriate constraints for effect categories
    fn create_constraints_for_effect_category(&self, effect_category: &prism_pir::effects::EffectCategory) -> VMResult<prism_runtime::authority::capability::ConstraintSet> {
        use prism_runtime::authority::capability::*;
        use prism_pir::effects::EffectCategory;
        
        let mut constraints = ConstraintSet::new();
        
        match effect_category {
            EffectCategory::Network(_) => {
                // Network operations get rate limiting and time constraints
                constraints.rate_limits.push(RateLimit::PerSecond(100));
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                ));
                constraints.resource_limits.push(ResourceLimit::NetworkBandwidth(1024 * 1024)); // 1MB/s
            }
            
            EffectCategory::FileSystem(_) | EffectCategory::IO(_) => {
                // File operations get resource limits
                constraints.resource_limits.push(ResourceLimit::FileDescriptors(100));
                constraints.resource_limits.push(ResourceLimit::DiskSpace(100 * 1024 * 1024)); // 100MB
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(7200) // 2 hours
                ));
            }
            
            EffectCategory::Database(_) => {
                // Database operations get connection and query limits
                constraints.rate_limits.push(RateLimit::PerSecond(50));
                constraints.resource_limits.push(ResourceLimit::Memory(10 * 1024 * 1024)); // 10MB
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(1800) // 30 minutes
                ));
            }
            
            EffectCategory::Memory(_) => {
                // Memory operations get strict memory limits
                constraints.resource_limits.push(ResourceLimit::Memory(50 * 1024 * 1024)); // 50MB
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                ));
            }
            
            EffectCategory::System(_) => {
                // System operations get process and time limits
                constraints.rate_limits.push(RateLimit::PerSecond(10)); // Strict rate limiting for system ops
                constraints.resource_limits.push(ResourceLimit::CpuTime(std::time::Duration::from_secs(60)));
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(1800) // 30 minutes
                ));
            }
            
            EffectCategory::Cryptography(_) => {
                // Crypto operations get CPU time limits
                constraints.resource_limits.push(ResourceLimit::CpuTime(std::time::Duration::from_secs(30)));
                constraints.resource_limits.push(ResourceLimit::Memory(20 * 1024 * 1024)); // 20MB
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                ));
            }
            
            EffectCategory::AI(_) => {
                // AI operations get network and memory limits
                constraints.rate_limits.push(RateLimit::PerMinute(60)); // API rate limiting
                constraints.resource_limits.push(ResourceLimit::Memory(100 * 1024 * 1024)); // 100MB
                constraints.resource_limits.push(ResourceLimit::NetworkBandwidth(2 * 1024 * 1024)); // 2MB/s
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(1800) // 30 minutes
                ));
            }
            
            EffectCategory::Time(_) | EffectCategory::Security(_) => {
                // Time and security operations get basic constraints
                constraints.time_constraints.push(TimeConstraint::ValidUntil(
                    std::time::SystemTime::now() + std::time::Duration::from_secs(3600)
                ));
                constraints.resource_limits.push(ResourceLimit::Memory(5 * 1024 * 1024)); // 5MB
            }
        }
        
        Ok(constraints)
    }

    /// Analyze capability flow through the function
    fn analyze_capability_flow(
        &mut self, 
        function: &FunctionDefinition,
        requirements: &HashMap<u32, CapabilityRequirement>
    ) -> VMResult<CapabilityFlow> {
        // Check cache first
        if let Some(cached_flow) = self.flow_cache.get(&function.id) {
            return Ok(cached_flow.clone());
        }

        let mut flow = CapabilityFlow::default();
        
        // Initialize entry capabilities from function metadata
        for cap in &function.capabilities {
            let runtime_cap = self.convert_pir_capability_to_runtime(cap)?;
            flow.entry_capabilities.add(runtime_cap);
        }

        // Propagate capabilities through instructions
        let mut current_capabilities = flow.entry_capabilities.clone();
        
        for (offset, requirement) in requirements {
            // Check if current capabilities satisfy requirements
            let context = self.create_execution_context(&current_capabilities)?;
            
            for required_cap in requirement.pre_capabilities.iter() {
                if !current_capabilities.contains(required_cap) {
                    return Err(PrismVMError::AnalysisError(
                        format!("Missing required capability at instruction {}: {:?}", offset, required_cap)
                    ));
                }
            }
            
            // Update capabilities based on instruction effects
            for consumed_cap in requirement.consumed_capabilities.iter() {
                // In a real implementation, we'd remove the capability
                // For now, we just track the consumption
            }
            
            for produced_cap in requirement.produced_capabilities.iter() {
                current_capabilities.add(produced_cap.clone());
            }
            
            flow.instruction_capabilities.insert(*offset, current_capabilities.clone());
        }

        // Cache the result
        self.flow_cache.insert(function.id, flow.clone());
        
        Ok(flow)
    }

    /// Create execution context for capability validation
    fn create_execution_context(
        &self, 
        capabilities: &CapabilitySet
    ) -> VMResult<prism_runtime::platform::execution::ExecutionContext> {
        // This is a simplified implementation
        // In practice, this would create a proper execution context
        Ok(prism_runtime::platform::execution::ExecutionContext {
            capabilities: capabilities.clone(),
            target: prism_runtime::platform::execution::ExecutionTarget::Native,
            effects: vec![],
            component_id: prism_runtime::authority::capability::ComponentId::new(0),
        })
    }

    /// Identify security boundaries within the function
    fn identify_security_boundaries(
        &self,
        function: &FunctionDefinition,
        flow: &CapabilityFlow
    ) -> VMResult<Vec<SecurityBoundary>> {
        let mut boundaries = Vec::new();
        
        // Look for capability operations that create boundaries
        for (offset, instruction) in function.instructions.iter().enumerate() {
            match instruction.opcode {
                PrismOpcode::CAP_CHECK(_) | PrismOpcode::CAP_ACQUIRE(_) => {
                    boundaries.push(SecurityBoundary {
                        start_offset: offset as u32,
                        end_offset: offset as u32 + 1,
                        boundary_type: SecurityBoundaryType::CapabilityBoundary {
                            capability_operation: CapabilityOperation::Check,
                        },
                        crossing_capabilities: CapabilitySet::new(),
                        internal_security_level: SecurityLevel::new(
                            "High".to_string(), 
                            2, 
                            vec![], 
                            vec![]
                        ),
                        external_security_level: SecurityLevel::new(
                            "Medium".to_string(), 
                            1, 
                            vec![], 
                            vec![]
                        ),
                        crossing_restrictions: vec![
                            "Requires capability validation".to_string()
                        ],
                    });
                }
                
                PrismOpcode::EFFECT_ENTER(_) | PrismOpcode::EFFECT_EXIT => {
                    boundaries.push(SecurityBoundary {
                        start_offset: offset as u32,
                        end_offset: offset as u32 + 1,
                        boundary_type: SecurityBoundaryType::IsolationBoundary {
                            isolation_type: IsolationType::Effect,
                        },
                        crossing_capabilities: CapabilitySet::new(),
                        internal_security_level: SecurityLevel::new(
                            "Effect".to_string(), 
                            3, 
                            vec![], 
                            vec![]
                        ),
                        external_security_level: SecurityLevel::new(
                            "Normal".to_string(), 
                            1, 
                            vec![], 
                            vec![]
                        ),
                        crossing_restrictions: vec![
                            "Effect isolation must be maintained".to_string()
                        ],
                    });
                }
                
                _ => {}
            }
        }
        
        Ok(boundaries)
    }

    /// Analyze information flow constraints
    fn analyze_information_flows(
        &self,
        function: &FunctionDefinition,
        flow: &CapabilityFlow
    ) -> VMResult<Vec<InformationFlowConstraint>> {
        let mut constraints = Vec::new();
        
        // Analyze data dependencies and information flows
        for (offset, instruction) in function.instructions.iter().enumerate() {
            if instruction.has_side_effects() {
                // Instructions with side effects create information flow constraints
                constraints.push(InformationFlowConstraint {
                    source_instruction: offset as u32,
                    target_instruction: offset as u32 + 1,
                    flow: InformationFlow::new(
                        SecurityLevel::new("Source".to_string(), 1, vec![], vec![]),
                        SecurityLevel::new("Target".to_string(), 1, vec![], vec![]),
                        "side_effect".to_string(),
                    ),
                    constraint: FlowConstraint::RequiresCapabilities(CapabilitySet::new()),
                    reason: "Side effect requires controlled information flow".to_string(),
                });
            }
        }
        
        Ok(constraints)
    }

    /// Generate security constraints for optimization
    fn generate_security_constraints(
        &self,
        function: &FunctionDefinition,
        flow: &CapabilityFlow,
        boundaries: &[SecurityBoundary]
    ) -> VMResult<Vec<SecurityConstraint>> {
        let mut constraints = Vec::new();
        
        // Generate constraints for each security boundary
        for boundary in boundaries {
            constraints.push(SecurityConstraint {
                location: boundary.start_offset,
                constraint_type: SecurityConstraintType::BoundaryCrossing {
                    from_level: boundary.external_security_level.clone(),
                    to_level: boundary.internal_security_level.clone(),
                    reason: "Security boundary must be respected during optimization".to_string(),
                },
                required_capabilities: boundary.crossing_capabilities.clone(),
                security_level: boundary.internal_security_level.clone(),
                description: format!("Security boundary at instruction {}", boundary.start_offset),
                severity: ConstraintSeverity::High,
            });
        }
        
        // Generate constraints for capability operations
        for (offset, instruction) in function.instructions.iter().enumerate() {
            match instruction.opcode {
                PrismOpcode::CAP_CHECK(_) => {
                    constraints.push(SecurityConstraint {
                        location: offset as u32,
                        constraint_type: SecurityConstraintType::RequiresCapability {
                            capability: "capability_check".to_string(),
                            reason: "Capability check must not be optimized away".to_string(),
                        },
                        required_capabilities: CapabilitySet::new(),
                        security_level: SecurityLevel::new("High".to_string(), 2, vec![], vec![]),
                        description: "Capability check constraint".to_string(),
                        severity: ConstraintSeverity::Critical,
                    });
                }
                
                _ => {}
            }
        }
        
        Ok(constraints)
    }

    /// Analyze optimization safety with respect to capabilities
    fn analyze_optimization_safety(
        &self,
        function: &FunctionDefinition,
        flow: &CapabilityFlow,
        constraints: &[SecurityConstraint]
    ) -> VMResult<OptimizationSafety> {
        let mut safe_optimizations = Vec::new();
        let mut unsafe_optimizations = Vec::new();
        let mut optimization_constraints = Vec::new();
        let mut transformation_rules = Vec::new();

        // Analyze each potential optimization
        for (offset, instruction) in function.instructions.iter().enumerate() {
            let offset = offset as u32;
            
            // Check if instruction can be safely optimized
            let has_critical_constraints = constraints.iter().any(|c| {
                c.location == offset && c.severity >= ConstraintSeverity::High
            });
            
            if has_critical_constraints {
                unsafe_optimizations.push(UnsafeOptimization {
                    optimization_type: OptimizationType::DeadCodeElimination,
                    forbidden_instructions: vec![offset],
                    violated_properties: vec![SecurityProperty::CapabilityConfinement],
                    violation_reason: "Critical security constraint present".to_string(),
                });
            } else {
                // Check if optimization preserves security properties
                if self.can_safely_optimize_instruction(instruction, flow)? {
                    safe_optimizations.push(SafeOptimization {
                        optimization_type: OptimizationType::ConstantFolding,
                        applicable_instructions: vec![offset],
                        preserved_properties: vec![
                            SecurityProperty::CapabilityConfinement,
                            SecurityProperty::BoundaryIntegrity,
                        ],
                        safety_conditions: vec![
                            SafetyCondition::CapabilitiesPresent(CapabilitySet::new()),
                            SafetyCondition::BoundariesRespected,
                        ],
                    });
                }
            }
        }

        // Generate transformation rules
        transformation_rules.push(TransformationRule {
            rule_type: TransformationRuleType::CapabilityPreserving,
            conditions: vec!["Capability requirements must be preserved".to_string()],
            actions: vec!["Maintain capability checks during optimization".to_string()],
            security_impact: "Preserves capability-based security".to_string(),
        });

        Ok(OptimizationSafety {
            safe_optimizations,
            unsafe_optimizations,
            optimization_constraints,
            transformation_rules,
        })
    }

    /// Check if an instruction can be safely optimized
    fn can_safely_optimize_instruction(
        &self,
        instruction: &Instruction,
        flow: &CapabilityFlow
    ) -> VMResult<bool> {
        // Instructions with capability requirements generally cannot be optimized away
        if !instruction.required_capabilities.is_empty() {
            return Ok(false);
        }
        
        // Instructions with security effects need careful analysis
        if instruction.has_side_effects() {
            match instruction.opcode {
                PrismOpcode::CAP_CHECK(_) | PrismOpcode::CAP_DELEGATE(_) | PrismOpcode::CAP_REVOKE(_) => {
                    return Ok(false); // Never optimize capability operations
                }
                _ => {}
            }
        }
        
        // Pure instructions without security implications can usually be optimized
        Ok(!instruction.has_side_effects() && !instruction.can_throw())
    }

    /// Build capability propagation graph with optimizations for large functions
    fn build_propagation_graph(
        &self,
        function: &FunctionDefinition,
        requirements: &HashMap<u32, CapabilityRequirement>
    ) -> VMResult<CapabilityPropagationGraph> {
        // Pre-allocate with capacity for better performance
        let estimated_size = requirements.len();
        let mut nodes = HashMap::with_capacity(estimated_size);
        let mut edges = Vec::with_capacity(estimated_size * 2); // Estimate 2 edges per node
        
        // Build nodes efficiently
        for (&offset, requirement) in requirements {
            let node = CapabilityNode {
                instruction_offset: offset,
                required_capabilities: requirement.pre_capabilities.clone(),
                provided_capabilities: requirement.produced_capabilities.clone(),
                security_level: SecurityLevel::Medium, // TODO: Derive from context
                effects: Vec::new(), // TODO: Extract from instruction
            };
            nodes.insert(offset, node);
        }
        
        // Build edges with optimized adjacency detection
        self.build_capability_edges(function, requirements, &mut edges)?;
        
        // Use optimized algorithms for large graphs
        let (components, topological_order) = if nodes.len() > 1000 {
            // For large functions, use optimized algorithms with caching
            let components = self.compute_strongly_connected_components_optimized(&nodes, &edges)?;
            let topological_order = self.compute_topological_order_optimized(&nodes, &edges)?;
            (components, topological_order)
        } else {
            // For small functions, use standard algorithms
            let components = self.compute_strongly_connected_components(&nodes, &edges)?;
            let topological_order = self.compute_topological_order(&nodes, &edges)?;
            (components, topological_order)
        };
        
        Ok(CapabilityPropagationGraph {
            nodes,
            edges,
            components,
            topological_order,
        })
    }

    /// Build capability edges efficiently
    fn build_capability_edges(
        &self,
        function: &FunctionDefinition,
        requirements: &HashMap<u32, CapabilityRequirement>,
        edges: &mut Vec<CapabilityEdge>,
    ) -> VMResult<()> {
        // Create adjacency tracking for efficiency
        let offsets: Vec<u32> = requirements.keys().cloned().collect();
        
        for (i, &from_offset) in offsets.iter().enumerate() {
            let from_req = requirements.get(&from_offset).unwrap();
            
            // Check subsequent instructions for capability flow
            for &to_offset in offsets.iter().skip(i + 1) {
                let to_req = requirements.get(&to_offset).unwrap();
                
                // Check if capabilities flow from 'from' to 'to'
                if self.capabilities_flow(&from_req.produced_capabilities, &to_req.pre_capabilities) {
                    let edge = CapabilityEdge {
                        from: from_offset,
                        to: to_offset,
                        propagated_capabilities: self.compute_propagated_capabilities(
                            &from_req.produced_capabilities, 
                            &to_req.pre_capabilities
                        ),
                        flow_type: CapabilityFlowType::Direct,
                        conditions: Vec::new(),
                    };
                    edges.push(edge);
                }
            }
        }
        
        Ok(())
    }

    /// Check if capabilities flow between two sets
    fn capabilities_flow(&self, produced: &CapabilitySet, required: &CapabilitySet) -> bool {
        // Simple implementation - check if any produced capability satisfies required
        for produced_cap in &produced.available {
            if required.required.contains(produced_cap) {
                return true;
            }
        }
        false
    }

    /// Compute capabilities that actually propagate
    fn compute_propagated_capabilities(
        &self,
        produced: &CapabilitySet,
        required: &CapabilitySet,
    ) -> CapabilitySet {
        let mut propagated = CapabilitySet::default();
        
        for produced_cap in &produced.available {
            if required.required.contains(produced_cap) {
                propagated.available.insert(produced_cap.clone());
            }
        }
        
        propagated
    }

    /// Optimized SCC computation for large functions
    fn compute_strongly_connected_components_optimized(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> VMResult<Vec<Vec<u32>>> {
        // Use cached adjacency list for better performance
        let adjacency_list = self.build_cached_adjacency_list(nodes, edges);
        
        // Tarjan's algorithm with optimizations
        let mut tarjan_state = TarjanState::with_capacity(nodes.len());
        let mut components = Vec::new();

        // Process nodes in a more cache-friendly order
        let mut node_vec: Vec<u32> = nodes.keys().cloned().collect();
        node_vec.sort_unstable(); // Better cache locality
        
        for node_id in node_vec {
            if !tarjan_state.visited.contains(&node_id) {
                self.tarjan_scc_optimized(
                    node_id, 
                    &adjacency_list, 
                    &mut tarjan_state, 
                    &mut components
                )?;
            }
        }

        Ok(components)
    }

    /// Optimized topological sort for large functions
    fn compute_topological_order_optimized(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> VMResult<Vec<u32>> {
        // Use cached adjacency structures
        let (adjacency_list, in_degree) = self.build_cached_graph_structures(nodes, edges);
        
        // Kahn's algorithm with capacity pre-allocation
        let mut queue = VecDeque::with_capacity(nodes.len() / 4); // Estimate queue size
        let mut result = Vec::with_capacity(nodes.len());

        // Add all zero in-degree nodes
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        // Process with optimized loop
        let mut local_in_degree = in_degree; // Move to avoid repeated lookups
        while let Some(current) = queue.pop_front() {
            result.push(current);

            if let Some(neighbors) = adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    let degree = local_in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Cycle detection
        if result.len() != nodes.len() {
            return Err(PrismVMError::AnalysisError(
                format!("Cycle detected in capability propagation graph: processed {}/{} nodes", 
                       result.len(), nodes.len())
            ));
        }

        Ok(result)
    }

    /// Build cached adjacency list for reuse
    fn build_cached_adjacency_list(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> HashMap<u32, Vec<u32>> {
        let mut adjacency_list = HashMap::with_capacity(nodes.len());
        
        // Initialize all nodes
        for &node_id in nodes.keys() {
            adjacency_list.insert(node_id, Vec::new());
        }
        
        // Add edges
        for edge in edges {
            adjacency_list.entry(edge.from).or_default().push(edge.to);
        }
        
        adjacency_list
    }

    /// Build cached graph structures for efficiency
    fn build_cached_graph_structures(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> (HashMap<u32, Vec<u32>>, HashMap<u32, usize>) {
        let mut adjacency_list = HashMap::with_capacity(nodes.len());
        let mut in_degree = HashMap::with_capacity(nodes.len());
        
        // Initialize structures
        for &node_id in nodes.keys() {
            adjacency_list.insert(node_id, Vec::new());
            in_degree.insert(node_id, 0);
        }
        
        // Build structures in single pass
        for edge in edges {
            adjacency_list.entry(edge.from).or_default().push(edge.to);
            *in_degree.entry(edge.to).or_default() += 1;
        }
        
        (adjacency_list, in_degree)
    }

    /// Optimized Tarjan's algorithm with better memory usage
    fn tarjan_scc_optimized(
        &self,
        node: u32,
        adjacency_list: &HashMap<u32, Vec<u32>>,
        state: &mut TarjanState,
        components: &mut Vec<Vec<u32>>,
    ) -> VMResult<()> {
        // Initialize node with pre-allocated capacity
        let index = state.index_counter;
        state.index_counter += 1;
        state.indices.insert(node, index);
        state.lowlinks.insert(node, index);
        state.visited.insert(node);
        state.stack.push(node);
        state.on_stack.insert(node);

        // Visit neighbors with optimized iteration
        if let Some(neighbors) = adjacency_list.get(&node) {
            for &neighbor in neighbors {
                if !state.indices.contains_key(&neighbor) {
                    // Recurse on unvisited neighbor
                    self.tarjan_scc_optimized(neighbor, adjacency_list, state, components)?;
                    
                    // Update lowlink efficiently
                    let neighbor_lowlink = state.lowlinks[&neighbor];
                    let current_lowlink = state.lowlinks.get_mut(&node).unwrap();
                    *current_lowlink = (*current_lowlink).min(neighbor_lowlink);
                } else if state.on_stack.contains(&neighbor) {
                    // Update lowlink for back edge
                    let neighbor_index = state.indices[&neighbor];
                    let current_lowlink = state.lowlinks.get_mut(&node).unwrap();
                    *current_lowlink = (*current_lowlink).min(neighbor_index);
                }
            }
        }

        // Create SCC if this is a root node
        if state.lowlinks[&node] == state.indices[&node] {
            let mut component = Vec::new();
            loop {
                let v = state.stack.pop().unwrap();
                state.on_stack.remove(&v);
                component.push(v);
                if v == node {
                    break;
                }
            }
            if !component.is_empty() {
                components.push(component);
            }
        }

        Ok(())
    }

    /// Compute strongly connected components using Tarjan's algorithm for efficiency
    fn compute_strongly_connected_components(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> VMResult<Vec<Vec<u32>>> {
        // Build adjacency list for efficient traversal
        let mut adjacency_list: HashMap<u32, Vec<u32>> = HashMap::new();
        for node_id in nodes.keys() {
            adjacency_list.insert(*node_id, Vec::new());
        }
        for edge in edges {
            adjacency_list.entry(edge.from).or_default().push(edge.to);
        }

        // Tarjan's algorithm state
        let mut tarjan_state = TarjanState::new();
        let mut components = Vec::new();

        // Run Tarjan's algorithm on each unvisited node
        for &node_id in nodes.keys() {
            if !tarjan_state.visited.contains(&node_id) {
                self.tarjan_scc(
                    node_id, 
                    &adjacency_list, 
                    &mut tarjan_state, 
                    &mut components
                )?;
            }
        }

        Ok(components)
    }

    /// Tarjan's strongly connected components algorithm
    fn tarjan_scc(
        &self,
        node: u32,
        adjacency_list: &HashMap<u32, Vec<u32>>,
        state: &mut TarjanState,
        components: &mut Vec<Vec<u32>>,
    ) -> VMResult<()> {
        // Initialize node
        let index = state.index_counter;
        state.index_counter += 1;
        state.indices.insert(node, index);
        state.lowlinks.insert(node, index);
        state.visited.insert(node);
        state.stack.push(node);
        state.on_stack.insert(node);

        // Visit neighbors
        if let Some(neighbors) = adjacency_list.get(&node) {
            for &neighbor in neighbors {
                if !state.indices.contains_key(&neighbor) {
                    // Neighbor not yet visited, recurse
                    self.tarjan_scc(neighbor, adjacency_list, state, components)?;
                    let neighbor_lowlink = *state.lowlinks.get(&neighbor).unwrap();
                    let current_lowlink = state.lowlinks.get_mut(&node).unwrap();
                    *current_lowlink = (*current_lowlink).min(neighbor_lowlink);
                } else if state.on_stack.contains(&neighbor) {
                    // Neighbor is on stack and hence in current SCC
                    let neighbor_index = *state.indices.get(&neighbor).unwrap();
                    let current_lowlink = state.lowlinks.get_mut(&node).unwrap();
                    *current_lowlink = (*current_lowlink).min(neighbor_index);
                }
            }
        }

        // If node is a root node, pop the stack and create an SCC
        if state.lowlinks.get(&node) == state.indices.get(&node) {
            let mut component = Vec::new();
            loop {
                let v = state.stack.pop().unwrap();
                state.on_stack.remove(&v);
                component.push(v);
                if v == node {
                    break;
                }
            }
            if !component.is_empty() {
                components.push(component);
            }
        }

        Ok(())
    }

    /// DFS helper for SCC computation (legacy fallback)
    fn dfs_component(
        &self,
        node: u32,
        component: &mut Vec<u32>,
        visited: &mut HashSet<u32>,
        edges: &[CapabilityEdge]
    ) {
        if visited.contains(&node) {
            return;
        }
        
        visited.insert(node);
        component.push(node);
        
        // Visit all neighbors
        for edge in edges {
            if edge.from == node {
                self.dfs_component(edge.to, component, visited, edges);
            }
        }
    }

    /// Compute topological order using optimized Kahn's algorithm
    fn compute_topological_order(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> VMResult<Vec<u32>> {
        // Build adjacency list and in-degree count for efficiency
        let mut adjacency_list: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut in_degree: HashMap<u32, usize> = HashMap::new();
        
        // Initialize all nodes
        for &node_id in nodes.keys() {
            adjacency_list.insert(node_id, Vec::new());
            in_degree.insert(node_id, 0);
        }
        
        // Build graph and count in-degrees
        for edge in edges {
            adjacency_list.entry(edge.from).or_default().push(edge.to);
            *in_degree.entry(edge.to).or_default() += 1;
        }

        // Kahn's algorithm with optimized queue
        let mut queue = VecDeque::new();
        let mut result = Vec::with_capacity(nodes.len());

        // Add all nodes with no incoming edges
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }

        // Process nodes in topological order
        while let Some(current) = queue.pop_front() {
            result.push(current);

            // Process all neighbors
            if let Some(neighbors) = adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    let neighbor_degree = in_degree.get_mut(&neighbor).unwrap();
                    *neighbor_degree -= 1;
                    if *neighbor_degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != nodes.len() {
            return Err(PrismVMError::AnalysisError(
                "Cycle detected in capability propagation graph during topological sort".to_string()
            ));
        }

        Ok(result)
    }

    /// Compute topological order for analysis
    fn compute_topological_order(
        &self,
        nodes: &HashMap<u32, CapabilityNode>,
        edges: &[CapabilityEdge]
    ) -> VMResult<Vec<u32>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        
        for node_id in nodes.keys() {
            if !visited.contains(node_id) {
                self.topological_sort_visit(
                    *node_id, 
                    &mut order, 
                    &mut visited, 
                    &mut temp_visited, 
                    edges
                )?;
            }
        }
        
        order.reverse();
        Ok(order)
    }

    /// DFS-based topological sort (legacy fallback for debugging)
    fn topological_sort_visit(
        &self,
        node: u32,
        order: &mut Vec<u32>,
        visited: &mut HashSet<u32>,
        temp_visited: &mut HashSet<u32>,
        edges: &[CapabilityEdge]
    ) -> VMResult<()> {
        if temp_visited.contains(&node) {
            return Err(PrismVMError::AnalysisError(
                "Cycle detected in capability propagation graph".to_string()
            ));
        }
        
        if visited.contains(&node) {
            return Ok(());
        }
        
        temp_visited.insert(node);
        
        // Visit all neighbors
        for edge in edges {
            if edge.from == node {
                self.topological_sort_visit(edge.to, order, visited, temp_visited, edges)?;
            }
        }
        
        temp_visited.remove(&node);
        visited.insert(node);
        order.push(node);
        
        Ok(())
    }
}

/// State for Tarjan's SCC algorithm
struct TarjanState {
    index_counter: usize,
    indices: HashMap<u32, usize>,
    lowlinks: HashMap<u32, usize>,
    visited: HashSet<u32>,
    stack: Vec<u32>,
    on_stack: HashSet<u32>,
}

impl TarjanState {
    fn new() -> Self {
        Self {
            index_counter: 0,
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            visited: HashSet::new(),
            stack: Vec::new(),
            on_stack: HashSet::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            index_counter: 0,
            indices: HashMap::with_capacity(capacity),
            lowlinks: HashMap::with_capacity(capacity),
            visited: HashSet::with_capacity(capacity),
            stack: Vec::with_capacity(capacity),
            on_stack: HashSet::with_capacity(capacity),
        }
    }
}

// Integration with JIT optimization pipeline
impl CapabilityAnalysis {
    /// Check if an optimization is safe from a capability perspective
    pub fn is_optimization_safe(
        &self, 
        optimization_type: OptimizationType, 
        instruction_offset: u32
    ) -> bool {
        // Check if instruction has critical security constraints
        let has_critical_constraint = self.security_constraints.iter().any(|constraint| {
            constraint.location == instruction_offset && 
            constraint.severity >= ConstraintSeverity::High
        });
        
        if has_critical_constraint {
            return false;
        }
        
        // Check if optimization is explicitly marked as safe
        self.optimization_safety.safe_optimizations.iter().any(|safe_opt| {
            safe_opt.optimization_type == optimization_type &&
            safe_opt.applicable_instructions.contains(&instruction_offset)
        })
    }
    
    /// Get capability requirements for an instruction
    pub fn get_capability_requirements(&self, instruction_offset: u32) -> Option<&CapabilityRequirement> {
        self.instruction_requirements.get(&instruction_offset)
    }
    
    /// Get security constraints affecting an instruction
    pub fn get_security_constraints(&self, instruction_offset: u32) -> Vec<&SecurityConstraint> {
        self.security_constraints.iter()
            .filter(|constraint| constraint.location == instruction_offset)
            .collect()
    }
    
    /// Check if two instructions can be reordered safely
    pub fn can_reorder_instructions(&self, offset1: u32, offset2: u32) -> bool {
        // Check for information flow constraints between the instructions
        let has_flow_constraint = self.information_flows.iter().any(|flow| {
            (flow.source_instruction == offset1 && flow.target_instruction == offset2) ||
            (flow.source_instruction == offset2 && flow.target_instruction == offset1)
        });
        
        if has_flow_constraint {
            return false;
        }
        
        // Check for security boundary crossings
        let crosses_boundary = self.security_boundaries.iter().any(|boundary| {
            (offset1 < boundary.start_offset && offset2 >= boundary.start_offset) ||
            (offset2 < boundary.start_offset && offset1 >= boundary.start_offset)
        });
        
        !crosses_boundary
    }
} 