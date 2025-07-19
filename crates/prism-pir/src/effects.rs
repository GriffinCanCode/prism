//! PIR Effect System Integration - PLD-003 Compliance
//!
//! This module implements the complete effect system integration for PIR,
//! including hierarchical effect categories, capability-based security,
//! supply chain security, and object capability model from PLD-003.

use crate::{PIRError, PIRResult, semantic::{PrismIR, Effect, EffectGraph, EffectNode, EffectEdge, EffectEdgeType}};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, span, Level};

/// PIR Effect types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectType {
    /// Network I/O operations
    Network(NetworkEffect),
    /// File system operations  
    FileSystem(FileSystemEffect),
    /// Cryptographic operations
    Cryptography(CryptographyEffect),
    /// Database operations
    Database(DatabaseEffect),
    /// System operations
    System(SystemEffect),
    /// Security operations
    Security(SecurityEffect),
}

/// Hierarchical effect categories from PLD-003
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EffectCategory {
    /// I/O operations with specific subcategories
    IO(IOEffect),
    /// Cryptographic operations
    Cryptography(CryptographyEffect),
    /// Network operations
    Network(NetworkEffect),
    /// System operations
    System(SystemEffect),
    /// Database operations
    Database(DatabaseEffect),
    /// File system operations
    FileSystem(FileSystemEffect),
    /// Memory operations
    Memory(MemoryEffect),
    /// Time operations
    Time(TimeEffect),
    /// Security operations
    Security(SecurityEffect),
    /// AI operations (for AI-first design)
    AI(AIEffect),
}

/// I/O effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IOEffect {
    /// Read from input stream
    Read(IOResource),
    /// Write to output stream
    Write(IOResource),
    /// Flush buffers
    Flush(IOResource),
    /// Close resource
    Close(IOResource),
}

/// Cryptography effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CryptographyEffect {
    /// Generate cryptographic keys
    KeyGeneration(CryptoAlgorithm),
    /// Encrypt data
    Encryption(CryptoAlgorithm),
    /// Decrypt data
    Decryption(CryptoAlgorithm),
    /// Digital signing
    Signing(CryptoAlgorithm),
    /// Signature verification
    Verification(CryptoAlgorithm),
    /// Generate random data
    Random(EntropySource),
    /// Hash data
    Hashing(HashAlgorithm),
}

/// Network effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NetworkEffect {
    /// Connect to endpoint
    Connect(NetworkEndpoint),
    /// Listen on port
    Listen(NetworkPort),
    /// DNS resolution
    DNS(DomainName),
    /// HTTP request
    HTTP(HTTPMethod, NetworkEndpoint),
    /// WebSocket connection
    WebSocket(NetworkEndpoint),
    /// Send data
    Send(NetworkEndpoint),
    /// Receive data
    Receive(NetworkEndpoint),
}

/// System effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SystemEffect {
    /// Access environment variables
    Environment(EnvironmentVariable),
    /// Execute process
    Process(ProcessCommand),
    /// Allocate memory
    MemoryAllocation(MemorySize),
    /// System call
    SystemCall(SystemCallType),
    /// Signal handling
    Signal(SignalType),
}

/// Database effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DatabaseEffect {
    /// Read from database
    Read(DatabaseQuery),
    /// Write to database
    Write(DatabaseQuery),
    /// Transaction operations
    Transaction(TransactionType),
    /// Schema operations
    Schema(SchemaOperation),
    /// Connection management
    Connection(ConnectionOperation),
}

/// File system effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FileSystemEffect {
    /// Read file
    ReadFile(FilePath),
    /// Write file
    WriteFile(FilePath),
    /// Delete file
    DeleteFile(FilePath),
    /// Create directory
    CreateDirectory(DirectoryPath),
    /// List directory
    ListDirectory(DirectoryPath),
    /// Change permissions
    ChangePermissions(FilePath, FilePermissions),
}

/// Memory effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryEffect {
    /// Allocate memory
    Allocate(MemorySize),
    /// Deallocate memory
    Deallocate(MemoryAddress),
    /// Memory mapping
    Map(MemoryRegion),
    /// Memory protection
    Protect(MemoryRegion, MemoryProtection),
}

/// Time effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TimeEffect {
    /// Get current time
    GetCurrentTime,
    /// Sleep/delay
    Sleep(Duration),
    /// Set timer
    SetTimer(Duration),
    /// Get system uptime
    GetUptime,
}

/// Security effect subcategories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SecurityEffect {
    /// Authentication
    Authenticate(AuthenticationMethod),
    /// Authorization check
    Authorize(Permission),
    /// Audit logging
    Audit(AuditEvent),
    /// Security policy enforcement
    PolicyEnforcement(SecurityPolicy),
}

/// AI effect subcategories (AI-first design)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AIEffect {
    /// AI model inference
    Inference(AIModel),
    /// AI training
    Training(AIModel),
    /// AI data processing
    DataProcessing(AIDataType),
    /// AI safety analysis
    SafetyAnalysis(AISafetyCheck),
}

// Supporting types for effect categories

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct IOResource {
    pub resource_type: IOResourceType,
    pub identifier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IOResourceType {
    File,
    Network,
    Pipe,
    Device,
    Memory,
}

/// Cryptographic algorithm specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CryptoAlgorithm {
    pub algorithm: String,
    pub key_size: Option<u32>,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EntropySource {
    pub source_type: String,
    pub quality: EntropyQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntropyQuality {
    Low,
    Medium,
    High,
    Cryptographic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HashAlgorithm {
    pub algorithm: String,
    pub output_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NetworkEndpoint {
    pub host: String,
    pub port: u16,
    pub protocol: NetworkProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NetworkProtocol {
    TCP,
    UDP,
    HTTP,
    HTTPS,
    WebSocket,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NetworkPort {
    pub port: u16,
    pub protocol: NetworkProtocol,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DomainName {
    pub domain: String,
    pub record_type: DNSRecordType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DNSRecordType {
    A,
    AAAA,
    CNAME,
    MX,
    TXT,
    SRV,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HTTPMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EnvironmentVariable {
    pub name: String,
    pub access_type: EnvironmentAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EnvironmentAccess {
    Read,
    Write,
    ReadWrite,
}

/// Process command specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessCommand {
    pub command: String,
    pub arguments: Vec<String>,
    pub environment: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemorySize {
    pub bytes: u64,
    pub alignment: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SystemCallType {
    Open,
    Close,
    Read,
    Write,
    Fork,
    Exec,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SignalType {
    SIGTERM,
    SIGINT,
    SIGUSR1,
    SIGUSR2,
    Custom(i32),
}

// Database-related types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DatabaseQuery {
    pub query_type: QueryType,
    pub table: String,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
    Create,
    Drop,
    Alter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TransactionType {
    Begin,
    Commit,
    Rollback,
    Savepoint(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SchemaOperation {
    CreateTable(String),
    DropTable(String),
    AlterTable(String),
    CreateIndex(String),
    DropIndex(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConnectionOperation {
    Connect,
    Disconnect,
    Pool,
    Transaction,
}

// File system types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FilePath {
    pub path: String,
    pub access_mode: FileAccessMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FileAccessMode {
    Read,
    Write,
    ReadWrite,
    Append,
    Create,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DirectoryPath {
    pub path: String,
    pub recursive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FilePermissions {
    pub owner: PermissionSet,
    pub group: PermissionSet,
    pub other: PermissionSet,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PermissionSet {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

// Memory types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryAddress {
    pub address: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryRegion {
    pub start: u64,
    pub size: u64,
    pub flags: MemoryFlags,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MemoryFlags {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryProtection {
    ReadOnly,
    ReadWrite,
    ReadExecute,
    ReadWriteExecute,
    NoAccess,
}

// Time types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Duration {
    pub milliseconds: u64,
}

// Security types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuthenticationMethod {
    Password,
    Certificate,
    Token,
    Biometric,
    MultiFactory,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Permission {
    pub resource: String,
    pub action: String,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AuditEvent {
    pub event_type: String,
    pub actor: String,
    pub resource: String,
    pub action: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SecurityPolicy {
    pub policy_name: String,
    pub rules: Vec<SecurityRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SecurityRule {
    pub condition: String,
    pub action: SecurityAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SecurityAction {
    Allow,
    Deny,
    Audit,
    Require(String),
}

// AI types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AIModel {
    pub model_name: String,
    pub model_type: AIModelType,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AIModelType {
    LanguageModel,
    VisionModel,
    AudioModel,
    MultiModal,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AIDataType {
    Text,
    Image,
    Audio,
    Video,
    Structured,
    Unstructured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AISafetyCheck {
    PromptInjection,
    ContentFilter,
    BiasDetection,
    PrivacyAnalysis,
    Custom(String),
}

/// Capability-based security from PLD-003
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySystem {
    /// Object capabilities
    pub object_capabilities: HashMap<String, ObjectCapability>,
    /// Capability attenuation rules
    pub attenuation_rules: Vec<AttenuationRule>,
    /// Trust levels
    pub trust_levels: HashMap<String, TrustLevel>,
    /// Supply chain security policies
    pub supply_chain_policies: Vec<SupplyChainPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectCapability {
    /// Capability identifier
    pub id: String,
    /// Allowed effects
    pub allowed_effects: HashSet<EffectCategory>,
    /// Resource constraints
    pub resource_constraints: Vec<ResourceConstraint>,
    /// Time constraints
    pub time_constraints: Option<TimeConstraint>,
    /// Delegation rules
    pub delegation_rules: Vec<DelegationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttenuationRule {
    /// Source capability
    pub source_capability: String,
    /// Attenuated capability
    pub target_capability: String,
    /// Attenuation type
    pub attenuation_type: AttenuationType,
    /// Conditions for attenuation
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttenuationType {
    /// Reduce allowed effects
    EffectReduction(HashSet<EffectCategory>),
    /// Add resource constraints
    ResourceRestriction(Vec<ResourceConstraint>),
    /// Add time limits
    TimeRestriction(TimeConstraint),
    /// Combine multiple attenuations
    Combined(Vec<AttenuationType>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraint {
    /// Resource type
    pub resource_type: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint value
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Maximum usage
    MaxUsage,
    /// Rate limit
    RateLimit,
    /// Access pattern
    AccessPattern,
    /// Custom constraint
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraint {
    /// Start time
    pub start_time: Option<String>,
    /// End time
    pub end_time: Option<String>,
    /// Duration limit
    pub duration_limit: Option<Duration>,
    /// Usage window
    pub usage_window: Option<TimeWindow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start of window
    pub start: String,
    /// End of window
    pub end: String,
    /// Recurring pattern
    pub recurring: Option<RecurringPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecurringPattern {
    Daily,
    Weekly,
    Monthly,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationRule {
    /// Can delegate to
    pub delegate_to: Vec<String>,
    /// Delegation constraints
    pub constraints: Vec<DelegationConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelegationConstraint {
    /// Maximum delegation depth
    MaxDepth(u32),
    /// Required trust level
    RequiredTrust(TrustLevel),
    /// Time limit for delegation
    TimeLimit(Duration),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TrustLevel {
    /// No trust
    None,
    /// Basic trust
    Basic,
    /// Standard trust
    Standard,
    /// High trust
    High,
    /// Maximum trust
    Maximum,
}

/// Supply chain security from PLD-003
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyChainPolicy {
    /// Policy name
    pub name: String,
    /// Allowed dependencies
    pub allowed_dependencies: HashSet<DependencySpec>,
    /// Capability restrictions for dependencies
    pub dependency_restrictions: HashMap<String, Vec<CapabilityRestriction>>,
    /// Verification requirements
    pub verification_requirements: Vec<VerificationRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DependencySpec {
    /// Package name
    pub name: String,
    /// Version constraint
    pub version: VersionConstraint,
    /// Source repository
    pub source: Option<String>,
    /// Required signatures
    pub signatures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum VersionConstraint {
    /// Exact version
    Exact(String),
    /// Minimum version
    Minimum(String),
    /// Version range
    Range(String, String),
    /// Semantic version constraint
    Semantic(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRestriction {
    /// Restricted capability
    pub capability: String,
    /// Restriction type
    pub restriction_type: RestrictionType,
    /// Justification
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestrictionType {
    /// Completely forbidden
    Forbidden,
    /// Requires explicit approval
    RequiresApproval,
    /// Limited usage
    Limited(ResourceConstraint),
    /// Audit required
    AuditRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRequirement {
    /// Requirement type
    pub requirement_type: VerificationType,
    /// Required evidence
    pub required_evidence: Vec<String>,
    /// Verification method
    pub verification_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationType {
    /// Code signing
    CodeSigning,
    /// Reproducible builds
    ReproducibleBuild,
    /// Static analysis
    StaticAnalysis,
    /// Dynamic analysis
    DynamicAnalysis,
    /// Manual review
    ManualReview,
    /// Custom verification
    Custom(String),
}

/// Effect system builder for PIR
#[derive(Debug)]
pub struct EffectSystemBuilder {
    /// Effect categories registry
    effect_categories: HashMap<String, EffectCategory>,
    /// Capability system
    capability_system: CapabilitySystem,
    /// Effect relationships
    effect_relationships: HashMap<String, Vec<EffectRelationship>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRelationship {
    /// Source effect
    pub source: String,
    /// Target effect
    pub target: String,
    /// Relationship type
    pub relationship_type: EffectRelationshipType,
    /// Relationship strength
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectRelationshipType {
    /// One effect requires another
    Requires,
    /// One effect provides another
    Provides,
    /// Effects conflict with each other
    Conflicts,
    /// One effect enhances another
    Enhances,
    /// Effects are mutually exclusive
    MutuallyExclusive,
}

impl EffectSystemBuilder {
    /// Create a new effect system builder
    pub fn new() -> Self {
        Self {
            effect_categories: HashMap::new(),
            capability_system: CapabilitySystem {
                object_capabilities: HashMap::new(),
                attenuation_rules: Vec::new(),
                trust_levels: HashMap::new(),
                supply_chain_policies: Vec::new(),
            },
            effect_relationships: HashMap::new(),
        }
    }

    /// Build effect graph from PIR
    pub fn build_effect_graph(&mut self, pir: &PrismIR) -> PIRResult<EffectGraph> {
        let _span = span!(Level::DEBUG, "build_effect_graph").entered();
        debug!("Building effect graph from PIR");

        let mut nodes = HashMap::new();
        let mut edges = Vec::new();

        // Extract effects from all modules
        for module in &pir.modules {
            for effect in &module.effects {
                let node = EffectNode {
                    name: effect.name.clone(),
                    effect_type: effect.effect_type.clone(),
                    capabilities: self.extract_effect_capabilities(&effect.name)?,
                    side_effects: self.analyze_side_effects(&effect.name)?,
                };
                nodes.insert(effect.name.clone(), node);
            }

            // Build relationships between effects in the module
            for (i, effect1) in module.effects.iter().enumerate() {
                for effect2 in module.effects.iter().skip(i + 1) {
                    if let Some(relationship) = self.analyze_effect_relationship(&effect1.name, &effect2.name)? {
                        let edge_type = match relationship.relationship_type {
                            EffectRelationshipType::Requires => EffectEdgeType::Requires,
                            EffectRelationshipType::Provides => EffectEdgeType::Provides,
                            EffectRelationshipType::Conflicts => EffectEdgeType::Conflicts,
                            EffectRelationshipType::Enhances => EffectEdgeType::Enhances,
                            EffectRelationshipType::MutuallyExclusive => EffectEdgeType::Conflicts,
                        };

                        edges.push(EffectEdge {
                            source: effect1.name.clone(),
                            target: effect2.name.clone(),
                            edge_type,
                        });
                    }
                }
            }
        }

        Ok(EffectGraph { nodes, edges })
    }

    /// Register effect category
    pub fn register_effect_category(&mut self, name: String, category: EffectCategory) {
        self.effect_categories.insert(name, category);
    }

    /// Create object capability
    pub fn create_object_capability(
        &mut self,
        id: String,
        allowed_effects: HashSet<EffectCategory>,
    ) -> ObjectCapability {
        let capability = ObjectCapability {
            id: id.clone(),
            allowed_effects,
            resource_constraints: Vec::new(),
            time_constraints: None,
            delegation_rules: Vec::new(),
        };

        self.capability_system.object_capabilities.insert(id.clone(), capability.clone());
        capability
    }

    /// Add supply chain policy
    pub fn add_supply_chain_policy(&mut self, policy: SupplyChainPolicy) {
        self.capability_system.supply_chain_policies.push(policy);
    }

    /// Validate effect against capabilities
    pub fn validate_effect_capability(
        &self,
        effect: &EffectCategory,
        capability_id: &str,
    ) -> PIRResult<bool> {
        if let Some(capability) = self.capability_system.object_capabilities.get(capability_id) {
            Ok(capability.allowed_effects.contains(effect))
        } else {
            Err(PIRError::Internal {
                message: format!("Unknown capability: {}", capability_id),
            })
        }
    }

    // Helper methods
    fn extract_effect_capabilities(&self, effect_name: &str) -> PIRResult<Vec<String>> {
        // Analyze effect to determine required capabilities
        let capabilities = match effect_name {
            name if name.contains("file") => vec!["FileSystem".to_string()],
            name if name.contains("network") => vec!["Network".to_string()],
            name if name.contains("crypto") => vec!["Cryptography".to_string()],
            _ => vec!["General".to_string()],
        };
        Ok(capabilities)
    }

    fn analyze_side_effects(&self, effect_name: &str) -> PIRResult<Vec<String>> {
        // Analyze potential side effects
        let side_effects = match effect_name {
            name if name.contains("write") => vec!["StateModification".to_string()],
            name if name.contains("network") => vec!["NetworkCommunication".to_string()],
            name if name.contains("log") => vec!["Logging".to_string()],
            _ => Vec::new(),
        };
        Ok(side_effects)
    }

    fn analyze_effect_relationship(
        &self,
        effect1: &str,
        effect2: &str,
    ) -> PIRResult<Option<EffectRelationship>> {
        // Simple relationship analysis
        if effect1.contains("read") && effect2.contains("write") {
            return Ok(Some(EffectRelationship {
                source: effect1.to_string(),
                target: effect2.to_string(),
                relationship_type: EffectRelationshipType::Conflicts,
                strength: 0.8,
            }));
        }

        if effect1.contains("init") && effect2.contains("cleanup") {
            return Ok(Some(EffectRelationship {
                source: effect1.to_string(),
                target: effect2.to_string(),
                relationship_type: EffectRelationshipType::Requires,
                strength: 0.9,
            }));
        }

        Ok(None)
    }
}

impl Default for EffectSystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize default effect categories from PLD-003
pub fn initialize_default_effect_categories() -> HashMap<String, EffectCategory> {
    let mut categories = HashMap::new();

    // I/O effects
    categories.insert(
        "io_file_read".to_string(),
        EffectCategory::IO(IOEffect::Read(IOResource {
            resource_type: IOResourceType::File,
            identifier: "file".to_string(),
        })),
    );

    // Cryptography effects
    categories.insert(
        "crypto_key_gen".to_string(),
        EffectCategory::Cryptography(CryptographyEffect::KeyGeneration(CryptoAlgorithm {
            algorithm: "AES".to_string(),
            key_size: Some(256),
            parameters: HashMap::new(),
        })),
    );

    // Network effects
    categories.insert(
        "network_connect".to_string(),
        EffectCategory::Network(NetworkEffect::Connect(NetworkEndpoint {
            host: "localhost".to_string(),
            port: 8080,
            protocol: NetworkProtocol::TCP,
        })),
    );

    // System effects
    categories.insert(
        "system_env_read".to_string(),
        EffectCategory::System(SystemEffect::Environment(EnvironmentVariable {
            name: "PATH".to_string(),
            access_type: EnvironmentAccess::Read,
        })),
    );

    categories
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_category_serialization() {
        let effect = EffectCategory::IO(IOEffect::Read(IOResource {
            resource_type: IOResourceType::File,
            identifier: "test.txt".to_string(),
        }));

        let serialized = serde_json::to_string(&effect).unwrap();
        let deserialized: EffectCategory = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(effect, deserialized);
    }

    #[test]
    fn test_capability_system() {
        let mut builder = EffectSystemBuilder::new();
        
        let mut allowed_effects = HashSet::new();
        allowed_effects.insert(EffectCategory::IO(IOEffect::Read(IOResource {
            resource_type: IOResourceType::File,
            identifier: "test".to_string(),
        })));

        let capability = builder.create_object_capability("test_cap".to_string(), allowed_effects);
        
        assert_eq!(capability.id, "test_cap");
        assert_eq!(capability.allowed_effects.len(), 1);
    }

    #[test]
    fn test_effect_system_builder() {
        let mut builder = EffectSystemBuilder::new();
        
        // Create a simple PIR for testing
        let pir = crate::types::PrismIR::new();
        
        let result = builder.build_effect_graph(&pir);
        assert!(result.is_ok());
        
        let graph = result.unwrap();
        assert!(graph.nodes.is_empty()); // Empty PIR should produce empty graph
    }

    #[test]
    fn test_default_effect_categories() {
        let categories = initialize_default_effect_categories();
        
        assert!(!categories.is_empty());
        assert!(categories.contains_key("io_file_read"));
        assert!(categories.contains_key("crypto_key_gen"));
        assert!(categories.contains_key("network_connect"));
        assert!(categories.contains_key("system_env_read"));
    }
} 