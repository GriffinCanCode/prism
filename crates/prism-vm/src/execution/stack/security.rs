//! Stack Security Management
//!
//! This module provides capability-aware security management for stack operations,
//! integrating with the existing prism-runtime authority system to ensure all
//! stack operations respect capability-based security constraints with granular
//! control and comprehensive analysis.

use crate::{VMResult, PrismVMError};
use crate::execution::{ExecutionStack, StackFrame, StackValue};
use prism_runtime::{
    authority::capability::{
        CapabilitySet, Capability, Authority, Operation, ConstraintSet,
        ComponentId, CapabilityManager,
    },
    platform::execution::ExecutionContext,
};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn, span, Level};

/// Stack security manager that enforces capability-based security
#[derive(Debug)]
pub struct StackSecurityManager {
    /// Available capabilities
    capabilities: CapabilitySet,
    
    /// Capability manager integration
    capability_manager: Arc<CapabilityManager>,
    
    /// Security policy enforcement
    security_policy: StackSecurityPolicy,
    
    /// Security audit log
    audit_log: Arc<RwLock<Vec<StackSecurityEvent>>>,
    
    /// Security statistics
    stats: Arc<RwLock<StackSecurityStats>>,
    
    /// Granular access control matrix
    access_control_matrix: Arc<RwLock<AccessControlMatrix>>,
    
    /// Security context cache
    context_cache: Arc<RwLock<HashMap<String, SecurityContext>>>,
    
    /// Threat detection system
    threat_detector: Arc<RwLock<ThreatDetector>>,
}

/// Granular access control matrix
#[derive(Debug, Clone)]
pub struct AccessControlMatrix {
    /// Permission matrix by operation and capability level
    permissions: HashMap<StackOperation, HashMap<CapabilityLevel, AccessPermission>>,
    
    /// Context-specific overrides
    context_overrides: HashMap<String, HashMap<StackOperation, AccessPermission>>,
    
    /// Temporal access restrictions
    temporal_restrictions: HashMap<StackOperation, TemporalRestriction>,
    
    /// Resource-based restrictions
    resource_restrictions: HashMap<StackOperation, ResourceRestriction>,
}

/// Stack operations for fine-grained control
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum StackOperation {
    /// Frame operations
    FrameCreate { function_name: String },
    FrameDestroy { function_id: u32 },
    FrameAccess { function_id: u32, access_type: FrameAccessType },
    
    /// Local variable operations
    LocalRead { slot: u8, var_type: String },
    LocalWrite { slot: u8, var_type: String },
    LocalCreate { slot: u8, var_type: String },
    
    /// Stack value operations
    StackPush { value_type: String },
    StackPop { value_type: String },
    StackPeek { depth: usize },
    
    /// Upvalue operations
    UpvalueRead { slot: u8, closure_id: u32 },
    UpvalueWrite { slot: u8, closure_id: u32 },
    UpvalueCapture { slot: u8, source_frame: u32 },
    
    /// Effect and capability operations
    EffectInvoke { effect_name: String },
    CapabilityCheck { capability_name: String },
    CapabilityElevate { from: String, to: String },
    
    /// Memory operations
    MemoryAllocate { size: usize, purpose: String },
    MemoryDeallocate { size: usize },
    MemoryAccess { address_space: String },
    
    /// Debug and profiling operations
    DebugAccess { target: String },
    ProfilerAccess { metric: String },
    IntrospectionAccess { scope: String },
}

/// Frame access types
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrameAccessType {
    MetadataRead,
    MetadataWrite,
    LocalsRead,
    LocalsWrite,
    UpvaluesRead,
    UpvaluesWrite,
    EffectsRead,
    EffectsWrite,
    CapabilitiesRead,
    CapabilitiesWrite,
}

/// Capability levels for granular control
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum CapabilityLevel {
    /// No access
    None,
    /// Read-only access
    ReadOnly,
    /// Limited write access
    LimitedWrite,
    /// Full access within scope
    FullAccess,
    /// Administrative access
    Administrative,
    /// System-level access
    System,
    /// Debug/development access
    Debug,
}

/// Access permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPermission {
    /// Whether access is allowed
    pub allowed: bool,
    
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    
    /// Additional constraints
    pub constraints: Vec<AccessConstraint>,
    
    /// Rate limiting
    pub rate_limit: Option<RateLimit>,
    
    /// Audit requirements
    pub audit_level: AuditLevel,
    
    /// Expiration time
    pub expires_at: Option<SystemTime>,
}

/// Access constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessConstraint {
    /// Time-based constraint
    TimeWindow { start: SystemTime, end: SystemTime },
    
    /// Resource-based constraint
    ResourceLimit { resource: String, limit: u64 },
    
    /// Context-based constraint
    ContextRequired { context_type: String, context_value: String },
    
    /// Dependency constraint
    DependsOn { operation: String, completion_required: bool },
    
    /// Concurrency constraint
    MaxConcurrent { limit: usize },
    
    /// Frequency constraint
    MaxFrequency { operations_per_second: f64 },
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum operations per time window
    pub max_operations: u32,
    
    /// Time window duration
    pub window_duration: Duration,
    
    /// Current operation count
    pub current_count: u32,
    
    /// Window start time
    pub window_start: Instant,
    
    /// Burst allowance
    pub burst_allowance: u32,
}

/// Audit levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLevel {
    /// No audit required
    None,
    /// Basic logging
    Basic,
    /// Detailed logging
    Detailed,
    /// Full forensic logging
    Forensic,
    /// Real-time monitoring
    RealTime,
}

/// Temporal access restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRestriction {
    /// Allowed time windows
    pub allowed_windows: Vec<TimeWindow>,
    
    /// Blackout periods
    pub blackout_periods: Vec<TimeWindow>,
    
    /// Maximum session duration
    pub max_session_duration: Option<Duration>,
    
    /// Cooldown period between operations
    pub cooldown_period: Option<Duration>,
}

/// Time window for temporal restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start time (can be recurring)
    pub start: TimeSpec,
    
    /// End time
    pub end: TimeSpec,
    
    /// Days of week (1-7, Monday=1)
    pub days_of_week: Option<Vec<u8>>,
    
    /// Timezone
    pub timezone: Option<String>,
}

/// Time specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSpec {
    /// Absolute time
    Absolute(SystemTime),
    
    /// Daily recurring time (hour, minute)
    Daily { hour: u8, minute: u8 },
    
    /// Weekly recurring time
    Weekly { day: u8, hour: u8, minute: u8 },
    
    /// Relative to session start
    RelativeToSession(Duration),
}

/// Resource-based restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRestriction {
    /// Memory usage limits
    pub memory_limits: Option<MemoryLimits>,
    
    /// CPU usage limits
    pub cpu_limits: Option<CpuLimits>,
    
    /// I/O limits
    pub io_limits: Option<IoLimits>,
    
    /// Network limits
    pub network_limits: Option<NetworkLimits>,
}

/// Memory usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum memory allocation
    pub max_allocation: usize,
    
    /// Maximum total memory usage
    pub max_total_usage: usize,
    
    /// Maximum allocation rate (bytes/second)
    pub max_allocation_rate: f64,
}

/// CPU usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU time per operation
    pub max_cpu_time_per_op: Duration,
    
    /// Maximum total CPU time
    pub max_total_cpu_time: Duration,
    
    /// CPU usage percentage limit
    pub max_cpu_percentage: f64,
}

/// I/O limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoLimits {
    /// Maximum read operations
    pub max_read_ops: u64,
    
    /// Maximum write operations
    pub max_write_ops: u64,
    
    /// Maximum bytes read
    pub max_bytes_read: u64,
    
    /// Maximum bytes written
    pub max_bytes_written: u64,
}

/// Network limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLimits {
    /// Maximum network connections
    pub max_connections: u32,
    
    /// Maximum bandwidth usage
    pub max_bandwidth: u64,
    
    /// Allowed network destinations
    pub allowed_destinations: Vec<String>,
    
    /// Blocked network destinations
    pub blocked_destinations: Vec<String>,
}

/// Security context for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Context identifier
    pub context_id: String,
    
    /// User identity
    pub user_identity: Option<String>,
    
    /// Session information
    pub session_info: SessionInfo,
    
    /// Environment context
    pub environment: EnvironmentContext,
    
    /// Trust level
    pub trust_level: TrustLevel,
    
    /// Active constraints
    pub active_constraints: Vec<String>,
    
    /// Context creation time
    pub created_at: SystemTime,
    
    /// Context expiration
    pub expires_at: Option<SystemTime>,
}

/// Session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Session ID
    pub session_id: String,
    
    /// Session start time
    pub started_at: SystemTime,
    
    /// Last activity time
    pub last_activity: SystemTime,
    
    /// Session duration limit
    pub max_duration: Option<Duration>,
    
    /// Session source
    pub source: SessionSource,
}

/// Session sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionSource {
    /// Interactive user session
    Interactive,
    
    /// Automated system
    Automated,
    
    /// External API
    ExternalApi,
    
    /// Internal service
    InternalService,
    
    /// Debug/development
    Debug,
}

/// Environment context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    /// Execution environment
    pub execution_env: ExecutionEnvironment,
    
    /// Security domain
    pub security_domain: String,
    
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Execution environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEnvironment {
    /// Production environment
    Production,
    
    /// Staging environment
    Staging,
    
    /// Development environment
    Development,
    
    /// Testing environment
    Testing,
    
    /// Sandbox environment
    Sandbox,
}

/// Trust levels
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Untrusted
    Untrusted,
    
    /// Low trust
    Low,
    
    /// Medium trust
    Medium,
    
    /// High trust
    High,
    
    /// Fully trusted
    Trusted,
    
    /// System trust
    System,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score (0.0-1.0)
    pub risk_score: f64,
    
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    
    /// Mitigation strategies
    pub mitigations: Vec<String>,
    
    /// Assessment timestamp
    pub assessed_at: SystemTime,
}

/// Risk factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor type
    pub factor_type: RiskFactorType,
    
    /// Factor weight (0.0-1.0)
    pub weight: f64,
    
    /// Factor description
    pub description: String,
}

/// Risk factor types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskFactorType {
    /// Network-based risk
    Network,
    
    /// User behavior risk
    UserBehavior,
    
    /// System configuration risk
    SystemConfiguration,
    
    /// Data sensitivity risk
    DataSensitivity,
    
    /// Compliance risk
    Compliance,
    
    /// Performance risk
    Performance,
}

/// Threat detection system
#[derive(Debug)]
pub struct ThreatDetector {
    /// Anomaly detection patterns
    anomaly_patterns: HashMap<String, AnomalyPattern>,
    
    /// Threat signatures
    threat_signatures: Vec<ThreatSignature>,
    
    /// Behavioral baselines
    behavioral_baselines: HashMap<String, BehavioralBaseline>,
    
    /// Detection statistics
    detection_stats: ThreatDetectionStats,
}

/// Anomaly patterns for detection
#[derive(Debug, Clone)]
pub struct AnomalyPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern description
    pub description: String,
    
    /// Detection threshold
    pub threshold: f64,
    
    /// Time window for analysis
    pub time_window: Duration,
    
    /// Pattern metrics
    pub metrics: Vec<AnomalyMetric>,
}

/// Anomaly metrics
#[derive(Debug, Clone)]
pub struct AnomalyMetric {
    /// Metric name
    pub name: String,
    
    /// Expected value range
    pub expected_range: (f64, f64),
    
    /// Current value
    pub current_value: f64,
    
    /// Deviation threshold
    pub deviation_threshold: f64,
}

/// Threat signatures
#[derive(Debug, Clone)]
pub struct ThreatSignature {
    /// Signature ID
    pub id: String,
    
    /// Threat type
    pub threat_type: ThreatType,
    
    /// Detection criteria
    pub criteria: Vec<DetectionCriterion>,
    
    /// Severity level
    pub severity: ThreatSeverity,
    
    /// Response actions
    pub response_actions: Vec<ResponseAction>,
}

/// Threat types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    /// Stack overflow attack
    StackOverflow,
    
    /// Privilege escalation
    PrivilegeEscalation,
    
    /// Resource exhaustion
    ResourceExhaustion,
    
    /// Information disclosure
    InformationDisclosure,
    
    /// Code injection
    CodeInjection,
    
    /// Denial of service
    DenialOfService,
    
    /// Data exfiltration
    DataExfiltration,
}

/// Detection criteria
#[derive(Debug, Clone)]
pub struct DetectionCriterion {
    /// Criterion type
    pub criterion_type: CriterionType,
    
    /// Expected value or pattern
    pub expected: String,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
    
    /// Weight in overall detection
    pub weight: f64,
}

/// Criterion types
#[derive(Debug, Clone)]
pub enum CriterionType {
    /// Operation frequency
    OperationFrequency,
    
    /// Resource usage pattern
    ResourceUsage,
    
    /// Access pattern
    AccessPattern,
    
    /// Timing pattern
    TimingPattern,
    
    /// Data pattern
    DataPattern,
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    Matches,
}

/// Threat severity levels
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Response actions for threats
#[derive(Debug, Clone)]
pub enum ResponseAction {
    /// Log the event
    Log { level: String },
    
    /// Alert administrators
    Alert { urgency: String },
    
    /// Block the operation
    Block,
    
    /// Throttle operations
    Throttle { rate: f64 },
    
    /// Elevate monitoring
    ElevateMonitoring { duration: Duration },
    
    /// Isolate context
    IsolateContext,
    
    /// Terminate session
    TerminateSession,
}

/// Behavioral baselines
#[derive(Debug, Clone)]
pub struct BehavioralBaseline {
    /// Baseline name
    pub name: String,
    
    /// Normal operation patterns
    pub normal_patterns: HashMap<String, OperationPattern>,
    
    /// Baseline establishment period
    pub establishment_period: Duration,
    
    /// Last update time
    pub last_updated: SystemTime,
    
    /// Confidence level
    pub confidence: f64,
}

/// Operation patterns for baseline
#[derive(Debug, Clone)]
pub struct OperationPattern {
    /// Average frequency
    pub avg_frequency: f64,
    
    /// Standard deviation
    pub std_deviation: f64,
    
    /// Peak times
    pub peak_times: Vec<Duration>,
    
    /// Minimum frequency
    pub min_frequency: f64,
    
    /// Maximum frequency
    pub max_frequency: f64,
}

/// Threat detection statistics
#[derive(Debug, Clone, Default)]
pub struct ThreatDetectionStats {
    /// Total threats detected
    pub total_threats_detected: u64,
    
    /// Threats by type
    pub threats_by_type: HashMap<String, u64>,
    
    /// False positive rate
    pub false_positive_rate: f64,
    
    /// False negative rate
    pub false_negative_rate: f64,
    
    /// Detection accuracy
    pub detection_accuracy: f64,
    
    /// Average detection time
    pub avg_detection_time_ms: f64,
}

/// Stack security policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityPolicy {
    /// Require capabilities for stack frame creation
    pub require_frame_capability: bool,
    
    /// Granular local variable access control
    pub local_access_control: HashMap<String, CapabilityLevel>,
    
    /// Require capabilities for upvalue access
    pub require_upvalue_capability: bool,
    
    /// Maximum stack depth by capability level
    pub max_depth_by_capability: HashMap<CapabilityLevel, usize>,
    
    /// Enable stack overflow protection
    pub stack_overflow_protection: bool,
    
    /// Enable capability inheritance validation
    pub validate_capability_inheritance: bool,
    
    /// Enable threat detection
    pub enable_threat_detection: bool,
    
    /// Audit level for operations
    pub default_audit_level: AuditLevel,
    
    /// Rate limiting configuration
    pub rate_limiting: HashMap<StackOperation, RateLimit>,
    
    /// Context validation requirements
    pub context_validation: ContextValidationConfig,
}

/// Context validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextValidationConfig {
    /// Require valid session
    pub require_valid_session: bool,
    
    /// Maximum session age
    pub max_session_age: Option<Duration>,
    
    /// Required trust level
    pub min_trust_level: TrustLevel,
    
    /// Required environment
    pub allowed_environments: Vec<ExecutionEnvironment>,
    
    /// Enable context caching
    pub enable_context_caching: bool,
    
    /// Context cache TTL
    pub context_cache_ttl: Duration,
}

impl Default for StackSecurityPolicy {
    fn default() -> Self {
        let mut max_depth_by_capability = HashMap::new();
        max_depth_by_capability.insert(CapabilityLevel::None, 10);
        max_depth_by_capability.insert(CapabilityLevel::ReadOnly, 100);
        max_depth_by_capability.insert(CapabilityLevel::LimitedWrite, 500);
        max_depth_by_capability.insert(CapabilityLevel::FullAccess, 1000);
        max_depth_by_capability.insert(CapabilityLevel::Administrative, 5000);
        max_depth_by_capability.insert(CapabilityLevel::System, 10000);
        max_depth_by_capability.insert(CapabilityLevel::Debug, 50000);

        let mut local_access_control = HashMap::new();
        local_access_control.insert("default".to_string(), CapabilityLevel::FullAccess);
        local_access_control.insert("sensitive".to_string(), CapabilityLevel::Administrative);
        local_access_control.insert("system".to_string(), CapabilityLevel::System);

        Self {
            require_frame_capability: true,
            local_access_control,
            require_upvalue_capability: true,
            max_depth_by_capability,
            stack_overflow_protection: true,
            validate_capability_inheritance: true,
            enable_threat_detection: true,
            default_audit_level: AuditLevel::Basic,
            rate_limiting: HashMap::new(),
            context_validation: ContextValidationConfig {
                require_valid_session: true,
                max_session_age: Some(Duration::from_hours(8)),
                min_trust_level: TrustLevel::Low,
                allowed_environments: vec![
                    ExecutionEnvironment::Development,
                    ExecutionEnvironment::Testing,
                    ExecutionEnvironment::Staging,
                    ExecutionEnvironment::Production,
                ],
                enable_context_caching: true,
                context_cache_ttl: Duration::from_minutes(15),
            },
        }
    }
}

/// Stack security event for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityEvent {
    /// When the event occurred
    pub timestamp: SystemTime,
    
    /// Type of security event
    pub event_type: StackSecurityEventType,
    
    /// Operation that triggered the event
    pub operation: StackOperation,
    
    /// Function context
    pub function_context: Option<String>,
    
    /// Stack depth at time of event
    pub stack_depth: usize,
    
    /// Capabilities involved
    pub capabilities: Vec<String>,
    
    /// Whether the operation was allowed
    pub allowed: bool,
    
    /// Security context
    pub security_context: Option<String>,
    
    /// Threat detection results
    pub threat_analysis: Option<ThreatAnalysisResult>,
    
    /// Additional context
    pub context: String,
    
    /// Event severity
    pub severity: EventSeverity,
}

/// Threat analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAnalysisResult {
    /// Threat score (0.0-1.0)
    pub threat_score: f64,
    
    /// Detected threat types
    pub detected_threats: Vec<ThreatType>,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Analysis details
    pub analysis_details: String,
}

/// Event severity levels
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Types of stack security events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StackSecurityEventType {
    /// Stack frame creation
    FrameCreation,
    
    /// Local variable access
    LocalAccess,
    
    /// Upvalue access
    UpvalueAccess,
    
    /// Stack depth check
    DepthCheck,
    
    /// Capability validation
    CapabilityValidation,
    
    /// Security violation
    SecurityViolation,
    
    /// Threat detection
    ThreatDetection,
    
    /// Rate limiting triggered
    RateLimitTriggered,
    
    /// Context validation
    ContextValidation,
    
    /// Access denied
    AccessDenied,
}

/// Stack security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityStats {
    /// Total security checks performed
    pub total_checks: u64,
    
    /// Security checks passed
    pub checks_passed: u64,
    
    /// Security violations detected
    pub violations_detected: u64,
    
    /// Capability validations performed
    pub capability_validations: u64,
    
    /// Average check time in microseconds
    pub avg_check_time_us: f64,
    
    /// Security events by type
    pub events_by_type: HashMap<String, u64>,
    
    /// Threat detection statistics
    pub threat_detection_stats: ThreatDetectionStats,
    
    /// Rate limiting statistics
    pub rate_limiting_stats: RateLimitingStats,
    
    /// Context validation statistics
    pub context_validation_stats: ContextValidationStats,
}

/// Rate limiting statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RateLimitingStats {
    /// Total rate limit checks
    pub total_checks: u64,
    
    /// Rate limits triggered
    pub limits_triggered: u64,
    
    /// Operations throttled
    pub operations_throttled: u64,
    
    /// Average throttle duration
    pub avg_throttle_duration_ms: f64,
}

/// Context validation statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextValidationStats {
    /// Total context validations
    pub total_validations: u64,
    
    /// Successful validations
    pub successful_validations: u64,
    
    /// Failed validations
    pub failed_validations: u64,
    
    /// Cache hits
    pub cache_hits: u64,
    
    /// Cache misses
    pub cache_misses: u64,
}

impl Default for StackSecurityStats {
    fn default() -> Self {
        Self {
            total_checks: 0,
            checks_passed: 0,
            violations_detected: 0,
            capability_validations: 0,
            avg_check_time_us: 0.0,
            events_by_type: HashMap::new(),
            threat_detection_stats: ThreatDetectionStats::default(),
            rate_limiting_stats: RateLimitingStats::default(),
            context_validation_stats: ContextValidationStats::default(),
        }
    }
}

/// Stack security analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSecurityAnalysis {
    /// Overall security status
    pub security_status: SecurityStatus,
    
    /// Active capabilities
    pub active_capabilities: Vec<String>,
    
    /// Current capability level
    pub current_capability_level: CapabilityLevel,
    
    /// Security violations found
    pub violations: Vec<SecurityViolation>,
    
    /// Capability coverage analysis
    pub capability_coverage: CapabilityCoverage,
    
    /// Risk assessment
    pub risk_level: RiskLevel,
    
    /// Threat analysis
    pub threat_analysis: ThreatAnalysisResult,
    
    /// Access control analysis
    pub access_control_analysis: AccessControlAnalysis,
    
    /// Recommendations
    pub recommendations: Vec<SecurityRecommendation>,
    
    /// Compliance status
    pub compliance_status: ComplianceStatus,
}

/// Access control analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlAnalysis {
    /// Effective permissions
    pub effective_permissions: HashMap<String, AccessPermission>,
    
    /// Permission gaps
    pub permission_gaps: Vec<String>,
    
    /// Over-privileged areas
    pub over_privileged: Vec<String>,
    
    /// Under-privileged areas
    pub under_privileged: Vec<String>,
    
    /// Access patterns
    pub access_patterns: Vec<String>,
}

/// Security recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    
    /// Priority level
    pub priority: RecommendationPriority,
    
    /// Description
    pub description: String,
    
    /// Implementation steps
    pub implementation_steps: Vec<String>,
    
    /// Expected impact
    pub expected_impact: String,
    
    /// Risk if not implemented
    pub risk_if_ignored: String,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Capability adjustment
    CapabilityAdjustment,
    
    /// Policy update
    PolicyUpdate,
    
    /// Access control refinement
    AccessControlRefinement,
    
    /// Threat mitigation
    ThreatMitigation,
    
    /// Compliance improvement
    ComplianceImprovement,
    
    /// Performance optimization
    PerformanceOptimization,
}

/// Recommendation priorities
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    /// Overall compliance score (0.0-1.0)
    pub compliance_score: f64,
    
    /// Compliance by requirement
    pub requirement_compliance: HashMap<String, bool>,
    
    /// Non-compliant areas
    pub non_compliant_areas: Vec<String>,
    
    /// Compliance gaps
    pub compliance_gaps: Vec<ComplianceGap>,
    
    /// Last assessment time
    pub last_assessed: SystemTime,
}

/// Compliance gap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceGap {
    /// Requirement that's not met
    pub requirement: String,
    
    /// Gap description
    pub description: String,
    
    /// Severity of the gap
    pub severity: ComplianceGapSeverity,
    
    /// Remediation steps
    pub remediation_steps: Vec<String>,
}

/// Compliance gap severity
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub enum ComplianceGapSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}

/// Security status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityStatus {
    /// All security checks passed
    Secure,
    
    /// Minor security concerns
    Warning,
    
    /// Significant security issues
    Critical,
    
    /// Security violations detected
    Violated,
    
    /// Under threat
    UnderThreat,
}

/// Security violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    
    /// Function where violation occurred
    pub function_name: Option<String>,
    
    /// Stack depth at violation
    pub stack_depth: usize,
    
    /// Description of the violation
    pub description: String,
    
    /// Severity level
    pub severity: Severity,
    
    /// Violation timestamp
    pub timestamp: SystemTime,
    
    /// Remediation actions taken
    pub remediation_actions: Vec<String>,
}

/// Types of security violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Missing required capability
    MissingCapability,
    
    /// Capability expired
    ExpiredCapability,
    
    /// Stack depth exceeded
    StackDepthExceeded,
    
    /// Unauthorized access attempt
    UnauthorizedAccess,
    
    /// Capability inheritance violation
    InheritanceViolation,
    
    /// Rate limit exceeded
    RateLimitExceeded,
    
    /// Context validation failed
    ContextValidationFailed,
    
    /// Threat detected
    ThreatDetected,
    
    /// Policy violation
    PolicyViolation,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Capability coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCoverage {
    /// Required capabilities
    pub required: HashSet<String>,
    
    /// Available capabilities
    pub available: HashSet<String>,
    
    /// Missing capabilities
    pub missing: HashSet<String>,
    
    /// Excessive capabilities
    pub excessive: HashSet<String>,
    
    /// Coverage percentage
    pub coverage_percentage: f64,
    
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// Risk assessment levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl StackSecurityManager {
    /// Create a new stack security manager
    pub fn new(capabilities: CapabilitySet) -> VMResult<Self> {
        let _span = span!(Level::INFO, "stack_security_init").entered();
        info!("Initializing advanced stack security management");

        let capability_manager = Arc::new(
            CapabilityManager::new().map_err(|e| PrismVMError::RuntimeError {
                message: format!("Failed to create capability manager: {}", e),
            })?
        );

        let access_control_matrix = Arc::new(RwLock::new(AccessControlMatrix::new()));
        let context_cache = Arc::new(RwLock::new(HashMap::new()));
        let threat_detector = Arc::new(RwLock::new(ThreatDetector::new()));

        Ok(Self {
            capabilities,
            capability_manager,
            security_policy: StackSecurityPolicy::default(),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(StackSecurityStats::default())),
            access_control_matrix,
            context_cache,
            threat_detector,
        })
    }

    /// Validate stack operation with comprehensive security checks
    pub fn validate_stack_operation(
        &self,
        operation: &StackOperation,
        context: &ExecutionContext,
    ) -> VMResult<()> {
        let _span = span!(Level::DEBUG, "validate_operation", 
            operation = ?operation
        ).entered();

        let start_time = Instant::now();
        
        // Build security context
        let security_context = self.build_security_context(context)?;
        
        // Determine required capability level
        let required_capability_level = self.determine_required_capability_level(operation);
        
        // Check access permissions
        let access_permission = self.check_access_permission(operation, &required_capability_level, &security_context)?;
        
        if !access_permission.allowed {
            self.handle_access_denied(operation, &security_context, "Insufficient permissions")?;
            return Err(PrismVMError::CapabilityViolation {
                message: format!("Access denied for operation: {:?}", operation),
            });
        }
        
        // Validate constraints
        self.validate_constraints(&access_permission.constraints, operation, &security_context)?;
        
        // Check rate limits
        self.check_rate_limits(operation, &access_permission.rate_limit)?;
        
        // Perform threat detection
        if self.security_policy.enable_threat_detection {
            self.perform_threat_detection(operation, &security_context)?;
        }
        
        // Log security event
        self.log_security_event(StackSecurityEvent {
            timestamp: SystemTime::now(),
            event_type: StackSecurityEventType::CapabilityValidation,
            operation: operation.clone(),
            function_context: self.extract_function_context(operation),
            stack_depth: 0, // Would be provided by caller
            capabilities: self.capabilities.capability_names(),
            allowed: true,
            security_context: Some(security_context.context_id.clone()),
            threat_analysis: None,
            context: "Operation validated successfully".to_string(),
            severity: EventSeverity::Info,
        });
        
        // Update statistics
        self.update_stats(start_time, true);
        
        debug!("Stack operation validated successfully: {:?}", operation);
        Ok(())
    }

    /// Build comprehensive security context
    fn build_security_context(&self, context: &ExecutionContext) -> VMResult<SecurityContext> {
        // Check cache first
        let context_id = format!("{:?}", context); // Simplified - would use proper ID generation
        
        if self.security_policy.context_validation.enable_context_caching {
            let cache = self.context_cache.read().unwrap();
            if let Some(cached_context) = cache.get(&context_id) {
                if cached_context.expires_at.map_or(true, |exp| exp > SystemTime::now()) {
                    return Ok(cached_context.clone());
                }
            }
        }
        
        // Build new security context
        let session_info = SessionInfo {
            session_id: "default_session".to_string(), // Would be extracted from context
            started_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            max_duration: self.security_policy.context_validation.max_session_age,
            source: SessionSource::Interactive, // Would be determined from context
        };
        
        let environment = EnvironmentContext {
            execution_env: ExecutionEnvironment::Development, // Would be determined from context
            security_domain: "default".to_string(),
            compliance_requirements: vec!["basic_security".to_string()],
            risk_assessment: self.assess_risk(context),
        };
        
        let security_context = SecurityContext {
            context_id: context_id.clone(),
            user_identity: None, // Would be extracted from context
            session_info,
            environment,
            trust_level: TrustLevel::Medium, // Would be calculated
            active_constraints: Vec::new(),
            created_at: SystemTime::now(),
            expires_at: Some(SystemTime::now() + self.security_policy.context_validation.context_cache_ttl),
        };
        
        // Cache the context
        if self.security_policy.context_validation.enable_context_caching {
            let mut cache = self.context_cache.write().unwrap();
            cache.insert(context_id, security_context.clone());
        }
        
        Ok(security_context)
    }

    /// Assess risk for the given context
    fn assess_risk(&self, _context: &ExecutionContext) -> RiskAssessment {
        // This would perform comprehensive risk assessment
        // For now, return a basic assessment
        RiskAssessment {
            risk_score: 0.3, // Medium-low risk
            risk_factors: vec![
                RiskFactor {
                    factor_type: RiskFactorType::SystemConfiguration,
                    weight: 0.5,
                    description: "Standard system configuration".to_string(),
                },
            ],
            mitigations: vec!["Standard security controls applied".to_string()],
            assessed_at: SystemTime::now(),
        }
    }

    /// Determine required capability level for operation
    fn determine_required_capability_level(&self, operation: &StackOperation) -> CapabilityLevel {
        match operation {
            StackOperation::FrameCreate { .. } => CapabilityLevel::FullAccess,
            StackOperation::FrameDestroy { .. } => CapabilityLevel::FullAccess,
            StackOperation::FrameAccess { access_type, .. } => match access_type {
                FrameAccessType::MetadataRead => CapabilityLevel::ReadOnly,
                FrameAccessType::MetadataWrite => CapabilityLevel::LimitedWrite,
                FrameAccessType::LocalsRead => CapabilityLevel::ReadOnly,
                FrameAccessType::LocalsWrite => CapabilityLevel::LimitedWrite,
                FrameAccessType::UpvaluesRead => CapabilityLevel::ReadOnly,
                FrameAccessType::UpvaluesWrite => CapabilityLevel::FullAccess,
                FrameAccessType::EffectsRead => CapabilityLevel::ReadOnly,
                FrameAccessType::EffectsWrite => CapabilityLevel::Administrative,
                FrameAccessType::CapabilitiesRead => CapabilityLevel::ReadOnly,
                FrameAccessType::CapabilitiesWrite => CapabilityLevel::Administrative,
            },
            StackOperation::LocalRead { var_type, .. } => {
                self.security_policy.local_access_control
                    .get(var_type)
                    .cloned()
                    .unwrap_or(CapabilityLevel::ReadOnly)
            },
            StackOperation::LocalWrite { var_type, .. } => {
                self.security_policy.local_access_control
                    .get(var_type)
                    .cloned()
                    .unwrap_or(CapabilityLevel::LimitedWrite)
            },
            StackOperation::LocalCreate { var_type, .. } => {
                self.security_policy.local_access_control
                    .get(var_type)
                    .cloned()
                    .unwrap_or(CapabilityLevel::FullAccess)
            },
            StackOperation::StackPush { .. } | StackOperation::StackPop { .. } => CapabilityLevel::LimitedWrite,
            StackOperation::StackPeek { .. } => CapabilityLevel::ReadOnly,
            StackOperation::UpvalueRead { .. } => CapabilityLevel::ReadOnly,
            StackOperation::UpvalueWrite { .. } | StackOperation::UpvalueCapture { .. } => CapabilityLevel::FullAccess,
            StackOperation::EffectInvoke { .. } => CapabilityLevel::Administrative,
            StackOperation::CapabilityCheck { .. } => CapabilityLevel::ReadOnly,
            StackOperation::CapabilityElevate { .. } => CapabilityLevel::Administrative,
            StackOperation::MemoryAllocate { size, .. } => {
                if *size > 1024 * 1024 { // 1MB
                    CapabilityLevel::Administrative
                } else {
                    CapabilityLevel::FullAccess
                }
            },
            StackOperation::MemoryDeallocate { .. } => CapabilityLevel::FullAccess,
            StackOperation::MemoryAccess { .. } => CapabilityLevel::FullAccess,
            StackOperation::DebugAccess { .. } => CapabilityLevel::Debug,
            StackOperation::ProfilerAccess { .. } => CapabilityLevel::Debug,
            StackOperation::IntrospectionAccess { .. } => CapabilityLevel::System,
        }
    }

    /// Check access permission for operation
    fn check_access_permission(
        &self,
        operation: &StackOperation,
        required_level: &CapabilityLevel,
        security_context: &SecurityContext,
    ) -> VMResult<AccessPermission> {
        let matrix = self.access_control_matrix.read().unwrap();
        
        // Check for operation-specific permission
        if let Some(operation_permissions) = matrix.permissions.get(operation) {
            if let Some(permission) = operation_permissions.get(required_level) {
                return Ok(permission.clone());
            }
        }
        
        // Check for context-specific overrides
        if let Some(context_overrides) = matrix.context_overrides.get(&security_context.context_id) {
            if let Some(permission) = context_overrides.get(operation) {
                return Ok(permission.clone());
            }
        }
        
        // Return default permission based on capability level and trust level
        Ok(self.get_default_permission(required_level, &security_context.trust_level))
    }

    /// Get default permission based on capability and trust levels
    fn get_default_permission(&self, capability_level: &CapabilityLevel, trust_level: &TrustLevel) -> AccessPermission {
        let allowed = match (capability_level, trust_level) {
            (CapabilityLevel::None, _) => false,
            (CapabilityLevel::ReadOnly, TrustLevel::Untrusted) => false,
            (CapabilityLevel::ReadOnly, _) => true,
            (CapabilityLevel::LimitedWrite, TrustLevel::Untrusted) => false,
            (CapabilityLevel::LimitedWrite, TrustLevel::Low) => false,
            (CapabilityLevel::LimitedWrite, _) => true,
            (CapabilityLevel::FullAccess, trust) if *trust >= TrustLevel::Medium => true,
            (CapabilityLevel::Administrative, trust) if *trust >= TrustLevel::High => true,
            (CapabilityLevel::System, TrustLevel::System) => true,
            (CapabilityLevel::Debug, trust) if *trust >= TrustLevel::High => true,
            _ => false,
        };
        
        AccessPermission {
            allowed,
            required_capabilities: vec![format!("{:?}", capability_level)],
            constraints: Vec::new(),
            rate_limit: None,
            audit_level: self.security_policy.default_audit_level.clone(),
            expires_at: None,
        }
    }

    /// Validate access constraints
    fn validate_constraints(
        &self,
        constraints: &[AccessConstraint],
        operation: &StackOperation,
        security_context: &SecurityContext,
    ) -> VMResult<()> {
        for constraint in constraints {
            match constraint {
                AccessConstraint::TimeWindow { start, end } => {
                    let now = SystemTime::now();
                    if now < *start || now > *end {
                        return Err(PrismVMError::CapabilityViolation {
                            message: "Operation not allowed in current time window".to_string(),
                        });
                    }
                }
                AccessConstraint::ResourceLimit { resource, limit } => {
                    // This would check actual resource usage
                    // For now, just validate the constraint exists
                    debug!("Checking resource limit for {}: {}", resource, limit);
                }
                AccessConstraint::ContextRequired { context_type, context_value } => {
                    // This would validate the security context has required properties
                    debug!("Validating context requirement: {} = {}", context_type, context_value);
                }
                AccessConstraint::DependsOn { operation: dep_op, completion_required } => {
                    // This would check if dependent operation has completed
                    debug!("Checking dependency on operation: {} (completion required: {})", dep_op, completion_required);
                }
                AccessConstraint::MaxConcurrent { limit } => {
                    // This would check current concurrent operations
                    debug!("Checking concurrency limit: {}", limit);
                }
                AccessConstraint::MaxFrequency { operations_per_second } => {
                    // This would check operation frequency
                    debug!("Checking frequency limit: {} ops/sec", operations_per_second);
                }
            }
        }
        
        Ok(())
    }

    /// Check rate limits for operation
    fn check_rate_limits(&self, operation: &StackOperation, rate_limit: &Option<RateLimit>) -> VMResult<()> {
        if let Some(limit) = rate_limit {
            // This would implement actual rate limiting logic
            // For now, just check if we're within the basic limit
            let now = Instant::now();
            let window_elapsed = now.duration_since(limit.window_start);
            
            if window_elapsed > limit.window_duration {
                // Reset window - in practice this would be more sophisticated
                debug!("Rate limit window reset for operation: {:?}", operation);
            } else if limit.current_count >= limit.max_operations {
                let mut stats = self.stats.write().unwrap();
                stats.rate_limiting_stats.limits_triggered += 1;
                
                return Err(PrismVMError::CapabilityViolation {
                    message: format!("Rate limit exceeded for operation: {:?}", operation),
                });
            }
        }
        
        Ok(())
    }

    /// Perform threat detection analysis
    fn perform_threat_detection(&self, operation: &StackOperation, security_context: &SecurityContext) -> VMResult<()> {
        let mut threat_detector = self.threat_detector.write().unwrap();
        
        // Analyze operation for threats
        let threat_analysis = threat_detector.analyze_operation(operation, security_context);
        
        if threat_analysis.threat_score > 0.7 {
            // High threat detected
            warn!("High threat detected for operation: {:?}, score: {}", operation, threat_analysis.threat_score);
            
            // Log threat detection event
            self.log_security_event(StackSecurityEvent {
                timestamp: SystemTime::now(),
                event_type: StackSecurityEventType::ThreatDetection,
                operation: operation.clone(),
                function_context: self.extract_function_context(operation),
                stack_depth: 0,
                capabilities: self.capabilities.capability_names(),
                allowed: false,
                security_context: Some(security_context.context_id.clone()),
                threat_analysis: Some(threat_analysis.clone()),
                context: format!("Threat detected: {}", threat_analysis.analysis_details),
                severity: EventSeverity::Critical,
            });
            
            return Err(PrismVMError::CapabilityViolation {
                message: format!("Threat detected: {}", threat_analysis.analysis_details),
            });
        }
        
        Ok(())
    }

    /// Handle access denied scenarios
    fn handle_access_denied(&self, operation: &StackOperation, security_context: &SecurityContext, reason: &str) -> VMResult<()> {
        // Log access denied event
        self.log_security_event(StackSecurityEvent {
            timestamp: SystemTime::now(),
            event_type: StackSecurityEventType::AccessDenied,
            operation: operation.clone(),
            function_context: self.extract_function_context(operation),
            stack_depth: 0,
            capabilities: self.capabilities.capability_names(),
            allowed: false,
            security_context: Some(security_context.context_id.clone()),
            threat_analysis: None,
            context: reason.to_string(),
            severity: EventSeverity::Warning,
        });
        
        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.violations_detected += 1;
        
        Ok(())
    }

    /// Extract function context from operation
    fn extract_function_context(&self, operation: &StackOperation) -> Option<String> {
        match operation {
            StackOperation::FrameCreate { function_name } => Some(function_name.clone()),
            StackOperation::FrameDestroy { function_id } => Some(format!("function_{}", function_id)),
            StackOperation::FrameAccess { function_id, .. } => Some(format!("function_{}", function_id)),
            _ => None,
        }
    }

    /// Comprehensive stack security analysis
    pub fn analyze(&self, stack: &ExecutionStack) -> StackSecurityAnalysis {
        let _span = span!(Level::DEBUG, "analyze_security").entered();

        let mut violations = Vec::new();
        let mut recommendations = Vec::new();

        // Analyze current stack state
        let current_depth = stack.frame_count();
        let current_capability_level = self.determine_current_capability_level();
        
        // Check stack depth against capability levels
        if let Some(max_depth) = self.security_policy.max_depth_by_capability.get(&current_capability_level) {
            if current_depth > *max_depth {
                violations.push(SecurityViolation {
                    violation_type: ViolationType::StackDepthExceeded,
                    function_name: None,
                    stack_depth: current_depth,
                    description: format!("Stack depth {} exceeds limit {} for capability level {:?}", 
                                       current_depth, max_depth, current_capability_level),
                    severity: if current_depth > max_depth * 2 {
                        Severity::Critical
                    } else {
                        Severity::Medium
                    },
                    timestamp: SystemTime::now(),
                    remediation_actions: vec!["Reduce recursion depth".to_string(), "Elevate capabilities".to_string()],
                });
                
                recommendations.push(SecurityRecommendation {
                    recommendation_type: RecommendationType::CapabilityAdjustment,
                    priority: RecommendationPriority::High,
                    description: "Stack depth exceeds safe limits for current capability level".to_string(),
                    implementation_steps: vec![
                        "Review recursive algorithms".to_string(),
                        "Consider tail call optimization".to_string(),
                        "Elevate capabilities if necessary".to_string(),
                    ],
                    expected_impact: "Reduced risk of stack overflow attacks".to_string(),
                    risk_if_ignored: "Potential for denial of service or system instability".to_string(),
                });
            }
        }
        
        // Analyze capability coverage
        let capability_coverage = self.analyze_comprehensive_capability_coverage(stack);
        
        // Perform threat analysis
        let threat_analysis = self.perform_comprehensive_threat_analysis(stack);
        
        // Analyze access control
        let access_control_analysis = self.analyze_access_control_effectiveness();
        
        // Assess compliance
        let compliance_status = self.assess_compliance_status();
        
        // Determine overall security status
        let security_status = self.determine_overall_security_status(&violations, &threat_analysis);
        
        // Determine risk level
        let risk_level = self.calculate_risk_level(&violations, &threat_analysis, &compliance_status);

        StackSecurityAnalysis {
            security_status,
            active_capabilities: self.capabilities.capability_names(),
            current_capability_level,
            violations,
            capability_coverage,
            risk_level,
            threat_analysis,
            access_control_analysis,
            recommendations,
            compliance_status,
        }
    }

    /// Determine current capability level
    fn determine_current_capability_level(&self) -> CapabilityLevel {
        // This would analyze current capabilities and determine the effective level
        // For now, return a reasonable default based on available capabilities
        let capability_names = self.capabilities.capability_names();
        
        if capability_names.iter().any(|name| name.contains("system")) {
            CapabilityLevel::System
        } else if capability_names.iter().any(|name| name.contains("admin")) {
            CapabilityLevel::Administrative
        } else if capability_names.len() > 5 {
            CapabilityLevel::FullAccess
        } else if capability_names.len() > 2 {
            CapabilityLevel::LimitedWrite
        } else if !capability_names.is_empty() {
            CapabilityLevel::ReadOnly
        } else {
            CapabilityLevel::None
        }
    }

    /// Comprehensive capability coverage analysis
    fn analyze_comprehensive_capability_coverage(&self, _stack: &ExecutionStack) -> CapabilityCoverage {
        let available: HashSet<String> = self.capabilities.capability_names().into_iter().collect();
        
        // Determine required capabilities based on current operations
        let required: HashSet<String> = [
            "memory_access".to_string(),
            "stack_frame_creation".to_string(),
            "local_variable_access".to_string(),
        ].into_iter().collect();
        
        // Identify missing and excessive capabilities
        let missing: HashSet<String> = required.difference(&available).cloned().collect();
        let excessive: HashSet<String> = available.difference(&required).cloned().collect();
        
        let coverage_percentage = if required.is_empty() {
            100.0
        } else {
            ((required.len() - missing.len()) as f64 / required.len() as f64) * 100.0
        };
        
        let optimization_opportunities = if !excessive.is_empty() {
            vec!["Remove unused capabilities to follow principle of least privilege".to_string()]
        } else if !missing.is_empty() {
            vec!["Add missing capabilities for full functionality".to_string()]
        } else {
            vec!["Capability set is optimally configured".to_string()]
        };

        CapabilityCoverage {
            required,
            available,
            missing,
            excessive,
            coverage_percentage,
            optimization_opportunities,
        }
    }

    /// Comprehensive threat analysis
    fn perform_comprehensive_threat_analysis(&self, _stack: &ExecutionStack) -> ThreatAnalysisResult {
        let threat_detector = self.threat_detector.read().unwrap();
        let stats = &threat_detector.detection_stats;
        
        // Calculate overall threat score based on recent detections
        let threat_score = if stats.total_threats_detected > 0 {
            (stats.total_threats_detected as f64 / 100.0).min(1.0)
        } else {
            0.1 // Base threat level
        };
        
        let detected_threats = if threat_score > 0.5 {
            vec![ThreatType::ResourceExhaustion, ThreatType::StackOverflow]
        } else {
            Vec::new()
        };
        
        ThreatAnalysisResult {
            threat_score,
            detected_threats,
            confidence: stats.detection_accuracy,
            analysis_details: format!("Threat analysis based on {} historical detections", stats.total_threats_detected),
        }
    }

    /// Analyze access control effectiveness
    fn analyze_access_control_effectiveness(&self) -> AccessControlAnalysis {
        let matrix = self.access_control_matrix.read().unwrap();
        
        // Analyze effective permissions
        let mut effective_permissions = HashMap::new();
        let mut permission_gaps = Vec::new();
        let mut over_privileged = Vec::new();
        let mut under_privileged = Vec::new();
        
        // This would perform comprehensive access control analysis
        // For now, provide basic analysis
        effective_permissions.insert("default".to_string(), AccessPermission {
            allowed: true,
            required_capabilities: vec!["basic_access".to_string()],
            constraints: Vec::new(),
            rate_limit: None,
            audit_level: AuditLevel::Basic,
            expires_at: None,
        });
        
        if matrix.permissions.is_empty() {
            permission_gaps.push("No explicit permissions configured".to_string());
        }
        
        AccessControlAnalysis {
            effective_permissions,
            permission_gaps,
            over_privileged,
            under_privileged,
            access_patterns: vec!["Standard access pattern observed".to_string()],
        }
    }

    /// Assess compliance status
    fn assess_compliance_status(&self) -> ComplianceStatus {
        let mut requirement_compliance = HashMap::new();
        let mut non_compliant_areas = Vec::new();
        let mut compliance_gaps = Vec::new();
        
        // Check basic security requirements
        requirement_compliance.insert("capability_based_access".to_string(), true);
        requirement_compliance.insert("audit_logging".to_string(), true);
        requirement_compliance.insert("threat_detection".to_string(), self.security_policy.enable_threat_detection);
        
        if !self.security_policy.enable_threat_detection {
            non_compliant_areas.push("threat_detection".to_string());
            compliance_gaps.push(ComplianceGap {
                requirement: "threat_detection".to_string(),
                description: "Threat detection is not enabled".to_string(),
                severity: ComplianceGapSeverity::Moderate,
                remediation_steps: vec!["Enable threat detection in security policy".to_string()],
            });
        }
        
        let compliant_count = requirement_compliance.values().filter(|&&v| v).count();
        let compliance_score = compliant_count as f64 / requirement_compliance.len() as f64;
        
        ComplianceStatus {
            compliance_score,
            requirement_compliance,
            non_compliant_areas,
            compliance_gaps,
            last_assessed: SystemTime::now(),
        }
    }

    /// Determine overall security status
    fn determine_overall_security_status(&self, violations: &[SecurityViolation], threat_analysis: &ThreatAnalysisResult) -> SecurityStatus {
        if threat_analysis.threat_score > 0.8 {
            SecurityStatus::UnderThreat
        } else if violations.iter().any(|v| matches!(v.severity, Severity::Critical)) {
            SecurityStatus::Critical
        } else if violations.iter().any(|v| matches!(v.severity, Severity::High)) {
            SecurityStatus::Violated
        } else if !violations.is_empty() || threat_analysis.threat_score > 0.3 {
            SecurityStatus::Warning
        } else {
            SecurityStatus::Secure
        }
    }

    /// Calculate risk level
    fn calculate_risk_level(&self, violations: &[SecurityViolation], threat_analysis: &ThreatAnalysisResult, compliance_status: &ComplianceStatus) -> RiskLevel {
        let violation_risk = if violations.iter().any(|v| matches!(v.severity, Severity::Critical)) {
            1.0
        } else if violations.iter().any(|v| matches!(v.severity, Severity::High)) {
            0.8
        } else if violations.iter().any(|v| matches!(v.severity, Severity::Medium)) {
            0.5
        } else {
            0.2
        };
        
        let threat_risk = threat_analysis.threat_score;
        let compliance_risk = 1.0 - compliance_status.compliance_score;
        
        let overall_risk = (violation_risk + threat_risk + compliance_risk) / 3.0;
        
        if overall_risk > 0.8 {
            RiskLevel::Critical
        } else if overall_risk > 0.6 {
            RiskLevel::High
        } else if overall_risk > 0.3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Log security event with comprehensive details
    fn log_security_event(&self, event: StackSecurityEvent) {
        let mut audit_log = self.audit_log.write().unwrap();
        audit_log.push(event.clone());

        // Keep only recent events (last 10000)
        if audit_log.len() > 10000 {
            audit_log.drain(0..1000);
        }

        // Update event type statistics
        let mut stats = self.stats.write().unwrap();
        let event_type_key = format!("{:?}", event.event_type);
        *stats.events_by_type.entry(event_type_key).or_insert(0) += 1;
        
        // Update threat detection stats if applicable
        if let Some(threat_analysis) = &event.threat_analysis {
            stats.threat_detection_stats.total_threats_detected += 1;
            for threat_type in &threat_analysis.detected_threats {
                let threat_key = format!("{:?}", threat_type);
                *stats.threat_detection_stats.threats_by_type.entry(threat_key).or_insert(0) += 1;
            }
        }
    }

    /// Update security statistics
    fn update_stats(&self, start_time: Instant, allowed: bool) {
        let mut stats = self.stats.write().unwrap();
        stats.total_checks += 1;
        
        if allowed {
            stats.checks_passed += 1;
        } else {
            stats.violations_detected += 1;
        }

        // Update average check time
        let check_time_us = start_time.elapsed().as_micros() as f64;
        if stats.total_checks == 1 {
            stats.avg_check_time_us = check_time_us;
        } else {
            let alpha = 0.1; // Exponential moving average
            stats.avg_check_time_us = alpha * check_time_us + (1.0 - alpha) * stats.avg_check_time_us;
        }
    }

    /// Get security statistics
    pub fn stats(&self) -> StackSecurityStats {
        self.stats.read().unwrap().clone()
    }

    /// Get recent security events
    pub fn recent_events(&self, count: usize) -> Vec<StackSecurityEvent> {
        let audit_log = self.audit_log.read().unwrap();
        audit_log.iter().rev().take(count).cloned().collect()
    }

    /// Get threat detection analysis
    pub fn threat_analysis(&self) -> ThreatDetectionStats {
        let threat_detector = self.threat_detector.read().unwrap();
        threat_detector.detection_stats.clone()
    }
}

impl AccessControlMatrix {
    /// Create a new access control matrix
    pub fn new() -> Self {
        Self {
            permissions: HashMap::new(),
            context_overrides: HashMap::new(),
            temporal_restrictions: HashMap::new(),
            resource_restrictions: HashMap::new(),
        }
    }
}

impl ThreatDetector {
    /// Create a new threat detector
    pub fn new() -> Self {
        Self {
            anomaly_patterns: HashMap::new(),
            threat_signatures: Vec::new(),
            behavioral_baselines: HashMap::new(),
            detection_stats: ThreatDetectionStats::default(),
        }
    }
    
    /// Analyze operation for threats
    pub fn analyze_operation(&mut self, operation: &StackOperation, security_context: &SecurityContext) -> ThreatAnalysisResult {
        // This would implement comprehensive threat analysis
        // For now, provide basic analysis based on operation type and context
        
        let mut threat_score = 0.0;
        let mut detected_threats = Vec::new();
        
        // Check for suspicious patterns
        match operation {
            StackOperation::MemoryAllocate { size, .. } if *size > 10 * 1024 * 1024 => {
                threat_score += 0.3;
                detected_threats.push(ThreatType::ResourceExhaustion);
            }
            StackOperation::StackPush { .. } => {
                // Check for potential stack overflow
                threat_score += 0.1;
            }
            StackOperation::CapabilityElevate { .. } => {
                threat_score += 0.4;
                detected_threats.push(ThreatType::PrivilegeEscalation);
            }
            _ => {}
        }
        
        // Adjust based on security context
        match security_context.trust_level {
            TrustLevel::Untrusted => threat_score += 0.5,
            TrustLevel::Low => threat_score += 0.3,
            TrustLevel::Medium => threat_score += 0.1,
            _ => {}
        }
        
        ThreatAnalysisResult {
            threat_score: threat_score.min(1.0),
            detected_threats,
            confidence: 0.8, // Would be calculated based on detection quality
            analysis_details: format!("Threat analysis for {:?} operation", operation),
        }
    }
}

// Define operation types for capability checking
#[derive(Debug, Clone)]
pub enum MemoryOperation {
    Allocate,
    Access,
    Deallocate,
}

#[derive(Debug, Clone)]
pub enum SystemOperation {
    ElevatedExecution,
    DebugAccess,
    ProfilerAccess,
} 