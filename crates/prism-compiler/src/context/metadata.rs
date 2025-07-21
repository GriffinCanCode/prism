//! AI Metadata Collection - Semantic Context and Business Rule Tracking
//!
//! This module implements AI metadata collection during compilation, gathering
//! semantic contexts, business rules, performance characteristics, and security
//! implications for external AI tool consumption.
//!
//! **Conceptual Responsibility**: AI metadata collection and aggregation
//! **What it does**: Collect semantic context, business rules, security implications
//! **What it doesn't do**: Generate code, manage compilation state, resolve symbols

use prism_common::span::Span;
use serde::{Serialize, Deserialize};

/// AI metadata collector for compilation process
#[derive(Debug, Clone)]
pub struct AIMetadataCollector {
    /// Enable metadata collection
    pub enabled: bool,
    /// Collected semantic contexts
    pub semantic_contexts: Vec<SemanticContextEntry>,
    /// Business rules discovered during compilation
    pub business_rules: Vec<BusinessRuleEntry>,
    /// Performance characteristics identified
    pub performance_characteristics: Vec<PerformanceCharacteristic>,
    /// Security implications found
    pub security_implications: Vec<SecurityImplication>,
}

/// Semantic context entry for AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContextEntry {
    /// Source location where context was identified
    pub location: Span,
    /// Type of semantic context
    pub context_type: SemanticContextType,
    /// Semantic information description
    pub semantic_info: String,
    /// Related concepts and entities
    pub related_concepts: Vec<String>,
    /// AI confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of semantic contexts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticContextType {
    /// Business logic implementation
    BusinessLogic,
    /// Data validation and constraints
    DataValidation,
    /// Error handling patterns
    ErrorHandling,
    /// Performance-critical code sections
    PerformanceCritical,
    /// Security-sensitive operations
    SecuritySensitive,
    /// User interface interactions
    UserInterface,
    /// Data persistence operations
    DataPersistence,
    /// External system integrations
    ExternalIntegration,
    /// Configuration management
    Configuration,
    /// Testing and quality assurance
    Testing,
}

/// Business rule entry discovered in code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessRuleEntry {
    /// Rule identifier or name
    pub name: String,
    /// Human-readable rule description
    pub description: String,
    /// Source location where rule is implemented
    pub location: Span,
    /// Category of business rule
    pub category: BusinessRuleCategory,
    /// Enforcement level requirement
    pub enforcement: EnforcementLevel,
    /// Rule dependencies
    pub dependencies: Vec<String>,
    /// Validation criteria
    pub validation_criteria: Vec<String>,
}

/// Categories of business rules
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessRuleCategory {
    /// Data validation rules
    Validation,
    /// Business constraints and limits
    Constraint,
    /// Workflow and process rules
    Workflow,
    /// Compliance and regulatory requirements
    Compliance,
    /// Security policies and rules
    Security,
    /// Authorization and access control
    Authorization,
    /// Data integrity rules
    DataIntegrity,
    /// Business calculation rules
    Calculation,
    /// Audit and logging requirements
    Audit,
    /// Performance requirements
    Performance,
}

/// Rule enforcement levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Must be enforced (compile error if violated)
    Required,
    /// Should be enforced (warning if violated)
    Recommended,
    /// Optional enforcement (hint if violated)
    Optional,
    /// Informational only (no enforcement)
    Informational,
}

/// Performance characteristic identified in code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristic {
    /// Source location of performance-relevant code
    pub location: Span,
    /// Type of performance characteristic
    pub characteristic_type: PerformanceCharacteristicType,
    /// Description of the characteristic
    pub description: String,
    /// Complexity analysis (e.g., "O(n)", "O(log n)")
    pub complexity: Option<String>,
    /// Optimization suggestions
    pub optimizations: Vec<String>,
    /// Performance impact assessment
    pub impact_assessment: PerformanceImpact,
    /// Resource usage patterns
    pub resource_usage: Vec<ResourceUsagePattern>,
}

/// Types of performance characteristics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceCharacteristicType {
    /// Time complexity concerns
    TimeComplexity,
    /// Space/memory complexity
    SpaceComplexity,
    /// I/O intensive operations
    IOIntensive,
    /// CPU intensive computations
    CPUIntensive,
    /// Memory intensive operations
    MemoryIntensive,
    /// Network intensive operations
    NetworkIntensive,
    /// Database query performance
    DatabaseQuery,
    /// Caching opportunities
    CachingOpportunity,
    /// Parallelization potential
    ParallelizationPotential,
    /// Bottleneck identification
    Bottleneck,
}

/// Performance impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceImpact {
    /// Critical performance impact
    Critical,
    /// High performance impact
    High,
    /// Medium performance impact
    Medium,
    /// Low performance impact
    Low,
    /// Negligible performance impact
    Negligible,
}

/// Resource usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsagePattern {
    /// Type of resource
    pub resource_type: ResourceType,
    /// Usage pattern description
    pub pattern: String,
    /// Estimated usage amount
    pub estimated_usage: Option<String>,
}

/// Types of computational resources
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU usage
    CPU,
    /// Memory usage
    Memory,
    /// Disk I/O
    DiskIO,
    /// Network I/O
    NetworkIO,
    /// Database connections
    DatabaseConnections,
    /// File handles
    FileHandles,
    /// Thread pool usage
    ThreadPool,
}

/// Security implication identified in code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplication {
    /// Source location of security-relevant code
    pub location: Span,
    /// Category of security concern
    pub category: SecurityCategory,
    /// Risk assessment level
    pub risk_level: RiskLevel,
    /// Description of the security implication
    pub description: String,
    /// Suggested mitigation strategies
    pub mitigations: Vec<String>,
    /// Compliance requirements affected
    pub compliance_impact: Vec<String>,
    /// Attack vectors potentially enabled
    pub attack_vectors: Vec<String>,
}

/// Security categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityCategory {
    /// Input validation and sanitization
    InputValidation,
    /// Authentication mechanisms
    Authentication,
    /// Authorization and access control
    Authorization,
    /// Data encryption and protection
    DataEncryption,
    /// SQL injection vulnerabilities
    SQLInjection,
    /// Cross-site scripting (XSS)
    XSS,
    /// Cross-site request forgery (CSRF)
    CSRF,
    /// Buffer overflow risks
    BufferOverflow,
    /// Information disclosure
    InformationDisclosure,
    /// Session management
    SessionManagement,
    /// Cryptographic usage
    Cryptography,
    /// File system security
    FileSystemSecurity,
}

/// Risk assessment levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Critical security risk
    Critical,
    /// High security risk
    High,
    /// Medium security risk
    Medium,
    /// Low security risk
    Low,
    /// Informational security note
    Info,
}

/// AI metadata export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMetadataExport {
    /// All semantic contexts collected
    pub semantic_contexts: Vec<SemanticContextEntry>,
    /// All business rules identified
    pub business_rules: Vec<BusinessRuleEntry>,
    /// All performance characteristics found
    pub performance_characteristics: Vec<PerformanceCharacteristic>,
    /// All security implications discovered
    pub security_implications: Vec<SecurityImplication>,
}

impl AIMetadataCollector {
    /// Create a new AI metadata collector
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            semantic_contexts: Vec::new(),
            business_rules: Vec::new(),
            performance_characteristics: Vec::new(),
            security_implications: Vec::new(),
        }
    }

    /// Add a semantic context entry
    pub fn add_semantic_context(&mut self, entry: SemanticContextEntry) {
        if self.enabled {
            self.semantic_contexts.push(entry);
        }
    }

    /// Add a business rule entry
    pub fn add_business_rule(&mut self, rule: BusinessRuleEntry) {
        if self.enabled {
            self.business_rules.push(rule);
        }
    }

    /// Add a performance characteristic
    pub fn add_performance_characteristic(&mut self, characteristic: PerformanceCharacteristic) {
        if self.enabled {
            self.performance_characteristics.push(characteristic);
        }
    }

    /// Add a security implication
    pub fn add_security_implication(&mut self, implication: SecurityImplication) {
        if self.enabled {
            self.security_implications.push(implication);
        }
    }

    /// Get total metadata entries collected
    pub fn total_entries(&self) -> usize {
        self.semantic_contexts.len() +
        self.business_rules.len() +
        self.performance_characteristics.len() +
        self.security_implications.len()
    }

    /// Clear all collected metadata
    pub fn clear(&mut self) {
        self.semantic_contexts.clear();
        self.business_rules.clear();
        self.performance_characteristics.clear();
        self.security_implications.clear();
    }

    /// Get metadata summary by category
    pub fn summary(&self) -> MetadataSummary {
        MetadataSummary {
            semantic_contexts: self.semantic_contexts.len(),
            business_rules: self.business_rules.len(),
            performance_characteristics: self.performance_characteristics.len(),
            security_implications: self.security_implications.len(),
            high_confidence_entries: self.count_high_confidence_entries(0.8),
            critical_security_risks: self.count_critical_security_risks(),
            required_business_rules: self.count_required_business_rules(),
        }
    }

    /// Export all metadata for external consumption
    pub fn export(&self) -> AIMetadataExport {
        AIMetadataExport {
            semantic_contexts: self.semantic_contexts.clone(),
            business_rules: self.business_rules.clone(),
            performance_characteristics: self.performance_characteristics.clone(),
            security_implications: self.security_implications.clone(),
        }
    }

    /// Count high-confidence entries
    fn count_high_confidence_entries(&self, threshold: f64) -> usize {
        self.semantic_contexts.iter()
            .filter(|entry| entry.confidence >= threshold)
            .count()
    }

    /// Count critical security risks
    fn count_critical_security_risks(&self) -> usize {
        self.security_implications.iter()
            .filter(|implication| implication.risk_level == RiskLevel::Critical)
            .count()
    }

    /// Count required business rules
    fn count_required_business_rules(&self) -> usize {
        self.business_rules.iter()
            .filter(|rule| rule.enforcement == EnforcementLevel::Required)
            .count()
    }

    /// Get semantic contexts by type
    pub fn get_contexts_by_type(&self, context_type: SemanticContextType) -> Vec<&SemanticContextEntry> {
        self.semantic_contexts.iter()
            .filter(|entry| entry.context_type == context_type)
            .collect()
    }

    /// Get business rules by category
    pub fn get_rules_by_category(&self, category: BusinessRuleCategory) -> Vec<&BusinessRuleEntry> {
        self.business_rules.iter()
            .filter(|rule| rule.category == category)
            .collect()
    }

    /// Get security implications by risk level
    pub fn get_security_by_risk(&self, risk_level: RiskLevel) -> Vec<&SecurityImplication> {
        self.security_implications.iter()
            .filter(|implication| implication.risk_level == risk_level)
            .collect()
    }
}

/// Metadata collection summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSummary {
    /// Number of semantic contexts
    pub semantic_contexts: usize,
    /// Number of business rules
    pub business_rules: usize,
    /// Number of performance characteristics
    pub performance_characteristics: usize,
    /// Number of security implications
    pub security_implications: usize,
    /// High-confidence entries (>= 0.8)
    pub high_confidence_entries: usize,
    /// Critical security risks
    pub critical_security_risks: usize,
    /// Required business rules
    pub required_business_rules: usize,
}

impl Default for AIMetadataCollector {
    fn default() -> Self {
        Self::new(true)
    }
}

impl std::fmt::Display for SemanticContextType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticContextType::BusinessLogic => write!(f, "business-logic"),
            SemanticContextType::DataValidation => write!(f, "data-validation"),
            SemanticContextType::ErrorHandling => write!(f, "error-handling"),
            SemanticContextType::PerformanceCritical => write!(f, "performance-critical"),
            SemanticContextType::SecuritySensitive => write!(f, "security-sensitive"),
            SemanticContextType::UserInterface => write!(f, "user-interface"),
            SemanticContextType::DataPersistence => write!(f, "data-persistence"),
            SemanticContextType::ExternalIntegration => write!(f, "external-integration"),
            SemanticContextType::Configuration => write!(f, "configuration"),
            SemanticContextType::Testing => write!(f, "testing"),
        }
    }
}

impl std::fmt::Display for BusinessRuleCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusinessRuleCategory::Validation => write!(f, "validation"),
            BusinessRuleCategory::Constraint => write!(f, "constraint"),
            BusinessRuleCategory::Workflow => write!(f, "workflow"),
            BusinessRuleCategory::Compliance => write!(f, "compliance"),
            BusinessRuleCategory::Security => write!(f, "security"),
            BusinessRuleCategory::Authorization => write!(f, "authorization"),
            BusinessRuleCategory::DataIntegrity => write!(f, "data-integrity"),
            BusinessRuleCategory::Calculation => write!(f, "calculation"),
            BusinessRuleCategory::Audit => write!(f, "audit"),
            BusinessRuleCategory::Performance => write!(f, "performance"),
        }
    }
}

impl std::fmt::Display for SecurityCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityCategory::InputValidation => write!(f, "input-validation"),
            SecurityCategory::Authentication => write!(f, "authentication"),
            SecurityCategory::Authorization => write!(f, "authorization"),
            SecurityCategory::DataEncryption => write!(f, "data-encryption"),
            SecurityCategory::SQLInjection => write!(f, "sql-injection"),
            SecurityCategory::XSS => write!(f, "xss"),
            SecurityCategory::CSRF => write!(f, "csrf"),
            SecurityCategory::BufferOverflow => write!(f, "buffer-overflow"),
            SecurityCategory::InformationDisclosure => write!(f, "information-disclosure"),
            SecurityCategory::SessionManagement => write!(f, "session-management"),
            SecurityCategory::Cryptography => write!(f, "cryptography"),
            SecurityCategory::FileSystemSecurity => write!(f, "filesystem-security"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_collector_creation() {
        let collector = AIMetadataCollector::new(true);
        
        assert!(collector.enabled);
        assert_eq!(collector.total_entries(), 0);
    }

    #[test]
    fn test_semantic_context_addition() {
        let mut collector = AIMetadataCollector::new(true);
        
        let context = SemanticContextEntry {
            location: Span::dummy(),
            context_type: SemanticContextType::BusinessLogic,
            semantic_info: "User validation logic".to_string(),
            related_concepts: vec!["validation".to_string(), "user".to_string()],
            confidence: 0.9,
        };
        
        collector.add_semantic_context(context);
        
        assert_eq!(collector.semantic_contexts.len(), 1);
        assert_eq!(collector.total_entries(), 1);
    }

    #[test]
    fn test_business_rule_addition() {
        let mut collector = AIMetadataCollector::new(true);
        
        let rule = BusinessRuleEntry {
            name: "age_validation".to_string(),
            description: "Users must be at least 18 years old".to_string(),
            location: Span::dummy(),
            category: BusinessRuleCategory::Validation,
            enforcement: EnforcementLevel::Required,
            dependencies: vec![],
            validation_criteria: vec!["age >= 18".to_string()],
        };
        
        collector.add_business_rule(rule);
        
        assert_eq!(collector.business_rules.len(), 1);
        assert_eq!(collector.total_entries(), 1);
    }

    #[test]
    fn test_security_implication_addition() {
        let mut collector = AIMetadataCollector::new(true);
        
        let implication = SecurityImplication {
            location: Span::dummy(),
            category: SecurityCategory::InputValidation,
            risk_level: RiskLevel::High,
            description: "Unvalidated user input".to_string(),
            mitigations: vec!["Sanitize input".to_string()],
            compliance_impact: vec!["OWASP".to_string()],
            attack_vectors: vec!["XSS".to_string()],
        };
        
        collector.add_security_implication(implication);
        
        assert_eq!(collector.security_implications.len(), 1);
        assert_eq!(collector.total_entries(), 1);
    }

    #[test]
    fn test_metadata_filtering() {
        let mut collector = AIMetadataCollector::new(true);
        
        // Add various entries
        collector.add_semantic_context(SemanticContextEntry {
            location: Span::dummy(),
            context_type: SemanticContextType::BusinessLogic,
            semantic_info: "Test".to_string(),
            related_concepts: vec![],
            confidence: 0.9,
        });
        
        collector.add_business_rule(BusinessRuleEntry {
            name: "test_rule".to_string(),
            description: "Test rule".to_string(),
            location: Span::dummy(),
            category: BusinessRuleCategory::Validation,
            enforcement: EnforcementLevel::Required,
            dependencies: vec![],
            validation_criteria: vec![],
        });
        
        // Test filtering
        let business_contexts = collector.get_contexts_by_type(SemanticContextType::BusinessLogic);
        assert_eq!(business_contexts.len(), 1);
        
        let validation_rules = collector.get_rules_by_category(BusinessRuleCategory::Validation);
        assert_eq!(validation_rules.len(), 1);
    }

    #[test]
    fn test_metadata_summary() {
        let mut collector = AIMetadataCollector::new(true);
        
        collector.add_semantic_context(SemanticContextEntry {
            location: Span::dummy(),
            context_type: SemanticContextType::SecuritySensitive,
            semantic_info: "Security check".to_string(),
            related_concepts: vec![],
            confidence: 0.95,
        });
        
        collector.add_security_implication(SecurityImplication {
            location: Span::dummy(),
            category: SecurityCategory::Authentication,
            risk_level: RiskLevel::Critical,
            description: "Critical security issue".to_string(),
            mitigations: vec![],
            compliance_impact: vec![],
            attack_vectors: vec![],
        });
        
        let summary = collector.summary();
        assert_eq!(summary.semantic_contexts, 1);
        assert_eq!(summary.security_implications, 1);
        assert_eq!(summary.high_confidence_entries, 1);
        assert_eq!(summary.critical_security_risks, 1);
    }
} 