//! Symbol Data Structures
//!
//! This module defines the core data structures for symbol information storage,
//! following PLT-004 specifications. It provides comprehensive symbol metadata
//! with AI-comprehensible context and semantic integration.
//!
//! ## Conceptual Responsibility
//! 
//! This module handles ONE thing: "Symbol Data Storage and Structure"
//! - Symbol data structure definitions
//! - Symbol identification and referencing
//! - Visibility and access control
//! - Effect and capability tracking
//! 
//! It does NOT handle:
//! - Symbol classification (delegated to kinds.rs)
//! - Symbol metadata (delegated to metadata.rs)
//! - Symbol storage/retrieval (delegated to table.rs)

use crate::symbols::kinds::SymbolKind;
use crate::symbols::metadata::SymbolMetadata;
use crate::context::CompilationPhase;
use prism_common::{NodeId, span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for symbols within the compiler
/// 
/// This provides a lightweight identifier separate from the interned Symbol
/// for internal compiler use and cross-referencing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolId(u64);

static NEXT_SYMBOL_ID: AtomicU64 = AtomicU64::new(1);

impl SymbolId {
    /// Create a new unique symbol ID
    pub fn new() -> Self {
        Self(NEXT_SYMBOL_ID.fetch_add(1, Ordering::Relaxed))
    }
    
    /// Create a symbol ID from a raw value (for testing/serialization)
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sym_id:{}", self.0)
    }
}

impl Default for SymbolId {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive symbol data with semantic information
/// 
/// This is the core data structure that holds all information about a symbol,
/// integrating with existing infrastructure while providing rich metadata
/// for AI comprehension and tooling support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolData {
    /// Unique symbol ID for internal referencing
    pub id: SymbolId,
    
    /// Interned symbol for string representation (from prism-common)
    pub symbol: Symbol,
    
    /// Symbol name for display and debugging
    pub name: String,
    
    /// Symbol classification and semantic meaning
    pub kind: SymbolKind,
    
    /// Source location where symbol is defined
    pub location: Span,
    
    /// Visibility level and access control
    pub visibility: SymbolVisibility,
    
    /// Associated AST node reference
    pub ast_node: Option<NodeId>,
    
    /// Semantic type reference (key into semantic database)
    pub semantic_type: Option<String>,
    
    /// Effect information for this symbol
    pub effects: Vec<SymbolEffect>,
    
    /// Required capabilities to use this symbol
    pub required_capabilities: Vec<String>,
    
    /// Symbol metadata for AI and tooling
    pub metadata: SymbolMetadata,
    
    /// Cross-references to related symbols
    pub references: SymbolReferences,
    
    /// Usage statistics and patterns
    pub usage_info: SymbolUsageInfo,
    
    /// Compilation context information
    pub compilation_context: SymbolCompilationMetadata,
}

/// Symbol visibility levels with semantic meaning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolVisibility {
    /// Public visibility - accessible from anywhere
    Public,
    /// Module visibility - accessible within the same module
    Module,
    /// Private visibility - accessible only within the same scope
    Private,
    /// Internal visibility - accessible within the same crate/package
    Internal,
    /// Protected visibility - accessible within inheritance hierarchy
    Protected,
    /// Package visibility - accessible within the same package
    Package,
}

impl SymbolVisibility {
    /// Check if this visibility level allows access from another visibility context
    pub fn allows_access_from(&self, from: SymbolVisibility, same_module: bool, same_package: bool) -> bool {
        match self {
            SymbolVisibility::Public => true,
            SymbolVisibility::Module => same_module,
            SymbolVisibility::Private => false,
            SymbolVisibility::Internal => same_package,
            SymbolVisibility::Protected => same_module || same_package, // Simplified
            SymbolVisibility::Package => same_package,
        }
    }
    
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            SymbolVisibility::Public => "public",
            SymbolVisibility::Module => "module",
            SymbolVisibility::Private => "private",
            SymbolVisibility::Internal => "internal",
            SymbolVisibility::Protected => "protected",
            SymbolVisibility::Package => "package",
        }
    }
    
    /// Check if this visibility is more restrictive than another
    pub fn is_more_restrictive_than(&self, other: &SymbolVisibility) -> bool {
        use SymbolVisibility::*;
        match (self, other) {
            (Private, _) => *other != Private,
            (Module, Public | Internal | Protected | Package) => true,
            (Protected, Public | Internal | Package) => true,
            (Package, Public | Internal) => true,
            (Internal, Public) => true,
            _ => false,
        }
    }
}

impl Default for SymbolVisibility {
    fn default() -> Self {
        SymbolVisibility::Private
    }
}

/// Effect information for symbols with detailed context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEffect {
    /// Effect name/identifier
    pub name: String,
    
    /// Effect category for classification
    pub category: EffectCategory,
    
    /// Effect parameters and configuration
    pub parameters: HashMap<String, String>,
    
    /// Effect composition information
    pub composition: EffectComposition,
    
    /// Safety requirements for this effect
    pub safety_requirements: Vec<SafetyRequirement>,
}

/// Effect category classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectCategory {
    /// I/O operations
    IO,
    /// Memory allocation/deallocation
    Memory,
    /// Network operations
    Network,
    /// File system operations
    FileSystem,
    /// Concurrency and threading
    Concurrency,
    /// Security-sensitive operations
    Security,
    /// Performance-critical operations
    Performance,
    /// User interface operations
    UI,
    /// Database operations
    Database,
    /// Custom effect category
    Custom(String),
}

/// Effect composition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectComposition {
    /// Whether this effect can be composed with others
    pub composable: bool,
    /// Effects this effect conflicts with
    pub conflicts_with: Vec<String>,
    /// Effects this effect requires
    pub requires: Vec<String>,
    /// Effects this effect provides
    pub provides: Vec<String>,
}

/// Safety requirement for effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRequirement {
    /// Requirement type
    pub requirement_type: SafetyRequirementType,
    /// Requirement description
    pub description: String,
    /// Enforcement level
    pub enforcement: crate::symbols::kinds::EnforcementLevel,
}

/// Types of safety requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyRequirementType {
    /// Memory safety requirement
    MemorySafety,
    /// Thread safety requirement
    ThreadSafety,
    /// Exception safety requirement
    ExceptionSafety,
    /// Resource safety requirement
    ResourceSafety,
    /// Security requirement
    Security,
    /// Custom safety requirement
    Custom(String),
}

/// Cross-references to related symbols
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolReferences {
    /// Symbols this symbol depends on
    pub dependencies: Vec<SymbolReference>,
    
    /// Symbols that depend on this symbol
    pub dependents: Vec<SymbolReference>,
    
    /// Symbols this symbol overrides or implements
    pub overrides: Vec<SymbolReference>,
    
    /// Symbols that override or implement this symbol
    pub overridden_by: Vec<SymbolReference>,
    
    /// Related symbols (similar functionality, same domain, etc.)
    pub related: Vec<SymbolReference>,
}

/// Reference to another symbol with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolReference {
    /// Referenced symbol ID
    pub symbol_id: SymbolId,
    
    /// Reference type and context
    pub reference_type: ReferenceType,
    
    /// Location where the reference occurs
    pub reference_location: Option<Span>,
    
    /// Strength of the relationship (0.0 to 1.0)
    pub relationship_strength: f64,
}

/// Types of symbol references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReferenceType {
    /// Direct dependency (calls, uses, etc.)
    Dependency,
    /// Type dependency
    TypeDependency,
    /// Inheritance relationship
    Inheritance,
    /// Implementation relationship
    Implementation,
    /// Override relationship
    Override,
    /// Composition relationship
    Composition,
    /// Association relationship
    Association,
    /// Custom relationship
    Custom(String),
}

/// Symbol usage information for optimization and analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolUsageInfo {
    /// Number of times this symbol is referenced
    pub reference_count: u32,
    
    /// Usage patterns observed
    pub usage_patterns: Vec<UsagePattern>,
    
    /// Performance characteristics
    pub performance_info: Option<PerformanceInfo>,
    
    /// Hot path information
    pub is_hot_path: bool,
    
    /// Optimization hints
    pub optimization_hints: Vec<String>,
}

/// Usage pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    /// Pattern type
    pub pattern_type: UsagePatternType,
    
    /// Pattern frequency
    pub frequency: u32,
    
    /// Pattern context
    pub context: String,
    
    /// Confidence in pattern detection (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsagePatternType {
    /// Frequent access pattern
    FrequentAccess,
    /// Batch processing pattern
    BatchProcessing,
    /// Initialization pattern
    Initialization,
    /// Cleanup pattern
    Cleanup,
    /// Error handling pattern
    ErrorHandling,
    /// Caching pattern
    Caching,
    /// Custom pattern
    Custom(String),
}

/// Performance information for symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInfo {
    /// Estimated time complexity
    pub time_complexity: Option<String>,
    
    /// Estimated space complexity
    pub space_complexity: Option<String>,
    
    /// Memory usage characteristics
    pub memory_usage: Option<MemoryUsageInfo>,
    
    /// Performance bottlenecks
    pub bottlenecks: Vec<String>,
    
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageInfo {
    /// Estimated memory usage in bytes
    pub estimated_bytes: Option<u64>,
    
    /// Memory allocation pattern
    pub allocation_pattern: AllocationPattern,
    
    /// Memory lifetime
    pub lifetime: MemoryLifetime,
}

/// Memory allocation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationPattern {
    /// Stack allocation
    Stack,
    /// Heap allocation
    Heap,
    /// Static allocation
    Static,
    /// Pool allocation
    Pool,
    /// Custom allocation
    Custom(String),
}

/// Memory lifetime patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLifetime {
    /// Short-lived (function scope)
    Short,
    /// Medium-lived (object scope)
    Medium,
    /// Long-lived (application scope)
    Long,
    /// Static lifetime
    Static,
    /// Custom lifetime
    Custom(String),
}

/// Symbol compilation metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolCompilationMetadata {
    /// Compilation phase where symbol was created
    pub creation_phase: CompilationPhase,
    
    /// Source file information
    pub source_file: Option<String>,
    
    /// Compilation timestamp
    pub compilation_timestamp: Option<std::time::SystemTime>,
    
    /// Compiler version
    pub compiler_version: Option<String>,
    
    /// Compilation flags and options
    pub compilation_flags: Vec<String>,
    
    /// Target platform information
    pub target_platform: Option<String>,
}



impl SymbolData {
    /// Create a new symbol data with minimal required information
    pub fn new(
        symbol: Symbol,
        name: String,
        kind: SymbolKind,
        location: Span,
    ) -> Self {
        Self {
            id: SymbolId::new(),
            symbol,
            name,
            kind,
            location,
            visibility: SymbolVisibility::default(),
            ast_node: None,
            semantic_type: None,
            effects: Vec::new(),
            required_capabilities: Vec::new(),
            metadata: SymbolMetadata::default(),
            references: SymbolReferences::default(),
            usage_info: SymbolUsageInfo::default(),
            compilation_context: SymbolCompilationMetadata::default(),
        }
    }
    
    /// Get a human-readable description of this symbol
    pub fn description(&self) -> String {
        format!("{} {} ({})", 
            self.visibility.description(),
            self.kind.description(),
            self.name
        )
    }
    
    /// Check if this symbol has any effects
    pub fn has_effects(&self) -> bool {
        !self.effects.is_empty()
    }
    
    /// Check if this symbol requires specific capabilities
    pub fn requires_capabilities(&self) -> bool {
        !self.required_capabilities.is_empty()
    }
    
    /// Check if this symbol is accessible with given capabilities
    pub fn is_accessible_with_capabilities(&self, available_capabilities: &[String]) -> bool {
        self.required_capabilities.iter()
            .all(|required| available_capabilities.contains(required))
    }
    
    /// Get AI-comprehensible summary of this symbol
    pub fn ai_summary(&self) -> String {
        let mut summary = format!(
            "Symbol '{}' is a {} {} defined at {}",
            self.name,
            self.visibility.description(),
            self.kind.ai_category(),
            self.location
        );
        
        if self.has_effects() {
            summary.push_str(&format!(", with {} effects", self.effects.len()));
        }
        
        if self.requires_capabilities() {
            summary.push_str(&format!(", requiring {} capabilities", self.required_capabilities.len()));
        }
        
        if let Some(ai_context) = &self.metadata.ai_context {
            summary.push_str(&format!(". Purpose: {}", ai_context.purpose));
        }
        
        summary
    }
    
    /// Update usage statistics
    pub fn record_usage(&mut self, pattern_type: UsagePatternType, context: String) {
        self.usage_info.reference_count += 1;
        
        // Find existing pattern or create new one
        if let Some(pattern) = self.usage_info.usage_patterns.iter_mut()
            .find(|p| std::mem::discriminant(&p.pattern_type) == std::mem::discriminant(&pattern_type)) {
            pattern.frequency += 1;
        } else {
            self.usage_info.usage_patterns.push(UsagePattern {
                pattern_type,
                frequency: 1,
                context,
                confidence: 1.0,
            });
        }
    }
    
    /// Add a symbol reference
    pub fn add_reference(&mut self, reference: SymbolReference) {
        match reference.reference_type {
            ReferenceType::Dependency | ReferenceType::TypeDependency => {
                self.references.dependencies.push(reference);
            }
            ReferenceType::Override => {
                self.references.overrides.push(reference);
            }
            _ => {
                self.references.related.push(reference);
            }
        }
    }
    
    /// Check if this symbol is related to another symbol
    pub fn is_related_to(&self, other_id: SymbolId) -> bool {
        self.references.dependencies.iter().any(|r| r.symbol_id == other_id) ||
        self.references.dependents.iter().any(|r| r.symbol_id == other_id) ||
        self.references.related.iter().any(|r| r.symbol_id == other_id)
    }
}

impl std::fmt::Display for SymbolData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// Default implementations for supporting types
impl Default for EffectComposition {
    fn default() -> Self {
        Self {
            composable: true,
            conflicts_with: Vec::new(),
            requires: Vec::new(),
            provides: Vec::new(),
        }
    }
}

impl Default for SymbolEffect {
    fn default() -> Self {
        Self {
            name: String::new(),
            category: EffectCategory::Custom("unknown".to_string()),
            parameters: HashMap::new(),
            composition: EffectComposition::default(),
            safety_requirements: Vec::new(),
        }
    }
} 