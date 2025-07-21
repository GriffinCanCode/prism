//! Conceptual Boundary Detection
//!
//! This module implements algorithms to detect conceptual boundaries within
//! and between modules, helping identify where code should be split or merged.
//!
//! **Conceptual Responsibility**: Detect and validate conceptual boundaries

use crate::{CohesionResult, CohesionError, AnalysisResult, CohesionMetrics};
use prism_ast::{Program, AstNode, Item, ModuleDecl, SectionDecl};
use prism_common::{span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Conceptual boundary detector
#[derive(Debug)]
pub struct BoundaryDetector {
    /// Boundary detection configuration
    config: BoundaryConfig,
    
    /// Detected boundary cache
    boundary_cache: HashMap<String, Vec<ConceptualBoundary>>,
}

/// Configuration for boundary detection
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    /// Minimum boundary strength to report
    pub min_boundary_strength: f64,
    
    /// Enable cross-module boundary detection
    pub enable_cross_module: bool,
    
    /// Enable section boundary analysis
    pub enable_section_analysis: bool,
    
    /// Boundary detection sensitivity
    pub sensitivity: BoundarySensitivity,
}

/// Boundary detection sensitivity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundarySensitivity {
    /// Conservative - only detect strong boundaries
    Conservative,
    /// Balanced - detect moderate boundaries
    Balanced,
    /// Aggressive - detect weak boundaries too
    Aggressive,
}

/// Detected conceptual boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualBoundary {
    /// Boundary type
    pub boundary_type: BoundaryType,
    
    /// Boundary strength (0-1)
    pub strength: f64,
    
    /// Location of the boundary
    pub location: BoundaryLocation,
    
    /// Description of the boundary
    pub description: String,
    
    /// Evidence for this boundary
    pub evidence: Vec<String>,
    
    /// Suggested action
    pub suggested_action: BoundaryAction,
    
    /// Confidence in this boundary detection
    pub confidence: f64,
}

/// Types of conceptual boundaries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Boundary between different business capabilities
    BusinessCapability,
    
    /// Boundary between different technical concerns
    TechnicalConcern,
    
    /// Boundary between different data domains
    DataDomain,
    
    /// Boundary between different abstraction levels
    AbstractionLevel,
    
    /// Boundary between different architectural layers
    ArchitecturalLayer,
    
    /// Boundary between different responsibilities
    Responsibility,
    
    /// Custom boundary type
    Custom(String),
}

/// Location of a boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryLocation {
    /// Boundary within a module, between sections
    WithinModule {
        module_name: String,
        section_before: String,
        section_after: String,
    },
    
    /// Boundary between modules
    BetweenModules {
        module_a: String,
        module_b: String,
    },
    
    /// Boundary at a specific location
    AtLocation(Span),
    
    /// Conceptual boundary (not tied to specific code location)
    Conceptual {
        area_a: String,
        area_b: String,
    },
}

/// Suggested action for a boundary
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryAction {
    /// Split the area along this boundary
    Split,
    
    /// Merge areas across this boundary
    Merge,
    
    /// Clarify the boundary with better organization
    Clarify,
    
    /// Add explicit boundary markers
    AddMarkers,
    
    /// No action needed - boundary is appropriate
    NoAction,
}

/// Scope of responsibility for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsibilityScope {
    /// Scope name
    pub name: String,
    
    /// Scope type
    pub scope_type: ScopeType,
    
    /// Elements included in this scope
    pub elements: Vec<String>,
    
    /// Responsibility description
    pub responsibility: String,
    
    /// Scope cohesion score
    pub cohesion_score: f64,
    
    /// Related scopes
    pub related_scopes: Vec<String>,
}

/// Types of responsibility scopes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScopeType {
    /// Module-level scope
    Module,
    
    /// Section-level scope
    Section,
    
    /// Function-level scope
    Function,
    
    /// Type-level scope
    Type,
    
    /// Cross-cutting scope
    CrossCutting,
}

impl BoundaryDetector {
    /// Create new boundary detector
    pub fn new() -> Self {
        Self {
            config: BoundaryConfig::default(),
            boundary_cache: HashMap::new(),
        }
    }
    
    /// Create boundary detector with custom configuration
    pub fn with_config(config: BoundaryConfig) -> Self {
        Self {
            config,
            boundary_cache: HashMap::new(),
        }
    }
    
    /// Detect boundaries in a complete program
    pub fn detect_program_boundaries(
        &mut self, 
        program: &Program, 
        analysis_result: &AnalysisResult
    ) -> CohesionResult<Vec<ConceptualBoundary>> {
        let mut boundaries = Vec::new();
        
        // Extract modules
        let modules: Vec<_> = program.items.iter()
            .filter_map(|item| {
                if let Item::Module(module_decl) = &item.kind {
                    Some((item, module_decl))
                } else {
                    None
                }
            })
            .collect();
        
        if modules.len() < 2 {
            return Ok(boundaries); // Need at least 2 modules to detect boundaries
        }
        
        // Detect boundaries between modules
        if self.config.enable_cross_module {
            let cross_module_boundaries = self.detect_cross_module_boundaries(&modules)?;
            boundaries.extend(cross_module_boundaries);
        }
        
        // Detect boundaries within modules
        for (module_item, module_decl) in modules {
            let module_boundaries = self.detect_module_boundaries(module_item, module_decl)?;
            boundaries.extend(module_boundaries);
        }
        
        // Filter boundaries by strength threshold
        boundaries.retain(|b| b.strength >= self.config.min_boundary_strength);
        
        // Sort by strength (strongest first)
        boundaries.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(boundaries)
    }
    
    /// Detect boundaries within a single module
    pub fn detect_module_boundaries(
        &mut self, 
        module_item: &AstNode<Item>, 
        module_decl: &ModuleDecl
    ) -> CohesionResult<Vec<ConceptualBoundary>> {
        let mut boundaries = Vec::new();
        
        // Check cache first
        let cache_key = format!("module_{}", module_decl.name);
        if let Some(cached_boundaries) = self.boundary_cache.get(&cache_key) {
            return Ok(cached_boundaries.clone());
        }
        
        if self.config.enable_section_analysis {
            let section_boundaries = self.detect_section_boundaries(module_decl)?;
            boundaries.extend(section_boundaries);
        }
        
        // Detect responsibility boundaries
        let responsibility_boundaries = self.detect_responsibility_boundaries(module_decl)?;
        boundaries.extend(responsibility_boundaries);
        
        // Cache results
        self.boundary_cache.insert(cache_key, boundaries.clone());
        
        Ok(boundaries)
    }
    
    /// Detect boundaries between modules
    fn detect_cross_module_boundaries(
        &self, 
        modules: &[(&AstNode<Item>, &ModuleDecl)]
    ) -> CohesionResult<Vec<ConceptualBoundary>> {
        let mut boundaries = Vec::new();
        
        // Compare each pair of modules
        for i in 0..modules.len() {
            for j in (i + 1)..modules.len() {
                let (_, module_a) = modules[i];
                let (_, module_b) = modules[j];
                
                let boundary = self.analyze_module_pair(module_a, module_b)?;
                if let Some(boundary) = boundary {
                    boundaries.push(boundary);
                }
            }
        }
        
        Ok(boundaries)
    }
    
    /// Analyze a pair of modules for boundary detection
    fn analyze_module_pair(
        &self, 
        module_a: &ModuleDecl, 
        module_b: &ModuleDecl
    ) -> CohesionResult<Option<ConceptualBoundary>> {
        // Calculate similarity between modules
        let semantic_similarity = self.calculate_module_semantic_similarity(module_a, module_b);
        let capability_similarity = self.calculate_capability_similarity(module_a, module_b);
        let dependency_similarity = self.calculate_dependency_similarity(module_a, module_b);
        
        // Overall similarity
        let overall_similarity = (semantic_similarity + capability_similarity + dependency_similarity) / 3.0;
        
        // Boundary strength is inverse of similarity
        let boundary_strength = 1.0 - overall_similarity;
        
        // Only report significant boundaries
        if boundary_strength < self.sensitivity_threshold() {
            return Ok(None);
        }
        
        let suggested_action = if boundary_strength > 0.8 {
            BoundaryAction::NoAction // Strong boundary is good
        } else if boundary_strength < 0.3 {
            BoundaryAction::Merge // Weak boundary suggests merge
        } else {
            BoundaryAction::Clarify // Medium boundary needs clarification
        };
        
        let boundary = ConceptualBoundary {
            boundary_type: self.determine_boundary_type(module_a, module_b),
            strength: boundary_strength,
            location: BoundaryLocation::BetweenModules {
                module_a: module_a.name.to_string(),
                module_b: module_b.name.to_string(),
            },
            description: format!("Boundary between {} and {}", module_a.name, module_b.name),
            evidence: self.generate_boundary_evidence(module_a, module_b, boundary_strength),
            suggested_action,
            confidence: self.calculate_boundary_confidence(boundary_strength),
        };
        
        Ok(Some(boundary))
    }
    
    /// Detect boundaries between sections within a module
    fn detect_section_boundaries(&self, module_decl: &ModuleDecl) -> CohesionResult<Vec<ConceptualBoundary>> {
        let mut boundaries = Vec::new();
        
        if module_decl.sections.len() < 2 {
            return Ok(boundaries);
        }
        
        // Analyze section organization
        for i in 0..module_decl.sections.len() {
            for j in (i + 1)..module_decl.sections.len() {
                let section_a = &module_decl.sections[i];
                let section_b = &module_decl.sections[j];
                
                let boundary = self.analyze_section_pair(module_decl, section_a, section_b)?;
                if let Some(boundary) = boundary {
                    boundaries.push(boundary);
                }
            }
        }
        
        Ok(boundaries)
    }
    
    /// Analyze a pair of sections for boundary detection
    fn analyze_section_pair(
        &self,
        module_decl: &ModuleDecl,
        section_a: &AstNode<SectionDecl>,
        section_b: &AstNode<SectionDecl>
    ) -> CohesionResult<Option<ConceptualBoundary>> {
        // Different section types naturally have boundaries
        let type_difference = if std::mem::discriminant(&section_a.kind.kind) != std::mem::discriminant(&section_b.kind.kind) {
            0.8 // Strong boundary between different section types
        } else {
            0.2 // Weak boundary between same section types
        };
        
        // TODO: Analyze content similarity between sections
        let content_similarity = 0.5; // Placeholder
        
        let boundary_strength = type_difference * (1.0 - content_similarity);
        
        if boundary_strength < self.sensitivity_threshold() {
            return Ok(None);
        }
        
        let boundary = ConceptualBoundary {
            boundary_type: BoundaryType::TechnicalConcern,
            strength: boundary_strength,
            location: BoundaryLocation::WithinModule {
                module_name: module_decl.name.to_string(),
                section_before: format!("{:?}", section_a.kind),
                section_after: format!("{:?}", section_b.kind),
            },
            description: format!("Section boundary in module {}", module_decl.name),
            evidence: vec![
                format!("Section type difference: {:.2}", type_difference),
                format!("Content similarity: {:.2}", content_similarity),
            ],
            suggested_action: BoundaryAction::Clarify,
            confidence: 0.8,
        };
        
        Ok(Some(boundary))
    }
    
    /// Detect responsibility boundaries
    fn detect_responsibility_boundaries(&self, module_decl: &ModuleDecl) -> CohesionResult<Vec<ConceptualBoundary>> {
        let mut boundaries = Vec::new();
        
        // TODO: Implement responsibility boundary detection
        // This would analyze how well-defined and separated the responsibilities are
        
        Ok(boundaries)
    }
    
    /// Calculate semantic similarity between modules
    fn calculate_module_semantic_similarity(&self, module_a: &ModuleDecl, module_b: &ModuleDecl) -> f64 {
        use strsim::jaro_winkler;
        
        // Compare module names
        let name_similarity = jaro_winkler(&module_a.name.to_string(), &module_b.name.to_string());
        
        // Compare capabilities
        let capability_similarity = match (&module_a.capability, &module_b.capability) {
            (Some(cap_a), Some(cap_b)) => jaro_winkler(cap_a, cap_b),
            (None, None) => 1.0,
            _ => 0.0,
        };
        
        // Compare descriptions
        let description_similarity = match (&module_a.description, &module_b.description) {
            (Some(desc_a), Some(desc_b)) => jaro_winkler(desc_a, desc_b),
            (None, None) => 1.0,
            _ => 0.5,
        };
        
        (name_similarity + capability_similarity + description_similarity) / 3.0
    }
    
    /// Calculate capability similarity between modules
    fn calculate_capability_similarity(&self, module_a: &ModuleDecl, module_b: &ModuleDecl) -> f64 {
        match (&module_a.capability, &module_b.capability) {
            (Some(cap_a), Some(cap_b)) => {
                if cap_a == cap_b {
                    1.0
                } else {
                    strsim::jaro_winkler(cap_a, cap_b)
                }
            }
            (None, None) => 0.5, // Both lack capability definition
            _ => 0.0, // One has capability, other doesn't
        }
    }
    
    /// Calculate dependency similarity between modules
    fn calculate_dependency_similarity(&self, module_a: &ModuleDecl, module_b: &ModuleDecl) -> f64 {
        let deps_a: std::collections::HashSet<_> = module_a.dependencies.iter()
            .map(|d| &d.path)
            .collect();
        let deps_b: std::collections::HashSet<_> = module_b.dependencies.iter()
            .map(|d| &d.path)
            .collect();
        
        if deps_a.is_empty() && deps_b.is_empty() {
            return 1.0; // Both have no dependencies
        }
        
        let intersection = deps_a.intersection(&deps_b).count();
        let union = deps_a.union(&deps_b).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64 // Jaccard similarity
        }
    }
    
    /// Determine the type of boundary between modules
    fn determine_boundary_type(&self, module_a: &ModuleDecl, module_b: &ModuleDecl) -> BoundaryType {
        // If both have capabilities, it's likely a business boundary
        if module_a.capability.is_some() && module_b.capability.is_some() {
            return BoundaryType::BusinessCapability;
        }
        
        // If one has many more dependencies, it's likely a technical concern boundary
        let dep_diff = (module_a.dependencies.len() as i32 - module_b.dependencies.len() as i32).abs();
        if dep_diff > 3 {
            return BoundaryType::TechnicalConcern;
        }
        
        // Default to responsibility boundary
        BoundaryType::Responsibility
    }
    
    /// Generate evidence for a boundary
    fn generate_boundary_evidence(&self, module_a: &ModuleDecl, module_b: &ModuleDecl, strength: f64) -> Vec<String> {
        let mut evidence = Vec::new();
        
        evidence.push(format!("Boundary strength: {:.2}", strength));
        
        if let (Some(cap_a), Some(cap_b)) = (&module_a.capability, &module_b.capability) {
            evidence.push(format!("Different capabilities: '{}' vs '{}'", cap_a, cap_b));
        }
        
        let dep_diff = (module_a.dependencies.len() as i32 - module_b.dependencies.len() as i32).abs();
        if dep_diff > 0 {
            evidence.push(format!("Dependency count difference: {}", dep_diff));
        }
        
        evidence
    }
    
    /// Calculate confidence in boundary detection
    fn calculate_boundary_confidence(&self, strength: f64) -> f64 {
        // Higher strength = higher confidence, but with some uncertainty
        (strength * 0.8) + 0.2
    }
    
    /// Get sensitivity threshold based on configuration
    fn sensitivity_threshold(&self) -> f64 {
        match self.config.sensitivity {
            BoundarySensitivity::Conservative => 0.7,
            BoundarySensitivity::Balanced => 0.5,
            BoundarySensitivity::Aggressive => 0.3,
        }
    }
}

impl BoundaryConfig {
    /// Create conservative boundary detection configuration
    pub fn conservative() -> Self {
        Self {
            min_boundary_strength: 0.7,
            enable_cross_module: true,
            enable_section_analysis: false,
            sensitivity: BoundarySensitivity::Conservative,
        }
    }
    
    /// Create balanced boundary detection configuration
    pub fn balanced() -> Self {
        Self {
            min_boundary_strength: 0.5,
            enable_cross_module: true,
            enable_section_analysis: true,
            sensitivity: BoundarySensitivity::Balanced,
        }
    }
    
    /// Create aggressive boundary detection configuration
    pub fn aggressive() -> Self {
        Self {
            min_boundary_strength: 0.3,
            enable_cross_module: true,
            enable_section_analysis: true,
            sensitivity: BoundarySensitivity::Aggressive,
        }
    }
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl Default for BoundaryDetector {
    fn default() -> Self {
        Self::new()
    }
} 