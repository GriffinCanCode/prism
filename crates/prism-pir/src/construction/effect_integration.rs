//! Effect System Integration - AST to PIR Effect Analysis
//!
//! This module implements effect system integration during PIR construction,
//! extracting effects from AST nodes and building comprehensive effect graphs.
//!
//! **Conceptual Responsibility**: Effect system analysis and integration
//! **What it does**: Extracts effects from AST, builds effect graphs, analyzes effect relationships
//! **What it doesn't do**: AST parsing, PIR construction, semantic analysis (focuses on effect analysis)

use crate::{PIRResult, PIRError};
use crate::semantic::{EffectGraph, EffectNode, EffectEdge, EffectSignature, PIREffect};
use prism_ast::{Program, AstNode, Item, ModuleDecl, FunctionDecl, TypeDecl};
use prism_effects::{EffectType, EffectCapability, SecurityClassification};
use std::collections::{HashMap, HashSet};

/// Effect system integrator for PIR construction
pub struct EffectSystemIntegrator {
    /// Configuration for effect integration
    config: EffectIntegrationConfig,
    /// Effect extractors for different AST node types
    extractors: EffectExtractors,
    /// Effect graph builder
    graph_builder: EffectGraphBuilder,
}

/// Configuration for effect system integration
#[derive(Debug, Clone)]
pub struct EffectIntegrationConfig {
    /// Enable function effect extraction
    pub enable_function_effects: bool,
    /// Enable type effect extraction
    pub enable_type_effects: bool,
    /// Enable security effect analysis
    pub enable_security_analysis: bool,
    /// Enable performance effect tracking
    pub enable_performance_tracking: bool,
    /// Enable effect composition analysis
    pub enable_composition_analysis: bool,
    /// Maximum effect analysis depth
    pub max_analysis_depth: usize,
}

/// Effect extractors for different AST constructs
pub struct EffectExtractors {
    /// Function effect extractor
    function_extractor: FunctionEffectExtractor,
    /// Type effect extractor
    type_extractor: TypeEffectExtractor,
    /// Module effect extractor
    module_extractor: ModuleEffectExtractor,
}

/// Function effect extractor
pub struct FunctionEffectExtractor {
    /// Effect patterns for function analysis
    effect_patterns: HashMap<String, Vec<EffectType>>,
    /// Security classification patterns
    security_patterns: HashMap<String, SecurityClassification>,
}

/// Type effect extractor
pub struct TypeEffectExtractor {
    /// Type-based effect inference
    type_effect_mapping: HashMap<String, Vec<EffectType>>,
}

/// Module effect extractor
pub struct ModuleEffectExtractor {
    /// Module-level effect patterns
    module_patterns: HashMap<String, Vec<EffectType>>,
}

/// Effect graph builder
pub struct EffectGraphBuilder {
    /// Current graph being built
    graph: EffectGraph,
    /// Node ID counter
    next_node_id: usize,
    /// Effect relationships
    relationships: Vec<EffectRelationship>,
}

/// Relationship between effects
#[derive(Debug, Clone)]
pub struct EffectRelationship {
    /// Source effect
    pub source: String,
    /// Target effect
    pub target: String,
    /// Relationship type
    pub relationship_type: EffectRelationshipType,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
}

/// Types of effect relationships
#[derive(Debug, Clone)]
pub enum EffectRelationshipType {
    /// One effect depends on another
    Dependency,
    /// Effects are mutually exclusive
    Exclusion,
    /// Effects compose together
    Composition,
    /// One effect enables another
    Enablement,
}

/// Effect integration result
#[derive(Debug, Clone)]
pub struct EffectIntegrationResult {
    /// Built effect graph
    pub effect_graph: EffectGraph,
    /// Effect signatures by function
    pub function_signatures: HashMap<String, EffectSignature>,
    /// Module-level effects
    pub module_effects: HashMap<String, Vec<PIREffect>>,
    /// Integration diagnostics
    pub diagnostics: Vec<EffectDiagnostic>,
    /// Effect analysis metrics
    pub metrics: EffectAnalysisMetrics,
}

/// Diagnostic from effect integration
#[derive(Debug, Clone)]
pub struct EffectDiagnostic {
    /// Diagnostic level
    pub level: EffectDiagnosticLevel,
    /// Diagnostic message
    pub message: String,
    /// Source location
    pub location: Option<prism_common::span::Span>,
    /// Related effect
    pub effect_name: Option<String>,
}

/// Effect diagnostic levels
#[derive(Debug, Clone)]
pub enum EffectDiagnosticLevel {
    /// Information
    Info,
    /// Warning about potential issues
    Warning,
    /// Error in effect analysis
    Error,
}

/// Effect analysis metrics
#[derive(Debug, Default, Clone)]
pub struct EffectAnalysisMetrics {
    /// Total effects analyzed
    pub total_effects: usize,
    /// Effects by type
    pub effects_by_type: HashMap<String, usize>,
    /// Security classifications found
    pub security_classifications: HashMap<String, usize>,
    /// Effect composition depth
    pub max_composition_depth: usize,
    /// Analysis coverage score
    pub coverage_score: f64,
}

impl EffectSystemIntegrator {
    /// Create a new effect system integrator
    pub fn new(config: EffectIntegrationConfig) -> Self {
        Self {
            extractors: EffectExtractors::new(),
            graph_builder: EffectGraphBuilder::new(),
            config,
        }
    }

    /// Integrate effects from program into PIR
    pub fn integrate_effects(&mut self, program: &Program) -> PIRResult<EffectIntegrationResult> {
        let mut function_signatures = HashMap::new();
        let mut module_effects = HashMap::new();
        let mut diagnostics = Vec::new();
        let mut metrics = EffectAnalysisMetrics::default();

        // Process modules
        for item in &program.items {
            if let Item::Module(module_decl) = &item.kind {
                self.process_module_effects(
                    module_decl,
                    &mut function_signatures,
                    &mut module_effects,
                    &mut diagnostics,
                    &mut metrics,
                )?;
            }
        }

        // Process global items
        let global_items: Vec<_> = program.items.iter()
            .filter(|item| !matches!(item.kind, Item::Module(_)))
            .collect();

        if !global_items.is_empty() {
            self.process_global_effects(
                &global_items,
                &mut function_signatures,
                &mut module_effects,
                &mut diagnostics,
                &mut metrics,
            )?;
        }

        // Build effect graph
        let effect_graph = self.graph_builder.build_graph(&function_signatures, &module_effects)?;

        // Calculate coverage score
        metrics.coverage_score = self.calculate_coverage_score(&function_signatures, &module_effects);

        Ok(EffectIntegrationResult {
            effect_graph,
            function_signatures,
            module_effects,
            diagnostics,
            metrics,
        })
    }

    /// Process effects for a module
    fn process_module_effects(
        &mut self,
        module_decl: &ModuleDecl,
        function_signatures: &mut HashMap<String, EffectSignature>,
        module_effects: &mut HashMap<String, Vec<PIREffect>>,
        diagnostics: &mut Vec<EffectDiagnostic>,
        metrics: &mut EffectAnalysisMetrics,
    ) -> PIRResult<()> {
        let mut module_level_effects = Vec::new();

        // Extract module-level effects
        if self.config.enable_function_effects {
            let extracted_effects = self.extractors.module_extractor.extract_module_effects(module_decl)?;
            module_level_effects.extend(extracted_effects);
        }

        // Process module items
        for item in &module_decl.items {
            match &item.kind {
                Item::Function(func_decl) => {
                    if self.config.enable_function_effects {
                        let signature = self.extract_function_effect_signature(func_decl, diagnostics)?;
                        let full_name = format!("{}::{}", module_decl.name, func_decl.name);
                        function_signatures.insert(full_name, signature);
                        metrics.total_effects += 1;
                    }
                }
                Item::Type(type_decl) => {
                    if self.config.enable_type_effects {
                        let type_effects = self.extract_type_effects(type_decl, diagnostics)?;
                        module_level_effects.extend(type_effects);
                    }
                }
                _ => {}
            }
        }

        // Store module effects
        if !module_level_effects.is_empty() {
            module_effects.insert(module_decl.name.clone(), module_level_effects);
        }

        Ok(())
    }

    /// Process effects for global items
    fn process_global_effects(
        &mut self,
        items: &[&AstNode<Item>],
        function_signatures: &mut HashMap<String, EffectSignature>,
        module_effects: &mut HashMap<String, Vec<PIREffect>>,
        diagnostics: &mut Vec<EffectDiagnostic>,
        metrics: &mut EffectAnalysisMetrics,
    ) -> PIRResult<()> {
        let mut global_effects = Vec::new();

        for item in items {
            match &item.kind {
                Item::Function(func_decl) => {
                    if self.config.enable_function_effects {
                        let signature = self.extract_function_effect_signature(func_decl, diagnostics)?;
                        function_signatures.insert(func_decl.name.to_string(), signature);
                        metrics.total_effects += 1;
                    }
                }
                Item::Type(type_decl) => {
                    if self.config.enable_type_effects {
                        let type_effects = self.extract_type_effects(type_decl, diagnostics)?;
                        global_effects.extend(type_effects);
                    }
                }
                _ => {}
            }
        }

        if !global_effects.is_empty() {
            module_effects.insert("global".to_string(), global_effects);
        }

        Ok(())
    }

    /// Extract effect signature from function
    fn extract_function_effect_signature(
        &self,
        func_decl: &FunctionDecl,
        diagnostics: &mut Vec<EffectDiagnostic>,
    ) -> PIRResult<EffectSignature> {
        let input_effects = self.extractors.function_extractor.extract_input_effects(func_decl)?;
        let output_effects = self.extractors.function_extractor.extract_output_effects(func_decl)?;
        let effect_dependencies = self.extractors.function_extractor.extract_effect_dependencies(func_decl)?;

        // Validate effect signature
        if input_effects.is_empty() && output_effects.is_empty() && self.config.enable_function_effects {
            diagnostics.push(EffectDiagnostic {
                level: EffectDiagnosticLevel::Info,
                message: format!("Function '{}' has no detected effects", func_decl.name),
                location: None, // TODO: Get function span
                effect_name: None,
            });
        }

        Ok(EffectSignature {
            input_effects,
            output_effects,
            effect_dependencies,
        })
    }

    /// Extract effects from type declaration
    fn extract_type_effects(
        &self,
        type_decl: &TypeDecl,
        _diagnostics: &mut Vec<EffectDiagnostic>,
    ) -> PIRResult<Vec<PIREffect>> {
        let effects = self.extractors.type_extractor.extract_type_effects(type_decl)?;
        Ok(effects)
    }

    /// Calculate coverage score for effect analysis
    fn calculate_coverage_score(
        &self,
        function_signatures: &HashMap<String, EffectSignature>,
        module_effects: &HashMap<String, Vec<PIREffect>>,
    ) -> f64 {
        let total_functions = function_signatures.len();
        let functions_with_effects = function_signatures.values()
            .filter(|sig| !sig.input_effects.is_empty() || !sig.output_effects.is_empty())
            .count();

        let total_modules = module_effects.len();
        let modules_with_effects = module_effects.values()
            .filter(|effects| !effects.is_empty())
            .count();

        if total_functions == 0 && total_modules == 0 {
            return 1.0; // Perfect score for empty program
        }

        let function_coverage = if total_functions > 0 {
            functions_with_effects as f64 / total_functions as f64
        } else {
            1.0
        };

        let module_coverage = if total_modules > 0 {
            modules_with_effects as f64 / total_modules as f64
        } else {
            1.0
        };

        (function_coverage + module_coverage) / 2.0
    }
}

impl EffectExtractors {
    fn new() -> Self {
        Self {
            function_extractor: FunctionEffectExtractor::new(),
            type_extractor: TypeEffectExtractor::new(),
            module_extractor: ModuleEffectExtractor::new(),
        }
    }
}

impl FunctionEffectExtractor {
    fn new() -> Self {
        let mut effect_patterns = HashMap::new();
        let mut security_patterns = HashMap::new();

        // Define common effect patterns
        effect_patterns.insert("io".to_string(), vec![EffectType::IO]);
        effect_patterns.insert("network".to_string(), vec![EffectType::Network]);
        effect_patterns.insert("file".to_string(), vec![EffectType::FileSystem]);
        effect_patterns.insert("database".to_string(), vec![EffectType::Database]);
        effect_patterns.insert("memory".to_string(), vec![EffectType::Memory]);

        // Define security patterns
        security_patterns.insert("public".to_string(), SecurityClassification::Public);
        security_patterns.insert("internal".to_string(), SecurityClassification::Internal);
        security_patterns.insert("confidential".to_string(), SecurityClassification::Confidential);
        security_patterns.insert("secret".to_string(), SecurityClassification::Secret);

        Self {
            effect_patterns,
            security_patterns,
        }
    }

    fn extract_input_effects(&self, func_decl: &FunctionDecl) -> PIRResult<Vec<PIREffect>> {
        let mut effects = Vec::new();
        let func_name = func_decl.name.to_lowercase();

        // Pattern-based effect detection
        for (pattern, effect_types) in &self.effect_patterns {
            if func_name.contains(pattern) {
                for effect_type in effect_types {
                    effects.push(PIREffect {
                        effect_type: effect_type.clone(),
                        description: format!("Input effect inferred from function name pattern: {}", pattern),
                        security_level: self.infer_security_level(&func_name),
                        capabilities: vec![EffectCapability::Read], // Default for input
                    });
                }
            }
        }

        Ok(effects)
    }

    fn extract_output_effects(&self, func_decl: &FunctionDecl) -> PIRResult<Vec<PIREffect>> {
        let mut effects = Vec::new();
        let func_name = func_decl.name.to_lowercase();

        // Pattern-based effect detection for outputs
        if func_name.starts_with("write") || func_name.starts_with("save") || func_name.starts_with("create") {
            effects.push(PIREffect {
                effect_type: EffectType::IO,
                description: "Output effect inferred from function name".to_string(),
                security_level: self.infer_security_level(&func_name),
                capabilities: vec![EffectCapability::Write],
            });
        }

        if func_name.contains("send") || func_name.contains("transmit") {
            effects.push(PIREffect {
                effect_type: EffectType::Network,
                description: "Network output effect inferred from function name".to_string(),
                security_level: self.infer_security_level(&func_name),
                capabilities: vec![EffectCapability::Network],
            });
        }

        Ok(effects)
    }

    fn extract_effect_dependencies(&self, _func_decl: &FunctionDecl) -> PIRResult<Vec<String>> {
        // TODO: Extract effect dependencies from function body analysis
        Ok(Vec::new())
    }

    fn infer_security_level(&self, func_name: &str) -> SecurityClassification {
        for (pattern, classification) in &self.security_patterns {
            if func_name.contains(pattern) {
                return classification.clone();
            }
        }

        // Default security classification
        if func_name.contains("admin") || func_name.contains("privileged") {
            SecurityClassification::Confidential
        } else if func_name.contains("internal") {
            SecurityClassification::Internal
        } else {
            SecurityClassification::Public
        }
    }
}

impl TypeEffectExtractor {
    fn new() -> Self {
        let mut type_effect_mapping = HashMap::new();

        // Define type-based effects
        type_effect_mapping.insert("File".to_string(), vec![EffectType::FileSystem]);
        type_effect_mapping.insert("Socket".to_string(), vec![EffectType::Network]);
        type_effect_mapping.insert("Database".to_string(), vec![EffectType::Database]);
        type_effect_mapping.insert("Memory".to_string(), vec![EffectType::Memory]);

        Self {
            type_effect_mapping,
        }
    }

    fn extract_type_effects(&self, type_decl: &TypeDecl) -> PIRResult<Vec<PIREffect>> {
        let mut effects = Vec::new();
        let type_name = type_decl.name.to_string();

        // Check for direct mappings
        if let Some(effect_types) = self.type_effect_mapping.get(&type_name) {
            for effect_type in effect_types {
                effects.push(PIREffect {
                    effect_type: effect_type.clone(),
                    description: format!("Effect inferred from type: {}", type_name),
                    security_level: SecurityClassification::Public, // Default
                    capabilities: vec![EffectCapability::Read, EffectCapability::Write],
                });
            }
        }

        Ok(effects)
    }
}

impl ModuleEffectExtractor {
    fn new() -> Self {
        let mut module_patterns = HashMap::new();

        // Define module-level effect patterns
        module_patterns.insert("io".to_string(), vec![EffectType::IO]);
        module_patterns.insert("net".to_string(), vec![EffectType::Network]);
        module_patterns.insert("fs".to_string(), vec![EffectType::FileSystem]);
        module_patterns.insert("db".to_string(), vec![EffectType::Database]);

        Self {
            module_patterns,
        }
    }

    fn extract_module_effects(&self, module_decl: &ModuleDecl) -> PIRResult<Vec<PIREffect>> {
        let mut effects = Vec::new();
        let module_name = module_decl.name.to_lowercase();

        // Pattern-based module effect detection
        for (pattern, effect_types) in &self.module_patterns {
            if module_name.contains(pattern) {
                for effect_type in effect_types {
                    effects.push(PIREffect {
                        effect_type: effect_type.clone(),
                        description: format!("Module-level effect inferred from name pattern: {}", pattern),
                        security_level: SecurityClassification::Internal, // Default for modules
                        capabilities: vec![EffectCapability::ModuleLevel],
                    });
                }
            }
        }

        Ok(effects)
    }
}

impl EffectGraphBuilder {
    fn new() -> Self {
        Self {
            graph: EffectGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
            },
            next_node_id: 0,
            relationships: Vec::new(),
        }
    }

    fn build_graph(
        &mut self,
        function_signatures: &HashMap<String, EffectSignature>,
        module_effects: &HashMap<String, Vec<PIREffect>>,
    ) -> PIRResult<EffectGraph> {
        // Build nodes for function effects
        for (func_name, signature) in function_signatures {
            self.add_function_effect_nodes(func_name, signature)?;
        }

        // Build nodes for module effects
        for (module_name, effects) in module_effects {
            self.add_module_effect_nodes(module_name, effects)?;
        }

        // Build edges based on relationships
        self.build_effect_edges()?;

        Ok(self.graph.clone())
    }

    fn add_function_effect_nodes(&mut self, func_name: &str, signature: &EffectSignature) -> PIRResult<()> {
        // Add input effect nodes
        for effect in &signature.input_effects {
            let node_id = format!("{}:input:{}", func_name, self.next_node_id);
            self.next_node_id += 1;

            let node = EffectNode {
                id: node_id.clone(),
                effect_type: effect.effect_type.clone(),
                description: effect.description.clone(),
                security_classification: effect.security_level.clone(),
                capabilities: effect.capabilities.clone(),
                source_location: None, // TODO: Add source location
            };

            self.graph.nodes.insert(node_id, node);
        }

        // Add output effect nodes
        for effect in &signature.output_effects {
            let node_id = format!("{}:output:{}", func_name, self.next_node_id);
            self.next_node_id += 1;

            let node = EffectNode {
                id: node_id.clone(),
                effect_type: effect.effect_type.clone(),
                description: effect.description.clone(),
                security_classification: effect.security_level.clone(),
                capabilities: effect.capabilities.clone(),
                source_location: None,
            };

            self.graph.nodes.insert(node_id, node);
        }

        Ok(())
    }

    fn add_module_effect_nodes(&mut self, module_name: &str, effects: &[PIREffect]) -> PIRResult<()> {
        for effect in effects {
            let node_id = format!("{}:module:{}", module_name, self.next_node_id);
            self.next_node_id += 1;

            let node = EffectNode {
                id: node_id.clone(),
                effect_type: effect.effect_type.clone(),
                description: effect.description.clone(),
                security_classification: effect.security_level.clone(),
                capabilities: effect.capabilities.clone(),
                source_location: None,
            };

            self.graph.nodes.insert(node_id, node);
        }

        Ok(())
    }

    fn build_effect_edges(&mut self) -> PIRResult<()> {
        // Build edges based on effect relationships
        for relationship in &self.relationships {
            if let (Some(source_node), Some(target_node)) = (
                self.graph.nodes.get(&relationship.source),
                self.graph.nodes.get(&relationship.target),
            ) {
                let edge = EffectEdge {
                    source: source_node.id.clone(),
                    target: target_node.id.clone(),
                    edge_type: match relationship.relationship_type {
                        EffectRelationshipType::Dependency => "dependency".to_string(),
                        EffectRelationshipType::Exclusion => "exclusion".to_string(),
                        EffectRelationshipType::Composition => "composition".to_string(),
                        EffectRelationshipType::Enablement => "enablement".to_string(),
                    },
                    strength: relationship.strength,
                };

                self.graph.edges.push(edge);
            }
        }

        Ok(())
    }
}

impl Default for EffectIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_function_effects: true,
            enable_type_effects: true,
            enable_security_analysis: true,
            enable_performance_tracking: false,
            enable_composition_analysis: true,
            max_analysis_depth: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_integrator_creation() {
        let config = EffectIntegrationConfig::default();
        let integrator = EffectSystemIntegrator::new(config);
        assert!(integrator.config.enable_function_effects);
    }

    #[test]
    fn test_function_effect_extraction() {
        let extractor = FunctionEffectExtractor::new();
        assert!(extractor.effect_patterns.contains_key("io"));
        assert!(extractor.security_patterns.contains_key("public"));
    }

    #[test]
    fn test_effect_graph_builder() {
        let builder = EffectGraphBuilder::new();
        assert_eq!(builder.next_node_id, 0);
        assert!(builder.graph.nodes.is_empty());
    }
}