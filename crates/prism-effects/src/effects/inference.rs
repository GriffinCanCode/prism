//! Effect inference and tracking
//!
//! This module provides effect inference capabilities that analyze AST nodes
//! to automatically determine what effects functions and expressions may have.

use crate::Effect;
use crate::effects::definition::{EffectRegistry, EffectInstanceMetadata};
use prism_common::{span::Span, symbol::Symbol, NodeId};
use prism_ast::{AstNode, Type, Expr, FunctionDecl, types::FunctionType};
use std::collections::{HashMap, HashSet};

/// Effect inference system
#[derive(Debug)]
pub struct EffectInference {
    /// Inference engine
    pub engine: EffectInferenceEngine,
}

impl EffectInference {
    /// Create new inference system
    pub fn new() -> Self {
        Self {
            engine: EffectInferenceEngine::new(),
        }
    }
}

impl Default for EffectInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Effect inference engine that analyzes code to determine effects
#[derive(Debug)]
pub struct EffectInferenceEngine {
    /// Configuration for inference
    pub config: InferenceConfig,
    /// Cache of previously inferred effects
    pub inference_cache: HashMap<NodeId, Vec<InferredEffect>>,
    /// Effect composition analyzer
    pub composition_analyzer: EffectCompositionAnalyzer,
    /// AI-assisted inference
    pub ai_assistant: AIInferenceAssistant,
}

impl EffectInferenceEngine {
    /// Create a new effect inference engine
    pub fn new() -> Self {
        Self {
            config: InferenceConfig::default(),
            inference_cache: HashMap::new(),
            composition_analyzer: EffectCompositionAnalyzer::new(),
            ai_assistant: AIInferenceAssistant::new(),
        }
    }

    /// Infer effects for a given AST node
    pub fn infer_effects(
        &mut self,
        node: &AstNode<Type>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, crate::EffectSystemError> {
        // Check cache first
        if let Some(cached_effects) = self.inference_cache.get(&node.id) {
            return Ok(cached_effects.iter().map(|ie| ie.effect.clone()).collect());
        }

        // Perform inference based on node type
        let inferred_effects = match &node.kind {
            Type::Function(func_type) => {
                self.infer_function_effects(func_type, registry)?
            }
            Type::Effect(effect_type) => {
                // Convert prism_ast::Effect to our Effect type
                effect_type.effects.iter().map(|ast_effect| {
                    // Convert prism_ast::Effect to our Effect type
                    let effect_name = match ast_effect {
                        prism_ast::Effect::IO(_) => "IO",
                        prism_ast::Effect::State(_) => "State", 
                        prism_ast::Effect::Exception(_) => "Exception",
                        prism_ast::Effect::Async(_) => "Async",
                        prism_ast::Effect::Database(_) => "Database",
                        prism_ast::Effect::Network(_) => "Network",
                        prism_ast::Effect::FileSystem(_) => "FileSystem",
                        prism_ast::Effect::Memory(_) => "Memory",
                        prism_ast::Effect::Computation(_) => "Computation",
                        prism_ast::Effect::Security(_) => "Security",
                        prism_ast::Effect::Custom(name) => &name.name,
                    };
                    Effect::new(effect_name.to_string(), node.span)
                }).collect()
            }
            _ => {
                // For other types, check if they reference effect-producing operations
                self.infer_type_effects(node, registry)?
            }
        };

        // Apply AI assistance if enabled
        let ai_enhanced_effects = if self.config.use_ai_assistance {
            self.ai_assistant.enhance_inference(&inferred_effects, node, registry)?
        } else {
            inferred_effects
        };

        // Cache the results
        let inferred_effect_entries: Vec<InferredEffect> = ai_enhanced_effects
            .iter()
            .map(|effect| InferredEffect {
                effect: effect.clone(),
                confidence: 0.8, // Default confidence
                source: InferenceSource::Analysis,
                reasoning: "Inferred from AST analysis".to_string(),
            })
            .collect();

        self.inference_cache.insert(node.id, inferred_effect_entries.clone());

        Ok(ai_enhanced_effects)
    }

    /// Infer effects for function types
    fn infer_function_effects(
        &mut self,
        func_type: &FunctionType,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, crate::EffectSystemError> {
        let mut effects = Vec::new();

        // Convert explicit effects from AST to our Effect type
        for ast_effect in &func_type.effects {
            let effect_name = match ast_effect {
                prism_ast::Effect::IO(_) => "IO",
                prism_ast::Effect::State(_) => "State", 
                prism_ast::Effect::Exception(_) => "Exception",
                prism_ast::Effect::Async(_) => "Async",
                prism_ast::Effect::Database(_) => "Database",
                prism_ast::Effect::Network(_) => "Network",
                prism_ast::Effect::FileSystem(_) => "FileSystem",
                prism_ast::Effect::Memory(_) => "Memory",
                prism_ast::Effect::Computation(_) => "Computation",
                prism_ast::Effect::Security(_) => "Security",
                prism_ast::Effect::Custom(name) => &name.name,
            };
            effects.push(Effect::new(effect_name.to_string(), prism_common::span::Span::dummy()));
        }

        // Analyze parameter types for implicit effects
        for param_type in &func_type.parameters {
            let param_effects = self.infer_effects(param_type, registry)?;
            effects.extend(param_effects);
        }

        // Analyze return type for implicit effects
        let return_effects = self.infer_effects(&func_type.return_type, registry)?;
        effects.extend(return_effects);

        // Apply composition rules
        let composed_effects = self.composition_analyzer.compose_effects(&effects, registry)?;

        Ok(composed_effects)
    }

    /// Infer effects for general types
    fn infer_type_effects(
        &mut self,
        node: &AstNode<Type>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, crate::EffectSystemError> {
        let mut effects = Vec::new();

        // Check if the type has semantic metadata that implies effects
        if let Some(ai_context) = &node.metadata.ai_context {
            // Look for effect-related keywords in AI context
            if ai_context.side_effects.contains(&"IO".to_string()) {
                effects.push(Effect::new("IO.General".to_string(), node.span));
            }
            if ai_context.side_effects.contains(&"Database".to_string()) {
                effects.push(Effect::new("Database.Query".to_string(), node.span));
            }
            if ai_context.side_effects.contains(&"Network".to_string()) {
                effects.push(Effect::new("IO.Network.Connect".to_string(), node.span));
            }
        }

        // Check business rules for effect implications
        for rule in &node.metadata.business_rules {
            if rule.contains("database") || rule.contains("query") {
                effects.push(Effect::new("Database.Query".to_string(), node.span));
            }
            if rule.contains("file") || rule.contains("storage") {
                effects.push(Effect::new("IO.FileSystem.Read".to_string(), node.span));
            }
            if rule.contains("network") || rule.contains("api") {
                effects.push(Effect::new("IO.Network.Connect".to_string(), node.span));
            }
        }

        Ok(effects)
    }

    /// Track effect usage in a function
    pub fn track_function_effects(
        &mut self,
        function: &FunctionDecl,
        registry: &EffectRegistry,
    ) -> Result<EffectUsageAnalysis, crate::EffectSystemError> {
        let mut analysis = EffectUsageAnalysis::new(function.name.clone());

        // Analyze function signature for declared effects
        if let Some(return_type) = &function.return_type {
            let declared_effects = self.infer_effects(return_type, registry)?;
            analysis.declared_effects = declared_effects;
        }

        // Analyze function body for actual effect usage
        // This would require analyzing the function body AST
        // For now, we'll use a simplified approach based on function attributes

        // Check attributes for effect annotations
        for attribute in &function.attributes {
            if attribute.name.as_str() == "effect" {
                // Parse effect from attribute
                let effect_name = format!("AttributeEffect.{}", attribute.name);
                let effect = Effect::new(effect_name, Span::dummy());
                analysis.actual_effects.push(effect);
            }
        }

        // Compare declared vs actual effects
        analysis.effect_consistency = self.check_effect_consistency(
            &analysis.declared_effects,
            &analysis.actual_effects,
        );

        Ok(analysis)
    }

    /// Check consistency between declared and actual effects
    fn check_effect_consistency(
        &self,
        declared: &[Effect],
        actual: &[Effect],
    ) -> EffectConsistency {
        let declared_names: HashSet<_> = declared.iter().map(|e| &e.definition).collect();
        let actual_names: HashSet<_> = actual.iter().map(|e| &e.definition).collect();

        let missing_declarations: Vec<_> = actual_names.difference(&declared_names).cloned().collect();
        let unused_declarations: Vec<_> = declared_names.difference(&actual_names).cloned().collect();

        EffectConsistency {
            is_consistent: missing_declarations.is_empty() && unused_declarations.is_empty(),
            missing_declarations: missing_declarations.into_iter().cloned().collect(),
            unused_declarations: unused_declarations.into_iter().cloned().collect(),
        }
    }
}

impl Default for EffectInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for effect inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Whether to use AI assistance for inference
    pub use_ai_assistance: bool,
    /// Confidence threshold for including inferred effects
    pub confidence_threshold: f64,
    /// Maximum depth for recursive inference
    pub max_inference_depth: u32,
    /// Whether to cache inference results
    pub cache_results: bool,
    /// Whether to infer effects from semantic metadata
    pub infer_from_metadata: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            use_ai_assistance: true,
            confidence_threshold: 0.7,
            max_inference_depth: 10,
            cache_results: true,
            infer_from_metadata: true,
        }
    }
}

/// An inferred effect with confidence and reasoning
#[derive(Debug, Clone)]
pub struct InferredEffect {
    /// The inferred effect
    pub effect: Effect,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Source of the inference
    pub source: InferenceSource,
    /// Reasoning for the inference
    pub reasoning: String,
}

/// Source of effect inference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceSource {
    /// Explicit declaration in code
    Explicit,
    /// Inferred from static analysis
    Analysis,
    /// Inferred from AI assistance
    AIAssisted,
    /// Inferred from semantic metadata
    Metadata,
    /// Inferred from usage patterns
    Usage,
}

/// Effect composition analyzer
#[derive(Debug)]
pub struct EffectCompositionAnalyzer {
    /// Composition rules
    pub composition_rules: Vec<CompositionRule>,
}

impl EffectCompositionAnalyzer {
    /// Create a new composition analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            composition_rules: Vec::new(),
        };
        analyzer.register_default_rules();
        analyzer
    }

    /// Compose effects according to composition rules
    pub fn compose_effects(
        &self,
        effects: &[Effect],
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, crate::EffectSystemError> {
        let mut composed_effects = effects.to_vec();

        // Apply composition rules
        for rule in &self.composition_rules {
            if rule.applies_to(effects) {
                let new_effects = rule.apply(effects)?;
                composed_effects.extend(new_effects);
            }
        }

        // Remove duplicates
        composed_effects.sort_by(|a, b| a.definition.cmp(&b.definition));
        composed_effects.dedup_by(|a, b| a.definition == b.definition);

        Ok(composed_effects)
    }

    /// Register default composition rules
    fn register_default_rules(&mut self) {
        // Rule: File read + File write = File modification
        self.composition_rules.push(CompositionRule {
            name: "FileReadWrite".to_string(),
            description: "File read and write compose to file modification".to_string(),
            input_patterns: vec![
                "IO.FileSystem.Read".to_string(),
                "IO.FileSystem.Write".to_string(),
            ],
            output_effect: "IO.FileSystem.Modify".to_string(),
            conditions: Vec::new(),
        });

        // Rule: Database query + Database transaction = Database modification
        self.composition_rules.push(CompositionRule {
            name: "DatabaseQueryTransaction".to_string(),
            description: "Database query in transaction context".to_string(),
            input_patterns: vec![
                "Database.Query".to_string(),
                "Database.Transaction".to_string(),
            ],
            output_effect: "Database.TransactionalQuery".to_string(),
            conditions: Vec::new(),
        });
    }
}

impl Default for EffectCompositionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Rule for composing effects
#[derive(Debug, Clone)]
pub struct CompositionRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Input effect patterns
    pub input_patterns: Vec<String>,
    /// Output effect
    pub output_effect: String,
    /// Conditions for applying the rule
    pub conditions: Vec<AstNode<Expr>>,
}

impl CompositionRule {
    /// Check if this rule applies to a set of effects
    pub fn applies_to(&self, effects: &[Effect]) -> bool {
        let effect_names: HashSet<_> = effects.iter().map(|e| &e.definition).collect();
        
        // Check if all input patterns are present
        self.input_patterns.iter().all(|pattern| effect_names.contains(pattern))
    }

    /// Apply this composition rule
    pub fn apply(&self, effects: &[Effect]) -> Result<Vec<Effect>, crate::EffectSystemError> {
        if !self.applies_to(effects) {
            return Ok(Vec::new());
        }

        // Create the composed effect
        // Use the span of the first matching effect
        let span = effects
            .iter()
            .find(|e| self.input_patterns.contains(&e.definition))
            .map(|e| e.span)
            .unwrap_or_else(|| Span::dummy());

        let composed_effect = Effect::new(self.output_effect.clone(), span)
            .with_metadata(EffectInstanceMetadata {
                ai_context: Some(format!("Composed from: {}", self.input_patterns.join(", "))),
                inferred: true,
                confidence: 0.9,
                inference_source: Some(format!("Composition rule: {}", self.name)),
                ..Default::default()
            });

        Ok(vec![composed_effect])
    }
}

/// AI-assisted inference component
#[derive(Debug)]
pub struct AIInferenceAssistant {
    /// AI inference patterns
    pub patterns: Vec<AIInferencePattern>,
    /// Confidence adjustment factors
    pub confidence_factors: HashMap<String, f64>,
}

impl AIInferenceAssistant {
    /// Create a new AI inference assistant
    pub fn new() -> Self {
        let mut assistant = Self {
            patterns: Vec::new(),
            confidence_factors: HashMap::new(),
        };
        assistant.initialize_patterns();
        assistant
    }

    /// Enhance effect inference with AI assistance
    pub fn enhance_inference(
        &self,
        base_effects: &[Effect],
        node: &AstNode<Type>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, crate::EffectSystemError> {
        let mut enhanced_effects = base_effects.to_vec();

        // Apply AI patterns
        for pattern in &self.patterns {
            if pattern.matches(node) {
                let suggested_effects = pattern.suggest_effects(node, registry)?;
                enhanced_effects.extend(suggested_effects);
            }
        }

        Ok(enhanced_effects)
    }

    /// Initialize AI inference patterns
    fn initialize_patterns(&mut self) {
        // Pattern for database-related operations
        self.patterns.push(AIInferencePattern {
            name: "DatabaseOperations".to_string(),
            description: "Detect database operations from semantic context".to_string(),
            keywords: vec!["database", "query", "sql", "table", "record"].into_iter().map(String::from).collect(),
            suggested_effects: vec!["Database.Query".to_string()],
            confidence: 0.8,
        });

        // Pattern for file operations
        self.patterns.push(AIInferencePattern {
            name: "FileOperations".to_string(),
            description: "Detect file operations from semantic context".to_string(),
            keywords: vec!["file", "read", "write", "path", "directory"].into_iter().map(String::from).collect(),
            suggested_effects: vec!["IO.FileSystem.Read".to_string(), "IO.FileSystem.Write".to_string()],
            confidence: 0.7,
        });

        // Pattern for network operations
        self.patterns.push(AIInferencePattern {
            name: "NetworkOperations".to_string(),
            description: "Detect network operations from semantic context".to_string(),
            keywords: vec!["network", "http", "api", "request", "response"].into_iter().map(String::from).collect(),
            suggested_effects: vec!["IO.Network.Connect".to_string()],
            confidence: 0.75,
        });
    }
}

impl Default for AIInferenceAssistant {
    fn default() -> Self {
        Self::new()
    }
}

/// AI inference pattern for detecting effects
#[derive(Debug, Clone)]
pub struct AIInferencePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Keywords that trigger this pattern
    pub keywords: HashSet<String>,
    /// Effects suggested by this pattern
    pub suggested_effects: Vec<String>,
    /// Confidence level of suggestions
    pub confidence: f64,
}

impl AIInferencePattern {
    /// Check if this pattern matches a node
    pub fn matches(&self, node: &AstNode<Type>) -> bool {
        // Check AI context for keywords
        if let Some(ai_context) = &node.metadata.ai_context {
            if let Some(description) = &ai_context.description {
                let description_lower = description.to_lowercase();
                if self.keywords.iter().any(|keyword| description_lower.contains(keyword)) {
                    return true;
                }
            }
        }

        // Check business rules for keywords
        for rule in &node.metadata.business_rules {
            let rule_lower = rule.to_lowercase();
            if self.keywords.iter().any(|keyword| rule_lower.contains(keyword)) {
                return true;
            }
        }

        false
    }

    /// Suggest effects for a matching node
    pub fn suggest_effects(
        &self,
        node: &AstNode<Type>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, crate::EffectSystemError> {
        let mut effects = Vec::new();

        for effect_name in &self.suggested_effects {
            if registry.get_effect(effect_name).is_some() {
                let effect = Effect::new(effect_name.clone(), node.span)
                    .with_metadata(EffectInstanceMetadata {
                        ai_context: Some(format!("Suggested by AI pattern: {}", self.name)),
                        inferred: true,
                        confidence: self.confidence,
                        inference_source: Some(format!("AI pattern: {}", self.name)),
                        ..Default::default()
                    });
                effects.push(effect);
            }
        }

        Ok(effects)
    }
}

/// Analysis of effect usage in a function
#[derive(Debug)]
pub struct EffectUsageAnalysis {
    /// Function name
    pub function_name: Symbol,
    /// Effects declared in function signature
    pub declared_effects: Vec<Effect>,
    /// Effects actually used in function body
    pub actual_effects: Vec<Effect>,
    /// Consistency analysis
    pub effect_consistency: EffectConsistency,
    /// AI-generated insights
    pub ai_insights: Vec<String>,
}

impl EffectUsageAnalysis {
    /// Create a new effect usage analysis
    pub fn new(function_name: Symbol) -> Self {
        Self {
            function_name,
            declared_effects: Vec::new(),
            actual_effects: Vec::new(),
            effect_consistency: EffectConsistency::default(),
            ai_insights: Vec::new(),
        }
    }
}

/// Effect consistency analysis
#[derive(Debug, Default)]
pub struct EffectConsistency {
    /// Whether declared and actual effects are consistent
    pub is_consistent: bool,
    /// Effects used but not declared
    pub missing_declarations: Vec<String>,
    /// Effects declared but not used
    pub unused_declarations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::NodeId;

    #[test]
    fn test_inference_engine_creation() {
        let engine = EffectInferenceEngine::new();
        assert!(engine.inference_cache.is_empty());
        assert!(!engine.composition_analyzer.composition_rules.is_empty());
    }

    #[test]
    fn test_composition_rule() {
        let rule = CompositionRule {
            name: "TestRule".to_string(),
            description: "Test composition rule".to_string(),
            input_patterns: vec!["Effect1".to_string(), "Effect2".to_string()],
            output_effect: "ComposedEffect".to_string(),
            conditions: Vec::new(),
        };

        let effects = vec![
            Effect::new("Effect1".to_string(), Span::dummy()),
            Effect::new("Effect2".to_string(), Span::dummy()),
        ];

        assert!(rule.applies_to(&effects));

        let composed = rule.apply(&effects).unwrap();
        assert_eq!(composed.len(), 1);
        assert_eq!(composed[0].definition, "ComposedEffect");
    }

    #[test]
    fn test_ai_inference_pattern() {
        let pattern = AIInferencePattern {
            name: "TestPattern".to_string(),
            description: "Test pattern".to_string(),
            keywords: ["database", "query"].iter().map(|s| s.to_string()).collect(),
            suggested_effects: vec!["Database.Query".to_string()],
            confidence: 0.8,
        };

        // Create a node with database-related context
        let mut node = AstNode::new(
            Type::Primitive(prism_ast::PrimitiveType::String),
            Span::dummy(),
            NodeId::new(1),
        );
        node.metadata.business_rules.push("This function queries the database".to_string());

        assert!(pattern.matches(&node));
    }
} 