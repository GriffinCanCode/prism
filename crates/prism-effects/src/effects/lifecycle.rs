//! Effect Lifecycle Management
//!
//! This Smart Module represents the complete lifecycle of effects in Prism's system.
//! It unifies effect definitions, inference, handling, and execution into a single,
//! conceptually cohesive unit that manages effects from creation to completion.
//!
//! ## Conceptual Cohesion
//! 
//! This module embodies the business concept of "Effect Lifecycle" by bringing together:
//! - Effect definitions and registry (what effects exist)
//! - Effect inference and discovery (finding effects in code)
//! - Effect handling and execution (how effects are processed)
//! - Effect composition and optimization (combining effects efficiently)

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use crate::security_trust::{SecurityOperation, SecureExecutionContext, ValidatedCapability};
use crate::effects::definition::{Effect, EffectDefinition, EffectCategory, EffectParameter, EffectRegistry, EffectHierarchy};
use crate::effects::registry::EffectCompositionRule;
use prism_ast::{AstNode, Type, Expr, SecurityClassification};
use prism_common::span::Span;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Capability: Effect Lifecycle Management
/// Description: Unified effect system managing the complete lifecycle from definition to execution  
/// Dependencies: prism-ast, prism-common, security_trust

/// Lifecycle rule for effect management
#[derive(Debug, Clone)]
pub struct LifecycleRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule condition
    pub condition: String,
    /// Rule action
    pub action: String,
}

/// Effect lifecycle management system
#[derive(Debug)]
pub struct EffectLifecycle {
    /// Lifecycle rules
    pub rules: Vec<LifecycleRule>,
}

impl EffectLifecycle {
    /// Create new lifecycle manager
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
        }
    }
}

impl Default for EffectLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

/// The unified Effect Lifecycle system that manages all aspects of effect processing
#[derive(Debug)]
pub struct EffectLifecycleSystem {
    /// Effect definition and registry subsystem
    pub effect_registry: EffectRegistry,
    /// Effect inference and discovery subsystem
    pub inference_engine: EffectInferenceEngine,
    /// Effect handling and execution subsystem
    pub execution_engine: EffectExecutionEngine,
    /// Effect composition and optimization subsystem
    pub composition_engine: EffectCompositionEngine,
}

impl EffectLifecycleSystem {
    /// Create a new Effect Lifecycle system
    pub fn new() -> Self {
        Self {
            effect_registry: EffectRegistry::new(),
            inference_engine: EffectInferenceEngine::new(),
            execution_engine: EffectExecutionEngine::new(),
            composition_engine: EffectCompositionEngine::new(),
        }
    }

    /// Process the complete effect lifecycle for a code unit
    pub fn process_effect_lifecycle(
        &mut self,
        code_unit: &AstNode<Type>,
        security_context: &SecureExecutionContext,
    ) -> Result<EffectLifecycleResult, EffectError> {
        // Phase 1: Discover and infer effects
        let discovered_effects = self.inference_engine
            .discover_effects(code_unit, &self.effect_registry)?;

        // Phase 2: Compose and optimize effects
        let optimized_effects = self.composition_engine
            .compose_and_optimize(discovered_effects, &self.effect_registry)?;

        // Phase 3: Validate against security context
        self.validate_effects_against_security(&optimized_effects, security_context)?;

        // Phase 4: Prepare for execution
        let execution_plan = self.execution_engine
            .prepare_execution_plan(&optimized_effects, security_context)?;

        Ok(EffectLifecycleResult {
            discovered_effects: optimized_effects.clone(),
            execution_plan,
            security_validation: EffectSecurityValidation {
                validated: true,
                violations: Vec::new(),
                recommendations: Vec::new(),
            },
            lifecycle_metadata: EffectLifecycleMetadata {
                processing_duration: std::time::Duration::from_millis(50), // Simulated
                phases_completed: 4,
                optimizations_applied: self.composition_engine.last_optimizations_count(),
            },
        })
    }

    /// Execute effects with full lifecycle management
    pub fn execute_effects_with_lifecycle(
        &mut self,
        effects: &[Effect],
        security_context: &SecureExecutionContext,
    ) -> Result<EffectExecutionResult, EffectError> {
        // Compose effects for optimal execution
        let composed_effects = self.composition_engine
            .compose_for_execution(effects, &self.effect_registry)?;

        // Execute with full lifecycle tracking
        self.execution_engine.execute_effects_with_tracking(
            &composed_effects,
            security_context,
            &self.effect_registry,
        )
    }

    /// Validate effects against security context
    fn validate_effects_against_security(
        &self,
        effects: &[Effect],
        security_context: &SecureExecutionContext,
    ) -> Result<(), EffectError> {
        for effect in effects {
            // Check if effect is registered
            let effect_def = self.effect_registry.get_effect(&effect.definition)
                .ok_or_else(|| EffectError::UnknownEffect { 
                    name: effect.definition.clone() 
                })?;

            // Validate capability requirements
            for (capability_name, _permissions) in &effect_def.capability_requirements {
                let has_capability = security_context.capabilities.iter()
                    .any(|cap| cap.definition.name == *capability_name);

                if !has_capability {
                    return Err(EffectError::InsufficientCapability {
                        effect: effect.definition.clone(),
                        capability: capability_name.clone(),
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// SECTION: Effect Registry
// Manages effect definitions and their metadata
// ============================================================================

/// Registry of all known effects in the system
#[derive(Debug, Default)]
pub struct EffectRegistry {
    /// Map of effect names to their definitions
    pub effects: HashMap<String, EffectDefinition>,
    /// Hierarchical organization of effects
    pub hierarchy: EffectHierarchy,
    /// Effect composition rules
    pub composition_rules: Vec<EffectCompositionRule>,
    /// Effect categories for organization
    pub categories: HashMap<EffectCategory, Vec<String>>,
}

impl EffectRegistry {
    /// Create a new effect registry with built-in effects
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_effects();
        registry
    }

    /// Register a new effect definition
    pub fn register(&mut self, effect: EffectDefinition) -> Result<(), EffectError> {
        if self.effects.contains_key(&effect.name) {
            return Err(EffectError::EffectAlreadyRegistered {
                name: effect.name.clone(),
            });
        }
        
        // Add to hierarchy
        self.hierarchy.add_effect(&effect);
        
        // Add to category
        self.categories
            .entry(effect.category.clone())
            .or_default()
            .push(effect.name.clone());
        
        // Store definition
        self.effects.insert(effect.name.clone(), effect);
        Ok(())
    }

    /// Get an effect definition by name
    pub fn get_effect(&self, name: &str) -> Option<&EffectDefinition> {
        self.effects.get(name)
    }

    /// Check if an effect is a subtype of another
    pub fn is_subeffect(&self, child: &str, parent: &str) -> bool {
        self.hierarchy.is_subeffect(child, parent)
    }

    /// Get all effects in a category
    pub fn get_effects_by_category(&self, category: &EffectCategory) -> Vec<&EffectDefinition> {
        self.categories
            .get(category)
            .map(|effect_names| {
                effect_names
                    .iter()
                    .filter_map(|name| self.effects.get(name))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Register all built-in effects from PLD-003
    fn register_builtin_effects(&mut self) {
        let builtin_effects = vec![
            // IO Effects
            EffectDefinition::new(
                "IO.FileSystem.Read".to_string(),
                "Read files from the file system".to_string(),
                EffectCategory::IO,
            )
            .with_ai_context("Provides read access to file system resources")
            .with_security_implication("Can access sensitive file contents")
            .with_capability_requirement("FileSystem", vec!["Read".to_string()])
            .with_parameter(EffectParameter {
                name: "path".to_string(),
                parameter_type: "Path".to_string(),
                description: "File path to read".to_string(),
                required: true,
                constraints: vec!["Must be within allowed paths".to_string()],
            }),

            EffectDefinition::new(
                "IO.FileSystem.Write".to_string(),
                "Write files to the file system".to_string(),
                EffectCategory::IO,
            )
            .with_ai_context("Provides write access to file system resources")
            .with_security_implication("Can modify or create files")
            .with_capability_requirement("FileSystem", vec!["Write".to_string()])
            .with_parameter(EffectParameter {
                name: "path".to_string(),
                parameter_type: "Path".to_string(),
                description: "File path to write".to_string(),
                required: true,
                constraints: vec!["Must be within allowed paths".to_string()],
            }),

            EffectDefinition::new(
                "IO.Network.Connect".to_string(),
                "Establish network connections".to_string(),
                EffectCategory::Network,
            )
            .with_ai_context("Enables network communication")
            .with_security_implication("Can send data over network")
            .with_capability_requirement("Network", vec!["Connect".to_string()]),

            // Database Effects
            EffectDefinition::new(
                "Database.Query".to_string(),
                "Execute database queries".to_string(),
                EffectCategory::Database,
            )
            .with_ai_context("Provides database query capabilities")
            .with_security_implication("Can access database contents")
            .with_capability_requirement("Database", vec!["Query".to_string()]),

            EffectDefinition::new(
                "Database.Transaction".to_string(),
                "Manage database transactions".to_string(),
                EffectCategory::Database,
            )
            .with_ai_context("Provides transactional database operations")
            .with_security_implication("Can modify database state")
            .with_capability_requirement("Database", vec!["Transaction".to_string()]),

            // Cryptography Effects
            EffectDefinition::new(
                "Cryptography.KeyGeneration".to_string(),
                "Generate cryptographic keys".to_string(),
                EffectCategory::Security,
            )
            .with_ai_context("Generates cryptographic material")
            .with_security_implication("Creates sensitive cryptographic keys")
            .with_capability_requirement("Cryptography", vec!["KeyGeneration".to_string()]),

            EffectDefinition::new(
                "Cryptography.Encryption".to_string(),
                "Encrypt data using cryptographic algorithms".to_string(),
                EffectCategory::Security,
            )
            .with_ai_context("Provides data encryption capabilities")
            .with_security_implication("Processes potentially sensitive data")
            .with_capability_requirement("Cryptography", vec!["Encryption".to_string()]),

            // External AI Integration Effects (for server-based AI tools)
            EffectDefinition::new(
                "ExternalAI.DataExport".to_string(),
                "Export data for external AI analysis".to_string(),
                EffectCategory::IO,
            )
            .with_ai_context("Prepares and exports data for external AI systems to analyze")
            .with_security_implication("May expose sensitive data to external AI services")
            .with_capability_requirement("Network", vec!["Send".to_string()])
            .with_business_rule("Data must be sanitized before export"),

            EffectDefinition::new(
                "ExternalAI.MetadataGeneration".to_string(),
                "Generate AI-comprehensible metadata".to_string(),
                EffectCategory::Pure,
            )
            .with_ai_context("Creates structured metadata for external AI systems to understand code")
            .with_security_implication("Metadata may reveal code structure and business logic")
            .with_capability_requirement("FileSystem", vec!["Write".to_string()])
            .with_business_rule("Generated metadata must not include sensitive implementation details"),
        ];

        for effect in builtin_effects {
            let _ = self.register(effect);
        }
    }
}

/// Hierarchical organization of effects
#[derive(Debug, Default)]
pub struct EffectHierarchy {
    /// Parent-child relationships between effects
    pub relationships: HashMap<String, Vec<String>>,
    /// Root effects (no parents)
    pub roots: HashSet<String>,
}

impl EffectHierarchy {
    /// Add an effect to the hierarchy
    pub fn add_effect(&mut self, effect: &EffectDefinition) {
        if let Some(parent) = &effect.parent_effect {
            self.relationships
                .entry(parent.clone())
                .or_default()
                .push(effect.name.clone());
        } else {
            self.roots.insert(effect.name.clone());
        }
    }

    /// Check if one effect is a subeffect of another
    pub fn is_subeffect(&self, child: &str, parent: &str) -> bool {
        if child == parent {
            return true;
        }

        if let Some(children) = self.relationships.get(parent) {
            for child_effect in children {
                if child_effect == child || self.is_subeffect(child, child_effect) {
                    return true;
                }
            }
        }

        false
    }
}

/// Definition of an effect type
#[derive(Debug, Clone)]
pub struct EffectDefinition {
    /// Unique name of the effect
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Effect category for classification
    pub category: EffectCategory,
    /// Parent effect in the hierarchy
    pub parent_effect: Option<String>,
    /// AI-comprehensible context
    pub ai_context: Option<String>,
    /// Security implications of this effect
    pub security_implications: Vec<String>,
    /// Business rules associated with this effect
    pub business_rules: Vec<String>,
    /// Capability requirements
    pub capability_requirements: HashMap<String, Vec<String>>,
    /// Effect parameters
    pub parameters: Vec<EffectParameter>,
    /// Examples of effect usage
    pub examples: Vec<String>,
    /// Common mistakes to avoid
    pub common_mistakes: Vec<String>,
}

impl EffectDefinition {
    /// Create a new effect definition
    pub fn new(name: String, description: String, category: EffectCategory) -> Self {
        Self {
            name,
            description,
            category,
            parent_effect: None,
            ai_context: None,
            security_implications: Vec::new(),
            business_rules: Vec::new(),
            capability_requirements: HashMap::new(),
            parameters: Vec::new(),
            examples: Vec::new(),
            common_mistakes: Vec::new(),
        }
    }

    /// Add AI context for better comprehension
    pub fn with_ai_context(mut self, context: impl Into<String>) -> Self {
        self.ai_context = Some(context.into());
        self
    }

    /// Add a security implication
    pub fn with_security_implication(mut self, implication: impl Into<String>) -> Self {
        self.security_implications.push(implication.into());
        self
    }

    /// Add a capability requirement
    pub fn with_capability_requirement(mut self, capability: impl Into<String>, permissions: Vec<String>) -> Self {
        self.capability_requirements.insert(capability.into(), permissions);
        self
    }

    /// Add an effect parameter
    pub fn with_parameter(mut self, parameter: EffectParameter) -> Self {
        self.parameters.push(parameter);
        self
    }
}

/// Categories of effects as defined in PLD-003
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectCategory {
    /// Pure computation with no side effects
    Pure,
    /// Input/Output operations
    IO,
    /// Database operations
    Database,
    /// Network operations
    Network,
    /// Cryptographic operations
    Security,
    /// AI and machine learning operations
    AI,
    /// Memory management operations
    Memory,
    /// System-level operations
    System,
    /// Unsafe operations requiring special handling
    Unsafe,
    /// Custom effect category
    Custom(String),
}

/// Parameter for an effect
#[derive(Debug, Clone)]
pub struct EffectParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: String,
    /// Parameter description
    pub description: String,
    /// Whether the parameter is required
    pub required: bool,
    /// Constraints on the parameter
    pub constraints: Vec<String>,
}

/// Concrete effect instance with runtime values
#[derive(Debug, Clone)]
pub struct Effect {
    /// The effect definition this instance is based on
    pub definition: String,
    /// Runtime parameters for this effect
    pub parameters: HashMap<String, String>, // Simplified - would be actual values
    /// Source location where this effect occurs
    pub span: Span,
    /// Metadata for this specific effect instance
    pub metadata: EffectInstanceMetadata,
}

impl Effect {
    /// Create a new effect instance
    pub fn new(definition: String, span: Span) -> Self {
        Self {
            definition,
            parameters: HashMap::new(),
            span,
            metadata: EffectInstanceMetadata::default(),
        }
    }

    /// Add a parameter to this effect
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(name.into(), value.into());
        self
    }
}

/// Metadata for a specific effect instance
#[derive(Debug, Clone, Default)]
pub struct EffectInstanceMetadata {
    /// AI-readable context for this specific instance
    pub ai_context: Option<String>,
    /// Security classification of data processed
    pub security_classification: SecurityClassification,
    /// Whether this effect was inferred or explicit
    pub inferred: bool,
    /// Confidence level in effect inference (0.0 to 1.0)
    pub confidence: f64,
    /// Source of effect inference
    pub inference_source: Option<String>,
}

/// Rule for composing multiple effects
#[derive(Debug, Clone)]
pub struct EffectCompositionRule {
    /// Rule name for identification
    pub name: String,
    /// Description of what this rule does
    pub description: String,
    /// Input effects that trigger this rule
    pub input_effects: Vec<String>,
    /// Output effect produced by composition
    pub output_effect: String,
    /// Conditions under which this rule applies
    pub conditions: Vec<String>, // Simplified - would be expressions
    /// AI-readable explanation of the composition
    pub ai_explanation: Option<String>,
}

// ============================================================================
// SECTION: Effect Inference Engine
// Discovers and infers effects from code analysis
// ============================================================================

/// Engine for discovering and inferring effects from code
#[derive(Debug)]
pub struct EffectInferenceEngine {
    /// Configuration for inference
    pub config: InferenceConfig,
    /// Cache of previously inferred effects
    pub inference_cache: HashMap<String, Vec<Effect>>, // Simplified key
    /// Effect composition analyzer
    pub composition_analyzer: EffectCompositionAnalyzer,
    /// AI-assisted inference patterns
    pub ai_patterns: Vec<AIInferencePattern>,
}

impl EffectInferenceEngine {
    /// Create a new inference engine
    pub fn new() -> Self {
        let mut engine = Self {
            config: InferenceConfig::default(),
            inference_cache: HashMap::new(),
            composition_analyzer: EffectCompositionAnalyzer::new(),
            ai_patterns: Vec::new(),
        };
        engine.initialize_ai_patterns();
        engine
    }

    /// Discover effects in a code unit
    pub fn discover_effects(
        &mut self,
        code_unit: &AstNode<Type>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, EffectError> {
        // Check cache first
        let cache_key = self.generate_cache_key(code_unit);
        if let Some(cached_effects) = self.inference_cache.get(&cache_key) {
            return Ok(cached_effects.clone());
        }

        let mut discovered_effects = Vec::new();

        // Apply AI patterns for inference
        for pattern in &self.ai_patterns {
            if pattern.matches_code_unit(code_unit) {
                let pattern_effects = pattern.infer_effects(code_unit, registry)?;
                discovered_effects.extend(pattern_effects);
            }
        }

        // Apply composition analysis
        let composed_effects = self.composition_analyzer
            .analyze_and_compose(&discovered_effects, registry)?;

        // Cache results
        if self.config.cache_results {
            self.inference_cache.insert(cache_key, composed_effects.clone());
        }

        Ok(composed_effects)
    }

    /// Generate cache key for code unit
    fn generate_cache_key(&self, code_unit: &AstNode<Type>) -> String {
        // Simplified - would use proper hashing of AST structure
        format!("code_unit_{}", code_unit.span.start)
    }

    /// Initialize AI inference patterns
    fn initialize_ai_patterns(&mut self) {
        self.ai_patterns = vec![
            AIInferencePattern {
                name: "FunctionCallPattern".to_string(),
                description: "Infers effects from function calls".to_string(),
                keywords: vec!["call".to_string(), "invoke".to_string()].into_iter().collect(),
                infer_fn: Box::new(|code_unit, registry| {
                    // Simplified inference - would analyze actual function calls
                    Ok(vec![Effect::new("IO.FileSystem.Read".to_string(), code_unit.span)])
                }),
            },
            AIInferencePattern {
                name: "DatabasePattern".to_string(),
                description: "Infers database effects from database operations".to_string(),
                keywords: vec!["query".to_string(), "select".to_string(), "insert".to_string()].into_iter().collect(),
                infer_fn: Box::new(|code_unit, registry| {
                    Ok(vec![Effect::new("Database.Query".to_string(), code_unit.span)])
                }),
            },
        ];
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

/// AI pattern for effect inference
#[derive(Debug)]
pub struct AIInferencePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Keywords that trigger this pattern
    pub keywords: HashSet<String>,
    /// Inference function
    pub infer_fn: Box<dyn Fn(&AstNode<Type>, &EffectRegistry) -> Result<Vec<Effect>, EffectError> + Send + Sync>,
}

impl AIInferencePattern {
    /// Check if this pattern matches the code unit
    pub fn matches_code_unit(&self, code_unit: &AstNode<Type>) -> bool {
        // Simplified - would analyze AST structure and content
        true // For now, all patterns match
    }

    /// Infer effects using this pattern
    pub fn infer_effects(
        &self,
        code_unit: &AstNode<Type>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, EffectError> {
        (self.infer_fn)(code_unit, registry)
    }
}

/// Analyzer for effect composition
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

    /// Analyze and compose effects
    pub fn analyze_and_compose(
        &self,
        effects: &[Effect],
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, EffectError> {
        let mut composed_effects = effects.to_vec();

        // Apply composition rules
        for rule in &self.composition_rules {
            if rule.applies_to(effects) {
                composed_effects = rule.apply(composed_effects)?;
            }
        }

        Ok(composed_effects)
    }

    /// Register default composition rules
    fn register_default_rules(&mut self) {
        self.composition_rules = vec![
            CompositionRule {
                name: "DatabaseTransactionComposition".to_string(),
                description: "Combines multiple database operations into a transaction".to_string(),
                input_patterns: vec!["Database.Query".to_string()],
                output_effect: "Database.Transaction".to_string(),
                apply_fn: Box::new(|effects| {
                    // If we have multiple database queries, combine into transaction
                    let db_effects: Vec<_> = effects.iter()
                        .filter(|e| e.definition.starts_with("Database"))
                        .collect();
                    
                    if db_effects.len() > 1 {
                        let mut result = effects.clone();
                        let span = db_effects[0].span;
                        result.push(Effect::new("Database.Transaction".to_string(), span));
                        Ok(result)
                    } else {
                        Ok(effects)
                    }
                }),
            },
        ];
    }
}

/// Rule for composing effects
#[derive(Debug)]
pub struct CompositionRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Input effect patterns
    pub input_patterns: Vec<String>,
    /// Output effect
    pub output_effect: String,
    /// Function to apply the rule
    pub apply_fn: Box<dyn Fn(Vec<Effect>) -> Result<Vec<Effect>, EffectError> + Send + Sync>,
}

impl CompositionRule {
    /// Check if this rule applies to the effects
    pub fn applies_to(&self, effects: &[Effect]) -> bool {
        self.input_patterns.iter().any(|pattern| {
            effects.iter().any(|effect| effect.definition.contains(pattern))
        })
    }

    /// Apply this rule to the effects
    pub fn apply(&self, effects: Vec<Effect>) -> Result<Vec<Effect>, EffectError> {
        (self.apply_fn)(effects)
    }
}

// ============================================================================
// SECTION: Effect Execution Engine
// Handles effect execution and lifecycle management
// ============================================================================

/// Engine for executing effects with lifecycle management
#[derive(Debug)]
pub struct EffectExecutionEngine {
    /// Registered effect handlers
    pub handlers: HashMap<String, Box<dyn EffectHandler>>,
    /// Execution configuration
    pub config: ExecutionConfig,
    /// Execution history for analysis
    pub execution_history: Vec<ExecutionRecord>,
}

impl EffectExecutionEngine {
    /// Create a new execution engine
    pub fn new() -> Self {
        let mut engine = Self {
            handlers: HashMap::new(),
            config: ExecutionConfig::default(),
            execution_history: Vec::new(),
        };
        engine.register_builtin_handlers();
        engine
    }

    /// Prepare an execution plan for effects
    pub fn prepare_execution_plan(
        &self,
        effects: &[Effect],
        security_context: &SecureExecutionContext,
    ) -> Result<EffectExecutionPlan, EffectError> {
        let mut execution_steps = Vec::new();

        for effect in effects {
            // Find appropriate handler
            let handler = self.handlers.get(&effect.definition)
                .ok_or_else(|| EffectError::NoHandlerFound {
                    effect: effect.definition.clone(),
                })?;

            // Create execution step
            execution_steps.push(ExecutionStep {
                effect: effect.clone(),
                handler_name: effect.definition.clone(),
                estimated_duration: handler.estimate_execution_time(effect),
                dependencies: Vec::new(), // Would analyze dependencies
                security_requirements: handler.get_security_requirements(effect),
            });
        }

        let total_duration = execution_steps.iter()
            .map(|step| step.estimated_duration)
            .sum();
        let parallel_groups = self.identify_parallel_groups(&execution_steps);
        let security_checkpoints = self.identify_security_checkpoints(&execution_steps);
            
        Ok(EffectExecutionPlan {
            steps: execution_steps,
            total_estimated_duration: total_duration,
            parallel_groups,
            security_checkpoints,
        })
    }

    /// Execute effects with full lifecycle tracking
    pub fn execute_effects_with_tracking(
        &mut self,
        effects: &[Effect],
        security_context: &SecureExecutionContext,
        registry: &EffectRegistry,
    ) -> Result<EffectExecutionResult, EffectError> {
        let start_time = std::time::Instant::now();
        let mut execution_results = Vec::new();
        let mut security_events = Vec::new();

        for effect in effects {
            // Get handler
            let handler = self.handlers.get(&effect.definition)
                .ok_or_else(|| EffectError::NoHandlerFound {
                    effect: effect.definition.clone(),
                })?;

            // Execute effect
            let effect_start = std::time::Instant::now();
            let result = handler.execute_effect(effect, security_context)?;
            let effect_duration = effect_start.elapsed();

            // Record execution
            execution_results.push(SingleEffectResult {
                effect: effect.clone(),
                success: result.success,
                output: result.output,
                duration: effect_duration,
                security_events: result.security_events.clone(),
            });

            security_events.extend(result.security_events);
        }

        let total_duration = start_time.elapsed();

        // Record execution history
        let execution_record = ExecutionRecord {
            effects: effects.to_vec(),
            duration: total_duration,
            success: execution_results.iter().all(|r| r.success),
            timestamp: std::time::Instant::now(),
        };
        self.execution_history.push(execution_record);

        Ok(EffectExecutionResult {
            individual_results: execution_results,
            total_duration,
            overall_success: security_events.is_empty(),
            security_events,
            performance_metrics: PerformanceMetrics {
                total_effects_executed: effects.len(),
                average_effect_duration: total_duration / effects.len() as u32,
                parallel_execution_efficiency: 1.0, // Simplified
            },
        })
    }

    /// Identify execution steps that can run in parallel
    fn identify_parallel_groups(&self, steps: &[ExecutionStep]) -> Vec<ParallelGroup> {
        // Simplified - would analyze dependencies
        vec![ParallelGroup {
            steps: (0..steps.len()).collect(),
            estimated_duration: steps.iter()
                .map(|s| s.estimated_duration)
                .max()
                .unwrap_or(std::time::Duration::from_millis(0)),
        }]
    }

    /// Identify security checkpoints in execution
    fn identify_security_checkpoints(&self, steps: &[ExecutionStep]) -> Vec<SecurityCheckpoint> {
        steps.iter().enumerate()
            .filter(|(_, step)| !step.security_requirements.is_empty())
            .map(|(index, step)| SecurityCheckpoint {
                step_index: index,
                requirements: step.security_requirements.clone(),
                checkpoint_type: SecurityCheckpointType::CapabilityValidation,
            })
            .collect()
    }

    /// Register built-in effect handlers
    fn register_builtin_handlers(&mut self) {
        // File system handler
        self.handlers.insert(
            "IO.FileSystem.Read".to_string(),
            Box::new(FileSystemReadHandler::new()),
        );
        self.handlers.insert(
            "IO.FileSystem.Write".to_string(),
            Box::new(FileSystemWriteHandler::new()),
        );

        // Network handler
        self.handlers.insert(
            "IO.Network.Connect".to_string(),
            Box::new(NetworkConnectHandler::new()),
        );

        // Database handlers
        self.handlers.insert(
            "Database.Query".to_string(),
            Box::new(DatabaseQueryHandler::new()),
        );
        self.handlers.insert(
            "Database.Transaction".to_string(),
            Box::new(DatabaseTransactionHandler::new()),
        );
    }
}

/// Configuration for effect execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum execution time for single effect
    pub max_effect_execution_time: std::time::Duration,
    /// Whether to enable parallel execution
    pub enable_parallel_execution: bool,
    /// Whether to record execution history
    pub record_execution_history: bool,
    /// Maximum number of concurrent effects
    pub max_concurrent_effects: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_effect_execution_time: std::time::Duration::from_secs(30),
            enable_parallel_execution: true,
            record_execution_history: true,
            max_concurrent_effects: 10,
        }
    }
}

/// Trait for effect handlers
pub trait EffectHandler: std::fmt::Debug + Send + Sync {
    /// Get the name of this handler
    fn name(&self) -> &str;

    /// Execute an effect
    fn execute_effect(
        &self,
        effect: &Effect,
        security_context: &SecureExecutionContext,
    ) -> Result<EffectHandlerResult, EffectError>;

    /// Estimate execution time for an effect
    fn estimate_execution_time(&self, effect: &Effect) -> std::time::Duration;

    /// Get security requirements for executing this effect
    fn get_security_requirements(&self, effect: &Effect) -> Vec<String>;
}

/// Result from effect handler execution
#[derive(Debug, Clone)]
pub struct EffectHandlerResult {
    /// Whether execution was successful
    pub success: bool,
    /// Output from the effect (if any)
    pub output: Option<String>,
    /// Security events generated
    pub security_events: Vec<SecurityEvent>,
}

/// Security event during effect execution
#[derive(Debug, Clone)]
pub enum SecurityEvent {
    /// Capability was used
    CapabilityUsed { capability: String, operation: String },
    /// Security policy was enforced
    PolicyEnforced { policy: String, action: String },
    /// Unauthorized access attempt
    UnauthorizedAccess { resource: String, capability: String },
}

/// Execution plan for effects
#[derive(Debug, Clone)]
pub struct EffectExecutionPlan {
    /// Individual execution steps
    pub steps: Vec<ExecutionStep>,
    /// Total estimated duration
    pub total_estimated_duration: std::time::Duration,
    /// Groups of steps that can execute in parallel
    pub parallel_groups: Vec<ParallelGroup>,
    /// Security checkpoints during execution
    pub security_checkpoints: Vec<SecurityCheckpoint>,
}

/// Single execution step
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Effect to execute
    pub effect: Effect,
    /// Name of handler to use
    pub handler_name: String,
    /// Estimated execution duration
    pub estimated_duration: std::time::Duration,
    /// Dependencies on other steps
    pub dependencies: Vec<usize>,
    /// Security requirements
    pub security_requirements: Vec<String>,
}

/// Group of steps that can execute in parallel
#[derive(Debug, Clone)]
pub struct ParallelGroup {
    /// Indices of steps in this group
    pub steps: Vec<usize>,
    /// Estimated duration for the group
    pub estimated_duration: std::time::Duration,
}

/// Security checkpoint during execution
#[derive(Debug, Clone)]
pub struct SecurityCheckpoint {
    /// Step index where checkpoint occurs
    pub step_index: usize,
    /// Security requirements to check
    pub requirements: Vec<String>,
    /// Type of security checkpoint
    pub checkpoint_type: SecurityCheckpointType,
}

/// Types of security checkpoints
#[derive(Debug, Clone)]
pub enum SecurityCheckpointType {
    CapabilityValidation,
    InformationFlowCheck,
    PolicyEnforcement,
}

/// Record of effect execution
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Effects that were executed
    pub effects: Vec<Effect>,
    /// Total execution duration
    pub duration: std::time::Duration,
    /// Whether execution was successful
    pub success: bool,
    /// When execution occurred
    pub timestamp: std::time::Instant,
}

// Example handler implementations
#[derive(Debug)]
pub struct FileSystemReadHandler {
    name: String,
}

impl FileSystemReadHandler {
    pub fn new() -> Self {
        Self {
            name: "FileSystemReadHandler".to_string(),
        }
    }
}

impl EffectHandler for FileSystemReadHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn execute_effect(
        &self,
        effect: &Effect,
        security_context: &SecureExecutionContext,
    ) -> Result<EffectHandlerResult, EffectError> {
        // Simplified implementation
        Ok(EffectHandlerResult {
            success: true,
            output: Some("File content".to_string()),
            security_events: vec![SecurityEvent::CapabilityUsed {
                capability: "FileSystem".to_string(),
                operation: "Read".to_string(),
            }],
        })
    }

    fn estimate_execution_time(&self, _effect: &Effect) -> std::time::Duration {
        std::time::Duration::from_millis(100)
    }

    fn get_security_requirements(&self, _effect: &Effect) -> Vec<String> {
        vec!["FileSystem".to_string()]
    }
}

// Similar simplified implementations for other handlers
#[derive(Debug)]
pub struct FileSystemWriteHandler { name: String }
impl FileSystemWriteHandler { pub fn new() -> Self { Self { name: "FileSystemWriteHandler".to_string() } } }
impl EffectHandler for FileSystemWriteHandler {
    fn name(&self) -> &str { &self.name }
    fn execute_effect(&self, _effect: &Effect, _security_context: &SecureExecutionContext) -> Result<EffectHandlerResult, EffectError> {
        Ok(EffectHandlerResult { success: true, output: None, security_events: vec![SecurityEvent::CapabilityUsed { capability: "FileSystem".to_string(), operation: "Write".to_string() }] })
    }
    fn estimate_execution_time(&self, _effect: &Effect) -> std::time::Duration { std::time::Duration::from_millis(150) }
    fn get_security_requirements(&self, _effect: &Effect) -> Vec<String> { vec!["FileSystem".to_string()] }
}

#[derive(Debug)]
pub struct NetworkConnectHandler { name: String }
impl NetworkConnectHandler { pub fn new() -> Self { Self { name: "NetworkConnectHandler".to_string() } } }
impl EffectHandler for NetworkConnectHandler {
    fn name(&self) -> &str { &self.name }
    fn execute_effect(&self, _effect: &Effect, _security_context: &SecureExecutionContext) -> Result<EffectHandlerResult, EffectError> {
        Ok(EffectHandlerResult { success: true, output: Some("Connected".to_string()), security_events: vec![SecurityEvent::CapabilityUsed { capability: "Network".to_string(), operation: "Connect".to_string() }] })
    }
    fn estimate_execution_time(&self, _effect: &Effect) -> std::time::Duration { std::time::Duration::from_millis(500) }
    fn get_security_requirements(&self, _effect: &Effect) -> Vec<String> { vec!["Network".to_string()] }
}

#[derive(Debug)]
pub struct DatabaseQueryHandler { name: String }
impl DatabaseQueryHandler { pub fn new() -> Self { Self { name: "DatabaseQueryHandler".to_string() } } }
impl EffectHandler for DatabaseQueryHandler {
    fn name(&self) -> &str { &self.name }
    fn execute_effect(&self, _effect: &Effect, _security_context: &SecureExecutionContext) -> Result<EffectHandlerResult, EffectError> {
        Ok(EffectHandlerResult { success: true, output: Some("Query result".to_string()), security_events: vec![SecurityEvent::CapabilityUsed { capability: "Database".to_string(), operation: "Query".to_string() }] })
    }
    fn estimate_execution_time(&self, _effect: &Effect) -> std::time::Duration { std::time::Duration::from_millis(200) }
    fn get_security_requirements(&self, _effect: &Effect) -> Vec<String> { vec!["Database".to_string()] }
}

#[derive(Debug)]
pub struct DatabaseTransactionHandler { name: String }
impl DatabaseTransactionHandler { pub fn new() -> Self { Self { name: "DatabaseTransactionHandler".to_string() } } }
impl EffectHandler for DatabaseTransactionHandler {
    fn name(&self) -> &str { &self.name }
    fn execute_effect(&self, _effect: &Effect, _security_context: &SecureExecutionContext) -> Result<EffectHandlerResult, EffectError> {
        Ok(EffectHandlerResult { success: true, output: Some("Transaction completed".to_string()), security_events: vec![SecurityEvent::CapabilityUsed { capability: "Database".to_string(), operation: "Transaction".to_string() }] })
    }
    fn estimate_execution_time(&self, _effect: &Effect) -> std::time::Duration { std::time::Duration::from_millis(300) }
    fn get_security_requirements(&self, _effect: &Effect) -> Vec<String> { vec!["Database".to_string()] }
}

// ============================================================================
// SECTION: Effect Composition Engine
// Optimizes and composes effects for efficient execution
// ============================================================================

/// Engine for composing and optimizing effects
#[derive(Debug)]
pub struct EffectCompositionEngine {
    /// Composition strategies
    pub strategies: Vec<CompositionStrategy>,
    /// Optimization rules
    pub optimization_rules: Vec<OptimizationRule>,
    /// Last optimization count for tracking
    pub last_optimizations_count: usize,
}

impl EffectCompositionEngine {
    /// Create a new composition engine
    pub fn new() -> Self {
        let mut engine = Self {
            strategies: Vec::new(),
            optimization_rules: Vec::new(),
            last_optimizations_count: 0,
        };
        engine.initialize_strategies();
        engine
    }

    /// Compose and optimize effects
    pub fn compose_and_optimize(
        &mut self,
        effects: Vec<Effect>,
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, EffectError> {
        let mut optimized_effects = effects;
        let mut optimizations_applied = 0;

        // Apply composition strategies
        for strategy in &self.strategies {
            if strategy.applies_to(&optimized_effects) {
                optimized_effects = strategy.apply(optimized_effects, registry)?;
                optimizations_applied += 1;
            }
        }

        // Apply optimization rules
        for rule in &self.optimization_rules {
            if rule.can_optimize(&optimized_effects) {
                optimized_effects = rule.optimize(optimized_effects)?;
                optimizations_applied += 1;
            }
        }

        self.last_optimizations_count = optimizations_applied;
        Ok(optimized_effects)
    }

    /// Compose effects specifically for execution
    pub fn compose_for_execution(
        &self,
        effects: &[Effect],
        registry: &EffectRegistry,
    ) -> Result<Vec<Effect>, EffectError> {
        // Apply execution-specific optimizations
        let mut composed = effects.to_vec();

        // Group similar effects
        composed = self.group_similar_effects(composed);

        // Optimize for parallel execution
        composed = self.optimize_for_parallelism(composed);

        Ok(composed)
    }

    /// Get the number of optimizations applied in the last operation
    pub fn last_optimizations_count(&self) -> usize {
        self.last_optimizations_count
    }

    /// Group similar effects together
    fn group_similar_effects(&self, effects: Vec<Effect>) -> Vec<Effect> {
        // Simplified grouping logic
        effects
    }

    /// Optimize effects for parallel execution
    fn optimize_for_parallelism(&self, effects: Vec<Effect>) -> Vec<Effect> {
        // Simplified parallelism optimization
        effects
    }

    /// Initialize composition strategies
    fn initialize_strategies(&mut self) {
        self.strategies = vec![
            CompositionStrategy {
                name: "DatabaseBatching".to_string(),
                description: "Batches multiple database operations".to_string(),
                applies_fn: Box::new(|effects| {
                    effects.iter().filter(|e| e.definition.starts_with("Database")).count() > 1
                }),
                apply_fn: Box::new(|effects, _registry| {
                    // Simplified batching
                    Ok(effects)
                }),
            },
        ];

        self.optimization_rules = vec![
            OptimizationRule {
                name: "DuplicateElimination".to_string(),
                description: "Eliminates duplicate effects".to_string(),
                can_optimize_fn: Box::new(|effects| {
                    let mut seen = HashSet::new();
                    effects.iter().any(|e| !seen.insert(&e.definition))
                }),
                optimize_fn: Box::new(|effects| {
                    let mut unique_effects = Vec::new();
                    let mut seen = HashSet::new();
                    
                    for effect in effects {
                        if seen.insert(effect.definition.clone()) {
                            unique_effects.push(effect);
                        }
                    }
                    
                    Ok(unique_effects)
                }),
            },
        ];
    }
}

/// Strategy for composing effects
#[derive(Debug)]
pub struct CompositionStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Function to check if strategy applies
    pub applies_fn: Box<dyn Fn(&[Effect]) -> bool + Send + Sync>,
    /// Function to apply the strategy
    pub apply_fn: Box<dyn Fn(Vec<Effect>, &EffectRegistry) -> Result<Vec<Effect>, EffectError> + Send + Sync>,
}

impl CompositionStrategy {
    /// Check if this strategy applies to the effects
    pub fn applies_to(&self, effects: &[Effect]) -> bool {
        (self.applies_fn)(effects)
    }

    /// Apply this strategy to the effects
    pub fn apply(&self, effects: Vec<Effect>, registry: &EffectRegistry) -> Result<Vec<Effect>, EffectError> {
        (self.apply_fn)(effects, registry)
    }
}

/// Rule for optimizing effects
#[derive(Debug)]
pub struct OptimizationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Function to check if optimization can be applied
    pub can_optimize_fn: Box<dyn Fn(&[Effect]) -> bool + Send + Sync>,
    /// Function to apply the optimization
    pub optimize_fn: Box<dyn Fn(Vec<Effect>) -> Result<Vec<Effect>, EffectError> + Send + Sync>,
}

impl OptimizationRule {
    /// Check if this rule can optimize the effects
    pub fn can_optimize(&self, effects: &[Effect]) -> bool {
        (self.can_optimize_fn)(effects)
    }

    /// Apply this optimization rule
    pub fn optimize(&self, effects: Vec<Effect>) -> Result<Vec<Effect>, EffectError> {
        (self.optimize_fn)(effects)
    }
}

// ============================================================================
// SECTION: Common Types and Results
// Shared types used across the effect lifecycle system
// ============================================================================

/// Result of complete effect lifecycle processing
#[derive(Debug, Clone)]
pub struct EffectLifecycleResult {
    /// Effects discovered and optimized
    pub discovered_effects: Vec<Effect>,
    /// Execution plan for the effects
    pub execution_plan: EffectExecutionPlan,
    /// Security validation results
    pub security_validation: EffectSecurityValidation,
    /// Lifecycle processing metadata
    pub lifecycle_metadata: EffectLifecycleMetadata,
}

/// Security validation results for effects
#[derive(Debug, Clone)]
pub struct EffectSecurityValidation {
    /// Whether all effects passed security validation
    pub validated: bool,
    /// Security violations found
    pub violations: Vec<String>,
    /// Security recommendations
    pub recommendations: Vec<String>,
}

/// Metadata about effect lifecycle processing
#[derive(Debug, Clone)]
pub struct EffectLifecycleMetadata {
    /// Total processing duration
    pub processing_duration: std::time::Duration,
    /// Number of lifecycle phases completed
    pub phases_completed: usize,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
}

/// Result of effect execution
#[derive(Debug, Clone)]
pub struct EffectExecutionResult {
    /// Results of individual effect executions
    pub individual_results: Vec<SingleEffectResult>,
    /// Total execution duration
    pub total_duration: std::time::Duration,
    /// Whether overall execution was successful
    pub overall_success: bool,
    /// Security events generated during execution
    pub security_events: Vec<SecurityEvent>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Result of executing a single effect
#[derive(Debug, Clone)]
pub struct SingleEffectResult {
    /// The effect that was executed
    pub effect: Effect,
    /// Whether execution was successful
    pub success: bool,
    /// Output from execution (if any)
    pub output: Option<String>,
    /// Duration of execution
    pub duration: std::time::Duration,
    /// Security events for this effect
    pub security_events: Vec<SecurityEvent>,
}

/// Performance metrics for effect execution
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total number of effects executed
    pub total_effects_executed: usize,
    /// Average duration per effect
    pub average_effect_duration: std::time::Duration,
    /// Efficiency of parallel execution (0.0 to 1.0)
    pub parallel_execution_efficiency: f64,
}

/// Errors that can occur in the effect lifecycle system
#[derive(Debug, Error)]
pub enum EffectError {
    /// Effect already registered
    #[error("Effect '{name}' is already registered")]
    EffectAlreadyRegistered { name: String },

    /// Unknown effect
    #[error("Unknown effect: {name}")]
    UnknownEffect { name: String },

    /// Insufficient capability for effect
    #[error("Effect '{effect}' requires capability '{capability}' which is not available")]
    InsufficientCapability { effect: String, capability: String },

    /// No handler found for effect
    #[error("No handler found for effect: {effect}")]
    NoHandlerFound { effect: String },

    /// Effect inference failed
    #[error("Effect inference failed: {reason}")]
    InferenceFailed { reason: String },

    /// Effect execution failed
    #[error("Effect execution failed: {reason}")]
    ExecutionFailed { reason: String },

    /// Effect composition failed
    #[error("Effect composition failed: {reason}")]
    CompositionFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_lifecycle_system_creation() {
        let system = EffectLifecycleSystem::new();
        assert!(!system.effect_registry.effects.is_empty());
        assert!(!system.inference_engine.ai_patterns.is_empty());
        assert!(!system.execution_engine.handlers.is_empty());
    }

    #[test]
    fn test_effect_registry() {
        let registry = EffectRegistry::new();
        assert!(registry.get_effect("IO.FileSystem.Read").is_some());
        assert!(registry.get_effect("Database.Query").is_some());
        assert!(registry.get_effect("NonExistent").is_none());
    }

    #[test]
    fn test_effect_hierarchy() {
        let mut hierarchy = EffectHierarchy::default();
        let parent_effect = EffectDefinition::new(
            "Parent".to_string(),
            "Parent effect".to_string(),
            EffectCategory::IO,
        );
        let mut child_effect = EffectDefinition::new(
            "Child".to_string(),
            "Child effect".to_string(),
            EffectCategory::IO,
        );
        child_effect.parent_effect = Some("Parent".to_string());

        hierarchy.add_effect(&parent_effect);
        hierarchy.add_effect(&child_effect);

        assert!(hierarchy.is_subeffect("Child", "Parent"));
        assert!(hierarchy.is_subeffect("Parent", "Parent"));
        assert!(!hierarchy.is_subeffect("Parent", "Child"));
    }

    #[test]
    fn test_effect_creation() {
        let span = Span::new(0, 0, 0.into());
        let effect = Effect::new("TestEffect".to_string(), span)
            .with_parameter("key", "value");
        
        assert_eq!(effect.definition, "TestEffect");
        assert_eq!(effect.parameters.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_composition_engine() {
        let mut engine = EffectCompositionEngine::new();
        let span = Span::new(0, 0, 0.into());
        let effects = vec![
            Effect::new("Database.Query".to_string(), span),
            Effect::new("Database.Query".to_string(), span), // Duplicate
        ];
        
        let registry = EffectRegistry::new();
        let result = engine.compose_and_optimize(effects, &registry);
        assert!(result.is_ok());
    }
} 