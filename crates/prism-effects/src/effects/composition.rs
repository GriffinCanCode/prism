//! Effect Composition System
//!
//! This module provides sophisticated effect composition with validation, dependency analysis,
//! and conflict detection to ensure safe and efficient effect combinations.
//!
//! ## Design Principles
//!
//! 1. **Safety First**: Detect and prevent conflicting effect compositions
//! 2. **Dependency Analysis**: Understand and respect effect dependencies
//! 3. **Performance Optimization**: Optimize composed effects for execution
//! 4. **Validation**: Ensure composed effects are semantically correct
//! 5. **AI-Comprehensible**: Generate structured metadata for analysis

use crate::effects::definition::{Effect, EffectDefinition, EffectRegistry};
use crate::EffectSystemError;
use prism_ast::{AstNode, Expr, SecurityClassification};
use prism_common::span::Span;
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Effect composition system with validation and optimization
#[derive(Debug)]
pub struct EffectComposition {
    /// Composition strategies
    pub strategies: Vec<CompositionStrategy>,
    /// Optimization rules
    pub optimization_rules: Vec<OptimizationRule>,
    /// Composition validator
    pub validator: CompositionValidator,
    /// Dependency analyzer
    pub dependency_analyzer: DependencyAnalyzer,
    /// Conflict detector
    pub conflict_detector: ConflictDetector,
}

impl EffectComposition {
    /// Create new composition system
    pub fn new() -> Self {
        let mut composition = Self {
            strategies: Vec::new(),
            optimization_rules: Vec::new(),
            validator: CompositionValidator::new(),
            dependency_analyzer: DependencyAnalyzer::new(),
            conflict_detector: ConflictDetector::new(),
        };
        composition.initialize_default_strategies();
        composition
    }

    /// Compose effects with full validation and optimization
    pub fn compose(&self, effects: Vec<Effect>, operator: CompositionOperator) -> Result<Effect, EffectSystemError> {
        if effects.is_empty() {
            return Err(EffectSystemError::EffectValidationFailed {
                reason: "Cannot compose empty effect list".to_string(),
            });
        }

        if effects.len() == 1 {
            return Ok(effects.into_iter().next().unwrap());
        }

        // Step 1: Validate composition is possible
        let validation_result = self.validator.validate_composition(&effects, &operator)?;
        if !validation_result.is_valid {
            return Err(EffectSystemError::EffectValidationFailed {
                reason: format!("Composition validation failed: {}", validation_result.error_message.unwrap_or_default()),
            });
        }

        // Step 2: Analyze dependencies
        let dependency_graph = self.dependency_analyzer.analyze_dependencies(&effects)?;
        let execution_order = dependency_graph.topological_sort()?;

        // Step 3: Detect conflicts
        let conflict_analysis = self.conflict_detector.detect_conflicts(&effects)?;
        if !conflict_analysis.conflicts.is_empty() {
            return Err(EffectSystemError::EffectValidationFailed {
                reason: format!("Effect conflicts detected: {}", 
                    conflict_analysis.conflicts.iter()
                        .map(|c| c.description.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        }

        // Step 4: Apply composition strategy
        let strategy = self.select_composition_strategy(&effects, &operator)?;
        let mut composed_effects = strategy.apply_composition(effects, &dependency_graph, &execution_order)?;

        // Step 5: Apply optimizations
        for rule in &self.optimization_rules {
            if rule.can_optimize(&composed_effects) {
                composed_effects = rule.optimize(composed_effects)?;
            }
        }

        // Step 6: Create final composed effect
        self.create_composed_effect(composed_effects, operator, validation_result.metadata)
    }

    /// Select appropriate composition strategy
    fn select_composition_strategy(&self, effects: &[Effect], operator: &CompositionOperator) -> Result<&CompositionStrategy, EffectSystemError> {
        self.strategies.iter()
            .find(|strategy| strategy.applies_to(effects, operator))
            .ok_or_else(|| EffectSystemError::EffectValidationFailed {
                reason: format!("No composition strategy found for {:?} with {} effects", operator, effects.len()),
            })
    }

    /// Create the final composed effect
    fn create_composed_effect(&self, effects: Vec<Effect>, operator: CompositionOperator, metadata: CompositionMetadata) -> Result<Effect, EffectSystemError> {
        let composed_name = self.generate_composed_name(&effects, &operator);
        let composed_span = self.merge_spans(&effects);
        
        let mut composed_effect = Effect::new(composed_name, composed_span);
        
        // Add composition metadata
        composed_effect.metadata.ai_context = Some(format!(
            "Composed effect using {:?} operator with {} sub-effects: {}",
            operator,
            effects.len(),
            effects.iter().map(|e| e.definition.as_str()).collect::<Vec<_>>().join(", ")
        ));
        
        // Set security classification to highest level among sub-effects
        let highest_classification = effects.iter()
            .map(|e| &e.metadata.security_classification)
            .fold(SecurityClassification::Public, |acc, classification| {
                use SecurityClassification::*;
                match (acc, classification) {
                    (TopSecret, _) | (_, TopSecret) => TopSecret,
                    (Confidential, _) | (_, Confidential) => Confidential,
                    (Restricted, _) | (_, Restricted) => Restricted,
                    (Internal, _) | (_, Internal) => Internal,
                    (Public, Public) => Public,
                }
            });
        
        composed_effect.metadata.security_classification = highest_classification;
        
        // Mark as inferred composition
        composed_effect.metadata.inferred = true;
        composed_effect.metadata.confidence = metadata.confidence_score;
        composed_effect.metadata.inference_source = Some("EffectComposition".to_string());

        Ok(composed_effect)
    }

    /// Generate name for composed effect
    fn generate_composed_name(&self, effects: &[Effect], operator: &CompositionOperator) -> String {
        let operator_str = match operator {
            CompositionOperator::Parallel => "Parallel",
            CompositionOperator::Sequential => "Sequential",
            CompositionOperator::Conditional(_) => "Conditional",
        };
        
        let effect_names: Vec<String> = effects.iter().map(|e| e.definition.clone()).collect();
        format!("{}Composition[{}]", operator_str, effect_names.join(","))
    }

    /// Merge spans from multiple effects
    fn merge_spans(&self, effects: &[Effect]) -> Span {
        if effects.is_empty() {
            return Span::dummy();
        }
        
        // For now, use the first effect's span
        // In a full implementation, this would merge all spans
        effects[0].span
    }

    /// Initialize default composition strategies
    fn initialize_default_strategies(&mut self) {
        // Parallel composition strategy
        self.strategies.push(CompositionStrategy {
            name: "ParallelComposition".to_string(),
            description: "Compose effects that can execute in parallel".to_string(),
            operator: CompositionOperator::Parallel,
            applies_fn: Box::new(|effects, operator| {
                matches!(operator, CompositionOperator::Parallel) && effects.len() > 1
            }),
            apply_fn: Box::new(|effects, dependency_graph, _execution_order| {
                // For parallel composition, effects should have minimal dependencies
                let parallel_effects = effects.into_iter()
                    .filter(|effect| dependency_graph.get_dependencies(&effect.definition).len() <= 1)
                    .collect();
                Ok(parallel_effects)
            }),
        });

        // Sequential composition strategy  
        self.strategies.push(CompositionStrategy {
            name: "SequentialComposition".to_string(),
            description: "Compose effects that must execute in sequence".to_string(),
            operator: CompositionOperator::Sequential,
            applies_fn: Box::new(|effects, operator| {
                matches!(operator, CompositionOperator::Sequential) && effects.len() > 1
            }),
            apply_fn: Box::new(|effects, _dependency_graph, execution_order| {
                // For sequential composition, reorder effects based on dependencies
                let mut ordered_effects = Vec::new();
                for effect_name in execution_order {
                    if let Some(effect) = effects.iter().find(|e| e.definition == *effect_name) {
                        ordered_effects.push(effect.clone());
                    }
                }
                Ok(ordered_effects)
            }),
        });

        // Conditional composition strategy
        self.strategies.push(CompositionStrategy {
            name: "ConditionalComposition".to_string(),
            description: "Compose effects with conditional execution".to_string(),
            operator: CompositionOperator::Conditional("default".to_string()),
            applies_fn: Box::new(|effects, operator| {
                matches!(operator, CompositionOperator::Conditional(_)) && !effects.is_empty()
            }),
            apply_fn: Box::new(|effects, _dependency_graph, _execution_order| {
                // For conditional composition, keep effects as-is but add conditional metadata
                Ok(effects)
            }),
        });

        // Initialize optimization rules
        self.optimization_rules.push(OptimizationRule {
            name: "RemoveDuplicates".to_string(),
            description: "Remove duplicate effects in composition".to_string(),
            can_optimize_fn: Box::new(|effects| {
                let mut seen = HashSet::new();
                effects.iter().any(|effect| !seen.insert(&effect.definition))
            }),
            optimize_fn: Box::new(|effects| {
                let mut seen = HashSet::new();
                let deduplicated: Vec<Effect> = effects.into_iter()
                    .filter(|effect| seen.insert(effect.definition.clone()))
                    .collect();
                Ok(deduplicated)
            }),
        });

        self.optimization_rules.push(OptimizationRule {
            name: "MergeCompatibleEffects".to_string(),
            description: "Merge effects that can be combined for efficiency".to_string(),
            can_optimize_fn: Box::new(|effects| {
                // Check if we have multiple effects from the same category that can be merged
                effects.len() > 1 && effects.iter()
                    .any(|e1| effects.iter().any(|e2| 
                        e1.definition != e2.definition && 
                        e1.definition.split('.').next() == e2.definition.split('.').next()
                    ))
            }),
            optimize_fn: Box::new(|effects| {
                // Group effects by category and merge where possible
                let mut category_groups: HashMap<String, Vec<Effect>> = HashMap::new();
                
                for effect in effects {
                    let category = effect.definition.split('.').next().unwrap_or("Unknown").to_string();
                    category_groups.entry(category).or_default().push(effect);
                }

                let mut optimized_effects = Vec::new();
                for (category, mut category_effects) in category_groups {
                    if category_effects.len() > 1 && (category == "IO" || category == "Database") {
                        // Merge multiple I/O or Database effects into a batch operation
                        let merged_effect = Effect::new(
                            format!("{}Batch", category),
                            category_effects[0].span
                        );
                        optimized_effects.push(merged_effect);
                    } else {
                        optimized_effects.append(&mut category_effects);
                    }
                }

                Ok(optimized_effects)
            }),
        });
    }
}

impl Default for EffectComposition {
    fn default() -> Self {
        Self::new()
    }
}

/// Composition operators for combining effects
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionOperator {
    /// Parallel composition (|) - effects execute concurrently
    Parallel,
    /// Sequential composition (;) - effects execute in order
    Sequential,
    /// Conditional composition - effects execute based on conditions
    Conditional(String),
}

/// Composition strategy for specific scenarios
pub struct CompositionStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Composition operator this strategy handles
    pub operator: CompositionOperator,
    /// Function to check if strategy applies
    pub applies_fn: Box<dyn Fn(&[Effect], &CompositionOperator) -> bool + Send + Sync>,
    /// Function to apply the strategy
    pub apply_fn: Box<dyn Fn(Vec<Effect>, &DependencyGraph, &[String]) -> Result<Vec<Effect>, EffectSystemError> + Send + Sync>,
}

impl std::fmt::Debug for CompositionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositionStrategy")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("operator", &self.operator)
            .finish()
    }
}

impl CompositionStrategy {
    /// Check if this strategy applies to the given effects and operator
    pub fn applies_to(&self, effects: &[Effect], operator: &CompositionOperator) -> bool {
        (self.applies_fn)(effects, operator)
    }

    /// Apply this composition strategy
    pub fn apply_composition(&self, effects: Vec<Effect>, dependency_graph: &DependencyGraph, execution_order: &[String]) -> Result<Vec<Effect>, EffectSystemError> {
        (self.apply_fn)(effects, dependency_graph, execution_order)
    }
}

/// Optimization rule for effect compositions
pub struct OptimizationRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Function to check if optimization can be applied
    pub can_optimize_fn: Box<dyn Fn(&[Effect]) -> bool + Send + Sync>,
    /// Function to apply the optimization
    pub optimize_fn: Box<dyn Fn(Vec<Effect>) -> Result<Vec<Effect>, EffectSystemError> + Send + Sync>,
}

impl std::fmt::Debug for OptimizationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl OptimizationRule {
    /// Check if this rule can optimize the given effects
    pub fn can_optimize(&self, effects: &[Effect]) -> bool {
        (self.can_optimize_fn)(effects)
    }

    /// Apply this optimization rule
    pub fn optimize(&self, effects: Vec<Effect>) -> Result<Vec<Effect>, EffectSystemError> {
        (self.optimize_fn)(effects)
    }
}

/// Validates effect compositions for safety and correctness
#[derive(Debug)]
pub struct CompositionValidator {
    /// Validation rules
    validation_rules: Vec<ValidationRule>,
}

impl CompositionValidator {
    /// Create new composition validator
    pub fn new() -> Self {
        let mut validator = Self {
            validation_rules: Vec::new(),
        };
        validator.initialize_validation_rules();
        validator
    }

    /// Validate an effect composition
    pub fn validate_composition(&self, effects: &[Effect], operator: &CompositionOperator) -> Result<ValidationResult, EffectSystemError> {
        let mut validation_result = ValidationResult {
            is_valid: true,
            error_message: None,
            warnings: Vec::new(),
            metadata: CompositionMetadata {
                confidence_score: 1.0,
                complexity_score: self.calculate_complexity(effects),
                risk_assessment: self.assess_risk(effects),
                performance_impact: self.estimate_performance_impact(effects, operator),
            },
        };

        // Apply all validation rules
        for rule in &self.validation_rules {
            let rule_result = rule.validate(effects, operator)?;
            
            if !rule_result.is_valid {
                validation_result.is_valid = false;
                validation_result.error_message = Some(rule_result.message);
                break;
            }
            
            if !rule_result.warnings.is_empty() {
                validation_result.warnings.extend(rule_result.warnings);
            }
            
            // Reduce confidence if rule has concerns
            if rule_result.confidence_impact < 1.0 {
                validation_result.metadata.confidence_score *= rule_result.confidence_impact;
            }
        }

        Ok(validation_result)
    }

    /// Initialize validation rules
    fn initialize_validation_rules(&mut self) {
        // Rule: Check for circular dependencies
        self.validation_rules.push(ValidationRule {
            name: "CircularDependencyCheck".to_string(),
            description: "Ensure no circular dependencies exist in effect composition".to_string(),
            validate_fn: Box::new(|effects, _operator| {
                // Simple circular dependency check based on effect names
                let effect_names: HashSet<String> = effects.iter().map(|e| e.definition.clone()).collect();
                
                // For now, just check that we don't have obviously circular patterns
                let has_circular = effects.iter().any(|effect| {
                    effect.definition.contains("Circular") || 
                    effect.parameters.values().any(|param| {
                        // This is a simplified check - in practice would analyze parameter dependencies
                        effect_names.iter().any(|name| format!("{:?}", param).contains(name))
                    })
                });

                RuleValidationResult {
                    is_valid: !has_circular,
                    message: if has_circular { 
                        "Circular dependency detected in effect composition".to_string() 
                    } else { 
                        String::new() 
                    },
                    warnings: Vec::new(),
                    confidence_impact: if has_circular { 0.0 } else { 1.0 },
                }
            }),
        });

        // Rule: Check security classification compatibility
        self.validation_rules.push(ValidationRule {
            name: "SecurityClassificationCheck".to_string(),
            description: "Ensure security classifications are compatible".to_string(),
            validate_fn: Box::new(|effects, _operator| {
                let classifications: Vec<&SecurityClassification> = effects.iter()
                    .map(|e| &e.metadata.security_classification)
                    .collect();
                
                let has_incompatible = classifications.iter().any(|&c1| {
                    classifications.iter().any(|&c2| {
                        // Check for incompatible security levels
                        matches!((c1, c2), 
                            (SecurityClassification::Public, SecurityClassification::TopSecret) |
                            (SecurityClassification::TopSecret, SecurityClassification::Public)
                        )
                    })
                });

                let warnings = if classifications.iter().any(|&c| matches!(c, SecurityClassification::TopSecret)) {
                    vec!["Composition contains TopSecret effects - ensure proper clearance".to_string()]
                } else {
                    Vec::new()
                };

                RuleValidationResult {
                    is_valid: !has_incompatible,
                    message: if has_incompatible {
                        "Incompatible security classifications in effect composition".to_string()
                    } else {
                        String::new()
                    },
                    warnings,
                    confidence_impact: if has_incompatible { 0.0 } else { 0.9 },
                }
            }),
        });

        // Rule: Check resource constraints
        self.validation_rules.push(ValidationRule {
            name: "ResourceConstraintCheck".to_string(),
            description: "Ensure composed effects don't exceed resource limits".to_string(),
            validate_fn: Box::new(|effects, operator| {
                let estimated_resource_usage = effects.len() as f64;
                let resource_multiplier = match operator {
                    CompositionOperator::Parallel => 1.0, // Parallel might use more resources simultaneously
                    CompositionOperator::Sequential => 0.5, // Sequential uses resources over time
                    CompositionOperator::Conditional(_) => 0.7, // Conditional usage depends on conditions
                };
                
                let total_resource_estimate = estimated_resource_usage * resource_multiplier;
                let exceeds_limit = total_resource_estimate > 100.0; // Arbitrary limit for example
                
                let warnings = if total_resource_estimate > 50.0 {
                    vec![format!("High resource usage estimated: {:.1} units", total_resource_estimate)]
                } else {
                    Vec::new()
                };

                RuleValidationResult {
                    is_valid: !exceeds_limit,
                    message: if exceeds_limit {
                        format!("Resource usage estimate ({:.1}) exceeds limits", total_resource_estimate)
                    } else {
                        String::new()
                    },
                    warnings,
                    confidence_impact: if exceeds_limit { 0.0 } else { 0.95 },
                }
            }),
        });
    }

    /// Calculate composition complexity
    fn calculate_complexity(&self, effects: &[Effect]) -> f64 {
        // Simple complexity calculation based on number of effects and their types
        let base_complexity = effects.len() as f64;
        let type_complexity: f64 = effects.iter()
            .map(|effect| {
                // More complex effects get higher scores
                if effect.definition.contains("Database") || effect.definition.contains("Network") {
                    2.0
                } else if effect.definition.contains("AI") || effect.definition.contains("Unsafe") {
                    3.0
                } else {
                    1.0
                }
            })
            .sum();
        
        (base_complexity + type_complexity) / (effects.len() as f64 + 1.0)
    }

    /// Assess risk level of composition
    fn assess_risk(&self, effects: &[Effect]) -> RiskLevel {
        let high_risk_count = effects.iter()
            .filter(|effect| {
                effect.definition.contains("Unsafe") ||
                effect.metadata.security_classification == SecurityClassification::TopSecret
            })
            .count();
        
        let medium_risk_count = effects.iter()
            .filter(|effect| {
                effect.definition.contains("Network") ||
                effect.definition.contains("AI") ||
                effect.metadata.security_classification == SecurityClassification::Confidential
            })
            .count();

        if high_risk_count > 0 {
            RiskLevel::High
        } else if medium_risk_count > 2 {
            RiskLevel::Medium
        } else if medium_risk_count > 0 {
            RiskLevel::Low
        } else {
            RiskLevel::VeryLow
        }
    }

    /// Estimate performance impact
    fn estimate_performance_impact(&self, effects: &[Effect], operator: &CompositionOperator) -> PerformanceImpact {
        let base_impact = effects.len() as f64;
        let operator_multiplier = match operator {
            CompositionOperator::Parallel => 0.8, // Parallel can be more efficient
            CompositionOperator::Sequential => 1.0, // Sequential is baseline
            CompositionOperator::Conditional(_) => 0.6, // Conditional might skip some effects
        };
        
        let total_impact = base_impact * operator_multiplier;
        
        if total_impact > 10.0 {
            PerformanceImpact::High
        } else if total_impact > 5.0 {
            PerformanceImpact::Medium
        } else {
            PerformanceImpact::Low
        }
    }
}

/// Analyzes dependencies between effects
#[derive(Debug)]
pub struct DependencyAnalyzer {
    /// Dependency rules for effect types
    dependency_rules: HashMap<String, Vec<String>>,
}

impl DependencyAnalyzer {
    /// Create new dependency analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            dependency_rules: HashMap::new(),
        };
        analyzer.initialize_dependency_rules();
        analyzer
    }

    /// Analyze dependencies between effects
    pub fn analyze_dependencies(&self, effects: &[Effect]) -> Result<DependencyGraph, EffectSystemError> {
        let mut graph = DependencyGraph::new();
        
        // Add all effects as nodes
        for effect in effects {
            graph.add_effect(effect.definition.clone());
        }

        // Add dependencies based on rules and effect analysis
        for effect in effects {
            let dependencies = self.find_dependencies(effect, effects)?;
            for dependency in dependencies {
                graph.add_dependency(effect.definition.clone(), dependency)?;
            }
        }

        Ok(graph)
    }

    /// Find dependencies for a specific effect
    fn find_dependencies(&self, effect: &Effect, all_effects: &[Effect]) -> Result<Vec<String>, EffectSystemError> {
        let mut dependencies = Vec::new();

        // Check rule-based dependencies
        if let Some(rule_deps) = self.dependency_rules.get(&effect.definition) {
            for dep in rule_deps {
                if all_effects.iter().any(|e| e.definition == *dep) {
                    dependencies.push(dep.clone());
                }
            }
        }

        // Check parameter-based dependencies
        for (param_name, param_value) in &effect.parameters {
            // Look for references to other effects in parameters
            let param_str = format!("{:?}", param_value);
            for other_effect in all_effects {
                if other_effect.definition != effect.definition && param_str.contains(&other_effect.definition) {
                    dependencies.push(other_effect.definition.clone());
                }
            }
        }

        // Check security classification dependencies
        for other_effect in all_effects {
            if other_effect.definition != effect.definition {
                if self.has_security_dependency(&effect.metadata.security_classification, &other_effect.metadata.security_classification) {
                    dependencies.push(other_effect.definition.clone());
                }
            }
        }

        Ok(dependencies)
    }

    /// Check if one security classification depends on another
    fn has_security_dependency(&self, dependent: &SecurityClassification, dependency: &SecurityClassification) -> bool {
        // Higher security levels depend on lower ones being processed first
        matches!(
            (dependent, dependency),
            (SecurityClassification::Confidential, SecurityClassification::Public) |
            (SecurityClassification::TopSecret, SecurityClassification::Confidential) |
            (SecurityClassification::TopSecret, SecurityClassification::Public)
        )
    }

    /// Initialize dependency rules
    fn initialize_dependency_rules(&mut self) {
        // Database operations often depend on connection establishment
        self.dependency_rules.insert("Database.Query".to_string(), vec!["Database.Connect".to_string()]);
        self.dependency_rules.insert("Database.Transaction".to_string(), vec!["Database.Connect".to_string()]);
        
        // File operations might depend on directory creation
        self.dependency_rules.insert("FileSystem.Write".to_string(), vec!["FileSystem.CreateDirectory".to_string()]);
        
        // Network operations might depend on connection establishment
        self.dependency_rules.insert("Network.Send".to_string(), vec!["Network.Connect".to_string()]);
        
        // AI operations might depend on model loading
        self.dependency_rules.insert("AI.Inference".to_string(), vec!["AI.LoadModel".to_string()]);
        
        // Cryptographic operations might depend on key generation
        self.dependency_rules.insert("Cryptography.Encrypt".to_string(), vec!["Cryptography.GenerateKey".to_string()]);
        self.dependency_rules.insert("Cryptography.Sign".to_string(), vec!["Cryptography.GenerateKey".to_string()]);
    }
}

/// Detects conflicts between effects
#[derive(Debug)]
pub struct ConflictDetector {
    /// Conflict rules
    conflict_rules: Vec<ConflictRule>,
}

impl ConflictDetector {
    /// Create new conflict detector
    pub fn new() -> Self {
        let mut detector = Self {
            conflict_rules: Vec::new(),
        };
        detector.initialize_conflict_rules();
        detector
    }

    /// Detect conflicts in effect composition
    pub fn detect_conflicts(&self, effects: &[Effect]) -> Result<ConflictAnalysis, EffectSystemError> {
        let mut conflicts = Vec::new();

        // Apply all conflict rules
        for rule in &self.conflict_rules {
            let rule_conflicts = rule.detect_conflicts(effects)?;
            conflicts.extend(rule_conflicts);
        }

        // Analyze conflict severity
        let max_severity = conflicts.iter()
            .map(|c| &c.severity)
            .max()
            .cloned()
            .unwrap_or(ConflictSeverity::None);

        let resolution_suggestions = self.generate_resolution_suggestions(&conflicts);
        
        Ok(ConflictAnalysis {
            conflicts,
            overall_severity: max_severity,
            resolution_suggestions,
        })
    }

    /// Generate suggestions for resolving conflicts
    fn generate_resolution_suggestions(&self, conflicts: &[EffectConflict]) -> Vec<String> {
        let mut suggestions = Vec::new();

        for conflict in conflicts {
            match conflict.conflict_type {
                ConflictType::ResourceContention => {
                    suggestions.push("Consider using resource pooling or queuing".to_string());
                },
                ConflictType::SecurityViolation => {
                    suggestions.push("Review security classifications and access controls".to_string());
                },
                ConflictType::DataRace => {
                    suggestions.push("Add synchronization or use sequential composition".to_string());
                },
                ConflictType::MutualExclusion => {
                    suggestions.push("Use conditional composition or mutex-like coordination".to_string());
                },
            }
        }

        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    /// Initialize conflict detection rules
    fn initialize_conflict_rules(&mut self) {
        // Rule: Detect resource contention
        self.conflict_rules.push(ConflictRule {
            name: "ResourceContentionRule".to_string(),
            description: "Detect when multiple effects compete for the same resource".to_string(),
            detect_fn: Box::new(|effects| {
                let mut conflicts = Vec::new();
                
                // Check for multiple file write operations to the same resource
                let file_writes: Vec<_> = effects.iter()
                    .filter(|e| e.definition.contains("FileSystem.Write"))
                    .collect();
                
                if file_writes.len() > 1 {
                    conflicts.push(EffectConflict {
                        conflict_type: ConflictType::ResourceContention,
                        severity: ConflictSeverity::Medium,
                        description: "Multiple file write operations detected - potential resource contention".to_string(),
                        affected_effects: file_writes.iter().map(|e| e.definition.clone()).collect(),
                    });
                }

                // Check for multiple database transactions
                let db_transactions: Vec<_> = effects.iter()
                    .filter(|e| e.definition.contains("Database.Transaction"))
                    .collect();
                
                if db_transactions.len() > 1 {
                    conflicts.push(EffectConflict {
                        conflict_type: ConflictType::ResourceContention,
                        severity: ConflictSeverity::High,
                        description: "Multiple database transactions detected - potential deadlock risk".to_string(),
                        affected_effects: db_transactions.iter().map(|e| e.definition.clone()).collect(),
                    });
                }

                Ok(conflicts)
            }),
        });

        // Rule: Detect security violations
        self.conflict_rules.push(ConflictRule {
            name: "SecurityViolationRule".to_string(),
            description: "Detect security classification conflicts".to_string(),
            detect_fn: Box::new(|effects| {
                let mut conflicts = Vec::new();
                
                for effect1 in effects {
                    for effect2 in effects {
                        if effect1.definition != effect2.definition {
                            let violation = matches!(
                                (&effect1.metadata.security_classification, &effect2.metadata.security_classification),
                                (SecurityClassification::Public, SecurityClassification::TopSecret) |
                                (SecurityClassification::TopSecret, SecurityClassification::Public)
                            );
                            
                            if violation {
                                conflicts.push(EffectConflict {
                                    conflict_type: ConflictType::SecurityViolation,
                                    severity: ConflictSeverity::Critical,
                                    description: "Security classification conflict between Public and TopSecret effects".to_string(),
                                    affected_effects: vec![effect1.definition.clone(), effect2.definition.clone()],
                                });
                            }
                        }
                    }
                }

                Ok(conflicts)
            }),
        });

        // Rule: Detect data races
        self.conflict_rules.push(ConflictRule {
            name: "DataRaceRule".to_string(),
            description: "Detect potential data races in parallel composition".to_string(),
            detect_fn: Box::new(|effects| {
                let mut conflicts = Vec::new();
                
                // Look for read-write conflicts
                let readers: Vec<_> = effects.iter()
                    .filter(|e| e.definition.contains("Read") || e.definition.contains("Query"))
                    .collect();
                let writers: Vec<_> = effects.iter()
                    .filter(|e| e.definition.contains("Write") || e.definition.contains("Insert") || e.definition.contains("Update"))
                    .collect();
                
                if !readers.is_empty() && !writers.is_empty() {
                    conflicts.push(EffectConflict {
                        conflict_type: ConflictType::DataRace,
                        severity: ConflictSeverity::Medium,
                        description: "Potential data race between read and write operations".to_string(),
                        affected_effects: readers.iter().chain(writers.iter())
                            .map(|e| e.definition.clone())
                            .collect(),
                    });
                }

                Ok(conflicts)
            }),
        });

        // Rule: Detect mutual exclusion violations
        self.conflict_rules.push(ConflictRule {
            name: "MutualExclusionRule".to_string(),
            description: "Detect effects that cannot execute simultaneously".to_string(),
            detect_fn: Box::new(|effects| {
                let mut conflicts = Vec::new();
                
                // Check for mutually exclusive operations
                let unsafe_effects: Vec<_> = effects.iter()
                    .filter(|e| e.definition.contains("Unsafe"))
                    .collect();
                
                if unsafe_effects.len() > 1 {
                    conflicts.push(EffectConflict {
                        conflict_type: ConflictType::MutualExclusion,
                        severity: ConflictSeverity::High,
                        description: "Multiple unsafe operations cannot execute simultaneously".to_string(),
                        affected_effects: unsafe_effects.iter().map(|e| e.definition.clone()).collect(),
                    });
                }

                Ok(conflicts)
            }),
        });
    }
}

/// Dependency graph for effects
#[derive(Debug)]
pub struct DependencyGraph {
    /// Effect nodes
    effects: HashSet<String>,
    /// Dependencies (effect -> dependencies)
    dependencies: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    /// Create new dependency graph
    pub fn new() -> Self {
        Self {
            effects: HashSet::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Add an effect to the graph
    pub fn add_effect(&mut self, effect: String) {
        self.effects.insert(effect);
    }

    /// Add a dependency relationship
    pub fn add_dependency(&mut self, effect: String, dependency: String) -> Result<(), EffectSystemError> {
        if !self.effects.contains(&effect) || !self.effects.contains(&dependency) {
            return Err(EffectSystemError::EffectValidationFailed {
                reason: "Cannot add dependency for non-existent effects".to_string(),
            });
        }
        
        self.dependencies.entry(effect).or_default().insert(dependency);
        Ok(())
    }

    /// Get dependencies for an effect
    pub fn get_dependencies(&self, effect: &str) -> Vec<String> {
        self.dependencies.get(effect)
            .map(|deps| deps.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Perform topological sort to get execution order
    pub fn topological_sort(&self) -> Result<Vec<String>, EffectSystemError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        for effect in &self.effects {
            if !visited.contains(effect) {
                self.visit(effect, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        result.reverse();
        Ok(result)
    }

    /// Visit node in topological sort (DFS)
    fn visit(
        &self,
        effect: &str,
        visited: &mut HashSet<String>,
        temp_visited: &mut HashSet<String>,
        result: &mut Vec<String>,
    ) -> Result<(), EffectSystemError> {
        if temp_visited.contains(effect) {
            return Err(EffectSystemError::EffectValidationFailed {
                reason: format!("Circular dependency detected involving effect: {}", effect),
            });
        }

        if visited.contains(effect) {
            return Ok(());
        }

        temp_visited.insert(effect.to_string());

        if let Some(dependencies) = self.dependencies.get(effect) {
            for dependency in dependencies {
                self.visit(dependency, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(effect);
        visited.insert(effect.to_string());
        result.push(effect.to_string());

        Ok(())
    }
}

// Supporting types and structures...

/// Validation rule for effect compositions
struct ValidationRule {
    /// Rule name
    name: String,
    /// Rule description
    description: String,
    /// Validation function
    validate_fn: Box<dyn Fn(&[Effect], &CompositionOperator) -> RuleValidationResult + Send + Sync>,
}

impl std::fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl ValidationRule {
    /// Validate effects using this rule
    pub fn validate(&self, effects: &[Effect], operator: &CompositionOperator) -> Result<RuleValidationResult, EffectSystemError> {
        Ok((self.validate_fn)(effects, operator))
    }
}

/// Result of applying a validation rule
struct RuleValidationResult {
    /// Whether the rule passed
    is_valid: bool,
    /// Error message if rule failed
    message: String,
    /// Warnings from the rule
    warnings: Vec<String>,
    /// Impact on confidence (0.0 to 1.0)
    confidence_impact: f64,
}

/// Overall validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether composition is valid
    pub is_valid: bool,
    /// Error message if invalid
    pub error_message: Option<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Composition metadata
    pub metadata: CompositionMetadata,
}

/// Metadata about a composition
#[derive(Debug)]
pub struct CompositionMetadata {
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,
    /// Complexity score
    pub complexity_score: f64,
    /// Risk assessment
    pub risk_assessment: RiskLevel,
    /// Performance impact estimate
    pub performance_impact: PerformanceImpact,
}

/// Risk levels for compositions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    Critical,
}

/// Performance impact levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceImpact {
    Low,
    Medium,
    High,
}

/// Conflict detection rule
struct ConflictRule {
    /// Rule name
    name: String,
    /// Rule description
    description: String,
    /// Function to detect conflicts
    detect_fn: Box<dyn Fn(&[Effect]) -> Result<Vec<EffectConflict>, EffectSystemError> + Send + Sync>,
}

impl std::fmt::Debug for ConflictRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConflictRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl ConflictRule {
    /// Detect conflicts using this rule
    fn detect_conflicts(&self, effects: &[Effect]) -> Result<Vec<EffectConflict>, EffectSystemError> {
        (self.detect_fn)(effects)
    }
}

/// Analysis of conflicts in effect composition
#[derive(Debug)]
pub struct ConflictAnalysis {
    /// Detected conflicts
    pub conflicts: Vec<EffectConflict>,
    /// Overall severity level
    pub overall_severity: ConflictSeverity,
    /// Suggestions for resolving conflicts
    pub resolution_suggestions: Vec<String>,
}

/// Individual effect conflict
#[derive(Debug)]
pub struct EffectConflict {
    /// Type of conflict
    pub conflict_type: ConflictType,
    /// Severity of the conflict
    pub severity: ConflictSeverity,
    /// Description of the conflict
    pub description: String,
    /// Effects involved in the conflict
    pub affected_effects: Vec<String>,
}

/// Types of effect conflicts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Resource contention between effects
    ResourceContention,
    /// Security policy violation
    SecurityViolation,
    /// Potential data race
    DataRace,
    /// Mutual exclusion violation
    MutualExclusion,
}

/// Conflict severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Legacy composition rule for backward compatibility
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
    pub conditions: Vec<AstNode<Expr>>,
    /// AI-readable explanation of the composition
    pub ai_explanation: Option<String>,
}

impl EffectCompositionRule {
    /// Create a new composition rule
    pub fn new(
        name: String,
        description: String,
        input_effects: Vec<String>,
        output_effect: String,
    ) -> Self {
        Self {
            name,
            description,
            input_effects,
            output_effect,
            conditions: Vec::new(),
            ai_explanation: None,
        }
    }

    /// Add a condition to this rule
    pub fn with_condition(mut self, condition: AstNode<Expr>) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Add AI explanation
    pub fn with_ai_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.ai_explanation = Some(explanation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_system_creation() {
        let composition = EffectComposition::new();
        assert!(!composition.strategies.is_empty());
        assert!(!composition.optimization_rules.is_empty());
    }

    #[test]
    fn test_parallel_composition() {
        let composition = EffectComposition::new();
        
        let effects = vec![
            Effect::new("IO.Read".to_string(), Span::dummy()),
            Effect::new("Network.Send".to_string(), Span::dummy()),
        ];
        
        let result = composition.compose(effects, CompositionOperator::Parallel);
        assert!(result.is_ok());
        
        let composed = result.unwrap();
        assert!(composed.definition.contains("Parallel"));
    }

    #[test]
    fn test_dependency_analysis() {
        let analyzer = DependencyAnalyzer::new();
        
        let effects = vec![
            Effect::new("Database.Connect".to_string(), Span::dummy()),
            Effect::new("Database.Query".to_string(), Span::dummy()),
        ];
        
        let graph = analyzer.analyze_dependencies(&effects).unwrap();
        let dependencies = graph.get_dependencies("Database.Query");
        assert!(dependencies.contains(&"Database.Connect".to_string()));
    }

    #[test]
    fn test_conflict_detection() {
        let detector = ConflictDetector::new();
        
        let effects = vec![
            Effect::new("FileSystem.Write".to_string(), Span::dummy()),
            Effect::new("FileSystem.Write".to_string(), Span::dummy()),
        ];
        
        let analysis = detector.detect_conflicts(&effects).unwrap();
        assert!(!analysis.conflicts.is_empty());
        assert_eq!(analysis.conflicts[0].conflict_type, ConflictType::ResourceContention);
    }

    #[test]
    fn test_validation() {
        let validator = CompositionValidator::new();
        
        let effects = vec![
            Effect::new("IO.Read".to_string(), Span::dummy()),
            Effect::new("Network.Send".to_string(), Span::dummy()),
        ];
        
        let result = validator.validate_composition(&effects, &CompositionOperator::Sequential).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_remove_duplicates_optimization() {
        let composition = EffectComposition::new();
        
        let effects = vec![
            Effect::new("IO.Read".to_string(), Span::dummy()),
            Effect::new("IO.Read".to_string(), Span::dummy()),
            Effect::new("Network.Send".to_string(), Span::dummy()),
        ];
        
        let result = composition.compose(effects, CompositionOperator::Sequential);
        assert!(result.is_ok());
        
        // The composed effect should handle duplicates
        let composed = result.unwrap();
        assert!(composed.metadata.ai_context.is_some());
    }
} 