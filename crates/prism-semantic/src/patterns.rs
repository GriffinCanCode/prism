//! Semantic Pattern Recognition
//!
//! This module embodies the single concept of "Semantic Pattern Recognition".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: recognizing semantic patterns in code.

use crate::{SemanticResult, SemanticConfig};
use crate::analyzer::AnalysisResult;
use prism_ast::Program;
use prism_common::{span::Span, symbol::Symbol};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Pattern recognizer for semantic analysis
#[derive(Debug)]
pub struct PatternRecognizer {
    /// Configuration
    config: PatternConfig,
    /// Pattern database
    pattern_database: PatternDatabase,
    /// Recognition statistics
    stats: RecognitionStats,
}

/// Configuration for pattern recognition
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Enable pattern recognition
    pub enable_recognition: bool,
    /// Minimum confidence threshold for pattern detection
    pub min_confidence: f64,
    /// Maximum patterns to detect per analysis
    pub max_patterns: usize,
    /// Enable AI-assisted pattern recognition
    pub enable_ai_assistance: bool,
}

/// Pattern database containing known semantic patterns
#[derive(Debug)]
pub struct PatternDatabase {
    /// Known patterns indexed by type
    patterns: HashMap<PatternType, Vec<PatternDefinition>>,
    /// Pattern matching rules
    matching_rules: Vec<MatchingRule>,
}

/// Pattern definition with matching criteria
#[derive(Debug, Clone)]
pub struct PatternDefinition {
    /// Pattern name
    pub name: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Matching criteria
    pub criteria: Vec<MatchingCriterion>,
    /// Expected confidence range
    pub confidence_range: (f64, f64),
    /// AI description
    pub ai_description: String,
}

/// Matching criterion for pattern detection
#[derive(Debug, Clone)]
pub struct MatchingCriterion {
    /// Criterion type
    pub criterion_type: CriterionType,
    /// Expected value or pattern
    pub expected_value: String,
    /// Matching weight (0.0 to 1.0)
    pub weight: f64,
}

/// Types of matching criteria
#[derive(Debug, Clone)]
pub enum CriterionType {
    /// Symbol name pattern
    SymbolName,
    /// Type pattern
    TypePattern,
    /// Function signature pattern
    FunctionSignature,
    /// Module organization pattern
    ModuleStructure,
    /// Business context pattern
    BusinessContext,
    /// AI hint pattern
    AIHint,
}

/// Matching rule for pattern detection
#[derive(Debug, Clone)]
pub struct MatchingRule {
    /// Rule name
    pub name: String,
    /// Pattern type this rule applies to
    pub applies_to: PatternType,
    /// Matching logic
    pub logic: MatchingLogic,
}

/// Matching logic types
#[derive(Debug, Clone)]
pub enum MatchingLogic {
    /// All criteria must match
    AllMatch,
    /// Any criteria can match
    AnyMatch,
    /// Weighted scoring
    WeightedScore { threshold: f64 },
    /// Custom logic
    Custom { expression: String },
}

/// Recognition statistics
#[derive(Debug, Default)]
pub struct RecognitionStats {
    /// Total patterns checked
    pub patterns_checked: usize,
    /// Patterns detected
    pub patterns_detected: usize,
    /// High confidence patterns
    pub high_confidence_patterns: usize,
    /// Recognition time in microseconds
    pub recognition_time_us: u64,
}

/// Semantic pattern detected in code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Location where pattern was detected
    pub location: Span,
    /// AI hints related to this pattern
    pub ai_hints: Vec<String>,
    /// Evidence supporting this pattern
    pub evidence: Vec<PatternEvidence>,
    /// Suggested improvements
    pub suggestions: Vec<String>,
}

/// Evidence supporting a pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEvidence {
    /// Evidence type
    pub evidence_type: String,
    /// Evidence description
    pub description: String,
    /// Confidence contribution
    pub confidence_contribution: f64,
    /// Source location
    pub location: Option<Span>,
}

/// Types of semantic patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Module organization pattern
    ModuleOrganization,
    /// Function naming pattern
    FunctionNaming,
    /// Type definition pattern
    TypeDefinition,
    /// Effect declaration pattern
    EffectDeclaration,
    /// Capability usage pattern
    CapabilityUsage,
    /// Business logic pattern
    BusinessLogic,
    /// Error handling pattern
    ErrorHandling,
    /// Configuration pattern
    Configuration,
    /// Data access pattern
    DataAccess,
    /// Validation pattern
    ValidationPattern,
    /// Factory pattern
    FactoryPattern,
    /// Builder pattern
    BuilderPattern,
    /// Observer pattern
    ObserverPattern,
}

impl PatternRecognizer {
    /// Create a new pattern recognizer
    pub fn new(config: &SemanticConfig) -> SemanticResult<Self> {
        let pattern_config = PatternConfig {
            enable_recognition: config.enable_pattern_recognition,
            min_confidence: 0.7,
            max_patterns: 50,
            enable_ai_assistance: config.enable_ai_metadata,
        };

        let mut recognizer = Self {
            config: pattern_config,
            pattern_database: PatternDatabase::new(),
            stats: RecognitionStats::default(),
        };

        recognizer.initialize_patterns()?;
        Ok(recognizer)
    }

    /// Initialize built-in patterns
    fn initialize_patterns(&mut self) -> SemanticResult<()> {
        // Function naming patterns
        self.pattern_database.add_pattern(PatternDefinition {
            name: "CRUD Operations".to_string(),
            pattern_type: PatternType::FunctionNaming,
            criteria: vec![
                MatchingCriterion {
                    criterion_type: CriterionType::SymbolName,
                    expected_value: r"^(create|read|update|delete|get|set|add|remove)".to_string(),
                    weight: 0.8,
                },
            ],
            confidence_range: (0.7, 0.95),
            ai_description: "CRUD operation pattern detected in function naming".to_string(),
        });

        // Business logic patterns
        self.pattern_database.add_pattern(PatternDefinition {
            name: "Business Context Usage".to_string(),
            pattern_type: PatternType::BusinessLogic,
            criteria: vec![
                MatchingCriterion {
                    criterion_type: CriterionType::BusinessContext,
                    expected_value: ".*".to_string(), // Any business context
                    weight: 1.0,
                },
            ],
            confidence_range: (0.8, 1.0),
            ai_description: "Function or type with explicit business context".to_string(),
        });

        // Validation patterns
        self.pattern_database.add_pattern(PatternDefinition {
            name: "Validation Function".to_string(),
            pattern_type: PatternType::ValidationPattern,
            criteria: vec![
                MatchingCriterion {
                    criterion_type: CriterionType::SymbolName,
                    expected_value: r".*validate.*|.*check.*|.*verify.*".to_string(),
                    weight: 0.9,
                },
                MatchingCriterion {
                    criterion_type: CriterionType::FunctionSignature,
                    expected_value: r".*bool.*|.*result.*|.*option.*".to_string(),
                    weight: 0.7,
                },
            ],
            confidence_range: (0.75, 0.95),
            ai_description: "Function that performs validation operations".to_string(),
        });

        // Module organization patterns
        self.pattern_database.add_pattern(PatternDefinition {
            name: "Domain Module".to_string(),
            pattern_type: PatternType::ModuleOrganization,
            criteria: vec![
                MatchingCriterion {
                    criterion_type: CriterionType::ModuleStructure,
                    expected_value: "domain_focused".to_string(),
                    weight: 0.8,
                },
            ],
            confidence_range: (0.6, 0.9),
            ai_description: "Module organized around domain concepts".to_string(),
        });

        Ok(())
    }

    /// Recognize patterns in a program
    pub fn recognize_patterns(&mut self, program: &Program, analysis: &AnalysisResult) -> SemanticResult<Vec<SemanticPattern>> {
        if !self.config.enable_recognition {
            return Ok(Vec::new());
        }

        let start_time = std::time::Instant::now();
        let mut detected_patterns = Vec::new();

        // Reset stats
        self.stats = RecognitionStats::default();

        // Analyze symbols for patterns
        for (symbol, symbol_info) in &analysis.symbols {
            let symbol_patterns = self.analyze_symbol_patterns(symbol, symbol_info)?;
            detected_patterns.extend(symbol_patterns);
        }

        // Analyze types for patterns
        for (type_id, type_info) in &analysis.types {
            let type_patterns = self.analyze_type_patterns(type_id, type_info)?;
            detected_patterns.extend(type_patterns);
        }

        // Analyze overall module structure
        let module_patterns = self.analyze_module_patterns(program, analysis)?;
        detected_patterns.extend(module_patterns);

        // Filter by confidence and limit results
        detected_patterns.retain(|p| p.confidence >= self.config.min_confidence);
        detected_patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        detected_patterns.truncate(self.config.max_patterns);

        // Update stats
        self.stats.recognition_time_us = start_time.elapsed().as_micros() as u64;
        self.stats.patterns_detected = detected_patterns.len();
        self.stats.high_confidence_patterns = detected_patterns.iter()
            .filter(|p| p.confidence >= 0.9)
            .count();

        Ok(detected_patterns)
    }

    /// Analyze patterns in a symbol
    fn analyze_symbol_patterns(
        &mut self, 
        symbol: &Symbol, 
        symbol_info: &crate::analyzer::SymbolInfo
    ) -> SemanticResult<Vec<SemanticPattern>> {
        let mut patterns = Vec::new();
        self.stats.patterns_checked += 1;

        // Check function naming patterns
        if matches!(symbol_info.symbol_type, crate::analyzer::SymbolType::Function) {
            if let Some(pattern) = self.check_function_naming_pattern(&symbol_info.name, symbol_info.location)? {
                patterns.push(pattern);
            }
        }

        // Check business logic patterns
        if symbol_info.business_context.is_some() {
            patterns.push(SemanticPattern {
                pattern_type: PatternType::BusinessLogic,
                description: format!("Symbol '{}' has business context", symbol_info.name),
                confidence: 0.85,
                location: symbol_info.location,
                ai_hints: vec![
                    "This symbol is associated with business logic".to_string(),
                    "Consider documenting business rules".to_string(),
                ],
                evidence: vec![
                    PatternEvidence {
                        evidence_type: "business_context".to_string(),
                        description: format!("Business context: {:?}", symbol_info.business_context),
                        confidence_contribution: 0.8,
                        location: Some(symbol_info.location),
                    }
                ],
                suggestions: vec![
                    "Ensure business rules are properly documented".to_string(),
                ],
            });
        }

        // Check validation patterns
        if symbol_info.name.contains("validate") || symbol_info.name.contains("check") || symbol_info.name.contains("verify") {
            patterns.push(SemanticPattern {
                pattern_type: PatternType::ValidationPattern,
                description: format!("Function '{}' appears to perform validation", symbol_info.name),
                confidence: 0.8,
                location: symbol_info.location,
                ai_hints: vec![
                    "This function performs validation".to_string(),
                    "Consider adding proper error handling".to_string(),
                ],
                evidence: vec![
                    PatternEvidence {
                        evidence_type: "naming_pattern".to_string(),
                        description: "Function name suggests validation operation".to_string(),
                        confidence_contribution: 0.7,
                        location: Some(symbol_info.location),
                    }
                ],
                suggestions: vec![
                    "Ensure validation functions return appropriate error types".to_string(),
                    "Consider using Result<T, E> for validation functions".to_string(),
                ],
            });
        }

        Ok(patterns)
    }

    /// Analyze patterns in a type
    fn analyze_type_patterns(
        &mut self,
        _type_id: &prism_common::NodeId,
        type_info: &crate::analyzer::TypeInfo
    ) -> SemanticResult<Vec<SemanticPattern>> {
        let mut patterns = Vec::new();
        self.stats.patterns_checked += 1;

        // Check for configuration patterns
        if type_info.domain.as_ref().map_or(false, |d| d.contains("config")) {
            patterns.push(SemanticPattern {
                pattern_type: PatternType::Configuration,
                description: "Type appears to be configuration-related".to_string(),
                confidence: 0.75,
                location: type_info.location,
                ai_hints: vec![
                    "This type is used for configuration".to_string(),
                    "Consider validation for configuration values".to_string(),
                ],
                evidence: vec![
                    PatternEvidence {
                        evidence_type: "domain_context".to_string(),
                        description: format!("Domain: {:?}", type_info.domain),
                        confidence_contribution: 0.7,
                        location: Some(type_info.location),
                    }
                ],
                suggestions: vec![
                    "Add validation for configuration parameters".to_string(),
                    "Consider using strong types for configuration values".to_string(),
                ],
            });
        }

        Ok(patterns)
    }

    /// Analyze module-level patterns
    fn analyze_module_patterns(
        &mut self,
        _program: &Program,
        analysis: &AnalysisResult
    ) -> SemanticResult<Vec<SemanticPattern>> {
        let mut patterns = Vec::new();

        // Check for cohesive module organization
        let symbol_count = analysis.symbols.len();
        let business_context_count = analysis.symbols.values()
            .filter(|s| s.business_context.is_some())
            .count();

        if symbol_count > 0 {
            let business_context_ratio = business_context_count as f64 / symbol_count as f64;
            
            if business_context_ratio > 0.7 {
                patterns.push(SemanticPattern {
                    pattern_type: PatternType::ModuleOrganization,
                    description: "Module shows strong business context organization".to_string(),
                    confidence: 0.8 + (business_context_ratio - 0.7) * 0.2,
                    location: Span::new(
                        prism_common::span::Position::new(1, 1, 0),
                        prism_common::span::Position::new(1, 1, 0),
                        prism_common::SourceId::new(0)
                    ),
                    ai_hints: vec![
                        "Module demonstrates good domain organization".to_string(),
                        "Business context is well-represented".to_string(),
                    ],
                    evidence: vec![
                        PatternEvidence {
                            evidence_type: "business_context_ratio".to_string(),
                            description: format!("{}% of symbols have business context", (business_context_ratio * 100.0) as u32),
                            confidence_contribution: business_context_ratio,
                            location: None,
                        }
                    ],
                    suggestions: vec![
                        "Continue maintaining strong business context".to_string(),
                    ],
                });
            }
        }

        Ok(patterns)
    }

    /// Check function naming patterns
    fn check_function_naming_pattern(
        &self,
        function_name: &str,
        location: Span
    ) -> SemanticResult<Option<SemanticPattern>> {
        // CRUD pattern detection
        let crud_regex = regex::Regex::new(r"^(create|read|update|delete|get|set|add|remove)").unwrap();
        if crud_regex.is_match(function_name) {
            return Ok(Some(SemanticPattern {
                pattern_type: PatternType::FunctionNaming,
                description: format!("Function '{}' follows CRUD naming pattern", function_name),
                confidence: 0.9,
                location,
                ai_hints: vec![
                    "CRUD operation detected".to_string(),
                    "Consider consistent error handling across CRUD operations".to_string(),
                ],
                evidence: vec![
                    PatternEvidence {
                        evidence_type: "naming_convention".to_string(),
                        description: "Function name matches CRUD pattern".to_string(),
                        confidence_contribution: 0.8,
                        location: Some(location),
                    }
                ],
                suggestions: vec![
                    "Ensure CRUD operations have consistent return types".to_string(),
                    "Consider using Result<T, E> for error handling".to_string(),
                ],
            }));
        }

        Ok(None)
    }

    /// Get recognition statistics
    pub fn get_statistics(&self) -> &RecognitionStats {
        &self.stats
    }
}

impl PatternDatabase {
    /// Create a new pattern database
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            matching_rules: Vec::new(),
        }
    }

    /// Add a pattern to the database
    fn add_pattern(&mut self, pattern: PatternDefinition) {
        self.patterns
            .entry(pattern.pattern_type.clone())
            .or_insert_with(Vec::new)
            .push(pattern);
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enable_recognition: true,
            min_confidence: 0.7,
            max_patterns: 50,
            enable_ai_assistance: true,
        }
    }
} 