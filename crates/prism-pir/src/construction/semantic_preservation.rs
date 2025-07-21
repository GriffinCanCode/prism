//! Semantic Preservation - AST to PIR Semantic Fidelity
//!
//! This module implements semantic preservation validation during AST to PIR transformation,
//! ensuring that no semantic information is lost during the conversion process.
//!
//! **Conceptual Responsibility**: Semantic preservation validation
//! **What it does**: Validates semantic fidelity, tracks information preservation, reports semantic losses
//! **What it doesn't do**: AST parsing, PIR construction, semantic analysis (validates existing transformations)

use crate::{PIRResult, PIRError};
use crate::semantic::{PrismIR, PIRModule, PIRSemanticType, PIRFunction};
use prism_ast::{Program, AstNode, Item, ModuleDecl, FunctionDecl, TypeDecl};
use prism_common::span::Span;
use std::collections::{HashMap, HashSet};

/// Semantic preservation validator for AST to PIR transformation
pub struct SemanticPreservationValidator {
    /// Configuration for preservation validation
    config: PreservationConfig,
    /// Preservation metrics collector
    metrics: PreservationMetrics,
}

/// Configuration for semantic preservation validation
#[derive(Debug, Clone)]
pub struct PreservationConfig {
    /// Minimum preservation score required (0.0 to 1.0)
    pub min_preservation_score: f64,
    /// Enable strict type preservation checking
    pub strict_type_preservation: bool,
    /// Enable business context preservation checking
    pub check_business_context: bool,
    /// Enable AI metadata preservation checking
    pub check_ai_metadata: bool,
    /// Enable effect preservation checking
    pub check_effect_preservation: bool,
    /// Report warnings for minor preservation issues
    pub report_warnings: bool,
}

/// Preservation metrics and statistics
#[derive(Debug, Default)]
pub struct PreservationMetrics {
    /// Total items processed
    pub total_items: usize,
    /// Items with full preservation
    pub fully_preserved: usize,
    /// Items with partial preservation
    pub partially_preserved: usize,
    /// Items with preservation issues
    pub preservation_issues: usize,
    /// Overall preservation score (0.0 to 1.0)
    pub overall_score: f64,
    /// Detailed preservation scores by category
    pub category_scores: HashMap<PreservationCategory, f64>,
}

/// Categories of semantic preservation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PreservationCategory {
    /// Type information preservation
    TypeInformation,
    /// Function signature preservation
    FunctionSignatures,
    /// Business context preservation
    BusinessContext,
    /// AI metadata preservation
    AIMetadata,
    /// Effect system preservation
    EffectSystem,
    /// Module structure preservation
    ModuleStructure,
}

/// Result of semantic preservation validation
#[derive(Debug, Clone)]
pub struct PreservationResult {
    /// Overall validation success
    pub success: bool,
    /// Preservation score (0.0 to 1.0)
    pub score: f64,
    /// Detailed findings
    pub findings: Vec<PreservationFinding>,
    /// Preservation metrics
    pub metrics: PreservationMetrics,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Individual preservation finding
#[derive(Debug, Clone)]
pub struct PreservationFinding {
    /// Finding category
    pub category: PreservationCategory,
    /// Finding severity
    pub severity: PreservationSeverity,
    /// Finding description
    pub description: String,
    /// Source location (if available)
    pub location: Option<Span>,
    /// PIR location (if available)
    pub pir_location: Option<String>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Severity levels for preservation findings
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PreservationSeverity {
    /// Information only
    Info,
    /// Warning about potential issues
    Warning,
    /// Error that affects preservation
    Error,
    /// Critical error that breaks preservation
    Critical,
}

/// Context for preservation validation
#[derive(Debug)]
pub struct PreservationContext {
    /// AST items by name for lookup
    ast_items: HashMap<String, ASTItemRef>,
    /// PIR items by name for lookup
    pir_items: HashMap<String, PIRItemRef>,
    /// Type mappings for cross-reference
    type_mappings: HashMap<String, String>,
    /// Function mappings for cross-reference
    function_mappings: HashMap<String, String>,
}

/// Reference to AST item for preservation checking
#[derive(Debug, Clone)]
pub enum ASTItemRef {
    /// Module reference
    Module(ModuleDecl),
    /// Function reference
    Function(FunctionDecl),
    /// Type reference
    Type(TypeDecl),
}

/// Reference to PIR item for preservation checking
#[derive(Debug, Clone)]
pub enum PIRItemRef {
    /// Module reference
    Module(PIRModule),
    /// Function reference
    Function(PIRFunction),
    /// Type reference
    Type(PIRSemanticType),
}

impl SemanticPreservationValidator {
    /// Create a new semantic preservation validator
    pub fn new(config: PreservationConfig) -> Self {
        Self {
            config,
            metrics: PreservationMetrics::default(),
        }
    }

    /// Validate semantic preservation between AST and PIR
    pub fn validate_preservation(&mut self, ast: &Program, pir: &PrismIR) -> PIRResult<PreservationResult> {
        let mut context = self.build_context(ast, pir)?;
        let mut findings = Vec::new();

        // Validate module structure preservation
        self.validate_module_structure(ast, pir, &mut context, &mut findings)?;

        // Validate type preservation
        if self.config.strict_type_preservation {
            self.validate_type_preservation(ast, pir, &mut context, &mut findings)?;
        }

        // Validate function preservation
        self.validate_function_preservation(ast, pir, &mut context, &mut findings)?;

        // Validate business context preservation
        if self.config.check_business_context {
            self.validate_business_context_preservation(ast, pir, &mut context, &mut findings)?;
        }

        // Validate AI metadata preservation
        if self.config.check_ai_metadata {
            self.validate_ai_metadata_preservation(ast, pir, &mut context, &mut findings)?;
        }

        // Validate effect preservation
        if self.config.check_effect_preservation {
            self.validate_effect_preservation(ast, pir, &mut context, &mut findings)?;
        }

        // Calculate overall preservation score
        let score = self.calculate_preservation_score(&findings);
        let success = score >= self.config.min_preservation_score;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&findings);

        Ok(PreservationResult {
            success,
            score,
            findings,
            metrics: self.metrics.clone(),
            recommendations,
        })
    }

    /// Build preservation context from AST and PIR
    fn build_context(&self, ast: &Program, pir: &PrismIR) -> PIRResult<PreservationContext> {
        let mut ast_items = HashMap::new();
        let mut pir_items = HashMap::new();
        let mut type_mappings = HashMap::new();
        let mut function_mappings = HashMap::new();

        // Index AST items
        for item in &ast.items {
            match &item.kind {
                Item::Module(module) => {
                    ast_items.insert(module.name.clone(), ASTItemRef::Module(module.clone()));
                    
                    // Index module contents
                    for module_item in &module.items {
                        match &module_item.kind {
                            Item::Function(func) => {
                                let full_name = format!("{}::{}", module.name, func.name);
                                ast_items.insert(full_name.clone(), ASTItemRef::Function(func.clone()));
                                function_mappings.insert(func.name.to_string(), full_name);
                            }
                            Item::Type(ty) => {
                                let full_name = format!("{}::{}", module.name, ty.name);
                                ast_items.insert(full_name.clone(), ASTItemRef::Type(ty.clone()));
                                type_mappings.insert(ty.name.to_string(), full_name);
                            }
                            _ => {}
                        }
                    }
                }
                Item::Function(func) => {
                    ast_items.insert(func.name.to_string(), ASTItemRef::Function(func.clone()));
                    function_mappings.insert(func.name.to_string(), func.name.to_string());
                }
                Item::Type(ty) => {
                    ast_items.insert(ty.name.to_string(), ASTItemRef::Type(ty.clone()));
                    type_mappings.insert(ty.name.to_string(), ty.name.to_string());
                }
                _ => {}
            }
        }

        // Index PIR items
        for module in &pir.modules {
            pir_items.insert(module.name.clone(), PIRItemRef::Module(module.clone()));

            // Index module sections
            for section in &module.sections {
                match section {
                    crate::semantic::PIRSection::Functions(func_section) => {
                        for func in &func_section.functions {
                            let full_name = format!("{}::{}", module.name, func.name);
                            pir_items.insert(full_name, PIRItemRef::Function(func.clone()));
                        }
                    }
                    crate::semantic::PIRSection::Types(type_section) => {
                        for ty in &type_section.types {
                            let full_name = format!("{}::{}", module.name, ty.name);
                            pir_items.insert(full_name, PIRItemRef::Type(ty.clone()));
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(PreservationContext {
            ast_items,
            pir_items,
            type_mappings,
            function_mappings,
        })
    }

    /// Validate module structure preservation
    fn validate_module_structure(
        &mut self,
        ast: &Program,
        pir: &PrismIR,
        context: &mut PreservationContext,
        findings: &mut Vec<PreservationFinding>,
    ) -> PIRResult<()> {
        let ast_modules: HashSet<String> = ast.items.iter()
            .filter_map(|item| match &item.kind {
                Item::Module(module) => Some(module.name.clone()),
                _ => None,
            })
            .collect();

        let pir_modules: HashSet<String> = pir.modules.iter()
            .map(|module| module.name.clone())
            .collect();

        // Check for missing modules in PIR
        for ast_module in &ast_modules {
            if !pir_modules.contains(ast_module) {
                findings.push(PreservationFinding {
                    category: PreservationCategory::ModuleStructure,
                    severity: PreservationSeverity::Error,
                    description: format!("Module '{}' from AST is missing in PIR", ast_module),
                    location: None, // TODO: Get actual location
                    pir_location: None,
                    suggested_fix: Some("Ensure module transformation includes all AST modules".to_string()),
                });
            }
        }

        // Check for extra modules in PIR (might be intentional, so warning only)
        for pir_module in &pir_modules {
            if !ast_modules.contains(pir_module) && pir_module != "global" {
                findings.push(PreservationFinding {
                    category: PreservationCategory::ModuleStructure,
                    severity: PreservationSeverity::Warning,
                    description: format!("PIR contains module '{}' not present in AST", pir_module),
                    location: None,
                    pir_location: Some(pir_module.clone()),
                    suggested_fix: Some("Verify that additional modules are intentionally created".to_string()),
                });
            }
        }

        self.metrics.total_items += ast_modules.len();
        Ok(())
    }

    /// Validate type preservation
    fn validate_type_preservation(
        &mut self,
        _ast: &Program,
        _pir: &PrismIR,
        context: &mut PreservationContext,
        findings: &mut Vec<PreservationFinding>,
    ) -> PIRResult<()> {
        // Compare AST types with PIR types
        for (ast_name, ast_item) in &context.ast_items {
            if let ASTItemRef::Type(ast_type) = ast_item {
                // Look for corresponding PIR type
                if let Some(PIRItemRef::Type(pir_type)) = context.pir_items.get(ast_name) {
                    // Validate type name preservation
                    if ast_type.name.to_string() != pir_type.name {
                        findings.push(PreservationFinding {
                            category: PreservationCategory::TypeInformation,
                            severity: PreservationSeverity::Warning,
                            description: format!("Type name changed from '{}' to '{}'", 
                                               ast_type.name, pir_type.name),
                            location: None, // TODO: Get AST span
                            pir_location: Some(pir_type.name.clone()),
                            suggested_fix: Some("Ensure type names are preserved during transformation".to_string()),
                        });
                    }

                    // TODO: Validate type structure preservation
                    // TODO: Validate type constraints preservation
                    // TODO: Validate type metadata preservation
                } else {
                    findings.push(PreservationFinding {
                        category: PreservationCategory::TypeInformation,
                        severity: PreservationSeverity::Error,
                        description: format!("Type '{}' from AST is missing in PIR", ast_type.name),
                        location: None, // TODO: Get AST span
                        pir_location: None,
                        suggested_fix: Some("Ensure all AST types are transformed to PIR".to_string()),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate function preservation
    fn validate_function_preservation(
        &mut self,
        _ast: &Program,
        _pir: &PrismIR,
        context: &mut PreservationContext,
        findings: &mut Vec<PreservationFinding>,
    ) -> PIRResult<()> {
        // Compare AST functions with PIR functions
        for (ast_name, ast_item) in &context.ast_items {
            if let ASTItemRef::Function(ast_func) = ast_item {
                // Look for corresponding PIR function
                if let Some(PIRItemRef::Function(pir_func)) = context.pir_items.get(ast_name) {
                    // Validate function name preservation
                    if ast_func.name.to_string() != pir_func.name {
                        findings.push(PreservationFinding {
                            category: PreservationCategory::FunctionSignatures,
                            severity: PreservationSeverity::Warning,
                            description: format!("Function name changed from '{}' to '{}'", 
                                               ast_func.name, pir_func.name),
                            location: None, // TODO: Get AST span
                            pir_location: Some(pir_func.name.clone()),
                            suggested_fix: Some("Ensure function names are preserved during transformation".to_string()),
                        });
                    }

                    // TODO: Validate parameter preservation
                    // TODO: Validate return type preservation
                    // TODO: Validate function attributes preservation
                } else {
                    findings.push(PreservationFinding {
                        category: PreservationCategory::FunctionSignatures,
                        severity: PreservationSeverity::Error,
                        description: format!("Function '{}' from AST is missing in PIR", ast_func.name),
                        location: None, // TODO: Get AST span
                        pir_location: None,
                        suggested_fix: Some("Ensure all AST functions are transformed to PIR".to_string()),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate business context preservation
    fn validate_business_context_preservation(
        &mut self,
        _ast: &Program,
        pir: &PrismIR,
        _context: &mut PreservationContext,
        findings: &mut Vec<PreservationFinding>,
    ) -> PIRResult<()> {
        // Check that PIR modules have business context
        for module in &pir.modules {
            if module.business_context.domain.is_empty() {
                findings.push(PreservationFinding {
                    category: PreservationCategory::BusinessContext,
                    severity: PreservationSeverity::Warning,
                    description: format!("Module '{}' lacks business domain context", module.name),
                    location: None,
                    pir_location: Some(module.name.clone()),
                    suggested_fix: Some("Extract business domain from AST attributes or documentation".to_string()),
                });
            }

            if module.capability.is_empty() {
                findings.push(PreservationFinding {
                    category: PreservationCategory::BusinessContext,
                    severity: PreservationSeverity::Warning,
                    description: format!("Module '{}' lacks capability definition", module.name),
                    location: None,
                    pir_location: Some(module.name.clone()),
                    suggested_fix: Some("Extract capability from AST attributes or infer from module content".to_string()),
                });
            }
        }

        Ok(())
    }

    /// Validate AI metadata preservation
    fn validate_ai_metadata_preservation(
        &mut self,
        _ast: &Program,
        pir: &PrismIR,
        _context: &mut PreservationContext,
        findings: &mut Vec<PreservationFinding>,
    ) -> PIRResult<()> {
        // Check that PIR has AI metadata
        if pir.ai_metadata.module_context.is_none() {
            findings.push(PreservationFinding {
                category: PreservationCategory::AIMetadata,
                severity: PreservationSeverity::Warning,
                description: "PIR lacks module-level AI context".to_string(),
                location: None,
                pir_location: Some("ai_metadata".to_string()),
                suggested_fix: Some("Extract AI context from AST documentation and attributes".to_string()),
            });
        }

        // Check function-level AI metadata
        let function_count = pir.modules.iter()
            .flat_map(|m| &m.sections)
            .filter_map(|s| match s {
                crate::semantic::PIRSection::Functions(fs) => Some(fs.functions.len()),
                _ => None,
            })
            .sum::<usize>();

        if function_count > 0 && pir.ai_metadata.function_contexts.is_empty() {
            findings.push(PreservationFinding {
                category: PreservationCategory::AIMetadata,
                severity: PreservationSeverity::Info,
                description: "PIR has functions but no function-level AI contexts".to_string(),
                location: None,
                pir_location: Some("ai_metadata.function_contexts".to_string()),
                suggested_fix: Some("Extract function AI contexts from AST documentation".to_string()),
            });
        }

        Ok(())
    }

    /// Validate effect preservation
    fn validate_effect_preservation(
        &mut self,
        _ast: &Program,
        pir: &PrismIR,
        _context: &mut PreservationContext,
        findings: &mut Vec<PreservationFinding>,
    ) -> PIRResult<()> {
        // Check that PIR has effect information
        if pir.effect_graph.nodes.is_empty() && !pir.modules.is_empty() {
            findings.push(PreservationFinding {
                category: PreservationCategory::EffectSystem,
                severity: PreservationSeverity::Warning,
                description: "PIR has modules but no effect graph nodes".to_string(),
                location: None,
                pir_location: Some("effect_graph".to_string()),
                suggested_fix: Some("Extract effects from AST function signatures and attributes".to_string()),
            });
        }

        Ok(())
    }

    /// Calculate overall preservation score
    fn calculate_preservation_score(&self, findings: &[PreservationFinding]) -> f64 {
        if findings.is_empty() {
            return 1.0; // Perfect preservation
        }

        let mut score = 1.0;
        let mut penalty_weight = 0.0;

        for finding in findings {
            let penalty = match finding.severity {
                PreservationSeverity::Info => 0.01,
                PreservationSeverity::Warning => 0.05,
                PreservationSeverity::Error => 0.15,
                PreservationSeverity::Critical => 0.30,
            };
            penalty_weight += penalty;
        }

        score = (score - penalty_weight).max(0.0);
        score
    }

    /// Generate recommendations for improving preservation
    fn generate_recommendations(&self, findings: &[PreservationFinding]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Group findings by category
        let mut category_counts: HashMap<PreservationCategory, usize> = HashMap::new();
        for finding in findings {
            *category_counts.entry(finding.category.clone()).or_insert(0) += 1;
        }

        // Generate category-specific recommendations
        for (category, count) in category_counts {
            match category {
                PreservationCategory::TypeInformation => {
                    recommendations.push(format!(
                        "Improve type preservation: {} type-related issues found. Consider enhancing type extraction from AST.",
                        count
                    ));
                }
                PreservationCategory::FunctionSignatures => {
                    recommendations.push(format!(
                        "Improve function preservation: {} function-related issues found. Verify parameter and return type extraction.",
                        count
                    ));
                }
                PreservationCategory::BusinessContext => {
                    recommendations.push(format!(
                        "Enhance business context extraction: {} business context issues found. Extract more information from AST attributes.",
                        count
                    ));
                }
                PreservationCategory::AIMetadata => {
                    recommendations.push(format!(
                        "Improve AI metadata extraction: {} AI metadata issues found. Consider extracting from documentation comments.",
                        count
                    ));
                }
                PreservationCategory::EffectSystem => {
                    recommendations.push(format!(
                        "Enhance effect system preservation: {} effect-related issues found. Extract effects from function signatures.",
                        count
                    ));
                }
                PreservationCategory::ModuleStructure => {
                    recommendations.push(format!(
                        "Improve module structure preservation: {} module issues found. Ensure all AST modules are transformed.",
                        count
                    ));
                }
            }
        }

        recommendations
    }
}

impl Default for PreservationConfig {
    fn default() -> Self {
        Self {
            min_preservation_score: 0.8,
            strict_type_preservation: true,
            check_business_context: true,
            check_ai_metadata: true,
            check_effect_preservation: true,
            report_warnings: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preservation_validator_creation() {
        let config = PreservationConfig::default();
        let validator = SemanticPreservationValidator::new(config);
        assert_eq!(validator.metrics.total_items, 0);
    }

    #[test]
    fn test_preservation_score_calculation() {
        let validator = SemanticPreservationValidator::new(PreservationConfig::default());
        
        // Test perfect preservation
        let findings = Vec::new();
        assert_eq!(validator.calculate_preservation_score(&findings), 1.0);
        
        // Test with warnings
        let findings = vec![
            PreservationFinding {
                category: PreservationCategory::TypeInformation,
                severity: PreservationSeverity::Warning,
                description: "Test warning".to_string(),
                location: None,
                pir_location: None,
                suggested_fix: None,
            }
        ];
        let score = validator.calculate_preservation_score(&findings);
        assert!(score < 1.0 && score > 0.9);
    }

    #[test]
    fn test_empty_program_preservation() {
        let mut validator = SemanticPreservationValidator::new(PreservationConfig::default());
        
        let ast = Program {
            items: Vec::new(),
            metadata: Default::default(),
        };
        
        let pir = PrismIR::new();
        
        let result = validator.validate_preservation(&ast, &pir).unwrap();
        assert!(result.success);
        assert_eq!(result.score, 1.0);
    }
} 