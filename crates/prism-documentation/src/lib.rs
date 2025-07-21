//! Documentation validation and JSDoc compatibility for Prism language.
//!
//! This crate implements PSG-003: PrismDoc Standards, providing comprehensive
//! documentation validation, JSDoc compatibility, and AI-readable metadata
//! generation for the Prism programming language.
//!
//! ## Features
//!
//! - **PSG-003 Compliance**: Full validation of PrismDoc standards
//! - **JSDoc Compatibility**: Support for JSDoc-style annotations
//! - **Required Annotations**: Enforcement of mandatory documentation
//! - **AI Integration**: Generation of AI-comprehensible metadata
//! - **Multi-Syntax Support**: Works with all Prism syntax styles
//! - **Compile-Time Validation**: Documentation errors as type errors
//!
//! ## Architecture
//!
//! The documentation system follows conceptual cohesion principles with
//! specialized modules for each responsibility:
//!
//! - `validation/` - Core validation engine and rules
//! - `jsdoc/` - JSDoc compatibility and conversion
//! - `extraction/` - Documentation extraction from AST
//! - `generation/` - Documentation generation and export
//! - `requirements/` - PSG-003 requirement checking
//! - `ai_integration/` - AI-readable metadata generation
//!
//! ## Examples
//!
//! ```rust
//! use prism_documentation::{DocumentationValidator, ValidationConfig};
//! use prism_ast::Program;
//!
//! let validator = DocumentationValidator::new(ValidationConfig::strict());
//! let validation_result = validator.validate_program(&program)?;
//!
//! if !validation_result.is_compliant() {
//!     for violation in validation_result.violations() {
//!         eprintln!("Documentation error: {}", violation.message());
//!     }
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod validation;
pub mod jsdoc;
pub mod extraction;
pub mod generation;
pub mod requirements;
pub mod ai_integration;

// Re-export main types for convenience
pub use validation::{
    DocumentationValidator, ValidationConfig, ValidationResult, ValidationError,
    ValidationViolation, ViolationSeverity
};
pub use jsdoc::{
    JSDocProcessor, JSDocInfo, JSDocTag, JSDocCompatibility
};
pub use extraction::{
    DocumentationExtractor, ExtractedDocumentation, DocumentationElement
};
pub use generation::{
    DocumentationGenerator, GenerationConfig, GeneratedDocumentation
};
pub use requirements::{
    RequirementChecker, RequiredAnnotationType, AnnotationRequirement
};
pub use ai_integration::{
    AIMetadataGenerator, AIDocumentationMetadata, AIContextExtractor
};

use prism_common::span::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use thiserror::Error;

/// Main error type for documentation processing
#[derive(Debug, Error)]
pub enum DocumentationError {
    /// Validation failed with specific violations
    #[error("Documentation validation failed: {violations_count} violations found")]
    ValidationFailed {
        violations_count: usize,
        violations: Vec<ValidationViolation>,
    },

    /// Required annotation is missing
    #[error("Missing required annotation '{annotation_type}' at {location}")]
    MissingRequiredAnnotation {
        annotation_type: String,
        location: Span,
    },

    /// JSDoc processing failed
    #[error("JSDoc processing failed: {reason}")]
    JSDocProcessingFailed { reason: String },

    /// Documentation extraction failed
    #[error("Documentation extraction failed: {reason}")]
    ExtractionFailed { reason: String },

    /// AI metadata generation failed
    #[error("AI metadata generation failed: {reason}")]
    AIMetadataFailed { reason: String },

    /// Invalid documentation format
    #[error("Invalid documentation format at {location}: {reason}")]
    InvalidFormat { location: Span, reason: String },

    /// Documentation inconsistent with code
    #[error("Documentation inconsistent with code at {location}: {reason}")]
    InconsistentWithCode { location: Span, reason: String },
}

/// Result type for documentation operations
pub type DocumentationResult<T> = Result<T, DocumentationError>;

/// Documentation system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationConfig {
    /// Validation configuration
    pub validation: ValidationConfig,

    /// JSDoc compatibility settings
    pub jsdoc_compatibility: JSDocCompatibility,

    /// AI integration settings
    pub ai_integration: AIIntegrationConfig,

    /// Generation settings
    pub generation: GenerationConfig,

    /// Custom requirement overrides
    pub custom_requirements: HashMap<String, AnnotationRequirement>,
}

/// AI integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIIntegrationConfig {
    /// Enable AI metadata generation
    pub enabled: bool,

    /// Include business context in AI metadata
    pub include_business_context: bool,

    /// Include architectural patterns
    pub include_architectural_patterns: bool,

    /// Include semantic relationships
    pub include_semantic_relationships: bool,

    /// AI context detail level
    pub detail_level: AIDetailLevel,
}

/// AI metadata detail levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AIDetailLevel {
    /// Minimal AI context
    Minimal,
    /// Standard AI context
    Standard,
    /// Comprehensive AI context
    Comprehensive,
    /// Maximum AI context for training
    Maximum,
}

/// Complete documentation processing system
#[derive(Debug)]
pub struct DocumentationSystem {
    /// Configuration
    config: DocumentationConfig,

    /// Validation engine
    validator: DocumentationValidator,

    /// JSDoc processor
    jsdoc_processor: JSDocProcessor,

    /// Documentation extractor
    extractor: DocumentationExtractor,

    /// Documentation generator
    generator: DocumentationGenerator,

    /// Requirement checker
    requirement_checker: RequirementChecker,

    /// AI metadata generator
    ai_metadata_generator: AIMetadataGenerator,
}

impl DocumentationSystem {
    /// Create a new documentation system with default configuration
    pub fn new() -> Self {
        let config = DocumentationConfig::default();
        Self::with_config(config)
    }

    /// Create a new documentation system with custom configuration
    pub fn with_config(config: DocumentationConfig) -> Self {
        Self {
            validator: DocumentationValidator::new(config.validation.clone()),
            jsdoc_processor: JSDocProcessor::new(config.jsdoc_compatibility),
            extractor: DocumentationExtractor::new(),
            generator: DocumentationGenerator::new(config.generation.clone()),
            requirement_checker: RequirementChecker::new(),
            ai_metadata_generator: AIMetadataGenerator::new(config.ai_integration.clone()),
            config,
        }
    }

    /// Process documentation for a complete program
    pub fn process_program(&mut self, program: &prism_ast::Program) -> DocumentationResult<ProcessingResult> {
        // Step 1: Extract documentation from AST
        let extracted_docs = self.extractor.extract_from_program(program)
            .map_err(|e| DocumentationError::ExtractionFailed { 
                reason: e.to_string() 
            })?;

        // Step 2: Validate against PSG-003 requirements
        let validation_result = self.validator.validate_extracted(&extracted_docs)?;

        // Step 3: Check required annotations
        let requirement_violations = self.requirement_checker.check_program(program)?;

        // Step 4: Process JSDoc compatibility
        let jsdoc_result = self.jsdoc_processor.process_extracted(&extracted_docs)
            .map_err(|e| DocumentationError::JSDocProcessingFailed { 
                reason: e.to_string() 
            })?;

        // Step 5: Generate AI metadata (if enabled)
        let ai_metadata = if self.config.ai_integration.enabled {
            Some(self.ai_metadata_generator.generate_for_program(program, &extracted_docs)
                .map_err(|e| DocumentationError::AIMetadataFailed { 
                    reason: e.to_string() 
                })?)
        } else {
            None
        };

        // Step 6: Generate documentation output
        let generated_docs = self.generator.generate_from_extracted(&extracted_docs)?;

        // Combine all results
        let mut all_violations = validation_result.violations;
        all_violations.extend(requirement_violations);

        let is_compliant = all_violations.is_empty();

        Ok(ProcessingResult {
            extracted_documentation: extracted_docs,
            validation_result: ValidationResult {
                is_compliant,
                violations: all_violations,
                warnings: validation_result.warnings,
                suggestions: validation_result.suggestions,
            },
            jsdoc_result,
            ai_metadata,
            generated_documentation: generated_docs,
        })
    }

    /// Validate documentation for a single module
    pub fn validate_module(&mut self, module: &prism_ast::AstNode<prism_ast::Item>) -> DocumentationResult<ValidationResult> {
        // Extract documentation from the module
        let extracted_docs = self.extractor.extract_from_item(module)
            .map_err(|e| DocumentationError::ExtractionFailed { 
                reason: e.to_string() 
            })?;

        // Validate the extracted documentation
        self.validator.validate_extracted(&extracted_docs)
    }

    /// Generate AI metadata for documentation
    pub fn generate_ai_metadata(&mut self, program: &prism_ast::Program) -> DocumentationResult<AIDocumentationMetadata> {
        let extracted_docs = self.extractor.extract_from_program(program)
            .map_err(|e| DocumentationError::ExtractionFailed { 
                reason: e.to_string() 
            })?;

        self.ai_metadata_generator.generate_for_program(program, &extracted_docs)
            .map_err(|e| DocumentationError::AIMetadataFailed { 
                reason: e.to_string() 
            })
    }

    /// Get current configuration
    pub fn config(&self) -> &DocumentationConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DocumentationConfig) {
        self.config = config.clone();
        self.validator = DocumentationValidator::new(config.validation.clone());
        self.jsdoc_processor = JSDocProcessor::new(config.jsdoc_compatibility);
        self.generator = DocumentationGenerator::new(config.generation.clone());
        self.ai_metadata_generator = AIMetadataGenerator::new(config.ai_integration.clone());
    }
}

/// Result of complete documentation processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Extracted documentation elements
    pub extracted_documentation: ExtractedDocumentation,

    /// Validation results
    pub validation_result: ValidationResult,

    /// JSDoc processing results
    pub jsdoc_result: jsdoc::ProcessingResult,

    /// AI metadata (if generated)
    pub ai_metadata: Option<AIDocumentationMetadata>,

    /// Generated documentation output
    pub generated_documentation: GeneratedDocumentation,
}

impl ProcessingResult {
    /// Check if documentation is fully compliant with PSG-003
    pub fn is_compliant(&self) -> bool {
        self.validation_result.is_compliant
    }

    /// Get all violations found during processing
    pub fn violations(&self) -> &[ValidationViolation] {
        &self.validation_result.violations
    }

    /// Get all warnings found during processing
    pub fn warnings(&self) -> &[String] {
        &self.validation_result.warnings
    }

    /// Get improvement suggestions
    pub fn suggestions(&self) -> &[String] {
        &self.validation_result.suggestions
    }

    /// Get AI metadata if available
    pub fn ai_metadata(&self) -> Option<&AIDocumentationMetadata> {
        self.ai_metadata.as_ref()
    }
}

impl Default for DocumentationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DocumentationConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            jsdoc_compatibility: JSDocCompatibility::default(),
            ai_integration: AIIntegrationConfig::default(),
            generation: GenerationConfig::default(),
            custom_requirements: HashMap::new(),
        }
    }
}

impl Default for AIIntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            include_business_context: true,
            include_architectural_patterns: true,
            include_semantic_relationships: true,
            detail_level: AIDetailLevel::Standard,
        }
    }
}

/// Validate documentation for a program using default configuration
pub fn validate_program(program: &prism_ast::Program) -> DocumentationResult<ValidationResult> {
    let mut system = DocumentationSystem::new();
    let result = system.process_program(program)?;
    Ok(result.validation_result)
}

/// Generate AI metadata for a program using default configuration
pub fn generate_ai_metadata(program: &prism_ast::Program) -> DocumentationResult<AIDocumentationMetadata> {
    let mut system = DocumentationSystem::new();
    system.generate_ai_metadata(program)
}

/// Check if documentation is PSG-003 compliant
pub fn is_psg003_compliant(program: &prism_ast::Program) -> DocumentationResult<bool> {
    let validation_result = validate_program(program)?;
    Ok(validation_result.is_compliant)
} 