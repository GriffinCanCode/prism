//! Prism Semantic Analysis Engine
//!
//! This crate implements the comprehensive semantic analysis system for the Prism programming language,
//! centralizing all semantic functionality that was previously scattered across multiple crates.
//! It embodies PLD-001's Semantic Type System and provides AI-first semantic analysis.
//!
//! ## Architecture
//!
//! The crate is organized around **Separation of Concerns** with each module having a single,
//! clear responsibility:
//!
//! - `analyzer/` - Core semantic analysis engine and orchestration
//! - `types/` - PLD-001 semantic type system implementation
//! - `inference/` - Type inference and constraint solving
//! - `validation/` - Semantic validation and business rule enforcement
//! - `context/` - AI context extraction and metadata generation
//! - `database/` - Semantic information storage and querying
//! - `patterns/` - Semantic pattern recognition and analysis
//!
//! ## Design Principles
//!
//! 1. **Conceptual Cohesion**: Each module represents one clear business concept
//! 2. **AI-First Design**: All APIs produce AI-comprehensible metadata
//! 3. **Zero-Cost Abstractions**: Semantic richness with no runtime overhead
//! 4. **Incremental Analysis**: Support for incremental compilation
//! 5. **Cross-Target Consistency**: Semantic preservation across compilation targets

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

// Core modules organized by separation of concerns
pub mod analyzer;
pub mod types;
pub mod inference;
pub mod type_inference;  // NEW: Complete type inference subsystem
pub mod validation;
pub mod context;
pub mod database;
pub mod patterns;
pub mod ai_integration;  // NEW: AI integration and metadata provider

// Re-export main types for convenience
pub use analyzer::{SemanticAnalyzer, AnalysisConfig, AnalysisResult};
pub use types::{SemanticType, TypeConstraint, BusinessRule, SemanticTypeSystem};
pub use inference::{TypeInferenceEngine, InferredType, InferenceConfig};
pub use type_inference::{
    TypeInferenceEngine as NewTypeInferenceEngine, 
    InferenceConfig as NewInferenceConfig,
    TypeInferenceResult, 
    InferredType as NewInferredType,
    TypeVar,
    ConstraintSolver,
    Unifier,
    constraints::ConstraintSet,
    TypeEnvironment,
    TypeError,
    TypeInferenceAI,
};
pub use validation::{SemanticValidator, ValidationResult, ValidationConfig};
pub use context::{SemanticContext};
pub use database::{SemanticDatabase};
pub use patterns::{SemanticPattern, PatternRecognizer};
pub use ai_integration::SemanticMetadataProvider;

// Common error type
use thiserror::Error;

/// Main error type for semantic analysis
#[derive(Debug, Error)]
pub enum SemanticError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Error message
        message: String,
    },
    
    /// Type inference error
    #[error("Type inference error: {message}")]
    TypeInferenceError {
        /// Error message
        message: String,
    },
    
    /// Validation error
    #[error("Validation error: {message}")]
    ValidationError {
        /// Error message
        message: String,
    },
    
    /// Pattern recognition error
    #[error("Pattern recognition error: {message}")]
    PatternRecognitionError {
        /// Error message
        message: String,
    },
    
    /// Database error
    #[error("Database error: {message}")]
    DatabaseError {
        /// Error message
        message: String,
    },
    
    /// Context extraction error
    #[error("Context extraction error: {message}")]
    ContextExtractionError {
        /// Error message
        message: String,
    },

    /// Type computation error
    #[error("Type computation error: {message}")]
    TypeComputationError {
        /// Error message
        message: String,
    },
}

/// Result type for semantic analysis operations
pub type SemanticResult<T> = Result<T, SemanticError>;

/// Semantic analysis configuration
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
    /// Enable business rule validation
    pub enable_business_rules: bool,
    /// Enable type inference
    pub enable_type_inference: bool,
    /// Enable pattern recognition
    pub enable_pattern_recognition: bool,
    /// Enable incremental analysis
    pub enable_incremental: bool,
    /// Maximum analysis depth
    pub max_analysis_depth: usize,
    /// AI context extraction level
    pub ai_context_level: AIContextLevel,
}

/// AI context extraction levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIContextLevel {
    /// Basic context only
    Basic,
    /// Standard context with business rules
    Standard,
    /// Full context with all metadata
    Full,
    /// Maximum context for AI training
    Maximum,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enable_ai_metadata: true,
            enable_business_rules: true,
            enable_type_inference: true,
            enable_pattern_recognition: true,
            enable_incremental: true,
            max_analysis_depth: 100,
            ai_context_level: AIContextLevel::Standard,
        }
    }
}

/// Main entry point for semantic analysis
pub struct SemanticEngine {
    /// Configuration
    config: SemanticConfig,
    /// Core analyzer
    analyzer: SemanticAnalyzer,
    /// Type system
    type_system: SemanticTypeSystem,
    /// Inference engine
    inference_engine: TypeInferenceEngine,
    /// Validator
    validator: SemanticValidator,
    /// Context extractor
    context_extractor: crate::context::AIContextExtractor,
    /// Database
    database: SemanticDatabase,
    /// Pattern recognizer
    pattern_recognizer: PatternRecognizer,
}

impl SemanticEngine {
    /// Create a new semantic engine
    pub fn new(config: SemanticConfig) -> SemanticResult<Self> {
        let analyzer = SemanticAnalyzer::new(&config)?;
        let type_system = SemanticTypeSystem::new(&config)?;
        let inference_engine = TypeInferenceEngine::new(&config)?;
        let validator = SemanticValidator::new(&config)?;
        let context_extractor = crate::context::AIContextExtractor::new(&config)?;
        let database = SemanticDatabase::new(&config)?;
        let pattern_recognizer = PatternRecognizer::new(&config)?;

        Ok(Self {
            config,
            analyzer,
            type_system,
            inference_engine,
            validator,
            context_extractor,
            database,
            pattern_recognizer,
        })
    }

    /// Analyze a program and return comprehensive semantic information
    pub fn analyze_program(&mut self, program: &prism_ast::Program) -> SemanticResult<crate::database::SemanticInfo> {
        // Step 1: Core semantic analysis
        let analysis_result = self.analyzer.analyze_program(program)?;

        // Step 2: Type inference
        let inferred_types = if self.config.enable_type_inference {
            Some(self.inference_engine.infer_types(program, &analysis_result)?)
        } else {
            None
        };

        // Step 3: Semantic validation
        let validation_result = if self.config.enable_business_rules {
            Some(self.validator.validate_program(program, &analysis_result)?)
        } else {
            None
        };

        // Step 4: Pattern recognition
        let patterns = if self.config.enable_pattern_recognition {
            self.pattern_recognizer.recognize_patterns(program, &analysis_result)?
        } else {
            Vec::new()
        };

        // Step 5: AI context extraction
        let ai_metadata = if self.config.enable_ai_metadata {
            Some(self.context_extractor.extract_context(program, &analysis_result)?)
        } else {
            None
        };

        // Step 6: Store in database
        let semantic_info = crate::database::SemanticInfo {
            symbols: analysis_result.symbols,
            types: analysis_result.types,
            inferred_types,
            validation_result,
            patterns,
            ai_metadata,
            analysis_metadata: analysis_result.metadata,
        };

        self.database.store_semantic_info(&semantic_info)?;

        Ok(semantic_info)
    }

    /// Get semantic information for a specific location
    pub fn get_semantic_info_at(&self, location: prism_common::span::Span) -> SemanticResult<Option<crate::database::SemanticInfo>> {
        self.database.get_semantic_info_at(location)
    }

    /// Export AI-readable context for external tools
    pub fn export_ai_context(&self, location: prism_common::span::Span) -> SemanticResult<crate::context::AIMetadata> {
        self.context_extractor.export_ai_context(location, &self.database)
    }

    /// Get database statistics
    pub fn get_statistics(&self) -> database::DatabaseStatistics {
        self.database.get_statistics()
    }
}

// Make SemanticEngine cloneable for async compatibility
impl Clone for SemanticEngine {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone implementation
        // In a real implementation, we'd need to handle the internal state properly
        Self::new(self.config.clone()).expect("Failed to clone SemanticEngine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ast::Program;

    #[test]
    fn test_semantic_engine_creation() {
        let config = SemanticConfig::default();
        let engine = SemanticEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_semantic_analysis_basic() {
        let config = SemanticConfig::default();
        let mut engine = SemanticEngine::new(config).unwrap();
        
        // Create a minimal program for testing
        let program = Program {
            items: Vec::new(),
            source_id: prism_common::SourceId::new(0),
            metadata: prism_ast::ProgramMetadata::default(),
        };

        let result = engine.analyze_program(&program);
        assert!(result.is_ok());
        
        let semantic_info = result.unwrap();
        assert!(semantic_info.symbols.is_empty()); // No items means no symbols
        assert!(semantic_info.types.is_empty());
    }

    #[test]
    fn test_ai_context_levels() {
        let mut config = SemanticConfig::default();
        config.ai_context_level = AIContextLevel::Maximum;
        
        let engine = SemanticEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_database_statistics() {
        let config = SemanticConfig::default();
        let engine = SemanticEngine::new(config).unwrap();
        
        let stats = engine.get_statistics();
        assert_eq!(stats.symbol_count, 0);
        assert_eq!(stats.type_count, 0);
    }
}
