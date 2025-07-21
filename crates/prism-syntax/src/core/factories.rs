//! Factory System for Syntax Components
//!
//! This module implements the factory pattern for creating syntax parsers and normalizers,
//! maintaining conceptual cohesion around "component creation and configuration".
//! It follows Prism's design principles by providing clean separation of concerns between
//! component creation and component usage.
//!
//! ## Design Principles
//!
//! 1. **Single Responsibility**: Each factory creates one type of component
//! 2. **Configuration-Driven**: Factories use configuration objects for customization
//! 3. **Type Safety**: Compile-time guarantees about syntax style support
//! 4. **Extensibility**: Easy to add new syntax styles without breaking existing code
//! 5. **Testability**: Factories can be easily mocked and tested

use crate::{
    detection::SyntaxStyle,
    styles::{StyleParser, CLikeParser, PythonLikeParser, RustLikeParser, CanonicalParser},
    normalization::{
        traits::StyleNormalizer,
        CLikeNormalizer, PythonLikeNormalizer, RustLikeNormalizer, CanonicalNormalizer,
        NormalizationConfig,
    },
    validation::{Validator, ValidationConfig},
    SyntaxError,
};
use std::collections::HashMap;
use thiserror::Error;

/// Factory for creating style-specific parsers
#[derive(Debug)]
pub struct ParserFactory {
    /// Parser configurations by syntax style
    configs: HashMap<SyntaxStyle, ParserConfig>,
}

/// Factory for creating style-specific normalizers
#[derive(Debug)]
pub struct NormalizerFactory {
    /// Base normalization configuration
    base_config: NormalizationConfig,
    /// Style-specific normalizer configurations
    style_configs: HashMap<SyntaxStyle, NormalizerStyleConfig>,
}

/// Factory for creating validators
#[derive(Debug)]
pub struct ValidatorFactory {
    /// Base validation configuration
    config: ValidationConfig,
}

/// Configuration for parser creation
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// Whether to enable error recovery
    pub enable_error_recovery: bool,
    /// Whether to generate AI metadata during parsing
    pub generate_ai_metadata: bool,
    /// Maximum nesting depth allowed
    pub max_nesting_depth: usize,
    /// Style-specific configuration
    pub style_specific: StyleSpecificConfig,
}

/// Style-specific configuration for parsers
#[derive(Debug, Clone)]
pub enum StyleSpecificConfig {
    /// C-like parser configuration
    CLike {
        /// Require semicolons after statements
        require_semicolons: bool,
        /// Allow trailing commas
        allow_trailing_commas: bool,
    },
    /// Python-like parser configuration
    PythonLike {
        /// Tab size for indentation
        tab_size: usize,
        /// Allow mixed indentation
        allow_mixed_indentation: bool,
        /// Enable error recovery
        enable_error_recovery: bool,
    },
    /// Rust-like parser configuration
    RustLike {
        /// Allow trailing commas
        allow_trailing_commas: bool,
        /// Parse ownership annotations
        parse_ownership: bool,
        /// Parse lifetime annotations
        parse_lifetimes: bool,
    },
    /// Canonical parser configuration
    Canonical {
        /// Enable error recovery
        error_recovery: bool,
        /// Enable semantic validation
        semantic_validation: bool,
    },
}

/// Configuration for normalizer creation
#[derive(Debug, Clone)]
pub struct NormalizerStyleConfig {
    /// Whether to preserve formatting hints
    pub preserve_formatting: bool,
    /// Whether to generate business insights
    pub generate_business_insights: bool,
    /// Style-specific normalization options
    pub style_options: NormalizerStyleOptions,
}

/// Style-specific normalizer options
#[derive(Debug, Clone)]
pub enum NormalizerStyleOptions {
    /// C-like normalizer options
    CLike {
        /// Preserve operator precedence information
        preserve_operator_precedence: bool,
        /// Normalize C-style casts
        normalize_casts: bool,
    },
    /// Python-like normalizer options
    PythonLike {
        /// Generate type metadata
        generate_type_metadata: bool,
        /// Track import dependencies
        track_import_dependencies: bool,
    },
    /// Rust-like normalizer options
    RustLike {
        /// Preserve ownership annotations
        preserve_ownership_annotations: bool,
        /// Normalize match expressions
        normalize_match_expressions: bool,
    },
    /// Canonical normalizer options
    Canonical {
        /// Strict validation of canonical constructs
        strict_validation: bool,
        /// Enhance AI metadata
        enhance_ai_metadata: bool,
    },
}

/// Errors that can occur during factory operations
#[derive(Debug, Error)]
pub enum FactoryError {
    /// Unsupported syntax style
    #[error("Unsupported syntax style: {style:?}")]
    UnsupportedStyle { style: SyntaxStyle },
    
    /// Invalid configuration
    #[error("Invalid configuration for {component}: {reason}")]
    InvalidConfiguration { component: String, reason: String },
    
    /// Component creation failed
    #[error("Failed to create {component}: {reason}")]
    CreationFailed { component: String, reason: String },
}

impl ParserFactory {
    /// Create a new parser factory with default configurations
    pub fn new() -> Self {
        let mut configs = HashMap::new();
        
        // Default configurations for each syntax style
        configs.insert(SyntaxStyle::CLike, ParserConfig {
            enable_error_recovery: true,
            generate_ai_metadata: true,
            max_nesting_depth: 256,
            style_specific: StyleSpecificConfig::CLike {
                require_semicolons: false,
                allow_trailing_commas: true,
            },
        });
        
        configs.insert(SyntaxStyle::PythonLike, ParserConfig {
            enable_error_recovery: true,
            generate_ai_metadata: true,
            max_nesting_depth: 256,
            style_specific: StyleSpecificConfig::PythonLike {
                tab_size: 4,
                allow_mixed_indentation: false,
                enable_error_recovery: true,
            },
        });
        
        configs.insert(SyntaxStyle::RustLike, ParserConfig {
            enable_error_recovery: true,
            generate_ai_metadata: true,
            max_nesting_depth: 256,
            style_specific: StyleSpecificConfig::RustLike {
                allow_trailing_commas: true,
                parse_ownership: true,
                parse_lifetimes: false,
            },
        });
        
        configs.insert(SyntaxStyle::Canonical, ParserConfig {
            enable_error_recovery: true,
            generate_ai_metadata: true,
            max_nesting_depth: 256,
            style_specific: StyleSpecificConfig::Canonical {
                error_recovery: true,
                semantic_validation: true,
            },
        });
        
        Self { configs }
    }
    
    /// Create a parser factory with custom configurations
    pub fn with_configs(configs: HashMap<SyntaxStyle, ParserConfig>) -> Self {
        Self { configs }
    }
    
    /// Create a parser for the specified syntax style
    pub fn create_parser(&self, style: SyntaxStyle) -> Result<Box<dyn StyleParser<Output = ParsedOutput, Error = ParseError>>, FactoryError> {
        let config = self.configs.get(&style)
            .ok_or(FactoryError::UnsupportedStyle { style })?;
        
        match style {
            SyntaxStyle::CLike => {
                if let StyleSpecificConfig::CLike { require_semicolons, allow_trailing_commas } = &config.style_specific {
                    let parser_config = crate::styles::c_like::CLikeConfig {
                        require_semicolons: *require_semicolons,
                        allow_trailing_commas: *allow_trailing_commas,
                        enable_error_recovery: config.enable_error_recovery,
                        generate_ai_metadata: config.generate_ai_metadata,
                    };
                    Ok(Box::new(CLikeParser::with_config(parser_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "CLikeParser".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
            SyntaxStyle::PythonLike => {
                if let StyleSpecificConfig::PythonLike { tab_size, allow_mixed_indentation, enable_error_recovery } = &config.style_specific {
                    let parser_config = crate::styles::python_like::PythonParserConfig {
                        tab_size: *tab_size,
                        allow_mixed_indentation: *allow_mixed_indentation,
                        enable_error_recovery: *enable_error_recovery,
                        generate_ai_metadata: config.generate_ai_metadata,
                        max_nesting_depth: config.max_nesting_depth,
                        ..Default::default()
                    };
                    Ok(Box::new(PythonLikeParser::with_config(parser_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "PythonLikeParser".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
            SyntaxStyle::RustLike => {
                if let StyleSpecificConfig::RustLike { allow_trailing_commas, parse_ownership, parse_lifetimes } = &config.style_specific {
                    let parser_config = crate::styles::rust_like::RustLikeConfig {
                        allow_trailing_commas: *allow_trailing_commas,
                        parse_ownership: *parse_ownership,
                        parse_lifetimes: *parse_lifetimes,
                        require_explicit_returns: false,
                    };
                    Ok(Box::new(RustLikeParser::with_config(parser_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "RustLikeParser".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
            SyntaxStyle::Canonical => {
                if let StyleSpecificConfig::Canonical { error_recovery, semantic_validation } = &config.style_specific {
                    let parser_config = crate::styles::canonical::CanonicalConfig {
                        error_recovery: *error_recovery,
                        semantic_validation: *semantic_validation,
                        ..Default::default()
                    };
                    Ok(Box::new(CanonicalParser::with_config(parser_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "CanonicalParser".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
        }
    }
    
    /// Update configuration for a specific syntax style
    pub fn update_config(&mut self, style: SyntaxStyle, config: ParserConfig) {
        self.configs.insert(style, config);
    }
    
    /// Get configuration for a specific syntax style
    pub fn get_config(&self, style: SyntaxStyle) -> Option<&ParserConfig> {
        self.configs.get(&style)
    }
}

impl NormalizerFactory {
    /// Create a new normalizer factory with default configurations
    pub fn new() -> Self {
        Self::with_config(NormalizationConfig::default())
    }
    
    /// Create a normalizer factory with custom base configuration
    pub fn with_config(base_config: NormalizationConfig) -> Self {
        let mut style_configs = HashMap::new();
        
        // Default style-specific configurations
        style_configs.insert(SyntaxStyle::CLike, NormalizerStyleConfig {
            preserve_formatting: true,
            generate_business_insights: true,
            style_options: NormalizerStyleOptions::CLike {
                preserve_operator_precedence: true,
                normalize_casts: true,
            },
        });
        
        style_configs.insert(SyntaxStyle::PythonLike, NormalizerStyleConfig {
            preserve_formatting: true,
            generate_business_insights: true,
            style_options: NormalizerStyleOptions::PythonLike {
                generate_type_metadata: true,
                track_import_dependencies: true,
            },
        });
        
        style_configs.insert(SyntaxStyle::RustLike, NormalizerStyleConfig {
            preserve_formatting: true,
            generate_business_insights: true,
            style_options: NormalizerStyleOptions::RustLike {
                preserve_ownership_annotations: true,
                normalize_match_expressions: true,
            },
        });
        
        style_configs.insert(SyntaxStyle::Canonical, NormalizerStyleConfig {
            preserve_formatting: true,
            generate_business_insights: true,
            style_options: NormalizerStyleOptions::Canonical {
                strict_validation: true,
                enhance_ai_metadata: true,
            },
        });
        
        Self {
            base_config,
            style_configs,
        }
    }
    
    /// Create a normalizer for the specified syntax style
    pub fn create_normalizer(&self, style: SyntaxStyle) -> Result<Box<dyn StyleNormalizer>, FactoryError> {
        let style_config = self.style_configs.get(&style)
            .ok_or(FactoryError::UnsupportedStyle { style })?;
        
        match style {
            SyntaxStyle::CLike => {
                if let NormalizerStyleOptions::CLike { preserve_operator_precedence, normalize_casts } = &style_config.style_options {
                    let normalizer_config = crate::normalization::c_like::CLikeNormalizerConfig {
                        preserve_operator_precedence: *preserve_operator_precedence,
                        normalize_casts: *normalize_casts,
                        normalize_arrays: true,
                        handle_pointers: true,
                        preserve_memory_hints: false,
                        custom_operator_mappings: Default::default(),
                    };
                    Ok(Box::new(CLikeNormalizer::with_config(normalizer_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "CLikeNormalizer".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
            SyntaxStyle::PythonLike => {
                if let NormalizerStyleOptions::PythonLike { generate_type_metadata, track_import_dependencies } = &style_config.style_options {
                    let normalizer_config = crate::normalization::python_like::PythonNormalizerConfig {
                        preserve_formatting_hints: style_config.preserve_formatting,
                        generate_type_metadata: *generate_type_metadata,
                        track_import_dependencies: *track_import_dependencies,
                        normalize_string_literals: true,
                        preserve_indentation_info: true,
                        generate_business_insights: style_config.generate_business_insights,
                        max_analysis_depth: 100,
                        enable_experimental_features: false,
                    };
                    Ok(Box::new(PythonLikeNormalizer::with_config(normalizer_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "PythonLikeNormalizer".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
            SyntaxStyle::RustLike => {
                if let NormalizerStyleOptions::RustLike { preserve_ownership_annotations, normalize_match_expressions } = &style_config.style_options {
                    let normalizer_config = crate::normalization::rust_like::RustLikeNormalizerConfig {
                        preserve_ownership_annotations: *preserve_ownership_annotations,
                        normalize_match_expressions: *normalize_match_expressions,
                        preserve_lifetimes: false,
                        normalize_error_handling: true,
                        preserve_trait_bounds: true,
                        handle_unsafe_blocks: true,
                        custom_construct_mappings: Default::default(),
                    };
                    Ok(Box::new(RustLikeNormalizer::with_config(normalizer_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "RustLikeNormalizer".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
            SyntaxStyle::Canonical => {
                if let NormalizerStyleOptions::Canonical { strict_validation, enhance_ai_metadata } = &style_config.style_options {
                    let normalizer_config = crate::normalization::canonical::CanonicalNormalizerConfig {
                        strict_validation: *strict_validation,
                        enhance_ai_metadata: *enhance_ai_metadata,
                        validate_semantics: true,
                        preserve_ast_metadata: true,
                        validate_module_structure: true,
                        custom_validation_rules: Default::default(),
                    };
                    Ok(Box::new(CanonicalNormalizer::with_config(normalizer_config)))
                } else {
                    Err(FactoryError::InvalidConfiguration {
                        component: "CanonicalNormalizer".to_string(),
                        reason: "Invalid style-specific configuration".to_string(),
                    })
                }
            }
        }
    }
    
    /// Update style-specific configuration
    pub fn update_style_config(&mut self, style: SyntaxStyle, config: NormalizerStyleConfig) {
        self.style_configs.insert(style, config);
    }
    
    /// Get style-specific configuration
    pub fn get_style_config(&self, style: SyntaxStyle) -> Option<&NormalizerStyleConfig> {
        self.style_configs.get(&style)
    }
    
    /// Update base configuration
    pub fn update_base_config(&mut self, config: NormalizationConfig) {
        self.base_config = config;
    }
    
    /// Get base configuration
    pub fn base_config(&self) -> &NormalizationConfig {
        &self.base_config
    }
}

impl ValidatorFactory {
    /// Create a new validator factory with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }
    
    /// Create a validator factory with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }
    
    /// Create a validator instance
    pub fn create_validator(&self) -> Result<Validator, FactoryError> {
        Ok(Validator::with_config(self.config.clone()))
    }
    
    /// Update validator configuration
    pub fn update_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }
    
    /// Get current validator configuration
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }
}

impl Default for ParserFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for NormalizerFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ValidatorFactory {
    fn default() -> Self {
        Self::new()
    }
}

// Type alias for parsed output to simplify the trait bounds
pub type ParsedOutput = Box<dyn std::any::Any + Send + Sync>;
pub type ParseError = crate::SyntaxError;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parser_factory_creation() {
        let factory = ParserFactory::new();
        
        // Test that we can create parsers for all supported styles
        assert!(factory.create_parser(SyntaxStyle::CLike).is_ok());
        assert!(factory.create_parser(SyntaxStyle::PythonLike).is_ok());
        assert!(factory.create_parser(SyntaxStyle::RustLike).is_ok());
        assert!(factory.create_parser(SyntaxStyle::Canonical).is_ok());
    }
    
    #[test]
    fn test_normalizer_factory_creation() {
        let factory = NormalizerFactory::new();
        
        // Test that we can create normalizers for all supported styles
        assert!(factory.create_normalizer(SyntaxStyle::CLike).is_ok());
        assert!(factory.create_normalizer(SyntaxStyle::PythonLike).is_ok());
        assert!(factory.create_normalizer(SyntaxStyle::RustLike).is_ok());
        assert!(factory.create_normalizer(SyntaxStyle::Canonical).is_ok());
    }
    
    #[test]
    fn test_validator_factory_creation() {
        let factory = ValidatorFactory::new();
        
        // Test that we can create a validator
        assert!(factory.create_validator().is_ok());
    }
    
    #[test]
    fn test_configuration_updates() {
        let mut factory = ParserFactory::new();
        
        let custom_config = ParserConfig {
            enable_error_recovery: false,
            generate_ai_metadata: false,
            max_nesting_depth: 128,
            style_specific: StyleSpecificConfig::CLike {
                require_semicolons: true,
                allow_trailing_commas: false,
            },
        };
        
        factory.update_config(SyntaxStyle::CLike, custom_config.clone());
        
        let retrieved_config = factory.get_config(SyntaxStyle::CLike).unwrap();
        assert_eq!(retrieved_config.enable_error_recovery, false);
        assert_eq!(retrieved_config.generate_ai_metadata, false);
        assert_eq!(retrieved_config.max_nesting_depth, 128);
    }
} 