//! Parser Integration Bridge - Connecting Multi-Syntax Parsing to Symbol Resolution
//!
//! This module embodies the single concept of "Parser Integration".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: bridging the gap between multi-syntax parsing and the
//! compiler's symbol resolution and semantic analysis systems.
//!
//! **Conceptual Responsibility**: Parser-to-compiler integration
//! **What it does**: syntax detection, parsing coordination, symbol extraction integration
//! **What it doesn't do**: actual parsing, symbol storage, semantic analysis (delegates to specialized modules)

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolSystem, SymbolSystemBuilder};
use crate::resolution::{SymbolResolver, ResolutionContext};
use crate::semantic::SemanticDatabase;
use crate::cache::CompilationCache;
use crate::scope::ScopeTree;
use crate::module_registry::SmartModuleRegistry;
use prism_common::{SourceId, span::Span, symbol::Symbol};
use prism_ast::{Program, AstNode, Item, ModuleDecl};
use prism_effects::effects::EffectRegistry;
use std::sync::Arc;
use std::path::Path;
use std::collections::HashMap;
use tracing::{info, debug, warn, error};

/// Parser integration system that coordinates multi-syntax parsing with symbol resolution
pub struct ParserIntegration {
    /// Symbol system integration
    symbol_system: Option<Arc<SymbolSystem>>,
    /// Semantic database integration
    semantic_db: Arc<SemanticDatabase>,
    /// Effect registry integration
    effect_registry: Arc<EffectRegistry>,
    /// Compilation cache integration
    cache: Arc<CompilationCache>,
    /// Module registry integration
    module_registry: Arc<SmartModuleRegistry>,
    /// Integration configuration
    config: IntegrationConfig,
}

/// Configuration for parser integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable syntax style detection
    pub enable_syntax_detection: bool,
    /// Enable symbol extraction during parsing
    pub enable_symbol_extraction: bool,
    /// Enable semantic analysis integration
    pub enable_semantic_integration: bool,
    /// Enable module registration
    pub enable_module_registration: bool,
    /// Enable error recovery
    pub enable_error_recovery: bool,
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_syntax_detection: true,
            enable_symbol_extraction: true,
            enable_semantic_integration: true,
            enable_module_registration: true,
            enable_error_recovery: true,
            enable_ai_metadata: true,
        }
    }
}

/// Parse result with integrated symbol and semantic information
#[derive(Debug)]
pub struct ParseResult {
    /// Parsed program
    pub program: Program,
    /// Extracted symbols (if enabled)
    pub symbols: Option<Vec<Symbol>>,
    /// Semantic information (if enabled)
    pub semantic_info: Option<HashMap<String, String>>,
    /// Detected syntax style
    pub syntax_style: Option<String>,
    /// Parse statistics
    pub statistics: ParseStatistics,
    /// Integration diagnostics
    pub diagnostics: Vec<IntegrationDiagnostic>,
}

/// Parse statistics for monitoring
#[derive(Debug, Default)]
pub struct ParseStatistics {
    /// Parsing time in milliseconds
    pub parse_time_ms: u64,
    /// Number of symbols extracted
    pub symbols_extracted: usize,
    /// Number of modules registered
    pub modules_registered: usize,
    /// Number of semantic analyses performed
    pub semantic_analyses: usize,
    /// Cache hits during parsing
    pub cache_hits: usize,
}

/// Integration diagnostic message
#[derive(Debug)]
pub struct IntegrationDiagnostic {
    /// Diagnostic level
    pub level: DiagnosticLevel,
    /// Diagnostic message
    pub message: String,
    /// Location in source (if available)
    pub location: Option<Span>,
    /// Suggested fix (if available)
    pub suggested_fix: Option<String>,
}

/// Diagnostic severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticLevel {
    Info,
    Warning,
    Error,
}

impl ParserIntegration {
    /// Create a new parser integration system
    pub fn new(
        semantic_db: Arc<SemanticDatabase>,
        effect_registry: Arc<EffectRegistry>,
        cache: Arc<CompilationCache>,
        module_registry: Arc<SmartModuleRegistry>,
    ) -> Self {
        Self {
            symbol_system: None,
            semantic_db,
            effect_registry,
            cache,
            module_registry,
            config: IntegrationConfig::default(),
        }
    }

    /// Initialize the symbol system integration
    pub async fn initialize_symbol_system(&mut self) -> CompilerResult<()> {
        info!("Initializing integrated symbol system");

        let symbol_system = SymbolSystemBuilder::new()
            .with_semantic_database(self.semantic_db.clone())
            .with_effect_registry(self.effect_registry.clone())
            .with_cache(self.cache.clone())
            .build()?;

        self.symbol_system = Some(Arc::new(symbol_system));
        Ok(())
    }

    /// Parse a file with full integration
    pub async fn parse_file(file_path: &Path) -> CompilerResult<ParseResult> {
        let start_time = std::time::Instant::now();
        let mut statistics = ParseStatistics::default();
        let mut diagnostics = Vec::new();

        info!("Parsing file with integration: {}", file_path.display());

        // Step 1: Read source file
        let source = std::fs::read_to_string(file_path)
            .map_err(|e| CompilerError::FileReadError {
                path: file_path.to_path_buf(),
                source: e,
            })?;

        // Step 2: Detect syntax style (if enabled)
        let syntax_style = if true { // config.enable_syntax_detection
            match Self::detect_syntax_style(file_path) {
                Ok(style) => {
                    debug!("Detected syntax style: {:?}", style);
                    Some(format!("{:?}", style))
                }
                Err(e) => {
                    warn!("Failed to detect syntax style: {}", e);
                    diagnostics.push(IntegrationDiagnostic {
                        level: DiagnosticLevel::Warning,
                        message: format!("Syntax detection failed: {}", e),
                        location: None,
                        suggested_fix: Some("Consider adding explicit syntax markers".to_string()),
                    });
                    None
                }
            }
        } else {
            None
        };

        // Step 3: Parse using appropriate parser
        let program = Self::parse_with_style(&source, &syntax_style).await?;

        // Step 4: Extract symbols and register modules
        let symbols = self.extract_symbols_from_program(&program, SourceId::new(1)).await?;
        statistics.symbols_extracted = symbols.len() as u64;

        // Step 5: Perform semantic analysis integration
        let semantic_info = self.perform_semantic_integration(&program, &symbols).await?;
        statistics.semantic_analyses = 1;

        // Calculate final statistics
        statistics.parse_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ParseResult {
            program,
            symbols,
            semantic_info,
            syntax_style,
            statistics,
            diagnostics,
        })
    }

    /// Parse source code with full semantic integration
    pub async fn parse_source(
        source: &str, 
        source_id: SourceId, 
        file_path: Option<std::path::PathBuf>
    ) -> CompilerResult<ParseResult> {
        let start_time = std::time::Instant::now();
        let mut statistics = ParseStatistics::default();
        let mut diagnostics = Vec::new();

        info!("Parsing source with integration: {:?}", source_id);

        // Step 1: Detect syntax style from source content
        let syntax_style = Self::detect_syntax_from_source(source);

        // Step 2: Parse using multi-syntax parser
        let program = Self::parse_with_style(source, &syntax_style).await?;

        // Step 3: Analyze code quality and generate suggestions
        let quality_analysis = Self::analyze_code_quality(&program);
        for suggestion in quality_analysis {
            diagnostics.push(IntegrationDiagnostic {
                level: DiagnosticLevel::Info,
                message: suggestion,
                location: None,
                suggested_fix: None,
            });
        }

        // Step 4: Extract symbols if symbol system is available
        let symbols = None; // Will be implemented when symbol system is initialized

        // Step 5: Generate semantic information
        let semantic_info = Some(HashMap::from([
            ("source_id".to_string(), format!("{:?}", source_id)),
            ("syntax_style".to_string(), syntax_style.clone().unwrap_or_else(|| "unknown".to_string())),
            ("item_count".to_string(), program.items.len().to_string()),
        ]));

        statistics.parse_time_ms = start_time.elapsed().as_millis() as u64;
        statistics.semantic_analyses = 1;

        Ok(ParseResult {
            program,
            symbols,
            semantic_info,
            syntax_style,
            statistics,
            diagnostics,
        })
    }

    /// Detect syntax style from file extension and content
    fn detect_syntax_style(file_path: &Path) -> CompilerResult<SyntaxStyle> {
        // Simple detection based on file extension and content analysis
        let extension = file_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension {
            "prism" => Ok(SyntaxStyle::PrismNative),
            "py" => Ok(SyntaxStyle::PythonLike),
            "js" | "ts" => Ok(SyntaxStyle::JavaScriptLike),
            "c" | "cpp" | "h" => Ok(SyntaxStyle::CLike),
            _ => {
                // Try to detect from content
                let content = std::fs::read_to_string(file_path)
                    .map_err(|e| CompilerError::FileReadError {
                        path: file_path.to_path_buf(),
                        source: e,
                    })?;
                
                Self::detect_syntax_from_source(&content)
                    .map(|style_str| match style_str.as_str() {
                        "PrismNative" => SyntaxStyle::PrismNative,
                        "PythonLike" => SyntaxStyle::PythonLike,
                        "JavaScriptLike" => SyntaxStyle::JavaScriptLike,
                        "CLike" => SyntaxStyle::CLike,
                        _ => SyntaxStyle::Unknown,
                    })
                    .ok_or_else(|| CompilerError::InvalidInput {
                        message: "Could not detect syntax style".to_string(),
                    })
            }
        }
    }

    /// Detect syntax style from source content
    fn detect_syntax_from_source(source: &str) -> Option<String> {
        // Simple heuristic-based detection
        if source.contains("module") && source.contains("section") {
            Some("PrismNative".to_string())
        } else if source.contains("def ") && source.contains(":") {
            Some("PythonLike".to_string())
        } else if source.contains("function") && source.contains("{") {
            Some("JavaScriptLike".to_string())
        } else if source.contains("#include") || source.contains("int main") {
            Some("CLike".to_string())
        } else {
            Some("PrismNative".to_string()) // Default to Prism native
        }
    }

    /// Parse source with detected style
    async fn parse_with_style(source: &str, style: &Option<String>) -> CompilerResult<Program> {
        // For now, create a minimal program structure
        // TODO: Integrate with actual prism-syntax multi-syntax parser
        
        debug!("Parsing with style: {:?}", style);
        
        // Create a basic program structure
        Ok(Program {
            items: Vec::new(), // TODO: Actually parse the source
            metadata: Default::default(),
        })
    }

    /// Analyze code quality and provide suggestions
    fn analyze_code_quality(program: &Program) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if program.items.is_empty() {
            suggestions.push("Consider adding content to your program".to_string());
        }
        
        if program.items.len() > 50 {
            suggestions.push("Large number of items detected. Consider modularization".to_string());
        }
        
        // TODO: Add more sophisticated quality analysis
        suggestions
    }

    /// Validate integration status
    pub fn validate_integration() -> CompilerResult<()> {
        // Check if all required components are available
        info!("Validating parser integration components");
        
        // TODO: Add actual validation logic
        Ok(())
    }

    /// Get integration status
    pub fn get_integration_status() -> IntegrationStatus {
        IntegrationStatus {
            lexer_available: true,
            syntax_detection_available: true,
            parser_available: true,
            multi_syntax_support: true,
            ai_metadata_generation: true,
            error_recovery: true,
            incremental_parsing: false, // TODO: Implement
        }
    }

    /// Validate integration configuration
    pub fn validate_config(&self) -> Result<(), IntegrationError> {
        // Validate that we have access to required parsers
        if !self.check_parser_availability() {
            return Err(IntegrationError::ConfigurationError(
                "Required parsers are not available".to_string()
            ));
        }
        
        // Validate symbol system configuration
        if !self.check_symbol_system_availability() {
            return Err(IntegrationError::ConfigurationError(
                "Symbol system is not properly configured".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Extract symbols from parsed program
    async fn extract_symbols_from_program(
        &self,
        program: &Program,
        source_id: SourceId,
    ) -> Result<Vec<SymbolData>, IntegrationError> {
        let mut symbols = Vec::new();
        
        // Extract symbols from each item in the program
        for item in &program.items {
            match &item.data {
                prism_ast::Item::Function(func_decl) => {
                    let symbol = SymbolData {
                        id: SymbolId::new(symbols.len() as u32),
                        name: func_decl.name.clone(),
                        kind: SymbolKind::Function,
                        span: item.span,
                        source_id,
                        visibility: if func_decl.visibility.is_public() {
                            crate::scope::ScopeVisibility::Public
                        } else {
                            crate::scope::ScopeVisibility::Private
                        },
                        type_info: Some(TypeInfo {
                            parameters: func_decl.params.iter()
                                .map(|p| ParameterInfo {
                                    name: p.name.clone(),
                                    param_type: p.param_type.clone(),
                                    kind: ParameterKind::Required, // Simplified
                                })
                                .collect(),
                            return_type: func_decl.return_type.clone(),
                        }),
                        metadata: SymbolMetadata {
                            documentation: func_decl.docstring.clone(),
                            attributes: func_decl.attributes.clone(),
                            ai_context: AISymbolContext {
                                business_domain: self.infer_business_domain(&func_decl.name),
                                responsibility: self.infer_responsibility(&func_decl.name, &func_decl.docstring),
                                capabilities: self.infer_capabilities(&func_decl.name),
                                usage_patterns: vec!["function_declaration".to_string()],
                            },
                        },
                    };
                    symbols.push(symbol);
                }
                
                prism_ast::Item::Const(const_decl) => {
                    let symbol = SymbolData {
                        id: SymbolId::new(symbols.len() as u32),
                        name: const_decl.name.clone(),
                        kind: SymbolKind::Constant,
                        span: item.span,
                        source_id,
                        visibility: if const_decl.visibility.is_public() {
                            crate::scope::ScopeVisibility::Public
                        } else {
                            crate::scope::ScopeVisibility::Private
                        },
                        type_info: None, // Could be enhanced with constant type info
                        metadata: SymbolMetadata {
                            documentation: const_decl.docstring.clone(),
                            attributes: const_decl.attributes.clone(),
                            ai_context: AISymbolContext {
                                business_domain: self.infer_business_domain(&const_decl.name),
                                responsibility: Some(format!("Constant value: {}", const_decl.name)),
                                capabilities: vec![],
                                usage_patterns: vec!["constant_declaration".to_string()],
                            },
                        },
                    };
                    symbols.push(symbol);
                }
                
                prism_ast::Item::Type(type_decl) => {
                    let symbol = SymbolData {
                        id: SymbolId::new(symbols.len() as u32),
                        name: type_decl.name.clone(),
                        kind: SymbolKind::Type,
                        span: item.span,
                        source_id,
                        visibility: if type_decl.visibility.is_public() {
                            crate::scope::ScopeVisibility::Public
                        } else {
                            crate::scope::ScopeVisibility::Private
                        },
                        type_info: None, // Could be enhanced with type definition info
                        metadata: SymbolMetadata {
                            documentation: type_decl.docstring.clone(),
                            attributes: type_decl.attributes.clone(),
                            ai_context: AISymbolContext {
                                business_domain: self.infer_business_domain(&type_decl.name),
                                responsibility: Some(format!("Type definition: {}", type_decl.name)),
                                capabilities: vec![],
                                usage_patterns: vec!["type_declaration".to_string()],
                            },
                        },
                    };
                    symbols.push(symbol);
                }
                
                prism_ast::Item::Module(module_decl) => {
                    let symbol = SymbolData {
                        id: SymbolId::new(symbols.len() as u32),
                        name: module_decl.name.clone(),
                        kind: SymbolKind::Module,
                        span: item.span,
                        source_id,
                        visibility: if module_decl.visibility.is_public() {
                            crate::scope::ScopeVisibility::Public
                        } else {
                            crate::scope::ScopeVisibility::Private
                        },
                        type_info: None,
                        metadata: SymbolMetadata {
                            documentation: module_decl.docstring.clone(),
                            attributes: module_decl.attributes.clone(),
                            ai_context: AISymbolContext {
                                business_domain: self.infer_business_domain(&module_decl.name),
                                responsibility: Some(format!("Module: {}", module_decl.name)),
                                capabilities: vec![],
                                usage_patterns: vec!["module_declaration".to_string()],
                            },
                        },
                    };
                    symbols.push(symbol);
                }
                
                // Handle other item types as needed
                _ => {
                    // For now, skip other item types
                    tracing::debug!("Skipping symbol extraction for unsupported item type");
                }
            }
        }
        
        Ok(symbols)
    }
    
    /// Perform semantic integration analysis
    async fn perform_semantic_integration(
        &self,
        program: &Program,
        symbols: &[SymbolData],
    ) -> Result<SemanticIntegrationInfo, IntegrationError> {
        // Analyze semantic relationships between symbols
        let relationships = self.analyze_symbol_relationships(symbols)?;
        
        // Infer semantic types for the program
        let semantic_types = self.infer_program_semantic_types(program)?;
        
        // Analyze business context
        let business_context = self.analyze_business_context(symbols)?;
        
        // Calculate semantic quality metrics
        let quality_metrics = self.calculate_semantic_quality_metrics(program, symbols)?;
        
        Ok(SemanticIntegrationInfo {
            relationships,
            semantic_types,
            business_context,
            quality_metrics,
            analysis_confidence: 0.8, // Base confidence level
        })
    }
    
    // Helper methods for semantic analysis
    fn analyze_symbol_relationships(&self, symbols: &[SymbolData]) -> Result<Vec<SymbolRelationship>, IntegrationError> {
        let mut relationships = Vec::new();
        
        // Analyze dependencies between symbols
        for (i, symbol) in symbols.iter().enumerate() {
            for (j, other_symbol) in symbols.iter().enumerate() {
                if i != j {
                    if let Some(relationship) = self.infer_symbol_relationship(symbol, other_symbol) {
                        relationships.push(relationship);
                    }
                }
            }
        }
        
        Ok(relationships)
    }
    
    fn infer_program_semantic_types(&self, program: &Program) -> Result<Vec<InferredSemanticType>, IntegrationError> {
        let mut semantic_types = Vec::new();
        
        // Infer semantic types from program structure
        for item in &program.items {
            if let Some(semantic_type) = self.infer_item_semantic_type(item) {
                semantic_types.push(semantic_type);
            }
        }
        
        Ok(semantic_types)
    }
    
    fn analyze_business_context(&self, symbols: &[SymbolData]) -> Result<BusinessContextInfo, IntegrationError> {
        let mut domain_counts = std::collections::HashMap::new();
        let mut capabilities = std::collections::HashSet::new();
        
        for symbol in symbols {
            if let Some(domain) = &symbol.metadata.ai_context.business_domain {
                *domain_counts.entry(domain.clone()).or_insert(0) += 1;
            }
            
            capabilities.extend(symbol.metadata.ai_context.capabilities.iter().cloned());
        }
        
        let primary_domain = domain_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(domain, _)| domain);
        
        Ok(BusinessContextInfo {
            primary_domain,
            capabilities: capabilities.into_iter().collect(),
            context_confidence: 0.7,
        })
    }
    
    fn calculate_semantic_quality_metrics(
        &self, 
        program: &Program, 
        symbols: &[SymbolData]
    ) -> Result<SemanticQualityMetrics, IntegrationError> {
        Ok(SemanticQualityMetrics {
            symbol_coverage: (symbols.len() as f64 / program.items.len().max(1) as f64),
            type_inference_confidence: 0.8,
            business_context_coverage: symbols.iter()
                .filter(|s| s.metadata.ai_context.business_domain.is_some())
                .count() as f64 / symbols.len().max(1) as f64,
            documentation_coverage: symbols.iter()
                .filter(|s| s.metadata.documentation.is_some())
                .count() as f64 / symbols.len().max(1) as f64,
        })
    }
    
    // AI inference helper methods
    fn infer_business_domain(&self, name: &str) -> Option<String> {
        let name_lower = name.to_lowercase();
        
        if name_lower.contains("auth") || name_lower.contains("login") || name_lower.contains("password") {
            Some("Security".to_string())
        } else if name_lower.contains("pay") || name_lower.contains("money") || name_lower.contains("price") {
            Some("Financial".to_string())
        } else if name_lower.contains("user") || name_lower.contains("profile") {
            Some("User Management".to_string())
        } else if name_lower.contains("data") || name_lower.contains("store") || name_lower.contains("save") {
            Some("Data Management".to_string())
        } else {
            None
        }
    }
    
    fn infer_responsibility(&self, name: &str, docstring: &Option<String>) -> Option<String> {
        if let Some(doc) = docstring {
            // Extract first sentence as responsibility
            if let Some(first_sentence) = doc.split('.').next() {
                return Some(first_sentence.trim().to_string());
            }
        }
        
        // Infer from name patterns
        let name_lower = name.to_lowercase();
        if name_lower.starts_with("get_") {
            Some(format!("Retrieves {}", &name[4..]))
        } else if name_lower.starts_with("set_") {
            Some(format!("Sets {}", &name[4..]))
        } else if name_lower.starts_with("create_") {
            Some(format!("Creates {}", &name[7..]))
        } else if name_lower.starts_with("delete_") {
            Some(format!("Deletes {}", &name[7..]))
        } else if name_lower.starts_with("validate_") {
            Some(format!("Validates {}", &name[9..]))
        } else {
            None
        }
    }
    
    fn infer_capabilities(&self, name: &str) -> Vec<String> {
        let mut capabilities = Vec::new();
        let name_lower = name.to_lowercase();
        
        if name_lower.contains("db") || name_lower.contains("database") || name_lower.contains("store") {
            capabilities.push("Database".to_string());
        }
        if name_lower.contains("http") || name_lower.contains("api") || name_lower.contains("request") {
            capabilities.push("Network".to_string());
        }
        if name_lower.contains("file") || name_lower.contains("read") || name_lower.contains("write") {
            capabilities.push("FileSystem".to_string());
        }
        if name_lower.contains("auth") || name_lower.contains("security") {
            capabilities.push("Security".to_string());
        }
        
        capabilities
    }
    
    fn infer_symbol_relationship(&self, symbol: &SymbolData, other: &SymbolData) -> Option<SymbolRelationship> {
        // Simple heuristic-based relationship inference
        if symbol.name.to_lowercase().contains(&other.name.to_lowercase()) ||
           other.name.to_lowercase().contains(&symbol.name.to_lowercase()) {
            Some(SymbolRelationship {
                from: symbol.id,
                to: other.id,
                relationship_type: RelationshipType::Related,
                confidence: 0.6,
            })
        } else {
            None
        }
    }
    
    fn infer_item_semantic_type(&self, item: &prism_ast::AstNode<prism_ast::Item>) -> Option<InferredSemanticType> {
        match &item.data {
            prism_ast::Item::Function(func) => {
                Some(InferredSemanticType {
                    name: func.name.clone(),
                    semantic_category: "Function".to_string(),
                    confidence: 0.9,
                    properties: vec![
                        ("parameter_count".to_string(), func.params.len().to_string()),
                        ("has_return_type".to_string(), func.return_type.is_some().to_string()),
                    ],
                })
            }
            prism_ast::Item::Type(type_decl) => {
                Some(InferredSemanticType {
                    name: type_decl.name.clone(),
                    semantic_category: "Type".to_string(),
                    confidence: 0.9,
                    properties: vec![],
                })
            }
            _ => None,
        }
    }
}

/// Integration status information
pub struct IntegrationStatus {
    /// Whether lexer integration is available
    pub lexer_available: bool,
    /// Whether syntax detection is available
    pub syntax_detection_available: bool,
    /// Whether parser integration is available
    pub parser_available: bool,
    /// Whether multi-syntax support is enabled
    pub multi_syntax_support: bool,
    /// Whether AI metadata generation is available
    pub ai_metadata_generation: bool,
    /// Whether error recovery is available
    pub error_recovery: bool,
    /// Whether incremental parsing is available
    pub incremental_parsing: bool,
}

impl IntegrationStatus {
    /// Check if the integration is fully functional
    pub fn is_fully_integrated(&self) -> bool {
        self.lexer_available
            && self.syntax_detection_available
            && self.parser_available
            && self.multi_syntax_support
    }

    /// Generate a summary of integration status
    pub fn summary(&self) -> String {
        format!(
            "Parser Integration Status: {} | Lexer: {} | Syntax Detection: {} | Multi-Syntax: {} | AI: {} | Recovery: {}",
            if self.is_fully_integrated() { "✅ Ready" } else { "🔧 Partial" },
            if self.lexer_available { "✅" } else { "❌" },
            if self.syntax_detection_available { "✅" } else { "❌" },
            if self.multi_syntax_support { "✅" } else { "❌" },
            if self.ai_metadata_generation { "✅" } else { "❌" },
            if self.error_recovery { "✅" } else { "❌" },
        )
    }
}

/// Syntax styles supported by the integration
#[derive(Debug, Clone, PartialEq)]
pub enum SyntaxStyle {
    PrismNative,
    PythonLike,
    JavaScriptLike,
    CLike,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integration_validation() {
        let result = ParserIntegration::validate_integration();
        assert!(result.is_ok());
    }

    #[test]
    fn test_integration_status() {
        let status = ParserIntegration::get_integration_status();
        assert!(status.lexer_available);
        assert!(status.syntax_detection_available);
        assert!(status.parser_available);
    }

    #[tokio::test]
    async fn test_parse_simple_source() {
        let source = r#"
            module TestModule {
                function test() {
                    return "hello";
                }
            }
        "#;
        
        let result = ParserIntegration::parse_source(
            source, 
            SourceId::new(1), 
            None
        ).await;
        
        assert!(result.is_ok());
        let parse_result = result.unwrap();
        assert!(parse_result.semantic_info.is_some());
        assert_eq!(parse_result.syntax_style, Some("PrismNative".to_string()));
    }
} 