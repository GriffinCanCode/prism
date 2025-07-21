//! AI Metadata Extraction - AST to PIR AI Intelligence
//!
//! This module implements AI metadata extraction from AST nodes, analyzing
//! code patterns, documentation, and structure for AI-relevant information.
//!
//! **Conceptual Responsibility**: AI metadata extraction and context analysis
//! **What it does**: Extracts AI contexts, function hints, type information, and learning patterns
//! **What it doesn't do**: AST parsing, PIR construction, semantic analysis (focuses on AI intelligence)

use crate::{PIRResult, PIRError};
use crate::ai_integration::AIMetadata;
use crate::semantic::{PIRTypeAIContext, PIRFunctionAIContext, PIRModuleAIContext};
use prism_ast::{Program, AstNode, Item, ModuleDecl, FunctionDecl, TypeDecl};
use std::collections::{HashMap, HashSet};

/// AI metadata extractor for PIR construction
pub struct AIMetadataExtractor {
    /// Configuration for AI extraction
    config: AIExtractionConfig,
    /// Context analyzers
    analyzers: AIContextAnalyzers,
    /// Pattern matchers for AI-relevant code
    pattern_matchers: AIPatternMatchers,
}

/// Configuration for AI metadata extraction
#[derive(Debug, Clone)]
pub struct AIExtractionConfig {
    /// Enable function context extraction
    pub enable_function_contexts: bool,
    /// Enable type context extraction
    pub enable_type_contexts: bool,
    /// Enable module context extraction
    pub enable_module_contexts: bool,
    /// Enable documentation analysis
    pub enable_documentation_analysis: bool,
    /// Enable code pattern recognition
    pub enable_pattern_recognition: bool,
    /// Enable learning hint extraction
    pub enable_learning_hints: bool,
    /// Minimum confidence threshold for AI insights
    pub min_confidence_threshold: f64,
}

/// AI context analyzers for different code constructs
pub struct AIContextAnalyzers {
    /// Function context analyzer
    function_analyzer: FunctionContextAnalyzer,
    /// Type context analyzer
    type_analyzer: TypeContextAnalyzer,
    /// Module context analyzer
    module_analyzer: ModuleContextAnalyzer,
    /// Documentation analyzer
    doc_analyzer: DocumentationAnalyzer,
}

/// Function context analyzer
pub struct FunctionContextAnalyzer {
    /// Intent pattern matchers
    intent_patterns: HashMap<String, Vec<String>>,
    /// Complexity analyzers
    complexity_patterns: Vec<ComplexityPattern>,
}

/// Type context analyzer
pub struct TypeContextAnalyzer {
    /// Domain type patterns
    domain_patterns: HashMap<String, Vec<String>>,
    /// Usage pattern matchers
    usage_patterns: HashMap<String, Vec<String>>,
}

/// Module context analyzer
pub struct ModuleContextAnalyzer {
    /// Architecture pattern matchers
    architecture_patterns: HashMap<String, Vec<String>>,
    /// Responsibility analyzers
    responsibility_patterns: HashMap<String, Vec<String>>,
}

/// Documentation analyzer
pub struct DocumentationAnalyzer {
    /// AI hint extractors
    hint_extractors: Vec<HintExtractor>,
    /// Example extractors
    example_extractors: Vec<ExampleExtractor>,
}

/// AI pattern matchers
pub struct AIPatternMatchers {
    /// Common code patterns
    code_patterns: HashMap<String, CodePattern>,
    /// Anti-pattern detectors
    anti_patterns: HashMap<String, AntiPattern>,
}

/// Complexity analysis pattern
#[derive(Debug, Clone)]
pub struct ComplexityPattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Complexity indicators
    pub indicators: Vec<String>,
}

/// Hint extractor for documentation
#[derive(Debug, Clone)]
pub struct HintExtractor {
    /// Extractor name
    pub name: String,
    /// Pattern to match
    pub pattern: String,
    /// Extraction confidence
    pub confidence: f64,
}

/// Example extractor for documentation
#[derive(Debug, Clone)]
pub struct ExampleExtractor {
    /// Extractor name
    pub name: String,
    /// Example pattern
    pub pattern: String,
    /// Context type
    pub context_type: String,
}

/// Code pattern for AI analysis
#[derive(Debug, Clone)]
pub struct CodePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// AI insights
    pub insights: Vec<String>,
    /// Learning opportunities
    pub learning_opportunities: Vec<String>,
}

/// Anti-pattern for AI analysis
#[derive(Debug, Clone)]
pub struct AntiPattern {
    /// Anti-pattern name
    pub name: String,
    /// Description
    pub description: String,
    /// Common mistakes
    pub common_mistakes: Vec<String>,
    /// Improvement suggestions
    pub suggestions: Vec<String>,
}

/// AI extraction result
#[derive(Debug, Clone)]
pub struct AIExtractionResult {
    /// Extracted AI metadata
    pub metadata: AIMetadata,
    /// Function-specific AI contexts
    pub function_contexts: HashMap<String, PIRFunctionAIContext>,
    /// Type-specific AI contexts
    pub type_contexts: HashMap<String, PIRTypeAIContext>,
    /// Module-specific AI contexts
    pub module_contexts: HashMap<String, PIRModuleAIContext>,
    /// Extraction diagnostics
    pub diagnostics: Vec<AIExtractionDiagnostic>,
    /// Extraction confidence scores
    pub confidence_scores: HashMap<String, f64>,
}

/// Diagnostic from AI extraction
#[derive(Debug, Clone)]
pub struct AIExtractionDiagnostic {
    /// Diagnostic level
    pub level: AIExtractionLevel,
    /// Diagnostic message
    pub message: String,
    /// Source location
    pub location: Option<prism_common::span::Span>,
    /// Confidence score
    pub confidence: Option<f64>,
    /// AI insight category
    pub category: String,
}

/// AI extraction diagnostic levels
#[derive(Debug, Clone)]
pub enum AIExtractionLevel {
    /// Information
    Info,
    /// Insight discovered
    Insight,
    /// Learning opportunity identified
    Learning,
    /// Potential improvement
    Improvement,
}

impl AIMetadataExtractor {
    /// Create a new AI metadata extractor
    pub fn new(config: AIExtractionConfig) -> Self {
        Self {
            analyzers: AIContextAnalyzers::new(),
            pattern_matchers: AIPatternMatchers::new(),
            config,
        }
    }

    /// Extract AI metadata from program
    pub fn extract_ai_metadata(&mut self, program: &Program) -> PIRResult<AIExtractionResult> {
        let mut function_contexts = HashMap::new();
        let mut type_contexts = HashMap::new();
        let mut module_contexts = HashMap::new();
        let mut diagnostics = Vec::new();
        let mut confidence_scores = HashMap::new();

        // Process modules
        for item in &program.items {
            if let Item::Module(module_decl) = &item.kind {
                self.process_module_ai_context(
                    module_decl,
                    &mut function_contexts,
                    &mut type_contexts,
                    &mut module_contexts,
                    &mut diagnostics,
                    &mut confidence_scores,
                )?;
            }
        }

        // Process global items
        let global_items: Vec<_> = program.items.iter()
            .filter(|item| !matches!(item.kind, Item::Module(_)))
            .collect();

        if !global_items.is_empty() {
            self.process_global_ai_context(
                &global_items,
                &mut function_contexts,
                &mut type_contexts,
                &mut diagnostics,
                &mut confidence_scores,
            )?;
        }

        // Build comprehensive AI metadata
        let metadata = self.build_ai_metadata(&function_contexts, &type_contexts, &module_contexts)?;

        Ok(AIExtractionResult {
            metadata,
            function_contexts,
            type_contexts,
            module_contexts,
            diagnostics,
            confidence_scores,
        })
    }

    /// Process AI context for a module
    fn process_module_ai_context(
        &mut self,
        module_decl: &ModuleDecl,
        function_contexts: &mut HashMap<String, PIRFunctionAIContext>,
        type_contexts: &mut HashMap<String, PIRTypeAIContext>,
        module_contexts: &mut HashMap<String, PIRModuleAIContext>,
        diagnostics: &mut Vec<AIExtractionDiagnostic>,
        confidence_scores: &mut HashMap<String, f64>,
    ) -> PIRResult<()> {
        // Extract module-level AI context
        if self.config.enable_module_contexts {
            let module_context = self.analyzers.module_analyzer.analyze_module(module_decl)?;
            let confidence = self.calculate_module_confidence(&module_context);
            
            module_contexts.insert(module_decl.name.clone(), module_context);
            confidence_scores.insert(format!("module:{}", module_decl.name), confidence);
        }

        // Process module items
        for item in &module_decl.items {
            match &item.kind {
                Item::Function(func_decl) => {
                    if self.config.enable_function_contexts {
                        let context = self.extract_function_ai_context(func_decl, diagnostics)?;
                        let confidence = self.calculate_function_confidence(&context);
                        let full_name = format!("{}::{}", module_decl.name, func_decl.name);
                        
                        function_contexts.insert(full_name.clone(), context);
                        confidence_scores.insert(format!("function:{}", full_name), confidence);
                    }
                }
                Item::Type(type_decl) => {
                    if self.config.enable_type_contexts {
                        let context = self.extract_type_ai_context(type_decl, diagnostics)?;
                        let confidence = self.calculate_type_confidence(&context);
                        let full_name = format!("{}::{}", module_decl.name, type_decl.name);
                        
                        type_contexts.insert(full_name.clone(), context);
                        confidence_scores.insert(format!("type:{}", full_name), confidence);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Process AI context for global items
    fn process_global_ai_context(
        &mut self,
        items: &[&AstNode<Item>],
        function_contexts: &mut HashMap<String, PIRFunctionAIContext>,
        type_contexts: &mut HashMap<String, PIRTypeAIContext>,
        diagnostics: &mut Vec<AIExtractionDiagnostic>,
        confidence_scores: &mut HashMap<String, f64>,
    ) -> PIRResult<()> {
        for item in items {
            match &item.kind {
                Item::Function(func_decl) => {
                    if self.config.enable_function_contexts {
                        let context = self.extract_function_ai_context(func_decl, diagnostics)?;
                        let confidence = self.calculate_function_confidence(&context);
                        
                        function_contexts.insert(func_decl.name.to_string(), context);
                        confidence_scores.insert(format!("function:{}", func_decl.name), confidence);
                    }
                }
                Item::Type(type_decl) => {
                    if self.config.enable_type_contexts {
                        let context = self.extract_type_ai_context(type_decl, diagnostics)?;
                        let confidence = self.calculate_type_confidence(&context);
                        
                        type_contexts.insert(type_decl.name.to_string(), context);
                        confidence_scores.insert(format!("type:{}", type_decl.name), confidence);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Extract AI context from function
    pub fn extract_function_ai_context(
        &self,
        func_decl: &FunctionDecl,
        diagnostics: &mut Vec<AIExtractionDiagnostic>,
    ) -> PIRResult<PIRFunctionAIContext> {
        let context = self.analyzers.function_analyzer.analyze_function(func_decl)?;
        
        // Add diagnostics for interesting patterns
        if !context.common_patterns.is_empty() {
            diagnostics.push(AIExtractionDiagnostic {
                level: AIExtractionLevel::Insight,
                message: format!("Function '{}' exhibits {} common patterns", 
                               func_decl.name, context.common_patterns.len()),
                location: None, // TODO: Get function span
                confidence: Some(0.8),
                category: "pattern_recognition".to_string(),
            });
        }

        Ok(context)
    }

    /// Extract AI context from type
    pub fn extract_type_ai_context(
        &self,
        type_decl: &TypeDecl,
        diagnostics: &mut Vec<AIExtractionDiagnostic>,
    ) -> PIRResult<PIRTypeAIContext> {
        let context = self.analyzers.type_analyzer.analyze_type(type_decl)?;
        
        // Add diagnostics for type insights
        if context.intent.is_some() {
            diagnostics.push(AIExtractionDiagnostic {
                level: AIExtractionLevel::Insight,
                message: format!("Type '{}' intent successfully inferred", type_decl.name),
                location: None, // TODO: Get type span
                confidence: Some(0.7),
                category: "intent_analysis".to_string(),
            });
        }

        Ok(context)
    }

    /// Build comprehensive AI metadata
    fn build_ai_metadata(
        &self,
        function_contexts: &HashMap<String, PIRFunctionAIContext>,
        type_contexts: &HashMap<String, PIRTypeAIContext>,
        module_contexts: &HashMap<String, PIRModuleAIContext>,
    ) -> PIRResult<AIMetadata> {
        let mut metadata = AIMetadata::default();

        // Set function contexts
        metadata.function_contexts = function_contexts.clone();

        // Set type contexts
        metadata.type_contexts = type_contexts.clone();

        // Set module context (use first module as primary, or create global)
        metadata.module_context = module_contexts.values().next().cloned();

        // Extract global patterns and insights
        metadata.global_patterns = self.extract_global_patterns(function_contexts, type_contexts)?;
        metadata.learning_opportunities = self.extract_learning_opportunities(function_contexts, type_contexts)?;

        Ok(metadata)
    }

    /// Extract global patterns from contexts
    fn extract_global_patterns(
        &self,
        function_contexts: &HashMap<String, PIRFunctionAIContext>,
        type_contexts: &HashMap<String, PIRTypeAIContext>,
    ) -> PIRResult<Vec<String>> {
        let mut patterns = HashSet::new();

        // Collect patterns from function contexts
        for context in function_contexts.values() {
            patterns.extend(context.common_patterns.clone());
        }

        // Collect patterns from type contexts
        for context in type_contexts.values() {
            if let Some(ref examples) = context.examples {
                patterns.extend(examples.iter().cloned());
            }
        }

        Ok(patterns.into_iter().collect())
    }

    /// Extract learning opportunities from contexts
    fn extract_learning_opportunities(
        &self,
        function_contexts: &HashMap<String, PIRFunctionAIContext>,
        type_contexts: &HashMap<String, PIRTypeAIContext>,
    ) -> PIRResult<Vec<String>> {
        let mut opportunities = HashSet::new();

        // Collect opportunities from function contexts
        for context in function_contexts.values() {
            opportunities.extend(context.learning_hints.clone());
        }

        // Collect opportunities from type contexts
        for context in type_contexts.values() {
            opportunities.extend(context.best_practices.clone());
        }

        Ok(opportunities.into_iter().collect())
    }

    /// Calculate confidence for function context
    fn calculate_function_confidence(&self, context: &PIRFunctionAIContext) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        if context.intent.is_some() {
            score += 0.8;
            factors += 1;
        }

        if !context.common_patterns.is_empty() {
            score += 0.7;
            factors += 1;
        }

        if !context.learning_hints.is_empty() {
            score += 0.6;
            factors += 1;
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.3 // Default low confidence
        }
    }

    /// Calculate confidence for type context
    fn calculate_type_confidence(&self, context: &PIRTypeAIContext) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        if context.intent.is_some() {
            score += 0.8;
            factors += 1;
        }

        if let Some(ref examples) = context.examples {
            if !examples.is_empty() {
                score += 0.7;
                factors += 1;
            }
        }

        if !context.best_practices.is_empty() {
            score += 0.6;
            factors += 1;
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.3
        }
    }

    /// Calculate confidence for module context
    fn calculate_module_confidence(&self, context: &PIRModuleAIContext) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        if context.primary_purpose.is_some() {
            score += 0.9;
            factors += 1;
        }

        if !context.architectural_patterns.is_empty() {
            score += 0.8;
            factors += 1;
        }

        if !context.usage_examples.is_empty() {
            score += 0.6;
            factors += 1;
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.4
        }
    }
}

impl AIContextAnalyzers {
    fn new() -> Self {
        Self {
            function_analyzer: FunctionContextAnalyzer::new(),
            type_analyzer: TypeContextAnalyzer::new(),
            module_analyzer: ModuleContextAnalyzer::new(),
            doc_analyzer: DocumentationAnalyzer::new(),
        }
    }
}

impl FunctionContextAnalyzer {
    fn new() -> Self {
        let mut intent_patterns = HashMap::new();
        
        // Define intent patterns for functions
        intent_patterns.insert("creation".to_string(), 
            vec!["create", "new", "make", "build", "construct"].iter().map(|s| s.to_string()).collect());
        intent_patterns.insert("retrieval".to_string(),
            vec!["get", "find", "search", "fetch", "retrieve"].iter().map(|s| s.to_string()).collect());
        intent_patterns.insert("modification".to_string(),
            vec!["update", "modify", "change", "edit", "alter"].iter().map(|s| s.to_string()).collect());
        intent_patterns.insert("validation".to_string(),
            vec!["validate", "check", "verify", "ensure", "assert"].iter().map(|s| s.to_string()).collect());

        let complexity_patterns = vec![
            ComplexityPattern {
                name: "high_branching".to_string(),
                description: "Function with high branching complexity".to_string(),
                indicators: vec!["multiple_if".to_string(), "nested_conditions".to_string()],
            },
            ComplexityPattern {
                name: "data_transformation".to_string(),
                description: "Function performing data transformation".to_string(),
                indicators: vec!["map".to_string(), "transform".to_string(), "convert".to_string()],
            },
        ];

        Self {
            intent_patterns,
            complexity_patterns,
        }
    }

    fn analyze_function(&self, func_decl: &FunctionDecl) -> PIRResult<PIRFunctionAIContext> {
        let func_name = func_decl.name.to_lowercase();
        
        // Infer intent from function name
        let intent = self.infer_function_intent(&func_name);
        
        // Extract common patterns
        let common_patterns = self.extract_function_patterns(&func_name);
        
        // Generate learning hints
        let learning_hints = self.generate_learning_hints(&func_name, &intent);

        Ok(PIRFunctionAIContext {
            intent,
            common_patterns,
            learning_hints,
            complexity_analysis: None, // TODO: Implement complexity analysis
            usage_examples: Vec::new(), // TODO: Extract from documentation
        })
    }

    fn infer_function_intent(&self, func_name: &str) -> Option<String> {
        for (intent, patterns) in &self.intent_patterns {
            for pattern in patterns {
                if func_name.contains(pattern) {
                    return Some(intent.clone());
                }
            }
        }
        None
    }

    fn extract_function_patterns(&self, func_name: &str) -> Vec<String> {
        let mut patterns = Vec::new();
        
        if func_name.starts_with("is_") || func_name.starts_with("has_") {
            patterns.push("predicate_function".to_string());
        }
        
        if func_name.ends_with("_async") || func_name.contains("async") {
            patterns.push("asynchronous_operation".to_string());
        }
        
        if func_name.contains("_mut") || func_name.ends_with("_mut") {
            patterns.push("mutable_operation".to_string());
        }

        patterns
    }

    fn generate_learning_hints(&self, func_name: &str, intent: &Option<String>) -> Vec<String> {
        let mut hints = Vec::new();
        
        if let Some(intent_str) = intent {
            hints.push(format!("Function performs {} operation", intent_str));
        }
        
        if func_name.len() > 20 {
            hints.push("Consider shorter, more descriptive function names".to_string());
        }
        
        if func_name.contains("_") {
            hints.push("Uses snake_case naming convention".to_string());
        }

        hints
    }
}

impl TypeContextAnalyzer {
    fn new() -> Self {
        let mut domain_patterns = HashMap::new();
        let mut usage_patterns = HashMap::new();

        // Define domain patterns
        domain_patterns.insert("identifier".to_string(),
            vec!["id", "identifier", "key", "uuid"].iter().map(|s| s.to_string()).collect());
        domain_patterns.insert("collection".to_string(),
            vec!["list", "vec", "array", "set", "map"].iter().map(|s| s.to_string()).collect());
        domain_patterns.insert("configuration".to_string(),
            vec!["config", "settings", "options", "params"].iter().map(|s| s.to_string()).collect());

        // Define usage patterns
        usage_patterns.insert("data_transfer".to_string(),
            vec!["dto", "request", "response", "message"].iter().map(|s| s.to_string()).collect());
        usage_patterns.insert("state_management".to_string(),
            vec!["state", "context", "store", "cache"].iter().map(|s| s.to_string()).collect());

        Self {
            domain_patterns,
            usage_patterns,
        }
    }

    fn analyze_type(&self, type_decl: &TypeDecl) -> PIRResult<PIRTypeAIContext> {
        let type_name = type_decl.name.to_lowercase();
        
        // Infer intent from type name
        let intent = self.infer_type_intent(&type_name);
        
        // Generate examples based on type patterns
        let examples = self.generate_type_examples(&type_name);
        
        // Extract best practices
        let best_practices = self.extract_best_practices(&type_name);
        
        // Identify common mistakes
        let common_mistakes = self.identify_common_mistakes(&type_name);

        Ok(PIRTypeAIContext {
            intent,
            examples: if examples.is_empty() { None } else { Some(examples) },
            common_mistakes,
            best_practices,
        })
    }

    fn infer_type_intent(&self, type_name: &str) -> Option<String> {
        for (domain, patterns) in &self.domain_patterns {
            for pattern in patterns {
                if type_name.contains(pattern) {
                    return Some(format!("Represents {} in the domain", domain));
                }
            }
        }
        
        for (usage, patterns) in &self.usage_patterns {
            for pattern in patterns {
                if type_name.contains(pattern) {
                    return Some(format!("Used for {}", usage));
                }
            }
        }
        
        None
    }

    fn generate_type_examples(&self, type_name: &str) -> Vec<String> {
        let mut examples = Vec::new();
        
        if type_name.contains("config") {
            examples.push("Configuration object with settings".to_string());
        }
        
        if type_name.contains("id") {
            examples.push("Unique identifier for entity".to_string());
        }
        
        if type_name.contains("request") {
            examples.push("Request object for API call".to_string());
        }

        examples
    }

    fn extract_best_practices(&self, type_name: &str) -> Vec<String> {
        let mut practices = Vec::new();
        
        if type_name.ends_with("id") {
            practices.push("Use strong typing for identifiers".to_string());
        }
        
        if type_name.contains("config") {
            practices.push("Provide sensible defaults for configuration".to_string());
        }
        
        practices.push("Use descriptive field names".to_string());
        practices
    }

    fn identify_common_mistakes(&self, type_name: &str) -> Vec<String> {
        let mut mistakes = Vec::new();
        
        if type_name.len() < 3 {
            mistakes.push("Type name too short, consider more descriptive names".to_string());
        }
        
        if type_name.chars().any(|c| c.is_uppercase()) && type_name.chars().any(|c| c.is_lowercase()) {
            // Mixed case might indicate inconsistent naming
            mistakes.push("Ensure consistent naming convention".to_string());
        }

        mistakes
    }
}

impl ModuleContextAnalyzer {
    fn new() -> Self {
        let mut architecture_patterns = HashMap::new();
        let mut responsibility_patterns = HashMap::new();

        // Define architecture patterns
        architecture_patterns.insert("service_layer".to_string(),
            vec!["service", "handler", "controller"].iter().map(|s| s.to_string()).collect());
        architecture_patterns.insert("data_layer".to_string(),
            vec!["repository", "dao", "model", "entity"].iter().map(|s| s.to_string()).collect());
        architecture_patterns.insert("utility_layer".to_string(),
            vec!["util", "helper", "common", "shared"].iter().map(|s| s.to_string()).collect());

        // Define responsibility patterns
        responsibility_patterns.insert("business_logic".to_string(),
            vec!["business", "domain", "logic", "rules"].iter().map(|s| s.to_string()).collect());
        responsibility_patterns.insert("infrastructure".to_string(),
            vec!["infra", "config", "setup", "init"].iter().map(|s| s.to_string()).collect());

        Self {
            architecture_patterns,
            responsibility_patterns,
        }
    }

    fn analyze_module(&self, module_decl: &ModuleDecl) -> PIRResult<PIRModuleAIContext> {
        let module_name = module_decl.name.to_lowercase();
        
        // Infer primary purpose
        let primary_purpose = self.infer_module_purpose(&module_name);
        
        // Extract architectural patterns
        let architectural_patterns = self.extract_architectural_patterns(&module_name);
        
        // Generate usage examples
        let usage_examples = self.generate_usage_examples(&module_name);
        
        // Extract integration patterns
        let integration_patterns = self.extract_integration_patterns(module_decl);

        Ok(PIRModuleAIContext {
            primary_purpose,
            architectural_patterns,
            usage_examples,
            integration_patterns,
        })
    }

    fn infer_module_purpose(&self, module_name: &str) -> Option<String> {
        for (pattern, keywords) in &self.architecture_patterns {
            for keyword in keywords {
                if module_name.contains(keyword) {
                    return Some(format!("Implements {} pattern", pattern));
                }
            }
        }
        
        for (responsibility, keywords) in &self.responsibility_patterns {
            for keyword in keywords {
                if module_name.contains(keyword) {
                    return Some(format!("Handles {}", responsibility));
                }
            }
        }
        
        None
    }

    fn extract_architectural_patterns(&self, module_name: &str) -> Vec<String> {
        let mut patterns = Vec::new();
        
        for (pattern, keywords) in &self.architecture_patterns {
            for keyword in keywords {
                if module_name.contains(keyword) {
                    patterns.push(pattern.clone());
                    break;
                }
            }
        }
        
        patterns
    }

    fn generate_usage_examples(&self, module_name: &str) -> Vec<String> {
        let mut examples = Vec::new();
        
        if module_name.contains("service") {
            examples.push("Import and use service functions".to_string());
        }
        
        if module_name.contains("util") {
            examples.push("Import utility functions as needed".to_string());
        }
        
        examples
    }

    fn extract_integration_patterns(&self, module_decl: &ModuleDecl) -> Vec<String> {
        let mut patterns = Vec::new();
        
        let function_count = module_decl.items.iter()
            .filter(|item| matches!(item.kind, Item::Function(_)))
            .count();
        
        let type_count = module_decl.items.iter()
            .filter(|item| matches!(item.kind, Item::Type(_)))
            .count();
        
        if function_count > type_count {
            patterns.push("function_heavy_module".to_string());
        } else if type_count > function_count {
            patterns.push("type_heavy_module".to_string());
        } else {
            patterns.push("balanced_module".to_string());
        }
        
        patterns
    }
}

impl DocumentationAnalyzer {
    fn new() -> Self {
        let hint_extractors = vec![
            HintExtractor {
                name: "todo_hints".to_string(),
                pattern: "TODO".to_string(),
                confidence: 0.9,
            },
            HintExtractor {
                name: "fixme_hints".to_string(),
                pattern: "FIXME".to_string(),
                confidence: 0.95,
            },
        ];

        let example_extractors = vec![
            ExampleExtractor {
                name: "code_examples".to_string(),
                pattern: "```".to_string(),
                context_type: "code".to_string(),
            },
        ];

        Self {
            hint_extractors,
            example_extractors,
        }
    }
}

impl AIPatternMatchers {
    fn new() -> Self {
        let mut code_patterns = HashMap::new();
        let mut anti_patterns = HashMap::new();

        // Define common code patterns
        code_patterns.insert("builder_pattern".to_string(), CodePattern {
            name: "Builder Pattern".to_string(),
            description: "Fluent interface for object construction".to_string(),
            insights: vec!["Provides flexible object creation".to_string()],
            learning_opportunities: vec!["Study fluent interface design".to_string()],
        });

        // Define anti-patterns
        anti_patterns.insert("god_function".to_string(), AntiPattern {
            name: "God Function".to_string(),
            description: "Function that does too many things".to_string(),
            common_mistakes: vec!["Single function handling multiple responsibilities".to_string()],
            suggestions: vec!["Break into smaller, focused functions".to_string()],
        });

        Self {
            code_patterns,
            anti_patterns,
        }
    }
}

impl Default for AIExtractionConfig {
    fn default() -> Self {
        Self {
            enable_function_contexts: true,
            enable_type_contexts: true,
            enable_module_contexts: true,
            enable_documentation_analysis: true,
            enable_pattern_recognition: true,
            enable_learning_hints: true,
            min_confidence_threshold: 0.5,
        }
    }
} 