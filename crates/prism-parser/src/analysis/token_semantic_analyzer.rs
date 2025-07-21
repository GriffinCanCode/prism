//! Token-based Semantic Analysis
//!
//! This module embodies the single concept of "Token Stream Semantic Analysis".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: analyzing token streams to extract semantic patterns and summaries.
//!
//! **Conceptual Responsibility**: Analyze token relationships and patterns
//! **What it does**: semantic summaries, token pattern recognition, identifier analysis
//! **What it doesn't do**: tokenization, AST construction, syntax detection

use prism_lexer::{Token, TokenKind};
use prism_common::{span::Span, symbol::Symbol};
use std::collections::HashMap;

/// Semantic summary of analyzed tokens
#[derive(Debug, Clone)]
pub struct TokenSemanticSummary {
    /// Identified modules
    pub modules: Vec<ModuleInfo>,
    /// Identified functions
    pub functions: Vec<FunctionInfo>,
    /// Identified types
    pub types: Vec<TypeInfo>,
    /// Identified capabilities
    pub capabilities: Vec<CapabilityInfo>,
    /// Overall semantic score
    pub semantic_score: f64,
    /// Detected patterns
    pub patterns: Vec<SemanticPattern>,
    /// Identifier usage statistics
    pub identifier_usage: HashMap<String, IdentifierUsage>,
}

/// Information about a detected module
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    /// Module span
    pub span: Span,
    /// Detected capability
    pub capability: Option<String>,
    /// Module description (if found in comments)
    pub description: Option<String>,
}

/// Information about a detected function
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Function span
    pub span: Span,
    /// Whether it's async
    pub is_async: bool,
    /// Inferred purpose from name analysis
    pub inferred_purpose: Option<String>,
    /// Parameter count
    pub parameter_count: usize,
}

/// Information about a detected type
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type span
    pub span: Span,
    /// Type kind (struct, enum, interface, etc.)
    pub kind: String,
    /// Inferred domain from name analysis
    pub inferred_domain: Option<String>,
}

/// Information about a detected capability
#[derive(Debug, Clone)]
pub struct CapabilityInfo {
    /// Capability name
    pub name: String,
    /// Capability span
    pub span: Span,
    /// Associated effects
    pub effects: Vec<String>,
}

/// Semantic patterns detected in token stream
#[derive(Debug, Clone)]
pub struct SemanticPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Description of the pattern
    pub description: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Span where pattern was detected
    pub span: Span,
    /// AI hints related to this pattern
    pub ai_hints: Vec<String>,
}

/// Types of semantic patterns
#[derive(Debug, Clone, PartialEq)]
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
}

/// Usage information for identifiers
#[derive(Debug, Clone)]
pub struct IdentifierUsage {
    /// How many times this identifier is used
    pub usage_count: usize,
    /// Contexts where this identifier appears
    pub contexts: Vec<String>,
    /// Whether this identifier follows naming conventions
    pub follows_conventions: bool,
    /// Suggested improvements
    pub suggestions: Vec<String>,
    /// Inferred semantic role
    pub semantic_role: Option<SemanticRole>,
}

/// Semantic role of an identifier
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticRole {
    /// Module name
    Module,
    /// Function name
    Function,
    /// Type name
    Type,
    /// Variable name
    Variable,
    /// Parameter name
    Parameter,
    /// Capability name
    Capability,
    /// Configuration key
    ConfigKey,
    /// Business concept
    BusinessConcept,
}

/// Token stream semantic analyzer
/// 
/// This analyzer processes token streams to extract semantic information
/// that was incorrectly placed in the lexer. It focuses on TOKEN RELATIONSHIPS
/// and PATTERNS rather than individual token classification.
pub struct TokenSemanticAnalyzer {
    /// Current analysis context
    current_context: AnalysisContext,
    /// Detected patterns
    patterns: Vec<SemanticPattern>,
    /// Identifier tracking
    identifier_usage: HashMap<String, IdentifierUsage>,
    /// Module information
    modules: Vec<ModuleInfo>,
    /// Function information
    functions: Vec<FunctionInfo>,
    /// Type information
    types: Vec<TypeInfo>,
    /// Capability information
    capabilities: Vec<CapabilityInfo>,
}

/// Current analysis context
#[derive(Debug, Clone)]
struct AnalysisContext {
    /// Current module being analyzed
    current_module: Option<String>,
    /// Current function being analyzed
    current_function: Option<String>,
    /// Current type being analyzed
    current_type: Option<String>,
    /// Nesting level
    nesting_level: usize,
    /// Whether we're in an async context
    in_async_context: bool,
}

impl TokenSemanticAnalyzer {
    /// Create a new token semantic analyzer
    pub fn new() -> Self {
        Self {
            current_context: AnalysisContext {
                current_module: None,
                current_function: None,
                current_type: None,
                nesting_level: 0,
                in_async_context: false,
            },
            patterns: Vec::new(),
            identifier_usage: HashMap::new(),
            modules: Vec::new(),
            functions: Vec::new(),
            types: Vec::new(),
            capabilities: Vec::new(),
        }
    }

    /// Analyze a stream of tokens to extract semantic information
    pub fn analyze_tokens(&mut self, tokens: &[Token]) -> TokenSemanticSummary {
        // Reset state for new analysis
        self.reset_analysis();

        // Process tokens sequentially to build context
        let mut i = 0;
        while i < tokens.len() {
            i = self.process_token_sequence(tokens, i);
        }

        // Generate final summary
        self.generate_summary()
    }

    /// Reset analysis state
    fn reset_analysis(&mut self) {
        self.current_context = AnalysisContext {
            current_module: None,
            current_function: None,
            current_type: None,
            nesting_level: 0,
            in_async_context: false,
        };
        self.patterns.clear();
        self.identifier_usage.clear();
        self.modules.clear();
        self.functions.clear();
        self.types.clear();
        self.capabilities.clear();
    }

    /// Process a sequence of tokens starting at the given index
    fn process_token_sequence(&mut self, tokens: &[Token], start_index: usize) -> usize {
        if start_index >= tokens.len() {
            return start_index;
        }

        let token = &tokens[start_index];
        
        match &token.kind {
            TokenKind::Module => self.process_module_declaration(tokens, start_index),
            TokenKind::Function | TokenKind::Fn => self.process_function_declaration(tokens, start_index),
            TokenKind::Type => self.process_type_declaration(tokens, start_index),
            TokenKind::Capability => self.process_capability_declaration(tokens, start_index),
            TokenKind::Identifier(name) => {
                self.process_identifier(name, token);
                start_index + 1
            }
            TokenKind::LeftBrace => {
                self.current_context.nesting_level += 1;
                start_index + 1
            }
            TokenKind::RightBrace => {
                self.current_context.nesting_level = self.current_context.nesting_level.saturating_sub(1);
                start_index + 1
            }
            TokenKind::Async => {
                self.current_context.in_async_context = true;
                start_index + 1
            }
            _ => start_index + 1,
        }
    }

    /// Process module declaration
    fn process_module_declaration(&mut self, tokens: &[Token], start_index: usize) -> usize {
        let mut index = start_index + 1; // Skip 'module' token
        
        // Look for module name
        if let Some(Token { kind: TokenKind::Identifier(name), span, .. }) = tokens.get(index) {
            let module_info = ModuleInfo {
                name: name.clone(),
                span: *span,
                capability: None, // Could be detected from following tokens
                description: None, // Could be extracted from preceding comments
            };
            
            self.modules.push(module_info);
            self.current_context.current_module = Some(name.clone());
            
            // Detect module organization pattern
            self.patterns.push(SemanticPattern {
                pattern_type: PatternType::ModuleOrganization,
                description: format!("Module '{}' follows standard organization", name),
                confidence: 0.8,
                span: *span,
                ai_hints: vec![
                    "Module represents single business capability".to_string(),
                    "Consider adding capability annotation".to_string(),
                ],
            });
            
            index += 1;
        }
        
        index
    }

    /// Process function declaration
    fn process_function_declaration(&mut self, tokens: &[Token], start_index: usize) -> usize {
        let mut index = start_index + 1; // Skip function keyword
        let is_async = self.current_context.in_async_context;
        
        // Look for function name
        if let Some(Token { kind: TokenKind::Identifier(name), span, .. }) = tokens.get(index) {
            let function_info = FunctionInfo {
                name: name.clone(),
                span: *span,
                is_async,
                inferred_purpose: self.infer_function_purpose(name),
                parameter_count: self.count_parameters(tokens, index),
            };
            
            self.functions.push(function_info);
            self.current_context.current_function = Some(name.clone());
            
            // Detect function naming patterns
            if self.follows_naming_conventions(name) {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::FunctionNaming,
                    description: format!("Function '{}' follows naming conventions", name),
                    confidence: 0.9,
                    span: *span,
                    ai_hints: vec![
                        "Function name is descriptive and follows conventions".to_string(),
                    ],
                });
            }
            
            index += 1;
        }
        
        // Reset async context after processing function
        self.current_context.in_async_context = false;
        index
    }

    /// Process type declaration
    fn process_type_declaration(&mut self, tokens: &[Token], start_index: usize) -> usize {
        let mut index = start_index + 1; // Skip 'type' token
        
        // Look for type name
        if let Some(Token { kind: TokenKind::Identifier(name), span, .. }) = tokens.get(index) {
            let type_info = TypeInfo {
                name: name.clone(),
                span: *span,
                kind: "type".to_string(), // Could be more specific based on following tokens
                inferred_domain: self.infer_type_domain(name),
            };
            
            self.types.push(type_info);
            self.current_context.current_type = Some(name.clone());
            
            // Detect type definition patterns
            self.patterns.push(SemanticPattern {
                pattern_type: PatternType::TypeDefinition,
                description: format!("Type '{}' definition", name),
                confidence: 0.8,
                span: *span,
                ai_hints: vec![
                    "Consider adding semantic constraints".to_string(),
                    "Type should express business meaning".to_string(),
                ],
            });
            
            index += 1;
        }
        
        index
    }

    /// Process capability declaration
    fn process_capability_declaration(&mut self, tokens: &[Token], start_index: usize) -> usize {
        let mut index = start_index + 1; // Skip 'capability' token
        
        // Look for capability name
        if let Some(Token { kind: TokenKind::Identifier(name), span, .. }) = tokens.get(index) {
            let capability_info = CapabilityInfo {
                name: name.clone(),
                span: *span,
                effects: Vec::new(), // Could be detected from following tokens
            };
            
            self.capabilities.push(capability_info);
            
            // Detect capability usage patterns
            self.patterns.push(SemanticPattern {
                pattern_type: PatternType::CapabilityUsage,
                description: format!("Capability '{}' declaration", name),
                confidence: 0.9,
                span: *span,
                ai_hints: vec![
                    "Capability enables secure operations".to_string(),
                    "Consider effect declarations".to_string(),
                ],
            });
            
            index += 1;
        }
        
        index
    }

    /// Process an identifier token
    fn process_identifier(&mut self, name: &str, _token: &Token) {
        let follows_conventions = self.follows_naming_conventions(name);
        let semantic_role = self.infer_semantic_role(name);
        
        let usage = self.identifier_usage.entry(name.to_string()).or_insert_with(|| {
            IdentifierUsage {
                usage_count: 0,
                contexts: Vec::new(),
                follows_conventions,
                suggestions: Vec::new(),
                semantic_role,
            }
        });
        
        usage.usage_count += 1;
        
        // Add context based on current analysis state
        if let Some(module) = &self.current_context.current_module {
            usage.contexts.push(format!("Module: {}", module));
        }
        if let Some(function) = &self.current_context.current_function {
            usage.contexts.push(format!("Function: {}", function));
        }
        if let Some(type_name) = &self.current_context.current_type {
            usage.contexts.push(format!("Type: {}", type_name));
        }
    }

    /// Infer function purpose from name
    fn infer_function_purpose(&self, name: &str) -> Option<String> {
        if name.starts_with("get") || name.starts_with("fetch") || name.starts_with("find") {
            Some("Data retrieval".to_string())
        } else if name.starts_with("set") || name.starts_with("update") || name.starts_with("save") {
            Some("Data modification".to_string())
        } else if name.starts_with("create") || name.starts_with("new") || name.starts_with("make") {
            Some("Object creation".to_string())
        } else if name.starts_with("delete") || name.starts_with("remove") || name.starts_with("destroy") {
            Some("Data deletion".to_string())
        } else if name.starts_with("validate") || name.starts_with("check") || name.starts_with("verify") {
            Some("Validation".to_string())
        } else if name.starts_with("calculate") || name.starts_with("compute") || name.starts_with("process") {
            Some("Computation".to_string())
        } else {
            None
        }
    }

    /// Infer type domain from name
    fn infer_type_domain(&self, name: &str) -> Option<String> {
        let lower_name = name.to_lowercase();
        
        if lower_name.contains("user") || lower_name.contains("account") || lower_name.contains("profile") {
            Some("User Management".to_string())
        } else if lower_name.contains("payment") || lower_name.contains("transaction") || lower_name.contains("billing") {
            Some("Financial".to_string())
        } else if lower_name.contains("product") || lower_name.contains("item") || lower_name.contains("catalog") {
            Some("Product Management".to_string())
        } else if lower_name.contains("order") || lower_name.contains("cart") || lower_name.contains("checkout") {
            Some("E-commerce".to_string())
        } else if lower_name.contains("config") || lower_name.contains("setting") || lower_name.contains("preference") {
            Some("Configuration".to_string())
        } else {
            None
        }
    }

    /// Check if identifier follows naming conventions
    fn follows_naming_conventions(&self, name: &str) -> bool {
        // Check for snake_case (common in Rust/Python style)
        let is_snake_case = name.chars().all(|c| c.is_lowercase() || c.is_ascii_digit() || c == '_') 
            && !name.starts_with('_') 
            && !name.ends_with('_')
            && !name.contains("__");
        
        // Check for camelCase
        let is_camel_case = name.chars().next().map_or(false, |c| c.is_lowercase())
            && name.chars().any(|c| c.is_uppercase())
            && !name.contains('_');
        
        // Check for PascalCase
        let is_pascal_case = name.chars().next().map_or(false, |c| c.is_uppercase())
            && !name.contains('_');
        
        // Check for descriptive length (at least 3 characters, avoid single letters except common cases)
        let is_descriptive = name.len() >= 3 || matches!(name, "i" | "j" | "x" | "y" | "z");
        
        (is_snake_case || is_camel_case || is_pascal_case) && is_descriptive
    }

    /// Infer semantic role of identifier
    fn infer_semantic_role(&self, name: &str) -> Option<SemanticRole> {
        // This is a simplified heuristic-based approach
        if name.ends_with("Module") || name.ends_with("_module") {
            Some(SemanticRole::Module)
        } else if name.ends_with("Type") || name.ends_with("_type") || name.chars().next().map_or(false, |c| c.is_uppercase()) {
            Some(SemanticRole::Type)
        } else if name.starts_with("CONFIG_") || name.ends_with("_config") {
            Some(SemanticRole::ConfigKey)
        } else {
            // Default based on context
            if self.current_context.current_function.is_some() {
                Some(SemanticRole::Variable)
            } else {
                Some(SemanticRole::BusinessConcept)
            }
        }
    }

    /// Count parameters in function declaration
    fn count_parameters(&self, tokens: &[Token], start_index: usize) -> usize {
        let mut count = 0;
        let mut paren_level = 0;
        let mut found_open_paren = false;
        
        for token in tokens.iter().skip(start_index) {
            match &token.kind {
                TokenKind::LeftParen => {
                    paren_level += 1;
                    found_open_paren = true;
                }
                TokenKind::RightParen => {
                    paren_level -= 1;
                    if paren_level == 0 && found_open_paren {
                        break;
                    }
                }
                TokenKind::Identifier(_) if paren_level == 1 && found_open_paren => {
                    count += 1;
                }
                TokenKind::LeftBrace if paren_level == 0 => break,
                _ => {}
            }
        }
        
        count
    }

    /// Generate final semantic summary
    fn generate_summary(&self) -> TokenSemanticSummary {
        let semantic_score = self.calculate_semantic_score();
        
        TokenSemanticSummary {
            modules: self.modules.clone(),
            functions: self.functions.clone(),
            types: self.types.clone(),
            capabilities: self.capabilities.clone(),
            semantic_score,
            patterns: self.patterns.clone(),
            identifier_usage: self.identifier_usage.clone(),
        }
    }

    /// Calculate overall semantic score
    fn calculate_semantic_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;
        
        // Factor in naming conventions
        let good_names = self.identifier_usage.values()
            .filter(|usage| usage.follows_conventions)
            .count();
        let total_names = self.identifier_usage.len();
        
        if total_names > 0 {
            score += (good_names as f64 / total_names as f64) * 0.3;
            factors += 1;
        }
        
        // Factor in pattern detection
        let pattern_confidence: f64 = self.patterns.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / self.patterns.len().max(1) as f64;
        score += pattern_confidence * 0.4;
        factors += 1;
        
        // Factor in structural organization
        let has_modules = !self.modules.is_empty();
        let has_types = !self.types.is_empty();
        let has_functions = !self.functions.is_empty();
        
        let structure_score = match (has_modules, has_types, has_functions) {
            (true, true, true) => 1.0,
            (true, true, false) | (true, false, true) | (false, true, true) => 0.7,
            (true, false, false) | (false, true, false) | (false, false, true) => 0.5,
            (false, false, false) => 0.2,
        };
        
        score += structure_score * 0.3;
        factors += 1;
        
        if factors > 0 {
            score / factors as f64
        } else {
            0.5 // Default neutral score
        }
    }
}

impl Default for TokenSemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_common::{span::Position, SourceId};

    fn create_test_token(kind: TokenKind, line: u32, col: u32) -> Token {
        Token::new(
            kind,
            prism_common::span::Span::new(
                Position::new(line, col, 0),
                Position::new(line, col + 1, 1),
                SourceId::new(1),
            ),
        )
    }

    #[test]
    fn test_module_analysis() {
        let mut analyzer = TokenSemanticAnalyzer::new();
        
        let tokens = vec![
            create_test_token(TokenKind::Module, 1, 1),
            create_test_token(TokenKind::Identifier("UserAuth".to_string()), 1, 8),
            create_test_token(TokenKind::LeftBrace, 1, 17),
            create_test_token(TokenKind::RightBrace, 2, 1),
        ];
        
        let summary = analyzer.analyze_tokens(&tokens);
        
        assert_eq!(summary.modules.len(), 1);
        assert_eq!(summary.modules[0].name, "UserAuth");
        assert!(!summary.patterns.is_empty());
    }

    #[test]
    fn test_function_analysis() {
        let mut analyzer = TokenSemanticAnalyzer::new();
        
        let tokens = vec![
            create_test_token(TokenKind::Function, 1, 1),
            create_test_token(TokenKind::Identifier("getUserById".to_string()), 1, 10),
            create_test_token(TokenKind::LeftParen, 1, 21),
            create_test_token(TokenKind::Identifier("id".to_string()), 1, 22),
            create_test_token(TokenKind::RightParen, 1, 24),
        ];
        
        let summary = analyzer.analyze_tokens(&tokens);
        
        assert_eq!(summary.functions.len(), 1);
        assert_eq!(summary.functions[0].name, "getUserById");
        assert_eq!(summary.functions[0].inferred_purpose, Some("Data retrieval".to_string()));
    }

    #[test]
    fn test_semantic_score_calculation() {
        let mut analyzer = TokenSemanticAnalyzer::new();
        
        let tokens = vec![
            create_test_token(TokenKind::Module, 1, 1),
            create_test_token(TokenKind::Identifier("user_management".to_string()), 1, 8),
            create_test_token(TokenKind::Function, 2, 1),
            create_test_token(TokenKind::Identifier("create_user".to_string()), 2, 10),
        ];
        
        let summary = analyzer.analyze_tokens(&tokens);
        
        assert!(summary.semantic_score > 0.5);
        assert!(!summary.modules.is_empty());
        assert!(!summary.functions.is_empty());
    }
} 