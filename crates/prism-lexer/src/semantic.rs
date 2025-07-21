//! Basic token-level semantic analysis and context extraction
//!
//! ## Clear Separation of Concerns
//!
//! **✅ What this module DOES:**
//! - Basic semantic context for individual tokens
//! - Simple identifier usage tracking
//! - Token-level pattern detection
//! - AI-comprehensible metadata for single tokens
//!
//! **❌ What this module does NOT do (moved to appropriate modules):**
//! - ❌ Cross-token relationship analysis (→ prism-semantic)
//! - ❌ Multi-token semantic patterns (→ prism-parser)
//! - ❌ Complex semantic analysis (→ prism-semantic)
//! - ❌ Type inference (→ prism-semantic)

use crate::token::{SemanticContext, Token, TokenKind};
use std::collections::HashMap;

/// Basic token-level semantic analyzer (complex analysis moved to prism-semantic)
pub struct SemanticAnalyzer {
    /// Current module context (basic tracking only)
    current_module: Option<String>,
    /// Current function context (basic tracking only)
    current_function: Option<String>,
    /// Simple identifier usage tracking
    identifier_usage: HashMap<String, IdentifierUsage>,
    /// Basic token-level semantic patterns
    patterns: Vec<SemanticPattern>,
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
}

/// Semantic patterns detected in the code
#[derive(Debug, Clone)]
pub struct SemanticPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Description of the pattern
    pub description: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Hints to help AI systems understand this pattern
    pub ai_comprehension_hints: Vec<String>,
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
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new() -> Self {
        Self {
            current_module: None,
            current_function: None,
            identifier_usage: HashMap::new(),
            patterns: Vec::new(),
        }
    }

    /// Analyze a token and enrich it with semantic context
    pub fn analyze_token(&mut self, token: &mut Token) {
        // Update current context
        self.update_context(token);
        
        // Add semantic context based on token type
        if let Some(context) = self.infer_semantic_context(token) {
            token.semantic_context = Some(context);
        }
        
        // Track identifier usage
        if let TokenKind::Identifier(name) = &token.kind {
            self.track_identifier_usage(name.clone());
        }
        
        // Detect semantic patterns
        self.detect_patterns(token);
    }

    /// Update the current analysis context
    fn update_context(&mut self, token: &Token) {
        match &token.kind {
            TokenKind::Module => {
                self.current_module = None; // Will be set when we see the identifier
            }
            TokenKind::Function | TokenKind::Fn => {
                self.current_function = None; // Will be set when we see the identifier
            }
            TokenKind::Identifier(name) => {
                // If we just saw a module keyword, this is the module name
                if self.current_module.is_none() {
                    self.current_module = Some(name.clone());
                }
                // If we just saw a function keyword, this is the function name
                if self.current_function.is_none() {
                    self.current_function = Some(name.clone());
                }
            }
            _ => {}
        }
    }

    /// Infer semantic context for a token
    fn infer_semantic_context(&self, token: &Token) -> Option<SemanticContext> {
        match &token.kind {
            TokenKind::Module => {
                let mut context = SemanticContext::with_purpose("Define module boundary and capabilities");
                context.domain = Some("Module System".to_string());
                context.add_concept("Conceptual Cohesion");
                context.add_concept("Smart Modules");
                context.add_concept("Business Capabilities");
                context.add_ai_comprehension_hint("Modules should represent single business capabilities");
                context.add_ai_comprehension_hint("Each module should have high conceptual cohesion");
                
                if let Some(module_name) = &self.current_module {
                    context.add_ai_comprehension_hint(&format!("Module '{}' groups related functionality", module_name));
                }
                
                Some(context)
            }
            TokenKind::Section => {
                let mut context = SemanticContext::with_purpose("Organize related code within a module");
                context.domain = Some("Module Organization".to_string());
                context.add_concept("Smart Modules");
                context.add_concept("Separation of Concerns");
                context.add_ai_comprehension_hint("Sections group related types, functions, or configurations");
                context.add_ai_comprehension_hint("Common sections: types, interface, operations, config");
                Some(context)
            }
            TokenKind::Function | TokenKind::Fn => {
                let mut context = SemanticContext::with_purpose("Define function with semantic contracts");
                context.domain = Some("Function Definition".to_string());
                context.add_concept("Single Responsibility");
                context.add_concept("Semantic Types");
                context.add_concept("Effect System");
                context.add_ai_comprehension_hint("Functions should have single responsibility");
                context.add_ai_comprehension_hint("All public functions require documentation");
                
                if let Some(function_name) = &self.current_function {
                    context.add_ai_comprehension_hint(&format!("Function '{}' should have clear purpose", function_name));
                    
                    // Analyze function name for semantic hints
                    if let Some(hints) = self.analyze_function_name(function_name) {
                        for hint in hints {
                            context.add_ai_comprehension_hint(&hint);
                        }
                    }
                }
                
                Some(context)
            }
            TokenKind::Type => {
                let mut context = SemanticContext::with_purpose("Define semantic type with business constraints");
                context.domain = Some("Type System".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Business Rules");
                context.add_concept("Domain Modeling");
                context.add_ai_comprehension_hint("Types should express business meaning");
                context.add_ai_comprehension_hint("Include validation constraints where applicable");
                context.add_ai_comprehension_hint("Types should be self-documenting");
                Some(context)
            }
            TokenKind::Capability => {
                let mut context = SemanticContext::with_purpose("Define security capability");
                context.domain = Some("Security".to_string());
                context.add_concept("Capability-Based Security");
                context.add_concept("Least Privilege");
                context.add_concept("Resource Access");
                context.add_security_implication("Capabilities control access to resources");
                context.add_security_implication("Should follow principle of least privilege");
                context.add_ai_comprehension_hint("Capabilities define what operations are allowed");
                Some(context)
            }
            TokenKind::Effects => {
                let mut context = SemanticContext::with_purpose("Declare computational effects");
                context.domain = Some("Effect System".to_string());
                context.add_concept("Side Effects");
                context.add_concept("Purity");
                context.add_concept("Resource Tracking");
                context.add_security_implication("Effects must be explicitly declared");
                context.add_security_implication("Enables static analysis of resource usage");
                context.add_ai_comprehension_hint("Effects track what resources functions access");
                Some(context)
            }
            TokenKind::Public | TokenKind::Pub => {
                let mut context = SemanticContext::with_purpose("Make item publicly accessible");
                context.domain = Some("Visibility".to_string());
                context.add_concept("API Design");
                context.add_concept("Encapsulation");
                context.add_security_implication("Public items form the external API");
                context.add_ai_comprehension_hint("Public items should be well-documented");
                context.add_ai_comprehension_hint("Consider if this needs to be public");
                Some(context)
            }
            TokenKind::Private => {
                let mut context = SemanticContext::with_purpose("Restrict access to implementation details");
                context.domain = Some("Visibility".to_string());
                context.add_concept("Encapsulation");
                context.add_concept("Information Hiding");
                context.add_ai_comprehension_hint("Private items are implementation details");
                context.add_ai_comprehension_hint("Helps maintain clean API boundaries");
                Some(context)
            }
            TokenKind::Identifier(name) => {
                self.analyze_identifier_context(name)
            }
            _ => None,
        }
    }

    /// Analyze function name for semantic hints
    fn analyze_function_name(&self, name: &str) -> Option<Vec<String>> {
        let mut hints = Vec::new();
        
        // Check for common prefixes and their meanings
        if name.starts_with("get") {
            hints.push("'get' prefix suggests data retrieval operation".to_string());
            hints.push("Consider if this function has side effects".to_string());
        } else if name.starts_with("set") {
            hints.push("'set' prefix suggests data modification operation".to_string());
            hints.push("Consider if this should return a result".to_string());
        } else if name.starts_with("create") {
            hints.push("'create' prefix suggests object instantiation".to_string());
            hints.push("Consider validation and error handling".to_string());
        } else if name.starts_with("validate") {
            hints.push("'validate' prefix suggests validation logic".to_string());
            hints.push("Should return clear validation results".to_string());
        } else if name.starts_with("calculate") {
            hints.push("'calculate' prefix suggests computation".to_string());
            hints.push("Consider if this is a pure function".to_string());
        } else if name.starts_with("handle") {
            hints.push("'handle' prefix suggests event or error handling".to_string());
            hints.push("Consider error recovery strategies".to_string());
        }
        
        // Check for common suffixes
        if name.ends_with("_async") || name.ends_with("Async") {
            hints.push("Async suffix indicates asynchronous operation".to_string());
            hints.push("Consider proper error handling for async operations".to_string());
        }
        
        // Check for business domain terms
        if name.contains("auth") || name.contains("Auth") {
            hints.push("Authentication-related function".to_string());
            hints.push("Consider security implications and audit requirements".to_string());
        } else if name.contains("payment") || name.contains("Payment") {
            hints.push("Payment-related function".to_string());
            hints.push("Consider PCI compliance and security requirements".to_string());
        } else if name.contains("user") || name.contains("User") {
            hints.push("User-related function".to_string());
            hints.push("Consider privacy and data protection requirements".to_string());
        }
        
        if hints.is_empty() {
            None
        } else {
            Some(hints)
        }
    }

    /// Analyze identifier context
    fn analyze_identifier_context(&self, name: &str) -> Option<SemanticContext> {
        let mut context = SemanticContext::with_purpose("Identifier reference");
        
        // Check for semantic type constraint keywords
        match name {
            "min_value" | "max_value" => {
                context.domain = Some("Range Constraints".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Value Bounds");
                context.add_ai_comprehension_hint("Range constraints ensure values fall within acceptable bounds");
                context.add_ai_comprehension_hint("Critical for domain validation and business rules");
            }
            "min_length" | "max_length" => {
                context.domain = Some("Length Constraints".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("String Validation");
                context.add_ai_comprehension_hint("Length constraints ensure strings meet size requirements");
                context.add_ai_comprehension_hint("Important for data validation and security");
            }
            "pattern" => {
                context.domain = Some("Pattern Constraints".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Regular Expressions");
                context.add_ai_comprehension_hint("Pattern constraints use regex for format validation");
                context.add_ai_comprehension_hint("Essential for email, phone, and ID validation");
            }
            "format" => {
                context.domain = Some("Format Constraints".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Data Formatting");
                context.add_ai_comprehension_hint("Format constraints specify data structure requirements");
                context.add_ai_comprehension_hint("Used for dates, currencies, and structured identifiers");
            }
            "precision" => {
                context.domain = Some("Numeric Precision".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Financial Types");
                context.add_ai_comprehension_hint("Precision specifies decimal places for monetary values");
                context.add_ai_comprehension_hint("Critical for financial calculations and compliance");
            }
            "currency" => {
                context.domain = Some("Currency Specification".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Financial Types");
                context.add_ai_comprehension_hint("Currency constraint ensures monetary type safety");
                context.add_ai_comprehension_hint("Prevents mixing different currency types");
            }
            "non_negative" => {
                context.domain = Some("Sign Constraints".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Business Rules");
                context.add_ai_comprehension_hint("Non-negative constraint prevents negative values");
                context.add_ai_comprehension_hint("Common for quantities, ages, and amounts");
            }
            "immutable" => {
                context.domain = Some("Mutability Constraints".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Data Integrity");
                context.add_ai_comprehension_hint("Immutable constraint prevents value changes");
                context.add_ai_comprehension_hint("Important for IDs and audit data");
            }
            "validated" => {
                context.domain = Some("Validation Status".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Data Quality");
                context.add_ai_comprehension_hint("Validated constraint ensures data has been verified");
                context.add_ai_comprehension_hint("Critical for user input and external data");
            }
            "business_rule" => {
                context.domain = Some("Business Logic".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Domain Rules");
                context.add_ai_comprehension_hint("Business rules encode domain-specific constraints");
                context.add_ai_comprehension_hint("Essential for modeling complex business requirements");
            }
            "security_classification" => {
                context.domain = Some("Security".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Data Classification");
                context.add_security_implication("Security classification controls data access");
                context.add_ai_comprehension_hint("Determines handling requirements for sensitive data");
            }
            "compliance" => {
                context.domain = Some("Regulatory Compliance".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Legal Requirements");
                context.add_security_implication("Compliance requirements must be met");
                context.add_ai_comprehension_hint("Ensures adherence to regulatory standards");
            }
            "ai_context" => {
                context.domain = Some("AI Integration".to_string());
                context.add_concept("Semantic Types");
                context.add_concept("Machine Comprehension");
                context.add_ai_comprehension_hint("AI context provides machine-readable explanations");
                context.add_ai_comprehension_hint("Enables better AI understanding of code intent");
            }
            _ => {
                // Existing naming convention analysis
                if self.is_snake_case(name) {
                    context.add_ai_comprehension_hint("Uses snake_case naming convention");
                } else if self.is_camel_case(name) {
                    context.add_ai_comprehension_hint("Uses camelCase naming convention");
                } else if self.is_pascal_case(name) {
                    context.add_ai_comprehension_hint("Uses PascalCase naming convention");
                }
                
                // Check for descriptive naming
                if name.len() < 3 {
                    context.add_ai_comprehension_hint("Very short identifier - consider more descriptive name");
                } else if name.len() > 30 {
                    context.add_ai_comprehension_hint("Very long identifier - consider if it can be shortened");
                }
                
                // Check for common abbreviations
                if name.contains("mgr") {
                    context.add_ai_comprehension_hint("'mgr' abbreviation - consider 'manager' for clarity");
                } else if name.contains("cfg") {
                    context.add_ai_comprehension_hint("'cfg' abbreviation - consider 'config' for clarity");
                } else if name.contains("svc") {
                    context.add_ai_comprehension_hint("'svc' abbreviation - consider 'service' for clarity");
                }
                
                return Some(context);
            }
        }
        
        Some(context)
    }

    /// Track identifier usage
    fn track_identifier_usage(&mut self, name: String) {
        let follows_conventions = self.follows_naming_conventions(&name);
        
        let usage = self.identifier_usage.entry(name.clone()).or_insert_with(|| {
            IdentifierUsage {
                usage_count: 0,
                contexts: Vec::new(),
                follows_conventions,
                suggestions: Vec::new(),
            }
        });
        
        usage.usage_count += 1;
        
        // Add current context
        if let Some(module) = &self.current_module {
            usage.contexts.push(format!("Module: {}", module));
        }
        if let Some(function) = &self.current_function {
            usage.contexts.push(format!("Function: {}", function));
        }
        
        // Generate suggestions
        if !usage.follows_conventions {
            usage.suggestions.push("Consider following naming conventions".to_string());
        }
        
        if usage.usage_count == 1 && name.len() < 3 {
            usage.suggestions.push("Consider more descriptive name".to_string());
        }
    }

    /// Detect semantic patterns in the code
    fn detect_patterns(&mut self, token: &Token) {
        match &token.kind {
            TokenKind::Module => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::ModuleOrganization,
                    description: "Module definition detected".to_string(),
                    confidence: 0.9,
                    ai_comprehension_hints: vec![
                        "Modules should represent single business capabilities".to_string(),
                        "Consider organizing related functionality together".to_string(),
                    ],
                });
            }
            TokenKind::Function | TokenKind::Fn => {
                if let Some(function_name) = &self.current_function {
                    let confidence = if self.is_descriptive_name(function_name) { 0.8 } else { 0.5 };
                    
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::FunctionNaming,
                        description: format!("Function '{}' defined", function_name),
                        confidence,
                        ai_comprehension_hints: vec![
                            "Function names should be descriptive".to_string(),
                            "Consider if this function has single responsibility".to_string(),
                        ],
                    });
                }
            }
            TokenKind::Type => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::TypeDefinition,
                    description: "Type definition detected".to_string(),
                    confidence: 0.9,
                    ai_comprehension_hints: vec![
                        "Types should express business domain concepts".to_string(),
                        "Consider adding validation constraints".to_string(),
                    ],
                });
            }
            TokenKind::Where => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::TypeDefinition,
                    description: "Semantic type constraints detected".to_string(),
                    confidence: 0.95,
                    ai_comprehension_hints: vec![
                        "Semantic constraints enhance type safety".to_string(),
                        "Constraints should reflect business rules".to_string(),
                        "Consider adding AI context for better comprehension".to_string(),
                    ],
                });
            }
            TokenKind::Requires | TokenKind::Ensures | TokenKind::Invariant => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::BusinessLogic,
                    description: "Formal verification constraint detected".to_string(),
                    confidence: 0.9,
                    ai_comprehension_hints: vec![
                        "Formal constraints enable compile-time verification".to_string(),
                        "Preconditions and postconditions improve reliability".to_string(),
                        "Invariants ensure data consistency".to_string(),
                    ],
                });
            }
            TokenKind::Effects => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::EffectDeclaration,
                    description: "Effect system usage detected".to_string(),
                    confidence: 0.9,
                    ai_comprehension_hints: vec![
                        "Effects track computational side effects".to_string(),
                        "Explicit effect tracking improves safety".to_string(),
                        "Consider capability-based security".to_string(),
                    ],
                });
            }
            TokenKind::Capability => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::CapabilityUsage,
                    description: "Capability-based security detected".to_string(),
                    confidence: 0.9,
                    ai_comprehension_hints: vec![
                        "Capabilities provide fine-grained access control".to_string(),
                        "Follow principle of least privilege".to_string(),
                        "Document capability requirements clearly".to_string(),
                    ],
                });
            }
            
            // Enhanced literal patterns
            TokenKind::RegexLiteral(pattern) => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::TypeDefinition,
                    description: format!("Regular expression pattern '{}' detected", pattern),
                    confidence: 0.85,
                    ai_comprehension_hints: vec![
                        "Regex patterns should be well-documented".to_string(),
                        "Consider validation for complex patterns".to_string(),
                        "Test regex patterns with edge cases".to_string(),
                        "Consider performance implications for complex patterns".to_string(),
                    ],
                });
            }
            
            TokenKind::MoneyLiteral(amount) => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::TypeDefinition,
                    description: format!("Money literal '{}' detected", amount),
                    confidence: 0.9,
                    ai_comprehension_hints: vec![
                        "Money literals should specify currency explicitly".to_string(),
                        "Use precise decimal arithmetic for financial calculations".to_string(),
                        "Consider compliance requirements (PCI, financial regulations)".to_string(),
                        "Add validation for reasonable monetary ranges".to_string(),
                        "Consider currency conversion and exchange rates".to_string(),
                    ],
                });
            }
            
            TokenKind::DurationLiteral(duration) => {
                self.patterns.push(SemanticPattern {
                    pattern_type: PatternType::TypeDefinition,
                    description: format!("Duration literal '{}' detected", duration),
                    confidence: 0.85,
                    ai_comprehension_hints: vec![
                        "Duration literals improve code readability".to_string(),
                        "Consider timeout and performance implications".to_string(),
                        "Validate duration ranges for business logic".to_string(),
                        "Consider timezone implications for longer durations".to_string(),
                        "Document expected duration ranges".to_string(),
                    ],
                });
            }
            
            TokenKind::Identifier(name) => {
                // Detect semantic type patterns based on identifier names
                if name.ends_with("Id") || name.ends_with("ID") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Identifier type '{}' detected", name),
                        confidence: 0.8,
                        ai_comprehension_hints: vec![
                            "ID types should have format constraints".to_string(),
                            "Consider immutability for identifiers".to_string(),
                            "Add validation patterns for ID formats".to_string(),
                        ],
                    });
                } else if name.to_lowercase().contains("money") || name.to_lowercase().contains("price") || name.to_lowercase().contains("amount") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Financial type '{}' detected", name),
                        confidence: 0.85,
                        ai_comprehension_hints: vec![
                            "Financial types should specify currency".to_string(),
                            "Use precise decimal arithmetic".to_string(),
                            "Consider non-negative constraints".to_string(),
                            "Add compliance requirements for financial data".to_string(),
                        ],
                    });
                } else if name.to_lowercase().contains("email") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Email type '{}' detected", name),
                        confidence: 0.9,
                        ai_comprehension_hints: vec![
                            "Email types should have pattern validation".to_string(),
                            "Consider normalization to lowercase".to_string(),
                            "Add length constraints for email addresses".to_string(),
                        ],
                    });
                } else if name.to_lowercase().contains("phone") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Phone number type '{}' detected", name),
                        confidence: 0.85,
                        ai_comprehension_hints: vec![
                            "Phone types should use E.164 format".to_string(),
                            "Add region validation constraints".to_string(),
                            "Consider international format support".to_string(),
                        ],
                    });
                } else if name.to_lowercase().contains("password") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Password type '{}' detected", name),
                        confidence: 0.9,
                        ai_comprehension_hints: vec![
                            "Password types should be hashed, not plain text".to_string(),
                            "Mark as sensitive to prevent logging".to_string(),
                            "Add memory protection constraints".to_string(),
                            "Consider security classification".to_string(),
                        ],
                    });
                } else if name.to_lowercase().contains("timeout") || name.to_lowercase().contains("duration") || name.to_lowercase().contains("delay") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Time-related type '{}' detected", name),
                        confidence: 0.8,
                        ai_comprehension_hints: vec![
                            "Time types should use duration literals for clarity".to_string(),
                            "Consider reasonable timeout ranges".to_string(),
                            "Document time unit expectations".to_string(),
                            "Consider timezone implications".to_string(),
                        ],
                    });
                } else if name.to_lowercase().contains("pattern") || name.to_lowercase().contains("regex") || name.to_lowercase().contains("regexp") {
                    self.patterns.push(SemanticPattern {
                        pattern_type: PatternType::TypeDefinition,
                        description: format!("Pattern type '{}' detected", name),
                        confidence: 0.85,
                        ai_comprehension_hints: vec![
                            "Pattern types should use regex literals".to_string(),
                            "Document pattern expectations clearly".to_string(),
                            "Test patterns with edge cases".to_string(),
                            "Consider pattern complexity and performance".to_string(),
                        ],
                    });
                }
            }
            _ => {}
        }
    }

    /// Check if name follows naming conventions
    fn follows_naming_conventions(&self, name: &str) -> bool {
        self.is_snake_case(name) || self.is_camel_case(name) || self.is_pascal_case(name)
    }

    /// Check if name is snake_case
    fn is_snake_case(&self, name: &str) -> bool {
        name.chars().all(|c| c.is_lowercase() || c.is_ascii_digit() || c == '_')
            && !name.starts_with('_')
            && !name.ends_with('_')
            && !name.contains("__")
    }

    /// Check if name is camelCase
    fn is_camel_case(&self, name: &str) -> bool {
        name.chars().next().map_or(false, |c| c.is_lowercase())
            && name.chars().any(|c| c.is_uppercase())
            && name.chars().all(|c| c.is_alphanumeric())
    }

    /// Check if name is PascalCase
    fn is_pascal_case(&self, name: &str) -> bool {
        name.chars().next().map_or(false, |c| c.is_uppercase())
            && name.chars().all(|c| c.is_alphanumeric())
    }

    /// Check if name is descriptive
    fn is_descriptive_name(&self, name: &str) -> bool {
        name.len() >= 4 && !name.chars().all(|c| c.is_ascii_digit())
    }

    /// Get semantic patterns detected
    pub fn get_patterns(&self) -> &[SemanticPattern] {
        &self.patterns
    }

    /// Get identifier usage statistics
    pub fn get_identifier_usage(&self) -> &HashMap<String, IdentifierUsage> {
        &self.identifier_usage
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{SyntaxStyle, Token, TokenKind};
    use prism_common::span::Span;

    #[test]
    fn test_module_semantic_analysis() {
        let mut analyzer = SemanticAnalyzer::new();
        let mut token = Token::new(
            TokenKind::Module,
            Span::new(prism_common::SourceId::new(1), prism_common::span::Position::new(1, 1), prism_common::span::Position::new(1, 7)),
            SyntaxStyle::Canonical,
        );
        
        analyzer.analyze_token(&mut token);
        
        assert!(token.semantic_context.is_some());
        let context = token.semantic_context.unwrap();
        assert_eq!(context.domain, Some("Module System".to_string()));
        assert!(context.related_concepts.contains(&"Conceptual Cohesion".to_string()));
    }

    #[test]
    fn test_function_name_analysis() {
        let analyzer = SemanticAnalyzer::new();
        let hints = analyzer.analyze_function_name("getUserById").unwrap();
        
        assert!(hints.iter().any(|h| h.contains("'get' prefix")));
    }

    #[test]
    fn test_naming_conventions() {
        let analyzer = SemanticAnalyzer::new();
        
        assert!(analyzer.is_snake_case("user_name"));
        assert!(analyzer.is_camel_case("userName"));
        assert!(analyzer.is_pascal_case("UserName"));
        
        assert!(!analyzer.is_snake_case("UserName"));
        assert!(!analyzer.is_camel_case("user_name"));
        assert!(!analyzer.is_pascal_case("userName"));
    }

    #[test]
    fn test_pattern_detection() {
        let mut analyzer = SemanticAnalyzer::new();
        let mut token = Token::new(
            TokenKind::Function,
            Span::new(prism_common::SourceId::new(1), prism_common::span::Position::new(1, 1), prism_common::span::Position::new(1, 9)),
            SyntaxStyle::Canonical,
        );
        
        analyzer.analyze_token(&mut token);
        
        let patterns = analyzer.get_patterns();
        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].pattern_type, PatternType::FunctionNaming);
    }
} 