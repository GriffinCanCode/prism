//! Semantic Preservation for JavaScript Backend
//!
//! This module handles semantic type preservation and business rule generation
//! for JavaScript output, maintaining Prism's semantic richness in generated code.

use super::{JavaScriptResult, JavaScriptError};
use crate::backends::{PIRSemanticType, BusinessRule, ValidationPredicate};
use serde::{Deserialize, Serialize};

/// Semantic preservation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Enable semantic type preservation
    pub enable_semantic_types: bool,
    /// Enable business rule generation
    pub enable_business_rules: bool,
    /// Enable runtime validation
    pub enable_runtime_validation: bool,
    /// Generate branded types for semantic safety
    pub generate_branded_types: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enable_semantic_types: true,
            enable_business_rules: true,
            enable_runtime_validation: true,
            generate_branded_types: true,
        }
    }
}

/// Semantic type preserver for JavaScript backend
pub struct SemanticTypePreserver {
    config: SemanticConfig,
}

impl SemanticTypePreserver {
    pub fn new(config: SemanticConfig) -> Self {
        Self { config }
    }

    /// Generate semantic type preservation code
    pub fn generate_semantic_type(&self, semantic_type: &PIRSemanticType) -> JavaScriptResult<String> {
        if !self.config.enable_semantic_types {
            return Ok(String::new());
        }

        let mut output = String::new();

        // Generate JSDoc with semantic metadata
        output.push_str(&format!(
            r#"/**
 * Semantic Type: {}
 * Domain: {}
 * Security Classification: {:?}
 * Base Type: JavaScript {}
 * 
 * Business Rules:
{business_rules}
 * 
 * Validation Predicates:
{validation_predicates}
 * 
 * @semantic_type {{
 *   name: "{name}",
 *   domain: "{domain}",
 *   baseType: "{base_type}",
 *   businessRules: [{business_rule_names}],
 *   validationPredicates: [{validation_predicate_names}]
 * }}
 */
"#,
            semantic_type.name,
            semantic_type.domain,
            semantic_type.security_classification,
            self.convert_base_type(&semantic_type.base_type),
            business_rules = semantic_type.business_rules.iter()
                .map(|rule| format!(" * - {}: {}", rule.name, rule.description))
                .collect::<Vec<_>>()
                .join("\n"),
            validation_predicates = semantic_type.validation_predicates.iter()
                .map(|pred| format!(" * - {}: {}", pred.name, pred.expression))
                .collect::<Vec<_>>()
                .join("\n"),
            name = semantic_type.name,
            domain = semantic_type.domain,
            base_type = self.convert_base_type(&semantic_type.base_type),
            business_rule_names = semantic_type.business_rules.iter()
                .map(|rule| format!("\"{}\"", rule.name))
                .collect::<Vec<_>>()
                .join(", "),
            validation_predicate_names = semantic_type.validation_predicates.iter()
                .map(|pred| format!("\"{}\"", pred.name))
                .collect::<Vec<_>>()
                .join(", "),
        ));

        if self.config.generate_branded_types {
            // Generate branded type using JavaScript classes
            output.push_str(&format!(
                r#"class {} {{
    constructor(value) {{
        // Runtime type validation
        if (!this.constructor.isValid(value)) {{
            throw new TypeError(`Invalid value for {}: ${{JSON.stringify(value)}}`);
        }}
        this.value = value;
        this.__semantic_type = '{}';
        Object.freeze(this);
    }}

    valueOf() {{
        return this.value;
    }}

    toString() {{
        return String(this.value);
    }}

    toJSON() {{
        return this.value;
    }}

    static isValid(value) {{
        // Base type validation
        if (!{base_validation}) {{
            return false;
        }}

        // Business rule validation
{business_rule_validations}

        // Validation predicate checks
{validation_predicate_checks}

        return true;
    }}

    static from(value) {{
        return new {}(value);
    }}

    static validate(value) {{
        return this.isValid(value);
    }}
}}

// Export the semantic type
export {{ {} }};

"#,
                semantic_type.name,
                semantic_type.name,
                semantic_type.name,
                base_validation = self.generate_base_type_validation(&semantic_type.base_type),
                business_rule_validations = self.generate_business_rule_validations(&semantic_type.business_rules),
                validation_predicate_checks = self.generate_validation_predicate_checks(&semantic_type.validation_predicates),
                semantic_type.name,
                semantic_type.name,
            ));
        }

        Ok(output)
    }

    fn convert_base_type(&self, base_type: &crate::backends::PIRTypeInfo) -> String {
        match base_type {
            crate::backends::PIRTypeInfo::Primitive(prim) => {
                match prim {
                    crate::backends::PIRPrimitiveType::Integer { .. } => "number",
                    crate::backends::PIRPrimitiveType::Float { .. } => "number",
                    crate::backends::PIRPrimitiveType::Boolean => "boolean",
                    crate::backends::PIRPrimitiveType::String => "string",
                    crate::backends::PIRPrimitiveType::Unit => "undefined",
                }
            }
            _ => "object",
        }.to_string()
    }

    fn generate_base_type_validation(&self, base_type: &crate::backends::PIRTypeInfo) -> String {
        match base_type {
            crate::backends::PIRTypeInfo::Primitive(prim) => {
                match prim {
                    crate::backends::PIRPrimitiveType::Integer { .. } => "typeof value === 'number' && Number.isInteger(value)",
                    crate::backends::PIRPrimitiveType::Float { .. } => "typeof value === 'number'",
                    crate::backends::PIRPrimitiveType::Boolean => "typeof value === 'boolean'",
                    crate::backends::PIRPrimitiveType::String => "typeof value === 'string'",
                    crate::backends::PIRPrimitiveType::Unit => "value === undefined",
                }
            }
            _ => "typeof value === 'object' && value !== null",
        }.to_string()
    }

    fn generate_business_rule_validations(&self, business_rules: &[BusinessRule]) -> String {
        if !self.config.enable_business_rules || business_rules.is_empty() {
            return "        // No business rules to validate".to_string();
        }

        business_rules.iter()
            .map(|rule| format!(
                r#"        // Business rule: {} - {}
        if (!this.validateBusinessRule_{}_{}(value)) {{
            return false;
        }}"#,
                rule.name,
                rule.description,
                rule.name.replace(" ", "_").replace("-", "_").to_lowercase(),
                rule.name.len() // Add length to ensure uniqueness
            ))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_validation_predicate_checks(&self, predicates: &[ValidationPredicate]) -> String {
        if predicates.is_empty() {
            return "        // No validation predicates to check".to_string();
        }

        predicates.iter()
            .map(|pred| format!(
                r#"        // Validation predicate: {}
        if (!this.validatePredicate_{}(value)) {{
            return false;
        }}"#,
                pred.name,
                pred.name.replace(" ", "_").replace("-", "_").to_lowercase()
            ))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Business rule generator for JavaScript backend
pub struct BusinessRuleGenerator {
    config: SemanticConfig,
}

impl BusinessRuleGenerator {
    pub fn new(config: SemanticConfig) -> Self {
        Self { config }
    }

    /// Generate business rule system
    pub fn generate_business_rule_system(&self, semantic_types: &[&PIRSemanticType]) -> JavaScriptResult<String> {
        if !self.config.enable_business_rules {
            return Ok(String::new());
        }

        let mut output = String::new();

        output.push_str(r#"
// === BUSINESS RULE SYSTEM ===

/**
 * Business Rule Engine for semantic type validation
 * Provides runtime enforcement of compile-time business rules
 */
class BusinessRuleEngine {
    constructor() {
        this.rules = new Map();
        this.validationCache = new Map();
        this.registerBuiltinRules();
    }

    registerBuiltinRules() {
"#);

        // Generate rule registrations for all semantic types
        for semantic_type in semantic_types {
            for rule in &semantic_type.business_rules {
                output.push_str(&format!(
                    r#"        this.registerRule('{}', '{}', {{
            name: '{}',
            description: '{}',
            expression: '{}',
            validator: this.createValidator_{}_{}.bind(this)
        }});
"#,
                    semantic_type.name,
                    rule.name,
                    rule.name,
                    rule.description,
                    rule.expression,
                    semantic_type.name.to_lowercase(),
                    rule.name.replace(" ", "_").replace("-", "_").to_lowercase()
                ));
            }
        }

        output.push_str(r#"    }

    registerRule(typeName, ruleName, ruleDefinition) {
        if (!this.rules.has(typeName)) {
            this.rules.set(typeName, new Map());
        }
        this.rules.get(typeName).set(ruleName, ruleDefinition);
    }

    validateBusinessRule(typeName, ruleName, value) {
        const cacheKey = `${typeName}:${ruleName}:${JSON.stringify(value)}`;
        
        if (this.validationCache.has(cacheKey)) {
            return this.validationCache.get(cacheKey);
        }

        const typeRules = this.rules.get(typeName);
        if (!typeRules) {
            throw new Error(`No rules registered for type: ${typeName}`);
        }

        const rule = typeRules.get(ruleName);
        if (!rule) {
            throw new Error(`No rule '${ruleName}' found for type '${typeName}'`);
        }

        const result = rule.validator(value);
        this.validationCache.set(cacheKey, result);
        return result;
    }

    validateAllRules(typeName, value) {
        const typeRules = this.rules.get(typeName);
        if (!typeRules) {
            return { valid: true, violations: [] };
        }

        const violations = [];
        for (const [ruleName, rule] of typeRules) {
            if (!rule.validator(value)) {
                violations.push({
                    rule: ruleName,
                    description: rule.description,
                    expression: rule.expression
                });
            }
        }

        return {
            valid: violations.length === 0,
            violations
        };
    }
"#);

        // Generate specific validator methods for each business rule
        for semantic_type in semantic_types {
            for rule in &semantic_type.business_rules {
                output.push_str(&format!(
                    r#"
    createValidator_{}_{} (value) {{
        // Business rule: {} - {}
        // Expression: {}
        try {{
            // TODO: Implement actual rule validation logic
            // For now, return true as placeholder
            return true;
        }} catch (error) {{
            console.error(`Business rule validation error for {} {}: ${{error.message}}`);
            return false;
        }}
    }}
"#,
                    semantic_type.name.to_lowercase(),
                    rule.name.replace(" ", "_").replace("-", "_").to_lowercase(),
                    rule.name,
                    rule.description,
                    rule.expression,
                    semantic_type.name,
                    rule.name
                ));
            }
        }

        output.push_str(r#"}

// Global business rule engine instance
export const businessRuleEngine = new BusinessRuleEngine();

// Export business rule validation utilities
export function validateBusinessRule(typeName, ruleName, value) {
    return businessRuleEngine.validateBusinessRule(typeName, ruleName, value);
}

export function validateAllBusinessRules(typeName, value) {
    return businessRuleEngine.validateAllRules(typeName, value);
}

"#);

        Ok(output)
    }
} 