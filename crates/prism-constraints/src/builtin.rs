//! Built-in constraint validators
//!
//! This module provides implementations of common constraint validators
//! that are used throughout the Prism type system.

use crate::{
    ConstraintValue, ConstraintResult, ConstraintValidator, ValidationContext, ValidationResult,
    ValidationError,
};
use regex::Regex;
use std::collections::HashMap;

/// Range constraint validator
#[derive(Debug)]
pub struct RangeValidator {
    /// Minimum value (inclusive)
    pub min: Option<ConstraintValue>,
    /// Maximum value (inclusive)
    pub max: Option<ConstraintValue>,
    /// Whether the range is inclusive
    pub inclusive: bool,
}

impl RangeValidator {
    /// Create a new range validator
    pub fn new(min: Option<ConstraintValue>, max: Option<ConstraintValue>, inclusive: bool) -> Self {
        Self { min, max, inclusive }
    }

    /// Create a minimum-only range validator
    pub fn min_only(min: ConstraintValue, inclusive: bool) -> Self {
        Self::new(Some(min), None, inclusive)
    }

    /// Create a maximum-only range validator
    pub fn max_only(max: ConstraintValue, inclusive: bool) -> Self {
        Self::new(None, Some(max), inclusive)
    }

    /// Create a full range validator
    pub fn range(min: ConstraintValue, max: ConstraintValue, inclusive: bool) -> Self {
        Self::new(Some(min), Some(max), inclusive)
    }

    /// Compare two constraint values
    fn compare_values(&self, left: &ConstraintValue, right: &ConstraintValue) -> Option<std::cmp::Ordering> {
        match (left, right) {
            (ConstraintValue::Integer(a), ConstraintValue::Integer(b)) => Some(a.cmp(b)),
            (ConstraintValue::Float(a), ConstraintValue::Float(b)) => a.partial_cmp(b),
            (ConstraintValue::Integer(a), ConstraintValue::Float(b)) => (*a as f64).partial_cmp(b),
            (ConstraintValue::Float(a), ConstraintValue::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (ConstraintValue::String(a), ConstraintValue::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}

impl ConstraintValidator for RangeValidator {
    fn validate(&self, value: &ConstraintValue, _context: &ValidationContext) -> ConstraintResult<ValidationResult> {
        let mut result = ValidationResult::success();

        // Check minimum bound
        if let Some(ref min_val) = self.min {
            if let Some(ordering) = self.compare_values(value, min_val) {
                let valid = if self.inclusive {
                    ordering != std::cmp::Ordering::Less
                } else {
                    ordering == std::cmp::Ordering::Greater
                };

                if !valid {
                    let error = ValidationError::new(
                        format!(
                            "Value {:?} is below minimum {}{}",
                            value,
                            if self.inclusive { "" } else { "(exclusive) " },
                            format!("{:?}", min_val)
                        ),
                        "range",
                    )
                    .with_expected(format!("{}{:?}", if self.inclusive { ">= " } else { "> " }, min_val))
                    .with_actual(format!("{:?}", value))
                    .with_suggestion("Increase the value to meet the minimum requirement");

                    result.add_error(error);
                }
            } else {
                let error = ValidationError::new(
                    format!("Cannot compare {:?} with minimum value {:?}", value, min_val),
                    "range",
                )
                .with_suggestion("Ensure the value type is compatible with the range bounds");

                result.add_error(error);
            }
        }

        // Check maximum bound
        if let Some(ref max_val) = self.max {
            if let Some(ordering) = self.compare_values(value, max_val) {
                let valid = if self.inclusive {
                    ordering != std::cmp::Ordering::Greater
                } else {
                    ordering == std::cmp::Ordering::Less
                };

                if !valid {
                    let error = ValidationError::new(
                        format!(
                            "Value {:?} is above maximum {}{}",
                            value,
                            if self.inclusive { "" } else { "(exclusive) " },
                            format!("{:?}", max_val)
                        ),
                        "range",
                    )
                    .with_expected(format!("{}{:?}", if self.inclusive { "<= " } else { "< " }, max_val))
                    .with_actual(format!("{:?}", value))
                    .with_suggestion("Decrease the value to meet the maximum requirement");

                    result.add_error(error);
                }
            } else {
                let error = ValidationError::new(
                    format!("Cannot compare {:?} with maximum value {:?}", value, max_val),
                    "range",
                )
                .with_suggestion("Ensure the value type is compatible with the range bounds");

                result.add_error(error);
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "range"
    }

    fn description(&self) -> &str {
        "Validates that a value falls within a specified range"
    }

    fn constraint_type(&self) -> &str {
        "range"
    }

    fn examples(&self) -> Vec<ConstraintValue> {
        let mut examples = Vec::new();
        
        if let Some(ref min_val) = self.min {
            examples.push(min_val.clone());
        }
        
        if let Some(ref max_val) = self.max {
            examples.push(max_val.clone());
        }
        
        // Add a middle value if we have both bounds
        if let (Some(ConstraintValue::Integer(min)), Some(ConstraintValue::Integer(max))) = 
            (&self.min, &self.max) {
            examples.push(ConstraintValue::Integer((min + max) / 2));
        }
        
        examples
    }
}

/// Pattern constraint validator using regular expressions
#[derive(Debug)]
pub struct PatternValidator {
    /// The regex pattern
    pattern: Regex,
    /// The original pattern string
    pattern_str: String,
    /// Examples of valid values
    examples: Vec<String>,
}

impl PatternValidator {
    /// Create a new pattern validator
    pub fn new(pattern: &str) -> ConstraintResult<Self> {
        let regex = Regex::new(pattern).map_err(|e| {
            crate::ConstraintError::InvalidRegex {
                location: prism_common::span::Span::dummy(),
                pattern: pattern.to_string(),
                regex_error: e.to_string(),
            }
        })?;

        Ok(Self {
            pattern: regex,
            pattern_str: pattern.to_string(),
            examples: Vec::new(),
        })
    }

    /// Create a pattern validator with examples
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }

    /// Create a pattern validator for email addresses
    pub fn email() -> ConstraintResult<Self> {
        Ok(Self::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")?
            .with_examples(vec![
                "user@example.com".to_string(),
                "test.email+tag@domain.co.uk".to_string(),
            ]))
    }

    /// Create a pattern validator for E.164 phone numbers
    pub fn phone_e164() -> ConstraintResult<Self> {
        Ok(Self::new(r"^\+[1-9]\d{1,14}$")?
            .with_examples(vec![
                "+1234567890".to_string(),
                "+447700123456".to_string(),
            ]))
    }

    /// Create a pattern validator for UUIDs
    pub fn uuid() -> ConstraintResult<Self> {
        Ok(Self::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")?
            .with_examples(vec![
                "550e8400-e29b-41d4-a716-446655440000".to_string(),
                "6ba7b810-9dad-11d1-80b4-00c04fd430c8".to_string(),
            ]))
    }
}

impl ConstraintValidator for PatternValidator {
    fn validate(&self, value: &ConstraintValue, _context: &ValidationContext) -> ConstraintResult<ValidationResult> {
        let mut result = ValidationResult::success();

        if let Some(string_val) = value.as_string() {
            if !self.pattern.is_match(string_val) {
                let error = ValidationError::new(
                    format!("Value '{}' does not match pattern '{}'", string_val, self.pattern_str),
                    "pattern",
                )
                .with_expected(format!("string matching pattern: {}", self.pattern_str))
                .with_actual(string_val.to_string())
                .with_suggestion(format!("Ensure the value matches the pattern: {}", self.pattern_str));

                // Add examples if available
                let error = if !self.examples.is_empty() {
                    error.with_suggestion(format!("Examples of valid values: {}", self.examples.join(", ")))
                } else {
                    error
                };

                result.add_error(error);
            }
        } else {
            let error = ValidationError::new(
                format!("Pattern validation requires a string value, found {}", value.type_name()),
                "pattern",
            )
            .with_expected("string")
            .with_actual(value.type_name())
            .with_suggestion("Convert the value to a string before applying pattern validation");

            result.add_error(error);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "pattern"
    }

    fn description(&self) -> &str {
        "Validates that a string value matches a regular expression pattern"
    }

    fn constraint_type(&self) -> &str {
        "pattern"
    }

    fn examples(&self) -> Vec<ConstraintValue> {
        self.examples
            .iter()
            .map(|s| ConstraintValue::String(s.clone()))
            .collect()
    }
}

/// Length constraint validator
#[derive(Debug)]
pub struct LengthValidator {
    /// Minimum length
    pub min_length: Option<usize>,
    /// Maximum length
    pub max_length: Option<usize>,
}

impl LengthValidator {
    /// Create a new length validator
    pub fn new(min_length: Option<usize>, max_length: Option<usize>) -> Self {
        Self { min_length, max_length }
    }

    /// Create a minimum-only length validator
    pub fn min_only(min_length: usize) -> Self {
        Self::new(Some(min_length), None)
    }

    /// Create a maximum-only length validator
    pub fn max_only(max_length: usize) -> Self {
        Self::new(None, Some(max_length))
    }

    /// Create a fixed length validator
    pub fn exact(length: usize) -> Self {
        Self::new(Some(length), Some(length))
    }

    /// Get the length of a constraint value
    fn get_length(&self, value: &ConstraintValue) -> Option<usize> {
        match value {
            ConstraintValue::String(s) => Some(s.len()),
            ConstraintValue::Array(arr) => Some(arr.len()),
            _ => None,
        }
    }
}

impl ConstraintValidator for LengthValidator {
    fn validate(&self, value: &ConstraintValue, _context: &ValidationContext) -> ConstraintResult<ValidationResult> {
        let mut result = ValidationResult::success();

        if let Some(length) = self.get_length(value) {
            // Check minimum length
            if let Some(min_len) = self.min_length {
                if length < min_len {
                    let error = ValidationError::new(
                        format!("Length {} is below minimum {}", length, min_len),
                        "length",
                    )
                    .with_expected(format!(">= {}", min_len))
                    .with_actual(length.to_string())
                    .with_suggestion(format!("Increase the length to at least {}", min_len));

                    result.add_error(error);
                }
            }

            // Check maximum length
            if let Some(max_len) = self.max_length {
                if length > max_len {
                    let error = ValidationError::new(
                        format!("Length {} is above maximum {}", length, max_len),
                        "length",
                    )
                    .with_expected(format!("<= {}", max_len))
                    .with_actual(length.to_string())
                    .with_suggestion(format!("Reduce the length to at most {}", max_len));

                    result.add_error(error);
                }
            }
        } else {
            let error = ValidationError::new(
                format!("Length validation requires a string or array value, found {}", value.type_name()),
                "length",
            )
            .with_expected("string or array")
            .with_actual(value.type_name())
            .with_suggestion("Convert the value to a string or array before applying length validation");

            result.add_error(error);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "length"
    }

    fn description(&self) -> &str {
        "Validates the length of strings or arrays"
    }

    fn constraint_type(&self) -> &str {
        "length"
    }

    fn examples(&self) -> Vec<ConstraintValue> {
        let mut examples = Vec::new();
        
        if let Some(min_len) = self.min_length {
            examples.push(ConstraintValue::String("a".repeat(min_len)));
        }
        
        if let Some(max_len) = self.max_length {
            examples.push(ConstraintValue::String("a".repeat(max_len)));
        }
        
        examples
    }
}

/// Format validator for structured data validation
pub struct FormatValidator {
    /// Format name
    format_name: String,
    /// Validation function
    validator: Box<dyn Fn(&str) -> bool + Send + Sync>,
    /// Description of the format
    description: String,
    /// Examples of valid values
    examples: Vec<String>,
}

impl std::fmt::Debug for FormatValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FormatValidator")
            .field("format_name", &self.format_name)
            .field("validator", &"<function>")
            .field("description", &self.description)
            .field("examples", &self.examples)
            .finish()
    }
}

impl FormatValidator {
    /// Create a new format validator
    pub fn new<F>(
        format_name: impl Into<String>,
        validator: F,
        description: impl Into<String>,
    ) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        Self {
            format_name: format_name.into(),
            validator: Box::new(validator),
            description: description.into(),
            examples: Vec::new(),
        }
    }

    /// Add examples to the validator
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }

    /// ISO 8601 date format validator
    pub fn iso8601_date() -> Self {
        Self::new(
            "ISO8601_DATE",
            |s| {
                // Simple ISO 8601 date validation (YYYY-MM-DD)
                if s.len() != 10 {
                    return false;
                }
                let parts: Vec<&str> = s.split('-').collect();
                if parts.len() != 3 {
                    return false;
                }
                
                // Check year (4 digits)
                if parts[0].len() != 4 || !parts[0].chars().all(|c| c.is_ascii_digit()) {
                    return false;
                }
                
                // Check month (2 digits, 01-12)
                if parts[1].len() != 2 || !parts[1].chars().all(|c| c.is_ascii_digit()) {
                    return false;
                }
                if let Ok(month) = parts[1].parse::<u32>() {
                    if month < 1 || month > 12 {
                        return false;
                    }
                } else {
                    return false;
                }
                
                // Check day (2 digits, 01-31)
                if parts[2].len() != 2 || !parts[2].chars().all(|c| c.is_ascii_digit()) {
                    return false;
                }
                if let Ok(day) = parts[2].parse::<u32>() {
                    if day < 1 || day > 31 {
                        return false;
                    }
                } else {
                    return false;
                }
                
                true
            },
            "Value must be in ISO 8601 date format (YYYY-MM-DD)",
        ).with_examples(vec!["2023-12-25".to_string(), "2024-01-01".to_string()])
    }

    /// URL format validator
    pub fn url() -> Self {
        Self::new(
            "URL",
            |s| {
                // Simple URL validation
                s.starts_with("http://") || s.starts_with("https://") || s.starts_with("ftp://")
            },
            "Value must be a valid URL",
        ).with_examples(vec![
            "https://example.com".to_string(),
            "http://localhost:8080/path".to_string(),
        ])
    }
}

impl ConstraintValidator for FormatValidator {
    fn validate(&self, value: &ConstraintValue, _context: &ValidationContext) -> ConstraintResult<ValidationResult> {
        let mut result = ValidationResult::success();

        if let Some(string_val) = value.as_string() {
            if !(self.validator)(string_val) {
                let error = ValidationError::new(
                    format!("{}: '{}'", self.description, string_val),
                    "format",
                )
                .with_expected(format!("{} format", self.format_name))
                .with_actual(string_val.to_string())
                .with_suggestion(format!("Ensure the value is in {} format", self.format_name));

                // Add examples if available
                let error = if !self.examples.is_empty() {
                    error.with_suggestion(format!("Examples of valid values: {}", self.examples.join(", ")))
                } else {
                    error
                };

                result.add_error(error);
            }
        } else {
            let error = ValidationError::new(
                format!("Format validation requires a string value, found {}", value.type_name()),
                "format",
            )
            .with_expected("string")
            .with_actual(value.type_name())
            .with_suggestion("Convert the value to a string before applying format validation");

            result.add_error(error);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.format_name
    }

    fn description(&self) -> &str {
        "Validates that a string value matches a specific format"
    }

    fn constraint_type(&self) -> &str {
        "format"
    }

    fn examples(&self) -> Vec<ConstraintValue> {
        self.examples
            .iter()
            .map(|s| ConstraintValue::String(s.clone()))
            .collect()
    }
}

/// Registry of built-in constraint validators
#[derive(Debug)]
pub struct BuiltinValidatorRegistry {
    validators: HashMap<String, Box<dyn ConstraintValidator>>,
}

impl BuiltinValidatorRegistry {
    /// Create a new registry with all built-in validators
    pub fn new() -> Self {
        let mut registry = Self {
            validators: HashMap::new(),
        };
        
        registry.register_builtin_validators();
        registry
    }

    /// Register all built-in validators
    fn register_builtin_validators(&mut self) {
        // Register common pattern validators
        if let Ok(email_validator) = PatternValidator::email() {
            self.validators.insert("email".to_string(), Box::new(email_validator));
        }
        
        if let Ok(phone_validator) = PatternValidator::phone_e164() {
            self.validators.insert("phone_e164".to_string(), Box::new(phone_validator));
        }
        
        if let Ok(uuid_validator) = PatternValidator::uuid() {
            self.validators.insert("uuid".to_string(), Box::new(uuid_validator));
        }

        // Register format validators
        self.validators.insert("iso8601_date".to_string(), Box::new(FormatValidator::iso8601_date()));
        self.validators.insert("url".to_string(), Box::new(FormatValidator::url()));
    }

    /// Get a validator by name
    pub fn get(&self, name: &str) -> Option<&dyn ConstraintValidator> {
        self.validators.get(name).map(|v| v.as_ref())
    }

    /// Register a custom validator
    pub fn register(&mut self, name: impl Into<String>, validator: Box<dyn ConstraintValidator>) {
        self.validators.insert(name.into(), validator);
    }

    /// List all available validator names
    pub fn list_validators(&self) -> Vec<&str> {
        self.validators.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for BuiltinValidatorRegistry {
    fn default() -> Self {
        Self::new()
    }
} 