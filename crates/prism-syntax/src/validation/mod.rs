//! Parsing validation and constraint checking.
//! 
//! This module provides validation of parsed syntax against Prism standards,
//! maintaining conceptual cohesion around "syntax validation and standards compliance".

pub mod validator;
pub mod rules;
pub mod diagnostics;

pub use validator::{Validator, ValidationResult, ValidationRule};
pub use rules::{ValidationRuleSet, RuleEngine, RuleType};
pub use diagnostics::{ValidationDiagnostic, DiagnosticSeverity, DiagnosticCode}; 