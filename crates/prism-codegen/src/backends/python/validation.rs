//! Python Validation Module
//!
//! This module handles validation of generated Python code using modern tools
//! like mypy for type checking and ruff for linting and code quality.

use super::{PythonResult, PythonError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::process::Command;
use std::path::Path;

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable mypy validation
    pub enable_mypy: bool,
    /// Enable ruff linting
    pub enable_ruff: bool,
    /// Enable black formatting checks
    pub enable_black: bool,
    /// Enable bandit security checks
    pub enable_bandit: bool,
    /// Enable pylint checks
    pub enable_pylint: bool,
    /// Python version for validation
    pub python_version: String,
    /// Strict mode for type checking
    pub strict_mode: bool,
    /// Custom mypy configuration
    pub mypy_config: Option<String>,
    /// Custom ruff configuration
    pub ruff_config: Option<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_mypy: true,
            enable_ruff: true,
            enable_black: false,
            enable_bandit: true,
            enable_pylint: false,
            python_version: "3.12".to_string(),
            strict_mode: true,
            mypy_config: None,
            ruff_config: None,
        }
    }
}

/// Python validator with comprehensive validation capabilities
pub struct PythonValidator {
    config: ValidationConfig,
    validation_cache: HashMap<String, ValidationResult>,
}

impl PythonValidator {
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            validation_cache: HashMap::new(),
        }
    }

    /// Validate Python code comprehensively
    pub fn validate(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Basic syntax validation
        issues.extend(self.validate_syntax(code)?);
        
        // Type checking with mypy
        if self.config.enable_mypy {
            issues.extend(self.validate_with_mypy(code)?);
        }
        
        // Linting with ruff
        if self.config.enable_ruff {
            issues.extend(self.validate_with_ruff(code)?);
        }
        
        // Security checks with bandit
        if self.config.enable_bandit {
            issues.extend(self.validate_with_bandit(code)?);
        }
        
        // Code formatting checks
        if self.config.enable_black {
            issues.extend(self.validate_formatting(code)?);
        }
        
        // Semantic validation for Prism-specific patterns
        issues.extend(self.validate_prism_patterns(code)?);
        
        Ok(issues)
    }

    /// Validate Python syntax
    fn validate_syntax(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Check for basic syntax issues
        if code.is_empty() {
            issues.push("Empty code provided".to_string());
            return Ok(issues);
        }
        
        // Check for common syntax patterns
        let lines: Vec<&str> = code.lines().collect();
        for (line_num, line) in lines.iter().enumerate() {
            let line_number = line_num + 1;
            
            // Check for missing colons
            if line.trim_end().ends_with("if ") || 
               line.trim_end().ends_with("else ") ||
               line.trim_end().ends_with("elif ") ||
               line.trim_end().ends_with("for ") ||
               line.trim_end().ends_with("while ") ||
               line.trim_end().ends_with("def ") ||
               line.trim_end().ends_with("class ") {
                issues.push(format!("Line {}: Missing colon at end of statement", line_number));
            }
            
            // Check for inconsistent indentation (basic check)
            if line.starts_with('\t') && code.contains("    ") {
                issues.push(format!("Line {}: Mixed tabs and spaces for indentation", line_number));
            }
            
            // Check for trailing whitespace
            if line.ends_with(' ') || line.ends_with('\t') {
                issues.push(format!("Line {}: Trailing whitespace", line_number));
            }
        }
        
        Ok(issues)
    }

    /// Validate with mypy type checker
    fn validate_with_mypy(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Create a temporary file for mypy validation
        let temp_file = self.create_temp_file(code, "validation.py")?;
        
        // Build mypy command
        let mut cmd = Command::new("mypy");
        cmd.arg(&temp_file);
        cmd.arg("--python-version").arg(&self.config.python_version);
        
        if self.config.strict_mode {
            cmd.arg("--strict");
        }
        
        // Add custom config if provided
        if let Some(config_path) = &self.config.mypy_config {
            cmd.arg("--config-file").arg(config_path);
        }
        
        // Execute mypy
        match cmd.output() {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    
                    // Parse mypy output
                    for line in stderr.lines().chain(stdout.lines()) {
                        if !line.is_empty() && line.contains("error:") {
                            issues.push(format!("MyPy: {}", line));
                        }
                    }
                }
            }
            Err(_) => {
                // MyPy not available, add a warning
                issues.push("MyPy type checker not available - skipping type validation".to_string());
            }
        }
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);
        
        Ok(issues)
    }

    /// Validate with ruff linter
    fn validate_with_ruff(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Create a temporary file for ruff validation
        let temp_file = self.create_temp_file(code, "ruff_validation.py")?;
        
        // Build ruff command
        let mut cmd = Command::new("ruff");
        cmd.arg("check");
        cmd.arg(&temp_file);
        cmd.arg("--output-format=text");
        
        // Add custom config if provided
        if let Some(config_path) = &self.config.ruff_config {
            cmd.arg("--config").arg(config_path);
        }
        
        // Execute ruff
        match cmd.output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                // Parse ruff output
                for line in stdout.lines().chain(stderr.lines()) {
                    if !line.is_empty() && !line.starts_with("Found") {
                        issues.push(format!("Ruff: {}", line));
                    }
                }
            }
            Err(_) => {
                // Ruff not available, add a warning
                issues.push("Ruff linter not available - skipping lint validation".to_string());
            }
        }
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);
        
        Ok(issues)
    }

    /// Validate with bandit security checker
    fn validate_with_bandit(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Create a temporary file for bandit validation
        let temp_file = self.create_temp_file(code, "bandit_validation.py")?;
        
        // Build bandit command
        let mut cmd = Command::new("bandit");
        cmd.arg("-f").arg("txt");
        cmd.arg(&temp_file);
        
        // Execute bandit
        match cmd.output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                
                // Parse bandit output
                for line in stdout.lines() {
                    if line.contains(">> Issue:") || line.contains("Severity:") {
                        issues.push(format!("Bandit: {}", line.trim()));
                    }
                }
            }
            Err(_) => {
                // Bandit not available, add a warning
                issues.push("Bandit security checker not available - skipping security validation".to_string());
            }
        }
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);
        
        Ok(issues)
    }

    /// Validate code formatting
    fn validate_formatting(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Create a temporary file for black validation
        let temp_file = self.create_temp_file(code, "black_validation.py")?;
        
        // Build black command
        let mut cmd = Command::new("black");
        cmd.arg("--check");
        cmd.arg("--diff");
        cmd.arg(&temp_file);
        
        // Execute black
        match cmd.output() {
            Ok(output) => {
                if !output.status.success() {
                    issues.push("Black: Code formatting issues detected".to_string());
                    
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if !stdout.is_empty() {
                        issues.push(format!("Black diff:\n{}", stdout));
                    }
                }
            }
            Err(_) => {
                // Black not available, add a warning
                issues.push("Black formatter not available - skipping format validation".to_string());
            }
        }
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_file);
        
        Ok(issues)
    }

    /// Validate Prism-specific patterns
    fn validate_prism_patterns(&self, code: &str) -> PythonResult<Vec<String>> {
        let mut issues = Vec::new();
        
        // Check for required Prism imports
        if !code.contains("from prism_runtime import") && !code.contains("import prism_runtime") {
            issues.push("Prism: Missing prism_runtime import".to_string());
        }
        
        // Check for proper semantic type usage
        if code.contains("@dataclass") && !code.contains("def validate(") {
            issues.push("Prism: Semantic types should include validation methods".to_string());
        }
        
        // Check for proper async patterns
        if code.contains("async def") && !code.contains("await") {
            issues.push("Prism: Async functions should use await for effect tracking".to_string());
        }
        
        // Check for capability validation
        if code.contains("CapabilityManager") && !code.contains("validate_capabilities") {
            issues.push("Prism: CapabilityManager usage should include validation".to_string());
        }
        
        // Check for effect tracking
        if code.contains("EffectTracker") && !code.contains("track_effects") {
            issues.push("Prism: EffectTracker usage should include effect tracking".to_string());
        }
        
        // Check for proper error handling
        if code.contains("ValidationError") && !code.contains("try:") {
            issues.push("Prism: ValidationError usage should include proper exception handling".to_string());
        }
        
        Ok(issues)
    }

    /// Create a temporary file for validation
    fn create_temp_file(&self, code: &str, filename: &str) -> PythonResult<String> {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("prism_{}", filename));
        
        std::fs::write(&temp_file, code)
            .map_err(|e| PythonError::Validation {
                message: format!("Failed to create temp file: {}", e),
            })?;
        
        Ok(temp_file.to_string_lossy().to_string())
    }

    /// Get validation summary
    pub fn get_validation_summary(&self, issues: &[String]) -> ValidationSummary {
        let mut summary = ValidationSummary::default();
        
        for issue in issues {
            if issue.starts_with("MyPy:") {
                summary.mypy_issues += 1;
            } else if issue.starts_with("Ruff:") {
                summary.ruff_issues += 1;
            } else if issue.starts_with("Bandit:") {
                summary.security_issues += 1;
            } else if issue.starts_with("Black:") {
                summary.formatting_issues += 1;
            } else if issue.starts_with("Prism:") {
                summary.prism_issues += 1;
            } else {
                summary.syntax_issues += 1;
            }
        }
        
        summary.total_issues = issues.len();
        summary
    }
}

/// Validation result summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_issues: usize,
    pub syntax_issues: usize,
    pub mypy_issues: usize,
    pub ruff_issues: usize,
    pub security_issues: usize,
    pub formatting_issues: usize,
    pub prism_issues: usize,
}

/// Validation result for caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub issues: Vec<String>,
    pub summary: ValidationSummary,
    pub timestamp: std::time::SystemTime,
}

/// Linting integration for external tools
pub struct LintingIntegration {
    config: ValidationConfig,
}

impl LintingIntegration {
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Check if external tools are available
    pub fn check_tool_availability(&self) -> HashMap<String, bool> {
        let mut availability = HashMap::new();
        
        if self.config.enable_mypy {
            availability.insert("mypy".to_string(), self.is_tool_available("mypy"));
        }
        
        if self.config.enable_ruff {
            availability.insert("ruff".to_string(), self.is_tool_available("ruff"));
        }
        
        if self.config.enable_bandit {
            availability.insert("bandit".to_string(), self.is_tool_available("bandit"));
        }
        
        if self.config.enable_black {
            availability.insert("black".to_string(), self.is_tool_available("black"));
        }
        
        availability
    }

    /// Check if a specific tool is available
    fn is_tool_available(&self, tool: &str) -> bool {
        Command::new(tool)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Generate configuration files for external tools
    pub fn generate_config_files(&self, output_dir: &Path) -> PythonResult<()> {
        // Generate mypy.ini
        if self.config.enable_mypy {
            self.generate_mypy_config(output_dir)?;
        }
        
        // Generate pyproject.toml with ruff config
        if self.config.enable_ruff {
            self.generate_ruff_config(output_dir)?;
        }
        
        // Generate .bandit config
        if self.config.enable_bandit {
            self.generate_bandit_config(output_dir)?;
        }
        
        Ok(())
    }

    /// Generate mypy configuration
    fn generate_mypy_config(&self, output_dir: &Path) -> PythonResult<()> {
        let config_content = format!(
            r#"[mypy]
python_version = {}
strict = {}
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True

# Prism-specific modules
[mypy-prism_runtime.*]
ignore_missing_imports = True

[mypy-uuid]
ignore_missing_imports = True
"#,
            self.config.python_version,
            self.config.strict_mode
        );
        
        let config_path = output_dir.join("mypy.ini");
        std::fs::write(config_path, config_content)
            .map_err(|e| PythonError::Validation {
                message: format!("Failed to write mypy config: {}", e),
            })?;
        
        Ok(())
    }

    /// Generate ruff configuration
    fn generate_ruff_config(&self, output_dir: &Path) -> PythonResult<()> {
        let config_content = r#"[tool.ruff]
target-version = "py312"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # imported but unused

[tool.ruff.isort]
known-first-party = ["prism_runtime"]
"#;
        
        let config_path = output_dir.join("ruff.toml");
        std::fs::write(config_path, config_content)
            .map_err(|e| PythonError::Validation {
                message: format!("Failed to write ruff config: {}", e),
            })?;
        
        Ok(())
    }

    /// Generate bandit configuration
    fn generate_bandit_config(&self, output_dir: &Path) -> PythonResult<()> {
        let config_content = r#"[bandit]
exclude_dirs = ["tests", "test"]
skips = ["B101"]  # Skip assert_used test

[bandit.assert_used]
skips = ["*_test.py", "test_*.py"]
"#;
        
        let config_path = output_dir.join(".bandit");
        std::fs::write(config_path, config_content)
            .map_err(|e| PythonError::Validation {
                message: format!("Failed to write bandit config: {}", e),
            })?;
        
        Ok(())
    }
} 