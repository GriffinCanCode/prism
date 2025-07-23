//! Context Extraction
//!
//! This module provides utilities for extracting contextual information
//! from Prism programs that can be used by external AI tools.

use crate::AIIntegrationError;
use async_trait::async_trait;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// Trait for extracting context from various sources
#[async_trait]
pub trait ContextExtractor: Send + Sync {
    /// Extract context from a project
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError>;
    
    /// Get the name of this context extractor
    fn name(&self) -> &str;
    
    /// Get the priority of this extractor (higher numbers run first)
    fn priority(&self) -> u32 { 0 }
}

/// Context extracted from a project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContext {
    /// Source of the context
    pub source: String,
    /// Context type
    pub context_type: ContextType,
    /// Extracted data
    pub data: ContextData,
    /// Confidence in the extraction (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of context that can be extracted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextType {
    /// Project structure and organization
    ProjectStructure,
    /// Code patterns and idioms
    CodePatterns,
    /// Business domain information
    BusinessDomain,
    /// Architectural patterns
    ArchitecturalPatterns,
    /// Dependencies and relationships
    Dependencies,
    /// Documentation and comments
    Documentation,
    /// Configuration and settings
    Configuration,
}

/// Context data extracted from various sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextData {
    /// Textual description
    Text(String),
    /// Structured data
    Structured(serde_json::Value),
    /// List of items
    List(Vec<String>),
    /// Key-value pairs
    KeyValue(std::collections::HashMap<String, String>),
}

/// Project structure context extractor
#[derive(Debug)]
pub struct ProjectStructureExtractor;

impl ProjectStructureExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for ProjectStructureExtractor {
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        let mut structure_data = std::collections::HashMap::new();
        
        // Analyze directory structure
        let directory_structure = self.analyze_directory_structure(project_root).await?;
        structure_data.insert("directories".to_string(), directory_structure);
        
        // Detect project type
        let project_type = self.detect_project_type(project_root).await?;
        structure_data.insert("project_type".to_string(), project_type);
        
        // Count files by type
        let file_counts = self.count_files_by_type(project_root).await?;
        structure_data.insert("file_counts".to_string(), file_counts);
        
        // Detect build system
        let build_system = self.detect_build_system(project_root).await?;
        structure_data.insert("build_system".to_string(), build_system);
        
        // Calculate project size metrics
        let size_metrics = self.calculate_size_metrics(project_root).await?;
        structure_data.insert("size_metrics".to_string(), size_metrics);
        
        Ok(ExtractedContext {
            source: "project_structure".to_string(),
            context_type: ContextType::ProjectStructure,
            data: ContextData::KeyValue(structure_data),
            confidence: 0.95,
        })
    }
    
    fn name(&self) -> &str {
        "project_structure"
    }
    
    fn priority(&self) -> u32 {
        100 // High priority - project structure is fundamental
    }
}

impl ProjectStructureExtractor {
    /// Analyze directory structure and return key directories
    async fn analyze_directory_structure(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let mut directories = Vec::new();
        
        let mut entries = tokio::fs::read_dir(project_root).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    // Skip hidden directories and common build artifacts
                    if !dir_name.starts_with('.') && !["target", "node_modules", "dist", "build"].contains(&dir_name) {
                        directories.push(dir_name.to_string());
                    }
                }
            }
        }
        
        Ok(directories.join(", "))
    }
    
    /// Detect project type based on files and structure
    async fn detect_project_type(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        // Check for Rust project
        if project_root.join("Cargo.toml").exists() {
            return Ok("Rust".to_string());
        }
        
        // Check for Node.js project
        if project_root.join("package.json").exists() {
            return Ok("Node.js".to_string());
        }
        
        // Check for Python project
        if project_root.join("setup.py").exists() || project_root.join("pyproject.toml").exists() {
            return Ok("Python".to_string());
        }
        
        // Check for Java project
        if project_root.join("pom.xml").exists() || project_root.join("build.gradle").exists() {
            return Ok("Java".to_string());
        }
        
        // Check for C++ project
        if project_root.join("CMakeLists.txt").exists() || project_root.join("Makefile").exists() {
            return Ok("C++".to_string());
        }
        
        Ok("Unknown".to_string())
    }
    
    /// Count files by extension/type
    async fn count_files_by_type(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let mut file_counts = std::collections::HashMap::new();
        
        self.count_files_recursive(project_root, &mut file_counts).await?;
        
        let counts_str = file_counts
            .iter()
            .map(|(ext, count)| format!("{}: {}", ext, count))
            .collect::<Vec<_>>()
            .join(", ");
        
        Ok(counts_str)
    }
    
    /// Recursively count files by extension
    async fn count_files_recursive(
        &self,
        dir: &PathBuf,
        counts: &mut std::collections::HashMap<String, usize>,
    ) -> Result<(), AIIntegrationError> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_file() {
                let extension = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("no_extension")
                    .to_string();
                
                *counts.entry(extension).or_insert(0) += 1;
            } else if path.is_dir() {
                let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Skip common directories that shouldn't be counted
                if !["target", "node_modules", ".git", "dist", "build"].contains(&dir_name) 
                   && !dir_name.starts_with('.') {
                                          Box::pin(self.count_files_recursive(&path, counts)).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect build system
    async fn detect_build_system(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        if project_root.join("Cargo.toml").exists() {
            Ok("Cargo".to_string())
        } else if project_root.join("package.json").exists() {
            Ok("npm/yarn".to_string())
        } else if project_root.join("pom.xml").exists() {
            Ok("Maven".to_string())
        } else if project_root.join("build.gradle").exists() {
            Ok("Gradle".to_string())
        } else if project_root.join("CMakeLists.txt").exists() {
            Ok("CMake".to_string())
        } else if project_root.join("Makefile").exists() {
            Ok("Make".to_string())
        } else {
            Ok("None detected".to_string())
        }
    }
    
    /// Calculate basic project size metrics
    async fn calculate_size_metrics(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let mut total_files = 0;
        let mut total_size = 0u64;
        
        self.calculate_size_recursive(project_root, &mut total_files, &mut total_size).await?;
        
        Ok(format!("files: {}, size: {} bytes", total_files, total_size))
    }
    
    /// Recursively calculate size metrics
    async fn calculate_size_recursive(
        &self,
        dir: &PathBuf,
        file_count: &mut usize,
        total_size: &mut u64,
    ) -> Result<(), AIIntegrationError> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_file() {
                *file_count += 1;
                if let Ok(metadata) = entry.metadata().await {
                    *total_size += metadata.len();
                }
            } else if path.is_dir() {
                let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !["target", "node_modules", ".git", "dist", "build"].contains(&dir_name) 
                   && !dir_name.starts_with('.') {
                                          Box::pin(self.calculate_size_recursive(&path, file_count, total_size)).await?;
                }
            }
        }
        
        Ok(())
    }
}

/// Code patterns context extractor
#[derive(Debug)]
pub struct CodePatternsExtractor;

impl CodePatternsExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for CodePatternsExtractor {
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        let mut patterns = Vec::new();
        
        // Analyze code files for patterns
        patterns.extend(self.analyze_rust_patterns(project_root).await?);
        patterns.extend(self.analyze_general_patterns(project_root).await?);
        patterns.extend(self.analyze_architectural_patterns(project_root).await?);
        
        Ok(ExtractedContext {
            source: "code_patterns".to_string(),
            context_type: ContextType::CodePatterns,
            data: ContextData::List(patterns),
            confidence: 0.8,
        })
    }
    
    fn name(&self) -> &str {
        "code_patterns"
    }
    
    fn priority(&self) -> u32 {
        50
    }
}

impl CodePatternsExtractor {
    /// Analyze Rust-specific patterns
    async fn analyze_rust_patterns(&self, project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        let mut patterns = Vec::new();
        
        // Look for Rust source files
        if let Ok(entries) = self.find_rust_files(project_root).await {
            for file_path in entries {
                if let Ok(content) = tokio::fs::read_to_string(&file_path).await {
                    // Detect common Rust patterns
                    if content.contains("impl ") {
                        patterns.push("Implementation blocks".to_string());
                    }
                    if content.contains("trait ") {
                        patterns.push("Trait definitions".to_string());
                    }
                    if content.contains("match ") {
                        patterns.push("Pattern matching".to_string());
                    }
                    if content.contains("Result<") || content.contains("Option<") {
                        patterns.push("Error handling with Result/Option".to_string());
                    }
                    if content.contains("async fn") || content.contains(".await") {
                        patterns.push("Async/await patterns".to_string());
                    }
                    if content.contains("macro_rules!") || content.contains("#[derive(") {
                        patterns.push("Macro usage".to_string());
                    }
                    if content.contains("Box<") || content.contains("Rc<") || content.contains("Arc<") {
                        patterns.push("Smart pointers".to_string());
                    }
                }
            }
        }
        
        // Remove duplicates
        patterns.sort();
        patterns.dedup();
        
        Ok(patterns)
    }
    
    /// Analyze general programming patterns
    async fn analyze_general_patterns(&self, project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        let mut patterns = Vec::new();
        
        // Look for common file patterns
        if project_root.join("tests").exists() || project_root.join("test").exists() {
            patterns.push("Test organization".to_string());
        }
        
        if project_root.join("examples").exists() {
            patterns.push("Example code organization".to_string());
        }
        
        if project_root.join("docs").exists() || project_root.join("doc").exists() {
            patterns.push("Documentation organization".to_string());
        }
        
        if project_root.join("src").exists() {
            patterns.push("Source code organization".to_string());
        }
        
        // Check for configuration patterns
        if project_root.join("config").exists() || project_root.join(".env").exists() {
            patterns.push("Configuration management".to_string());
        }
        
        Ok(patterns)
    }
    
    /// Analyze architectural patterns
    async fn analyze_architectural_patterns(&self, project_root: &PathBuf) -> Result<Vec<String>, AIIntegrationError> {
        let mut patterns = Vec::new();
        
        // Check for common architectural patterns based on directory structure
        let src_dir = project_root.join("src");
        if src_dir.exists() {
            if let Ok(mut entries) = tokio::fs::read_dir(&src_dir).await {
                let mut has_modules = false;
                let mut has_lib = false;
                let mut has_main = false;
                
                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        match name {
                            "lib.rs" => has_lib = true,
                            "main.rs" => has_main = true,
                            _ if path.is_dir() => has_modules = true,
                            _ => {}
                        }
                    }
                }
                
                if has_lib && has_main {
                    patterns.push("Library with binary".to_string());
                } else if has_lib {
                    patterns.push("Library crate".to_string());
                } else if has_main {
                    patterns.push("Binary crate".to_string());
                }
                
                if has_modules {
                    patterns.push("Modular architecture".to_string());
                }
            }
        }
        
        // Check for workspace pattern
        if project_root.join("Cargo.toml").exists() {
            if let Ok(content) = tokio::fs::read_to_string(project_root.join("Cargo.toml")).await {
                if content.contains("[workspace]") {
                    patterns.push("Cargo workspace".to_string());
                }
            }
        }
        
        Ok(patterns)
    }
    
    /// Find all Rust source files in the project
    async fn find_rust_files(&self, project_root: &PathBuf) -> Result<Vec<PathBuf>, AIIntegrationError> {
        let mut rust_files = Vec::new();
        self.find_rust_files_recursive(project_root, &mut rust_files).await?;
        Ok(rust_files)
    }
    
    /// Recursively find Rust files
    async fn find_rust_files_recursive(
        &self,
        dir: &PathBuf,
        rust_files: &mut Vec<PathBuf>,
    ) -> Result<(), AIIntegrationError> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                    if extension == "rs" {
                        rust_files.push(path);
                    }
                }
            } else if path.is_dir() {
                let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Skip target and other build directories
                if !["target", "node_modules", ".git", "dist", "build"].contains(&dir_name) 
                   && !dir_name.starts_with('.') {
                                          Box::pin(self.find_rust_files_recursive(&path, rust_files)).await?;
                }
            }
        }
        
        Ok(())
    }
}

/// Business domain context extractor
#[derive(Debug)]
pub struct BusinessDomainExtractor;

impl BusinessDomainExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for BusinessDomainExtractor {
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        let mut domain_data = std::collections::HashMap::new();
        
        // Analyze project name and description for domain hints
        let domain_hints = self.analyze_project_metadata(project_root).await?;
        domain_data.insert("domain_hints".to_string(), domain_hints);
        
        // Analyze code comments and documentation for business terms
        let business_terms = self.extract_business_terms(project_root).await?;
        domain_data.insert("business_terms".to_string(), business_terms);
        
        // Analyze directory and file names for domain indicators
        let domain_indicators = self.analyze_naming_patterns(project_root).await?;
        domain_data.insert("domain_indicators".to_string(), domain_indicators);
        
        // Infer primary domain
        let primary_domain = self.infer_primary_domain(&domain_data);
        domain_data.insert("primary_domain".to_string(), primary_domain);
        
        Ok(ExtractedContext {
            source: "business_domain".to_string(),
            context_type: ContextType::BusinessDomain,
            data: ContextData::KeyValue(domain_data),
            confidence: 0.7,
        })
    }
    
    fn name(&self) -> &str {
        "business_domain"
    }
    
    fn priority(&self) -> u32 {
        30
    }
}

impl BusinessDomainExtractor {
    /// Analyze project metadata for domain hints
    async fn analyze_project_metadata(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let mut hints = Vec::new();
        
        // Check Cargo.toml description
        let cargo_toml = project_root.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&cargo_toml).await {
                if let Ok(parsed) = content.parse::<toml::Value>() {
                    if let Some(description) = parsed.get("package")
                        .and_then(|p| p.get("description"))
                        .and_then(|d| d.as_str()) {
                        hints.extend(self.extract_domain_from_text(description));
                    }
                }
            }
        }
        
        // Check README for project description
        for readme_name in &["README.md", "readme.md", "README.txt", "readme.txt"] {
            let readme_path = project_root.join(readme_name);
            if readme_path.exists() {
                if let Ok(content) = tokio::fs::read_to_string(&readme_path).await {
                    // Take first few lines as they usually contain project description
                    let first_lines: String = content.lines().take(10).collect::<Vec<_>>().join(" ");
                    hints.extend(self.extract_domain_from_text(&first_lines));
                }
                break;
            }
        }
        
        Ok(hints.join(", "))
    }
    
    /// Extract business terms from code and documentation
    async fn extract_business_terms(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let mut terms = std::collections::HashSet::new();
        
        // Common business domain terms to look for
        let business_keywords = [
            "user", "customer", "account", "order", "payment", "invoice", "product", "service",
            "authentication", "authorization", "login", "signup", "profile", "dashboard",
            "analytics", "report", "metrics", "api", "database", "cache", "queue",
            "notification", "email", "message", "chat", "workflow", "process", "task",
            "inventory", "catalog", "search", "filter", "admin", "management", "configuration"
        ];
        
        // Scan source files for business terms
        if let Ok(rust_files) = self.find_source_files(project_root).await {
            for file_path in rust_files.iter().take(20) { // Limit to avoid performance issues
                if let Ok(content) = tokio::fs::read_to_string(file_path).await {
                    let content_lower = content.to_lowercase();
                    for keyword in &business_keywords {
                        if content_lower.contains(keyword) {
                            terms.insert(keyword.to_string());
                        }
                    }
                }
            }
        }
        
        let mut terms_vec: Vec<String> = terms.into_iter().collect();
        terms_vec.sort();
        Ok(terms_vec.join(", "))
    }
    
    /// Analyze naming patterns for domain indicators
    async fn analyze_naming_patterns(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let mut indicators = Vec::new();
        
        // Analyze directory names
        if let Ok(mut entries) = tokio::fs::read_dir(project_root).await {
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                        if let Some(domain) = self.classify_directory_name(dir_name) {
                            indicators.push(domain);
                        }
                    }
                }
            }
        }
        
        // Analyze src directory structure if it exists
        let src_dir = project_root.join("src");
        if src_dir.exists() {
            if let Ok(mut entries) = tokio::fs::read_dir(&src_dir).await {
                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                        if let Some(domain) = self.classify_module_name(name) {
                            indicators.push(domain);
                        }
                    }
                }
            }
        }
        
        indicators.sort();
        indicators.dedup();
        Ok(indicators.join(", "))
    }
    
    /// Extract domain hints from text
    fn extract_domain_from_text(&self, text: &str) -> Vec<String> {
        let mut domains = Vec::new();
        let text_lower = text.to_lowercase();
        
        let domain_patterns = [
            ("web", "Web Development"),
            ("api", "API Development"),
            ("database", "Data Management"),
            ("auth", "Authentication/Security"),
            ("payment", "Financial Services"),
            ("e-commerce", "E-commerce"),
            ("analytics", "Analytics/BI"),
            ("ml", "Machine Learning"),
            ("ai", "Artificial Intelligence"),
            ("game", "Gaming"),
            ("mobile", "Mobile Development"),
            ("desktop", "Desktop Applications"),
            ("cli", "Command Line Tools"),
            ("library", "Software Library"),
            ("framework", "Software Framework"),
        ];
        
        for (pattern, domain) in &domain_patterns {
            if text_lower.contains(pattern) {
                domains.push(domain.to_string());
            }
        }
        
        domains
    }
    
    /// Classify directory name for domain indicators
    fn classify_directory_name(&self, dir_name: &str) -> Option<String> {
        match dir_name.to_lowercase().as_str() {
            "api" | "apis" => Some("API Layer".to_string()),
            "web" | "frontend" | "ui" => Some("Web Frontend".to_string()),
            "backend" | "server" => Some("Backend Services".to_string()),
            "database" | "db" | "data" => Some("Data Layer".to_string()),
            "auth" | "authentication" => Some("Authentication".to_string()),
            "admin" | "management" => Some("Administration".to_string()),
            "tests" | "test" => Some("Testing".to_string()),
            "docs" | "documentation" => Some("Documentation".to_string()),
            "examples" | "samples" => Some("Examples".to_string()),
            "config" | "configuration" => Some("Configuration".to_string()),
            _ => None,
        }
    }
    
    /// Classify module name for domain indicators
    fn classify_module_name(&self, module_name: &str) -> Option<String> {
        match module_name.to_lowercase().as_str() {
            "user" | "users" => Some("User Management".to_string()),
            "auth" | "authentication" => Some("Authentication".to_string()),
            "api" => Some("API Layer".to_string()),
            "db" | "database" => Some("Database".to_string()),
            "config" | "configuration" => Some("Configuration".to_string()),
            "error" | "errors" => Some("Error Handling".to_string()),
            "util" | "utils" | "utilities" => Some("Utilities".to_string()),
            "model" | "models" => Some("Data Models".to_string()),
            "service" | "services" => Some("Business Services".to_string()),
            "controller" | "controllers" => Some("Controllers".to_string()),
            _ => None,
        }
    }
    
    /// Infer primary domain from collected data
    fn infer_primary_domain(&self, domain_data: &std::collections::HashMap<String, String>) -> String {
        // Simple heuristic: if we find specific indicators, classify accordingly
        let all_text = domain_data.values().map(|s| s.as_str()).collect::<Vec<_>>().join(" ").to_lowercase();
        
        if all_text.contains("web") || all_text.contains("api") || all_text.contains("frontend") {
            "Web Development".to_string()
        } else if all_text.contains("data") || all_text.contains("analytics") || all_text.contains("database") {
            "Data Processing".to_string()
        } else if all_text.contains("auth") || all_text.contains("security") {
            "Security/Authentication".to_string()
        } else if all_text.contains("game") {
            "Gaming".to_string()
        } else if all_text.contains("ml") || all_text.contains("ai") {
            "Machine Learning/AI".to_string()
        } else {
            "General Software Development".to_string()
        }
    }
    
    /// Find source files for analysis
    async fn find_source_files(&self, project_root: &PathBuf) -> Result<Vec<PathBuf>, AIIntegrationError> {
        let mut source_files = Vec::new();
        self.find_source_files_recursive(project_root, &mut source_files).await?;
        Ok(source_files)
    }
    
    /// Recursively find source files
    async fn find_source_files_recursive(
        &self,
        dir: &PathBuf,
        source_files: &mut Vec<PathBuf>,
    ) -> Result<(), AIIntegrationError> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                    if ["rs", "py", "js", "ts", "java", "cpp", "c", "go"].contains(&extension) {
                        source_files.push(path);
                    }
                }
            } else if path.is_dir() {
                let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !["target", "node_modules", ".git", "dist", "build"].contains(&dir_name) 
                   && !dir_name.starts_with('.') {
                                          Box::pin(self.find_source_files_recursive(&path, source_files)).await?;
                }
            }
        }
        
        Ok(())
    }
}

/// Dependencies context extractor
#[derive(Debug)]
pub struct DependenciesExtractor;

impl DependenciesExtractor {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContextExtractor for DependenciesExtractor {
    async fn extract_context(&self, project_root: &PathBuf) -> Result<ExtractedContext, AIIntegrationError> {
        let mut dependency_data = std::collections::HashMap::new();
        
        // Detect build system
        let build_system = self.detect_build_system(project_root).await?;
        dependency_data.insert("build_system".to_string(), build_system.clone());
        
        // Extract dependencies based on build system
        let dependencies = match build_system.as_str() {
            "Cargo" => self.extract_cargo_dependencies(project_root).await?,
            "npm/yarn" => self.extract_npm_dependencies(project_root).await?,
            "Maven" => self.extract_maven_dependencies(project_root).await?,
            "Gradle" => self.extract_gradle_dependencies(project_root).await?,
            _ => "No dependencies found".to_string(),
        };
        dependency_data.insert("dependencies".to_string(), dependencies);
        
        // Analyze dependency categories
        let categories = self.categorize_dependencies(&dependency_data).await?;
        dependency_data.insert("categories".to_string(), categories);
        
        // Calculate dependency metrics
        let metrics = self.calculate_dependency_metrics(&dependency_data);
        dependency_data.insert("metrics".to_string(), metrics);
        
        let confidence = if build_system == "None detected" { 0.3 } else { 0.9 };
        
        Ok(ExtractedContext {
            source: "dependencies".to_string(),
            context_type: ContextType::Dependencies,
            data: ContextData::KeyValue(dependency_data),
            confidence,
        })
    }
    
    fn name(&self) -> &str {
        "dependencies"
    }
    
    fn priority(&self) -> u32 {
        80
    }
}

impl DependenciesExtractor {
    /// Detect build system
    async fn detect_build_system(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        if project_root.join("Cargo.toml").exists() {
            Ok("Cargo".to_string())
        } else if project_root.join("package.json").exists() {
            Ok("npm/yarn".to_string())
        } else if project_root.join("pom.xml").exists() {
            Ok("Maven".to_string())
        } else if project_root.join("build.gradle").exists() {
            Ok("Gradle".to_string())
        } else if project_root.join("CMakeLists.txt").exists() {
            Ok("CMake".to_string())
        } else {
            Ok("None detected".to_string())
        }
    }
    
    /// Extract Cargo dependencies
    async fn extract_cargo_dependencies(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let cargo_toml = project_root.join("Cargo.toml");
        let content = tokio::fs::read_to_string(&cargo_toml).await?;
        
        if let Ok(parsed) = content.parse::<toml::Value>() {
            let mut deps = Vec::new();
            
            // Extract regular dependencies
            if let Some(dependencies) = parsed.get("dependencies").and_then(|d| d.as_table()) {
                for (name, _) in dependencies {
                    deps.push(format!("{} (runtime)", name));
                }
            }
            
            // Extract dev dependencies
            if let Some(dev_deps) = parsed.get("dev-dependencies").and_then(|d| d.as_table()) {
                for (name, _) in dev_deps {
                    deps.push(format!("{} (dev)", name));
                }
            }
            
            Ok(deps.join(", "))
        } else {
            Ok("Failed to parse Cargo.toml".to_string())
        }
    }
    
    /// Extract npm dependencies
    async fn extract_npm_dependencies(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let package_json = project_root.join("package.json");
        let content = tokio::fs::read_to_string(&package_json).await?;
        
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content) {
            let mut deps = Vec::new();
            
            if let Some(dependencies) = parsed.get("dependencies").and_then(|d| d.as_object()) {
                for (name, _) in dependencies {
                    deps.push(format!("{} (runtime)", name));
                }
            }
            
            if let Some(dev_deps) = parsed.get("devDependencies").and_then(|d| d.as_object()) {
                for (name, _) in dev_deps {
                    deps.push(format!("{} (dev)", name));
                }
            }
            
            Ok(deps.join(", "))
        } else {
            Ok("Failed to parse package.json".to_string())
        }
    }
    
    /// Extract Maven dependencies (simplified)
    async fn extract_maven_dependencies(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let pom_xml = project_root.join("pom.xml");
        if let Ok(content) = tokio::fs::read_to_string(&pom_xml).await {
            // Simple text-based extraction for demo purposes
            let dependency_count = content.matches("<dependency>").count();
            Ok(format!("Maven project with {} dependencies", dependency_count))
        } else {
            Ok("Failed to read pom.xml".to_string())
        }
    }
    
    /// Extract Gradle dependencies (simplified)
    async fn extract_gradle_dependencies(&self, project_root: &PathBuf) -> Result<String, AIIntegrationError> {
        let build_gradle = project_root.join("build.gradle");
        if let Ok(content) = tokio::fs::read_to_string(&build_gradle).await {
            // Simple text-based extraction for demo purposes
            let implementation_count = content.matches("implementation").count();
            let test_count = content.matches("testImplementation").count();
            Ok(format!("Gradle project with {} implementation and {} test dependencies", implementation_count, test_count))
        } else {
            Ok("Failed to read build.gradle".to_string())
        }
    }
    
    /// Categorize dependencies
    async fn categorize_dependencies(&self, dependency_data: &std::collections::HashMap<String, String>) -> Result<String, AIIntegrationError> {
        let empty_string = String::new();
        let dependencies = dependency_data.get("dependencies").unwrap_or(&empty_string);
        let mut categories = Vec::new();
        
        if dependencies.contains("serde") || dependencies.contains("json") {
            categories.push("Serialization");
        }
        if dependencies.contains("tokio") || dependencies.contains("async") {
            categories.push("Async Runtime");
        }
        if dependencies.contains("reqwest") || dependencies.contains("hyper") || dependencies.contains("axum") {
            categories.push("HTTP/Web");
        }
        if dependencies.contains("sqlx") || dependencies.contains("diesel") || dependencies.contains("mongodb") {
            categories.push("Database");
        }
        if dependencies.contains("clap") || dependencies.contains("structopt") {
            categories.push("CLI");
        }
        if dependencies.contains("test") || dependencies.contains("mock") {
            categories.push("Testing");
        }
        
        Ok(categories.join(", "))
    }
    
    /// Calculate basic dependency metrics
    fn calculate_dependency_metrics(&self, dependency_data: &std::collections::HashMap<String, String>) -> String {
        let empty_string = String::new();
        let dependencies = dependency_data.get("dependencies").unwrap_or(&empty_string);
        let dep_count = dependencies.matches(",").count() + if dependencies.is_empty() { 0 } else { 1 };
        let dev_count = dependencies.matches("(dev)").count();
        let runtime_count = dependencies.matches("(runtime)").count();
        
        format!("Total: {}, Runtime: {}, Dev: {}", dep_count, runtime_count, dev_count)
    }
}

/// Context aggregator that combines multiple extractors
pub struct ContextAggregator {
    extractors: Vec<Box<dyn ContextExtractor>>,
}

impl ContextAggregator {
    /// Create a new context aggregator
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
        }
    }
    
    /// Add a context extractor
    pub fn add_extractor(&mut self, extractor: Box<dyn ContextExtractor>) {
        self.extractors.push(extractor);
        // Sort by priority (highest first)
        self.extractors.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }
    
    /// Extract context using all registered extractors
    pub async fn extract_all_context(&self, project_root: &PathBuf) -> Result<Vec<ExtractedContext>, AIIntegrationError> {
        let mut contexts = Vec::new();
        
        for extractor in &self.extractors {
            match extractor.extract_context(project_root).await {
                Ok(context) => contexts.push(context),
                Err(e) => {
                    // Log error but continue with other extractors
                    eprintln!("Warning: Context extractor '{}' failed: {}", extractor.name(), e);
                }
            }
        }
        
        Ok(contexts)
    }
    
    /// Create a default context aggregator with common extractors
    pub fn with_default_extractors() -> Self {
        let mut aggregator = Self::new();
        aggregator.add_extractor(Box::new(ProjectStructureExtractor::new()));
        aggregator.add_extractor(Box::new(CodePatternsExtractor::new()));
        aggregator.add_extractor(Box::new(BusinessDomainExtractor::new()));
        aggregator.add_extractor(Box::new(DependenciesExtractor::new()));
        aggregator
    }
}

impl Default for ContextAggregator {
    fn default() -> Self {
        Self::with_default_extractors()
    }
}

impl Default for ProjectStructureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CodePatternsExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BusinessDomainExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DependenciesExtractor {
    fn default() -> Self {
        Self::new()
    }
} 