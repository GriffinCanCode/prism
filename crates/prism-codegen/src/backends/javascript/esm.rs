//! ES Modules (ESM) Generation for JavaScript Backend
//!
//! This module handles modern ES module generation with proper
//! imports, exports, and module resolution.

use super::{JavaScriptResult, JavaScriptError, JavaScriptTarget};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ESM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ESMConfig {
    /// Use ES modules instead of CommonJS
    pub use_esm: bool,
    /// Target JavaScript version for module features
    pub target: JavaScriptTarget,
    /// Enable tree shaking friendly exports
    pub tree_shaking_friendly: bool,
    /// Use dynamic imports where appropriate
    pub dynamic_imports: bool,
}

impl Default for ESMConfig {
    fn default() -> Self {
        Self {
            use_esm: true,
            target: JavaScriptTarget::ES2022,
            tree_shaking_friendly: true,
            dynamic_imports: true,
        }
    }
}

/// ES Module generator
pub struct ESMGenerator {
    config: ESMConfig,
    imports: Vec<ImportDeclaration>,
    exports: Vec<ExportDeclaration>,
}

impl ESMGenerator {
    pub fn new(config: ESMConfig) -> Self {
        Self {
            config,
            imports: Vec::new(),
            exports: Vec::new(),
        }
    }

    pub fn add_import(&mut self, import: ImportDeclaration) {
        self.imports.push(import);
    }

    pub fn add_export(&mut self, export: ExportDeclaration) {
        self.exports.push(export);
    }

    pub fn generate_imports(&self) -> JavaScriptResult<String> {
        if !self.config.use_esm {
            return self.generate_commonjs_imports();
        }

        let mut output = String::new();
        
        for import in &self.imports {
            match import {
                ImportDeclaration::Named { specifiers, source } => {
                    output.push_str(&format!(
                        "import {{ {} }} from '{}';\n",
                        specifiers.join(", "),
                        source
                    ));
                }
                ImportDeclaration::Default { name, source } => {
                    output.push_str(&format!("import {} from '{}';\n", name, source));
                }
                ImportDeclaration::Namespace { name, source } => {
                    output.push_str(&format!("import * as {} from '{}';\n", name, source));
                }
                ImportDeclaration::Side { source } => {
                    output.push_str(&format!("import '{}';\n", source));
                }
            }
        }

        Ok(output)
    }

    pub fn generate_exports(&self) -> JavaScriptResult<String> {
        if !self.config.use_esm {
            return self.generate_commonjs_exports();
        }

        let mut output = String::new();
        
        for export in &self.exports {
            match export {
                ExportDeclaration::Named { specifiers } => {
                    output.push_str(&format!(
                        "export {{ {} }};\n",
                        specifiers.join(", ")
                    ));
                }
                ExportDeclaration::Default { name } => {
                    output.push_str(&format!("export default {};\n", name));
                }
                ExportDeclaration::All { source } => {
                    output.push_str(&format!("export * from '{}';\n", source));
                }
                ExportDeclaration::Function { name, function_code } => {
                    output.push_str(&format!("export {};\n", function_code));
                }
            }
        }

        Ok(output)
    }

    fn generate_commonjs_imports(&self) -> JavaScriptResult<String> {
        let mut output = String::new();
        
        for import in &self.imports {
            match import {
                ImportDeclaration::Named { specifiers, source } => {
                    output.push_str(&format!(
                        "const {{ {} }} = require('{}');\n",
                        specifiers.join(", "),
                        source
                    ));
                }
                ImportDeclaration::Default { name, source } => {
                    output.push_str(&format!("const {} = require('{}');\n", name, source));
                }
                ImportDeclaration::Namespace { name, source } => {
                    output.push_str(&format!("const {} = require('{}');\n", name, source));
                }
                ImportDeclaration::Side { source } => {
                    output.push_str(&format!("require('{}');\n", source));
                }
            }
        }

        Ok(output)
    }

    fn generate_commonjs_exports(&self) -> JavaScriptResult<String> {
        let mut output = String::new();
        let mut export_object = HashMap::new();
        
        for export in &self.exports {
            match export {
                ExportDeclaration::Named { specifiers } => {
                    for spec in specifiers {
                        export_object.insert(spec.clone(), spec.clone());
                    }
                }
                ExportDeclaration::Default { name } => {
                    export_object.insert("default".to_string(), name.clone());
                }
                ExportDeclaration::Function { name, .. } => {
                    export_object.insert(name.clone(), name.clone());
                }
                _ => {} // Skip re-exports for CommonJS
            }
        }

        if !export_object.is_empty() {
            output.push_str("module.exports = {\n");
            for (key, value) in export_object {
                output.push_str(&format!("  {}: {},\n", key, value));
            }
            output.push_str("};\n");
        }

        Ok(output)
    }

    pub fn generate_dynamic_import(&self, module: &str, condition: Option<&str>) -> JavaScriptResult<String> {
        if !self.config.dynamic_imports {
            return Ok(format!("require('{}')", module));
        }

        let import_code = format!("import('{}')", module);
        
        if let Some(condition) = condition {
            Ok(format!(
                "if ({}) {{\n  const module = await {};\n  // Use module\n}}",
                condition, import_code
            ))
        } else {
            Ok(import_code)
        }
    }
}

/// Import declaration types
#[derive(Debug, Clone)]
pub enum ImportDeclaration {
    Named {
        specifiers: Vec<String>,
        source: String,
    },
    Default {
        name: String,
        source: String,
    },
    Namespace {
        name: String,
        source: String,
    },
    Side {
        source: String,
    },
}

/// Export declaration types
#[derive(Debug, Clone)]
pub enum ExportDeclaration {
    Named {
        specifiers: Vec<String>,
    },
    Default {
        name: String,
    },
    All {
        source: String,
    },
    Function {
        name: String,
        function_code: String,
    },
} 