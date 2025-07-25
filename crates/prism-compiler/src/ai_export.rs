//! AI context export APIs
//!
//! This module provides APIs to export AI-readable semantic context and metadata
//! for external AI development tools and systems.

use crate::error::{CompilerError, CompilerResult};
use crate::semantic::{AIMetadata, SemanticDatabase, SemanticInfo};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// AI context export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format for general consumption
    Json,
    /// YAML format for human readability
    Yaml,
    /// XML format for structured data
    Xml,
    /// Custom binary format for efficiency
    Binary,
    /// OpenAPI specification
    OpenApi,
    /// GraphQL schema
    GraphQL,
}

/// AI context export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    /// Include source code in export
    pub include_source: bool,
    /// Include AST representation
    pub include_ast: bool,
    /// Include type information
    pub include_types: bool,
    /// Include semantic relationships
    pub include_relationships: bool,
    /// Include AI metadata
    pub include_ai_metadata: bool,
    /// Include performance metrics
    pub include_metrics: bool,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Output file path
    pub output_path: Option<PathBuf>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            include_source: true,
            include_ast: true,
            include_types: true,
            include_relationships: true,
            include_ai_metadata: true,
            include_metrics: false,
            compression_level: 6,
            output_path: None,
        }
    }
}

/// Exported AI context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIContext {
    /// Context metadata
    pub metadata: ContextMetadata,
    /// Project structure
    pub project: ProjectContext,
    /// Code modules
    pub modules: Vec<ModuleContext>,
    /// Type system information
    pub types: TypeSystemContext,
    /// Semantic relationships
    pub relationships: RelationshipGraph,
    /// AI-specific metadata
    pub ai_metadata: HashMap<String, AIMetadata>,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
}

/// Context metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Export timestamp
    pub timestamp: String,
    /// Prism compiler version
    pub compiler_version: String,
    /// Export format version
    pub format_version: String,
    /// Project name
    pub project_name: String,
    /// Project version
    pub project_version: Option<String>,
    /// Export configuration used
    pub export_config: ExportConfig,
}

/// Project context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    /// Root directory
    pub root_path: PathBuf,
    /// Source files
    pub source_files: Vec<SourceFileInfo>,
    /// Dependencies
    pub dependencies: Vec<DependencyInfo>,
    /// Build configuration
    pub build_config: HashMap<String, serde_json::Value>,
}

/// Source file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFileInfo {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp
    pub last_modified: String,
    /// Language/file type
    pub language: String,
    /// Source code (if included)
    pub source: Option<String>,
    /// AST representation (if included)
    pub ast: Option<serde_json::Value>,
}

/// Dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// Dependency name
    pub name: String,
    /// Version
    pub version: String,
    /// Dependency type (direct, transitive, dev, etc.)
    pub dependency_type: String,
    /// Source (registry, git, local, etc.)
    pub source: String,
}

/// Module context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleContext {
    /// Module name
    pub name: String,
    /// Module path
    pub path: PathBuf,
    /// Exported symbols
    pub exports: Vec<SymbolInfo>,
    /// Imported symbols
    pub imports: Vec<ImportInfo>,
    /// Functions defined in module
    pub functions: Vec<FunctionInfo>,
    /// Types defined in module
    pub types: Vec<TypeInfo>,
    /// Constants defined in module
    pub constants: Vec<ConstantInfo>,
    /// Module-level AI metadata
    pub ai_metadata: Option<AIMetadata>,
}

/// Symbol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: String,
    /// Visibility (public, private, etc.)
    pub visibility: String,
    /// Location in source
    pub location: LocationInfo,
    /// AI metadata
    pub ai_metadata: Option<AIMetadata>,
}

/// Import information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    /// Module being imported
    pub module: String,
    /// Specific symbols imported (if any)
    pub symbols: Option<Vec<String>>,
    /// Import alias
    pub alias: Option<String>,
    /// Import location
    pub location: LocationInfo,
}

/// Function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Parameters
    pub parameters: Vec<ParameterInfo>,
    /// Return type
    pub return_type: String,
    /// Function signature
    pub signature: String,
    /// Documentation
    pub documentation: Option<String>,
    /// Complexity metrics
    pub complexity: Option<ComplexityMetrics>,
    /// Location in source
    pub location: LocationInfo,
    /// AI metadata
    pub ai_metadata: Option<AIMetadata>,
}

/// Parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Default value (if any)
    pub default_value: Option<String>,
    /// Documentation
    pub documentation: Option<String>,
}

/// Type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type name
    pub name: String,
    /// Type kind (struct, enum, interface, etc.)
    pub kind: String,
    /// Fields/members
    pub fields: Vec<FieldInfo>,
    /// Methods
    pub methods: Vec<FunctionInfo>,
    /// Type constraints
    pub constraints: Vec<String>,
    /// Location in source
    pub location: LocationInfo,
    /// AI metadata
    pub ai_metadata: Option<AIMetadata>,
}

/// Field information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: String,
    /// Visibility
    pub visibility: String,
    /// Documentation
    pub documentation: Option<String>,
}

/// Constant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantInfo {
    /// Constant name
    pub name: String,
    /// Constant type
    pub const_type: String,
    /// Constant value
    pub value: String,
    /// Location in source
    pub location: LocationInfo,
}

/// Location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationInfo {
    /// File path
    pub file: PathBuf,
    /// Start line (1-based)
    pub start_line: u32,
    /// Start column (1-based)
    pub start_column: u32,
    /// End line (1-based)
    pub end_line: u32,
    /// End column (1-based)
    pub end_column: u32,
}

/// Type system context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSystemContext {
    /// Built-in types
    pub builtin_types: Vec<String>,
    /// User-defined types
    pub user_types: Vec<TypeInfo>,
    /// Type aliases
    pub type_aliases: HashMap<String, String>,
    /// Generic type parameters
    pub generic_parameters: HashMap<String, Vec<String>>,
    /// Type constraints
    pub constraints: HashMap<String, Vec<String>>,
}

/// Relationship graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipGraph {
    /// Call graph edges
    pub call_graph: Vec<CallRelation>,
    /// Dependency graph edges
    pub dependency_graph: Vec<DependencyRelation>,
    /// Inheritance relationships
    pub inheritance: Vec<InheritanceRelation>,
    /// Usage relationships
    pub usage: Vec<UsageRelation>,
}

/// Call relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRelation {
    /// Caller function
    pub caller: String,
    /// Called function
    pub callee: String,
    /// Call location
    pub location: LocationInfo,
    /// Call frequency (if available)
    pub frequency: Option<u64>,
}

/// Dependency relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyRelation {
    /// Dependent module
    pub dependent: String,
    /// Dependency module
    pub dependency: String,
    /// Dependency type
    pub relation_type: String,
}

/// Inheritance relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceRelation {
    /// Child type
    pub child: String,
    /// Parent type
    pub parent: String,
    /// Inheritance type (extends, implements, etc.)
    pub relation_type: String,
}

/// Usage relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRelation {
    /// User symbol
    pub user: String,
    /// Used symbol
    pub used: String,
    /// Usage type (field access, method call, etc.)
    pub usage_type: String,
    /// Usage location
    pub location: LocationInfo,
}

/// Complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic: u32,
    /// Cognitive complexity
    pub cognitive: u32,
    /// Lines of code
    pub lines_of_code: u32,
    /// Number of parameters
    pub parameter_count: u32,
    /// Nesting depth
    pub nesting_depth: u32,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Compilation time
    pub compilation_time: u64,
    /// Memory usage
    pub memory_usage: u64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
}

/// AI context exporter trait
#[async_trait]
pub trait AIContextExporter: Send + Sync {
    /// Export AI context
    async fn export_context(
        &self,
        semantic_db: &SemanticDatabase,
        config: &ExportConfig,
    ) -> CompilerResult<AIContext>;

    /// Export to file
    async fn export_to_file(
        &self,
        context: &AIContext,
        config: &ExportConfig,
    ) -> CompilerResult<PathBuf>;

    /// Export to string
    async fn export_to_string(
        &self,
        context: &AIContext,
        config: &ExportConfig,
    ) -> CompilerResult<String>;

    /// Validate exported context
    async fn validate_export(&self, context: &AIContext) -> CompilerResult<Vec<String>>;
}

/// Default AI context exporter
pub struct DefaultAIExporter {
    project_root: PathBuf,
}

impl DefaultAIExporter {
    /// Create a new AI context exporter
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    /// Build context metadata
    async fn build_metadata(&self, config: &ExportConfig) -> ContextMetadata {
        ContextMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            format_version: "1.0.0".to_string(),
            project_name: self.project_root.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            project_version: None,
            export_config: config.clone(),
        }
    }

    /// Build project context
    async fn build_project_context(&self, config: &ExportConfig) -> CompilerResult<ProjectContext> {
        let mut source_files = Vec::new();
        
        // Walk the project directory to find source files
        if let Ok(entries) = std::fs::read_dir(&self.project_root) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        let path = entry.path();
                        if let Some(extension) = path.extension() {
                            if extension == "prism" {
                                source_files.push(SourceFileInfo {
                                    path: path.clone(),
                                    size: metadata.len(),
                                    last_modified: format!("{:?}", metadata.modified().unwrap_or(std::time::UNIX_EPOCH)),
                                    language: "prism".to_string(),
                                    source: if config.include_source {
                                        std::fs::read_to_string(&path).ok()
                                    } else {
                                        None
                                    },
                                    ast: if config.include_ast {
                                        // Would parse and include AST
                                        None
                                    } else {
                                        None
                                    },
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(ProjectContext {
            root_path: self.project_root.clone(),
            source_files,
            dependencies: Vec::new(), // Would be populated from actual dependencies
            build_config: HashMap::new(),
        })
    }

    /// Extract AI context from semantic database with actual data
    async fn build_modules_context(
        &self,
        semantic_db: &SemanticDatabase,
        config: &ExportConfig,
    ) -> CompilerResult<Vec<ModuleContext>> {
        let mut modules = Vec::new();

        // Get all modules from semantic database
        let module_infos = semantic_db.get_all_modules().await?;
        
        for module_info in module_infos {
            let module_context = self.build_single_module_context(&module_info, semantic_db, config).await?;
            modules.push(module_context);
        }

        Ok(modules)
    }

    /// Build context for a single module with comprehensive metadata
    async fn build_single_module_context(
        &self,
        module_info: &crate::semantic::ModuleInfo,
        semantic_db: &SemanticDatabase,
        config: &ExportConfig,
    ) -> CompilerResult<ModuleContext> {
        // Extract basic module information
        let mut module_context = ModuleContext {
            name: module_info.name.clone(),
            path: module_info.file_path.clone(),
            exports: Vec::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            types: Vec::new(),
            constants: Vec::new(),
            ai_metadata: None,
        };

        // Extract symbols from semantic database
        let symbols = semantic_db.get_symbols_for_module(&module_info.name).await?;
        
        for symbol in symbols {
            match symbol.kind {
                crate::semantic::SymbolKind::Function => {
                    let function_info = self.extract_function_info(&symbol, semantic_db).await?;
                    module_context.functions.push(function_info);
                }
                crate::semantic::SymbolKind::Type => {
                    let type_info = self.extract_type_info(&symbol, semantic_db).await?;
                    module_context.types.push(type_info);
                }
                crate::semantic::SymbolKind::Constant => {
                    let constant_info = self.extract_constant_info(&symbol, semantic_db).await?;
                    module_context.constants.push(constant_info);
                }
                _ => {} // Handle other symbol types as needed
            }
        }

        // Extract exports and imports
        module_context.exports = self.extract_module_exports(module_info, semantic_db).await?;
        module_context.imports = self.extract_module_imports(module_info, semantic_db).await?;

        // Generate AI metadata if enabled
        if config.include_ai_metadata {
            module_context.ai_metadata = Some(self.generate_module_ai_metadata(module_info, semantic_db).await?);
        }

        Ok(module_context)
    }

    /// Extract function information with semantic context
    async fn extract_function_info(
        &self,
        symbol: &crate::semantic::SymbolInfo,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<FunctionInfo> {
        // Get function details from semantic database
        let function_details = semantic_db.get_function_details(&symbol.name).await?;

        let mut function_info = FunctionInfo {
            name: symbol.name.clone(),
            parameters: Vec::new(),
            return_type: function_details.return_type.clone(),
            signature: function_details.signature.clone(),
            documentation: function_details.documentation.clone(),
            complexity: None,
            location: LocationInfo {
                file: symbol.location.file.clone(),
                start_line: symbol.location.start.line,
                start_column: symbol.location.start.column,
                end_line: symbol.location.end.line,
                end_column: symbol.location.end.column,
            },
            ai_metadata: None,
        };

        // Extract parameter information
        for param in &function_details.parameters {
            function_info.parameters.push(ParameterInfo {
                name: param.name.clone(),
                param_type: param.type_annotation.clone(),
                default_value: param.default_value.clone(),
                documentation: param.documentation.clone(),
            });
        }

        // Calculate complexity metrics if available
        if let Some(complexity_data) = semantic_db.get_complexity_metrics(&symbol.name).await? {
            function_info.complexity = Some(ComplexityMetrics {
                cyclomatic: complexity_data.cyclomatic,
                cognitive: complexity_data.cognitive,
                lines_of_code: complexity_data.lines_of_code,
                parameter_count: function_info.parameters.len() as u32,
                nesting_depth: complexity_data.nesting_depth,
            });
        }

        // Generate AI metadata
        function_info.ai_metadata = Some(self.generate_function_ai_metadata(&function_details).await?);

        Ok(function_info)
    }

    /// Extract type information with semantic context
    async fn extract_type_info(
        &self,
        symbol: &crate::semantic::SymbolInfo,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<TypeInfo> {
        // Get type details from semantic database
        let type_details = semantic_db.get_type_details(&symbol.name).await?;

        let mut type_info = TypeInfo {
            name: symbol.name.clone(),
            kind: type_details.kind.clone(),
            fields: Vec::new(),
            methods: Vec::new(),
            constraints: type_details.constraints.clone(),
            location: LocationInfo {
                file: symbol.location.file.clone(),
                start_line: symbol.location.start.line,
                start_column: symbol.location.start.column,
                end_line: symbol.location.end.line,
                end_column: symbol.location.end.column,
            },
            ai_metadata: None,
        };

        // Extract field information
        for field in &type_details.fields {
            type_info.fields.push(FieldInfo {
                name: field.name.clone(),
                field_type: field.type_annotation.clone(),
                visibility: field.visibility.clone(),
                documentation: field.documentation.clone(),
            });
        }

        // Extract method information
        for method in &type_details.methods {
            let method_info = self.extract_function_info(method, semantic_db).await?;
            type_info.methods.push(method_info);
        }

        // Generate AI metadata
        type_info.ai_metadata = Some(self.generate_type_ai_metadata(&type_details).await?);

        Ok(type_info)
    }

    /// Extract constant information
    async fn extract_constant_info(
        &self,
        symbol: &crate::semantic::SymbolInfo,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<ConstantInfo> {
        let constant_details = semantic_db.get_constant_details(&symbol.name).await?;

        Ok(ConstantInfo {
            name: symbol.name.clone(),
            const_type: constant_details.type_annotation.clone(),
            value: constant_details.value.clone(),
            location: LocationInfo {
                file: symbol.location.file.clone(),
                start_line: symbol.location.start.line,
                start_column: symbol.location.start.column,
                end_line: symbol.location.end.line,
                end_column: symbol.location.end.column,
            },
        })
    }

    /// Extract module exports with semantic information
    async fn extract_module_exports(
        &self,
        module_info: &crate::semantic::ModuleInfo,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<Vec<SymbolInfo>> {
        let exports = semantic_db.get_module_exports(&module_info.name).await?;
        let mut export_infos = Vec::new();

        for export in exports {
            export_infos.push(SymbolInfo {
                name: export.name.clone(),
                symbol_type: export.kind.to_string(),
                visibility: export.visibility.to_string(),
                location: LocationInfo {
                    file: export.location.file.clone(),
                    start_line: export.location.start.line,
                    start_column: export.location.start.column,
                    end_line: export.location.end.line,
                    end_column: export.location.end.column,
                },
                ai_metadata: Some(self.generate_symbol_ai_metadata(&export).await?),
            });
        }

        Ok(export_infos)
    }

    /// Extract module imports with semantic information
    async fn extract_module_imports(
        &self,
        module_info: &crate::semantic::ModuleInfo,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<Vec<ImportInfo>> {
        let imports = semantic_db.get_module_imports(&module_info.name).await?;
        let mut import_infos = Vec::new();

        for import in imports {
            import_infos.push(ImportInfo {
                module: import.module_name.clone(),
                symbols: import.imported_symbols,
                alias: import.alias,
                location: LocationInfo {
                    file: import.location.file.clone(),
                    start_line: import.location.start.line,
                    start_column: import.location.start.column,
                    end_line: import.location.end.line,
                    end_column: import.location.end.column,
                },
            });
        }

        Ok(import_infos)
    }

    /// Build type system context with actual semantic data
    async fn build_type_system_context(
        &self,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<TypeSystemContext> {
        let builtin_types = semantic_db.get_builtin_types().await?;
        let user_types = semantic_db.get_user_defined_types().await?;
        let type_aliases = semantic_db.get_type_aliases().await?;
        let generic_parameters = semantic_db.get_generic_parameters().await?;
        let constraints = semantic_db.get_type_constraints().await?;

        let mut user_type_infos = Vec::new();
        for user_type in user_types {
            let type_info = self.extract_type_info(&user_type, semantic_db).await?;
            user_type_infos.push(type_info);
        }

        Ok(TypeSystemContext {
            builtin_types: builtin_types.into_iter().map(|t| t.name).collect(),
            user_types: user_type_infos,
            type_aliases,
            generic_parameters,
            constraints,
        })
    }

    /// Build relationship graph with actual semantic relationships
    async fn build_relationship_graph(
        &self,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<RelationshipGraph> {
        // Extract call relationships
        let call_relations = semantic_db.get_call_relationships().await?
            .into_iter()
            .map(|rel| CallRelation {
                caller: rel.caller,
                callee: rel.callee,
                location: LocationInfo {
                    file: rel.location.file,
                    start_line: rel.location.start.line,
                    start_column: rel.location.start.column,
                    end_line: rel.location.end.line,
                    end_column: rel.location.end.column,
                },
                frequency: rel.frequency,
            })
            .collect();

        // Extract dependency relationships
        let dependency_relations = semantic_db.get_dependency_relationships().await?
            .into_iter()
            .map(|rel| DependencyRelation {
                dependent: rel.dependent,
                dependency: rel.dependency,
                relation_type: rel.relation_type,
            })
            .collect();

        // Extract inheritance relationships
        let inheritance_relations = semantic_db.get_inheritance_relationships().await?
            .into_iter()
            .map(|rel| InheritanceRelation {
                child: rel.child,
                parent: rel.parent,
                relation_type: rel.relation_type,
            })
            .collect();

        // Extract usage relationships
        let usage_relations = semantic_db.get_usage_relationships().await?
            .into_iter()
            .map(|rel| UsageRelation {
                user: rel.user,
                used: rel.used,
                usage_type: rel.usage_type,
                location: LocationInfo {
                    file: rel.location.file,
                    start_line: rel.location.start.line,
                    start_column: rel.location.start.column,
                    end_line: rel.location.end.line,
                    end_column: rel.location.end.column,
                },
            })
            .collect();

        Ok(RelationshipGraph {
            call_graph: call_relations,
            dependency_graph: dependency_relations,
            inheritance: inheritance_relations,
            usage: usage_relations,
        })
    }

    /// Generate AI metadata for a module
    async fn generate_module_ai_metadata(
        &self,
        module_info: &crate::semantic::ModuleInfo,
        semantic_db: &SemanticDatabase,
    ) -> CompilerResult<AIMetadata> {
        // Extract business context
        let business_context = semantic_db.get_module_business_context(&module_info.name).await?;
        
        // Extract architectural patterns
        let architectural_patterns = semantic_db.get_module_patterns(&module_info.name).await?;
        
        // Extract complexity metrics
        let complexity_metrics = semantic_db.get_module_complexity(&module_info.name).await?;

        // Generate AI-readable summary
        let summary = format!(
            "Module '{}' provides {} with {} functions, {} types, and {} constants. Primary business capability: {}. Architectural style: {}.",
            module_info.name,
            business_context.primary_capability,
            module_info.function_count,
            module_info.type_count,
            module_info.constant_count,
            business_context.domain.unwrap_or_else(|| "general".to_string()),
            architectural_patterns.first().unwrap_or(&"unknown".to_string())
        );

        Ok(AIMetadata {
            summary,
            business_context: Some(business_context),
            architectural_patterns,
            complexity_metrics: Some(complexity_metrics),
            generated_at: chrono::Utc::now().to_rfc3339(),
            confidence: 0.85, // High confidence for semantic database data
        })
    }

    /// Generate AI metadata for a function
    async fn generate_function_ai_metadata(
        &self,
        function_details: &crate::semantic::FunctionDetails,
    ) -> CompilerResult<AIMetadata> {
        let summary = format!(
            "Function '{}' takes {} parameters and returns {}. {}",
            function_details.name,
            function_details.parameters.len(),
            function_details.return_type,
            function_details.documentation.clone().unwrap_or_else(|| "No documentation available.".to_string())
        );

        Ok(AIMetadata {
            summary,
            business_context: function_details.business_context.clone(),
            architectural_patterns: function_details.patterns.clone(),
            complexity_metrics: function_details.complexity.clone(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            confidence: 0.80,
        })
    }

    /// Generate AI metadata for a type
    async fn generate_type_ai_metadata(
        &self,
        type_details: &crate::semantic::TypeDetails,
    ) -> CompilerResult<AIMetadata> {
        let summary = format!(
            "{} '{}' with {} fields and {} methods. {}",
            type_details.kind,
            type_details.name,
            type_details.fields.len(),
            type_details.methods.len(),
            type_details.documentation.clone().unwrap_or_else(|| "No documentation available.".to_string())
        );

        Ok(AIMetadata {
            summary,
            business_context: type_details.business_context.clone(),
            architectural_patterns: type_details.patterns.clone(),
            complexity_metrics: None, // Types don't have complexity metrics
            generated_at: chrono::Utc::now().to_rfc3339(),
            confidence: 0.75,
        })
    }

    /// Generate AI metadata for a symbol
    async fn generate_symbol_ai_metadata(
        &self,
        symbol: &crate::semantic::SymbolInfo,
    ) -> CompilerResult<AIMetadata> {
        let summary = format!(
            "{} symbol '{}' of type {} with {} visibility",
            symbol.kind.to_string(),
            symbol.name,
            symbol.type_info.clone().unwrap_or_else(|| "unknown".to_string()),
            symbol.visibility.to_string()
        );

        Ok(AIMetadata {
            summary,
            business_context: symbol.business_context.clone(),
            architectural_patterns: Vec::new(),
            complexity_metrics: None,
            generated_at: chrono::Utc::now().to_rfc3339(),
            confidence: 0.70,
        })
    }

    /// Serialize context to string
    async fn serialize_context(
        &self,
        context: &AIContext,
        format: &ExportFormat,
    ) -> CompilerResult<String> {
        match format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(context).map_err(|e| {
                    CompilerError::InternalError(format!("JSON serialization failed: {}", e))
                })
            }
            ExportFormat::Yaml => {
                serde_yaml::to_string(context).map_err(|e| {
                    CompilerError::InternalError(format!("YAML serialization failed: {}", e))
                })
            }
            ExportFormat::Xml => {
                // Basic XML export by wrapping JSON in XML structure
                let json_data = serde_json::to_string_pretty(context).map_err(|e| {
                    CompilerError::InternalError(format!("JSON serialization for XML failed: {}", e))
                })?;
                
                Ok(format!(
                    r#"<?xml version="1.0" encoding="UTF-8"?>
<prism_ai_context>
    <format>xml</format>
    <version>1.0.0</version>
    <exported_at>{}</exported_at>
    <json_data><![CDATA[{}]]></json_data>
</prism_ai_context>"#,
                    chrono::Utc::now().to_rfc3339(),
                    json_data
                ))
            }
            ExportFormat::Binary => {
                // For binary export, we'll use a simple approach of converting to JSON bytes
                // A proper implementation would use MessagePack or similar
                let json_data = serde_json::to_string(context).map_err(|e| {
                    CompilerError::InternalError(format!("JSON serialization for binary failed: {}", e))
                })?;
                
                // For now, just return the JSON as a string since we don't have base64 dependency
                // A proper implementation would use MessagePack or similar binary format
                Ok(format!("BINARY_DATA:{}", json_data))
            }
            ExportFormat::OpenApi => {
                // Generate basic OpenAPI specification for the AI context
                let openapi_spec = serde_json::json!({
                    "openapi": "3.0.0",
                    "info": {
                        "title": "Prism AI Context API",
                        "version": "1.0.0",
                        "description": "API specification for Prism AI context data"
                    },
                    "paths": {
                        "/context": {
                            "get": {
                                "summary": "Get AI context",
                                "responses": {
                                    "200": {
                                        "description": "AI context data",
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                    "type": "object",
                                                    "properties": {
                                                        "metadata": { "type": "object" },
                                                        "project": { "type": "object" },
                                                        "modules": { "type": "array" },
                                                        "types": { "type": "object" },
                                                        "relationships": { "type": "object" }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
                
                serde_json::to_string_pretty(&openapi_spec).map_err(|e| {
                    CompilerError::InternalError(format!("OpenAPI serialization failed: {}", e))
                })
            }
            ExportFormat::GraphQL => {
                // Generate basic GraphQL schema for the AI context
                let schema = r#"
# Prism AI Context GraphQL Schema

type Query {
    context: AIContext
}

type AIContext {
    metadata: ContextMetadata!
    project: ProjectContext!
    modules: [ModuleContext!]!
    types: TypeSystemContext!
    relationships: RelationshipGraph!
}

type ContextMetadata {
    timestamp: String!
    compilerVersion: String!
    formatVersion: String!
    projectName: String!
    projectVersion: String
}

type ProjectContext {
    rootPath: String!
    sourceFiles: [SourceFileInfo!]!
    dependencies: [DependencyInfo!]!
}

type SourceFileInfo {
    path: String!
    size: Int!
    lastModified: String!
    language: String!
}

type DependencyInfo {
    name: String!
    version: String!
    dependencyType: String!
}

type ModuleContext {
    name: String!
    path: String!
    exports: [String!]!
    imports: [String!]!
    functions: [String!]!
    types: [String!]!
    constants: [String!]!
}

type TypeSystemContext {
    builtinTypes: [String!]!
    userTypes: [String!]!
}

type RelationshipGraph {
    callGraph: [String!]!
    dependencyGraph: [String!]!
    inheritance: [String!]!
    usage: [String!]!
}
"#;
                
                Ok(schema.to_string())
            }
        }
    }
}

#[async_trait]
impl AIContextExporter for DefaultAIExporter {
    async fn export_context(
        &self,
        semantic_db: &SemanticDatabase,
        config: &ExportConfig,
    ) -> CompilerResult<AIContext> {
        info!("Exporting AI context with format: {:?}", config.format);

        let metadata = self.build_metadata(config).await;
        let project = self.build_project_context(config).await?;
        let modules = self.build_modules_context(semantic_db, config).await?;
        let types = self.build_type_system_context(semantic_db).await?;
        let relationships = self.build_relationship_graph(semantic_db).await?;

        // Collect AI metadata
        let mut ai_metadata = HashMap::new();
        for symbol in semantic_db.get_all_symbols().await {
            if let Some(metadata) = semantic_db.get_ai_metadata(&symbol.id).await {
                ai_metadata.insert(symbol.name, metadata);
            }
        }

        let context = AIContext {
            metadata,
            project,
            modules,
            types,
            relationships,
            ai_metadata,
            metrics: if config.include_metrics {
                Some(PerformanceMetrics {
                    compilation_time: 0, // Would be populated from actual metrics
                    memory_usage: 0,
                    cache_hit_ratio: 0.0,
                    parallel_efficiency: 0.0,
                })
            } else {
                None
            },
        };

        debug!("AI context export completed");
        Ok(context)
    }

    async fn export_to_file(
        &self,
        context: &AIContext,
        config: &ExportConfig,
    ) -> CompilerResult<PathBuf> {
        let content = self.export_to_string(context, config).await?;
        
        let output_path = config.output_path.clone().unwrap_or_else(|| {
            let extension = match config.format {
                ExportFormat::Json => "json",
                ExportFormat::Yaml => "yaml",
                ExportFormat::Xml => "xml",
                ExportFormat::Binary => "bin",
                ExportFormat::OpenApi => "json",
                ExportFormat::GraphQL => "graphql",
            };
            PathBuf::from(format!("ai_context.{}", extension))
        });

        tokio::fs::write(&output_path, content).await.map_err(|e| {
            CompilerError::InternalError(format!("Failed to write export file: {}", e))
        })?;

        info!("AI context exported to: {:?}", output_path);
        Ok(output_path)
    }

    async fn export_to_string(
        &self,
        context: &AIContext,
        config: &ExportConfig,
    ) -> CompilerResult<String> {
        self.serialize_context(context, &config.format).await
    }

    async fn validate_export(&self, context: &AIContext) -> CompilerResult<Vec<String>> {
        let mut warnings = Vec::new();

        if context.modules.is_empty() {
            warnings.push("No modules found in export".to_string());
        }

        if context.ai_metadata.is_empty() {
            warnings.push("No AI metadata found in export".to_string());
        }

        if context.types.user_types.is_empty() {
            warnings.push("No user-defined types found in export".to_string());
        }

        Ok(warnings)
    }
}

/// AI context export utilities
pub struct AIExportUtils;

impl AIExportUtils {
    /// Create export configuration for different use cases
    pub fn config_for_code_completion() -> ExportConfig {
        ExportConfig {
            format: ExportFormat::Json,
            include_source: false,
            include_ast: true,
            include_types: true,
            include_relationships: true,
            include_ai_metadata: true,
            include_metrics: false,
            compression_level: 0,
            output_path: None,
        }
    }

    /// Create export configuration for documentation generation
    pub fn config_for_documentation() -> ExportConfig {
        ExportConfig {
            format: ExportFormat::Yaml,
            include_source: true,
            include_ast: false,
            include_types: true,
            include_relationships: false,
            include_ai_metadata: true,
            include_metrics: false,
            compression_level: 0,
            output_path: None,
        }
    }

    /// Create export configuration for static analysis
    pub fn config_for_analysis() -> ExportConfig {
        ExportConfig {
            format: ExportFormat::Json,
            include_source: false,
            include_ast: true,
            include_types: true,
            include_relationships: true,
            include_ai_metadata: false,
            include_metrics: true,
            compression_level: 6,
            output_path: None,
        }
    }

    /// Merge multiple AI contexts
    pub fn merge_contexts(contexts: Vec<AIContext>) -> CompilerResult<AIContext> {
        if contexts.is_empty() {
            return Err(CompilerError::InternalError("No contexts to merge".to_string()));
        }

        let first = contexts.first().unwrap();
        let mut merged = first.clone();

        for context in contexts.iter().skip(1) {
            merged.modules.extend(context.modules.clone());
            merged.types.user_types.extend(context.types.user_types.clone());
            merged.ai_metadata.extend(context.ai_metadata.clone());
        }

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::SemanticDatabase;

    #[tokio::test]
    async fn test_ai_context_export() {
        let exporter = DefaultAIExporter::new(PathBuf::from("/tmp/test"));
        let semantic_db = SemanticDatabase::new();
        let config = ExportConfig::default();

        let context = exporter.export_context(&semantic_db, &config).await.unwrap();
        assert_eq!(context.metadata.format_version, "1.0.0");
    }

    #[test]
    fn test_export_config_presets() {
        let completion_config = AIExportUtils::config_for_code_completion();
        assert!(completion_config.include_ast);
        assert!(!completion_config.include_source);

        let doc_config = AIExportUtils::config_for_documentation();
        assert!(doc_config.include_source);
        assert!(!doc_config.include_ast);

        let analysis_config = AIExportUtils::config_for_analysis();
        assert!(analysis_config.include_metrics);
        assert!(!analysis_config.include_ai_metadata);
    }
} 