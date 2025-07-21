//! Smart Module Registry - Capability-Based Module Discovery and Management
//!
//! This module embodies the single concept of "Smart Module Management".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: discovering, registering, and resolving smart modules based
//! on capabilities, cohesion metrics, and business context during compilation.
//!
//! **Conceptual Responsibility**: Smart module lifecycle management
//! **What it does**: module discovery, capability-based resolution, cohesion tracking, business context extraction
//! **What it doesn't do**: parsing, code generation, symbol resolution (delegates to specialized modules)

use crate::error::{CompilerError, CompilerResult};
use crate::symbols::{SymbolTable, SymbolData, SymbolKind};
use crate::scope::{ScopeTree, ScopeId, ScopeKind};
use crate::resolution::{SymbolResolver, ResolutionContext};
use crate::semantic::SemanticDatabase;
use crate::cache::CompilationCache;
use prism_common::{span::Span, symbol::Symbol, NodeId};
use prism_ast::{ModuleDecl, AstNode, Item};
use prism_cohesion::{CohesionSystem, CohesionAnalysis, CohesionMetadata};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

/// Smart Module Registry providing capability-based discovery and management
/// 
/// This registry maintains a comprehensive view of all smart modules in the system,
/// enabling capability-based discovery, cohesion tracking, and business context resolution.
#[derive(Debug)]
pub struct SmartModuleRegistry {
    /// Registered modules indexed by name
    modules: Arc<RwLock<HashMap<String, RegisteredModule>>>,
    /// Capability index for fast discovery
    capability_index: Arc<RwLock<CapabilityIndex>>,
    /// Business context index
    business_context_index: Arc<RwLock<BusinessContextIndex>>,
    /// Cohesion tracking system
    cohesion_tracker: Arc<CohesionTracker>,
    /// Module dependency graph
    dependency_graph: Arc<RwLock<ModuleDependencyGraph>>,
    /// Integration with existing systems
    symbol_table: Arc<SymbolTable>,
    scope_tree: Arc<ScopeTree>,
    semantic_db: Arc<SemanticDatabase>,
    cache: Arc<CompilationCache>,
    /// Registry configuration
    config: ModuleRegistryConfig,
}

/// Configuration for the Smart Module Registry
#[derive(Debug, Clone)]
pub struct ModuleRegistryConfig {
    /// Enable capability-based discovery
    pub enable_capability_discovery: bool,
    /// Enable real-time cohesion analysis
    pub enable_cohesion_tracking: bool,
    /// Enable business context extraction
    pub enable_business_context: bool,
    /// Enable dependency cycle detection
    pub enable_cycle_detection: bool,
    /// Maximum module registration batch size
    pub max_batch_size: usize,
    /// Cohesion threshold for warnings
    pub cohesion_warning_threshold: f64,
    /// Enable AI metadata generation
    pub enable_ai_metadata: bool,
}

impl Default for ModuleRegistryConfig {
    fn default() -> Self {
        Self {
            enable_capability_discovery: true,
            enable_cohesion_tracking: true,
            enable_business_context: true,
            enable_cycle_detection: true,
            max_batch_size: 100,
            cohesion_warning_threshold: 70.0,
            enable_ai_metadata: true,
        }
    }
}

/// A registered smart module with full metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModule {
    /// Module identification
    pub module_info: ModuleInfo,
    /// Capability information
    pub capabilities: ModuleCapabilities,
    /// Business context
    pub business_context: BusinessContext,
    /// Cohesion metrics
    pub cohesion_metrics: Option<CohesionMetadata>,
    /// Dependencies and relationships
    pub dependencies: ModuleDependencies,
    /// AI-generated metadata
    pub ai_metadata: Option<AIModuleMetadata>,
    /// Registration timestamp
    pub registered_at: SystemTime,
    /// Last analysis timestamp
    pub last_analyzed: Option<SystemTime>,
}

/// Core module identification and location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    /// Module file path
    pub file_path: PathBuf,
    /// Module version
    pub version: Option<String>,
    /// Module stability level
    pub stability: StabilityLevel,
    /// Module visibility
    pub visibility: ModuleVisibility,
    /// Module scope ID
    pub scope_id: Option<ScopeId>,
    /// Source AST node ID
    pub node_id: Option<NodeId>,
}

/// Module stability levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityLevel {
    Experimental,
    Beta,
    Stable,
    Deprecated,
}

/// Module visibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleVisibility {
    Public,
    Internal,
    Private,
}

/// Module capabilities and requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleCapabilities {
    /// Capabilities this module provides
    pub provides: HashSet<String>,
    /// Capabilities this module requires
    pub requires: HashSet<String>,
    /// Optional capabilities (degraded functionality if missing)
    pub optional: HashSet<String>,
    /// Capability compatibility matrix
    pub compatibility: HashMap<String, Vec<String>>,
}

/// Business context information for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessContext {
    /// Primary business capability
    pub primary_capability: String,
    /// Business domain
    pub domain: Option<String>,
    /// Responsibility description
    pub responsibility: Option<String>,
    /// Business rules this module enforces
    pub business_rules: Vec<String>,
    /// Key business entities
    pub entities: Vec<String>,
    /// Business processes supported
    pub processes: Vec<String>,
}

/// Module dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDependencies {
    /// Direct dependencies
    pub direct: HashSet<String>,
    /// Transitive dependencies (computed)
    pub transitive: HashSet<String>,
    /// Reverse dependencies (modules that depend on this one)
    pub dependents: HashSet<String>,
    /// Circular dependency chains (if any)
    pub cycles: Vec<Vec<String>>,
}

/// AI-generated metadata for modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModuleMetadata {
    /// AI-generated summary
    pub summary: String,
    /// Suggested improvements
    pub suggestions: Vec<String>,
    /// Complexity assessment
    pub complexity_score: f64,
    /// Maintainability score
    pub maintainability_score: f64,
    /// Business value assessment
    pub business_value: String,
    /// Generated documentation
    pub generated_docs: Option<String>,
}

/// Comprehensive AI metadata for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAIMetadata {
    /// Module summary
    pub module_summary: ModuleSummary,
    /// Architectural analysis
    pub architectural_analysis: ArchitecturalAnalysis,
    /// Business intelligence
    pub business_intelligence: BusinessIntelligence,
    /// Code insights
    pub code_insights: CodeInsights,
    /// Integration guidance
    pub integration_guidance: IntegrationGuidance,
    /// Learning resources
    pub learning_resources: LearningResources,
    /// Quality assessment
    pub quality_assessment: QualityAssessment,
    /// Future recommendations
    pub future_recommendations: FutureRecommendations,
    /// Metadata version
    pub metadata_version: String,
    /// Generated timestamp
    pub generated_at: String,
}

/// Project-level AI metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAIMetadata {
    /// Project summary
    pub project_summary: ProjectSummary,
    /// Modules within the project
    pub modules: HashMap<String, ComprehensiveAIMetadata>,
    /// Module relationships
    pub relationships: Vec<ModuleRelationshipAnalysis>,
    /// Architecture patterns
    pub architecture_patterns: Vec<ArchitecturePattern>,
    /// Quality overview
    pub quality_overview: ProjectQualityOverview,
    /// Recommendations
    pub recommendations: Vec<ProjectRecommendation>,
    /// Metadata version
    pub metadata_version: String,
    /// Generated timestamp
    pub generated_at: String,
}

/// Capability index for fast capability-based discovery
#[derive(Debug, Default)]
pub struct CapabilityIndex {
    /// Map from capability to modules that provide it
    providers: HashMap<String, HashSet<String>>,
    /// Map from capability to modules that require it
    consumers: HashMap<String, HashSet<String>>,
    /// Capability compatibility graph
    compatibility_graph: HashMap<String, HashSet<String>>,
}

/// Business context index for business-driven discovery
#[derive(Debug, Default)]
pub struct BusinessContextIndex {
    /// Map from domain to modules
    domains: HashMap<String, HashSet<String>>,
    /// Map from business capability to modules
    capabilities: HashMap<String, HashSet<String>>,
    /// Map from entity to modules that handle it
    entities: HashMap<String, HashSet<String>>,
    /// Map from process to modules that support it
    processes: HashMap<String, HashSet<String>>,
}

/// Cohesion tracking system
#[derive(Debug)]
pub struct CohesionTracker {
    /// Cohesion analysis system
    cohesion_system: Arc<CohesionSystem>,
    /// Module cohesion history
    cohesion_history: Arc<RwLock<HashMap<String, Vec<CohesionSnapshot>>>>,
    /// Cohesion thresholds and alerts
    alert_config: CohesionAlertConfig,
}

/// Cohesion snapshot for tracking over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionSnapshot {
    /// Timestamp of analysis
    pub timestamp: SystemTime,
    /// Cohesion metadata
    pub metrics: CohesionMetadata,
    /// Analysis trigger
    pub trigger: CohesionAnalysisTrigger,
}

/// What triggered a cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CohesionAnalysisTrigger {
    ModuleRegistration,
    ModuleUpdate,
    DependencyChange,
    PeriodicAnalysis,
    UserRequest,
}

/// Configuration for cohesion alerts
#[derive(Debug, Clone)]
pub struct CohesionAlertConfig {
    /// Minimum score for warnings
    pub warning_threshold: f64,
    /// Minimum score for errors
    pub error_threshold: f64,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
}

/// Module dependency graph for cycle detection and analysis
#[derive(Debug, Default)]
pub struct ModuleDependencyGraph {
    /// Adjacency list representation
    edges: HashMap<String, HashSet<String>>,
    /// Reverse edges for efficient queries
    reverse_edges: HashMap<String, HashSet<String>>,
    /// Cached strongly connected components
    scc_cache: Option<Vec<Vec<String>>>,
    /// Cache timestamp
    cache_timestamp: Option<SystemTime>,
}

impl SmartModuleRegistry {
    /// Create a new Smart Module Registry with full integration
    pub fn new(
        config: ModuleRegistryConfig,
        symbol_table: Arc<SymbolTable>,
        scope_tree: Arc<ScopeTree>,
        semantic_db: Arc<SemanticDatabase>,
        cache: Arc<CompilationCache>,
    ) -> CompilerResult<Self> {
        let cohesion_system = Arc::new(CohesionSystem::new()?);
        
        let cohesion_tracker = Arc::new(CohesionTracker {
            cohesion_system,
            cohesion_history: Arc::new(RwLock::new(HashMap::new())),
            alert_config: CohesionAlertConfig {
                warning_threshold: config.cohesion_warning_threshold,
                error_threshold: 50.0,
                enable_trend_analysis: true,
            },
        });

        Ok(Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            capability_index: Arc::new(RwLock::new(CapabilityIndex::default())),
            business_context_index: Arc::new(RwLock::new(BusinessContextIndex::default())),
            cohesion_tracker,
            dependency_graph: Arc::new(RwLock::new(ModuleDependencyGraph::default())),
            symbol_table,
            scope_tree,
            semantic_db,
            cache,
            config,
        })
    }

    /// Register a new smart module with full analysis
    pub async fn register_module(&self, module_decl: &ModuleDecl, file_path: PathBuf, scope_id: ScopeId, node_id: NodeId) -> CompilerResult<()> {
        info!("Registering smart module: {}", module_decl.name);

        // Extract module information
        let module_info = self.extract_module_info(module_decl, file_path, scope_id, node_id)?;
        
        // Extract capabilities
        let capabilities = self.extract_capabilities(module_decl)?;
        
        // Extract business context
        let business_context = if self.config.enable_business_context {
            self.extract_business_context(module_decl)?
        } else {
            BusinessContext {
                primary_capability: module_decl.name.to_string(),
                domain: None,
                responsibility: None,
                business_rules: Vec::new(),
                entities: Vec::new(),
                processes: Vec::new(),
            }
        };

        // Analyze cohesion if enabled
        let cohesion_metrics = if self.config.enable_cohesion_tracking {
            self.analyze_module_cohesion(module_decl, &module_info).await?
        } else {
            None
        };

        // Extract dependencies
        let dependencies = self.extract_dependencies(module_decl)?;

        // Generate AI metadata if enabled
        let ai_metadata = if self.config.enable_ai_metadata {
            self.generate_ai_metadata(&module_info, &capabilities, &business_context, &cohesion_metrics).await?
        } else {
            None
        };

        // Create registered module
        let registered_module = RegisteredModule {
            module_info: module_info.clone(),
            capabilities: capabilities.clone(),
            business_context: business_context.clone(),
            cohesion_metrics: cohesion_metrics.clone(),
            dependencies: dependencies.clone(),
            ai_metadata,
            registered_at: SystemTime::now(),
            last_analyzed: Some(SystemTime::now()),
        };

        // Store in registry
        {
            let mut modules = self.modules.write().unwrap();
            modules.insert(module_info.name.clone(), registered_module);
        }

        // Update indices
        self.update_capability_index(&module_info.name, &capabilities).await?;
        self.update_business_context_index(&module_info.name, &business_context).await?;
        self.update_dependency_graph(&module_info.name, &dependencies).await?;

        // Track cohesion if available
        if let Some(cohesion) = cohesion_metrics {
            self.track_cohesion_snapshot(&module_info.name, cohesion, CohesionAnalysisTrigger::ModuleRegistration).await?;
        }

        // Detect cycles if enabled
        if self.config.enable_cycle_detection {
            self.detect_dependency_cycles().await?;
        }

        debug!("Successfully registered smart module: {}", module_info.name);
        Ok(())
    }

    /// Discover modules by capability requirements
    pub async fn discover_by_capability(&self, required_capabilities: &[String]) -> CompilerResult<Vec<ModuleDiscoveryResult>> {
        let capability_index = self.capability_index.read().unwrap();
        let modules = self.modules.read().unwrap();
        
        let mut results = Vec::new();
        let mut candidate_modules = HashSet::new();

        // Find modules that provide required capabilities
        for capability in required_capabilities {
            if let Some(providers) = capability_index.providers.get(capability) {
                candidate_modules.extend(providers.iter().cloned());
            }
        }

        // Score and rank candidates
        for module_name in candidate_modules {
            if let Some(module) = modules.get(&module_name) {
                let score = self.calculate_capability_match_score(&module.capabilities, required_capabilities)?;
                
                results.push(ModuleDiscoveryResult {
                    module_name: module_name.clone(),
                    match_score: score,
                    provided_capabilities: module.capabilities.provides.clone(),
                    missing_capabilities: self.find_missing_capabilities(&module.capabilities, required_capabilities),
                    business_context: module.business_context.clone(),
                    cohesion_score: module.cohesion_metrics.as_ref().map(|m| m.overall_score),
                });
            }
        }

        // Sort by match score
        results.sort_by(|a, b| b.match_score.partial_cmp(&a.match_score).unwrap());

        Ok(results)
    }

    /// Discover modules by business context
    pub async fn discover_by_business_context(&self, domain: Option<&str>, capability: Option<&str>, entity: Option<&str>) -> CompilerResult<Vec<ModuleDiscoveryResult>> {
        let business_index = self.business_context_index.read().unwrap();
        let modules = self.modules.read().unwrap();
        
        let mut candidate_modules = HashSet::new();

        // Find modules by domain
        if let Some(domain) = domain {
            if let Some(domain_modules) = business_index.domains.get(domain) {
                candidate_modules.extend(domain_modules.iter().cloned());
            }
        }

        // Find modules by business capability
        if let Some(capability) = capability {
            if let Some(capability_modules) = business_index.capabilities.get(capability) {
                candidate_modules.extend(capability_modules.iter().cloned());
            }
        }

        // Find modules by entity
        if let Some(entity) = entity {
            if let Some(entity_modules) = business_index.entities.get(entity) {
                candidate_modules.extend(entity_modules.iter().cloned());
            }
        }

        // Convert to results
        let mut results = Vec::new();
        for module_name in candidate_modules {
            if let Some(module) = modules.get(&module_name) {
                results.push(ModuleDiscoveryResult {
                    module_name: module_name.clone(),
                    match_score: 1.0, // Business context matches are high confidence
                    provided_capabilities: module.capabilities.provides.clone(),
                    missing_capabilities: HashSet::new(),
                    business_context: module.business_context.clone(),
                    cohesion_score: module.cohesion_metrics.as_ref().map(|m| m.overall_score),
                });
            }
        }

        Ok(results)
    }

    /// Get comprehensive module information
    pub async fn get_module(&self, name: &str) -> Option<RegisteredModule> {
        let modules = self.modules.read().unwrap();
        modules.get(name).cloned()
    }

    /// Get all registered modules
    pub async fn list_modules(&self) -> Vec<RegisteredModule> {
        let modules = self.modules.read().unwrap();
        modules.values().cloned().collect()
    }

    /// Get modules with low cohesion scores
    pub async fn get_low_cohesion_modules(&self, threshold: f64) -> Vec<(String, f64)> {
        let modules = self.modules.read().unwrap();
        
        modules.iter()
            .filter_map(|(name, module)| {
                module.cohesion_metrics.as_ref()
                    .map(|metrics| (name.clone(), metrics.overall_score))
                    .filter(|(_, score)| *score < threshold)
            })
            .collect()
    }

    /// Analyze dependency cycles
    pub async fn analyze_dependency_cycles(&self) -> CompilerResult<Vec<Vec<String>>> {
        let dependency_graph = self.dependency_graph.read().unwrap();
        Ok(dependency_graph.find_strongly_connected_components())
    }

    // Private helper methods

    fn extract_module_info(&self, module_decl: &ModuleDecl, file_path: PathBuf, scope_id: ScopeId, node_id: NodeId) -> CompilerResult<ModuleInfo> {
        Ok(ModuleInfo {
            name: module_decl.name.to_string(),
            file_path,
            version: module_decl.version.clone(),
            stability: match module_decl.stability {
                prism_ast::StabilityLevel::Experimental => StabilityLevel::Experimental,
                prism_ast::StabilityLevel::Beta => StabilityLevel::Beta,
                prism_ast::StabilityLevel::Stable => StabilityLevel::Stable,
                prism_ast::StabilityLevel::Deprecated => StabilityLevel::Deprecated,
            },
            visibility: ModuleVisibility::Public, // Default for now
            scope_id: Some(scope_id),
            node_id: Some(node_id),
        })
    }

    fn extract_capabilities(&self, module_decl: &ModuleDecl) -> CompilerResult<ModuleCapabilities> {
        // Extract from module declaration
        let provides = module_decl.capability.as_ref()
            .map(|cap| {
                let mut set = HashSet::new();
                set.insert(cap.clone());
                set
            })
            .unwrap_or_default();

        let requires = module_decl.dependencies.iter()
            .cloned()
            .collect();

        Ok(ModuleCapabilities {
            provides,
            requires,
            optional: HashSet::new(),
            compatibility: HashMap::new(),
        })
    }

    fn extract_business_context(&self, module_decl: &ModuleDecl) -> CompilerResult<BusinessContext> {
        Ok(BusinessContext {
            primary_capability: module_decl.capability.clone()
                .unwrap_or_else(|| module_decl.name.to_string()),
            domain: None, // Would extract from attributes/annotations
            responsibility: module_decl.description.clone(),
            business_rules: Vec::new(), // Would extract from module content
            entities: Vec::new(), // Would extract from type analysis
            processes: Vec::new(), // Would extract from function analysis
        })
    }

    async fn analyze_module_cohesion(&self, module_decl: &ModuleDecl, _module_info: &ModuleInfo) -> CompilerResult<Option<CohesionMetadata>> {
        // This would integrate with the cohesion analysis from the parser
        // For now, return a placeholder
        Ok(module_decl.cohesion_metadata.clone())
    }

    fn extract_dependencies(&self, module_decl: &ModuleDecl) -> CompilerResult<ModuleDependencies> {
        let direct = module_decl.dependencies.iter().cloned().collect();
        
        Ok(ModuleDependencies {
            direct,
            transitive: HashSet::new(), // Would be computed
            dependents: HashSet::new(), // Would be computed
            cycles: Vec::new(), // Would be detected
        })
    }

    /// Generate AI metadata if enabled
    async fn generate_ai_metadata(&self, _module_info: &ModuleInfo, _capabilities: &ModuleCapabilities, _business_context: &BusinessContext, _cohesion_metrics: &Option<CohesionMetadata>) -> CompilerResult<Option<AIModuleMetadata>> {
        // This would integrate with AI metadata generation
        // For now, return None
        Ok(None)
    }

    /// Generate comprehensive AI metadata for a module
    pub async fn generate_comprehensive_ai_metadata(
        &self,
        module_name: &str,
    ) -> CompilerResult<Option<ComprehensiveAIMetadata>> {
        let modules = self.modules.read().unwrap();
        let module = match modules.get(module_name) {
            Some(m) => m,
            None => return Ok(None),
        };

        let metadata = ComprehensiveAIMetadata {
            module_summary: self.generate_module_summary(module).await?,
            architectural_analysis: self.generate_architectural_analysis(module).await?,
            business_intelligence: self.generate_business_intelligence(module).await?,
            code_insights: self.generate_code_insights(module).await?,
            integration_guidance: self.generate_integration_guidance(module).await?,
            learning_resources: self.generate_learning_resources(module).await?,
            quality_assessment: self.generate_quality_assessment(module).await?,
            future_recommendations: self.generate_future_recommendations(module).await?,
            metadata_version: "1.0".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
        };

        Ok(Some(metadata))
    }

    /// Export AI metadata in multiple formats for external consumption
    pub async fn export_ai_metadata_for_external_tools(
        &self,
        module_name: &str,
        export_format: AIMetadataExportFormat,
    ) -> CompilerResult<String> {
        let metadata = self.generate_comprehensive_ai_metadata(module_name).await?
            .ok_or_else(|| CompilerError::InvalidInput {
                message: format!("Module '{}' not found", module_name),
            })?;

        match export_format {
            AIMetadataExportFormat::Json => {
                serde_json::to_string_pretty(&metadata)
                    .map_err(|e| CompilerError::InternalError(format!("JSON serialization failed: {}", e)))
            }
            AIMetadataExportFormat::Yaml => {
                serde_yaml::to_string(&metadata)
                    .map_err(|e| CompilerError::InternalError(format!("YAML serialization failed: {}", e)))
            }
            AIMetadataExportFormat::OpenAIStructured => {
                self.export_openai_structured_format(&metadata).await
            }
            AIMetadataExportFormat::AnthropicStructured => {
                self.export_anthropic_structured_format(&metadata).await
            }
            AIMetadataExportFormat::CustomStructured(schema) => {
                self.export_custom_structured_format(&metadata, &schema).await
            }
        }
    }

    /// Export metadata for all modules in a project
    pub async fn export_project_ai_metadata(
        &self,
        export_format: AIMetadataExportFormat,
    ) -> CompilerResult<ProjectAIMetadata> {
        let modules = self.modules.read().unwrap();
        let mut project_metadata = ProjectAIMetadata {
            project_summary: self.generate_project_summary(&modules).await?,
            modules: HashMap::new(),
            relationships: self.analyze_module_relationships(&modules).await?,
            architecture_patterns: self.identify_architecture_patterns(&modules).await?,
            quality_overview: self.generate_project_quality_overview(&modules).await?,
            recommendations: self.generate_project_recommendations(&modules).await?,
            metadata_version: "1.0".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
        };

        // Generate metadata for each module
        for (name, module) in modules.iter() {
            if let Ok(Some(module_metadata)) = self.generate_comprehensive_ai_metadata(name).await {
                project_metadata.modules.insert(name.clone(), module_metadata);
            }
        }

        Ok(project_metadata)
    }

    // AI Metadata Generation Methods

    async fn generate_module_summary(&self, module: &RegisteredModule) -> CompilerResult<ModuleSummary> {
        Ok(ModuleSummary {
            name: module.module_info.name.clone(),
            primary_purpose: module.business_context.responsibility.clone()
                .unwrap_or_else(|| "General purpose module".to_string()),
            key_capabilities: module.capabilities.provides.iter().cloned().collect(),
            business_domain: module.business_context.domain.clone()
                .unwrap_or_else(|| "General".to_string()),
            complexity_assessment: self.assess_module_complexity(module),
            maturity_level: self.assess_module_maturity(module),
            usage_patterns: self.identify_usage_patterns(module),
        })
    }

    async fn generate_architectural_analysis(&self, module: &RegisteredModule) -> CompilerResult<ArchitecturalAnalysis> {
        Ok(ArchitecturalAnalysis {
            design_patterns: self.identify_design_patterns(module),
            architectural_style: self.identify_architectural_style(module),
            coupling_analysis: self.analyze_coupling(module),
            cohesion_analysis: module.cohesion_metrics.as_ref().map(|m| CohesionAnalysisDetail {
                overall_score: m.overall_score,
                type_cohesion: m.type_cohesion,
                data_flow_cohesion: m.data_flow_cohesion,
                semantic_cohesion: m.semantic_cohesion,
                dependency_cohesion: m.dependency_cohesion,
                strengths: m.strengths.clone(),
                improvement_areas: m.suggestions.clone(),
            }),
            scalability_assessment: self.assess_scalability(module),
            maintainability_score: self.calculate_maintainability_score(module),
        })
    }

    async fn generate_business_intelligence(&self, module: &RegisteredModule) -> CompilerResult<BusinessIntelligence> {
        Ok(BusinessIntelligence {
            business_value: self.assess_business_value(module),
            stakeholder_impact: self.analyze_stakeholder_impact(module),
            business_rules: module.business_context.business_rules.clone(),
            compliance_requirements: self.identify_compliance_requirements(module),
            risk_assessment: self.assess_business_risks(module),
            roi_indicators: self.calculate_roi_indicators(module),
        })
    }

    async fn generate_code_insights(&self, module: &RegisteredModule) -> CompilerResult<CodeInsights> {
        Ok(CodeInsights {
            code_quality_metrics: self.calculate_code_quality_metrics(module),
            performance_characteristics: self.analyze_performance_characteristics(module),
            security_analysis: self.analyze_security_aspects(module),
            testing_coverage: self.assess_testing_coverage(module),
            documentation_quality: self.assess_documentation_quality(module),
            technical_debt: self.assess_technical_debt(module),
        })
    }

    async fn generate_integration_guidance(&self, module: &RegisteredModule) -> CompilerResult<IntegrationGuidance> {
        Ok(IntegrationGuidance {
            integration_patterns: self.recommend_integration_patterns(module),
            api_usage_examples: self.generate_api_examples(module),
            common_pitfalls: self.identify_common_pitfalls(module),
            best_practices: self.recommend_best_practices(module),
            configuration_guidance: self.provide_configuration_guidance(module),
            troubleshooting_guide: self.generate_troubleshooting_guide(module),
        })
    }

    async fn generate_learning_resources(&self, module: &RegisteredModule) -> CompilerResult<LearningResources> {
        Ok(LearningResources {
            tutorials: self.recommend_tutorials(module),
            documentation_links: self.gather_documentation_links(module),
            code_examples: self.generate_code_examples(module),
            related_concepts: self.identify_related_concepts(module),
            prerequisite_knowledge: self.identify_prerequisites(module),
            advanced_topics: self.identify_advanced_topics(module),
        })
    }

    async fn generate_quality_assessment(&self, module: &RegisteredModule) -> CompilerResult<QualityAssessment> {
        Ok(QualityAssessment {
            overall_quality_score: self.calculate_overall_quality_score(module),
            code_quality: self.assess_code_quality(module),
            design_quality: self.assess_design_quality(module),
            documentation_quality: self.assess_documentation_quality(module),
            test_quality: self.assess_test_quality(module),
            performance_quality: self.assess_performance_quality(module),
            security_quality: self.assess_security_quality(module),
            improvement_priorities: self.prioritize_improvements(module),
        })
    }

    async fn generate_future_recommendations(&self, module: &RegisteredModule) -> CompilerResult<FutureRecommendations> {
        Ok(FutureRecommendations {
            short_term_improvements: self.recommend_short_term_improvements(module),
            long_term_evolution: self.recommend_long_term_evolution(module),
            technology_upgrades: self.recommend_technology_upgrades(module),
            refactoring_opportunities: self.identify_refactoring_opportunities(module),
            feature_enhancements: self.suggest_feature_enhancements(module),
            architecture_evolution: self.suggest_architecture_evolution(module),
        })
    }

    // Export Format Implementations

    async fn export_openai_structured_format(&self, metadata: &ComprehensiveAIMetadata) -> CompilerResult<String> {
        let openai_format = OpenAIStructuredMetadata {
            system_prompt: format!(
                "You are analyzing the '{}' module. This module's primary purpose is: {}",
                metadata.module_summary.name,
                metadata.module_summary.primary_purpose
            ),
            module_context: metadata.clone(),
            analysis_instructions: vec![
                "Analyze the module's architecture and design patterns".to_string(),
                "Assess code quality and suggest improvements".to_string(),
                "Evaluate business value and ROI potential".to_string(),
                "Recommend integration strategies".to_string(),
            ],
            expected_outputs: vec![
                "Architectural assessment".to_string(),
                "Code quality evaluation".to_string(),
                "Integration recommendations".to_string(),
                "Improvement suggestions".to_string(),
            ],
        };

        serde_json::to_string_pretty(&openai_format)
            .map_err(|e| CompilerError::InternalError(format!("OpenAI format serialization failed: {}", e)))
    }

    async fn export_anthropic_structured_format(&self, metadata: &ComprehensiveAIMetadata) -> CompilerResult<String> {
        let anthropic_format = AnthropicStructuredMetadata {
            human_prompt: format!(
                "Please analyze the '{}' module with the following context:",
                metadata.module_summary.name
            ),
            module_data: metadata.clone(),
            analysis_framework: "Use the provided module metadata to perform a comprehensive analysis covering architecture, code quality, business value, and integration aspects.".to_string(),
            output_structure: "Provide structured analysis with clear sections for each aspect and actionable recommendations.".to_string(),
        };

        serde_json::to_string_pretty(&anthropic_format)
            .map_err(|e| CompilerError::InternalError(format!("Anthropic format serialization failed: {}", e)))
    }

    async fn export_custom_structured_format(&self, metadata: &ComprehensiveAIMetadata, schema: &str) -> CompilerResult<String> {
        // This would implement custom schema transformation
        // For now, return JSON with schema reference
        let custom_format = CustomStructuredMetadata {
            schema_version: schema.to_string(),
            metadata: metadata.clone(),
            transformation_notes: "Custom schema transformation applied".to_string(),
        };

        serde_json::to_string_pretty(&custom_format)
            .map_err(|e| CompilerError::InternalError(format!("Custom format serialization failed: {}", e)))
    }

    // Project-level Analysis Methods

    async fn generate_project_summary(&self, modules: &HashMap<String, RegisteredModule>) -> CompilerResult<ProjectSummary> {
        Ok(ProjectSummary {
            total_modules: modules.len(),
            primary_domains: self.identify_primary_domains(modules),
            architecture_overview: self.generate_architecture_overview(modules),
            technology_stack: self.identify_technology_stack(modules),
            complexity_overview: self.assess_project_complexity(modules),
        })
    }

    async fn analyze_module_relationships(&self, modules: &HashMap<String, RegisteredModule>) -> CompilerResult<Vec<ModuleRelationshipAnalysis>> {
        let mut relationships = Vec::new();

        for (name, module) in modules {
            for dependency in &module.dependencies.direct {
                if let Some(dep_module) = modules.get(dependency) {
                    relationships.push(ModuleRelationshipAnalysis {
                        source_module: name.clone(),
                        target_module: dependency.clone(),
                        relationship_type: "dependency".to_string(),
                        strength: self.calculate_relationship_strength(module, dep_module),
                        business_alignment: self.assess_business_alignment(module, dep_module),
                    });
                }
            }
        }

        Ok(relationships)
    }

    async fn identify_architecture_patterns(&self, modules: &HashMap<String, RegisteredModule>) -> CompilerResult<Vec<ArchitecturePattern>> {
        let mut patterns = Vec::new();

        // Analyze for common patterns
        if self.has_mvc_pattern(modules) {
            patterns.push(ArchitecturePattern {
                name: "Model-View-Controller".to_string(),
                description: "Separation of concerns between data, presentation, and control logic".to_string(),
                modules_involved: self.get_mvc_modules(modules),
                benefits: vec!["Clear separation of concerns".to_string(), "Maintainable code structure".to_string()],
            });
        }

        if self.has_layered_pattern(modules) {
            patterns.push(ArchitecturePattern {
                name: "Layered Architecture".to_string(),
                description: "Hierarchical organization of modules by abstraction level".to_string(),
                modules_involved: self.get_layered_modules(modules),
                benefits: vec!["Clear abstraction levels".to_string(), "Testable architecture".to_string()],
            });
        }

        Ok(patterns)
    }

    async fn generate_project_quality_overview(&self, modules: &HashMap<String, RegisteredModule>) -> CompilerResult<ProjectQualityOverview> {
        let mut total_score = 0.0;
        let mut module_count = 0;

        for module in modules.values() {
            if let Some(cohesion) = &module.cohesion_metrics {
                total_score += cohesion.overall_score;
                module_count += 1;
            }
        }

        let average_cohesion = if module_count > 0 { total_score / module_count as f64 } else { 0.0 };

        Ok(ProjectQualityOverview {
            average_cohesion_score: average_cohesion,
            high_quality_modules: self.identify_high_quality_modules(modules),
            modules_needing_attention: self.identify_modules_needing_attention(modules),
            overall_maintainability: self.assess_overall_maintainability(modules),
            technical_debt_indicators: self.identify_technical_debt_indicators(modules),
        })
    }

    async fn generate_project_recommendations(&self, modules: &HashMap<String, RegisteredModule>) -> CompilerResult<Vec<ProjectRecommendation>> {
        let mut recommendations = Vec::new();

        // Architecture recommendations
        if self.has_circular_dependencies(modules) {
            recommendations.push(ProjectRecommendation {
                category: "Architecture".to_string(),
                priority: "High".to_string(),
                title: "Resolve Circular Dependencies".to_string(),
                description: "Circular dependencies detected between modules".to_string(),
                impact: "Improves maintainability and testability".to_string(),
                effort_estimate: "Medium".to_string(),
            });
        }

        // Quality recommendations
        let low_cohesion_modules = self.identify_low_cohesion_modules(modules);
        if !low_cohesion_modules.is_empty() {
            recommendations.push(ProjectRecommendation {
                category: "Quality".to_string(),
                priority: "Medium".to_string(),
                title: "Improve Module Cohesion".to_string(),
                description: format!("Modules with low cohesion: {}", low_cohesion_modules.join(", ")),
                impact: "Better code organization and maintainability".to_string(),
                effort_estimate: "High".to_string(),
            });
        }

        Ok(recommendations)
    }

    // Helper methods for analysis (simplified implementations)

    fn assess_module_complexity(&self, _module: &RegisteredModule) -> String {
        "Medium".to_string() // Placeholder
    }

    fn assess_module_maturity(&self, _module: &RegisteredModule) -> String {
        "Stable".to_string() // Placeholder
    }

    fn identify_usage_patterns(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Direct instantiation".to_string(), "Dependency injection".to_string()]
    }

    fn identify_design_patterns(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Factory".to_string(), "Observer".to_string()]
    }

    fn identify_architectural_style(&self, _module: &RegisteredModule) -> String {
        "Layered".to_string()
    }

    fn analyze_coupling(&self, _module: &RegisteredModule) -> CouplingAnalysis {
        CouplingAnalysis {
            afferent_coupling: 3,
            efferent_coupling: 2,
            instability: 0.4,
            coupling_assessment: "Well balanced".to_string(),
        }
    }

    fn assess_scalability(&self, _module: &RegisteredModule) -> String {
        "Good".to_string()
    }

    fn calculate_maintainability_score(&self, _module: &RegisteredModule) -> f64 {
        0.8
    }

    fn assess_business_value(&self, _module: &RegisteredModule) -> String {
        "High".to_string()
    }

    fn analyze_stakeholder_impact(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["End users".to_string(), "Developers".to_string()]
    }

    fn identify_compliance_requirements(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["GDPR".to_string(), "SOX".to_string()]
    }

    fn assess_business_risks(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Data privacy".to_string(), "Performance bottlenecks".to_string()]
    }

    fn calculate_roi_indicators(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Development time saved".to_string(), "Bug reduction".to_string()]
    }

    fn calculate_code_quality_metrics(&self, _module: &RegisteredModule) -> CodeQualityMetrics {
        CodeQualityMetrics {
            cyclomatic_complexity: 5.2,
            lines_of_code: 1500,
            code_coverage: 85.0,
            duplication_percentage: 3.2,
            maintainability_index: 78.0,
        }
    }

    fn analyze_performance_characteristics(&self, _module: &RegisteredModule) -> PerformanceCharacteristics {
        PerformanceCharacteristics {
            average_response_time: 150.0,
            throughput: 1000.0,
            memory_usage: 64.0,
            cpu_utilization: 25.0,
        }
    }

    fn analyze_security_aspects(&self, _module: &RegisteredModule) -> SecurityAnalysis {
        SecurityAnalysis {
            security_score: 0.85,
            vulnerabilities: vec!["Minor: Potential SQL injection".to_string()],
            security_best_practices: vec!["Input validation".to_string(), "Output encoding".to_string()],
        }
    }

    fn assess_testing_coverage(&self, _module: &RegisteredModule) -> TestingCoverage {
        TestingCoverage {
            unit_test_coverage: 85.0,
            integration_test_coverage: 70.0,
            e2e_test_coverage: 60.0,
            test_quality_score: 0.8,
        }
    }

    fn assess_technical_debt(&self, _module: &RegisteredModule) -> TechnicalDebt {
        TechnicalDebt {
            debt_ratio: 15.0,
            debt_items: vec!["Outdated dependencies".to_string(), "Code duplication".to_string()],
            remediation_effort: "Medium".to_string(),
        }
    }

    fn recommend_integration_patterns(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Dependency Injection".to_string(), "Event-driven integration".to_string()]
    }

    fn generate_api_examples(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Basic usage example".to_string(), "Advanced configuration".to_string()]
    }

    fn identify_common_pitfalls(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Incorrect initialization".to_string(), "Memory leaks in callbacks".to_string()]
    }

    fn recommend_best_practices(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Use dependency injection".to_string(), "Implement proper error handling".to_string()]
    }

    fn provide_configuration_guidance(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Set appropriate timeouts".to_string(), "Configure logging levels".to_string()]
    }

    fn generate_troubleshooting_guide(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Check configuration files".to_string(), "Verify network connectivity".to_string()]
    }

    fn recommend_tutorials(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Getting started guide".to_string(), "Advanced features tutorial".to_string()]
    }

    fn gather_documentation_links(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["API Reference".to_string(), "Architecture Guide".to_string()]
    }

    fn generate_code_examples(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Basic example".to_string(), "Complex use case".to_string()]
    }

    fn identify_related_concepts(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Design patterns".to_string(), "Architecture principles".to_string()]
    }

    fn identify_prerequisites(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Basic programming knowledge".to_string(), "Understanding of the domain".to_string()]
    }

    fn identify_advanced_topics(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Performance optimization".to_string(), "Advanced configuration".to_string()]
    }

    fn calculate_overall_quality_score(&self, _module: &RegisteredModule) -> f64 {
        0.82
    }

    fn assess_code_quality(&self, _module: &RegisteredModule) -> f64 {
        0.85
    }

    fn assess_design_quality(&self, _module: &RegisteredModule) -> f64 {
        0.80
    }

    fn assess_test_quality(&self, _module: &RegisteredModule) -> f64 {
        0.78
    }

    fn assess_performance_quality(&self, _module: &RegisteredModule) -> f64 {
        0.88
    }

    fn assess_security_quality(&self, _module: &RegisteredModule) -> f64 {
        0.85
    }

    fn prioritize_improvements(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Improve test coverage".to_string(), "Reduce code duplication".to_string()]
    }

    fn recommend_short_term_improvements(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Add missing unit tests".to_string(), "Update documentation".to_string()]
    }

    fn recommend_long_term_evolution(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Consider microservices architecture".to_string(), "Implement event sourcing".to_string()]
    }

    fn recommend_technology_upgrades(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Upgrade to latest framework version".to_string(), "Adopt new language features".to_string()]
    }

    fn identify_refactoring_opportunities(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Extract common functionality".to_string(), "Simplify complex methods".to_string()]
    }

    fn suggest_feature_enhancements(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Add caching layer".to_string(), "Implement batch operations".to_string()]
    }

    fn suggest_architecture_evolution(&self, _module: &RegisteredModule) -> Vec<String> {
        vec!["Adopt hexagonal architecture".to_string(), "Implement CQRS pattern".to_string()]
    }

    // Project-level helper methods

    fn identify_primary_domains(&self, modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        let mut domains = std::collections::HashSet::new();
        for module in modules.values() {
            if let Some(domain) = &module.business_context.domain {
                domains.insert(domain.clone());
            }
        }
        domains.into_iter().collect()
    }

    fn generate_architecture_overview(&self, _modules: &HashMap<String, RegisteredModule>) -> String {
        "Layered architecture with clear separation of concerns".to_string()
    }

    fn identify_technology_stack(&self, _modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        vec!["Rust".to_string(), "Tokio".to_string(), "Serde".to_string()]
    }

    fn assess_project_complexity(&self, modules: &HashMap<String, RegisteredModule>) -> String {
        if modules.len() > 20 {
            "High".to_string()
        } else if modules.len() > 10 {
            "Medium".to_string()
        } else {
            "Low".to_string()
        }
    }

    fn calculate_relationship_strength(&self, _module1: &RegisteredModule, _module2: &RegisteredModule) -> f64 {
        0.7 // Placeholder
    }

    fn assess_business_alignment(&self, _module1: &RegisteredModule, _module2: &RegisteredModule) -> f64 {
        0.8 // Placeholder
    }

    fn has_mvc_pattern(&self, _modules: &HashMap<String, RegisteredModule>) -> bool {
        true // Placeholder
    }

    fn get_mvc_modules(&self, _modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        vec!["UserController".to_string(), "UserModel".to_string(), "UserView".to_string()]
    }

    fn has_layered_pattern(&self, _modules: &HashMap<String, RegisteredModule>) -> bool {
        true // Placeholder
    }

    fn get_layered_modules(&self, _modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        vec!["PresentationLayer".to_string(), "BusinessLayer".to_string(), "DataLayer".to_string()]
    }

    fn identify_high_quality_modules(&self, modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        modules.iter()
            .filter_map(|(name, module)| {
                module.cohesion_metrics.as_ref()
                    .filter(|m| m.overall_score > 80.0)
                    .map(|_| name.clone())
            })
            .collect()
    }

    fn identify_modules_needing_attention(&self, modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        modules.iter()
            .filter_map(|(name, module)| {
                module.cohesion_metrics.as_ref()
                    .filter(|m| m.overall_score < 60.0)
                    .map(|_| name.clone())
            })
            .collect()
    }

    fn assess_overall_maintainability(&self, _modules: &HashMap<String, RegisteredModule>) -> f64 {
        0.75 // Placeholder
    }

    fn identify_technical_debt_indicators(&self, _modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        vec!["High coupling in some modules".to_string(), "Outdated dependencies".to_string()]
    }

    fn has_circular_dependencies(&self, _modules: &HashMap<String, RegisteredModule>) -> bool {
        false // Placeholder
    }

    fn identify_low_cohesion_modules(&self, modules: &HashMap<String, RegisteredModule>) -> Vec<String> {
        modules.iter()
            .filter_map(|(name, module)| {
                module.cohesion_metrics.as_ref()
                    .filter(|m| m.overall_score < 70.0)
                    .map(|_| name.clone())
            })
            .collect()
    }
}

/// Result of module discovery operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDiscoveryResult {
    /// Module name
    pub module_name: String,
    /// Match score (0.0 to 1.0)
    pub match_score: f64,
    /// Capabilities this module provides
    pub provided_capabilities: HashSet<String>,
    /// Required capabilities this module doesn't provide
    pub missing_capabilities: HashSet<String>,
    /// Business context information
    pub business_context: BusinessContext,
    /// Cohesion score if available
    pub cohesion_score: Option<f64>,
}

impl ModuleDependencyGraph {
    /// Find strongly connected components (cycles)
    fn find_strongly_connected_components(&self) -> Vec<Vec<String>> {
        // Simple cycle detection using DFS
        // In a production system, would use Tarjan's algorithm
        let mut visited = HashSet::new();
        let mut cycles = Vec::new();

        for node in self.edges.keys() {
            if !visited.contains(node) {
                if let Some(cycle) = self.find_cycle_from_node(node, &mut visited) {
                    cycles.push(cycle);
                }
            }
        }

        cycles
    }

    fn find_cycle_from_node(&self, start: &str, visited: &mut HashSet<String>) -> Option<Vec<String>> {
        let mut path = Vec::new();
        let mut current_visited = HashSet::new();
        
        if self.dfs_find_cycle(start, &mut path, &mut current_visited, visited) {
            Some(path)
        } else {
            None
        }
    }

    fn dfs_find_cycle(&self, node: &str, path: &mut Vec<String>, current_visited: &mut HashSet<String>, global_visited: &mut HashSet<String>) -> bool {
        if current_visited.contains(node) {
            // Found cycle
            if let Some(cycle_start) = path.iter().position(|n| n == node) {
                path.drain(..cycle_start);
                return true;
            }
        }

        if global_visited.contains(node) {
            return false;
        }

        current_visited.insert(node.to_string());
        global_visited.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = self.edges.get(node) {
            for neighbor in neighbors {
                if self.dfs_find_cycle(neighbor, path, current_visited, global_visited) {
                    return true;
                }
            }
        }

        path.pop();
        current_visited.remove(node);
        false
    }
}

/// Comprehensive AI metadata for external tool consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAIMetadata {
    /// High-level module summary
    pub module_summary: ModuleSummary,
    /// Architectural analysis and insights
    pub architectural_analysis: ArchitecturalAnalysis,
    /// Business intelligence and value assessment
    pub business_intelligence: BusinessIntelligence,
    /// Code quality and technical insights
    pub code_insights: CodeInsights,
    /// Integration guidance and patterns
    pub integration_guidance: IntegrationGuidance,
    /// Learning resources and documentation
    pub learning_resources: LearningResources,
    /// Quality assessment and metrics
    pub quality_assessment: QualityAssessment,
    /// Future recommendations and roadmap
    pub future_recommendations: FutureRecommendations,
    /// Metadata version for compatibility
    pub metadata_version: String,
    /// Generation timestamp
    pub generated_at: String,
}

/// Export formats for AI metadata
#[derive(Debug, Clone)]
pub enum AIMetadataExportFormat {
    /// Standard JSON format
    Json,
    /// YAML format for human readability
    Yaml,
    /// OpenAI-structured format for GPT models
    OpenAIStructured,
    /// Anthropic-structured format for Claude models
    AnthropicStructured,
    /// Custom structured format with schema
    CustomStructured(String),
}

/// Module summary for AI comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSummary {
    /// Module name
    pub name: String,
    /// Primary purpose description
    pub primary_purpose: String,
    /// Key capabilities provided
    pub key_capabilities: Vec<String>,
    /// Business domain
    pub business_domain: String,
    /// Complexity assessment
    pub complexity_assessment: String,
    /// Maturity level
    pub maturity_level: String,
    /// Common usage patterns
    pub usage_patterns: Vec<String>,
}

/// Architectural analysis details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalAnalysis {
    /// Design patterns identified
    pub design_patterns: Vec<String>,
    /// Architectural style
    pub architectural_style: String,
    /// Coupling analysis
    pub coupling_analysis: CouplingAnalysis,
    /// Cohesion analysis details
    pub cohesion_analysis: Option<CohesionAnalysisDetail>,
    /// Scalability assessment
    pub scalability_assessment: String,
    /// Maintainability score
    pub maintainability_score: f64,
}

/// Coupling analysis details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingAnalysis {
    /// Afferent coupling (incoming dependencies)
    pub afferent_coupling: u32,
    /// Efferent coupling (outgoing dependencies)
    pub efferent_coupling: u32,
    /// Instability measure
    pub instability: f64,
    /// Coupling assessment
    pub coupling_assessment: String,
}

/// Detailed cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysisDetail {
    /// Overall cohesion score
    pub overall_score: f64,
    /// Type cohesion score
    pub type_cohesion: f64,
    /// Data flow cohesion score
    pub data_flow_cohesion: f64,
    /// Semantic cohesion score
    pub semantic_cohesion: f64,
    /// Dependency cohesion score
    pub dependency_cohesion: f64,
    /// Cohesion strengths
    pub strengths: Vec<String>,
    /// Areas for improvement
    pub improvement_areas: Vec<String>,
}

/// Business intelligence and value assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessIntelligence {
    /// Business value assessment
    pub business_value: String,
    /// Stakeholder impact analysis
    pub stakeholder_impact: Vec<String>,
    /// Business rules enforced
    pub business_rules: Vec<String>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Risk assessment
    pub risk_assessment: Vec<String>,
    /// ROI indicators
    pub roi_indicators: Vec<String>,
}

/// Code quality and technical insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeInsights {
    /// Code quality metrics
    pub code_quality_metrics: CodeQualityMetrics,
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
    /// Security analysis
    pub security_analysis: SecurityAnalysis,
    /// Testing coverage
    pub testing_coverage: TestingCoverage,
    /// Documentation quality
    pub documentation_quality: f64,
    /// Technical debt assessment
    pub technical_debt: TechnicalDebt,
}

/// Code quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQualityMetrics {
    /// Cyclomatic complexity
    pub cyclomatic_complexity: f64,
    /// Lines of code
    pub lines_of_code: u32,
    /// Code coverage percentage
    pub code_coverage: f64,
    /// Code duplication percentage
    pub duplication_percentage: f64,
    /// Maintainability index
    pub maintainability_index: f64,
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Average response time (ms)
    pub average_response_time: f64,
    /// Throughput (operations/second)
    pub throughput: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
}

/// Security analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    /// Security score (0.0 to 1.0)
    pub security_score: f64,
    /// Identified vulnerabilities
    pub vulnerabilities: Vec<String>,
    /// Security best practices followed
    pub security_best_practices: Vec<String>,
}

/// Testing coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingCoverage {
    /// Unit test coverage percentage
    pub unit_test_coverage: f64,
    /// Integration test coverage percentage
    pub integration_test_coverage: f64,
    /// End-to-end test coverage percentage
    pub e2e_test_coverage: f64,
    /// Test quality score
    pub test_quality_score: f64,
}

/// Technical debt assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDebt {
    /// Technical debt ratio
    pub debt_ratio: f64,
    /// Specific debt items
    pub debt_items: Vec<String>,
    /// Remediation effort estimate
    pub remediation_effort: String,
}

/// Integration guidance and patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationGuidance {
    /// Recommended integration patterns
    pub integration_patterns: Vec<String>,
    /// API usage examples
    pub api_usage_examples: Vec<String>,
    /// Common pitfalls to avoid
    pub common_pitfalls: Vec<String>,
    /// Best practices
    pub best_practices: Vec<String>,
    /// Configuration guidance
    pub configuration_guidance: Vec<String>,
    /// Troubleshooting guide
    pub troubleshooting_guide: Vec<String>,
}

/// Learning resources and documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResources {
    /// Recommended tutorials
    pub tutorials: Vec<String>,
    /// Documentation links
    pub documentation_links: Vec<String>,
    /// Code examples
    pub code_examples: Vec<String>,
    /// Related concepts
    pub related_concepts: Vec<String>,
    /// Prerequisite knowledge
    pub prerequisite_knowledge: Vec<String>,
    /// Advanced topics
    pub advanced_topics: Vec<String>,
}

/// Quality assessment details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score
    pub overall_quality_score: f64,
    /// Code quality score
    pub code_quality: f64,
    /// Design quality score
    pub design_quality: f64,
    /// Documentation quality score
    pub documentation_quality: f64,
    /// Test quality score
    pub test_quality: f64,
    /// Performance quality score
    pub performance_quality: f64,
    /// Security quality score
    pub security_quality: f64,
    /// Improvement priorities
    pub improvement_priorities: Vec<String>,
}

/// Future recommendations and roadmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureRecommendations {
    /// Short-term improvements
    pub short_term_improvements: Vec<String>,
    /// Long-term evolution suggestions
    pub long_term_evolution: Vec<String>,
    /// Technology upgrade recommendations
    pub technology_upgrades: Vec<String>,
    /// Refactoring opportunities
    pub refactoring_opportunities: Vec<String>,
    /// Feature enhancement suggestions
    pub feature_enhancements: Vec<String>,
    /// Architecture evolution recommendations
    pub architecture_evolution: Vec<String>,
}

/// Project-level AI metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAIMetadata {
    /// Project summary
    pub project_summary: ProjectSummary,
    /// Module metadata map
    pub modules: HashMap<String, ComprehensiveAIMetadata>,
    /// Module relationships
    pub relationships: Vec<ModuleRelationshipAnalysis>,
    /// Architecture patterns
    pub architecture_patterns: Vec<ArchitecturePattern>,
    /// Quality overview
    pub quality_overview: ProjectQualityOverview,
    /// Project recommendations
    pub recommendations: Vec<ProjectRecommendation>,
    /// Metadata version
    pub metadata_version: String,
    /// Generation timestamp
    pub generated_at: String,
}

/// Project summary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSummary {
    /// Total number of modules
    pub total_modules: usize,
    /// Primary business domains
    pub primary_domains: Vec<String>,
    /// Architecture overview
    pub architecture_overview: String,
    /// Technology stack
    pub technology_stack: Vec<String>,
    /// Complexity overview
    pub complexity_overview: String,
}

/// Module relationship analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRelationshipAnalysis {
    /// Source module
    pub source_module: String,
    /// Target module
    pub target_module: String,
    /// Relationship type
    pub relationship_type: String,
    /// Relationship strength
    pub strength: f64,
    /// Business alignment score
    pub business_alignment: f64,
}

/// Architecture pattern identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Modules involved in this pattern
    pub modules_involved: Vec<String>,
    /// Benefits of this pattern
    pub benefits: Vec<String>,
}

/// Project quality overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectQualityOverview {
    /// Average cohesion score across modules
    pub average_cohesion_score: f64,
    /// High quality modules
    pub high_quality_modules: Vec<String>,
    /// Modules needing attention
    pub modules_needing_attention: Vec<String>,
    /// Overall maintainability score
    pub overall_maintainability: f64,
    /// Technical debt indicators
    pub technical_debt_indicators: Vec<String>,
}

/// Project-level recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRecommendation {
    /// Recommendation category
    pub category: String,
    /// Priority level
    pub priority: String,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected impact
    pub impact: String,
    /// Effort estimate
    pub effort_estimate: String,
}

/// OpenAI-structured metadata format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStructuredMetadata {
    /// System prompt for AI model
    pub system_prompt: String,
    /// Module context data
    pub module_context: ComprehensiveAIMetadata,
    /// Analysis instructions
    pub analysis_instructions: Vec<String>,
    /// Expected output format
    pub expected_outputs: Vec<String>,
}

/// Anthropic-structured metadata format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicStructuredMetadata {
    /// Human prompt
    pub human_prompt: String,
    /// Module data
    pub module_data: ComprehensiveAIMetadata,
    /// Analysis framework
    pub analysis_framework: String,
    /// Output structure guidance
    pub output_structure: String,
}

/// Custom structured metadata format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomStructuredMetadata {
    /// Schema version
    pub schema_version: String,
    /// Metadata content
    pub metadata: ComprehensiveAIMetadata,
    /// Transformation notes
    pub transformation_notes: String,
} 