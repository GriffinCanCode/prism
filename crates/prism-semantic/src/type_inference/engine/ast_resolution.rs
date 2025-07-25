//! AST Type Resolution
//!
//! This module handles the conversion of AST type representations to semantic types.
//! It provides comprehensive type resolution with proper error handling and semantic enrichment.
//!
//! **Single Responsibility**: AST type to semantic type conversion
//! **What it does**: Resolve AST types, handle generics, manage type contexts
//! **What it doesn't do**: Perform inference, manage environments, handle constraints

use crate::{
    SemanticResult, SemanticError,
    types::{SemanticType, SemanticTypeMetadata, BaseType},
    type_inference::{
        TypeVar, InferredType, InferenceSource, TypeVarGenerator,
        environment::TypeEnvironment,
        constraints::ConstraintSet,
        constraints::{TypeConstraint, ConstraintType, ConstraintReason},
        unification::Substitution,
    },
};
use prism_ast::{Type as AstType, TypeParameter, ArrayType, TupleType, UnionType, GenericType};
use prism_ast::types::FunctionType;
use prism_common::Span;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// AST type resolver for converting AST types to semantic types
#[derive(Debug)]
pub struct ASTTypeResolver {
    /// Configuration for type resolution
    config: TypeResolutionConfig,
    /// Type variable generator for generics
    type_var_gen: TypeVarGenerator,
    /// Type context stack for nested resolution
    context_stack: Vec<TypeResolutionContext>,
    /// Generic parameter mappings
    generic_mappings: HashMap<String, SemanticType>,
    /// Type alias resolution cache
    alias_cache: HashMap<String, SemanticType>,
    /// Built-in type mappings
    builtin_types: BuiltinTypeMappings,
    /// Resolution statistics
    resolution_stats: ResolutionStatistics,
}

/// Configuration for type resolution
#[derive(Debug, Clone)]
pub struct TypeResolutionConfig {
    /// Enable detailed error reporting
    pub enable_detailed_errors: bool,
    /// Enable type alias resolution
    pub enable_alias_resolution: bool,
    /// Enable generic type resolution
    pub enable_generic_resolution: bool,
    /// Enable semantic type enrichment
    pub enable_semantic_enrichment: bool,
    /// Maximum resolution depth
    pub max_resolution_depth: usize,
    /// Enable resolution caching
    pub enable_caching: bool,
}

/// Context for type resolution
#[derive(Debug, Clone)]
struct TypeResolutionContext {
    /// Context type
    context_type: ResolutionContextType,
    /// Generic parameters in scope
    generic_parameters: HashMap<String, TypeParameter>,
    /// Type variables in scope
    type_variables: HashMap<String, TypeVar>,
    /// Current resolution depth
    depth: usize,
    /// Context span
    span: Span,
}

/// Types of resolution contexts
#[derive(Debug, Clone)]
enum ResolutionContextType {
    /// Top-level resolution
    TopLevel,
    /// Function type resolution
    Function,
    /// Generic type resolution
    Generic,
    /// Array type resolution
    Array,
    /// Tuple type resolution
    Tuple,
    /// Union type resolution
    Union,
    /// Record type resolution
    Record,
}

/// Built-in type mappings
#[derive(Debug)]
struct BuiltinTypeMappings {
    /// Primitive type mappings
    primitive_mappings: HashMap<String, prism_ast::PrimitiveType>,
    /// Collection type mappings
    collection_mappings: HashMap<String, CollectionTypeInfo>,
    /// Special type mappings
    special_mappings: HashMap<String, SpecialTypeInfo>,
}

/// Information about collection types
#[derive(Debug, Clone)]
struct CollectionTypeInfo {
    /// Collection name
    name: String,
    /// Number of type parameters
    type_param_count: usize,
    /// Semantic representation template
    semantic_template: String,
    /// Default constraints
    constraints: Vec<String>,
}

/// Information about special types
#[derive(Debug, Clone)]
struct SpecialTypeInfo {
    /// Special type name
    name: String,
    /// Semantic representation
    semantic_type: SemanticType,
    /// Special handling required
    requires_special_handling: bool,
}

/// Result of type resolution
#[derive(Debug, Clone)]
pub struct TypeResolutionResult {
    /// Resolved semantic type
    pub semantic_type: SemanticType,
    /// Resolution metadata
    pub metadata: ResolutionMetadata,
    /// Generated constraints
    pub constraints: Vec<TypeResolutionConstraint>,
    /// Resolution warnings
    pub warnings: Vec<ResolutionWarning>,
}

/// Metadata about type resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionMetadata {
    /// Original AST type representation
    pub original_ast_type: String,
    /// Resolution complexity score
    pub complexity_score: f64,
    /// Generic parameters resolved
    pub generic_parameters_resolved: Vec<String>,
    /// Type aliases resolved
    pub aliases_resolved: Vec<String>,
    /// Resolution confidence
    pub confidence: f64,
    /// Resolution source
    pub resolution_source: ResolutionSource,
}

/// Source of type resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionSource {
    /// Built-in type
    Builtin,
    /// User-defined type
    UserDefined,
    /// Generic instantiation
    GenericInstantiation,
    /// Type alias expansion
    AliasExpansion,
    /// Inferred type
    Inferred,
    /// Default resolution
    Default,
}

/// Type resolution constraint
#[derive(Debug, Clone)]
pub struct TypeResolutionConstraint {
    /// Constraint type
    pub constraint_type: ResolutionConstraintType,
    /// Constraint description
    pub description: String,
    /// Source span
    pub span: Span,
    /// Constraint priority
    pub priority: u32,
}

/// Types of resolution constraints
#[derive(Debug, Clone)]
pub enum ResolutionConstraintType {
    /// Generic parameter bounds
    GenericBounds,
    /// Type compatibility
    Compatibility,
    /// Semantic consistency
    SemanticConsistency,
    /// Resource constraints
    ResourceConstraints,
}

/// Resolution warning
#[derive(Debug, Clone)]
pub struct ResolutionWarning {
    /// Warning type
    pub warning_type: ResolutionWarningType,
    /// Warning message
    pub message: String,
    /// Source span
    pub span: Span,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Types of resolution warnings
#[derive(Debug, Clone)]
pub enum ResolutionWarningType {
    /// Deprecated type usage
    DeprecatedType,
    /// Ambiguous resolution
    AmbiguousResolution,
    /// Performance concern
    PerformanceConcern,
    /// Semantic inconsistency
    SemanticInconsistency,
    /// Missing type parameters
    MissingTypeParameters,
}

/// Resolution statistics
#[derive(Debug, Clone)]
struct ResolutionStatistics {
    /// Total resolutions performed
    total_resolutions: usize,
    /// Successful resolutions
    successful_resolutions: usize,
    /// Failed resolutions
    failed_resolutions: usize,
    /// Cache hits
    cache_hits: usize,
    /// Cache misses
    cache_misses: usize,
    /// Average resolution time
    avg_resolution_time: std::time::Duration,
}

impl ASTTypeResolver {
    /// Create a new AST type resolver
    pub fn new() -> Self {
        Self {
            config: TypeResolutionConfig::default(),
            type_var_gen: TypeVarGenerator::new(),
            context_stack: Vec::new(),
            generic_mappings: HashMap::new(),
            alias_cache: HashMap::new(),
            builtin_types: BuiltinTypeMappings::new(),
            resolution_stats: ResolutionStatistics::default(),
        }
    }

    /// Create AST type resolver with custom configuration
    pub fn with_config(config: TypeResolutionConfig) -> Self {
        Self {
            config,
            type_var_gen: TypeVarGenerator::new(),
            context_stack: Vec::new(),
            generic_mappings: HashMap::new(),
            alias_cache: HashMap::new(),
            builtin_types: BuiltinTypeMappings::new(),
            resolution_stats: ResolutionStatistics::default(),
        }
    }

    /// Resolve an AST type to a semantic type
    pub fn resolve_ast_type(&mut self, ast_type: &AstType) -> SemanticResult<SemanticType> {
        let start_time = std::time::Instant::now();
        self.resolution_stats.total_resolutions += 1;

        let result = self.resolve_ast_type_internal(ast_type, Span::dummy());
        
        let elapsed = start_time.elapsed();
        self.update_resolution_stats(elapsed, result.is_ok());

        match result {
            Ok(semantic_type) => {
                self.resolution_stats.successful_resolutions += 1;
                Ok(semantic_type)
            }
            Err(error) => {
                self.resolution_stats.failed_resolutions += 1;
                Err(error)
            }
        }
    }

    /// Resolve an AST type with full result information
    pub fn resolve_ast_type_with_metadata(&mut self, ast_type: &AstType, span: Span) -> SemanticResult<TypeResolutionResult> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let cache_key = self.create_cache_key(ast_type);
        if self.config.enable_caching {
            if let Some(cached_type) = self.alias_cache.get(&cache_key) {
                self.resolution_stats.cache_hits += 1;
                return Ok(TypeResolutionResult {
                    semantic_type: cached_type.clone(),
                    metadata: ResolutionMetadata {
                        original_ast_type: format!("{:?}", ast_type),
                        complexity_score: 1.0,
                        generic_parameters_resolved: Vec::new(),
                        aliases_resolved: Vec::new(),
                        confidence: 1.0,
                        resolution_source: ResolutionSource::Builtin,
                    },
                    constraints: Vec::new(),
                    warnings: Vec::new(),
                });
            }
            self.resolution_stats.cache_misses += 1;
        }

        // Perform resolution
        let semantic_type = self.resolve_ast_type_internal(ast_type, span)?;
        
        // Generate metadata
        let metadata = self.generate_resolution_metadata(ast_type, &semantic_type, start_time.elapsed());
        
        // Generate constraints
        let constraints = self.generate_resolution_constraints(ast_type, &semantic_type, span);
        
        // Generate warnings
        let warnings = self.generate_resolution_warnings(ast_type, &semantic_type, span);
        
        // Cache result
        if self.config.enable_caching {
            self.alias_cache.insert(cache_key, semantic_type.clone());
        }

        Ok(TypeResolutionResult {
            semantic_type,
            metadata,
            constraints,
            warnings,
        })
    }

    /// Set generic parameter mappings
    pub fn set_generic_mappings(&mut self, mappings: HashMap<String, SemanticType>) {
        self.generic_mappings = mappings;
    }

    /// Add a generic parameter mapping
    pub fn add_generic_mapping(&mut self, name: String, semantic_type: SemanticType) {
        self.generic_mappings.insert(name, semantic_type);
    }

    /// Clear generic parameter mappings
    pub fn clear_generic_mappings(&mut self) {
        self.generic_mappings.clear();
    }

    /// Reset the resolver state
    pub fn reset(&mut self) {
        self.type_var_gen = TypeVarGenerator::new();
        self.context_stack.clear();
        self.generic_mappings.clear();
        self.alias_cache.clear();
        self.resolution_stats = ResolutionStatistics::default();
    }

    /// Get resolution statistics
    pub fn get_statistics(&self) -> &ResolutionStatistics {
        &self.resolution_stats
    }

    // Private implementation methods

    fn resolve_ast_type_internal(&mut self, ast_type: &AstType, span: Span) -> SemanticResult<SemanticType> {
        // Check recursion depth
        if self.context_stack.len() >= self.config.max_resolution_depth {
            return Err(SemanticError::TypeInferenceError {
                message: format!("Maximum type resolution depth exceeded: {}", self.config.max_resolution_depth),
            });
        }

        match ast_type {
            AstType::Named(named_type) => {
                self.resolve_named_type(named_type, span)
            }
            AstType::Function(func_type) => {
                self.resolve_function_type(func_type, span)
            }
            AstType::Array(array_type) => {
                self.resolve_array_type(array_type, span)
            }
            AstType::Tuple(tuple_type) => {
                self.resolve_tuple_type(tuple_type, span)
            }
            AstType::Union(union_type) => {
                self.resolve_union_type(union_type, span)
            }
            AstType::Generic(generic_type) => {
                self.resolve_generic_type(generic_type, span)
            }
            AstType::Primitive(prim_type) => {
                Ok(SemanticType::Primitive(prim_type.clone()))
            }
            AstType::Composite(comp_type) => {
                // For now, treat composite types as complex types
                Ok(SemanticType::Complex {
                    name: "CompositeType".to_string(),
                    base_type: BaseType::Primitive(crate::types::PrimitiveType::Custom { 
                        name: "CompositeType".to_string(), 
                        base: "composite".to_string() 
                    }),
                    constraints: Vec::new(),
                    business_rules: Vec::new(),
                    metadata: SemanticTypeMetadata::default(),
                    ai_context: None,
                    verification_properties: Vec::new(),
                    location: prism_common::Span::dummy(),
                })
            }
            AstType::Semantic(_sem_type) => {
                // TODO: Convert between prism_ast::SemanticType and crate::types::SemanticType
                // For now, use a placeholder
                Ok(SemanticType::Variable("semantic_type".to_string()))
            }
            AstType::Dependent(_dep_type) => {
                // For now, treat dependent types as type variables
                Ok(SemanticType::Variable("dependent_type".to_string()))
            }
            AstType::Effect(_eff_type) => {
                // For now, treat effect types as type variables
                Ok(SemanticType::Variable("effect_type".to_string()))
            }
            AstType::Intersection(_int_type) => {
                // For now, treat intersection types as type variables
                Ok(SemanticType::Variable("intersection_type".to_string()))
            }
            AstType::Computed(_comp_type) => {
                // For now, treat computed types as type variables
                Ok(SemanticType::Variable("computed_type".to_string()))
            }
            AstType::Error(_err_type) => {
                // Error types should be handled as type variables
                Ok(SemanticType::Variable("error_type".to_string()))
            }
        }
    }

    fn resolve_named_type(&mut self, named_type: &prism_ast::NamedType, span: Span) -> SemanticResult<SemanticType> {
        let type_name = named_type.name.resolve().unwrap_or_else(|| "unknown".to_string());

        // Check if it's a generic parameter
        if let Some(semantic_type) = self.generic_mappings.get(&type_name) {
            return Ok(semantic_type.clone());
        }

        // Check built-in primitive types
        if let Some(primitive_type) = self.builtin_types.primitive_mappings.get(&type_name) {
            return Ok(SemanticType::Primitive(primitive_type.clone()));
        }

        // Check built-in collection types
        if let Some(collection_info) = self.builtin_types.collection_mappings.get(&type_name) {
            return self.resolve_collection_type(collection_info, &[], span);
        }

        // Check special types
        if let Some(special_info) = self.builtin_types.special_mappings.get(&type_name) {
            return Ok(special_info.semantic_type.clone());
        }

        // Default case - create a complex type
        Ok(SemanticType::Complex {
            name: type_name.clone(),
            base_type: crate::types::BaseType::Primitive(crate::types::PrimitiveType::Custom {
                name: type_name.clone(),
                base: "unknown".to_string(),
            }),
            constraints: Vec::new(),
            business_rules: Vec::new(),
            metadata: crate::types::SemanticTypeMetadata::default(),
            ai_context: None,
            verification_properties: Vec::new(),
            location: span,
        })
    }

    fn resolve_function_type(&mut self, func_type: &FunctionType, span: Span) -> SemanticResult<SemanticType> {
        self.enter_context(ResolutionContextType::Function, span);

        // Resolve parameter types
        let mut param_types = Vec::new();
        for param in &func_type.parameters {
            let param_type = self.resolve_ast_type_internal(&param.kind, param.span)?;
            param_types.push(param_type);
        }

        // Resolve return type
        let return_type = self.resolve_ast_type_internal(&func_type.return_type.kind, func_type.return_type.span)?;

        // Resolve effects if present
        let effects = func_type.effects.iter()
            .map(|effect| format!("{:?}", effect))
            .collect();

        self.exit_context();

        Ok(SemanticType::Function {
            params: param_types,
            return_type: Box::new(return_type),
            effects,
        })
    }

    fn resolve_array_type(&mut self, array_type: &ArrayType, span: Span) -> SemanticResult<SemanticType> {
        self.enter_context(ResolutionContextType::Array, span);

        let element_type = self.resolve_ast_type_internal(&array_type.element_type.kind, array_type.element_type.span)?;

        self.exit_context();

        Ok(SemanticType::List(Box::new(element_type)))
    }

    fn resolve_tuple_type(&mut self, tuple_type: &TupleType, span: Span) -> SemanticResult<SemanticType> {
        self.enter_context(ResolutionContextType::Tuple, span);

        let mut fields = HashMap::new();
        for (i, element) in tuple_type.elements.iter().enumerate() {
            let element_type = self.resolve_ast_type_internal(&element.kind, element.span)?;
            fields.insert(i.to_string(), element_type);
        }

        self.exit_context();

        Ok(SemanticType::Record(fields))
    }

    fn resolve_union_type(&mut self, union_type: &UnionType, span: Span) -> SemanticResult<SemanticType> {
        self.enter_context(ResolutionContextType::Union, span);

        let mut variant_types = Vec::new();
        for member in &union_type.members {
            let member_type = self.resolve_ast_type_internal(&member.kind, member.span)?;
            variant_types.push(member_type);
        }

        self.exit_context();

        Ok(SemanticType::Union(variant_types))
    }

    fn resolve_generic_type(&mut self, generic_type: &GenericType, span: Span) -> SemanticResult<SemanticType> {
        self.enter_context(ResolutionContextType::Generic, span);

        // Resolve base type
        let base_type = self.resolve_ast_type_internal(&generic_type.base_type.kind, generic_type.base_type.span)?;

        // Resolve type parameters
        let mut type_parameters = Vec::new();
        for param in &generic_type.parameters {
            // For now, create type variables for parameters
            // In a full implementation, this would resolve parameter constraints
            let type_var = self.type_var_gen.fresh(Span::dummy()); // Use dummy span since TypeParameter doesn't have span
            type_parameters.push(SemanticType::Variable(type_var.id.to_string()));
        }

        self.exit_context();

        // Extract base type name
        let base_name = match base_type {
            SemanticType::Complex { name, .. } => name,
            _ => "Generic".to_string(),
        };

        Ok(SemanticType::Generic {
            name: base_name,
            parameters: type_parameters,
        })
    }

    fn resolve_collection_type(&self, collection_info: &CollectionTypeInfo, type_args: &[SemanticType], span: Span) -> SemanticResult<SemanticType> {
        match collection_info.name.as_str() {
            "Array" | "List" => {
                let element_type = type_args.get(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::Variable("T".to_string()));
                Ok(SemanticType::List(Box::new(element_type)))
            }
            "Map" => {
                let key_type = type_args.get(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::Variable("K".to_string()));
                let value_type = type_args.get(1)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::Variable("V".to_string()));
                
                // Represent Map as a generic type
                Ok(SemanticType::Generic {
                    name: "Map".to_string(),
                    parameters: vec![key_type, value_type],
                })
            }
            "Set" => {
                let element_type = type_args.get(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::Variable("T".to_string()));
                Ok(SemanticType::Generic {
                    name: "Set".to_string(),
                    parameters: vec![element_type],
                })
            }
            _ => {
                Ok(SemanticType::Generic {
                    name: collection_info.name.clone(),
                    parameters: type_args.to_vec(),
                })
            }
        }
    }

    fn enter_context(&mut self, context_type: ResolutionContextType, span: Span) {
        let depth = self.context_stack.len();
        let context = TypeResolutionContext {
            context_type,
            generic_parameters: HashMap::new(),
            type_variables: HashMap::new(),
            depth,
            span,
        };
        self.context_stack.push(context);
    }

    fn exit_context(&mut self) {
        self.context_stack.pop();
    }

    fn create_cache_key(&self, ast_type: &AstType) -> String {
        format!("{:?}", ast_type) // Simplified - would be more sophisticated
    }

    fn generate_resolution_metadata(&self, ast_type: &AstType, semantic_type: &SemanticType, elapsed: std::time::Duration) -> ResolutionMetadata {
        ResolutionMetadata {
            original_ast_type: format!("{:?}", ast_type),
            complexity_score: self.calculate_complexity_score(semantic_type),
            generic_parameters_resolved: self.extract_generic_parameters(semantic_type),
            aliases_resolved: Vec::new(), // Would track actual aliases resolved
            confidence: 0.9, // Would be calculated based on resolution quality
            resolution_source: self.determine_resolution_source(ast_type),
        }
    }

    fn generate_resolution_constraints(&self, _ast_type: &AstType, _semantic_type: &SemanticType, _span: Span) -> Vec<TypeResolutionConstraint> {
        // Would generate actual constraints based on type analysis
        Vec::new()
    }

    fn generate_resolution_warnings(&self, _ast_type: &AstType, _semantic_type: &SemanticType, _span: Span) -> Vec<ResolutionWarning> {
        // Would generate actual warnings based on type analysis
        Vec::new()
    }

    fn calculate_complexity_score(&self, semantic_type: &SemanticType) -> f64 {
        match semantic_type {
            SemanticType::Primitive(_) => 1.0,
            SemanticType::Function { params, .. } => 1.5 + (params.len() as f64 * 0.2),
            SemanticType::List(_) => 1.2,
            SemanticType::Record(fields) => 1.3 + (fields.len() as f64 * 0.1),
            SemanticType::Union(variants) => 1.4 + (variants.len() as f64 * 0.15),
            SemanticType::Generic { parameters, .. } => 1.2 + (parameters.len() as f64 * 0.3),
            SemanticType::Complex { .. } => 1.1, // Complex types have base complexity
            _ => 1.0,
        }
    }

    fn extract_generic_parameters(&self, semantic_type: &SemanticType) -> Vec<String> {
        match semantic_type {
            SemanticType::Generic { parameters, .. } => {
                parameters.iter()
                    .enumerate()
                    .map(|(i, _)| format!("T{}", i))
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    fn determine_resolution_source(&self, ast_type: &AstType) -> ResolutionSource {
        match ast_type {
            AstType::Named(named) => {
                let name = named.name.resolve().unwrap_or_else(|| "unknown".to_string());
                if self.builtin_types.primitive_mappings.contains_key(&name) ||
                   self.builtin_types.collection_mappings.contains_key(&name) ||
                   self.builtin_types.special_mappings.contains_key(&name) {
                    ResolutionSource::Builtin
                } else {
                    ResolutionSource::UserDefined
                }
            }
            AstType::Generic(_) => ResolutionSource::GenericInstantiation,
            _ => ResolutionSource::Default,
        }
    }

    fn update_resolution_stats(&mut self, elapsed: std::time::Duration, success: bool) {
        if self.resolution_stats.total_resolutions == 1 {
            self.resolution_stats.avg_resolution_time = elapsed;
        } else {
            let total = self.resolution_stats.total_resolutions as f64;
            let current_avg = self.resolution_stats.avg_resolution_time.as_nanos() as f64;
            let new_avg = (current_avg * (total - 1.0) + elapsed.as_nanos() as f64) / total;
            self.resolution_stats.avg_resolution_time = std::time::Duration::from_nanos(new_avg as u64);
        }
    }
}

impl BuiltinTypeMappings {
    fn new() -> Self {
        let mut primitive_mappings = HashMap::new();
        let mut collection_mappings = HashMap::new();
        let mut special_mappings = HashMap::new();

        // Initialize primitive types
        primitive_mappings.insert("Int".to_string(), prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)));
        primitive_mappings.insert("Integer".to_string(), prism_ast::PrimitiveType::Integer(prism_ast::IntegerType::Signed(32)));
        primitive_mappings.insert("Float".to_string(), prism_ast::PrimitiveType::Float(prism_ast::FloatType::F64));
        primitive_mappings.insert("String".to_string(), prism_ast::PrimitiveType::String);
        primitive_mappings.insert("Bool".to_string(), prism_ast::PrimitiveType::Boolean);
        primitive_mappings.insert("Boolean".to_string(), prism_ast::PrimitiveType::Boolean);
        primitive_mappings.insert("Char".to_string(), prism_ast::PrimitiveType::Char);
        primitive_mappings.insert("Unit".to_string(), prism_ast::PrimitiveType::Unit);

        // Initialize collection types
        collection_mappings.insert("Array".to_string(), CollectionTypeInfo {
            name: "Array".to_string(),
            type_param_count: 1,
            semantic_template: "List<T>".to_string(),
            constraints: Vec::new(),
        });
        collection_mappings.insert("List".to_string(), CollectionTypeInfo {
            name: "List".to_string(),
            type_param_count: 1,
            semantic_template: "List<T>".to_string(),
            constraints: Vec::new(),
        });
        collection_mappings.insert("Map".to_string(), CollectionTypeInfo {
            name: "Map".to_string(),
            type_param_count: 2,
            semantic_template: "Map<K, V>".to_string(),
            constraints: Vec::new(),
        });
        collection_mappings.insert("Set".to_string(), CollectionTypeInfo {
            name: "Set".to_string(),
            type_param_count: 1,
            semantic_template: "Set<T>".to_string(),
            constraints: Vec::new(),
        });

        // Initialize special types
        special_mappings.insert("Option".to_string(), SpecialTypeInfo {
            name: "Option".to_string(),
            semantic_type: SemanticType::Generic {
                name: "Option".to_string(),
                parameters: vec![SemanticType::Variable("T".to_string())],
            },
            requires_special_handling: true,
        });
        special_mappings.insert("Result".to_string(), SpecialTypeInfo {
            name: "Result".to_string(),
            semantic_type: SemanticType::Generic {
                name: "Result".to_string(),
                parameters: vec![
                    SemanticType::Variable("T".to_string()),
                    SemanticType::Variable("E".to_string()),
                ],
            },
            requires_special_handling: true,
        });

        Self {
            primitive_mappings,
            collection_mappings,
            special_mappings,
        }
    }
}

// Default implementations

impl Default for TypeResolutionConfig {
    fn default() -> Self {
        Self {
            enable_detailed_errors: true,
            enable_alias_resolution: true,
            enable_generic_resolution: true,
            enable_semantic_enrichment: true,
            max_resolution_depth: 100,
            enable_caching: true,
        }
    }
}

impl Default for ResolutionStatistics {
    fn default() -> Self {
        Self {
            total_resolutions: 0,
            successful_resolutions: 0,
            failed_resolutions: 0,
            cache_hits: 0,
            cache_misses: 0,
            avg_resolution_time: std::time::Duration::ZERO,
        }
    }
}

impl Default for ASTTypeResolver {
    fn default() -> Self {
        Self::new()
    }
}