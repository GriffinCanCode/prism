//! Symbol Resolution Engine - Multi-Phase Symbol Resolution
//!
//! This module embodies the single concept of "Symbol Resolution".
//! Following Prism's Conceptual Cohesion principle, this file is responsible
//! for ONE thing: resolving symbol references to their definitions through
//! a sophisticated multi-phase algorithm.
//!
//! **Conceptual Responsibility**: Symbol reference resolution
//! **What it does**: symbol resolution, candidate ranking, access validation, resolution caching
//! **What it doesn't do**: symbol storage, scope management, semantic analysis (delegates to specialized modules)

use crate::error::{CompilerError, CompilerResult};
use crate::symbol_table::{SymbolTable, SymbolData, SymbolKind, SymbolVisibility};
use crate::scope::{ScopeTree, ScopeId, ScopeData, ScopeKind};
use crate::cache::CompilationCache;
use crate::semantic::SemanticDatabase;
use prism_common::{NodeId, span::Span, symbol::Symbol};
use prism_effects::effects::EffectRegistry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;
use serde::{Serialize, Deserialize};

/// Multi-phase symbol resolution engine
/// 
/// Implements the sophisticated resolution algorithm from PLT-004,
/// integrating with existing infrastructure while maintaining modularity
#[derive(Debug)]
pub struct SymbolResolver {
    /// Symbol table integration
    symbol_table: Arc<SymbolTable>,
    /// Scope tree integration
    scope_tree: Arc<ScopeTree>,
    /// Semantic database integration
    semantic_db: Arc<SemanticDatabase>,
    /// Effect system integration
    effect_registry: Arc<EffectRegistry>,
    /// Resolution cache integration
    cache: Arc<CompilationCache>,
    /// Resolver configuration
    config: ResolverConfig,
}

/// Configuration for symbol resolver
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Enable lexical scope resolution
    pub enable_lexical_resolution: bool,
    /// Enable module import resolution
    pub enable_import_resolution: bool,
    /// Enable semantic type resolution
    pub enable_semantic_resolution: bool,
    /// Enable effect system resolution
    pub enable_effect_resolution: bool,
    /// Enable resolution caching
    pub enable_caching: bool,
    /// Maximum resolution depth
    pub max_resolution_depth: usize,
    /// Minimum confidence threshold for results
    pub min_confidence_threshold: f64,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            enable_lexical_resolution: true,
            enable_import_resolution: true,
            enable_semantic_resolution: true,
            enable_effect_resolution: true,
            enable_caching: true,
            max_resolution_depth: 50,
            min_confidence_threshold: 0.5,
        }
    }
}

/// Resolution context providing context for symbol resolution
#[derive(Debug, Clone)]
pub struct ResolutionContext {
    /// Current scope being analyzed
    pub current_scope: Option<ScopeId>,
    /// Current module context
    pub current_module: Option<String>,
    /// Available capabilities in current context
    pub available_capabilities: Vec<String>,
    /// Current effect context
    pub effect_context: Option<String>,
    /// Syntax style being used
    pub syntax_style: String,
    /// Resolution preferences
    pub preferences: ResolutionPreferences,
}

impl Default for ResolutionContext {
    fn default() -> Self {
        Self {
            current_scope: None,
            current_module: None,
            available_capabilities: Vec::new(),
            effect_context: None,
            syntax_style: "canonical".to_string(),
            preferences: ResolutionPreferences::default(),
        }
    }
}

/// Resolution preferences for customizing behavior
#[derive(Debug, Clone)]
pub struct ResolutionPreferences {
    /// Prefer semantic matches over lexical matches
    pub prefer_semantic: bool,
    /// Strict capability checking
    pub strict_capabilities: bool,
    /// Enable AI metadata generation during resolution
    pub enable_ai_metadata: bool,
    /// Maximum resolution candidates to consider
    pub max_candidates: usize,
}

impl Default for ResolutionPreferences {
    fn default() -> Self {
        Self {
            prefer_semantic: false,
            strict_capabilities: true,
            enable_ai_metadata: true,
            max_candidates: 10,
        }
    }
}

/// Resolution candidate with metadata
#[derive(Debug, Clone)]
pub struct ResolutionCandidate {
    /// The candidate symbol
    pub symbol: Symbol,
    /// Symbol data
    pub symbol_data: SymbolData,
    /// How this candidate was found
    pub resolution_path: ResolutionPath,
    /// Confidence in this candidate (0.0 to 1.0)
    pub confidence: f64,
    /// Resolution kind
    pub resolution_kind: ResolutionKind,
}

/// Resolution path tracking how symbol was found
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionPath {
    /// Direct lexical scope resolution
    Lexical { 
        scope_chain: Vec<ScopeId>,
        depth: usize,
    },
    /// Import resolution
    Import { 
        import_scope: ScopeId,
        source_module: String,
        import_alias: Option<String>,
    },
    /// Module export resolution
    ModuleExport { 
        module_scope: ScopeId,
        export_name: Option<String>,
    },
    /// Section-based resolution (PLD-002 integration)
    Section { 
        section_scope: ScopeId,
        section_type: String,
    },
    /// Semantic type resolution (PLD-001 integration)
    Semantic { 
        semantic_type: String,
        constraints: Vec<String>,
        business_rules: Vec<String>,
    },
    /// Effect system resolution (PLD-003 integration)
    Effect { 
        effects: Vec<String>,
        capabilities: Vec<String>,
        security_policy: Option<String>,
    },
}

/// Resolution kind classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionKind {
    Direct,        // Direct symbol in scope
    Import,        // Through import statement
    ModuleExport,  // Through module export
    Section,       // Through module section
    Semantic,      // Through semantic type system
    Effect,        // Through effect system
    Fallback,      // Fallback resolution
}

/// Comprehensive resolution result with AI metadata
#[derive(Debug, Clone)]
pub struct ResolvedSymbol {
    /// The resolved symbol
    pub symbol: Symbol,
    /// Symbol data with full context
    pub symbol_data: SymbolData,
    /// How the symbol was resolved
    pub resolution_path: ResolutionPath,
    /// Confidence in resolution (0.0 to 1.0)
    pub confidence: f64,
    /// Resolution kind
    pub resolution_kind: ResolutionKind,
    /// Access permissions validated
    pub access_validated: bool,
    /// Effect requirements checked
    pub effects_validated: bool,
    /// AI-readable resolution metadata
    pub ai_metadata: Option<String>,
    /// Resolution timestamp for caching
    pub resolved_at: SystemTime,
}

impl SymbolResolver {
    /// Create a new symbol resolver with all integrations
    pub fn new(
        symbol_table: Arc<SymbolTable>,
        scope_tree: Arc<ScopeTree>,
        semantic_db: Arc<SemanticDatabase>,
        effect_registry: Arc<EffectRegistry>,
        cache: Arc<CompilationCache>,
    ) -> CompilerResult<Self> {
        Ok(Self {
            symbol_table,
            scope_tree,
            semantic_db,
            effect_registry,
            cache,
            config: ResolverConfig::default(),
        })
    }

    /// Main entry point: Resolve a symbol reference with full semantic context
    /// 
    /// Implements the 8-phase resolution algorithm from PLT-004
    pub async fn resolve_symbol(
        &self,
        name: &str,
        context: &ResolutionContext,
    ) -> CompilerResult<ResolvedSymbol> {
        // Phase 1: Check resolution cache
        if self.config.enable_caching {
            if let Some(cached) = self.check_resolution_cache(name, context).await? {
                return Ok(cached);
            }
        }

        // Phase 2: Collect resolution candidates from all sources
        let mut all_candidates = Vec::new();

        // Phase 2a: Lexical scope resolution
        if self.config.enable_lexical_resolution {
            let lexical_candidates = self.resolve_lexical_scope(name, context)?;
            all_candidates.extend(lexical_candidates);
        }

        // Phase 2b: Module and import resolution
        if self.config.enable_import_resolution {
            let import_candidates = self.resolve_imports(name, context)?;
            all_candidates.extend(import_candidates);
        }

        // Phase 2c: Semantic type resolution (integrates with semantic database)
        if self.config.enable_semantic_resolution {
            let semantic_candidates = self.resolve_semantic_types(name, context)?;
            all_candidates.extend(semantic_candidates);
        }

        // Phase 2d: Effect and capability resolution (integrates with effect system)
        if self.config.enable_effect_resolution {
            let effect_candidates = self.resolve_effects_and_capabilities(name, context)?;
            all_candidates.extend(effect_candidates);
        }

        // Phase 3: Filter and rank candidates
        let valid_candidates = self.filter_and_rank_candidates(all_candidates, context)?;

        // Phase 4: Apply resolution rules and select best match
        let resolved = self.select_best_candidate(valid_candidates, context)?;

        // Phase 5: Validate access permissions and capabilities
        let access_validated = self.validate_access_permissions(&resolved, context)?;
        let effects_validated = self.validate_effect_requirements(&resolved, context)?;

        // Phase 6: Generate AI metadata for resolution
        let ai_metadata = if context.preferences.enable_ai_metadata {
            Some(self.generate_ai_metadata(&resolved, context)?)
        } else {
            None
        };

        // Phase 7: Create final result
        let final_result = ResolvedSymbol {
            symbol: resolved.symbol,
            symbol_data: resolved.symbol_data,
            resolution_path: resolved.resolution_path,
            confidence: resolved.confidence,
            resolution_kind: resolved.resolution_kind,
            access_validated,
            effects_validated,
            ai_metadata,
            resolved_at: SystemTime::now(),
        };

        // Phase 8: Cache result
        if self.config.enable_caching {
            self.cache_resolution_result(name, context, &final_result).await?;
        }

        Ok(final_result)
    }

    /// Phase 2a: Lexical scope resolution following Rust-like semantics
    fn resolve_lexical_scope(
        &self,
        name: &str,
        context: &ResolutionContext,
    ) -> CompilerResult<Vec<ResolutionCandidate>> {
        let mut candidates = Vec::new();
        
        let Some(current_scope) = context.current_scope else {
            return Ok(candidates);
        };

        // Walk up the scope chain
        let scope_chain = self.scope_tree.get_scope_chain(current_scope);
        
        for (depth, &scope_id) in scope_chain.iter().enumerate() {
            if depth > self.config.max_resolution_depth {
                break;
            }

            let Some(scope_data) = self.scope_tree.get_scope(scope_id) else {
                continue;
            };

            // Check for direct symbols in this scope
            for symbol in &scope_data.symbols {
                if let Some(symbol_data) = self.symbol_table.get_symbol(symbol) {
                    if symbol_data.name == name {
                        candidates.push(ResolutionCandidate {
                            symbol: *symbol,
                            symbol_data,
                            resolution_path: ResolutionPath::Lexical { 
                                scope_chain: scope_chain.clone(),
                                depth,
                            },
                            confidence: 1.0 - (depth as f64 * 0.1), // Closer scopes have higher confidence
                            resolution_kind: ResolutionKind::Direct,
                        });
                    }
                }
            }

            // Check imports in this scope
            if let Some(&imported_symbol) = scope_data.imports.get(name) {
                if let Some(symbol_data) = self.symbol_table.get_symbol(&imported_symbol) {
                    candidates.push(ResolutionCandidate {
                        symbol: imported_symbol,
                        symbol_data,
                        resolution_path: ResolutionPath::Import { 
                            import_scope: scope_id,
                            source_module: "unknown".to_string(), // Would need to track this
                            import_alias: None,
                        },
                        confidence: 0.9 - (depth as f64 * 0.1),
                        resolution_kind: ResolutionKind::Import,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Phase 2b: Module and import resolution with TypeScript-like semantics
    fn resolve_imports(
        &self,
        name: &str,
        _context: &ResolutionContext,
    ) -> CompilerResult<Vec<ResolutionCandidate>> {
        let mut candidates = Vec::new();

        // Find module scopes and check their exports
        let module_scopes = self.scope_tree.get_scopes_by_kind(|kind| matches!(kind, ScopeKind::Module { .. }));

        for scope_data in module_scopes {
            // Check exports
            if let Some(&exported_symbol) = scope_data.exports.get(name) {
                if let Some(symbol_data) = self.symbol_table.get_symbol(&exported_symbol) {
                    candidates.push(ResolutionCandidate {
                        symbol: exported_symbol,
                        symbol_data,
                        resolution_path: ResolutionPath::ModuleExport { 
                            module_scope: scope_data.id,
                            export_name: None,
                        },
                        confidence: 0.8,
                        resolution_kind: ResolutionKind::ModuleExport,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Phase 2c: Semantic type resolution (integrates with semantic database)
    fn resolve_semantic_types(
        &self,
        name: &str,
        _context: &ResolutionContext,
    ) -> CompilerResult<Vec<ResolutionCandidate>> {
        let mut candidates = Vec::new();

        // Query semantic database for matching types
        if let Some(symbol_info) = self.semantic_db.get_symbol(&Symbol::intern(name)) {
            // Find the symbol in our symbol table
            if let Some(symbol_data) = self.symbol_table.get_symbol(&symbol_info.id) {
                candidates.push(ResolutionCandidate {
                    symbol: symbol_info.id,
                    symbol_data,
                    resolution_path: ResolutionPath::Semantic { 
                        semantic_type: "unknown".to_string(),
                        constraints: Vec::new(),
                        business_rules: Vec::new(),
                    },
                    confidence: 0.85,
                    resolution_kind: ResolutionKind::Semantic,
                });
            }
        }

        Ok(candidates)
    }

    /// Phase 2d: Effect and capability resolution (integrates with effect system)
    fn resolve_effects_and_capabilities(
        &self,
        name: &str,
        context: &ResolutionContext,
    ) -> CompilerResult<Vec<ResolutionCandidate>> {
        let mut candidates = Vec::new();

        // Query effect registry for matching effects
        if let Some(_effect_def) = self.effect_registry.get_effect(name) {
            // Find symbols that use this effect
            let symbols_with_effect = self.symbol_table.find_symbols(|symbol_data| {
                symbol_data.effects.iter().any(|effect| effect.name == name)
            });

            for symbol_data in symbols_with_effect {
                // Validate capability requirements
                if self.validate_capability_requirements(&symbol_data, context) {
                    candidates.push(ResolutionCandidate {
                        symbol: symbol_data.symbol,
                        symbol_data,
                        resolution_path: ResolutionPath::Effect { 
                            effects: vec![name.to_string()],
                            capabilities: context.available_capabilities.clone(),
                            security_policy: None,
                        },
                        confidence: 0.75,
                        resolution_kind: ResolutionKind::Effect,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Phase 3: Filter and rank candidates
    fn filter_and_rank_candidates(
        &self,
        mut candidates: Vec<ResolutionCandidate>,
        context: &ResolutionContext,
    ) -> CompilerResult<Vec<ResolutionCandidate>> {
        // Filter by minimum confidence
        candidates.retain(|candidate| candidate.confidence >= self.config.min_confidence_threshold);

        // Filter by access permissions
        candidates.retain(|candidate| self.is_symbol_accessible(&candidate.symbol_data, context));

        // Sort by confidence (highest first)
        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to max candidates
        candidates.truncate(context.preferences.max_candidates);

        Ok(candidates)
    }

    /// Phase 4: Select best candidate based on resolution rules
    fn select_best_candidate(
        &self,
        candidates: Vec<ResolutionCandidate>,
        context: &ResolutionContext,
    ) -> CompilerResult<ResolutionCandidate> {
        if candidates.is_empty() {
            return Err(CompilerError::SymbolNotFound { 
                symbol: "symbol not found in any scope".to_string()
            });
        }

        // Apply resolution preferences
        let best_candidate = if context.preferences.prefer_semantic {
            candidates.into_iter()
                .find(|c| c.resolution_kind == ResolutionKind::Semantic)
                .or_else(|| candidates.into_iter().next())
        } else {
            candidates.into_iter().next()
        };

        best_candidate.ok_or_else(|| CompilerError::SymbolNotFound { 
            symbol: "no suitable candidate found".to_string()
        })
    }

    /// Helper: Check if symbol is accessible in current context
    fn is_symbol_accessible(&self, symbol_data: &SymbolData, context: &ResolutionContext) -> bool {
        match symbol_data.visibility {
            SymbolVisibility::Public => true,
            SymbolVisibility::Module => {
                // Check if we're in the same module
                context.current_module.as_ref()
                    .map(|_current| {
                        // Would need to determine symbol's module
                        true // Simplified for now
                    })
                    .unwrap_or(false)
            }
            SymbolVisibility::Private => false, // Only accessible in same scope
            SymbolVisibility::Internal => true, // Internal to crate
        }
    }

    /// Helper: Validate capability requirements
    fn validate_capability_requirements(&self, symbol_data: &SymbolData, context: &ResolutionContext) -> bool {
        // Check if all required capabilities are available
        symbol_data.required_capabilities.iter()
            .all(|required| context.available_capabilities.contains(required))
    }

    /// Phase 5: Validate access permissions
    fn validate_access_permissions(&self, resolved: &ResolutionCandidate, context: &ResolutionContext) -> CompilerResult<bool> {
        Ok(self.is_symbol_accessible(&resolved.symbol_data, context))
    }

    /// Phase 5: Validate effect requirements
    fn validate_effect_requirements(&self, resolved: &ResolutionCandidate, context: &ResolutionContext) -> CompilerResult<bool> {
        Ok(self.validate_capability_requirements(&resolved.symbol_data, context))
    }

    /// Phase 6: Generate AI metadata for resolution
    fn generate_ai_metadata(&self, resolved: &ResolutionCandidate, context: &ResolutionContext) -> CompilerResult<String> {
        let metadata = format!(
            "Resolved '{}' via {} with confidence {:.2} in {} context",
            resolved.symbol_data.name,
            match resolved.resolution_kind {
                ResolutionKind::Direct => "direct scope lookup",
                ResolutionKind::Import => "import resolution",
                ResolutionKind::ModuleExport => "module export",
                ResolutionKind::Section => "section lookup",
                ResolutionKind::Semantic => "semantic type matching",
                ResolutionKind::Effect => "effect system",
                ResolutionKind::Fallback => "fallback resolution",
            },
            resolved.confidence,
            context.syntax_style
        );
        
        Ok(metadata)
    }

    /// Check resolution cache
    async fn check_resolution_cache(&self, _name: &str, _context: &ResolutionContext) -> CompilerResult<Option<ResolvedSymbol>> {
        // Would implement cache lookup based on name and context hash
        Ok(None) // Simplified for now
    }

    /// Cache resolution result
    async fn cache_resolution_result(&self, _name: &str, _context: &ResolutionContext, _result: &ResolvedSymbol) -> CompilerResult<()> {
        // Would implement cache storage
        Ok(()) // Simplified for now
    }

    /// Get resolver configuration
    pub fn config(&self) -> &ResolverConfig {
        &self.config
    }
}

/// Resolution statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolverStats {
    /// Total number of resolutions performed
    pub total_resolutions: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average resolution time in milliseconds
    pub average_resolution_time_ms: f64,
    /// Resolutions by kind
    pub resolution_by_kind: HashMap<String, u64>,
} 