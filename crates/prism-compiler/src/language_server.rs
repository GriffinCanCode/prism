//! Built-in Language Server Protocol (LSP) implementation
//!
//! This module provides a comprehensive language server with AI-aware IDE features,
//! including intelligent code completion, semantic analysis, and real-time diagnostics.

use crate::context::CompilationContext;
use crate::error::{CompilerError, CompilerResult};
use crate::query::QueryEngine;
use crate::semantic::{AIMetadata, SemanticDatabase};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// LSP message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum LSPRequest {
    Initialize {
        params: InitializeParams,
    },
    TextDocumentDidOpen {
        params: DidOpenTextDocumentParams,
    },
    TextDocumentDidChange {
        params: DidChangeTextDocumentParams,
    },
    TextDocumentCompletion {
        params: CompletionParams,
    },
    TextDocumentHover {
        params: HoverParams,
    },
    TextDocumentDefinition {
        params: DefinitionParams,
    },
    TextDocumentReferences {
        params: ReferenceParams,
    },
    TextDocumentDiagnostics {
        params: DiagnosticParams,
    },
    WorkspaceSymbol {
        params: WorkspaceSymbolParams,
    },
}

/// LSP response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LSPResponse {
    Initialize(InitializeResult),
    Completion(CompletionList),
    Hover(Option<Hover>),
    Definition(Vec<Location>),
    References(Vec<Location>),
    Diagnostics(Vec<Diagnostic>),
    WorkspaceSymbols(Vec<SymbolInformation>),
    Error(LSPError),
}

/// LSP initialization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    pub process_id: Option<u32>,
    pub root_uri: Option<String>,
    pub capabilities: ClientCapabilities,
    pub initialization_options: Option<serde_json::Value>,
}

/// Client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    pub text_document: Option<TextDocumentClientCapabilities>,
    pub workspace: Option<WorkspaceClientCapabilities>,
    pub experimental: Option<serde_json::Value>,
}

/// Text document client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentClientCapabilities {
    pub completion: Option<CompletionClientCapabilities>,
    pub hover: Option<HoverClientCapabilities>,
    pub definition: Option<DefinitionClientCapabilities>,
    pub references: Option<ReferenceClientCapabilities>,
    pub diagnostics: Option<DiagnosticClientCapabilities>,
}

/// Workspace client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceClientCapabilities {
    pub symbol: Option<SymbolClientCapabilities>,
    pub workspace_folders: Option<bool>,
    pub configuration: Option<bool>,
}

/// Completion client capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionClientCapabilities {
    pub completion_item: Option<CompletionItemCapabilities>,
    pub context_support: Option<bool>,
}

/// Completion item capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItemCapabilities {
    pub snippet_support: Option<bool>,
    pub documentation_format: Option<Vec<MarkupKind>>,
}

/// Markup kind for documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkupKind {
    PlainText,
    Markdown,
}

/// Other capability structs (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverClientCapabilities {
    pub content_format: Option<Vec<MarkupKind>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefinitionClientCapabilities {
    pub link_support: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceClientCapabilities {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticClientCapabilities {
    pub related_information: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolClientCapabilities {
    pub symbol_kind: Option<SymbolKindCapabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolKindCapabilities {
    pub value_set: Option<Vec<SymbolKind>>,
}

/// Initialize result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    pub capabilities: ServerCapabilities,
    pub server_info: Option<ServerInfo>,
}

/// Server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub text_document_sync: Option<TextDocumentSyncOptions>,
    pub completion_provider: Option<CompletionOptions>,
    pub hover_provider: Option<bool>,
    pub definition_provider: Option<bool>,
    pub references_provider: Option<bool>,
    pub diagnostic_provider: Option<DiagnosticOptions>,
    pub workspace_symbol_provider: Option<bool>,
}

/// Text document sync options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentSyncOptions {
    pub open_close: Option<bool>,
    pub change: Option<TextDocumentSyncKind>,
    pub save: Option<SaveOptions>,
}

/// Text document sync kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextDocumentSyncKind {
    None = 0,
    Full = 1,
    Incremental = 2,
}

/// Save options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveOptions {
    pub include_text: Option<bool>,
}

/// Completion options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionOptions {
    pub resolve_provider: Option<bool>,
    pub trigger_characters: Option<Vec<String>>,
}

/// Diagnostic options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticOptions {
    pub inter_file_dependencies: Option<bool>,
    pub workspace_diagnostics: Option<bool>,
}

/// Server info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: Option<String>,
}

/// Document parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidOpenTextDocumentParams {
    pub text_document: TextDocumentItem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DidChangeTextDocumentParams {
    pub text_document: VersionedTextDocumentIdentifier,
    pub content_changes: Vec<TextDocumentContentChangeEvent>,
}

/// Text document item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentItem {
    pub uri: String,
    pub language_id: String,
    pub version: i32,
    pub text: String,
}

/// Versioned text document identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedTextDocumentIdentifier {
    pub uri: String,
    pub version: i32,
}

/// Text document content change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentContentChangeEvent {
    pub range: Option<Range>,
    pub range_length: Option<u32>,
    pub text: String,
}

/// Position in document
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// Range in document
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// Location in workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub uri: String,
    pub range: Range,
}

/// Completion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
    pub context: Option<CompletionContext>,
}

/// Text document identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentIdentifier {
    pub uri: String,
}

/// Completion context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionContext {
    pub trigger_kind: CompletionTriggerKind,
    pub trigger_character: Option<String>,
}

/// Completion trigger kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionTriggerKind {
    Invoked = 1,
    TriggerCharacter = 2,
    TriggerForIncompleteCompletions = 3,
}

/// Completion list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionList {
    pub is_incomplete: bool,
    pub items: Vec<CompletionItem>,
}

/// Completion item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: Option<CompletionItemKind>,
    pub detail: Option<String>,
    pub documentation: Option<MarkupContent>,
    pub insert_text: Option<String>,
    pub ai_metadata: Option<AIMetadata>,
}

/// Completion item kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionItemKind {
    Text = 1,
    Method = 2,
    Function = 3,
    Constructor = 4,
    Field = 5,
    Variable = 6,
    Class = 7,
    Interface = 8,
    Module = 9,
    Property = 10,
    Unit = 11,
    Value = 12,
    Enum = 13,
    Keyword = 14,
    Snippet = 15,
    Color = 16,
    File = 17,
    Reference = 18,
}

/// Markup content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkupContent {
    pub kind: MarkupKind,
    pub value: String,
}

/// Hover parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
}

/// Hover information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hover {
    pub contents: MarkupContent,
    pub range: Option<Range>,
}

/// Definition parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefinitionParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
}

/// Reference parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
    pub context: ReferenceContext,
}

/// Reference context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceContext {
    pub include_declaration: bool,
}

/// Diagnostic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticParams {
    pub text_document: TextDocumentIdentifier,
}

/// Diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: Option<DiagnosticSeverity>,
    pub code: Option<String>,
    pub source: Option<String>,
    pub message: String,
    pub related_information: Option<Vec<DiagnosticRelatedInformation>>,
    pub ai_suggestion: Option<String>,
}

/// Diagnostic severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

/// Diagnostic related information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticRelatedInformation {
    pub location: Location,
    pub message: String,
}

/// Workspace symbol parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSymbolParams {
    pub query: String,
}

/// Symbol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInformation {
    pub name: String,
    pub kind: SymbolKind,
    pub location: Location,
    pub container_name: Option<String>,
}

/// Symbol kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolKind {
    File = 1,
    Module = 2,
    Namespace = 3,
    Package = 4,
    Class = 5,
    Method = 6,
    Property = 7,
    Field = 8,
    Constructor = 9,
    Enum = 10,
    Interface = 11,
    Function = 12,
    Variable = 13,
    Constant = 14,
    String = 15,
    Number = 16,
    Boolean = 17,
    Array = 18,
    Object = 19,
    Key = 20,
    Null = 21,
    EnumMember = 22,
    Struct = 23,
    Event = 24,
    Operator = 25,
    TypeParameter = 26,
}

/// LSP error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSPError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Language server trait
#[async_trait]
pub trait LanguageServer: Send + Sync {
    /// Handle LSP request
    async fn handle_request(&self, request: LSPRequest) -> LSPResponse;

    /// Initialize the language server
    async fn initialize(&self, params: InitializeParams) -> InitializeResult;

    /// Handle document open
    async fn did_open(&self, params: DidOpenTextDocumentParams) -> CompilerResult<()>;

    /// Handle document change
    async fn did_change(&self, params: DidChangeTextDocumentParams) -> CompilerResult<()>;

    /// Provide code completion
    async fn completion(&self, params: CompletionParams) -> CompilerResult<CompletionList>;

    /// Provide hover information
    async fn hover(&self, params: HoverParams) -> CompilerResult<Option<Hover>>;

    /// Find definition
    async fn definition(&self, params: DefinitionParams) -> CompilerResult<Vec<Location>>;

    /// Find references
    async fn references(&self, params: ReferenceParams) -> CompilerResult<Vec<Location>>;

    /// Get diagnostics
    async fn diagnostics(&self, params: DiagnosticParams) -> CompilerResult<Vec<Diagnostic>>;

    /// Find workspace symbols
    async fn workspace_symbols(&self, params: WorkspaceSymbolParams) -> CompilerResult<Vec<SymbolInformation>>;
}

/// Prism language server implementation
pub struct PrismLanguageServer {
    /// Compilation context
    context: Arc<CompilationContext>,
    /// Query engine for semantic queries
    query_engine: Arc<QueryEngine>,
    /// Open documents
    documents: Arc<RwLock<HashMap<String, TextDocumentItem>>>,
    /// Client capabilities
    client_capabilities: Arc<Mutex<Option<ClientCapabilities>>>,
}

impl PrismLanguageServer {
    /// Create a new Prism language server
    pub fn new(context: Arc<CompilationContext>, query_engine: Arc<QueryEngine>) -> Self {
        Self {
            context,
            query_engine,
            documents: Arc::new(RwLock::new(HashMap::new())),
            client_capabilities: Arc::new(Mutex::new(None)),
        }
    }

    /// Get semantic information at position
    async fn get_semantic_info(&self, uri: &str, position: Position) -> CompilerResult<Option<AIMetadata>> {
        // This would query the semantic database for information at the given position
        // For now, return a placeholder
        Ok(Some(AIMetadata {
            description: "Semantic information would be provided here".to_string(),
            tags: vec!["semantic".to_string()],
            complexity_score: 0.5,
            maintainability_score: 0.8,
            performance_hints: vec!["Consider optimizing this code".to_string()],
            semantic_context: HashMap::new(),
        }))
    }

    /// Generate AI-powered completion suggestions
    async fn generate_ai_completions(&self, uri: &str, position: Position) -> CompilerResult<Vec<CompletionItem>> {
        // This would use AI to generate intelligent completion suggestions
        // For now, return some basic completions
        Ok(vec![
            CompletionItem {
                label: "function".to_string(),
                kind: Some(CompletionItemKind::Keyword),
                detail: Some("Function declaration".to_string()),
                documentation: Some(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Declares a new function".to_string(),
                }),
                insert_text: Some("function ${1:name}(${2:params}) {\n\t$0\n}".to_string()),
                ai_metadata: Some(AIMetadata {
                    description: "AI-suggested function template".to_string(),
                    tags: vec!["ai-suggestion".to_string(), "template".to_string()],
                    complexity_score: 0.3,
                    maintainability_score: 0.9,
                    performance_hints: vec![],
                    semantic_context: HashMap::new(),
                }),
            },
            CompletionItem {
                label: "if".to_string(),
                kind: Some(CompletionItemKind::Keyword),
                detail: Some("Conditional statement".to_string()),
                documentation: Some(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: "Conditional execution".to_string(),
                }),
                insert_text: Some("if (${1:condition}) {\n\t$0\n}".to_string()),
                ai_metadata: None,
            },
        ])
    }
}

#[async_trait]
impl LanguageServer for PrismLanguageServer {
    async fn handle_request(&self, request: LSPRequest) -> LSPResponse {
        match request {
            LSPRequest::Initialize { params } => {
                LSPResponse::Initialize(self.initialize(params).await)
            }
            LSPRequest::TextDocumentDidOpen { params } => {
                match self.did_open(params).await {
                    Ok(_) => LSPResponse::Initialize(InitializeResult {
                        capabilities: ServerCapabilities {
                            text_document_sync: None,
                            completion_provider: None,
                            hover_provider: None,
                            definition_provider: None,
                            references_provider: None,
                            diagnostic_provider: None,
                            workspace_symbol_provider: None,
                        },
                        server_info: None,
                    }),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::TextDocumentDidChange { params } => {
                match self.did_change(params).await {
                    Ok(_) => LSPResponse::Initialize(InitializeResult {
                        capabilities: ServerCapabilities {
                            text_document_sync: None,
                            completion_provider: None,
                            hover_provider: None,
                            definition_provider: None,
                            references_provider: None,
                            diagnostic_provider: None,
                            workspace_symbol_provider: None,
                        },
                        server_info: None,
                    }),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::TextDocumentCompletion { params } => {
                match self.completion(params).await {
                    Ok(result) => LSPResponse::Completion(result),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::TextDocumentHover { params } => {
                match self.hover(params).await {
                    Ok(result) => LSPResponse::Hover(result),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::TextDocumentDefinition { params } => {
                match self.definition(params).await {
                    Ok(result) => LSPResponse::Definition(result),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::TextDocumentReferences { params } => {
                match self.references(params).await {
                    Ok(result) => LSPResponse::References(result),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::TextDocumentDiagnostics { params } => {
                match self.diagnostics(params).await {
                    Ok(result) => LSPResponse::Diagnostics(result),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
            LSPRequest::WorkspaceSymbol { params } => {
                match self.workspace_symbols(params).await {
                    Ok(result) => LSPResponse::WorkspaceSymbols(result),
                    Err(e) => LSPResponse::Error(LSPError {
                        code: -32603,
                        message: e.to_string(),
                        data: None,
                    }),
                }
            }
        }
    }

    async fn initialize(&self, params: InitializeParams) -> InitializeResult {
        // Store client capabilities
        {
            let mut capabilities = self.client_capabilities.lock().await;
            *capabilities = Some(params.capabilities);
        }

        info!("Language server initialized");

        InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncOptions {
                    open_close: Some(true),
                    change: Some(TextDocumentSyncKind::Incremental),
                    save: Some(SaveOptions {
                        include_text: Some(true),
                    }),
                }),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(true),
                    trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                }),
                hover_provider: Some(true),
                definition_provider: Some(true),
                references_provider: Some(true),
                diagnostic_provider: Some(DiagnosticOptions {
                    inter_file_dependencies: Some(true),
                    workspace_diagnostics: Some(true),
                }),
                workspace_symbol_provider: Some(true),
            },
            server_info: Some(ServerInfo {
                name: "Prism Language Server".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) -> CompilerResult<()> {
        let mut documents = self.documents.write().await;
        documents.insert(params.text_document.uri.clone(), params.text_document);
        
        debug!("Document opened: {}", params.text_document.uri);
        Ok(())
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) -> CompilerResult<()> {
        let mut documents = self.documents.write().await;
        
        if let Some(doc) = documents.get_mut(&params.text_document.uri) {
            // Apply changes (simplified - would handle incremental changes)
            if let Some(change) = params.content_changes.first() {
                doc.text = change.text.clone();
                doc.version = params.text_document.version;
            }
        }
        
        debug!("Document changed: {}", params.text_document.uri);
        Ok(())
    }

    async fn completion(&self, params: CompletionParams) -> CompilerResult<CompletionList> {
        let ai_completions = self.generate_ai_completions(&params.text_document.uri, params.position).await?;
        
        Ok(CompletionList {
            is_incomplete: false,
            items: ai_completions,
        })
    }

    async fn hover(&self, params: HoverParams) -> CompilerResult<Option<Hover>> {
        if let Some(semantic_info) = self.get_semantic_info(&params.text_document.uri, params.position).await? {
            Ok(Some(Hover {
                contents: MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: format!("**{}**\n\n{}", "Symbol Information", semantic_info.description),
                },
                range: None,
            }))
        } else {
            Ok(None)
        }
    }

    async fn definition(&self, params: DefinitionParams) -> CompilerResult<Vec<Location>> {
        // Would query semantic database for definition location
        Ok(vec![])
    }

    async fn references(&self, params: ReferenceParams) -> CompilerResult<Vec<Location>> {
        // Would query semantic database for all references
        Ok(vec![])
    }

    async fn diagnostics(&self, params: DiagnosticParams) -> CompilerResult<Vec<Diagnostic>> {
        // Would run compilation and return diagnostics
        Ok(vec![
            Diagnostic {
                range: Range {
                    start: Position { line: 0, character: 0 },
                    end: Position { line: 0, character: 10 },
                },
                severity: Some(DiagnosticSeverity::Warning),
                code: Some("W001".to_string()),
                source: Some("prism-compiler".to_string()),
                message: "Example diagnostic message".to_string(),
                related_information: None,
                ai_suggestion: Some("Consider using a more descriptive variable name".to_string()),
            }
        ])
    }

    async fn workspace_symbols(&self, params: WorkspaceSymbolParams) -> CompilerResult<Vec<SymbolInformation>> {
        // Would search workspace for symbols matching the query
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::QueryEngine;
    use crate::context::CompilationContext;

    #[tokio::test]
    async fn test_language_server_initialization() {
        let context = Arc::new(CompilationContext::new());
        let query_engine = Arc::new(QueryEngine::new());
        let server = PrismLanguageServer::new(context, query_engine);

        let params = InitializeParams {
            process_id: Some(1234),
            root_uri: Some("file:///test".to_string()),
            capabilities: ClientCapabilities {
                text_document: None,
                workspace: None,
                experimental: None,
            },
            initialization_options: None,
        };

        let result = server.initialize(params).await;
        assert_eq!(result.server_info.unwrap().name, "Prism Language Server");
    }

    #[tokio::test]
    async fn test_completion() {
        let context = Arc::new(CompilationContext::new());
        let query_engine = Arc::new(QueryEngine::new());
        let server = PrismLanguageServer::new(context, query_engine);

        let params = CompletionParams {
            text_document: TextDocumentIdentifier {
                uri: "file:///test.prism".to_string(),
            },
            position: Position { line: 0, character: 0 },
            context: None,
        };

        let result = server.completion(params).await.unwrap();
        assert!(!result.items.is_empty());
    }
} 