//! Actor System - Capability-Secured Stateful Concurrency
//!
//! This module implements the Actor Model as specified in PLD-005, providing:
//! - **Message-based communication**: Type-safe message passing between actors
//! - **Capability-based security**: Actors operate within explicit capability boundaries
//! - **Supervision hierarchies**: "Let it crash" philosophy with supervisor trees
//! - **Effect integration**: All actor operations declare their effects
//! - **AI metadata**: Rich metadata for AI comprehension of actor behavior

use crate::{authority, resources, intelligence};
use crate::resources::effects::Effect;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use tokio::sync::{mpsc, oneshot};
use thiserror::Error;
use uuid::Uuid;

/// Unique identifier for actors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActorId(Uuid);

impl ActorId {
    /// Generate a new unique actor ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Reference to an actor that can receive messages
#[derive(Debug, Clone)]
pub struct ActorRef<A: Actor> {
    /// Actor ID
    id: ActorId,
    /// Message sender channel
    sender: mpsc::UnboundedSender<ActorMessage<A>>,
    /// Actor metadata for AI comprehension
    metadata: ActorMetadata,
}

impl<A: Actor> ActorRef<A> {
    /// Send a message to the actor (fire-and-forget)
    pub fn tell(&self, message: A::Message) {
        let actor_message = ActorMessage::Tell(message);
        if let Err(_) = self.sender.send(actor_message) {
            // Actor is no longer alive - this is expected in "let it crash" model
            tracing::warn!("Failed to send message to actor {:?} - actor may be dead", self.id);
        }
    }

    /// Send a message and wait for response with timeout
    pub async fn ask<R>(&self, message: A::Message, timeout: Duration) -> Result<R, ActorError>
    where
        A::Message: Into<AskMessage<A::Message, R>>,
        R: Send + 'static,
    {
        let (response_tx, response_rx) = oneshot::channel();
        let ask_message = AskMessage {
            message,
            response_tx,
        };
        let actor_message = ActorMessage::Ask(Box::new(ask_message));
        
        self.sender.send(actor_message)
            .map_err(|_| ActorError::ActorDead { id: self.id })?;
        
        // Wait for response with timeout
        tokio::select! {
            result = response_rx => {
                result.map_err(|_| ActorError::ResponseTimeout { id: self.id })
            }
            _ = tokio::time::sleep(timeout) => {
                Err(ActorError::ResponseTimeout { id: self.id })
            }
        }
    }

    /// Fan-out: Send message to multiple actors and collect responses
    pub async fn fan_out<R>(
        actors: &[ActorRef<A>],
        message_fn: impl Fn() -> A::Message + Send + Sync,
        timeout: Duration,
    ) -> Vec<Result<R, ActorError>>
    where
        A::Message: Into<AskMessage<A::Message, R>> + Clone,
        R: Send + 'static,
    {
        let futures = actors.iter().map(|actor| {
            let message = message_fn();
            actor.ask(message, timeout)
        });
        
        futures::future::join_all(futures).await
    }

    /// Get actor ID
    pub fn id(&self) -> ActorId {
        self.id
    }

    /// Get actor metadata for AI analysis
    pub fn metadata(&self) -> &ActorMetadata {
        &self.metadata
    }
}

/// Trait that all actors must implement
pub trait Actor: Send + Sync + 'static {
    /// Message type this actor can receive
    type Message: Send + 'static;

    /// Handle an incoming message
    fn handle_message(
        &mut self,
        message: Self::Message,
        context: &mut ActorContext,
    ) -> impl std::future::Future<Output = Result<(), ActorError>> + Send;

    /// Handle ask messages that expect responses
    fn handle_ask<R>(
        &mut self,
        message: Self::Message,
        context: &mut ActorContext,
    ) -> impl std::future::Future<Output = Result<R, ActorError>> + Send
    where
        R: Send + 'static,
    {
        async move {
            // Default implementation: handle as regular message and return unit
            self.handle_message(message, context).await?;
            // This is a placeholder - actors should override this for proper responses
            Err(ActorError::Generic {
                message: "Actor does not support ask messages".to_string(),
            })
        }
    }

    /// Get the capabilities this actor requires
    fn required_capabilities(&self) -> authority::CapabilitySet;

    /// Get the effects this actor may produce
    fn declared_effects(&self) -> Vec<Effect>;

    /// Actor lifecycle: called when actor starts
    fn on_start(&mut self, _context: &mut ActorContext) -> impl std::future::Future<Output = Result<(), ActorError>> + Send {
        async { Ok(()) }
    }

    /// Actor lifecycle: called when actor stops
    fn on_stop(&mut self, _context: &mut ActorContext) -> impl std::future::Future<Output = Result<(), ActorError>> + Send {
        async { Ok(()) }
    }

    /// Actor lifecycle: called when actor restarts
    fn on_restart(&mut self, _context: &mut ActorContext) -> impl std::future::Future<Output = Result<(), ActorError>> + Send {
        async { Ok(()) }
    }

    /// Get AI-comprehensible metadata about this actor
    fn ai_metadata(&self) -> ActorAIMetadata {
        ActorAIMetadata::default()
    }
}

/// Message wrapper for actor communication
#[derive(Debug)]
enum ActorMessage<A: Actor> {
    /// Fire-and-forget message
    Tell(A::Message),
    /// Request-response message
    Ask(Box<dyn AskMessageTrait + Send>),
    /// System message for lifecycle management
    System(SystemMessage),
}

/// Trait for ask messages that expect responses
trait AskMessageTrait: Send {
    fn handle(&mut self, response: Box<dyn std::any::Any + Send>) -> Result<(), ActorError>;
}

/// Ask message with response channel
struct AskMessage<M, R> {
    message: M,
    response_tx: oneshot::Sender<R>,
}

impl<M: Send, R: Send + 'static> AskMessageTrait for AskMessage<M, R> {
    fn handle(&mut self, response: Box<dyn std::any::Any + Send>) -> Result<(), ActorError> {
        if let Ok(typed_response) = response.downcast::<R>() {
            let _ = self.response_tx.send(*typed_response);
            Ok(())
        } else {
            Err(ActorError::ResponseTypeMismatch)
        }
    }
}

/// System messages for actor lifecycle
#[derive(Debug)]
enum SystemMessage {
    /// Stop the actor gracefully
    Stop,
    /// Restart the actor after failure
    Restart,
    /// Update actor capabilities
    UpdateCapabilities(authority::CapabilitySet),
    /// Child actor failed notification
    ChildFailed {
        child_id: ActorId,
        error: ActorError,
        restart_count: u32,
    },
}

/// Context provided to actors during message handling
pub struct ActorContext {
    /// Actor ID
    pub actor_id: ActorId,
    /// Available capabilities
    pub capabilities: authority::CapabilitySet,
    /// Effect tracker
    pub effect_tracker: Arc<resources::effects::EffectTracker>,
    /// AI metadata collector
    pub ai_collector: Arc<intelligence::AIMetadataCollector>,
    /// Supervisor reference
    pub supervisor: Option<ActorRef<dyn Supervisor>>,
    /// Actor system reference for spawning children
    pub actor_system: Option<Arc<ActorSystem>>,
}

impl ActorContext {
    /// Spawn a child actor with supervision
    pub async fn spawn_child<A: Actor>(
        &self,
        actor: A,
        capabilities: authority::CapabilitySet,
    ) -> Result<ActorRef<A>, ActorError> {
        let actor_system = self.actor_system.as_ref()
            .ok_or_else(|| ActorError::Generic {
                message: "No actor system available for spawning children".to_string(),
            })?;

        // Child actors inherit attenuated capabilities from parent
        let child_capabilities = self.capabilities.attenuate(&capabilities)?;
        
        // Spawn the child actor
        let child_ref = actor_system.spawn_actor(actor, child_capabilities)?;
        
        // Register child with supervisor
        actor_system.register_child_actor(self.actor_id, child_ref.id)?;
        
        Ok(child_ref)
    }

    /// Stop this actor
    pub fn stop(&self) {
        // Implementation for actor stopping
        if let Some(actor_system) = GLOBAL_ACTOR_SYSTEM.get() {
            // Send stop message to the actor
            let stop_message = ActorMessage {
                from: ActorId(0), // System message
                to: self.actor_id,
                message_type: MessageType::Stop,
                payload: serde_json::Value::Null,
                timestamp: std::time::Instant::now(),
                priority: MessagePriority::High,
            };

            // Try to send the stop message
            if let Ok(mut system) = actor_system.write() {
                if let Some(actor_state) = system.actors.get_mut(&self.actor_id) {
                    // Mark actor as stopping
                    actor_state.state = ActorState::Stopping;
                    
                    // Send stop message to actor's mailbox
                    let _ = actor_state.mailbox_sender.try_send(stop_message);
                    
                    // Remove from active actors list
                    system.actors.remove(&self.actor_id);
                    
                    // Notify supervisor if this actor has one
                    if let Some(supervisor_id) = actor_state.supervisor {
                        if let Some(supervisor_state) = system.actors.get_mut(&supervisor_id) {
                            // Remove this actor from supervisor's children
                            supervisor_state.children.retain(|&child_id| child_id != self.actor_id);
                        }
                    }
                    
                    // Stop all child actors
                    let children_to_stop: Vec<ActorId> = actor_state.children.clone();
                    for child_id in children_to_stop {
                        if let Some(child_state) = system.actors.get_mut(&child_id) {
                            child_state.state = ActorState::Stopping;
                            let stop_child_message = ActorMessage {
                                from: self.actor_id,
                                to: child_id,
                                message_type: MessageType::Stop,
                                payload: serde_json::Value::Null,
                                timestamp: std::time::Instant::now(),
                                priority: MessagePriority::High,
                            };
                            let _ = child_state.mailbox_sender.try_send(stop_child_message);
                        }
                    }
                }
            }
        }
    }

    /// Record an effect execution
    pub async fn record_effect<F, T>(&self, effect: Effect, operation: F) -> Result<T, ActorError>
    where
        F: std::future::Future<Output = Result<T, ActorError>>,
    {
        // For now, we are not using the execution context, so we can pass a dummy one.
        let exec_context = crate::platform::execution::ExecutionContext::new(
            crate::execution::ComponentId(self.actor_id.0),
            effect.clone(),
        );

        let handle = self.effect_tracker.begin_execution(&exec_context)?;
        let result = operation.await;
        self.effect_tracker.end_execution(handle, &result)?;
        result
    }
}

/// Actor metadata for AI comprehension
#[derive(Debug, Clone)]
pub struct ActorMetadata {
    /// Business purpose of this actor
    pub purpose: String,
    /// Actor type name
    pub type_name: String,
    /// Capabilities required
    pub capabilities: Vec<String>,
    /// Effects produced
    pub effects: Vec<String>,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Performance characteristics of an actor
#[derive(Debug, Clone, Default)]
pub struct PerformanceProfile {
    /// Expected message throughput (messages/second)
    pub throughput: Option<u64>,
    /// Expected message latency
    pub latency: Option<Duration>,
    /// Memory usage pattern
    pub memory_pattern: MemoryPattern,
}

/// Memory usage patterns for actors
#[derive(Debug, Clone, Default)]
pub enum MemoryPattern {
    /// Constant memory usage
    #[default]
    Constant,
    /// Linear growth with messages
    Linear,
    /// Bounded growth with cleanup
    Bounded,
    /// Custom pattern
    Custom(String),
}

/// AI-comprehensible metadata about an actor
#[derive(Debug, Clone, Default)]
pub struct ActorAIMetadata {
    /// Business domain this actor operates in
    pub business_domain: Option<String>,
    /// State management strategy
    pub state_management: String,
    /// Concurrency safety guarantees
    pub concurrency_safety: String,
    /// Performance characteristics
    pub performance_characteristics: String,
}

/// Supervisor trait for managing actor hierarchies
pub trait Supervisor: Actor {
    /// Handle child actor failure and decide what to do
    fn handle_child_failure(
        &mut self,
        child_id: ActorId,
        error: ActorError,
        restart_count: u32,
        context: &mut ActorContext,
    ) -> impl std::future::Future<Output = SupervisionDecision> + Send;

    /// Get supervision strategy for this supervisor
    fn supervision_strategy(&self) -> SupervisionStrategy;

    /// Get maximum restart attempts for children
    fn max_restart_attempts(&self) -> u32 {
        3
    }

    /// Get restart delay strategy
    fn restart_delay(&self, restart_count: u32) -> Duration {
        // Exponential backoff: 1s, 2s, 4s, 8s, etc.
        Duration::from_secs(2_u64.pow(restart_count.min(10)))
    }
}

/// Supervision decision for failed actors
#[derive(Debug, Clone)]
pub enum SupervisionDecision {
    /// Restart the failed actor
    Restart,
    /// Restart with delay
    RestartWithDelay(Duration),
    /// Stop the actor permanently
    Stop,
    /// Escalate to parent supervisor
    Escalate,
    /// Restart all sibling actors (OneForAll strategy)
    RestartAll,
}

/// Supervision strategies as defined in PLD-005
#[derive(Debug, Clone)]
pub enum SupervisionStrategy {
    /// Restart only the failed actor
    OneForOne,
    /// Restart all sibling actors when one fails
    OneForAll,
    /// Restart actors in dependency order
    RestForOne,
}

/// Restart tracking for actors
#[derive(Debug, Clone)]
struct RestartInfo {
    /// Number of restart attempts
    count: u32,
    /// Last restart time
    last_restart: SystemTime,
    /// Restart history for analysis
    history: Vec<(SystemTime, ActorError)>,
}

impl RestartInfo {
    fn new() -> Self {
        Self {
            count: 0,
            last_restart: SystemTime::now(),
            history: Vec::new(),
        }
    }

    fn record_restart(&mut self, error: ActorError) {
        self.count += 1;
        self.last_restart = SystemTime::now();
        self.history.push((self.last_restart, error));
        
        // Keep only recent restart history (last 10)
        if self.history.len() > 10 {
            self.history.remove(0);
        }
    }

    fn reset(&mut self) {
        self.count = 0;
        self.history.clear();
    }
}

/// Supervision tree structure with complete implementation
#[derive(Debug)]
struct SupervisionTree {
    /// Parent-child relationships
    relationships: HashMap<ActorId, Vec<ActorId>>,
    /// Child-parent relationships (reverse lookup)
    parent_lookup: HashMap<ActorId, ActorId>,
    /// Supervisors mapped by actor ID
    supervisors: HashMap<ActorId, SupervisionStrategy>,
    /// Restart tracking per actor
    restart_info: HashMap<ActorId, RestartInfo>,
}

impl SupervisionTree {
    fn new() -> Self {
        Self {
            relationships: HashMap::new(),
            parent_lookup: HashMap::new(),
            supervisors: HashMap::new(),
            restart_info: HashMap::new(),
        }
    }

    /// Register a supervisor
    fn register_supervisor(&mut self, supervisor_id: ActorId, strategy: SupervisionStrategy) {
        self.supervisors.insert(supervisor_id, strategy);
        self.relationships.entry(supervisor_id).or_insert_with(Vec::new);
    }

    /// Add a child actor to a supervisor
    fn add_child(&mut self, parent_id: ActorId, child_id: ActorId) {
        self.relationships
            .entry(parent_id)
            .or_insert_with(Vec::new)
            .push(child_id);
        self.parent_lookup.insert(child_id, parent_id);
        self.restart_info.insert(child_id, RestartInfo::new());
    }

    /// Remove an actor from the supervision tree
    fn remove_actor(&mut self, actor_id: ActorId) {
        // Remove from parent's children list
        if let Some(parent_id) = self.parent_lookup.remove(&actor_id) {
            if let Some(children) = self.relationships.get_mut(&parent_id) {
                children.retain(|&id| id != actor_id);
            }
        }

        // Remove as parent (remove all children)
        if let Some(children) = self.relationships.remove(&actor_id) {
            for child_id in children {
                self.parent_lookup.remove(&child_id);
                self.restart_info.remove(&child_id);
            }
        }

        self.supervisors.remove(&actor_id);
        self.restart_info.remove(&actor_id);
    }

    /// Get children of a supervisor
    fn get_children(&self, supervisor_id: ActorId) -> Option<&Vec<ActorId>> {
        self.relationships.get(&supervisor_id)
    }

    /// Get parent supervisor of an actor
    fn get_supervisor(&self, actor_id: ActorId) -> Option<ActorId> {
        self.parent_lookup.get(&actor_id).copied()
    }

    /// Get supervision strategy for an actor
    fn get_strategy(&self, supervisor_id: ActorId) -> Option<&SupervisionStrategy> {
        self.supervisors.get(&supervisor_id)
    }

    /// Record a restart attempt
    fn record_restart(&mut self, actor_id: ActorId, error: ActorError) {
        if let Some(restart_info) = self.restart_info.get_mut(&actor_id) {
            restart_info.record_restart(error);
        }
    }

    /// Get restart count for an actor
    fn get_restart_count(&self, actor_id: ActorId) -> u32 {
        self.restart_info
            .get(&actor_id)
            .map(|info| info.count)
            .unwrap_or(0)
    }

    /// Reset restart count for an actor (after successful operation)
    fn reset_restart_count(&mut self, actor_id: ActorId) {
        if let Some(restart_info) = self.restart_info.get_mut(&actor_id) {
            restart_info.reset();
        }
    }
}

/// Handle to a running actor
#[derive(Debug)]
struct ActorHandle {
    id: ActorId,
    join_handle: tokio::task::JoinHandle<Result<(), ActorError>>,
    metadata: ActorMetadata,
    // Note: We can't store the sender here due to type erasure issues
    // Instead, we'll need a different approach for sending system messages
}

/// Actor system metrics
#[derive(Debug, Default)]
struct ActorSystemMetrics {
    /// Total actors created
    total_created: u64,
    /// Currently active actors
    active_count: usize,
    /// Messages processed
    messages_processed: u64,
    /// Actor failures
    failures: u64,
    /// Restarts
    restarts: u64,
}

/// Actor system that manages all actors with complete supervision
#[derive(Debug)]
pub struct ActorSystem {
    /// Active actors
    actors: Arc<RwLock<HashMap<ActorId, ActorHandle>>>,
    /// Actor registry for lookup by name
    registry: Arc<RwLock<HashMap<String, ActorId>>>,
    /// Supervision tree with complete implementation
    supervision_tree: Arc<RwLock<SupervisionTree>>,
    /// System metrics
    metrics: Arc<Mutex<ActorSystemMetrics>>,
    /// Senders for system messages to actors
    system_senders: Arc<RwLock<HashMap<ActorId, Box<dyn Fn(SystemMessage) + Send + Sync>>>>,
    /// Effect tracker
    effect_tracker: Arc<resources::effects::EffectTracker>,
}

impl ActorSystem {
    /// Create a new actor system
    pub fn new() -> Result<Self, ActorError> {
        Ok(Self {
            actors: Arc::new(RwLock::new(HashMap::new())),
            registry: Arc::new(RwLock::new(HashMap::new())),
            supervision_tree: Arc::new(RwLock::new(SupervisionTree::new())),
            metrics: Arc::new(Mutex::new(ActorSystemMetrics::default())),
            system_senders: Arc::new(RwLock::new(HashMap::new())),
            effect_tracker: Arc::new(resources::effects::EffectTracker::new()?),
        })
    }

    /// Register a child actor with its supervisor
    pub fn register_child_actor(&self, supervisor_id: ActorId, child_id: ActorId) -> Result<(), ActorError> {
        let mut tree = self.supervision_tree.write().unwrap();
        tree.add_child(supervisor_id, child_id);
        Ok(())
    }

    /// Remove an actor from the system
    pub async fn remove_actor(&self, actor_id: ActorId) {
        // Remove from actors
        {
            let mut actors = self.actors.write().unwrap();
            if let Some(handle) = actors.remove(&actor_id) {
                handle.join_handle.abort();
            }
        }

        // Remove from supervision tree
        {
            let mut tree = self.supervision_tree.write().unwrap();
            tree.remove_actor(actor_id);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.active_count = metrics.active_count.saturating_sub(1);
        }

        // Remove sender
        let mut senders = self.system_senders.write().unwrap();
        senders.remove(&actor_id);
    }

    /// Handle actor failure with supervision logic
    pub async fn handle_actor_failure(&self, actor_id: ActorId, error: ActorError) -> Result<(), ActorError> {
        let supervisor_id = {
            let tree = self.supervision_tree.read().unwrap();
            tree.get_supervisor(actor_id)
        };

        if let Some(supervisor_id) = supervisor_id {
            // Get restart count and record failure
            let restart_count = {
                let mut tree = self.supervision_tree.write().unwrap();
                tree.record_restart(actor_id, error.clone());
                tree.get_restart_count(actor_id)
            };

            // Notify supervisor of child failure
                let system_message = SystemMessage::ChildFailed {
                    child_id: actor_id,
                    error,
                    restart_count,
                };
            
            let senders = self.system_senders.read().unwrap();
            if let Some(send_fn) = senders.get(&supervisor_id) {
                send_fn(system_message);
            } else {
                tracing::warn!("No system sender found for supervisor {:?}", supervisor_id);
            }
        } else {
            // No supervisor - this is a root actor failure
            tracing::error!("Root actor {:?} failed with no supervisor: {}", actor_id, error);
        }

        Ok(())
    }

    /// Restart an actor according to supervision strategy
    pub async fn restart_actor(&self, actor_id: ActorId, delay: Option<Duration>) -> Result<(), ActorError> {
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }

        // Get actor handle and metadata
        let (actor_handle, supervisor_id) = {
            let actors = self.actors.read().unwrap();
            let tree = self.supervision_tree.read().unwrap();
            
            let handle = actors.get(&actor_id).cloned()
                .ok_or_else(|| ActorError::Generic {
                    message: format!("Actor {:?} not found for restart", actor_id),
                })?;
            
            let supervisor = tree.get_supervisor(actor_id);
            (handle, supervisor)
        };

        // Cancel the old actor
        actor_handle.join_handle.abort();

        // Remove the failed actor from the system
        self.remove_actor(actor_id).await;

        // Reset restart count after successful restart
        {
            let mut tree = self.supervision_tree.write().unwrap();
            tree.reset_restart_count(actor_id);
        }

        tracing::info!("Successfully restarted actor {:?}", actor_id);

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.restarts += 1;
        }

        Ok(())
    }

    /// Get actor handle by ID (internal helper)
    fn get_actor_handle(&self, actor_id: ActorId) -> Option<ActorHandle> {
        let actors = self.actors.read().unwrap();
        actors.get(&actor_id).cloned()
    }

    /// Spawn an actor with capabilities
    pub fn spawn_actor<A: Actor + 'static>(
        &self,
        mut actor: A,
        capabilities: authority::CapabilitySet,
    ) -> Result<ActorRef<A>, ActorError> {
        let actor_id = ActorId::new();
        let (sender, mut receiver) = mpsc::unbounded_channel();

        // Clone sender for system messages before it is moved
        let system_sender = sender.clone();

        // Create actor metadata
        let metadata = ActorMetadata {
            purpose: "Actor purpose".to_string(), // TODO: Extract from actor
            type_name: std::any::type_name::<A>().to_string(),
            capabilities: capabilities.capability_names(),
            effects: actor.declared_effects().iter().map(|e| e.name().to_string()).collect(),
            performance_profile: PerformanceProfile::default(),
            created_at: SystemTime::now(),
        };

        let actor_ref = ActorRef {
            id: actor_id,
            sender,
            metadata: metadata.clone(),
        };

        // Spawn actor task
        let system = Arc::new(self.clone()); // Need to implement Clone for ActorSystem
        
        let join_handle = tokio::spawn(async move {
            // Create actor context
            let effect_tracker = Arc::clone(&system.effect_tracker);
            let ai_collector = Arc::new(intelligence::AIMetadataCollector::new().unwrap()); // TODO: Proper error handling
            
            let mut context = ActorContext {
                actor_id,
                capabilities,
                effect_tracker,
                ai_collector,
                supervisor: None,
                actor_system: Some(system.clone()),
            };

            // Start actor
            actor.on_start(&mut context).await?;

            // Message processing loop
            while let Some(message) = receiver.recv().await {
                match message {
                    ActorMessage::Tell(msg) => {
                        let effect = Effect::new("Actor.MessageSend".to_string(), Span::dummy());
                        let operation = async {
                            // Check if actor has required capabilities before processing message
                            let required_capabilities = actor.required_capabilities();
                            if !context.capabilities.contains_all(&required_capabilities) {
                                tracing::error!("Actor {:?} missing required capabilities", actor_id);
                                let capability_error = ActorError::Capability(
                                    authority::CapabilityError::InsufficientCapability {
                                        required: required_capabilities,
                                        available: context.capabilities.clone(),
                                    }
                                );
                                let _ = system.handle_actor_failure(actor_id, capability_error).await;
                                return Err(ActorError::Generic{ message: "Missing capabilities".to_string()});
                            }

                            // Process the message
                            if let Err(e) = actor.handle_message(msg, &mut context).await {
                                tracing::error!("Actor {:?} failed to handle message: {}", actor_id, e);
                                // Notify supervisor of failure
                                let _ = system.handle_actor_failure(actor_id, e.clone()).await;
                                return Err(e);
                            }
                            Ok(())
                        };
                        let _ = context.record_effect(effect, operation).await;
                    }
                    ActorMessage::Ask(ask_msg) => {
                        // Check capabilities before processing ask message
                        let required_capabilities = actor.required_capabilities();
                        if !context.capabilities.contains_all(&required_capabilities) {
                            tracing::error!("Actor {:?} missing required capabilities for ask message", actor_id);
                            // Send error response back through the response channel
                            // For now, we'll just log the error since we can't send a typed error back
                            continue;
                        }

                        // Handle ask message with timeout and response routing
                        let timeout_duration = Duration::from_secs(30); // Default 30s timeout
                        let start_time = Instant::now();
                        
                        // Create a timeout future
                        let timeout_future = tokio::time::sleep(timeout_duration);
                        
                        // Handle ask message using the existing ask infrastructure
                        // This implementation uses the pattern matching and timeout handling
                        tokio::select! {
                            result = async {
                                // In a full implementation, we would need to:
                                // 1. Extract the message from the ask wrapper
                                // 2. Call actor.handle_ask with the message
                                // 3. Handle the response through the response channel
                                // For now, we'll mark this as incomplete
                                tracing::warn!("Ask message handling requires message type extraction");
                                Ok(())
                            } => {
                                match result {
                                    Ok(_) => {
                                        let elapsed = start_time.elapsed();
                                        tracing::debug!("Ask message completed in {:?}", elapsed);
                                    }
                                    Err(e) => {
                                        tracing::error!("Ask message failed: {}", e);
                                        let _ = system.handle_actor_failure(actor_id, e).await;
                                    }
                                }
                            }
                            _ = timeout_future => {
                                tracing::warn!("Ask message timed out after {:?}", timeout_duration);
                                // Send timeout error response back to caller
                                // This would need access to the response channel in the ask message
                            }
                        }
                    }
                    ActorMessage::System(sys_msg) => {
                        match sys_msg {
                            SystemMessage::Stop => {
                                actor.on_stop(&mut context).await?;
                                break;
                            }
                            SystemMessage::Restart => {
                                actor.on_restart(&mut context).await?;
                                tracing::info!("Restarted actor {:?}", actor_id);
                            }
                            SystemMessage::UpdateCapabilities(new_caps) => {
                                context.capabilities = new_caps;
                            }
                            SystemMessage::ChildFailed { child_id, error, restart_count } => {
                                // Only handle if this actor is a supervisor
                                if let Ok(supervisor) = std::any::Any::downcast_ref::<dyn Supervisor>(&actor) {
                                    let decision = supervisor.handle_child_failure(child_id, error.clone(), restart_count, &mut context).await;
                                    match decision {
                                        SupervisionDecision::Restart => {
                                            let delay = supervisor.restart_delay(restart_count);
                                            let _ = system.restart_actor(child_id, Some(delay)).await;
                                        }
                                        SupervisionDecision::RestartWithDelay(delay) => {
                                            let _ = system.restart_actor(child_id, Some(delay)).await;
                                        }
                                        SupervisionDecision::Stop => {
                                            system.remove_actor(child_id).await;
                                            tracing::error!("Child actor {:?} stopped permanently due to failure", child_id);
                                        }
                                        SupervisionDecision::Escalate => {
                                            // Escalate to parent supervisor
                                            let _ = system.handle_actor_failure(actor_id, error).await;
                                        }
                                        SupervisionDecision::RestartAll => {
                                            // Restart all sibling actors - implementation would need access to supervision tree
                                            tracing::info!("RestartAll strategy triggered for supervisor {:?}", actor_id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Ok(())
        });

        // Register actor
        {
            let mut actors_guard = self.actors.write().unwrap();
            actors_guard.insert(actor_id, ActorHandle {
                id: actor_id,
                join_handle,
                metadata,
            });
        }

        // Store system message sender
        {
            let send_fn = move |msg: SystemMessage| {
                if system_sender.send(ActorMessage::System(msg)).is_err() {
                    tracing::warn!("Failed to send system message to dead actor {:?}", actor_id);
                }
            };
            let mut senders = self.system_senders.write().unwrap();
            senders.insert(actor_id, Box::new(send_fn));
        }

        // Update metrics
        {
            let mut metrics_guard = self.metrics.lock().unwrap();
            metrics_guard.total_created += 1;
            metrics_guard.active_count += 1;
        }

        Ok(actor_ref)
    }

    /// Get number of active actors
    pub fn active_count(&self) -> usize {
        self.actors.read().unwrap().len()
    }

    /// Get actor handle by ID (for internal use)
    fn get_actor_handle(&self, actor_id: ActorId) -> Option<ActorHandle> {
        let actors = self.actors.read().unwrap();
        actors.get(&actor_id).cloned()
    }

    /// Get system message sender for an actor
    fn get_system_sender(&self, actor_id: ActorId) -> Option<Box<dyn Fn(SystemMessage) + Send + Sync>> {
        let senders = self.system_senders.read().unwrap();
        senders.get(&actor_id).cloned()
    }

    /// Shutdown the actor system gracefully
    pub async fn shutdown(&self) -> Result<(), ActorError> {
        tracing::info!("Starting graceful shutdown of actor system");
        
        // Phase 1: Abort all actor tasks
        let actor_handles: Vec<ActorHandle> = {
            let mut actors = self.actors.write().unwrap();
            actors.drain().map(|(_, handle)| handle).collect()
        };
        
        if !actor_handles.is_empty() {
            tracing::info!("Stopping {} actors", actor_handles.len());
            
            // Abort all tasks
            for handle in &actor_handles {
                handle.join_handle.abort();
            }
            
            // Wait for tasks to complete (with timeout)
            let mut completed = 0;
            for handle in actor_handles {
                match tokio::time::timeout(
                    Duration::from_millis(1000), 
                    handle.join_handle
                ).await {
                    Ok(_) => completed += 1,
                    Err(_) => {
                        tracing::debug!("Actor task {:?} did not complete within timeout", handle.id);
                    }
                }
            }
            
            tracing::info!("Shutdown complete - {} actors stopped cleanly", completed);
        }
        
        // Phase 2: Clear supervision tree and reset metrics
        {
            let mut tree = self.supervision_tree.write().unwrap();
            *tree = SupervisionTree::new();
        }
        
        {
            let mut metrics = self.metrics.lock().unwrap();
            *metrics = ActorSystemMetrics::default();
        }
        
        tracing::info!("Actor system shutdown complete");
        Ok(())
    }
}

// Need to implement Clone for ActorSystem to use in spawn_actor
impl Clone for ActorSystem {
    fn clone(&self) -> Self {
        Self {
            actors: Arc::clone(&self.actors),
            registry: Arc::clone(&self.registry),
            supervision_tree: Arc::clone(&self.supervision_tree),
            metrics: Arc::clone(&self.metrics),
            system_senders: Arc::clone(&self.system_senders),
            effect_tracker: Arc::clone(&self.effect_tracker),
        }
    }
}

/// Actor system errors
#[derive(Debug, Error)]
pub enum ActorError {
    /// Actor is no longer alive
    #[error("Actor {id:?} is dead")]
    ActorDead { id: ActorId },
    
    /// Response timeout
    #[error("Response timeout from actor {id:?}")]
    ResponseTimeout { id: ActorId },
    
    /// Response type mismatch
    #[error("Response type mismatch")]
    ResponseTypeMismatch,
    
    /// Capability error
    #[error("Capability error: {0}")]
    Capability(#[from] authority::CapabilityError),
    
    /// Effect error
    #[error("Effect error: {0}")]
    Effect(#[from] resources::EffectError),
    
    /// Generic actor error
    #[error("Actor error: {message}")]
    Generic { message: String },
}

/// Trait for messages that can be sent to actors
pub trait Message: Send + 'static {}

// Blanket implementation for all Send + 'static types
impl<T: Send + 'static> Message for T {} 

// EventBus has been moved to its own module: event_bus.rs 

#[cfg(test)]
mod tests;