//! Event Bus - Publish-Subscribe Messaging System
//!
//! This module implements the publish-subscribe event bus as specified in PLD-005, providing:
//! - **Topic-based routing**: Events are routed based on topic names
//! - **Filtered subscriptions**: Subscribers can filter events based on properties
//! - **Async delivery**: Non-blocking event delivery to subscribers
//! - **AI metadata**: Rich metadata for AI comprehension of event patterns
//! - **Metrics tracking**: Comprehensive metrics for monitoring and analysis

use crate::concurrency::actor_system::ActorError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, Instant};
use std::marker::PhantomData;
use thiserror::Error;
use serde::{Serialize, Deserialize};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Event bus for publish-subscribe messaging patterns as specified in PLD-005
#[derive(Debug)]
pub struct EventBus<T> 
where 
    T: Send + Sync + Clone + 'static 
{
    /// Subscribers mapped by topic
    subscribers: Arc<RwLock<HashMap<String, Vec<EventSubscriber>>>>,
    /// Event bus metrics
    metrics: Arc<Mutex<EventBusMetrics>>,
    /// AI metadata for comprehension
    ai_metadata: EventBusAIMetadata,
    /// Phantom data to use the generic type
    _phantom: PhantomData<T>,
}

/// Event subscriber information
#[derive(Debug, Clone)]
struct EventSubscriber {
    /// Subscriber ID for tracking
    id: SubscriberId,
    /// Channel to send events
    sender: mpsc::UnboundedSender<EventMessage>,
    /// Subscription metadata
    metadata: SubscriptionMetadata,
}

/// Unique identifier for event subscribers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriberId(Uuid);

impl SubscriberId {
    /// Generate a new unique subscriber ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Event message wrapper
#[derive(Debug, Clone)]
struct EventMessage {
    /// Event topic
    topic: String,
    /// Event payload
    payload: Box<dyn std::any::Any + Send + Sync>,
    /// Event metadata
    metadata: EventMetadata,
}

/// Metadata about an event subscription
#[derive(Debug, Clone)]
struct SubscriptionMetadata {
    /// Subscriber name for identification
    name: String,
    /// When subscription was created
    created_at: SystemTime,
    /// Number of events received
    events_received: u64,
    /// Subscription filters (if any)
    filters: Vec<EventFilter>,
}

/// Event metadata
#[derive(Debug, Clone)]
struct EventMetadata {
    /// Event ID for tracking
    id: EventId,
    /// When event was published
    published_at: SystemTime,
    /// Publisher information
    publisher: Option<String>,
    /// Event priority
    priority: EventPriority,
}

/// Unique identifier for events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(Uuid);

impl EventId {
    /// Generate a new unique event ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    /// Critical events
    Critical = 0,
    /// High priority events
    High = 1,
    /// Normal priority events
    Normal = 2,
    /// Low priority events
    Low = 3,
}

impl Default for EventPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Event filter for selective subscription
#[derive(Debug, Clone)]
pub enum EventFilter {
    /// Filter by event property
    Property { key: String, value: String },
    /// Filter by regex pattern
    Pattern(String),
    /// Custom filter function
    Custom(String), // Function name for AI metadata
}

/// Event bus metrics for monitoring
#[derive(Debug, Default, Clone)]
pub struct EventBusMetrics {
    /// Total events published
    pub total_published: u64,
    /// Total events delivered
    pub total_delivered: u64,
    /// Total subscribers
    pub total_subscribers: usize,
    /// Events by topic
    pub events_by_topic: HashMap<String, u64>,
    /// Failed deliveries
    pub failed_deliveries: u64,
}

/// AI metadata for event bus comprehension
#[derive(Debug, Clone)]
pub struct EventBusAIMetadata {
    /// Business purpose of this event bus
    pub purpose: String,
    /// Event patterns supported
    pub supported_patterns: Vec<String>,
    /// Typical event flow
    pub event_flow: String,
    /// Performance characteristics
    pub performance_characteristics: String,
}

impl Default for EventBusAIMetadata {
    fn default() -> Self {
        Self {
            purpose: "Publish-subscribe event bus for decoupled communication".to_string(),
            supported_patterns: vec![
                "Publish-Subscribe".to_string(),
                "Event Broadcasting".to_string(),
                "Topic-based Routing".to_string(),
            ],
            event_flow: "Publishers send events to topics, subscribers receive matching events".to_string(),
            performance_characteristics: "Async delivery, topic-based routing, filtered subscriptions".to_string(),
        }
    }
}

impl<T> EventBus<T> 
where 
    T: Send + Sync + Clone + 'static 
{
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(EventBusMetrics::default())),
            ai_metadata: EventBusAIMetadata::default(),
            _phantom: PhantomData,
        }
    }

    /// Create a new event bus with custom AI metadata
    pub fn with_metadata(ai_metadata: EventBusAIMetadata) -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(EventBusMetrics::default())),
            ai_metadata,
            _phantom: PhantomData,
        }
    }

    /// Subscribe to a topic
    pub fn subscribe(
        &self,
        topic: impl Into<String>,
        subscriber_name: impl Into<String>,
        filters: Vec<EventFilter>,
    ) -> Result<EventSubscription<T>, ActorError> {
        let topic = topic.into();
        let subscriber_name = subscriber_name.into();
        let subscriber_id = SubscriberId::new();
        
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let subscription_metadata = SubscriptionMetadata {
            name: subscriber_name,
            created_at: SystemTime::now(),
            events_received: 0,
            filters,
        };

        let subscriber = EventSubscriber {
            id: subscriber_id,
            sender,
            metadata: subscription_metadata,
        };

        // Add subscriber to topic
        {
            let mut subscribers = self.subscribers.write().unwrap();
            subscribers
                .entry(topic.clone())
                .or_insert_with(Vec::new)
                .push(subscriber);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_subscribers += 1;
        }

        Ok(EventSubscription {
            id: subscriber_id,
            topic,
            receiver,
            event_bus: Arc::new(self.clone()),
            _phantom: PhantomData,
        })
    }

    /// Publish an event to a topic
    pub async fn publish(
        &self,
        topic: impl Into<String>,
        event: T,
        publisher: Option<String>,
        priority: EventPriority,
    ) -> Result<PublishResult, ActorError> {
        let topic = topic.into();
        let event_id = EventId::new();
        
        let event_metadata = EventMetadata {
            id: event_id,
            published_at: SystemTime::now(),
            publisher,
            priority,
        };

        // Get subscribers for this topic
        let subscribers = {
            let subscribers_guard = self.subscribers.read().unwrap();
            subscribers_guard.get(&topic).cloned().unwrap_or_default()
        };

        let mut delivered_count = 0;
        let mut failed_count = 0;

        // Deliver to all subscribers
        for subscriber in subscribers {
            // Apply filters
            if self.passes_filters(&event, &subscriber.metadata.filters) {
                let event_message = EventMessage {
                    topic: topic.clone(),
                    payload: Box::new(event.clone()),
                    metadata: event_metadata.clone(),
                };

                match subscriber.sender.send(event_message) {
                    Ok(()) => delivered_count += 1,
                    Err(_) => {
                        failed_count += 1;
                        // Subscriber channel is closed - should remove it
                        self.cleanup_dead_subscriber(subscriber.id, &topic).await;
                    }
                }
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_published += 1;
            metrics.total_delivered += delivered_count;
            metrics.failed_deliveries += failed_count;
            *metrics.events_by_topic.entry(topic.clone()).or_insert(0) += 1;
        }

        Ok(PublishResult {
            event_id,
            topic,
            delivered_count,
            failed_count,
        })
    }

    /// Unsubscribe from a topic
    pub fn unsubscribe(&self, topic: &str, subscriber_id: SubscriberId) -> Result<(), ActorError> {
        let mut subscribers = self.subscribers.write().unwrap();
        if let Some(topic_subscribers) = subscribers.get_mut(topic) {
            topic_subscribers.retain(|sub| sub.id != subscriber_id);
            
            // Remove topic if no subscribers left
            if topic_subscribers.is_empty() {
                subscribers.remove(topic);
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_subscribers = metrics.total_subscribers.saturating_sub(1);
        }

        Ok(())
    }

    /// Get event bus metrics
    pub fn get_metrics(&self) -> EventBusMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Get AI metadata
    pub fn ai_metadata(&self) -> &EventBusAIMetadata {
        &self.ai_metadata
    }

    /// Check if event passes subscriber filters
    fn passes_filters(&self, event: &T, filters: &[EventFilter]) -> bool {
        // If no filters, event passes
        if filters.is_empty() {
            return true;
        }

        // Event must pass all filters (AND logic)
        for filter in filters {
            if !self.passes_single_filter(event, filter) {
                return false;
            }
        }

        true
    }

    /// Check if event passes a single filter
    fn passes_single_filter(&self, event: &T, filter: &EventFilter) -> bool {
        match filter {
            EventFilter::Property { key, value } => {
                // For demonstration, we'll use a simple string representation check
                // In a real implementation, this would extract properties from the event
                let event_str = format!("{:?}", event);
                event_str.contains(&format!("{}:{}", key, value)) || 
                event_str.contains(&format!("{}: {}", key, value)) ||
                event_str.contains(&format!("\"{}\":\"{}\"", key, value))
            }
            EventFilter::Pattern(pattern) => {
                // Simple pattern matching using string contains
                // In a real implementation, this would use regex
                let event_str = format!("{:?}", event);
                event_str.contains(pattern)
            }
            EventFilter::Custom(function_name) => {
                // For demonstration, we'll match against the function name in the event
                // In a real implementation, this would call a registered custom filter function
                let event_str = format!("{:?}", event);
                event_str.contains(function_name)
            }
        }
    }

    /// Clean up dead subscriber
    async fn cleanup_dead_subscriber(&self, subscriber_id: SubscriberId, topic: &str) {
        let _ = self.unsubscribe(topic, subscriber_id);
        tracing::warn!("Removed dead subscriber {:?} from topic '{}'", subscriber_id, topic);
    }
}

impl<T> Clone for EventBus<T> 
where 
    T: Send + Sync + Clone + 'static 
{
    fn clone(&self) -> Self {
        Self {
            subscribers: Arc::clone(&self.subscribers),
            metrics: Arc::clone(&self.metrics),
            ai_metadata: self.ai_metadata.clone(),
            _phantom: PhantomData,
        }
    }
}

/// Event subscription handle
pub struct EventSubscription<T> 
where 
    T: Send + Sync + Clone + 'static 
{
    /// Subscription ID
    id: SubscriberId,
    /// Topic subscribed to
    topic: String,
    /// Event receiver
    receiver: mpsc::UnboundedReceiver<EventMessage>,
    /// Reference to event bus for unsubscribing
    event_bus: Arc<EventBus<T>>,
    /// Phantom data to use the generic type
    _phantom: PhantomData<T>,
}

impl<T> EventSubscription<T> 
where 
    T: Send + Sync + Clone + 'static 
{
    /// Receive the next event
    pub async fn recv(&mut self) -> Option<ReceivedEvent<T>> {
        match self.receiver.recv().await {
            Some(event_message) => {
                // Try to downcast the payload
                if let Ok(payload) = event_message.payload.downcast::<T>() {
                    Some(ReceivedEvent {
                        topic: event_message.topic,
                        payload: *payload,
                        metadata: event_message.metadata,
                    })
                } else {
                    // Type mismatch - should not happen with proper usage
                    None
                }
            }
            None => None,
        }
    }

    /// Get subscription ID
    pub fn id(&self) -> SubscriberId {
        self.id
    }

    /// Get subscribed topic
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

impl<T> Drop for EventSubscription<T> 
where 
    T: Send + Sync + Clone + 'static 
{
    fn drop(&mut self) {
        // Automatically unsubscribe when dropped
        let _ = self.event_bus.unsubscribe(&self.topic, self.id);
    }
}

/// Received event with metadata
#[derive(Debug)]
pub struct ReceivedEvent<T> {
    /// Event topic
    pub topic: String,
    /// Event payload
    pub payload: T,
    /// Event metadata
    pub metadata: EventMetadata,
}

/// Result of publishing an event
#[derive(Debug)]
pub struct PublishResult {
    /// Event ID
    pub event_id: EventId,
    /// Topic published to
    pub topic: String,
    /// Number of subscribers that received the event
    pub delivered_count: u64,
    /// Number of failed deliveries
    pub failed_count: u64,
} 