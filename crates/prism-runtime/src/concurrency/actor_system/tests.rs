//! Actor System Tests
//!
//! Comprehensive tests for the actor system including:
//! - Actor lifecycle management
//! - Message passing patterns
//! - Supervision hierarchies  
//! - Error handling and recovery
//! - Performance and scalability

use super::*;
use crate::authority;
use crate::resources;
use crate::intelligence;
use prism_effects::Effect;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;

// ============================================================================
// Test Actor Implementations
// ============================================================================

/// Simple counter actor for testing
#[derive(Debug)]
struct CounterActor {
    count: AtomicUsize,
    name: String,
}

#[derive(Debug, Clone)]
enum CounterMessage {
    Increment,
    Decrement,
    GetCount(oneshot::Sender<usize>),
    Reset,
    AddValue(usize),
}

impl CounterActor {
    fn new(name: String) -> Self {
        Self {
            count: AtomicUsize::new(0),
            name,
        }
    }
}

impl Actor for CounterActor {
    type Message = CounterMessage;

    async fn handle_message(
        &mut self,
        message: Self::Message,
        _context: &mut ActorContext,
    ) -> Result<(), ActorError> {
        match message {
            CounterMessage::Increment => {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            CounterMessage::Decrement => {
                self.count.fetch_sub(1, Ordering::SeqCst);
            }
            CounterMessage::GetCount(sender) => {
                let count = self.count.load(Ordering::SeqCst);
                let _ = sender.send(count);
            }
            CounterMessage::Reset => {
                self.count.store(0, Ordering::SeqCst);
            }
            CounterMessage::AddValue(value) => {
                self.count.fetch_add(value, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    fn required_capabilities(&self) -> authority::CapabilitySet {
        authority::CapabilitySet::new()
    }

    fn declared_effects(&self) -> Vec<Effect> {
        vec![]
    }

    fn ai_metadata(&self) -> ActorAIMetadata {
        ActorAIMetadata {
            business_domain: Some("Testing".to_string()),
            state_management: "Atomic counter".to_string(),
            concurrency_safety: "Thread-safe atomic operations".to_string(),
            performance_characteristics: "Low latency, high throughput".to_string(),
        }
    }
}

/// Echo actor that responds to messages
#[derive(Debug)]
struct EchoActor {
    name: String,
    message_count: AtomicUsize,
}

#[derive(Debug, Clone)]
enum EchoMessage {
    Echo(String, oneshot::Sender<String>),
    Ping,
    GetMessageCount(oneshot::Sender<usize>),
}

impl EchoActor {
    fn new(name: String) -> Self {
        Self {
            name,
            message_count: AtomicUsize::new(0),
        }
    }
}

impl Actor for EchoActor {
    type Message = EchoMessage;

    async fn handle_message(
        &mut self,
        message: Self::Message,
        _context: &mut ActorContext,
    ) -> Result<(), ActorError> {
        self.message_count.fetch_add(1, Ordering::SeqCst);
        
        match message {
            EchoMessage::Echo(text, sender) => {
                let response = format!("{}: {}", self.name, text);
                let _ = sender.send(response);
            }
            EchoMessage::Ping => {
                println!("Actor {} received ping", self.name);
            }
            EchoMessage::GetMessageCount(sender) => {
                let count = self.message_count.load(Ordering::SeqCst);
                let _ = sender.send(count);
            }
        }
        Ok(())
    }

    fn required_capabilities(&self) -> authority::CapabilitySet {
        authority::CapabilitySet::new()
    }

    fn declared_effects(&self) -> Vec<Effect> {
        vec![]
    }
}

/// Failing actor for testing error handling
#[derive(Debug)]
struct FailingActor {
    name: String,
    should_fail: Arc<AtomicBool>,
    failure_count: AtomicUsize,
}

#[derive(Debug, Clone)]
enum FailingMessage {
    DoWork,
    ToggleFailure(bool),
    GetFailureCount(oneshot::Sender<usize>),
}

impl FailingActor {
    fn new(name: String) -> Self {
        Self {
            name,
            should_fail: Arc::new(AtomicBool::new(false)),
            failure_count: AtomicUsize::new(0),
        }
    }
}

impl Actor for FailingActor {
    type Message = FailingMessage;

    async fn handle_message(
        &mut self,
        message: Self::Message,
        _context: &mut ActorContext,
    ) -> Result<(), ActorError> {
        match message {
            FailingMessage::DoWork => {
                if self.should_fail.load(Ordering::SeqCst) {
                    self.failure_count.fetch_add(1, Ordering::SeqCst);
                    return Err(ActorError::Generic {
                        message: format!("Actor {} intentionally failed", self.name),
                    });
                }
                println!("Actor {} completed work successfully", self.name);
            }
            FailingMessage::ToggleFailure(should_fail) => {
                self.should_fail.store(should_fail, Ordering::SeqCst);
            }
            FailingMessage::GetFailureCount(sender) => {
                let count = self.failure_count.load(Ordering::SeqCst);
                let _ = sender.send(count);
            }
        }
        Ok(())
    }

    fn required_capabilities(&self) -> authority::CapabilitySet {
        authority::CapabilitySet::new()
    }

    fn declared_effects(&self) -> Vec<Effect> {
        vec![]
    }
}

// ============================================================================
// Basic Actor System Tests
// ============================================================================

#[tokio::test]
async fn test_actor_system_creation() {
    let system = ActorSystem::new().unwrap();
    assert_eq!(system.active_count(), 0);
}

#[tokio::test]
async fn test_single_actor_lifecycle() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let counter = CounterActor::new("test_counter".to_string());
    let actor_ref = system.spawn_actor(counter, capabilities).unwrap();
    
    assert_eq!(system.active_count(), 1);
    
    // Test message sending
    actor_ref.tell(CounterMessage::Increment);
    actor_ref.tell(CounterMessage::Increment);
    actor_ref.tell(CounterMessage::AddValue(5));
    
    // Give time for processing
    sleep(Duration::from_millis(10)).await;
    
    // Query the count
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(CounterMessage::GetCount(tx));
    let count = rx.await.unwrap();
    assert_eq!(count, 7); // 2 increments + 5 added
}

#[tokio::test]
async fn test_multiple_actors() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    // Spawn multiple actors
    let mut actor_refs = Vec::new();
    for i in 0..5 {
        let counter = CounterActor::new(format!("counter_{}", i));
        let actor_ref = system.spawn_actor(counter, capabilities.clone()).unwrap();
        actor_refs.push(actor_ref);
    }
    
    assert_eq!(system.active_count(), 5);
    
    // Send different numbers of increments to each actor
    for (i, actor_ref) in actor_refs.iter().enumerate() {
        for _ in 0..=i {
            actor_ref.tell(CounterMessage::Increment);
        }
    }
    
    // Give time for processing
    sleep(Duration::from_millis(20)).await;
    
    // Verify each actor has the correct count
    for (i, actor_ref) in actor_refs.iter().enumerate() {
        let (tx, rx) = oneshot::channel();
        actor_ref.tell(CounterMessage::GetCount(tx));
        let count = rx.await.unwrap();
        assert_eq!(count, i + 1);
    }
}

#[tokio::test]
async fn test_actor_metadata() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let echo = EchoActor::new("metadata_test".to_string());
    let actor_ref = system.spawn_actor(echo, capabilities).unwrap();
    
    let metadata = actor_ref.metadata();
    assert!(!metadata.purpose.is_empty());
    assert!(!metadata.type_name.is_empty());
    assert!(metadata.type_name.contains("EchoActor"));
    assert!(metadata.created_at.elapsed().unwrap() < Duration::from_secs(1));
}

// ============================================================================
// Message Passing Tests
// ============================================================================

#[tokio::test]
async fn test_fire_and_forget_messaging() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let echo = EchoActor::new("fire_forget_test".to_string());
    let actor_ref = system.spawn_actor(echo, capabilities).unwrap();
    
    // Send multiple fire-and-forget messages
    for i in 0..100 {
        actor_ref.tell(EchoMessage::Ping);
    }
    
    // Give time for processing
    sleep(Duration::from_millis(50)).await;
    
    // Verify all messages were processed
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(EchoMessage::GetMessageCount(tx));
    let count = rx.await.unwrap();
    assert_eq!(count, 101); // 100 pings + 1 GetMessageCount
}

#[tokio::test]
async fn test_request_response_pattern() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let echo = EchoActor::new("request_response_test".to_string());
    let actor_ref = system.spawn_actor(echo, capabilities).unwrap();
    
    // Test request-response
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(EchoMessage::Echo("Hello World".to_string(), tx));
    
    let response = rx.await.unwrap();
    assert_eq!(response, "request_response_test: Hello World");
}

#[tokio::test]
async fn test_concurrent_messaging() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let counter = CounterActor::new("concurrent_test".to_string());
    let actor_ref = Arc::new(system.spawn_actor(counter, capabilities).unwrap());
    
    // Spawn multiple tasks sending messages concurrently
    let mut handles = Vec::new();
    for _ in 0..10 {
        let actor_ref_clone = Arc::clone(&actor_ref);
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                actor_ref_clone.tell(CounterMessage::Increment);
            }
        }));
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Give time for message processing
    sleep(Duration::from_millis(100)).await;
    
    // Verify final count
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(CounterMessage::GetCount(tx));
    let count = rx.await.unwrap();
    assert_eq!(count, 1000); // 10 tasks * 100 increments each
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_actor_error_handling() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let failing_actor = FailingActor::new("error_test".to_string());
    let actor_ref = system.spawn_actor(failing_actor, capabilities).unwrap();
    
    // Enable failure mode
    actor_ref.tell(FailingMessage::ToggleFailure(true));
    
    // Send work that will fail
    actor_ref.tell(FailingMessage::DoWork);
    actor_ref.tell(FailingMessage::DoWork);
    
    // Give time for error handling
    sleep(Duration::from_millis(50)).await;
    
    // Disable failure mode
    actor_ref.tell(FailingMessage::ToggleFailure(false));
    
    // Send work that should succeed
    actor_ref.tell(FailingMessage::DoWork);
    
    sleep(Duration::from_millis(20)).await;
    
    // Check failure count
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(FailingMessage::GetFailureCount(tx));
    let failure_count = rx.await.unwrap();
    assert_eq!(failure_count, 2);
}

// ============================================================================
// Supervision Tests
// ============================================================================

#[tokio::test]
async fn test_supervision_tree_creation() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    // Create supervisor actor
    let supervisor = CounterActor::new("supervisor".to_string());
    let supervisor_ref = system.spawn_actor(supervisor, capabilities.clone()).unwrap();
    let supervisor_id = supervisor_ref.id();
    
    // Create child actor
    let child = CounterActor::new("child".to_string());
    let child_ref = system.spawn_actor(child, capabilities).unwrap();
    let child_id = child_ref.id();
    
    // Register child with supervisor
    system.register_child_actor(supervisor_id, child_id).unwrap();
    
    assert_eq!(system.active_count(), 2);
}

#[tokio::test]
async fn test_actor_removal() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let actor = CounterActor::new("removal_test".to_string());
    let actor_ref = system.spawn_actor(actor, capabilities).unwrap();
    let actor_id = actor_ref.id();
    
    assert_eq!(system.active_count(), 1);
    
    // Remove the actor
    system.remove_actor(actor_id);
    
    assert_eq!(system.active_count(), 0);
}

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
async fn test_high_throughput_messaging() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let counter = CounterActor::new("throughput_test".to_string());
    let actor_ref = system.spawn_actor(counter, capabilities).unwrap();
    
    let start = std::time::Instant::now();
    
    // Send a large number of messages
    for _ in 0..10000 {
        actor_ref.tell(CounterMessage::Increment);
    }
    
    // Wait for processing to complete
    sleep(Duration::from_millis(500)).await;
    
    let elapsed = start.elapsed();
    println!("Processed 10000 messages in {:?}", elapsed);
    
    // Verify all messages were processed
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(CounterMessage::GetCount(tx));
    let count = rx.await.unwrap();
    assert_eq!(count, 10000);
    
    // Calculate throughput
    let throughput = 10000.0 / elapsed.as_secs_f64();
    println!("Throughput: {:.0} messages/second", throughput);
    
    // Should achieve reasonable throughput
    assert!(throughput > 1000.0);
}

#[tokio::test]
async fn test_memory_usage_patterns() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    // Create actors with different memory patterns
    let mut actor_refs = Vec::new();
    
    for i in 0..100 {
        let counter = CounterActor::new(format!("memory_test_{}", i));
        let actor_ref = system.spawn_actor(counter, capabilities.clone()).unwrap();
        actor_refs.push(actor_ref);
    }
    
    assert_eq!(system.active_count(), 100);
    
    // Send messages to all actors
    for actor_ref in &actor_refs {
        for _ in 0..10 {
            actor_ref.tell(CounterMessage::Increment);
        }
    }
    
    // Give time for processing
    sleep(Duration::from_millis(100)).await;
    
    // Verify all actors are still active
    assert_eq!(system.active_count(), 100);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

#[tokio::test]
async fn test_actor_with_no_messages() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let counter = CounterActor::new("idle_test".to_string());
    let actor_ref = system.spawn_actor(counter, capabilities).unwrap();
    
    // Don't send any messages, just wait
    sleep(Duration::from_millis(100)).await;
    
    // Actor should still be active
    assert_eq!(system.active_count(), 1);
    
    // Should be able to send a message after idle period
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(CounterMessage::GetCount(tx));
    let count = rx.await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_rapid_actor_creation_and_destruction() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    for i in 0..50 {
        // Create actor
        let counter = CounterActor::new(format!("rapid_{}", i));
        let actor_ref = system.spawn_actor(counter, capabilities.clone()).unwrap();
        let actor_id = actor_ref.id();
        
        // Send a few messages
        actor_ref.tell(CounterMessage::Increment);
        actor_ref.tell(CounterMessage::Increment);
        
        // Give minimal time for processing
        sleep(Duration::from_millis(1)).await;
        
        // Remove actor
        system.remove_actor(actor_id);
    }
    
    // System should be clean
    assert_eq!(system.active_count(), 0);
}

// ============================================================================
// Integration with Other Systems Tests
// ============================================================================

#[tokio::test]
async fn test_actor_with_capabilities() {
    let system = ActorSystem::new().unwrap();
    let mut capabilities = authority::CapabilitySet::new();
    // TODO: Add actual capabilities when capability system is implemented
    
    let counter = CounterActor::new("capability_test".to_string());
    let actor_ref = system.spawn_actor(counter, capabilities).unwrap();
    
    // Actor should be created successfully with capabilities
    assert_eq!(system.active_count(), 1);
    
    // Should be able to send messages
    actor_ref.tell(CounterMessage::Increment);
    
    sleep(Duration::from_millis(10)).await;
    
    let (tx, rx) = oneshot::channel();
    actor_ref.tell(CounterMessage::GetCount(tx));
    let count = rx.await.unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn test_actor_effect_declaration() {
    let system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let counter = CounterActor::new("effects_test".to_string());
    let effects = counter.declared_effects();
    let actor_ref = system.spawn_actor(counter, capabilities).unwrap();
    
    // Effects should be properly declared (empty in this case)
    assert_eq!(effects.len(), 0);
    
    // Actor metadata should include effects
    let metadata = actor_ref.metadata();
    assert_eq!(metadata.effects.len(), 0);
} 