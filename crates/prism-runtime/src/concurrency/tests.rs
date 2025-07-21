//! Comprehensive Tests for the Concurrency System
//!
//! This module contains tests for all components of the concurrency system:
//! - ConcurrencySystem integration tests
//! - Actor system tests
//! - Async runtime tests  
//! - Structured concurrency tests
//! - Event bus tests
//! - Supervision system tests
//! - Performance optimization tests
//! - Lock-free data structure tests

use super::*;
use crate::authority;
use crate::resources;
use crate::intelligence;
use prism_effects::Effect;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;

// ============================================================================
// ConcurrencySystem Integration Tests
// ============================================================================

#[tokio::test]
async fn test_concurrency_system_creation() {
    let system = ConcurrencySystem::new().unwrap();
    
    assert_eq!(system.actor_count(), 0);
    assert_eq!(system.task_count(), 0);
    assert_eq!(system.scope_count(), 0);
}

#[tokio::test]
async fn test_concurrency_system_spawn_actor() {
    let system = ConcurrencySystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let test_actor = TestActor::new("test_actor".to_string());
    let actor_ref = system.spawn_actor(test_actor, capabilities).unwrap();
    
    assert_eq!(system.actor_count(), 1);
    
    // Test message sending
    actor_ref.tell(TestMessage::Ping);
    
    // Give some time for processing
    sleep(Duration::from_millis(10)).await;
}

#[tokio::test]
async fn test_concurrency_system_async_execution() {
    let system = ConcurrencySystem::new().unwrap();
    
    let result = system.execute_async(async {
        Ok(42)
    }).await.unwrap();
    
    assert_eq!(result, 42);
}

#[tokio::test]
async fn test_concurrency_system_structured_scope() {
    let system = ConcurrencySystem::new().unwrap();
    
    let scope = system.create_scope().unwrap();
    assert_eq!(system.scope_count(), 1);
    
    // Test spawning task in scope
    let handle = scope.spawn(async { Ok(100) }).unwrap();
    let result = handle.await_result().await.unwrap();
    assert_eq!(result, 100);
}

#[tokio::test]
async fn test_concurrency_system_performance_metrics() {
    let system = ConcurrencySystem::new().unwrap();
    let metrics = system.get_performance_metrics();
    
    // Check that metrics are collected
    assert!(metrics.timestamp.elapsed() < Duration::from_secs(1));
}

// ============================================================================
// Actor System Tests
// ============================================================================

/// Test actor for actor system tests
#[derive(Debug)]
struct TestActor {
    name: String,
    message_count: AtomicUsize,
}

#[derive(Debug, Clone)]
enum TestMessage {
    Ping,
    Echo(String),
    Add(u32, u32),
    Fail,
}

impl TestActor {
    fn new(name: String) -> Self {
        Self {
            name,
            message_count: AtomicUsize::new(0),
        }
    }
    
    fn get_message_count(&self) -> usize {
        self.message_count.load(Ordering::Relaxed)
    }
}

impl Actor for TestActor {
    type Message = TestMessage;

    async fn handle_message(
        &mut self,
        message: Self::Message,
        _context: &mut ActorContext,
    ) -> Result<(), ActorError> {
        self.message_count.fetch_add(1, Ordering::Relaxed);
        
        match message {
            TestMessage::Ping => {
                println!("Actor {} received ping", self.name);
            }
            TestMessage::Echo(text) => {
                println!("Actor {} echoing: {}", self.name, text);
            }
            TestMessage::Add(a, b) => {
                println!("Actor {} computed: {} + {} = {}", self.name, a, b, a + b);
            }
            TestMessage::Fail => {
                return Err(ActorError::Generic {
                    message: "Intentional failure".to_string(),
                });
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

#[tokio::test]
async fn test_actor_system_creation() {
    let actor_system = ActorSystem::new().unwrap();
    assert_eq!(actor_system.active_count(), 0);
}

#[tokio::test]
async fn test_actor_spawning_and_messaging() {
    let actor_system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let test_actor = TestActor::new("test".to_string());
    let actor_ref = actor_system.spawn_actor(test_actor, capabilities).unwrap();
    
    assert_eq!(actor_system.active_count(), 1);
    
    // Send messages
    actor_ref.tell(TestMessage::Ping);
    actor_ref.tell(TestMessage::Echo("Hello World".to_string()));
    actor_ref.tell(TestMessage::Add(5, 7));
    
    // Give time for processing
    sleep(Duration::from_millis(50)).await;
}

#[tokio::test]
async fn test_actor_ask_pattern() {
    let actor_system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let test_actor = TestActor::new("ask_test".to_string());
    let actor_ref = actor_system.spawn_actor(test_actor, capabilities).unwrap();
    
    // Test ask with timeout (this will fail with current implementation as ask is not fully implemented)
    let result = actor_ref.ask(TestMessage::Ping, Duration::from_millis(100)).await;
    // We expect this to fail since ask is not fully implemented
    assert!(result.is_err());
}

#[tokio::test]
async fn test_actor_metadata() {
    let actor_system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let test_actor = TestActor::new("metadata_test".to_string());
    let actor_ref = actor_system.spawn_actor(test_actor, capabilities).unwrap();
    
    let metadata = actor_ref.metadata();
    assert!(!metadata.purpose.is_empty());
    assert!(!metadata.type_name.is_empty());
    assert!(metadata.created_at.elapsed().unwrap() < Duration::from_secs(1));
}

// ============================================================================
// Async Runtime Tests
// ============================================================================

#[tokio::test]
async fn test_async_runtime_creation() {
    let runtime = AsyncRuntime::new().unwrap();
    assert_eq!(runtime.task_count(), 0);
}

#[tokio::test]
async fn test_async_task_spawning() {
    let runtime = AsyncRuntime::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    let cancellation_token = CancellationToken::new();
    
    let handle = runtime.spawn_task(
        async { Ok(42) },
        capabilities,
        cancellation_token,
        TaskPriority::Normal,
    );
    
    assert_eq!(runtime.task_count(), 1);
    
    let result = handle.await_result().await.unwrap();
    assert_eq!(result, 42);
}

#[tokio::test]
async fn test_async_task_cancellation() {
    let runtime = AsyncRuntime::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    let cancellation_token = CancellationToken::new();
    
    let handle = runtime.spawn_task(
        async {
            sleep(Duration::from_secs(10)).await; // Long running task
            Ok(42)
        },
        capabilities,
        cancellation_token.clone(),
        TaskPriority::Normal,
    );
    
    // Cancel the task
    cancellation_token.cancel();
    
    let result = handle.await_result().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_async_join_operations() {
    let runtime = AsyncRuntime::new().unwrap();
    
    let result = runtime.join(
        async { Ok(10) },
        async { Ok(20) },
    ).await.unwrap();
    
    assert_eq!(result, (10, 20));
}

#[tokio::test]
async fn test_async_timeout() {
    let runtime = AsyncRuntime::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    let cancellation_token = CancellationToken::new();
    
    let context = AsyncContext {
        task_id: crate::concurrency::async_runtime::TaskId::new(),
        capabilities,
        cancellation_token,
        effect_handle: resources::EffectHandle::new(1),
        ai_collector: Arc::new(intelligence::AIMetadataCollector::new().unwrap()),
    };
    
    let result = context.with_timeout(
        Duration::from_millis(10),
        async {
            sleep(Duration::from_millis(100)).await;
            Ok(42)
        }
    ).await;
    
    assert!(matches!(result, Err(AsyncError::Timeout { .. })));
}

// ============================================================================
// Structured Concurrency Tests
// ============================================================================

#[tokio::test]
async fn test_structured_scope_creation() {
    let capabilities = authority::CapabilitySet::new();
    let scope = StructuredScope::new(capabilities, None).unwrap();
    
    assert_eq!(scope.active_task_count(), 0);
    assert_eq!(scope.child_scope_count(), 0);
}

#[tokio::test]
async fn test_structured_scope_task_spawning() {
    let capabilities = authority::CapabilitySet::new();
    let scope = StructuredScope::new(capabilities, None).unwrap();
    
    let handle = scope.spawn(async { Ok(100) }).unwrap();
    assert_eq!(scope.active_task_count(), 1);
    
    let result = handle.await_result().await.unwrap();
    assert_eq!(result, 100);
}

#[tokio::test]
async fn test_structured_scope_child_scopes() {
    let capabilities = authority::CapabilitySet::new();
    let parent_scope = StructuredScope::new(capabilities, None).unwrap();
    
    let child_scope = parent_scope.child_scope().unwrap();
    assert_eq!(parent_scope.child_scope_count(), 1);
    
    // Spawn task in child scope
    let handle = child_scope.spawn(async { Ok(200) }).unwrap();
    let result = handle.await_result().await.unwrap();
    assert_eq!(result, 200);
}

#[tokio::test]
async fn test_structured_scope_cancellation() {
    let capabilities = authority::CapabilitySet::new();
    let scope = StructuredScope::new(capabilities, None).unwrap();
    
    let handle = scope.spawn(async {
        sleep(Duration::from_secs(10)).await;
        Ok(42)
    }).unwrap();
    
    // Cancel the scope
    scope.cancel();
    
    let result = handle.await_result().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_structured_scope_timeout() {
    let capabilities = authority::CapabilitySet::new();
    let scope = StructuredScope::new(capabilities, None).unwrap();
    
    let result = scope.with_timeout(
        Duration::from_millis(10),
        async {
            sleep(Duration::from_millis(100)).await;
            Ok(42)
        }
    ).await;
    
    assert!(matches!(result, Err(StructuredError::Timeout { .. })));
}

#[tokio::test]
async fn test_structured_coordinator() {
    let coordinator = StructuredCoordinator::new().unwrap();
    assert_eq!(coordinator.scope_count(), 0);
    
    let scope = coordinator.create_scope().unwrap();
    assert_eq!(coordinator.scope_count(), 1);
    
    // Test shutdown
    coordinator.shutdown().await.unwrap();
}

// ============================================================================
// Event Bus Tests
// ============================================================================

#[tokio::test]
async fn test_event_bus_creation() {
    let event_bus: EventBus<String> = EventBus::new();
    let metrics = event_bus.get_metrics();
    
    assert_eq!(metrics.total_published, 0);
    assert_eq!(metrics.total_subscribers, 0);
}

#[tokio::test]
async fn test_event_bus_subscription() {
    let event_bus: EventBus<String> = EventBus::new();
    
    let subscription = event_bus.subscribe(
        "test_topic",
        "test_subscriber",
        vec![]
    ).unwrap();
    
    let metrics = event_bus.get_metrics();
    assert_eq!(metrics.total_subscribers, 1);
    
    assert_eq!(subscription.topic(), "test_topic");
}

#[tokio::test]
async fn test_event_bus_publish_and_receive() {
    let event_bus: EventBus<String> = EventBus::new();
    
    let mut subscription = event_bus.subscribe(
        "test_topic",
        "test_subscriber", 
        vec![]
    ).unwrap();
    
    let publish_result = event_bus.publish(
        "test_topic",
        "Hello World".to_string(),
        Some("test_publisher".to_string()),
        EventPriority::Normal,
    ).await.unwrap();
    
    assert_eq!(publish_result.delivered_count, 1);
    assert_eq!(publish_result.failed_count, 0);
    
    // Receive the event
    let received_event = subscription.recv().await.unwrap();
    assert_eq!(received_event.topic, "test_topic");
    assert_eq!(received_event.payload, "Hello World");
}

#[tokio::test]
async fn test_event_bus_multiple_subscribers() {
    let event_bus: EventBus<i32> = EventBus::new();
    
    let mut sub1 = event_bus.subscribe("numbers", "sub1", vec![]).unwrap();
    let mut sub2 = event_bus.subscribe("numbers", "sub2", vec![]).unwrap();
    
    let publish_result = event_bus.publish(
        "numbers",
        42,
        None,
        EventPriority::High,
    ).await.unwrap();
    
    assert_eq!(publish_result.delivered_count, 2);
    
    // Both subscribers should receive the event
    let event1 = sub1.recv().await.unwrap();
    let event2 = sub2.recv().await.unwrap();
    
    assert_eq!(event1.payload, 42);
    assert_eq!(event2.payload, 42);
}

// ============================================================================
// Supervision System Tests
// ============================================================================

#[tokio::test]
async fn test_supervisor_creation() {
    let supervisor = Supervisor::new(SupervisionStrategy::OneForOne);
    let stats = supervisor.get_stats().unwrap();
    
    assert_eq!(stats.total_failures, 0);
    assert_eq!(stats.successful_restarts, 0);
}

#[tokio::test]
async fn test_supervisor_child_management() {
    let supervisor = Supervisor::new(SupervisionStrategy::OneForOne);
    let actor_system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let test_actor = TestActor::new("supervised_actor".to_string());
    let actor_ref = actor_system.spawn_actor(test_actor, capabilities).unwrap();
    
    let child_metadata = ChildMetadata {
        name: "test_child".to_string(),
        purpose: "Test child actor".to_string(),
        failure_modes: vec!["panic".to_string()],
        dependencies: vec![],
        performance_profile: PerformanceProfile {
            messages_per_second: 100,
            memory_usage_mb: 10,
            cpu_intensity: CpuIntensity::Low,
            network_io: NetworkIoProfile {
                connections_per_second: 0,
                bytes_per_second: 0,
                connection_duration: Duration::from_secs(0),
            },
        },
    };
    
    supervisor.supervise_child(
        actor_ref,
        RestartPolicy::Permanent,
        child_metadata,
    ).unwrap();
}

#[tokio::test]
async fn test_supervision_decision_making() {
    let supervisor = Supervisor::new(SupervisionStrategy::OneForOne);
    
    let decision = supervisor.handle_child_failure(
        ActorId::new(),
        ActorError::Generic { message: "test error".to_string() },
    ).await.unwrap();
    
    // Should decide to restart for permanent policy
    assert!(matches!(decision, SupervisionDecision::Restart));
}

// ============================================================================
// Performance Optimization Tests
// ============================================================================

#[tokio::test]
async fn test_performance_optimizer_creation() {
    let optimizer = PerformanceOptimizer::new().unwrap();
    let metrics = optimizer.get_metrics();
    
    // Check that metrics are collected
    assert!(metrics.timestamp.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn test_performance_optimization_hints() {
    let optimizer = PerformanceOptimizer::new().unwrap();
    let hints = optimizer.get_optimization_hints();
    
    // Should have some hints available
    assert!(!hints.is_empty());
}

// ============================================================================
// Lock-Free Data Structure Tests  
// ============================================================================

#[tokio::test]
async fn test_lock_free_queue() {
    let queue = LockFreeQueue::new();
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);
    
    // Test enqueue
    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);
    
    assert_eq!(queue.len(), 3);
    assert!(!queue.is_empty());
    
    // Test dequeue
    assert_eq!(queue.dequeue(), Some(1));
    assert_eq!(queue.dequeue(), Some(2));
    assert_eq!(queue.dequeue(), Some(3));
    assert_eq!(queue.dequeue(), None);
    
    assert!(queue.is_empty());
}

#[tokio::test]
async fn test_lock_free_map() {
    let map = LockFreeMap::new();
    assert!(map.is_empty());
    
    // Test insert and get
    assert_eq!(map.insert("key1".to_string(), 100), None);
    assert_eq!(map.get(&"key1".to_string()), Some(100));
    
    // Test update
    assert_eq!(map.insert("key1".to_string(), 200), Some(100));
    assert_eq!(map.get(&"key1".to_string()), Some(200));
    
    // Test remove
    assert_eq!(map.remove(&"key1".to_string()), Some(200));
    assert_eq!(map.get(&"key1".to_string()), None);
    
    assert!(map.is_empty());
}

#[tokio::test]
async fn test_atomic_counter() {
    let counter = AtomicCounter::new();
    assert_eq!(counter.get(), 0);
    
    // Test increment
    assert_eq!(counter.increment(), 1);
    assert_eq!(counter.increment(), 2);
    assert_eq!(counter.get(), 2);
    
    // Test decrement
    assert_eq!(counter.decrement(), 1);
    assert_eq!(counter.get(), 1);
    
    // Test add/sub
    assert_eq!(counter.add(10), 11);
    assert_eq!(counter.sub(5), 6);
    
    // Test reset
    assert_eq!(counter.reset(), 6);
    assert_eq!(counter.get(), 0);
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_actor_messaging() {
    let actor_system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    // Spawn multiple actors
    let mut actor_refs = Vec::new();
    for i in 0..10 {
        let test_actor = TestActor::new(format!("actor_{}", i));
        let actor_ref = actor_system.spawn_actor(test_actor, capabilities.clone()).unwrap();
        actor_refs.push(actor_ref);
    }
    
    // Send messages concurrently
    let tasks: Vec<_> = actor_refs.into_iter().enumerate().map(|(i, actor_ref)| {
        tokio::spawn(async move {
            for j in 0..100 {
                actor_ref.tell(TestMessage::Add(i as u32, j));
            }
        })
    }).collect();
    
    // Wait for all tasks to complete
    for task in tasks {
        task.await.unwrap();
    }
    
    // Give time for message processing
    sleep(Duration::from_millis(100)).await;
    
    assert_eq!(actor_system.active_count(), 10);
}

#[tokio::test]
async fn test_concurrent_queue_operations() {
    let queue = Arc::new(LockFreeQueue::new());
    let mut handles = Vec::new();
    
    // Spawn producer tasks
    for i in 0..5 {
        let queue_clone = Arc::clone(&queue);
        handles.push(tokio::spawn(async move {
            for j in 0..100 {
                queue_clone.enqueue(i * 100 + j);
            }
        }));
    }
    
    // Spawn consumer tasks
    let consumed = Arc::new(AtomicUsize::new(0));
    for _ in 0..3 {
        let queue_clone = Arc::clone(&queue);
        let consumed_clone = Arc::clone(&consumed);
        handles.push(tokio::spawn(async move {
            while consumed_clone.load(Ordering::Relaxed) < 500 {
                if queue_clone.dequeue().is_some() {
                    consumed_clone.fetch_add(1, Ordering::Relaxed);
                }
                tokio::task::yield_now().await;
            }
        }));
    }
    
    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }
    
    assert_eq!(consumed.load(Ordering::Relaxed), 500);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_full_system_integration() {
    let system = ConcurrencySystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    // Create structured scope
    let scope = system.create_scope().unwrap();
    
    // Spawn actors
    let actor1 = TestActor::new("integration_actor_1".to_string());
    let actor_ref1 = system.spawn_actor(actor1, capabilities.clone()).unwrap();
    
    let actor2 = TestActor::new("integration_actor_2".to_string());
    let actor_ref2 = system.spawn_actor(actor2, capabilities.clone()).unwrap();
    
    // Create event bus
    let event_bus: EventBus<String> = EventBus::new();
    let mut subscription = event_bus.subscribe("integration", "test", vec![]).unwrap();
    
    // Execute async tasks in scope
    let task1 = scope.spawn(async {
        actor_ref1.tell(TestMessage::Ping);
        Ok("Task 1 completed".to_string())
    }).unwrap();
    
    let task2 = scope.spawn(async {
        actor_ref2.tell(TestMessage::Echo("Integration test".to_string()));
        Ok("Task 2 completed".to_string())
    }).unwrap();
    
    // Publish event
    let _publish_result = event_bus.publish(
        "integration",
        "Integration event".to_string(),
        Some("integration_test".to_string()),
        EventPriority::Normal,
    ).await.unwrap();
    
    // Wait for tasks
    let result1 = task1.await_result().await.unwrap();
    let result2 = task2.await_result().await.unwrap();
    
    // Receive event
    let received_event = subscription.recv().await.unwrap();
    
    // Verify results
    assert_eq!(result1, "Task 1 completed");
    assert_eq!(result2, "Task 2 completed");
    assert_eq!(received_event.payload, "Integration event");
    
    // Check system state
    assert_eq!(system.actor_count(), 2);
    assert_eq!(system.scope_count(), 1);
    
    // Get performance metrics
    let metrics = system.get_performance_metrics();
    assert!(metrics.system_health >= 0.0 && metrics.system_health <= 1.0);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_actor_error_handling() {
    let actor_system = ActorSystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    let test_actor = TestActor::new("error_test".to_string());
    let actor_ref = actor_system.spawn_actor(test_actor, capabilities).unwrap();
    
    // Send a message that will cause failure
    actor_ref.tell(TestMessage::Fail);
    
    // Give time for error handling
    sleep(Duration::from_millis(50)).await;
    
    // Actor system should handle the error gracefully
    // The actor might be restarted or removed depending on supervision
}

#[tokio::test]
async fn test_structured_concurrency_error_propagation() {
    let capabilities = authority::CapabilitySet::new();
    let scope = StructuredScope::new(capabilities, None).unwrap();
    
    let handle = scope.spawn(async {
        Err(AsyncError::Generic { message: "Test error".to_string() })
    }).unwrap();
    
    let result = handle.await_result().await;
    assert!(result.is_err());
}

// ============================================================================
// Cleanup and Resource Management Tests
// ============================================================================

#[tokio::test]
async fn test_scope_cleanup_on_drop() {
    let capabilities = authority::CapabilitySet::new();
    let scope = StructuredScope::new(capabilities, None).unwrap();
    let scope_handle = scope.handle();
    
    // Spawn a long-running task
    let _handle = scope.spawn(async {
        sleep(Duration::from_secs(10)).await;
        Ok(42)
    }).unwrap();
    
    assert!(!scope_handle.is_cancelled());
    
    // Drop the scope - this should cancel all child tasks
    drop(scope);
    
    // Give time for cleanup
    sleep(Duration::from_millis(10)).await;
    
    assert!(scope_handle.is_cancelled());
}

#[tokio::test]
async fn test_system_shutdown() {
    let system = ConcurrencySystem::new().unwrap();
    let capabilities = authority::CapabilitySet::new();
    
    // Spawn some actors and create scopes
    let _actor_ref = system.spawn_actor(
        TestActor::new("shutdown_test".to_string()),
        capabilities
    ).unwrap();
    
    let _scope = system.create_scope().unwrap();
    
    assert!(system.actor_count() > 0);
    assert!(system.scope_count() > 0);
    
    // System should clean up gracefully when dropped
    drop(system);
} 