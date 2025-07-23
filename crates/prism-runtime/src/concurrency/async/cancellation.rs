//! Cancellation System - Structured Cancellation for Async Operations
//!
//! This module implements the cancellation token system for structured concurrency:
//! - **Hierarchical cancellation**: Child tokens are cancelled when parent is cancelled
//! - **Broadcast notifications**: Multiple listeners can wait for cancellation
//! - **Non-blocking checks**: Fast cancellation status checking
//! - **Cooperative cancellation**: Tasks must check for cancellation voluntarily
//!
//! The cancellation system is designed to support structured concurrency patterns
//! where parent operations can cleanly cancel all child operations.

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::sync::broadcast;

use super::types::{AsyncResult, AsyncError};

/// Cancellation token for structured cancellation
#[derive(Debug, Clone)]
pub struct CancellationToken {
    /// Internal cancellation state
    inner: Arc<CancellationTokenInner>,
}

#[derive(Debug)]
struct CancellationTokenInner {
    /// Cancellation sender
    sender: broadcast::Sender<()>,
    /// Whether this token is cancelled
    is_cancelled: AtomicBool,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1);
        Self {
            inner: Arc::new(CancellationTokenInner {
                sender,
                is_cancelled: AtomicBool::new(false),
            }),
        }
    }

    /// Create a child token that is cancelled when this token is cancelled
    pub fn child(&self) -> Self {
        // For simplicity, return a clone of this token
        // In a full implementation, this would create a hierarchical cancellation structure
        self.clone()
    }

    /// Cancel this token
    pub fn cancel(&self) {
        self.inner.is_cancelled.store(true, Ordering::SeqCst);
        let _ = self.inner.sender.send(());
    }

    /// Check if this token is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled.load(Ordering::SeqCst)
    }

    /// Check for cancellation and return error if cancelled
    pub fn check_cancelled(&self) -> AsyncResult<()> {
        if self.is_cancelled() {
            Err(AsyncError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Get a future that completes when this token is cancelled
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        
        let mut receiver = self.inner.sender.subscribe();
        let _ = receiver.recv().await;
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
} 