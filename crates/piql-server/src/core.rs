//! ServerCore - main public API for piql-server

use std::sync::Arc;

use polars::prelude::*;
use tokio::sync::broadcast;

use crate::state::{DfUpdate, SharedState};

/// Main server core providing DataFrame management and query execution
#[derive(Clone)]
pub struct ServerCore {
    state: Arc<SharedState>,
}

impl ServerCore {
    /// Create a new ServerCore
    pub fn new() -> Self {
        let (state, _) = SharedState::new();
        Self { state }
    }

    /// Create a new ServerCore with max rows limit
    pub fn with_max_rows(max_rows: Option<u32>) -> Self {
        let (state, _) = SharedState::with_max_rows(max_rows);
        Self { state }
    }

    /// Create a new ServerCore and return an update receiver
    pub fn with_update_receiver() -> (Self, broadcast::Receiver<()>) {
        let (state, rx) = SharedState::new();
        (Self { state }, rx)
    }

    /// Get the underlying shared state
    pub fn state(&self) -> Arc<SharedState> {
        self.state.clone()
    }

    /// Get a receiver for update notifications
    pub fn subscribe_updates(&self) -> broadcast::Receiver<()> {
        self.state.subscribe_updates()
    }

    /// Insert a DataFrame
    pub async fn insert_df(&self, name: impl Into<String>, df: DataFrame) {
        self.state.insert_df(name, df).await;
    }

    /// Remove a DataFrame
    pub async fn remove_df(&self, name: &str) {
        self.state.remove_df(name).await;
    }

    /// Apply a DfUpdate
    pub async fn apply_update(&self, update: DfUpdate) {
        self.state.apply_update(update).await;
    }

    /// List all DataFrame names
    pub async fn list_dataframes(&self) -> Vec<String> {
        self.state.list_dataframes().await
    }

    /// Execute a query and return collected DataFrame
    pub async fn execute_query(&self, query: &str) -> Result<DataFrame, piql::PiqlError> {
        self.state.execute_query(query).await
    }
}

impl Default for ServerCore {
    fn default() -> Self {
        Self::new()
    }
}
