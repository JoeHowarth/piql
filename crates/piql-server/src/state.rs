//! Server state with channel-based DataFrame updates

use std::sync::Arc;

use piql::{DataFrameEntry, EvalContext};
use polars::prelude::*;
use serde::Serialize;
use tokio::sync::{RwLock, broadcast};
use utoipa::ToSchema;

/// DataFrame update message
#[derive(Clone)]
pub enum DfUpdate {
    Insert { name: String, df: DataFrame },
    Remove { name: String },
    Reload { name: String, df: DataFrame },
}

/// Shared server state
pub struct SharedState {
    pub(crate) ctx: RwLock<EvalContext>,
    update_tx: broadcast::Sender<()>,
    /// Maximum rows to return from queries (None = unlimited)
    max_rows: Option<u32>,
}

impl SharedState {
    pub fn new() -> (Arc<Self>, broadcast::Receiver<()>) {
        Self::with_max_rows(None)
    }

    pub fn with_max_rows(max_rows: Option<u32>) -> (Arc<Self>, broadcast::Receiver<()>) {
        let (update_tx, update_rx) = broadcast::channel(16);
        let state = Arc::new(Self {
            ctx: RwLock::new(EvalContext::new()),
            update_tx,
            max_rows,
        });
        (state, update_rx)
    }

    /// Get a receiver for update notifications
    pub fn subscribe_updates(&self) -> broadcast::Receiver<()> {
        self.update_tx.subscribe()
    }

    /// Apply a DataFrame update
    pub async fn apply_update(&self, update: DfUpdate) {
        let mut ctx = self.ctx.write().await;
        match update {
            DfUpdate::Insert { name, df } => {
                ctx.dataframes.insert(
                    name,
                    DataFrameEntry {
                        df,
                        time_series: None,
                    },
                );
            }
            DfUpdate::Remove { name } => {
                ctx.dataframes.remove(&name);
            }
            DfUpdate::Reload { name, df } => {
                if let Some(entry) = ctx.dataframes.get_mut(&name) {
                    entry.df = df;
                } else {
                    ctx.dataframes.insert(
                        name,
                        DataFrameEntry {
                            df,
                            time_series: None,
                        },
                    );
                }
            }
        }
        drop(ctx);
        // Notify subscribers (ignore if no receivers)
        let _ = self.update_tx.send(());
    }

    /// Insert a DataFrame
    pub async fn insert_df(&self, name: impl Into<String>, df: DataFrame) {
        self.apply_update(DfUpdate::Insert {
            name: name.into(),
            df,
        })
        .await;
    }

    /// Remove a DataFrame
    pub async fn remove_df(&self, name: &str) {
        self.apply_update(DfUpdate::Remove {
            name: name.to_string(),
        })
        .await;
    }

    /// List all DataFrame names
    pub async fn list_dataframes(&self) -> Vec<String> {
        let ctx = self.ctx.read().await;
        let mut names: Vec<String> = ctx.dataframes.keys().cloned().collect();
        names.sort();
        names
    }

    /// Execute a query and collect results (runs on blocking thread pool)
    pub async fn execute_query(&self, query: &str) -> Result<DataFrame, piql::PiqlError> {
        let ctx = self.ctx.read().await.clone();
        let query = query.to_string();
        let max_rows = self.max_rows;

        tokio::task::spawn_blocking(move || {
            let result = piql::run(&query, &ctx)?;
            match result {
                piql::Value::DataFrame(lf, _) => {
                    let lf = if let Some(limit) = max_rows {
                        lf.limit(limit)
                    } else {
                        lf
                    };
                    lf.collect()
                        .map_err(piql::EvalError::from)
                        .map_err(piql::PiqlError::from)
                }
                _ => Err(piql::PiqlError::Eval(piql::EvalError::TypeError {
                    expected: "DataFrame".to_string(),
                    got: "other value".to_string(),
                })),
            }
        })
        .await
        .map_err(|e| piql::PiqlError::Eval(piql::EvalError::Other(format!("task failed: {e}"))))?
    }
}

// ============ API Types ============

#[derive(Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize, ToSchema)]
pub struct DataframesResponse {
    pub names: Vec<String>,
}
