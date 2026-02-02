//! WebSocket protocol message types

use serde::{Deserialize, Serialize};

/// Messages from client to server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// List available dataframes
    ListDfs,
    /// Subscribe to a query
    Subscribe { name: String, query: String },
    /// Unsubscribe from a query
    Unsubscribe { name: String },
    /// One-off query
    Query { query: String },
}

/// Messages from server to client (text/JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// List of available dataframes
    Dfs { names: Vec<String> },
    /// Subscription confirmed
    Subscribed { name: String },
    /// Unsubscription confirmed
    Unsubscribed { name: String },
    /// Error response
    Error { message: String },
    /// Header for a result (followed by binary Arrow IPC)
    ResultHeader {
        name: String,
        tick: i64,
        size: usize,
    },
    /// Header for a one-off query result (followed by binary Arrow IPC)
    QueryResultHeader { size: usize },
}
