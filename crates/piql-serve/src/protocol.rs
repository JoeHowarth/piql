//! WebSocket protocol message types

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Message too short: need at least {0} bytes")]
    TooShort(usize),
    #[error("Invalid header JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
}

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

/// Encode a header and Arrow IPC payload into a single binary message.
///
/// Format: [4-byte header length (big-endian u32)][JSON header][Arrow IPC payload]
pub fn encode_binary_result(header: &ServerMessage, payload: &[u8]) -> Vec<u8> {
    let header_json = serde_json::to_vec(header).expect("ServerMessage serialization cannot fail");
    let header_len = header_json.len() as u32;

    let mut buf = Vec::with_capacity(4 + header_json.len() + payload.len());
    buf.extend_from_slice(&header_len.to_be_bytes());
    buf.extend_from_slice(&header_json);
    buf.extend_from_slice(payload);
    buf
}

/// Decode a binary message into header and Arrow IPC payload.
///
/// Returns the parsed ServerMessage and a slice of the payload bytes.
pub fn decode_binary_result(data: &[u8]) -> Result<(ServerMessage, &[u8]), ProtocolError> {
    if data.len() < 4 {
        return Err(ProtocolError::TooShort(4));
    }

    let header_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let header_end = 4 + header_len;

    if data.len() < header_end {
        return Err(ProtocolError::TooShort(header_end));
    }

    let header: ServerMessage = serde_json::from_slice(&data[4..header_end])?;
    let payload = &data[header_end..];

    Ok((header, payload))
}
