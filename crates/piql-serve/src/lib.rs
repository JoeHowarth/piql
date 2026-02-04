//! PiQL Server - Networking, file I/O, and utilities for PiQL
//!
//! This crate provides:
//! - WebSocket server for query subscriptions
//! - File I/O for loading/saving DataFrames
//! - Utilities for simulation integration

pub mod io;
pub mod protocol;
pub mod server;

pub use protocol::{
    decode_binary_result, encode_binary_result, ClientMessage, ProtocolError, ServerMessage,
};
pub use server::{PiqlServer, ServerError};

pub use piql::{QueryEngine, TimeSeriesConfig};
