//! PiQL Server - Networking, file I/O, and utilities for PiQL
//!
//! This crate provides:
//! - WebSocket server for query subscriptions
//! - File I/O for loading/saving DataFrames
//! - Utilities for simulation integration

pub mod io;
pub mod protocol;
pub mod server;
pub mod util;

pub use protocol::{ClientMessage, ServerMessage};
pub use server::{PiqlServer, ServerError};

pub use piql::{QueryEngine, TimeSeriesConfig};
