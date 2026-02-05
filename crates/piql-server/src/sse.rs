//! SSE subscription handler

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use base64::Engine;
use futures::stream::{self, Stream, StreamExt};
use log::{debug, info, warn};
use polars::prelude::*;
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use utoipa::IntoParams;

use crate::core::ServerCore;

#[derive(Deserialize, IntoParams)]
pub struct SubscribeParams {
    /// PiQL query to subscribe to
    pub query: String,
}

/// Subscribe to query results via SSE
///
/// Returns a stream of events. Each event contains base64-encoded Arrow IPC data.
/// Events are emitted:
/// - Immediately with initial results
/// - Whenever any DataFrame is updated
#[utoipa::path(
    get,
    path = "/subscribe",
    params(SubscribeParams),
    responses(
        (status = 200, description = "SSE stream of query results"),
        (status = 400, description = "Error")
    )
)]
pub async fn subscribe(
    State(core): State<Arc<ServerCore>>,
    Query(params): Query<SubscribeParams>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let query = params.query;
    info!("GET /subscribe: {}", query);
    let update_rx = core.subscribe_updates();

    // Create a stream that emits on updates
    let update_stream = BroadcastStream::new(update_rx)
        .filter_map(|_| async { Some(()) });

    // Prepend an immediate trigger to emit initial results
    let trigger_stream = stream::once(async { () }).chain(update_stream);

    // For each trigger, execute the query and emit results
    let query_for_log = query.clone();
    let event_stream = trigger_stream.then(move |_| {
        let core = core.clone();
        let query = query.clone();
        async move {
            match execute_and_encode(&core, &query).await {
                Ok(data) => {
                    debug!("SSE result: {} bytes", data.len());
                    Event::default()
                        .event("result")
                        .data(data)
                }
                Err(e) => {
                    warn!("SSE error: {}", e);
                    Event::default()
                        .event("error")
                        .data(e)
                }
            }
        }
    });

    debug!("SSE subscription started for: {}", query_for_log);
    Sse::new(event_stream.map(Ok))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(30)))
}

/// Execute query and encode result as base64 Arrow IPC
async fn execute_and_encode(core: &ServerCore, query: &str) -> Result<String, String> {
    let mut df = core.execute_query(query).await.map_err(|e| e.to_string())?;

    let buf = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, PolarsError> {
        let mut buf = Vec::new();
        IpcStreamWriter::new(&mut buf).finish(&mut df)?;
        Ok(buf)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?
    .map_err(|e| e.to_string())?;

    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}
