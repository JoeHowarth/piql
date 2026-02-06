//! HTTP REST handlers

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use log::{debug, info, warn};

use crate::core::ServerCore;
use crate::error::AppError;
use crate::ipc::dataframe_to_ipc_bytes;
use crate::state::{DataframesResponse, ErrorResponse};

/// Execute a piql query
#[utoipa::path(
    post,
    path = "/query",
    request_body(content = String, content_type = "text/plain", description = "PiQL query string"),
    responses(
        (status = 200, description = "Arrow IPC stream", content_type = "application/vnd.apache.arrow.stream"),
        (status = 400, description = "Query error", body = ErrorResponse)
    )
)]
pub async fn query(
    State(core): State<Arc<ServerCore>>,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    let start = Instant::now();
    info!("POST /query: {}", body.lines().next().unwrap_or(&body));
    debug!("Full query: {}", body);

    let df = match core.execute_query(&body).await {
        Ok(df) => df,
        Err(e) => {
            warn!("Query failed in {:.2?}: {}", start.elapsed(), e);
            return Err(e.into());
        }
    };

    let buf = dataframe_to_ipc_bytes(df).await?;

    info!(
        "Query succeeded in {:.2?}, {} bytes",
        start.elapsed(),
        buf.len()
    );
    Ok((
        [(header::CONTENT_TYPE, "application/vnd.apache.arrow.stream")],
        buf,
    ))
}

/// List available DataFrames
#[utoipa::path(
    get,
    path = "/dataframes",
    responses(
        (status = 200, description = "List of available dataframe names", body = DataframesResponse)
    )
)]
pub async fn list_dataframes(State(core): State<Arc<ServerCore>>) -> Json<DataframesResponse> {
    info!("GET /dataframes");
    let names = core.list_dataframes().await;
    debug!("Available dataframes: {:?}", names);
    Json(DataframesResponse { names })
}
