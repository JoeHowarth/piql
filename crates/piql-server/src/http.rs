//! HTTP REST handlers

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use log::{debug, info, warn};
use polars::prelude::*;

use crate::core::ServerCore;
use crate::state::{DataframesResponse, ErrorResponse};

/// Application error type
pub struct AppError(pub String);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: self.0 }),
        )
            .into_response()
    }
}

impl From<piql::PiqlError> for AppError {
    fn from(e: piql::PiqlError) -> Self {
        AppError(e.to_string())
    }
}

impl From<PolarsError> for AppError {
    fn from(e: PolarsError) -> Self {
        AppError(e.to_string())
    }
}

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

    let mut df = match core.execute_query(&body).await {
        Ok(df) => df,
        Err(e) => {
            warn!("Query failed in {:.2?}: {}", start.elapsed(), e);
            return Err(e.into());
        }
    };

    // Serialize to Arrow IPC (blocking but fast for in-memory data)
    let buf = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, PolarsError> {
        let mut buf = Vec::new();
        IpcStreamWriter::new(&mut buf).finish(&mut df)?;
        Ok(buf)
    })
    .await
    .map_err(|e| AppError(format!("Task join error: {}", e)))?
    .map_err(AppError::from)?;

    info!("Query succeeded in {:.2?}, {} bytes", start.elapsed(), buf.len());
    Ok((
        [(
            header::CONTENT_TYPE,
            "application/vnd.apache.arrow.stream",
        )],
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
