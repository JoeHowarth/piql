//! HTTP request handlers

use axum::extract::{Query, State};
use axum::http::header;
use axum::response::IntoResponse;
use axum::Json;
use piql::Value;
use polars::prelude::*;
use serde::Deserialize;
use utoipa::IntoParams;

use crate::llm::{build_system_prompt, generate_query, get_schema_and_examples};
use crate::state::{AppError, DataframesResponse, SharedState};

// ============ HTTP Handlers ============

#[utoipa::path(
    post,
    path = "/query",
    request_body(content = String, content_type = "text/plain", description = "PiQL query string"),
    responses(
        (status = 200, description = "Arrow IPC stream", content_type = "application/vnd.apache.arrow.stream"),
        (status = 400, description = "Query error", body = crate::state::ErrorResponse)
    )
)]
pub async fn query(State(state): State<SharedState>, body: String) -> Result<impl IntoResponse, AppError> {
    let state = state.read().await;
    let result = piql::run(&body, &state.ctx)?;

    let lf = match result {
        Value::DataFrame(lf, _) => lf,
        _ => {
            return Err(AppError(
                "Query did not return a DataFrame".to_string(),
            ))
        }
    };

    // Run collect in a blocking task to avoid nested runtime issues
    let buf = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, PolarsError> {
        let mut df = lf.collect()?;
        let mut buf = Vec::new();
        IpcStreamWriter::new(&mut buf).finish(&mut df)?;
        Ok(buf)
    })
    .await
    .map_err(|e| AppError(format!("Task join error: {}", e)))?
    .map_err(AppError::from)?;

    Ok((
        [(
            header::CONTENT_TYPE,
            "application/vnd.apache.arrow.stream",
        )],
        buf,
    ))
}

#[utoipa::path(
    get,
    path = "/dataframes",
    responses(
        (status = 200, description = "List of available dataframe names", body = crate::state::DataframesResponse)
    )
)]
pub async fn list_dataframes(State(state): State<SharedState>) -> Json<DataframesResponse> {
    let state = state.read().await;
    let mut names: Vec<String> = state.ctx.dataframes.keys().cloned().collect();
    names.sort();
    Json(DataframesResponse { names })
}

#[derive(Deserialize, IntoParams)]
pub struct AskParams {
    /// Execute the generated query and return results
    #[serde(default)]
    pub execute: bool,
}

#[utoipa::path(
    post,
    path = "/ask",
    request_body(content = String, content_type = "text/plain", description = "Natural language question"),
    params(AskParams),
    responses(
        (status = 200, description = "Generated query (in X-Piql-Query header) and optionally results"),
        (status = 400, description = "Error", body = crate::state::ErrorResponse)
    )
)]
pub async fn ask(
    State(state): State<SharedState>,
    Query(params): Query<AskParams>,
    body: String,
) -> Result<impl IntoResponse, AppError> {
    use axum::http::{HeaderName, HeaderValue};

    // Get schema info, samples, and generated examples for the prompt
    let state_guard = state.read().await;
    let (schema_info, examples) = get_schema_and_examples(&state_guard.ctx).await;
    drop(state_guard); // Release lock before async LLM call

    let system_prompt = build_system_prompt(&schema_info, &examples);
    let query = generate_query(&body, &system_prompt).await?;

    let response_body = if params.execute {
        // Execute and return Arrow IPC
        let state_guard = state.read().await;
        let result = piql::run(&query, &state_guard.ctx)?;
        let lf = match result {
            Value::DataFrame(lf, _) => lf,
            _ => return Err(AppError("Query did not return DataFrame".into())),
        };

        tokio::task::spawn_blocking(move || -> Result<Vec<u8>, PolarsError> {
            let mut df = lf.collect()?;
            let mut buf = Vec::new();
            IpcStreamWriter::new(&mut buf).finish(&mut df)?;
            Ok(buf)
        })
        .await
        .map_err(|e| AppError(format!("Task join error: {}", e)))?
        .map_err(AppError::from)?
    } else {
        // Empty body
        Vec::new()
    };

    Ok((
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/vnd.apache.arrow.stream"),
            ),
            (
                HeaderName::from_static("x-piql-query"),
                HeaderValue::from_str(&query).unwrap_or_else(|_| HeaderValue::from_static("")),
            ),
        ],
        response_body,
    ))
}
