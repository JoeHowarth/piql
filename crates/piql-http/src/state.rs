//! Application state and error types

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use piql::EvalContext;
use polars::prelude::PolarsError;
use serde::Serialize;
use tokio::sync::RwLock;
use utoipa::ToSchema;

// ============ State ============

pub struct AppState {
    pub ctx: EvalContext,
    /// Maps canonical file paths to their df names for reload tracking
    pub paths: HashMap<PathBuf, String>,
}

pub type SharedState = Arc<RwLock<AppState>>;

// ============ API Types ============

#[derive(Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize, ToSchema)]
pub struct DataframesResponse {
    pub names: Vec<String>,
}

// ============ Error Handling ============

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
