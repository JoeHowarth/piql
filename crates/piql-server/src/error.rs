//! Shared API error type for HTTP handlers.

use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use polars::prelude::PolarsError;

use crate::ipc::IpcEncodeError;
use crate::state::ErrorResponse;

/// Application error type surfaced by handlers.
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

impl From<IpcEncodeError> for AppError {
    fn from(e: IpcEncodeError) -> Self {
        AppError(e.to_string())
    }
}
