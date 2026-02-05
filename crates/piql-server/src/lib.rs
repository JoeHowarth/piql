//! PiQL Server - DataFrame query server with REST and SSE support
//!
//! This crate provides a server for executing PiQL queries against DataFrames
//! with support for HTTP REST endpoints and SSE subscriptions.
//!
//! # Features
//!
//! - `llm` - Natural language to PiQL query generation
//! - `file-watcher` - Automatic DataFrame reloading on file changes
//! - `full` - All features enabled
//!
//! # Example
//!
//! ```ignore
//! use piql_server::{ServerCore, build_router};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let core = Arc::new(ServerCore::new());
//!
//!     // Load a DataFrame (already collected/materialized)
//!     let df = polars::prelude::df!("a" => [1, 2, 3]).unwrap();
//!     core.insert_df("test", df).await;
//!
//!     // Build and run the router
//!     let router = build_router(core);
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
//!     axum::serve(listener, router).await.unwrap();
//! }
//! ```

pub mod core;
pub mod http;
pub mod loader;
pub mod sse;
pub mod state;

#[cfg(feature = "llm")]
pub mod llm;

#[cfg(feature = "file-watcher")]
pub mod watcher;

// Re-exports for convenience
pub use core::ServerCore;
pub use state::{DfUpdate, SharedState};

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use utoipa::OpenApi;

/// OpenAPI documentation (base endpoints)
#[derive(OpenApi)]
#[openapi(
    paths(
        http::query,
        http::list_dataframes,
        sse::subscribe,
    ),
    components(schemas(
        state::DataframesResponse,
        state::ErrorResponse,
    ))
)]
struct ApiDocBase;

/// Build OpenAPI spec with feature-gated endpoints merged in
pub fn openapi_spec() -> utoipa::openapi::OpenApi {
    #[allow(unused_mut)]
    let mut doc = ApiDocBase::openapi();
    #[cfg(feature = "llm")]
    {
        use utoipa::OpenApi;
        let llm_doc = llm::LlmApiDoc::openapi();
        doc.paths.paths.extend(llm_doc.paths.paths);
    }
    doc
}

/// Build the axum router with all endpoints
pub fn build_router(core: Arc<ServerCore>) -> Router {
    #[allow(unused_mut)]
    let mut router = Router::new()
        .route("/query", post(http::query))
        .route("/dataframes", get(http::list_dataframes))
        .route("/subscribe", get(sse::subscribe));

    #[cfg(feature = "llm")]
    {
        router = router.route("/ask", post(llm::ask));
    }

    router.with_state(core)
}

/// Build the router with OpenAPI documentation endpoint
pub fn build_router_with_docs(core: Arc<ServerCore>) -> Router {
    use utoipa_swagger_ui::SwaggerUi;

    build_router(core).merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", openapi_spec()))
}
