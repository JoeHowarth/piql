//! piql-http: HTTP server for piql queries with file watching

mod files;
mod handlers;
mod llm;
mod state;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use piql::{DataFrameEntry, EvalContext};
use tokio::sync::RwLock;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::files::{collect_files, df_name_from_path, load_file, watch_files};
use crate::state::{AppState, DataframesResponse, ErrorResponse, SharedState};

// ============ CLI ============

#[derive(Parser)]
#[command(name = "piql-http")]
#[command(about = "HTTP server for piql queries with file watching")]
struct Args {
    /// Files or directories to watch (parquet, ipc, arrow, csv)
    #[arg(required = true)]
    paths: Vec<std::path::PathBuf>,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

// ============ OpenAPI ============

#[derive(OpenApi)]
#[openapi(
    paths(handlers::query, handlers::list_dataframes, handlers::ask),
    components(schemas(ErrorResponse, DataframesResponse))
)]
struct ApiDoc;

// ============ Router ============

fn create_router(state: SharedState) -> Router {
    Router::new()
        .route("/query", post(handlers::query))
        .route("/ask", post(handlers::ask))
        .route("/dataframes", get(handlers::list_dataframes))
        .merge(SwaggerUi::new("/swagger-ui").url("/openapi.json", ApiDoc::openapi()))
        .with_state(state)
}

// ============ Main ============

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Collect all files to load
    let files = collect_files(&args.paths);
    if files.is_empty() {
        log::error!("No supported files found in provided paths");
        std::process::exit(1);
    }

    // Build initial state
    let mut ctx = EvalContext::new();
    let mut paths = HashMap::new();

    for file in &files {
        let canonical = file.canonicalize()?;
        let df_name = df_name_from_path(&canonical);

        match load_file(&canonical) {
            Ok(df) => {
                ctx.dataframes.insert(
                    df_name.clone(),
                    DataFrameEntry {
                        df,
                        time_series: None,
                    },
                );
                paths.insert(canonical.clone(), df_name.clone());
                log::info!("Loaded: {} -> {}", canonical.display(), df_name);
            }
            Err(e) => {
                log::warn!("Failed to load {}: {}", canonical.display(), e);
            }
        }
    }

    if ctx.dataframes.is_empty() {
        log::error!("No dataframes loaded successfully");
        std::process::exit(1);
    }

    let state: SharedState = Arc::new(RwLock::new(AppState { ctx, paths }));

    // Spawn file watcher
    let watcher_state = state.clone();
    let watch_paths = args.paths.clone();
    tokio::spawn(async move {
        if let Err(e) = watch_files(watcher_state, watch_paths).await {
            log::error!("File watcher error: {}", e);
        }
    });

    // Start HTTP server
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    log::info!("Starting server at http://{}", addr);
    log::info!("Swagger UI: http://{}/swagger-ui", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, create_router(state)).await?;

    Ok(())
}
