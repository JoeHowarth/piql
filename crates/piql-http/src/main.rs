//! piql-http: HTTP server for piql queries with file watching

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars::prelude::PlPath;

use axum::extract::{Query, State};
use axum::http::{header, HeaderName, HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use piql::{DataFrameEntry, EvalContext, Value};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use utoipa::{IntoParams, OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

// ============ CLI ============

#[derive(Parser)]
#[command(name = "piql-http")]
#[command(about = "HTTP server for piql queries with file watching")]
struct Args {
    /// Files or directories to watch (parquet, ipc, arrow, csv)
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

// ============ State ============

struct AppState {
    ctx: EvalContext,
    /// Maps canonical file paths to their df names for reload tracking
    paths: HashMap<PathBuf, String>,
}

type SharedState = Arc<RwLock<AppState>>;

// ============ File Loading ============

fn load_file(path: &Path) -> Result<LazyFrame, PolarsError> {
    let pl_path = PlPath::Local(Arc::from(path));
    match path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => LazyFrame::scan_parquet(pl_path, Default::default()),
        Some("csv") => LazyCsvReader::new(pl_path).finish(),
        Some("ipc" | "arrow") => LazyFrame::scan_ipc(pl_path, Default::default(), Default::default()),
        Some(ext) => Err(PolarsError::ComputeError(
            format!("unsupported file extension: {ext}").into(),
        )),
        None => Err(PolarsError::ComputeError(
            "file has no extension".to_string().into(),
        )),
    }
}

fn df_name_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn is_supported_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("parquet" | "csv" | "ipc" | "arrow")
    )
}

/// Collect all supported files from paths (files or directories)
fn collect_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() && is_supported_file(&p) {
                        files.push(p);
                    }
                }
            }
        } else if path.is_file() && is_supported_file(path) {
            files.push(path.clone());
        }
    }
    files
}

// ============ File Watcher ============

async fn watch_files(
    state: SharedState,
    paths: Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (tx, mut rx) = mpsc::channel::<Event>(100);

    let mut watcher = RecommendedWatcher::new(
        move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                let _ = tx.blocking_send(event);
            }
        },
        notify::Config::default(),
    )?;

    // Watch directories containing our files
    let mut watched_dirs = std::collections::HashSet::new();
    for path in &paths {
        let dir = if path.is_dir() {
            path.clone()
        } else {
            path.parent().map(|p| p.to_path_buf()).unwrap_or_default()
        };
        if watched_dirs.insert(dir.clone()) {
            watcher.watch(&dir, RecursiveMode::NonRecursive)?;
            log::info!("Watching directory: {}", dir.display());
        }
    }

    // Keep watcher alive
    let _watcher = watcher;

    while let Some(event) = rx.recv().await {
        if matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        ) {
            for path in event.paths {
                if !is_supported_file(&path) {
                    continue;
                }

                let canonical = match path.canonicalize() {
                    Ok(p) => p,
                    Err(_) => path.clone(),
                };

                let mut state = state.write().await;
                if let Some(df_name) = state.paths.get(&canonical).cloned() {
                    // Reload existing file
                    match load_file(&canonical) {
                        Ok(df) => {
                            state.ctx.dataframes.insert(
                                df_name.clone(),
                                DataFrameEntry {
                                    df,
                                    time_series: None,
                                },
                            );
                            log::info!("Reloaded: {} -> {}", canonical.display(), df_name);
                        }
                        Err(e) => {
                            log::warn!("Failed to reload {}: {}", canonical.display(), e);
                        }
                    }
                } else if event.kind.is_create() {
                    // New file in watched directory
                    let df_name = df_name_from_path(&canonical);
                    match load_file(&canonical) {
                        Ok(df) => {
                            state.ctx.dataframes.insert(
                                df_name.clone(),
                                DataFrameEntry {
                                    df,
                                    time_series: None,
                                },
                            );
                            state.paths.insert(canonical.clone(), df_name.clone());
                            log::info!("Loaded new file: {} -> {}", canonical.display(), df_name);
                        }
                        Err(e) => {
                            log::warn!("Failed to load new file {}: {}", canonical.display(), e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ============ API Types ============

#[derive(Serialize, ToSchema)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize, ToSchema)]
struct DataframesResponse {
    names: Vec<String>,
}

// ============ Error Handling ============

struct AppError(String);

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

// ============ HTTP Handlers ============

#[utoipa::path(
    post,
    path = "/query",
    request_body(content = String, content_type = "text/plain", description = "PiQL query string"),
    responses(
        (status = 200, description = "Arrow IPC stream", content_type = "application/vnd.apache.arrow.stream"),
        (status = 400, description = "Query error", body = ErrorResponse)
    )
)]
async fn query(State(state): State<SharedState>, body: String) -> Result<impl IntoResponse, AppError> {
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
        (status = 200, description = "List of available dataframe names", body = DataframesResponse)
    )
)]
async fn list_dataframes(State(state): State<SharedState>) -> Json<DataframesResponse> {
    let state = state.read().await;
    let mut names: Vec<String> = state.ctx.dataframes.keys().cloned().collect();
    names.sort();
    Json(DataframesResponse { names })
}

// ============ Natural Language to PiQL ============

const PIQL_DOCS: &str = r#"PiQL is a text query language for Polars dataframes. Write queries that look like Python Polars.

## Supported Features

**DataFrame methods**
`filter`, `select`, `with_columns`, `head`, `tail`, `sort`, `drop`, `explode`, `group_by`, `join`, `rename`, `drop_nulls`, `reverse`, `top`

**Expr methods**
`alias`, `over`, `is_between`, `diff`, `shift`, `sum`, `mean`, `min`, `max`, `count`, `first`, `last`, `cast`, `fill_null`, `is_null`, `is_not_null`, `unique`, `abs`, `round`, `len`, `n_unique`, `cum_sum`, `cum_max`, `cum_min`, `rank`, `clip`, `reverse`

**pl functions**
`col`, `lit`, `when`/`then`/`otherwise`

**str namespace**
`starts_with`, `ends_with`, `to_lowercase`, `to_uppercase`, `len_chars`, `contains`, `replace`, `slice`

**dt namespace**
`year`, `month`, `day`, `hour`, `minute`, `second`

**Operators**
`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`, `~`

**Sugar**
- `$col` → `pl.col("col")`
"#;

/// Example query templates - {table}, {str_col}, {num_col}, {cat_col} are placeholders
const EXAMPLE_TEMPLATES: &[(&str, &str)] = &[
    ("Filter numeric", "{table}.filter(pl.col(\"{num_col}\") > {num_val})"),
    ("Filter string equality", "{table}.filter(pl.col(\"{cat_col}\") == \"{cat_val}\")"),
    ("Select columns", "{table}.select(pl.col(\"{str_col}\"), pl.col(\"{num_col}\"))"),
    ("Sort descending", "{table}.sort(\"{num_col}\", descending=True)"),
    ("Head", "{table}.head(5)"),
    ("Group by + count", "{table}.group_by(\"{cat_col}\").agg(pl.col(\"{str_col}\").count().alias(\"count\"))"),
    ("String contains", "{table}.filter(pl.col(\"{str_col}\").str.contains(\"{str_fragment}\"))"),
];

/// Column info extracted from a dataframe
struct ColumnInfo {
    str_cols: Vec<String>,
    num_cols: Vec<String>,
    cat_col: Option<String>,  // A string column good for grouping (few unique values)
    sample_str: Option<String>,
    sample_num: Option<i64>,
    sample_cat: Option<String>,
}

/// Get schema info, sample data, and generate examples from actual data
async fn get_schema_and_examples(ctx: &EvalContext) -> (String, String) {
    // Clone dataframes for blocking task
    let dfs: Vec<(String, LazyFrame)> = ctx
        .dataframes
        .iter()
        .map(|(name, entry)| (name.clone(), entry.df.clone()))
        .collect();

    // Collect samples and column info in blocking task
    let results: Vec<(String, String, ColumnInfo)> = tokio::task::spawn_blocking(move || {
        dfs.into_iter()
            .filter_map(|(name, mut lf)| {
                let df = lf.clone().slice(0, 5).collect().ok()?;
                let sample_str = format!("{}", df);

                // Analyze columns
                let schema = lf.collect_schema().ok()?;
                let mut str_cols = Vec::new();
                let mut num_cols = Vec::new();

                for (col_name, dtype) in schema.iter() {
                    match dtype {
                        DataType::String => str_cols.push(col_name.to_string()),
                        DataType::Int64 | DataType::Int32 | DataType::Float64 => {
                            num_cols.push(col_name.to_string())
                        }
                        _ => {}
                    }
                }

                // Try to find a good categorical column and sample values
                let mut cat_col = None;
                let mut sample_cat = None;
                let mut sample_str_val = None;
                let mut sample_num = None;

                // Get sample values from first row
                if df.height() > 0 {
                    for col in &str_cols {
                        if let Ok(series) = df.column(col) {
                            if let Ok(val) = series.str() {
                                if let Some(s) = val.get(0) {
                                    if sample_str_val.is_none() {
                                        sample_str_val = Some(s.to_string());
                                    }
                                    // Check if this could be a categorical column (short values)
                                    if s.len() < 20 && cat_col.is_none() {
                                        cat_col = Some(col.clone());
                                        sample_cat = Some(s.to_string());
                                    }
                                }
                            }
                        }
                    }
                    for col in &num_cols {
                        if let Ok(series) = df.column(col) {
                            if let Ok(val) = series.i64() {
                                sample_num = val.get(0);
                                break;
                            }
                        }
                    }
                }

                Some((name, sample_str, ColumnInfo {
                    str_cols,
                    num_cols,
                    cat_col,
                    sample_str: sample_str_val,
                    sample_num,
                    sample_cat,
                }))
            })
            .collect()
    })
    .await
    .unwrap_or_default();

    // Build schema info
    let mut schema_info = String::new();
    for (name, sample, _) in &results {
        schema_info.push_str(&format!("## {}\n{}\n\n", name, sample));
    }

    // Generate examples from first suitable table
    let mut examples = String::new();
    if let Some((table, _, info)) = results.iter().find(|(_, _, i)| !i.str_cols.is_empty() && !i.num_cols.is_empty()) {
        let str_col = info.str_cols.first().map(|s| s.as_str()).unwrap_or("name");
        let num_col = info.num_cols.first().map(|s| s.as_str()).unwrap_or("id");
        let cat_col = info.cat_col.as_deref().unwrap_or(str_col);
        let num_val = info.sample_num.unwrap_or(100);
        let cat_val = info.sample_cat.as_deref().unwrap_or("value");
        let str_fragment = info.sample_str.as_deref()
            .and_then(|s| s.get(0..3))
            .unwrap_or("abc");

        for (desc, template) in EXAMPLE_TEMPLATES {
            let query = template
                .replace("{table}", table)
                .replace("{str_col}", str_col)
                .replace("{num_col}", num_col)
                .replace("{cat_col}", cat_col)
                .replace("{num_val}", &num_val.to_string())
                .replace("{cat_val}", cat_val)
                .replace("{str_fragment}", str_fragment);
            examples.push_str(&format!("# {}\n{}\n\n", desc, query));
        }
    }

    (schema_info, examples)
}

/// Build the system prompt with piql docs, examples, and schema
fn build_system_prompt(schema_info: &str, examples: &str) -> String {
    format!(
        r#"You are a PiQL query generator. Given a natural language question about data, respond with ONLY a valid PiQL query string.

<piql_description>
{}
</piql_description>

<examples>
{}
</examples>

<available_dataframes>
{}
</available_dataframes>

IMPORTANT:
- Respond with ONLY the PiQL query string
- Do NOT include any explanation, markdown formatting, or code blocks
- Do NOT wrap the query in quotes or backticks
- Just output the raw query that can be executed directly"#,
        PIQL_DOCS, examples, schema_info
    )
}

/// Call LLM to generate query
async fn generate_query(prompt: &str, system: &str) -> Result<String, AppError> {
    if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
        call_openrouter(&api_key, prompt, system).await
    } else {
        call_claude_cli(prompt, system).await
    }
}

async fn call_openrouter(api_key: &str, prompt: &str, system: &str) -> Result<String, AppError> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "model": "anthropic/claude-sonnet-4",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        }))
        .send()
        .await
        .map_err(|e| AppError(format!("OpenRouter request failed: {}", e)))?;

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AppError(format!("Failed to parse OpenRouter response: {}", e)))?;

    let query = json["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AppError("No response content from LLM".into()))?
        .trim()
        .to_string();

    Ok(query)
}

async fn call_claude_cli(prompt: &str, system: &str) -> Result<String, AppError> {
    let full_prompt = format!("{}\n\nUser question: {}", system, prompt);
    let output = tokio::process::Command::new("claude")
        .args(["-p", &full_prompt])
        .output()
        .await
        .map_err(|e| AppError(format!("Failed to run claude CLI: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError(format!("claude CLI failed: {}", stderr)));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[derive(Deserialize, IntoParams)]
struct AskParams {
    /// Execute the generated query and return results
    #[serde(default)]
    execute: bool,
}

#[utoipa::path(
    post,
    path = "/ask",
    request_body(content = String, content_type = "text/plain", description = "Natural language question"),
    params(AskParams),
    responses(
        (status = 200, description = "Generated query (in X-Piql-Query header) and optionally results"),
        (status = 400, description = "Error", body = ErrorResponse)
    )
)]
async fn ask(
    State(state): State<SharedState>,
    Query(params): Query<AskParams>,
    body: String,
) -> Result<impl IntoResponse, AppError> {
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

// ============ OpenAPI ============

#[derive(OpenApi)]
#[openapi(
    paths(query, list_dataframes, ask),
    components(schemas(ErrorResponse, DataframesResponse))
)]
struct ApiDoc;

// ============ Router ============

fn create_router(state: SharedState) -> Router {
    Router::new()
        .route("/query", post(query))
        .route("/ask", post(ask))
        .route("/dataframes", get(list_dataframes))
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
