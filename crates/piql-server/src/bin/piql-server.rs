//! PiQL Server CLI
//!
//! A thin wrapper around the piql-server library.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

#[derive(Parser)]
#[command(name = "piql-server")]
#[command(about = "PiQL DataFrame query server")]
#[command(after_help = "\
EXAMPLES:
    # Serve files from a directory
    piql-server ./data/

    # Serve with concat mode (combine chunked data)
    piql-server --concat ~/dfs/

    # Given:  ~/dfs/50_000_000/tx.parquet
    #         ~/dfs/50_100_000/tx.parquet
    # Creates: 'tx' DataFrame with all chunks concatenated

    # Multiple directories
    piql-server --concat ~/dfs/chain1/ ~/dfs/chain2/

    # Run-aware mode (watches for new run subdirectories)
    piql-server --runs ./data/

    # Each subdir with a _ready sentinel is a run:
    #   data/0206_1430_basic/fill.parquet  → fill, _0206_1430_basic::fill, _all::fill
")]
struct Args {
    /// Paths to parquet/csv/ipc files or directories
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Recursively scan directories and concatenate files with the same name.
    /// Use this when data is split across multiple chunk directories.
    #[arg(long, conflicts_with = "runs")]
    concat: bool,

    /// Run-aware mode: watch a parent directory for run subdirectories.
    /// Each subdirectory with a _ready sentinel file is loaded as a run.
    /// Tables are exposed as: table (latest), run::table, _all::table.
    #[arg(long, conflicts_with = "concat")]
    runs: bool,

    /// In --runs mode, automatically drop an existing _run column before labeling _all:: tables.
    /// By default, loading fails if a source table already contains _run.
    #[arg(long, requires = "runs")]
    runs_drop_existing_run_col: bool,

    /// Maximum rows to return from queries. Default 100000. Use 0 for unlimited.
    #[arg(long, default_value = "100000")]
    max_rows: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    let max_rows = if args.max_rows == 0 {
        None
    } else {
        Some(args.max_rows)
    };
    let core = Arc::new(piql_server::ServerCore::with_max_rows(max_rows));
    log::info!(
        "Max rows per query: {}",
        max_rows.map_or("unlimited".to_string(), |n| n.to_string())
    );

    if args.runs {
        // Run-aware mode: watch parent dir for run subdirectories
        #[cfg(feature = "file-watcher")]
        {
            let parent = &args.paths[0];
            log::info!("Starting in run-aware mode, watching: {}", parent.display());
            let _watcher = piql_server::watcher::load_and_watch_runs(
                core.clone(),
                parent.clone(),
                piql_server::watcher::RunModeOptions {
                    drop_existing_run_label_column: args.runs_drop_existing_run_col,
                },
            )
            .await?;
        }

        #[cfg(not(feature = "file-watcher"))]
        {
            anyhow::bail!("--runs requires the file-watcher feature");
        }
    } else if args.concat {
        // Concat mode: recursively scan and concatenate files with same name
        for path in &args.paths {
            match piql_server::loader::load_concat_dir(path).await {
                Ok(dfs) => {
                    for (name, df) in dfs {
                        log::info!("Loaded concatenated df: {}", name);
                        core.insert_df(name, df).await;
                    }
                }
                Err(e) => {
                    log::error!("Failed to load concat dir {}: {}", path.display(), e);
                }
            }
        }
    } else {
        // Normal mode: load files and optionally start watching
        #[cfg(feature = "file-watcher")]
        let _watcher = piql_server::watcher::load_and_watch(core.clone(), args.paths).await?;

        #[cfg(not(feature = "file-watcher"))]
        {
            // Just load files once without watching
            let files = piql_server::loader::collect_files(&args.paths);
            for path in files {
                if let Ok(df) = piql_server::loader::load_file(&path).await {
                    let name = piql_server::loader::df_name_from_path(&path);
                    core.insert_df(name, df).await;
                }
            }
        }
    }

    let router = piql_server::build_router_with_docs(core);

    let addr = format!("{}:{}", args.host, args.port);
    println!("Starting server on {}", addr);
    println!("  POST /query - Execute PiQL query");
    println!("  GET  /dataframes - List available DataFrames");
    println!("  GET  /subscribe?query=<query> - SSE subscription");
    #[cfg(feature = "llm")]
    println!("  POST /ask - Natural language query");
    println!("  GET  /swagger-ui - API documentation");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
