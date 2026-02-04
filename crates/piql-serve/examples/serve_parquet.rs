//! Load parquet files from a directory and serve them via WebSocket.
//!
//! Usage: cargo run --example serve_parquet -- <directory> [port]
//!
//! Example:
//!   cargo run --example serve_parquet -- ./data 9000

use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};

use piql_serve::io::load_parquet;
use piql_serve::{PiqlServer, QueryEngine};

fn load_parquet_dir(engine: &mut QueryEngine, dir: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let mut count = 0;

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "parquet") {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("Invalid filename")?;

            let df = load_parquet(&path)?;
            engine.add_base_df(name, df);
            println!("Loaded: {} <- {}", name, path.display());
            count += 1;
        }
    }

    Ok(count)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <directory> [port]", args[0]);
        eprintln!("  directory  Path to directory containing .parquet files");
        eprintln!("  port       Port to listen on (default: 9000)");
        std::process::exit(1);
    }

    let dir = Path::new(&args[1]);
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9000);

    if !dir.is_dir() {
        eprintln!("Error: {} is not a directory", dir.display());
        std::process::exit(1);
    }

    let mut engine = QueryEngine::new();
    let count = load_parquet_dir(&mut engine, dir)?;

    if count == 0 {
        eprintln!("Warning: No .parquet files found in {}", dir.display());
    }

    let engine = Arc::new(Mutex::new(engine));
    let server = PiqlServer::new(engine);

    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    println!("Listening on ws://{}", addr);

    server.listen(addr).await?;

    Ok(())
}
