//! File loading utilities

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars::prelude::*;

/// Load a DataFrame from a file path (sync, collects immediately)
pub fn load_file_sync(path: &Path) -> Result<DataFrame, PolarsError> {
    let pl_path = PlPath::Local(Arc::from(path));
    let lf = match path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => LazyFrame::scan_parquet(pl_path, Default::default())?,
        Some("csv") => LazyCsvReader::new(pl_path).finish()?,
        Some("ipc" | "arrow") => LazyFrame::scan_ipc(pl_path, Default::default(), Default::default())?,
        Some(ext) => return Err(PolarsError::ComputeError(
            format!("unsupported file extension: {ext}").into(),
        )),
        None => return Err(PolarsError::ComputeError(
            "file has no extension".to_string().into(),
        )),
    };
    lf.collect()
}

/// Load a DataFrame from a file path (async, runs on blocking thread pool)
pub async fn load_file(path: &Path) -> Result<DataFrame, PolarsError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || load_file_sync(&path))
        .await
        .map_err(|e| PolarsError::ComputeError(format!("blocking task failed: {e}").into()))?
}

/// Extract DataFrame name from path (file stem)
pub fn df_name_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Check if a file has a supported extension
pub fn is_supported_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("parquet" | "csv" | "ipc" | "arrow")
    )
}

/// Collect all supported files from paths (files or directories)
pub fn collect_files(paths: &[PathBuf]) -> Vec<PathBuf> {
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

/// Recursively collect all supported files from a directory tree
fn collect_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_files_recursive(&path));
            } else if path.is_file() && is_supported_file(&path) {
                files.push(path);
            }
        }
    }

    files
}

/// Load a directory tree, concatenating all files with the same name (sync).
fn load_concat_dir_sync(dir: &Path) -> Result<HashMap<String, DataFrame>, PolarsError> {
    let files = collect_files_recursive(dir);

    // Group files by name (stem)
    let mut by_name: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for path in files {
        let name = df_name_from_path(&path);
        by_name.entry(name).or_default().push(path);
    }

    // Concat each group
    let mut result = HashMap::new();
    for (name, paths) in by_name {
        if paths.is_empty() {
            continue;
        }

        log::info!(
            "Loading {} ({} files)",
            name,
            paths.len()
        );

        // Load all files as DataFrames
        let frames: Vec<DataFrame> = paths
            .iter()
            .filter_map(|p| {
                match load_file_sync(p) {
                    Ok(df) => Some(df),
                    Err(e) => {
                        log::warn!("Failed to load {}: {}", p.display(), e);
                        None
                    }
                }
            })
            .collect();

        if frames.is_empty() {
            log::warn!("No valid files for {}", name);
            continue;
        }

        // Concat all frames
        let combined = if frames.len() == 1 {
            frames.into_iter().next().unwrap()
        } else {
            // Convert to lazy, concat, collect
            let lazy_frames: Vec<LazyFrame> = frames.into_iter().map(|df| df.lazy()).collect();
            concat(&lazy_frames, UnionArgs::default())?.collect()?
        };

        result.insert(name, combined);
    }

    Ok(result)
}

/// Load a directory tree, concatenating all files with the same name.
///
/// Given a structure like:
/// ```text
/// dir/
///   50_000_000/
///     slot_updates.parquet
///     tx_header.parquet
///   50_100_000/
///     slot_updates.parquet
///     tx_header.parquet
/// ```
///
/// Returns a map: {"slot_updates" -> concat(all slot_updates), "tx_header" -> concat(all tx_header)}
pub async fn load_concat_dir(dir: &Path) -> Result<HashMap<String, DataFrame>, PolarsError> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || load_concat_dir_sync(&dir))
        .await
        .map_err(|e| PolarsError::ComputeError(format!("blocking task failed: {e}").into()))?
}
