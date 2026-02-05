//! File loading and watching

use std::path::{Path, PathBuf};
use std::sync::Arc;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use piql::DataFrameEntry;
use polars::prelude::*;
use tokio::sync::mpsc;

use crate::state::SharedState;

// ============ File Loading ============

pub fn load_file(path: &Path) -> Result<LazyFrame, PolarsError> {
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

pub fn df_name_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

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

// ============ File Watcher ============

pub async fn watch_files(
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
