//! File watcher for automatic DataFrame reloading
//!
//! This module is feature-gated behind the `file-watcher` feature.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::core::ServerCore;
use crate::loader::{df_name_from_path, is_supported_file, load_file};
use crate::state::DfUpdate;

/// Watch paths for changes and send updates to ServerCore
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
}

impl FileWatcher {
    /// Create a new file watcher that monitors the given paths
    pub fn new(core: Arc<ServerCore>, paths: Vec<PathBuf>) -> notify::Result<Self> {
        let (tx, mut rx) = mpsc::channel::<PathBuf>(100);

        // Set up the notify watcher
        let tx_clone = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                        for path in event.paths {
                            if is_supported_file(&path) {
                                let _ = tx_clone.blocking_send(path);
                            }
                        }
                    }
                    _ => {}
                }
            }
        })?;

        // Watch all provided paths
        for path in &paths {
            watcher.watch(path, RecursiveMode::NonRecursive)?;
        }

        // Spawn task to process file change events
        tokio::spawn(async move {
            // Debounce: collect events for a short window
            let mut pending: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
            let debounce_duration = Duration::from_millis(100);

            loop {
                tokio::select! {
                    Some(path) = rx.recv() => {
                        pending.insert(path);
                    }
                    _ = tokio::time::sleep(debounce_duration), if !pending.is_empty() => {
                        for path in pending.drain() {
                            let update = if path.exists() {
                                // load_file is async and uses spawn_blocking internally
                                match load_file(&path).await {
                                    Ok(df) => {
                                        let name = df_name_from_path(&path);
                                        DfUpdate::Reload { name, df }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to reload {}: {}", path.display(), e);
                                        continue;
                                    }
                                }
                            } else {
                                let name = df_name_from_path(&path);
                                DfUpdate::Remove { name }
                            };
                            core.apply_update(update).await;
                        }
                    }
                }
            }
        });

        Ok(Self { _watcher: watcher })
    }
}

/// Load initial files and start watching for changes
pub async fn load_and_watch(
    core: Arc<ServerCore>,
    paths: Vec<PathBuf>,
) -> notify::Result<FileWatcher> {
    // Load initial files (load_file is async, uses spawn_blocking internally)
    let files = crate::loader::collect_files(&paths);
    for path in files {
        if let Ok(df) = load_file(&path).await {
            let name = df_name_from_path(&path);
            core.insert_df(name, df).await;
        }
    }

    // Start watching
    FileWatcher::new(core, paths)
}
