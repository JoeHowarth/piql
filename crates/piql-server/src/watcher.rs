//! File watcher for automatic DataFrame reloading
//!
//! This module is feature-gated behind the `file-watcher` feature.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::core::ServerCore;
use crate::loader::{
    collect_files, df_name_from_path, is_supported_file, load_file, load_file_sync,
};
use crate::runs::{RunRegistry, RunRegistryOptions};
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

// ============ Run Watcher ============

/// Event types sent from the notify callback to the async handler
enum RunEvent {
    /// A _ready sentinel appeared in a subdirectory
    Ready(PathBuf),
    /// A subdirectory was removed
    Removed(PathBuf),
}

/// Watches a parent directory for run subdirectories (sentinel-based).
/// Owns the RunRegistry — sole mutator, no locking needed.
pub struct RunWatcher {
    _watcher: RecommendedWatcher,
}

#[derive(Debug, Clone, Default)]
pub struct RunModeOptions {
    pub drop_existing_run_label_column: bool,
}

/// Load all parquet files from a run directory into the registry.
async fn load_run_dir(
    registry: &mut RunRegistry,
    run_name: &str,
    run_dir: &std::path::Path,
    core: &ServerCore,
) {
    let run_dir_owned = run_dir.to_path_buf();
    let tables = match tokio::task::spawn_blocking(move || {
        let files = collect_files(&[run_dir_owned]);
        let mut tables = std::collections::HashMap::new();
        for path in files {
            match load_file_sync(&path) {
                Ok(df) => {
                    let name = df_name_from_path(&path);
                    tables.insert(name, df);
                }
                Err(e) => {
                    log::warn!("Failed to load {}: {}", path.display(), e);
                }
            }
        }
        tables
    })
    .await
    {
        Ok(tables) if !tables.is_empty() => tables,
        Ok(_) => {
            log::warn!("Run '{}' has no loadable tables, skipping", run_name);
            return;
        }
        Err(e) => {
            log::error!("Failed to load run '{}': {}", run_name, e);
            return;
        }
    };

    if let Err(err) = registry.load_run(run_name, tables, core).await {
        log::error!("Failed to load run '{}': {}", run_name, err);
    }
}

/// Scan for existing ready runs, load them, then start watching for new ones.
pub async fn load_and_watch_runs(
    core: Arc<ServerCore>,
    parent: PathBuf,
    options: RunModeOptions,
) -> notify::Result<RunWatcher> {
    let mut registry = RunRegistry::with_options(RunRegistryOptions {
        drop_existing_run_label_column: options.drop_existing_run_label_column,
        ..Default::default()
    });

    // Scan for existing run subdirs with _ready sentinel
    let mut run_dirs: Vec<_> = std::fs::read_dir(&parent)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().is_dir() && entry.path().join("_ready").exists())
        .collect();

    // Sort by name (timestamp prefix → chronological order)
    run_dirs.sort_by_key(|e| e.file_name());

    for entry in &run_dirs {
        let run_dir = entry.path();
        let Some(run_name) = run_dir.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        load_run_dir(&mut registry, run_name, &run_dir, &core).await;
    }

    // Now start watching — but we need to hand off the registry to the watcher.
    // The watcher creates its own registry, so we need a different approach:
    // we build the watcher with pre-loaded registry.
    RunWatcher::with_registry(core, parent, registry)
}

impl RunWatcher {
    fn with_registry(
        core: Arc<ServerCore>,
        parent: PathBuf,
        initial_registry: RunRegistry,
    ) -> notify::Result<Self> {
        let (tx, mut rx) = mpsc::channel::<RunEvent>(100);

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                for path in &event.paths {
                    if path.file_name().and_then(|f| f.to_str()) == Some("_ready") {
                        match event.kind {
                            EventKind::Create(_) => {
                                let _ = tx.blocking_send(RunEvent::Ready(path.clone()));
                            }
                            EventKind::Remove(_) => {
                                let _ = tx.blocking_send(RunEvent::Removed(path.clone()));
                            }
                            _ => {}
                        }
                    }
                    if event.kind == EventKind::Remove(notify::event::RemoveKind::Folder) {
                        let _ = tx.blocking_send(RunEvent::Removed(path.clone()));
                    }
                }
            }
        })?;

        watcher.watch(&parent, RecursiveMode::Recursive)?;

        tokio::spawn(async move {
            let mut registry = initial_registry;

            while let Some(event) = rx.recv().await {
                match event {
                    RunEvent::Ready(sentinel_path) => {
                        let Some(run_dir) = sentinel_path.parent() else {
                            continue;
                        };
                        let Some(run_name) = run_dir.file_name().and_then(|f| f.to_str()) else {
                            continue;
                        };
                        load_run_dir(&mut registry, run_name, run_dir, &core).await;
                    }
                    RunEvent::Removed(path) => {
                        let dir = if path.file_name().and_then(|f| f.to_str()) == Some("_ready") {
                            path.parent().unwrap_or(&path)
                        } else {
                            &path
                        };
                        let Some(run_name) = dir.file_name().and_then(|f| f.to_str()) else {
                            continue;
                        };
                        registry.remove_run(run_name, &core).await;
                    }
                }
            }
        });

        Ok(Self { _watcher: watcher })
    }
}
