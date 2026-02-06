//! Run registry for multi-run mode
//!
//! Tracks loaded simulation runs and manages the three-tier naming:
//! - `table` → latest run's version
//! - `run_name::table` → specific run's version
//! - `_all::table` → all runs concatenated with `_run` column

use std::collections::HashMap;

use polars::prelude::*;

use crate::core::ServerCore;
use crate::state::DfUpdate;

pub struct RunRegistry {
    /// Loaded runs in insertion order (oldest first)
    runs: Vec<RunInfo>,
    /// Name of the most recently loaded run
    latest: Option<String>,
}

struct RunInfo {
    name: String,
    /// Table name → DataFrame with `_run` column added (for _all:: rebuilds)
    tables: HashMap<String, DataFrame>,
}

impl RunRegistry {
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            latest: None,
        }
    }

    /// Load a new run, registering all three naming tiers.
    pub async fn load_run(
        &mut self,
        run_name: &str,
        tables: HashMap<String, DataFrame>,
        core: &ServerCore,
    ) {
        if run_name == "_all" {
            log::warn!("Skipping run named '_all' — reserved name");
            return;
        }

        if self.runs.iter().any(|r| r.name == run_name) {
            log::debug!("Run '{}' already loaded, skipping", run_name);
            return;
        }

        let mut annotated = HashMap::new();

        for (table, df) in &tables {
            // 1. Register run-specific: run_name::table
            let specific_name = format!("{run_name}::{table}");
            core.insert_df(&specific_name, df.clone()).await;

            // 2. Update latest bare name
            core.apply_update(DfUpdate::Reload {
                name: table.clone(),
                df: df.clone(),
            })
            .await;

            // 3. Build _run-annotated version for _all:: concat
            let with_run = df
                .clone()
                .lazy()
                .with_column(lit(run_name).alias("_run"))
                .collect()
                .unwrap_or_else(|e| {
                    log::error!("Failed to add _run column for {specific_name}: {e}");
                    df.clone()
                });
            annotated.insert(table.clone(), with_run);
        }

        self.runs.push(RunInfo {
            name: run_name.to_string(),
            tables: annotated,
        });
        self.latest = Some(run_name.to_string());

        // 4. Rebuild _all:: for each table in this run
        let table_names: Vec<String> = tables.keys().cloned().collect();
        for table in &table_names {
            self.rebuild_all(table, core).await;
        }

        log::info!(
            "Loaded run '{}' with {} tables (now {} runs total)",
            run_name,
            table_names.len(),
            self.runs.len()
        );
    }

    /// Remove a run and clean up all three tiers.
    pub async fn remove_run(&mut self, run_name: &str, core: &ServerCore) {
        let Some(idx) = self.runs.iter().position(|r| r.name == run_name) else {
            log::debug!("Run '{}' not loaded, nothing to remove", run_name);
            return;
        };

        let removed = self.runs.remove(idx);
        let table_names: Vec<String> = removed.tables.keys().cloned().collect();

        // Remove run-specific DFs
        for table in &table_names {
            let specific_name = format!("{run_name}::{table}");
            core.apply_update(DfUpdate::Remove {
                name: specific_name,
            })
            .await;
        }

        // Update latest if we removed the current latest
        let was_latest = self.latest.as_deref() == Some(run_name);
        if was_latest {
            self.latest = self.runs.last().map(|r| r.name.clone());
        }

        // Rebuild _all:: and bare names for affected tables
        for table in &table_names {
            self.rebuild_all(table, core).await;

            if was_latest {
                // Update bare name to new latest (or remove if no runs left)
                match self.latest_df_for(table) {
                    Some(df) => {
                        core.apply_update(DfUpdate::Reload {
                            name: table.clone(),
                            df,
                        })
                        .await;
                    }
                    None => {
                        core.apply_update(DfUpdate::Remove {
                            name: table.clone(),
                        })
                        .await;
                    }
                }
            }
        }

        log::info!(
            "Removed run '{}' ({} tables, {} runs remaining)",
            run_name,
            table_names.len(),
            self.runs.len()
        );
    }

    /// Rebuild `_all::{table}` by concatenating all runs' annotated versions.
    async fn rebuild_all(&self, table: &str, core: &ServerCore) {
        let all_name = format!("_all::{table}");

        let frames: Vec<DataFrame> = self
            .runs
            .iter()
            .filter_map(|run| run.tables.get(table).cloned())
            .collect();

        if frames.is_empty() {
            core.apply_update(DfUpdate::Remove { name: all_name })
                .await;
            return;
        }

        let combined = if frames.len() == 1 {
            frames.into_iter().next().unwrap()
        } else {
            let lazy: Vec<LazyFrame> = frames.into_iter().map(|df| df.lazy()).collect();
            match concat(
                &lazy,
                UnionArgs {
                    rechunk: false,
                    ..Default::default()
                },
            )
            .and_then(|lf| lf.collect())
            {
                Ok(df) => df,
                Err(e) => {
                    log::error!("Failed to concat _all::{table}: {e}");
                    return;
                }
            }
        };

        core.apply_update(DfUpdate::Reload {
            name: all_name,
            df: combined,
        })
        .await;
    }

    /// Get the raw (un-annotated) DF for a table from the latest run.
    fn latest_df_for(&self, table: &str) -> Option<DataFrame> {
        let latest_name = self.latest.as_deref()?;
        let run = self.runs.iter().find(|r| r.name == latest_name)?;
        let annotated = run.tables.get(table)?;
        // Strip the _run column to get the bare version
        Some(annotated.drop("_run").unwrap_or_else(|_| annotated.clone()))
    }
}
