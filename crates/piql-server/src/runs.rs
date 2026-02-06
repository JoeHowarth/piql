//! Run registry for multi-run mode
//!
//! Tracks loaded simulation runs and manages the three-tier naming:
//! - `table` → latest run's version
//! - `run_name::table` → specific run's version
//! - `_all::table` → all runs concatenated with a run-label column (`_run` by default)

use std::collections::HashMap;

use polars::prelude::*;
use thiserror::Error;

use crate::core::ServerCore;
use crate::state::DfUpdate;

pub const DEFAULT_RUN_LABEL_COLUMN: &str = "_run";

#[derive(Debug, Clone)]
pub struct RunRegistryOptions {
    pub run_label_column: String,
    pub drop_existing_run_label_column: bool,
}

impl Default for RunRegistryOptions {
    fn default() -> Self {
        Self {
            run_label_column: DEFAULT_RUN_LABEL_COLUMN.to_string(),
            drop_existing_run_label_column: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum RunRegistryError {
    #[error(
        "run '{run_name}' table '{table}' already contains reserved run-label column '{column}'"
    )]
    RunLabelColumnConflict {
        run_name: String,
        table: String,
        column: String,
    },
    #[error(
        "failed to drop existing run-label column '{column}' from run '{run_name}' table '{table}': {source}"
    )]
    DropRunLabelColumnFailed {
        run_name: String,
        table: String,
        column: String,
        source: PolarsError,
    },
}

pub struct RunRegistry {
    /// Loaded runs in insertion order (oldest first)
    runs: Vec<RunInfo>,
    /// Name of the most recently loaded run
    latest: Option<String>,
    options: RunRegistryOptions,
}

struct RunInfo {
    name: String,
    /// Table name → DataFrame with `_run` column added (for _all:: rebuilds)
    tables: HashMap<String, DataFrame>,
}

impl RunRegistry {
    pub fn new() -> Self {
        Self::with_options(RunRegistryOptions::default())
    }

    pub fn with_options(options: RunRegistryOptions) -> Self {
        Self {
            runs: Vec::new(),
            latest: None,
            options,
        }
    }

    /// Load a new run, registering all three naming tiers.
    pub async fn load_run(
        &mut self,
        run_name: &str,
        tables: HashMap<String, DataFrame>,
        core: &ServerCore,
    ) -> Result<(), RunRegistryError> {
        if run_name == "_all" {
            log::warn!("Skipping run named '_all' — reserved name");
            return Ok(());
        }

        if self.runs.iter().any(|r| r.name == run_name) {
            log::debug!("Run '{}' already loaded, skipping", run_name);
            return Ok(());
        }

        let known_tables_before = self.all_table_names();
        let mut normalized_tables: HashMap<String, DataFrame> = HashMap::new();
        let mut annotated = HashMap::new();

        for (table, df) in tables {
            let normalized = self.normalize_run_table(run_name, &table, df)?;
            normalized_tables.insert(table, normalized);
        }

        for (table, df) in &normalized_tables {
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
                .with_column(lit(run_name).alias(&self.options.run_label_column))
                .collect()
                .unwrap_or_else(|e| {
                    log::error!(
                        "Failed to add {} column for {specific_name}: {e}",
                        self.options.run_label_column
                    );
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
        let table_names: Vec<String> = normalized_tables.keys().cloned().collect();
        for table in &table_names {
            self.rebuild_all(table, core).await;
        }
        self.rebuild_latest_bare_names(core, known_tables_before)
            .await;

        log::info!(
            "Loaded run '{}' with {} tables (now {} runs total)",
            run_name,
            table_names.len(),
            self.runs.len()
        );

        Ok(())
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

        let known_tables_before = self.all_table_names_with(&removed.tables);
        self.latest = self.runs.last().map(|r| r.name.clone());

        // Rebuild _all:: and bare names for affected tables
        for table in &table_names {
            self.rebuild_all(table, core).await;
        }
        self.rebuild_latest_bare_names(core, known_tables_before)
            .await;

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
            core.apply_update(DfUpdate::Remove { name: all_name }).await;
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
        // Strip the run-label column to get the bare version
        Some(
            annotated
                .drop(&self.options.run_label_column)
                .unwrap_or_else(|_| annotated.clone()),
        )
    }

    fn normalize_run_table(
        &self,
        run_name: &str,
        table: &str,
        df: DataFrame,
    ) -> Result<DataFrame, RunRegistryError> {
        let run_label_column = &self.options.run_label_column;
        if df.get_column_index(run_label_column).is_some() {
            if self.options.drop_existing_run_label_column {
                return df.drop(run_label_column).map_err(|source| {
                    RunRegistryError::DropRunLabelColumnFailed {
                        run_name: run_name.to_string(),
                        table: table.to_string(),
                        column: run_label_column.clone(),
                        source,
                    }
                });
            }
            return Err(RunRegistryError::RunLabelColumnConflict {
                run_name: run_name.to_string(),
                table: table.to_string(),
                column: run_label_column.clone(),
            });
        }
        Ok(df)
    }

    fn all_table_names(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        for run in &self.runs {
            seen.extend(run.tables.keys().cloned());
        }
        seen.into_iter().collect()
    }

    fn all_table_names_with(&self, extra: &HashMap<String, DataFrame>) -> Vec<String> {
        let mut names = self.all_table_names();
        names.extend(extra.keys().cloned());
        names.sort();
        names.dedup();
        names
    }

    async fn rebuild_latest_bare_names(&self, core: &ServerCore, known_tables: Vec<String>) {
        for table in known_tables {
            match self.latest_df_for(&table) {
                Some(df) => {
                    core.apply_update(DfUpdate::Reload { name: table, df })
                        .await;
                }
                None => {
                    core.apply_update(DfUpdate::Remove { name: table }).await;
                }
            }
        }
    }
}

impl Default for RunRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;
    use std::collections::HashSet;

    #[tokio::test]
    async fn latest_bare_names_are_rebuilt_on_new_run() {
        let core = ServerCore::new();
        let mut registry = RunRegistry::new();

        let mut r1 = HashMap::new();
        r1.insert("a".to_string(), df! { "x" => &[1] }.unwrap());
        registry.load_run("r1", r1, &core).await.unwrap();

        let mut r2 = HashMap::new();
        r2.insert("b".to_string(), df! { "y" => &[2] }.unwrap());
        registry.load_run("r2", r2, &core).await.unwrap();

        let names: HashSet<_> = core.list_dataframes().await.into_iter().collect();
        assert!(names.contains("b"));
        assert!(!names.contains("a"));
    }

    #[tokio::test]
    async fn removing_old_run_cleans_stale_bare_names() {
        let core = ServerCore::new();
        let mut registry = RunRegistry::new();

        let mut r1 = HashMap::new();
        r1.insert("a".to_string(), df! { "x" => &[1] }.unwrap());
        registry.load_run("r1", r1, &core).await.unwrap();

        let mut r2 = HashMap::new();
        r2.insert("b".to_string(), df! { "y" => &[2] }.unwrap());
        registry.load_run("r2", r2, &core).await.unwrap();

        registry.remove_run("r1", &core).await;
        let names: HashSet<_> = core.list_dataframes().await.into_iter().collect();
        assert!(names.contains("b"));
        assert!(!names.contains("a"));
    }

    #[tokio::test]
    async fn load_run_rejects_existing_run_label_column_by_default() {
        let core = ServerCore::new();
        let mut registry = RunRegistry::new();

        let mut run = HashMap::new();
        run.insert(
            "a".to_string(),
            df! { "x" => &[1], "_run" => &["existing"] }.unwrap(),
        );
        let result = registry.load_run("r1", run, &core).await;

        assert!(matches!(
            result,
            Err(RunRegistryError::RunLabelColumnConflict { .. })
        ));
        assert!(core.list_dataframes().await.is_empty());
    }

    #[tokio::test]
    async fn load_run_can_drop_existing_run_label_column() {
        let core = ServerCore::new();
        let mut registry = RunRegistry::with_options(RunRegistryOptions {
            drop_existing_run_label_column: true,
            ..Default::default()
        });

        let mut run = HashMap::new();
        run.insert(
            "a".to_string(),
            df! { "x" => &[1], "_run" => &["existing"] }.unwrap(),
        );
        registry.load_run("r1", run, &core).await.unwrap();

        let bare = core.execute_query("a").await.unwrap();
        assert!(bare.column("_run").is_err());

        let combined = core.execute_query("_all::a").await.unwrap();
        let run_col = combined.column("_run").unwrap().str().unwrap();
        assert_eq!(run_col.get(0), Some("r1"));
    }
}
