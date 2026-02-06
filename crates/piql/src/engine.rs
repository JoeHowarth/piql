//! Query engine with materialized tables and subscriptions
//!
//! For simulation use cases where:
//! - Base tables grow each tick
//! - Multiple queries share intermediate results
//! - Clients subscribe to query results

use indexmap::IndexMap;
use polars::prelude::*;
use std::collections::HashMap;

use crate::eval::{EvalContext, TimeSeriesConfig};
use crate::{PiqlError, Value, run};

/// State for a base table that grows each tick
#[derive(Clone)]
struct BaseTableState {
    /// All historical data (grows via concat each tick, None until first append)
    all: Option<LazyFrame>,
    /// Current tick's data only (None until first append)
    now: Option<LazyFrame>,
}

/// Query engine with materialized tables and subscriptions
///
/// # Example
///
/// ```ignore
/// let mut engine = QueryEngine::new();
/// engine.add_base_df("entities", df);
///
/// // Materialized table (re-evaluated each tick)
/// engine.materialize("merchants", "entities.filter(@merchant)")?;
///
/// // Subscriptions (results pushed each tick)
/// engine.subscribe("rich_now", "merchants.filter(@now & $gold > 100)");
///
/// // Each simulation tick
/// let results = engine.on_tick(tick)?;
/// for (name, df) in results {
///     push_to_client(name, df);
/// }
/// ```
pub struct QueryEngine {
    ctx: EvalContext,

    /// Base tables that grow each tick
    /// Stores both `all` (full history) and `now` (current tick) ptrs
    base_tables: HashMap<String, BaseTableState>,

    /// Materialized tables: name -> query
    /// Evaluated in insertion order each tick (user ensures correct dependency order)
    materialized: IndexMap<String, String>,

    /// Subscribed queries: name -> query
    subscriptions: HashMap<String, String>,
}

impl QueryEngine {
    pub fn new() -> Self {
        Self {
            ctx: EvalContext::new(),
            base_tables: HashMap::new(),
            materialized: IndexMap::new(),
            subscriptions: HashMap::new(),
        }
    }

    /// Add a base dataframe (not time-series, collects immediately)
    pub fn add_base_df(&mut self, name: impl Into<String>, df: LazyFrame) {
        let collected = df.collect().expect("failed to collect DataFrame");
        self.ctx.dataframes.insert(
            name.into(),
            crate::eval::DataFrameEntry {
                df: collected,
                time_series: None,
            },
        );
    }

    /// Add a time-series dataframe (collects immediately)
    pub fn add_time_series_df(
        &mut self,
        name: impl Into<String>,
        df: LazyFrame,
        config: TimeSeriesConfig,
    ) {
        let collected = df.collect().expect("failed to collect DataFrame");
        self.ctx.dataframes.insert(
            name.into(),
            crate::eval::DataFrameEntry {
                df: collected,
                time_series: Some(config),
            },
        );
    }

    /// Update a base dataframe (e.g., after appending new rows, collects immediately)
    pub fn update_df(&mut self, name: &str, df: LazyFrame) {
        if let Some(entry) = self.ctx.dataframes.get_mut(name) {
            entry.df = df.collect().expect("failed to collect DataFrame");
        }
    }

    /// Access the sugar registry for registering custom directives
    pub fn sugar(&mut self) -> &mut crate::sugar::SugarRegistry {
        &mut self.ctx.sugar
    }

    /// Register a base table that grows each tick
    ///
    /// Base tables support implicit "now" scoping:
    /// - `entities.filter(...)` → uses current tick data only
    /// - `entities.all().filter(...)` → uses full history
    /// - `entities.window(-10, 0).filter(...)` → uses history with tick filter
    pub fn register_base(&mut self, name: impl Into<String>, config: TimeSeriesConfig) {
        let name = name.into();
        // Register config in eval context (it holds the config for scope method routing)
        self.ctx.register_base_table(name.clone(), config);
        // Initialize with empty state (will be populated on first append_tick)
        self.base_tables.insert(
            name,
            BaseTableState {
                all: None,
                now: None,
            },
        );
    }

    /// Append new tick data to a base table
    ///
    /// - Concatenates rows into `all` (full history)
    /// - Replaces `now` with the new rows
    /// - Both share the underlying Arrow arrays (no data copy)
    pub fn append_tick(&mut self, name: &str, rows: LazyFrame) -> Result<(), PiqlError> {
        let state = self
            .base_tables
            .get_mut(name)
            .ok_or_else(|| crate::eval::EvalError::UnknownIdent(name.to_string()))?;

        // Concat or initialize
        state.all = Some(match state.all.take() {
            Some(existing) => concat([existing, rows.clone()], UnionArgs::default())
                .map_err(crate::eval::EvalError::from)?,
            None => rows.clone(),
        });

        // Now ptr is just the new rows
        state.now = Some(rows);

        // Update eval context with current ptrs
        self.ctx.update_base_table_ptrs(
            name,
            state.all.clone().unwrap(),
            state.now.clone().unwrap(),
        );

        Ok(())
    }

    /// Add a materialized table
    ///
    /// The query is evaluated immediately and stored. It will be re-evaluated
    /// each tick before subscriptions.
    /// Add them in dependency order (if A depends on B, add B first).
    pub fn materialize(
        &mut self,
        name: impl Into<String>,
        query: impl Into<String>,
    ) -> Result<(), PiqlError> {
        let name = name.into();
        let query = query.into();

        // Evaluate immediately
        let result = run(&query, &self.ctx)?;
        if let Value::DataFrame(lf, _) = result {
            let collected = lf.collect().map_err(crate::eval::EvalError::from)?;
            self.ctx.dataframes.insert(
                name.clone(),
                crate::eval::DataFrameEntry {
                    df: collected,
                    time_series: None,
                },
            );
        }

        self.materialized.insert(name, query);
        Ok(())
    }

    /// Subscribe to a query's results
    ///
    /// Results are computed each tick and returned from `on_tick()`.
    pub fn subscribe(&mut self, name: impl Into<String>, query: impl Into<String>) {
        self.subscriptions.insert(name.into(), query.into());
    }

    /// Unsubscribe from a query
    pub fn unsubscribe(&mut self, name: &str) {
        self.subscriptions.remove(name);
    }

    /// Process a tick: re-evaluate materialized tables and subscriptions
    ///
    /// Returns results for all subscribed queries.
    pub fn on_tick(&mut self, tick: i64) -> Result<HashMap<String, DataFrame>, PiqlError> {
        self.ctx.tick = Some(tick);

        // 1. Re-evaluate materialized tables in order
        for (name, query) in &self.materialized {
            let result = run(query, &self.ctx)?;
            if let Value::DataFrame(lf, _) = result {
                // Store as new DF entry (no time-series config for derived tables)
                let collected = lf.collect().map_err(crate::eval::EvalError::from)?;
                self.ctx.dataframes.insert(
                    name.clone(),
                    crate::eval::DataFrameEntry {
                        df: collected,
                        time_series: None,
                    },
                );
            }
        }

        // 2. Evaluate all subscriptions
        let mut results = HashMap::new();
        for (name, query) in &self.subscriptions {
            let result = run(query, &self.ctx)?;
            if let Value::DataFrame(df, _) = result {
                let collected = df.collect().map_err(crate::eval::EvalError::from)?;
                results.insert(name.clone(), collected);
            }
        }

        Ok(results)
    }

    /// Run a one-off query without subscribing
    pub fn query(&self, query: &str) -> Result<Value, PiqlError> {
        run(query, &self.ctx)
    }

    /// Get current tick
    pub fn tick(&self) -> Option<i64> {
        self.ctx.tick
    }

    /// Set current tick (for queries outside of on_tick)
    pub fn set_tick(&mut self, tick: i64) {
        self.ctx.tick = Some(tick);
    }

    /// Set default tick column for scope methods when table config is unavailable.
    pub fn set_default_tick_column(&mut self, tick_column: impl Into<String>) {
        self.ctx.default_tick_column = Some(tick_column.into());
    }

    /// Set default partition key for sugar methods when table config is unavailable.
    pub fn set_default_partition_key(&mut self, partition_key: impl Into<String>) {
        self.ctx.default_partition_key = Some(partition_key.into());
    }

    /// Get names of all registered dataframes
    pub fn dataframe_names(&self) -> Vec<String> {
        self.ctx.dataframes.keys().cloned().collect()
    }
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}
