//! Interpreter that evaluates PiQL AST against Polars dataframes
//!
//! Evaluates core::Expr (the transformed AST) against Polars dataframes.

use std::collections::HashMap;

use polars::prelude::*;
use polars::series::ops::NullBehavior;
use polars_ops::series::RoundMode;
use thiserror::Error;

use crate::ast::core::{CoreArg, Expr};
use crate::ast::{Arg, BinOp, Literal, UnaryOp};

#[derive(Error, Debug)]
pub enum EvalError {
    #[error("Unknown identifier: {0}")]
    UnknownIdent(String),

    #[error("Unknown method '{method}' on {target}")]
    UnknownMethod { target: String, method: String },

    #[error("Type error: expected {expected}, got {got}")]
    TypeError { expected: String, got: String },

    #[error("Argument error: {0}")]
    ArgError(String),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("{0}")]
    Other(String),
}

type Result<T> = std::result::Result<T, EvalError>;

/// Runtime value produced by evaluation
#[derive(Clone)]
pub enum Value {
    /// A Polars LazyFrame with source lineage metadata
    DataFrame(LazyFrame, DataFrameLineage),
    /// A Polars LazyGroupBy (from group_by, before agg), retaining lineage
    GroupBy(LazyGroupBy, DataFrameLineage),
    /// A Polars column expression
    Expr(polars::prelude::Expr),
    /// A scalar/literal value (for use in expressions)
    Scalar(ScalarValue),
    /// The `pl` namespace object
    PlNamespace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataFrameLineage {
    /// Direct table source by name.
    Table(String),
    /// Derived from a single table source.
    DerivedFrom(String),
    /// Derived from multiple sources (e.g., join).
    Ambiguous,
    /// Source is unknown.
    Unknown,
}

impl DataFrameLineage {
    fn derived(&self) -> Self {
        match self {
            Self::Table(name) | Self::DerivedFrom(name) => Self::DerivedFrom(name.clone()),
            Self::Ambiguous => Self::Ambiguous,
            Self::Unknown => Self::Unknown,
        }
    }

    fn source_name(&self) -> Option<&str> {
        match self {
            Self::Table(name) | Self::DerivedFrom(name) => Some(name),
            Self::Ambiguous | Self::Unknown => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ScalarValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

/// Configuration for time-series dataframes
#[derive(Debug, Clone)]
pub struct TimeSeriesConfig {
    /// Column name containing tick values
    pub tick_column: String,
    /// Partition key for windowed operations (e.g., "entity_id")
    pub partition_key: String,
}

/// A registered dataframe with optional time-series config
#[derive(Clone)]
pub struct DataFrameEntry {
    /// Materialized DataFrame (collected on insert for fast repeated access)
    pub df: DataFrame,
    pub time_series: Option<TimeSeriesConfig>,
}

/// State for a base table tracked in eval context
#[derive(Clone)]
pub struct BaseTableEntry {
    /// All historical data (None until first data appended)
    pub all: Option<LazyFrame>,
    /// Current tick's data only (None until first data appended)
    pub now: Option<LazyFrame>,
    /// Time-series configuration
    pub config: TimeSeriesConfig,
}

/// Evaluation context - holds named dataframes and configuration
#[derive(Clone)]
pub struct EvalContext {
    pub dataframes: HashMap<String, DataFrameEntry>,
    /// Base tables with all/now ptrs for implicit now scoping
    pub base_tables: HashMap<String, BaseTableEntry>,
    /// Current simulation tick (for @now, .window, etc.)
    pub tick: Option<i64>,
    /// Default tick column for scope methods when source table config is unavailable
    pub default_tick_column: Option<String>,
    /// Default partition key for sugar methods when source table config is unavailable
    pub default_partition_key: Option<String>,
    /// Sugar registry for directive expansion
    pub sugar: crate::sugar::SugarRegistry,
}

impl EvalContext {
    pub fn new() -> Self {
        Self {
            dataframes: HashMap::new(),
            base_tables: HashMap::new(),
            tick: None,
            default_tick_column: None,
            default_partition_key: None,
            sugar: crate::sugar::SugarRegistry::new(),
        }
    }

    /// Add a regular (non-time-series) dataframe (collects immediately)
    pub fn with_df(mut self, name: impl Into<String>, df: LazyFrame) -> Self {
        let collected = df.collect().expect("failed to collect DataFrame");
        self.dataframes.insert(
            name.into(),
            DataFrameEntry {
                df: collected,
                time_series: None,
            },
        );
        self
    }

    /// Add a pre-collected dataframe
    pub fn with_materialized_df(mut self, name: impl Into<String>, df: DataFrame) -> Self {
        self.dataframes.insert(
            name.into(),
            DataFrameEntry {
                df,
                time_series: None,
            },
        );
        self
    }

    /// Add a time-series dataframe with tick column and partition key (collects immediately)
    pub fn with_time_series_df(
        mut self,
        name: impl Into<String>,
        df: LazyFrame,
        config: TimeSeriesConfig,
    ) -> Self {
        let collected = df.collect().expect("failed to collect DataFrame");
        self.dataframes.insert(
            name.into(),
            DataFrameEntry {
                df: collected,
                time_series: Some(config),
            },
        );
        self
    }

    /// Set the current tick for time-based queries
    pub fn with_tick(mut self, tick: i64) -> Self {
        self.tick = Some(tick);
        self
    }

    /// Set default tick column used by scope methods when table config is unavailable
    pub fn with_default_tick_column(mut self, tick_column: impl Into<String>) -> Self {
        self.default_tick_column = Some(tick_column.into());
        self
    }

    /// Set default partition key used by sugar methods when table config is unavailable
    pub fn with_default_partition_key(mut self, partition_key: impl Into<String>) -> Self {
        self.default_partition_key = Some(partition_key.into());
        self
    }

    /// Get time-series config for a dataframe (if registered as time-series)
    pub fn get_time_series_config(&self, name: &str) -> Option<&TimeSeriesConfig> {
        self.dataframes
            .get(name)
            .and_then(|entry| entry.time_series.as_ref())
    }

    /// Build a SugarContext from this EvalContext for a specific dataframe
    pub fn sugar_context(&self, df_name: Option<&str>) -> crate::sugar::SugarContext {
        let partition_key = df_name
            .and_then(|name| self.get_time_series_config(name))
            .map(|ts| ts.partition_key.clone())
            .or_else(|| self.default_partition_key.clone());

        crate::sugar::SugarContext {
            tick: self.tick,
            partition_key,
        }
    }

    /// Register a base table (called by QueryEngine::register_base)
    pub fn register_base_table(&mut self, name: String, config: TimeSeriesConfig) {
        self.base_tables.insert(
            name,
            BaseTableEntry {
                all: None,
                now: None,
                config,
            },
        );
    }

    /// Update base table ptrs (called by QueryEngine::append_tick)
    pub fn update_base_table_ptrs(&mut self, name: &str, all: LazyFrame, now: LazyFrame) {
        if let Some(entry) = self.base_tables.get_mut(name) {
            entry.all = Some(all.clone());
            entry.now = Some(now);
            // Also update dataframes to point to `all` (for non-base-table-aware code paths)
            let collected = all.collect().expect("failed to collect base table");
            self.dataframes.insert(
                name.to_string(),
                DataFrameEntry {
                    df: collected,
                    time_series: Some(entry.config.clone()),
                },
            );
        }
    }

    /// Check if a name is a base table
    pub fn is_base_table(&self, name: &str) -> bool {
        self.base_tables.contains_key(name)
    }

    /// Get the `now` ptr for a base table
    pub fn get_base_now(&self, name: &str) -> Option<LazyFrame> {
        self.base_tables.get(name).and_then(|e| e.now.clone())
    }

    /// Get the `all` ptr for a base table
    pub fn get_base_all(&self, name: &str) -> Option<LazyFrame> {
        self.base_tables.get(name).and_then(|e| e.all.clone())
    }
}

impl Default for EvalContext {
    fn default() -> Self {
        Self::new()
    }
}

pub fn eval(expr: &Expr, ctx: &EvalContext) -> Result<Value> {
    match expr {
        Expr::Ident(name) => eval_ident(name, ctx),
        Expr::Literal(lit) => Ok(Value::Scalar(literal_to_scalar(lit))),
        Expr::List(items) => eval_list(items, ctx),
        Expr::Attr(base, attr) => eval_attr(base, attr, ctx),
        Expr::Call(callee, args) => eval_call(callee, args, ctx),
        Expr::BinaryOp(lhs, op, rhs) => eval_binop(lhs, *op, rhs, ctx),
        Expr::UnaryOp(op, operand) => eval_unaryop(*op, operand, ctx),
        Expr::WhenThenOtherwise {
            branches,
            otherwise,
        } => eval_when_then_otherwise(branches, otherwise, ctx),
        Expr::Invalid(message) => Err(EvalError::Other(message.clone())),
    }
}

fn eval_ident(name: &str, ctx: &EvalContext) -> Result<Value> {
    match name {
        "pl" => Ok(Value::PlNamespace),
        _ => {
            // Check if it's a base table - return `now` ptr for implicit now
            if let Some(now_df) = ctx.get_base_now(name) {
                return Ok(Value::DataFrame(
                    now_df,
                    DataFrameLineage::Table(name.to_string()),
                ));
            }
            // Otherwise check regular dataframes
            if let Some(entry) = ctx.dataframes.get(name) {
                Ok(Value::DataFrame(
                    entry.df.clone().lazy(),
                    DataFrameLineage::Table(name.to_string()),
                ))
            } else {
                Err(EvalError::UnknownIdent(name.to_string()))
            }
        }
    }
}

fn literal_to_scalar(lit: &Literal) -> ScalarValue {
    match lit {
        Literal::String(s) => ScalarValue::String(s.clone()),
        Literal::Int(n) => ScalarValue::Int(*n),
        Literal::Float(f) => ScalarValue::Float(*f),
        Literal::Bool(b) => ScalarValue::Bool(*b),
        Literal::Null => ScalarValue::Null,
    }
}

fn eval_list(items: &[Expr], ctx: &EvalContext) -> Result<Value> {
    if items.is_empty() {
        return Err(EvalError::Other("Empty list".to_string()));
    }
    // Lists are handled specially at call sites (e.g., select([a, b, c]))
    eval(&items[0], ctx)
}

fn eval_attr(base: &Expr, attr: &str, ctx: &EvalContext) -> Result<Value> {
    let base_val = eval(base, ctx)?;

    match base_val {
        Value::PlNamespace => Err(EvalError::Other(format!(
            "pl.{attr} must be called as a function"
        ))),
        Value::Expr(e) => {
            // Namespace markers - these get handled by the subsequent method call
            match attr {
                "str" | "dt" | "list" => Ok(Value::Expr(e)),
                _ => Err(EvalError::UnknownMethod {
                    target: "Expr".to_string(),
                    method: attr.to_string(),
                }),
            }
        }
        Value::DataFrame(_, _) => Err(EvalError::UnknownMethod {
            target: "DataFrame".to_string(),
            method: attr.to_string(),
        }),
        Value::GroupBy(_, _) => Err(EvalError::UnknownMethod {
            target: "GroupBy".to_string(),
            method: attr.to_string(),
        }),
        Value::Scalar(_) => Err(EvalError::TypeError {
            expected: "Expr or DataFrame".to_string(),
            got: "Scalar".to_string(),
        }),
    }
}

fn eval_call(callee: &Expr, args: &[CoreArg], ctx: &EvalContext) -> Result<Value> {
    if let Expr::Attr(base, method) = callee {
        return eval_method_call(base, method, args, ctx);
    }

    Err(EvalError::Other(
        "Direct function calls not yet supported".to_string(),
    ))
}

fn eval_method_call(
    base_expr: &Expr,
    method: &str,
    args: &[CoreArg],
    ctx: &EvalContext,
) -> Result<Value> {
    // Special case: namespace methods like .str.contains(), .dt.year()
    if let Expr::Attr(inner_base, namespace) = base_expr {
        if namespace == "str" {
            let e = eval_to_expr(inner_base, ctx)?;
            return eval_str_method(e, method, args);
        }
        if namespace == "dt" {
            let e = eval_to_expr(inner_base, ctx)?;
            return eval_dt_method(e, method);
        }
    }

    let base_val = eval(base_expr, ctx)?;
    let base_is_direct_ident = matches!(base_expr, Expr::Ident(_));

    match base_val {
        Value::PlNamespace => eval_pl_function(method, args, ctx),
        Value::DataFrame(df, lineage) => {
            eval_df_method(df, lineage, method, args, ctx, base_is_direct_ident)
        }
        Value::GroupBy(gb, lineage) => eval_groupby_method(gb, lineage, method, args, ctx),
        Value::Expr(e) => eval_expr_method(e, method, args, ctx),
        Value::Scalar(_) => Err(EvalError::TypeError {
            expected: "Expr or DataFrame".to_string(),
            got: "Scalar".to_string(),
        }),
    }
}

fn eval_pl_function(name: &str, args: &[CoreArg], ctx: &EvalContext) -> Result<Value> {
    match name {
        "col" => {
            let col_names = collect_string_args(args)?;
            if col_names.len() == 1 {
                Ok(Value::Expr(col(&col_names[0])))
            } else {
                // cols() returns Selector in polars 0.52+
                let names: Arc<[PlSmallStr]> =
                    col_names.into_iter().map(PlSmallStr::from).collect();
                let selector = Selector::ByName {
                    names,
                    strict: true,
                };
                Ok(Value::Expr(polars::prelude::Expr::Selector(selector)))
            }
        }
        "lit" => {
            let val = get_positional_arg(args, 0, "lit")?;
            let scalar = eval(val, ctx)?;
            match scalar {
                Value::Scalar(s) => Ok(Value::Expr(scalar_to_lit(s))),
                _ => Err(EvalError::ArgError(
                    "lit() expects a literal value".to_string(),
                )),
            }
        }
        "len" => {
            // pl.len() returns row count expression (like SQL COUNT(*))
            Ok(Value::Expr(polars::prelude::len()))
        }
        _ => Err(EvalError::UnknownMethod {
            target: "pl".to_string(),
            method: name.to_string(),
        }),
    }
}

fn eval_df_method(
    df: LazyFrame,
    lineage: DataFrameLineage,
    method: &str,
    args: &[CoreArg],
    ctx: &EvalContext,
    base_is_direct_ident: bool,
) -> Result<Value> {
    match method {
        "filter" => {
            let pred = get_positional_arg(args, 0, "filter")?;
            let pred_expr = eval_to_expr(pred, ctx)?;
            Ok(df_value(df.filter(pred_expr), &lineage))
        }
        "select" => {
            let exprs = collect_expr_args(args, ctx)?;
            Ok(df_value(df.select(exprs), &lineage))
        }
        "with_columns" => {
            let exprs = collect_expr_args(args, ctx)?;
            Ok(df_value(df.with_columns(exprs), &lineage))
        }
        "head" => {
            let n = get_int_arg(args, 0, "head").unwrap_or(10) as u32;
            Ok(df_value(df.limit(n), &lineage))
        }
        "sort" => {
            let col_names = get_strings_arg(args, 0, "sort")?;
            let descending = get_kwarg_bool(args, "descending").unwrap_or(false);
            let opts = SortMultipleOptions::new().with_order_descending(descending);
            Ok(df_value(df.sort(&col_names, opts), &lineage))
        }
        "tail" => {
            let n = get_int_arg(args, 0, "tail").unwrap_or(10) as u32;
            Ok(df_value(df.tail(n), &lineage))
        }
        "drop" => {
            let col_names = collect_string_args(args)?;
            let names: Arc<[PlSmallStr]> = col_names.into_iter().map(PlSmallStr::from).collect();
            let selector = Selector::ByName {
                names,
                strict: true,
            };
            Ok(df_value(df.drop(selector), &lineage))
        }
        "explode" => {
            let col_names = collect_string_args(args)?;
            let names: Arc<[PlSmallStr]> = col_names.into_iter().map(PlSmallStr::from).collect();
            let selector = Selector::ByName {
                names,
                strict: true,
            };
            Ok(df_value(df.explode(selector), &lineage))
        }
        "drop_nulls" => Ok(df_value(df.drop_nulls(None), &lineage)),
        "reverse" => Ok(df_value(df.reverse(), &lineage)),
        "unique" => {
            // df.unique() or df.unique(["col1", "col2"])
            let subset = if args.is_empty() {
                None
            } else {
                let col_names = get_strings_arg(args, 0, "unique")?;
                let names: Arc<[PlSmallStr]> =
                    col_names.into_iter().map(PlSmallStr::from).collect();
                Some(Selector::ByName {
                    names,
                    strict: true,
                })
            };
            Ok(df_value(
                df.unique(subset, UniqueKeepStrategy::Any),
                &lineage,
            ))
        }
        "count" => {
            // Returns non-null count per column (like pandas df.count())
            let schema = df.clone().collect_schema()?;
            let count_exprs: Vec<polars::prelude::Expr> = schema
                .iter()
                .map(|(name, _)| col(name.as_str()).count().alias(name.as_str()))
                .collect();
            Ok(df_value(df.select(count_exprs), &lineage))
        }
        "height" => {
            // Returns single-row DataFrame with row count
            Ok(df_value(
                df.select([polars::prelude::len().alias("height")]),
                &lineage,
            ))
        }
        "group_by" => {
            let col_names = collect_string_args(args)?;
            let col_exprs: Vec<_> = col_names.iter().map(col).collect();
            Ok(Value::GroupBy(df.group_by(col_exprs), lineage.derived()))
        }
        "rename" => {
            // Collect kwargs: rename(gold="coins", name="id")
            let renames: Vec<(String, String)> = args
                .iter()
                .filter_map(|arg| {
                    if let Arg::Keyword(old, Expr::Literal(Literal::String(new))) = arg {
                        Some((old.clone(), new.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            if !renames.is_empty() {
                let (old_names, new_names): (Vec<_>, Vec<_>) = renames.into_iter().unzip();
                Ok(df_value(df.rename(old_names, new_names, false), &lineage))
            } else {
                // Positional fallback: rename("old", "new")
                let old = get_string_arg(args, 0, "rename")?;
                let new = get_string_arg(args, 1, "rename")?;
                Ok(df_value(df.rename([old], [new], false), &lineage))
            }
        }
        // Scope methods for time-series data
        "all" => {
            // For direct base-table access, swap to `all` ptr; otherwise keep current df.
            if base_is_direct_ident
                && let Some(name) = lineage.source_name()
                && let Some(all_df) = ctx.get_base_all(name)
            {
                return Ok(df_value(all_df, &lineage));
            }
            Ok(df_value(df, &lineage))
        }
        "window" => {
            // For direct base-table access, scope against `all`; otherwise scope current df.
            let a = get_int_arg(args, 0, "window")?;
            let b = get_int_arg(args, 1, "window")?;
            let tick = ctx
                .tick
                .ok_or_else(|| EvalError::Other(".window() requires tick in context".into()))?;
            let tick_col = resolve_scope_tick_column(&lineage, ctx, "window")?;
            let target_df = scope_target_df(df, &lineage, ctx, base_is_direct_ident);

            let filtered = target_df.filter(col(&tick_col).is_between(
                lit(tick + a),
                lit(tick + b),
                ClosedInterval::Both,
            ));
            Ok(df_value(filtered, &lineage))
        }
        "since" => {
            // For direct base-table access, scope against `all`; otherwise scope current df.
            let n = get_int_arg(args, 0, "since")?;
            let tick_col = resolve_scope_tick_column(&lineage, ctx, "since")?;
            let target_df = scope_target_df(df, &lineage, ctx, base_is_direct_ident);

            let filtered = target_df.filter(col(&tick_col).gt_eq(lit(n)));
            Ok(df_value(filtered, &lineage))
        }
        "at" => {
            // For direct base-table access, scope against `all`; otherwise scope current df.
            let n = get_int_arg(args, 0, "at")?;
            let tick_col = resolve_scope_tick_column(&lineage, ctx, "at")?;
            let target_df = scope_target_df(df, &lineage, ctx, base_is_direct_ident);

            let filtered = target_df.filter(col(&tick_col).eq(lit(n)));
            Ok(df_value(filtered, &lineage))
        }
        // Convenience method
        "top" => {
            // .top(n, col) -> .sort(col, descending=True).head(n)
            let n = get_int_arg(args, 0, "top")? as u32;
            let sort_col = get_string_arg(args, 1, "top")?;
            let opts = SortMultipleOptions::new().with_order_descending(true);
            Ok(df_value(df.sort([sort_col], opts).limit(n), &lineage))
        }
        "describe" => {
            // Build describe statistics via lazy aggregations (no blocking collect)
            // Returns: statistic, col1, col2, ... for numeric columns
            let schema = df.clone().collect_schema()?;
            let numeric_cols: Vec<_> = schema
                .iter()
                .filter(|(_, dtype)| dtype.is_primitive_numeric() || dtype.is_float())
                .map(|(name, _)| name.to_string())
                .collect();

            if numeric_cols.is_empty() {
                return Err(EvalError::Other(
                    "describe() requires at least one numeric column".into(),
                ));
            }

            // Build a lazy row for each statistic
            let stats = ["count", "null_count", "mean", "std", "min", "max"];
            let rows: Vec<LazyFrame> = stats
                .iter()
                .map(|&stat| {
                    let exprs: Vec<polars::prelude::Expr> = numeric_cols
                        .iter()
                        .map(|c| {
                            let e = col(c);
                            match stat {
                                "count" => e.count().cast(DataType::Float64).alias(c),
                                "null_count" => e.null_count().cast(DataType::Float64).alias(c),
                                "mean" => e.mean().alias(c),
                                "std" => e.std(1).alias(c),
                                "min" => e.min().cast(DataType::Float64).alias(c),
                                "max" => e.max().cast(DataType::Float64).alias(c),
                                _ => unreachable!(),
                            }
                        })
                        .collect();

                    df.clone()
                        .select(exprs)
                        .with_column(lit(stat).alias("statistic"))
                })
                .collect();

            // Concat all rows lazily
            let result = polars::prelude::concat(rows, UnionArgs::default())?;

            // Reorder columns to put statistic first
            let mut col_order: Vec<polars::prelude::Expr> = vec![col("statistic")];
            col_order.extend(numeric_cols.iter().map(col));
            let result = result.select(col_order);

            Ok(df_value(result, &lineage))
        }
        "join" => {
            // Get the other dataframe (first positional arg)
            let other_expr = get_positional_arg(args, 0, "join")?;
            let other = match eval(other_expr, ctx)? {
                Value::DataFrame(lf, _) => lf,
                _ => {
                    return Err(EvalError::ArgError(
                        "join() first argument must be a DataFrame".to_string(),
                    ));
                }
            };

            // Get join type
            let how = get_kwarg_string(args, "how").unwrap_or_else(|| "inner".to_string());
            let join_type = match how.as_str() {
                "inner" => JoinType::Inner,
                "left" => JoinType::Left,
                "right" => JoinType::Right,
                "outer" | "full" => JoinType::Full,
                "cross" => JoinType::Cross,
                _ => return Err(EvalError::ArgError(format!("Unknown join type: {how}"))),
            };

            // Get join columns - supports single string or list
            let result = if let Some(on_cols) = get_kwarg_strings(args, "on") {
                // Same column name(s) on both sides
                let on_exprs: Vec<_> = on_cols.iter().map(col).collect();
                df.join(other, on_exprs.clone(), on_exprs, JoinArgs::new(join_type))
            } else {
                // Different column names
                let left_cols = get_kwarg_strings(args, "left_on").ok_or_else(|| {
                    EvalError::ArgError(
                        "join() requires 'on' or 'left_on'/'right_on' kwargs".to_string(),
                    )
                })?;
                let right_cols = get_kwarg_strings(args, "right_on").ok_or_else(|| {
                    EvalError::ArgError(
                        "join() requires 'right_on' when 'left_on' is specified".to_string(),
                    )
                })?;
                let left_exprs: Vec<_> = left_cols.iter().map(col).collect();
                let right_exprs: Vec<_> = right_cols.iter().map(col).collect();
                df.join(other, left_exprs, right_exprs, JoinArgs::new(join_type))
            };

            Ok(Value::DataFrame(result, DataFrameLineage::Ambiguous))
        }
        _ => Err(EvalError::UnknownMethod {
            target: "DataFrame".to_string(),
            method: method.to_string(),
        }),
    }
}

fn df_value(df: LazyFrame, lineage: &DataFrameLineage) -> Value {
    Value::DataFrame(df, lineage.derived())
}

fn scope_target_df(
    df: LazyFrame,
    lineage: &DataFrameLineage,
    ctx: &EvalContext,
    base_is_direct_ident: bool,
) -> LazyFrame {
    if base_is_direct_ident
        && let Some(name) = lineage.source_name()
        && let Some(entry) = ctx.base_tables.get(name)
        && let Some(all_df) = entry.all.clone()
    {
        return all_df;
    }

    df
}

fn resolve_scope_tick_column(
    lineage: &DataFrameLineage,
    ctx: &EvalContext,
    method: &str,
) -> Result<String> {
    if matches!(lineage, DataFrameLineage::Ambiguous) {
        return Err(EvalError::Other(format!(
            ".{method}() has ambiguous lineage; call .at/.since/.window before joins or configure an explicit tick column"
        )));
    }

    if let Some(name) = lineage.source_name() {
        if let Some(entry) = ctx.base_tables.get(name) {
            return Ok(entry.config.tick_column.clone());
        }

        if let Some(cfg) = ctx.get_time_series_config(name) {
            return Ok(cfg.tick_column.clone());
        }
    }

    if let Some(default_tick) = &ctx.default_tick_column {
        return Ok(default_tick.clone());
    }

    Err(EvalError::Other(format!(
        ".{method}() requires tick column configuration; register a time-series dataframe or set EvalContext::with_default_tick_column(...)"
    )))
}

fn eval_groupby_method(
    gb: LazyGroupBy,
    lineage: DataFrameLineage,
    method: &str,
    args: &[CoreArg],
    ctx: &EvalContext,
) -> Result<Value> {
    match method {
        "agg" => {
            let exprs = collect_expr_args(args, ctx)?;
            Ok(Value::DataFrame(gb.agg(exprs), lineage.derived())) // agg produces new shape
        }
        _ => Err(EvalError::UnknownMethod {
            target: "GroupBy".to_string(),
            method: method.to_string(),
        }),
    }
}

fn eval_expr_method(
    e: polars::prelude::Expr,
    method: &str,
    args: &[CoreArg],
    ctx: &EvalContext,
) -> Result<Value> {
    match method {
        "alias" => {
            let name = get_string_arg(args, 0, "alias")?;
            Ok(Value::Expr(e.alias(&name)))
        }
        "over" => {
            let partition_cols = get_strings_arg(args, 0, "over")?;
            let partition_exprs: Vec<_> = partition_cols.iter().map(col).collect();
            Ok(Value::Expr(e.over(partition_exprs)))
        }
        "is_between" => {
            let low = eval_to_expr(get_positional_arg(args, 0, "is_between")?, ctx)?;
            let high = eval_to_expr(get_positional_arg(args, 1, "is_between")?, ctx)?;
            Ok(Value::Expr(e.is_between(low, high, ClosedInterval::Both)))
        }
        "diff" => Ok(Value::Expr(e.diff(lit(1), NullBehavior::Ignore))),
        "shift" => {
            let n = get_int_arg(args, 0, "shift")?;
            Ok(Value::Expr(e.shift(lit(n))))
        }
        "sum" => Ok(Value::Expr(e.sum())),
        "mean" => Ok(Value::Expr(e.mean())),
        "min" => Ok(Value::Expr(e.min())),
        "max" => Ok(Value::Expr(e.max())),
        "count" => Ok(Value::Expr(e.count())),
        "first" => Ok(Value::Expr(e.first())),
        "last" => Ok(Value::Expr(e.last())),
        "cast" => {
            // Simple cast support - just integers and floats for now
            let type_name = get_string_arg(args, 0, "cast")?;
            let dtype = match type_name.as_str() {
                "int" | "i64" => DataType::Int64,
                "float" | "f64" => DataType::Float64,
                "str" | "string" => DataType::String,
                "bool" => DataType::Boolean,
                _ => {
                    return Err(EvalError::ArgError(format!(
                        "Unknown type for cast: {type_name}"
                    )));
                }
            };
            Ok(Value::Expr(e.cast(dtype)))
        }
        "fill_null" => {
            let fill_val = eval_to_expr(get_positional_arg(args, 0, "fill_null")?, ctx)?;
            Ok(Value::Expr(e.fill_null(fill_val)))
        }
        "is_null" => Ok(Value::Expr(e.is_null())),
        "is_not_null" => Ok(Value::Expr(e.is_not_null())),
        "unique" => Ok(Value::Expr(e.unique())),
        "abs" => Ok(Value::Expr(e.abs())),
        "round" => {
            let decimals = get_int_arg(args, 0, "round")? as u32;
            Ok(Value::Expr(e.round(decimals, RoundMode::HalfToEven)))
        }
        "len" => Ok(Value::Expr(e.len())),
        "n_unique" => Ok(Value::Expr(e.n_unique())),
        "cum_sum" => Ok(Value::Expr(e.cum_sum(false))),
        "cum_max" => Ok(Value::Expr(e.cum_max(false))),
        "cum_min" => Ok(Value::Expr(e.cum_min(false))),
        "rank" => Ok(Value::Expr(e.rank(Default::default(), None))),
        "clip" => {
            let min_val = eval_to_expr(get_positional_arg(args, 0, "clip")?, ctx)?;
            let max_val = eval_to_expr(get_positional_arg(args, 1, "clip")?, ctx)?;
            Ok(Value::Expr(e.clip(min_val, max_val)))
        }
        "reverse" => Ok(Value::Expr(e.reverse())),
        _ => Err(EvalError::UnknownMethod {
            target: "Expr".to_string(),
            method: method.to_string(),
        }),
    }
}

fn eval_str_method(e: polars::prelude::Expr, method: &str, args: &[CoreArg]) -> Result<Value> {
    let str_ns = e.str();
    match method {
        "starts_with" => {
            let prefix = get_string_arg(args, 0, "starts_with")?;
            Ok(Value::Expr(str_ns.starts_with(lit(prefix))))
        }
        "ends_with" => {
            let suffix = get_string_arg(args, 0, "ends_with")?;
            Ok(Value::Expr(str_ns.ends_with(lit(suffix))))
        }
        "to_lowercase" => Ok(Value::Expr(str_ns.to_lowercase())),
        "to_uppercase" => Ok(Value::Expr(str_ns.to_uppercase())),
        "len_chars" => Ok(Value::Expr(str_ns.len_chars())),
        "contains" => {
            let pattern = get_string_arg(args, 0, "contains")?;
            Ok(Value::Expr(str_ns.contains(lit(pattern), false)))
        }
        "replace" => {
            let pattern = get_string_arg(args, 0, "replace")?;
            let replacement = get_string_arg(args, 1, "replace")?;
            Ok(Value::Expr(str_ns.replace(
                lit(pattern),
                lit(replacement),
                false,
            )))
        }
        "slice" => {
            let offset = get_int_arg(args, 0, "slice")?;
            let length = get_int_arg(args, 1, "slice")? as u64;
            Ok(Value::Expr(str_ns.slice(lit(offset), lit(length))))
        }
        _ => Err(EvalError::UnknownMethod {
            target: "str".to_string(),
            method: method.to_string(),
        }),
    }
}

fn eval_dt_method(e: polars::prelude::Expr, method: &str) -> Result<Value> {
    let dt_ns = e.dt();
    match method {
        "year" => Ok(Value::Expr(dt_ns.year())),
        "month" => Ok(Value::Expr(dt_ns.month())),
        "day" => Ok(Value::Expr(dt_ns.day())),
        "hour" => Ok(Value::Expr(dt_ns.hour())),
        "minute" => Ok(Value::Expr(dt_ns.minute())),
        "second" => Ok(Value::Expr(dt_ns.second())),
        _ => Err(EvalError::UnknownMethod {
            target: "dt".to_string(),
            method: method.to_string(),
        }),
    }
}

fn eval_binop(lhs: &Expr, op: BinOp, rhs: &Expr, ctx: &EvalContext) -> Result<Value> {
    let l = eval_to_expr(lhs, ctx)?;
    let r = eval_to_expr(rhs, ctx)?;

    let result = match op {
        BinOp::Add => l + r,
        BinOp::Sub => l - r,
        BinOp::Mul => l * r,
        BinOp::Div => l / r,
        BinOp::Mod => l % r,
        BinOp::Eq => l.eq(r),
        BinOp::Ne => l.neq(r),
        BinOp::Lt => l.lt(r),
        BinOp::Le => l.lt_eq(r),
        BinOp::Gt => l.gt(r),
        BinOp::Ge => l.gt_eq(r),
        BinOp::And => l.and(r),
        BinOp::Or => l.or(r),
    };

    Ok(Value::Expr(result))
}

fn eval_unaryop(op: UnaryOp, operand: &Expr, ctx: &EvalContext) -> Result<Value> {
    let e = eval_to_expr(operand, ctx)?;

    let result = match op {
        UnaryOp::Neg => lit(0) - e,
        UnaryOp::Not => e.not(),
    };

    Ok(Value::Expr(result))
}

fn eval_when_then_otherwise(
    branches: &[(Box<Expr>, Box<Expr>)],
    otherwise: &Expr,
    ctx: &EvalContext,
) -> Result<Value> {
    if branches.is_empty() {
        return Err(EvalError::Other(
            "when/then/otherwise requires at least one branch".to_string(),
        ));
    }

    let otherwise_expr = eval_to_expr(otherwise, ctx)?;

    // Handle single branch case (Then) vs multiple branches (ChainedThen)
    if branches.len() == 1 {
        let (cond, val) = &branches[0];
        let cond_expr = eval_to_expr(cond, ctx)?;
        let then_expr = eval_to_expr(val, ctx)?;
        let result = when(cond_expr).then(then_expr).otherwise(otherwise_expr);
        Ok(Value::Expr(result))
    } else {
        // Multiple branches - first one gives Then, rest give ChainedThen
        let (first_cond, first_val) = &branches[0];
        let cond_expr = eval_to_expr(first_cond, ctx)?;
        let then_expr = eval_to_expr(first_val, ctx)?;
        let first_then = when(cond_expr).then(then_expr);

        // Chain remaining branches
        let (second_cond, second_val) = &branches[1];
        let cond_expr = eval_to_expr(second_cond, ctx)?;
        let then_expr = eval_to_expr(second_val, ctx)?;
        let mut chain = first_then.when(cond_expr).then(then_expr);

        for (cond, val) in &branches[2..] {
            let cond_expr = eval_to_expr(cond, ctx)?;
            let then_expr = eval_to_expr(val, ctx)?;
            chain = chain.when(cond_expr).then(then_expr);
        }

        let result = chain.otherwise(otherwise_expr);
        Ok(Value::Expr(result))
    }
}

fn eval_to_expr(expr: &Expr, ctx: &EvalContext) -> Result<polars::prelude::Expr> {
    match eval(expr, ctx)? {
        Value::Expr(e) => Ok(e),
        Value::Scalar(s) => Ok(scalar_to_lit(s)),
        Value::DataFrame(_, _) => Err(EvalError::TypeError {
            expected: "Expr".to_string(),
            got: "DataFrame".to_string(),
        }),
        Value::GroupBy(_, _) => Err(EvalError::TypeError {
            expected: "Expr".to_string(),
            got: "GroupBy".to_string(),
        }),
        Value::PlNamespace => Err(EvalError::TypeError {
            expected: "Expr".to_string(),
            got: "pl namespace".to_string(),
        }),
    }
}

fn scalar_to_lit(s: ScalarValue) -> polars::prelude::Expr {
    match s {
        ScalarValue::String(v) => lit(v),
        ScalarValue::Int(v) => lit(v),
        ScalarValue::Float(v) => lit(v),
        ScalarValue::Bool(v) => lit(v),
        ScalarValue::Null => lit(NULL),
    }
}

fn collect_expr_args(args: &[CoreArg], ctx: &EvalContext) -> Result<Vec<polars::prelude::Expr>> {
    let mut exprs = Vec::new();

    for arg in args {
        match arg {
            Arg::Positional(e) => {
                if let Expr::List(items) = e {
                    for item in items {
                        exprs.push(eval_to_expr(item, ctx)?);
                    }
                } else {
                    exprs.push(eval_to_expr(e, ctx)?);
                }
            }
            Arg::Keyword(_, _) => {
                // Skip keyword args for collect
            }
        }
    }

    Ok(exprs)
}

fn get_positional_arg<'a>(args: &'a [CoreArg], idx: usize, fn_name: &str) -> Result<&'a Expr> {
    let mut pos_idx = 0;
    for arg in args {
        if let Arg::Positional(e) = arg {
            if pos_idx == idx {
                return Ok(e);
            }
            pos_idx += 1;
        }
    }
    Err(EvalError::ArgError(format!(
        "{fn_name}() missing required positional argument {idx}"
    )))
}

/// Try to extract column name from either:
/// - String literal: "col_name"
/// - pl.col call: pl.col("col_name") or $col_name (desugared)
fn try_extract_col_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(Literal::String(s)) => Some(s.clone()),
        // pl.col("name") -> Call(Attr(Ident("pl"), "col"), [Positional(Literal(String(name)))])
        Expr::Call(callee, args) => {
            if let Expr::Attr(base, method) = callee.as_ref()
                && method == "col"
                && let Expr::Ident(ident) = base.as_ref()
                && ident == "pl"
                && args.len() == 1
                && let Arg::Positional(arg_expr) = &args[0]
                && let Expr::Literal(Literal::String(name)) = arg_expr
            {
                Some(name.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn get_string_arg(args: &[CoreArg], idx: usize, fn_name: &str) -> Result<String> {
    let expr = get_positional_arg(args, idx, fn_name)?;
    try_extract_col_name(expr)
        .ok_or_else(|| EvalError::ArgError(format!("{fn_name}() argument {idx} must be a string")))
}

/// Get a positional arg that can be either a single string/col or a list of strings/cols
fn get_strings_arg(args: &[CoreArg], idx: usize, fn_name: &str) -> Result<Vec<String>> {
    let expr = get_positional_arg(args, idx, fn_name)?;

    // Single value (string or pl.col)
    if let Some(name) = try_extract_col_name(expr) {
        return Ok(vec![name]);
    }

    // List of strings/cols
    if let Expr::List(items) = expr {
        return items
            .iter()
            .map(|e| {
                try_extract_col_name(e).ok_or_else(|| {
                    EvalError::ArgError(format!(
                        "{fn_name}() argument {idx} must be a column name or list of column names"
                    ))
                })
            })
            .collect();
    }

    Err(EvalError::ArgError(format!(
        "{fn_name}() argument {idx} must be a column name or list of column names"
    )))
}

fn get_int_arg(args: &[CoreArg], idx: usize, fn_name: &str) -> Result<i64> {
    let expr = get_positional_arg(args, idx, fn_name)?;
    match expr {
        Expr::Literal(Literal::Int(n)) => Ok(*n),
        // Handle negative integers: -3 parses as UnaryOp(Neg, Int(3))
        Expr::UnaryOp(crate::ast::UnaryOp::Neg, inner) => {
            if let Expr::Literal(Literal::Int(n)) = inner.as_ref() {
                Ok(-n)
            } else {
                Err(EvalError::ArgError(format!(
                    "{fn_name}() argument {idx} must be an integer"
                )))
            }
        }
        _ => Err(EvalError::ArgError(format!(
            "{fn_name}() argument {idx} must be an integer"
        ))),
    }
}

fn get_kwarg_bool(args: &[CoreArg], name: &str) -> Option<bool> {
    for arg in args {
        if let Arg::Keyword(k, v) = arg
            && k == name
            && let Expr::Literal(Literal::Bool(b)) = v
        {
            return Some(*b);
        }
    }
    None
}

fn get_kwarg_string(args: &[CoreArg], name: &str) -> Option<String> {
    for arg in args {
        if let Arg::Keyword(k, v) = arg
            && k == name
        {
            return try_extract_col_name(v);
        }
    }
    None
}

/// Get a kwarg that can be either a single string/col or a list of strings/cols
fn get_kwarg_strings(args: &[CoreArg], name: &str) -> Option<Vec<String>> {
    for arg in args {
        if let Arg::Keyword(k, v) = arg
            && k == name
        {
            // Single value
            if let Some(s) = try_extract_col_name(v) {
                return Some(vec![s]);
            }
            // List of values
            if let Expr::List(items) = v {
                let strings: Option<Vec<_>> = items.iter().map(try_extract_col_name).collect();
                return strings;
            }
            return None;
        }
    }
    None
}

fn collect_string_args(args: &[CoreArg]) -> Result<Vec<String>> {
    let mut strings = Vec::new();
    for arg in args {
        if let Arg::Positional(e) = arg {
            // Check if it's a list of strings: ["a", "b", "c"]
            if let Expr::List(items) = e {
                for item in items {
                    if let Some(name) = try_extract_col_name(item) {
                        strings.push(name);
                    } else {
                        return Err(EvalError::ArgError(
                            "Expected column name or pl.col() expression in list".to_string(),
                        ));
                    }
                }
            } else if let Some(name) = try_extract_col_name(e) {
                strings.push(name);
            } else {
                return Err(EvalError::ArgError(
                    "Expected column name or pl.col() expression".to_string(),
                ));
            }
        }
    }
    Ok(strings)
}

// ============ Sanity Tests ============
// Most testing is done via integration tests in tests/integration.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn eval_ident_lookup() {
        let df = df! { "x" => &[1, 2, 3] }.unwrap().lazy();
        let ctx = EvalContext::new().with_df("test", df);

        // Known df returns DataFrame
        let result = eval(&Expr::Ident("test".to_string()), &ctx).unwrap();
        assert!(matches!(result, Value::DataFrame(_, _)));

        // "pl" returns namespace
        let result = eval(&Expr::Ident("pl".to_string()), &ctx).unwrap();
        assert!(matches!(result, Value::PlNamespace));

        // Unknown ident is error
        let result = eval(&Expr::Ident("unknown".to_string()), &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn eval_pl_col() {
        let ctx = EvalContext::new();
        let query = Expr::Ident("pl".to_string())
            .attr("col")
            .call(vec![CoreArg::pos(Expr::Literal(Literal::String(
                "x".to_string(),
            )))]);

        let result = eval(&query, &ctx).unwrap();
        assert!(matches!(result, Value::Expr(_)));
    }
}
