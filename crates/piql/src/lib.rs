//! PiQL - Polars-inspired Query Language
//!
//! A text query language for interacting with Polars dataframes.
//!
//! ## Quick Start
//!
//! ```ignore
//! use piql::{QueryEngine, TimeSeriesConfig};
//!
//! let mut engine = QueryEngine::new();
//! engine.add_time_series_df("entities", df, TimeSeriesConfig {
//!     tick_column: "tick".into(),
//!     partition_key: "entity_id".into(),
//! });
//!
//! // Register custom directives
//! engine.sugar().register_directive("merchant", |_, _| { /* ... */ });
//!
//! // Materialized intermediate results
//! engine.materialize("merchants", "entities.filter(@merchant)")?;
//!
//! // Subscribe to queries
//! engine.subscribe("top_merchants", "merchants.filter(@now).top(10, 'gold')");
//!
//! // Each tick
//! let results = engine.on_tick(current_tick)?;
//! ```
//!
//! ## Standalone Usage
//!
//! For one-off queries without the engine:
//!
//! ```ignore
//! use piql::{run, EvalContext};
//!
//! let ctx = EvalContext::new()
//!     .with_df("entities", df)
//!     .with_tick(1000);
//!
//! let result = run(r#"entities.filter($gold > 100)"#, &ctx)?;
//! ```
//!
//! ## Sugar Syntax
//!
//! - `$col` → `pl.col("col")`
//! - `$col.delta` → `col.diff().over(partition)`
//! - `@directive(args)` → custom filter (registered at runtime)
//! - `.window(a, b)`, `.since(n)`, `.at(n)`, `.all()` → time scope
//! - `.top(n, col)` → sort descending + head

mod ast;
mod engine;
mod eval;
mod parse;
#[doc(hidden)]
mod sugar;
mod transform;

use thiserror::Error;

// ============ Primary Public API ============

pub use engine::QueryEngine;
pub use eval::{DataFrameEntry, EvalContext, TimeSeriesConfig, Value};

/// Run a one-off query
pub fn run(query: &str, ctx: &EvalContext) -> Result<Value, PiqlError> {
    let surface = parse::parse(query)?;
    let sugar_ctx = ctx.sugar_context(None);
    let core = transform::transform_with_sugar(surface, &ctx.sugar, &sugar_ctx);
    let result = eval::eval(&core, ctx)?;
    Ok(result)
}

// ============ Errors ============

#[derive(Error, Debug)]
pub enum PiqlError {
    #[error("Parse error: {0}")]
    Parse(#[from] parse::ParseError),
    #[error("Eval error: {0}")]
    Eval(#[from] eval::EvalError),
}

pub use eval::EvalError;
pub use parse::ParseError;

// ============ Sugar System ============

pub use crate::sugar::{SugarContext, SugarRegistry};

/// Helpers for building expressions in custom directives
pub mod expr_helpers {
    pub use crate::sugar::helpers::*;
}

// Re-export BinOp for use in custom directives
pub use ast::BinOp;

// ============ Advanced: AST Access ============

/// Low-level AST types (for custom transforms or introspection)
pub mod advanced {
    pub use crate::ast::core::{CoreArg, Expr as CoreExpr};
    pub use crate::ast::surface::{Expr as SurfaceExpr, SurfaceArg};
    pub use crate::ast::{Arg, Literal, UnaryOp};
    pub use crate::eval::eval;
    pub use crate::parse::parse;
    pub use crate::transform::{transform, transform_with_sugar};
}
