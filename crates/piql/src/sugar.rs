//! Sugar system for PiQL
//!
//! Provides:
//! - SugarContext: Runtime values for sugar expansion (tick, partition_key)
//! - SugarRegistry: Handlers for @directives and $col.method sugar

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::core::{CoreArg, Expr as CoreExpr};
use crate::ast::{Arg, BinOp, Literal};

/// Context available during sugar expansion
#[derive(Debug, Clone, Default)]
pub struct SugarContext {
    /// Current simulation tick (for @now, @window, etc.)
    pub tick: Option<i64>,
    /// Partition key for windowed operations (from current DF's TimeSeriesConfig)
    pub partition_key: Option<String>,
}

impl SugarContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tick(mut self, tick: i64) -> Self {
        self.tick = Some(tick);
        self
    }

    pub fn with_partition_key(mut self, key: impl Into<String>) -> Self {
        self.partition_key = Some(key.into());
        self
    }
}

/// Handler for @directive(args) sugar
pub type DirectiveHandler =
    Arc<dyn Fn(&[CoreArg], &SugarContext) -> CoreExpr + Send + Sync + 'static>;

/// Handler for $col.method(args) sugar
pub type ColMethodHandler =
    Arc<dyn Fn(CoreExpr, &[CoreArg], &SugarContext) -> CoreExpr + Send + Sync + 'static>;

/// Registry of sugar handlers
#[derive(Default, Clone)]
pub struct SugarRegistry {
    /// @directive handlers by name
    directives: HashMap<String, DirectiveHandler>,
    /// $col.method handlers by method name
    col_methods: HashMap<String, ColMethodHandler>,
}

impl SugarRegistry {
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_col_methods();
        registry
    }

    /// Register a custom @directive handler
    pub fn register_directive<F>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(&[CoreArg], &SugarContext) -> CoreExpr + Send + Sync + 'static,
    {
        self.directives.insert(name.into(), Arc::new(handler));
    }

    /// Register a custom $col.method handler
    pub fn register_col_method<F>(&mut self, name: impl Into<String>, handler: F)
    where
        F: Fn(CoreExpr, &[CoreArg], &SugarContext) -> CoreExpr + Send + Sync + 'static,
    {
        self.col_methods.insert(name.into(), Arc::new(handler));
    }

    /// Expand a @directive(args)
    pub fn expand_directive(
        &self,
        name: &str,
        args: &[CoreArg],
        ctx: &SugarContext,
    ) -> Option<CoreExpr> {
        self.directives.get(name).map(|handler| handler(args, ctx))
    }

    /// Expand a $col.method(args)
    pub fn expand_col_method(
        &self,
        col_expr: CoreExpr,
        method: &str,
        args: &[CoreArg],
        ctx: &SugarContext,
    ) -> Option<CoreExpr> {
        self.col_methods
            .get(method)
            .map(|handler| handler(col_expr, args, ctx))
    }

    /// Check if a method name is a registered col method
    pub fn has_col_method(&self, name: &str) -> bool {
        self.col_methods.contains_key(name)
    }

    /// Register built-in $col.method handlers
    fn register_builtin_col_methods(&mut self) {
        // $col.delta -> col.diff().over(partition)
        // $col.delta(n) -> col - col.shift(n).over(partition)
        self.register_col_method("delta", |col_expr, args, ctx| {
            let partition = ctx.partition_key.as_deref().unwrap_or("entity_id");

            if args.is_empty() {
                // $col.delta -> col.diff().over(partition)
                helpers::method_call(
                    helpers::method_call(col_expr, "diff", vec![]),
                    "over",
                    vec![Arg::pos(helpers::lit_str(partition))],
                )
            } else {
                // $col.delta(n) -> col - col.shift(n).over(partition)
                let shifted = helpers::method_call(
                    helpers::method_call(col_expr.clone(), "shift", args.to_vec()),
                    "over",
                    vec![Arg::pos(helpers::lit_str(partition))],
                );
                helpers::binop(col_expr, BinOp::Sub, shifted)
            }
        });

        // $col.pct(n) -> (col - col.shift(n)) / col.shift(n), all over partition
        self.register_col_method("pct", |col_expr, args, ctx| {
            let partition = ctx.partition_key.as_deref().unwrap_or("entity_id");

            let shifted = helpers::method_call(
                helpers::method_call(col_expr.clone(), "shift", args.to_vec()),
                "over",
                vec![Arg::pos(helpers::lit_str(partition))],
            );
            let diff = helpers::binop(col_expr, BinOp::Sub, shifted.clone());
            helpers::binop(diff, BinOp::Div, shifted)
        });
    }
}

/// Helper functions for building CoreExpr nodes
pub mod helpers {
    use super::*;

    /// Build pl.col("name")
    pub fn pl_col(name: &str) -> CoreExpr {
        CoreExpr::Call(
            Box::new(CoreExpr::Attr(
                Box::new(CoreExpr::Ident("pl".into())),
                "col".into(),
            )),
            vec![Arg::pos(lit_str(name))],
        )
    }

    /// Build a string literal
    pub fn lit_str(s: &str) -> CoreExpr {
        CoreExpr::Literal(Literal::String(s.into()))
    }

    /// Build an integer literal
    pub fn lit_int(n: i64) -> CoreExpr {
        CoreExpr::Literal(Literal::Int(n))
    }

    /// Build a binary operation
    pub fn binop(left: CoreExpr, op: BinOp, right: CoreExpr) -> CoreExpr {
        CoreExpr::BinaryOp(Box::new(left), op, Box::new(right))
    }

    /// Build a method call: base.method(args)
    pub fn method_call(base: CoreExpr, method: &str, args: Vec<CoreArg>) -> CoreExpr {
        CoreExpr::Call(
            Box::new(CoreExpr::Attr(Box::new(base), method.into())),
            args,
        )
    }

    /// Extract integer from first positional arg
    pub fn get_int_arg(args: &[CoreArg], idx: usize) -> Option<i64> {
        let mut pos_idx = 0;
        for arg in args {
            if let Arg::Positional(CoreExpr::Literal(Literal::Int(n))) = arg {
                if pos_idx == idx {
                    return Some(*n);
                }
                pos_idx += 1;
            }
        }
        None
    }
}
