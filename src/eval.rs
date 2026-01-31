//! Interpreter that evaluates PiQL AST against Polars dataframes
//!
//! Evaluates core::Expr (the transformed AST) against Polars dataframes.

use std::collections::HashMap;

use polars::prelude::*;
use polars::series::ops::NullBehavior;
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
    /// A Polars LazyFrame
    DataFrame(LazyFrame),
    /// A Polars LazyGroupBy (from group_by, before agg)
    GroupBy(LazyGroupBy),
    /// A Polars column expression
    Expr(polars::prelude::Expr),
    /// A scalar/literal value (for use in expressions)
    Scalar(ScalarValue),
    /// The `pl` namespace object
    PlNamespace,
}

#[derive(Debug, Clone)]
pub enum ScalarValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

/// Evaluation context - holds named dataframes and configuration
pub struct EvalContext {
    pub dataframes: HashMap<String, LazyFrame>,
}

impl EvalContext {
    pub fn new() -> Self {
        Self {
            dataframes: HashMap::new(),
        }
    }

    pub fn with_df(mut self, name: impl Into<String>, df: LazyFrame) -> Self {
        self.dataframes.insert(name.into(), df);
        self
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
    }
}

fn eval_ident(name: &str, ctx: &EvalContext) -> Result<Value> {
    match name {
        "pl" => Ok(Value::PlNamespace),
        _ => {
            if let Some(df) = ctx.dataframes.get(name) {
                Ok(Value::DataFrame(df.clone()))
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
        Value::DataFrame(_) => Err(EvalError::UnknownMethod {
            target: "DataFrame".to_string(),
            method: attr.to_string(),
        }),
        Value::GroupBy(_) => Err(EvalError::UnknownMethod {
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

    match base_val {
        Value::PlNamespace => eval_pl_function(method, args, ctx),
        Value::DataFrame(df) => eval_df_method(df, method, args, ctx),
        Value::GroupBy(gb) => eval_groupby_method(gb, method, args, ctx),
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
            let col_name = get_string_arg(args, 0, "col")?;
            Ok(Value::Expr(col(&col_name)))
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
        _ => Err(EvalError::UnknownMethod {
            target: "pl".to_string(),
            method: name.to_string(),
        }),
    }
}

fn eval_df_method(
    df: LazyFrame,
    method: &str,
    args: &[CoreArg],
    ctx: &EvalContext,
) -> Result<Value> {
    match method {
        "filter" => {
            let pred = get_positional_arg(args, 0, "filter")?;
            let pred_expr = eval_to_expr(pred, ctx)?;
            Ok(Value::DataFrame(df.filter(pred_expr)))
        }
        "select" => {
            let exprs = collect_expr_args(args, ctx)?;
            Ok(Value::DataFrame(df.select(exprs)))
        }
        "with_columns" => {
            let exprs = collect_expr_args(args, ctx)?;
            Ok(Value::DataFrame(df.with_columns(exprs)))
        }
        "head" => {
            let n = get_int_arg(args, 0, "head").unwrap_or(10) as u32;
            Ok(Value::DataFrame(df.limit(n)))
        }
        "sort" => {
            let col_name = get_string_arg(args, 0, "sort")?;
            let descending = get_kwarg_bool(args, "descending").unwrap_or(false);
            let opts = SortMultipleOptions::new().with_order_descending(descending);
            Ok(Value::DataFrame(df.sort([&col_name], opts)))
        }
        "tail" => {
            let n = get_int_arg(args, 0, "tail").unwrap_or(10) as u32;
            Ok(Value::DataFrame(df.tail(n)))
        }
        "drop" => {
            let col_names = collect_string_args(args)?;
            Ok(Value::DataFrame(df.drop(col_names)))
        }
        "explode" => {
            let col_names = collect_string_args(args)?;
            let col_exprs: Vec<_> = col_names.iter().map(col).collect();
            Ok(Value::DataFrame(df.explode(col_exprs)))
        }
        "group_by" => {
            let col_names = collect_string_args(args)?;
            let col_exprs: Vec<_> = col_names.iter().map(col).collect();
            Ok(Value::GroupBy(df.group_by(col_exprs)))
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
                Ok(Value::DataFrame(df.rename(old_names, new_names, false)))
            } else {
                // Positional fallback: rename("old", "new")
                let old = get_string_arg(args, 0, "rename")?;
                let new = get_string_arg(args, 1, "rename")?;
                Ok(Value::DataFrame(df.rename([old], [new], false)))
            }
        }
        "join" => {
            // Get the other dataframe (first positional arg)
            let other_expr = get_positional_arg(args, 0, "join")?;
            let other = match eval(other_expr, ctx)? {
                Value::DataFrame(lf) => lf,
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

            // Get join columns
            let result = if let Some(on) = get_kwarg_string(args, "on") {
                // Same column name on both sides
                df.join(other, [col(&on)], [col(&on)], JoinArgs::new(join_type))
            } else {
                // Different column names
                let left_on = get_kwarg_string(args, "left_on").ok_or_else(|| {
                    EvalError::ArgError(
                        "join() requires 'on' or 'left_on'/'right_on' kwargs".to_string(),
                    )
                })?;
                let right_on = get_kwarg_string(args, "right_on").ok_or_else(|| {
                    EvalError::ArgError(
                        "join() requires 'right_on' when 'left_on' is specified".to_string(),
                    )
                })?;
                df.join(
                    other,
                    [col(&left_on)],
                    [col(&right_on)],
                    JoinArgs::new(join_type),
                )
            };

            Ok(Value::DataFrame(result))
        }
        _ => Err(EvalError::UnknownMethod {
            target: "DataFrame".to_string(),
            method: method.to_string(),
        }),
    }
}

fn eval_groupby_method(
    gb: LazyGroupBy,
    method: &str,
    args: &[CoreArg],
    ctx: &EvalContext,
) -> Result<Value> {
    match method {
        "agg" => {
            let exprs = collect_expr_args(args, ctx)?;
            Ok(Value::DataFrame(gb.agg(exprs)))
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
            let partition = get_string_arg(args, 0, "over")?;
            Ok(Value::Expr(e.over([col(&partition)])))
        }
        "is_between" => {
            let low = eval_to_expr(get_positional_arg(args, 0, "is_between")?, ctx)?;
            let high = eval_to_expr(get_positional_arg(args, 1, "is_between")?, ctx)?;
            Ok(Value::Expr(e.is_between(low, high, ClosedInterval::Both)))
        }
        "diff" => Ok(Value::Expr(e.diff(1, NullBehavior::Ignore))),
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
        Value::DataFrame(_) => Err(EvalError::TypeError {
            expected: "Expr".to_string(),
            got: "DataFrame".to_string(),
        }),
        Value::GroupBy(_) => Err(EvalError::TypeError {
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

fn get_string_arg(args: &[CoreArg], idx: usize, fn_name: &str) -> Result<String> {
    let expr = get_positional_arg(args, idx, fn_name)?;
    if let Expr::Literal(Literal::String(s)) = expr {
        Ok(s.clone())
    } else {
        Err(EvalError::ArgError(format!(
            "{fn_name}() argument {idx} must be a string"
        )))
    }
}

fn get_int_arg(args: &[CoreArg], idx: usize, fn_name: &str) -> Result<i64> {
    let expr = get_positional_arg(args, idx, fn_name)?;
    if let Expr::Literal(Literal::Int(n)) = expr {
        Ok(*n)
    } else {
        Err(EvalError::ArgError(format!(
            "{fn_name}() argument {idx} must be an integer"
        )))
    }
}

fn get_kwarg_bool(args: &[CoreArg], name: &str) -> Option<bool> {
    for arg in args {
        if let Arg::Keyword(k, v) = arg
            && k == name
                && let Expr::Literal(Literal::Bool(b)) = v {
                    return Some(*b);
                }
    }
    None
}

fn get_kwarg_string(args: &[CoreArg], name: &str) -> Option<String> {
    for arg in args {
        if let Arg::Keyword(k, v) = arg
            && k == name
                && let Expr::Literal(Literal::String(s)) = v {
                    return Some(s.clone());
                }
    }
    None
}

fn collect_string_args(args: &[CoreArg]) -> Result<Vec<String>> {
    let mut strings = Vec::new();
    for arg in args {
        if let Arg::Positional(e) = arg {
            if let Expr::Literal(Literal::String(s)) = e {
                strings.push(s.clone());
            } else {
                return Err(EvalError::ArgError("Expected string argument".to_string()));
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
        assert!(matches!(result, Value::DataFrame(_)));

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
