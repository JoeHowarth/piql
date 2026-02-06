//! Transform surface AST to core AST
//!
//! This pass:
//! - Recognizes when/then/otherwise chains and converts to WhenThenOtherwise
//! - Expands sugar: $col, @directive, $col.method

use crate::ast::Arg;
use crate::ast::core::{CoreArg, Expr as CoreExpr};
use crate::ast::surface::{Expr as SurfaceExpr, SurfaceArg};
use crate::sugar::{SugarContext, SugarRegistry};

/// Transform a surface AST into a core AST (without sugar registry)
pub fn transform(expr: SurfaceExpr) -> CoreExpr {
    let registry = SugarRegistry::new();
    let ctx = SugarContext::new();
    transform_with_sugar(expr, &registry, &ctx)
}

/// Transform a surface AST into a core AST with sugar expansion
pub fn transform_with_sugar(
    expr: SurfaceExpr,
    registry: &SugarRegistry,
    ctx: &SugarContext,
) -> CoreExpr {
    transform_expr(expr, registry, ctx)
}

fn transform_expr(expr: SurfaceExpr, registry: &SugarRegistry, ctx: &SugarContext) -> CoreExpr {
    match expr {
        SurfaceExpr::Ident(s) => CoreExpr::Ident(s),
        SurfaceExpr::Literal(lit) => CoreExpr::Literal(lit),
        SurfaceExpr::List(items) => CoreExpr::List(
            items
                .into_iter()
                .map(|e| transform_expr(e, registry, ctx))
                .collect(),
        ),
        SurfaceExpr::Attr(base, name) => {
            // Check for $col.method pattern (no args - like $col.delta)
            if let SurfaceExpr::ColShorthand(ref col_name) = *base
                && let Some(expanded) =
                    registry.expand_col_method(build_pl_col(col_name), &name, &[], ctx)
            {
                return expanded;
            }
            CoreExpr::Attr(Box::new(transform_expr(*base, registry, ctx)), name)
        }
        SurfaceExpr::BinaryOp(lhs, op, rhs) => CoreExpr::BinaryOp(
            Box::new(transform_expr(*lhs, registry, ctx)),
            op,
            Box::new(transform_expr(*rhs, registry, ctx)),
        ),
        SurfaceExpr::UnaryOp(op, operand) => {
            CoreExpr::UnaryOp(op, Box::new(transform_expr(*operand, registry, ctx)))
        }
        // Sugar: $col -> pl.col("col")
        SurfaceExpr::ColShorthand(name) => build_pl_col(&name),
        // Sugar: @directive(args) -> expanded via registry
        SurfaceExpr::Directive(name, args) => {
            let core_args: Vec<CoreArg> = args
                .into_iter()
                .map(|a| transform_arg(a, registry, ctx))
                .collect();
            registry
                .expand_directive(&name, &core_args, ctx)
                .unwrap_or_else(|| CoreExpr::Invalid(format!("Unknown directive: @{name}")))
        }
        SurfaceExpr::Call(callee, args) => {
            // Check for .otherwise() pattern - signals end of when chain
            if let SurfaceExpr::Attr(ref base, ref method) = *callee
                && method == "otherwise"
                && let Some(when_chain) = try_extract_when_chain(base)
            {
                return build_when_then_otherwise(when_chain, args, registry, ctx);
            }

            // Check for $col.method() pattern - sugar for col methods
            if let SurfaceExpr::Attr(ref base, ref method) = *callee
                && let SurfaceExpr::ColShorthand(ref col_name) = **base
            {
                let core_args: Vec<CoreArg> = args
                    .into_iter()
                    .map(|a| transform_arg(a, registry, ctx))
                    .collect();

                // Try col method handler first
                if let Some(expanded) =
                    registry.expand_col_method(build_pl_col(col_name), method, &core_args, ctx)
                {
                    return expanded;
                }

                // Fall through to normal method call on expanded col
                return CoreExpr::Call(
                    Box::new(CoreExpr::Attr(
                        Box::new(build_pl_col(col_name)),
                        method.clone(),
                    )),
                    core_args,
                );
            }

            // Normal call
            CoreExpr::Call(
                Box::new(transform_expr(*callee, registry, ctx)),
                args.into_iter()
                    .map(|a| transform_arg(a, registry, ctx))
                    .collect(),
            )
        }
    }
}

fn transform_arg(arg: SurfaceArg, registry: &SugarRegistry, ctx: &SugarContext) -> CoreArg {
    match arg {
        Arg::Positional(e) => Arg::Positional(transform_expr(e, registry, ctx)),
        Arg::Keyword(name, e) => Arg::Keyword(name, transform_expr(e, registry, ctx)),
    }
}

/// A when/then pair extracted from the chain
struct WhenThenPair {
    condition: SurfaceExpr,
    then_value: SurfaceExpr,
}

/// Try to extract a when/then chain from an expression
/// Returns None if the expression doesn't match the pattern
fn try_extract_when_chain(expr: &SurfaceExpr) -> Option<Vec<WhenThenPair>> {
    let mut pairs = Vec::new();
    let mut current = expr;

    loop {
        // Expect: Call(Attr(inner, "then"), [then_value])
        if let SurfaceExpr::Call(then_callee, then_args) = current
            && let SurfaceExpr::Attr(when_expr, method) = then_callee.as_ref()
        {
            if method != "then" {
                return None;
            }
            let then_value = get_single_positional_arg(then_args)?;

            // Now check what's before .then()
            // It should be Call(Attr(_, "when"), [condition])
            if let SurfaceExpr::Call(when_callee, when_args) = when_expr.as_ref()
                && let SurfaceExpr::Attr(inner, when_method) = when_callee.as_ref()
            {
                if when_method != "when" {
                    return None;
                }
                let condition = get_single_positional_arg(when_args)?;

                pairs.push(WhenThenPair {
                    condition: condition.clone(),
                    then_value: then_value.clone(),
                });

                // Check if inner is pl.when (root) or another then (chained)
                if is_pl_ident(inner) {
                    // We've reached the root: pl.when(...)
                    pairs.reverse();
                    return Some(pairs);
                } else {
                    // Continue up the chain
                    current = inner;
                    continue;
                }
            }
        }
        return None;
    }
}

fn get_single_positional_arg(args: &[SurfaceArg]) -> Option<&SurfaceExpr> {
    if args.len() == 1
        && let Arg::Positional(e) = &args[0]
    {
        return Some(e);
    }
    None
}

fn is_pl_ident(expr: &SurfaceExpr) -> bool {
    matches!(expr, SurfaceExpr::Ident(s) if s == "pl")
}

/// Build pl.col("name") as CoreExpr
fn build_pl_col(name: &str) -> CoreExpr {
    use crate::ast::Arg;
    use crate::ast::Literal;

    // pl.col("name")
    CoreExpr::Call(
        Box::new(CoreExpr::Attr(
            Box::new(CoreExpr::Ident("pl".into())),
            "col".into(),
        )),
        vec![Arg::Positional(CoreExpr::Literal(Literal::String(
            name.into(),
        )))],
    )
}

fn build_when_then_otherwise(
    chain: Vec<WhenThenPair>,
    otherwise_args: Vec<SurfaceArg>,
    registry: &SugarRegistry,
    ctx: &SugarContext,
) -> CoreExpr {
    let Some(otherwise_value) = otherwise_args
        .into_iter()
        .find_map(|arg| {
            if let Arg::Positional(e) = arg {
                Some(e)
            } else {
                None
            }
        })
    else {
        return CoreExpr::Invalid("otherwise() requires an argument".to_string());
    };

    let branches = chain
        .into_iter()
        .map(|pair| {
            (
                Box::new(transform_expr(pair.condition, registry, ctx)),
                Box::new(transform_expr(pair.then_value, registry, ctx)),
            )
        })
        .collect();

    CoreExpr::WhenThenOtherwise {
        branches,
        otherwise: Box::new(transform_expr(otherwise_value, registry, ctx)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    #[test]
    fn transform_simple_expr() {
        let surface = parse("x + y").unwrap();
        let core = transform(surface);
        assert!(matches!(core, CoreExpr::BinaryOp(_, _, _)));
    }

    #[test]
    fn transform_when_then_otherwise() {
        let surface =
            parse(r#"pl.when(pl.col("x") > 0).then(pl.lit("pos")).otherwise(pl.lit("neg"))"#)
                .unwrap();
        let core = transform(surface);

        if let CoreExpr::WhenThenOtherwise { branches, .. } = core {
            assert_eq!(branches.len(), 1);
        } else {
            panic!("Expected WhenThenOtherwise, got {:?}", core);
        }
    }

    #[test]
    fn transform_chained_when() {
        let surface = parse(r#"pl.when(a > 10).then(x).when(a > 5).then(y).otherwise(z)"#).unwrap();
        let core = transform(surface);

        if let CoreExpr::WhenThenOtherwise { branches, .. } = core {
            assert_eq!(branches.len(), 2);
        } else {
            panic!("Expected WhenThenOtherwise, got {:?}", core);
        }
    }

    #[test]
    fn transform_non_when_chain_unchanged() {
        // A normal method chain should pass through unchanged
        let surface = parse(r#"df.filter(x).select(y)"#).unwrap();
        let core = transform(surface);
        assert!(matches!(core, CoreExpr::Call(_, _)));
    }
}
