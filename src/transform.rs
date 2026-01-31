//! Transform surface AST to core AST
//!
//! This pass:
//! - Recognizes when/then/otherwise chains and converts to WhenThenOtherwise
//! - (Future) Expands sugar like $col, @now, etc.

use crate::ast::Arg;
use crate::ast::core::{CoreArg, Expr as CoreExpr};
use crate::ast::surface::{Expr as SurfaceExpr, SurfaceArg};

/// Transform a surface AST into a core AST
pub fn transform(expr: SurfaceExpr) -> CoreExpr {
    transform_expr(expr)
}

fn transform_expr(expr: SurfaceExpr) -> CoreExpr {
    match expr {
        SurfaceExpr::Ident(s) => CoreExpr::Ident(s),
        SurfaceExpr::Literal(lit) => CoreExpr::Literal(lit),
        SurfaceExpr::List(items) => CoreExpr::List(items.into_iter().map(transform_expr).collect()),
        SurfaceExpr::Attr(base, name) => CoreExpr::Attr(Box::new(transform_expr(*base)), name),
        SurfaceExpr::BinaryOp(lhs, op, rhs) => CoreExpr::BinaryOp(
            Box::new(transform_expr(*lhs)),
            op,
            Box::new(transform_expr(*rhs)),
        ),
        SurfaceExpr::UnaryOp(op, operand) => {
            CoreExpr::UnaryOp(op, Box::new(transform_expr(*operand)))
        }
        SurfaceExpr::Call(callee, args) => {
            // Check for .otherwise() pattern - signals end of when chain
            if let SurfaceExpr::Attr(ref base, ref method) = *callee
                && method == "otherwise"
                    && let Some(when_chain) = try_extract_when_chain(base) {
                        return build_when_then_otherwise(when_chain, args);
                    }
            // Normal call
            CoreExpr::Call(
                Box::new(transform_expr(*callee)),
                args.into_iter().map(transform_arg).collect(),
            )
        }
    }
}

fn transform_arg(arg: SurfaceArg) -> CoreArg {
    match arg {
        Arg::Positional(e) => Arg::Positional(transform_expr(e)),
        Arg::Keyword(name, e) => Arg::Keyword(name, transform_expr(e)),
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
            && let SurfaceExpr::Attr(when_expr, method) = then_callee.as_ref() {
                if method != "then" {
                    return None;
                }
                let then_value = get_single_positional_arg(then_args)?;

                // Now check what's before .then()
                // It should be Call(Attr(_, "when"), [condition])
                if let SurfaceExpr::Call(when_callee, when_args) = when_expr.as_ref()
                    && let SurfaceExpr::Attr(inner, when_method) = when_callee.as_ref() {
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
        && let Arg::Positional(e) = &args[0] {
            return Some(e);
        }
    None
}

fn is_pl_ident(expr: &SurfaceExpr) -> bool {
    matches!(expr, SurfaceExpr::Ident(s) if s == "pl")
}

fn build_when_then_otherwise(
    chain: Vec<WhenThenPair>,
    otherwise_args: Vec<SurfaceArg>,
) -> CoreExpr {
    let otherwise_value = otherwise_args
        .into_iter()
        .find_map(|arg| {
            if let Arg::Positional(e) = arg {
                Some(e)
            } else {
                None
            }
        })
        .expect("otherwise() requires an argument");

    let branches = chain
        .into_iter()
        .map(|pair| {
            (
                Box::new(transform_expr(pair.condition)),
                Box::new(transform_expr(pair.then_value)),
            )
        })
        .collect();

    CoreExpr::WhenThenOtherwise {
        branches,
        otherwise: Box::new(transform_expr(otherwise_value)),
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
