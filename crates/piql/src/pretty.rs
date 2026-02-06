//! Pretty printing for PiQL AST
//!
//! Provides both single-line (`Display`) and intelligent multi-line formatting
//! that breaks method chains when they exceed a width threshold.

use crate::ast::surface::{Expr, SurfaceArg};
use crate::ast::{Arg, BinOp, Literal, UnaryOp};
use std::fmt::{self, Display, Write};

// ============ Display (single-line) ============

impl Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::String(s) => write!(f, "\"{}\"", escape_string(s)),
            Literal::Int(n) => write!(f, "{}", n),
            Literal::Float(n) => {
                if n.is_finite() && n.fract() == 0.0 {
                    write!(f, "{n:.1}")
                } else {
                    write!(f, "{}", n)
                }
            }
            Literal::Bool(b) => {
                if *b {
                    write!(f, "True")
                } else {
                    write!(f, "False")
                }
            }
            Literal::Null => write!(f, "None"),
        }
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&",
            BinOp::Or => "|",
        };
        write!(f, "{}", s)
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "~",
        };
        write!(f, "{}", s)
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Ident(name) => write!(f, "{}", name),
            Expr::Literal(lit) => write!(f, "{}", lit),
            Expr::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Expr::Attr(base, name) => {
                // BinaryOp/UnaryOp need parens when used as method receiver
                let needs_parens = matches!(base.as_ref(), Expr::BinaryOp(..) | Expr::UnaryOp(..));
                if needs_parens {
                    write!(f, "({}).{}", base, name)
                } else {
                    write!(f, "{}.{}", base, name)
                }
            }
            Expr::Call(callee, args) => {
                write!(f, "{}(", callee)?;
                write_args(f, args)?;
                write!(f, ")")
            }
            Expr::BinaryOp(lhs, op, rhs) => {
                let needs_parens_lhs = matches!(lhs.as_ref(), Expr::BinaryOp(..));
                let needs_parens_rhs = matches!(rhs.as_ref(), Expr::BinaryOp(..));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " {} ", op)?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)?;
                } else {
                    write!(f, "{}", rhs)?;
                }
                Ok(())
            }
            Expr::UnaryOp(op, expr) => {
                let needs_parens = matches!(expr.as_ref(), Expr::BinaryOp(..));
                if needs_parens {
                    write!(f, "{}({})", op, expr)
                } else {
                    write!(f, "{}{}", op, expr)
                }
            }
            Expr::ColShorthand(name) => write!(f, "${}", name),
            Expr::Directive(name, args) => {
                write!(f, "@{}", name)?;
                if !args.is_empty() {
                    write!(f, "(")?;
                    write_args(f, args)?;
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }
}

impl<E: Display> Display for Arg<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Positional(expr) => write!(f, "{}", expr),
            Arg::Keyword(name, expr) => write!(f, "{}={}", name, expr),
        }
    }
}

fn write_args(f: &mut fmt::Formatter<'_>, args: &[SurfaceArg]) -> fmt::Result {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    Ok(())
}

fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

// ============ Intelligent line breaking ============

/// A segment of a method chain: either the base expression or a method call
#[derive(Debug)]
enum ChainSegment<'a> {
    /// Base expression (leftmost)
    Base(&'a Expr),
    /// Attribute access: `.name`
    Attr(&'a str),
    /// Method call: `.name(args)`
    Call(&'a str, &'a [SurfaceArg]),
}

/// Flatten a method chain into segments
fn flatten_chain(expr: &Expr) -> Vec<ChainSegment<'_>> {
    let mut segments = Vec::new();
    collect_chain(expr, &mut segments);
    segments
}

fn collect_chain<'a>(expr: &'a Expr, segments: &mut Vec<ChainSegment<'a>>) {
    match expr {
        Expr::Call(callee, args) => {
            if let Expr::Attr(base, name) = callee.as_ref() {
                collect_chain(base, segments);
                segments.push(ChainSegment::Call(name, args));
            } else {
                // Not a method call, treat as base
                segments.push(ChainSegment::Base(expr));
            }
        }
        Expr::Attr(base, name) => {
            collect_chain(base, segments);
            segments.push(ChainSegment::Attr(name));
        }
        _ => {
            segments.push(ChainSegment::Base(expr));
        }
    }
}

fn format_args(args: &[SurfaceArg]) -> String {
    args.iter()
        .map(|a| a.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Pretty print an expression with intelligent line breaking
///
/// Method chains that exceed `width` characters will be broken across lines.
pub fn pretty(expr: &Expr, width: usize) -> String {
    let one_line = expr.to_string();

    // If it fits, use single line
    if one_line.len() <= width {
        return one_line;
    }

    // Try to break method chains
    let segments = flatten_chain(expr);

    // Only break if we have a real chain (more than just base)
    if segments.len() <= 1 {
        return one_line;
    }

    // Build multi-line version
    let mut result = String::new();
    for (i, seg) in segments.iter().enumerate() {
        match seg {
            ChainSegment::Base(base) => {
                // Base might itself be complex, recursively pretty print
                write!(result, "{}", pretty(base, width)).unwrap();
            }
            ChainSegment::Attr(name) => {
                // First method (i==1) stays on same line as base, rest break
                if i > 1 {
                    result.push_str("\n    ");
                }
                write!(result, ".{}", name).unwrap();
            }
            ChainSegment::Call(name, args) => {
                if i > 1 {
                    result.push_str("\n    ");
                }
                let args_str = format_args(args);

                // If args are long, consider breaking them too
                let call_line = format!(".{}({})", name, args_str);
                if call_line.len() > width - 4 && args.len() > 1 {
                    // Break args across lines
                    write!(result, ".{}(\n        ", name).unwrap();
                    for (j, arg) in args.iter().enumerate() {
                        if j > 0 {
                            result.push_str(",\n        ");
                        }
                        // Recursively pretty print arg expressions
                        let arg_str = match arg {
                            Arg::Positional(e) => pretty(e, width.saturating_sub(8)),
                            Arg::Keyword(k, e) => {
                                format!(
                                    "{}={}",
                                    k,
                                    pretty(e, width.saturating_sub(8 + k.len() + 1))
                                )
                            }
                        };
                        result.push_str(&arg_str);
                    }
                    result.push_str("\n    )");
                } else {
                    result.push_str(&call_line);
                }
            }
        }
    }

    result
}

impl Expr {
    /// Pretty print with intelligent line breaking at the given width
    pub fn pretty(&self, width: usize) -> String {
        pretty(self, width)
    }
}

#[cfg(test)]
mod tests {
    use crate::parse::parse;

    #[test]
    fn test_display_simple() {
        let expr = parse("df.filter($gold > 100)").unwrap();
        assert_eq!(expr.to_string(), "df.filter($gold > 100)");
    }

    #[test]
    fn test_display_directive() {
        let expr = parse("@merchant").unwrap();
        assert_eq!(expr.to_string(), "@merchant");

        let expr = parse("@entity(42)").unwrap();
        assert_eq!(expr.to_string(), "@entity(42)");
    }

    #[test]
    fn test_display_list() {
        let expr = parse(r#"df.select(["a", "b"])"#).unwrap();
        assert_eq!(expr.to_string(), r#"df.select(["a", "b"])"#);
    }

    #[test]
    fn test_pretty_short_chain() {
        let expr = parse("df.filter($x > 1).head(10)").unwrap();
        // Short enough, stays on one line
        assert_eq!(expr.pretty(80), "df.filter($x > 1).head(10)");
    }

    #[test]
    fn test_pretty_long_chain() {
        let expr =
            parse("df.filter($gold > 100).select($name, $gold, $level).sort($gold).head(10)")
                .unwrap();
        let pretty = expr.pretty(40);
        assert!(
            pretty.contains('\n'),
            "Should break into multiple lines: {}",
            pretty
        );
        assert!(pretty.contains(".filter("), "Should have filter");
        assert!(pretty.contains(".select("), "Should have select");

        // Verify the structure: first method on same line as base
        let lines: Vec<&str> = pretty.lines().collect();
        assert!(lines[0].starts_with("df.filter("), "First method on same line: {}", pretty);
        assert!(lines[1].trim().starts_with(".select("));
    }

    #[test]
    fn demo_pretty_output() {
        let queries = [
            ("short", "df.filter($x > 1).head(10)"),
            (
                "long chain",
                "df.filter($gold > 100).select($name, $gold, $level).sort($gold).head(10)",
            ),
            (
                "many args",
                "df.with_columns($a.alias(\"x\"), $b.alias(\"y\"), $c.alias(\"z\"), $d.alias(\"w\"))",
            ),
        ];
        for (name, q) in queries {
            let expr = parse(q).unwrap();
            println!("\n=== {} ===", name);
            println!("Input:  {}", q);
            println!("Width 80:\n{}", expr.pretty(80));
            println!("Width 40:\n{}", expr.pretty(40));
        }
    }

    #[test]
    fn test_binop_method_call_parens() {
        // BinaryOp as method receiver needs parentheses
        let expr = parse(r#"(pl.col("a") - pl.col("b")).alias("diff")"#).unwrap();
        let s = expr.to_string();
        assert!(
            s.contains("(pl.col(\"a\") - pl.col(\"b\")).alias"),
            "Should preserve parens: {}",
            s
        );

        // Verify round-trip
        let reparsed = parse(&s).unwrap();
        assert_eq!(expr.to_string(), reparsed.to_string());
    }

    #[test]
    fn test_pretty_preserves_semantics() {
        let queries = [
            "df.filter($x > 1)",
            "$col.delta(3)",
            "pl.col(\"a\", \"b\")",
            "df.filter($x > 1 & $y < 2)",
            "(pl.col(\"a\") - pl.col(\"b\")).alias(\"diff\")",
        ];
        for q in queries {
            let expr = parse(q).unwrap();
            let pretty = expr.pretty(80);
            // Should be able to re-parse the pretty output
            let reparsed = parse(&pretty).unwrap();
            assert_eq!(
                expr.to_string(),
                reparsed.to_string(),
                "Round-trip failed for: {}",
                q
            );
        }
    }

    #[test]
    fn literal_display_round_trip() {
        for q in ["True", "False", "None", "1.0"] {
            let expr = parse(q).unwrap();
            let printed = expr.to_string();
            let reparsed = parse(&printed).unwrap();
            assert_eq!(expr, reparsed, "round trip failed for: {q}");
        }
    }
}
