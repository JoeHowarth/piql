//! Parser for PiQL expressions
//!
//! Produces surface::Expr which is then transformed to core::Expr before eval.

use winnow::ascii::{digit1, multispace0};
use winnow::combinator::{alt, delimited, opt, preceded, repeat, separated, terminated};
use winnow::prelude::*;
use winnow::token::{one_of, take_while};

use crate::ast::surface::{Expr, SurfaceArg};
use crate::ast::{BinOp, Literal, UnaryOp};

type PResult<T> = winnow::ModalResult<T>;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
    pub line: usize,
    pub column: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (line {}, column {}, offset {})",
            self.message, self.line, self.column, self.offset
        )
    }
}

impl std::error::Error for ParseError {}

/// Parse a PiQL expression from a string
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let input = input.trim();
    let mut stream = input;
    match expr.parse_next(&mut stream) {
        Ok(parsed) => {
            if stream.trim().is_empty() {
                Ok(parsed)
            } else {
                let offset = trailing_input_offset(input, stream);
                Err(build_parse_error(
                    "unexpected trailing input".to_string(),
                    input,
                    offset,
                ))
            }
        }
        Err(e) => {
            let offset = input.len().saturating_sub(stream.len());
            Err(build_parse_error(format!("{:?}", e), input, offset))
        }
    }
}

fn build_parse_error(message: String, input: &str, offset: usize) -> ParseError {
    let (line, column) = offset_to_line_column(input, offset);
    ParseError {
        message,
        offset,
        line,
        column,
    }
}

fn offset_to_line_column(input: &str, offset: usize) -> (usize, usize) {
    let bounded = offset.min(input.len());
    let mut line = 1usize;
    let mut column = 1usize;

    for ch in input[..bounded].chars() {
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }

    (line, column)
}

fn trailing_input_offset(input: &str, trailing: &str) -> usize {
    let base = input.len().saturating_sub(trailing.len());
    let non_ws = trailing
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    base + non_ws
}

// ============ Top-level expression (handles precedence) ============

fn expr(input: &mut &str) -> PResult<Expr> {
    or_expr.parse_next(input)
}

fn or_expr(input: &mut &str) -> PResult<Expr> {
    let first = and_expr.parse_next(input)?;
    let rest: Vec<Expr> = repeat(0.., preceded((ws, '|', ws), and_expr)).parse_next(input)?;
    Ok(rest.into_iter().fold(first, |l, r| {
        Expr::BinaryOp(Box::new(l), BinOp::Or, Box::new(r))
    }))
}

fn and_expr(input: &mut &str) -> PResult<Expr> {
    let first = cmp_expr.parse_next(input)?;
    let rest: Vec<Expr> = repeat(0.., preceded((ws, '&', ws), cmp_expr)).parse_next(input)?;
    Ok(rest.into_iter().fold(first, |l, r| {
        Expr::BinaryOp(Box::new(l), BinOp::And, Box::new(r))
    }))
}

fn cmp_expr(input: &mut &str) -> PResult<Expr> {
    let left = add_expr.parse_next(input)?;
    let rest: Option<(BinOp, Expr)> =
        opt((ws, cmp_op, ws, add_expr).map(|(_, op, _, e)| (op, e))).parse_next(input)?;
    match rest {
        Some((op, right)) => Ok(Expr::BinaryOp(Box::new(left), op, Box::new(right))),
        None => Ok(left),
    }
}

fn cmp_op(input: &mut &str) -> PResult<BinOp> {
    alt((
        "==".value(BinOp::Eq),
        "!=".value(BinOp::Ne),
        "<=".value(BinOp::Le),
        ">=".value(BinOp::Ge),
        "<".value(BinOp::Lt),
        ">".value(BinOp::Gt),
    ))
    .parse_next(input)
}

fn add_expr(input: &mut &str) -> PResult<Expr> {
    let first = mul_expr.parse_next(input)?;
    let rest: Vec<(BinOp, Expr)> =
        repeat(0.., (ws, add_op, ws, mul_expr).map(|(_, op, _, e)| (op, e))).parse_next(input)?;
    Ok(rest.into_iter().fold(first, |l, (op, r)| {
        Expr::BinaryOp(Box::new(l), op, Box::new(r))
    }))
}

fn add_op(input: &mut &str) -> PResult<BinOp> {
    alt(('+'.value(BinOp::Add), '-'.value(BinOp::Sub))).parse_next(input)
}

fn mul_expr(input: &mut &str) -> PResult<Expr> {
    let first = unary_expr.parse_next(input)?;
    let rest: Vec<(BinOp, Expr)> = repeat(
        0..,
        (ws, mul_op, ws, unary_expr).map(|(_, op, _, e)| (op, e)),
    )
    .parse_next(input)?;
    Ok(rest.into_iter().fold(first, |l, (op, r)| {
        Expr::BinaryOp(Box::new(l), op, Box::new(r))
    }))
}

fn mul_op(input: &mut &str) -> PResult<BinOp> {
    alt((
        '*'.value(BinOp::Mul),
        '/'.value(BinOp::Div),
        '%'.value(BinOp::Mod),
    ))
    .parse_next(input)
}

fn unary_expr(input: &mut &str) -> PResult<Expr> {
    alt((
        preceded(('-', ws), unary_expr).map(|e| Expr::UnaryOp(UnaryOp::Neg, Box::new(e))),
        preceded(('~', ws), unary_expr).map(|e| Expr::UnaryOp(UnaryOp::Not, Box::new(e))),
        postfix_expr,
    ))
    .parse_next(input)
}

// ============ Postfix expressions (.attr and (call)) ============

enum Postfix {
    Attr(String),
    Call(Vec<SurfaceArg>),
}

fn postfix_expr(input: &mut &str) -> PResult<Expr> {
    let base = primary.parse_next(input)?;
    let ops: Vec<Postfix> = repeat(0.., postfix_op).parse_next(input)?;

    Ok(ops.into_iter().fold(base, |acc, op| match op {
        Postfix::Attr(name) => Expr::Attr(Box::new(acc), name),
        Postfix::Call(args) => Expr::Call(Box::new(acc), args),
    }))
}

fn postfix_op(input: &mut &str) -> PResult<Postfix> {
    preceded(ws, alt((attr_access, call_expr))).parse_next(input)
}

fn attr_access(input: &mut &str) -> PResult<Postfix> {
    preceded('.', ident_str)
        .map(Postfix::Attr)
        .parse_next(input)
}

fn call_expr(input: &mut &str) -> PResult<Postfix> {
    delimited(
        '(',
        (ws, opt(call_args), ws).map(|(_, args, _)| args.unwrap_or_default()),
        ')',
    )
    .map(Postfix::Call)
    .parse_next(input)
}

fn call_args(input: &mut &str) -> PResult<Vec<SurfaceArg>> {
    terminated(
        separated(1.., call_arg, (ws, ',', ws)),
        opt((ws, ',')), // trailing comma
    )
    .parse_next(input)
}

fn call_arg(input: &mut &str) -> PResult<SurfaceArg> {
    alt((
        // keyword arg: name=expr
        (ident_str, ws, '=', ws, expr).map(|(name, _, _, _, e)| SurfaceArg::Keyword(name, e)),
        // positional arg
        expr.map(SurfaceArg::Positional),
    ))
    .parse_next(input)
}

// ============ Primary expressions ============

fn primary(input: &mut &str) -> PResult<Expr> {
    preceded(
        ws,
        alt((
            paren_expr,
            list_expr,
            col_shorthand,
            directive,
            literal.map(Expr::Literal),
            ident.map(Expr::Ident),
        )),
    )
    .parse_next(input)
}

/// Parse column shorthand: $gold -> ColShorthand("gold")
fn col_shorthand(input: &mut &str) -> PResult<Expr> {
    preceded('$', ident_str)
        .map(Expr::ColShorthand)
        .parse_next(input)
}

/// Parse directive: @merchant, @entity(42)
fn directive(input: &mut &str) -> PResult<Expr> {
    (
        preceded('@', ident_str),
        opt(delimited(('(', ws), call_args, (ws, ')'))),
    )
        .map(|(name, args)| Expr::Directive(name, args.unwrap_or_default()))
        .parse_next(input)
}

fn paren_expr(input: &mut &str) -> PResult<Expr> {
    delimited(('(', ws), expr, (ws, ')')).parse_next(input)
}

fn list_expr(input: &mut &str) -> PResult<Expr> {
    delimited(
        ('[', ws),
        opt(terminated(
            separated(1.., expr, (ws, ',', ws)),
            opt((ws, ',')),
        ))
        .map(|items| items.unwrap_or_default()),
        (ws, ']'),
    )
    .map(Expr::List)
    .parse_next(input)
}

// ============ Identifiers ============

fn ident(input: &mut &str) -> PResult<String> {
    ident_str.parse_next(input)
}

fn namespace_segment<'a>(input: &mut &'a str) -> PResult<&'a str> {
    preceded(
        "::",
        (
            one_of(|c: char| c.is_ascii_alphabetic() || c == '_'),
            take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
        )
            .take(),
    )
    .parse_next(input)
}

fn ident_str(input: &mut &str) -> PResult<String> {
    let first = (
        one_of(|c: char| c.is_ascii_alphabetic() || c == '_'),
        take_while(0.., |c: char| c.is_ascii_alphanumeric() || c == '_'),
    )
        .take()
        .parse_next(input)?;

    let mut result = first.to_string();
    while let Some(seg) = opt(namespace_segment).parse_next(input)? {
        result.push_str("::");
        result.push_str(seg);
    }
    Ok(result)
}

// ============ Literals ============

fn literal(input: &mut &str) -> PResult<Literal> {
    alt((
        "True".value(Literal::Bool(true)),
        "False".value(Literal::Bool(false)),
        "None".value(Literal::Null),
        float_lit,
        int_lit,
        string_lit,
    ))
    .parse_next(input)
}

fn int_lit(input: &mut &str) -> PResult<Literal> {
    digit1
        .try_map(|s: &str| s.parse::<i64>())
        .map(Literal::Int)
        .parse_next(input)
}

fn float_lit(input: &mut &str) -> PResult<Literal> {
    (digit1, '.', digit1)
        .take()
        .try_map(|s: &str| s.parse::<f64>())
        .map(Literal::Float)
        .parse_next(input)
}

fn string_lit(input: &mut &str) -> PResult<Literal> {
    alt((
        delimited('"', string_contents('"'), '"'),
        delimited('\'', string_contents('\''), '\''),
    ))
    .map(Literal::String)
    .parse_next(input)
}

fn string_contents<'a>(quote: char) -> impl FnMut(&mut &'a str) -> PResult<String> {
    move |input: &mut &'a str| {
        let mut result = String::new();
        loop {
            if input.is_empty() {
                return Err(winnow::error::ErrMode::Backtrack(
                    winnow::error::ContextError::new(),
                ));
            }
            let c = input.chars().next().unwrap();
            if c == quote {
                break;
            } else if c == '\\' {
                *input = &input[1..];
                if input.is_empty() {
                    return Err(winnow::error::ErrMode::Backtrack(
                        winnow::error::ContextError::new(),
                    ));
                }
                let escaped = input.chars().next().unwrap();
                let unescaped = match escaped {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    '\'' => '\'',
                    '0' => '\0',
                    _ => escaped, // Unknown escapes pass through
                };
                result.push(unescaped);
                *input = &input[escaped.len_utf8()..];
            } else {
                result.push(c);
                *input = &input[c.len_utf8()..];
            }
        }
        Ok(result)
    }
}

// ============ Whitespace ============

fn ws(input: &mut &str) -> PResult<()> {
    multispace0.void().parse_next(input)
}

// ============ Sanity Tests ============
// Most testing is done via integration tests in tests/integration.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_literals() {
        assert!(matches!(
            parse("123").unwrap(),
            Expr::Literal(Literal::Int(123))
        ));
        assert!(matches!(
            parse("3.14").unwrap(),
            Expr::Literal(Literal::Float(_))
        ));
        assert!(matches!(
            parse("True").unwrap(),
            Expr::Literal(Literal::Bool(true))
        ));
        assert!(matches!(
            parse(r#""hello""#).unwrap(),
            Expr::Literal(Literal::String(_))
        ));
    }

    #[test]
    fn parse_operator_precedence() {
        // a * b + c should parse as (a * b) + c
        let result = parse("a * b + c").unwrap();
        if let Expr::BinaryOp(left, BinOp::Add, _) = result {
            assert!(matches!(*left, Expr::BinaryOp(_, BinOp::Mul, _)));
        } else {
            panic!("Expected Add at top level");
        }

        // a & b | c should parse as (a & b) | c
        let result = parse("a & b | c").unwrap();
        assert!(matches!(result, Expr::BinaryOp(_, BinOp::Or, _)));
    }

    #[test]
    fn parse_method_chain() {
        assert!(parse(r#"df.filter(x).select(y)"#).is_ok());
    }

    #[test]
    fn parse_kwargs() {
        let result = parse(r#"f(a, b=True)"#).unwrap();
        if let Expr::Call(_, args) = result {
            assert!(matches!(&args[0], SurfaceArg::Positional(_)));
            assert!(matches!(&args[1], SurfaceArg::Keyword(_, _)));
        } else {
            panic!("Expected call");
        }
    }

    #[test]
    fn parse_col_shorthand() {
        let result = parse("$gold").unwrap();
        assert!(matches!(result, Expr::ColShorthand(ref s) if s == "gold"));

        // With method chain
        let result = parse("$gold.sum()").unwrap();
        assert!(matches!(result, Expr::Call(_, _)));
    }

    #[test]
    fn parse_directive() {
        // No args
        let result = parse("@merchant").unwrap();
        if let Expr::Directive(name, args) = result {
            assert_eq!(name, "merchant");
            assert!(args.is_empty());
        } else {
            panic!("Expected directive");
        }

        // With args
        let result = parse("@entity(42)").unwrap();
        if let Expr::Directive(name, args) = result {
            assert_eq!(name, "entity");
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected directive");
        }
    }

    #[test]
    fn parse_string_unknown_escape_non_ascii() {
        let result = parse("\"\\é\"").unwrap();
        assert!(matches!(result, Expr::Literal(Literal::String(ref s)) if s == "é"));
    }
}
