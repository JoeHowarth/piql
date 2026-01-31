//! PiQL - Polars-inspired Query Language
//!
//! A text query language for interacting with Polars dataframes.
//!
//! Pipeline: parse() -> surface::Expr -> transform() -> core::Expr -> eval()

pub mod ast;
pub mod eval;
pub mod parse;
pub mod transform;

// Re-export commonly used types
pub use ast::core::Expr as CoreExpr;
pub use ast::surface::Expr as SurfaceExpr;
pub use ast::{Arg, BinOp, Literal, UnaryOp};
pub use eval::{EvalContext, EvalError, Value, eval};
pub use parse::{ParseError, parse};
pub use transform::transform;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PiqlError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),
    #[error("Eval error: {0}")]
    Eval(#[from] EvalError),
}

/// Parse, transform, and evaluate a PiQL query string
pub fn run(query: &str, ctx: &EvalContext) -> Result<Value, PiqlError> {
    let surface = parse(query)?;
    let core = transform(surface);
    let result = eval(&core, ctx)?;
    Ok(result)
}
