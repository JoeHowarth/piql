//! Surface AST - what the parser produces
//!
//! This mirrors the source syntax closely. It will later include sugar like:
//! - $col -> ColShorthand
//! - @now, @tick(n) -> Directive
//! - etc.

use super::{Arg, BinOp, Literal, UnaryOp};

pub type SurfaceArg = Arg<Expr>;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Identifier: `df`, `pl`, `foo`
    Ident(String),

    /// Literal value
    Literal(Literal),

    /// List expression: `["a", "b", "c"]`
    List(Vec<Expr>),

    /// Attribute access: `expr.name`
    Attr(Box<Expr>, String),

    /// Function/method call: `expr(args...)`
    Call(Box<Expr>, Vec<SurfaceArg>),

    /// Binary operation: `a + b`, `a == b`
    BinaryOp(Box<Expr>, BinOp, Box<Expr>),

    /// Unary operation: `-x`, `~x`
    UnaryOp(UnaryOp, Box<Expr>),

    // === Sugar ===
    /// Column shorthand: `$gold` -> `pl.col("gold")`
    ColShorthand(String),

    /// Directive: `@merchant`, `@entity(42)`
    Directive(String, Vec<SurfaceArg>),
}

impl Expr {
    pub fn attr(self, name: impl Into<String>) -> Self {
        Expr::Attr(Box::new(self), name.into())
    }

    pub fn call(self, args: Vec<SurfaceArg>) -> Self {
        Expr::Call(Box::new(self), args)
    }

    pub fn binop(self, op: BinOp, rhs: Expr) -> Self {
        Expr::BinaryOp(Box::new(self), op, Box::new(rhs))
    }
}
