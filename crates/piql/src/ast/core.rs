//! Core AST - what eval consumes
//!
//! This is the desugared, pattern-recognized form. Transform converts
//! surface::Expr into core::Expr before evaluation.

use super::{Arg, BinOp, Literal, UnaryOp};

pub type CoreArg = Arg<Expr>;

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
    Call(Box<Expr>, Vec<CoreArg>),

    /// Binary operation: `a + b`, `a == b`
    BinaryOp(Box<Expr>, BinOp, Box<Expr>),

    /// Unary operation: `-x`, `~x`
    UnaryOp(UnaryOp, Box<Expr>),

    // === Recognized patterns ===
    /// when/then/otherwise chain (recognized from method chain)
    /// pl.when(c1).then(v1).when(c2).then(v2).otherwise(else)
    WhenThenOtherwise {
        /// Condition-value pairs: [(c1, v1), (c2, v2), ...]
        branches: Vec<(Box<Expr>, Box<Expr>)>,
        /// The else value
        otherwise: Box<Expr>,
    },
}

impl Expr {
    pub fn attr(self, name: impl Into<String>) -> Self {
        Expr::Attr(Box::new(self), name.into())
    }

    pub fn call(self, args: Vec<CoreArg>) -> Self {
        Expr::Call(Box::new(self), args)
    }

    pub fn binop(self, op: BinOp, rhs: Expr) -> Self {
        Expr::BinaryOp(Box::new(self), op, Box::new(rhs))
    }
}
