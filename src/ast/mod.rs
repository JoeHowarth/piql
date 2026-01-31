//! AST types for PiQL
//!
//! Split into:
//! - `surface`: What the parser produces (raw syntax, will include sugar)
//! - `core`: What eval consumes (desugared, patterns recognized)

pub mod core;
pub mod surface;

// Shared types used by both surface and core ASTs

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Arg<E> {
    Positional(E),
    Keyword(String, E),
}

impl<E> Arg<E> {
    pub fn pos(expr: E) -> Self {
        Arg::Positional(expr)
    }

    pub fn kw(name: impl Into<String>, expr: E) -> Self {
        Arg::Keyword(name.into(), expr)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}
