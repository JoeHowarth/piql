# PiQL Development Notes

## Architecture

```
src/
├── lib.rs          # Public API: run(), re-exports
├── parse.rs        # Winnow parser -> surface::Expr
├── transform.rs    # surface::Expr -> core::Expr (desugaring)
├── sugar.rs        # SugarRegistry, SugarContext, directive handlers
├── eval.rs         # core::Expr interpreter against Polars
└── ast/
    ├── mod.rs      # Shared types: Literal, BinOp, UnaryOp, Arg<E>
    ├── surface.rs  # Parser output AST (includes ColShorthand, Directive)
    └── core.rs     # Eval input AST (has WhenThenOtherwise)

tests/
└── integration.rs  # Black-box tests: query string -> DataFrame assertions
```

## Pipeline

```
query string -> parse() -> surface::Expr -> transform() -> core::Expr -> eval() -> Value
```

## Testing Philosophy

Prefer black-box integration tests over unit tests. Tests should:
- Take a query string and EvalContext
- Assert on the resulting DataFrame
- Not test internal representations

Unit tests only for tricky parser/transform edge cases.

## Sugar System

The surface/core AST split enables sugar expansion in the transform pass.

**Implemented:**

| Sugar | Expansion | Where |
|-------|-----------|-------|
| `$col` | `pl.col("col")` | transform |
| `$col.delta` | `col.diff().over(partition)` | transform (SugarRegistry) |
| `$col.delta(n)` | `col - col.shift(n).over(partition)` | transform (SugarRegistry) |
| `$col.pct(n)` | percent change formula | transform (SugarRegistry) |
| `@directive(args)` | custom (registered at runtime) | transform (SugarRegistry) |
| `.window(a, b)` | tick filter | eval |
| `.since(n)` | tick filter | eval |
| `.at(n)` | tick filter | eval |
| `.all()` | no time filter | eval |
| `.top(n, col)` | sort desc + head | eval |

Transform pass handles:
- Pattern recognition (when/then/otherwise chains)
- `$col` and `@directive` expansion via SugarRegistry

Eval pass handles:
- Scope methods (window, since, at, all)
- Convenience methods (top)

## Supported Syntax

- Multi-column kwargs: `on=["a", "b"]`, `over(["a", "b"])`, `sort(["a", "b"])`
- String escapes: `\n`, `\t`, `\r`, `\\`, `\"`, `\'`
- `pl.col("a", "b")` multi-column select

## Deferred Features

**pivot / unpivot** - Reshape long ↔ wide format. Useful for:
- Event logs → per-entity event columns
- Resource history → tick-by-tick comparison columns
- EAV attribute tables → wide entity tables for filtering
- Wide resource columns → long format for aggregation

Not implemented yet. Add when needed for game/simulation use cases.
