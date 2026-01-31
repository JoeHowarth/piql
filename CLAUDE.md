# PiQL Development Notes

## Architecture

```
src/
├── lib.rs          # Public API: run(), re-exports
├── parse.rs        # Winnow parser -> surface::Expr
├── transform.rs    # surface::Expr -> core::Expr (desugaring)
├── eval.rs         # core::Expr interpreter against Polars
└── ast/
    ├── mod.rs      # Shared types: Literal, BinOp, UnaryOp, Arg<E>
    ├── surface.rs  # Parser output AST
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

## Desugaring Strategy

The surface/core AST split exists for future sugar expansion:

| Sugar | Expansion |
|-------|-----------|
| `$gold` | `pl.col("gold")` |
| `@now` | `pl.lit(ctx.now)` |
| `@tick` | `pl.lit(ctx.tick)` |
| `@entity.location` | context-dependent column access |

Transform pass handles pattern recognition (e.g., when/then/otherwise chains).
Future: add sugar variants to surface::Expr, expand in transform().

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
