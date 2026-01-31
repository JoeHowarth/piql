# PiQL

A text query language for Polars dataframes. Write queries that look like Python Polars, evaluate them against Rust dataframes.

## Why

Polars expressions as strings enable:
- User-defined queries at runtime
- Storing queries in config/database
- Building query UIs without code generation

## Examples

```python
# Filter and select
entities.filter(pl.col("gold") > 100).select(["name", "gold"])

# Arithmetic and logic
entities.filter((pl.col("type") == "merchant") & (pl.col("gold") >= 50))

# Aggregation
entities.group_by("type").agg(
    pl.col("gold").sum().alias("total"),
    pl.col("name").count().alias("count")
)

# Conditional expressions
entities.with_columns(
    pl.when(pl.col("gold") > 200).then(pl.lit("rich"))
      .when(pl.col("gold") > 100).then(pl.lit("comfortable"))
      .otherwise(pl.lit("poor"))
      .alias("wealth")
)

# Joins
entities.join(locations, left_on="location_id", right_on="id")

# Window functions
entities.with_columns(
    pl.col("gold").sum().over("type").alias("type_total")
)
```

## Supported Features

**DataFrame methods**
`filter`, `select`, `with_columns`, `head`, `tail`, `sort`, `drop`, `explode`, `group_by`, `join`, `rename`, `drop_nulls`, `reverse`

**Expr methods**
`alias`, `over`, `is_between`, `diff`, `shift`, `sum`, `mean`, `min`, `max`, `count`, `first`, `last`, `cast`, `fill_null`, `is_null`, `is_not_null`, `unique`, `abs`, `round`, `len`, `n_unique`, `cum_sum`, `cum_max`, `cum_min`, `rank`, `clip`, `reverse`

**pl functions**
`col`, `lit`, `when`/`then`/`otherwise`

**str namespace**
`starts_with`, `ends_with`, `to_lowercase`, `to_uppercase`, `len_chars`, `contains`, `replace`, `slice`

**dt namespace**
`year`, `month`, `day`, `hour`, `minute`, `second`

**Operators**
`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`, `~`

## Usage

```rust
use piql::{run, EvalContext};

let ctx = EvalContext::new()
    .with_df("entities", entities_df)
    .with_df("locations", locations_df);

let result = run(r#"entities.filter(pl.col("gold") > 100)"#, &ctx)?;
```
