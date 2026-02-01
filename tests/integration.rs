//! Black-box integration tests for PiQL
//!
//! These tests exercise the full parse → eval pipeline.

use piql::expr_helpers::{binop, lit_int, lit_str, pl_col};
use piql::{BinOp, EvalContext, QueryEngine, TimeSeriesConfig, Value, run};
use polars::prelude::*;

fn setup_test_df() -> EvalContext {
    let df = df! {
        "name" => &["alice", "bob", "charlie"],
        "gold" => &[100, 250, 50],
        "type" => &["merchant", "producer", "merchant"],
    }
    .unwrap()
    .lazy();

    EvalContext::new().with_df("entities", df)
}

fn run_to_df(query: &str, ctx: &EvalContext) -> DataFrame {
    match run(query, ctx).unwrap() {
        Value::DataFrame(lf, _) => lf.collect().unwrap(),
        other => panic!(
            "Expected DataFrame, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

// ============ Filter ============

#[test]
fn filter_simple() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("gold") > 100)"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

#[test]
fn filter_equality() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("type") == "merchant")"#, &ctx);
    assert_eq!(df.height(), 2);
}

#[test]
fn filter_chained() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.filter(pl.col("type") == "merchant").filter(pl.col("gold") > 50)"#,
        &ctx,
    );
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "alice"
    );
}

#[test]
fn filter_with_arithmetic() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("gold") + 50 > 200)"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

// ============ Logical operators ============

#[test]
fn logical_and() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.filter((pl.col("type") == "merchant") & (pl.col("gold") >= 100))"#,
        &ctx,
    );
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "alice"
    );
}

#[test]
fn logical_or() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.filter((pl.col("gold") > 200) | (pl.col("gold") < 60))"#,
        &ctx,
    );
    assert_eq!(df.height(), 2); // bob (250) and charlie (50)
}

// ============ Select ============

#[test]
fn select_columns() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.select(pl.col("name"), pl.col("gold"))"#, &ctx);
    assert_eq!(df.width(), 2);
    assert!(df.column("name").is_ok());
    assert!(df.column("gold").is_ok());
}

#[test]
fn select_with_alias() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.select(pl.col("gold").alias("money"))"#, &ctx);
    assert!(df.column("money").is_ok());
}

#[test]
fn select_list_syntax() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.select([pl.col("name"), pl.col("gold")])"#, &ctx);
    assert_eq!(df.width(), 2);
}

// ============ With columns ============

#[test]
fn with_columns_arithmetic() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.with_columns(pl.col("gold") * 2)"#, &ctx);
    assert!(df.column("name").is_ok());
    let gold = df.column("gold").unwrap().i32().unwrap();
    assert_eq!(gold.get(0).unwrap(), 200);
}

// ============ Sort ============

#[test]
fn sort_ascending() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.sort("gold")"#, &ctx);
    let gold = df.column("gold").unwrap().i32().unwrap();
    assert_eq!(gold.get(0).unwrap(), 50);
    assert_eq!(gold.get(2).unwrap(), 250);
}

#[test]
fn sort_descending() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.sort("gold", descending=True)"#, &ctx);
    let gold = df.column("gold").unwrap().i32().unwrap();
    assert_eq!(gold.get(0).unwrap(), 250);
    assert_eq!(gold.get(1).unwrap(), 100);
    assert_eq!(gold.get(2).unwrap(), 50);
}

// ============ Head ============

#[test]
fn head_explicit() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.head(2)"#, &ctx);
    assert_eq!(df.height(), 2);
}

#[test]
fn head_default() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.head()"#, &ctx);
    assert_eq!(df.height(), 3); // default is 10, we only have 3
}

// ============ String namespace ============

#[test]
fn str_starts_with() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.filter(pl.col("type").str.starts_with("prod"))"#,
        &ctx,
    );
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

#[test]
fn str_ends_with() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.filter(pl.col("type").str.ends_with("er"))"#,
        &ctx,
    );
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

// ============ Complex queries ============

#[test]
fn multiline_query() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"
        entities
            .filter(pl.col("gold") > 50)
            .sort("gold", descending=True)
            .head(2)
        "#,
        &ctx,
    );
    assert_eq!(df.height(), 2);
    let names = df.column("name").unwrap().str().unwrap();
    assert_eq!(names.get(0).unwrap(), "bob");
    assert_eq!(names.get(1).unwrap(), "alice");
}

#[test]
fn full_pipeline() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"
        entities
            .filter(pl.col("type") == "merchant")
            .with_columns(pl.col("gold") * 2)
            .select(pl.col("name"), pl.col("gold").alias("doubled"))
            .sort("doubled", descending=True)
        "#,
        &ctx,
    );
    assert_eq!(df.height(), 2);
    assert_eq!(df.width(), 2);
    let doubled = df.column("doubled").unwrap().i32().unwrap();
    assert_eq!(doubled.get(0).unwrap(), 200); // alice: 100 * 2
    assert_eq!(doubled.get(1).unwrap(), 100); // charlie: 50 * 2
}

// ============ Binary operators ============

#[test]
fn binop_subtract() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("gold") - 50 > 100)"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

#[test]
fn binop_divide() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("gold") / 2 > 100)"#, &ctx);
    assert_eq!(df.height(), 1);
}

#[test]
fn binop_not_equal() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("type") != "merchant")"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

#[test]
fn binop_less_than() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("gold") < 100)"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "charlie"
    );
}

#[test]
fn binop_less_equal() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(pl.col("gold") <= 100)"#, &ctx);
    assert_eq!(df.height(), 2);
}

// ============ Unary operators ============

#[test]
fn unary_not() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.filter(~(pl.col("type") == "merchant"))"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

// ============================================================================
// UNIMPLEMENTED FEATURES
// These tests document Polars features we plan to support but haven't yet.
// ============================================================================

// ============ pl.lit ============

#[test]
fn pl_lit_in_expression() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.with_columns(pl.lit(999).alias("constant"))"#,
        &ctx,
    );
    assert!(df.column("constant").is_ok());
}

// ============ when/then/otherwise ============

#[test]
fn when_then_otherwise_simple() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"
        entities.with_columns(
            pl.when(pl.col("gold") > 100)
              .then(pl.lit("rich"))
              .otherwise(pl.lit("poor"))
              .alias("wealth")
        )
        "#,
        &ctx,
    );
    let wealth = df.column("wealth").unwrap().str().unwrap();
    assert_eq!(wealth.get(1).unwrap(), "rich"); // bob has 250
}

#[test]
fn when_then_otherwise_chained() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"
        entities.with_columns(
            pl.when(pl.col("gold") > 200).then(pl.lit("rich"))
              .when(pl.col("gold") > 75).then(pl.lit("comfortable"))
              .otherwise(pl.lit("poor"))
              .alias("wealth")
        )
        "#,
        &ctx,
    );
    let wealth = df.column("wealth").unwrap().str().unwrap();
    assert_eq!(wealth.get(0).unwrap(), "comfortable"); // alice has 100
    assert_eq!(wealth.get(1).unwrap(), "rich"); // bob has 250
    assert_eq!(wealth.get(2).unwrap(), "poor"); // charlie has 50
}

// ============ group_by / agg ============

#[test]
fn group_by_agg_simple() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.group_by("type").agg(pl.col("gold").sum())"#,
        &ctx,
    );
    assert_eq!(df.height(), 2); // merchant and producer
}

#[test]
fn group_by_agg_multiple() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"
        entities.group_by("type").agg(
            pl.col("gold").sum().alias("total"),
            pl.col("gold").mean().alias("avg"),
            pl.col("name").count().alias("count")
        )
        "#,
        &ctx,
    );
    assert_eq!(df.width(), 4); // type + 3 aggregations
}

#[test]
fn group_by_multiple_columns() {
    let df = df! {
        "a" => &["x", "x", "y", "y"],
        "b" => &[1, 2, 1, 2],
        "val" => &[10, 20, 30, 40],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.group_by("a", "b").agg(pl.col("val").sum().alias("total"))"#,
        &ctx,
    );
    assert_eq!(result.height(), 4); // 4 unique (a, b) combinations
}

// ============ join ============

#[test]
fn join_simple() {
    let entities = df! {
        "id" => &[1, 2, 3],
        "name" => &["alice", "bob", "charlie"],
        "location_id" => &[10, 20, 10],
    }
    .unwrap()
    .lazy();

    let locations = df! {
        "loc_id" => &[10, 20],
        "loc_name" => &["town", "city"],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new()
        .with_df("entities", entities)
        .with_df("locations", locations);

    let df = run_to_df(
        r#"entities.join(locations, left_on="location_id", right_on="loc_id")"#,
        &ctx,
    );
    assert_eq!(df.height(), 3);
    assert!(df.column("loc_name").is_ok());
}

#[test]
fn join_with_how() {
    let left = df! { "a" => &[1, 2, 3] }.unwrap().lazy();
    let right = df! { "a" => &[2, 3, 4], "b" => &["x", "y", "z"] }
        .unwrap()
        .lazy();

    let ctx = EvalContext::new()
        .with_df("left", left)
        .with_df("right", right);

    let df = run_to_df(r#"left.join(right, on="a", how="left")"#, &ctx);
    assert_eq!(df.height(), 3);
}

// ============ over (window functions) ============

#[test]
fn over_partition() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"
        entities.with_columns(
            pl.col("gold").sum().over("type").alias("type_total")
        )
        "#,
        &ctx,
    );
    // merchants total: 100 + 50 = 150
    // producer total: 250
    assert!(df.column("type_total").is_ok());
}

// ============ diff / shift ============

#[test]
fn diff_column() {
    let df = df! {
        "tick" => &[1, 2, 3, 4, 5],
        "value" => &[10, 15, 25, 30, 50],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("value").diff().alias("delta"))"#,
        &ctx,
    );
    assert!(result.column("delta").is_ok());
}

#[test]
fn shift_column() {
    let df = df! {
        "tick" => &[1, 2, 3, 4, 5],
        "value" => &[10, 15, 25, 30, 50],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("value").shift(1).alias("prev_value"))"#,
        &ctx,
    );
    assert!(result.column("prev_value").is_ok());
}

// ============ is_between ============

#[test]
fn is_between_filter() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.filter(pl.col("gold").is_between(75, 150))"#,
        &ctx,
    );
    assert_eq!(df.height(), 1); // alice with 100
}

// ============ fill_null / is_null ============

#[test]
fn fill_null_value() {
    let df = df! {
        "a" => &[Some(1), None, Some(3)],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(r#"df.with_columns(pl.col("a").fill_null(0))"#, &ctx);
    let a = result.column("a").unwrap().i32().unwrap();
    assert_eq!(a.get(1).unwrap(), 0);
}

#[test]
fn filter_is_null() {
    let df = df! {
        "a" => &[Some(1), None, Some(3)],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(r#"df.filter(pl.col("a").is_null())"#, &ctx);
    assert_eq!(result.height(), 1);
}

// ============ cast ============

#[test]
fn cast_to_float() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.with_columns(pl.col("gold").cast("float").alias("gold_f"))"#,
        &ctx,
    );
    assert!(df.column("gold_f").is_ok());
}

// ============ unique / n_unique ============

#[test]
fn unique_values() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.select(pl.col("type").unique())"#, &ctx);
    assert_eq!(df.height(), 2); // merchant, producer
}

// ============ limit / tail ============

#[test]
fn tail_rows() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.tail(2)"#, &ctx);
    assert_eq!(df.height(), 2);
}

// ============ drop / drop_nulls ============

#[test]
fn drop_columns() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.drop("type")"#, &ctx);
    assert!(df.column("type").is_err());
    assert!(df.column("name").is_ok());
}

// ============ rename ============

#[test]
fn rename_positional() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.rename("gold", "coins")"#, &ctx);
    assert!(df.column("coins").is_ok());
    assert!(df.column("gold").is_err());
}

#[test]
fn rename_kwarg_single() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.rename(gold="coins")"#, &ctx);
    assert!(df.column("coins").is_ok());
    assert!(df.column("gold").is_err());
}

#[test]
fn rename_kwarg_multiple() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.rename(gold="coins", name="entity_name")"#, &ctx);
    assert!(df.column("coins").is_ok());
    assert!(df.column("entity_name").is_ok());
    assert!(df.column("gold").is_err());
    assert!(df.column("name").is_err());
}

// ============ explode ============

#[test]
fn explode_list() {
    let df = df! {
        "id" => &[1, 2],
        "tags" => &[Series::new("".into(), &["a", "b"]), Series::new("".into(), &["c"])],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(r#"df.explode("tags")"#, &ctx);
    assert_eq!(result.height(), 3); // 2 tags for id=1, 1 tag for id=2
}

// ============ Edge cases / regression tests ============

#[test]
fn chain_after_group_by_agg() {
    let ctx = setup_test_df();
    let df = run_to_df(
        r#"entities.group_by("type").agg(pl.col("gold").sum().alias("total")).filter(pl.col("total") > 100)"#,
        &ctx,
    );
    // merchant total: 150, producer total: 250 - both > 100
    assert_eq!(df.height(), 2);
}

#[test]
fn negative_number_in_filter() {
    let df = df! {
        "val" => &[-5, 0, 5, 10],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(r#"df.filter(pl.col("val") > -3)"#, &ctx);
    assert_eq!(result.height(), 3); // 0, 5, 10
}

#[test]
fn none_literal_in_fill_null() {
    let df = df! {
        "a" => &[Some(1), None, Some(3)],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    // Fill null with literal 0
    let result = run_to_df(r#"df.with_columns(pl.col("a").fill_null(pl.lit(0)))"#, &ctx);
    let a = result.column("a").unwrap().i32().unwrap();
    assert_eq!(a.get(1).unwrap(), 0);
}

#[test]
fn over_multiple_columns() {
    let df = df! {
        "a" => &["x", "x", "y", "y"],
        "b" => &[1, 1, 1, 2],
        "val" => &[10, 20, 30, 40],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    // Would need: over(["a", "b"]) syntax
    let _result = run_to_df(
        r#"df.with_columns(pl.col("val").sum().over(["a", "b"]).alias("group_sum"))"#,
        &ctx,
    );
}

#[test]
fn join_multiple_columns() {
    let left = df! {
        "a" => &[1, 1, 2],
        "b" => &[10, 20, 10],
        "val" => &[100, 200, 300],
    }
    .unwrap()
    .lazy();

    let right = df! {
        "a" => &[1, 2],
        "b" => &[10, 10],
        "other" => &["x", "y"],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new()
        .with_df("left", left)
        .with_df("right", right);

    let _result = run_to_df(r#"left.join(right, on=["a", "b"])"#, &ctx);
}

// ============ String escapes ============

#[test]
fn string_escape_sequences() {
    let df = df! {
        "text" => &["hello\nworld", "tab\there", "quote\"test"],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);

    // Filter for string containing newline
    let result = run_to_df(
        r#"df.filter(pl.col("text").str.starts_with("hello\nw"))"#,
        &ctx,
    );
    assert_eq!(result.height(), 1);
}

#[test]
fn string_with_escaped_quote() {
    let df = df! {
        "name" => &["test"],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);

    // Use escaped quote in string literal
    let result = run_to_df(
        r#"df.with_columns(pl.lit("say \"hello\"").alias("greeting"))"#,
        &ctx,
    );
    let greeting = result.column("greeting").unwrap().str().unwrap();
    assert_eq!(greeting.get(0).unwrap(), "say \"hello\"");
}

// ============ Math expr methods ============

#[test]
fn expr_abs() {
    let df = df! {
        "val" => &[-5, 0, 5, -10],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").abs().alias("abs_val"))"#,
        &ctx,
    );
    let abs_val = result.column("abs_val").unwrap().i32().unwrap();
    assert_eq!(abs_val.get(0).unwrap(), 5);
    assert_eq!(abs_val.get(3).unwrap(), 10);
}

#[test]
fn expr_round() {
    let df = df! {
        "val" => &[1.234, 5.678, 9.999],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").round(1).alias("rounded"))"#,
        &ctx,
    );
    let rounded = result.column("rounded").unwrap().f64().unwrap();
    assert!((rounded.get(0).unwrap() - 1.2).abs() < 0.01);
    assert!((rounded.get(1).unwrap() - 5.7).abs() < 0.01);
}

#[test]
fn expr_len() {
    let ctx = setup_test_df();
    let result = run_to_df(
        r#"entities.select(pl.col("name").len().alias("count"))"#,
        &ctx,
    );
    assert_eq!(
        result
            .column("count")
            .unwrap()
            .u32()
            .unwrap()
            .get(0)
            .unwrap(),
        3
    );
}

#[test]
fn expr_n_unique() {
    let ctx = setup_test_df();
    let result = run_to_df(
        r#"entities.select(pl.col("type").n_unique().alias("unique_types"))"#,
        &ctx,
    );
    assert_eq!(
        result
            .column("unique_types")
            .unwrap()
            .u32()
            .unwrap()
            .get(0)
            .unwrap(),
        2
    );
}

// ============ String methods ============

#[test]
fn str_contains() {
    let ctx = setup_test_df();
    let result = run_to_df(
        r#"entities.filter(pl.col("name").str.contains("li"))"#,
        &ctx,
    );
    assert_eq!(result.height(), 2); // alice and charlie
}

#[test]
fn str_replace() {
    let df = df! {
        "text" => &["hello world", "foo bar"],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("text").str.replace("o", "0").alias("replaced"))"#,
        &ctx,
    );
    let replaced = result.column("replaced").unwrap().str().unwrap();
    assert_eq!(replaced.get(0).unwrap(), "hell0 world");
}

#[test]
fn str_slice() {
    let df = df! {
        "text" => &["hello", "world"],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("text").str.slice(0, 3).alias("sliced"))"#,
        &ctx,
    );
    let sliced = result.column("sliced").unwrap().str().unwrap();
    assert_eq!(sliced.get(0).unwrap(), "hel");
    assert_eq!(sliced.get(1).unwrap(), "wor");
}

// ============ DataFrame methods ============

#[test]
fn drop_nulls_df() {
    let df = df! {
        "a" => &[Some(1), None, Some(3)],
        "b" => &[Some("x"), Some("y"), None],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(r#"df.drop_nulls()"#, &ctx);
    assert_eq!(result.height(), 1); // only row 0 has no nulls... wait, row 0 has all values
}

// ============ Cumulative functions ============

#[test]
fn expr_cum_sum() {
    let df = df! {
        "val" => &[1, 2, 3, 4],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").cum_sum().alias("running_total"))"#,
        &ctx,
    );
    let running = result.column("running_total").unwrap().i32().unwrap();
    assert_eq!(running.get(0).unwrap(), 1);
    assert_eq!(running.get(1).unwrap(), 3);
    assert_eq!(running.get(2).unwrap(), 6);
    assert_eq!(running.get(3).unwrap(), 10);
}

#[test]
fn expr_cum_max() {
    let df = df! {
        "val" => &[1, 3, 2, 5, 4],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").cum_max().alias("running_max"))"#,
        &ctx,
    );
    let running = result.column("running_max").unwrap().i32().unwrap();
    assert_eq!(running.get(0).unwrap(), 1);
    assert_eq!(running.get(1).unwrap(), 3);
    assert_eq!(running.get(2).unwrap(), 3);
    assert_eq!(running.get(3).unwrap(), 5);
}

#[test]
fn expr_cum_min() {
    let df = df! {
        "val" => &[5, 3, 4, 1, 2],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").cum_min().alias("running_min"))"#,
        &ctx,
    );
    let running = result.column("running_min").unwrap().i32().unwrap();
    assert_eq!(running.get(0).unwrap(), 5);
    assert_eq!(running.get(1).unwrap(), 3);
    assert_eq!(running.get(2).unwrap(), 3);
    assert_eq!(running.get(3).unwrap(), 1);
}

// ============ Rank, clip, reverse ============

#[test]
fn expr_rank() {
    let df = df! {
        "val" => &[10, 30, 20, 40],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").rank().alias("ranked"))"#,
        &ctx,
    );
    let ranked = result.column("ranked").unwrap().u32().unwrap();
    assert_eq!(ranked.get(0).unwrap(), 1); // 10 is rank 1
    assert_eq!(ranked.get(1).unwrap(), 3); // 30 is rank 3
    assert_eq!(ranked.get(2).unwrap(), 2); // 20 is rank 2
    assert_eq!(ranked.get(3).unwrap(), 4); // 40 is rank 4
}

#[test]
fn expr_clip() {
    let df = df! {
        "val" => &[1, 5, 10, 15, 20],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").clip(5, 15).alias("clipped"))"#,
        &ctx,
    );
    let clipped = result.column("clipped").unwrap().i32().unwrap();
    assert_eq!(clipped.get(0).unwrap(), 5); // 1 clipped to 5
    assert_eq!(clipped.get(1).unwrap(), 5); // 5 stays
    assert_eq!(clipped.get(2).unwrap(), 10); // 10 stays
    assert_eq!(clipped.get(3).unwrap(), 15); // 15 stays
    assert_eq!(clipped.get(4).unwrap(), 15); // 20 clipped to 15
}

#[test]
fn expr_reverse() {
    let df = df! {
        "val" => &[1, 2, 3, 4],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(
        r#"df.with_columns(pl.col("val").reverse().alias("reversed"))"#,
        &ctx,
    );
    let reversed = result.column("reversed").unwrap().i32().unwrap();
    assert_eq!(reversed.get(0).unwrap(), 4);
    assert_eq!(reversed.get(1).unwrap(), 3);
    assert_eq!(reversed.get(2).unwrap(), 2);
    assert_eq!(reversed.get(3).unwrap(), 1);
}

#[test]
fn df_reverse() {
    let df = df! {
        "val" => &[1, 2, 3],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("df", df);
    let result = run_to_df(r#"df.reverse()"#, &ctx);
    let val = result.column("val").unwrap().i32().unwrap();
    assert_eq!(val.get(0).unwrap(), 3);
    assert_eq!(val.get(1).unwrap(), 2);
    assert_eq!(val.get(2).unwrap(), 1);
}

// ============ Sugar: $col ============

#[test]
fn sugar_col_shorthand() {
    let ctx = setup_test_df();
    // $gold should expand to pl.col("gold")
    let df = run_to_df(r#"entities.filter($gold > 100)"#, &ctx);
    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("name").unwrap().str().unwrap().get(0).unwrap(),
        "bob"
    );
}

#[test]
fn sugar_col_shorthand_in_select() {
    let ctx = setup_test_df();
    let df = run_to_df(r#"entities.select($name, $gold)"#, &ctx);
    assert_eq!(df.width(), 2);
    assert!(df.column("name").is_ok());
    assert!(df.column("gold").is_ok());
}

#[test]
fn sugar_col_shorthand_with_method() {
    let ctx = setup_test_df();
    // $gold.sum() should work
    let df = run_to_df(r#"entities.select($gold.sum().alias("total"))"#, &ctx);
    assert_eq!(
        df.column("total").unwrap().i32().unwrap().get(0).unwrap(),
        400
    );
}

#[test]
fn sugar_col_delta() {
    // $col.delta -> col.diff().over(partition)
    let df = df! {
        "entity_id" => &[1, 1, 1, 2, 2],
        "tick" => &[1, 2, 3, 1, 2],
        "gold" => &[100, 150, 120, 200, 250],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("entities", df);
    let result = run_to_df(
        r#"entities.with_columns($gold.delta.alias("gold_change"))"#,
        &ctx,
    );

    let changes = result.column("gold_change").unwrap().i32().unwrap();
    // First row of each entity should be null (no previous)
    assert!(changes.get(0).is_none());
    assert_eq!(changes.get(1).unwrap(), 50); // 150 - 100
    assert_eq!(changes.get(2).unwrap(), -30); // 120 - 150
    assert!(changes.get(3).is_none()); // first of entity 2
    assert_eq!(changes.get(4).unwrap(), 50); // 250 - 200
}

// ============ Scope Methods ============

#[test]
fn scope_window() {
    let df = df! {
        "tick" => &[95, 96, 97, 98, 99, 100, 101, 102],
        "value" => &[1, 2, 3, 4, 5, 6, 7, 8],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("data", df).with_tick(100);
    // window(-3, 0) should get ticks 97-100
    let result = run_to_df(r#"data.window(-3, 0)"#, &ctx);
    assert_eq!(result.height(), 4);
}

#[test]
fn scope_since() {
    let df = df! {
        "tick" => &[1, 2, 3, 4, 5],
        "value" => &[10, 20, 30, 40, 50],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("data", df);
    // since(3) should get ticks >= 3
    let result = run_to_df(r#"data.since(3)"#, &ctx);
    assert_eq!(result.height(), 3);
}

#[test]
fn scope_at() {
    let df = df! {
        "tick" => &[1, 2, 3, 4, 5],
        "value" => &[10, 20, 30, 40, 50],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("data", df);
    // at(3) should get only tick == 3
    let result = run_to_df(r#"data.at(3)"#, &ctx);
    assert_eq!(result.height(), 1);
    assert_eq!(
        result
            .column("value")
            .unwrap()
            .i32()
            .unwrap()
            .get(0)
            .unwrap(),
        30
    );
}

#[test]
fn scope_all() {
    let df = df! {
        "tick" => &[1, 2, 3, 4, 5],
        "value" => &[10, 20, 30, 40, 50],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new().with_df("data", df);
    // all() should return all rows
    let result = run_to_df(r#"data.all()"#, &ctx);
    assert_eq!(result.height(), 5);
}

#[test]
fn scope_top() {
    let ctx = setup_test_df();
    // top(2, "gold") should get top 2 by gold descending
    let result = run_to_df(r#"entities.top(2, "gold")"#, &ctx);
    assert_eq!(result.height(), 2);
    let gold = result.column("gold").unwrap().i32().unwrap();
    assert_eq!(gold.get(0).unwrap(), 250); // bob
    assert_eq!(gold.get(1).unwrap(), 100); // alice
}

// ============ Custom Directives ============

#[test]
fn custom_directive_merchant() {
    let mut ctx = setup_test_df();

    // Register @merchant directive
    ctx.sugar.register_directive("merchant", |_, _| {
        // pl.col("type") == "merchant"
        binop(pl_col("type"), BinOp::Eq, lit_str("merchant"))
    });

    let result = run_to_df(r#"entities.filter(@merchant)"#, &ctx);
    assert_eq!(result.height(), 2); // alice and charlie are merchants
}

#[test]
fn custom_directive_entity_with_arg() {
    let df = df! {
        "entity_id" => &[1, 2, 3, 4],
        "name" => &["a", "b", "c", "d"],
    }
    .unwrap()
    .lazy();

    let mut ctx = EvalContext::new().with_df("entities", df);

    // Register @entity(id) directive
    ctx.sugar.register_directive("entity", |args, _| {
        // pl.col("entity_id") == id
        let id = piql::expr_helpers::get_int_arg(args, 0).unwrap_or(0);
        binop(pl_col("entity_id"), BinOp::Eq, lit_int(id))
    });

    let result = run_to_df(r#"entities.filter(@entity(2))"#, &ctx);
    assert_eq!(result.height(), 1);
    assert_eq!(
        result
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap(),
        "b"
    );
}

#[test]
fn time_series_df_with_config() {
    let df = df! {
        "entity_id" => &[1, 1, 1, 2, 2, 2],
        "tick" => &[1, 2, 3, 1, 2, 3],
        "gold" => &[100, 150, 200, 50, 75, 100],
    }
    .unwrap()
    .lazy();

    let ctx = EvalContext::new()
        .with_time_series_df(
            "entities",
            df,
            TimeSeriesConfig {
                tick_column: "tick".into(),
                partition_key: "entity_id".into(),
            },
        )
        .with_tick(2);

    // Use window to filter and delta which uses partition_key from config
    let result = run_to_df(
        r#"entities.window(-1, 0).with_columns($gold.delta.alias("change"))"#,
        &ctx,
    );
    // Should have ticks 1 and 2 for both entities
    assert_eq!(result.height(), 4);
}

// ============ QueryEngine ============

#[test]
fn query_engine_basic() {
    let df = df! {
        "name" => &["alice", "bob", "charlie"],
        "gold" => &[100, 250, 50],
        "type" => &["merchant", "producer", "merchant"],
    }
    .unwrap()
    .lazy();

    let mut engine = QueryEngine::new();
    engine.add_base_df("entities", df);

    // One-off query
    let result = engine.query(r#"entities.filter($gold > 100)"#).unwrap();
    if let Value::DataFrame(lf, _) = result {
        assert_eq!(lf.collect().unwrap().height(), 1);
    } else {
        panic!("Expected DataFrame");
    }
}

#[test]
fn query_engine_materialized() {
    let df = df! {
        "name" => &["alice", "bob", "charlie", "dave"],
        "gold" => &[100, 250, 50, 300],
        "type" => &["merchant", "producer", "merchant", "merchant"],
    }
    .unwrap()
    .lazy();

    let mut engine = QueryEngine::new();
    engine.add_base_df("entities", df);

    // Register @merchant directive
    engine.sugar().register_directive("merchant", |_, _| {
        binop(pl_col("type"), BinOp::Eq, lit_str("merchant"))
    });

    // Materialize intermediate result
    engine
        .materialize("merchants", "entities.filter(@merchant)")
        .unwrap();

    // Query the materialized table
    let result = engine.query(r#"merchants.filter($gold > 100)"#).unwrap();
    if let Value::DataFrame(lf, _) = result {
        let df = lf.collect().unwrap();
        assert_eq!(df.height(), 1); // only dave (300 gold merchant)
    } else {
        panic!("Expected DataFrame");
    }
}

#[test]
fn query_engine_subscriptions() {
    let df = df! {
        "tick" => &[1, 1, 2, 2, 3, 3],
        "entity_id" => &[1, 2, 1, 2, 1, 2],
        "gold" => &[100, 50, 150, 75, 200, 100],
    }
    .unwrap()
    .lazy();

    let mut engine = QueryEngine::new();
    engine.add_time_series_df(
        "entities",
        df,
        TimeSeriesConfig {
            tick_column: "tick".into(),
            partition_key: "entity_id".into(),
        },
    );

    // Subscribe to queries
    engine.subscribe("rich", r#"entities.at(2).filter($gold > 100)"#);
    engine.subscribe("all_current", r#"entities.at(2)"#);

    // Process tick 2
    let results = engine.on_tick(2).unwrap();

    assert_eq!(results.get("rich").unwrap().height(), 1); // entity 1 with 150 gold
    assert_eq!(results.get("all_current").unwrap().height(), 2); // both entities at tick 2
}

#[test]
fn query_engine_materialized_chain() {
    let df = df! {
        "tick" => &[1, 1, 1, 2, 2, 2],
        "entity_id" => &[1, 2, 3, 1, 2, 3],
        "gold" => &[100, 200, 50, 150, 250, 75],
        "type" => &["a", "b", "a", "a", "b", "a"],
    }
    .unwrap()
    .lazy();

    let mut engine = QueryEngine::new();
    engine.add_base_df("entities", df);

    // Chain of materialized tables
    engine
        .materialize("type_a", r#"entities.filter($type == "a")"#)
        .unwrap();
    engine
        .materialize("rich_a", r#"type_a.filter($gold > 100)"#)
        .unwrap();

    // Subscribe to final result
    engine.subscribe("report", r#"rich_a.at(2)"#);

    let results = engine.on_tick(2).unwrap();
    let report = results.get("report").unwrap();

    // At tick 2, type_a entities are 1 (150) and 3 (75)
    // rich_a filters to gold > 100, so only entity 1
    assert_eq!(report.height(), 1);
}

// ============ Base Table Routing ============

#[test]
fn base_table_implicit_now() {
    // Test that queries on base tables without scope use `now` ptr
    let mut engine = QueryEngine::new();
    engine.register_base(
        "entities",
        TimeSeriesConfig {
            tick_column: "tick".into(),
            partition_key: "entity_id".into(),
        },
    );

    // Tick 1: add some entities
    let tick1 = df! {
        "tick" => &[1, 1],
        "entity_id" => &[1, 2],
        "gold" => &[100, 200],
    }
    .unwrap()
    .lazy();
    engine.append_tick("entities", tick1).unwrap();
    engine.set_tick(1);

    // Query without scope - should only see tick 1 data
    let result = engine.query("entities").unwrap();
    if let Value::DataFrame(lf, _) = result {
        assert_eq!(lf.collect().unwrap().height(), 2);
    } else {
        panic!("Expected DataFrame");
    }

    // Tick 2: add more entities
    let tick2 = df! {
        "tick" => &[2, 2],
        "entity_id" => &[1, 2],
        "gold" => &[150, 250],
    }
    .unwrap()
    .lazy();
    engine.append_tick("entities", tick2).unwrap();
    engine.set_tick(2);

    // Query without scope - should only see tick 2 data (implicit now)
    let result = engine.query("entities").unwrap();
    if let Value::DataFrame(lf, _) = result {
        let df = lf.collect().unwrap();
        assert_eq!(df.height(), 2); // only tick 2 rows
        let gold: Vec<i32> = df
            .column("gold")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(gold, vec![150, 250]); // tick 2 values
    } else {
        panic!("Expected DataFrame");
    }
}

#[test]
fn base_table_all_scope() {
    // Test that .all() returns full history
    let mut engine = QueryEngine::new();
    engine.register_base(
        "entities",
        TimeSeriesConfig {
            tick_column: "tick".into(),
            partition_key: "entity_id".into(),
        },
    );

    // Add data for tick 1 and 2
    let tick1 = df! {
        "tick" => &[1, 1],
        "entity_id" => &[1, 2],
        "gold" => &[100, 200],
    }
    .unwrap()
    .lazy();
    engine.append_tick("entities", tick1).unwrap();

    let tick2 = df! {
        "tick" => &[2, 2],
        "entity_id" => &[1, 2],
        "gold" => &[150, 250],
    }
    .unwrap()
    .lazy();
    engine.append_tick("entities", tick2).unwrap();
    engine.set_tick(2);

    // .all() should return all 4 rows
    let result = engine.query("entities.all()").unwrap();
    if let Value::DataFrame(lf, _) = result {
        assert_eq!(lf.collect().unwrap().height(), 4);
    } else {
        panic!("Expected DataFrame");
    }
}

#[test]
fn base_table_window_scope() {
    // Test that .window() returns filtered history
    let mut engine = QueryEngine::new();
    engine.register_base(
        "entities",
        TimeSeriesConfig {
            tick_column: "tick".into(),
            partition_key: "entity_id".into(),
        },
    );

    // Add data for ticks 1, 2, 3
    for tick in 1..=3 {
        let data = df! {
            "tick" => &[tick, tick],
            "entity_id" => &[1, 2],
            "gold" => &[tick * 100, tick * 100 + 50],
        }
        .unwrap()
        .lazy();
        engine.append_tick("entities", data).unwrap();
    }
    engine.set_tick(3);

    // .window(-1, 0) at tick 3 should return ticks 2 and 3
    let result = engine.query("entities.window(-1, 0)").unwrap();
    if let Value::DataFrame(lf, _) = result {
        let df = lf.collect().unwrap();
        assert_eq!(df.height(), 4); // 2 entities x 2 ticks
    } else {
        panic!("Expected DataFrame");
    }
}
