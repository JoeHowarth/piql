//! Black-box integration tests for PiQL
//!
//! These tests exercise the full parse → eval pipeline.

use piql::{EvalContext, Value, run};
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
        Value::DataFrame(lf) => lf.collect().unwrap(),
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
