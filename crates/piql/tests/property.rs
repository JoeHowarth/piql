use piql::advanced::{parse, pretty};
use piql::{EvalContext, Value, run};
use polars::df;
use polars::prelude::IntoLazy;
use proptest::prelude::*;

fn arb_atom() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("x".to_string()),
        Just("y".to_string()),
        (0i64..1000).prop_map(|n| n.to_string()),
        (0i64..1000).prop_map(|n| format!("-{n}")),
    ]
}

fn arb_expr(depth: u32) -> BoxedStrategy<String> {
    if depth == 0 {
        return arb_atom().boxed();
    }

    let leaf = arb_atom();
    let nested = (
        arb_expr(depth - 1),
        prop_oneof![Just("+"), Just("-"), Just("*"), Just("/")],
        arb_expr(depth - 1),
    )
        .prop_map(|(lhs, op, rhs)| format!("({lhs} {op} {rhs})"));
    prop_oneof![leaf, nested].boxed()
}

fn test_ctx() -> EvalContext {
    let df = df! {
        "x" => &[1, 2, 3, 4, 5],
        "y" => &[10, 20, 30, 40, 50],
    }
    .unwrap()
    .lazy();
    EvalContext::new().with_df("t", df)
}

fn assert_df(value: Value) {
    match value {
        Value::DataFrame(_, _) => {}
        other => panic!(
            "expected dataframe result, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

proptest! {
    #[test]
    fn parse_pretty_roundtrip(expr in arb_expr(3)) {
        let parsed = parse(&expr).expect("generated expression should parse");
        let rendered = pretty(&parsed, 120);
        let reparsed = parse(&rendered).expect("pretty output should reparse");
        prop_assert_eq!(parsed, reparsed);
    }

    #[test]
    fn eval_generated_filters_stay_valid(threshold in 0i32..500) {
        let ctx = test_ctx();
        let query = format!("t.filter($x > {threshold})");
        let value = run(&query, &ctx).expect("generated filter query should evaluate");
        assert_df(value);
    }
}
