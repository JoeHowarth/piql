use criterion::{Criterion, black_box, criterion_group, criterion_main};
use piql::{EvalContext, QueryEngine, TimeSeriesConfig, compile, run, run_compiled};
use polars::df;
use polars::prelude::IntoLazy;

fn large_eval_context() -> EvalContext {
    let df = df! {
        "x" => (0..10_000).collect::<Vec<i32>>(),
        "y" => (0..10_000).map(|n| n * 2).collect::<Vec<i32>>(),
    }
    .unwrap()
    .lazy();
    EvalContext::new().with_df("t", df)
}

fn seeded_engine() -> QueryEngine {
    let mut engine = QueryEngine::new();
    engine.register_base(
        "events",
        TimeSeriesConfig {
            tick_column: "tick".into(),
            partition_key: "entity_id".into(),
        },
    );
    engine.subscribe("report", r#"events.window(-2, 0).filter($value > 10)"#);
    engine.set_tick(100);

    for tick in 95..=100 {
        let rows = df! {
            "tick" => &[tick, tick],
            "entity_id" => &[1, 2],
            "value" => &[tick * 2, tick * 3],
        }
        .unwrap()
        .lazy();
        engine.append_tick("events", rows).unwrap();
    }
    engine
}

fn bench_run_filter(c: &mut Criterion) {
    let ctx = large_eval_context();
    let query = "t.filter($x > 1000).with_columns(($y / 2).alias(\"z\"))";

    c.bench_function("run_filter_query", |b| {
        b.iter(|| run(black_box(query), black_box(&ctx)).unwrap())
    });
}

fn bench_compiled_query(c: &mut Criterion) {
    let ctx = large_eval_context();
    let query = "t.filter($x > 1000).with_columns(($y / 2).alias(\"z\"))";
    let compiled = compile(query, &ctx).unwrap();

    c.bench_function("run_compiled_filter_query", |b| {
        b.iter(|| run_compiled(black_box(&compiled), black_box(&ctx)).unwrap())
    });
}

fn bench_engine_tick(c: &mut Criterion) {
    let mut engine = seeded_engine();

    c.bench_function("query_engine_on_tick", |b| {
        b.iter(|| {
            let _ = engine.on_tick(black_box(100)).unwrap();
        })
    });
}

criterion_group!(
    hot_paths,
    bench_run_filter,
    bench_compiled_query,
    bench_engine_tick
);
criterion_main!(hot_paths);
