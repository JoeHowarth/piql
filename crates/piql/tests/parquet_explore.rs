//! Exploratory tests for parquet data issues
//!
//! Run with: cargo test -p piql --test parquet_explore -- --nocapture

use polars::prelude::*;
use std::path::Path;
use std::sync::Arc;

const DATA_DIR: &str = "/Users/jh/personal/archive/event-data/dfs";

fn parquet_path(name: &str) -> std::path::PathBuf {
    Path::new(DATA_DIR).join(format!("{}.parquet", name))
}

fn scan_parquet(name: &str) -> PolarsResult<LazyFrame> {
    let path = parquet_path(name);
    let pl_path = PlPath::Local(Arc::from(path.as_path()));
    LazyFrame::scan_parquet(pl_path, Default::default())
}

#[test]
fn list_available_files() {
    let dir = Path::new(DATA_DIR);
    if !dir.exists() {
        println!("Data directory not found: {}", DATA_DIR);
        return;
    }

    println!("\n=== Available parquet files ===");
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map(|e| e == "parquet").unwrap_or(false) {
            let size = entry.metadata().unwrap().len();
            println!("  {} ({:.1} MB)", path.file_name().unwrap().to_string_lossy(), size as f64 / 1_000_000.0);
        }
    }
}

#[test]
fn inspect_slot_updates_schema() {
    let path = parquet_path("slot_updates");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }

    println!("\n=== slot_updates schema ===");
    let mut lf = scan_parquet("slot_updates").unwrap();
    let schema = lf.collect_schema().unwrap();

    for (name, dtype) in schema.iter() {
        println!("  {}: {:?}", name, dtype);
    }
}

#[test]
fn inspect_all_schemas() {
    let files = ["slot_updates", "tx_header", "tx_gas", "tx_perf", "block_perf"];

    for file in files {
        let path = parquet_path(file);
        if !path.exists() {
            println!("\n=== {} === (not found)", file);
            continue;
        }

        println!("\n=== {} schema ===", file);
        match scan_parquet(file) {
            Ok(mut lf) => match lf.collect_schema() {
                Ok(schema) => {
                    for (name, dtype) in schema.iter() {
                        println!("  {}: {:?}", name, dtype);
                    }
                }
                Err(e) => println!("  Error getting schema: {}", e),
            },
            Err(e) => println!("  Error scanning: {}", e),
        }
    }
}

#[test]
fn try_slot_updates_head() {
    let path = parquet_path("slot_updates");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }

    println!("\n=== Trying slot_updates.head(2) ===");
    let lf = scan_parquet("slot_updates").unwrap();

    // First check schema
    let schema = lf.clone().collect_schema().unwrap();
    println!("Schema:");
    for (name, dtype) in schema.iter() {
        println!("  {}: {:?}", name, dtype);
    }

    // Try to collect head
    println!("\nAttempting head(2).collect()...");
    match lf.limit(2).collect() {
        Ok(df) => {
            println!("Success! Got {} rows", df.height());
            println!("{}", df);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

#[test]
fn try_each_column_separately() {
    let path = parquet_path("slot_updates");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }

    println!("\n=== Testing each column separately ===");
    let mut lf = scan_parquet("slot_updates").unwrap();
    let schema = lf.collect_schema().unwrap();

    for (name, dtype) in schema.iter() {
        let name_str = name.to_string();
        let dtype_clone = dtype.clone();

        // Use catch_unwind to handle panics from unimplemented dtypes
        let result = std::panic::catch_unwind(|| {
            let lf = scan_parquet("slot_updates").unwrap();
            lf.select([col(&name_str)]).limit(2).collect()
        });

        match result {
            Ok(Ok(_df)) => {
                println!("  ✓ {}: {:?} - OK", name, dtype_clone);
            }
            Ok(Err(e)) => {
                println!("  ✗ {}: {:?} - ERROR: {}", name, dtype_clone, e);
            }
            Err(_) => {
                println!("  💥 {}: {:?} - PANIC (unimplemented)", name, dtype_clone);
            }
        }
    }
}

#[test]
fn try_slot_updates_via_piql() {
    let path = parquet_path("slot_updates");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }

    println!("\n=== Testing via piql ===");
    let lf = scan_parquet("slot_updates").unwrap();

    let ctx = piql::EvalContext::new().with_df("slot_updates", lf);

    match piql::run("slot_updates.head(2)", &ctx) {
        Ok(piql::Value::DataFrame(result_lf, _)) => match result_lf.collect() {
            Ok(df) => {
                println!("Success via piql! Got {} rows", df.height());
                println!("{}", df);
            }
            Err(e) => println!("Collect failed: {}", e),
        },
        Ok(_) => println!("Unexpected result type"),
        Err(e) => println!("piql error: {}", e),
    }
}

#[test]
fn find_problematic_dtype() {
    let path = parquet_path("slot_updates");
    if !path.exists() {
        println!("File not found: {:?}", path);
        return;
    }

    println!("\n=== Looking for problematic dtypes ===");
    let mut lf = scan_parquet("slot_updates").unwrap();
    let schema = lf.collect_schema().unwrap();

    let suspicious_dtypes = ["Decimal", "Duration", "List", "Array", "Struct", "Object"];

    for (name, dtype) in schema.iter() {
        let dtype_str = format!("{:?}", dtype);
        for suspicious in suspicious_dtypes {
            if dtype_str.contains(suspicious) {
                println!("  ⚠ {}: {:?} (contains {})", name, dtype, suspicious);
            }
        }
    }
}

#[test]
fn scan_all_files_for_problematic_dtypes() {
    let dir = Path::new(DATA_DIR);
    if !dir.exists() {
        println!("Data directory not found: {}", DATA_DIR);
        return;
    }

    println!("\n=== Scanning all parquet files for problematic dtypes ===\n");

    let suspicious_dtypes = ["Array", "Decimal", "Duration", "Struct", "Object", "List"];
    let mut total_problems = 0;

    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false)
        })
        .collect();
    files.sort_by_key(|e| e.file_name());

    for entry in files {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy();

        let pl_path = PlPath::Local(Arc::from(path.as_path()));
        let mut lf = match LazyFrame::scan_parquet(pl_path, Default::default()) {
            Ok(lf) => lf,
            Err(e) => {
                println!("{}:", name);
                println!("  ❌ Failed to scan: {}\n", e);
                continue;
            }
        };

        let schema = match lf.collect_schema() {
            Ok(s) => s,
            Err(e) => {
                println!("{}:", name);
                println!("  ❌ Failed to get schema: {}\n", e);
                continue;
            }
        };

        let mut file_problems: Vec<(String, String, String)> = Vec::new();

        for (col_name, dtype) in schema.iter() {
            let dtype_str = format!("{:?}", dtype);
            for suspicious in suspicious_dtypes {
                if dtype_str.contains(suspicious) {
                    file_problems.push((
                        col_name.to_string(),
                        dtype_str.clone(),
                        suspicious.to_string(),
                    ));
                    break;
                }
            }
        }

        if !file_problems.is_empty() {
            println!("{}:", name);
            for (col, dtype, issue) in &file_problems {
                println!("  ⚠ {}: {} ({})", col, dtype, issue);
            }
            println!();
            total_problems += file_problems.len();
        } else {
            println!("{}: ✓ all columns OK", name);
        }
    }

    println!("\n=== Summary ===");
    println!("Total problematic columns: {}", total_problems);
}
