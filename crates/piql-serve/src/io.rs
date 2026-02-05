//! File I/O utilities for loading and saving DataFrames

use polars::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Load a DataFrame from a Parquet file
pub fn load_parquet(path: impl AsRef<Path>) -> Result<LazyFrame, PolarsError> {
    let pl_path = PlPath::Local(Arc::from(path.as_ref()));
    LazyFrame::scan_parquet(pl_path, Default::default())
}

/// Load a DataFrame from a CSV file
pub fn load_csv(path: impl AsRef<Path>) -> Result<LazyFrame, PolarsError> {
    let pl_path = PlPath::Local(Arc::from(path.as_ref()));
    LazyCsvReader::new(pl_path).finish()
}
