//! File I/O utilities for loading and saving DataFrames

use polars::prelude::*;
use std::path::Path;

/// Load a DataFrame from a Parquet file
pub fn load_parquet(path: impl AsRef<Path>) -> Result<LazyFrame, PolarsError> {
    LazyFrame::scan_parquet(path, Default::default())
}

/// Load a DataFrame from a CSV file
pub fn load_csv(path: impl AsRef<Path>) -> Result<LazyFrame, PolarsError> {
    LazyCsvReader::new(path).finish()
}
