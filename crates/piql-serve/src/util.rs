//! Utilities for simulation integration

use polars::prelude::*;

/// Convert a Vec of structs to a DataFrame
///
/// Helper for the common pattern of accumulating simulation state
/// and converting to columnar format at tick end.
pub trait IntoDataFrame {
    fn into_dataframe(self) -> Result<DataFrame, PolarsError>;
}
