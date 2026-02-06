//! Arrow IPC serialization helpers

use base64::Engine;
use polars::prelude::*;

/// Error while encoding a DataFrame as Arrow IPC.
#[derive(Debug)]
pub enum IpcEncodeError {
    Join(tokio::task::JoinError),
    Polars(PolarsError),
}

impl std::fmt::Display for IpcEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Join(e) => write!(f, "Task join error: {e}"),
            Self::Polars(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for IpcEncodeError {}

impl From<PolarsError> for IpcEncodeError {
    fn from(value: PolarsError) -> Self {
        Self::Polars(value)
    }
}

/// Serialize a DataFrame as Arrow IPC stream bytes.
pub async fn dataframe_to_ipc_bytes(mut df: DataFrame) -> Result<Vec<u8>, IpcEncodeError> {
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, PolarsError> {
        let mut buf = Vec::new();
        IpcStreamWriter::new(&mut buf).finish(&mut df)?;
        Ok(buf)
    })
    .await
    .map_err(IpcEncodeError::Join)?
    .map_err(IpcEncodeError::Polars)?;

    Ok(bytes)
}

/// Serialize a DataFrame as base64-encoded Arrow IPC stream.
pub async fn dataframe_to_base64_ipc(df: DataFrame) -> Result<String, IpcEncodeError> {
    let buf = dataframe_to_ipc_bytes(df).await?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[tokio::test]
    async fn round_trip_ipc() {
        let df = df! {
            "a" => &[1i32, 2, 3],
            "b" => &["x", "y", "z"],
        }
        .unwrap();

        let buf = dataframe_to_ipc_bytes(df).await.unwrap();
        let decoded = IpcStreamReader::new(Cursor::new(buf)).finish().unwrap();
        assert_eq!(decoded.height(), 3);
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.column("a").unwrap().i32().unwrap().get(1), Some(2));
    }
}
