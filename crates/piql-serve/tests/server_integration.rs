//! Integration tests for PiQL WebSocket server
//!
//! These tests verify the black-box behavior of the server:
//! - Client connects, subscribes, receives results on tick
//! - Unsubscribe stops results
//! - One-off queries work independently

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use piql::QueryEngine;
use piql_serve::{decode_binary_result, ClientMessage, PiqlServer, ServerMessage};
use polars::io::ipc::IpcStreamReader;
use polars::prelude::*;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Helper to set up a test server with sample data
async fn setup_test_server() -> (Arc<PiqlServer>, SocketAddr, Arc<Mutex<QueryEngine>>) {
    let mut engine = QueryEngine::new();

    // Add test dataframe
    let df = df! {
        "entity_id" => &[1, 2, 3],
        "name" => &["alice", "bob", "charlie"],
        "gold" => &[100, 250, 50],
    }
    .unwrap()
    .lazy();
    engine.add_base_df("entities", df);

    let engine = Arc::new(Mutex::new(engine));
    let server = Arc::new(PiqlServer::new(engine.clone()));

    // Find available port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    // Start server in background
    let server_clone = server.clone();
    tokio::spawn(async move {
        server_clone.listen(addr).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    (server, addr, engine)
}

/// Connect a WebSocket client to the server
async fn connect_client(addr: SocketAddr) -> WsStream {
    let url = format!("ws://{}", addr);
    let (ws, _) = connect_async(&url).await.expect("Failed to connect");
    ws
}

/// Send a client message
async fn send(ws: &mut WsStream, msg: &ClientMessage) {
    let json = serde_json::to_string(msg).unwrap();
    ws.send(Message::Text(json.into())).await.unwrap();
}

/// Receive a server message (text), with timeout
async fn recv_text(ws: &mut WsStream) -> ServerMessage {
    let msg = timeout(Duration::from_secs(2), ws.next())
        .await
        .expect("Timeout waiting for message")
        .expect("Stream closed")
        .expect("WebSocket error");

    match msg {
        Message::Text(text) => serde_json::from_str(text.as_str()).expect("Invalid JSON"),
        other => panic!("Expected text message, got {:?}", other),
    }
}

/// Receive a combined binary message (header + Arrow IPC), with timeout
async fn recv_result(ws: &mut WsStream) -> (ServerMessage, Vec<u8>) {
    let msg = timeout(Duration::from_secs(2), ws.next())
        .await
        .expect("Timeout waiting for message")
        .expect("Stream closed")
        .expect("WebSocket error");

    match msg {
        Message::Binary(data) => {
            let (header, payload) = decode_binary_result(&data).expect("Invalid binary message");
            (header, payload.to_vec())
        }
        other => panic!("Expected binary message, got {:?}", other),
    }
}

// ============ Tests ============

/// Test: list_dfs returns the registered dataframe names
#[tokio::test]
async fn test_list_dfs() {
    let (_server, addr, _engine) = setup_test_server().await;
    let mut ws = connect_client(addr).await;

    send(&mut ws, &ClientMessage::ListDfs).await;

    let response = recv_text(&mut ws).await;
    match response {
        ServerMessage::Dfs { names } => {
            assert!(names.contains(&"entities".to_string()));
        }
        other => panic!("Expected Dfs response, got {:?}", other),
    }
}

/// Test: subscribe → broadcast_tick → receive result with Arrow IPC
#[tokio::test]
async fn test_subscribe_and_receive() {
    let (server, addr, engine) = setup_test_server().await;
    let mut ws = connect_client(addr).await;

    // Subscribe to a query
    send(
        &mut ws,
        &ClientMessage::Subscribe {
            name: "rich".into(),
            query: "entities.filter($gold > 100)".into(),
        },
    )
    .await;

    // Should get subscription confirmation
    let response = recv_text(&mut ws).await;
    assert!(matches!(response, ServerMessage::Subscribed { name } if name == "rich"));

    // Set tick and broadcast
    {
        let mut eng = engine.lock().unwrap();
        eng.set_tick(1);
    }
    server.broadcast_tick(1).await.unwrap();

    // Should receive combined result
    let (header, ipc_data) = recv_result(&mut ws).await;
    match header {
        ServerMessage::ResultHeader { name, tick, size } => {
            assert_eq!(name, "rich");
            assert_eq!(tick, 1);
            assert_eq!(ipc_data.len(), size);
        }
        other => panic!("Expected ResultHeader, got {:?}", other),
    }

    // Verify we can deserialize the Arrow IPC
    let cursor = std::io::Cursor::new(ipc_data);
    let df = IpcStreamReader::new(cursor).finish().unwrap();
    assert_eq!(df.height(), 1); // Only bob with 250 gold
}

/// Test: multiple subscriptions all receive results
#[tokio::test]
async fn test_multiple_subscriptions() {
    let (server, addr, engine) = setup_test_server().await;
    let mut ws = connect_client(addr).await;

    // Subscribe to two queries
    send(
        &mut ws,
        &ClientMessage::Subscribe {
            name: "rich".into(),
            query: "entities.filter($gold > 100)".into(),
        },
    )
    .await;
    let _ = recv_text(&mut ws).await; // Subscribed confirmation

    send(
        &mut ws,
        &ClientMessage::Subscribe {
            name: "all".into(),
            query: "entities".into(),
        },
    )
    .await;
    let _ = recv_text(&mut ws).await; // Subscribed confirmation

    // Broadcast tick
    {
        let mut eng = engine.lock().unwrap();
        eng.set_tick(1);
    }
    server.broadcast_tick(1).await.unwrap();

    // Should receive two results (order may vary)
    let mut received_names = Vec::new();
    for _ in 0..2 {
        let (header, ipc_data) = recv_result(&mut ws).await;
        if let ServerMessage::ResultHeader { name, size, .. } = header {
            received_names.push(name);
            assert_eq!(ipc_data.len(), size);
        }
    }

    assert!(received_names.contains(&"rich".to_string()));
    assert!(received_names.contains(&"all".to_string()));
}

/// Test: unsubscribe stops receiving that result
#[tokio::test]
async fn test_unsubscribe() {
    let (server, addr, engine) = setup_test_server().await;
    let mut ws = connect_client(addr).await;

    // Subscribe
    send(
        &mut ws,
        &ClientMessage::Subscribe {
            name: "rich".into(),
            query: "entities.filter($gold > 100)".into(),
        },
    )
    .await;
    let _ = recv_text(&mut ws).await; // Subscribed

    // Unsubscribe
    send(
        &mut ws,
        &ClientMessage::Unsubscribe {
            name: "rich".into(),
        },
    )
    .await;

    let response = recv_text(&mut ws).await;
    assert!(matches!(response, ServerMessage::Unsubscribed { name } if name == "rich"));

    // Broadcast tick
    {
        let mut eng = engine.lock().unwrap();
        eng.set_tick(1);
    }
    server.broadcast_tick(1).await.unwrap();

    // Should NOT receive any result - timeout expected
    let result = timeout(Duration::from_millis(200), ws.next()).await;
    assert!(
        result.is_err(),
        "Should not receive message after unsubscribe"
    );
}

/// Test: one-off query returns immediately without broadcast_tick
#[tokio::test]
async fn test_oneoff_query() {
    let (_server, addr, _engine) = setup_test_server().await;
    let mut ws = connect_client(addr).await;

    // Send query (no subscribe, no broadcast_tick needed)
    send(
        &mut ws,
        &ClientMessage::Query {
            query: "entities.filter($gold > 100)".into(),
        },
    )
    .await;

    // Should receive combined result immediately
    let (header, ipc_data) = recv_result(&mut ws).await;
    match header {
        ServerMessage::QueryResultHeader { size } => {
            assert_eq!(ipc_data.len(), size);
        }
        other => panic!("Expected QueryResultHeader, got {:?}", other),
    }

    // Verify data
    let cursor = std::io::Cursor::new(ipc_data);
    let df = IpcStreamReader::new(cursor).finish().unwrap();
    assert_eq!(df.height(), 1); // Only bob
}
