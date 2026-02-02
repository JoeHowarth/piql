//! WebSocket server for PiQL query subscriptions

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::{SinkExt, StreamExt};
use piql::{QueryEngine, Value};
use polars::io::ipc::IpcStreamWriter;
use polars::prelude::SerWriter;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

use crate::protocol::{ClientMessage, ServerMessage};

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("Query error: {0}")]
    Query(#[from] piql::PiqlError),
}

type ClientId = u64;

/// Per-client state
struct ClientState {
    /// Subscriptions: name -> query
    subscriptions: HashMap<String, String>,
    /// Channel to send messages to this client
    tx: mpsc::UnboundedSender<Message>,
}

/// Shared state across all client handlers
struct SharedState {
    clients: Mutex<HashMap<ClientId, ClientState>>,
    next_id: AtomicU64,
}

/// WebSocket server for PiQL subscriptions
pub struct PiqlServer {
    engine: Arc<Mutex<QueryEngine>>,
    state: Arc<SharedState>,
}

impl PiqlServer {
    pub fn new(engine: Arc<Mutex<QueryEngine>>) -> Self {
        Self {
            engine,
            state: Arc::new(SharedState {
                clients: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Start listening for WebSocket connections
    pub async fn listen(&self, addr: SocketAddr) -> Result<(), ServerError> {
        let listener = TcpListener::bind(addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let engine = self.engine.clone();
            let state = self.state.clone();

            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, engine, state).await {
                    eprintln!("Connection error: {}", e);
                }
            });
        }
    }

    /// Broadcast tick results to all subscribed clients
    pub async fn broadcast_tick(&self, tick: i64) -> Result<(), ServerError> {
        let clients = self.state.clients.lock().unwrap();

        for (_, client) in clients.iter() {
            for (name, query) in &client.subscriptions {
                // Evaluate the query
                let result = {
                    let engine = self.engine.lock().unwrap();
                    engine.query(query)
                };

                match result {
                    Ok(Value::DataFrame(df, _)) => {
                        // Serialize to Arrow IPC
                        let mut buf = Vec::new();
                        {
                            let collected = df.collect().unwrap();
                            let mut writer = IpcStreamWriter::new(&mut buf);
                            writer.finish(&mut collected.clone()).unwrap();
                        }

                        // Send header
                        let header = ServerMessage::ResultHeader {
                            name: name.clone(),
                            tick,
                            size: buf.len(),
                        };
                        let header_json = serde_json::to_string(&header).unwrap();
                        let _ = client.tx.send(Message::Text(header_json.into()));

                        // Send binary data
                        let _ = client.tx.send(Message::Binary(buf.into()));
                    }
                    Ok(_) => {
                        let err = ServerMessage::Error {
                            message: format!("Query '{}' did not return a DataFrame", name),
                        };
                        let _ = client
                            .tx
                            .send(Message::Text(serde_json::to_string(&err).unwrap().into()));
                    }
                    Err(e) => {
                        let err = ServerMessage::Error {
                            message: format!("Query '{}' failed: {}", name, e),
                        };
                        let _ = client
                            .tx
                            .send(Message::Text(serde_json::to_string(&err).unwrap().into()));
                    }
                }
            }
        }

        Ok(())
    }
}

async fn handle_connection(
    stream: TcpStream,
    engine: Arc<Mutex<QueryEngine>>,
    state: Arc<SharedState>,
) -> Result<(), ServerError> {
    let ws = accept_async(stream).await?;
    let (mut ws_tx, mut ws_rx) = ws.split();

    // Create channel for sending messages to this client
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

    // Register client
    let client_id = state.next_id.fetch_add(1, Ordering::SeqCst);
    {
        let mut clients = state.clients.lock().unwrap();
        clients.insert(
            client_id,
            ClientState {
                subscriptions: HashMap::new(),
                tx,
            },
        );
    }

    // Spawn task to forward messages from channel to WebSocket
    let forward_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if ws_tx.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(_) => break,
        };

        let client_msg: ClientMessage = match serde_json::from_str(msg.as_str()) {
            Ok(m) => m,
            Err(e) => {
                send_error(&state, client_id, format!("Invalid message: {}", e));
                continue;
            }
        };

        handle_message(&engine, &state, client_id, client_msg);
    }

    // Cleanup: remove client
    {
        let mut clients = state.clients.lock().unwrap();
        clients.remove(&client_id);
    }

    forward_task.abort();
    Ok(())
}

fn handle_message(
    engine: &Arc<Mutex<QueryEngine>>,
    state: &Arc<SharedState>,
    client_id: ClientId,
    msg: ClientMessage,
) {
    match msg {
        ClientMessage::ListDfs => {
            let names: Vec<String> = {
                let eng = engine.lock().unwrap();
                eng.dataframe_names()
            };
            let response = ServerMessage::Dfs { names };
            send_message(state, client_id, &response);
        }

        ClientMessage::Subscribe { name, query } => {
            // Validate query first
            {
                let eng = engine.lock().unwrap();
                if let Err(e) = eng.query(&query) {
                    send_error(state, client_id, format!("Invalid query: {}", e));
                    return;
                }
            }

            // Add subscription
            {
                let mut clients = state.clients.lock().unwrap();
                if let Some(client) = clients.get_mut(&client_id) {
                    client.subscriptions.insert(name.clone(), query);
                }
            }

            let response = ServerMessage::Subscribed { name };
            send_message(state, client_id, &response);
        }

        ClientMessage::Unsubscribe { name } => {
            {
                let mut clients = state.clients.lock().unwrap();
                if let Some(client) = clients.get_mut(&client_id) {
                    client.subscriptions.remove(&name);
                }
            }

            let response = ServerMessage::Unsubscribed { name };
            send_message(state, client_id, &response);
        }

        ClientMessage::Query { query } => {
            let result = {
                let eng = engine.lock().unwrap();
                eng.query(&query)
            };

            match result {
                Ok(Value::DataFrame(df, _)) => {
                    // Serialize to Arrow IPC
                    let mut buf = Vec::new();
                    {
                        let collected = df.collect().unwrap();
                        let mut writer = IpcStreamWriter::new(&mut buf);
                        writer.finish(&mut collected.clone()).unwrap();
                    }

                    // Send header
                    let header = ServerMessage::QueryResultHeader { size: buf.len() };
                    send_message(state, client_id, &header);

                    // Send binary
                    let clients = state.clients.lock().unwrap();
                    if let Some(client) = clients.get(&client_id) {
                        let _ = client.tx.send(Message::Binary(buf.into()));
                    }
                }
                Ok(_) => {
                    send_error(state, client_id, "Query did not return a DataFrame".into());
                }
                Err(e) => {
                    send_error(state, client_id, format!("Query failed: {}", e));
                }
            }
        }
    }
}

fn send_message(state: &Arc<SharedState>, client_id: ClientId, msg: &ServerMessage) {
    let json = serde_json::to_string(msg).unwrap();
    let clients = state.clients.lock().unwrap();
    if let Some(client) = clients.get(&client_id) {
        let _ = client.tx.send(Message::Text(json.into()));
    }
}

fn send_error(state: &Arc<SharedState>, client_id: ClientId, message: String) {
    let err = ServerMessage::Error { message };
    send_message(state, client_id, &err);
}
