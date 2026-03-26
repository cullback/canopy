//! Colonist.io CDP connector — reads the game log from a running Chrome tab.

pub(crate) mod board;
pub(crate) mod log;
pub(crate) mod state;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

use crate::game::dice::{BalancedDice, Dice};
use crate::heuristic::HeuristicEvaluator;
use crate::presenter::CatanPresenter;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

static MSG_ID: AtomicU64 = AtomicU64::new(1);

/// Discover the colonist.io tab and return its `webSocketDebuggerUrl`.
async fn discover_tab(port: u16) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("cannot connect to Chrome debug port — is the SSH tunnel up?");

    let request =
        format!("GET /json HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    tokio::io::AsyncWriteExt::write_all(&mut stream, request.as_bytes())
        .await
        .expect("failed to write HTTP request");

    let mut buf = vec![0u8; 8192];
    let mut data = Vec::new();
    loop {
        let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
            .await
            .expect("failed to read HTTP response");
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        let text = String::from_utf8_lossy(&data);
        if let Some(start) = text.find('[') {
            if serde_json::from_str::<Vec<serde_json::Value>>(&text[start..]).is_ok() {
                break;
            }
        }
    }

    let body = String::from_utf8_lossy(&data);
    let json_start = body.find('[').expect("no JSON array in /json response");
    let tabs: Vec<serde_json::Value> =
        serde_json::from_str(&body[json_start..]).expect("failed to parse /json response");

    for tab in &tabs {
        let url = tab["url"].as_str().unwrap_or("");
        if url.contains("colonist.io") && tab["type"].as_str() == Some("page") {
            return tab["webSocketDebuggerUrl"]
                .as_str()
                .expect("tab missing webSocketDebuggerUrl")
                .to_string();
        }
    }
    panic!("no colonist.io tab found among {} tabs", tabs.len());
}

/// Send a `Runtime.evaluate` CDP command and return the result value.
async fn evaluate(ws: &mut WsStream, expression: &str) -> serde_json::Value {
    let id = MSG_ID.fetch_add(1, Ordering::Relaxed);
    let msg = serde_json::json!({
        "id": id,
        "method": "Runtime.evaluate",
        "params": { "expression": expression, "returnByValue": true }
    });
    ws.send(Message::Text(msg.to_string().into()))
        .await
        .expect("ws send failed");

    loop {
        let frame = ws.next().await.expect("ws closed").expect("ws error");
        if let Message::Text(text) = frame {
            let resp: serde_json::Value =
                serde_json::from_str(&text).expect("invalid JSON from CDP");
            if resp["id"].as_u64() == Some(id) {
                if let Some(err) = resp.get("error") {
                    panic!("CDP error: {err}");
                }
                return resp["result"]["result"]["value"].clone();
            }
        }
    }
}

/// Extract the structured game log JSON from the React virtual scroller.
const EXTRACT_LOG_JS: &str = r#"(() => {
    let vs = document.querySelector('[class*="virtualScroller"]');
    if (!vs) return '[]';
    let fiberKey = Object.keys(vs).find(k => k.startsWith('__reactFiber'));
    if (!fiberKey) return '[]';
    let node = vs[fiberKey].return?.return;
    let children = node?.memoizedProps?.children;
    if (!Array.isArray(children)) return '[]';
    return JSON.stringify(children.map(c => c?.props?.gameLogData).filter(Boolean));
})()"#;

/// Extracted game data from a colonist.io session.
struct ColonistData {
    board: board::BoardData,
    events: Vec<log::GameEvent>,
}

/// Connect to Chrome via CDP and extract game data.
async fn extract_game_data(port: u16) -> ColonistData {
    let ws_url = discover_tab(port).await;
    eprintln!("connected to: {ws_url}");

    let (mut ws, _) = connect_async(&ws_url)
        .await
        .expect("WebSocket connect failed");

    // Extract game log
    let result = evaluate(&mut ws, EXTRACT_LOG_JS).await;
    let json_str = result.as_str().unwrap_or("[]");
    let entries: Vec<serde_json::Value> = serde_json::from_str(json_str).unwrap_or_default();
    eprintln!("{} log entries", entries.len());
    let events = log::parse(&entries);

    // Extract board state
    let board_json = evaluate(&mut ws, board::EXTRACT_JS).await;
    let board_str = board_json.as_str().unwrap_or("{}");
    let board = board::parse(board_str).expect("failed to parse board data");

    ws.close(None).await.ok();

    ColonistData { board, events }
}

/// Entry point: print game log and board state to console.
pub fn run(port: u16) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let data = rt.block_on(extract_game_data(port));
    log::print(&data.events);
    board::print(&data.board);
}

/// Entry point: extract game data and serve via web analysis board.
pub fn run_serve(cdp_port: u16, serve_port: u16) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let data = rt.block_on(extract_game_data(cdp_port));
    log::print(&data.events);
    board::print(&data.board);

    let game_state = state::build_game_state(&data.board, &data.events);

    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/catan/web");
    let dice = Dice::Balanced(BalancedDice::new());
    let presenter = Arc::new(CatanPresenter::new(static_dir, dice));
    let evaluator: Arc<dyn canopy::eval::Evaluator<crate::game::state::GameState> + Sync> =
        Arc::new(HeuristicEvaluator::default());

    eprintln!("serving colonist board on port {serve_port}");
    rt.block_on(canopy::server::serve_with_state(
        serve_port,
        game_state,
        evaluator,
        presenter,
        200,
        [true, true],
    ));
}
