mod protocol;
mod session;
mod traits;

pub use protocol::{ClientMsg, ServerMsg};
pub use session::GameSession;
pub use traits::GamePresenter;

use std::sync::Arc;

use axum::{
    Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
};
use tokio::sync::Mutex;
use tower_http::services::ServeDir;

use crate::eval::Evaluator;
use crate::game::Game;
use crate::game_log::GameLog;
use crate::mcts::Config;

/// Send a progress snapshot every N simulations.
const PROGRESS_INTERVAL: u32 = 10;

/// Launch the web analysis board server.
///
/// Serves static files from the presenter's `static_dir()` and provides
/// a WebSocket endpoint at `/ws` for real-time game interaction.
pub async fn serve<G: Game + 'static>(
    port: u16,
    evaluator: Arc<dyn Evaluator<G> + Sync>,
    eval_name: &str,
    presenter: Arc<dyn GamePresenter<G>>,
    human_players: [bool; 2],
    replay: Option<GameLog>,
) {
    let static_dir = presenter.static_dir().to_path_buf();
    let replay = replay.map(Arc::new);

    let mut initial_session = GameSession::new(evaluator, eval_name, presenter, human_players);
    if let Some(log) = &replay {
        initial_session.load_replay(log);
    }
    let session = Arc::new(Mutex::new(initial_session));

    let app = Router::new()
        .route(
            "/ws",
            axum::routing::get({
                let session = session.clone();
                move |ws: WebSocketUpgrade| {
                    let session = session.clone();
                    async move { ws.on_upgrade(move |socket| handle_socket(socket, session)) }
                }
            }),
        )
        .fallback_service(ServeDir::new(&static_dir));

    let addr = format!("0.0.0.0:{port}");
    println!("Analysis board: http://localhost:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Launch the web analysis board with a pre-built game state.
pub async fn serve_with_state<G: Game + 'static>(
    port: u16,
    state: G,
    evaluator: Arc<dyn Evaluator<G> + Sync>,
    eval_name: &str,
    presenter: Arc<dyn GamePresenter<G>>,
    human_players: [bool; 2],
) {
    let static_dir = presenter.static_dir().to_path_buf();
    let session = Arc::new(Mutex::new(GameSession::with_state(
        state,
        evaluator,
        eval_name,
        presenter,
        human_players,
        Config::default(),
    )));

    let app = Router::new()
        .route(
            "/ws",
            axum::routing::get({
                let session = session.clone();
                move |ws: WebSocketUpgrade| {
                    let session = session.clone();
                    async move { ws.on_upgrade(move |socket| handle_socket(socket, session)) }
                }
            }),
        )
        .fallback_service(ServeDir::new(&static_dir));

    let addr = format!("0.0.0.0:{port}");
    println!("Analysis board: http://localhost:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Launch the web analysis board with a pre-built timeline of game states.
pub async fn serve_with_timeline<G: Game + 'static>(
    port: u16,
    timeline: Vec<(String, G)>,
    evaluator: Arc<dyn Evaluator<G> + Sync>,
    presenter: Arc<dyn GamePresenter<G>>,
    human_players: [bool; 2],
) {
    let static_dir = presenter.static_dir().to_path_buf();
    let mut initial_session = GameSession::with_state(
        timeline[0].1.clone(),
        evaluator,
        "unknown",
        presenter,
        human_players,
        Config::default(),
    );
    initial_session.load_timeline(timeline);
    let session = Arc::new(Mutex::new(initial_session));

    let app = Router::new()
        .route(
            "/ws",
            axum::routing::get({
                let session = session.clone();
                move |ws: WebSocketUpgrade| {
                    let session = session.clone();
                    async move { ws.on_upgrade(move |socket| handle_socket(socket, session)) }
                }
            }),
        )
        .fallback_service(ServeDir::new(&static_dir));

    let addr = format!("0.0.0.0:{port}");
    println!("Analysis board: http://localhost:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Run a streaming MCTS search, sending progress updates over the socket.
///
/// Handles `BotMove` and `RunSims` messages. Returns the final response
/// messages, or `Err(())` if the socket disconnected during search.
pub async fn run_search<G: Game + 'static>(
    socket: &mut WebSocket,
    session: &mut GameSession<G>,
    msg: &ClientMsg,
) -> Result<Vec<ServerMsg>, ()> {
    match session.begin_search(msg) {
        Err(msgs) => Ok(msgs),
        Ok(sims_total) => {
            let mut evals = vec![];
            let mut last_progress = 0;
            let result = loop {
                if let Some(result) = session.search_tick(&mut evals) {
                    break result;
                }
                if let Some((snap, labels)) = session.snapshot_with_labels() {
                    if snap.total_simulations >= last_progress + PROGRESS_INTERVAL {
                        last_progress = snap.total_simulations;
                        send_msg(
                            socket,
                            &ServerMsg::SearchProgress {
                                snapshot: snap,
                                action_labels: labels,
                                sims_total,
                            },
                        )
                        .await?;
                        if let Some(subtree_msg) = session.explore_subtree_msg() {
                            let _ = send_msg(socket, &subtree_msg).await;
                        }
                    }
                }
            };
            Ok(session.finish_search(msg, result))
        }
    }
}

async fn handle_socket<G: Game + 'static>(
    mut socket: WebSocket,
    session: Arc<Mutex<GameSession<G>>>,
) {
    // Send initial state.
    {
        let mut session = session.lock().await;
        let init_msgs = session.handle(ClientMsg::GetState);
        for msg in init_msgs {
            if send_msg(&mut socket, &msg).await.is_err() {
                return;
            }
        }
    }

    while let Some(Ok(ws_msg)) = socket.recv().await {
        let text = match ws_msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg: ClientMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let _ = send_msg(
                    &mut socket,
                    &ServerMsg::Error {
                        message: format!("Invalid message: {e}"),
                    },
                )
                .await;
                continue;
            }
        };

        let mut session = session.lock().await;
        let responses = match &client_msg {
            ClientMsg::BotMove { .. } | ClientMsg::RunSims { .. } => {
                match run_search(&mut socket, &mut session, &client_msg).await {
                    Ok(msgs) => msgs,
                    Err(()) => return,
                }
            }
            _ => session.handle(client_msg),
        };

        for msg in responses {
            if send_msg(&mut socket, &msg).await.is_err() {
                return;
            }
        }
    }
}

pub async fn send_msg(socket: &mut WebSocket, msg: &ServerMsg) -> Result<(), ()> {
    let json = serde_json::to_string(msg).map_err(|_| ())?;
    socket
        .send(Message::Text(json.into()))
        .await
        .map_err(|_| ())
}
