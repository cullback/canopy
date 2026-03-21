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

/// Launch the web analysis board server.
///
/// Serves static files from the presenter's `static_dir()` and provides
/// a WebSocket endpoint at `/ws` for real-time game interaction.
pub async fn serve<G: Game + 'static>(
    port: u16,
    evaluator: Arc<dyn Evaluator<G> + Sync>,
    presenter: Arc<dyn GamePresenter<G>>,
    default_sims: u32,
    human_players: [bool; 2],
    replay: Option<GameLog>,
) {
    let static_dir = presenter.static_dir().to_path_buf();
    let replay = replay.map(Arc::new);

    let session = Arc::new(Mutex::new(GameSession::new(
        evaluator.clone(),
        presenter.clone(),
        default_sims,
        human_players,
    )));

    let evaluator_for_ws = evaluator;
    let presenter_for_ws = presenter;

    let app = Router::new()
        .route(
            "/ws",
            axum::routing::get({
                let session = session.clone();
                let evaluator = evaluator_for_ws;
                let presenter = presenter_for_ws;
                let replay = replay.clone();
                move |ws: WebSocketUpgrade| {
                    let session = session.clone();
                    let evaluator = evaluator.clone();
                    let presenter = presenter.clone();
                    let replay = replay.clone();
                    async move {
                        ws.on_upgrade(move |socket| {
                            handle_socket(
                                socket,
                                session,
                                evaluator,
                                presenter,
                                default_sims,
                                human_players,
                                replay,
                            )
                        })
                    }
                }
            }),
        )
        .fallback_service(ServeDir::new(&static_dir));

    let addr = format!("0.0.0.0:{port}");
    println!("Analysis board: http://localhost:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_socket<G: Game + 'static>(
    mut socket: WebSocket,
    _shared_session: Arc<Mutex<GameSession<G>>>,
    evaluator: Arc<dyn Evaluator<G> + Sync>,
    presenter: Arc<dyn GamePresenter<G>>,
    default_sims: u32,
    human_players: [bool; 2],
    replay: Option<Arc<GameLog>>,
) {
    // Each WebSocket connection gets its own session for isolation.
    let mut session = GameSession::new(evaluator, presenter, default_sims, human_players);
    if let Some(log) = &replay {
        session.load_replay(log);
    }

    // Send initial state.
    let init_msgs = session.handle(ClientMsg::GetState);
    for msg in init_msgs {
        if send_msg(&mut socket, &msg).await.is_err() {
            return;
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

        let responses = session.handle(client_msg);
        for msg in responses {
            if send_msg(&mut socket, &msg).await.is_err() {
                return;
            }
        }
    }
}

async fn send_msg(socket: &mut WebSocket, msg: &ServerMsg) -> Result<(), ()> {
    let json = serde_json::to_string(msg).map_err(|_| ())?;
    socket
        .send(Message::Text(json.into()))
        .await
        .map_err(|_| ())
}
