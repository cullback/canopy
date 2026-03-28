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

use crate::game::dev_card::DevCardKind;
use crate::game::dice::{BalancedDice, Dice};
use crate::presenter::CatanPresenter;
use canopy::eval::RolloutEvaluator;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

static MSG_ID: AtomicU64 = AtomicU64::new(1);

/// Discover the colonist.io tab and return its `webSocketDebuggerUrl`.
async fn discover_tab(port: u16) -> String {
    try_discover_tab(port)
        .await
        .expect("failed to discover colonist.io tab")
}

/// Fallible version of `discover_tab`.
async fn try_discover_tab(port: u16) -> Result<String, String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .map_err(|e| format!("cannot connect to Chrome debug port: {e}"))?;

    let request =
        format!("GET /json HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    tokio::io::AsyncWriteExt::write_all(&mut stream, request.as_bytes())
        .await
        .map_err(|e| format!("failed to write HTTP request: {e}"))?;

    let mut buf = vec![0u8; 8192];
    let mut data = Vec::new();
    loop {
        let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
            .await
            .map_err(|e| format!("failed to read HTTP response: {e}"))?;
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
    let json_start = body.find('[').ok_or("no JSON array in /json response")?;
    let tabs: Vec<serde_json::Value> = serde_json::from_str(&body[json_start..])
        .map_err(|e| format!("failed to parse /json response: {e}"))?;

    for tab in &tabs {
        let url = tab["url"].as_str().unwrap_or("");
        if url.contains("colonist.io") && tab["type"].as_str() == Some("page") {
            return tab["webSocketDebuggerUrl"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "tab missing webSocketDebuggerUrl".to_string());
        }
    }
    Err(format!(
        "no colonist.io tab found among {} tabs",
        tabs.len()
    ))
}

/// Send a `Runtime.evaluate` CDP command and return the result value.
async fn evaluate(ws: &mut WsStream, expression: &str) -> serde_json::Value {
    try_evaluate(ws, expression)
        .await
        .expect("CDP evaluate failed")
}

/// Fallible version of `evaluate`.
async fn try_evaluate(ws: &mut WsStream, expression: &str) -> Result<serde_json::Value, String> {
    let id = MSG_ID.fetch_add(1, Ordering::Relaxed);
    let msg = serde_json::json!({
        "id": id,
        "method": "Runtime.evaluate",
        "params": { "expression": expression, "returnByValue": true }
    });
    ws.send(Message::Text(msg.to_string().into()))
        .await
        .map_err(|e| format!("ws send failed: {e}"))?;

    loop {
        let frame = ws
            .next()
            .await
            .ok_or("ws closed")?
            .map_err(|e| format!("ws error: {e}"))?;
        if let Message::Text(text) = frame {
            let resp: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("invalid JSON from CDP: {e}"))?;
            if resp["id"].as_u64() == Some(id) {
                if let Some(err) = resp.get("error") {
                    return Err(format!("CDP error: {err}"));
                }
                return Ok(resp["result"]["result"]["value"].clone());
            }
        }
    }
}

/// Extract the structured game log JSON.
///
/// Strategy 1: walk up from the virtual scroller's fiber — a parent component
/// should hold the full log array as a prop (not just visible children).
/// Strategy 2: find `gameValidator` and search `gameState` properties.
/// Fallback: read visible virtual scroller children (may miss scrolled entries).
const EXTRACT_LOG_JS: &str = r#"(() => {
    function isLogArray(v) {
        return Array.isArray(v) && v.length > 0 && v[0]?.text?.type != null;
    }
    function searchProps(p) {
        if (!p) return null;
        for (let key of Object.keys(p)) {
            if (isLogArray(p[key])) return p[key];
        }
        return null;
    }
    // Strategy 1: walk up from virtual scroller to find full log prop
    let vs = document.querySelector('[class*="virtualScroller"]');
    if (vs) {
        let fk = Object.keys(vs).find(k => k.startsWith('__reactFiber'));
        if (fk) {
            let node = vs[fk];
            for (let d = 0; d < 30 && node; d++) {
                let found = searchProps(node.memoizedProps);
                if (found) return JSON.stringify(found);
                node = node.return;
            }
        }
    }
    // Strategy 2: find gameValidator.gameState, search its properties
    let seen = new Set();
    for (let el of document.querySelectorAll('*')) {
        let fk = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        if (!fk) continue;
        let node = el[fk];
        for (let d = 0; d < 50 && node; d++) {
            if (seen.has(node)) { node = node.return; continue; }
            seen.add(node);
            let p = node.memoizedProps;
            if (p?.gameValidator) {
                let gv = p.gameValidator;
                // Check gameValidator top-level props
                let found = searchProps(gv);
                if (found) return JSON.stringify(found);
                // Check inside gameState
                if (gv.gameState) {
                    found = searchProps(gv.gameState);
                    if (found) return JSON.stringify(found);
                    // One level deeper
                    for (let key of Object.keys(gv.gameState)) {
                        let v = gv.gameState[key];
                        if (v && typeof v === 'object' && !Array.isArray(v)) {
                            found = searchProps(v);
                            if (found) return JSON.stringify(found);
                        }
                    }
                }
                break;
            }
            node = node.return;
        }
    }
    // Fallback: virtual scroller visible children
    if (vs) {
        let fk = Object.keys(vs).find(k => k.startsWith('__reactFiber'));
        if (fk) {
            let n = vs[fk].return?.return;
            let children = n?.memoizedProps?.children;
            if (Array.isArray(children)) {
                return JSON.stringify(children.map(c => c?.props?.gameLogData).filter(Boolean));
            }
        }
    }
    return '[]';
})()"#;

/// Extracted game data from a colonist.io session.
struct ColonistData {
    board: board::BoardData,
    events: Vec<log::GameEvent>,
    /// Dev card identities extracted from the React state (may be empty).
    dev_cards: Vec<DevCardKind>,
    /// Player names keyed by colonist color.
    player_names: Vec<(u8, String)>,
    /// Local player's colonist color (the browser session owner).
    local_color: Option<u8>,
    /// Current turn player's colonist color.
    current_turn_color: Option<u8>,
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

    // Try extracting dev card hand (best-effort).
    let cards_json = evaluate(&mut ws, board::EXTRACT_CARDS_JS).await;
    let cards_str = cards_json.as_str().unwrap_or("[]");
    let card_enums: Vec<u64> = serde_json::from_str(cards_str).unwrap_or_default();
    let dev_cards: Vec<DevCardKind> = card_enums
        .iter()
        .filter_map(|&c| log::card_to_dev(c))
        .collect();
    if !dev_cards.is_empty() {
        eprintln!("extracted {} dev cards from React state", dev_cards.len());
    }

    // Extract player names, local player identity, current turn, and robber.
    let players_json = evaluate(&mut ws, board::EXTRACT_LIVE_JS).await;
    let players_str = players_json.as_str().unwrap_or("{}");
    let players_obj: serde_json::Value = serde_json::from_str(players_str).unwrap_or_default();
    let local_color: Option<u8> = players_obj["localColor"].as_u64().map(|c| c as u8);
    let current_turn_color: Option<u8> = players_obj["currentTurnColor"].as_u64().map(|c| c as u8);
    let player_names: Vec<(u8, String)> = players_obj["players"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let color = p["color"].as_u64()? as u8;
                    let name = p["username"].as_str()?.to_string();
                    Some((color, name))
                })
                .collect()
        })
        .unwrap_or_default();
    for (color, name) in &player_names {
        let me = if local_color == Some(*color) {
            " (local)"
        } else {
            ""
        };
        eprintln!("player: {name} (color {color}){me}");
    }

    ws.close(None).await.ok();

    ColonistData {
        board,
        events,
        dev_cards,
        player_names,
        local_color,
        current_turn_color,
    }
}

/// Resolve the local player's colonist color to an internal `Player`.
fn local_player(
    local_color: Option<u8>,
    color_map: &[(u8, canopy::player::Player)],
) -> canopy::player::Player {
    let color = local_color.expect("failed to detect local player from localStorage");
    state::player_of_color(color_map, color)
        .unwrap_or_else(|| panic!("local color {color} not found in game log"))
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

/// Apply live robber, current turn, and dev card state to a game state.
///
/// Returns whether anything changed. Buildings are handled separately
/// (via `sync_buildings`) because they have their own logging.
fn apply_live_state(
    state: &mut crate::game::state::GameState,
    poll: &PollData,
    color_map: &[(u8, canopy::player::Player)],
    local_color: Option<u8>,
    mapper: &board::CoordMapper,
) -> bool {
    let mut changed = false;

    if let Some((rx, ry)) = poll.robber_hex {
        if let Some(idx) = mapper.tile_index(rx, ry) {
            let new_robber = crate::game::board::TileId(idx as u8);
            if state.robber != new_robber {
                state.robber = new_robber;
                changed = true;
            }
        }
    }

    // During setup, sync_setup_phase sets current_player from building counts.
    // Only override post-setup.
    if state.setup_count >= 4
        && let Some(color) = poll.current_turn_color
        && let Some(pid) = state::player_of_color(color_map, color)
        && state.current_player != pid
    {
        state.current_player = pid;
        changed = true;
    }

    if !poll.dev_cards.is_empty()
        && let Some(lp) = local_color.and_then(|c| state::player_of_color(color_map, c))
        && state.players[lp].hidden_dev_cards > 0
    {
        state::apply_dev_cards(state, lp, &poll.dev_cards);
        changed = true;
    }

    changed
}

/// Lightweight CDP extraction for polling.
struct PollData {
    events: Vec<log::GameEvent>,
    dev_cards: Vec<DevCardKind>,
    current_turn_color: Option<u8>,
    robber_hex: Option<(i32, i32)>,
    buildings: board::BuildingData,
}

/// Connect to Chrome and extract only what's needed for incremental updates.
async fn poll_game_data(port: u16) -> Result<PollData, String> {
    let ws_url = try_discover_tab(port).await?;
    let (mut ws, _) = connect_async(&ws_url)
        .await
        .map_err(|e| format!("CDP connect: {e}"))?;

    // Log entries
    let result = try_evaluate(&mut ws, EXTRACT_LOG_JS).await?;
    let json_str = result.as_str().unwrap_or("[]");
    let entries: Vec<serde_json::Value> = serde_json::from_str(json_str).unwrap_or_default();
    let events = log::parse(&entries);

    // Dev cards
    let cards_json = try_evaluate(&mut ws, board::EXTRACT_CARDS_JS).await?;
    let cards_str = cards_json.as_str().unwrap_or("[]");
    let card_enums: Vec<u64> = serde_json::from_str(cards_str).unwrap_or_default();
    let dev_cards: Vec<DevCardKind> = card_enums
        .iter()
        .filter_map(|&c| log::card_to_dev(c))
        .collect();

    // Players, current turn, and robber (single CDP call)
    let live_json = try_evaluate(&mut ws, board::EXTRACT_LIVE_JS).await?;
    let live_str = live_json.as_str().unwrap_or("{}");
    let live_obj: serde_json::Value = serde_json::from_str(live_str).unwrap_or_default();
    let current_turn_color: Option<u8> = live_obj["currentTurnColor"].as_u64().map(|c| c as u8);
    let robber_hex: Option<(i32, i32)> = live_obj["robberHex"].as_object().and_then(|o| {
        let x = o.get("x")?.as_i64()? as i32;
        let y = o.get("y")?.as_i64()? as i32;
        Some((x, y))
    });

    // Buildings from board snapshot
    let buildings_json = try_evaluate(&mut ws, board::EXTRACT_BUILDINGS_JS).await?;
    let buildings_str = buildings_json.as_str().unwrap_or("{}");
    let buildings = board::parse_buildings_poll(buildings_str);

    ws.close(None).await.ok();

    Ok(PollData {
        events,
        dev_cards,
        current_turn_color,
        robber_hex,
        buildings,
    })
}

/// Check if a canopy action index matches a colonist `GameEvent`.
///
/// Uses the corner_map/edge_map to reverse-lookup coordinates for settlements,
/// cities, and roads. Other action types match by event type alone.
fn match_action_to_event(
    action: usize,
    event: &log::GameEvent,
    corner_map: &std::collections::HashMap<(i32, i32, u8), crate::game::board::NodeId>,
    edge_map: &std::collections::HashMap<(i32, i32, u8), crate::game::board::EdgeId>,
    mapper: &board::CoordMapper,
) -> bool {
    use crate::game::action::*;
    let a = action as u8;
    match a {
        SETTLEMENT_START..SETTLEMENT_END => {
            let nid = crate::game::board::NodeId(a - SETTLEMENT_START);
            match event {
                log::GameEvent::PlaceSettlement {
                    corner: Some((x, y, z)),
                    ..
                }
                | log::GameEvent::BuildSettlement {
                    corner: Some((x, y, z)),
                    ..
                } => {
                    let mapped = mapper.map_corner(*x, *y, *z);
                    corner_map.get(&mapped) == Some(&nid)
                }
                // Accept without coordinates — trust event type match.
                log::GameEvent::PlaceSettlement { corner: None, .. }
                | log::GameEvent::BuildSettlement { corner: None, .. } => true,
                _ => false,
            }
        }
        ROAD_START..ROAD_END => {
            let eid = crate::game::board::EdgeId(a - ROAD_START);
            match event {
                log::GameEvent::PlaceRoad {
                    edge: Some((x, y, z)),
                    ..
                }
                | log::GameEvent::BuildRoad {
                    edge: Some((x, y, z)),
                    ..
                } => {
                    let mapped = mapper.map_edge(*x, *y, *z);
                    edge_map.get(&mapped) == Some(&eid)
                }
                log::GameEvent::PlaceRoad { edge: None, .. }
                | log::GameEvent::BuildRoad { edge: None, .. } => true,
                _ => false,
            }
        }
        CITY_START..CITY_END => {
            let nid = crate::game::board::NodeId(a - CITY_START);
            match event {
                log::GameEvent::BuildCity {
                    corner: Some((x, y, z)),
                    ..
                } => {
                    let mapped = mapper.map_corner(*x, *y, *z);
                    corner_map.get(&mapped) == Some(&nid)
                }
                log::GameEvent::BuildCity { corner: None, .. } => true,
                _ => false,
            }
        }
        BUY_DEV_CARD => matches!(event, log::GameEvent::BuyDevCard { .. }),
        PLAY_KNIGHT => matches!(event, log::GameEvent::PlayedKnight { .. }),
        PLAY_ROAD_BUILDING => matches!(event, log::GameEvent::PlayedRoadBuilding { .. }),
        YOP_START..YOP_END => matches!(event, log::GameEvent::PlayedYearOfPlenty { .. }),
        MONOPOLY_START..MONOPOLY_END => matches!(event, log::GameEvent::PlayedMonopoly { .. }),
        MARITIME_START..MARITIME_END => matches!(event, log::GameEvent::BankTrade { .. }),
        _ => false,
    }
}

/// Try to match pending canopy actions against new colonist events.
///
/// Returns `Ok(matched_count)` if events confirm pending actions in order,
/// or `Err(())` on the first mismatch.
fn match_pending_actions(
    pending: &[usize],
    new_events: &[log::GameEvent],
    corner_map: &std::collections::HashMap<(i32, i32, u8), crate::game::board::NodeId>,
    edge_map: &std::collections::HashMap<(i32, i32, u8), crate::game::board::EdgeId>,
    mapper: &board::CoordMapper,
) -> Result<usize, ()> {
    // Filter new events to "significant" ones (the ones that correspond to
    // player actions, not silent resource gains etc.).
    let significant: Vec<&log::GameEvent> = new_events
        .iter()
        .filter(|e| {
            matches!(
                e,
                log::GameEvent::PlaceSettlement { .. }
                    | log::GameEvent::PlaceRoad { .. }
                    | log::GameEvent::BuildRoad { .. }
                    | log::GameEvent::BuildSettlement { .. }
                    | log::GameEvent::BuildCity { .. }
                    | log::GameEvent::BuyDevCard { .. }
                    | log::GameEvent::PlayedKnight { .. }
                    | log::GameEvent::PlayedMonopoly { .. }
                    | log::GameEvent::PlayedRoadBuilding { .. }
                    | log::GameEvent::PlayedYearOfPlenty { .. }
                    | log::GameEvent::BankTrade { .. }
            )
        })
        .collect();

    let mut matched = 0;
    for (i, &action) in pending.iter().enumerate() {
        if i >= significant.len() {
            // Not enough events yet to confirm all pending — partial match is ok.
            break;
        }
        if match_action_to_event(action, significant[i], corner_map, edge_map, mapper) {
            matched += 1;
        } else {
            eprintln!(
                "pending action mismatch: canopy action {} vs event {:?}",
                action, significant[i]
            );
            return Err(());
        }
    }
    Ok(matched)
}

/// Entry point: extract game data and serve via web analysis board with live polling.
pub fn run_serve(cdp_port: u16, serve_port: u16) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let data = rt.block_on(extract_game_data(cdp_port));
    log::print(&data.events);
    board::print(&data.board);

    let mapper = board::CoordMapper::detect(&data.board.tiles);
    let mut timeline = state::build_timeline(&data.board, &data.events, &mapper);
    // Derive color map from player list (first player → P1, second → P2).
    let color_map: Vec<(u8, canopy::player::Player)> = data
        .player_names
        .iter()
        .take(2)
        .enumerate()
        .map(|(i, &(color, _))| {
            let pid = if i == 0 {
                canopy::player::Player::One
            } else {
                canopy::player::Player::Two
            };
            (color, pid)
        })
        .collect();
    let initial_event_count = data.events.len();

    // Apply dev card identities to the final timeline state if available.
    if !data.dev_cards.is_empty() {
        if let Some(last) = timeline.last_mut() {
            let local_player = local_player(data.local_color, &color_map);
            state::apply_dev_cards(&mut last.state, local_player, &data.dev_cards);
        }
    }

    // Set current turn player on final state from live game data.
    if let Some(color) = data.current_turn_color {
        if let Some(pid) = state::player_of_color(&color_map, color) {
            if let Some(last) = timeline.last_mut() {
                last.state.current_player = pid;
            }
        }
    }

    // Save the last state for incremental processing.
    let last_state = timeline
        .last()
        .map(|e| e.state.clone())
        .expect("empty timeline");

    let timeline_pairs: Vec<(String, crate::game::state::GameState)> =
        timeline.into_iter().map(|e| (e.label, e.state)).collect();
    let mut names = ["P1".to_string(), "P2".to_string()];
    for &(color, pid) in &color_map {
        if let Some((_, name)) = data.player_names.iter().find(|(c, _)| *c == color) {
            names[pid as usize] = name.clone();
        }
    }

    let static_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/catan/web");
    let dice = Dice::Balanced(BalancedDice::new());
    let presenter = Arc::new(CatanPresenter::new(static_dir, dice).with_player_names(names));
    let evaluator: Arc<dyn canopy::eval::Evaluator<crate::game::state::GameState> + Sync> =
        Arc::new(RolloutEvaluator::default());

    // Create session manually with the initial timeline.
    let mut initial_session = canopy::server::GameSession::with_state(
        timeline_pairs[0].1.clone(),
        evaluator,
        "rollout",
        presenter.clone(),
        [true, true],
    );
    initial_session.load_timeline(timeline_pairs);
    let session = Arc::new(tokio::sync::Mutex::new(initial_session));

    let (notify_tx, notify_rx) = tokio::sync::watch::channel(0u64);
    let (action_tx, mut action_rx) = tokio::sync::mpsc::unbounded_channel::<usize>();

    // Build coordinate maps for incremental processing.
    let (terrains, numbers, port_resources) = board::to_layout(&data.board, &mapper);
    let topology = Arc::new(crate::game::topology::Topology::from_layout(
        terrains,
        numbers,
        port_resources,
    ));
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    // Spawn polling task.
    let poll_session = session.clone();
    let poll_board = data.board;
    let poll_local_color = data.local_color;
    let poll_mapper = mapper;
    rt.spawn(async move {
        let mut committed_event_count = initial_event_count;
        let mut last_state = last_state;
        let mut generation = 0u64;
        let mut pending_actions: Vec<usize> = Vec::new();
        let mut pre_pending_cursor: Option<usize> = None;

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            // Drain any actions the user played on the serve board.
            while let Ok(action) = action_rx.try_recv() {
                if pending_actions.is_empty() {
                    // Save cursor before first pending action.
                    let session = poll_session.lock().await;
                    pre_pending_cursor = Some(session.cursor());
                }
                pending_actions.push(action);
            }

            let poll = match poll_game_data(cdp_port).await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("poll error: {e}");
                    continue;
                }
            };

            let total_events = poll.events.len();
            eprintln!("poll ok: {total_events} events (committed {committed_event_count})");

            // Sync buildings from board snapshot (handles setup placements
            // that lack coordinates in log entries).
            let (ns, nc, nr, last_settle) = state::sync_buildings(
                &mut last_state,
                &poll.buildings,
                &color_map,
                &corner_map,
                &edge_map,
                &poll_mapper,
            );
            let buildings_changed = ns + nc + nr > 0;
            if buildings_changed {
                eprintln!("poll: synced {ns} settlements, {nc} cities, {nr} roads from board");
                state::sync_setup_phase(&mut last_state, last_settle);
            }

            if total_events <= committed_event_count {
                // No new events — still update robber/turn/dev cards/buildings.
                let live_changed = apply_live_state(
                    &mut last_state,
                    &poll,
                    &color_map,
                    poll_local_color,
                    &poll_mapper,
                );
                if buildings_changed || live_changed {
                    let mut session = poll_session.lock().await;
                    session.update_final_state(|s| {
                        let (_, _, _, ls) = state::sync_buildings(
                            s,
                            &poll.buildings,
                            &color_map,
                            &corner_map,
                            &edge_map,
                            &poll_mapper,
                        );
                        state::sync_setup_phase(s, ls);
                        apply_live_state(s, &poll, &color_map, poll_local_color, &poll_mapper);
                    });
                    generation += 1;
                    let _ = notify_tx.send(generation);
                }
                continue;
            }

            // Process new events.
            let new_events = &poll.events[committed_event_count..];
            eprintln!(
                "poll: {} new events (total {})",
                new_events.len(),
                total_events
            );

            // Check pending actions against new colonist events.
            if !pending_actions.is_empty() {
                match match_pending_actions(
                    &pending_actions,
                    new_events,
                    &corner_map,
                    &edge_map,
                    &poll_mapper,
                ) {
                    Ok(matched) if matched > 0 => {
                        eprintln!("poll: {matched} pending actions confirmed by colonist");
                        pending_actions.drain(..matched);
                        if pending_actions.is_empty() {
                            pre_pending_cursor = None;
                        }
                        // Skip the confirmed events — they're already applied
                        // on the session via PlayAction. Just update our tracking
                        // state from the full event set.
                        let new_entries = state::process_new_events(
                            &mut last_state,
                            new_events,
                            &color_map,
                            &corner_map,
                            &edge_map,
                            &poll_board,
                            &poll_mapper,
                        );
                        apply_live_state(
                            &mut last_state,
                            &poll,
                            &color_map,
                            poll_local_color,
                            &poll_mapper,
                        );
                        committed_event_count = total_events;

                        // Only extend timeline with entries beyond what was
                        // already pending. If matched == new_entries.len(),
                        // there's nothing extra to add.
                        let extra: Vec<_> = new_entries
                            .into_iter()
                            .skip(matched)
                            .map(|e| (e.label, e.state))
                            .collect();
                        if !extra.is_empty() {
                            let mut session = poll_session.lock().await;
                            session.extend_timeline(extra);
                        }

                        {
                            let mut session = poll_session.lock().await;
                            session.update_final_state(|s| {
                                let (_, _, _, ls) = state::sync_buildings(
                                    s,
                                    &poll.buildings,
                                    &color_map,
                                    &corner_map,
                                    &edge_map,
                                    &poll_mapper,
                                );
                                state::sync_setup_phase(s, ls);
                                apply_live_state(
                                    s,
                                    &poll,
                                    &color_map,
                                    poll_local_color,
                                    &poll_mapper,
                                );
                            });
                        }

                        generation += 1;
                        let _ = notify_tx.send(generation);
                        continue;
                    }
                    Ok(_) => {
                        // No events matched yet (0 matched) — keep waiting.
                    }
                    Err(()) => {
                        // Mismatch: rollback and re-process normally.
                        eprintln!("poll: pending action mismatch — rolling back");
                        if let Some(cursor) = pre_pending_cursor.take() {
                            let mut session = poll_session.lock().await;
                            session.rollback_to_cursor(cursor);
                        }
                        pending_actions.clear();
                        // Fall through to normal processing below.
                    }
                }
            }

            let new_entries = state::process_new_events(
                &mut last_state,
                new_events,
                &color_map,
                &corner_map,
                &edge_map,
                &poll_board,
                &poll_mapper,
            );

            apply_live_state(
                &mut last_state,
                &poll,
                &color_map,
                poll_local_color,
                &poll_mapper,
            );
            committed_event_count = total_events;

            {
                let mut session = poll_session.lock().await;

                if !new_entries.is_empty() {
                    let pairs: Vec<(String, crate::game::state::GameState)> = new_entries
                        .into_iter()
                        .map(|e| (e.label, e.state))
                        .collect();
                    session.extend_timeline(pairs);
                }

                session.update_final_state(|s| {
                    let (_, _, _, ls) = state::sync_buildings(
                        s,
                        &poll.buildings,
                        &color_map,
                        &corner_map,
                        &edge_map,
                        &poll_mapper,
                    );
                    state::sync_setup_phase(s, ls);
                    apply_live_state(s, &poll, &color_map, poll_local_color, &poll_mapper);
                });
            }

            generation += 1;
            let _ = notify_tx.send(generation);
        }
    });

    eprintln!("serving colonist board on port {serve_port} (live polling every 5s)");
    rt.block_on(canopy::server::serve_live(
        serve_port,
        session,
        presenter,
        notify_rx,
        Some(action_tx),
    ));
}
