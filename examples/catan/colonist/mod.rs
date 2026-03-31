//! Colonist.io CDP connector — reads the game log from a running Chrome tab.

pub(crate) mod board;
pub(crate) mod log;
pub(crate) mod state;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::ws::{Message, WebSocket};
use futures_util::{FutureExt, SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

use crate::game::dev_card::DevCardKind;
use crate::game::dice::{BalancedDice, Dice};
use crate::game::state::Phase;
use crate::presenter::CatanPresenter;

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

static MSG_ID: AtomicU64 = AtomicU64::new(1);

/// Discover the colonist.io tab and return its `webSocketDebuggerUrl`.
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
async fn try_evaluate(ws: &mut WsStream, expression: &str) -> Result<serde_json::Value, String> {
    let id = MSG_ID.fetch_add(1, Ordering::Relaxed);
    let msg = serde_json::json!({
        "id": id,
        "method": "Runtime.evaluate",
        "params": { "expression": expression, "returnByValue": true }
    });
    ws.send(tungstenite::Message::Text(msg.to_string().into()))
        .await
        .map_err(|e| format!("ws send failed: {e}"))?;

    loop {
        let frame = ws
            .next()
            .await
            .ok_or("ws closed")?
            .map_err(|e| format!("ws error: {e}"))?;
        if let tungstenite::Message::Text(text) = frame {
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
struct GameData {
    board: board::BoardData,
    buildings: board::BuildingData,
    events: Vec<log::GameEvent>,
    /// Dev card identities extracted from the React state (may be empty).
    dev_cards: Vec<DevCardKind>,
    /// Dev cards bought this turn (cannot be played yet).
    dev_cards_bought_this_turn: Vec<DevCardKind>,
    /// Player names keyed by colonist color.
    player_names: Vec<(u8, String)>,
    /// Local player's colonist color (the browser session owner).
    local_color: u8,
    /// Current turn player's colonist color.
    current_turn_color: u8,
    /// Whether the dice have been thrown this turn.
    dice_thrown: bool,
    /// Robber hex coordinates from the live board.
    robber_hex: (i32, i32),
}

/// Connect to Chrome via CDP and extract game data.
async fn extract_game_data(port: u16) -> Result<GameData, String> {
    let ws_url = try_discover_tab(port).await?;
    eprintln!("connected to: {ws_url}");

    let (mut ws, _) = connect_async(&ws_url)
        .await
        .map_err(|e| format!("CDP connect: {e}"))?;

    // Extract game log
    let result = try_evaluate(&mut ws, EXTRACT_LOG_JS).await?;
    let json_str = result.as_str().unwrap_or("[]");
    let entries: Vec<serde_json::Value> = serde_json::from_str(json_str).unwrap_or_default();
    eprintln!("{} log entries", entries.len());
    let events = log::parse(&entries);

    // Extract board state
    let board_json = try_evaluate(&mut ws, board::EXTRACT_JS).await?;
    let board_str = board_json.as_str().unwrap_or("{}");
    let board = board::parse(board_str).ok_or("failed to parse board data")?;

    // Extract buildings
    let buildings_json = try_evaluate(&mut ws, board::EXTRACT_BUILDINGS_JS).await?;
    let buildings_str = buildings_json.as_str().unwrap_or("{}");
    let buildings = board::parse_buildings_poll(buildings_str);

    // Dev cards
    let cards_json = try_evaluate(&mut ws, board::EXTRACT_CARDS_JS).await?;
    let cards_str = cards_json.as_str().unwrap_or("{}");
    let dcs = parse_dev_card_state(cards_str);
    if !dcs.cards.is_empty() {
        eprintln!(
            "extracted {} dev cards from React state (bought this turn: {})",
            dcs.cards.len(),
            dcs.bought_this_turn.len(),
        );
    }

    // Extract player names, local player identity, current turn, dice, robber.
    let live_json = try_evaluate(&mut ws, board::EXTRACT_LIVE_JS).await?;
    let live_str = live_json.as_str().unwrap_or("{}");
    let live_obj: serde_json::Value = serde_json::from_str(live_str).unwrap_or_default();
    let local_color = live_obj["localColor"]
        .as_u64()
        .map(|c| c as u8)
        .ok_or("missing localColor")?;
    let current_turn_color = live_obj["currentTurnColor"]
        .as_u64()
        .map(|c| c as u8)
        .ok_or("missing currentTurnColor")?;
    let dice_thrown = live_obj["diceThrown"]
        .as_bool()
        .ok_or("missing diceThrown")?;
    let robber_hex = live_obj["robberHex"]
        .as_object()
        .and_then(|o| {
            let x = o.get("x")?.as_i64()? as i32;
            let y = o.get("y")?.as_i64()? as i32;
            Some((x, y))
        })
        .ok_or("missing robberHex")?;
    let player_names: Vec<(u8, String)> = live_obj["players"]
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
        let me = if *color == local_color {
            " (local)"
        } else {
            ""
        };
        eprintln!("player: {name} (color {color}){me}");
    }

    ws.close(None).await.ok();

    Ok(GameData {
        board,
        buildings,
        events,
        dev_cards: dcs.cards,
        dev_cards_bought_this_turn: dcs.bought_this_turn,
        player_names,
        local_color,
        current_turn_color,
        dice_thrown,
        robber_hex,
    })
}

/// Parsed dev card state from the React extraction.
struct DevCardState {
    cards: Vec<DevCardKind>,
    bought_this_turn: Vec<DevCardKind>,
}

/// Parse the JSON result from EXTRACT_CARDS_JS.
fn parse_dev_card_state(json_str: &str) -> DevCardState {
    let obj: serde_json::Value = serde_json::from_str(json_str).unwrap_or_default();

    let cards = obj
        .get("cards")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().and_then(log::card_to_dev))
                .collect()
        })
        .unwrap_or_default();
    let bought_this_turn = obj
        .get("bought_this_turn")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().and_then(log::card_to_dev))
                .collect()
        })
        .unwrap_or_default();

    DevCardState {
        cards,
        bought_this_turn,
    }
}

/// Entry point: print game log and board state to console.
pub fn run(port: u16) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let data = rt
        .block_on(extract_game_data(port))
        .expect("failed to extract game data");
    log::print(&data.events);
    board::print(&data.board);
}

/// Apply live robber, current turn, and dev card state to a game state.
///
/// Returns whether anything changed. Buildings are handled separately
/// (via `sync_buildings`) because they have their own logging.
fn apply_live_state(
    state: &mut crate::game::state::GameState,
    data: &GameData,
    color_map: &[(u8, canopy::player::Player)],
    mapper: &board::CoordMapper,
) -> bool {
    let mut changed = false;

    let (rx, ry) = data.robber_hex;
    if let Some(idx) = mapper.tile_index(rx, ry) {
        let new_robber = crate::game::board::TileId(idx as u8);
        if state.robber != new_robber {
            state.robber = new_robber;
            changed = true;
        }
    }

    // During setup, sync_setup_phase sets current_player from building counts.
    // Only override post-setup (phase has left PlaceSettlement/PlaceRoad).
    if !matches!(state.phase, Phase::PlaceSettlement | Phase::PlaceRoad)
        && let Some(pid) = state::player_of_color(color_map, data.current_turn_color)
    {
        if state.current_player != pid {
            eprintln!("live: turn changed to {pid:?}");
            state.current_player = pid;
            changed = true;
        }

        // Use colonist's diceState.diceThrown to detect pre-roll vs post-roll.
        let want_pre_roll = !data.dice_thrown;
        if state.pre_roll != want_pre_roll {
            eprintln!(
                "live: dice_thrown={} → pre_roll={want_pre_roll} (was {})",
                data.dice_thrown, state.pre_roll
            );
            state.pre_roll = want_pre_roll;
            if want_pre_roll {
                state.phase = Phase::PreRoll;
            } else if matches!(state.phase, Phase::PreRoll) {
                state.phase = Phase::Main;
            }
            changed = true;
        }
    }

    if let Some(lp) = state::player_of_color(color_map, data.local_color) {
        state::apply_dev_cards(state, lp, &data.dev_cards, &data.dev_cards_bought_this_turn);
        if state.players[lp].hidden_dev_cards == 0 && !data.dev_cards.is_empty() {
            changed = true;
        }
    }

    // Derive has_played_dev_card_this_turn from the event log.
    if !matches!(state.phase, Phase::PlaceSettlement | Phase::PlaceRoad) {
        let played =
            state::played_dev_card_this_turn(&data.events, color_map, state.current_player);
        if state.players[state.current_player].has_played_dev_card_this_turn != played {
            state.players[state.current_player].has_played_dev_card_this_turn = played;
            changed = true;
        }
    }

    // Phase::Roll is an internal chance node for auto-resolving dice.
    // In colonist mode dice come from events, so map back to PreRoll/Main.
    if matches!(state.phase, Phase::Roll) {
        state.phase = if state.pre_roll {
            Phase::PreRoll
        } else {
            Phase::Main
        };
        changed = true;
    }

    // Diagnostic: dev card legality
    if state.setup_count >= 4 {
        let cp = state.current_player;
        let p = &state.players[cp];
        let kn = p.dev_cards[crate::game::dev_card::DevCardKind::Knight];
        let kn_bought = p.dev_cards_bought_this_turn[crate::game::dev_card::DevCardKind::Knight];
        let played = p.has_played_dev_card_this_turn;
        let hidden = p.hidden_dev_cards;
        eprintln!(
            "live dev: cp={cp:?} phase={:?} knights={kn} bought_this_turn={kn_bought} played={played} hidden={hidden}",
            state.phase
        );
    }

    changed
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

/// Persistent state for polling colonist.io via CDP.
struct ColonistPollState {
    cdp_port: u16,
    committed_event_count: usize,
    committed_state: crate::game::state::GameState,
    color_map: Vec<(u8, canopy::player::Player)>,
    corner_map: std::collections::HashMap<(i32, i32, u8), crate::game::board::NodeId>,
    edge_map: std::collections::HashMap<(i32, i32, u8), crate::game::board::EdgeId>,
    mapper: board::CoordMapper,
    pending_actions: Vec<usize>,
    pre_pending_cursor: Option<usize>,
}

impl ColonistPollState {
    /// Poll CDP, sync state, and update the session. Returns response messages.
    async fn poll(
        &mut self,
        session: &mut canopy::server::GameSession<crate::game::state::GameState>,
    ) -> Vec<canopy::server::ServerMsg> {
        let poll = match extract_game_data(self.cdp_port).await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("poll error: {e}");
                return vec![session.state_msg()];
            }
        };

        let total_events = poll.events.len();
        eprintln!(
            "poll ok: {total_events} events (committed {})",
            self.committed_event_count
        );

        // --- Mutate committed_state (single copy, single place) ---

        let was_setup = matches!(
            self.committed_state.phase,
            Phase::PlaceSettlement | Phase::PlaceRoad
        );
        let sync = state::sync_buildings(
            &mut self.committed_state,
            &poll.buildings,
            &self.color_map,
            &self.corner_map,
            &self.edge_map,
            &self.mapper,
        );
        let building_count = sync.settlements + sync.cities + sync.roads;
        if building_count > 0 {
            eprintln!(
                "poll: synced {} settlements, {} cities, {} roads from board",
                sync.settlements, sync.cities, sync.roads
            );
            state::sync_setup_phase(&mut self.committed_state);
        }

        let mut entries_to_extend = Vec::new();
        let mut actions_to_walk: Vec<usize> = Vec::new();
        let mut needs_rollback = false;

        // During setup, sync_buildings places buildings directly and returns
        // the corresponding action IDs. Prepend them so the tree pointer
        // advances to match the new committed_state.
        if building_count > 0 && was_setup {
            actions_to_walk.extend(sync.walk_actions);
        }

        if total_events > self.committed_event_count {
            let new_events = &poll.events[self.committed_event_count..];
            eprintln!(
                "poll: {} new events (total {})",
                new_events.len(),
                total_events
            );

            let (rx, ry) = poll.robber_hex;
            let live_robber = self
                .mapper
                .tile_index(rx, ry)
                .map(|i| crate::game::board::TileId(i as u8));
            let (new_entries, new_actions) = state::process_new_events(
                &mut self.committed_state,
                new_events,
                &self.color_map,
                &self.corner_map,
                &self.edge_map,
                &self.mapper,
                live_robber,
            );
            self.committed_event_count = total_events;

            if !self.pending_actions.is_empty() {
                match match_pending_actions(
                    &self.pending_actions,
                    new_events,
                    &self.corner_map,
                    &self.edge_map,
                    &self.mapper,
                ) {
                    Ok(matched) if matched > 0 => {
                        eprintln!("poll: {matched} pending actions confirmed by colonist");
                        // The session already walked the tree for these pending
                        // actions (via PlayAction). But the colonist events may
                        // produce additional walk actions beyond the matched ones
                        // (e.g. a Roll event generates [END_TURN, ROLL, dice]
                        // but only END_TURN was the pending action). Strip the
                        // matched action IDs from the front of new_actions and
                        // walk the remainder so the tree pointer reaches the
                        // correct depth.
                        let matched_ids: Vec<usize> = self.pending_actions[..matched].to_vec();
                        self.pending_actions.drain(..matched);
                        if self.pending_actions.is_empty() {
                            self.pre_pending_cursor = None;
                        }
                        entries_to_extend = new_entries.into_iter().skip(matched).collect();
                        let mut skip = 0;
                        for &pa in &matched_ids {
                            if skip < new_actions.len() && new_actions[skip] == pa {
                                skip += 1;
                            }
                        }
                        if skip < new_actions.len() {
                            actions_to_walk = new_actions[skip..].to_vec();
                        }
                    }
                    Ok(_) => {
                        // No events matched yet — keep waiting.
                    }
                    Err(()) => {
                        eprintln!("poll: pending action mismatch — rolling back");
                        needs_rollback = true;
                        self.pending_actions.clear();
                        entries_to_extend = new_entries;
                        actions_to_walk = new_actions;
                    }
                }
            } else {
                entries_to_extend = new_entries;
                actions_to_walk = new_actions;
            }
        }

        // Recompute derived state from board bits.
        {
            use canopy::player::Player;
            for &pid in &[Player::One, Player::Two] {
                let s = self.committed_state.boards[pid].settlements.count_ones() as u8;
                let c = self.committed_state.boards[pid].cities.count_ones() as u8;
                self.committed_state.players[pid].building_vps = s + c * 2;
            }
            let len1 = self.committed_state.boards[Player::One]
                .road_network
                .longest_road();
            let len2 = self.committed_state.boards[Player::Two]
                .road_network
                .longest_road();
            self.committed_state.longest_road = match (len1 >= 5, len2 >= 5) {
                (false, false) => None,
                (true, false) => Some((Player::One, len1)),
                (false, true) => Some((Player::Two, len2)),
                (true, true) => {
                    if len1 > len2 {
                        Some((Player::One, len1))
                    } else if len2 > len1 {
                        Some((Player::Two, len2))
                    } else {
                        self.committed_state
                            .longest_road
                            .map(|(pid, _)| (pid, len1))
                    }
                }
            };
        }

        apply_live_state(
            &mut self.committed_state,
            &poll,
            &self.color_map,
            &self.mapper,
        );

        // --- Sync to session ---

        let has_updates =
            needs_rollback || !entries_to_extend.is_empty() || self.pending_actions.is_empty();
        if !has_updates {
            return vec![session.state_msg()];
        }

        if needs_rollback && let Some(cursor) = self.pre_pending_cursor.take() {
            session.rollback_to_cursor(cursor);
        }

        if !actions_to_walk.is_empty() {
            let pre_visits = session.root_visits();
            session.walk_tree(&actions_to_walk);
            let post_visits = session.root_visits();
            eprintln!(
                "poll: walk_tree {:?} — visits {} → {}",
                actions_to_walk, pre_visits, post_visits
            );
        }

        if !entries_to_extend.is_empty() {
            let pairs: Vec<(String, crate::game::state::GameState)> = entries_to_extend
                .into_iter()
                .map(|e| (e.label, e.state))
                .collect();
            session.extend_timeline(pairs);
        }

        if self.pending_actions.is_empty() {
            session.set_final_state(self.committed_state.clone());
        }

        let mut msgs = vec![session.state_msg()];
        // Include a snapshot so the tree view refreshes after state changes
        // (otherwise the client keeps displaying the stale search snapshot).
        if let Some((snap, labels)) = session.snapshot_with_labels() {
            msgs.push(canopy::server::ServerMsg::Snapshot {
                snapshot: snap,
                action_labels: labels,
            });
        }
        msgs
    }
}

/// Poll interval during search — how often to check colonist for state changes.
const SEARCH_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Progress update interval (matches server::PROGRESS_INTERVAL).
const PROGRESS_INTERVAL: u32 = 10;

/// Run MCTS search with interleaved colonist polling.
///
/// Every `SEARCH_POLL_INTERVAL`, cancels the active search, polls colonist for
/// state changes (which may reroot the tree), resets the sim budget, and
/// resumes searching. This ensures state changes on colonist are detected
/// during long searches.
async fn colonist_run_search(
    socket: &mut WebSocket,
    session: &mut canopy::server::GameSession<crate::game::state::GameState>,
    msg: &canopy::server::ClientMsg,
    poll: &mut ColonistPollState,
) -> Result<Vec<canopy::server::ServerMsg>, ()> {
    let sims_total = match session.begin_search(msg) {
        Err(msgs) => return Ok(msgs),
        Ok(n) => n,
    };

    // Track sims across poll-interrupted segments. Each segment measures
    // root_visits() before/after to count sims done, since poll() may
    // reroot or reset the tree (changing root_visits discontinuously).
    let mut total_sims_done = 0u32;
    let mut segment_baseline = session.root_visits();

    let mut evals = vec![];
    let mut last_progress = 0u32;
    let mut last_poll = std::time::Instant::now();

    let result = loop {
        // Check if it's time to poll colonist.
        if last_poll.elapsed() >= SEARCH_POLL_INTERVAL {
            // Count sims completed in the segment that just ended.
            total_sims_done += session.root_visits().saturating_sub(segment_baseline);

            session.cancel_search();
            let poll_msgs = poll.poll(session).await;
            for m in &poll_msgs {
                canopy::server::send_msg(socket, m).await?;
            }

            // Start a new segment: capture baseline after any tree changes
            // from poll (walk_tree, rollback, reset).
            segment_baseline = session.root_visits();
            let remaining = sims_total.saturating_sub(total_sims_done);
            session.set_num_simulations(remaining);
            last_poll = std::time::Instant::now();
            // Reset so progress messages fire from the new baseline.
            last_progress = 0;
            evals.clear();
        }

        if let Some(result) = session.search_tick(&mut evals) {
            break result;
        }

        // Send progress updates.
        if let Some((snap, labels)) = session.snapshot_with_labels() {
            if snap.total_simulations >= last_progress + PROGRESS_INTERVAL {
                last_progress = snap.total_simulations;
                canopy::server::send_msg(
                    socket,
                    &canopy::server::ServerMsg::SearchProgress {
                        snapshot: snap,
                        action_labels: labels,
                        sims_total,
                    },
                )
                .await?;
                if let Some(subtree_msg) = session.explore_subtree_msg() {
                    let _ = canopy::server::send_msg(socket, &subtree_msg).await;
                }
            }
        }

        // Non-blocking: process interactive messages (tree exploration)
        // that arrive while the search is running.
        if let Some(Some(Ok(ws_msg))) = socket.recv().now_or_never() {
            if let Message::Text(text) = ws_msg {
                if let Ok(client_msg) = serde_json::from_str::<canopy::server::ClientMsg>(&text) {
                    let responses = match client_msg {
                        canopy::server::ClientMsg::ExploreSubtree { .. }
                        | canopy::server::ClientMsg::GetSnapshot => session.handle(client_msg),
                        _ => vec![],
                    };
                    for m in responses {
                        canopy::server::send_msg(socket, &m).await?;
                    }
                }
            }
        }
    };

    Ok(session.finish_search(msg, result))
}

/// Handle a colonist WebSocket connection (sequential, single-threaded).
async fn handle_colonist_socket(
    mut socket: WebSocket,
    session: &mut canopy::server::GameSession<crate::game::state::GameState>,
    poll: &mut ColonistPollState,
) {
    // Send initial state.
    let msg = session.state_msg();
    if canopy::server::send_msg(&mut socket, &msg).await.is_err() {
        return;
    }

    loop {
        let ws_msg = match socket.recv().await {
            Some(Ok(msg)) => msg,
            _ => break,
        };
        let text = match ws_msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };
        let client_msg: canopy::server::ClientMsg = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let _ = canopy::server::send_msg(
                    &mut socket,
                    &canopy::server::ServerMsg::Error {
                        message: format!("Invalid message: {e}"),
                    },
                )
                .await;
                continue;
            }
        };

        let responses = match &client_msg {
            canopy::server::ClientMsg::PollState => {
                eprintln!("ws: PollState");
                poll.poll(session).await
            }
            canopy::server::ClientMsg::BotMove { .. }
            | canopy::server::ClientMsg::RunSims { .. } => {
                eprintln!("ws: {:?}", client_msg);
                match colonist_run_search(&mut socket, session, &client_msg, poll).await {
                    Ok(msgs) => {
                        for m in &msgs {
                            match m {
                                canopy::server::ServerMsg::Snapshot { snapshot, .. } => {
                                    eprintln!(
                                        "ws: → Snapshot ({} sims, {} edges)",
                                        snapshot.total_simulations,
                                        snapshot.edges.len()
                                    );
                                }
                                canopy::server::ServerMsg::Error { message } => {
                                    eprintln!("ws: → Error: {message}");
                                }
                                _ => {}
                            }
                        }
                        msgs
                    }
                    Err(()) => return,
                }
            }
            canopy::server::ClientMsg::PlayAction { action } => {
                if poll.pending_actions.is_empty() {
                    poll.pre_pending_cursor = Some(session.cursor());
                }
                poll.pending_actions.push(*action);
                session.handle(client_msg)
            }
            _ => session.handle(client_msg),
        };

        for msg in responses {
            if canopy::server::send_msg(&mut socket, &msg).await.is_err() {
                return;
            }
        }
    }
}

/// Entry point: extract game data and serve via web analysis board with client-driven polling.
pub fn run_serve(
    cdp_port: u16,
    serve_port: u16,
    evaluator: Arc<dyn canopy::eval::Evaluator<crate::game::state::GameState> + Sync>,
    eval_name: &str,
) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let data = rt
        .block_on(extract_game_data(cdp_port))
        .expect("failed to extract game data");
    log::print(&data.events);
    board::print(&data.board);

    let mapper = board::CoordMapper::detect(&data.board.tiles);
    let mut timeline = state::build_timeline(&data.board, &data.events, &mapper);
    // Derive color → Player mapping from the event log (first to act = P1).
    let mut color_map = state::discover_colors(&data.events);
    for &(color, _) in &data.player_names {
        if color_map.iter().any(|&(c, _)| c == color) {
            continue;
        }
        let pid = if color_map.is_empty() {
            canopy::player::Player::One
        } else {
            canopy::player::Player::Two
        };
        color_map.push((color, pid));
        if color_map.len() >= 2 {
            break;
        }
    }
    let initial_event_count = data.events.len();

    // Apply live state to the final timeline entry.
    if let Some(last) = timeline.last_mut() {
        apply_live_state(&mut last.state, &data, &color_map, &mapper);
    }

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
    let presenter =
        Arc::new(CatanPresenter::new(static_dir.clone(), dice).with_player_names(names));
    let mcts_config = canopy::mcts::Config {
        filter_legal: true,
        ..canopy::mcts::Config::default()
    };
    let mut session = canopy::server::GameSession::with_state(
        timeline_pairs[0].1.clone(),
        evaluator,
        eval_name,
        presenter,
        [true, true],
        mcts_config,
    );
    session.load_timeline(timeline_pairs);
    session.seek_to_end();

    // Build coordinate maps for incremental processing.
    let (terrains, numbers, port_resources, port_specs) = board::to_layout(&data.board, &mapper);
    let topology = Arc::new(crate::game::topology::Topology::from_layout_with_ports(
        terrains,
        numbers,
        port_resources,
        port_specs,
    ));
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    let poll_state = ColonistPollState {
        cdp_port,
        committed_event_count: initial_event_count,
        committed_state: last_state,
        color_map,
        corner_map,
        edge_map,
        mapper,
        pending_actions: Vec::new(),
        pre_pending_cursor: None,
    };

    // Wrap in Mutex for the axum handler (single client, no real contention).
    let session = Arc::new(tokio::sync::Mutex::new(session));
    let poll_state = Arc::new(tokio::sync::Mutex::new(poll_state));

    let app = axum::Router::new()
        .route(
            "/ws",
            axum::routing::get({
                let session = session.clone();
                let poll_state = poll_state.clone();
                move |ws: axum::extract::ws::WebSocketUpgrade| {
                    let session = session.clone();
                    let poll_state = poll_state.clone();
                    async move {
                        ws.on_upgrade(move |socket| async move {
                            let mut session = session.lock().await;
                            let mut poll = poll_state.lock().await;
                            handle_colonist_socket(socket, &mut session, &mut poll).await;
                        })
                    }
                }
            }),
        )
        .fallback_service(tower_http::services::ServeDir::new(&static_dir));

    eprintln!("serving colonist board on port {serve_port} (client-driven polling)");
    let addr = format!("0.0.0.0:{serve_port}");
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        println!("Analysis board: http://localhost:{serve_port}");
        axum::serve(listener, app).await.unwrap();
    });
}
