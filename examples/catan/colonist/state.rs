//! Build a `GameState` from colonist.io board data.
//!
//! Creates a topology from the colonist board layout, places buildings from the
//! tileCornerStates/tileEdgeStates snapshot, and replays the game log to track
//! resource hands and dice state.
//!
//! `build_timeline` produces a sequence of labelled snapshots for undo/redo
//! navigation. Full-log mode replays setup through the game action system;
//! late-join mode falls back to the board snapshot.

use std::collections::HashMap;
use std::sync::Arc;

use canopy::game::Game;
use canopy::player::{PerPlayer, Player};

use crate::game::board::{EdgeId, NodeId, TileId};
use crate::game::dev_card::{DevCardDeck, DevCardKind};
use crate::game::dice::{BalancedDice, Dice};
use crate::game::resource::ResourceArray;
use crate::game::state::{GameState, Phase};
use crate::game::topology::Topology;

use super::board::{self, BoardData, CoordMapper};
use super::log::GameEvent;

/// A snapshot in the game timeline for undo/redo navigation.
#[derive(Clone)]
pub struct TimelineEntry {
    pub label: String,
    pub state: GameState,
}

/// Build a timeline of game states from colonist.io data.
///
/// Uses search-based replay: walks the log event by event, mapping each to
/// engine actions. The engine handles all side effects (resources, VPs, etc.).
/// Falls back to the legacy board-snapshot approach on failure.
pub fn build_timeline(
    board: &BoardData,
    events: &[GameEvent],
    mapper: &CoordMapper,
) -> Vec<TimelineEntry> {
    let (terrains, numbers, port_resources, port_specs) = board::to_layout(board, mapper);
    let topology = Arc::new(Topology::from_layout_with_ports(
        terrains,
        numbers,
        port_resources,
        port_specs,
    ));
    let color_map = discover_colors(events);
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    if let Some(timeline) = replay_search(
        topology.clone(),
        board,
        events,
        &color_map,
        &corner_map,
        &edge_map,
        mapper,
    ) {
        timeline
    } else {
        eprintln!("replay_search failed — falling back to board snapshot");
        vec![TimelineEntry {
            label: "Current position".into(),
            state: build_game_state(board, events, mapper),
        }]
    }
}

/// Build a `GameState` from colonist board data and game log.
///
/// Places buildings from the board snapshot (tileCornerStates/tileEdgeStates),
/// replays the log for resource tracking, and sets up dice state.
pub fn build_game_state(
    board: &BoardData,
    events: &[GameEvent],
    mapper: &CoordMapper,
) -> GameState {
    let (terrains, numbers, port_resources, port_specs) = board::to_layout(board, mapper);
    let topology = Arc::new(Topology::from_layout_with_ports(
        terrains,
        numbers,
        port_resources,
        port_specs,
    ));
    let dev_deck = DevCardDeck::new();
    let dice = Dice::Balanced(BalancedDice::new());

    let mut state = GameState::new(topology.clone(), dev_deck, dice);

    // Build coordinate maps for colonist → canopy mapping
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    // Discover color → Player mapping from the log
    let color_map = discover_colors(events);

    // Place buildings from board snapshot (dedup checks are no-ops on fresh state)
    let buildings = board::extract_buildings(board, mapper);
    let sync = sync_buildings(
        &mut state,
        &buildings,
        &color_map,
        &corner_map,
        &edge_map,
        mapper,
    );
    if sync.settlements + sync.cities + sync.roads > 0 {
        eprintln!(
            "placed {} settlements, {} cities, {} roads from board snapshot",
            sync.settlements, sync.cities, sync.roads
        );
    }

    // Replay log for resource hands + dice
    replay_log(&mut state, events, &color_map);

    // Robber
    if let Some(idx) = buildings.robber_tile_index {
        state.robber = TileId(idx);
    }

    // Derive phase from building counts.
    sync_setup_phase(&mut state);

    // Override phase based on last log event — sync_setup_phase only looks at
    // building counts, so it can't distinguish pre-roll from post-roll.
    sync_phase_from_log(&mut state, events, &color_map);

    state
}

/// Discover the first two unique player colors from the log, mapping them to
/// Player::One and Player::Two.
///
/// Uses only placement events (PlaceSettlement/PlaceRoad) — colonist logs
/// "roll for first player" before setup, and those don't reflect turn order.
pub fn discover_colors(events: &[GameEvent]) -> Vec<(u8, Player)> {
    let mut map: Vec<(u8, Player)> = Vec::new();
    for event in events {
        let color = match event {
            GameEvent::PlaceSettlement { player, .. } | GameEvent::PlaceRoad { player, .. } => {
                *player
            }
            _ => continue,
        };
        if map.iter().any(|&(c, _)| c == color) {
            continue;
        }
        let p = if map.is_empty() {
            Player::One
        } else {
            Player::Two
        };
        map.push((color, p));
        if map.len() >= 2 {
            break;
        }
    }
    map
}

/// Look up a colonist color in the color map, returning the internal `Player`.
pub fn player_of_color(color_map: &[(u8, Player)], color: u8) -> Option<Player> {
    color_map
        .iter()
        .find(|&&(c, _)| c == color)
        .map(|&(_, p)| p)
}

/// Record a dev card being played: decrement hidden count, increment played count,
/// and update largest army when a knight is played.
///
/// Used by `replay_log` (late-join path) which doesn't route through the engine.
fn play_dev_card(state: &mut GameState, pid: Player, kind: DevCardKind) {
    state.players[pid].hidden_dev_cards = state.players[pid].hidden_dev_cards.saturating_sub(1);
    state.players[pid].dev_cards_played[kind] += 1;
    if kind == DevCardKind::Knight {
        state.players[pid].knights_played += 1;
        let knights = state.players[pid].knights_played;
        if knights >= 3 {
            let beats = match state.largest_army {
                Some((_, n)) => knights > n,
                None => true,
            };
            if beats {
                state.largest_army = Some((pid, knights));
            }
        }
    }
}

/// Walk the log to track resource hands and dice state.
fn replay_log(state: &mut GameState, events: &[GameEvent], color_map: &[(u8, Player)]) {
    use crate::game::resource::{CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST};

    for event in events {
        match event {
            GameEvent::Roll { player, d1, d2 } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    // Clear per-turn state for the previous player (turn change).
                    let prev = pid.opponent();
                    state.players[prev].dev_cards_bought_this_turn = Default::default();
                    state.players[prev].hidden_dev_cards_bought_this_turn = 0;
                    state.players[prev].has_played_dev_card_this_turn = false;
                    if let Dice::Balanced(ref mut b) = state.dice {
                        b.draw(d1 + d2, pid);
                    }
                }
            }
            GameEvent::StartingResources { player, resources }
            | GameEvent::GotResources { player, resources } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }
            GameEvent::BuildRoad { player, .. } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(ROAD_COST);
                    state.bank.add(ROAD_COST);
                }
            }
            GameEvent::BuildSettlement { player, .. } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(SETTLEMENT_COST);
                    state.bank.add(SETTLEMENT_COST);
                }
            }
            GameEvent::BuildCity { player, .. } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(CITY_COST);
                    state.bank.add(CITY_COST);
                }
            }
            GameEvent::BuyDevCard { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(DEV_CARD_COST);
                    state.bank.add(DEV_CARD_COST);
                    state.players[pid].hidden_dev_cards += 1;
                    state.players[pid].hidden_dev_cards_bought_this_turn += 1;
                    state.dev_deck.total -= 1;
                }
            }
            GameEvent::Stole {
                player,
                victim,
                resources,
            } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                }
                if let Some(vid) = player_of_color(color_map, *victim) {
                    state.players[vid].hand.sub(*resources);
                }
            }
            GameEvent::YearOfPlentyGain { player, resources } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }
            GameEvent::MonopolyResult {
                player,
                count,
                resource,
            } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    let mut gained = ResourceArray::default();
                    gained[*resource] = *count;
                    state.players[pid].hand.add(gained);
                    let opp = pid.opponent();
                    let lost = state.players[opp].hand[*resource].min(*count);
                    let mut sub = ResourceArray::default();
                    sub[*resource] = lost;
                    state.players[opp].hand.sub(sub);
                }
            }
            GameEvent::PlayerTrade {
                player,
                counterparty,
                given,
                received,
            } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                }
                if let Some(cpid) = player_of_color(color_map, *counterparty) {
                    state.players[cpid].hand.sub(*received);
                    state.players[cpid].hand.add(*given);
                }
            }
            GameEvent::BankTrade {
                player,
                given,
                received,
            } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                    state.bank.add(*given);
                    state.bank.sub(*received);
                }
            }
            GameEvent::Discard {
                player, resources, ..
            } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(*resources);
                    state.bank.add(*resources);
                }
            }
            GameEvent::PlayedKnight { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    play_dev_card(state, pid, DevCardKind::Knight);
                }
            }
            GameEvent::PlayedMonopoly { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    play_dev_card(state, pid, DevCardKind::Monopoly);
                }
            }
            GameEvent::PlayedRoadBuilding { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    play_dev_card(state, pid, DevCardKind::RoadBuilding);
                }
            }
            GameEvent::PlayedYearOfPlenty { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    play_dev_card(state, pid, DevCardKind::YearOfPlenty);
                }
            }
            _ => {}
        }
    }
}

/// Update trade ratios for a player based on a node's port.
fn update_port_access(state: &mut GameState, pid: Player, nid: NodeId) {
    if let Some(port) = state.topology.nodes[nid.0 as usize].port {
        match port {
            crate::game::board::Port::Specific(r) => {
                state.players[pid].trade_ratios[r as usize] =
                    state.players[pid].trade_ratios[r as usize].min(2);
            }
            crate::game::board::Port::Generic => {
                for ratio in &mut state.players[pid].trade_ratios {
                    *ratio = (*ratio).min(3);
                }
            }
        }
    }
}

/// Place a settlement on the board without going through the game action system.
fn place_settlement(state: &mut GameState, pid: Player, nid: NodeId) {
    state.boards[pid].settlements |= 1u64 << nid.0;
    state.players[pid].building_vps += 1;
    state.players[pid].settlements_left = state.players[pid].settlements_left.saturating_sub(1);
    update_port_access(state, pid, nid);

    let opp = pid.opponent();
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid]
        .road_network
        .add_building(nid, &state.topology.adj, opp_roads);

    // Break opponent's road continuity through this node.
    let opp_own_buildings = state.boards[opp].settlements | state.boards[opp].cities;
    let pid_buildings = state.boards[pid].settlements | state.boards[pid].cities;
    state.boards[opp].road_network.on_opponent_build(
        nid,
        &state.topology.adj,
        opp_own_buildings,
        pid_buildings,
    );
}

/// Place a city on the board.
fn place_city(state: &mut GameState, pid: Player, nid: NodeId) {
    state.boards[pid].settlements &= !(1u64 << nid.0);
    state.boards[pid].cities |= 1u64 << nid.0;
    // City gives 2 VP total (1 from settlement + 1 upgrade).
    // If settlement wasn't already counted, give full 2.
    if state.players[pid].building_vps == 0 {
        state.players[pid].building_vps = 2;
    } else {
        state.players[pid].building_vps += 1;
    }
    state.players[pid].cities_left = state.players[pid].cities_left.saturating_sub(1);
    // Upgrading a settlement returns the settlement piece.
    state.players[pid].settlements_left += 1;
    update_port_access(state, pid, nid);

    let opp = pid.opponent();
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid]
        .road_network
        .add_building(nid, &state.topology.adj, opp_roads);
}

/// Place a road on the board.
fn place_road(state: &mut GameState, pid: Player, eid: crate::game::board::EdgeId) {
    let opp = pid.opponent();
    let opp_buildings = state.player_buildings(opp);
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid].road_network.add_road(
        eid,
        &state.topology.adj,
        opp_buildings,
        opp_roads,
        true,
    );
    state.players[pid].roads_placed += 1;
    state.players[pid].roads_left = state.players[pid].roads_left.saturating_sub(1);

    // Remove from opponent's frontier
    state.boards[opp].road_network.remove_edge(eid);
}

/// Result of syncing buildings from a board snapshot.
pub struct SyncBuildingsResult {
    pub settlements: u32,
    pub cities: u32,
    pub roads: u32,
    pub last_settlement: Option<NodeId>,
    /// Action IDs corresponding to each newly placed building, in placement order.
    /// Useful for walking the MCTS tree during setup.
    pub walk_actions: Vec<usize>,
}

/// Sync buildings from a board snapshot onto the game state.
///
/// Only places buildings that are not already present (checked via bitfields).
/// Returns a `SyncBuildingsResult` with counts, the last settlement node, and
/// action IDs for tree walking.
pub fn sync_buildings(
    state: &mut GameState,
    buildings: &board::BuildingData,
    color_map: &[(u8, Player)],
    corner_map: &std::collections::HashMap<(i32, i32, u8), NodeId>,
    edge_map: &std::collections::HashMap<(i32, i32, u8), EdgeId>,
    mapper: &CoordMapper,
) -> SyncBuildingsResult {
    use crate::game::action;

    let mut result = SyncBuildingsResult {
        settlements: 0,
        cities: 0,
        roads: 0,
        last_settlement: None,
        walk_actions: Vec::new(),
    };

    for &(color, x, y, z) in &buildings.settlements {
        let Some(pid) = player_of_color(color_map, color) else {
            continue;
        };
        let (mx, my, mz) = mapper.map_corner(x, y, z);
        let Some(&nid) = corner_map.get(&(mx, my, mz)) else {
            continue;
        };
        if state.boards[pid].settlements & (1u64 << nid.0) == 0 {
            place_settlement(state, pid, nid);
            result.last_settlement = Some(nid);
            result.settlements += 1;
            result
                .walk_actions
                .push(action::settlement_id(nid).0 as usize);
        }
    }

    for &(color, x, y, z) in &buildings.cities {
        let Some(pid) = player_of_color(color_map, color) else {
            continue;
        };
        let (mx, my, mz) = mapper.map_corner(x, y, z);
        let Some(&nid) = corner_map.get(&(mx, my, mz)) else {
            continue;
        };
        if state.boards[pid].cities & (1u64 << nid.0) == 0 {
            place_city(state, pid, nid);
            result.cities += 1;
            result.walk_actions.push(action::city_id(nid).0 as usize);
        }
    }

    for &(color, x, y, z) in &buildings.roads {
        let Some(pid) = player_of_color(color_map, color) else {
            continue;
        };
        let (mx, my, mz) = mapper.map_edge(x, y, z);
        let Some(&eid) = edge_map.get(&(mx, my, mz)) else {
            continue;
        };
        if state.boards[pid].road_network.roads & (1u128 << eid.0) == 0 {
            place_road(state, pid, eid);
            result.roads += 1;
            result.walk_actions.push(action::road_id(eid).0 as usize);
        }
    }

    result
}

/// Derive the correct setup phase from building counts on the board.
///
/// During setup (4 settlements + 4 roads), the phase alternates between
/// PlaceSettlement and PlaceRoad. The turn order is snake-draft:
/// P1, P2, P2, P1. Once setup is complete, transitions to PreRoll.
///
/// `last_settlement` is the NodeId of the last settlement placed by
/// `sync_buildings`. When the phase is `PlaceRoad`, this is stored in
/// `state.last_setup_node` so `populate_place_road` picks the right node.
pub fn sync_setup_phase(state: &mut GameState) {
    let total_settlements = state.boards[Player::One].settlements.count_ones()
        + state.boards[Player::Two].settlements.count_ones();
    let total_cities = state.boards[Player::One].cities.count_ones()
        + state.boards[Player::Two].cities.count_ones();
    let total_corners = total_settlements + total_cities;
    let total_roads = state.boards[Player::One].road_network.roads.count_ones()
        + state.boards[Player::Two].road_network.roads.count_ones();

    if total_corners >= 4 && total_roads >= 4 {
        // Setup complete.
        state.setup_count = 4;
        // Only transition to PreRoll if the game is still in a setup phase.
        // After the first roll, the phase will already be Main (or later) and
        // we must not overwrite it.
        if matches!(state.phase, Phase::PlaceSettlement | Phase::PlaceRoad) {
            state.phase = Phase::PreRoll;
            state.pre_roll = true;
            state.turn_number = state.turn_number.max(1);
            state.current_player = Player::One;
        }
        return;
    }

    state.setup_count = total_corners as u8;

    if total_corners > total_roads {
        state.phase = Phase::PlaceRoad;
    } else {
        state.phase = Phase::PlaceSettlement;
    }

    // Snake draft turn order: P1, P2, P2, P1.
    // setup_count = total corners placed.
    //
    // PlaceSettlement: setup_count indexes the *next* settlement to place.
    //   0 → P1, 1 → P2, 2 → P2, 3 → P1
    //
    // PlaceRoad: setup_count indexes the settlement that was *just* placed,
    // so the road-placer is one step behind (setup_count - 1).
    let idx = if state.phase == Phase::PlaceRoad {
        state.setup_count.saturating_sub(1)
    } else {
        state.setup_count
    };
    state.current_player = match idx {
        0 | 3 => Player::One,
        1 | 2 => Player::Two,
        _ => Player::One,
    };

    // Find the current player's settlement that still needs a road.
    // sync_buildings returns the last settlement in iteration order, which
    // may belong to the opponent.
    if state.phase == Phase::PlaceRoad {
        let pid = state.current_player;
        let my_roads = state.boards[pid].road_network.roads;
        let adj = &state.topology.adj;
        let mut settlements = state.boards[pid].settlements;
        while settlements != 0 {
            let bit = settlements.trailing_zeros() as u8;
            settlements &= settlements - 1;
            if adj.node_adj_edges[bit as usize] & my_roads == 0 {
                state.last_setup_node = Some(NodeId(bit));
                break;
            }
        }
    }
}

/// Check whether the current player played a dev card this turn by scanning
/// the event log backwards. Returns true if a dev card play event appears
/// before the opponent's roll (which marks the turn boundary).
///
/// Pre-roll dev cards happen *before* the current player's own roll, so the
/// current player's roll is NOT a turn boundary — the opponent's roll is.
pub fn played_dev_card_this_turn(
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    current_player: Player,
) -> bool {
    let opponent = current_player.opponent();
    for event in events.iter().rev() {
        match event {
            GameEvent::PlayedKnight { player }
            | GameEvent::PlayedMonopoly { player }
            | GameEvent::PlayedRoadBuilding { player }
            | GameEvent::PlayedYearOfPlenty { player } => {
                if player_of_color(color_map, *player) == Some(current_player) {
                    return true;
                }
            }
            // Opponent's roll marks the start of the current player's turn.
            GameEvent::Roll { player, .. } => {
                if player_of_color(color_map, *player) == Some(opponent) {
                    return false;
                }
            }
            _ => {}
        }
    }
    false
}

/// Refine the game phase by scanning the log tail for recent events.
///
/// `sync_setup_phase` sets PreRoll once setup is complete, but can't tell if a
/// roll already happened. This function walks backwards from the last event to
/// determine the actual phase:
///   - Last significant event is Roll → Phase::Main, pre_roll=false
///   - Last significant event is EndTurn / no post-setup events → PreRoll
///   - Build/Buy/Trade after a Roll → Phase::Main (still that player's turn)
fn sync_phase_from_log(state: &mut GameState, events: &[GameEvent], color_map: &[(u8, Player)]) {
    // Only relevant once setup is complete.
    if state.setup_count < 4 {
        return;
    }

    // Walk backwards to find the last phase-determining event.
    // Track whether we passed through robber-flow events (MoveRobber, Stole,
    // Discard) before hitting the Roll, which tells us the 7-flow completed.
    let mut saw_move_robber = false;
    for event in events.iter().rev() {
        match event {
            GameEvent::Roll { player, d1, d2 } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.pre_roll = false;
                    state.current_player = pid;
                    if d1 + d2 == 7 && !saw_move_robber {
                        // Rolled 7 but no MoveRobber yet — still in robber flow.
                        crate::game::handle_seven(state);
                    } else {
                        state.phase = Phase::Main;
                    }
                }
                return;
            }
            GameEvent::MoveRobber { .. } => {
                saw_move_robber = true;
                continue;
            }
            GameEvent::Stole { .. } => continue,
            GameEvent::Discard { .. } | GameEvent::RolledSeven => continue,
            GameEvent::PlayedKnight { player } => {
                if !saw_move_robber {
                    // Knight played but no MoveRobber yet → MoveRobber phase.
                    if let Some(pid) = player_of_color(color_map, *player) {
                        state.current_player = pid;
                        state.phase = Phase::MoveRobber;
                    }
                    return;
                }
                continue;
            }
            // These happen within a turn (after a roll) — still Main phase.
            GameEvent::BuildRoad { .. }
            | GameEvent::BuildSettlement { .. }
            | GameEvent::BuildCity { .. }
            | GameEvent::BuyDevCard { .. }
            | GameEvent::PlayerTrade { .. }
            | GameEvent::BankTrade { .. }
            | GameEvent::PlayedMonopoly { .. }
            | GameEvent::PlayedRoadBuilding { .. }
            | GameEvent::PlayedYearOfPlenty { .. } => continue,
            // Resource events, starting resources, etc. — not phase-determining.
            _ => continue,
        }
    }
    // No roll found — stay at PreRoll (start of first post-setup turn).
}

// -- Timeline building --------------------------------------------------------

fn player_label(color: u8, color_map: &[(u8, Player)]) -> &'static str {
    match player_of_color(color_map, color) {
        Some(Player::One) => "P1",
        Some(Player::Two) => "P2",
        None => "P?",
    }
}

/// Format a `ResourceArray` as a compact string like "2 wool, 1 grain".
fn format_resources(r: ResourceArray) -> String {
    use crate::game::resource::ALL_RESOURCES;
    let parts: Vec<String> = ALL_RESOURCES
        .iter()
        .filter(|&&res| r[res] > 0)
        .map(|&res| format!("{} {res}", r[res]))
        .collect();
    parts.join(", ")
}

// -- Search-based log replay ---------------------------------------------------

/// Shared context for replay, avoiding repeated parameter passing.
pub(crate) struct ReplayCtx<'a> {
    pub color_map: &'a [(u8, Player)],
    pub corner_map: &'a HashMap<(i32, i32, u8), NodeId>,
    pub edge_map: &'a HashMap<(i32, i32, u8), EdgeId>,
    pub mapper: &'a CoordMapper,
    pub dom: board::BuildingData,
    pub dom_settlements: PerPlayer<Vec<NodeId>>,
    pub dom_roads: PerPlayer<Vec<EdgeId>>,
}

/// Replay new events onto an existing state using engine actions.
/// Used by the live polling path to process incremental events.
/// Returns `Some(entries)` on success (state updated), `None` on failure
/// (state unchanged — caller should retry with more events later).
pub fn replay_events(
    state: &mut GameState,
    events: &[GameEvent],
    ctx: &ReplayCtx,
) -> Option<Vec<TimelineEntry>> {
    let timeline = Vec::new();
    let entries = try_replay(state.clone(), events, 0, ctx, timeline)?;
    if let Some(last) = entries.last() {
        *state = last.state.clone();
    }
    Some(entries)
}

impl<'a> ReplayCtx<'a> {
    pub fn from_buildings(
        dom: board::BuildingData,
        color_map: &'a [(u8, Player)],
        corner_map: &'a HashMap<(i32, i32, u8), NodeId>,
        edge_map: &'a HashMap<(i32, i32, u8), EdgeId>,
        mapper: &'a CoordMapper,
    ) -> Self {
        let mut dom_settlements: PerPlayer<Vec<NodeId>> = PerPlayer::default();
        let mut dom_roads: PerPlayer<Vec<EdgeId>> = PerPlayer::default();
        for &(color, x, y, z) in &dom.settlements {
            let Some(pid) = player_of_color(color_map, color) else {
                continue;
            };
            let mapped = mapper.map_corner(x, y, z);
            if let Some(&nid) = corner_map.get(&mapped) {
                dom_settlements[pid].push(nid);
            }
        }
        for &(color, x, y, z) in &dom.cities {
            let Some(pid) = player_of_color(color_map, color) else {
                continue;
            };
            let mapped = mapper.map_corner(x, y, z);
            if let Some(&nid) = corner_map.get(&mapped) {
                dom_settlements[pid].push(nid);
            }
        }
        for &(color, x, y, z) in &dom.roads {
            let Some(pid) = player_of_color(color_map, color) else {
                continue;
            };
            let mapped = mapper.map_edge(x, y, z);
            if let Some(&eid) = edge_map.get(&mapped) {
                dom_roads[pid].push(eid);
            }
        }
        ReplayCtx {
            color_map,
            corner_map,
            edge_map,
            mapper,
            dom,
            dom_settlements,
            dom_roads,
        }
    }
}

/// Replay the game log through the engine's action system, using search to
/// resolve ambiguous events (placements without coordinates, robber moves).
///
/// Returns `Some(timeline)` on success, `None` if no consistent replay exists.
fn replay_search(
    topology: Arc<Topology>,
    board: &BoardData,
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
    mapper: &CoordMapper,
) -> Option<Vec<TimelineEntry>> {
    let dev_deck = DevCardDeck::new();
    let dice = Dice::Balanced(BalancedDice::new());
    let state = GameState::new(topology, dev_deck, dice);

    let dom = board::extract_buildings(board, mapper);
    let ctx = ReplayCtx::from_buildings(dom, color_map, corner_map, edge_map, mapper);

    let timeline = vec![TimelineEntry {
        label: "Game start".into(),
        state: state.clone(),
    }];

    let result = try_replay(state, events, 0, &ctx, timeline)?;
    eprintln!("replay_search: {} timeline entries", result.len(),);
    Some(result)
}

/// Recursive replay: process events from `idx`, branching at ambiguous points.
/// Returns `None` on contradiction (GotResources mismatch, no valid candidate).
fn try_replay(
    mut state: GameState,
    events: &[GameEvent],
    idx: usize,
    ctx: &ReplayCtx,
    mut timeline: Vec<TimelineEntry>,
) -> Option<Vec<TimelineEntry>> {
    use crate::game::action::{self, END_TURN, ROLL};
    use crate::game::resource::ALL_RESOURCES;

    // Ensure current_player matches the event's player. Pre-roll actions
    // (Knight, etc.) from the next player arrive before their Roll event,
    // so we need to inject END_TURN when we see a different player acting.
    let ensure_player = |state: &mut GameState, player: u8| {
        if let Some(pid) = player_of_color(ctx.color_map, player) {
            if state.current_player != pid && matches!(state.phase, Phase::Main) {
                state.apply_action(END_TURN as usize);
            }
        }
    };

    let mut i = idx;
    while i < events.len() {
        match &events[i] {
            // -- Setup placements (with backtracking) ----------------------
            GameEvent::PlaceSettlement { player, corner }
                if matches!(state.phase, Phase::PlaceSettlement) =>
            {
                let pid = player_of_color(ctx.color_map, *player);
                let nid_from_coords = corner
                    .map(|(x, y, z)| ctx.mapper.map_corner(x, y, z))
                    .and_then(|c| ctx.corner_map.get(&c).copied());

                let mut candidates: Vec<NodeId> = if let Some(nid) = nid_from_coords {
                    vec![nid]
                } else if let Some(pid) = pid {
                    ctx.dom_settlements[pid]
                        .iter()
                        .filter(|&&nid| {
                            state.boards[pid].settlements & (1u64 << nid.0) == 0
                                && state.boards[pid].cities & (1u64 << nid.0) == 0
                        })
                        .copied()
                        .collect()
                } else {
                    vec![]
                };

                // Filter 2nd settlements by StartingResources.
                let setup_after = state.setup_count + 1;
                let is_second = setup_after == 3 || setup_after == 4;
                if is_second {
                    let starting = events[i + 1..].iter().find_map(|e| match e {
                        GameEvent::StartingResources {
                            player: p,
                            resources,
                        } if *p == *player => Some(*resources),
                        _ => None,
                    });
                    if let Some(expected) = starting {
                        candidates.retain(|&nid| {
                            let node = &state.topology.nodes[nid.0 as usize];
                            let mut would_give = ResourceArray::default();
                            for &tid in &node.adjacent_tiles {
                                let tile = &state.topology.tiles[tid.0 as usize];
                                if let Some(resource) = tile.terrain.resource() {
                                    would_give[resource] += 1;
                                }
                            }
                            would_give == expected
                        });
                    }
                }

                if candidates.len() == 1 {
                    state.apply_action(candidates[0].0 as usize);
                    timeline.push(TimelineEntry {
                        label: format!(
                            "{} places settlement",
                            player_label(*player, ctx.color_map)
                        ),
                        state: state.clone(),
                    });
                } else if candidates.is_empty() {
                    return None;
                } else {
                    // Branch: try each candidate, recurse with full replay.
                    for &nid in &candidates {
                        let mut trial = state.clone();
                        trial.apply_action(nid.0 as usize);
                        let mut trial_tl = timeline.clone();
                        trial_tl.push(TimelineEntry {
                            label: format!(
                                "{} places settlement",
                                player_label(*player, ctx.color_map)
                            ),
                            state: trial.clone(),
                        });
                        if let Some(result) = try_replay(trial, events, i + 1, ctx, trial_tl) {
                            return Some(result);
                        }
                    }
                    return None; // all branches failed
                }
            }

            GameEvent::PlaceRoad { player, edge } if matches!(state.phase, Phase::PlaceRoad) => {
                let eid = edge
                    .map(|(x, y, z)| ctx.mapper.map_edge(x, y, z))
                    .and_then(|e| ctx.edge_map.get(&e).copied())
                    .or_else(|| {
                        let pid = player_of_color(ctx.color_map, *player)?;
                        ctx.dom_roads[pid]
                            .iter()
                            .find(|&&eid| {
                                state.boards[pid].road_network.roads & (1u128 << eid.0) == 0
                            })
                            .copied()
                    });
                if let Some(eid) = eid {
                    state.apply_action((54 + eid.0) as usize);
                    timeline.push(TimelineEntry {
                        label: format!("{} places road", player_label(*player, ctx.color_map)),
                        state: state.clone(),
                    });
                }
            }

            GameEvent::StartingResources { player, resources } => {
                if let Some(pid) = player_of_color(ctx.color_map, *player) {
                    if state.players[pid].hand != *resources {
                        return None; // contradiction
                    }
                }
            }

            // -- Rolls with validation -------------------------------------
            GameEvent::Roll { player, d1, d2 } => {
                if let Some(pid) = player_of_color(ctx.color_map, *player) {
                    ensure_player(&mut state, *player);
                    if matches!(state.phase, Phase::PreRoll) {
                        state.apply_action(ROLL as usize);
                    }

                    let hands_before: [ResourceArray; 2] = [
                        state.players[Player::One].hand,
                        state.players[Player::Two].hand,
                    ];

                    let chance = (*d1 + *d2 - 2) as usize;
                    state.apply_action(chance);

                    let mut log_gains: [ResourceArray; 2] = Default::default();
                    for e in &events[i + 1..] {
                        match e {
                            GameEvent::GotResources {
                                player: p,
                                resources,
                            } => {
                                if let Some(pid) = player_of_color(ctx.color_map, *p) {
                                    log_gains[pid as usize].add(*resources);
                                }
                            }
                            GameEvent::Roll { .. }
                            | GameEvent::BuildRoad { .. }
                            | GameEvent::BuildSettlement { .. }
                            | GameEvent::BuildCity { .. }
                            | GameEvent::BuyDevCard { .. }
                            | GameEvent::PlayerTrade { .. }
                            | GameEvent::BankTrade { .. }
                            | GameEvent::PlayedKnight { .. }
                            | GameEvent::PlayedMonopoly { .. }
                            | GameEvent::PlayedRoadBuilding { .. }
                            | GameEvent::PlayedYearOfPlenty { .. } => break,
                            _ => {}
                        }
                    }

                    let total = d1 + d2;
                    if total != 7 {
                        for &p in &[Player::One, Player::Two] {
                            let mut engine_gain = state.players[p].hand;
                            engine_gain.sub(hands_before[p as usize]);
                            if engine_gain != log_gains[p as usize] {
                                return None; // resource mismatch → backtrack
                            }
                        }
                    }

                    let _ = pid;
                    let mut label =
                        format!("{} rolls {total}", player_label(*player, ctx.color_map));
                    let p1 = format_resources(log_gains[0]);
                    let p2 = format_resources(log_gains[1]);
                    if !p1.is_empty() {
                        label.push_str(&format!("\n  P1: {p1}"));
                    }
                    if !p2.is_empty() {
                        label.push_str(&format!("\n  P2: {p2}"));
                    }
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                }
            }

            GameEvent::GotResources { .. } | GameEvent::TileBlocked { .. } => {}
            GameEvent::RolledSeven => {}

            // -- Discard ---------------------------------------------------
            GameEvent::Discard { player, resources } => {
                if player_of_color(ctx.color_map, *player).is_some() {
                    for &res in &ALL_RESOURCES {
                        for _ in 0..resources[res] {
                            state.apply_action(action::discard_id(res).0 as usize);
                        }
                    }
                }
            }

            // -- Robber flow (with backtracking) ---------------------------
            GameEvent::MoveRobber { player } => {
                if player_of_color(ctx.color_map, *player).is_some()
                    && matches!(state.phase, Phase::MoveRobber)
                {
                    // If this is the last MoveRobber in the events, the DOM
                    // robber position is authoritative — use it directly.
                    let is_last_robber_move = !events[i + 1..]
                        .iter()
                        .any(|e| matches!(e, GameEvent::MoveRobber { .. }));
                    if is_last_robber_move {
                        if let Some(tile) = ctx.dom.robber_tile_index.map(TileId) {
                            if tile != state.robber {
                                state.apply_action(action::robber_id(tile).0 as usize);
                            } else {
                                // Same tile (shouldn't happen in standard rules).
                                state.phase = if state.pre_roll {
                                    Phase::PreRoll
                                } else {
                                    Phase::Main
                                };
                            }
                            // Skip the search — handled by DOM.
                            i += 1;
                            continue;
                        }
                    }

                    let mut candidates = all_robber_candidates(&state);
                    let opp = state.current_player.opponent();

                    // Filter by TileBlocked: if a subsequent roll shows
                    // "resource (N) blocked by robber", the robber must be
                    // on a tile with that terrain and dice number.
                    for e in &events[i + 1..] {
                        match e {
                            GameEvent::TileBlocked {
                                dice_number,
                                resource: Some(resource),
                            } => {
                                candidates.retain(|&tile| {
                                    let t = &state.topology.tiles[tile.0 as usize];
                                    t.terrain.resource() == Some(*resource)
                                        && state.topology.dice_to_tiles[*dice_number as usize]
                                            .contains(&tile)
                                });
                                break;
                            }
                            GameEvent::MoveRobber { .. } | GameEvent::PlayedKnight { .. } => break,
                            _ => {}
                        }
                    }

                    // Peek for steal outcome: constrains whether the tile
                    // has opponent buildings.
                    let steal_outcome = events[i + 1..].iter().find_map(|e| match e {
                        GameEvent::Stole { .. } => Some(true),
                        GameEvent::StoleNothing { .. } => Some(false),
                        GameEvent::Roll { .. }
                        | GameEvent::PlayedKnight { .. }
                        | GameEvent::MoveRobber { .. } => None,
                        _ => None,
                    });
                    let opp_buildings = state.player_buildings(opp);
                    let opp_has_cards = state.players[opp].hand.total() > 0;
                    let friendly =
                        state.players[opp].building_vps < crate::game::FRIENDLY_ROBBER_VP;
                    candidates.retain(|&tile| {
                        let tile_mask = state.topology.adj.tile_nodes[tile.0 as usize];
                        let on_tile = (tile_mask & opp_buildings) != 0;
                        let can_steal = on_tile && opp_has_cards && !friendly;
                        match steal_outcome {
                            Some(true) => can_steal,   // stole → must be stealable
                            Some(false) => !can_steal, // stole nothing → not stealable
                            None => true,
                        }
                    });

                    // Pre-filter: check all non-7 rolls until the next event
                    // that changes the board or robber position.
                    let check_rolls = pre_filter_rolls(&events[i + 1..], ctx.color_map);
                    if !check_rolls.is_empty() {
                        candidates.retain(|&tile| {
                            let mut trial = state.clone();
                            trial.robber = tile;
                            check_rolls
                                .iter()
                                .all(|(roll, gains)| validate_distribution(&trial, *roll, gains))
                        });
                    }

                    if candidates.len() == 1 {
                        let aid = action::robber_id(candidates[0]).0 as usize;
                        state.apply_action(aid);
                    } else if candidates.is_empty() {
                        // No valid tile — an earlier robber placement was wrong.
                        // Backtrack to try a different assignment.
                        return None;
                    } else {
                        // Multiple candidates survive — branch and backtrack.
                        for &tile in &candidates {
                            let mut trial = state.clone();
                            trial.apply_action(action::robber_id(tile).0 as usize);
                            let trial_tl = timeline.clone();
                            if let Some(result) = try_replay(trial, events, i + 1, ctx, trial_tl) {
                                return Some(result);
                            }
                        }
                        return None; // all robber branches failed
                    }
                }
            }

            GameEvent::Stole { resources, .. } => {
                if matches!(state.phase, Phase::StealResolve) {
                    if let Some(idx) = ALL_RESOURCES.iter().position(|&r| resources[r] > 0) {
                        state.apply_action(idx);
                    }
                }
                // After steal, engine may set Phase::Roll for pre_roll knight.
                // We want PreRoll so colonist events drive dice resolution.
                if matches!(state.phase, Phase::Roll) {
                    state.phase = Phase::PreRoll;
                }
            }

            GameEvent::StoleNothing { .. } | GameEvent::StoleUnknown { .. } => {
                // Engine may have entered StealResolve (it thought opponent
                // had cards + buildings on the tile). Force out.
                if matches!(state.phase, Phase::StealResolve) {
                    state.phase = if state.pre_roll {
                        Phase::PreRoll
                    } else {
                        Phase::Main
                    };
                }
            }

            // -- Post-setup builds -----------------------------------------
            GameEvent::BuildRoad { player, edge } | GameEvent::PlaceRoad { player, edge } => {
                ensure_player(&mut state, *player);
                let pid = player_of_color(ctx.color_map, *player);
                let from_coords = edge
                    .map(|(x, y, z)| ctx.mapper.map_edge(x, y, z))
                    .and_then(|e| ctx.edge_map.get(&e).copied());

                let verb = if matches!(&events[i], GameEvent::BuildRoad { .. }) {
                    "builds"
                } else {
                    "places"
                };

                if let Some(eid) = from_coords {
                    let aid = action::road_id(eid).0 as usize;
                    crate::game::apply_with_chance(&mut state, aid, None);
                    let label = format!("{} {verb} road", player_label(*player, ctx.color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                } else if let Some(p) = pid {
                    let candidates: Vec<EdgeId> = ctx.dom_roads[p]
                        .iter()
                        .filter(|&&eid| state.boards[p].road_network.roads & (1u128 << eid.0) == 0)
                        .copied()
                        .collect();
                    if candidates.len() == 1 {
                        let aid = action::road_id(candidates[0]).0 as usize;
                        crate::game::apply_with_chance(&mut state, aid, None);
                        let label = format!("{} {verb} road", player_label(*player, ctx.color_map));
                        timeline.push(TimelineEntry {
                            label,
                            state: state.clone(),
                        });
                    } else if candidates.is_empty() {
                        return None;
                    } else {
                        for &eid in &candidates {
                            let mut trial = state.clone();
                            let aid = action::road_id(eid).0 as usize;
                            crate::game::apply_with_chance(&mut trial, aid, None);
                            let mut trial_tl = timeline.clone();
                            trial_tl.push(TimelineEntry {
                                label: format!(
                                    "{} {verb} road",
                                    player_label(*player, ctx.color_map)
                                ),
                                state: trial.clone(),
                            });
                            if let Some(result) = try_replay(trial, events, i + 1, ctx, trial_tl) {
                                return Some(result);
                            }
                        }
                        return None;
                    }
                }
            }

            GameEvent::BuildSettlement { player, corner } => {
                ensure_player(&mut state, *player);
                let pid = player_of_color(ctx.color_map, *player);
                let from_coords = corner
                    .map(|(x, y, z)| ctx.mapper.map_corner(x, y, z))
                    .and_then(|c| ctx.corner_map.get(&c).copied());

                if let Some(nid) = from_coords {
                    let aid = action::settlement_id(nid).0 as usize;
                    crate::game::apply_with_chance(&mut state, aid, None);
                    let label =
                        format!("{} builds settlement", player_label(*player, ctx.color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                } else if let Some(p) = pid {
                    // No coordinates — branch on unplaced DOM settlements.
                    let candidates: Vec<NodeId> = ctx.dom_settlements[p]
                        .iter()
                        .filter(|&&nid| {
                            state.boards[p].settlements & (1u64 << nid.0) == 0
                                && state.boards[p].cities & (1u64 << nid.0) == 0
                        })
                        .copied()
                        .collect();
                    if candidates.len() == 1 {
                        let aid = action::settlement_id(candidates[0]).0 as usize;
                        crate::game::apply_with_chance(&mut state, aid, None);
                        let label =
                            format!("{} builds settlement", player_label(*player, ctx.color_map));
                        timeline.push(TimelineEntry {
                            label,
                            state: state.clone(),
                        });
                    } else if candidates.is_empty() {
                        return None;
                    } else {
                        for &nid in &candidates {
                            let mut trial = state.clone();
                            let aid = action::settlement_id(nid).0 as usize;
                            crate::game::apply_with_chance(&mut trial, aid, None);
                            let mut trial_tl = timeline.clone();
                            trial_tl.push(TimelineEntry {
                                label: format!(
                                    "{} builds settlement",
                                    player_label(*player, ctx.color_map)
                                ),
                                state: trial.clone(),
                            });
                            if let Some(result) = try_replay(trial, events, i + 1, ctx, trial_tl) {
                                return Some(result);
                            }
                        }
                        return None;
                    }
                }
            }

            GameEvent::BuildCity { player, corner } => {
                ensure_player(&mut state, *player);
                let pid = player_of_color(ctx.color_map, *player);
                let from_coords = corner
                    .map(|(x, y, z)| ctx.mapper.map_corner(x, y, z))
                    .and_then(|c| ctx.corner_map.get(&c).copied());

                if let Some(nid) = from_coords {
                    let aid = action::city_id(nid).0 as usize;
                    crate::game::apply_with_chance(&mut state, aid, None);
                    let label = format!("{} builds city", player_label(*player, ctx.color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                } else if let Some(p) = pid {
                    let candidates: Vec<NodeId> = ctx
                        .dom
                        .cities
                        .iter()
                        .filter_map(|&(color, x, y, z)| {
                            if player_of_color(ctx.color_map, color) != Some(p) {
                                return None;
                            }
                            let mapped = ctx.mapper.map_corner(x, y, z);
                            let &nid = ctx.corner_map.get(&mapped)?;
                            if state.boards[p].cities & (1u64 << nid.0) == 0 {
                                Some(nid)
                            } else {
                                None
                            }
                        })
                        .collect();
                    if candidates.len() == 1 {
                        let aid = action::city_id(candidates[0]).0 as usize;
                        crate::game::apply_with_chance(&mut state, aid, None);
                        let label = format!("{} builds city", player_label(*player, ctx.color_map));
                        timeline.push(TimelineEntry {
                            label,
                            state: state.clone(),
                        });
                    } else if candidates.is_empty() {
                        return None;
                    } else {
                        for &nid in &candidates {
                            let mut trial = state.clone();
                            let aid = action::city_id(nid).0 as usize;
                            crate::game::apply_with_chance(&mut trial, aid, None);
                            let mut trial_tl = timeline.clone();
                            trial_tl.push(TimelineEntry {
                                label: format!(
                                    "{} builds city",
                                    player_label(*player, ctx.color_map)
                                ),
                                state: trial.clone(),
                            });
                            if let Some(result) = try_replay(trial, events, i + 1, ctx, trial_tl) {
                                return Some(result);
                            }
                        }
                        return None;
                    }
                }
            }

            // -- Dev cards -------------------------------------------------
            GameEvent::BuyDevCard { player } => {
                ensure_player(&mut state, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    crate::game::apply_hidden_dev_card_buy(&mut state);
                }
            }
            GameEvent::PlayedKnight { player } => {
                ensure_player(&mut state, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    state.apply_action(action::PLAY_KNIGHT as usize);
                }
            }
            GameEvent::PlayedRoadBuilding { player } => {
                ensure_player(&mut state, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    state.apply_action(action::PLAY_ROAD_BUILDING as usize);
                }
            }
            GameEvent::PlayedMonopoly { player } => {
                ensure_player(&mut state, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    let resource = events[i + 1..].iter().find_map(|e| match e {
                        GameEvent::MonopolyResult { resource, .. } => Some(*resource),
                        _ => None,
                    });
                    if let Some(res) = resource {
                        state.apply_action(action::monopoly_id(res).0 as usize);
                    }
                }
            }
            GameEvent::MonopolyResult { .. } => {}
            GameEvent::PlayedYearOfPlenty { player } => {
                ensure_player(&mut state, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    let gain = events[i + 1..].iter().find_map(|e| match e {
                        GameEvent::YearOfPlentyGain { resources, .. } => Some(*resources),
                        _ => None,
                    });
                    if let Some(resources) = gain {
                        let mut r1 = None;
                        let mut r2 = None;
                        for &res in &ALL_RESOURCES {
                            for _ in 0..resources[res] {
                                if r1.is_none() {
                                    r1 = Some(res);
                                } else {
                                    r2 = Some(res);
                                }
                            }
                        }
                        if let (Some(a), Some(b)) = (r1, r2) {
                            state.apply_action(action::yop_id(a, b).0 as usize);
                        }
                    }
                }
            }
            GameEvent::YearOfPlentyGain { .. } => {}
            GameEvent::PlayedDevCard { .. } => {}

            // -- Trades ----------------------------------------------------
            GameEvent::BankTrade {
                player,
                given,
                received,
            } => {
                ensure_player(&mut state, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    // Decompose multi-resource trades into individual
                    // maritime actions. A combined event like "L L L L B B → G G"
                    // is two trades: 4L→1G + 2B→1G at different ratios.
                    let hand_before = state.players[state.current_player].hand;
                    let mut remaining = *given;
                    for &recv in &ALL_RESOURCES {
                        for _ in 0..received[recv] {
                            for &give in &ALL_RESOURCES {
                                if give == recv {
                                    continue;
                                }
                                let ratio =
                                    state.players[state.current_player].trade_ratios[give as usize];
                                if remaining[give] >= ratio {
                                    remaining[give] -= ratio;
                                    let aid = action::maritime_id(give, recv).0 as usize;
                                    crate::game::apply_with_chance(&mut state, aid, None);
                                    break;
                                }
                            }
                        }
                    }
                    // Verify: hand should have decreased by given and
                    // increased by received.
                    let mut expected = hand_before;
                    expected.sub(*given);
                    expected.add(*received);
                    if state.players[state.current_player].hand != expected {
                        eprintln!(
                            "  bank trade mismatch[{i}]: player={player} \
                             ratios={:?} expected={expected:?} actual={:?}",
                            state.players[state.current_player].trade_ratios,
                            state.players[state.current_player].hand,
                        );
                    }
                    let label = format!("{} bank trade", player_label(*player, ctx.color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                }
            }

            GameEvent::PlayerTrade { .. }
            | GameEvent::PlaceSettlement { .. }
            | GameEvent::LongestRoad { .. }
            | GameEvent::LongestRoadChanged { .. }
            | GameEvent::TradeOffer { .. }
            | GameEvent::EmbargoSet { .. }
            | GameEvent::EmbargoLifted { .. }
            | GameEvent::Unknown { .. } => {}
        }

        i += 1;
    }

    Some(timeline)
}

/// Collect all (roll, per-player gains) pairs until the next MoveRobber or
/// PlayedKnight. Used to validate robber placement against multiple rolls.
#[cfg(test)]
fn peek_all_roll_gains(
    events: &[GameEvent],
    color_map: &[(u8, Player)],
) -> Vec<(u8, [ResourceArray; 2])> {
    let mut results = Vec::new();
    let mut current_roll: Option<u8> = None;
    let mut gains: [ResourceArray; 2] = Default::default();

    for event in events {
        match event {
            // Stop at events that change the robber or the board.
            GameEvent::MoveRobber { .. }
            | GameEvent::PlayedKnight { .. }
            | GameEvent::BuildSettlement { .. }
            | GameEvent::BuildCity { .. } => {
                if let Some(roll) = current_roll.take() {
                    results.push((roll, gains));
                }
                break;
            }
            GameEvent::Roll { d1, d2, .. } => {
                // Flush previous roll if any.
                if let Some(roll) = current_roll {
                    results.push((roll, gains));
                    gains = Default::default();
                }
                let total = d1 + d2;
                if total != 7 {
                    current_roll = Some(total);
                } else {
                    current_roll = None;
                }
            }
            GameEvent::GotResources { player, resources } if current_roll.is_some() => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    gains[pid as usize].add(*resources);
                }
            }
            _ => {}
        }
    }
    // Flush last roll if we ran out of events.
    if let Some(roll) = current_roll {
        results.push((roll, gains));
    }
    results
}

/// Collect (roll, gains) pairs for robber pre-filtering. Stops at events that
/// change the board (builds, robber moves, knight plays) since those invalidate
/// the static distribution check.
fn pre_filter_rolls(
    events: &[GameEvent],
    color_map: &[(u8, Player)],
) -> Vec<(u8, [ResourceArray; 2])> {
    let mut results = Vec::new();
    let mut current_roll: Option<u8> = None;
    let mut gains: [ResourceArray; 2] = Default::default();

    for event in events {
        match event {
            GameEvent::MoveRobber { .. }
            | GameEvent::PlayedKnight { .. }
            | GameEvent::BuildSettlement { .. }
            | GameEvent::BuildCity { .. } => {
                if let Some(roll) = current_roll.take() {
                    results.push((roll, gains));
                }
                break;
            }
            GameEvent::Roll { d1, d2, .. } => {
                if let Some(roll) = current_roll.take() {
                    results.push((roll, gains));
                    gains = Default::default();
                }
                let total = d1 + d2;
                if total != 7 {
                    current_roll = Some(total);
                }
            }
            GameEvent::GotResources { player, resources } if current_roll.is_some() => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    gains[pid as usize].add(*resources);
                }
            }
            _ => {}
        }
    }
    if let Some(roll) = current_roll {
        results.push((roll, gains));
    }
    results
}

/// All legal robber tile candidates (any tile except current robber position).
fn all_robber_candidates(state: &GameState) -> Vec<TileId> {
    (0..state.topology.tiles.len() as u8)
        .map(TileId)
        .filter(|&t| t != state.robber)
        .collect()
}

/// Check if a roll would produce the expected resource gains with the given state.
fn validate_distribution(state: &GameState, roll: u8, expected: &[ResourceArray; 2]) -> bool {
    use crate::game::resource::ALL_RESOURCES;

    let topo = &state.topology;
    let mut total_demand = [0u8; 5];
    let mut player_gains = [[0u8; 5]; 2];

    for &tid in &topo.dice_to_tiles[roll as usize] {
        if tid == state.robber {
            continue;
        }
        let tile = &topo.tiles[tid.0 as usize];
        let Some(resource) = tile.terrain.resource() else {
            continue;
        };
        let ri = resource as usize;
        let tile_mask = topo.adj.tile_nodes[tid.0 as usize];

        for (pi, &pid) in [Player::One, Player::Two].iter().enumerate() {
            let s = (state.boards[pid].settlements & tile_mask).count_ones() as u8;
            let c = (state.boards[pid].cities & tile_mask).count_ones() as u8;
            let amount = s + c * 2;
            total_demand[ri] += amount;
            player_gains[pi][ri] += amount;
        }
    }

    // Only distribute if bank can cover total demand (same rule as engine).
    for &r in &ALL_RESOURCES {
        let ri = r as usize;
        if total_demand[ri] == 0 || state.bank[r] < total_demand[ri] {
            player_gains[0][ri] = 0;
            player_gains[1][ri] = 0;
        }
    }

    let actual: [ResourceArray; 2] = [
        ResourceArray(player_gains[0]),
        ResourceArray(player_gains[1]),
    ];
    actual == *expected
}

/// Apply extracted dev card identities and bought-this-turn info to a game state.
///
/// When React provides a non-empty card list, it's authoritative — set the
/// player's `dev_cards` to match exactly (clearing any hidden card tracking).
/// `has_played_dev_card_this_turn` is left to event processing (PlayedKnight
/// etc. set it; DiceRoll clears it on turn change).
pub fn apply_dev_cards(
    state: &mut GameState,
    player: Player,
    cards: &[DevCardKind],
    bought_this_turn: &[DevCardKind],
) {
    let ps = &mut state.players[player];
    // React's card list is authoritative — set dev_cards to match exactly.
    if !cards.is_empty() {
        ps.dev_cards = Default::default();
        for &kind in cards {
            ps.dev_cards[kind] += 1;
        }
        ps.hidden_dev_cards = 0;
        ps.hidden_dev_cards_bought_this_turn = 0;
    }
    // Always sync bought-this-turn from colonist's authoritative data.
    ps.dev_cards_bought_this_turn = Default::default();
    for &kind in bought_this_turn {
        ps.dev_cards_bought_this_turn[kind] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a fresh GameState on a deterministic board.
    fn test_state() -> GameState {
        let topo = Arc::new(Topology::from_seed(42));
        let deck = DevCardDeck::new();
        GameState::new(topo, deck, Dice::default())
    }

    #[test]
    fn validate_distribution_empty_board() {
        let state = test_state();
        let empty = [ResourceArray::default(); 2];
        // No buildings → no resources for any roll.
        for roll in 2..=12 {
            assert!(validate_distribution(&state, roll, &empty));
        }
    }

    #[test]
    fn validate_distribution_one_settlement() {
        let mut state = test_state();
        // Place P1 settlement at node 0 during setup.
        state.apply_action(0); // PlaceSettlement at node 0

        // Figure out what resources node 0 should produce.
        let node = &state.topology.nodes[0];
        for &tid in &node.adjacent_tiles {
            let tile = &state.topology.tiles[tid.0 as usize];
            if let Some(resource) = tile.terrain.resource() {
                // This tile has a dice number. Find it.
                let dice_num = state
                    .topology
                    .dice_to_tiles
                    .iter()
                    .enumerate()
                    .find(|(_, tiles)| tiles.contains(&tid))
                    .map(|(i, _)| i as u8)
                    .unwrap();

                let mut expected = [ResourceArray::default(); 2];
                expected[0][resource] = 1;
                assert!(
                    validate_distribution(&state, dice_num, &expected),
                    "roll={dice_num} resource={resource:?} should give P1 1"
                );
            }
        }
    }

    #[test]
    fn validate_distribution_robber_blocks() {
        let mut state = test_state();
        state.apply_action(0); // P1 settlement at node 0

        let node = &state.topology.nodes[0];
        // Find a producing adjacent tile and block it.
        for &tid in &node.adjacent_tiles {
            let tile = &state.topology.tiles[tid.0 as usize];
            if tile.terrain.resource().is_some() {
                let dice_num = state
                    .topology
                    .dice_to_tiles
                    .iter()
                    .enumerate()
                    .find(|(_, tiles)| tiles.contains(&tid))
                    .map(|(i, _)| i as u8)
                    .unwrap();

                // Put robber on this tile.
                state.robber = tid;
                let empty = [ResourceArray::default(); 2];
                // With only one tile of this number adjacent, blocking it
                // should produce zero (if no other tiles share the number).
                // Just verify it doesn't produce the unblocked amount.
                let mut unblocked = [ResourceArray::default(); 2];
                unblocked[0][tile.terrain.resource().unwrap()] = 1;
                // Distribution should differ from unblocked.
                assert!(
                    !validate_distribution(&state, dice_num, &unblocked)
                        || validate_distribution(&state, dice_num, &empty),
                    "robber should block production"
                );
                break;
            }
        }
    }

    #[test]
    fn peek_all_roll_gains_collects_one_roll() {
        let color_map = vec![(1, Player::One), (5, Player::Two)];
        let events = vec![
            GameEvent::Roll {
                player: 1,
                d1: 3,
                d2: 6,
            },
            GameEvent::GotResources {
                player: 1,
                resources: ResourceArray::new(0, 0, 0, 1, 0),
            },
            GameEvent::GotResources {
                player: 5,
                resources: ResourceArray::new(0, 1, 0, 0, 0),
            },
            GameEvent::Roll {
                player: 5,
                d1: 4,
                d2: 3,
            },
        ];
        let rolls = peek_all_roll_gains(&events, &color_map);
        assert_eq!(rolls.len(), 1, "should stop at second Roll");
        assert_eq!(rolls[0].0, 9);
        assert_eq!(rolls[0].1[0], ResourceArray::new(0, 0, 0, 1, 0)); // P1
        assert_eq!(rolls[0].1[1], ResourceArray::new(0, 1, 0, 0, 0)); // P2
    }

    #[test]
    fn peek_all_roll_gains_skips_seven() {
        let color_map = vec![(1, Player::One), (5, Player::Two)];
        let events = vec![
            GameEvent::Roll {
                player: 1,
                d1: 3,
                d2: 4,
            }, // 7
            GameEvent::MoveRobber { player: 1 },
        ];
        let rolls = peek_all_roll_gains(&events, &color_map);
        assert!(rolls.is_empty(), "7-roll should not produce gains entry");
    }

    #[test]
    fn peek_all_roll_gains_stops_at_move_robber() {
        let color_map = vec![(1, Player::One), (5, Player::Two)];
        let events = vec![
            GameEvent::Roll {
                player: 1,
                d1: 3,
                d2: 6,
            },
            GameEvent::GotResources {
                player: 1,
                resources: ResourceArray::new(0, 0, 0, 1, 0),
            },
            GameEvent::Roll {
                player: 5,
                d1: 2,
                d2: 4,
            },
            GameEvent::GotResources {
                player: 5,
                resources: ResourceArray::new(0, 0, 1, 0, 0),
            },
            GameEvent::Roll {
                player: 1,
                d1: 3,
                d2: 4,
            }, // 7
            GameEvent::MoveRobber { player: 1 },
            // After this — should NOT be collected.
            GameEvent::Roll {
                player: 5,
                d1: 5,
                d2: 5,
            },
            GameEvent::GotResources {
                player: 5,
                resources: ResourceArray::new(1, 0, 0, 0, 0),
            },
        ];
        let rolls = peek_all_roll_gains(&events, &color_map);
        assert_eq!(rolls.len(), 2, "should stop at MoveRobber");
        assert_eq!(rolls[0].0, 9);
        assert_eq!(rolls[1].0, 6);
    }

    #[test]
    fn peek_all_roll_gains_stops_at_build() {
        let color_map = vec![(1, Player::One), (5, Player::Two)];
        let events = vec![
            GameEvent::Roll {
                player: 1,
                d1: 3,
                d2: 6,
            },
            GameEvent::GotResources {
                player: 1,
                resources: ResourceArray::new(0, 0, 0, 1, 0),
            },
            GameEvent::BuildSettlement {
                player: 1,
                corner: None,
            },
            // After build — should NOT be collected.
            GameEvent::Roll {
                player: 5,
                d1: 2,
                d2: 4,
            },
        ];
        let rolls = peek_all_roll_gains(&events, &color_map);
        assert_eq!(rolls.len(), 1, "should stop at BuildSettlement");
    }
}
