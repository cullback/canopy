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
use canopy::player::Player;

use crate::game::board::{EdgeId, NodeId, TileId};
use crate::game::dev_card::{DevCardDeck, DevCardKind};
use crate::game::dice::{BalancedDice, Dice};
use crate::game::resource::{Resource, ResourceArray};
use crate::game::state::{GameState, Phase};
use crate::game::topology::Topology;

use super::board::{self, BoardData, CoordMapper};
use super::log::GameEvent;

/// A snapshot in the game timeline for undo/redo navigation.
pub struct TimelineEntry {
    pub label: String,
    pub state: GameState,
}

/// Build a timeline of game states from colonist.io data.
///
/// In **full log mode** (setup placement coordinates present), replays the
/// entire game from scratch through the action system for setup, then tracks
/// post-setup events incrementally.
///
/// In **late join mode** (incomplete setup data), builds from the board snapshot
/// and returns a single-entry timeline.
pub fn build_timeline(
    board: &BoardData,
    events: &[GameEvent],
    mapper: &CoordMapper,
) -> Vec<TimelineEntry> {
    let (terrains, numbers, port_resources) = board::to_layout(board, mapper);
    let topology = Arc::new(Topology::from_layout(terrains, numbers, port_resources));
    let color_map = discover_colors(events);
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    if is_full_log(events, &corner_map, &edge_map, mapper) {
        eprintln!("full log detected — building complete timeline");
        build_full_timeline(
            board,
            topology,
            events,
            &color_map,
            &corner_map,
            &edge_map,
            mapper,
        )
    } else {
        eprintln!("late join — single-state timeline from board snapshot");
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
    let (terrains, numbers, port_resources) = board::to_layout(board, mapper);
    let topology = Arc::new(Topology::from_layout(terrains, numbers, port_resources));
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
    let (ns, nc, nr, _) = sync_buildings(
        &mut state,
        &buildings,
        &color_map,
        &corner_map,
        &edge_map,
        mapper,
    );
    if ns + nc + nr > 0 {
        eprintln!("placed {ns} settlements, {nc} cities, {nr} roads from board snapshot");
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
pub fn discover_colors(events: &[GameEvent]) -> Vec<(u8, Player)> {
    let mut map: Vec<(u8, Player)> = Vec::new();
    for event in events {
        let color = match event {
            GameEvent::PlaceSettlement { player, .. }
            | GameEvent::PlaceRoad { player, .. }
            | GameEvent::Roll { player, .. } => *player,
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

/// Ensure a dev card is in the player's concrete `dev_cards` array so the
/// engine can decrement it. Moves one card from `hidden_dev_cards` if needed.
fn reveal_dev_card(state: &mut GameState, pid: Player, kind: DevCardKind) {
    if state.players[pid].dev_cards[kind] == 0 && state.players[pid].hidden_dev_cards > 0 {
        state.players[pid].hidden_dev_cards -= 1;
        state.players[pid].dev_cards[kind] += 1;
    }
}

/// Decode up to two resources from a `ResourceArray` (for Year of Plenty).
fn decode_two_resources(resources: &ResourceArray) -> (Option<Resource>, Option<Resource>) {
    use crate::game::resource::ALL_RESOURCES;
    let mut result = [None; 2];
    let mut idx = 0;
    for &res in &ALL_RESOURCES {
        for _ in 0..resources[res] {
            if idx < 2 {
                result[idx] = Some(res);
                idx += 1;
            }
        }
    }
    (result[0], result[1])
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

/// Sync buildings from a board snapshot onto the game state.
///
/// Only places buildings that are not already present (checked via bitfields).
/// Returns `(settlements, cities, roads, last_settlement_node)` — the last
/// settlement NodeId placed (if any), used to set `last_setup_node`.
pub fn sync_buildings(
    state: &mut GameState,
    buildings: &board::BuildingData,
    color_map: &[(u8, Player)],
    corner_map: &std::collections::HashMap<(i32, i32, u8), NodeId>,
    edge_map: &std::collections::HashMap<(i32, i32, u8), EdgeId>,
    mapper: &CoordMapper,
) -> (u32, u32, u32, Option<NodeId>) {
    let mut placed = (0u32, 0u32, 0u32);
    let mut last_settlement: Option<NodeId> = None;

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
            last_settlement = Some(nid);
            placed.0 += 1;
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
            placed.1 += 1;
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
            placed.2 += 1;
        }
    }

    (placed.0, placed.1, placed.2, last_settlement)
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
    let mut saw_discard = false;
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
            GameEvent::Discard { .. } | GameEvent::RolledSeven => {
                saw_discard = true;
                continue;
            }
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

/// Check if the log has enough setup placement events with valid coordinates
/// to replay the full game from scratch.
fn is_full_log(
    events: &[GameEvent],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
    mapper: &CoordMapper,
) -> bool {
    let mut settlements = 0u32;
    let mut roads = 0u32;
    for event in events {
        match event {
            GameEvent::PlaceSettlement {
                corner: Some((x, y, z)),
                ..
            } => {
                let mapped = mapper.map_corner(*x, *y, *z);
                if corner_map.contains_key(&mapped) {
                    settlements += 1;
                }
            }
            GameEvent::PlaceRoad {
                edge: Some((x, y, z)),
                ..
            } => {
                let mapped = mapper.map_edge(*x, *y, *z);
                if edge_map.contains_key(&mapped) {
                    roads += 1;
                }
            }
            _ => {}
        }
    }
    settlements >= 4 && roads >= 4
}

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

/// Build complete timeline: replay setup via game actions, then track post-setup
/// events with snapshots at each significant moment.
fn build_full_timeline(
    board: &BoardData,
    topology: Arc<Topology>,
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
    mapper: &CoordMapper,
) -> Vec<TimelineEntry> {
    let dev_deck = DevCardDeck::new();
    let dice = Dice::Balanced(BalancedDice::new());
    let mut state = GameState::new(topology, dev_deck, dice);

    let mut timeline = vec![TimelineEntry {
        label: "Game start".into(),
        state: state.clone(),
    }];

    // Phase 1: Replay setup through the game action system.
    let mut event_idx = 0;
    while event_idx < events.len()
        && matches!(state.phase, Phase::PlaceSettlement | Phase::PlaceRoad)
    {
        match &events[event_idx] {
            GameEvent::PlaceSettlement {
                player,
                corner: Some((x, y, z)),
            } => {
                let mapped = mapper.map_corner(*x, *y, *z);
                if let Some(&nid) = corner_map.get(&mapped) {
                    state.apply_action(nid.0 as usize);
                    let label = format!("{} places settlement", player_label(*player, color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                }
            }
            GameEvent::PlaceRoad {
                player,
                edge: Some((x, y, z)),
            } => {
                let mapped = mapper.map_edge(*x, *y, *z);
                if let Some(&eid) = edge_map.get(&mapped) {
                    state.apply_action((54 + eid.0) as usize);
                    let label = format!("{} places road", player_label(*player, color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                }
            }
            // StartingResources handled by the game engine during 2nd-round placements.
            _ => {}
        }
        event_idx += 1;
    }

    eprintln!(
        "setup complete: {} entries, phase={:?}, setup_count={}",
        timeline.len(),
        state.phase,
        state.setup_count
    );

    // Phase 2: Post-setup events routed through the engine action system.
    // State is in PreRoll after setup — the first Roll event transitions
    // through ROLL → resolve, so no manual phase override needed.
    let robber_tile = {
        let buildings = board::extract_buildings(board, mapper);
        buildings.robber_tile_index.map(TileId)
    };
    let mut _actions = Vec::new();
    process_post_setup(
        &mut state,
        &events[event_idx..],
        color_map,
        corner_map,
        edge_map,
        &mut timeline,
        mapper,
        robber_tile,
        &mut _actions,
    );

    eprintln!("timeline: {} total entries", timeline.len());
    timeline
}

/// Process post-setup log events, routing through the engine's action system
/// where possible and appending timeline entries at each significant event.
///
/// Actions like builds, dev card plays, discards, and bank trades flow through
/// `apply_with_chance` so the engine handles costs, phase transitions, road
/// networks, and longest road/army. Rolls use the engine for turn switching
/// (END_TURN) and the PreRoll→Roll transition, but resolve dice manually to
/// preserve exact resource tracking from GotResources events.
///
/// Exceptions that stay as direct mutation:
///   - `PlayerTrade` (no action ID in the engine)
///   - `BuyDevCard` (uses `apply_hidden_dev_card_buy` — card identity unknown)
///   - `Stole` fallback when engine didn't enter StealResolve
fn process_post_setup(
    state: &mut GameState,
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
    timeline: &mut Vec<TimelineEntry>,
    mapper: &CoordMapper,
    robber_tile: Option<TileId>,
    actions: &mut Vec<usize>,
) {
    use crate::game::action::{
        self, BUY_DEV_CARD, END_TURN, PLAY_KNIGHT, PLAY_ROAD_BUILDING, ROLL,
    };
    use crate::game::resource::{ALL_RESOURCES, CITY_COST, ROAD_COST, SETTLEMENT_COST};

    let mut pending_label: Option<String> = None;
    let mut roll_gains: [ResourceArray; 2] = [ResourceArray::default(); 2];

    for event in events {
        let is_entry = matches!(
            event,
            GameEvent::Roll { .. }
                | GameEvent::PlaceRoad { .. }
                | GameEvent::BuildRoad { .. }
                | GameEvent::BuildSettlement { .. }
                | GameEvent::BuildCity { .. }
                | GameEvent::BuyDevCard { .. }
                | GameEvent::PlayerTrade { .. }
                | GameEvent::BankTrade { .. }
                | GameEvent::PlayedKnight { .. }
                | GameEvent::PlayedMonopoly { .. }
                | GameEvent::PlayedRoadBuilding { .. }
                | GameEvent::PlayedYearOfPlenty { .. }
        );

        // Flush pending entry before the next entry-producing event.
        if is_entry && let Some(mut label) = pending_label.take() {
            let p1 = format_resources(roll_gains[0]);
            let p2 = format_resources(roll_gains[1]);
            if !p1.is_empty() {
                label.push_str(&format!("\n  P1: {p1}"));
            }
            if !p2.is_empty() {
                label.push_str(&format!("\n  P2: {p2}"));
            }
            roll_gains = [ResourceArray::default(); 2];
            timeline.push(TimelineEntry {
                label,
                state: state.clone(),
            });
        }

        match event {
            // --- Entry-producing events ---
            GameEvent::Roll { player, d1, d2 } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    // Use the engine for turn switching and PreRoll→Roll.
                    if state.phase != Phase::Roll {
                        if state.current_player != pid {
                            state.apply_action(END_TURN as usize);
                            actions.push(END_TURN as usize);
                        }
                        state.apply_action(ROLL as usize);
                        actions.push(ROLL as usize);
                    }
                    // Record dice outcome for tree walk (not applied to
                    // committed state — GotResources tracks resources).
                    actions.push((*d1 + *d2 - 2) as usize);
                    // Resolve dice manually — GotResources provides exact
                    // resource tracking, so we skip distribute_resources.
                    state.pre_roll = false;
                    let total = d1 + d2;
                    if let Dice::Balanced(ref mut b) = state.dice {
                        b.draw(total, pid);
                    }
                    if total == 7 {
                        crate::game::handle_seven(state);
                    } else {
                        state.phase = Phase::Main;
                    }
                }
                pending_label = Some(format!(
                    "{} rolls {}",
                    player_label(*player, color_map),
                    d1 + d2
                ));
            }

            GameEvent::BuildRoad { player, edge } => {
                if let Some(&eid) = edge
                    .map(|(x, y, z)| mapper.map_edge(x, y, z))
                    .and_then(|e| edge_map.get(&e))
                {
                    let aid = action::road_id(eid).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                } else if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(ROAD_COST);
                    state.bank.add(ROAD_COST);
                }
                pending_label = Some(format!("{} builds road", player_label(*player, color_map)));
            }

            // Road Building dev card uses "place" events (type 4), not "build"
            // (type 5).  Apply as a road action — apply_build_road handles the
            // free cost and phase transition.
            GameEvent::PlaceRoad { player, edge } if state.setup_count >= 4 => {
                if let Some(&eid) = edge
                    .map(|(x, y, z)| mapper.map_edge(x, y, z))
                    .and_then(|e| edge_map.get(&e))
                {
                    let aid = action::road_id(eid).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                }
                pending_label = Some(format!("{} places road", player_label(*player, color_map)));
            }

            GameEvent::BuildSettlement { player, corner, .. } => {
                if let Some(&nid) = corner
                    .map(|(x, y, z)| mapper.map_corner(x, y, z))
                    .and_then(|c| corner_map.get(&c))
                {
                    let aid = action::settlement_id(nid).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                } else if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(SETTLEMENT_COST);
                    state.bank.add(SETTLEMENT_COST);
                }
                pending_label = Some(format!(
                    "{} builds settlement",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::BuildCity { player, corner, .. } => {
                if let Some(&nid) = corner
                    .map(|(x, y, z)| mapper.map_corner(x, y, z))
                    .and_then(|c| corner_map.get(&c))
                {
                    let aid = action::city_id(nid).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                } else if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(CITY_COST);
                    state.bank.add(CITY_COST);
                }
                pending_label = Some(format!("{} builds city", player_label(*player, color_map)));
            }

            GameEvent::BuyDevCard { player } => {
                if player_of_color(color_map, *player).is_some() {
                    crate::game::apply_hidden_dev_card_buy(state);
                    actions.push(BUY_DEV_CARD as usize);
                }
                pending_label = Some(format!(
                    "{} buys dev card",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::PlayerTrade {
                player,
                counterparty,
                given,
                received,
            } => {
                // No action ID — direct mutation only.
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                }
                if let Some(cpid) = player_of_color(color_map, *counterparty) {
                    state.players[cpid].hand.sub(*received);
                    state.players[cpid].hand.add(*given);
                }
                pending_label = Some(format!(
                    "{} trades with {}",
                    player_label(*player, color_map),
                    player_label(*counterparty, color_map)
                ));
            }

            GameEvent::BankTrade {
                player,
                given,
                received,
            } => {
                let give_r = ALL_RESOURCES.iter().find(|&&r| given[r] > 0).copied();
                let recv_r = ALL_RESOURCES.iter().find(|&&r| received[r] > 0).copied();
                if let (Some(give), Some(recv)) = (give_r, recv_r) {
                    let aid = action::maritime_id(give, recv).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                }
                pending_label = Some(format!("{} bank trade", player_label(*player, color_map)));
            }

            GameEvent::PlayedKnight { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    reveal_dev_card(state, pid, DevCardKind::Knight);
                    crate::game::apply_with_chance(state, PLAY_KNIGHT as usize, None);
                    actions.push(PLAY_KNIGHT as usize);
                }
                pending_label = Some(format!("{} plays Knight", player_label(*player, color_map)));
            }

            GameEvent::PlayedRoadBuilding { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    reveal_dev_card(state, pid, DevCardKind::RoadBuilding);
                    crate::game::apply_with_chance(state, PLAY_ROAD_BUILDING as usize, None);
                    actions.push(PLAY_ROAD_BUILDING as usize);
                }
                pending_label = Some(format!(
                    "{} plays Road Building",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::PlayedYearOfPlenty { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    // Reveal the card now; action deferred to YearOfPlentyGain.
                    reveal_dev_card(state, pid, DevCardKind::YearOfPlenty);
                }
                pending_label = Some(format!(
                    "{} plays Year of Plenty",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::PlayedMonopoly { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    // Reveal the card now; action deferred to MonopolyResult.
                    reveal_dev_card(state, pid, DevCardKind::Monopoly);
                }
                pending_label = Some(format!(
                    "{} plays Monopoly",
                    player_label(*player, color_map)
                ));
            }

            // --- Silent events: engine handles via preceding action ---
            GameEvent::YearOfPlentyGain { resources, .. } => {
                let (r1, r2) = decode_two_resources(resources);
                if let (Some(r1), Some(r2)) = (r1, r2) {
                    let aid = action::yop_id(r1, r2).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                } else if let Some(r1) = r1 {
                    // Bank had only 1 resource; use same resource for both.
                    let aid = action::yop_id(r1, r1).0 as usize;
                    crate::game::apply_with_chance(state, aid, None);
                    actions.push(aid);
                }
            }

            GameEvent::MonopolyResult { resource, .. } => {
                let aid = action::monopoly_id(*resource).0 as usize;
                crate::game::apply_with_chance(state, aid, None);
                actions.push(aid);
            }

            GameEvent::Discard {
                player, resources, ..
            } => {
                if player_of_color(color_map, *player).is_some() {
                    for &res in &ALL_RESOURCES {
                        for _ in 0..resources[res] {
                            if matches!(state.phase, Phase::Discard { .. }) {
                                let aid = action::discard_id(res).0 as usize;
                                crate::game::apply_with_chance(state, aid, None);
                                actions.push(aid);
                            }
                        }
                    }
                }
            }

            GameEvent::MoveRobber { player } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.current_player = pid;
                    if matches!(state.phase, Phase::MoveRobber) {
                        if let Some(tile) = robber_tile {
                            if tile != state.robber {
                                let aid = action::robber_id(tile).0 as usize;
                                crate::game::apply_with_chance(state, aid, None);
                                actions.push(aid);
                                // Engine sets Phase::Roll when pre_roll (auto-dice).
                                // Colonist resolves dice via events — use PreRoll.
                                if matches!(state.phase, Phase::Roll) {
                                    state.phase = Phase::PreRoll;
                                }
                            } else {
                                // Same tile — can't use engine. Manual transition.
                                state.phase = if state.pre_roll {
                                    Phase::PreRoll
                                } else {
                                    Phase::Main
                                };
                            }
                        } else {
                            state.phase = if state.pre_roll {
                                Phase::PreRoll
                            } else {
                                Phase::Main
                            };
                        }
                    }
                }
            }

            GameEvent::Stole {
                player,
                victim,
                resources,
            } => {
                if matches!(state.phase, Phase::StealResolve) {
                    // Find the resource index for the chance outcome.
                    if let Some(idx) = ALL_RESOURCES.iter().position(|&r| resources[r] > 0) {
                        state.apply_action(idx);
                        actions.push(idx);
                    }
                    // Engine sets Phase::Roll when pre_roll (auto-dice).
                    // Colonist resolves dice via events — use PreRoll.
                    if matches!(state.phase, Phase::Roll) {
                        state.phase = Phase::PreRoll;
                    }
                } else {
                    // Fallback: manual resource transfer.
                    if let Some(pid) = player_of_color(color_map, *player) {
                        state.players[pid].hand.add(*resources);
                    }
                    if let Some(vid) = player_of_color(color_map, *victim) {
                        state.players[vid].hand.sub(*resources);
                    }
                }
            }

            GameEvent::StoleNothing { .. } | GameEvent::StoleUnknown { .. } => {
                // If engine entered StealResolve unexpectedly, force out.
                if matches!(state.phase, Phase::StealResolve) {
                    state.phase = if state.pre_roll {
                        Phase::PreRoll
                    } else {
                        Phase::Main
                    };
                }
            }

            GameEvent::GotResources { player, resources } => {
                // Manual resource tracking — more reliable than engine's
                // distribute_resources when intermediate robber position is
                // unknown.
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                    roll_gains[pid as usize].add(*resources);
                }
            }

            GameEvent::StartingResources { player, resources } => {
                if let Some(pid) = player_of_color(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }

            // Engine computes longest road/army via build/knight actions.
            GameEvent::LongestRoad { .. } | GameEvent::LongestRoadChanged { .. } => {}

            _ => {}
        }
    }

    // Flush final pending entry.
    if let Some(mut label) = pending_label {
        let p1 = format_resources(roll_gains[0]);
        let p2 = format_resources(roll_gains[1]);
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

    // Apply robber position from board snapshot to the final state.
    if let Some(tile) = robber_tile
        && let Some(last) = timeline.last_mut()
    {
        last.state.robber = tile;
    }
}

/// Process new post-setup events incrementally, returning new timeline entries.
///
/// `state` is mutated to reflect the new events. The returned entries can be
/// passed to `GameSession::extend_timeline()`.
pub fn process_new_events(
    state: &mut GameState,
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
    mapper: &CoordMapper,
    robber_tile: Option<TileId>,
) -> (Vec<TimelineEntry>, Vec<usize>) {
    let mut timeline = Vec::new();
    let mut actions = Vec::new();
    process_post_setup(
        state,
        events,
        color_map,
        corner_map,
        edge_map,
        &mut timeline,
        mapper,
        robber_tile,
        &mut actions,
    );
    (timeline, actions)
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
