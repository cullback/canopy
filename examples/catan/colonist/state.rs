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
use crate::game::resource::ResourceArray;
use crate::game::state::{GameState, Phase};
use crate::game::topology::Topology;

use super::board::{self, BoardData};
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
pub fn build_timeline(board: &BoardData, events: &[GameEvent]) -> Vec<TimelineEntry> {
    let (terrains, numbers, port_resources) = board::to_layout(board);
    let topology = Arc::new(Topology::from_layout(terrains, numbers, port_resources));
    let color_map = discover_colors(events);
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    if is_full_log(events, &corner_map, &edge_map) {
        eprintln!("full log detected — building complete timeline");
        build_full_timeline(board, topology, events, &color_map, &corner_map, &edge_map)
    } else {
        eprintln!("late join — single-state timeline from board snapshot");
        vec![TimelineEntry {
            label: "Current position".into(),
            state: build_game_state(board, events),
        }]
    }
}

/// Build a `GameState` from colonist board data and game log.
///
/// Places buildings from the board snapshot (tileCornerStates/tileEdgeStates),
/// replays the log for resource tracking, and sets up dice state.
pub fn build_game_state(board: &BoardData, events: &[GameEvent]) -> GameState {
    let (terrains, numbers, port_resources) = board::to_layout(board);
    let topology = Arc::new(Topology::from_layout(terrains, numbers, port_resources));
    let dev_deck = DevCardDeck::new();
    let dice = Dice::Balanced(BalancedDice::new());

    let mut state = GameState::new(topology.clone(), dev_deck, dice);

    // Build coordinate maps for colonist → canopy mapping
    let corner_map = board::build_corner_map(&topology);
    let edge_map = board::build_edge_map(&topology);

    // Discover color → Player mapping from the log
    let color_map = discover_colors(events);

    // Place buildings from board snapshot
    let buildings = board::extract_buildings(board);
    place_buildings(&mut state, &buildings, &color_map, &corner_map, &edge_map);

    // Replay log for resource hands + dice
    replay_log(&mut state, events, &color_map);

    // Robber
    if let Some(idx) = buildings.robber_tile_index {
        state.robber = TileId(idx);
    }

    // If buildings exist, we joined mid-game: skip to main phase.
    // Otherwise keep the initial PlaceSettlement phase from GameState::new().
    let has_buildings = !buildings.settlements.is_empty()
        || !buildings.cities.is_empty()
        || !buildings.roads.is_empty();
    if has_buildings {
        state.phase = Phase::PreRoll;
        state.turn_number = 1;
        state.pre_roll = true;
        state.setup_count = 4;
        state.current_player = Player::One;
    }

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

fn player_of(color_map: &[(u8, Player)], color: u8) -> Option<Player> {
    color_map
        .iter()
        .find(|&&(c, _)| c == color)
        .map(|&(_, p)| p)
}

/// Look up a colonist color in the color map, returning the internal `Player`.
pub fn player_of_color(color_map: &[(u8, Player)], color: u8) -> Option<Player> {
    player_of(color_map, color)
}

/// Place buildings and roads from the board snapshot.
fn place_buildings(
    state: &mut GameState,
    buildings: &board::BuildingData,
    color_map: &[(u8, Player)],
    corner_map: &std::collections::HashMap<(i32, i32, u8), NodeId>,
    edge_map: &std::collections::HashMap<(i32, i32, u8), EdgeId>,
) {
    let mut placed = (0u32, 0u32, 0u32);

    for &(color, x, y, z) in &buildings.settlements {
        let Some(pid) = player_of(color_map, color) else {
            continue;
        };
        let Some(nid) = corner_map.get(&(x, y, z)).copied() else {
            eprintln!("warning: settlement ({x},{y},{z}) not mapped");
            continue;
        };
        place_settlement(state, pid, nid);
        placed.0 += 1;
    }

    for &(color, x, y, z) in &buildings.cities {
        let Some(pid) = player_of(color_map, color) else {
            continue;
        };
        let Some(nid) = corner_map.get(&(x, y, z)).copied() else {
            eprintln!("warning: city ({x},{y},{z}) not mapped");
            continue;
        };
        place_city(state, pid, nid);
        placed.1 += 1;
    }

    for &(color, x, y, z) in &buildings.roads {
        let Some(pid) = player_of(color_map, color) else {
            continue;
        };
        let Some(eid) = edge_map.get(&(x, y, z)).copied() else {
            eprintln!("warning: road ({x},{y},{z}) not mapped");
            continue;
        };
        place_road(state, pid, eid);
        placed.2 += 1;
    }

    eprintln!(
        "placed {} settlements, {} cities, {} roads from board snapshot",
        placed.0, placed.1, placed.2
    );
}

/// Walk the log to track resource hands and dice state.
fn replay_log(state: &mut GameState, events: &[GameEvent], color_map: &[(u8, Player)]) {
    use crate::game::resource::{CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST};

    for event in events {
        match event {
            GameEvent::Roll { player, d1, d2 } => {
                if let Some(pid) = player_of(color_map, *player) {
                    if let Dice::Balanced(ref mut b) = state.dice {
                        b.draw(d1 + d2, pid);
                    }
                }
            }
            GameEvent::StartingResources { player, resources }
            | GameEvent::GotResources { player, resources } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }
            GameEvent::BuildRoad { player, .. } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(ROAD_COST);
                    state.bank.add(ROAD_COST);
                }
            }
            GameEvent::BuildSettlement { player, .. } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(SETTLEMENT_COST);
                    state.bank.add(SETTLEMENT_COST);
                }
            }
            GameEvent::BuildCity { player, .. } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(CITY_COST);
                    state.bank.add(CITY_COST);
                }
            }
            GameEvent::BuyDevCard { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(DEV_CARD_COST);
                    state.bank.add(DEV_CARD_COST);
                    state.players[pid].hidden_dev_cards += 1;
                    state.dev_deck.remove_unknown();
                }
            }
            GameEvent::Stole {
                player,
                victim,
                resources,
            } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                }
                if let Some(vid) = player_of(color_map, *victim) {
                    state.players[vid].hand.sub(*resources);
                }
            }
            GameEvent::YearOfPlentyGain { player, resources } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }
            GameEvent::MonopolyResult {
                player,
                count,
                resource,
            } => {
                if let Some(pid) = player_of(color_map, *player) {
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
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                }
                if let Some(cpid) = player_of(color_map, *counterparty) {
                    state.players[cpid].hand.sub(*received);
                    state.players[cpid].hand.add(*given);
                }
            }
            GameEvent::BankTrade {
                player,
                given,
                received,
            } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                    state.bank.add(*given);
                    state.bank.sub(*received);
                }
            }
            GameEvent::Discard {
                player, resources, ..
            } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(*resources);
                    state.bank.add(*resources);
                }
            }
            GameEvent::PlayedKnight { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].knights_played += 1;
                    state.players[pid].dev_cards_played[DevCardKind::Knight] += 1;
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
            GameEvent::PlayedMonopoly { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].dev_cards_played[DevCardKind::Monopoly] += 1;
                }
            }
            GameEvent::PlayedRoadBuilding { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].dev_cards_played[DevCardKind::RoadBuilding] += 1;
                }
            }
            GameEvent::PlayedYearOfPlenty { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].dev_cards_played[DevCardKind::YearOfPlenty] += 1;
                }
            }
            _ => {}
        }
    }
}

/// Place a settlement on the board without going through the game action system.
fn place_settlement(state: &mut GameState, pid: Player, nid: NodeId) {
    state.boards[pid].settlements |= 1u64 << nid.0;
    state.players[pid].building_vps += 1;
    state.players[pid].settlements_left = state.players[pid].settlements_left.saturating_sub(1);

    // Update port access
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

    // Update road network frontier
    let opp = pid.opponent();
    let opp_roads = state.boards[opp].road_network.roads;
    state.boards[pid]
        .road_network
        .add_building(nid, &state.topology.adj, opp_roads);
}

/// Place a city on the board.
fn place_city(state: &mut GameState, pid: Player, nid: NodeId) {
    // Remove settlement if present, add city
    state.boards[pid].settlements &= !(1u64 << nid.0);
    state.boards[pid].cities |= 1u64 << nid.0;
    // City gives 2 VP total (1 from settlement + 1 upgrade)
    // If settlement wasn't already counted, give full 2
    if state.players[pid].building_vps == 0 {
        state.players[pid].building_vps = 2;
    } else {
        state.players[pid].building_vps += 1; // +1 for upgrade
    }
    state.players[pid].cities_left = state.players[pid].cities_left.saturating_sub(1);

    // Port access (same as settlement)
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

    // Update road network frontier
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

// -- Timeline building --------------------------------------------------------

/// Check if the log has enough setup placement events with valid coordinates
/// to replay the full game from scratch.
fn is_full_log(
    events: &[GameEvent],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
) -> bool {
    let mut settlements = 0u32;
    let mut roads = 0u32;
    for event in events {
        match event {
            GameEvent::PlaceSettlement {
                corner: Some(c), ..
            } if corner_map.contains_key(c) => {
                settlements += 1;
            }
            GameEvent::PlaceRoad { edge: Some(e), .. } if edge_map.contains_key(e) => {
                roads += 1;
            }
            _ => {}
        }
    }
    settlements >= 4 && roads >= 4
}

fn player_label(color: u8, color_map: &[(u8, Player)]) -> &'static str {
    match player_of(color_map, color) {
        Some(Player::One) => "P1",
        Some(Player::Two) => "P2",
        None => "P?",
    }
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
                corner: Some(c),
            } => {
                if let Some(&nid) = corner_map.get(c) {
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
                edge: Some(e),
            } => {
                if let Some(&eid) = edge_map.get(e) {
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

    // Phase 2: Post-setup events via direct state mutation.
    // Override phase to Main so each snapshot is MCTS-ready.
    state.phase = Phase::Main;
    state.pre_roll = false;

    process_post_setup(
        &mut state,
        &events[event_idx..],
        color_map,
        corner_map,
        edge_map,
        board,
        &mut timeline,
    );

    eprintln!("timeline: {} total entries", timeline.len());
    timeline
}

/// Process post-setup log events, mutating state and appending timeline entries
/// at each significant event.
///
/// "Significant" events (rolls, builds, buys, trades, dev card plays) produce
/// entries. Silent events (resource gains, steals, monopoly results) mutate
/// state but are absorbed into the preceding significant event's snapshot.
fn process_post_setup(
    state: &mut GameState,
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    corner_map: &HashMap<(i32, i32, u8), NodeId>,
    edge_map: &HashMap<(i32, i32, u8), EdgeId>,
    board: &BoardData,
    timeline: &mut Vec<TimelineEntry>,
) {
    use crate::game::resource::{CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST};

    let mut pending_label: Option<String> = None;
    let mut turn = 0u16;

    for event in events {
        let is_entry = matches!(
            event,
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
                | GameEvent::PlayedYearOfPlenty { .. }
        );

        // Flush pending entry before the next entry-producing event.
        if is_entry {
            if let Some(label) = pending_label.take() {
                timeline.push(TimelineEntry {
                    label,
                    state: state.clone(),
                });
            }
        }

        match event {
            // --- Entry-producing events ---
            GameEvent::Roll { player, d1, d2 } => {
                if let Some(pid) = player_of(color_map, *player) {
                    if let Dice::Balanced(ref mut b) = state.dice {
                        b.draw(d1 + d2, pid);
                    }
                    state.current_player = pid;
                    turn += 1;
                    state.turn_number = turn;
                    state.pre_roll = false;
                    state.phase = Phase::Main;
                }
                pending_label = Some(format!(
                    "{} rolls {}",
                    player_label(*player, color_map),
                    d1 + d2
                ));
            }

            GameEvent::BuildRoad { player, edge } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(ROAD_COST);
                    state.bank.add(ROAD_COST);
                    if let Some(&eid) = edge.and_then(|e| edge_map.get(&e)) {
                        place_road(state, pid, eid);
                    }
                }
                pending_label = Some(format!("{} builds road", player_label(*player, color_map)));
            }

            GameEvent::BuildSettlement { player, corner, .. } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(SETTLEMENT_COST);
                    state.bank.add(SETTLEMENT_COST);
                    if let Some(&nid) = corner.and_then(|c| corner_map.get(&c)) {
                        place_settlement(state, pid, nid);
                    }
                }
                pending_label = Some(format!(
                    "{} builds settlement",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::BuildCity { player, corner, .. } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(CITY_COST);
                    state.bank.add(CITY_COST);
                    if let Some(&nid) = corner.and_then(|c| corner_map.get(&c)) {
                        place_city(state, pid, nid);
                    }
                }
                pending_label = Some(format!("{} builds city", player_label(*player, color_map)));
            }

            GameEvent::BuyDevCard { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(DEV_CARD_COST);
                    state.bank.add(DEV_CARD_COST);
                    state.players[pid].hidden_dev_cards += 1;
                    state.dev_deck.remove_unknown();
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
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                }
                if let Some(cpid) = player_of(color_map, *counterparty) {
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
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(*given);
                    state.players[pid].hand.add(*received);
                    state.bank.add(*given);
                    state.bank.sub(*received);
                }
                pending_label = Some(format!("{} bank trade", player_label(*player, color_map)));
            }

            GameEvent::PlayedKnight { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].knights_played += 1;
                    state.players[pid].dev_cards_played[DevCardKind::Knight] += 1;
                    // Check largest army.
                    let knights = state.players[pid].knights_played;
                    if knights >= 3 {
                        let beats_current = match state.largest_army {
                            Some((_, n)) => knights > n,
                            None => true,
                        };
                        if beats_current {
                            state.largest_army = Some((pid, knights));
                        }
                    }
                }
                pending_label = Some(format!("{} plays Knight", player_label(*player, color_map)));
            }

            GameEvent::PlayedMonopoly { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].dev_cards_played[DevCardKind::Monopoly] += 1;
                }
                pending_label = Some(format!(
                    "{} plays Monopoly",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::PlayedRoadBuilding { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].dev_cards_played[DevCardKind::RoadBuilding] += 1;
                }
                pending_label = Some(format!(
                    "{} plays Road Building",
                    player_label(*player, color_map)
                ));
            }

            GameEvent::PlayedYearOfPlenty { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hidden_dev_cards =
                        state.players[pid].hidden_dev_cards.saturating_sub(1);
                    state.players[pid].dev_cards_played[DevCardKind::YearOfPlenty] += 1;
                }
                pending_label = Some(format!(
                    "{} plays Year of Plenty",
                    player_label(*player, color_map)
                ));
            }

            // --- Silent events: mutate state, no timeline entry ---
            GameEvent::StartingResources { player, resources }
            | GameEvent::GotResources { player, resources } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }

            GameEvent::Stole {
                player,
                victim,
                resources,
            } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                }
                if let Some(vid) = player_of(color_map, *victim) {
                    state.players[vid].hand.sub(*resources);
                }
            }

            GameEvent::YearOfPlentyGain { player, resources } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.add(*resources);
                    state.bank.sub(*resources);
                }
            }

            GameEvent::MonopolyResult {
                player,
                count,
                resource,
            } => {
                if let Some(pid) = player_of(color_map, *player) {
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

            GameEvent::Discard {
                player, resources, ..
            } => {
                if let Some(pid) = player_of(color_map, *player) {
                    state.players[pid].hand.sub(*resources);
                    state.bank.add(*resources);
                }
            }

            GameEvent::LongestRoad { player } => {
                if let Some(pid) = player_of(color_map, *player) {
                    let len = state.boards[pid].road_network.longest_road();
                    state.longest_road = Some((pid, len));
                }
            }

            GameEvent::LongestRoadChanged { to, .. } => {
                if let Some(pid) = player_of(color_map, *to) {
                    let len = state.boards[pid].road_network.longest_road();
                    state.longest_road = Some((pid, len));
                }
            }

            _ => {}
        }
    }

    // Flush final pending entry.
    if let Some(label) = pending_label {
        timeline.push(TimelineEntry {
            label,
            state: state.clone(),
        });
    }

    // Apply robber position from board snapshot to the final state.
    let buildings = board::extract_buildings(board);
    if let Some(idx) = buildings.robber_tile_index {
        if let Some(last) = timeline.last_mut() {
            last.state.robber = TileId(idx);
        }
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
    board: &BoardData,
) -> Vec<TimelineEntry> {
    let mut timeline = Vec::new();
    process_post_setup(
        state,
        events,
        color_map,
        corner_map,
        edge_map,
        board,
        &mut timeline,
    );
    timeline
}

/// Apply extracted dev card identities to a game state.
///
/// Converts `hidden_dev_cards` into concrete `dev_cards` entries and removes
/// specific card types from the deck instead of `remove_unknown()`.
pub fn apply_dev_cards(state: &mut GameState, player: Player, cards: &[DevCardKind]) {
    let ps = &mut state.players[player];
    let concrete_count = cards.len() as u8;
    // Only apply if we have hidden cards to resolve.
    if ps.hidden_dev_cards == 0 {
        return;
    }
    // Move cards from hidden to concrete.
    for &kind in cards {
        ps.dev_cards[kind] += 1;
        ps.hidden_dev_cards = ps.hidden_dev_cards.saturating_sub(1);
    }
    eprintln!(
        "applied {} dev card identities for {:?} ({} remain hidden)",
        concrete_count, player, ps.hidden_dev_cards
    );
}
