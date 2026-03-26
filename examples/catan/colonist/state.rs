//! Build a `GameState` from colonist.io board data.
//!
//! Creates a topology from the colonist board layout, places buildings from the
//! tileCornerStates/tileEdgeStates snapshot, and replays the game log to track
//! resource hands and dice state.

use std::sync::Arc;

use canopy::player::Player;

use crate::game::board::{EdgeId, NodeId, TileId};
use crate::game::dev_card::DevCardDeck;
use crate::game::dice::{BalancedDice, Dice};
use crate::game::resource::ResourceArray;
use crate::game::state::{GameState, Phase};
use crate::game::topology::Topology;

use super::board::{self, BoardData};
use super::log::GameEvent;

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

    // Set phase to Main so the board renders
    state.phase = Phase::Main;
    state.turn_number = 1;
    state.pre_roll = false;
    state.setup_count = 4;
    state.current_player = Player::One;

    state
}

/// Discover the first two unique player colors from the log, mapping them to
/// Player::One and Player::Two.
fn discover_colors(events: &[GameEvent]) -> Vec<(u8, Player)> {
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
