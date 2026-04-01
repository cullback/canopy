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

/// Move one hidden dev card to the revealed hand so the engine can play it.
fn reveal_hidden_card(state: &mut GameState, pid: Player, kind: DevCardKind) {
    let ps = &mut state.players[pid];
    if ps.dev_cards[kind] == 0 && ps.hidden_dev_cards > 0 {
        ps.hidden_dev_cards -= 1;
        ps.dev_cards[kind] += 1;
    }
}

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

    replay_search(
        topology.clone(),
        board,
        events,
        &color_map,
        &corner_map,
        &edge_map,
        mapper,
    )
    .expect("replay_search: no valid path found")
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
    /// Tiles where each player gets stolen from. Precomputed from the full log.
    /// Used to sort building candidates — nodes adjacent to steal tiles first.
    pub steal_tiles: PerPlayer<Vec<TileId>>,
    /// Event indices of Stole events where the robber position is known.
    /// Only these should trigger backtracking; unknown-position steals
    /// are applied directly since we can't validate adjacency.
    pub known_steal_indices: Vec<usize>,
}

/// Replay new events onto an existing state using engine actions.
/// Used by the live polling path to process incremental events.
/// Returns `Some(entries)` on success (state updated), `None` on failure
/// (state unchanged — caller should retry with more events later).
pub fn replay_events(
    state: &mut GameState,
    events: &[GameEvent],
    ctx: &ReplayCtx,
) -> Option<(Vec<TimelineEntry>, Vec<usize>)> {
    let timeline = Vec::new();
    let (final_state, entries, walk_actions) =
        try_replay(state.clone(), events, 0, ctx, timeline, Vec::new())?;
    *state = final_state;
    Some((entries, walk_actions))
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
            steal_tiles: PerPlayer::default(),
            known_steal_indices: Vec::new(),
        }
    }
}

/// Scan the full log to find tiles where each player gets stolen from.
/// Uses TileBlocked to infer intermediate robber positions and DOM for the last.
fn precompute_steal_tiles(
    events: &[GameEvent],
    color_map: &[(u8, Player)],
    topology: &Topology,
    ctx: &ReplayCtx,
) -> (PerPlayer<Vec<TileId>>, Vec<usize>) {
    let mut result: PerPlayer<Vec<TileId>> = PerPlayer::default();
    let mut known_indices: Vec<usize> = Vec::new();

    // Track current robber tile through the log.
    // Start on desert (topology default).
    let desert = (0..topology.tiles.len())
        .find(|&i| topology.tiles[i].terrain.resource().is_none())
        .unwrap_or(0);
    let mut robber: Option<TileId> = Some(TileId(desert as u8));

    // Find the last MoveRobber index to use DOM position.
    let last_robber_idx = events
        .iter()
        .rposition(|e| matches!(e, GameEvent::MoveRobber { .. }));

    for (i, event) in events.iter().enumerate() {
        match event {
            GameEvent::MoveRobber { .. } => {
                if Some(i) == last_robber_idx {
                    // Last move: DOM is authoritative.
                    robber = ctx.dom.robber_tile_index.map(TileId);
                } else {
                    // Intermediate: infer from TileBlocked after this move.
                    robber = None; // unknown until TileBlocked found
                    for e in &events[i + 1..] {
                        match e {
                            GameEvent::TileBlocked {
                                dice_number,
                                resource: Some(resource),
                            } => {
                                for &tid in &topology.dice_to_tiles[*dice_number as usize] {
                                    let t = &topology.tiles[tid.0 as usize];
                                    if t.terrain.resource() == Some(*resource) {
                                        robber = Some(tid);
                                        break;
                                    }
                                }
                                break;
                            }
                            GameEvent::MoveRobber { .. } | GameEvent::PlayedKnight { .. } => break,
                            _ => {}
                        }
                    }
                }
            }
            GameEvent::Stole { victim, .. } => {
                if let Some(tile) = robber {
                    // Known robber position → constraint + backtrackable.
                    if let Some(pid) = player_of_color(color_map, *victim) {
                        if !result[pid].contains(&tile) {
                            result[pid].push(tile);
                        }
                    }
                    known_indices.push(i);
                }
                // Unknown position → will be applied directly, no backtrack.
            }
            _ => {}
        }
    }

    (result, known_indices)
}

/// Sort building candidates: nodes adjacent to steal-required tiles first.
fn sort_by_steal_coverage(
    candidates: &mut [NodeId],
    player: Player,
    topology: &Topology,
    steal_tiles: &PerPlayer<Vec<TileId>>,
) {
    let required = &steal_tiles[player];
    if required.is_empty() {
        return;
    }
    candidates.sort_by_key(|&nid| {
        let node = &topology.nodes[nid.0 as usize];
        let covers = node
            .adjacent_tiles
            .iter()
            .filter(|&&tid| required.contains(&tid))
            .count();
        std::cmp::Reverse(covers) // more coverage first
    });
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
    let mut ctx = ReplayCtx::from_buildings(dom, color_map, corner_map, edge_map, mapper);
    let (steal_tiles, known_steal_indices) =
        precompute_steal_tiles(events, color_map, &state.topology, &ctx);
    ctx.steal_tiles = steal_tiles;
    ctx.known_steal_indices = known_steal_indices;

    let timeline = vec![TimelineEntry {
        label: "Game start".into(),
        state: state.clone(),
    }];

    let (_final_state, entries, _actions) =
        try_replay(state, events, 0, &ctx, timeline, Vec::new())?;
    eprintln!("replay_search: {} timeline entries", entries.len());
    Some(entries)
}

/// Recursive replay: process events from `idx`, branching at ambiguous points.
/// Returns `None` on contradiction (GotResources mismatch, no valid candidate).
/// On success returns (final_state, timeline_entries, walk_actions).
fn try_replay(
    mut state: GameState,
    events: &[GameEvent],
    idx: usize,
    ctx: &ReplayCtx,
    mut timeline: Vec<TimelineEntry>,
    mut actions: Vec<usize>,
) -> Option<(GameState, Vec<TimelineEntry>, Vec<usize>)> {
    use crate::game::action::{self, END_TURN, ROLL};
    use crate::game::resource::ALL_RESOURCES;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed);
    if n > 0 && n % 10000 == 0 {
        eprintln!("  try_replay: {n} calls, at event {idx}/{}", events.len());
    }
    // After many calls, log what's failing at the hot event, then bail.
    if n == 500000 && idx < events.len() {
        for ei in 140..=150.min(events.len() - 1) {
            eprintln!("  event[{ei}]: {:?}", std::mem::discriminant(&events[ei]),);
        }
    }
    if n > 600000 {
        return None;
    }

    // Ensure current_player matches the event's player. Pre-roll actions
    // (Knight, etc.) from the next player arrive before their Roll event,
    // so we need to inject END_TURN when we see a different player acting.
    let ensure_player = |state: &mut GameState, actions: &mut Vec<usize>, player: u8| {
        if let Some(pid) = player_of_color(ctx.color_map, player) {
            if state.current_player != pid && matches!(state.phase, Phase::Main) {
                actions.push(END_TURN as usize);
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
                    actions.push(candidates[0].0 as usize);
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
                    if let Some(pid) = pid {
                        sort_by_steal_coverage(
                            &mut candidates,
                            pid,
                            &state.topology,
                            &ctx.steal_tiles,
                        );
                    }
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
                        if let Some(result) =
                            try_replay(trial, events, i + 1, ctx, trial_tl, actions.clone())
                        {
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
                    actions.push((54 + eid.0) as usize);
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
                    ensure_player(&mut state, &mut actions, *player);
                    if matches!(state.phase, Phase::PreRoll) {
                        actions.push(ROLL as usize);
                        state.apply_action(ROLL as usize);
                    }

                    let hands_before: [ResourceArray; 2] = [
                        state.players[Player::One].hand,
                        state.players[Player::Two].hand,
                    ];

                    let chance = (*d1 + *d2 - 2) as usize;
                    actions.push(chance);
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
                            actions.push(action::discard_id(res).0 as usize);
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
                                actions.push(action::robber_id(tile).0 as usize);
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

                    // Inverse constraint: if a roll after this MoveRobber
                    // produces resources from a tile, the robber is NOT on
                    // that tile. Scan all rolls until next MoveRobber.
                    for e in &events[i + 1..] {
                        match e {
                            GameEvent::MoveRobber { .. } | GameEvent::PlayedKnight { .. } => break,
                            GameEvent::Roll { d1, d2, .. } => {
                                let total = d1 + d2;
                                if total != 7 {
                                    // Any tile with this number that has buildings
                                    // and produced resources is NOT the robber tile.
                                    for &tid in &state.topology.dice_to_tiles[total as usize] {
                                        let tile = &state.topology.tiles[tid.0 as usize];
                                        if tile.terrain.resource().is_some() {
                                            let mask =
                                                state.topology.adj.tile_nodes[tid.0 as usize];
                                            let all_buildings = state.player_buildings(Player::One)
                                                | state.player_buildings(Player::Two);
                                            if mask & all_buildings != 0 {
                                                candidates.retain(|&c| c != tid);
                                            }
                                        }
                                    }
                                }
                            }
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
                        actions.push(aid);
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
                            if let Some(result) =
                                try_replay(trial, events, i + 1, ctx, trial_tl, actions.clone())
                            {
                                return Some(result);
                            }
                        }
                        return None; // all robber branches failed
                    }
                }
            }

            GameEvent::Stole {
                player,
                victim,
                resources,
            } => {
                if matches!(state.phase, Phase::StealResolve) {
                    if let Some(idx) = ALL_RESOURCES.iter().position(|&r| resources[r] > 0) {
                        actions.push(idx);
                        state.apply_action(idx);
                    }
                } else if ctx.known_steal_indices.contains(&i) {
                    // Robber position is known but engine didn't enter
                    // StealResolve → building placement is wrong. Backtrack.
                    return None;
                } else {
                    // Robber position unknown for this steal. Apply directly
                    // since we can't validate adjacency.
                    if let Some(pid) = player_of_color(ctx.color_map, *player) {
                        state.players[pid].hand.add(*resources);
                    }
                    if let Some(vid) = player_of_color(ctx.color_map, *victim) {
                        state.players[vid].hand.sub(*resources);
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
                ensure_player(&mut state, &mut actions, *player);
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
                    actions.push(aid);
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
                        actions.push(aid);
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
                            if let Some(result) =
                                try_replay(trial, events, i + 1, ctx, trial_tl, actions.clone())
                            {
                                return Some(result);
                            }
                        }
                        return None;
                    }
                }
            }

            GameEvent::BuildSettlement { player, corner } => {
                ensure_player(&mut state, &mut actions, *player);
                let pid = player_of_color(ctx.color_map, *player);
                let from_coords = corner
                    .map(|(x, y, z)| ctx.mapper.map_corner(x, y, z))
                    .and_then(|c| ctx.corner_map.get(&c).copied());

                if let Some(nid) = from_coords {
                    let aid = action::settlement_id(nid).0 as usize;
                    actions.push(aid);
                    crate::game::apply_with_chance(&mut state, aid, None);
                    let label =
                        format!("{} builds settlement", player_label(*player, ctx.color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                } else if let Some(p) = pid {
                    // No coordinates — branch on unplaced DOM settlements.
                    let mut candidates: Vec<NodeId> = ctx.dom_settlements[p]
                        .iter()
                        .filter(|&&nid| {
                            state.boards[p].settlements & (1u64 << nid.0) == 0
                                && state.boards[p].cities & (1u64 << nid.0) == 0
                        })
                        .copied()
                        .collect();
                    if candidates.len() == 1 {
                        let aid = action::settlement_id(candidates[0]).0 as usize;
                        actions.push(aid);
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
                        if let Some(p) = pid {
                            sort_by_steal_coverage(
                                &mut candidates,
                                p,
                                &state.topology,
                                &ctx.steal_tiles,
                            );
                        }
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
                            if let Some(result) =
                                try_replay(trial, events, i + 1, ctx, trial_tl, actions.clone())
                            {
                                return Some(result);
                            }
                        }
                        return None;
                    }
                }
            }

            GameEvent::BuildCity { player, corner } => {
                ensure_player(&mut state, &mut actions, *player);
                let pid = player_of_color(ctx.color_map, *player);
                let from_coords = corner
                    .map(|(x, y, z)| ctx.mapper.map_corner(x, y, z))
                    .and_then(|c| ctx.corner_map.get(&c).copied());

                if let Some(nid) = from_coords {
                    let aid = action::city_id(nid).0 as usize;
                    actions.push(aid);
                    crate::game::apply_with_chance(&mut state, aid, None);
                    let label = format!("{} builds city", player_label(*player, ctx.color_map));
                    timeline.push(TimelineEntry {
                        label,
                        state: state.clone(),
                    });
                } else if let Some(p) = pid {
                    let mut candidates: Vec<NodeId> = ctx
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
                        actions.push(aid);
                        crate::game::apply_with_chance(&mut state, aid, None);
                        let label = format!("{} builds city", player_label(*player, ctx.color_map));
                        timeline.push(TimelineEntry {
                            label,
                            state: state.clone(),
                        });
                    } else if candidates.is_empty() {
                        return None;
                    } else {
                        if let Some(p) = pid {
                            sort_by_steal_coverage(
                                &mut candidates,
                                p,
                                &state.topology,
                                &ctx.steal_tiles,
                            );
                        }
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
                            if let Some(result) =
                                try_replay(trial, events, i + 1, ctx, trial_tl, actions.clone())
                            {
                                return Some(result);
                            }
                        }
                        return None;
                    }
                }
            }

            // -- Dev cards -------------------------------------------------
            GameEvent::BuyDevCard { player } => {
                ensure_player(&mut state, &mut actions, *player);
                if player_of_color(ctx.color_map, *player).is_some() {
                    // hidden dev card buy has no single action ID
                    crate::game::apply_hidden_dev_card_buy(&mut state);
                }
            }
            GameEvent::PlayedKnight { player } => {
                ensure_player(&mut state, &mut actions, *player);
                if let Some(pid) = player_of_color(ctx.color_map, *player) {
                    reveal_hidden_card(&mut state, pid, DevCardKind::Knight);
                    actions.push(action::PLAY_KNIGHT as usize);
                    state.apply_action(action::PLAY_KNIGHT as usize);
                }
            }
            GameEvent::PlayedRoadBuilding { player } => {
                ensure_player(&mut state, &mut actions, *player);
                if let Some(pid) = player_of_color(ctx.color_map, *player) {
                    reveal_hidden_card(&mut state, pid, DevCardKind::RoadBuilding);
                    actions.push(action::PLAY_ROAD_BUILDING as usize);
                    state.apply_action(action::PLAY_ROAD_BUILDING as usize);
                }
            }
            GameEvent::PlayedMonopoly { player } => {
                ensure_player(&mut state, &mut actions, *player);
                if let Some(pid) = player_of_color(ctx.color_map, *player) {
                    reveal_hidden_card(&mut state, pid, DevCardKind::Monopoly);
                    let resource = events[i + 1..].iter().find_map(|e| match e {
                        GameEvent::MonopolyResult { resource, .. } => Some(*resource),
                        _ => None,
                    });
                    if let Some(res) = resource {
                        actions.push(action::monopoly_id(res).0 as usize);
                        state.apply_action(action::monopoly_id(res).0 as usize);
                    }
                }
            }
            GameEvent::MonopolyResult { .. } => {}
            GameEvent::PlayedYearOfPlenty { player } => {
                ensure_player(&mut state, &mut actions, *player);
                if let Some(pid) = player_of_color(ctx.color_map, *player) {
                    reveal_hidden_card(&mut state, pid, DevCardKind::YearOfPlenty);
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
                            actions.push(action::yop_id(a, b).0 as usize);
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
                ensure_player(&mut state, &mut actions, *player);
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
                                    actions.push(aid);
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

    Some((state, timeline, actions))
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
