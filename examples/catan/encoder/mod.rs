//! # Feature Space Reference
//!
//! ## Phase (7 features) — one-hot with scalar value
//!
//! | Idx | Phase           | Value                 |
//! |-----|-----------------|-----------------------|
//! |  0  | PlaceSettlement | 1.0                   |
//! |  1  | PlaceRoad       | 1.0                   |
//! |  2  | PreRoll         | 1.0                   |
//! |  3  | Discard         | min(1, remaining/10)  |
//! |  4  | MoveRobber      | 1.0                   |
//! |  5  | Main            | 1.0                   |
//! |  6  | RoadBuilding    | 1.0                   |
//!
//! ## Per-player (49 × 2 = 98 features) — current player first
//!
//! | Features              | Count | Normalization           |
//! |-----------------------|-------|-------------------------|
//! | Resources (L/B/W/G/O) | 5     | / 19                    |
//! | Dev cards playable    | 5     | / deck_max (14,5,2,2,2) |
//! | Dev cards played      | 5     | / deck_max              |
//! | Settlements remaining | 1     | / 5                     |
//! | Cities remaining      | 1     | / 4                     |
//! | Roads remaining       | 1     | / 15                    |
//! | Longest-road award    | 1     | binary                  |
//! | Largest-army award    | 1     | binary                  |
//! | Longest road length   | 1     | / 15                    |
//! | Victory points        | 1     | / 15 (total self, public opp) |
//! | Trade ratios          | 5     | / 4                     |
//! | Per-number production | 11    | / 10                    |
//! | Per-resource prod.    | 5     | / 35                    |
//! | Dev bought this turn  | 5     | / deck_max              |
//! | Played dev this turn  | 1     | binary                  |
//!
//! Self dev-playable = held − bought this turn; opponent = hypergeometric expected
//! values over the unknown pool (deck + opponent hand).
//!
//! ## Normalization quick-reference
//!
//! | Divisor | Meaning                           |
//! |---------|-----------------------------------|
//! |      19 | max resource cards of one type    |
//! |  14/5/2 | original deck count per dev type  |
//! |       5 | max settlements                   |
//! |       4 | max cities                        |
//! |      15 | max roads / win threshold (VP)    |
//! |      13 | max pips at a single node (5+4+4) |
//! |       5 | max single-tile pips (for 6 or 8) |
//! |      35 | max per-resource production       |
//! |       6 | max tile corner nodes / BFS cap   |
//! |    5/36 | max single-tile dice probability  |

use canopy::player::Player;

use crate::game::board::{Node, Port};
use crate::game::dice::Dice;
use crate::game::resource::ALL_RESOURCES;
use crate::game::state::{GameState, Phase, PlayerBoards};
use crate::game::topology::Topology;

mod nexus;

pub use nexus::NexusEncoder;

/// Number of pips (dots on the number token) for each dice sum.
/// pips[n] = 6 - |7 - n| for n in 2..=12, 0 otherwise.
/// Maximum single-tile pips = 5 (for numbers 6 and 8).
pub(crate) const PIPS: [u8; 13] = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];

/// Original deck composition per card type.
pub(crate) const ORIGINAL_DECK: [f32; 5] = [14.0, 5.0, 2.0, 2.0, 2.0];

/// Push phase one-hot (7 features).
/// Chance nodes (Roll, StealResolve) are excluded — the agent never sees those states.
pub(crate) fn encode_phase(state: &GameState, out: &mut Vec<f32>) {
    let (phase_idx, phase_value) = match &state.phase {
        Phase::PlaceSettlement => (0, 1.0),
        Phase::PlaceRoad => (1, 1.0),
        Phase::PreRoll => (2, 1.0),
        Phase::Discard { remaining, .. } => (3, (*remaining as f32 / 10.0).min(1.0)),
        Phase::MoveRobber => (4, 1.0),
        Phase::Main => (5, 1.0),
        Phase::RoadBuilding { .. } => (6, 1.0),
        Phase::Roll | Phase::StealResolve => unreachable!(),
        Phase::GameOver(_) => unreachable!(),
    };
    for i in 0..7 {
        out.push(if i == phase_idx { phase_value } else { 0.0 });
    }
}

/// Push per-player features (22).
///
/// Dev card playable values are exact (held − bought this turn) for self,
/// or hypergeometric expected values for the opponent.
/// VP is total (including hidden VP cards) for self, public-only for opponent.
pub(crate) fn encode_player(state: &GameState, player_to_encode: Player, out: &mut Vec<f32>) {
    let player = &state.players[player_to_encode];
    let is_self = player_to_encode == state.current_player;

    // Resources (5)
    for &r in &ALL_RESOURCES {
        out.push(player.hand[r] as f32 / 19.0);
    }

    // Dev cards playable (5) — exact for self, expected for opponent
    let dev_cards_playable = if is_self {
        self_dev_cards_playable(state, player_to_encode)
    } else {
        opponent_expected_dev_cards(state, player_to_encode.opponent(), player_to_encode)
    };
    out.extend_from_slice(&dev_cards_playable);

    // Dev cards played — exact counts, visible for both (5)
    for (count, max) in player.dev_cards_played.0.iter().zip(&ORIGINAL_DECK) {
        out.push(*count as f32 / max);
    }

    // Settlements left (1)
    out.push(player.settlements_left as f32 / 5.0);

    // Cities left (1)
    out.push(player.cities_left as f32 / 4.0);

    // Roads left (1)
    out.push(player.roads_left as f32 / 15.0);

    // Has longest road award (1)
    let has_lr = state
        .longest_road
        .is_some_and(|(lr_pid, _)| lr_pid == player_to_encode);
    out.push(f32::from(has_lr));

    // Has largest army award (1)
    let has_la = state
        .largest_army
        .is_some_and(|(la_pid, _)| la_pid == player_to_encode);
    out.push(f32::from(has_la));

    // Longest road path length (1)
    out.push(state.boards[player_to_encode].road_network.longest_road() as f32 / 15.0);

    // Victory points (1) — total for self, public-only for opponent
    let vps = if is_self {
        state.total_vps(player_to_encode)
    } else {
        state.public_vps(player_to_encode)
    };
    out.push(vps as f32 / 15.0);
}

/// Compute normalized playable dev card values for the perspective player (exact).
/// Playable = held − bought this turn (cards bought this turn can't be played yet).
fn self_dev_cards_playable(state: &GameState, pid: Player) -> [f32; 5] {
    let player = &state.players[pid];
    let mut out = [0.0; 5];
    for (i, max) in ORIGINAL_DECK.iter().enumerate() {
        let playable = player.dev_cards.0[i].saturating_sub(player.dev_cards_bought_this_turn.0[i]);
        out[i] = playable as f32 / max;
    }
    out
}

/// Compute normalized expected dev card held values for the opponent
/// using hypergeometric proportions over the unknown card pool.
fn opponent_expected_dev_cards(state: &GameState, perspective: Player, opp: Player) -> [f32; 5] {
    let self_player = &state.players[perspective];
    let opp_player = &state.players[opp];

    let opp_hand_size: f32 = opp_player.dev_cards.0.iter().sum::<u8>() as f32
        + opp_player.dev_cards_bought_this_turn.0.iter().sum::<u8>() as f32;
    let deck_remaining = state.dev_deck.remaining() as f32;
    let total_unknown = deck_remaining + opp_hand_size;

    let mut out = [0.0; 5];
    if total_unknown > 0.0 {
        for t in 0..5 {
            let unknown_of_type = ORIGINAL_DECK[t]
                - self_player.dev_cards.0[t] as f32
                - self_player.dev_cards_bought_this_turn.0[t] as f32
                - self_player.dev_cards_played.0[t] as f32
                - opp_player.dev_cards_played.0[t] as f32;
            out[t] = (unknown_of_type * opp_hand_size / total_unknown) / ORIGINAL_DECK[t];
        }
    }
    out
}

/// Push 6 extra dev card features per player:
/// - Dev cards bought this turn (5): exact for self, 0 for opponent
/// - Has played dev card this turn (1): binary, visible for both
pub(crate) fn encode_player_dev_extra(
    state: &GameState,
    player_to_encode: Player,
    out: &mut Vec<f32>,
) {
    let player = &state.players[player_to_encode];
    let is_self = player_to_encode == state.current_player;

    // Dev cards bought this turn (5) — exact for self, 0 for opponent
    for (i, max) in ORIGINAL_DECK.iter().enumerate() {
        if is_self {
            out.push(player.dev_cards_bought_this_turn.0[i] as f32 / max);
        } else {
            out.push(0.0);
        }
    }

    // Has played dev card this turn (1)
    out.push(f32::from(player.has_played_dev_card_this_turn));
}

/// Building value for a single node.
/// 0.0 = empty, 0.5 = settlement, 1.0 = city.
pub(crate) fn node_value(boards: &PlayerBoards, i: u8) -> f32 {
    let mask = 1u64 << i;
    if boards.cities & mask != 0 {
        1.0
    } else if boards.settlements & mask != 0 {
        0.5
    } else {
        0.0
    }
}

/// Precompute the dice number for each tile (0 if none, e.g. desert).
pub(crate) fn tile_numbers(topo: &Topology) -> [u8; 19] {
    let mut numbers = [0u8; 19];
    for roll in 2..=12u8 {
        for &tid in &topo.dice_to_tiles[roll as usize] {
            numbers[tid.0 as usize] = roll;
        }
    }
    numbers
}

/// Push per-resource production (5 features): total building_weight × pips per resource.
/// Normalized by MAX_RESOURCE_PRODUCTION.
pub(crate) fn encode_per_resource_production(
    boards: &PlayerBoards,
    topo: &Topology,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    let mut res_prod = [0.0f32; 5];
    for (i, tile) in topo.tiles.iter().enumerate() {
        if let Some(r) = tile.terrain.resource() {
            let number = tile_numbers[i];
            if number == 0 {
                continue;
            }
            let pips = PIPS[number as usize] as f32;
            let mut tile_bw = 0.0f32;
            for &nid in &tile.nodes {
                tile_bw += building_weight(boards, nid.0);
            }
            res_prod[r as usize] += tile_bw * pips;
        }
    }
    for &v in &res_prod {
        out.push(v / MAX_RESOURCE_PRODUCTION);
    }
}

/// Maximum per-resource production (building_weight × pips summed over all tiles of that resource).
pub(crate) const MAX_RESOURCE_PRODUCTION: f32 = 35.0;

/// Push tile building weight features (2 features: own/6, opp/6).
/// Sum of building_weight() for the tile's 6 nodes, for each player.
pub(crate) fn encode_tile_building_weights(
    tile: &crate::game::board::Tile,
    cur_boards: &PlayerBoards,
    opp_boards: &PlayerBoards,
    out: &mut Vec<f32>,
) {
    let mut cur_bw = 0.0f32;
    let mut opp_bw = 0.0f32;
    for &nid in &tile.nodes {
        cur_bw += building_weight(cur_boards, nid.0);
        opp_bw += building_weight(opp_boards, nid.0);
    }
    out.push(cur_bw / 6.0);
    out.push(opp_bw / 6.0);
}

/// Compute network distances from each node to the nearest on-network node.
///
/// Uses multi-source BFS over the 54-node graph with bitmask operations.
/// Seeds: all nodes with a building (settlement or city) OR adjacent to a road.
/// Returns `min(dist, 6) / 6.0` per node; on-network = 0.0, unreachable = 1.0.
pub(crate) fn compute_network_distances(
    adj: &crate::game::board::AdjacencyBitboards,
    boards: &PlayerBoards,
) -> [f32; 54] {
    let buildings = boards.settlements | boards.cities;
    let roads = boards.road_network.roads;

    // Seed: nodes with buildings or adjacent to at least one road
    let mut visited: u64 = buildings;
    for n in 0..54u8 {
        if adj.node_adj_edges[n as usize] & roads != 0 {
            visited |= 1u64 << n;
        }
    }

    let mut dist = [0u8; 54];
    if visited == 0 {
        // No network exists — all nodes get max distance
        return [1.0; 54];
    }

    // Initialize distances: on-network = 0, others = 255 (unvisited)
    for n in 0..54u8 {
        if visited & (1u64 << n) == 0 {
            dist[n as usize] = 255;
        }
    }

    // BFS
    let mut frontier = visited;
    let mut current_dist = 0u8;
    while frontier != 0 && current_dist < 6 {
        current_dist += 1;
        let mut next_frontier: u64 = 0;
        let mut bits = frontier;
        while bits != 0 {
            let n = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let neighbors = adj.node_adj_nodes[n] & !visited;
            next_frontier |= neighbors;
        }
        // Set distances for newly discovered nodes
        let mut new_bits = next_frontier;
        while new_bits != 0 {
            let n = new_bits.trailing_zeros() as usize;
            new_bits &= new_bits - 1;
            dist[n] = current_dist;
        }
        visited |= next_frontier;
        frontier = next_frontier;
    }

    let mut result = [0.0f32; 54];
    for n in 0..54 {
        result[n] = if dist[n] == 255 {
            1.0
        } else {
            dist[n] as f32 / 6.0
        };
    }
    result
}

/// Push per-resource production features for a node (5 features).
/// For each adjacent tile with a resource, adds pips/13 to that resource's slot.
pub(crate) fn encode_node_production(
    node: &Node,
    topo: &Topology,
    tile_numbers: &[u8; 19],
    robber: crate::game::board::TileId,
    out: &mut Vec<f32>,
) {
    let mut prod = [0.0f32; 5];
    for &tid in &node.adjacent_tiles {
        let tile = &topo.tiles[tid.0 as usize];
        if tile.id == robber {
            continue;
        }
        if let Some(r) = tile.terrain.resource() {
            let number = tile_numbers[tid.0 as usize];
            if number > 0 {
                prod[r as usize] += PIPS[number as usize] as f32 / 13.0;
            }
        }
    }
    out.extend_from_slice(&prod);
}

/// Push per-resource blocked production features for a node (5 features).
/// Same as production but only for tiles where the robber sits. Normalized by /5.
pub(crate) fn encode_node_blocked_production(
    node: &Node,
    topo: &Topology,
    tile_numbers: &[u8; 19],
    robber: crate::game::board::TileId,
    out: &mut Vec<f32>,
) {
    let mut blocked = [0.0f32; 5];
    for &tid in &node.adjacent_tiles {
        let tile = &topo.tiles[tid.0 as usize];
        if tile.id != robber {
            continue;
        }
        if let Some(r) = tile.terrain.resource() {
            let number = tile_numbers[tid.0 as usize];
            if number > 0 {
                blocked[r as usize] += PIPS[number as usize] as f32 / 5.0;
            }
        }
    }
    out.extend_from_slice(&blocked);
}

/// Push port ratio features (5): 0.5 for matching specific resource, 0.25 for generic, 0.0 otherwise.
pub(crate) fn encode_port_ratios(node: &Node, out: &mut Vec<f32>) {
    match node.port {
        Some(Port::Specific(r)) => {
            for ri in 0..5 {
                out.push(if ri == r as usize { 0.5 } else { 0.0 });
            }
        }
        Some(Port::Generic) => {
            for _ in 0..5 {
                out.push(0.25);
            }
        }
        None => {
            for _ in 0..5 {
                out.push(0.0);
            }
        }
    }
}

/// Push per-number production (11 features): total building_weight per dice value 2..=12.
pub(crate) fn encode_per_number_production(
    boards: &PlayerBoards,
    topo: &Topology,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    let mut number_prod = [0.0f32; 11];
    for (i, tile) in topo.tiles.iter().enumerate() {
        if tile.terrain.resource().is_some() {
            let number = tile_numbers[i];
            if number == 0 {
                continue;
            }
            let mut tile_bw = 0.0f32;
            for &nid in &tile.nodes {
                tile_bw += building_weight(boards, nid.0);
            }
            number_prod[(number - 2) as usize] += tile_bw;
        }
    }
    for &v in &number_prod {
        out.push(v / MAX_NUMBER_PRODUCTION);
    }
}

/// Building weight: 0 = empty, 1 = settlement, 2 = city.
pub(crate) fn building_weight(boards: &PlayerBoards, node: u8) -> f32 {
    let mask = 1u64 << node;
    if boards.cities & mask != 0 {
        2.0
    } else if boards.settlements & mask != 0 {
        1.0
    } else {
        0.0
    }
}

/// Random dice weights (unnormalized): [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], total 36.
pub(crate) const RANDOM_DICE_PROBS: [f32; 11] = [
    1.0 / 36.0,
    2.0 / 36.0,
    3.0 / 36.0,
    4.0 / 36.0,
    5.0 / 36.0,
    6.0 / 36.0,
    5.0 / 36.0,
    4.0 / 36.0,
    3.0 / 36.0,
    2.0 / 36.0,
    1.0 / 36.0,
];

/// Maximum building weight on tiles sharing a dice number (~10).
pub(crate) const MAX_NUMBER_PRODUCTION: f32 = 10.0;

/// Push dice state features (12): roll probabilities (11) + deck fraction (1).
pub(crate) fn encode_dice(state: &GameState, out: &mut Vec<f32>) {
    match &state.dice {
        Dice::Random => {
            out.extend_from_slice(&RANDOM_DICE_PROBS);
            out.push(1.0);
        }
        Dice::Balanced(b) => {
            let ws = b.weights(state.current_player);
            let total: u32 = ws.iter().map(|(_, w)| w).sum();
            if total > 0 {
                let inv = 1.0 / total as f32;
                for &(_, w) in &ws {
                    out.push(w as f32 * inv);
                }
            } else {
                out.extend_from_slice(&RANDOM_DICE_PROBS);
            }
            out.push(b.cards_left() as f32 / 36.0);
        }
    }
}
