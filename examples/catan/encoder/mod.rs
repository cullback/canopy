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
//! ## Per-player (21 × 2 = 42 features) — current player first
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
//!
//! Self dev-playable = held − bought this turn; opponent = hypergeometric expected
//! values over the unknown pool (deck + opponent hand).
//!
//! ## BasicEncoder (479 features)
//!
//! | Block           | Shape   | Features |
//! |-----------------|---------|----------|
//! | Phase           |       7 |        7 |
//! | Players         | 21 x 2  |       42 |
//! | Tiles           | 19 x 7  |      133 |
//! | Nodes           | 54 x 2  |      108 |
//! | Edges           | 72 x 2  |      144 |
//! | Ports           |  9 x 5  |       45 |
//!
//! - **Tile (7)**: resource one-hot (5), dice_prob / max_prob (1), robber (1)
//! - **Node (2)**: building_cur, building_opp — 0.0 empty, 0.5 settlement, 1.0 city
//! - **Edge (2)**: road_cur, road_opp — binary
//! - **Port (5)**: resource one-hot; generic 3:1 = all zeros
//!
//! ## RichNodeEncoder (1489 features)
//!
//! See [`rich_node`] module for the full feature table.
//!
//! ## Normalization quick-reference
//!
//! | Divisor | Meaning                           |
//! |---------|-----------------------------------|
//! |      19 | max resource cards of one type    |
//! |  14/5/2 | original deck count per dev type  |
//! |       5 | max settlements                   |
//! |       4 | max cities                        |
//! |      15 | max roads                         |
//! |      13 | max pips at a single node (5+4+4) |
//! |       3 | max adjacent edges/nodes per node |
//! |       5 | hop distance BFS cap              |
//! |    5/36 | max single-tile dice probability  |

use std::collections::VecDeque;

use canopy::player::Player;

use crate::game::board::{Node, Port};
use crate::game::dice::Dice;
use crate::game::resource::ALL_RESOURCES;
use crate::game::state::{GameState, Phase, PlayerBoards};
use crate::game::topology::Topology;

mod basic;
mod gnn;
mod gnn2;
mod nexus;
mod rich_node;

pub use basic::BasicEncoder;
pub use gnn::GnnEncoder;
pub use gnn2::Gnn2Encoder;
pub use nexus::NexusEncoder;
pub use rich_node::RichNodeEncoder;

/// Dice probability for each sum (indices 2..=12, 0 and 1 unused).
pub(crate) const DICE_PROB: [f32; 13] = [
    0.0,
    0.0,
    1.0 / 36.0, // 2
    2.0 / 36.0, // 3
    3.0 / 36.0, // 4
    4.0 / 36.0, // 5
    5.0 / 36.0, // 6
    0.0,        // 7
    5.0 / 36.0, // 8
    4.0 / 36.0, // 9
    3.0 / 36.0, // 10
    2.0 / 36.0, // 11
    1.0 / 36.0, // 12
];

pub(crate) const MAX_DICE_PROB: f32 = 5.0 / 36.0;

/// Number of pips (dots on the number token) for each dice sum.
/// pips[n] = 6 - |7 - n| for n in 2..=12, 0 otherwise.
/// Maximum single-tile pips = 5 (for numbers 6 and 8).
pub(crate) const PIPS: [u8; 13] = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];

/// Max total pips at a single node: 13 (5+4+4, due to no-adjacent-6-8 rule).
pub(crate) const MAX_NODE_PIPS: f32 = 13.0;

/// Compute shortest hop-distance from each node to the nearest building in
/// `buildings` (bitmask). Uses multi-source BFS on the board topology graph.
/// Returns distances capped at `cap`.
pub(crate) fn compute_hop_distances(topo: &Topology, buildings: u64, cap: u8) -> [u8; 54] {
    let mut dist = [cap; 54];
    let mut queue = VecDeque::new();

    for i in 0..54u8 {
        if buildings & (1u64 << i) != 0 {
            dist[i as usize] = 0;
            queue.push_back(i);
        }
    }

    while let Some(node_id) = queue.pop_front() {
        let d = dist[node_id as usize];
        if d >= cap {
            continue;
        }
        for &adj in &topo.nodes[node_id as usize].adjacent_nodes {
            let adj_idx = adj.0 as usize;
            if dist[adj_idx] > d + 1 {
                dist[adj_idx] = d + 1;
                queue.push_back(adj.0);
            }
        }
    }

    dist
}

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

/// Push per-player features (21).
///
/// Dev card playable values are exact (held − bought this turn) for self,
/// or hypergeometric expected values for the opponent.
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

/// Push per-slot road features for a node (3 slots × 2 players = 6 features).
/// Padded to 3 for degree-2 nodes.
pub(crate) fn encode_road_slots(node: &Node, cur_roads: u128, opp_roads: u128, out: &mut Vec<f32>) {
    for slot in 0..3 {
        if slot < node.adjacent_edges.len() {
            let mask = 1u128 << node.adjacent_edges[slot].0;
            out.push(f32::from(cur_roads & mask != 0));
            out.push(f32::from(opp_roads & mask != 0));
        } else {
            out.push(0.0);
            out.push(0.0);
        }
    }
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
