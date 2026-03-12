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
//! | Dev cards held        | 5     | / deck_max (14,5,2,2,2) |
//! | Dev cards played      | 5     | / deck_max              |
//! | Settlements remaining | 1     | / 5                     |
//! | Cities remaining      | 1     | / 4                     |
//! | Roads remaining       | 1     | / 15                    |
//! | Longest-road award    | 1     | binary                  |
//! | Largest-army award    | 1     | binary                  |
//! | Longest road length   | 1     | / 15                    |
//!
//! Self dev-held = exact counts; opponent dev-held = hypergeometric expected
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
use std::sync::Arc;

use canopy2::nn::StateEncoder;
use canopy2::player::Player;

use crate::game::resource::ALL_RESOURCES;
use crate::game::state::{GameState, Phase, PlayerBoards};
use crate::game::topology::Topology;

mod basic;
mod gnn;
mod gnn2;
mod rich_node;

pub use basic::BasicEncoder;
pub use gnn::GnnEncoder;
pub use gnn2::Gnn2Encoder;
pub use rich_node::RichNodeEncoder;

/// Load a neural evaluator from a checkpoint, dispatching on encoder name.
pub fn load_evaluator(
    name: &str,
    path: &str,
    device: &canopy2::train::Device,
) -> Arc<dyn canopy2::eval::Evaluator<GameState> + Sync> {
    use canopy2::nn::NeuralEvaluator;
    use canopy2::train::InferBackend;

    match name {
        "basic" => {
            let encoder: Arc<dyn StateEncoder<GameState>> = Arc::new(BasicEncoder);
            Arc::new(
                NeuralEvaluator::<GameState, InferBackend, _>::from_checkpoint(
                    encoder,
                    crate::model::init_simple(device),
                    path,
                    device.clone(),
                ),
            )
        }
        "rich" => {
            let encoder: Arc<dyn StateEncoder<GameState>> = Arc::new(RichNodeEncoder);
            Arc::new(
                NeuralEvaluator::<GameState, InferBackend, _>::from_checkpoint(
                    encoder,
                    crate::model::init_simple_with(
                        RichNodeEncoder::NODES_F,
                        RichNodeEncoder::EDGES_F,
                        RichNodeEncoder::TILES_F,
                        RichNodeEncoder::PORTS_F,
                        device,
                    ),
                    path,
                    device.clone(),
                ),
            )
        }
        "gnn" => {
            let encoder: Arc<dyn StateEncoder<GameState>> = Arc::new(GnnEncoder);
            Arc::new(
                NeuralEvaluator::<GameState, InferBackend, _>::from_checkpoint(
                    encoder,
                    crate::model::init_gnn(device),
                    path,
                    device.clone(),
                ),
            )
        }
        "gnn2" => {
            let encoder: Arc<dyn StateEncoder<GameState>> = Arc::new(Gnn2Encoder);
            Arc::new(
                NeuralEvaluator::<GameState, InferBackend, _>::from_checkpoint(
                    encoder,
                    crate::model::init_gnn_with::<_, 101, 34>(device),
                    path,
                    device.clone(),
                ),
            )
        }
        other => {
            let known = ["basic", "rich", "gnn", "gnn2"];
            panic!("unknown encoder '{other}', available: {known:?}");
        }
    }
}

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
/// Dev card held values are exact counts for self, or hypergeometric
/// expected values for the opponent.
pub(crate) fn encode_player(state: &GameState, player_to_encode: Player, out: &mut Vec<f32>) {
    let player = &state.players[player_to_encode];
    let is_self = player_to_encode == state.current_player;

    // Resources (5)
    for &r in &ALL_RESOURCES {
        out.push(player.hand[r] as f32 / 19.0);
    }

    // Dev cards held (5) — exact for self, expected for opponent
    let dev_cards_held = if is_self {
        self_dev_cards(state, player_to_encode)
    } else {
        opponent_expected_dev_cards(state, player_to_encode.opponent(), player_to_encode)
    };
    out.extend_from_slice(&dev_cards_held);

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

/// Compute normalized dev card held values for the perspective player (exact).
fn self_dev_cards(state: &GameState, pid: Player) -> [f32; 5] {
    let player = &state.players[pid];
    let mut out = [0.0; 5];
    for (i, (count, max)) in player.dev_cards.0.iter().zip(&ORIGINAL_DECK).enumerate() {
        out[i] = *count as f32 / max;
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

/// Push tile stream (19 × 7 = 133 features).
pub(crate) fn encode_tiles(state: &GameState, out: &mut Vec<f32>) {
    let topo = &state.topology;
    for tile in &topo.tiles {
        // Resource one-hot (5)
        let resource_idx = tile.terrain.resource().map(|r| r as usize);
        for i in 0..5 {
            out.push(f32::from(resource_idx == Some(i)));
        }
        // Dice probability (1)
        let mut prob = 0.0f32;
        for roll in 2..=12u8 {
            if topo.dice_to_tiles[roll as usize].contains(&tile.id) {
                prob += DICE_PROB[roll as usize];
            }
        }
        out.push(prob / MAX_DICE_PROB);
        // Robber (1)
        out.push(f32::from(state.robber == tile.id));
    }
}

/// Push edge stream (72 × 2 = 144 features).
pub(crate) fn encode_edges(state: &GameState, out: &mut Vec<f32>) {
    let current = state.current_player;
    let opp = current.opponent();
    let cur_board = &state.boards[current];
    let opp_board = &state.boards[opp];
    for i in 0..72u8 {
        let mask = 1u128 << i;
        out.push(f32::from(cur_board.road_network.roads & mask != 0));
        out.push(f32::from(opp_board.road_network.roads & mask != 0));
    }
}

/// Push port stream (9 × 5 = 45 features).
pub(crate) fn encode_ports(state: &GameState, out: &mut Vec<f32>) {
    for &port_type in &state.topology.port_types {
        let resource_idx = port_type.map(|r| r as usize);
        for i in 0..5 {
            out.push(f32::from(resource_idx == Some(i)));
        }
    }
}
