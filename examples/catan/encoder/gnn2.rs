//! # Gnn2Encoder (1949 features)
//!
//! Restructured GNN encoder that preserves tile-level identity and adds
//! global production/trade features.
//!
//! ## Global (113 features)
//!
//! | Block               | Count | Source                    |
//! |---------------------|-------|---------------------------|
//! | Phase one-hot       |     7 | `encode_phase` (shared)   |
//! | Per-player std × 2  |    42 | `encode_player` (shared)  |
//! | Per-player ext × 2  |    52 | `encode_player_extended`  |
//! | Dice state          |    12 | `encode_dice`             |
//!
//! ### Per-player extended features (26)
//!
//! | Feature              | Count | Norm | Description                                |
//! |----------------------|-------|------|--------------------------------------------|
//! | trade_ratios         |     5 | /4   | Best maritime ratio per resource            |
//! | total_production     |     5 | /26  | Sum of pips × building_weight per resource  |
//! | robbed_production    |     5 | /26  | Same, only for tiles where robber sits      |
//! | per_number_production|    11 | /10  | Total building_weight per dice value (2-12) |
//!
//! ## Per-Node (34 features × 54 nodes = 1836)
//!
//! | Feature                  | Count | Description                        |
//! |--------------------------|-------|------------------------------------|
//! | building_cur, building_opp |   2 | 0/0.5/1.0                          |
//! | tile_slot × 3            |    21 | 7 per slot: resource(5)+number+robber |
//! | port_ratios              |     5 | Per-resource trade improvement      |
//! | road_slot × 3 × 2       |     6 | Binary per player per adj edge      |

use canopy2::nn::StateEncoder;

use crate::game::board::Port;
use crate::game::dice::Dice;
use crate::game::state::GameState;

use super::{PIPS, encode_phase, encode_player, node_value};

/// Random dice weights (unnormalized): [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], total 36.
const RANDOM_DICE_PROBS: [f32; 11] = [
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

/// Maximum production normalization: MAX_NODE_PIPS (13) × city weight (2) = 26.
const MAX_PRODUCTION: f32 = 26.0;

/// Maximum building weight on tiles sharing a dice number (~10).
const MAX_NUMBER_PRODUCTION: f32 = 10.0;

pub struct Gnn2Encoder;

#[allow(dead_code)]
impl Gnn2Encoder {
    pub const NODES_F: usize = 34;
    pub const EDGES_F: usize = 0;
    pub const TILES_F: usize = 0;
    pub const PORTS_F: usize = 0;
}

/// Building weight: 0 = empty, 1 = settlement, 2 = city.
fn building_weight(boards: &crate::game::state::PlayerBoards, node: u8) -> f32 {
    let mask = 1u64 << node;
    if boards.cities & mask != 0 {
        2.0
    } else if boards.settlements & mask != 0 {
        1.0
    } else {
        0.0
    }
}

/// Push 26 extended per-player features: trade ratios (5), total production (5),
/// robbed production (5), per-number production (11).
fn encode_player_extended(
    state: &GameState,
    player: canopy2::player::Player,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    let boards = &state.boards[player];
    let topo = &state.topology;

    // Trade ratios (5): (4 - ratio) / 4, so better = higher
    for &ratio in &state.players[player].trade_ratios {
        out.push((4 - ratio.min(4)) as f32 / 4.0);
    }

    // Total production per resource (5) and robbed production per resource (5)
    let mut total_prod = [0.0f32; 5];
    let mut robbed_prod = [0.0f32; 5];
    // Per-number production (indices 0-10 → dice values 2-12)
    let mut number_prod = [0.0f32; 11];

    for (i, tile) in topo.tiles.iter().enumerate() {
        if let Some(resource) = tile.terrain.resource() {
            let number = tile_numbers[i];
            if number == 0 {
                continue;
            }
            let pips = PIPS[number as usize] as f32;

            // Sum building_weight × pips across all 6 nodes of this tile
            let mut tile_bw = 0.0f32;
            for &nid in &tile.nodes {
                tile_bw += building_weight(boards, nid.0);
            }

            total_prod[resource as usize] += pips * tile_bw;
            if tile.id == state.robber {
                robbed_prod[resource as usize] += pips * tile_bw;
            }
            // Per-number: index = number - 2
            number_prod[(number - 2) as usize] += tile_bw;
        }
    }

    for &v in &total_prod {
        out.push(v / MAX_PRODUCTION);
    }
    for &v in &robbed_prod {
        out.push(v / MAX_PRODUCTION);
    }
    for &v in &number_prod {
        out.push(v / MAX_NUMBER_PRODUCTION);
    }
}

/// Push dice state features (12): roll probabilities (11) + deck fraction (1).
fn encode_dice(state: &GameState, out: &mut Vec<f32>) {
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

impl StateEncoder<GameState> for Gnn2Encoder {
    // Global: 7 + 42 + 52 + 12 = 113
    // Node stream: 54 × 34 = 1836
    // Total: 1949
    fn feature_size(&self) -> usize {
        1949
    }

    fn encode(&self, state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();
        let topo = &state.topology;

        // === Precompute tile_numbers: dice number for each tile (0 if none) ===
        let mut tile_numbers = [0u8; 19];
        for roll in 2..=12u8 {
            for &tid in &topo.dice_to_tiles[roll as usize] {
                tile_numbers[tid.0 as usize] = roll;
            }
        }

        // === Global features ===

        // Phase one-hot (7)
        encode_phase(state, out);

        // Per-player standard (21 × 2 = 42)
        encode_player(state, current, out);
        encode_player(state, opp, out);

        // Per-player extended (26 × 2 = 52)
        encode_player_extended(state, current, &tile_numbers, out);
        encode_player_extended(state, opp, &tile_numbers, out);

        // Dice state (12)
        encode_dice(state, out);

        // === Node stream (54 × 34 = 1836) ===
        let cur_board = &state.boards[current];
        let opp_board = &state.boards[opp];
        let cur_roads = cur_board.road_network.roads;
        let opp_roads = opp_board.road_network.roads;

        for i in 0..54u8 {
            let node = &topo.nodes[i as usize];

            // 1. building_cur, building_opp (2)
            out.push(node_value(cur_board, i));
            out.push(node_value(opp_board, i));

            // 2. tile_slot × 3 (7 per slot = 21)
            //    Each slot: resource one-hot (5) + number/12 (1) + robber (1)
            for slot in 0..3 {
                if slot < node.adjacent_tiles.len() {
                    let tid = node.adjacent_tiles[slot];
                    let tile = &topo.tiles[tid.0 as usize];
                    let resource_idx = tile.terrain.resource().map(|r| r as usize);
                    // Resource one-hot (5)
                    for ri in 0..5 {
                        out.push(f32::from(resource_idx == Some(ri)));
                    }
                    // Number / 12
                    let number = tile_numbers[tid.0 as usize];
                    out.push(number as f32 / 12.0);
                    // Robber
                    out.push(f32::from(tid == state.robber));
                } else {
                    // Pad empty slot
                    for _ in 0..7 {
                        out.push(0.0);
                    }
                }
            }

            // 3. port_ratios (5): per-resource trade improvement
            //    (4 - ratio) / 4 for matching resources, 0 for no port
            match node.port {
                Some(Port::Specific(r)) => {
                    for ri in 0..5 {
                        if ri == r as usize {
                            out.push(0.5); // (4 - 2) / 4
                        } else {
                            out.push(0.0);
                        }
                    }
                }
                Some(Port::Generic) => {
                    // (4 - 3) / 4 = 0.25 for all resources
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

            // 4. road_slot × 3 × 2 players (6)
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

        debug_assert_eq!(
            out.len(),
            self.feature_size(),
            "feature vector length mismatch: expected {}, got {}",
            self.feature_size(),
            out.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Port;
    use crate::game::dev_card::DevCardDeck;
    use crate::game::dice::Dice;
    use crate::game::state::Phase;
    use crate::game::topology::Topology;
    use canopy2::game::Game;
    use canopy2::player::Player;
    use std::sync::Arc;

    fn make_state() -> GameState {
        let topo = Arc::new(Topology::from_seed(42));
        let mut rng = fastrand::Rng::with_seed(42);
        let deck = DevCardDeck::new(&mut rng);
        GameState::new(topo, deck, Dice::default())
    }

    fn play_setup(state: &mut GameState) {
        let mut actions = Vec::new();
        for _ in 0..4 {
            state.legal_actions(&mut actions);
            state.apply_action(actions[0]);
            state.legal_actions(&mut actions);
            state.apply_action(actions[0]);
        }
    }

    #[test]
    fn feature_vector_length() {
        let state = make_state();
        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);
        assert_eq!(features.len(), Gnn2Encoder.feature_size());
    }

    #[test]
    fn feature_vector_length_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);
        assert_eq!(features.len(), Gnn2Encoder.feature_size());
    }

    #[test]
    fn values_in_range() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);
        for (i, &v) in features.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "feature {i} out of [0.0, 1.0]: {v}"
            );
        }
    }

    #[test]
    fn values_in_range_before_setup() {
        let state = make_state();
        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);
        for (i, &v) in features.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "feature {i} out of [0.0, 1.0]: {v}"
            );
        }
    }

    #[test]
    fn perspective_symmetry() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);
        let p1_features = features.clone();
        state.current_player = state.current_player.opponent();
        Gnn2Encoder.encode(&state, &mut features);
        assert_ne!(p1_features, features);
        assert_eq!(p1_features.len(), features.len());
    }

    // ── Feature offset helpers ───────────────────────────────────────

    const GLOBAL_OFF: usize = 113;

    fn node_feat(n: usize, f: usize) -> usize {
        GLOBAL_OFF + n * 34 + f
    }

    fn make_main_state() -> GameState {
        let mut state = make_state();
        state.phase = Phase::Main;
        state.current_player = Player::One;
        state
    }

    // ── Tile slot correctness ────────────────────────────────────────

    #[test]
    fn tile_slots_match_adjacent_tiles() {
        let state = make_main_state();
        let topo = &state.topology;

        // Precompute tile_numbers
        let mut tile_numbers = [0u8; 19];
        for roll in 2..=12u8 {
            for &tid in &topo.dice_to_tiles[roll as usize] {
                tile_numbers[tid.0 as usize] = roll;
            }
        }

        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);

        for n in 0..54 {
            let node = &topo.nodes[n];
            for slot in 0..3 {
                let slot_start = node_feat(n, 2 + slot * 7);
                if slot < node.adjacent_tiles.len() {
                    let tid = node.adjacent_tiles[slot];
                    let tile = &topo.tiles[tid.0 as usize];
                    let resource_idx = tile.terrain.resource().map(|r| r as usize);

                    // Check resource one-hot
                    for ri in 0..5 {
                        let expected = f32::from(resource_idx == Some(ri));
                        assert_eq!(
                            features[slot_start + ri],
                            expected,
                            "node {n} slot {slot} resource {ri}"
                        );
                    }

                    // Check number
                    let expected_number = tile_numbers[tid.0 as usize] as f32 / 12.0;
                    assert_eq!(
                        features[slot_start + 5],
                        expected_number,
                        "node {n} slot {slot} number"
                    );

                    // Check robber
                    let expected_robber = f32::from(tid == state.robber);
                    assert_eq!(
                        features[slot_start + 6],
                        expected_robber,
                        "node {n} slot {slot} robber"
                    );
                } else {
                    // Padded slot: all zeros
                    for f in 0..7 {
                        assert_eq!(
                            features[slot_start + f],
                            0.0,
                            "node {n} slot {slot} padded feature {f}"
                        );
                    }
                }
            }
        }
    }

    // ── Port ratio correctness ───────────────────────────────────────

    #[test]
    fn port_ratios_correct() {
        let state = make_main_state();
        let topo = &state.topology;
        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);

        for n in 0..54 {
            let node = &topo.nodes[n];
            let port_start = node_feat(n, 23); // 2 + 21 = 23
            let port_slice = &features[port_start..port_start + 5];

            match node.port {
                None => {
                    for ri in 0..5 {
                        assert_eq!(
                            port_slice[ri], 0.0,
                            "node {n} has no port but port_ratio[{ri}] = {}",
                            port_slice[ri]
                        );
                    }
                }
                Some(Port::Specific(r)) => {
                    for ri in 0..5 {
                        let expected = if ri == r as usize { 0.5 } else { 0.0 };
                        assert_eq!(
                            port_slice[ri], expected,
                            "node {n} specific port {r:?}: port_ratio[{ri}] = {}, expected {expected}",
                            port_slice[ri]
                        );
                    }
                }
                Some(Port::Generic) => {
                    for ri in 0..5 {
                        assert_eq!(
                            port_slice[ri], 0.25,
                            "node {n} generic port: port_ratio[{ri}] = {}, expected 0.25",
                            port_slice[ri]
                        );
                    }
                }
            }
        }
    }

    // ── Trade ratio features ─────────────────────────────────────────

    #[test]
    fn trade_ratios_reflect_player_ratios() {
        let mut state = make_main_state();
        // Set P1 to have a 2:1 lumber port and 3:1 generic
        state.players[Player::One].trade_ratios = [2, 3, 3, 4, 4];
        state.players[Player::Two].trade_ratios = [4, 4, 4, 4, 4];

        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);

        // Global layout: phase(7) + player_std(21×2=42) + player_ext(26×2=52)
        // Current player extended starts at offset 49
        let cur_ext = 49;
        // Trade ratios are first 5 features of extended block
        assert_eq!(features[cur_ext + 0], 0.5, "lumber 2:1 → 0.5");
        assert_eq!(features[cur_ext + 1], 0.25, "brick 3:1 → 0.25");
        assert_eq!(features[cur_ext + 2], 0.25, "wool 3:1 → 0.25");
        assert_eq!(features[cur_ext + 3], 0.0, "grain 4:1 → 0.0");
        assert_eq!(features[cur_ext + 4], 0.0, "ore 4:1 → 0.0");

        // Opponent extended starts at 49 + 26 = 75
        let opp_ext = 75;
        for i in 0..5 {
            assert_eq!(features[opp_ext + i], 0.0, "opp all 4:1 → 0.0");
        }
    }

    // ── Building features ────────────────────────────────────────────

    #[test]
    fn building_features_correct() {
        let mut state = make_main_state();
        let s_node = 10usize;
        let c_node = 20usize;
        let opp_node = 30usize;
        state.boards[Player::One].settlements = 1u64 << s_node;
        state.boards[Player::One].cities = 1u64 << c_node;
        state.boards[Player::Two].settlements = 1u64 << opp_node;

        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);

        assert_eq!(features[node_feat(s_node, 0)], 0.5, "settlement = 0.5");
        assert_eq!(features[node_feat(c_node, 0)], 1.0, "city = 1.0");
        assert_eq!(features[node_feat(opp_node, 0)], 0.0, "opp in cur slot");
        assert_eq!(features[node_feat(opp_node, 1)], 0.5, "opp settlement");
    }

    // ── Road slot features ───────────────────────────────────────────

    #[test]
    fn road_slot_features() {
        let mut state = make_main_state();
        let topo = &state.topology;
        let n = 5usize;
        let node = &topo.nodes[n];

        let cur_edge = node.adjacent_edges[0];
        let opp_edge = node.adjacent_edges[1];
        state.boards[Player::One].road_network.roads = 1u128 << cur_edge.0;
        state.boards[Player::Two].road_network.roads = 1u128 << opp_edge.0;

        let mut features = Vec::new();
        Gnn2Encoder.encode(&state, &mut features);

        // Road slots start at per-node offset 28 (2 + 21 + 5)
        assert_eq!(features[node_feat(n, 28)], 1.0, "slot 0 cur road");
        assert_eq!(features[node_feat(n, 29)], 0.0, "slot 0 opp road");
        assert_eq!(features[node_feat(n, 30)], 0.0, "slot 1 cur road");
        assert_eq!(features[node_feat(n, 31)], 1.0, "slot 1 opp road");
        assert_eq!(features[node_feat(n, 32)], 0.0, "slot 2 cur road");
        assert_eq!(features[node_feat(n, 33)], 0.0, "slot 2 opp road");
    }
}
