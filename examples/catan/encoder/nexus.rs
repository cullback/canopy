//! # NexusEncoder (928 features)
//!
//! Heterogeneous encoder that keeps tiles and nodes as separate entity streams.
//!
//! ## Global (93 features)
//!
//! | Block                  | Count | Source                    |
//! |------------------------|-------|---------------------------|
//! | Phase one-hot          |     7 | `encode_phase` (shared)   |
//! | Per-player std × 2     |    42 | `encode_player` (shared)  |
//! | Per-player ext × 2     |    32 | `encode_player_ext_trim`  |
//! | Dice state             |    12 | `encode_dice` (shared)    |
//!
//! ### Per-player extended trimmed (16)
//!
//! | Feature              | Count | Norm | Description                                |
//! |----------------------|-------|------|--------------------------------------------|
//! | trade_ratios         |     5 | /4   | Best maritime ratio per resource            |
//! | per_number_production|    11 | /10  | Total building_weight per dice value (2-12) |
//!
//! ## Tiles (19 × 7 = 133 features)
//!
//! | Feature       | Count | Description                        |
//! |---------------|-------|------------------------------------|
//! | resource      |     5 | one-hot                            |
//! | pips          |     1 | pips / 5                           |
//! | robber        |     1 | binary                             |
//!
//! ## Nodes (54 × 13 = 702 features)
//!
//! | Feature       | Count | Description                        |
//! |---------------|-------|------------------------------------|
//! | building_cur  |     1 | 0/0.5/1.0                          |
//! | building_opp  |     1 | 0/0.5/1.0                          |
//! | port_ratios   |     5 | per-resource trade improvement     |
//! | road_slots    |     6 | 3 adj edges × 2 players            |

use canopy2::nn::StateEncoder;

use crate::game::state::GameState;

use super::{
    PIPS, encode_dice, encode_per_number_production, encode_phase, encode_player,
    encode_port_ratios, encode_road_slots, node_value, tile_numbers,
};

pub struct NexusEncoder;

#[allow(dead_code)]
impl NexusEncoder {
    pub const FEATURE_SIZE: usize = 928;
    pub const GLOBAL_LEN: usize = 93;
    pub const TILES_F: usize = 7;
    pub const NODES_F: usize = 13;
}

/// Push 16 trimmed extended per-player features: trade ratios (5) + per-number production (11).
fn encode_player_ext_trimmed(
    state: &GameState,
    player: canopy2::player::Player,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    // Trade ratios (5): (4 - ratio) / 4, so better = higher
    for &ratio in &state.players[player].trade_ratios {
        out.push((4 - ratio.min(4)) as f32 / 4.0);
    }

    // Per-number production (11)
    encode_per_number_production(&state.boards[player], &state.topology, tile_numbers, out);
}

impl StateEncoder<GameState> for NexusEncoder {
    fn feature_size(&self) -> usize {
        Self::FEATURE_SIZE
    }

    fn encode(&self, state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();
        let topo = &state.topology;

        let tile_numbers = tile_numbers(topo);

        // === Global features (93) ===

        // Phase one-hot (7)
        encode_phase(state, out);

        // Per-player standard (21 × 2 = 42)
        encode_player(state, current, out);
        encode_player(state, opp, out);

        // Per-player extended trimmed (16 × 2 = 32)
        encode_player_ext_trimmed(state, current, &tile_numbers, out);
        encode_player_ext_trimmed(state, opp, &tile_numbers, out);

        // Dice state (12)
        encode_dice(state, out);

        debug_assert_eq!(out.len(), Self::GLOBAL_LEN);

        // === Tile stream (19 × 7 = 133) ===
        for (i, tile) in topo.tiles.iter().enumerate() {
            // Resource one-hot (5)
            let resource_idx = tile.terrain.resource().map(|r| r as usize);
            for ri in 0..5 {
                out.push(f32::from(resource_idx == Some(ri)));
            }
            // Pips / 5 (1)
            let number = tile_numbers[i];
            let pips = if number > 0 {
                PIPS[number as usize] as f32 / 5.0
            } else {
                0.0
            };
            out.push(pips);
            // Robber (1)
            out.push(f32::from(state.robber == tile.id));
        }

        // === Node stream (54 × 13 = 702) ===
        let cur_board = &state.boards[current];
        let opp_board = &state.boards[opp];
        let cur_roads = cur_board.road_network.roads;
        let opp_roads = opp_board.road_network.roads;

        for i in 0..54u8 {
            let node = &topo.nodes[i as usize];

            // building_cur, building_opp (2)
            out.push(node_value(cur_board, i));
            out.push(node_value(opp_board, i));

            // port_ratios (5)
            encode_port_ratios(node, out);

            // road_slot × 3 × 2 players (6)
            encode_road_slots(node, cur_roads, opp_roads, out);
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
        NexusEncoder.encode(&state, &mut features);
        assert_eq!(features.len(), NexusEncoder.feature_size());
    }

    #[test]
    fn feature_vector_length_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);
        assert_eq!(features.len(), NexusEncoder.feature_size());
    }

    #[test]
    fn values_in_range() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);
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
        NexusEncoder.encode(&state, &mut features);
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
        NexusEncoder.encode(&state, &mut features);
        let p1_features = features.clone();
        state.current_player = state.current_player.opponent();
        NexusEncoder.encode(&state, &mut features);
        assert_ne!(p1_features, features);
        assert_eq!(p1_features.len(), features.len());
    }

    // ── Feature offset helpers ───────────────────────────────────────
    const GLOBAL_OFF: usize = 93;
    const TILES_OFF: usize = GLOBAL_OFF + 19 * 7; // 93 + 133 = 226
    const NODES_OFF: usize = TILES_OFF; // tiles end = nodes start... no:
    // tiles: 93..226, nodes: 226..928

    fn tile_feat(t: usize, f: usize) -> usize {
        GLOBAL_OFF + t * 7 + f
    }

    fn node_feat(n: usize, f: usize) -> usize {
        GLOBAL_OFF + 19 * 7 + n * 13 + f
    }

    fn make_main_state() -> GameState {
        let mut state = make_state();
        state.phase = Phase::Main;
        state.current_player = Player::One;
        state
    }

    #[test]
    fn tile_features_correct() {
        let state = make_main_state();
        let topo = &state.topology;
        let tile_numbers = super::super::tile_numbers(topo);

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        for (i, tile) in topo.tiles.iter().enumerate() {
            let resource_idx = tile.terrain.resource().map(|r| r as usize);
            // Resource one-hot
            for ri in 0..5 {
                let expected = f32::from(resource_idx == Some(ri));
                assert_eq!(
                    features[tile_feat(i, ri)],
                    expected,
                    "tile {i} resource {ri}"
                );
            }
            // Pips
            let number = tile_numbers[i];
            let expected_pips = if number > 0 {
                super::PIPS[number as usize] as f32 / 5.0
            } else {
                0.0
            };
            assert_eq!(features[tile_feat(i, 5)], expected_pips, "tile {i} pips");
            // Robber
            assert_eq!(
                features[tile_feat(i, 6)],
                f32::from(state.robber == tile.id),
                "tile {i} robber"
            );
        }
    }

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
        NexusEncoder.encode(&state, &mut features);

        assert_eq!(features[node_feat(s_node, 0)], 0.5, "settlement = 0.5");
        assert_eq!(features[node_feat(c_node, 0)], 1.0, "city = 1.0");
        assert_eq!(features[node_feat(opp_node, 0)], 0.0, "opp in cur slot");
        assert_eq!(features[node_feat(opp_node, 1)], 0.5, "opp settlement");
    }

    #[test]
    fn port_ratios_correct() {
        let state = make_main_state();
        let topo = &state.topology;
        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        for n in 0..54 {
            let node = &topo.nodes[n];
            let port_start = node_feat(n, 2);
            let port_slice = &features[port_start..port_start + 5];

            match node.port {
                None => {
                    for ri in 0..5 {
                        assert_eq!(port_slice[ri], 0.0, "node {n} no port [{ri}]");
                    }
                }
                Some(Port::Specific(r)) => {
                    for ri in 0..5 {
                        let expected = if ri == r as usize { 0.5 } else { 0.0 };
                        assert_eq!(port_slice[ri], expected, "node {n} specific [{ri}]");
                    }
                }
                Some(Port::Generic) => {
                    for ri in 0..5 {
                        assert_eq!(port_slice[ri], 0.25, "node {n} generic [{ri}]");
                    }
                }
            }
        }
    }

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
        NexusEncoder.encode(&state, &mut features);

        // Road slots start at per-node offset 7 (2 + 5)
        assert_eq!(features[node_feat(n, 7)], 1.0, "slot 0 cur road");
        assert_eq!(features[node_feat(n, 8)], 0.0, "slot 0 opp road");
        assert_eq!(features[node_feat(n, 9)], 0.0, "slot 1 cur road");
        assert_eq!(features[node_feat(n, 10)], 1.0, "slot 1 opp road");
        assert_eq!(features[node_feat(n, 11)], 0.0, "slot 2 cur road");
        assert_eq!(features[node_feat(n, 12)], 0.0, "slot 2 opp road");
    }

    #[test]
    fn trade_ratios_reflect_player_ratios() {
        let mut state = make_main_state();
        state.players[Player::One].trade_ratios = [2, 3, 3, 4, 4];
        state.players[Player::Two].trade_ratios = [4, 4, 4, 4, 4];

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        // Global: phase(7) + player_std(42) = 49, then ext_trimmed starts
        let cur_ext = 49;
        assert_eq!(features[cur_ext + 0], 0.5, "lumber 2:1 → 0.5");
        assert_eq!(features[cur_ext + 1], 0.25, "brick 3:1 → 0.25");
        assert_eq!(features[cur_ext + 2], 0.25, "wool 3:1 → 0.25");
        assert_eq!(features[cur_ext + 3], 0.0, "grain 4:1 → 0.0");
        assert_eq!(features[cur_ext + 4], 0.0, "ore 4:1 → 0.0");

        // Opp ext_trimmed starts at 49 + 16 = 65
        let opp_ext = 65;
        for i in 0..5 {
            assert_eq!(features[opp_ext + i], 0.0, "opp all 4:1 → 0.0");
        }
    }
}
