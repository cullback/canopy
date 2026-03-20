//! # NexusEncoder (1266 features)
//!
//! Heterogeneous encoder that keeps tiles and nodes as separate entity streams.
//!
//! ## Global (107 features)
//!
//! | Block                  | Count | Source                         |
//! |------------------------|-------|--------------------------------|
//! | Phase one-hot          |     7 | `encode_phase` (shared)        |
//! | Per-player nexus × 2   |    88 | `encode_player_nexus`          |
//! | Dice state             |    12 | `encode_dice` (shared)         |
//!
//! ### Per-player nexus (44)
//!
//! | Feature              | Count | Source                         |
//! |----------------------|-------|--------------------------------|
//! | standard             |    22 | `encode_player` (shared)       |
//! | trade_ratios         |     5 | /4, best maritime ratio        |
//! | per_number_production|    11 | /10, building_weight per dice  |
//! | dev_bought_this_turn |     5 | /deck_max, exact self, 0 opp   |
//! | played_dev_this_turn |     1 | binary                         |
//!
//! ## Tiles (19 × 7 = 133 features)
//!
//! | Feature       | Count | Description                        |
//! |---------------|-------|------------------------------------|
//! | resource      |     5 | one-hot                            |
//! | pips          |     1 | pips / 5                           |
//! | robber        |     1 | binary                             |
//!
//! ## Nodes (54 × 19 = 1026 features)
//!
//! | Feature            | Count | Description                        |
//! |--------------------|-------|------------------------------------|
//! | building_cur       |     1 | 0/0.5/1.0                          |
//! | building_opp       |     1 | 0/0.5/1.0                          |
//! | port_ratios        |     5 | per-resource trade improvement     |
//! | production         |     5 | sum of pips from adj tiles / 13    |
//! | blocked_production |     5 | robber-blocked tile pips / 5       |
//! | road_count_cur     |     1 | adj edges with cur road / 3        |
//! | road_count_opp     |     1 | adj edges with opp road / 3        |

use canopy::nn::StateEncoder;

use crate::game::state::GameState;

use super::{
    PIPS, encode_dice, encode_node_blocked_production, encode_node_production,
    encode_per_number_production, encode_phase, encode_player, encode_player_dev_extra,
    encode_port_ratios, encode_road_counts, node_value, tile_numbers,
};

pub struct NexusEncoder;

#[allow(dead_code)]
impl NexusEncoder {
    pub const FEATURE_SIZE: usize = 1266;
    pub const GLOBAL_LEN: usize = 107;
    pub const TILES_F: usize = 7;
    pub const NODES_F: usize = 19;
}

/// Push 44 unified per-player features: std(22) + ext_trimmed(16) + dev_extra(6).
fn encode_player_nexus(
    state: &GameState,
    player: canopy::player::Player,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    // Standard (21)
    encode_player(state, player, out);

    // Trade ratios (5): (4 - ratio) / 4, so better = higher
    for &ratio in &state.players[player].trade_ratios {
        out.push((4 - ratio.min(4)) as f32 / 4.0);
    }

    // Per-number production (11)
    encode_per_number_production(&state.boards[player], &state.topology, tile_numbers, out);

    // Dev extra (6)
    encode_player_dev_extra(state, player, out);
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

        // === Global features (105) ===

        // Phase one-hot (7)
        encode_phase(state, out);

        // Per-player nexus (43 × 2 = 86)
        encode_player_nexus(state, current, &tile_numbers, out);
        encode_player_nexus(state, opp, &tile_numbers, out);

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

        // === Node stream (54 × 19 = 1026) ===
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

            // production (5)
            encode_node_production(node, topo, &tile_numbers, state.robber, out);

            // blocked_production (5)
            encode_node_blocked_production(node, topo, &tile_numbers, state.robber, out);

            // road_count_cur, road_count_opp (2)
            encode_road_counts(node, cur_roads, opp_roads, out);
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
    use canopy::game::Game;
    use canopy::player::Player;
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
    const GLOBAL_OFF: usize = 107;

    fn tile_feat(t: usize, f: usize) -> usize {
        GLOBAL_OFF + t * 7 + f
    }

    fn node_feat(n: usize, f: usize) -> usize {
        GLOBAL_OFF + 19 * 7 + n * 19 + f
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
    fn road_count_features() {
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

        // Road counts at per-node offset 17-18 (2 + 5 + 5 + 5 = 17)
        assert_eq!(
            features[node_feat(n, 17)],
            1.0 / 3.0,
            "cur road count = 1/3"
        );
        assert_eq!(
            features[node_feat(n, 18)],
            1.0 / 3.0,
            "opp road count = 1/3"
        );
    }

    #[test]
    fn node_production_features() {
        let state = make_main_state();
        let topo = &state.topology;
        let tile_numbers = super::super::tile_numbers(topo);

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        // Pick an interior node that has 3 adjacent tiles
        for n in 0..54 {
            let node = &topo.nodes[n];
            // Compute expected production per resource
            let mut expected = [0.0f32; 5];
            for &tid in &node.adjacent_tiles {
                let tile = &topo.tiles[tid.0 as usize];
                if tile.id == state.robber {
                    continue;
                }
                if let Some(r) = tile.terrain.resource() {
                    let number = tile_numbers[tid.0 as usize];
                    if number > 0 {
                        expected[r as usize] += super::PIPS[number as usize] as f32 / 13.0;
                    }
                }
            }
            // Production starts at per-node offset 7 (2 + 5)
            for ri in 0..5 {
                assert!(
                    (features[node_feat(n, 7 + ri)] - expected[ri]).abs() < 1e-6,
                    "node {n} production resource {ri}: got {}, expected {}",
                    features[node_feat(n, 7 + ri)],
                    expected[ri]
                );
            }
        }
    }

    #[test]
    fn node_blocked_production_features() {
        let mut state = make_main_state();
        let topo = &state.topology;
        let tile_numbers = super::super::tile_numbers(topo);

        // Move robber to a resource tile (find one that isn't desert)
        let robber_tile = topo
            .tiles
            .iter()
            .find(|t| t.terrain.resource().is_some())
            .unwrap();
        state.robber = robber_tile.id;

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        // Nodes adjacent to the robber tile should have nonzero blocked_production
        for &nid in &robber_tile.nodes {
            let n = nid.0 as usize;
            let r = robber_tile.terrain.resource().unwrap() as usize;
            let number = tile_numbers[robber_tile.id.0 as usize];
            let expected = super::PIPS[number as usize] as f32 / 5.0;
            // blocked_production starts at per-node offset 12 (2 + 5 + 5)
            assert!(
                (features[node_feat(n, 12 + r)] - expected).abs() < 1e-6,
                "node {n} blocked resource {r}: got {}, expected {}",
                features[node_feat(n, 12 + r)],
                expected
            );
        }

        // A node NOT adjacent to the robber tile should have all-zero blocked_production
        let non_adj_node = (0..54u8)
            .find(|i| !robber_tile.nodes.contains(&crate::game::board::NodeId(*i)))
            .unwrap() as usize;
        for ri in 0..5 {
            assert_eq!(
                features[node_feat(non_adj_node, 12 + ri)],
                0.0,
                "non-adj node {non_adj_node} blocked resource {ri} should be 0"
            );
        }
    }

    #[test]
    fn trade_ratios_reflect_player_ratios() {
        let mut state = make_main_state();
        state.players[Player::One].trade_ratios = [2, 3, 3, 4, 4];
        state.players[Player::Two].trade_ratios = [4, 4, 4, 4, 4];

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        // Global: phase(7) + player_nexus cur starts at 7
        // Inside player_nexus: std(22) then trade ratios at offset 22
        let cur_trade = 7 + 22;
        assert_eq!(features[cur_trade + 0], 0.5, "lumber 2:1 → 0.5");
        assert_eq!(features[cur_trade + 1], 0.25, "brick 3:1 → 0.25");
        assert_eq!(features[cur_trade + 2], 0.25, "wool 3:1 → 0.25");
        assert_eq!(features[cur_trade + 3], 0.0, "grain 4:1 → 0.0");
        assert_eq!(features[cur_trade + 4], 0.0, "ore 4:1 → 0.0");

        // Opp player_nexus starts at 7 + 44 = 51, trade ratios at 51 + 22 = 73
        let opp_trade = 51 + 22;
        for i in 0..5 {
            assert_eq!(features[opp_trade + i], 0.0, "opp all 4:1 → 0.0");
        }
    }
}
