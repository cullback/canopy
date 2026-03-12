//! # GnnEncoder (1345 features)
//!
//! | Block           | Shape   | Features |
//! |-----------------|---------|----------|
//! | Phase           |       7 |        7 |
//! | Players         | 21 × 2  |       42 |
//! | Nodes           | 54 × 24 |     1296 |
//!
//! No tile, edge, or port streams — the GNN learns spatial relationships
//! through message passing on the board graph.
//!
//! ## Per-node features (24)
//!
//! | Features                              | Count | Norm      |
//! |---------------------------------------|-------|-----------|
//! | building_cur, building_opp            |     2 | 0/0.5/1.0 |
//! | production_pips per resource          |     5 | pips/13   |
//! | robbed_production per resource        |     5 | pips/13   |
//! | port_type one-hot (L/B/W/G/O/generic) |     6 | binary    |
//! | road_slot per neighbor (×3×2 players) |     6 | binary    |

use canopy2::nn::StateEncoder;

use crate::game::board::Port;
use crate::game::state::GameState;

use super::{MAX_NODE_PIPS, PIPS, encode_phase, encode_player, node_value};

pub struct GnnEncoder;

#[allow(dead_code)]
impl GnnEncoder {
    /// Per-node feature count for this encoder.
    pub const NODES_F: usize = 24;
    /// No edge stream (GNN uses graph structure directly).
    pub const EDGES_F: usize = 0;
    /// No tile stream (production info is in per-node features).
    pub const TILES_F: usize = 0;
    /// No port stream (port info is in per-node features).
    pub const PORTS_F: usize = 0;
}

impl StateEncoder<GameState> for GnnEncoder {
    // Global: 7 + 42 = 49
    // Node stream: 54 × 24 = 1296
    // Total: 1345
    fn feature_size(&self) -> usize {
        1345
    }

    fn encode(&self, state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();

        // === Phase one-hot (7) ===
        encode_phase(state, out);

        // === Per-player features (21 × 2 = 42) ===
        encode_player(state, current, out);
        encode_player(state, opp, out);

        // === Pre-compute per-node production data ===
        let topo = &state.topology;
        let cur_board = &state.boards[current];
        let opp_board = &state.boards[opp];

        let mut node_prod_pips = [[0u8; 5]; 54];
        let mut node_robbed_pips = [[0u8; 5]; 54];

        for i in 0..54usize {
            let node = &topo.nodes[i];
            for &tile_id in &node.adjacent_tiles {
                let tile = &topo.tiles[tile_id.0 as usize];
                if let Some(resource) = tile.terrain.resource() {
                    let mut tile_pips = 0u8;
                    for roll in 2..=12u8 {
                        if topo.dice_to_tiles[roll as usize].contains(&tile_id) {
                            tile_pips = PIPS[roll as usize];
                            break;
                        }
                    }
                    node_prod_pips[i][resource as usize] += tile_pips;
                    if tile_id == state.robber {
                        node_robbed_pips[i][resource as usize] += tile_pips;
                    }
                }
            }
        }

        let cur_roads = cur_board.road_network.roads;
        let opp_roads = opp_board.road_network.roads;

        // === Node stream (54 × 24 = 1296) ===
        for i in 0..54u8 {
            let node = &topo.nodes[i as usize];
            let idx = i as usize;

            // 1. building_cur, building_opp (2)
            out.push(node_value(cur_board, i));
            out.push(node_value(opp_board, i));

            // 2. production_pips per resource (5) — /13 pips
            for &pips in &node_prod_pips[idx] {
                out.push(pips as f32 / MAX_NODE_PIPS);
            }

            // 3. robbed_production per resource (5) — /13 pips
            for &pips in &node_robbed_pips[idx] {
                out.push(pips as f32 / MAX_NODE_PIPS);
            }

            // 4. port_type one-hot (6):
            //    lumber_2:1, brick_2:1, wool_2:1, grain_2:1, ore_2:1, generic_3:1
            match node.port {
                Some(Port::Specific(r)) => {
                    for ri in 0..5 {
                        out.push(f32::from(ri == r as usize));
                    }
                    out.push(0.0);
                }
                Some(Port::Generic) => {
                    for _ in 0..5 {
                        out.push(0.0);
                    }
                    out.push(1.0);
                }
                None => {
                    for _ in 0..6 {
                        out.push(0.0);
                    }
                }
            }

            // 5. road_slot per neighbor (3 slots × 2 players = 6)
            //    Slot order matches adjacent_edges; padded to 3 for degree-2 nodes.
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
        GnnEncoder.encode(&state, &mut features);
        assert_eq!(features.len(), GnnEncoder.feature_size());
    }

    #[test]
    fn feature_vector_length_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        GnnEncoder.encode(&state, &mut features);
        assert_eq!(features.len(), GnnEncoder.feature_size());
    }

    #[test]
    fn values_in_range() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        GnnEncoder.encode(&state, &mut features);
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
        GnnEncoder.encode(&state, &mut features);
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
        GnnEncoder.encode(&state, &mut features);
        let p1_features = features.clone();
        state.current_player = state.current_player.opponent();
        GnnEncoder.encode(&state, &mut features);
        assert_ne!(p1_features, features);
        assert_eq!(p1_features.len(), features.len());
    }

    // ── Feature offset helpers ───────────────────────────────────────

    const NODE_OFF: usize = 49;

    fn node_feat(n: usize, f: usize) -> usize {
        NODE_OFF + n * 24 + f
    }

    fn make_main_state() -> GameState {
        let mut state = make_state();
        state.phase = Phase::Main;
        state.current_player = Player::One;
        state
    }

    // ── Production pips ──────────────────────────────────────────────

    #[test]
    fn production_pips_reflect_adjacent_tiles() {
        let state = make_main_state();
        let topo = &state.topology;
        let mut features = Vec::new();
        GnnEncoder.encode(&state, &mut features);

        for n in 0..54 {
            let node = &topo.nodes[n];
            let mut expected_pips = [0u8; 5];
            for &tile_id in &node.adjacent_tiles {
                let tile = &topo.tiles[tile_id.0 as usize];
                if let Some(resource) = tile.terrain.resource() {
                    for roll in 2..=12u8 {
                        if topo.dice_to_tiles[roll as usize].contains(&tile_id) {
                            expected_pips[resource as usize] += PIPS[roll as usize];
                            break;
                        }
                    }
                }
            }
            for r in 0..5 {
                let feat_idx = node_feat(n, 2 + r);
                let expected = expected_pips[r] as f32 / MAX_NODE_PIPS;
                assert_eq!(
                    features[feat_idx], expected,
                    "node {n} resource {r}: expected {expected}, got {}",
                    features[feat_idx]
                );
            }
        }
    }

    // ── Robber blocks production ─────────────────────────────────────

    #[test]
    fn robbed_production_matches_robber_tile() {
        let mut state = make_main_state();
        let topo = &state.topology;

        let robber_tile = topo
            .tiles
            .iter()
            .find(|t| t.terrain.resource().is_some())
            .unwrap()
            .id;
        state.robber = robber_tile;

        let mut features = Vec::new();
        GnnEncoder.encode(&state, &mut features);

        let tile = &topo.tiles[robber_tile.0 as usize];
        let resource = tile.terrain.resource().unwrap();
        let mut tile_pips = 0u8;
        for roll in 2..=12u8 {
            if topo.dice_to_tiles[roll as usize].contains(&robber_tile) {
                tile_pips = PIPS[roll as usize];
                break;
            }
        }

        for n in 0..54 {
            let node = &topo.nodes[n];
            let is_adjacent = node.adjacent_tiles.contains(&robber_tile);
            let robbed_feat = node_feat(n, 7 + resource as usize);
            if is_adjacent {
                assert_eq!(
                    features[robbed_feat],
                    tile_pips as f32 / MAX_NODE_PIPS,
                    "node {n} should have robbed production for resource {resource:?}"
                );
            }
        }

        for n in 0..54 {
            let node = &topo.nodes[n];
            if !node.adjacent_tiles.contains(&robber_tile) {
                for r in 0..5 {
                    let robbed_feat = node_feat(n, 7 + r);
                    assert_eq!(
                        features[robbed_feat], 0.0,
                        "node {n} resource {r}: should have 0 robbed production"
                    );
                }
            }
        }
    }

    // ── Port features ────────────────────────────────────────────────

    #[test]
    fn port_one_hot_correct() {
        let state = make_main_state();
        let topo = &state.topology;
        let mut features = Vec::new();
        GnnEncoder.encode(&state, &mut features);

        for n in 0..54 {
            let node = &topo.nodes[n];
            let port_start = node_feat(n, 12);
            let port_slice = &features[port_start..port_start + 6];
            let sum: f32 = port_slice.iter().sum();

            match node.port {
                None => {
                    assert_eq!(
                        sum, 0.0,
                        "node {n} has no port but port features sum to {sum}"
                    );
                }
                Some(Port::Specific(r)) => {
                    assert_eq!(
                        sum, 1.0,
                        "node {n} has specific port but features sum to {sum}"
                    );
                    assert_eq!(
                        port_slice[r as usize], 1.0,
                        "node {n} specific port resource {r:?} not set"
                    );
                }
                Some(Port::Generic) => {
                    assert_eq!(
                        sum, 1.0,
                        "node {n} has generic port but features sum to {sum}"
                    );
                    assert_eq!(port_slice[5], 1.0, "node {n} generic port bit not set");
                }
            }
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
        GnnEncoder.encode(&state, &mut features);

        assert_eq!(features[node_feat(s_node, 0)], 0.5, "settlement = 0.5");
        assert_eq!(features[node_feat(c_node, 0)], 1.0, "city = 1.0");
        assert_eq!(
            features[node_feat(opp_node, 0)],
            0.0,
            "opp building in cur slot"
        );
        assert_eq!(
            features[node_feat(opp_node, 1)],
            0.5,
            "opp settlement = 0.5"
        );
        assert_eq!(
            features[node_feat(s_node, 1)],
            0.0,
            "no opp building at cur settlement"
        );

        let empty = 0usize;
        assert_eq!(features[node_feat(empty, 0)], 0.0);
        assert_eq!(features[node_feat(empty, 1)], 0.0);
    }

    // ── Road slot features ────────────────────────────────────────────

    #[test]
    fn road_slot_features() {
        let mut state = make_main_state();
        let topo = &state.topology;

        // Pick a node and place roads on some of its adjacent edges.
        let n = 5usize;
        let node = &topo.nodes[n];

        // Place cur road on slot 0, opp road on slot 1.
        let cur_edge = node.adjacent_edges[0];
        let opp_edge = node.adjacent_edges[1];
        state.boards[Player::One].road_network.roads = 1u128 << cur_edge.0;
        state.boards[Player::Two].road_network.roads = 1u128 << opp_edge.0;

        let mut features = Vec::new();
        GnnEncoder.encode(&state, &mut features);

        // Road slots start at per-node offset 18, each slot is 2 features (cur, opp).
        // Slot 0: cur=1, opp=0
        assert_eq!(features[node_feat(n, 18)], 1.0, "slot 0 cur road");
        assert_eq!(features[node_feat(n, 19)], 0.0, "slot 0 opp road");
        // Slot 1: cur=0, opp=1
        assert_eq!(features[node_feat(n, 20)], 0.0, "slot 1 cur road");
        assert_eq!(features[node_feat(n, 21)], 1.0, "slot 1 opp road");
        // Slot 2: no road (or padded if degree-2 node)
        assert_eq!(features[node_feat(n, 22)], 0.0, "slot 2 cur road");
        assert_eq!(features[node_feat(n, 23)], 0.0, "slot 2 opp road");
    }

    #[test]
    fn road_slot_perspective_swap() {
        let mut state = make_main_state();
        let topo = &state.topology;
        let n = 5usize;
        let edge = topo.nodes[n].adjacent_edges[0];

        // P1 has a road on slot 0.
        state.boards[Player::One].road_network.roads = 1u128 << edge.0;

        let mut f1 = Vec::new();
        GnnEncoder.encode(&state, &mut f1);

        state.current_player = Player::Two;
        let mut f2 = Vec::new();
        GnnEncoder.encode(&state, &mut f2);

        // P1 view: slot 0 cur=1, opp=0
        assert_eq!(f1[node_feat(n, 18)], 1.0, "P1 view: cur road");
        assert_eq!(f1[node_feat(n, 19)], 0.0, "P1 view: opp road");
        // P2 view: slot 0 cur=0, opp=1
        assert_eq!(f2[node_feat(n, 18)], 0.0, "P2 view: cur road");
        assert_eq!(f2[node_feat(n, 19)], 1.0, "P2 view: opp road");
    }

    // ── Perspective swap ─────────────────────────────────────────────

    #[test]
    fn perspective_swap_flips_cur_opp() {
        let mut state = make_main_state();

        let cur_node = 10usize;
        let opp_node = 20usize;
        state.boards[Player::One].settlements = 1u64 << cur_node;
        state.boards[Player::Two].settlements = 1u64 << opp_node;

        let mut f1 = Vec::new();
        GnnEncoder.encode(&state, &mut f1);

        state.current_player = Player::Two;
        let mut f2 = Vec::new();
        GnnEncoder.encode(&state, &mut f2);

        assert_eq!(f1[node_feat(cur_node, 0)], 0.5, "P1 view: cur settlement");
        assert_eq!(
            f2[node_feat(cur_node, 1)],
            0.5,
            "P2 view: opp settlement (was P1's)"
        );
        assert_eq!(f1[node_feat(opp_node, 1)], 0.5, "P1 view: opp settlement");
        assert_eq!(
            f2[node_feat(opp_node, 0)],
            0.5,
            "P2 view: cur settlement (was P2's)"
        );
    }
}
