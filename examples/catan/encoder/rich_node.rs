//! # RichNodeEncoder (1489 features)
//!
//! | Block           | Shape   | Features |
//! |-----------------|---------|----------|
//! | Phase           |       7 |        7 |
//! | Players         | 21 × 2  |       42 |
//! | Nodes           | 54 × 24 |     1296 |
//! | Edges           | 72 × 2  |      144 |
//!
//! No tile or port streams — that information is folded into per-node features.
//!
//! ## Rich node (24 features per node)
//!
//! | Features                              | Count | Norm      |
//! |---------------------------------------|-------|-----------|
//! | building_cur, building_opp            |     2 | 0/0.5/1.0 |
//! | potential_production per resource     |     5 | pips/13   |
//! | robbed_production per resource        |     5 | pips/13   |
//! | port_type one-hot (L/B/W/G/O/generic) |     6 | binary    |
//! | road_count_cur, road_count_opp        |     2 | /3        |
//! | adj_buildings_cur, adj_buildings_opp  |     2 | /3        |
//! | hop_distance_cur, hop_distance_opp    |     2 | /5 capped |
//!
//! ## Rich edge (2 features per edge)
//!
//! | Feature                    | Norm      | Note                                        |
//! |----------------------------|-----------|---------------------------------------------|
//! | best_endpoint_production   | pips/13   | zeroed if endpoint occupied or adj-occupied |
//! | best_endpoint_hop_distance | /5 capped | min hop_cur of two endpoints                |

use canopy2::nn::StateEncoder;

use crate::game::board::Port;
use crate::game::state::GameState;

use super::{MAX_NODE_PIPS, PIPS, compute_hop_distances, encode_phase, encode_player, node_value};

pub struct RichNodeEncoder;

impl RichNodeEncoder {
    /// Per-node feature count for this encoder.
    pub const NODES_F: usize = 24;
    /// Per-edge feature count for this encoder.
    pub const EDGES_F: usize = 2;
    /// No tile stream (production info is in the per-node features).
    pub const TILES_F: usize = 0;
    /// No port stream (port info is in the per-node features).
    pub const PORTS_F: usize = 0;
}

/// Features intentionally excluded as linear combinations of existing inputs:
///
/// - **Victory points**: `(5 - settlements_left) + 2*(4 - cities_left)
///   + has_longest_road*2 + has_largest_army*2 + dev_cards_played[VP]`,
///   all within the per-player block.
/// - **Bank supply per resource**: `19 - cur_resources - opp_resources`
///   in a 2-player game.
/// - **Can-afford flags**: simple thresholds on the 5 resource counts
///   already in the per-player block.
/// - **Turn number**: VP serves as a sufficient proxy for game progress.
impl StateEncoder<GameState> for RichNodeEncoder {
    // Global: 7
    // Per-player (x2): 21 x 2 = 42
    // Node stream: 54 x 24 = 1296
    // Edge stream: 72 x 2 = 144
    // Total: 7 + 42 + 1296 + 144 = 1489
    const FEATURE_SIZE: usize = 1489;

    fn encode(state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();

        // === Phase one-hot (7) ===
        encode_phase(state, out);

        // === Per-player features (21 x 2 = 42) ===
        encode_player(state, current, out);
        encode_player(state, opp, out);

        // === Pre-compute per-node data ===
        let topo = &state.topology;
        let cur_board = &state.boards[current];
        let opp_board = &state.boards[opp];
        let cur_buildings = cur_board.settlements | cur_board.cities;
        let opp_buildings = opp_board.settlements | opp_board.cities;
        let occupied = cur_buildings | opp_buildings;

        // Hop distances (multi-source BFS, capped at 5)
        let hop_dist_cur = compute_hop_distances(topo, cur_buildings, 5);
        let hop_dist_opp = compute_hop_distances(topo, opp_buildings, 5);

        // Per-node pips by resource, total pips, and robbed pips
        let mut node_prod_pips = [[0u8; 5]; 54]; // per resource
        let mut node_total_pips = [0u8; 54]; // across all resources
        let mut node_robbed_pips = [[0u8; 5]; 54]; // pips from robbed tile

        for i in 0..54usize {
            let node = &topo.nodes[i];
            for &tile_id in &node.adjacent_tiles {
                let tile = &topo.tiles[tile_id.0 as usize];
                if let Some(resource) = tile.terrain.resource() {
                    // Find pips for this tile
                    let mut tile_pips = 0u8;
                    for roll in 2..=12u8 {
                        if topo.dice_to_tiles[roll as usize].contains(&tile_id) {
                            tile_pips = PIPS[roll as usize];
                            break;
                        }
                    }
                    node_prod_pips[i][resource as usize] += tile_pips;
                    node_total_pips[i] += tile_pips;
                    if tile_id == state.robber {
                        node_robbed_pips[i][resource as usize] += tile_pips;
                    }
                }
            }
        }

        // === Node stream (54 x 24 = 1296) ===
        for i in 0..54u8 {
            let node = &topo.nodes[i as usize];
            let idx = i as usize;

            // 1. building_cur, building_opp (2)
            out.push(node_value(cur_board, i));
            out.push(node_value(opp_board, i));

            // 2. potential_production per resource (5) — /13 pips, ignores robber
            for r in 0..5 {
                out.push(node_prod_pips[idx][r] as f32 / MAX_NODE_PIPS);
            }

            // 3. robbed_production per resource (5) — /13 pips
            for r in 0..5 {
                out.push(node_robbed_pips[idx][r] as f32 / MAX_NODE_PIPS);
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

            // 5. road_count_cur, road_count_opp (2) — /3
            let mut my_roads = 0u8;
            let mut opp_roads = 0u8;
            for &edge_id in &node.adjacent_edges {
                let mask = 1u128 << edge_id.0;
                if cur_board.road_network.roads & mask != 0 {
                    my_roads += 1;
                }
                if opp_board.road_network.roads & mask != 0 {
                    opp_roads += 1;
                }
            }
            out.push(my_roads as f32 / 3.0);
            out.push(opp_roads as f32 / 3.0);

            // 6. adjacent_buildings_cur, adjacent_buildings_opp (2) — /3
            let adj_mask = topo.adj.node_adj_nodes[idx];
            let adj_cur = (cur_buildings & adj_mask).count_ones();
            let adj_opp = (opp_buildings & adj_mask).count_ones();
            out.push(adj_cur as f32 / 3.0);
            out.push(adj_opp as f32 / 3.0);

            // 7. hop_distance_cur, hop_distance_opp (2) — /5 capped
            out.push(hop_dist_cur[idx] as f32 / 5.0);
            out.push(hop_dist_opp[idx] as f32 / 5.0);
        }

        // === Edge stream (72 x 2 = 144) ===
        for i in 0..72u8 {
            let endpoints = topo.adj.edge_endpoints[i as usize];

            // best_endpoint_production (1) — /13 pips
            // Endpoints with any building or adjacent_buildings > 0 are zeroed.
            let mut best_prod = 0u8;
            let mut node_idx = endpoints;
            while node_idx != 0 {
                let n = node_idx.trailing_zeros() as usize;
                node_idx &= node_idx - 1;
                let has_building = occupied & (1u64 << n) != 0;
                let has_adj_buildings = occupied & topo.adj.node_adj_nodes[n] != 0;
                if !has_building && !has_adj_buildings {
                    best_prod = best_prod.max(node_total_pips[n]);
                }
            }
            out.push(best_prod as f32 / MAX_NODE_PIPS);

            // best_endpoint_hop_distance (1) — /5 capped
            // Min hop_distance_cur of the two endpoints.
            let mut min_hop = 5u8;
            let mut node_idx = endpoints;
            while node_idx != 0 {
                let n = node_idx.trailing_zeros() as usize;
                node_idx &= node_idx - 1;
                min_hop = min_hop.min(hop_dist_cur[n]);
            }
            out.push(min_hop as f32 / 5.0);
        }

        debug_assert_eq!(
            out.len(),
            Self::FEATURE_SIZE,
            "feature vector length mismatch: expected {}, got {}",
            Self::FEATURE_SIZE,
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
        RichNodeEncoder::encode(&state, &mut features);
        assert_eq!(features.len(), RichNodeEncoder::FEATURE_SIZE);
    }

    #[test]
    fn feature_vector_length_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);
        assert_eq!(features.len(), RichNodeEncoder::FEATURE_SIZE);
    }

    #[test]
    fn values_in_range() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);
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
        RichNodeEncoder::encode(&state, &mut features);
        let p1_features = features.clone();
        state.current_player = state.current_player.opponent();
        RichNodeEncoder::encode(&state, &mut features);
        assert_ne!(p1_features, features);
        assert_eq!(p1_features.len(), features.len());
    }

    #[test]
    fn hop_distance_no_buildings() {
        // Before setup: no buildings exist, hop distances should all be capped at 5.
        let state = make_state();
        let topo = &state.topology;
        let buildings = 0u64;
        let dist = super::super::compute_hop_distances(topo, buildings, 5);
        assert!(dist.iter().all(|&d| d == 5));
    }

    #[test]
    fn hop_distance_single_building() {
        let state = make_state();
        let topo = &state.topology;
        let node = 0u8;
        let buildings = 1u64 << node;
        let dist = super::super::compute_hop_distances(topo, buildings, 5);

        // Source node has distance 0.
        assert_eq!(dist[node as usize], 0);
        // Its neighbors have distance 1.
        for &adj in &topo.nodes[node as usize].adjacent_nodes {
            assert_eq!(dist[adj.0 as usize], 1);
        }
        // All distances are in [0, 5].
        assert!(dist.iter().all(|&d| d <= 5));
    }

    #[test]
    fn values_in_range_before_setup() {
        // Encoding with no buildings (initial placement phase) must still
        // produce values in [0, 1] — hop_distance returns cap, not 0.
        let state = make_state();
        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);
        for (i, &v) in features.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "feature {i} out of [0.0, 1.0]: {v}"
            );
        }
    }

    // ── Feature offset helpers ───────────────────────────────────────

    const NODE_OFF: usize = 49;
    const EDGE_OFF: usize = 49 + 54 * 24; // 1345

    /// Feature index for node `n`, feature `f` within the 24-feature block.
    fn node_feat(n: usize, f: usize) -> usize {
        NODE_OFF + n * 24 + f
    }

    /// Feature index for edge `e`, feature `f` within the 2-feature block.
    fn edge_feat(e: usize, f: usize) -> usize {
        EDGE_OFF + e * 2 + f
    }

    /// Set up a state in Main phase with buildings placed (bypassing setup).
    fn make_main_state() -> GameState {
        let mut state = make_state();
        // Place buildings directly, skip to Main phase.
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
        RichNodeEncoder::encode(&state, &mut features);

        // For each node, manually compute expected production pips per resource
        // and verify against the encoded features.
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
                let feat_idx = node_feat(n, 2 + r); // prod starts at offset 2
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

        // Find a non-desert tile with production to put the robber on.
        let robber_tile = topo
            .tiles
            .iter()
            .find(|t| t.terrain.resource().is_some())
            .unwrap()
            .id;
        state.robber = robber_tile;

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        // Nodes adjacent to the robber tile should have nonzero robbed_production.
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

        // Nodes NOT adjacent to robber tile should have all robbed_production = 0
        // (assuming robber is only on one tile).
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
        RichNodeEncoder::encode(&state, &mut features);

        for n in 0..54 {
            let node = &topo.nodes[n];
            let port_start = node_feat(n, 12); // port one-hot starts at offset 12
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

        // Place a settlement and a city for P1, a settlement for P2.
        let s_node = 10usize;
        let c_node = 20usize;
        let opp_node = 30usize;
        state.boards[Player::One].settlements = 1u64 << s_node;
        state.boards[Player::One].cities = 1u64 << c_node;
        state.boards[Player::Two].settlements = 1u64 << opp_node;

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        // building_cur = 0.5 for settlement, 1.0 for city
        assert_eq!(features[node_feat(s_node, 0)], 0.5, "settlement = 0.5");
        assert_eq!(features[node_feat(c_node, 0)], 1.0, "city = 1.0");
        assert_eq!(
            features[node_feat(opp_node, 0)],
            0.0,
            "opp building in cur slot"
        );
        // building_opp
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

        // Empty node
        let empty = 0usize;
        assert_eq!(features[node_feat(empty, 0)], 0.0);
        assert_eq!(features[node_feat(empty, 1)], 0.0);
    }

    // ── Road count features ──────────────────────────────────────────

    #[test]
    fn road_count_features() {
        let mut state = make_main_state();
        let topo = &state.topology;

        // Pick a node and place roads on some of its adjacent edges.
        let n = 5usize;
        let node = &topo.nodes[n];
        let num_adj_edges = node.adjacent_edges.len();

        // Place roads for current player on all adjacent edges of node n.
        let mut road_mask = 0u128;
        for &eid in &node.adjacent_edges {
            road_mask |= 1u128 << eid.0;
        }
        state.boards[Player::One].road_network.roads = road_mask;

        // Place one road for opponent on an adjacent edge too.
        let opp_edge = node.adjacent_edges[0];
        state.boards[Player::Two].road_network.roads = 1u128 << opp_edge.0;

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        let road_cur = features[node_feat(n, 18)];
        let road_opp = features[node_feat(n, 19)];
        assert_eq!(
            road_cur,
            num_adj_edges as f32 / 3.0,
            "cur road count for node {n}"
        );
        assert_eq!(road_opp, 1.0 / 3.0, "opp road count for node {n}");
    }

    // ── Adjacent building features ───────────────────────────────────

    #[test]
    fn adjacent_building_features() {
        let mut state = make_main_state();
        let topo = &state.topology;

        // Place a building for P1 at node 0.
        state.boards[Player::One].settlements = 1u64 << 0;

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        // All neighbors of node 0 should have adj_buildings_cur = 1/3.
        for &adj in &topo.nodes[0].adjacent_nodes {
            let adj_cur = features[node_feat(adj.0 as usize, 20)];
            assert_eq!(
                adj_cur,
                1.0 / 3.0,
                "node {} adj_cur should be 1/3 (neighbor of node 0)",
                adj.0
            );
        }

        // Node 0 itself should have adj_buildings_cur = 0 (no buildings adjacent
        // to itself, only at itself).
        assert_eq!(
            features[node_feat(0, 20)],
            0.0,
            "node 0 adj_cur should be 0 (own building doesn't count)"
        );
    }

    // ── Hop distance features ────────────────────────────────────────

    #[test]
    fn hop_distance_features_match_bfs() {
        let mut state = make_main_state();
        let topo = &state.topology;

        // Place buildings at specific nodes.
        let cur_buildings = (1u64 << 0) | (1u64 << 20);
        let opp_buildings = 1u64 << 40;
        state.boards[Player::One].settlements = cur_buildings;
        state.boards[Player::Two].settlements = opp_buildings;

        let expected_cur = super::super::compute_hop_distances(topo, cur_buildings, 5);
        let expected_opp = super::super::compute_hop_distances(topo, opp_buildings, 5);

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        for n in 0..54 {
            let hop_cur = features[node_feat(n, 22)];
            let hop_opp = features[node_feat(n, 23)];
            assert_eq!(
                hop_cur,
                expected_cur[n] as f32 / 5.0,
                "node {n} hop_cur mismatch"
            );
            assert_eq!(
                hop_opp,
                expected_opp[n] as f32 / 5.0,
                "node {n} hop_opp mismatch"
            );
        }
    }

    #[test]
    fn hop_distance_multi_source() {
        // Two buildings: BFS should find shortest path to either.
        let state = make_state();
        let topo = &state.topology;

        // Pick two distant nodes.
        let buildings = (1u64 << 0) | (1u64 << 53);
        let dist = super::super::compute_hop_distances(topo, buildings, 5);

        assert_eq!(dist[0], 0);
        assert_eq!(dist[53], 0);
        // Every node should be closer than or equal to its single-source distance.
        let dist_from_0 = super::super::compute_hop_distances(topo, 1u64 << 0, 5);
        let dist_from_53 = super::super::compute_hop_distances(topo, 1u64 << 53, 5);
        for n in 0..54 {
            assert_eq!(dist[n], dist_from_0[n].min(dist_from_53[n]));
        }
    }

    // ── Edge features ────────────────────────────────────────────────

    #[test]
    fn edge_best_endpoint_production_zeroed_when_occupied() {
        let mut state = make_main_state();
        let topo = &state.topology;

        // Find an edge and place a building on one endpoint.
        let e = 0usize;
        let endpoints = topo.adj.edge_endpoints[e];
        let endpoint_node = endpoints.trailing_zeros() as usize;
        state.boards[Player::One].settlements = 1u64 << endpoint_node;

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        // Check the other endpoint: if it has adj buildings (our settlement is
        // adjacent via the edge), it should also be zeroed.
        let occupied = 1u64 << endpoint_node;

        // The best_prod should only count endpoints that are not occupied
        // AND have no adjacent buildings.
        let mut expected_best = 0u8;
        let mut ep = endpoints;
        while ep != 0 {
            let n = ep.trailing_zeros() as usize;
            ep &= ep - 1;
            let has_building = occupied & (1u64 << n) != 0;
            let has_adj = occupied & topo.adj.node_adj_nodes[n] != 0;
            if !has_building && !has_adj {
                let mut total_pips = 0u8;
                for &tid in &topo.nodes[n].adjacent_tiles {
                    let tile = &topo.tiles[tid.0 as usize];
                    if tile.terrain.resource().is_some() {
                        for roll in 2..=12u8 {
                            if topo.dice_to_tiles[roll as usize].contains(&tid) {
                                total_pips += PIPS[roll as usize];
                                break;
                            }
                        }
                    }
                }
                expected_best = expected_best.max(total_pips);
            }
        }

        assert_eq!(
            features[edge_feat(e, 0)],
            expected_best as f32 / MAX_NODE_PIPS,
            "edge {e} best_endpoint_production"
        );
    }

    #[test]
    fn edge_hop_distance_uses_current_player() {
        let mut state = make_main_state();
        let topo = &state.topology;

        // Place a building for current player at node 0.
        state.boards[Player::One].settlements = 1u64 << 0;
        let hop_dist = super::super::compute_hop_distances(topo, 1u64 << 0, 5);

        let mut features = Vec::new();
        RichNodeEncoder::encode(&state, &mut features);

        // For each edge, best_endpoint_hop_distance = min hop_cur of endpoints.
        for e in 0..72 {
            let mut min_hop = 5u8;
            let mut ep = topo.adj.edge_endpoints[e];
            while ep != 0 {
                let n = ep.trailing_zeros() as usize;
                ep &= ep - 1;
                min_hop = min_hop.min(hop_dist[n]);
            }
            assert_eq!(
                features[edge_feat(e, 1)],
                min_hop as f32 / 5.0,
                "edge {e} best_endpoint_hop_distance"
            );
        }
    }

    // ── Perspective swap ─────────────────────────────────────────────

    #[test]
    fn perspective_swap_flips_cur_opp() {
        let mut state = make_main_state();

        // Place buildings for both players.
        let cur_node = 10usize;
        let opp_node = 20usize;
        state.boards[Player::One].settlements = 1u64 << cur_node;
        state.boards[Player::Two].settlements = 1u64 << opp_node;

        let mut f1 = Vec::new();
        RichNodeEncoder::encode(&state, &mut f1);

        state.current_player = Player::Two;
        let mut f2 = Vec::new();
        RichNodeEncoder::encode(&state, &mut f2);

        // P1's building_cur should become P2's building_opp and vice versa.
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

        // Hop distances should swap too.
        assert_eq!(
            f1[node_feat(cur_node, 22)],
            0.0,
            "P1 view: hop_cur at own building"
        );
        assert_eq!(
            f2[node_feat(cur_node, 23)],
            0.0,
            "P2 view: hop_opp at P1's building"
        );
    }
}
