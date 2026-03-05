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
    // Global: 8
    // Per-player (x2): 21 x 2 = 42
    // Node stream: 54 x 24 = 1296
    // Edge stream: 72 x 2 = 144
    // Total: 8 + 42 + 1296 + 144 = 1490
    const FEATURE_SIZE: usize = 1490;

    fn encode(state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();

        // === Phase one-hot (8) ===
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
    use crate::game::topology::Topology;
    use canopy2::game::Game;
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
}
