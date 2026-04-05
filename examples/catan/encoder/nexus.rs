//! # NexusEncoder (1949 features)
//!
//! Heterogeneous encoder that keeps tiles, nodes, and edges as separate entity streams.
//!
//! ## Global (121 = 7 + 51×2 + 12)
//!
//! | Block              | Count |
//! |--------------------|-------|
//! | phase              |     7 |
//! | per-player × 2     |   102 |
//! | dice               |    12 |
//!
//! ### Per-player (51) — cur player first
//!
//! | Feature              | Count | Norm      |
//! |----------------------|-------|-----------|
//! | resource_count       |     5 | /19       |
//! | trade_ratio          |     5 | /4        |
//! | resource_prod        |     5 | /35       |
//! | number_prod          |    11 | /10       |
//! | settlement_count     |     1 | /5        |
//! | city_count           |     1 | /4        |
//! | road_count           |     1 | /15       |
//! | longest_road_award   |     1 | binary    |
//! | longest_road_length  |     1 | /15       |
//! | largest_army_award   |     1 | binary    |
//! | victory_points       |     1 | /15       |
//! | vp_remaining         |     1 | /vp_limit |
//! | cards_over_threshold |     1 | /19 cap 1 |
//! | dev_playable         |     5 | /deck_max |
//! | dev_played           |     5 | /deck_max |
//! | dev_bought_turn      |     5 | /deck_max |
//! | dev_played_turn      |     1 | binary    |
//!
//! ## Tiles (19 × 10 = 190)
//!
//! | Feature              | Count | Norm    |
//! |----------------------|-------|---------|
//! | resource             |     5 | one-hot |
//! | pips                 |     1 | /5      |
//! | roll_prob            |     1 | raw     |
//! | robber               |     1 | binary  |
//! | cur_building_weight  |     1 | /6      |
//! | opp_building_weight  |     1 | /6      |
//!
//! ## Nodes (54 × 25 = 1350)
//!
//! | Feature              | Count | Norm    |
//! |----------------------|-------|---------|
//! | cur_building         |     1 | 0/½/1   |
//! | opp_building         |     1 | 0/½/1   |
//! | port_ratio           |     5 | .5/.25  |
//! | resource_prod        |     5 | /13     |
//! | blocked_prod         |     5 | /5      |
//! | cur_road_count       |     1 | /3      |
//! | opp_road_count       |     1 | /3      |
//! | cur_network_dist     |     1 | /6      |
//! | opp_network_dist     |     1 | /6      |
//! | cur_settle_legal     |     1 | binary  |
//! | opp_settle_legal     |     1 | binary  |
//! | cur_on_longest_road  |     1 | binary  |
//! | opp_on_longest_road  |     1 | binary  |
//!
//! ## Edges (72 × 4 = 288)
//!
//! | Feature              | Count | Norm    |
//! |----------------------|-------|---------|
//! | cur_road             |     1 | binary  |
//! | opp_road             |     1 | binary  |
//! | cur_frontier         |     1 | binary  |
//! | opp_frontier         |     1 | binary  |

use canopy::nn::StateEncoder;

use crate::game::state::GameState;

use super::{
    ORIGINAL_DECK, PIPS, compute_network_distances, encode_dice, encode_node_blocked_production,
    encode_node_production, encode_per_number_production, encode_per_resource_production,
    encode_phase, encode_port_ratios, encode_tile_building_weights, node_value,
    opponent_expected_dev_cards, roll_probabilities, self_dev_cards_playable, tile_numbers,
};

pub struct NexusEncoder;

#[allow(dead_code)]
impl NexusEncoder {
    pub const FEATURE_SIZE: usize = 1949;
    pub const GLOBAL_LEN: usize = 121;
    pub const TILES_F: usize = 10;
    pub const NODES_F: usize = 25;
    pub const EDGES_F: usize = 4;
}

/// Push 51 per-player features grouped by category.
///
/// Economy (26): resource_count(5), trade_ratio(5), resource_prod(5), number_prod(11)
/// Board (9): settlement_count(1), city_count(1), road_count(1),
///            longest_road_award(1), longest_road_length(1), largest_army_award(1), victory_points(1),
///            vp_remaining(1), cards_over_threshold(1)
/// Dev cards (16): dev_playable(5), dev_played(5), dev_bought_turn(5), dev_played_turn(1)
fn encode_player_nexus(
    state: &GameState,
    player: canopy::player::Player,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    let p = &state.players[player];
    let is_cur = player == state.current_player;

    // ── Economy (26) ─────────────────────────────────────────────────

    // resource_count (5): per resource, /19
    for &r in &crate::game::resource::ALL_RESOURCES {
        out.push(p.hand[r] as f32 / 19.0);
    }

    // trade_ratio (5): (4 − ratio) / 4, higher = better
    for &ratio in &p.trade_ratios {
        out.push((4 - ratio.min(4)) as f32 / 4.0);
    }

    // resource_prod (5): building_wt × pips per resource, /35
    encode_per_resource_production(&state.boards[player], &state.topology, tile_numbers, out);

    // number_prod (11): building_wt per dice value 2-12, /10
    encode_per_number_production(&state.boards[player], &state.topology, tile_numbers, out);

    // ── Board (7) ────────────────────────────────────────────────────

    // settlement_count (1): remaining, /5
    out.push(p.settlements_left as f32 / 5.0);

    // city_count (1): remaining, /4
    out.push(p.cities_left as f32 / 4.0);

    // road_count (1): remaining, /15
    out.push(p.roads_left as f32 / 15.0);

    // longest_road_award (1): binary
    let has_lr = state.longest_road.is_some_and(|(pid, _)| pid == player);
    out.push(f32::from(has_lr));

    // longest_road_length (1): /15
    out.push(state.boards[player].road_network.longest_road() as f32 / 15.0);

    // largest_army_award (1): binary
    let has_la = state.largest_army.is_some_and(|(pid, _)| pid == player);
    out.push(f32::from(has_la));

    // victory_points (1): /15, total for cur, public for opp
    let vps = if is_cur {
        state.total_vps(player)
    } else {
        state.public_vps(player)
    };
    out.push(vps as f32 / 15.0);

    // vp_remaining (1): (vp_limit − vps) / vp_limit
    let vp_remaining = state.vp_limit.saturating_sub(vps);
    out.push(vp_remaining as f32 / state.vp_limit as f32);

    // cards_over_threshold (1): min(1, max(0, hand_total − discard_threshold) / 19)
    let hand_total = p.hand.total();
    let over = hand_total.saturating_sub(state.discard_threshold);
    out.push((over as f32 / 19.0).min(1.0));

    // ── Dev cards (16) ───────────────────────────────────────────────

    // dev_playable (5): per type, /deck_max, exact for cur / hypergeo for opp
    let playable = if is_cur {
        self_dev_cards_playable(state, player)
    } else {
        opponent_expected_dev_cards(state, player.opponent(), player)
    };
    out.extend_from_slice(&playable);

    // dev_played (5): per type, /deck_max, visible both
    for (count, max) in p.dev_cards_played.0.iter().zip(&ORIGINAL_DECK) {
        out.push(*count as f32 / max);
    }

    // dev_bought_turn (5): per type, /deck_max, exact for cur / 0 for opp
    for (i, max) in ORIGINAL_DECK.iter().enumerate() {
        if is_cur {
            out.push(p.dev_cards_bought_this_turn.0[i] as f32 / max);
        } else {
            out.push(0.0);
        }
    }

    // dev_played_turn (1): binary, visible both
    out.push(f32::from(p.has_played_dev_card_this_turn));
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

        // === Global features (117) ===

        // Phase one-hot (7)
        encode_phase(state, out);

        // Per-player nexus (51 × 2 = 102)
        encode_player_nexus(state, current, &tile_numbers, out);
        encode_player_nexus(state, opp, &tile_numbers, out);

        // Dice state (12)
        encode_dice(state, out);

        debug_assert_eq!(out.len(), Self::GLOBAL_LEN);

        // === Tiles (19 × 10 = 190) ===
        let cur_boards = &state.boards[current];
        let opp_boards = &state.boards[opp];
        let roll_probs = roll_probabilities(state);

        for (i, tile) in topo.tiles.iter().enumerate() {
            // resource (5): one-hot
            let resource_idx = tile.terrain.resource().map(|r| r as usize);
            for ri in 0..5 {
                out.push(f32::from(resource_idx == Some(ri)));
            }
            // pips (1): /5
            let number = tile_numbers[i];
            let pips = if number > 0 {
                PIPS[number as usize] as f32 / 5.0
            } else {
                0.0
            };
            out.push(pips);
            // roll_prob (1): current probability of this tile's number (raw)
            let roll_prob = if number > 0 {
                roll_probs[(number - 2) as usize]
            } else {
                0.0
            };
            out.push(roll_prob);
            // robber (1): binary
            out.push(f32::from(state.robber == tile.id));
            // cur_building_weight, opp_building_weight (2): /6
            encode_tile_building_weights(tile, cur_boards, opp_boards, out);
        }

        // === Nodes (54 × 25 = 1350) ===
        let cur_dist = compute_network_distances(&topo.adj, cur_boards);
        let opp_dist = compute_network_distances(&topo.adj, opp_boards);

        // Precompute settlement legality bitmasks
        let occupied = state.occupied_nodes();
        let mut neighbor_blocked = 0u64;
        {
            let mut bits = occupied;
            while bits != 0 {
                let node = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                neighbor_blocked |= topo.adj.node_adj_nodes[node];
            }
        }
        let node_mask: u64 = (1u64 << 54) - 1;
        let base_legal = !occupied & !neighbor_blocked & node_mask;

        // cur_settle_legal: on current player's road network
        let mut cur_on_road = 0u64;
        {
            let mut roads = cur_boards.road_network.roads;
            while roads != 0 {
                let eid = roads.trailing_zeros() as usize;
                roads &= roads - 1;
                cur_on_road |= topo.adj.edge_endpoints[eid];
            }
        }
        let cur_settle_legal = base_legal & cur_on_road;

        // opp_settle_legal: on opponent's road network
        let mut opp_on_road = 0u64;
        {
            let mut roads = opp_boards.road_network.roads;
            while roads != 0 {
                let eid = roads.trailing_zeros() as usize;
                roads &= roads - 1;
                opp_on_road |= topo.adj.edge_endpoints[eid];
            }
        }
        let opp_settle_legal = base_legal & opp_on_road;

        // Precompute longest road node bitmasks
        let opp_buildings = opp_boards.settlements | opp_boards.cities;
        let cur_buildings = cur_boards.settlements | cur_boards.cities;
        let cur_lr_nodes = cur_boards
            .road_network
            .longest_road_nodes(&topo.adj, opp_buildings);
        let opp_lr_nodes = opp_boards
            .road_network
            .longest_road_nodes(&topo.adj, cur_buildings);

        for i in 0..54u8 {
            let node = &topo.nodes[i as usize];
            let bit = 1u64 << i;

            // cur_building, opp_building (2)
            out.push(node_value(cur_boards, i));
            out.push(node_value(opp_boards, i));

            // port_ratio (5)
            encode_port_ratios(node, out);

            // resource_prod (5)
            encode_node_production(node, topo, &tile_numbers, state.robber, out);

            // blocked_prod (5)
            encode_node_blocked_production(node, topo, &tile_numbers, state.robber, out);

            // cur_road_count, opp_road_count (2): incident roads at this node, /3
            let adj_edges = topo.adj.node_adj_edges[i as usize];
            out.push((adj_edges & cur_boards.road_network.roads).count_ones() as f32 / 3.0);
            out.push((adj_edges & opp_boards.road_network.roads).count_ones() as f32 / 3.0);

            // cur_network_dist, opp_network_dist (2)
            out.push(cur_dist[i as usize]);
            out.push(opp_dist[i as usize]);

            // cur_settle_legal, opp_settle_legal (2)
            out.push(f32::from(cur_settle_legal & bit != 0));
            out.push(f32::from(opp_settle_legal & bit != 0));

            // cur_on_longest_road, opp_on_longest_road (2)
            out.push(f32::from(cur_lr_nodes & bit != 0));
            out.push(f32::from(opp_lr_nodes & bit != 0));
        }

        // === Edges (72 × 4 = 288) ===
        let cur_roads = cur_boards.road_network.roads;
        let opp_roads = opp_boards.road_network.roads;
        let cur_frontier = cur_boards.road_network.reachable_edges();
        let opp_frontier = opp_boards.road_network.reachable_edges();

        for e in 0..72u8 {
            let edge_bit = 1u128 << e;
            out.push(f32::from(cur_roads & edge_bit != 0));
            out.push(f32::from(opp_roads & edge_bit != 0));
            out.push(f32::from(cur_frontier & edge_bit != 0));
            out.push(f32::from(opp_frontier & edge_bit != 0));
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
        let deck = DevCardDeck::new();
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
    const GLOBAL_OFF: usize = 121;

    fn tile_feat(t: usize, f: usize) -> usize {
        GLOBAL_OFF + t * 10 + f
    }

    fn node_feat(n: usize, f: usize) -> usize {
        GLOBAL_OFF + 19 * 10 + n * 25 + f
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
            // Robber (offset 7, after roll_prob)
            assert_eq!(
                features[tile_feat(i, 7)],
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
    fn network_distance_features() {
        let mut state = make_main_state();

        // Place a settlement for Player::One at node 10
        state.boards[Player::One].settlements = 1u64 << 10;

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        // Network distance at per-node offset 19-20 (2 + 5 + 5 + 5 + 2 = 19)
        // Node 10 is on-network → own_dist = 0.0
        assert_eq!(
            features[node_feat(10, 19)],
            0.0,
            "on-network node should have dist 0.0"
        );

        // Player::Two has no buildings → all opp distances = 1.0 (no network)
        assert_eq!(
            features[node_feat(10, 20)],
            1.0,
            "no opp network → dist 1.0"
        );

        // A node adjacent to node 10 should have own_dist = 1/6
        let topo = &state.topology;
        let adj_node = topo.nodes[10].adjacent_nodes[0].0 as usize;
        assert_eq!(
            features[node_feat(adj_node, 19)],
            1.0 / 6.0,
            "adjacent to network → dist 1/6"
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

        // Global: phase(7) + cur player starts at 7
        // Inside per-player: resource_count(5) then trade_ratio at offset 5
        let cur_trade = 7 + 5;
        assert_eq!(features[cur_trade + 0], 0.5, "lumber 2:1 → 0.5");
        assert_eq!(features[cur_trade + 1], 0.25, "brick 3:1 → 0.25");
        assert_eq!(features[cur_trade + 2], 0.25, "wool 3:1 → 0.25");
        assert_eq!(features[cur_trade + 3], 0.0, "grain 4:1 → 0.0");
        assert_eq!(features[cur_trade + 4], 0.0, "ore 4:1 → 0.0");

        // Opp per-player starts at 7 + 51 = 58, trade_ratio at 58 + 5 = 63
        let opp_trade = 58 + 5;
        for i in 0..5 {
            assert_eq!(features[opp_trade + i], 0.0, "opp all 4:1 → 0.0");
        }
    }

    fn edge_feat(e: usize, f: usize) -> usize {
        GLOBAL_OFF + 19 * 10 + 54 * 25 + e * 4 + f
    }

    #[test]
    fn new_node_features_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        // cur_settle_legal (offset 21): right after setup, all road endpoints
        // are either occupied (settlements) or neighbor-blocked (distance rule).
        // Feature is correctly all-zero — it only lights up once roads extend
        // beyond the distance-rule exclusion zone.
        let cur_legal_count: usize = (0..54)
            .filter(|&n| features[node_feat(n, 21)] > 0.0)
            .count();
        let opp_legal_count: usize = (0..54)
            .filter(|&n| features[node_feat(n, 22)] > 0.0)
            .count();
        assert_eq!(
            cur_legal_count, 0,
            "no legal spots right after setup (distance rule)"
        );
        assert_eq!(
            opp_legal_count, 0,
            "no legal spots right after setup (distance rule)"
        );

        // cur_on_longest_road (offset 23): all zero with < 5 roads
        for n in 0..54 {
            assert_eq!(
                features[node_feat(n, 23)],
                0.0,
                "node {n}: lr should be 0 with < 5 roads"
            );
            assert_eq!(
                features[node_feat(n, 24)],
                0.0,
                "node {n}: opp lr should be 0"
            );
        }
    }

    #[test]
    fn edge_features_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);

        let mut features = Vec::new();
        NexusEncoder.encode(&state, &mut features);

        let mut cur_roads = 0;
        let mut opp_roads = 0;
        let mut cur_frontier = 0;
        let mut opp_frontier = 0;

        for e in 0..72 {
            let cr = features[edge_feat(e, 0)];
            let or = features[edge_feat(e, 1)];
            let cf = features[edge_feat(e, 2)];
            let of = features[edge_feat(e, 3)];

            assert!(cr == 0.0 || cr == 1.0, "edge {e} cur_road not binary: {cr}");
            assert!(or == 0.0 || or == 1.0, "edge {e} opp_road not binary: {or}");
            assert!(
                cf == 0.0 || cf == 1.0,
                "edge {e} cur_frontier not binary: {cf}"
            );
            assert!(
                of == 0.0 || of == 1.0,
                "edge {e} opp_frontier not binary: {of}"
            );

            // Road and frontier are mutually exclusive per player
            assert!(!(cr == 1.0 && cf == 1.0), "edge {e}: cur road AND frontier");
            assert!(!(or == 1.0 && of == 1.0), "edge {e}: opp road AND frontier");

            if cr == 1.0 {
                cur_roads += 1;
            }
            if or == 1.0 {
                opp_roads += 1;
            }
            if cf == 1.0 {
                cur_frontier += 1;
            }
            if of == 1.0 {
                opp_frontier += 1;
            }
        }

        assert_eq!(cur_roads, 2, "cur should have 2 roads after setup");
        assert_eq!(opp_roads, 2, "opp should have 2 roads after setup");
        assert!(cur_frontier > 0, "cur should have frontier edges");
        assert!(opp_frontier > 0, "opp should have frontier edges");
    }
}
