//! # NexusEncoderV3 (1553 features)
//!
//! Simplified heterogeneous encoder — drops edge features and settle_legal.
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
//! ## Nodes (54 × 23 = 1242)
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
//! | cur_on_longest_road  |     1 | binary  |
//! | opp_on_longest_road  |     1 | binary  |

use canopy::nn::StateEncoder;

use crate::game::state::GameState;

use super::{
    ORIGINAL_DECK, PIPS, compute_network_distances, encode_dice, encode_node_blocked_production,
    encode_node_production, encode_per_number_production, encode_per_resource_production,
    encode_phase, encode_port_ratios, encode_tile_building_weights, node_value,
    opponent_expected_dev_cards, roll_probabilities, self_dev_cards_playable, tile_numbers,
};

pub struct NexusEncoderV3;

#[allow(dead_code)]
impl NexusEncoderV3 {
    pub const FEATURE_SIZE: usize = 1553;
    pub const GLOBAL_LEN: usize = 121;
    pub const TILES_F: usize = 10;
    pub const NODES_F: usize = 23;
}

/// Push 51 per-player features grouped by category.
///
/// Economy (26): resource_count(5), trade_ratio(5), resource_prod(5), number_prod(11)
/// Board (9): settlement_count(1), city_count(1), road_count(1),
///            longest_road_award(1), longest_road_length(1), largest_army_award(1), victory_points(1),
///            vp_remaining(1), cards_over_threshold(1)
/// Dev cards (16): dev_playable(5), dev_played(5), dev_bought_turn(5), dev_played_turn(1)
fn encode_player_nexus_v3(
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

impl StateEncoder<GameState> for NexusEncoderV3 {
    fn feature_size(&self) -> usize {
        Self::FEATURE_SIZE
    }

    fn encode(&self, state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();
        let topo = &state.topology;

        let tile_numbers = tile_numbers(topo);

        // === Global features (121) ===

        // Phase one-hot (7)
        encode_phase(state, out);

        // Per-player nexus (51 × 2 = 102)
        encode_player_nexus_v3(state, current, &tile_numbers, out);
        encode_player_nexus_v3(state, opp, &tile_numbers, out);

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

        // === Nodes (54 × 23 = 1242) ===
        let cur_dist = compute_network_distances(&topo.adj, cur_boards);
        let opp_dist = compute_network_distances(&topo.adj, opp_boards);

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

            // cur_on_longest_road, opp_on_longest_road (2)
            out.push(f32::from(cur_lr_nodes & bit != 0));
            out.push(f32::from(opp_lr_nodes & bit != 0));
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
