//! # NexusEncoderV1 (1445 features)
//!
//! Original heterogeneous encoder with tiles and nodes (no edges).
//!
//! ## Global (121 = 7 + 51×2 + 12) — same as current
//!
//! ## Tiles (19 × 10 = 190) — same as current
//!
//! ## Nodes (54 × 21 = 1134)
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
//!
//! No edges, no settle_legal, no longest_road_nodes.

use canopy::nn::StateEncoder;

use crate::game::state::GameState;

use super::{
    ORIGINAL_DECK, PIPS, compute_network_distances, encode_dice, encode_node_blocked_production,
    encode_node_production, encode_per_number_production, encode_per_resource_production,
    encode_phase, encode_port_ratios, encode_tile_building_weights, node_value,
    opponent_expected_dev_cards, roll_probabilities, self_dev_cards_playable, tile_numbers,
};

pub struct NexusEncoderV1;

#[allow(dead_code)]
impl NexusEncoderV1 {
    pub const FEATURE_SIZE: usize = 1445;
    pub const GLOBAL_LEN: usize = 121;
    pub const TILES_F: usize = 10;
    pub const NODES_F: usize = 21;
}

/// Push 51 per-player features grouped by category (same as current).
fn encode_player_nexus_v1(
    state: &GameState,
    player: canopy::player::Player,
    tile_numbers: &[u8; 19],
    out: &mut Vec<f32>,
) {
    let p = &state.players[player];
    let is_cur = player == state.current_player;

    // ── Economy (26) ─────────────────────────────────────────────────

    for &r in &crate::game::resource::ALL_RESOURCES {
        out.push(p.hand[r] as f32 / 19.0);
    }

    for &ratio in &p.trade_ratios {
        out.push((4 - ratio.min(4)) as f32 / 4.0);
    }

    encode_per_resource_production(&state.boards[player], &state.topology, tile_numbers, out);
    encode_per_number_production(&state.boards[player], &state.topology, tile_numbers, out);

    // ── Board (9) ────────────────────────────────────────────────────

    out.push(p.settlements_left as f32 / 5.0);
    out.push(p.cities_left as f32 / 4.0);
    out.push(p.roads_left as f32 / 15.0);

    let has_lr = state.longest_road.is_some_and(|(pid, _)| pid == player);
    out.push(f32::from(has_lr));
    out.push(state.boards[player].road_network.longest_road() as f32 / 15.0);

    let has_la = state.largest_army.is_some_and(|(pid, _)| pid == player);
    out.push(f32::from(has_la));

    let vps = if is_cur {
        state.total_vps(player)
    } else {
        state.public_vps(player)
    };
    out.push(vps as f32 / 15.0);

    let vp_remaining = state.vp_limit.saturating_sub(vps);
    out.push(vp_remaining as f32 / state.vp_limit as f32);

    let hand_total = p.hand.total();
    let over = hand_total.saturating_sub(state.discard_threshold);
    out.push((over as f32 / 19.0).min(1.0));

    // ── Dev cards (16) ───────────────────────────────────────────────

    let playable = if is_cur {
        self_dev_cards_playable(state, player)
    } else {
        opponent_expected_dev_cards(state, player.opponent(), player)
    };
    out.extend_from_slice(&playable);

    for (count, max) in p.dev_cards_played.0.iter().zip(&ORIGINAL_DECK) {
        out.push(*count as f32 / max);
    }

    for (i, max) in ORIGINAL_DECK.iter().enumerate() {
        if is_cur {
            out.push(p.dev_cards_bought_this_turn.0[i] as f32 / max);
        } else {
            out.push(0.0);
        }
    }

    out.push(f32::from(p.has_played_dev_card_this_turn));
}

impl StateEncoder<GameState> for NexusEncoderV1 {
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

        encode_phase(state, out);
        encode_player_nexus_v1(state, current, &tile_numbers, out);
        encode_player_nexus_v1(state, opp, &tile_numbers, out);
        encode_dice(state, out);

        debug_assert_eq!(out.len(), Self::GLOBAL_LEN);

        // === Tiles (19 × 10 = 190) ===
        let cur_boards = &state.boards[current];
        let opp_boards = &state.boards[opp];
        let roll_probs = roll_probabilities(state);

        for (i, tile) in topo.tiles.iter().enumerate() {
            let resource_idx = tile.terrain.resource().map(|r| r as usize);
            for ri in 0..5 {
                out.push(f32::from(resource_idx == Some(ri)));
            }
            let number = tile_numbers[i];
            let pips = if number > 0 {
                PIPS[number as usize] as f32 / 5.0
            } else {
                0.0
            };
            out.push(pips);
            let roll_prob = if number > 0 {
                roll_probs[(number - 2) as usize]
            } else {
                0.0
            };
            out.push(roll_prob);
            out.push(f32::from(state.robber == tile.id));
            encode_tile_building_weights(tile, cur_boards, opp_boards, out);
        }

        // === Nodes (54 × 21 = 1134) ===
        let cur_dist = compute_network_distances(&topo.adj, cur_boards);
        let opp_dist = compute_network_distances(&topo.adj, opp_boards);

        for i in 0..54u8 {
            let node = &topo.nodes[i as usize];

            // cur_building, opp_building (2)
            out.push(node_value(cur_boards, i));
            out.push(node_value(opp_boards, i));

            // port_ratio (5)
            encode_port_ratios(node, out);

            // resource_prod (5)
            encode_node_production(node, topo, &tile_numbers, state.robber, out);

            // blocked_prod (5)
            encode_node_blocked_production(node, topo, &tile_numbers, state.robber, out);

            // cur_road_count, opp_road_count (2)
            let adj_edges = topo.adj.node_adj_edges[i as usize];
            out.push((adj_edges & cur_boards.road_network.roads).count_ones() as f32 / 3.0);
            out.push((adj_edges & opp_boards.road_network.roads).count_ones() as f32 / 3.0);

            // cur_network_dist, opp_network_dist (2)
            out.push(cur_dist[i as usize]);
            out.push(opp_dist[i as usize]);
        }

        // No edges in V1.

        debug_assert_eq!(
            out.len(),
            self.feature_size(),
            "feature vector length mismatch: expected {}, got {}",
            self.feature_size(),
            out.len()
        );
    }
}
