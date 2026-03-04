use canopy2::nn::StateEncoder;
use canopy2::player::Player;

use crate::game::resource::ALL_RESOURCES;
use crate::game::state::{GameState, Phase};

/// Dice probability for each sum (indices 2..=12, 0 and 1 unused).
const DICE_PROB: [f32; 13] = [
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

/// Original deck composition per card type.
const ORIGINAL_DECK: [f32; 5] = [14.0, 5.0, 2.0, 2.0, 2.0];

fn b(x: bool) -> f32 {
    x as u8 as f32
}

pub struct CatanEncoder;

impl CatanEncoder {
    /// Push per-player features (21).
    ///
    /// Dev card held values are exact counts for self, or hypergeometric
    /// expected values for the opponent.
    fn encode_player(state: &GameState, player_to_encode: Player, out: &mut Vec<f32>) {
        let player = &state.players[player_to_encode];
        let is_self = player_to_encode == state.current_player;

        // Resources (5)
        for &r in &ALL_RESOURCES {
            out.push(player.hand[r] as f32 / 19.0);
        }

        // Dev cards held (5) — exact for self, expected for opponent
        let dev_cards_held = if is_self {
            Self::self_dev_cards(state, player_to_encode)
        } else {
            Self::opponent_expected_dev_cards(state, player_to_encode.opponent(), player_to_encode)
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
        out.push(b(has_lr));

        // Has largest army award (1)
        let has_la = state
            .largest_army
            .is_some_and(|(la_pid, _)| la_pid == player_to_encode);
        out.push(b(has_la));

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
    fn opponent_expected_dev_cards(
        state: &GameState,
        perspective: Player,
        opp: Player,
    ) -> [f32; 5] {
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

    /// Push building value for a single node.
    /// 0.0 = empty, 0.5 = settlement, 1.0 = city.
    fn node_value(boards: &crate::game::state::PlayerBoards, i: u8) -> f32 {
        let mask = 1u64 << i;
        if boards.cities & mask != 0 {
            1.0
        } else if boards.settlements & mask != 0 {
            0.5
        } else {
            0.0
        }
    }
}

impl StateEncoder<GameState> for CatanEncoder {
    // Global: 8
    // Per-player (x2): 21 x 2 = 42
    // Tile stream: 19 x 7 = 133  (resource one-hot 5 + dice prob 1 + robber 1)
    // Node stream: 54 x 2 = 108  (current building + opponent building)
    // Edge stream: 72 x 2 = 144  (current road + opponent road)
    // Port stream: 9 x 5 = 45
    // Total: 8 + 42 + 133 + 108 + 144 + 45 = 480
    const FEATURE_SIZE: usize = 480;

    fn encode(state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();

        // === Phase one-hot (8) ===
        let phase_idx = match &state.phase {
            Phase::PlaceSettlement => 0,
            Phase::PlaceRoad => 1,
            Phase::Roll => 2,
            Phase::Discard { .. } => 3,
            Phase::MoveRobber => 4,
            Phase::StealResolve => 5,
            Phase::Main => 6,
            Phase::RoadBuilding { .. } => 7,
            Phase::GameOver(_) => unreachable!(),
        };
        for i in 0..8 {
            out.push(b(i == phase_idx));
        }

        // === Per-player features (21 x 2 = 42) ===
        Self::encode_player(state, current, out);
        Self::encode_player(state, opp, out);

        // === Tile stream (19 x 7 = 133) ===
        let topo = &state.topology;
        const MAX_DICE_PROB: f32 = 5.0 / 36.0;
        for tile in &topo.tiles {
            // Resource one-hot (5)
            let resource_idx = tile.terrain.resource().map(|r| r as usize);
            for i in 0..5 {
                out.push(b(resource_idx == Some(i)));
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
            out.push(b(state.robber == tile.id));
        }

        // === Node stream (54 x 2 = 108) ===
        let cur_board = &state.boards[current];
        let opp_board = &state.boards[opp];
        for i in 0..54u8 {
            out.push(Self::node_value(cur_board, i));
            out.push(Self::node_value(opp_board, i));
        }

        // === Edge stream (72 x 2 = 144) ===
        for i in 0..72u8 {
            let mask = 1u128 << i;
            out.push(b(cur_board.road_network.roads & mask != 0));
            out.push(b(opp_board.road_network.roads & mask != 0));
        }

        // === Port stream (9 x 5 = 45) ===
        for &port_type in &topo.port_types {
            let resource_idx = port_type.map(|r| r as usize);
            for i in 0..5 {
                out.push(b(resource_idx == Some(i)));
            }
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
        CatanEncoder::encode(&state, &mut features);
        assert_eq!(features.len(), CatanEncoder::FEATURE_SIZE);
    }

    #[test]
    fn feature_vector_length_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        CatanEncoder::encode(&state, &mut features);
        assert_eq!(features.len(), CatanEncoder::FEATURE_SIZE);
    }

    #[test]
    fn perspective_symmetry() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        CatanEncoder::encode(&state, &mut features);
        let p1_features = features.clone();
        state.current_player = state.current_player.opponent();
        CatanEncoder::encode(&state, &mut features);
        assert_ne!(p1_features, features);
        assert_eq!(p1_features.len(), features.len());
    }

    #[test]
    fn values_in_range() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        CatanEncoder::encode(&state, &mut features);
        for (i, &v) in features.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "feature {i} out of [0.0, 1.0]: {v}"
            );
        }
    }

    #[test]
    fn opponent_expected_dev_cards_initial() {
        // In the initial state, no one has drawn any dev cards.
        // opp_hand_size = 0, so all expected values should be 0.
        let state = make_state();
        let mut features = Vec::new();
        CatanEncoder::encode(&state, &mut features);

        // Opponent dev card features start at offset:
        // 8 (phase) + 21 (self) + 5 (opp resources) = 34
        for i in 0..5 {
            assert_eq!(
                features[34 + i],
                0.0,
                "opponent expected dev card {i} should be 0.0 when they hold no cards"
            );
        }
    }
}
