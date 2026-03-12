use canopy2::nn::StateEncoder;

use crate::game::state::GameState;

use super::{encode_edges, encode_phase, encode_player, encode_ports, encode_tiles, node_value};

pub struct BasicEncoder;

impl BasicEncoder {
    pub const NODES_F: usize = 2;
    pub const EDGES_F: usize = 2;
    pub const TILES_F: usize = 7;
    pub const PORTS_F: usize = 5;
}

impl StateEncoder<GameState> for BasicEncoder {
    // Global: 7
    // Per-player (x2): 21 x 2 = 42
    // Tile stream: 19 x 7 = 133  (resource one-hot 5 + dice prob 1 + robber 1)
    // Node stream: 54 x 2 = 108  (current building + opponent building)
    // Edge stream: 72 x 2 = 144  (current road + opponent road)
    // Port stream: 9 x 5 = 45
    // Total: 7 + 42 + 133 + 108 + 144 + 45 = 479
    fn feature_size(&self) -> usize {
        479
    }

    fn encode(&self, state: &GameState, out: &mut Vec<f32>) {
        out.clear();
        let current = state.current_player;
        let opp = current.opponent();

        // === Phase one-hot (7) ===
        encode_phase(state, out);

        // === Per-player features (21 x 2 = 42) ===
        encode_player(state, current, out);
        encode_player(state, opp, out);

        // === Tile stream (19 x 7 = 133) ===
        encode_tiles(state, out);

        // === Node stream (54 x 2 = 108) ===
        let cur_board = &state.boards[current];
        let opp_board = &state.boards[opp];
        for i in 0..54u8 {
            out.push(node_value(cur_board, i));
            out.push(node_value(opp_board, i));
        }

        // === Edge stream (72 x 2 = 144) ===
        encode_edges(state, out);

        // === Port stream (9 x 5 = 45) ===
        encode_ports(state, out);

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

    fn enc() -> BasicEncoder {
        BasicEncoder
    }

    #[test]
    fn feature_vector_length() {
        let state = make_state();
        let mut features = Vec::new();
        enc().encode(&state, &mut features);
        assert_eq!(features.len(), enc().feature_size());
    }

    #[test]
    fn feature_vector_length_after_setup() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        enc().encode(&state, &mut features);
        assert_eq!(features.len(), enc().feature_size());
    }

    #[test]
    fn perspective_symmetry() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        enc().encode(&state, &mut features);
        let p1_features = features.clone();
        state.current_player = state.current_player.opponent();
        enc().encode(&state, &mut features);
        assert_ne!(p1_features, features);
        assert_eq!(p1_features.len(), features.len());
    }

    #[test]
    fn values_in_range() {
        let mut state = make_state();
        play_setup(&mut state);
        let mut features = Vec::new();
        enc().encode(&state, &mut features);
        for (i, &v) in features.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "feature {i} out of [0.0, 1.0]: {v}"
            );
        }
    }

    #[test]
    fn opponent_expected_dev_cards_initial() {
        let state = make_state();
        let mut features = Vec::new();
        enc().encode(&state, &mut features);

        // Opponent dev card features start at offset:
        // 7 (phase) + 21 (self) + 5 (opp resources) = 33
        for i in 0..5 {
            assert_eq!(
                features[33 + i],
                0.0,
                "opponent expected dev card {i} should be 0.0 when they hold no cards"
            );
        }
    }
}
