use canopy2::nn::StateEncoder;

use crate::game::{PigGame, Player};

pub struct PigEncoder;

impl StateEncoder<PigGame> for PigEncoder {
    const FEATURE_SIZE: usize = 3;

    fn encode(state: &PigGame, out: &mut Vec<f32>) {
        out.clear();
        let scores = state.scores();
        let (my, opp) = match state.current_player() {
            Player::One => (scores[0], scores[1]),
            Player::Two => (scores[1], scores[0]),
        };
        out.push(my as f32 / 100.0);
        out.push(opp as f32 / 100.0);
        out.push(state.turn_total() as f32 / 100.0);

        debug_assert_eq!(out.len(), Self::FEATURE_SIZE);
    }
}
