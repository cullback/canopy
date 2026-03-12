use canopy2::nn::StateEncoder;

use crate::game::{PigGame, Player};

pub struct PigEncoder;

impl StateEncoder<PigGame> for PigEncoder {
    fn feature_size(&self) -> usize {
        3
    }

    fn encode(&self, state: &PigGame, out: &mut Vec<f32>) {
        out.clear();
        let scores = state.scores();
        let (my, opp) = match state.current_player() {
            Player::One => (scores[0], scores[1]),
            Player::Two => (scores[1], scores[0]),
        };
        out.push(my as f32 / 100.0);
        out.push(opp as f32 / 100.0);
        out.push(state.turn_total() as f32 / 100.0);

        debug_assert_eq!(out.len(), self.feature_size());
    }
}
