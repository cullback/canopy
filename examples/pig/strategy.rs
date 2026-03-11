use canopy2::eval::{Evaluation, Evaluator};

use crate::game::{self, PigGame};

/// Hold-at-N strategy: roll until turn total reaches the threshold, then hold.
pub struct HoldAt(pub u32);

impl Evaluator<PigGame> for HoldAt {
    fn evaluate(&self, state: &PigGame, _rng: &mut fastrand::Rng) -> Evaluation {
        let hold = state.turn_total() >= self.0;
        let mut logits = vec![-10.0; game::NUM_ACTIONS];
        logits[if hold { game::HOLD } else { game::ROLL }] = 10.0;
        Evaluation {
            policy_logits: logits,
            value: 0.0,
        }
    }
}
