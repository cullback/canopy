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

/// "End race or keep pace" strategy (~0.9% disadvantage vs optimal).
///
/// If either player has a score of 71+, roll to win (hold at 100 - my_score).
/// Otherwise, hold at 21 + (my_score - opponent_score) / 8.
pub struct EndRaceKeepPace;

impl Evaluator<PigGame> for EndRaceKeepPace {
    fn evaluate(&self, state: &PigGame, _rng: &mut fastrand::Rng) -> Evaluation {
        let scores = state.scores();
        let i = state.current_player().index();
        let my_score = scores[i];
        let opp_score = scores[1 - i];

        let threshold = if my_score >= 71 || opp_score >= 71 {
            100u32.saturating_sub(my_score)
        } else {
            (21 + (my_score as i32 - opp_score as i32) / 8) as u32
        };

        let hold = state.turn_total() >= threshold;
        let mut logits = vec![-10.0; game::NUM_ACTIONS];
        logits[if hold { game::HOLD } else { game::ROLL }] = 10.0;
        Evaluation {
            policy_logits: logits,
            value: 0.0,
        }
    }
}
