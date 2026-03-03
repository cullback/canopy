use canopy2::eval::{Evaluator, NnOutput};
use canopy2::game::Game;
use canopy2::player::Player;

use crate::game::action::{
    self, BUY_DEV_CARD, CITY_END, CITY_START, END_TURN, MARITIME_END, MARITIME_START, PLAY_KNIGHT,
    PLAY_ROAD_BUILDING, ROAD_END, ROAD_START, SETTLEMENT_END, SETTLEMENT_START,
};
use crate::game::state::GameState;

/// Static heuristic evaluator: no rollouts, scores leaves by VP difference.
pub struct HeuristicEvaluator;

impl Evaluator<GameState> for HeuristicEvaluator {
    fn evaluate(&self, state: &GameState, _rng: &mut fastrand::Rng) -> NnOutput {
        let vp1 = f32::from(state.total_vps(Player::One));
        let vp2 = f32::from(state.total_vps(Player::Two));
        NnOutput {
            policy_logits: vec![0.0; GameState::NUM_ACTIONS],
            value: (vp1 - vp2).tanh(),
        }
    }
}

/// Rollout evaluator with a hand-crafted policy prior.
///
/// Uses random rollouts for the value (like `RolloutEvaluator`) but returns
/// non-uniform policy logits that bias MCTS toward building actions. This
/// makes the MCTS tree search itself smarter without changing the rollouts.
pub struct PolicyEvaluator {
    pub rollout: canopy2::eval::RolloutEvaluator,
}

impl Evaluator<GameState> for PolicyEvaluator {
    fn evaluate(&self, state: &GameState, rng: &mut fastrand::Rng) -> NnOutput {
        let mut out = self.rollout.evaluate(state, rng);
        out.policy_logits = POLICY_LOGITS.to_vec();
        out
    }
}

/// Hand-crafted policy logits over the Catan action space.
///
/// Higher logits make MCTS explore those actions earlier (via softmax prior).
/// These are relative — only the differences matter after softmax.
const POLICY_LOGITS: [f32; action::ACTION_SPACE] = {
    let mut logits = [0.0f32; action::ACTION_SPACE];

    // Building = strong preference
    let mut i = SETTLEMENT_START;
    while i < SETTLEMENT_END {
        logits[i as usize] = 3.0;
        i += 1;
    }
    i = CITY_START;
    while i < CITY_END {
        logits[i as usize] = 3.0;
        i += 1;
    }

    // Roads = moderate (needed for expansion but don't score directly)
    i = ROAD_START;
    while i < ROAD_END {
        logits[i as usize] = 1.0;
        i += 1;
    }

    // Dev cards = moderate
    logits[BUY_DEV_CARD as usize] = 2.0;

    // Playing dev cards = good
    logits[PLAY_KNIGHT as usize] = 2.0;
    logits[PLAY_ROAD_BUILDING as usize] = 1.5;

    // Maritime trade = slight preference over doing nothing
    i = MARITIME_START;
    while i < MARITIME_END {
        logits[i as usize] = 0.5;
        i += 1;
    }

    // End turn = low priority (explore building options first)
    logits[END_TURN as usize] = -1.0;

    // Everything else (robber, discard, steal, yop, monopoly) stays at 0.0
    // — these are usually forced or have few options so prior doesn't matter much.

    logits
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game;
    use crate::game::dice::Dice;
    use canopy2::eval::Evaluator;

    #[test]
    fn heuristic_returns_bounded_value() {
        let state = game::new_game(42, Dice::default());
        let eval = HeuristicEvaluator;
        let mut rng = fastrand::Rng::with_seed(0);
        let out = eval.evaluate(&state, &mut rng);
        assert!(out.value >= -1.0 && out.value <= 1.0);
        assert_eq!(out.policy_logits.len(), GameState::NUM_ACTIONS);
    }

    #[test]
    fn policy_returns_bounded_value() {
        let state = game::new_game(42, Dice::default());
        let eval = PolicyEvaluator {
            rollout: canopy2::eval::RolloutEvaluator { num_rollouts: 1 },
        };
        let mut rng = fastrand::Rng::with_seed(0);
        let out = eval.evaluate(&state, &mut rng);
        assert!(out.value >= -1.0 && out.value <= 1.0);
        assert_eq!(out.policy_logits.len(), GameState::NUM_ACTIONS);
    }

    #[test]
    fn policy_prefers_building_over_end_turn() {
        let settle = POLICY_LOGITS[SETTLEMENT_START as usize];
        let end = POLICY_LOGITS[END_TURN as usize];
        assert!(
            settle > end,
            "settlement ({settle}) should have higher prior than end_turn ({end})"
        );
    }
}
