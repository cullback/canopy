use canopy::eval::{Evaluation, Evaluator, wdl_from_scalar};

use crate::game::{DESTROY_A, DESTROY_B, DONE, K, NUM_ACTIONS, Phase, TenureGame, optimal_value};

/// Optimal defender: destroys the more valuable partition. Uniform attacker policy.
pub struct OptimalDefender;

impl Evaluator<TenureGame> for OptimalDefender {
    fn evaluate(&self, state: &TenureGame, _rng: &mut fastrand::Rng) -> Evaluation {
        let mut logits = vec![-10.0; NUM_ACTIONS];

        match state.phase {
            Phase::Attacker => {
                for l in 0..K {
                    if state.board[l] > 0 {
                        logits[l] = 0.0;
                    }
                }
                logits[DONE] = 0.0;
            }
            Phase::Defender => {
                let va = optimal_value(&state.board);
                let vb = optimal_value(&state.partition);
                if va >= vb {
                    logits[DESTROY_A] = 10.0;
                } else {
                    logits[DESTROY_B] = 10.0;
                }
            }
        }

        Evaluation {
            policy_logits: logits,
            wdl: wdl_from_scalar(0.0),
        }
    }
}

/// Balanced attacker: greedily moves pieces to minimize |v(A) - v(B)|.
/// Falls back to optimal defender in defender phase.
pub struct BalancedAttacker;

impl Evaluator<TenureGame> for BalancedAttacker {
    fn evaluate(&self, state: &TenureGame, _rng: &mut fastrand::Rng) -> Evaluation {
        let mut logits = vec![-10.0; NUM_ACTIONS];

        match state.phase {
            Phase::Attacker => {
                let va = optimal_value(&state.board);
                let vb = optimal_value(&state.partition);
                let current_diff = (va - vb).abs();
                let mut best_diff = current_diff;
                let mut best_action = DONE;

                for l in 0..K {
                    if state.board[l] > 0 {
                        let piece_value = 1.0 / (1u32 << (l + 1)) as f32;
                        let new_diff = (va - piece_value - (vb + piece_value)).abs();
                        if new_diff < best_diff {
                            best_diff = new_diff;
                            best_action = l;
                        }
                    }
                }

                logits[best_action] = 10.0;
            }
            Phase::Defender => {
                let va = optimal_value(&state.board);
                let vb = optimal_value(&state.partition);
                if va >= vb {
                    logits[DESTROY_A] = 10.0;
                } else {
                    logits[DESTROY_B] = 10.0;
                }
            }
        }

        Evaluation {
            policy_logits: logits,
            wdl: wdl_from_scalar(0.0),
        }
    }
}
