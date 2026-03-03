use crate::game::{Game, Status};

/// Output from a neural network (or any position evaluator).
pub struct NnOutput {
    /// Logits over the full action space `[0, NUM_ACTIONS)`.
    pub policy_logits: Vec<f32>,
    /// Value estimate from P1's perspective.
    pub value: f32,
}

/// Evaluates a game state, producing policy logits and a value estimate.
pub trait Evaluator<G: Game> {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> NnOutput;
}

/// Default evaluator: random rollouts with uniform policy logits.
pub struct RolloutEvaluator {
    pub num_rollouts: u32,
}

impl<G: Game> Evaluator<G> for RolloutEvaluator {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> NnOutput {
        let mut total = 0.0f32;
        for _ in 0..self.num_rollouts {
            let mut s = state.clone();
            total += rollout(&mut s, rng);
        }
        NnOutput {
            policy_logits: vec![0.0; G::NUM_ACTIONS],
            value: total / self.num_rollouts as f32,
        }
    }
}

fn rollout<G: Game>(state: &mut G, rng: &mut fastrand::Rng) -> f32 {
    let mut action_buf = Vec::new();
    let mut chance_buf = Vec::new();
    loop {
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing(_) => {
                chance_buf.clear();
                state.chance_outcomes(&mut chance_buf);
                let action = if !chance_buf.is_empty() {
                    sample_weighted(&chance_buf, rng)
                } else {
                    action_buf.clear();
                    state.legal_actions(&mut action_buf);
                    action_buf[rng.usize(..action_buf.len())]
                };
                state.apply_action(action);
            }
        }
    }
}

fn sample_weighted(items: &[(usize, f32)], rng: &mut fastrand::Rng) -> usize {
    let total: f32 = items.iter().map(|(_, p)| p).sum();
    let mut r = rng.f32() * total;
    for &(item, p) in items {
        r -= p;
        if r <= 0.0 {
            return item;
        }
    }
    items.last().unwrap().0
}
