use crate::game::{Game, Status};

/// Policy logits and value estimate produced by any [`Evaluator`].
pub struct Evaluation {
    /// Logits over the full action space `[0, NUM_ACTIONS)`.
    pub policy_logits: Vec<f32>,
    /// Value estimate from P1's perspective.
    pub value: f32,
}

impl Evaluation {
    /// Uniform-logit evaluation (all zeros) with the given value.
    ///
    /// Used for terminal states and rollout evaluators where there is no
    /// meaningful policy signal — zero logits become a uniform distribution
    /// after softmax.
    pub fn uniform(num_actions: usize, value: f32) -> Self {
        Self {
            policy_logits: vec![0.0; num_actions],
            value,
        }
    }
}

/// Evaluates a game state, producing policy logits and a value estimate.
pub trait Evaluator<G: Game> {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> Evaluation;
}

/// Default evaluator: random rollouts with uniform policy logits.
pub struct RolloutEvaluator {
    pub num_rollouts: u32,
}

impl<G: Game> Evaluator<G> for RolloutEvaluator {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> Evaluation {
        assert!(self.num_rollouts > 0, "num_rollouts must be at least 1");
        let mut action_buf = Vec::with_capacity(G::NUM_ACTIONS);
        let mut chance_buf = Vec::with_capacity(G::NUM_ACTIONS);
        let mut total = 0.0f32;
        for _ in 0..self.num_rollouts {
            let mut s = state.clone();
            total += rollout(&mut s, &mut action_buf, &mut chance_buf, rng);
        }
        Evaluation::uniform(G::NUM_ACTIONS, total / self.num_rollouts as f32)
    }
}

fn rollout<G: Game>(
    state: &mut G,
    action_buf: &mut Vec<usize>,
    chance_buf: &mut Vec<(usize, f32)>,
    rng: &mut fastrand::Rng,
) -> f32 {
    loop {
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing(_) => {
                chance_buf.clear();
                state.chance_outcomes(chance_buf);
                let action = if !chance_buf.is_empty() {
                    sample_weighted(chance_buf, rng)
                } else {
                    action_buf.clear();
                    state.legal_actions(action_buf);
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
