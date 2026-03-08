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
pub trait Evaluator<G: Game>: Send {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> Evaluation;

    /// Evaluate multiple states in a single batch.
    ///
    /// The default implementation loops over each state individually.
    /// Neural evaluators override this with a single batched forward pass.
    fn evaluate_batch(&self, states: &[&G], rng: &mut fastrand::Rng) -> Vec<Evaluation> {
        states.iter().map(|s| self.evaluate(s, rng)).collect()
    }

    /// Run raw inference on pre-encoded features.
    ///
    /// Takes flat features `[batch_size * feature_size]` and returns
    /// `(flat_policy_logits, flat_values)`. Only supported by neural evaluators.
    fn infer_features(
        &self,
        _features: Vec<f32>,
        _batch_size: usize,
        _feature_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        unimplemented!("infer_features not supported for this evaluator")
    }
}

/// Default evaluator: random rollouts with uniform policy logits.
#[derive(Clone)]
pub struct RolloutEvaluator {
    pub num_rollouts: u32,
}

impl<G: Game> Evaluator<G> for RolloutEvaluator {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> Evaluation {
        assert!(self.num_rollouts > 0, "num_rollouts must be at least 1");
        let mut action_buf = Vec::with_capacity(G::NUM_ACTIONS);
        let mut total = 0.0f32;
        for _ in 0..self.num_rollouts {
            let mut s = state.clone();
            total += rollout(&mut s, &mut action_buf, rng);
        }
        Evaluation::uniform(G::NUM_ACTIONS, total / self.num_rollouts as f32)
    }
}

fn rollout<G: Game>(state: &mut G, action_buf: &mut Vec<usize>, rng: &mut fastrand::Rng) -> f32 {
    loop {
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing(_) => {
                if let Some(action) = state.sample_chance(rng) {
                    state.apply_action(action);
                } else {
                    action_buf.clear();
                    state.legal_actions(action_buf);
                    let action = action_buf[rng.usize(..action_buf.len())];
                    state.apply_action(action);
                }
            }
        }
    }
}
