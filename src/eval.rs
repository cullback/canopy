use std::sync::Arc;

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

/// Instant evaluator: uniform policy, zero value. No rollout, no cloning.
pub struct RandomEvaluator;

impl<G: Game> Evaluator<G> for RandomEvaluator {
    fn evaluate(&self, _state: &G, _rng: &mut fastrand::Rng) -> Evaluation {
        Evaluation::uniform(G::NUM_ACTIONS, 0.0)
    }
}

/// Default evaluator: random rollouts with uniform policy logits.
#[derive(Clone)]
pub struct RolloutEvaluator {
    pub num_rollouts: u32,
}

impl Default for RolloutEvaluator {
    fn default() -> Self {
        Self { num_rollouts: 1 }
    }
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

/// Named registry of evaluators for tournament and benchmark dispatch.
pub struct Evaluators<G: Game> {
    entries: Vec<(String, Arc<dyn Evaluator<G> + Sync>)>,
}

impl<G: Game> Default for Evaluators<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game> Evaluators<G> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, name: impl Into<String>, eval: impl Evaluator<G> + Sync + 'static) {
        self.entries.push((name.into(), Arc::new(eval)));
    }

    pub fn add_arc(&mut self, name: impl Into<String>, eval: Arc<dyn Evaluator<G> + Sync>) {
        self.entries.push((name.into(), eval));
    }

    /// Look up an evaluator by name. Panics with available names on miss.
    pub fn get(&self, name: &str) -> &(dyn Evaluator<G> + Sync) {
        for (n, e) in &self.entries {
            if n == name {
                return &**e;
            }
        }
        let names: Vec<&str> = self.entries.iter().map(|(n, _)| n.as_str()).collect();
        panic!("unknown evaluator '{name}', available: {names:?}");
    }

    /// Look up an evaluator by name and clone the `Arc`. Panics on miss.
    pub fn get_arc(&self, name: &str) -> Arc<dyn Evaluator<G> + Sync> {
        for (n, e) in &self.entries {
            if n == name {
                return Arc::clone(e);
            }
        }
        let names: Vec<&str> = self.entries.iter().map(|(n, _)| n.as_str()).collect();
        panic!("unknown evaluator '{name}', available: {names:?}");
    }
}

fn rollout<G: Game>(state: &mut G, action_buf: &mut Vec<usize>, rng: &mut fastrand::Rng) -> f32 {
    loop {
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing => {
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
