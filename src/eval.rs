use std::sync::Arc;

use crate::game::{Game, Status};

/// Convert a scalar value in [-1, 1] to a WDL distribution.
///
/// Linear mapping: w = (1+v)/2, d = 0, l = (1-v)/2.
/// Works for any terminal reward in [-1, 1], not just {-1, 0, 1}.
pub fn wdl_from_scalar(v: f32) -> [f32; 3] {
    let w = ((1.0 + v) / 2.0).clamp(0.0, 1.0);
    let l = ((1.0 - v) / 2.0).clamp(0.0, 1.0);
    [w, 0.0, l]
}

/// Flip a WDL distribution: swap Win and Loss (perspective change).
pub fn flip_wdl(wdl: [f32; 3]) -> [f32; 3] {
    [wdl[2], wdl[1], wdl[0]]
}

/// Policy logits and value estimate produced by any [`Evaluator`].
pub struct Evaluation {
    /// Logits over the full action space `[0, NUM_ACTIONS)`.
    pub policy_logits: Vec<f32>,
    /// Win/Draw/Loss probabilities from P1's perspective.
    pub wdl: [f32; 3],
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
            wdl: wdl_from_scalar(value),
        }
    }

    /// Uniform-logit evaluation with an explicit WDL distribution.
    pub fn uniform_wdl(num_actions: usize, wdl: [f32; 3]) -> Self {
        Self {
            policy_logits: vec![0.0; num_actions],
            wdl,
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
    /// `(flat_policy_logits, flat_wdl)`. Only supported by neural evaluators.
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
        let mut wdl = [0.0f32; 3];
        let n = self.num_rollouts as f32;
        for _ in 0..self.num_rollouts {
            let mut s = state.clone();
            let reward = rollout(&mut s, &mut action_buf, rng);
            if reward > 0.0 {
                wdl[0] += 1.0 / n;
            } else if reward < 0.0 {
                wdl[2] += 1.0 / n;
            } else {
                wdl[1] += 1.0 / n;
            }
        }
        Evaluation::uniform_wdl(G::NUM_ACTIONS, wdl)
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

/// Maximum actions per rollout before declaring a draw.
const ROLLOUT_MAX_STEPS: u32 = 10_000;

fn rollout<G: Game>(state: &mut G, action_buf: &mut Vec<usize>, rng: &mut fastrand::Rng) -> f32 {
    for _ in 0..ROLLOUT_MAX_STEPS {
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
    0.0 // draw if rollout exceeds step limit
}
