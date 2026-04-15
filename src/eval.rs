use std::sync::Arc;

use crate::game::{Game, Status};

/// Win/Draw/Loss probabilities from P1's perspective.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Wdl {
    pub w: f32,
    pub d: f32,
    pub l: f32,
}

impl Wdl {
    pub const DRAW: Wdl = Wdl {
        w: 0.0,
        d: 1.0,
        l: 0.0,
    };

    /// Convert a scalar value in [-1, 1] to a WDL distribution.
    pub fn from_value(v: f32) -> Self {
        let w = ((1.0 + v) / 2.0).clamp(0.0, 1.0);
        let l = ((1.0 - v) / 2.0).clamp(0.0, 1.0);
        Wdl { w, d: 0.0, l }
    }

    pub fn q(&self) -> f32 {
        self.w - self.l
    }

    pub fn flip(self) -> Self {
        Wdl {
            w: self.l,
            d: self.d,
            l: self.w,
        }
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.w, self.d, self.l]
    }

    pub fn from_array(a: [f32; 3]) -> Self {
        Wdl {
            w: a[0],
            d: a[1],
            l: a[2],
        }
    }
}

/// Policy logits and value estimate produced by any [`Evaluator`].
pub struct Evaluation {
    pub policy_logits: Vec<f32>,
    pub wdl: Wdl,
}

impl Evaluation {
    pub fn uniform(num_actions: usize, value: f32) -> Self {
        Self {
            policy_logits: vec![0.0; num_actions],
            wdl: Wdl::from_value(value),
        }
    }

    pub fn uniform_wdl(num_actions: usize, wdl: Wdl) -> Self {
        Self {
            policy_logits: vec![0.0; num_actions],
            wdl,
        }
    }
}

/// Evaluates a game state, producing policy logits and a value estimate.
pub trait Evaluator<G: Game>: Send {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> Evaluation;

    fn evaluate_batch(&self, states: &[&G], rng: &mut fastrand::Rng) -> Vec<Evaluation> {
        states.iter().map(|s| self.evaluate(s, rng)).collect()
    }

    fn infer_features(
        &self,
        _features: Vec<f32>,
        _batch_size: usize,
        _feature_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        unimplemented!("infer_features not supported for this evaluator")
    }
}

/// Instant evaluator: uniform policy, zero value.
pub struct RandomEvaluator;

impl<G: Game> Evaluator<G> for RandomEvaluator {
    fn evaluate(&self, _state: &G, _rng: &mut fastrand::Rng) -> Evaluation {
        Evaluation::uniform(G::NUM_ACTIONS, 0.0)
    }
}

/// Random rollouts with uniform policy logits.
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
        assert!(self.num_rollouts > 0);
        let mut action_buf = Vec::with_capacity(G::NUM_ACTIONS);
        let mut wdl = Wdl {
            w: 0.0,
            d: 0.0,
            l: 0.0,
        };
        let n = self.num_rollouts as f32;
        for _ in 0..self.num_rollouts {
            let mut s = state.clone();
            let reward = rollout(&mut s, &mut action_buf, rng);
            if reward > 0.0 {
                wdl.w += 1.0 / n;
            } else if reward < 0.0 {
                wdl.l += 1.0 / n;
            } else {
                wdl.d += 1.0 / n;
            }
        }
        Evaluation::uniform_wdl(G::NUM_ACTIONS, wdl)
    }
}

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

    pub fn get(&self, name: &str) -> &(dyn Evaluator<G> + Sync) {
        for (n, e) in &self.entries {
            if n == name {
                return &**e;
            }
        }
        let names: Vec<&str> = self.entries.iter().map(|(n, _)| n.as_str()).collect();
        panic!("unknown evaluator '{name}', available: {names:?}");
    }

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

const ROLLOUT_MAX_STEPS: u32 = 10_000;

fn rollout<G: Game>(state: &mut G, action_buf: &mut Vec<usize>, rng: &mut fastrand::Rng) -> f32 {
    for _ in 0..ROLLOUT_MAX_STEPS {
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Chance => match state.sample_chance(rng) {
                Some(a) => state.apply_action(a),
                None => return 0.0,
            },
            Status::Decision(_) => {
                action_buf.clear();
                state.legal_actions(action_buf);
                let action = action_buf[rng.usize(..action_buf.len())];
                state.apply_action(action);
            }
        }
    }
    0.0
}
