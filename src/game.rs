#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    /// Decision node: the given player acts. `+1.0` = maximizer, `-1.0` = minimizer.
    Decision(f32),
    /// Chance node: the environment acts.
    Chance,
    /// Game over. Reward from P1's perspective; P2's reward is `-reward` (zero-sum).
    Terminal(f32),
}

/// Core trait: every game implements this.
pub trait Game: Clone + Send + Sync {
    /// Total number of distinct actions in `[0, NUM_ACTIONS)`.
    const NUM_ACTIONS: usize;

    /// Current node type and acting player.
    fn status(&self) -> Status;

    /// Append legal actions to `buf`. Only meaningful for decision nodes.
    fn legal_actions(&self, buf: &mut Vec<usize>);

    /// Apply an action (player decision or chance outcome).
    fn apply_action(&mut self, action: usize);

    /// Transposition key. `None` disables transposition detection (default).
    fn state_key(&self) -> Option<u64> {
        None
    }

    /// Enumerate chance outcomes as `(outcome, weight)` pairs.
    /// Outcomes are passed back to `apply_action`.
    fn chance_outcomes(&self, _buf: &mut Vec<(usize, u32)>) {}

    /// Sample a single chance outcome. Returns `None` for non-chance nodes.
    /// Override to avoid the intermediate `Vec` from `chance_outcomes`.
    fn sample_chance(&self, rng: &mut fastrand::Rng) -> Option<usize> {
        let mut buf = Vec::new();
        self.chance_outcomes(&mut buf);
        crate::utils::sample_weighted(&buf, rng)
    }

    /// Resample hidden information before an MCTS simulation.
    ///
    /// Called once per simulation on a clone of the root state. Returns
    /// `true` if resampling occurred — search then filters tree edges
    /// against `legal_actions()` during descent (SO-ISMCTS).
    fn determinize(&mut self, _rng: &mut fastrand::Rng) -> bool {
        false
    }
}
