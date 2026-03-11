#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Ongoing,
    /// Reward from P1's perspective. P2's reward is `-reward` (zero-sum).
    Terminal(f32),
}

/// Core trait: every game implements this.
///
/// Actions are `usize` in `[0, NUM_ACTIONS)`.
pub trait Game: Clone + Send + Sync {
    // ── Core (required) ───────────────────────────────────────────

    /// Total number of distinct actions. Actions map to indices in `[0, NUM_ACTIONS)`.
    const NUM_ACTIONS: usize;

    fn status(&self) -> Status;
    fn legal_actions(&self, buf: &mut Vec<usize>);
    fn apply_action(&mut self, action: usize);

    /// Sign of the current player: `1.0` for the maximizing player, `-1.0`
    /// for the minimizing player. Single-player games can leave the default.
    fn current_sign(&self) -> f32 {
        1.0
    }

    // ── Transposition ─────────────────────────────────────────────

    /// Transposition key. Returns `None` (no transpositions) by default.
    fn state_key(&self) -> Option<u64> {
        None
    }

    // ── Stochastic ────────────────────────────────────────────────

    /// Fills `buf` with `(outcome, weight)` pairs (unnormalized integer weights).
    /// An empty buffer means this is a decision node; non-empty means chance node.
    /// Outcomes are passed back to `apply_action` — the game knows it's in a chance
    /// state and interprets the `usize` accordingly.
    ///
    /// Used by MCTS to enumerate all chance branches. For sampling a single
    /// outcome, use [`sample_chance`](Game::sample_chance) instead.
    fn chance_outcomes(&self, _buf: &mut Vec<(usize, u32)>) {}

    /// Sample a single chance outcome. Returns `None` for decision/terminal nodes.
    ///
    /// The default delegates to [`chance_outcomes`](Game::chance_outcomes).
    /// Stochastic games should override this to avoid the intermediate `Vec`.
    fn sample_chance(&self, rng: &mut fastrand::Rng) -> Option<usize> {
        let mut buf = Vec::new();
        self.chance_outcomes(&mut buf);
        crate::utils::sample_weighted(&buf, rng)
    }
}
