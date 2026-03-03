use rand::RngCore;

use crate::player::Player;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Ongoing(Player),
    /// Reward from P1's perspective. P2's reward is `-reward` (zero-sum).
    Terminal(f32),
}

/// Core trait: every game implements this.
///
/// Actions are `usize` in `[0, NUM_ACTIONS)`. Players are [`Player`].
pub trait Game: Clone + Send + Sync {
    // ── Core (required) ───────────────────────────────────────────

    /// Total number of distinct actions. Actions map to indices in `[0, NUM_ACTIONS)`.
    const NUM_ACTIONS: usize;

    fn status(&self) -> Status;
    fn legal_actions(&self, buf: &mut Vec<usize>);
    fn apply_action(&mut self, action: usize);

    // ── Transposition ─────────────────────────────────────────────

    /// Transposition key. Returns `None` (no transpositions) by default.
    fn state_key(&self) -> Option<u64> {
        None
    }

    // ── Stochastic ────────────────────────────────────────────────

    /// Fills `buf` with `(outcome, probability)` pairs. Probabilities must sum to 1.
    /// An empty buffer means this is a decision node; non-empty means chance node.
    /// Outcomes are passed back to `apply_action` — the game knows it's in a chance
    /// state and interprets the `usize` accordingly.
    fn chance_outcomes(&self, _buf: &mut Vec<(usize, f32)>) {}

    // ── Imperfect information ─────────────────────────────────────

    /// Sample a determinization consistent with `observer`'s information set.
    fn determinize(&self, _observer: Player, _rng: &mut dyn RngCore) -> Self
    where
        Self: Sized,
    {
        unimplemented!()
    }

    /// Key that identifies the information set for `observer`.
    fn info_set_key(&self, _observer: Player) -> Option<u64> {
        None
    }
}
