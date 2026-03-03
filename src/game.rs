use rand::Rng;

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
    /// Total number of distinct actions. Actions map to indices in `[0, NUM_ACTIONS)`.
    const NUM_ACTIONS: usize;

    fn status(&self) -> Status;
    fn legal_actions(&self, buf: &mut Vec<usize>);
    fn apply_action(&mut self, action: usize);
}

/// Extension for games that support transposition tables / graph search.
pub trait TransposableGame: Game {
    fn state_key(&self) -> u64;
}

/// Extension for stochastic games (dice rolls, card draws, etc.).
///
/// Chance outcomes are `usize`, just like actions. The game maps them
/// internally to domain types. When `is_chance_node()` returns true, the
/// engine samples an outcome and applies it via `apply_chance` — the game's
/// `legal_actions` / `apply_action` are not called at chance nodes.
pub trait StochasticGame: Game {
    fn is_chance_node(&self) -> bool;
    /// Fills `buf` with `(outcome, probability)` pairs. Probabilities must sum to 1.
    fn chance_outcomes(&self, buf: &mut Vec<(usize, f32)>);
    fn apply_chance(&mut self, outcome: usize);
}

/// Extension for imperfect-information games.
pub trait ImperfectInfoGame: Game {
    /// Sample a determinization consistent with `observer`'s information set.
    fn determinize(&self, observer: Player, rng: &mut impl Rng) -> Self;
    /// Key that identifies the information set for `observer`.
    fn info_set_key(&self, observer: Player) -> u64;
}
