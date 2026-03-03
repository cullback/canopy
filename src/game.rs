use std::collections::HashMap;
use std::hash::Hash;

use rand::Rng;

/// Core trait: every game implements this.
pub trait Game: Clone + Send + Sync {
    type Action: Copy + Eq + Hash + Send;
    type Player: Copy + Eq + Hash + Send;

    fn current_player(&self) -> Self::Player;
    fn legal_actions(&self) -> Vec<Self::Action>;
    fn apply_action(&mut self, action: Self::Action);
    fn is_terminal(&self) -> bool;
    fn rewards(&self) -> HashMap<Self::Player, f32>;
    /// Key for transposition table / graph search.
    fn state_key(&self) -> u64;
    /// Maps an action to a dense index in `0..action_space_size()`.
    fn action_index(action: &Self::Action) -> usize;
    /// Total number of distinct actions in the game.
    fn action_space_size() -> usize;
}

/// Extension for stochastic games (dice rolls, card draws, etc.).
pub trait StochasticGame: Game {
    fn is_chance_node(&self) -> bool;
    /// Returns (action, probability) pairs. Probabilities must sum to 1.
    fn chance_outcomes(&self) -> Vec<(Self::Action, f32)>;
}

/// Extension for imperfect-information games.
pub trait ImperfectInfoGame: Game {
    /// Sample a determinization consistent with `observer`'s information set.
    fn determinize(&self, observer: Self::Player, rng: &mut impl Rng) -> Self;
    /// Key that identifies the information set for `observer`.
    fn info_set_key(&self, observer: Self::Player) -> u64;
}
