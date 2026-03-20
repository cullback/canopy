use std::path::Path;

use crate::game::Game;

/// Game-specific presentation layer for the web analysis board.
///
/// The framework doesn't require `Game` to implement `Serialize` — instead,
/// each game provides a `GamePresenter` that knows how to produce JSON state,
/// human-readable action labels, and phase descriptions.
pub trait GamePresenter<G: Game>: Send + Sync {
    /// Serialize the game state to a JSON value for the frontend.
    fn serialize_state(&self, state: &G) -> serde_json::Value;

    /// Human-readable label for an action in the given state (includes player prefix).
    fn action_label(&self, state: &G, action: usize) -> String;

    /// Action description without player prefix (for tree explorer where
    /// the acting player varies by depth).
    fn action_description(&self, state: &G, action: usize) -> String {
        self.action_label(state, action)
    }

    /// Human-readable label for a chance outcome (dice roll, random steal, etc.).
    /// Return empty string to omit from the game log.
    fn chance_label(&self, _state: &G, _outcome: usize) -> String {
        String::new()
    }

    /// Human-readable label for the current phase.
    fn phase_label(&self, state: &G) -> String;

    /// Directory containing static web assets (index.html, JS, CSS).
    fn static_dir(&self) -> &Path;

    /// Create a new game from a seed.
    fn new_game(&self, seed: u64) -> G;
}
