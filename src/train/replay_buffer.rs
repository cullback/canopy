//! Sample-capped replay buffer.
//!
//! Stores completed games as `GameRecord`s. Capped by total sample count —
//! oldest games are evicted first when the buffer exceeds capacity.

use std::collections::VecDeque;

use super::Sample;

/// A completed game stored in the replay buffer.
pub struct GameRecord {
    /// Terminal reward (P1 perspective).
    pub reward: f32,
    pub samples: Vec<Sample>,
    /// Initial state string for game replay.
    pub initial_state: String,
    /// Action log (all actions including chance) for game replay.
    pub actions: Vec<usize>,
}

/// Sample-capped replay buffer.
pub struct ReplayBuffer {
    games: VecDeque<GameRecord>,
    max_samples: usize,
    cached_samples: usize,
}

impl ReplayBuffer {
    pub fn new(max_samples: usize) -> Self {
        Self {
            games: VecDeque::new(),
            max_samples,
            cached_samples: 0,
        }
    }

    /// Add games to the buffer, evicting oldest if over capacity.
    pub fn push_games(&mut self, games: Vec<GameRecord>) {
        for game in games {
            self.cached_samples += game.samples.len();
            self.games.push_back(game);
        }
        self.evict();
    }

    /// All samples across all games (for training).
    pub fn all_samples(&self) -> Vec<&Sample> {
        self.games.iter().flat_map(|g| g.samples.iter()).collect()
    }

    pub fn total_samples(&self) -> usize {
        self.cached_samples
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn games(&self) -> &VecDeque<GameRecord> {
        &self.games
    }

    /// Evict oldest games until total samples <= max_samples.
    fn evict(&mut self) {
        while self.cached_samples > self.max_samples {
            if let Some(game) = self.games.pop_front() {
                self.cached_samples -= game.samples.len();
            } else {
                break;
            }
        }
    }
}
