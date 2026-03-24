//! Sample-capped replay buffer with monotonic game IDs.
//!
//! Stores completed games as `GameRecord`s. Capped by total sample count —
//! oldest games are evicted first when the buffer exceeds capacity. Supports
//! staleness-prioritized selection for reanalyze: games with high value
//! correction and stale analysis are chosen more often.

use std::collections::VecDeque;

use super::Sample;

/// A completed game stored in the replay buffer.
pub struct GameRecord {
    /// Monotonic, unique, never reused.
    pub id: u64,
    pub seed: u64,
    /// All actions (chance + player) in sequence — needed for reanalyze replay.
    pub actions: Vec<usize>,
    /// Terminal reward (P1 perspective).
    pub reward: f32,
    pub samples: Vec<Sample>,
    /// Priority proxy: mean |Q - V| across samples.
    pub mean_value_correction: f32,
    /// Iteration when this game was created or last reanalyzed.
    pub iteration_analyzed: usize,
}

/// Sample-capped replay buffer with monotonic game IDs.
pub struct ReplayBuffer {
    games: VecDeque<GameRecord>,
    max_samples: usize,
    next_id: u64,
    cached_samples: usize,
}

impl ReplayBuffer {
    pub fn new(max_samples: usize) -> Self {
        Self {
            games: VecDeque::new(),
            max_samples,
            next_id: 0,
            cached_samples: 0,
        }
    }

    /// Add games to the buffer, assigning monotonic IDs and evicting oldest if over capacity.
    pub fn push_games(&mut self, games: Vec<GameRecord>) {
        for mut game in games {
            game.id = self.next_id;
            self.next_id += 1;
            self.cached_samples += game.samples.len();
            self.games.push_back(game);
        }
        self.evict();
    }

    /// Replace samples for a game by ID (after reanalyze). No-op if evicted.
    pub fn replace_samples(&mut self, game_id: u64, new_samples: Vec<Sample>, iteration: usize) {
        if let Some(game) = self.games.iter_mut().find(|g| g.id == game_id) {
            self.cached_samples -= game.samples.len();
            self.cached_samples += new_samples.len();
            let mean_vc = if new_samples.is_empty() {
                0.0
            } else {
                new_samples.iter().map(|s| s.value_correction).sum::<f32>()
                    / new_samples.len() as f32
            };
            game.samples = new_samples;
            game.mean_value_correction = mean_vc;
            game.iteration_analyzed = iteration;
        }
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

    /// Select games for reanalyze, prioritized by value correction * staleness.
    ///
    /// Priority = mean_value_correction * min((current_iter - iteration_analyzed) / 4, 1.0)
    pub fn select_for_reanalyze(
        &self,
        count: usize,
        current_iteration: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<(u64, u64, Vec<usize>, f32)> {
        if self.games.is_empty() || count == 0 {
            return Vec::new();
        }

        // Build weighted list
        let weights: Vec<(usize, f32)> = self
            .games
            .iter()
            .enumerate()
            .map(|(idx, g)| {
                let staleness = (current_iteration.saturating_sub(g.iteration_analyzed)) as f32;
                let priority = g.mean_value_correction * (staleness / 4.0).min(1.0);
                (idx, priority.max(1e-6))
            })
            .collect();

        let total_weight: f32 = weights.iter().map(|(_, w)| w).sum();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let mut roll = rng.f32() * total_weight;
            let mut pick = weights.len() - 1;
            for (i, &(_, w)) in weights.iter().enumerate() {
                roll -= w;
                if roll <= 0.0 {
                    pick = i;
                    break;
                }
            }
            let (game_idx, _) = weights[pick];
            let g = &self.games[game_idx];
            selected.push((g.id, g.seed, g.actions.clone(), g.reward));
        }

        selected
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
