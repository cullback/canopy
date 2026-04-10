//! Persistent worker tasks and work dispatch types.
//!
//! Each worker owns a `tokio::sync::mpsc` receiver and loops over `WorkItem`s,
//! sending `GameRecord`s back to the coordinator. The coordinator round-robins
//! work across per-worker senders — no shared receiver, no mutex. Also provides
//! `IterGameResults` for aggregating game-level stats from a batch of
//! `GameRecord`s.

use std::sync::Arc;

use crate::game::Game;
use crate::mcts::Search;
use crate::nn::StateEncoder;

use super::game::{ActorConfig, play_game};
use super::inference::InferRequest;
use super::replay_buffer::GameRecord;

/// Aggregated results from one iteration's games (for metrics/logging).
pub(super) struct IterGameResults {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub total_actions: u32,
    pub min_game_length: Option<u32>,
    pub max_game_length: u32,
    pub game_length_stddev: f64,
    pub search_pv_depth: f32,
    pub search_depth_max: u32,
}

impl IterGameResults {
    pub fn aggregate(games: &[GameRecord]) -> Self {
        let mut wins = 0u32;
        let mut losses = 0u32;
        let mut draws = 0u32;
        let mut total_actions = 0u32;
        let mut sum_actions_sq = 0u64;
        let mut min_game_length: Option<u32> = None;
        let mut max_game_length = 0u32;

        let mut depth_sum = 0.0f64;
        let mut depth_max = 0u32;
        let mut depth_count = 0u32;

        for game in games {
            let game_len = game.samples.last().map_or(0, |s| s.game_length);
            total_actions += game_len;
            sum_actions_sq += (game_len as u64) * (game_len as u64);
            min_game_length = Some(min_game_length.map_or(game_len, |m: u32| m.min(game_len)));
            max_game_length = max_game_length.max(game_len);
            if game.reward > 0.0 {
                wins += 1;
            } else if game.reward < 0.0 {
                losses += 1;
            } else {
                draws += 1;
            }
            for s in &game.samples {
                depth_sum += s.search_pv_depth as f64;
                depth_max = depth_max.max(s.search_depth_max);
                depth_count += 1;
            }
        }

        let num_games = wins + losses + draws;
        let game_length_stddev = if num_games > 0 {
            let mean = total_actions as f64 / num_games as f64;
            let var = sum_actions_sq as f64 / num_games as f64 - mean * mean;
            var.max(0.0).sqrt()
        } else {
            0.0
        };

        let search_pv_depth = if depth_count > 0 {
            (depth_sum / depth_count as f64) as f32
        } else {
            0.0
        };

        Self {
            wins,
            losses,
            draws,
            total_actions,
            min_game_length,
            max_game_length,
            game_length_stddev,
            search_pv_depth,
            search_depth_max: depth_max,
        }
    }
}

// ---------------------------------------------------------------------------
// Work items and results
// ---------------------------------------------------------------------------

pub(super) enum WorkItem {
    SelfPlay { seed: u64, effective_sims: u32 },
    Shutdown,
}

// ---------------------------------------------------------------------------
// Persistent worker task
// ---------------------------------------------------------------------------

/// Persistent worker task. Each worker owns its own mpsc receiver.
pub(super) async fn worker_loop<G: Game + std::fmt::Display + 'static>(
    mut work_rx: tokio::sync::mpsc::Receiver<WorkItem>,
    result_tx: tokio::sync::mpsc::UnboundedSender<GameRecord>,
    request_tx: tokio::sync::mpsc::Sender<InferRequest>,
    encoder: Arc<dyn StateEncoder<G>>,
    actor_config: Arc<ActorConfig>,
    mcts_config: crate::mcts::Config,
    new_state: Arc<dyn Fn(u64) -> G + Send + Sync>,
    worker_seed: u64,
) {
    let mut rng = fastrand::Rng::with_seed(worker_seed);
    let mut search: Option<Search<G>> = None;

    while let Some(item) = work_rx.recv().await {
        match item {
            WorkItem::SelfPlay {
                seed,
                effective_sims,
            } => {
                let state = (new_state)(seed);
                let fast_sims = actor_config.playout_cap_fast_sims_base.min(effective_sims);

                // Reuse or create Search
                match &mut search {
                    Some(s) => s.reset(state),
                    None => {
                        search = Some(Search::new(state, mcts_config.clone()));
                    }
                }
                let s = search.as_mut().unwrap();

                let tx = request_tx.clone();
                let game = play_game(
                    s,
                    effective_sims,
                    fast_sims,
                    &actor_config,
                    &*encoder,
                    |flat_features, batch_size| {
                        let tx = tx.clone();
                        async move {
                            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                            tx.send(InferRequest {
                                flat_features,
                                batch_size,
                                response_tx: resp_tx,
                            })
                            .await
                            .unwrap();
                            let resp = resp_rx.await.unwrap();
                            (resp.flat_policy_logits, resp.flat_wdl)
                        }
                    },
                    &mut rng,
                )
                .await;

                if result_tx.send(game).is_err() {
                    break;
                }
            }

            WorkItem::Shutdown => break,
        }
    }
}
