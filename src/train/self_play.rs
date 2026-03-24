//! Persistent worker tasks and work dispatch types.
//!
//! Each worker owns a `tokio::sync::mpsc` receiver and loops over `WorkItem`s
//! (self-play or reanalyze), sending `WorkResult`s back to the coordinator.
//! The coordinator round-robins work across per-worker senders — no shared
//! receiver, no mutex. Also provides `IterGameResults` for aggregating
//! game-level stats from a batch of `GameRecord`s.

use std::sync::Arc;

use crate::game::Game;
use crate::mcts::Search;
use crate::nn::StateEncoder;

use super::game::{ActorConfig, play_game, reanalyze_game};
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
        }

        let num_games = wins + losses + draws;
        let game_length_stddev = if num_games > 0 {
            let mean = total_actions as f64 / num_games as f64;
            let var = sum_actions_sq as f64 / num_games as f64 - mean * mean;
            var.max(0.0).sqrt()
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
        }
    }
}

// ---------------------------------------------------------------------------
// Work items and results
// ---------------------------------------------------------------------------

pub(super) enum WorkItem {
    SelfPlay {
        seed: u64,
        effective_sims: u32,
    },
    Reanalyze {
        game_id: u64,
        seed: u64,
        actions: Vec<usize>,
        reward: f32,
        effective_sims: u32,
    },
    Shutdown,
}

pub(super) enum WorkResult {
    SelfPlay(GameRecord),
    Reanalyze {
        game_id: u64,
        samples: Vec<super::Sample>,
    },
}

// ---------------------------------------------------------------------------
// Persistent worker task
// ---------------------------------------------------------------------------

/// Persistent worker task. Each worker owns its own mpsc receiver.
pub(super) async fn worker_loop<G: Game + 'static>(
    mut work_rx: tokio::sync::mpsc::Receiver<WorkItem>,
    result_tx: tokio::sync::mpsc::UnboundedSender<WorkResult>,
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
                let game_seed = seed;
                let state = (new_state)(game_seed);
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
                    game_seed,
                )
                .await;

                let mean_vc = if game.samples.is_empty() {
                    0.0
                } else {
                    game.samples.iter().map(|s| s.value_correction).sum::<f32>()
                        / game.samples.len() as f32
                };

                let record = GameRecord {
                    id: 0, // assigned by ReplayBuffer
                    seed: game.seed,
                    actions: game.actions,
                    reward: game.reward,
                    samples: game.samples,
                    mean_value_correction: mean_vc,
                    iteration_analyzed: 0, // set by coordinator
                };

                if result_tx.send(WorkResult::SelfPlay(record)).is_err() {
                    break;
                }
            }

            WorkItem::Reanalyze {
                game_id,
                seed,
                actions,
                reward,
                effective_sims,
            } => {
                let state = (new_state)(seed);

                match &mut search {
                    Some(s) => s.reset(state),
                    None => {
                        search = Some(Search::new(state, mcts_config.clone()));
                    }
                }
                let s = search.as_mut().unwrap();

                let tx = request_tx.clone();
                let samples = reanalyze_game(
                    s,
                    effective_sims,
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
                    &actions,
                    reward,
                )
                .await;

                if result_tx
                    .send(WorkResult::Reanalyze { game_id, samples })
                    .is_err()
                {
                    break;
                }
            }

            WorkItem::Shutdown => break,
        }
    }
}
