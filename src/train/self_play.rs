use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::eval::Evaluator;
use crate::game::Game;
use crate::nn::StateEncoder;
use crate::player::Player;

use super::TrainConfig;
use super::game::{ActorConfig, GameResult, play_game};
use super::inference::{BatcherStats, InferRequest, batcher_loop, gpu_worker_loop};

/// Aggregated results from one self-play iteration (many games).
pub(super) struct IterGameResults {
    pub samples: Vec<super::Sample>,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub total_turns: u32,
    pub min_game_length: Option<u32>,
    pub max_game_length: u32,
    pub game_length_stddev: f64,
    pub num_tasks: usize,
    pub total_batches: u64,
    pub total_evals: u64,
}

impl IterGameResults {
    pub fn avg_batch_size(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.total_evals as f64 / self.total_batches as f64
        }
    }
}

/// Run self-play games using async actor tasks with batched multi-GPU inference.
///
/// Architecture:
/// - N async actor tasks play games, sending `InferRequest`s via tokio mpsc
/// - 1 batcher thread collects requests into batches, dispatches via crossbeam SPMC
/// - M GPU worker threads receive batches and run forward passes
///
/// Shutdown: actors finish → drop mpsc sender → batcher exits → drops crossbeam
/// sender → workers exit.
pub(super) fn run_self_play_iteration<G, E, Ev>(
    evaluator: Ev,
    config: &TrainConfig,
    effective_sims: u32,
    iteration: usize,
    rng: &mut fastrand::Rng,
    new_state: &Arc<dyn Fn(&mut fastrand::Rng) -> G + Send + Sync>,
) -> IterGameResults
where
    G: Game + 'static,
    E: StateEncoder<G> + 'static,
    Ev: Evaluator<G> + 'static,
{
    let pb = indicatif::ProgressBar::new(config.games_per_iter as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message(format!(
        "iter {}/{} self-play (sims={})",
        iteration + 1,
        config.iterations,
        effective_sims,
    ));

    let base_config = crate::mcts::Config {
        num_simulations: effective_sims,
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        leaf_batch_size: config.leaf_batch_size,
        ..Default::default()
    };
    let fast_sims = config.playout_cap_fast_sims.min(effective_sims);

    let actor_config = Arc::new(ActorConfig {
        explore_moves: config.explore_moves,
        playout_cap_full_prob: config.playout_cap_full_prob,
        playout_cap_fast_sims: fast_sims,
        effective_sims,
    });

    let concurrent_games = config.concurrent_games;
    let max_batch_size = config.max_batch_size;
    let num_actions = G::NUM_ACTIONS;
    let games_remaining = Arc::new(AtomicU32::new(config.games_per_iter as u32));
    let completed = Arc::new(AtomicU32::new(0));
    let batcher_stats = Arc::new(BatcherStats::new());
    let pb = Arc::new(pb);

    // Actor → Batcher channel (tokio mpsc)
    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(2 * max_batch_size);

    // Batcher → GPU workers channel (crossbeam SPMC, bounded to limit memory)
    // Currently 1 worker; multi-GPU: clone evaluator per worker + spawn more.
    let (work_tx, work_rx) = crossbeam_channel::bounded(2);

    let worker_handle = std::thread::spawn(move || {
        gpu_worker_loop(
            work_rx,
            |features, batch_size, feature_size| {
                evaluator.infer_features(features, batch_size, feature_size)
            },
            num_actions,
        );
    });

    // Spawn batcher thread
    let batcher_stats_ref = batcher_stats.clone();
    let batcher_handle = std::thread::spawn(move || {
        batcher_loop(
            &mut request_rx,
            &work_tx,
            max_batch_size,
            &batcher_stats_ref,
        );
    });

    // Tokio runtime for async actor tasks
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let game_results: Vec<GameResult> = rt.block_on(async {
        let mut join_set = tokio::task::JoinSet::new();

        for _ in 0..concurrent_games {
            let seed = rng.u64(..);
            let request_tx = request_tx.clone();
            let games_remaining = games_remaining.clone();
            let completed = completed.clone();
            let pb = pb.clone();
            let batcher_stats = batcher_stats.clone();
            let new_state = new_state.clone();
            let base_config = base_config.clone();
            let actor_config = actor_config.clone();

            join_set.spawn(async move {
                let mut rng = fastrand::Rng::with_seed(seed);
                let mut results = Vec::new();

                loop {
                    // Claim a game via atomic decrement
                    let claimed = games_remaining
                        .fetch_update(Relaxed, Relaxed, |n| if n > 0 { Some(n - 1) } else { None });
                    if claimed.is_err() {
                        break;
                    }

                    let state = new_state(&mut rng);
                    let tx = request_tx.clone();
                    let game = play_game::<G, E, _, _>(
                        state,
                        base_config.clone(),
                        &actor_config,
                        |features| {
                            let tx = tx.clone();
                            async move {
                                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                                tx.send(InferRequest {
                                    features,
                                    response_tx: resp_tx,
                                })
                                .await
                                .unwrap();
                                let resp = resp_rx.await.unwrap();
                                (resp.policy_logits, resp.value)
                            }
                        },
                        &mut rng,
                    )
                    .await;

                    results.push(game);

                    let done = completed.fetch_add(1, Relaxed) + 1;
                    pb.set_position(done as u64);
                    let avg_batch = batcher_stats.avg_batch_size();
                    let elapsed = pb.elapsed().as_secs_f64();
                    let evals = batcher_stats.evals.load(Relaxed);
                    let evals_per_sec = if elapsed > 0.0 {
                        evals as f64 / elapsed
                    } else {
                        0.0
                    };
                    pb.set_message(format!(
                        "avg_batch={avg_batch:.1}, evals/s={evals_per_sec:.0}"
                    ));
                }

                results
            });
        }

        // Drop our sender so batcher sees disconnect when all tasks finish
        drop(request_tx);

        let mut all_results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            all_results.extend(result.unwrap());
        }
        all_results
    });

    // Batcher + worker threads exit once all senders are dropped
    batcher_handle.join().unwrap();
    worker_handle.join().unwrap();

    pb.finish();

    // Aggregate results
    let mut samples = Vec::new();
    let mut p1_wins = 0u32;
    let mut p2_wins = 0u32;
    let mut draws = 0u32;
    let mut total_turns = 0u32;
    let mut sum_turns_sq = 0u64;
    let mut min_game_length: Option<u32> = None;
    let mut max_game_length = 0u32;
    let mut num_games = 0u32;

    for game in game_results {
        let game_len = game.samples.last().map_or(0, |s| s.game_length);
        total_turns += game_len;
        sum_turns_sq += (game_len as u64) * (game_len as u64);
        num_games += 1;
        min_game_length = Some(min_game_length.map_or(game_len, |m: u32| m.min(game_len)));
        max_game_length = max_game_length.max(game_len);
        match game.winner {
            Some(Player::One) => p1_wins += 1,
            Some(Player::Two) => p2_wins += 1,
            None => draws += 1,
        }
        samples.extend(game.samples);
    }

    let game_length_stddev = if num_games > 0 {
        let mean = total_turns as f64 / num_games as f64;
        let var = sum_turns_sq as f64 / num_games as f64 - mean * mean;
        var.max(0.0).sqrt()
    } else {
        0.0
    };

    IterGameResults {
        samples,
        p1_wins,
        p2_wins,
        draws,
        total_turns,
        min_game_length,
        max_game_length,
        game_length_stddev,
        num_tasks: concurrent_games,
        total_batches: batcher_stats.batches.load(Relaxed),
        total_evals: batcher_stats.evals.load(Relaxed),
    }
}
