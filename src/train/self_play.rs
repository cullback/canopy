use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::eval::Evaluator;
use crate::game::Game;
use crate::mcts::Search;
use crate::nn::StateEncoder;

use super::TrainConfig;
use super::game::{ActorConfig, GameResult, play_game};
use super::inference::{BatcherStats, InferRequest, batcher_loop, gpu_worker_loop};

/// Aggregated results from one self-play iteration (many games).
pub(super) struct IterGameResults {
    pub samples: Vec<super::Sample>,
    pub wins: u32,
    pub losses: u32,
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

    fn aggregate(games: Vec<GameResult>, stats: &BatcherStats, num_tasks: usize) -> Self {
        let mut samples = Vec::new();
        let mut wins = 0u32;
        let mut losses = 0u32;
        let mut draws = 0u32;
        let mut total_turns = 0u32;
        let mut sum_turns_sq = 0u64;
        let mut min_game_length: Option<u32> = None;
        let mut max_game_length = 0u32;

        for game in games {
            let game_len = game.samples.last().map_or(0, |s| s.game_length);
            total_turns += game_len;
            sum_turns_sq += (game_len as u64) * (game_len as u64);
            min_game_length = Some(min_game_length.map_or(game_len, |m: u32| m.min(game_len)));
            max_game_length = max_game_length.max(game_len);
            if game.reward > 0.0 {
                wins += 1;
            } else if game.reward < 0.0 {
                losses += 1;
            } else {
                draws += 1;
            }
            samples.extend(game.samples);
        }

        let num_games = wins + losses + draws;
        let game_length_stddev = if num_games > 0 {
            let mean = total_turns as f64 / num_games as f64;
            let var = sum_turns_sq as f64 / num_games as f64 - mean * mean;
            var.max(0.0).sqrt()
        } else {
            0.0
        };

        Self {
            samples,
            wins,
            losses,
            draws,
            total_turns,
            min_game_length,
            max_game_length,
            game_length_stddev,
            num_tasks,
            total_batches: stats.batches.load(Relaxed),
            total_evals: stats.evals.load(Relaxed),
        }
    }
}

/// Shared state cloned once into each actor task.
struct TaskContext<G: Game> {
    request_tx: tokio::sync::mpsc::Sender<InferRequest>,
    encoder: Arc<dyn StateEncoder<G>>,
    games_remaining: Arc<AtomicU32>,
    completed: Arc<AtomicU32>,
    span: tracing::Span,
    batcher_stats: Arc<BatcherStats>,
    new_state: Arc<dyn Fn(u64) -> G + Send + Sync>,
    actor_config: Arc<ActorConfig>,
}

/// Run self-play games using async actor tasks with batched multi-GPU inference.
///
/// Shutdown: actors finish → drop mpsc sender → batcher exits → drops crossbeam
/// sender → workers exit.
pub(super) fn run_self_play_iteration<G>(
    evaluators: Vec<Arc<dyn Evaluator<G> + Sync>>,
    encoder: Arc<dyn StateEncoder<G>>,
    config: &TrainConfig,
    effective_sims: u32,
    iteration: usize,
    rng: &mut fastrand::Rng,
    new_state: &Arc<dyn Fn(u64) -> G + Send + Sync>,
) -> IterGameResults
where
    G: Game + 'static,
{
    let iter_label = format!(
        "iter {}/{} self-play (sims={})",
        iteration + 1,
        config.iterations,
        effective_sims,
    );
    let span = tracing::info_span!("self_play");
    span.pb_set_style(
        &indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len} {per_sec}  {msg}  [{elapsed} < {eta}]",
        )
        .unwrap(),
    );
    span.pb_set_length(config.games_per_iter as u64);
    span.pb_set_message(&iter_label);
    span.pb_start();

    let mcts_config = crate::mcts::Config {
        num_simulations: effective_sims,
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        leaf_batch_size: config.leaf_batch_size,
        ..Default::default()
    };
    let fast_sims = config.playout_cap_fast_sims.min(effective_sims);

    let concurrent_games = config.concurrent_games;
    let max_batch_size = config.max_batch_size;
    let num_actions = G::NUM_ACTIONS;

    // Actor → Batcher channel (tokio mpsc)
    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(2 * max_batch_size);

    let batcher_stats = Arc::new(BatcherStats::new());
    let ctx = Arc::new(TaskContext {
        request_tx,
        encoder,
        games_remaining: Arc::new(AtomicU32::new(config.games_per_iter as u32)),
        completed: Arc::new(AtomicU32::new(0)),
        span: span.clone(),
        batcher_stats: batcher_stats.clone(),
        new_state: new_state.clone(),
        actor_config: Arc::new(ActorConfig {
            explore_moves: config.explore_moves,
            playout_cap_full_prob: config.playout_cap_full_prob,
            playout_cap_fast_sims: fast_sims,
            effective_sims,
        }),
    });

    // Batcher → GPU workers channel (crossbeam SPMC, bounded to limit memory)
    let (work_tx, work_rx) = crossbeam_channel::bounded(2);

    let mut worker_handles = Vec::new();
    for evaluator in evaluators {
        let rx = work_rx.clone();
        worker_handles.push(std::thread::spawn(move || {
            gpu_worker_loop(
                rx,
                |features, batch_size, feature_size| {
                    evaluator.infer_features(features, batch_size, feature_size)
                },
                num_actions,
            );
        }));
    }
    drop(work_rx);

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
            let ctx = ctx.clone();
            let mcts_config = mcts_config.clone();

            join_set.spawn(async move {
                let mut rng = fastrand::Rng::with_seed(seed);
                let mut results = Vec::new();
                let mut search = Search::new((ctx.new_state)(rng.u64(..)), mcts_config);

                loop {
                    let claimed = ctx
                        .games_remaining
                        .fetch_update(Relaxed, Relaxed, |n| if n > 0 { Some(n - 1) } else { None });
                    if claimed.is_err() {
                        break;
                    }

                    search.reset((ctx.new_state)(rng.u64(..)));
                    let tx = ctx.request_tx.clone();
                    let game = play_game(
                        &mut search,
                        &ctx.actor_config,
                        &*ctx.encoder,
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
                                (resp.flat_policy_logits, resp.values)
                            }
                        },
                        &mut rng,
                    )
                    .await;

                    results.push(game);

                    let done = ctx.completed.fetch_add(1, Relaxed) + 1;
                    ctx.span.pb_set_position(done as u64);
                }

                results
            });
        }

        // Drop our copy so batcher sees disconnect when all tasks finish
        let tick_span = ctx.span.clone();
        let stats_ref = ctx.batcher_stats.clone();
        drop(ctx);

        // Periodic stats tick so the user sees progress before any game finishes
        let tick_label = iter_label.clone();
        let tick_start = std::time::Instant::now();
        let tick_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                let evals = stats_ref.evals.load(Relaxed);
                let avg_batch = stats_ref.avg_batch_size();
                let elapsed = tick_start.elapsed().as_secs_f64();
                let evals_per_sec = if elapsed > 0.0 {
                    evals as f64 / elapsed
                } else {
                    0.0
                };
                tick_span.pb_set_message(&format!(
                    "{tick_label}  avg_batch={avg_batch:.1}, evals/s={evals_per_sec:.0}"
                ));
            }
        });

        let mut all_results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            all_results.extend(result.unwrap());
        }
        tick_handle.abort();
        all_results
    });

    batcher_handle.join().unwrap();
    for h in worker_handles {
        h.join().unwrap();
    }

    drop(span);

    IterGameResults::aggregate(game_results, &batcher_stats, concurrent_games)
}
