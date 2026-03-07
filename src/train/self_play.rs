use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

use tokio::sync::mpsc;

use crate::eval::{Evaluation, Evaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, PendingEval, Search, Step};
use crate::nn::StateEncoder;
use crate::player::Player;

use super::{Sample, TrainConfig};

/// Shared atomic counters updated by the batcher, read by tasks for live stats.
pub(super) struct BatcherStats {
    pub batches: AtomicU64,
    pub evals: AtomicU64,
}

impl BatcherStats {
    pub fn new() -> Self {
        Self {
            batches: AtomicU64::new(0),
            evals: AtomicU64::new(0),
        }
    }

    pub fn avg_batch_size(&self) -> f64 {
        let b = self.batches.load(Relaxed);
        let e = self.evals.load(Relaxed);
        if b == 0 { 0.0 } else { e as f64 / b as f64 }
    }
}

/// Result of a single self-play game.
pub(super) struct GameResult {
    pub samples: Vec<Sample>,
    pub winner: Option<Player>,
    pub num_turns: u32,
}

/// Aggregated results from one self-play iteration (many games).
pub(super) struct IterGameResults {
    pub samples: Vec<Sample>,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub total_turns: u32,
    pub min_game_length: Option<u32>,
    pub max_game_length: u32,
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

/// A leaf-node evaluation request sent from an async game task to the batcher.
pub(super) struct EvalRequest<G: Game> {
    pub pending: PendingEval<G>,
    pub response_tx: tokio::sync::oneshot::Sender<(Evaluation, PendingEval<G>)>,
}

/// Sender half of the bounded eval request channel.
pub(super) type EvalSender<G> = mpsc::Sender<EvalRequest<G>>;

fn sample_from_policy(policy: &[f32], rng: &mut fastrand::Rng) -> usize {
    let mut roll = rng.f32();
    for (i, &p) in policy.iter().enumerate() {
        roll -= p;
        if roll <= 0.0 {
            return i;
        }
    }
    // Fallback: return argmax
    policy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Async task: plays sequential games, sending leaf evals to the batcher.
#[allow(clippy::too_many_arguments)]
async fn game_task<G: Game, E: StateEncoder<G>>(
    request_tx: mpsc::Sender<EvalRequest<G>>,
    games_remaining: Arc<AtomicU32>,
    completed: Arc<AtomicU32>,
    pb: Arc<indicatif::ProgressBar>,
    batcher_stats: Arc<BatcherStats>,
    config: Arc<TrainConfig>,
    base_config: Config,
    effective_sims: u32,
    fast_sims: u32,
    new_state: Arc<dyn Fn(&mut fastrand::Rng) -> G + Send + Sync>,
    seed: u64,
) -> Vec<GameResult> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut results = Vec::new();
    let mut actions_buf = Vec::new();

    loop {
        // Claim a game via atomic decrement
        let claimed = games_remaining
            .fetch_update(Relaxed, Relaxed, |n| if n > 0 { Some(n - 1) } else { None });
        if claimed.is_err() {
            break;
        }

        let mut search = Search::new(new_state(&mut rng));
        let mut samples: Vec<Sample> = Vec::new();
        let mut turn_count: u32 = 0;
        let mut last_player: Option<Player> = None;

        loop {
            // Resolve chance nodes
            if let Some(action) = search.state().sample_chance(&mut rng) {
                search.apply_action(action);
                continue;
            }

            // Terminal check
            if let Status::Terminal(reward) = search.state().status() {
                for s in &mut samples {
                    s.z *= reward;
                }
                let winner = if reward > 0.0 {
                    Some(Player::One)
                } else if reward < 0.0 {
                    Some(Player::Two)
                } else {
                    None
                };
                results.push(GameResult {
                    samples: std::mem::take(&mut samples),
                    winner,
                    num_turns: turn_count,
                });
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
                break;
            }

            let current = match search.state().status() {
                Status::Ongoing(p) => p,
                Status::Terminal(_) => unreachable!(),
            };

            // Track turn count via player changes
            if last_player != Some(current) {
                turn_count += 1;
                last_player = Some(current);
            }

            // Skip forced moves (single legal action)
            actions_buf.clear();
            search.state().legal_actions(&mut actions_buf);
            if actions_buf.len() == 1 {
                search.apply_action(actions_buf[0]);
                continue;
            }

            // Playout cap randomization
            let is_full_search = rng.f32() < config.playout_cap_full_prob;
            let move_config = Config {
                num_simulations: if is_full_search {
                    effective_sims
                } else {
                    fast_sims
                },
                ..base_config.clone()
            };

            // Encode features before search
            let mut features_buf = Vec::with_capacity(E::FEATURE_SIZE);
            E::encode(search.state(), &mut features_buf);

            // MCTS search with eval requests sent to batcher
            let mut step = search.run(&move_config, &mut rng);
            let result = loop {
                match step {
                    Step::NeedsEval(pendings) => {
                        let mut receivers = Vec::with_capacity(pendings.len());
                        for pending in pendings {
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            request_tx
                                .send(EvalRequest {
                                    pending,
                                    response_tx: tx,
                                })
                                .await
                                .unwrap();
                            receivers.push(rx);
                        }
                        let mut evals = Vec::with_capacity(receivers.len());
                        for rx in receivers {
                            evals.push(rx.await.unwrap());
                        }
                        step = search.supply(evals, &mut rng);
                    }
                    Step::Done(result) => break result,
                }
            };

            // Create training sample
            let q = result.value * current.sign();
            let chosen = if turn_count <= config.explore_moves {
                sample_from_policy(&result.policy, &mut rng)
            } else {
                result.selected_action
            };

            samples.push(Sample {
                features: features_buf.into_boxed_slice(),
                policy_target: result.policy.into_boxed_slice(),
                z: current.sign(),
                q,
                full_search: is_full_search,
            });

            search.apply_action(chosen);
        }
    }

    results
}

/// Batcher thread: collects eval requests and runs batched NN inference.
pub(super) fn batcher_loop<G: Game, Ev: Evaluator<G>>(
    request_rx: &mut mpsc::Receiver<EvalRequest<G>>,
    evaluator: &Ev,
    rng: &mut fastrand::Rng,
    max_batch_size: usize,
    stats: &BatcherStats,
) {
    let mut batch: Vec<EvalRequest<G>> = Vec::with_capacity(max_batch_size);

    loop {
        // Block for the first request (None = all senders dropped)
        let first = match request_rx.blocking_recv() {
            Some(req) => req,
            None => return,
        };
        batch.push(first);

        // Drain additional requests without blocking, up to max_batch_size
        while batch.len() < max_batch_size {
            match request_rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        stats.evals.fetch_add(batch.len() as u64, Relaxed);
        stats.batches.fetch_add(1, Relaxed);

        // Batched evaluation
        let states: Vec<&G> = batch.iter().map(|r| &r.pending.state).collect();
        let evals = evaluator.evaluate_batch(&states, rng);

        // Send results back to tasks
        for (req, eval) in batch.drain(..).zip(evals) {
            // Ignore send errors (task may have been cancelled)
            let _ = req.response_tx.send((eval, req.pending));
        }
    }
}

/// Run self-play games using async tasks with batched NN inference.
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

    let base_config = Config {
        num_simulations: effective_sims,
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        leaf_batch_size: config.leaf_batch_size,
    };
    let fast_sims = config.playout_cap_fast_sims.min(effective_sims);

    let concurrent_games = config.concurrent_games;
    let max_batch_size = config.max_batch_size;
    let games_remaining = Arc::new(AtomicU32::new(config.games_per_iter as u32));
    let completed = Arc::new(AtomicU32::new(0));
    let batcher_stats = Arc::new(BatcherStats::new());
    let config = Arc::new(config.clone());
    let pb = Arc::new(pb);

    let (request_tx, mut request_rx) = mpsc::channel(2 * max_batch_size);

    // Seed batcher rng before spawning
    let batcher_seed = rng.u64(..);

    // Batcher on a plain std::thread (owns evaluator)
    let batcher_stats_ref = batcher_stats.clone();
    let batcher_handle = std::thread::spawn(move || {
        let mut batcher_rng = fastrand::Rng::with_seed(batcher_seed);
        batcher_loop(
            &mut request_rx,
            &evaluator,
            &mut batcher_rng,
            max_batch_size,
            &batcher_stats_ref,
        );
    });

    // Tokio runtime for async game tasks
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let game_results: Vec<GameResult> = rt.block_on(async {
        let mut join_set = tokio::task::JoinSet::new();

        for _ in 0..concurrent_games {
            let seed = rng.u64(..);
            join_set.spawn(game_task::<G, E>(
                request_tx.clone(),
                games_remaining.clone(),
                completed.clone(),
                pb.clone(),
                batcher_stats.clone(),
                config.clone(),
                base_config.clone(),
                effective_sims,
                fast_sims,
                new_state.clone(),
                seed,
            ));
        }

        // Drop our sender so batcher sees disconnect when all tasks finish
        drop(request_tx);

        let mut all_results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            all_results.extend(result.unwrap());
        }
        all_results
    });

    // Batcher thread exits once all senders are dropped
    batcher_handle.join().unwrap();

    pb.finish();

    // Aggregate results
    let mut samples = Vec::new();
    let mut p1_wins = 0u32;
    let mut p2_wins = 0u32;
    let mut draws = 0u32;
    let mut total_turns = 0u32;
    let mut min_game_length: Option<u32> = None;
    let mut max_game_length = 0u32;

    for game in game_results {
        total_turns += game.num_turns;
        min_game_length =
            Some(min_game_length.map_or(game.num_turns, |m: u32| m.min(game.num_turns)));
        max_game_length = max_game_length.max(game.num_turns);
        match game.winner {
            Some(Player::One) => p1_wins += 1,
            Some(Player::Two) => p2_wins += 1,
            None => draws += 1,
        }
        samples.extend(game.samples);
    }

    IterGameResults {
        samples,
        p1_wins,
        p2_wins,
        draws,
        total_turns,
        min_game_length,
        max_game_length,
        num_tasks: concurrent_games,
        total_batches: batcher_stats.batches.load(Relaxed),
        total_evals: batcher_stats.evals.load(Relaxed),
    }
}
