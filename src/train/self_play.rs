use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

use tokio::sync::mpsc;

use crate::eval::{Evaluation, Evaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, Step};
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

/// A pre-encoded inference request sent from an async game task to the batcher.
/// Game-independent: carries raw features, not game states.
pub(super) struct InferRequest {
    pub features: Vec<f32>,
    pub response_tx: tokio::sync::oneshot::Sender<InferResponse>,
}

/// Raw inference result returned from the batcher to a game task.
pub(super) struct InferResponse {
    pub policy_logits: Vec<f32>,
    pub value: f32,
}

/// Sender half of the bounded inference request channel.
pub(super) type InferSender = mpsc::Sender<InferRequest>;

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
/// Encodes features locally before sending — batcher only does GPU work.
#[allow(clippy::too_many_arguments)]
async fn game_task<G: Game, E: StateEncoder<G>>(
    request_tx: InferSender,
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
    let mut encode_buf = Vec::with_capacity(E::FEATURE_SIZE);

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
                    s.game_length = turn_count;
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
                        let mut pending_store = Vec::with_capacity(pendings.len());
                        for pending in pendings {
                            // Pre-encode features in the task
                            let sign = match pending.state.status() {
                                Status::Ongoing(p) => p.sign(),
                                Status::Terminal(_) => 1.0,
                            };
                            encode_buf.clear();
                            E::encode(&pending.state, &mut encode_buf);
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            request_tx
                                .send(InferRequest {
                                    features: encode_buf.clone(),
                                    response_tx: tx,
                                })
                                .await
                                .unwrap();
                            receivers.push((rx, sign));
                            pending_store.push(pending);
                        }
                        let mut evals = Vec::with_capacity(receivers.len());
                        for ((rx, sign), pending) in receivers.into_iter().zip(pending_store) {
                            let resp = rx.await.unwrap();
                            let eval = Evaluation {
                                policy_logits: resp.policy_logits,
                                value: resp.value * sign,
                            };
                            evals.push((eval, pending));
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

            let value_correction = (result.value - result.network_value).abs();
            let q_std = if result.children_q.len() >= 2 {
                let mean = result.children_q.iter().map(|&(_, q)| q).sum::<f32>()
                    / result.children_q.len() as f32;
                let var = result
                    .children_q
                    .iter()
                    .map(|&(_, q)| (q - mean).powi(2))
                    .sum::<f32>()
                    / result.children_q.len() as f32;
                var.sqrt()
            } else {
                0.0
            };
            let prior_agrees = result.prior_top1_action == result.selected_action;

            samples.push(Sample {
                features: features_buf.into_boxed_slice(),
                policy_target: result.policy.into_boxed_slice(),
                z: current.sign(),
                q,
                full_search: is_full_search,
                move_number: turn_count,
                game_length: 0, // backfilled at game end
                network_value: result.network_value * current.sign(),
                value_correction,
                q_std,
                prior_agrees,
            });

            search.apply_action(chosen);
        }
    }

    results
}

/// Batcher thread: collects pre-encoded feature vectors and runs batched NN inference.
/// Game-independent — only touches raw floats and the provided inference function.
pub(super) fn batcher_loop(
    request_rx: &mut mpsc::Receiver<InferRequest>,
    infer_fn: impl Fn(Vec<f32>, usize) -> (Vec<f32>, Vec<f32>),
    num_actions: usize,
    max_batch_size: usize,
    stats: &BatcherStats,
) {
    use std::time::Instant;

    let mut batch: Vec<InferRequest> = Vec::with_capacity(max_batch_size);

    let mut total_wait_us = 0u64;
    let mut total_infer_us = 0u64;
    let mut total_overhead_us = 0u64;
    let mut num_cycles = 0u64;
    let mut batch_size_sum = 0u64;
    let mut max_batch_seen = 0usize;

    loop {
        // Block for the first request (None = all senders dropped)
        let t_wait = Instant::now();
        let first = match request_rx.blocking_recv() {
            Some(req) => req,
            None => break,
        };
        total_wait_us += t_wait.elapsed().as_micros() as u64;

        let feature_size = first.features.len();
        batch.push(first);

        // Drain additional requests without blocking, up to max_batch_size
        while batch.len() < max_batch_size {
            match request_rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        let bs = batch.len();
        batch_size_sum += bs as u64;
        max_batch_seen = max_batch_seen.max(bs);

        stats.evals.fetch_add(bs as u64, Relaxed);
        stats.batches.fetch_add(1, Relaxed);

        // Collect flat features from pre-encoded requests
        let t_overhead = Instant::now();
        let mut flat_features = Vec::with_capacity(bs * feature_size);
        for req in &batch {
            flat_features.extend_from_slice(&req.features);
        }
        total_overhead_us += t_overhead.elapsed().as_micros() as u64;

        // Batched inference (GPU forward pass only — no encoding)
        let t_infer = Instant::now();
        let (flat_logits, flat_values) = infer_fn(flat_features, bs);
        total_infer_us += t_infer.elapsed().as_micros() as u64;

        // Send results back to tasks
        let t_send = Instant::now();
        for (i, req) in batch.drain(..).enumerate() {
            let logits_start = i * num_actions;
            let policy_logits = flat_logits[logits_start..logits_start + num_actions].to_vec();
            let value = flat_values[i];
            let _ = req.response_tx.send(InferResponse {
                policy_logits,
                value,
            });
        }
        total_overhead_us += t_send.elapsed().as_micros() as u64;

        num_cycles += 1;
    }

    if num_cycles > 0 {
        let avg_batch = batch_size_sum as f64 / num_cycles as f64;
        let total_us = total_wait_us + total_infer_us + total_overhead_us;
        let wait_pct = total_wait_us as f64 / total_us as f64 * 100.0;
        let infer_pct = total_infer_us as f64 / total_us as f64 * 100.0;
        let overhead_pct = total_overhead_us as f64 / total_us as f64 * 100.0;
        let total_ms = total_us as f64 / 1000.0;
        eprintln!(
            "batcher: {num_cycles} cycles, avg_batch={avg_batch:.1}, max_batch={max_batch_seen}, \
             total={total_ms:.0}ms | wait={wait_pct:.1}% infer={infer_pct:.1}% overhead={overhead_pct:.1}%"
        );
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
    let num_actions = G::NUM_ACTIONS;
    let feature_size = E::FEATURE_SIZE;
    let games_remaining = Arc::new(AtomicU32::new(config.games_per_iter as u32));
    let completed = Arc::new(AtomicU32::new(0));
    let batcher_stats = Arc::new(BatcherStats::new());
    let config = Arc::new(config.clone());
    let pb = Arc::new(pb);

    let (request_tx, mut request_rx) = mpsc::channel(2 * max_batch_size);

    // Batcher on a plain std::thread — receives pre-encoded features, runs GPU inference
    let batcher_stats_ref = batcher_stats.clone();
    let batcher_handle = std::thread::spawn(move || {
        batcher_loop(
            &mut request_rx,
            |features, batch_size| evaluator.infer_features(features, batch_size, feature_size),
            num_actions,
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
    let mut sum_turns_sq = 0u64;
    let mut min_game_length: Option<u32> = None;
    let mut max_game_length = 0u32;
    let mut num_games = 0u32;

    for game in game_results {
        total_turns += game.num_turns;
        sum_turns_sq += (game.num_turns as u64) * (game.num_turns as u64);
        num_games += 1;
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
