//! Training coordinator: owns the main loop, shared types, and public API.
//!
//! Wires together all other `train::*` modules. Sets up persistent inference
//! servers and worker tasks once, then loops: collect samples → train →
//! swap evaluators → repeat. Also defines the types that cross module
//! boundaries (`Sample`, `TrainStepConfig`, `TrainMetrics`, `TrainableModel`).

mod burn_backend;
mod checkpoint;
pub mod config;
mod game;
pub mod inference;
mod metrics;
mod replay_buffer;
mod self_play;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::eval::Evaluator;
use crate::game::Game;
use crate::nn::StateEncoder;
use crate::utils::HumanDuration;

pub use burn_backend::{BurnTrainableModel, Device, InferBackend, TrainBackend, default_device};
pub use config::TrainConfig;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub struct Sample {
    pub features: Box<[f32]>,
    pub policy_target: Box<[f32]>,
    /// Value target: game outcome from current player's perspective (+1, -1, or 0).
    /// During self-play construction this temporarily holds the player's sign,
    /// then is multiplied by the terminal reward once the game ends.
    pub z: f32,
    /// Root WDL from current player's perspective (search-refined).
    pub q_wdl: [f32; 3],
    /// Whether this position used full search (for playout cap randomization)
    pub full_search: bool,
    /// Turn number when this sample was generated (1-indexed).
    pub move_number: u32,
    /// Total game length (backfilled after game ends).
    pub game_length: u32,
    /// Raw network value for this position (current player's perspective).
    pub network_value: f32,
    /// |Q_search - V_network| — how much search corrected the network's value.
    pub value_correction: f32,
    /// Std of Q values across visited root children (value head discriminative power).
    pub q_std: f32,
    /// Whether network's top-1 action matches MCTS selected action.
    pub prior_agrees: bool,
    /// Raw KL(MCTS target || network prior). Used for policy surprise weighting.
    pub policy_surprise: f32,
    /// Pre-normalized per-game surprise weight (1.0 if disabled).
    pub surprise_weight: f32,
    /// Short-term value targets (length = aux_value_horizons.len()).
    pub aux_targets: Box<[f32]>,
}

/// Per-iteration config passed to the model's train_step method.
pub struct TrainStepConfig {
    pub lr: f64,
    pub train_batch_size: usize,
    pub epochs: usize,
    /// Weight of Q in value target: 0.0 = pure Z, 1.0 = pure Q.
    pub q_weight: f32,
    /// Temperature for soft policy target (0.0 = disabled).
    pub soft_policy_temperature: f32,
    /// Weight of soft policy loss.
    pub soft_policy_weight: f32,
    /// Fraction of weight budget from surprise (0.0 = disabled).
    pub surprise_weight_fraction: f32,
    /// Per-head weight for auxiliary value losses.
    pub aux_value_weight: f32,
    /// Number of auxiliary value targets per sample.
    pub num_aux_targets: usize,
}

/// Metrics returned from a training step.
pub struct TrainMetrics {
    pub loss_policy_train: f32,
    pub loss_wdl_train: f32,
    pub loss_policy_val: f32,
    pub loss_wdl_val: f32,
    /// Soft policy loss (0.0 if disabled).
    pub loss_soft_policy_train: f32,
    pub loss_soft_policy_val: f32,
    /// Auxiliary value loss (0.0 if disabled).
    pub loss_aux_value_train: f32,
    pub loss_aux_value_val: f32,
    /// Per-horizon auxiliary value MSE (empty if no aux heads).
    pub aux_value_losses_train: Vec<f32>,
    pub aux_value_losses_val: Vec<f32>,
    /// Total gradient steps (optimizer updates) across all epochs
    pub gradient_steps: usize,
}

#[derive(Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub iteration: usize,
    pub rng_seed: u64,
}

// ---------------------------------------------------------------------------
// TrainableModel trait
// ---------------------------------------------------------------------------

pub trait TrainableModel<G: Game>: Send {
    fn evaluator(&self, encoder: Arc<dyn StateEncoder<G>>) -> Arc<dyn Evaluator<G> + Sync>;
    fn evaluators(
        &self,
        encoder: Arc<dyn StateEncoder<G>>,
        count: usize,
    ) -> Vec<Arc<dyn Evaluator<G> + Sync>> {
        (0..count)
            .map(|_| self.evaluator(encoder.clone()))
            .collect()
    }
    fn train_step(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> TrainMetrics;
    fn save(&self, dir: &Path, iteration: usize);
    fn load(&mut self, dir: &Path, iteration: usize);
    /// Load only model weights (no optimizer state). Use for inference.
    fn load_weights(&mut self, dir: &Path, iteration: usize);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linear ramp from 0 to 1 over warmup_iters, uncapped.
fn warmup_frac(config: &TrainConfig, iteration: usize) -> f64 {
    if config.warmup_iters == 0 {
        return 1.0;
    }
    (iteration as f64 / config.warmup_iters as f64).min(1.0)
}

/// Q-weight for value target mixing: ramps from 0 to q_weight_max.
fn q_weight(config: &TrainConfig, iteration: usize) -> f64 {
    warmup_frac(config, iteration) * config.q_weight_max as f64
}

fn progressive_sims(config: &TrainConfig, iteration: usize) -> u32 {
    let start = config.mcts_sims_start;
    if start >= config.mcts_sims {
        return config.mcts_sims;
    }
    let t = warmup_frac(config, iteration);
    let sims = start as f64 + t * (config.mcts_sims - start) as f64;
    (sims.round() as u32).max(1)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_training<G>(
    config: TrainConfig,
    model: &mut dyn TrainableModel<G>,
    encoder: Arc<dyn StateEncoder<G>>,
    new_state: impl Fn(u64) -> G + Send + Sync + 'static,
) where
    G: Game + 'static,
{
    let new_state: Arc<dyn Fn(u64) -> G + Send + Sync> = Arc::new(new_state);
    crate::init_logging();

    let (mut rng, start_iteration) = checkpoint::resume_if_requested(&config, model);
    let run_dir = checkpoint::setup_run_dir(&config);
    info!(path = %run_dir.display(), "run directory");
    checkpoint::save_config(&run_dir, &config);

    let mut csv = metrics::CsvLogger::open(&run_dir, start_iteration);
    let training_start = Instant::now();

    // === Setup (once) ===

    // 1. Create PauseControl for GPU 0
    let pause_control = Arc::new(inference::PauseControl::new());

    // 2. Create InferenceServer per GPU (GPU 0 gets pause)
    let evaluators = model.evaluators(encoder.clone(), config.inference_workers);
    let mut servers = Vec::with_capacity(evaluators.len());
    for (i, eval) in evaluators.into_iter().enumerate() {
        let pause = if i == 0 {
            Some(pause_control.clone())
        } else {
            None
        };
        servers.push(inference::InferenceServer::start::<G>(
            eval,
            config.inference_batch_size,
            pause,
        ));
    }

    // 3. Create tokio runtime (persistent)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    // 4. Create per-worker channels + unbounded result channel
    let num_workers = config.concurrent_games;
    let (result_tx, mut result_rx) = tokio::sync::mpsc::unbounded_channel();

    // Per-worker work channels (bounded capacity 2 for backpressure)
    let mut work_txs: Vec<tokio::sync::mpsc::Sender<self_play::WorkItem>> =
        Vec::with_capacity(num_workers);

    let mcts_config = crate::mcts::Config {
        num_simulations: config.mcts_sims, // overridden per-game by effective_sims
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
        leaf_batch_size: config.leaf_batch_size,
        ..Default::default()
    };

    let actor_config = Arc::new(game::ActorConfig {
        explore_moves: config.explore_moves,
        playout_cap_full_prob: config.playout_cap_full_prob,
        playout_cap_fast_sims_base: config.playout_cap_fast_sims,
        num_aux_targets: config.aux_value_horizons.len(),
        aux_value_horizons: config.aux_value_horizons.clone(),
        surprise_weight_fraction: config.surprise_weight_fraction,
    });

    // 5. Spawn persistent worker tasks (round-robin GPU assignment)
    let mut worker_handles = Vec::with_capacity(num_workers);
    for i in 0..num_workers {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        work_txs.push(tx);

        let server_idx = i % servers.len();
        let request_tx = servers[server_idx].sender();
        let result_tx = result_tx.clone();
        let encoder = encoder.clone();
        let actor_config = actor_config.clone();
        let mcts_config = mcts_config.clone();
        let new_state = new_state.clone();
        let worker_seed = rng.u64(..);

        worker_handles.push(rt.spawn(self_play::worker_loop::<G>(
            rx,
            result_tx,
            request_tx,
            encoder,
            actor_config,
            mcts_config,
            new_state,
            worker_seed,
        )));
    }
    // Drop our copy so unbounded channel closes when all workers finish
    drop(result_tx);

    // 6. Create ReplayBuffer
    let mut replay_buffer = replay_buffer::ReplayBuffer::new(config.replay_buffer_samples);

    // === Main loop ===
    for iteration in start_iteration..config.iterations {
        let iter_start = Instant::now();
        let effective_sims = progressive_sims(&config, iteration);

        // Progress bar for self-play collection
        let sp_span = tracing::info_span!("self_play");
        sp_span.pb_set_style(
            &indicatif::ProgressStyle::with_template(
                "{bar:40.cyan/dim} {pos}/{len} samples  {msg}  [{elapsed} < {eta}]",
            )
            .unwrap(),
        );
        sp_span.pb_set_length(config.train_samples_per_iter as u64);
        sp_span.pb_set_message(&format!(
            "iter {}/{} sims={}",
            iteration + 1,
            config.iterations,
            effective_sims
        ));
        sp_span.pb_start();

        let stats_start_evals: u64 = servers.iter().map(|s| s.stats().evals.load(Relaxed)).sum();

        // Keep work queues fed
        refill_work_queues(&work_txs, effective_sims, &mut rng);

        // Collect results until sample threshold
        let mut fresh_samples = 0usize;
        let mut collected_games = Vec::new();
        while fresh_samples < config.train_samples_per_iter {
            match result_rx.blocking_recv() {
                Some(self_play::WorkResult::SelfPlay(mut record)) => {
                    fresh_samples += record.samples.len();
                    record.iteration_analyzed = iteration;
                    collected_games.push(record);
                    sp_span.pb_set_position(fresh_samples as u64);
                    let elapsed_secs = iter_start.elapsed().as_secs_f64();
                    if elapsed_secs > 0.5 {
                        let evals_now: u64 =
                            servers.iter().map(|s| s.stats().evals.load(Relaxed)).sum();
                        let evals_per_sec = (evals_now - stats_start_evals) as f64 / elapsed_secs;
                        sp_span.pb_set_message(&format!(
                            "iter {}/{} sims={} | {:.0} evals/s",
                            iteration + 1,
                            config.iterations,
                            effective_sims,
                            evals_per_sec
                        ));
                    }
                }
                Some(self_play::WorkResult::Reanalyze { game_id, samples }) => {
                    replay_buffer.replace_samples(game_id, samples, iteration);
                }
                None => panic!("all workers died unexpectedly"),
            }
            // Keep queues fed while collecting
            refill_work_queues(&work_txs, effective_sims, &mut rng);
        }
        // No drain. In-flight results buffer in the channel for next iteration.
        let evals_now: u64 = servers.iter().map(|s| s.stats().evals.load(Relaxed)).sum();
        let evals_per_sec =
            (evals_now - stats_start_evals) as f64 / iter_start.elapsed().as_secs_f64();
        sp_span.pb_set_finish_message(&format!(
            "iter {}/{} sims={} | {:.0} evals/s  [{}]",
            iteration + 1,
            config.iterations,
            effective_sims,
            evals_per_sec,
            HumanDuration(iter_start.elapsed()),
        ));
        drop(sp_span);

        let self_play_elapsed = iter_start.elapsed();
        let games_done = collected_games.len();
        let samples_this_iter = collected_games
            .iter()
            .map(|g| g.samples.len())
            .sum::<usize>();

        // Compute game stats before pushing to buffer
        let sp = self_play::IterGameResults::aggregate(&collected_games);

        // Push new games to replay buffer
        replay_buffer.push_games(collected_games);

        // Submit reanalyze work (phased)
        if config.reanalyze_games > 0 {
            let reanalyze_items =
                replay_buffer.select_for_reanalyze(config.reanalyze_games, iteration, &mut rng);
            let mut next_worker = 0;
            for (game_id, seed, actions, reward) in reanalyze_items {
                let item = self_play::WorkItem::Reanalyze {
                    game_id,
                    seed,
                    actions,
                    reward,
                    effective_sims,
                };
                // Round-robin across workers
                let _ = work_txs[next_worker].blocking_send(item);
                next_worker = (next_worker + 1) % work_txs.len();
            }
        }

        // Stats from fresh samples
        let fresh_sample_refs: Vec<&Sample> = replay_buffer
            .games()
            .iter()
            .rev()
            .take(games_done)
            .flat_map(|g| g.samples.iter())
            .collect();
        let stats = metrics::compute_iter_stats(&fresh_sample_refs);

        // Pause GPU 0 → train → swap evaluators → resume
        let train_start = Instant::now();
        pause_control.pause();

        let mut all_samples: Vec<&Sample> = replay_buffer.all_samples();
        let q_weight = q_weight(&config, iteration + 1) as f32;
        fastrand::shuffle(&mut all_samples);
        let step_cfg = TrainStepConfig {
            lr: config.lr,
            train_batch_size: config.train_batch_size,
            epochs: config.epochs,
            q_weight,
            soft_policy_temperature: config.soft_policy_temperature,
            soft_policy_weight: config.soft_policy_weight,
            surprise_weight_fraction: config.surprise_weight_fraction,
            aux_value_weight: config.aux_value_weight,
            num_aux_targets: config.aux_value_horizons.len(),
        };
        let train_metrics = model.train_step(&all_samples, &step_cfg);

        // Swap evaluators on all servers
        let new_evals = model.evaluators(encoder.clone(), config.inference_workers);
        for (server, eval) in servers.iter().zip(new_evals) {
            server.swap_infer_fn(inference::make_infer_fn::<G>(eval));
        }

        pause_control.resume();
        let train_elapsed = train_start.elapsed();

        // Checkpoint
        let iter_num = iteration + 1;
        let is_last = iter_num == config.iterations;
        if is_last || iter_num % config.checkpoint_interval == 0 {
            checkpoint::save_checkpoint(model, &run_dir, iter_num, &mut rng);
        }

        // Timing / ETA
        let total_elapsed = training_start.elapsed();
        let iters_done = iteration + 1;
        let iters_this_session = iters_done - start_iteration;
        let iters_remaining = config.iterations - iters_done;
        let avg_iter_time = total_elapsed / iters_this_session as u32;
        let eta = avg_iter_time * iters_remaining as u32;
        let avg_turns = if sp.wins + sp.losses + sp.draws > 0 {
            sp.total_turns / (sp.wins + sp.losses + sp.draws)
        } else {
            0
        };
        let num_games_total = sp.wins + sp.losses + sp.draws;

        info!(
            "iter {}/{}: {} games (W:{} L:{} D:{}, avg {} turns) {} samples (buffer: {} samples, {} games) | self-play {}, train {} | total {}, ETA {}",
            iters_done,
            config.iterations,
            num_games_total,
            sp.wins,
            sp.losses,
            sp.draws,
            avg_turns,
            all_samples.len(),
            replay_buffer.total_samples(),
            replay_buffer.len(),
            HumanDuration(self_play_elapsed),
            HumanDuration(train_elapsed),
            HumanDuration(total_elapsed),
            HumanDuration(eta),
        );

        // CSV
        let min_game_length = sp.min_game_length.unwrap_or(0);
        let avg_game_length = if num_games_total > 0 {
            sp.total_turns as f64 / num_games_total as f64
        } else {
            0.0
        };
        let aux_t = &train_metrics.aux_value_losses_train;
        let aux_v = &train_metrics.aux_value_losses_val;
        csv.write_row(&metrics::CsvRow {
            // Core training
            iteration: iters_done,
            loss_policy_train: train_metrics.loss_policy_train,
            loss_wdl_train: train_metrics.loss_wdl_train,
            loss_policy_val: train_metrics.loss_policy_val,
            loss_wdl_val: train_metrics.loss_wdl_val,
            loss_soft_policy_train: train_metrics.loss_soft_policy_train,
            loss_soft_policy_val: train_metrics.loss_soft_policy_val,
            loss_aux_value_train: train_metrics.loss_aux_value_train,
            loss_aux_value_val: train_metrics.loss_aux_value_val,
            loss_aux_value_0_train: aux_t.first().copied().unwrap_or(0.0),
            loss_aux_value_0_val: aux_v.first().copied().unwrap_or(0.0),
            loss_aux_value_1_train: aux_t.get(1).copied().unwrap_or(0.0),
            loss_aux_value_1_val: aux_v.get(1).copied().unwrap_or(0.0),
            loss_aux_value_2_train: aux_t.get(2).copied().unwrap_or(0.0),
            loss_aux_value_2_val: aux_v.get(2).copied().unwrap_or(0.0),
            loss_aux_value_3_train: aux_t.get(3).copied().unwrap_or(0.0),
            loss_aux_value_3_val: aux_v.get(3).copied().unwrap_or(0.0),
            gradient_steps: train_metrics.gradient_steps,
            // Self-play game stats
            game_length_avg: avg_game_length,
            game_length_stddev: sp.game_length_stddev,
            game_length_min: min_game_length,
            game_length_max: sp.max_game_length,
            game_wins: sp.wins,
            game_losses: sp.losses,
            game_draws: sp.draws,
            // Policy diagnostics
            policy_entropy_avg: stats.policy_entropy_avg,
            policy_max_prob_avg: stats.policy_max_prob_avg,
            policy_entropy_high_branch_avg: stats.policy_entropy_high_branch_avg,
            policy_max_prob_high_branch_avg: stats.policy_max_prob_high_branch_avg,
            policy_agreement_avg: stats.policy_agreement_avg,
            policy_agreement_high_branch_avg: stats.policy_agreement_high_branch_avg,
            policy_surprise_avg: stats.policy_surprise_avg,
            // Value diagnostics
            value_z_avg: stats.value_z_avg,
            value_q_avg: stats.value_q_avg,
            value_z_stddev: stats.value_z_stddev,
            value_q_stddev: stats.value_q_stddev,
            value_correction_avg: stats.value_correction_avg,
            value_correction_high_branch_avg: stats.value_correction_high_branch_avg,
            value_q_spread_avg: stats.value_q_spread_avg,
            value_q_spread_high_branch_avg: stats.value_q_spread_high_branch_avg,
            value_error_early_avg: stats.value_error_early_avg,
            value_error_mid_avg: stats.value_error_mid_avg,
            value_error_late_avg: stats.value_error_late_avg,
            value_network_stddev: stats.value_network_stddev,
            // Config/infra
            lr: config.lr,
            q_weight,
            mcts_sims: effective_sims,
            replay_samples: all_samples.len(),
            samples_iter: samples_this_iter,
            reanalyze_games: config.reanalyze_games,
            replay_buffer_games: replay_buffer.len(),
            time_selfplay_secs: self_play_elapsed.as_secs_f64(),
            time_train_secs: train_elapsed.as_secs_f64(),
        });
    }

    // === Shutdown ===
    for tx in &work_txs {
        let _ = tx.blocking_send(self_play::WorkItem::Shutdown);
    }
    drop(work_txs);
    rt.block_on(async {
        for h in worker_handles {
            let _ = h.await;
        }
    });
    for server in servers {
        server.shutdown();
    }
}

fn refill_work_queues(
    work_txs: &[tokio::sync::mpsc::Sender<self_play::WorkItem>],
    effective_sims: u32,
    rng: &mut fastrand::Rng,
) {
    for tx in work_txs {
        // Fill each worker's queue until full (capacity 2)
        loop {
            match tx.try_send(self_play::WorkItem::SelfPlay {
                seed: rng.u64(..),
                effective_sims,
            }) {
                Ok(()) => {}
                Err(_) => break,
            }
        }
    }
}
