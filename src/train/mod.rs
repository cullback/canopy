mod burn_backend;
mod checkpoint;
pub mod config;
mod game;
pub mod inference;
mod metrics;
mod self_play;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::info;

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
    /// Root Q from current player's perspective, in [-1, 1]
    pub q: f32,
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
    pub loss_value_train: f32,
    pub loss_policy_val: f32,
    pub loss_value_val: f32,
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
    let mut replay_buffer: VecDeque<Vec<Sample>> = VecDeque::new();
    let training_start = Instant::now();

    for iteration in start_iteration..config.iterations {
        let iter_start = Instant::now();
        let effective_sims = progressive_sims(&config, iteration);
        let evaluators = model.evaluators(encoder.clone(), config.inference_workers);

        // Self-play
        let sp = self_play::run_self_play_iteration(
            evaluators,
            encoder.clone(),
            &config,
            effective_sims,
            iteration,
            &mut rng,
            &new_state,
        );
        let self_play_elapsed = iter_start.elapsed();
        let games_done = sp.wins + sp.losses + sp.draws;
        let samples_this_iter = sp.samples.len();
        let num_tasks = sp.num_tasks;
        let avg_batch_size = sp.avg_batch_size();

        // Stats
        let stats = metrics::compute_iter_stats(&sp.samples);

        // Train
        let train_start = Instant::now();
        replay_buffer.push_back(sp.samples);
        while replay_buffer.len() > config.replay_window {
            replay_buffer.pop_front();
        }
        let mut samples: Vec<&Sample> = replay_buffer.iter().flat_map(|v| v.iter()).collect();
        let q_weight = q_weight(&config, iteration + 1) as f32;
        fastrand::shuffle(&mut samples);
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
        let train_metrics = model.train_step(&samples, &step_cfg);
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
        let avg_turns = if games_done > 0 {
            sp.total_turns / games_done
        } else {
            0
        };

        info!(
            "iter {}/{}: {} games (W:{} L:{} D:{}, avg {} turns) {} samples, entropy={:.6} | tasks={}, avg_batch={:.1} | self-play {}, train {} | total {}, ETA {}",
            iters_done,
            config.iterations,
            games_done,
            sp.wins,
            sp.losses,
            sp.draws,
            avg_turns,
            samples.len(),
            stats.policy_entropy_avg,
            num_tasks,
            avg_batch_size,
            HumanDuration(self_play_elapsed),
            HumanDuration(train_elapsed),
            HumanDuration(total_elapsed),
            HumanDuration(eta),
        );

        // CSV
        let min_game_length = sp.min_game_length.unwrap_or(0);
        let avg_game_length = if games_done > 0 {
            sp.total_turns as f64 / games_done as f64
        } else {
            0.0
        };
        let aux_t = &train_metrics.aux_value_losses_train;
        let aux_v = &train_metrics.aux_value_losses_val;
        csv.write_row(&metrics::CsvRow {
            // Core training
            iteration: iters_done,
            loss_policy_train: train_metrics.loss_policy_train,
            loss_value_train: train_metrics.loss_value_train,
            loss_policy_val: train_metrics.loss_policy_val,
            loss_value_val: train_metrics.loss_value_val,
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
            replay_samples: samples.len(),
            samples_iter: samples_this_iter,
            time_selfplay_secs: self_play_elapsed.as_secs_f64(),
            time_train_secs: train_elapsed.as_secs_f64(),
        });
    }
}
