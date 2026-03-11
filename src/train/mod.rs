mod benchmark;
mod burn_backend;
mod checkpoint;
mod game;
mod inference;
mod metrics;
mod self_play;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::eval::Evaluator;
use crate::game::Game;
use crate::nn::StateEncoder;
use crate::utils::HumanDuration;

pub use burn_backend::{BurnTrainableModel, Device, InferBackend, default_device};

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
}

#[derive(Clone, Serialize)]
pub struct TrainConfig {
    // -- Infrastructure --
    /// Total training iterations (self-play + train cycles).
    pub iterations: usize,
    /// Directory where run folders and checkpoints are written.
    pub output_dir: PathBuf,
    /// Resume training from this checkpoint path (e.g. `checkpoints/run/model_iter_10`).
    #[serde(skip)]
    pub resume: Option<PathBuf>,

    // -- Training --
    /// Training epochs over the replay buffer per iteration.
    pub epochs: usize,
    /// Mini-batch size for training.
    pub batch_size: usize,
    /// Peak learning rate (cosine-annealed from here to `lr_min`).
    pub lr: f64,
    /// Minimum learning rate at end of cosine cycle (default: `lr / 10`).
    pub lr_min: f64,
    /// Number of most-recent iterations kept in the replay buffer.
    pub replay_window: usize,
    /// Iterations over which MCTS sims ramp from `mcts_sims_start` to `mcts_sims`
    /// and the value target transitions from pure Z (game outcome) to pure Q
    /// (search value). 0 = no ramp (full sims and pure Z from the start).
    ///
    /// Both ramps are synchronized because they address the same issue: early in
    /// training the value head is unreliable, so Q (derived from search) is
    /// near-zero garbage and extra sims just average more noise. Z (game outcome)
    /// is noisy but carries real signal. As the network improves, Q becomes a
    /// better per-position target than Z (averaging many sims vs one game result),
    /// and deeper search produces higher-quality Q. See "Lessons From AlphaZero
    /// (part 4): Improving the Training Target" (Abrams, 2018).
    pub warmup_iters: usize,

    // -- Self-play --
    /// Self-play games generated per iteration.
    pub games_per_iter: usize,
    /// Maximum async game tasks running concurrently during self-play.
    pub concurrent_games: usize,
    /// Maximum evaluations per GPU forward pass. Also determines the
    /// eval request queue capacity (2x this value).
    pub max_batch_size: usize,
    /// Early-game turns where action is sampled from improved policy
    /// (for exploration diversity). After this, the best action is used deterministically.
    pub explore_moves: u32,
    /// Probability of full search per move (playout cap randomization).
    pub playout_cap_full_prob: f32,
    /// Simulations for fast (non-full) search moves (playout cap randomization).
    pub playout_cap_fast_sims: u32,

    // -- MCTS --
    /// MCTS simulations per move during self-play (full-search budget).
    pub mcts_sims: u32,
    /// Starting MCTS simulations for progressive ramp (ramps linearly to `mcts_sims`).
    /// Set equal to `mcts_sims` for no ramp.
    pub mcts_sims_start: u32,
    /// Gumbel-Top-k sampled actions at root. Clamped to the number of
    /// legal actions, so values larger than `NUM_ACTIONS` are safe.
    pub gumbel_m: u32,
    /// Sigma scaling parameter for completed-Q transform.
    pub c_visit: f32,
    /// Sigma scaling parameter for completed-Q transform.
    pub c_scale: f32,
    /// Leaves to collect per MCTS batch before requesting evaluation.
    pub leaf_batch_size: u32,

    // -- Benchmark --
    /// Benchmark games against random-rollout bot (0 = skip).
    pub bench_games: u32,
    /// Run benchmark every N iterations (1 = every iteration).
    pub bench_interval: usize,
    /// MCTS simulations for the benchmark baseline opponent.
    pub bench_baseline_sims: u32,
    /// Rollouts per evaluation for the benchmark baseline opponent.
    pub bench_baseline_rollouts: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            // Infrastructure
            iterations: 1000,
            output_dir: PathBuf::from("checkpoints"),
            resume: None,

            // Training
            epochs: 3,
            batch_size: 256,
            lr: 0.001,
            lr_min: 0.0001,
            replay_window: 40,
            warmup_iters: 100,

            // Self-play
            games_per_iter: 150,
            concurrent_games: 256,
            max_batch_size: 1024,
            explore_moves: 30,
            playout_cap_full_prob: 0.25,
            playout_cap_fast_sims: 32,

            // MCTS
            mcts_sims: 800,
            mcts_sims_start: 50,
            gumbel_m: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            leaf_batch_size: 16,

            // Benchmark
            bench_games: 10,
            bench_interval: 10,
            bench_baseline_sims: 200,
            bench_baseline_rollouts: 1,
        }
    }
}

/// Per-iteration config passed to the model's train_step method.
pub struct TrainStepConfig {
    pub lr: f64,
    pub batch_size: usize,
    pub epochs: usize,
    /// Weight of Q in value target: 0.0 = pure Z, 1.0 = pure Q.
    pub q_weight: f32,
}

/// Metrics returned from a training step.
pub struct TrainMetrics {
    pub loss_policy_train: f32,
    pub loss_value_train: f32,
    pub loss_policy_val: f32,
    pub loss_value_val: f32,
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
    type Encoder: StateEncoder<G> + 'static;
    type Evaluator: Evaluator<G> + 'static;

    fn evaluator(&self) -> Self::Evaluator;
    fn train_step(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> TrainMetrics;
    fn save(&self, dir: &Path, iteration: usize);
    fn load(&mut self, dir: &Path, iteration: usize);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cosine_lr(config: &TrainConfig, iteration: usize) -> f64 {
    let t = if config.iterations > 1 {
        iteration as f64 / (config.iterations - 1) as f64
    } else {
        1.0
    };
    config.lr_min + 0.5 * (config.lr - config.lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
}

fn warmup_t(config: &TrainConfig, iteration: usize) -> f64 {
    if config.warmup_iters == 0 {
        return 1.0;
    }
    (iteration as f64 / config.warmup_iters as f64).min(1.0)
}

fn progressive_sims(config: &TrainConfig, iteration: usize) -> u32 {
    let start = config.mcts_sims_start;
    if start >= config.mcts_sims {
        return config.mcts_sims;
    }
    let t = warmup_t(config, iteration);
    let sims = start as f64 + t * (config.mcts_sims - start) as f64;
    (sims.round() as u32).max(1)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_training<G, M>(
    config: TrainConfig,
    model: &mut M,
    new_state: impl Fn(&mut fastrand::Rng) -> G + Send + Sync + 'static,
) where
    G: Game + 'static,
    M: TrainableModel<G>,
{
    let new_state: Arc<dyn Fn(&mut fastrand::Rng) -> G + Send + Sync> = Arc::new(new_state);
    let (mut rng, start_iteration) = checkpoint::resume_if_requested(&config, model);
    let run_dir = checkpoint::setup_run_dir(&config);
    eprintln!("run directory: {}", run_dir.display());
    checkpoint::save_config(&run_dir, &config);

    let mut csv = metrics::CsvLogger::open(&run_dir, start_iteration);
    let mut replay_buffer: VecDeque<Vec<Sample>> = VecDeque::new();
    let training_start = Instant::now();

    for iteration in start_iteration..config.iterations {
        let iter_start = Instant::now();
        let effective_lr = cosine_lr(&config, iteration);
        let effective_sims = progressive_sims(&config, iteration);
        let evaluator = model.evaluator();

        // Self-play
        let sp = self_play::run_self_play_iteration::<G, M::Encoder, M::Evaluator>(
            evaluator,
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
        let q_weight = warmup_t(&config, iteration + 1) as f32;
        fastrand::shuffle(&mut samples);
        let step_cfg = TrainStepConfig {
            lr: effective_lr,
            batch_size: config.batch_size,
            epochs: config.epochs,
            q_weight,
        };
        let train_metrics = model.train_step(&samples, &step_cfg);
        let train_elapsed = train_start.elapsed();

        // Checkpoint
        let iter_num = iteration + 1;
        checkpoint::save_checkpoint(model, &run_dir, iter_num, &mut rng);

        // Benchmark
        let run_bench = config.bench_games > 0
            && config.bench_interval > 0
            && iter_num % config.bench_interval == 0;
        let (bench_wins, bench_losses, bench_draws, bench_elapsed) = if run_bench {
            let bench_start = Instant::now();
            let eval = model.evaluator();
            let result = benchmark::run_benchmark::<G, M::Encoder, M::Evaluator>(
                eval, &config, &mut rng, &new_state,
            );
            (result.0, result.1, result.2, bench_start.elapsed())
        } else {
            (0, 0, 0, std::time::Duration::ZERO)
        };

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

        let bench_str = if run_bench {
            format!(
                " | bench: {}/{} ({:.0}%)",
                bench_wins,
                config.bench_games,
                bench_wins as f64 / config.bench_games as f64 * 100.0,
            )
        } else {
            String::new()
        };
        eprintln!(
            "iter {}/{}: {} games (W:{} L:{} D:{}, avg {} turns) {} samples, entropy={:.6}{} | tasks={}, avg_batch={:.1} | self-play {}, train {}, bench {} | total {}, ETA {}",
            iters_done,
            config.iterations,
            games_done,
            sp.wins,
            sp.losses,
            sp.draws,
            avg_turns,
            samples.len(),
            stats.policy_entropy_avg,
            bench_str,
            num_tasks,
            avg_batch_size,
            HumanDuration(self_play_elapsed),
            HumanDuration(train_elapsed),
            HumanDuration(bench_elapsed),
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
        csv.write_row(&metrics::CsvRow {
            // Core training
            iteration: iters_done,
            loss_policy_train: train_metrics.loss_policy_train,
            loss_value_train: train_metrics.loss_value_train,
            loss_policy_val: train_metrics.loss_policy_val,
            loss_value_val: train_metrics.loss_value_val,
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
            // Value diagnostics
            value_z_avg: stats.value_z_avg,
            value_q_avg: stats.value_q_avg,
            value_z_stddev: stats.value_z_stddev,
            value_q_stddev: stats.value_q_stddev,
            value_correction_avg: stats.value_correction_avg,
            value_q_spread_avg: stats.value_q_spread_avg,
            value_error_early_avg: stats.value_error_early_avg,
            value_error_late_avg: stats.value_error_late_avg,
            value_network_stddev: stats.value_network_stddev,
            // Benchmark
            bench_wins,
            bench_losses,
            bench_draws,
            // Config/infra
            lr: effective_lr,
            q_weight,
            mcts_sims: effective_sims,
            replay_samples: samples.len(),
            samples_iter: samples_this_iter,
            time_selfplay_secs: self_play_elapsed.as_secs_f64(),
            time_train_secs: train_elapsed.as_secs_f64(),
            time_bench_secs: bench_elapsed.as_secs_f64(),
        });
    }
}
