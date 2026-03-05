mod benchmark;
mod burn_backend;
mod checkpoint;
mod metrics;
mod self_play;

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::eval::Evaluator;
use crate::game::Game;
use crate::nn::StateEncoder;
use crate::utils::HumanDuration;

pub use burn_backend::BurnTrainableModel;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub struct Sample {
    pub features: Vec<f32>,
    pub policy_target: Vec<f32>,
    /// Game outcome from current player's perspective: +1, -1, or 0
    pub z: f32,
    /// Root Q from current player's perspective, in [-1, 1]
    pub q: f32,
    /// Whether this position used full search (for playout cap randomization)
    pub full_search: bool,
}

#[derive(Serialize)]
pub struct TrainConfig {
    pub iterations: usize,
    pub games_per_iter: usize,
    pub mcts_sims: u32,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// Minimum learning rate at end of cosine cycle
    pub lr_min: f64,
    pub replay_window: usize,
    pub output_dir: String,
    #[serde(skip)]
    pub resume: Option<String>,
    /// Iteration at which q fully replaces z as value target (0 = pure z always)
    pub q_blend_generations: usize,
    /// Benchmark games against random-rollout bot per iteration (0 = skip)
    pub bench_games: u32,
    /// Gumbel-Top-k sampled actions at root
    pub gumbel_m: u32,
    /// σ scaling parameter
    pub c_visit: f32,
    /// σ scaling parameter
    pub c_scale: f32,
    /// Number of early-game turns where action is sampled from improved policy
    /// (for exploration diversity). After this, use selected_action deterministically.
    pub explore_moves: u32,
    /// Probability of full search per move (playout cap randomization)
    pub playout_cap_full_prob: f32,
    /// Simulations for fast (non-full) search moves
    pub playout_cap_fast_sims: u32,
}

/// Per-iteration config passed to the model's train_step method.
pub struct TrainStepConfig {
    pub lr: f64,
    pub batch_size: usize,
    pub epochs: usize,
    /// z/q blend factor: 0.0 = pure z, 1.0 = pure q
    pub alpha: f32,
}

/// Metrics returned from a training step.
pub struct TrainMetrics {
    pub train_policy_loss: f32,
    pub train_value_loss: f32,
    pub val_policy_loss: f32,
    pub val_value_loss: f32,
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
    type Encoder: StateEncoder<G>;
    type Evaluator: Evaluator<G> + Clone + Send;

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

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_training<G, M>(
    config: TrainConfig,
    model: &mut M,
    new_state: impl Fn(&mut fastrand::Rng) -> G + Sync,
) where
    G: Game,
    M: TrainableModel<G>,
    M::Evaluator: 'static,
{
    let (mut rng, start_iteration) = checkpoint::resume_if_requested(&config, model);
    let run_dir = checkpoint::setup_run_dir(&config);
    eprintln!("run directory: {}", run_dir.display());
    checkpoint::save_config(&run_dir, &config);

    let mut csv = metrics::CsvLogger::open(&run_dir, start_iteration);
    let mut replay_buffer: VecDeque<Vec<Sample>> = VecDeque::new();
    let training_start = Instant::now();
    let mut total_games = 0u64;
    let mut total_gradient_steps = 0u64;

    for iteration in start_iteration..config.iterations {
        let iter_start = Instant::now();
        let effective_lr = cosine_lr(&config, iteration);
        let evaluator = model.evaluator();

        // Self-play
        let sp = self_play::run_self_play_iteration::<G, M::Encoder, M::Evaluator>(
            evaluator,
            &config,
            config.mcts_sims,
            iteration,
            &mut rng,
            &new_state,
        );
        let self_play_elapsed = iter_start.elapsed();
        let games_done = sp.p1_wins + sp.p2_wins + sp.draws;
        let samples_this_iter = sp.samples.len();
        total_games += games_done as u64;

        // Stats
        let stats = metrics::compute_iter_stats(&sp.samples);

        // Train
        let train_start = Instant::now();
        replay_buffer.push_back(sp.samples);
        while replay_buffer.len() > config.replay_window {
            replay_buffer.pop_front();
        }
        let mut samples: Vec<&Sample> = replay_buffer.iter().flat_map(|v| v.iter()).collect();
        let alpha = if config.q_blend_generations > 0 {
            ((iteration + 1) as f32 / config.q_blend_generations as f32).min(1.0)
        } else {
            0.0
        };
        fastrand::shuffle(&mut samples);
        let step_cfg = TrainStepConfig {
            lr: effective_lr,
            batch_size: config.batch_size,
            epochs: config.epochs,
            alpha,
        };
        let train_metrics = model.train_step(&samples, &step_cfg);
        total_gradient_steps += train_metrics.gradient_steps as u64;
        let train_elapsed = train_start.elapsed();

        // Checkpoint
        let iter_num = iteration + 1;
        checkpoint::save_checkpoint(model, &run_dir, iter_num, &mut rng);

        // Benchmark
        let bench_start = Instant::now();
        let (bench_wins, bench_losses, bench_draws) = if config.bench_games > 0 {
            let eval = model.evaluator();
            benchmark::run_benchmark::<G, M::Evaluator>(
                &eval,
                config.mcts_sims,
                config.bench_games,
                &mut rng,
                &new_state,
            )
        } else {
            (0, 0, 0)
        };
        let bench_elapsed = bench_start.elapsed();

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

        let bench_str = if config.bench_games > 0 {
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
            "iter {}/{}: {} games (P1:{} P2:{} D:{}, avg {} turns) {} samples, entropy={:.2}{} | self-play {}, train {}, bench {} | total {}, ETA {}",
            iters_done,
            config.iterations,
            games_done,
            sp.p1_wins,
            sp.p2_wins,
            sp.draws,
            avg_turns,
            samples.len(),
            stats.avg_entropy,
            bench_str,
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
        let iter_total_secs = iter_start.elapsed().as_secs_f64();
        let games_per_sec = if self_play_elapsed.as_secs_f64() > 0.0 {
            games_done as f64 / self_play_elapsed.as_secs_f64()
        } else {
            0.0
        };
        let samples_per_sec = if iter_total_secs > 0.0 {
            samples_this_iter as f64 / iter_total_secs
        } else {
            0.0
        };
        let avg_value_target = (1.0 - alpha as f64) * stats.avg_z + alpha as f64 * stats.avg_q;
        let bench_win_rate: f64 = if config.bench_games > 0 {
            bench_wins as f64 / config.bench_games as f64
        } else {
            f64::NAN
        };

        csv.write_row(&metrics::CsvRow {
            iteration: iters_done,
            train_policy_loss: train_metrics.train_policy_loss,
            train_value_loss: train_metrics.train_value_loss,
            val_policy_loss: train_metrics.val_policy_loss,
            val_value_loss: train_metrics.val_value_loss,
            avg_game_length,
            p1_wins: sp.p1_wins,
            p2_wins: sp.p2_wins,
            draws: sp.draws,
            avg_policy_entropy: stats.avg_entropy,
            replay_buffer_samples: samples.len(),
            bench_wins,
            bench_losses,
            bench_draws,
            lr: effective_lr,
            q_alpha: alpha,
            self_play_secs: self_play_elapsed.as_secs_f64(),
            train_secs: train_elapsed.as_secs_f64(),
            bench_secs: bench_elapsed.as_secs_f64(),
            games: games_done,
            samples_this_iter,
            min_game_length,
            max_game_length: sp.max_game_length,
            avg_z: stats.avg_z,
            avg_q: stats.avg_q,
            stddev_z: stats.stddev_z,
            stddev_q: stats.stddev_q,
            avg_value_target,
            avg_policy_max_prob: stats.avg_policy_max_prob,
            games_per_sec,
            samples_per_sec,
            total_elapsed_secs: total_elapsed.as_secs_f64(),
            bench_win_rate,
            mcts_sims: config.mcts_sims,
            gumbel_m: config.gumbel_m,
            c_visit: config.c_visit,
            c_scale: config.c_scale,
            epochs: config.epochs,
            batch_size: config.batch_size,
            replay_window: config.replay_window,
            explore_moves: config.explore_moves,
            games_per_iter: config.games_per_iter,
            total_games,
            total_gradient_steps,
        });
    }
}
