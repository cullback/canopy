use std::collections::VecDeque;
use std::fmt;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, SearchResult, Step};
use crate::nn::StateEncoder;
use crate::player::Player;

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
    /// MCTS simulations for the NN bot during benchmark games
    pub bench_mcts: u32,
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
    /// Starting simulation budget for progressive ramp (defaults to mcts_sims = no ramp)
    pub mcts_sims_init: u32,
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
    type Evaluator: Evaluator<G> + Clone + Send;

    fn evaluator(&self) -> Self::Evaluator;
    fn train_step(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> TrainMetrics;
    fn save(&self, dir: &Path, iteration: usize);
    fn load(&mut self, dir: &Path, iteration: usize);
}

// ---------------------------------------------------------------------------
// Search helper
// ---------------------------------------------------------------------------

/// Drive an in-progress MCTS search state machine to completion.
fn drive_search<G: Game, E: Evaluator<G>>(
    search: &mut Search<G>,
    mut step: Step<G>,
    evaluator: &E,
    rng: &mut fastrand::Rng,
) -> SearchResult {
    loop {
        match step {
            Step::NeedsEval(pending) => {
                let output = evaluator.evaluate(&pending.state, rng);
                step = search.supply(output, pending, rng);
            }
            Step::Done(result) => return result,
        }
    }
}

/// Create a fresh search and drive it to completion.
fn run_search<G: Game, E: Evaluator<G>>(
    state: &G,
    evaluator: &E,
    config: &Config,
    rng: &mut fastrand::Rng,
) -> SearchResult {
    let (mut search, step) = Search::start(state, config, rng);
    drive_search(&mut search, step, evaluator, rng)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

struct GameRecord {
    features: Vec<f32>,
    policy_target: Vec<f32>,
    player: Player,
    q: f32,
    full_search: bool,
}

struct GameStats {
    winner: Option<Player>,
    num_turns: u32,
}

struct HumanDuration(std::time::Duration);

impl fmt::Display for HumanDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs();
        if secs >= 3600 {
            write!(f, "{}h{:02}m", secs / 3600, (secs % 3600) / 60)
        } else if secs >= 60 {
            write!(f, "{}m{:02}s", secs / 60, secs % 60)
        } else {
            write!(f, "{}s", secs)
        }
    }
}

fn self_play_game<G, Ev, E>(
    evaluator: &Ev,
    config: &TrainConfig,
    effective_sims: u32,
    seed: u64,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> (Vec<Sample>, GameStats)
where
    G: Game,
    Ev: Evaluator<G> + Clone,
    E: StateEncoder<G>,
{
    let mut rng = fastrand::Rng::with_seed(seed);
    let base_config = Config {
        num_simulations: effective_sims,
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
    };
    let fast_sims = config.playout_cap_fast_sims.min(effective_sims);

    let mut state = new_state(&mut rng);
    let mut actions = Vec::new();
    let mut chance_buf = Vec::new();
    let mut records = Vec::new();
    let mut features_buf = Vec::new();
    let mut turn_count: u32 = 0;
    let mut last_player: Option<Player> = None;

    // Tree reuse: persist search across moves, track intermediate actions
    let mut search: Option<Search<G>> = None;
    let mut actions_since_search: Vec<usize> = Vec::new();

    loop {
        // Resolve chance events
        chance_buf.clear();
        state.chance_outcomes(&mut chance_buf);
        if !chance_buf.is_empty() {
            let action = sample_chance(&chance_buf, &mut rng);
            actions_since_search.push(action);
            state.apply_action(action);
            continue;
        }

        let current = match state.status() {
            Status::Terminal(_) => break,
            Status::Ongoing(p) => p,
        };

        // Track turn count via player changes
        if last_player != Some(current) {
            turn_count += 1;
            last_player = Some(current);
        }

        actions.clear();
        state.legal_actions(&mut actions);

        // Skip forced moves (single legal action)
        if actions.len() == 1 {
            actions_since_search.push(actions[0]);
            state.apply_action(actions[0]);
            continue;
        }

        // Playout cap randomization: coin flip for full vs fast search
        let is_full = rng.f32() < config.playout_cap_full_prob;
        let move_config = Config {
            num_simulations: if is_full { effective_sims } else { fast_sims },
            ..base_config.clone()
        };

        // Run MCTS, reusing tree from previous search
        let result = match search {
            Some(ref mut s) => {
                let step = s.step_to(&state, &actions_since_search, &move_config, &mut rng);
                actions_since_search.clear();
                drive_search(s, step, evaluator, &mut rng)
            }
            None => {
                let (mut s, step) = Search::start(&state, &move_config, &mut rng);
                let result = drive_search(&mut s, step, evaluator, &mut rng);
                search = Some(s);
                actions_since_search.clear();
                result
            }
        };

        // Use improved policy directly as training target
        let policy_target = result.policy.clone();

        // Root Q from current player's perspective.
        // result.value is in [-1,1] from P1's perspective.
        let q = result.value * current.sign();

        // Encode state from current player's perspective
        E::encode(&state, &mut features_buf);

        records.push(GameRecord {
            features: features_buf.clone(),
            policy_target,
            player: current,
            q,
            full_search: is_full,
        });

        // Early-game: sample from improved policy for diversity.
        // After explore_moves: use SH survivor deterministically.
        let chosen = if turn_count <= config.explore_moves {
            sample_from_policy(&result.policy, &mut rng)
        } else {
            result.selected_action
        };
        actions_since_search.push(chosen);
        state.apply_action(chosen);
    }

    // Assign value targets from game outcome
    let (terminal_value, winner) = match state.status() {
        Status::Terminal(reward) => {
            let winner = if reward > 0.0 {
                Some(Player::One)
            } else if reward < 0.0 {
                Some(Player::Two)
            } else {
                None
            };
            (reward, winner)
        }
        _ => (0.0, None),
    };

    let stats = GameStats {
        winner,
        num_turns: turn_count,
    };

    let samples = records
        .into_iter()
        .map(|r| {
            // terminal_value is from P1's perspective in [-1,1].
            // Convert to current player's perspective.
            let z = terminal_value * r.player.sign();
            Sample {
                features: r.features,
                policy_target: r.policy_target,
                z,
                q: r.q,
                full_search: r.full_search,
            }
        })
        .collect();
    (samples, stats)
}

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

fn sample_chance(outcomes: &[(usize, f32)], rng: &mut fastrand::Rng) -> usize {
    let total: f32 = outcomes.iter().map(|(_, p)| p).sum();
    let mut r = rng.f32() * total;
    for &(outcome, p) in outcomes {
        r -= p;
        if r <= 0.0 {
            return outcome;
        }
    }
    outcomes.last().unwrap().0
}

/// Play benchmark games: NN evaluator vs RolloutEvaluator, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
fn run_benchmark<G: Game, Ev: Evaluator<G>>(
    evaluator: &Ev,
    mcts_sims: u32,
    num_games: u32,
    rng: &mut fastrand::Rng,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> (u32, u32, u32) {
    use crate::eval::RolloutEvaluator;

    let nn_config = Config {
        num_simulations: mcts_sims,
        ..Default::default()
    };
    let baseline_config = Config {
        num_simulations: 200,
        ..Default::default()
    };
    let baseline_eval = RolloutEvaluator { num_rollouts: 1 };

    let mut nn_wins = 0u32;
    let mut nn_losses = 0u32;
    let mut draws = 0u32;

    let pb = indicatif::ProgressBar::new(num_games as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message("bench W:0 L:0 D:0".to_string());

    for i in 0..num_games {
        let mut state = new_state(rng);
        let nn_is_p1 = i % 2 == 0;
        let mut chance_buf = Vec::new();

        loop {
            // Resolve chance
            chance_buf.clear();
            state.chance_outcomes(&mut chance_buf);
            if !chance_buf.is_empty() {
                let action = sample_chance(&chance_buf, rng);
                state.apply_action(action);
                continue;
            }

            let current = match state.status() {
                Status::Terminal(_) => break,
                Status::Ongoing(p) => p,
            };

            let is_nn_turn = (current == Player::One) == nn_is_p1;
            let result = if is_nn_turn {
                run_search(&state, evaluator, &nn_config, rng)
            } else {
                run_search(&state, &baseline_eval, &baseline_config, rng)
            };

            state.apply_action(result.selected_action);
        }

        if let Status::Terminal(reward) = state.status() {
            // Map to nn's result
            let nn_reward = if nn_is_p1 { reward } else { -reward };
            if nn_reward > 0.0 {
                nn_wins += 1;
            } else if nn_reward < 0.0 {
                nn_losses += 1;
            } else {
                draws += 1;
            }
        }

        pb.set_message(format!("bench W:{nn_wins} L:{nn_losses} D:{draws}"));
        pb.inc(1);
    }
    pb.finish();

    (nn_wins, nn_losses, draws)
}

// ---------------------------------------------------------------------------
// Generic burn training model
// ---------------------------------------------------------------------------

mod burn_train {
    use std::marker::PhantomData;
    use std::path::Path;

    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use burn::module::AutodiffModule;
    use burn::optim::adaptor::OptimizerAdaptor;
    use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
    use burn::prelude::*;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
    use burn::tensor::activation::log_softmax;

    use crate::game::Game;
    use crate::nn::{NeuralEvaluator, PolicyValueNet, StateEncoder};

    use super::{Sample, TrainMetrics, TrainStepConfig, TrainableModel};

    type TrainBackend = Autodiff<NdArray>;
    type InferBackend = NdArray;

    #[allow(clippy::too_many_arguments)]
    fn train_epoch<M, O>(
        mut model: M,
        samples: &[&Sample],
        lr: f64,
        batch_size: usize,
        alpha: f32,
        feature_size: usize,
        num_actions: usize,
        optimizer: &mut O,
        device: &NdArrayDevice,
    ) -> (M, f32, f32)
    where
        M: AutodiffModule<TrainBackend> + PolicyValueNet<TrainBackend>,
        O: Optimizer<M, TrainBackend>,
    {
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        fastrand::shuffle(&mut indices);

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut num_batches = 0usize;

        for batch_start in (0..indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(indices.len());
            let batch_indices = &indices[batch_start..batch_end];
            let bs = batch_indices.len();
            if bs == 0 {
                continue;
            }

            let features: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| samples[i].features.iter().copied())
                .collect();
            let features_tensor = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(features, [bs, feature_size]),
                device,
            );

            let policy_targets: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| samples[i].policy_target.iter().copied())
                .collect();
            let policy_tensor = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(policy_targets, [bs, num_actions]),
                device,
            );

            let value_targets: Vec<f32> = batch_indices
                .iter()
                .map(|&i| (1.0 - alpha) * samples[i].z + alpha * samples[i].q)
                .collect();
            let value_tensor = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(value_targets, [bs, 1]),
                device,
            );

            let (policy_logits, value_pred) = model.forward(features_tensor);

            // Per-sample cross-entropy, masked by full_search (playout cap)
            let log_probs = log_softmax(policy_logits, 1);
            let per_sample_ce = policy_tensor.mul(log_probs).sum_dim(1).neg();
            let mask_data: Vec<f32> = batch_indices
                .iter()
                .map(|&i| if samples[i].full_search { 1.0 } else { 0.0 })
                .collect();
            let mask_tensor =
                Tensor::<TrainBackend, 2>::from_data(TensorData::new(mask_data, [bs, 1]), device);
            let num_full: usize = batch_indices
                .iter()
                .filter(|&&i| samples[i].full_search)
                .count();
            let denom = if num_full > 0 { num_full as f32 } else { 1.0 };
            let policy_loss = per_sample_ce.mul(mask_tensor).sum().div_scalar(denom);

            let value_diff = value_pred.sub(value_tensor);
            let value_loss = value_diff.clone().mul(value_diff).mean();

            total_policy_loss += policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            total_value_loss += value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            num_batches += 1;

            let loss = policy_loss.add(value_loss);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);
        }

        let avg_policy = if num_batches > 0 {
            total_policy_loss / num_batches as f32
        } else {
            0.0
        };
        let avg_value = if num_batches > 0 {
            total_value_loss / num_batches as f32
        } else {
            0.0
        };
        (model, avg_policy, avg_value)
    }

    fn validate<M>(
        model: &M,
        samples: &[&Sample],
        batch_size: usize,
        alpha: f32,
        feature_size: usize,
        num_actions: usize,
        device: &NdArrayDevice,
    ) -> (f32, f32)
    where
        M: PolicyValueNet<InferBackend>,
    {
        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut num_batches = 0usize;

        for batch_start in (0..samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(samples.len());
            let batch = &samples[batch_start..batch_end];
            let bs = batch.len();
            if bs == 0 {
                continue;
            }

            let features: Vec<f32> = batch
                .iter()
                .flat_map(|s| s.features.iter().copied())
                .collect();
            let features_tensor = Tensor::<InferBackend, 2>::from_data(
                TensorData::new(features, [bs, feature_size]),
                device,
            );

            let policy_targets: Vec<f32> = batch
                .iter()
                .flat_map(|s| s.policy_target.iter().copied())
                .collect();
            let policy_tensor = Tensor::<InferBackend, 2>::from_data(
                TensorData::new(policy_targets, [bs, num_actions]),
                device,
            );

            let value_targets: Vec<f32> = batch
                .iter()
                .map(|s| (1.0 - alpha) * s.z + alpha * s.q)
                .collect();
            let value_tensor = Tensor::<InferBackend, 2>::from_data(
                TensorData::new(value_targets, [bs, 1]),
                device,
            );

            let (policy_logits, value_pred) = model.forward(features_tensor);

            // Per-sample cross-entropy, masked by full_search (playout cap)
            let log_probs = log_softmax(policy_logits, 1);
            let per_sample_ce = policy_tensor.mul(log_probs).sum_dim(1).neg();
            let mask_data: Vec<f32> = batch
                .iter()
                .map(|s| if s.full_search { 1.0 } else { 0.0 })
                .collect();
            let mask_tensor =
                Tensor::<InferBackend, 2>::from_data(TensorData::new(mask_data, [bs, 1]), device);
            let num_full: usize = batch.iter().filter(|s| s.full_search).count();
            let denom = if num_full > 0 { num_full as f32 } else { 1.0 };
            let policy_loss = per_sample_ce.mul(mask_tensor).sum().div_scalar(denom);

            let value_diff = value_pred.sub(value_tensor);
            let value_loss = value_diff.clone().mul(value_diff).mean();

            total_policy_loss += policy_loss.into_data().to_vec::<f32>().unwrap()[0];
            total_value_loss += value_loss.into_data().to_vec::<f32>().unwrap()[0];
            num_batches += 1;
        }

        let avg_policy = if num_batches > 0 {
            total_policy_loss / num_batches as f32
        } else {
            0.0
        };
        let avg_value = if num_batches > 0 {
            total_value_loss / num_batches as f32
        } else {
            0.0
        };
        (avg_policy, avg_value)
    }

    pub struct BurnTrainableModel<G, E, M>
    where
        M: AutodiffModule<TrainBackend>,
    {
        model: M,
        optimizer: OptimizerAdaptor<AdamW, M, TrainBackend>,
        device: NdArrayDevice,
        model_init: Box<dyn Fn(&NdArrayDevice) -> M + Send>,
        _marker: PhantomData<fn() -> (G, E)>,
    }

    impl<G, E, M> BurnTrainableModel<G, E, M>
    where
        M: AutodiffModule<TrainBackend>,
    {
        pub fn new(
            model_init: impl Fn(&NdArrayDevice) -> M + Send + 'static,
            device: &NdArrayDevice,
        ) -> Self {
            let model = model_init(device);
            let optimizer = AdamWConfig::new().with_weight_decay(0.0004).init();
            Self {
                model,
                optimizer,
                device: *device,
                model_init: Box::new(model_init),
                _marker: PhantomData,
            }
        }
    }

    impl<G, E, M> TrainableModel<G> for BurnTrainableModel<G, E, M>
    where
        G: Game,
        E: StateEncoder<G> + Send,
        M: AutodiffModule<TrainBackend> + PolicyValueNet<TrainBackend> + Send + 'static,
        M::InnerModule: PolicyValueNet<InferBackend> + Send,
    {
        type Evaluator = NeuralEvaluator<InferBackend, E, M::InnerModule>;

        fn evaluator(&self) -> Self::Evaluator {
            let infer_model = self.model.valid();
            NeuralEvaluator::new(infer_model, self.device)
        }

        fn train_step(&mut self, samples: &[&Sample], cfg: &TrainStepConfig) -> TrainMetrics {
            let val_split = (samples.len() * 4 / 5)
                .max(1)
                .min(samples.len().saturating_sub(1));
            let (train_samples, val_samples) = samples.split_at(val_split);

            let mut final_train_ploss = 0.0f32;
            let mut final_train_vloss = 0.0f32;
            let mut final_val_ploss = 0.0f32;
            let mut final_val_vloss = 0.0f32;

            for epoch in 0..cfg.epochs {
                let (new_model, ploss, vloss) = train_epoch(
                    std::mem::replace(&mut self.model, (self.model_init)(&self.device)),
                    train_samples,
                    cfg.lr,
                    cfg.batch_size,
                    cfg.alpha,
                    E::FEATURE_SIZE,
                    G::NUM_ACTIONS,
                    &mut self.optimizer,
                    &self.device,
                );
                self.model = new_model;

                let (val_ploss, val_vloss) = validate(
                    &self.model.valid(),
                    val_samples,
                    cfg.batch_size,
                    cfg.alpha,
                    E::FEATURE_SIZE,
                    G::NUM_ACTIONS,
                    &self.device,
                );

                final_train_ploss = ploss;
                final_train_vloss = vloss;
                final_val_ploss = val_ploss;
                final_val_vloss = val_vloss;

                eprintln!(
                    "  epoch {}/{}: train(p={:.4} v={:.4}) val(p={:.4} v={:.4})",
                    epoch + 1,
                    cfg.epochs,
                    ploss,
                    vloss,
                    val_ploss,
                    val_vloss
                );
            }

            TrainMetrics {
                train_policy_loss: final_train_ploss,
                train_value_loss: final_train_vloss,
                val_policy_loss: final_val_ploss,
                val_value_loss: final_val_vloss,
            }
        }

        fn save(&self, dir: &Path, iteration: usize) {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

            let path = dir.join(format!("model_iter_{iteration}"));
            self.model
                .clone()
                .save_file(path.to_str().unwrap(), &recorder)
                .expect("failed to save checkpoint");

            let optim_path = dir.join(format!("optim_iter_{iteration}"));
            recorder
                .record(self.optimizer.to_record(), optim_path)
                .expect("failed to save optimizer state");
        }

        fn load(&mut self, dir: &Path, iteration: usize) {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

            let path = dir.join(format!("model_iter_{iteration}"));
            self.model = (self.model_init)(&self.device)
                .load_file(path.to_str().unwrap(), &recorder, &self.device)
                .expect("failed to load checkpoint");

            let optim_path = dir.join(format!("optim_iter_{iteration}"));
            match recorder.load(optim_path, &self.device) {
                Ok(optim_record) => {
                    let optimizer = std::mem::replace(
                        &mut self.optimizer,
                        AdamWConfig::new().with_weight_decay(0.0004).init(),
                    );
                    self.optimizer = optimizer.load_record(optim_record);
                    eprintln!("restored optimizer state");
                }
                Err(_) => {
                    eprintln!("warning: no optimizer state found, starting fresh optimizer");
                }
            }
        }
    }
}

pub use burn_train::BurnTrainableModel;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_training<G, E, M>(
    config: TrainConfig,
    model: &mut M,
    new_state: impl Fn(&mut fastrand::Rng) -> G + Sync,
) where
    G: Game,
    E: StateEncoder<G>,
    M: TrainableModel<G>,
    M::Evaluator: 'static,
{
    let mut rng = fastrand::Rng::new();
    let mut start_iteration = 0usize;

    // Resume from checkpoint if requested
    if let Some(ref path) = config.resume {
        let checkpoint_path = std::path::Path::new(path);
        let stem = checkpoint_path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("invalid checkpoint path");
        start_iteration = stem
            .strip_prefix("model_iter_")
            .expect("checkpoint filename must be model_iter_N")
            .parse::<usize>()
            .expect("failed to parse iteration from checkpoint filename");

        let checkpoint_dir = checkpoint_path.parent().unwrap();
        model.load(checkpoint_dir, start_iteration);

        let meta_path = checkpoint_dir.join(format!("checkpoint_iter_{start_iteration}.json"));
        if let Ok(meta_bytes) = std::fs::read(&meta_path) {
            let meta: CheckpointMeta =
                serde_json::from_slice(&meta_bytes).expect("failed to parse checkpoint metadata");
            rng = fastrand::Rng::with_seed(meta.rng_seed);
            eprintln!(
                "resumed from iteration {}, rng_seed={}",
                meta.iteration, meta.rng_seed
            );
        } else {
            eprintln!(
                "warning: no checkpoint metadata at {}, using fresh RNG",
                meta_path.display()
            );
        }
    }

    // Build run directory
    let run_dir = if let Some(ref path) = config.resume {
        std::path::Path::new(path).parent().unwrap().to_path_buf()
    } else {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let run_name = format!(
            "{ts}_mcts{}_g{}_lr{}",
            config.mcts_sims, config.games_per_iter, config.lr,
        );
        let dir = PathBuf::from(&config.output_dir).join(&run_name);
        std::fs::create_dir_all(&dir).expect("failed to create run directory");
        dir
    };
    eprintln!("run directory: {}", run_dir.display());

    // Save config snapshot
    let config_path = run_dir.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
        .expect("failed to write config.json");

    // CSV metrics file
    let csv_path = run_dir.join("metrics.csv");
    let mut csv = if start_iteration > 0 && csv_path.exists() {
        BufWriter::new(
            std::fs::OpenOptions::new()
                .append(true)
                .open(&csv_path)
                .expect("failed to open metrics.csv for append"),
        )
    } else {
        let mut w =
            BufWriter::new(std::fs::File::create(&csv_path).expect("failed to create metrics.csv"));
        writeln!(
            w,
            "iteration,train_policy_loss,train_value_loss,val_policy_loss,val_value_loss,\
             avg_game_length,p1_wins,p2_wins,draws,avg_policy_entropy,replay_buffer_samples,\
             bench_wins,bench_losses,bench_draws,lr,q_alpha,self_play_secs,train_secs,bench_secs,\
             games,samples_this_iter,min_game_length,max_game_length,\
             avg_z,avg_q,stddev_z,stddev_q,avg_value_target,\
             avg_policy_max_prob,\
             games_per_sec,samples_per_sec,total_elapsed_secs,\
             bench_win_rate,\
             mcts_sims,gumbel_m,c_visit,c_scale,epochs,batch_size,replay_window,explore_moves,games_per_iter"
        )
        .expect("failed to write CSV header");
        w
    };

    let mut replay_buffer: VecDeque<Vec<Sample>> = VecDeque::new();
    let training_start = Instant::now();

    for iteration in start_iteration..config.iterations {
        let iter_start = Instant::now();

        // Progressive simulation ramp
        let t = if config.iterations > 1 {
            iteration as f64 / (config.iterations - 1) as f64
        } else {
            1.0
        };
        let effective_sims =
            config.mcts_sims_init + ((config.mcts_sims - config.mcts_sims_init) as f64 * t) as u32;

        // Cosine LR schedule
        let effective_lr = config.lr_min
            + 0.5 * (config.lr - config.lr_min) * (1.0 + (std::f64::consts::PI * t).cos());

        let evaluator = model.evaluator();

        let pb = indicatif::ProgressBar::new(config.games_per_iter as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
            )
            .unwrap()
        );
        pb.set_message(format!(
            "iter {}/{} self-play (sims={})",
            iteration + 1,
            config.iterations,
            effective_sims,
        ));

        let seeds: Vec<u64> = (0..config.games_per_iter).map(|_| rng.u64(..)).collect();
        let completed = std::sync::atomic::AtomicU32::new(0);

        let results: Vec<(Vec<Sample>, GameStats)> = seeds
            .into_par_iter()
            .map_with(evaluator, |ev, seed| {
                let result = self_play_game::<G, M::Evaluator, E>(
                    ev,
                    &config,
                    effective_sims,
                    seed,
                    &new_state,
                );
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                pb.set_position(done as u64);
                result
            })
            .collect();
        pb.finish();

        let mut iter_samples = Vec::new();
        let mut p1_wins = 0u32;
        let mut p2_wins = 0u32;
        let mut draws = 0u32;
        let mut total_turns = 0u32;
        let mut min_game_length = u32::MAX;
        let mut max_game_length = 0u32;

        for (game_samples, stats) in results {
            total_turns += stats.num_turns;
            min_game_length = min_game_length.min(stats.num_turns);
            max_game_length = max_game_length.max(stats.num_turns);
            match stats.winner {
                Some(Player::One) => p1_wins += 1,
                Some(Player::Two) => p2_wins += 1,
                None => draws += 1,
            }
            iter_samples.extend(game_samples);
        }

        let self_play_elapsed = iter_start.elapsed();
        let games_done = p1_wins + p2_wins + draws;
        let samples_this_iter = iter_samples.len();

        // Fix min_game_length when no games played
        if games_done == 0 {
            min_game_length = 0;
        }

        // Compute average policy entropy
        let avg_entropy: f64 = if iter_samples.is_empty() {
            0.0
        } else {
            iter_samples
                .iter()
                .map(|s| {
                    s.policy_target
                        .iter()
                        .filter(|&&p| p > 0.0)
                        .map(|&p| -(p as f64) * (p as f64).ln())
                        .sum::<f64>()
                })
                .sum::<f64>()
                / iter_samples.len() as f64
        };

        // Value diagnostics
        let (avg_z, avg_q, stddev_z, stddev_q) = if iter_samples.is_empty() {
            (0.0f64, 0.0f64, 0.0f64, 0.0f64)
        } else {
            let n = iter_samples.len() as f64;
            let sum_z: f64 = iter_samples.iter().map(|s| s.z as f64).sum();
            let sum_q: f64 = iter_samples.iter().map(|s| s.q as f64).sum();
            let mean_z = sum_z / n;
            let mean_q = sum_q / n;
            let var_z: f64 = iter_samples
                .iter()
                .map(|s| (s.z as f64 - mean_z).powi(2))
                .sum::<f64>()
                / n;
            let var_q: f64 = iter_samples
                .iter()
                .map(|s| (s.q as f64 - mean_q).powi(2))
                .sum::<f64>()
                / n;
            (mean_z, mean_q, var_z.sqrt(), var_q.sqrt())
        };

        // Policy confidence: mean max probability in improved policy
        let avg_policy_max_prob: f64 = if iter_samples.is_empty() {
            0.0
        } else {
            iter_samples
                .iter()
                .map(|s| s.policy_target.iter().copied().fold(0.0f32, f32::max) as f64)
                .sum::<f64>()
                / iter_samples.len() as f64
        };

        let train_start = Instant::now();

        // Rolling replay buffer
        replay_buffer.push_back(iter_samples);
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

        let metrics = model.train_step(&samples, &step_cfg);

        let train_elapsed = train_start.elapsed();

        // Checkpoint every iteration
        let iter_num = iteration + 1;

        model.save(&run_dir, iter_num);

        let meta = CheckpointMeta {
            iteration: iter_num,
            rng_seed: rng.u64(..),
        };
        let meta_path = run_dir.join(format!("checkpoint_iter_{iter_num}.json"));
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
            .expect("failed to save checkpoint metadata");

        eprintln!(
            "checkpoint: {}",
            run_dir.join(format!("model_iter_{iter_num}")).display()
        );

        // Benchmark
        let bench_start = Instant::now();
        let (bench_wins, bench_losses, bench_draws) = if config.bench_games > 0 {
            let eval = model.evaluator();
            run_benchmark::<G, M::Evaluator>(
                &eval,
                config.bench_mcts,
                config.bench_games,
                &mut rng,
                &new_state,
            )
        } else {
            (0, 0, 0)
        };
        let bench_elapsed = bench_start.elapsed();

        let total_elapsed = training_start.elapsed();
        let iters_done = iteration + 1;
        let iters_this_session = iters_done - start_iteration;
        let iters_remaining = config.iterations - iters_done;
        let avg_iter_time = total_elapsed / iters_this_session as u32;
        let eta = avg_iter_time * iters_remaining as u32;

        let avg_turns = if games_done > 0 {
            total_turns / games_done
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
            p1_wins,
            p2_wins,
            draws,
            avg_turns,
            samples.len(),
            avg_entropy,
            bench_str,
            HumanDuration(self_play_elapsed),
            HumanDuration(train_elapsed),
            HumanDuration(bench_elapsed),
            HumanDuration(total_elapsed),
            HumanDuration(eta),
        );

        // Append CSV row
        let avg_game_length = if games_done > 0 {
            total_turns as f64 / games_done as f64
        } else {
            0.0
        };

        // Throughput
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

        // avg_value_target = (1-alpha)*avg_z + alpha*avg_q
        let avg_value_target = (1.0 - alpha as f64) * avg_z + alpha as f64 * avg_q;

        // Bench win rate (NaN if bench disabled)
        let bench_win_rate: f64 = if config.bench_games > 0 {
            bench_wins as f64 / config.bench_games as f64
        } else {
            f64::NAN
        };

        writeln!(
            csv,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{},{},{},{},{:.6},{:.4},{:.3},{:.3},{:.3},\
             {},{},{},{},\
             {:.6},{:.6},{:.6},{:.6},{:.6},\
             {:.6},\
             {:.3},{:.3},{:.3},\
             {:.6},\
             {},{},{},{},{},{},{},{},{}",
            iters_done,
            metrics.train_policy_loss,
            metrics.train_value_loss,
            metrics.val_policy_loss,
            metrics.val_value_loss,
            avg_game_length,
            p1_wins,
            p2_wins,
            draws,
            avg_entropy,
            samples.len(),
            bench_wins,
            bench_losses,
            bench_draws,
            effective_lr,
            alpha,
            self_play_elapsed.as_secs_f64(),
            train_elapsed.as_secs_f64(),
            bench_elapsed.as_secs_f64(),
            // New columns
            games_done,
            samples_this_iter,
            min_game_length,
            max_game_length,
            avg_z,
            avg_q,
            stddev_z,
            stddev_q,
            avg_value_target,
            avg_policy_max_prob,
            games_per_sec,
            samples_per_sec,
            total_elapsed.as_secs_f64(),
            bench_win_rate,
            effective_sims,
            config.gumbel_m,
            config.c_visit,
            config.c_scale,
            config.epochs,
            config.batch_size,
            config.replay_window,
            config.explore_moves,
            config.games_per_iter,
        )
        .expect("failed to write CSV row");
        csv.flush().ok();
    }
}
