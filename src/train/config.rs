//! Training configuration: all hyperparameters and infrastructure settings.
//!
//! `TrainConfig` is the single source of truth for a training run. Games
//! set defaults via `add_config` in their CLI setup; the user overrides
//! individual fields from the command line.

use std::path::PathBuf;

use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct TrainConfig {
    // -- Infrastructure --
    /// Total training iterations (self-play + train cycles).
    pub iterations: usize,
    /// Directory where run folders and checkpoints are written.
    pub output_dir: PathBuf,
    /// Game name used in run directory names.
    pub game_name: String,
    /// Resume training from this checkpoint path (e.g. `checkpoints/run/model_iter_10`).
    #[serde(skip)]
    pub resume: Option<PathBuf>,

    // -- Training --
    /// Training epochs over the replay buffer per iteration.
    pub epochs: usize,
    /// Mini-batch size for training.
    pub train_batch_size: usize,
    /// Learning rate.
    pub lr: f64,
    /// Fresh self-play samples to collect before triggering a training step.
    ///
    /// Each iteration generates this many samples via self-play, then trains
    /// `epochs` passes over the full replay buffer. Gradient steps per epoch =
    /// `replay_buffer_samples / train_batch_size`, so this controls how much
    /// new data arrives between training passes, not the training volume itself.
    /// Larger values mean more diverse fresh data per iteration but longer
    /// self-play phases.
    pub train_samples_per_iter: usize,
    /// Maximum samples retained in the replay buffer. Oldest games evicted first.
    ///
    /// Sized as a multiple of `train_samples_per_iter`: the ratio determines how
    /// many iterations a sample survives before eviction. 5-10x is typical.
    /// Too small risks overfitting to recent play patterns; too large dilutes
    /// training with stale positions from weaker networks.
    pub replay_buffer_samples: usize,
    /// Iterations over which the value target transitions from pure Z (game
    /// outcome) toward Q (search value), capped at `q_weight_max`. 0 = no ramp.
    ///
    /// Early in training the value head is unreliable, so Q (derived from search)
    /// is noise. Z (game outcome) is noisy but carries real signal. As the network
    /// improves, Q becomes a better per-position target than Z (averaging many
    /// sims vs one game result), and deeper search produces higher-quality Q. A
    /// small Z anchor (via `q_weight_max < 1.0`) prevents value drift from purely
    /// self-generated Q targets.
    pub q_weight_ramp_iters: usize,
    /// Maximum q_weight after warmup ramp. Values < 1.0 retain a Z (game outcome)
    /// anchor to prevent value drift from purely self-generated Q targets.
    pub q_weight_max: f32,

    // -- Self-play --
    /// Maximum async game tasks running concurrently during self-play.
    pub concurrent_games: usize,
    /// Maximum evaluations per GPU forward pass. Also determines the
    /// eval request queue capacity (2x this value).
    pub inference_batch_size: usize,
    /// Terminate games exceeding this many actions (including chance nodes) as a
    /// draw (reward = 0). 0 = no limit.
    pub max_actions: u32,
    /// Early-game actions where the action is sampled from improved policy
    /// (for exploration diversity). After this, the best action is used deterministically.
    pub explore_actions: u32,
    /// Probability of full search per action (playout cap randomization).
    pub playout_cap_full_prob: f32,
    /// Simulations for fast (non-full) search actions (playout cap randomization).
    pub playout_cap_fast_sims: u32,

    // -- MCTS --
    /// Enable SO-ISMCTS legal-action filtering at interior nodes. Required
    /// for imperfect-information games where determinization can make stored
    /// tree edges illegal.
    pub filter_legal: bool,
    /// MCTS simulations per action during self-play (full-search budget).
    pub mcts_sims: u32,
    /// Gumbel-Top-k sampled actions at root. Clamped to the number of
    /// legal actions, so values larger than `NUM_ACTIONS` are safe.
    pub gumbel_m: u32,
    /// Sigma scaling parameter for completed-Q transform.
    pub c_visit: f32,
    /// Sigma scaling parameter for completed-Q transform.
    pub c_scale: f32,
    /// Leaves to collect per MCTS batch before requesting evaluation.
    pub leaf_batch_size: u32,

    // -- Checkpointing --
    /// Save model checkpoint every N iterations (1 = every iteration).
    /// The final iteration is always checkpointed regardless of this setting.
    pub checkpoint_interval: usize,

    /// Number of GPU inference workers for self-play (default: 1).
    #[serde(skip)]
    pub inference_workers: usize,

    // -- Auxiliary short-term value heads --
    /// EMA horizons for auxiliary value heads (empty = disabled). E.g. [6, 16, 50].
    pub aux_value_horizons: Vec<u32>,
    /// Per-head weight for auxiliary value losses. Default 0.5.
    pub aux_value_weight: f32,

    // -- Soft policy --
    /// Temperature for soft policy target: policy^(1/T), renormalized.
    /// 0.0 = disabled (no soft policy head). KataGo uses T=4.
    pub soft_policy_temperature: f32,
    /// Weight for soft policy loss relative to hard policy loss.
    /// KataGo uses 8.0 (compensates for smaller gradients from softer target).
    pub soft_policy_weight: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            // Infrastructure
            iterations: 1000,
            output_dir: PathBuf::from("checkpoints"),
            game_name: String::from("unknown"),
            resume: None,

            // Training
            epochs: 2,
            train_batch_size: 1024,
            lr: 0.001,
            train_samples_per_iter: 20_000,
            replay_buffer_samples: 100_000,
            q_weight_ramp_iters: 25,
            q_weight_max: 0.85,

            // Self-play
            concurrent_games: 256,
            inference_batch_size: 1024,
            max_actions: 0,
            explore_actions: 30,
            playout_cap_full_prob: 0.25,
            playout_cap_fast_sims: 64,

            // MCTS
            filter_legal: false,
            mcts_sims: 200,
            gumbel_m: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            leaf_batch_size: 4,

            // Checkpointing
            checkpoint_interval: 1,

            inference_workers: 1,

            // Auxiliary short-term value heads
            aux_value_horizons: vec![],
            aux_value_weight: 0.5,

            // Soft policy
            soft_policy_temperature: 0.0,
            soft_policy_weight: 0.0,
        }
    }
}
