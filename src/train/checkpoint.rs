//! Checkpoint save/load and run directory management.
//!
//! Handles resuming from a checkpoint (restoring model weights, optimizer
//! state, and RNG seed), creating timestamped run directories, saving
//! config snapshots, and writing model checkpoints with metadata.

use std::path::PathBuf;

use serde::Serialize;
use tracing::{info, warn};

use super::{CheckpointMeta, TrainConfig, TrainableModel};
use crate::game::Game;

/// Resume from a checkpoint if `config.resume` is set.
/// Returns (rng, start_iteration).
pub(super) fn resume_if_requested<G: Game>(
    config: &TrainConfig,
    model: &mut dyn TrainableModel<G>,
) -> (fastrand::Rng, usize) {
    let mut rng = fastrand::Rng::new();
    let mut start_iteration = 0usize;

    if let Some(ref checkpoint_path) = config.resume {
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
            info!(
                iteration = meta.iteration,
                rng_seed = meta.rng_seed,
                "resumed from checkpoint"
            );
        } else {
            warn!(
                path = %meta_path.display(),
                "no checkpoint metadata found, using fresh RNG"
            );
        }
    }

    (rng, start_iteration)
}

/// Build or reuse the run directory.
pub(super) fn setup_run_dir(config: &TrainConfig) -> PathBuf {
    if let Some(ref path) = config.resume {
        path.parent().unwrap().to_path_buf()
    } else {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let run_name = format!("{ts}-{}", config.game_name);
        let dir = config.output_dir.join(&run_name);
        std::fs::create_dir_all(&dir).expect("failed to create run directory");
        dir
    }
}

/// Save config snapshot to the run directory.
pub(super) fn save_config(run_dir: &PathBuf, config: &impl Serialize) {
    let config_path = run_dir.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(config).unwrap())
        .expect("failed to write config.json");
}

/// Save model checkpoint and metadata.
pub(super) fn save_checkpoint<G: Game>(
    model: &dyn TrainableModel<G>,
    run_dir: &PathBuf,
    iter_num: usize,
    rng: &mut fastrand::Rng,
    model_name: Option<&str>,
    encoder_name: Option<&str>,
) {
    model.save(run_dir, iter_num);

    let meta = CheckpointMeta {
        iteration: iter_num,
        rng_seed: rng.u64(..),
        model: model_name.map(|s| s.to_string()),
        encoder: encoder_name.map(|s| s.to_string()),
    };
    let meta_path = run_dir.join(format!("checkpoint_iter_{iter_num}.json"));
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
        .expect("failed to save checkpoint metadata");

    info!(
        path = %run_dir.join(format!("model_iter_{iter_num}")).display(),
        "checkpoint saved"
    );
}
