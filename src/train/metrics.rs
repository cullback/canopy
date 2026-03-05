use std::io::{BufWriter, Write};
use std::path::PathBuf;

use super::Sample;

/// Declare a CSV row struct with automatic header and serialization.
///
/// Usage:
/// ```ignore
/// define_csv_row! {
///     struct CsvRow {
///         iteration: usize,
///         loss: f64,
///     }
/// }
/// ```
/// Generates `CsvRow` with `fn header() -> &'static str` and `fn to_csv_line(&self) -> String`.
macro_rules! define_csv_row {
    (struct $name:ident { $( $field:ident : $ty:ty ),* $(,)? }) => {
        pub(super) struct $name {
            $( pub $field: $ty, )*
        }

        impl $name {
            pub fn header() -> &'static str {
                concat!( $( stringify!($field), ",", )* )
            }

            pub fn to_csv_line(&self) -> String {
                let mut parts: Vec<String> = Vec::new();
                $( parts.push(format!("{}", self.$field)); )*
                parts.join(",")
            }
        }
    };
}

define_csv_row! {
    struct CsvRow {
        iteration: usize,
        train_policy_loss: f32,
        train_value_loss: f32,
        val_policy_loss: f32,
        val_value_loss: f32,
        avg_game_length: f64,
        p1_wins: u32,
        p2_wins: u32,
        draws: u32,
        avg_policy_entropy: f64,
        replay_buffer_samples: usize,
        bench_wins: u32,
        bench_losses: u32,
        bench_draws: u32,
        lr: f64,
        q_alpha: f32,
        self_play_secs: f64,
        train_secs: f64,
        bench_secs: f64,
        games: u32,
        samples_this_iter: usize,
        min_game_length: u32,
        max_game_length: u32,
        avg_z: f64,
        avg_q: f64,
        stddev_z: f64,
        stddev_q: f64,
        avg_value_target: f64,
        avg_policy_max_prob: f64,
        games_per_sec: f64,
        samples_per_sec: f64,
        total_elapsed_secs: f64,
        bench_win_rate: f64,
        mcts_sims: u32,
        gumbel_m: u32,
        c_visit: f32,
        c_scale: f32,
        epochs: usize,
        batch_size: usize,
        replay_window: usize,
        explore_moves: u32,
        games_per_iter: usize,
        total_games: u64,
        total_gradient_steps: u64,
    }
}

pub(super) struct CsvLogger {
    writer: BufWriter<std::fs::File>,
}

impl CsvLogger {
    pub fn open(run_dir: &PathBuf, start_iteration: usize) -> Self {
        let csv_path = run_dir.join("metrics.csv");
        let writer = if start_iteration > 0 && csv_path.exists() {
            BufWriter::new(
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&csv_path)
                    .expect("failed to open metrics.csv for append"),
            )
        } else {
            let mut w = BufWriter::new(
                std::fs::File::create(&csv_path).expect("failed to create metrics.csv"),
            );
            // Trim trailing comma from header
            let header = CsvRow::header();
            let header = header.trim_end_matches(',');
            writeln!(w, "{header}").expect("failed to write CSV header");
            w
        };
        CsvLogger { writer }
    }

    pub fn write_row(&mut self, row: &CsvRow) {
        writeln!(self.writer, "{}", row.to_csv_line()).expect("failed to write CSV row");
        self.writer.flush().ok();
    }
}

pub(super) struct IterStats {
    pub avg_entropy: f64,
    pub avg_z: f64,
    pub avg_q: f64,
    pub stddev_z: f64,
    pub stddev_q: f64,
    pub avg_policy_max_prob: f64,
}

pub(super) fn compute_iter_stats(samples: &[Sample]) -> IterStats {
    if samples.is_empty() {
        return IterStats {
            avg_entropy: 0.0,
            avg_z: 0.0,
            avg_q: 0.0,
            stddev_z: 0.0,
            stddev_q: 0.0,
            avg_policy_max_prob: 0.0,
        };
    }

    let n = samples.len() as f64;

    let avg_entropy = samples
        .iter()
        .map(|s| {
            s.policy_target
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -(p as f64) * (p as f64).ln())
                .sum::<f64>()
        })
        .sum::<f64>()
        / n;

    let sum_z: f64 = samples.iter().map(|s| s.z as f64).sum();
    let sum_q: f64 = samples.iter().map(|s| s.q as f64).sum();
    let mean_z = sum_z / n;
    let mean_q = sum_q / n;
    let var_z: f64 = samples
        .iter()
        .map(|s| (s.z as f64 - mean_z).powi(2))
        .sum::<f64>()
        / n;
    let var_q: f64 = samples
        .iter()
        .map(|s| (s.q as f64 - mean_q).powi(2))
        .sum::<f64>()
        / n;

    let avg_policy_max_prob = samples
        .iter()
        .map(|s| s.policy_target.iter().copied().fold(0.0f32, f32::max) as f64)
        .sum::<f64>()
        / n;

    IterStats {
        avg_entropy,
        avg_z: mean_z,
        avg_q: mean_q,
        stddev_z: var_z.sqrt(),
        stddev_q: var_q.sqrt(),
        avg_policy_max_prob,
    }
}
