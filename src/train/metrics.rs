use std::io::BufWriter;
use std::path::PathBuf;

use super::Sample;

#[derive(serde::Serialize)]
pub(super) struct CsvRow {
    pub iteration: usize,
    pub train_policy_loss: f32,
    pub train_value_loss: f32,
    pub val_policy_loss: f32,
    pub val_value_loss: f32,
    pub avg_game_length: f64,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub avg_policy_entropy: f64,
    pub replay_buffer_samples: usize,
    pub bench_wins: u32,
    pub bench_losses: u32,
    pub bench_draws: u32,
    pub lr: f64,
    pub q_alpha: f32,
    pub self_play_secs: f64,
    pub train_secs: f64,
    pub bench_secs: f64,
    pub games: u32,
    pub samples_this_iter: usize,
    pub min_game_length: u32,
    pub max_game_length: u32,
    pub avg_z: f64,
    pub avg_q: f64,
    pub stddev_z: f64,
    pub stddev_q: f64,
    pub avg_value_target: f64,
    pub avg_policy_max_prob: f64,
    pub games_per_sec: f64,
    pub samples_per_sec: f64,
    pub total_elapsed_secs: f64,
    pub bench_win_rate: f64,
    pub mcts_sims: u32,
    pub gumbel_m: u32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub replay_window: usize,
    pub explore_moves: u32,
    pub games_per_iter: usize,
    pub total_games: u64,
    pub total_gradient_steps: u64,
}

pub(super) struct CsvLogger {
    writer: csv::Writer<BufWriter<std::fs::File>>,
}

impl CsvLogger {
    pub fn open(run_dir: &PathBuf, start_iteration: usize) -> Self {
        let csv_path = run_dir.join("metrics.csv");
        let writer = if start_iteration > 0 && csv_path.exists() {
            let buf = BufWriter::new(
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&csv_path)
                    .expect("failed to open metrics.csv for append"),
            );
            csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(buf)
        } else {
            let buf = BufWriter::new(
                std::fs::File::create(&csv_path).expect("failed to create metrics.csv"),
            );
            csv::Writer::from_writer(buf)
        };
        CsvLogger { writer }
    }

    pub fn write_row(&mut self, row: &CsvRow) {
        self.writer.serialize(row).expect("failed to write CSV row");
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
