use std::io::BufWriter;
use std::path::PathBuf;

use super::Sample;

mod round6 {
    use serde::Serializer;

    pub fn f64<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64((v * 1e6).round() / 1e6)
    }

    pub fn f32<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f32((v * 1e6).round() / 1e6)
    }
}

#[derive(Default, serde::Serialize)]
pub(super) struct CsvRow {
    pub iteration: usize,
    #[serde(serialize_with = "round6::f32")]
    pub train_policy_loss: f32,
    #[serde(serialize_with = "round6::f32")]
    pub train_value_loss: f32,
    #[serde(serialize_with = "round6::f32")]
    pub val_policy_loss: f32,
    #[serde(serialize_with = "round6::f32")]
    pub val_value_loss: f32,
    #[serde(serialize_with = "round6::f64")]
    pub avg_game_length: f64,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    #[serde(serialize_with = "round6::f64")]
    pub avg_policy_entropy: f64,
    pub replay_buffer_samples: usize,
    pub bench_wins: u32,
    pub bench_losses: u32,
    pub bench_draws: u32,
    #[serde(serialize_with = "round6::f64")]
    pub lr: f64,
    #[serde(serialize_with = "round6::f32")]
    pub q_alpha: f32,
    #[serde(serialize_with = "round6::f64")]
    pub self_play_secs: f64,
    #[serde(serialize_with = "round6::f64")]
    pub train_secs: f64,
    #[serde(serialize_with = "round6::f64")]
    pub bench_secs: f64,
    pub samples_this_iter: usize,
    pub min_game_length: u32,
    pub max_game_length: u32,
    #[serde(serialize_with = "round6::f64")]
    pub avg_z: f64,
    #[serde(serialize_with = "round6::f64")]
    pub avg_q: f64,
    #[serde(serialize_with = "round6::f64")]
    pub stddev_z: f64,
    #[serde(serialize_with = "round6::f64")]
    pub stddev_q: f64,
    #[serde(serialize_with = "round6::f64")]
    pub avg_policy_max_prob: f64,
    #[serde(serialize_with = "round6::f64")]
    pub avg_entropy_high_branch: f64,
    #[serde(serialize_with = "round6::f64")]
    pub avg_max_prob_high_branch: f64,
    pub mcts_sims: u32,
    pub gradient_steps: usize,
}

pub(super) struct CsvLogger {
    writer: csv::Writer<BufWriter<std::fs::File>>,
}

/// Get the canonical header by serializing `CsvRow::default()`.
fn canonical_header() -> Vec<String> {
    let mut wtr = csv::Writer::from_writer(Vec::new());
    wtr.serialize(&CsvRow::default()).unwrap();
    let data = String::from_utf8(wtr.into_inner().unwrap()).unwrap();
    data.lines()
        .next()
        .unwrap()
        .split(',')
        .map(String::from)
        .collect()
}

/// Migrate an existing CSV to match the current header.
///
/// Reorders columns to match the canonical order, drops columns that no longer
/// exist, and fills new columns with empty strings for old rows.
fn migrate_csv(csv_path: &std::path::Path, header: &[String]) {
    let content = std::fs::read_to_string(csv_path).expect("failed to read metrics.csv");
    let mut reader = csv::Reader::from_reader(content.as_bytes());
    let old_header: Vec<String> = reader
        .headers()
        .expect("failed to read CSV header")
        .iter()
        .map(String::from)
        .collect();

    if old_header == header {
        return;
    }

    eprintln!(
        "migrating metrics.csv: {} -> {} columns",
        old_header.len(),
        header.len()
    );

    // Build mapping: for each canonical column, find its index in the old header (if any)
    let old_index: Vec<Option<usize>> = header
        .iter()
        .map(|col| old_header.iter().position(|old| old == col))
        .collect();

    let mut output = csv::Writer::from_writer(BufWriter::new(
        std::fs::File::create(csv_path).expect("failed to rewrite metrics.csv"),
    ));
    // Write new header
    output.write_record(header).unwrap();

    // Rewrite each row
    for result in reader.records() {
        let record = result.expect("failed to read CSV record");
        let row: Vec<&str> = old_index
            .iter()
            .map(|idx| idx.map_or("", |i| record.get(i).unwrap_or("")))
            .collect();
        output.write_record(&row).unwrap();
    }
    output.flush().unwrap();
}

impl CsvLogger {
    pub fn open(run_dir: &PathBuf, start_iteration: usize) -> Self {
        let csv_path = run_dir.join("metrics.csv");
        let header = canonical_header();

        let writer = if start_iteration > 0 && csv_path.exists() {
            migrate_csv(&csv_path, &header);
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
    /// Entropy averaged over samples with ≥6 legal actions.
    pub avg_entropy_high_branch: f64,
    /// Max policy prob averaged over samples with ≥6 legal actions.
    pub avg_max_prob_high_branch: f64,
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
            avg_entropy_high_branch: 0.0,
            avg_max_prob_high_branch: 0.0,
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

    // Stats over high-branching positions only (≥6 legal actions)
    let mut hb_entropy_sum = 0.0f64;
    let mut hb_max_prob_sum = 0.0f64;
    let mut hb_count = 0u64;
    for s in samples {
        let legal = s.policy_target.iter().filter(|&&p| p > 0.0).count();
        if legal >= 6 {
            hb_entropy_sum += s
                .policy_target
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -(p as f64) * (p as f64).ln())
                .sum::<f64>();
            hb_max_prob_sum += s.policy_target.iter().copied().fold(0.0f32, f32::max) as f64;
            hb_count += 1;
        }
    }
    let hb_n = hb_count.max(1) as f64;

    IterStats {
        avg_entropy,
        avg_z: mean_z,
        avg_q: mean_q,
        stddev_z: var_z.sqrt(),
        stddev_q: var_q.sqrt(),
        avg_policy_max_prob,
        avg_entropy_high_branch: hb_entropy_sum / hb_n,
        avg_max_prob_high_branch: hb_max_prob_sum / hb_n,
    }
}
