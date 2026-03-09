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

/// One row of `metrics.csv`. Field order determines CSV column order.
///
/// Columns are grouped logically:
/// - **Core training**: losses, gradient steps
/// - **Self-play game stats**: game lengths, win/draw counts
/// - **Policy diagnostics**: entropy, confidence, network-vs-search agreement
/// - **Value diagnostics**: Z/Q targets, search correction, phase-split error
/// - **Benchmark**: win/loss/draw vs baseline opponent
/// - **Config/infra**: hyperparams and timing
#[derive(Default, serde::Serialize)]
pub(super) struct CsvRow {
    // ── Core training ────────────────────────────────────────────────
    pub iteration: usize,
    #[serde(serialize_with = "round6::f32")]
    pub loss_policy_train: f32,
    #[serde(serialize_with = "round6::f32")]
    pub loss_value_train: f32,
    /// Validation policy loss (held-out split of replay buffer).
    #[serde(serialize_with = "round6::f32")]
    pub loss_policy_val: f32,
    /// Validation value loss (held-out split of replay buffer).
    #[serde(serialize_with = "round6::f32")]
    pub loss_value_val: f32,
    /// Total optimizer updates across all epochs this iteration.
    pub gradient_steps: usize,

    // ── Self-play game stats ─────────────────────────────────────────
    #[serde(serialize_with = "round6::f64")]
    pub game_length_avg: f64,
    /// Spread of game lengths. Collapse toward zero signals a degenerate
    /// strategy; bimodal distributions show up as high stddev.
    #[serde(serialize_with = "round6::f64")]
    pub game_length_stddev: f64,
    pub game_length_min: u32,
    pub game_length_max: u32,
    pub game_p1_wins: u32,
    pub game_p2_wins: u32,
    pub game_draws: u32,

    // ── Policy diagnostics ───────────────────────────────────────────
    /// Entropy of the MCTS improved policy (training target). Healthy runs
    /// dip then plateau; collapse to ~0 means exploration is dead.
    #[serde(serialize_with = "round6::f64")]
    pub policy_entropy_avg: f64,
    /// Average max probability in the improved policy. Complement of entropy:
    /// high values mean the policy concentrates on a single move.
    #[serde(serialize_with = "round6::f64")]
    pub policy_max_prob_avg: f64,
    /// Entropy restricted to positions with ≥6 legal actions, filtering out
    /// forced/near-forced moves that naturally have low entropy.
    #[serde(serialize_with = "round6::f64")]
    pub policy_entropy_high_branch_avg: f64,
    /// Max prob restricted to positions with ≥6 legal actions.
    #[serde(serialize_with = "round6::f64")]
    pub policy_max_prob_high_branch_avg: f64,
    /// Fraction of moves where network's top-1 matches MCTS's selected action.
    /// Should rise over training; plateau at ~40-50% means the network isn't
    /// distilling search. Jump to ~95%+ early means search isn't contributing.
    #[serde(serialize_with = "round6::f64")]
    pub policy_agreement_avg: f64,

    // ── Value diagnostics ────────────────────────────────────────────
    /// Mean game outcome (Z) from current player's perspective. Near 0 =
    /// balanced play; persistent bias means one side is stronger.
    #[serde(serialize_with = "round6::f64")]
    pub value_z_avg: f64,
    /// Mean MCTS root Q from current player's perspective.
    #[serde(serialize_with = "round6::f64")]
    pub value_q_avg: f64,
    #[serde(serialize_with = "round6::f64")]
    pub value_z_stddev: f64,
    #[serde(serialize_with = "round6::f64")]
    pub value_q_stddev: f64,
    /// Mean |Q_search − V_network|. How much search corrects the raw network
    /// value. Should shrink over training but never reach zero.
    #[serde(serialize_with = "round6::f64")]
    pub value_correction_avg: f64,
    /// Std of Q values across visited root children. Measures value head
    /// discriminative power: very small = can't distinguish moves, very
    /// large = playing refutation-style (one good move, rest terrible).
    #[serde(serialize_with = "round6::f64")]
    pub value_q_spread_avg: f64,
    /// Mean |Q − Z| for first-half-of-game positions. Compare with late to
    /// catch the failure mode where openings are confidently wrong.
    #[serde(serialize_with = "round6::f64")]
    pub value_error_early_avg: f64,
    /// Mean |Q − Z| for second-half-of-game positions.
    #[serde(serialize_with = "round6::f64")]
    pub value_error_late_avg: f64,
    /// Stddev of raw network value outputs. Near zero = value head outputs
    /// a constant; spread comparable to value_q_stddev = actually differentiating.
    #[serde(serialize_with = "round6::f64")]
    pub value_network_stddev: f64,

    // ── Benchmark ────────────────────────────────────────────────────
    pub bench_wins: u32,
    pub bench_losses: u32,
    pub bench_draws: u32,

    // ── Config/infra ─────────────────────────────────────────────────
    /// Cosine-annealed learning rate for this iteration.
    #[serde(serialize_with = "round6::f64")]
    pub lr: f64,
    /// Weight of Q in value target blend (0 = pure Z, 1 = pure Q).
    /// Ramps from 0→1 over `warmup_iters`.
    #[serde(serialize_with = "round6::f32")]
    pub q_weight: f32,
    /// Effective MCTS simulations this iteration (may ramp up during warmup).
    pub mcts_sims: u32,
    /// Total samples in the replay buffer (across all retained iterations).
    pub replay_samples: usize,
    /// Samples generated this iteration only.
    pub samples_iter: usize,
    #[serde(serialize_with = "round6::f64")]
    pub time_selfplay_secs: f64,
    #[serde(serialize_with = "round6::f64")]
    pub time_train_secs: f64,
    #[serde(serialize_with = "round6::f64")]
    pub time_bench_secs: f64,
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
    pub policy_entropy_avg: f64,
    pub policy_max_prob_avg: f64,
    /// Entropy averaged over samples with ≥6 legal actions.
    pub policy_entropy_high_branch_avg: f64,
    /// Max policy prob averaged over samples with ≥6 legal actions.
    pub policy_max_prob_high_branch_avg: f64,
    /// Fraction of moves where network top-1 == MCTS selected action.
    pub policy_agreement_avg: f64,
    pub value_z_avg: f64,
    pub value_q_avg: f64,
    pub value_z_stddev: f64,
    pub value_q_stddev: f64,
    /// Mean |Q_search - V_network|.
    pub value_correction_avg: f64,
    /// Mean std of Q across visited root children.
    pub value_q_spread_avg: f64,
    /// Mean |q - z| for early-game samples (move_number <= game_length / 2).
    pub value_error_early_avg: f64,
    /// Mean |q - z| for late-game samples (move_number > game_length / 2).
    pub value_error_late_avg: f64,
    /// Stddev of raw network value outputs. Near zero means the value head
    /// is outputting a constant; healthy spread means it differentiates positions.
    pub value_network_stddev: f64,
}

pub(super) fn compute_iter_stats(samples: &[Sample]) -> IterStats {
    if samples.is_empty() {
        return IterStats {
            policy_entropy_avg: 0.0,
            policy_max_prob_avg: 0.0,
            policy_entropy_high_branch_avg: 0.0,
            policy_max_prob_high_branch_avg: 0.0,
            policy_agreement_avg: 0.0,
            value_z_avg: 0.0,
            value_q_avg: 0.0,
            value_z_stddev: 0.0,
            value_q_stddev: 0.0,
            value_correction_avg: 0.0,
            value_q_spread_avg: 0.0,
            value_error_early_avg: 0.0,
            value_error_late_avg: 0.0,
            value_network_stddev: 0.0,
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

    let avg_policy_agreement = samples.iter().filter(|s| s.prior_agrees).count() as f64 / n;
    let avg_value_correction = samples
        .iter()
        .map(|s| s.value_correction as f64)
        .sum::<f64>()
        / n;
    let avg_q_std = samples.iter().map(|s| s.q_std as f64).sum::<f64>() / n;

    // Value error stratified by game phase
    let mut early_err_sum = 0.0f64;
    let mut early_count = 0u64;
    let mut late_err_sum = 0.0f64;
    let mut late_count = 0u64;
    for s in samples {
        let err = (s.q - s.z).abs() as f64;
        if s.game_length > 0 && s.move_number <= s.game_length / 2 {
            early_err_sum += err;
            early_count += 1;
        } else {
            late_err_sum += err;
            late_count += 1;
        }
    }
    let avg_value_error_early = if early_count > 0 {
        early_err_sum / early_count as f64
    } else {
        0.0
    };
    let avg_value_error_late = if late_count > 0 {
        late_err_sum / late_count as f64
    } else {
        0.0
    };

    let sum_nv: f64 = samples.iter().map(|s| s.network_value as f64).sum();
    let mean_nv = sum_nv / n;
    let var_nv: f64 = samples
        .iter()
        .map(|s| (s.network_value as f64 - mean_nv).powi(2))
        .sum::<f64>()
        / n;

    IterStats {
        policy_entropy_avg: avg_entropy,
        policy_max_prob_avg: avg_policy_max_prob,
        policy_entropy_high_branch_avg: hb_entropy_sum / hb_n,
        policy_max_prob_high_branch_avg: hb_max_prob_sum / hb_n,
        policy_agreement_avg: avg_policy_agreement,
        value_z_avg: mean_z,
        value_q_avg: mean_q,
        value_z_stddev: var_z.sqrt(),
        value_q_stddev: var_q.sqrt(),
        value_correction_avg: avg_value_correction,
        value_q_spread_avg: avg_q_std,
        value_error_early_avg: avg_value_error_early,
        value_error_late_avg: avg_value_error_late,
        value_network_stddev: var_nv.sqrt(),
    }
}
