use std::path::{Path, PathBuf};
use std::sync::Arc;

use canopy2::eval::Evaluator;
use canopy2::mcts::Config;
use canopy2::nn::NeuralEvaluator;
use canopy2::tournament::play_match;
use canopy2::train::{InferBackend, default_device};
use indicatif::{ProgressBar, ProgressStyle};

use crate::encoder::TenureEncoder;
use crate::game::TenureGame;
use crate::model::init_tenure;

struct Contestant {
    name: String,
    evaluator: Arc<dyn Evaluator<TenureGame> + Sync>,
}

/// Load all model checkpoints from a directory, sorted by iteration.
fn load_checkpoints(dir: &Path) -> Vec<Contestant> {
    let mut entries: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)
        .expect("failed to read checkpoint directory")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("model_iter_") && name.ends_with(".mpk") {
                let iter_str = name.strip_prefix("model_iter_")?.strip_suffix(".mpk")?;
                let iter: u32 = iter_str.parse().ok()?;
                // Path without .mpk extension (burn expects stem only)
                let path = dir.join(format!("model_iter_{iter}"));
                Some((iter, path))
            } else {
                None
            }
        })
        .filter(|(iter, _)| iter % 20 == 0)
        .collect();

    entries.sort_by_key(|(iter, _)| *iter);

    let device = default_device();
    let encoder = Arc::new(TenureEncoder);

    entries
        .into_iter()
        .map(|(iter, path)| {
            let model = init_tenure::<InferBackend>(&device);
            let eval = NeuralEvaluator::from_checkpoint(
                encoder.clone() as Arc<_>,
                model,
                path,
                device.clone(),
            );
            Contestant {
                name: format!("iter_{iter}"),
                evaluator: Arc::new(eval),
            }
        })
        .collect()
}

/// Run a round-robin tournament between all model checkpoints.
pub fn run(dir: &Path, num_games: u32, simulations: u32) {
    let contestants = load_checkpoints(dir);
    let n = contestants.len();
    println!(
        "Round-robin: {} checkpoints, {} games per pairing, {} simulations\n",
        n, num_games, simulations
    );

    if n < 2 {
        println!("Need at least 2 checkpoints.");
        return;
    }

    let config = Config {
        num_simulations: simulations,
        ..Config::default()
    };
    let configs = [config.clone(), config];

    let total_pairings = n * (n - 1) / 2;
    let pb = ProgressBar::new(total_pairings as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} pairings | {elapsed} < {eta} | {msg}")
            .unwrap(),
    );

    // wins[i][j] = wins for contestant i against contestant j
    let mut wins = vec![vec![0u32; n]; n];
    let mut draws = vec![vec![0u32; n]; n];
    let mut rng = fastrand::Rng::new();

    for i in 0..n {
        for j in (i + 1)..n {
            pb.set_message(format!(
                "{} vs {}",
                contestants[i].name, contestants[j].name
            ));

            let evaluators: [&dyn Evaluator<TenureGame>; 2] =
                [&*contestants[i].evaluator, &*contestants[j].evaluator];

            for g in 0..num_games {
                let swap = g % 2 == 1;
                let seed = rng.u64(..);
                let game_rng = &mut fastrand::Rng::with_seed(seed);
                let game = TenureGame::random(game_rng);
                let (reward, _) = play_match(&game, &evaluators, &configs, swap, &mut rng);

                let seat0_reward = if swap { -reward } else { reward };
                if seat0_reward > 0.0 {
                    wins[i][j] += 1;
                } else if seat0_reward < 0.0 {
                    wins[j][i] += 1;
                } else {
                    draws[i][j] += 1;
                    draws[j][i] += 1;
                }
            }

            pb.inc(1);
        }
    }

    pb.finish_and_clear();

    // Aggregate stats
    struct Stats {
        name: String,
        total_wins: u32,
        total_losses: u32,
        total_draws: u32,
        win_rate: f64,
    }

    let mut stats: Vec<Stats> = (0..n)
        .map(|i| {
            let total_wins: u32 = wins[i].iter().sum();
            let total_losses: u32 = (0..n).map(|j| wins[j][i]).sum();
            let total_draws: u32 = draws[i].iter().sum::<u32>() / 2; // counted twice
            let games_played = (n as u32 - 1) * num_games;
            Stats {
                name: contestants[i].name.clone(),
                total_wins,
                total_losses,
                total_draws,
                win_rate: total_wins as f64 / games_played as f64,
            }
        })
        .collect();

    stats.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap());

    // Print leaderboard
    println!(
        "\n{:>4}  {:<12} {:>6} {:>6} {:>6} {:>8}",
        "Rank", "Model", "W", "L", "D", "Win%"
    );
    println!("{}", "-".repeat(50));
    for (rank, s) in stats.iter().enumerate() {
        println!(
            "{:>4}  {:<12} {:>6} {:>6} {:>6} {:>7.1}%",
            rank + 1,
            s.name,
            s.total_wins,
            s.total_losses,
            s.total_draws,
            s.win_rate * 100.0,
        );
    }

    // Print head-to-head matrix
    println!("\nHead-to-head (row wins vs column):");
    print!("{:<12}", "");
    for c in &contestants {
        print!(" {:>9}", c.name);
    }
    println!();
    for i in 0..n {
        print!("{:<12}", contestants[i].name);
        for j in 0..n {
            if i == j {
                print!(" {:>9}", "-");
            } else {
                print!(" {:>4}-{:<4}", wins[i][j], wins[j][i],);
            }
        }
        println!();
    }
}
