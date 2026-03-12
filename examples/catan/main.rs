//! # Catan — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots playing Catan.
//!
//! ```text
//! cargo run --example catan -- --p1-simulations 200 --p2-simulations 1000
//! cargo run --example catan -- visualize logs/game_0.log
//! cargo run --example catan -- train --iterations 10 --games 20
//! ```

use std::sync::Arc;

use clap::{Arg, Command};

use canopy2::cli::GameSetup;
use canopy2::eval::RolloutEvaluator;
use canopy2::game_log::GameLog;
use canopy2::train::TrainConfig;

mod encoder;
mod game;
mod heuristic;
mod model;
mod visualize;

use encoder::{BasicEncoder, Gnn2Encoder, GnnEncoder, RichNodeEncoder};
use game::dice::Dice;
use model::{init_gnn, init_gnn_with, init_resnet, init_simple};

fn main() {
    let mut setup = GameSetup::new("catan", "Catan tournament between two MCTS bots");
    setup.add_evaluator("rollout", RolloutEvaluator::default());
    setup.add_evaluator("heuristic", heuristic::HeuristicEvaluator::default());

    // Encoders
    setup.add_encoder("basic", Arc::new(BasicEncoder));
    setup.add_encoder("rich", Arc::new(RichNodeEncoder));
    setup.add_encoder("gnn", Arc::new(GnnEncoder));
    setup.add_encoder("gnn2", Arc::new(Gnn2Encoder));

    // Models
    setup.add_model("simple", init_simple);
    setup.add_model("resnet", init_resnet);
    setup.add_model("gnn", init_gnn);
    setup.add_model("gnn2", init_gnn_with::<_, 101, 34>);

    // Configs
    setup.add_config(
        "small",
        TrainConfig {
            iterations: 1000,
            epochs: 2,
            batch_size: 128,
            replay_window: 10,
            games_per_iter: 100,
            bench_interval: 30,
            bench_baseline_sims: 50,
            ..TrainConfig::default()
        },
    );
    setup.add_config(
        "default",
        TrainConfig {
            iterations: 1000,
            epochs: 3,
            lr: 0.001,
            batch_size: 256,
            replay_window: 10,
            games_per_iter: 150,
            mcts_sims: 800,
            mcts_sims_start: 400,
            bench_games: 20,
            bench_interval: 10,
            bench_baseline_sims: 800,
            ..TrainConfig::default()
        },
    );
    setup.add_config(
        "resnet",
        TrainConfig {
            iterations: 1000,
            epochs: 2,
            lr: 0.0005,
            batch_size: 256,
            replay_window: 20,
            games_per_iter: 150,
            mcts_sims: 800,
            mcts_sims_start: 400,
            bench_games: 20,
            bench_interval: 20,
            bench_baseline_sims: 800,
            ..TrainConfig::default()
        },
    );

    let viz = Command::new("visualize")
        .about("Convert a game log to an HTML visualization")
        .arg(Arg::new("log-file").required(true))
        .arg(
            Arg::new("encoder")
                .long("encoder")
                .help("Embed encoder features in the visualization (gnn or gnn2)"),
        );

    let matches = setup
        .command()
        .subcommand(viz)
        .arg(
            Arg::new("nn-model")
                .long("nn-model")
                .help("Path to neural network checkpoint (for nn evaluator)"),
        )
        .arg(
            Arg::new("encoder")
                .long("encoder")
                .default_value("rich")
                .help("Encoder used for nn evaluator: basic or rich"),
        )
        .arg(
            Arg::new("balanced")
                .long("balanced")
                .action(clap::ArgAction::SetTrue)
                .help("Use balanced dice instead of random"),
        )
        .get_matches();

    // Visualize subcommand
    if let Some(sub) = matches.subcommand_matches("visualize") {
        let log_path = std::path::PathBuf::from(sub.get_one::<String>("log-file").unwrap());
        let game_log = GameLog::read(&log_path);
        let html_path = log_path.with_extension("html");

        if let Some(enc) = sub.get_one::<String>("encoder") {
            visualize::render_with_encoder_dispatch(enc, &game_log, &html_path);
            println!("Wrote {} (with {} features)", html_path.display(), enc);
            return;
        }

        visualize::render(&game_log, &html_path);
        println!("Wrote {}", html_path.display());
        return;
    }

    let dice = if matches.get_flag("balanced") {
        Dice::Balanced(game::dice::BalancedDice::new())
    } else {
        Dice::Random
    };

    // Train subcommand
    if let Some(sub) = matches.subcommand_matches("train") {
        setup.run_train(sub, move |rng| game::new_game(rng.u64(..), dice));
        return;
    }

    // Tournament mode
    if let Some(path) = matches.get_one::<String>("nn-model") {
        let enc = matches.get_one::<String>("encoder").unwrap().as_str();
        let device = canopy2::train::default_device();
        setup.add_evaluator_arc("nn", encoder::load_evaluator(enc, path, &device));
    }
    setup.run_tournament(&matches, move |seed| game::new_game(seed, dice));
}
