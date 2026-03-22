//! # Catan — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots playing Catan.
//!
//! ```text
//! cargo run --example catan -- --p1-sims 200 --p2-sims 1000
//! cargo run --example catan -- visualize logs/game_0.log
//! cargo run --example catan -- train --iterations 10 --games 20
//! ```

use std::sync::Arc;

use clap::{Arg, Command};

use canopy::cli::GameCli;
use canopy::game_log::GameLog;
use canopy::train::TrainConfig;

mod encoder;
mod game;
mod heuristic;
mod model;
mod presenter;
mod visualize;

use encoder::NexusEncoder;
use game::dice::Dice;
use model::init_nexus_with;

fn main() {
    let mut setup = GameCli::new("catan", "Catan tournament between two MCTS bots");
    setup.add_evaluator("heuristic", heuristic::HeuristicEvaluator::default());

    // Encoders
    setup.add_encoder("nexus", Arc::new(NexusEncoder));

    // Models
    setup.add_model("nexus", init_nexus_with);

    // Configs
    setup.add_config(
        "nexus",
        TrainConfig {
            iterations: 1000,
            games_per_iter: 500,
            epochs: 1,
            lr: 0.0005,
            replay_window: 10,
            mcts_sims: 400,
            mcts_sims_start: 400,
            train_batch_size: 4096,
            leaf_batch_size: 32,
            bench_games: 0,
            warmup_iters: 60,
            ..TrainConfig::default()
        },
    );

    let viz = Command::new("visualize")
        .about("Convert a game log to an HTML visualization")
        .arg(Arg::new("log-file").required(true));

    let matches = setup
        .command()
        .subcommand(viz)
        .arg(
            Arg::new("random-dice")
                .long("random-dice")
                .action(clap::ArgAction::SetTrue)
                .help("Use random dice instead of balanced (default)"),
        )
        .get_matches();

    // Visualize subcommand
    if let Some(sub) = matches.subcommand_matches("visualize") {
        let log_path = std::path::PathBuf::from(sub.get_one::<String>("log-file").unwrap());
        let game_log = GameLog::read(&log_path);
        let html_path = log_path.with_extension("html");

        visualize::render(&game_log, &html_path);
        println!("Wrote {}", html_path.display());
        return;
    }

    let dice = if matches.get_flag("random-dice") {
        Dice::Random
    } else {
        Dice::Balanced(game::dice::BalancedDice::new())
    };

    // Serve subcommand
    if let Some(sub) = matches.subcommand_matches("serve") {
        let static_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/catan/web");
        let presenter = Arc::new(presenter::CatanPresenter::new(static_dir, dice));
        setup.run_serve(&matches, sub, presenter);
        return;
    }

    setup.run(&matches, move |seed| game::new_game(seed, dice));
}
