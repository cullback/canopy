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

use encoder::{BasicEncoder, Gnn2Encoder, GnnEncoder, NexusEncoder, RichNodeEncoder};
use game::dice::Dice;
use model::{init_gnn, init_gnn_with, init_nexus_with, init_resnet, init_simple, init_simple_rich};

fn main() {
    let mut setup = GameCli::new("catan", "Catan tournament between two MCTS bots");
    setup.add_evaluator("heuristic", heuristic::HeuristicEvaluator::default());

    // Encoders
    setup.add_encoder("basic", Arc::new(BasicEncoder));
    setup.add_encoder("rich", Arc::new(RichNodeEncoder));
    setup.add_encoder("gnn", Arc::new(GnnEncoder));
    setup.add_encoder("gnn2", Arc::new(Gnn2Encoder));
    setup.add_encoder("nexus", Arc::new(NexusEncoder));

    // Models
    setup.add_model("simple", init_simple);
    setup.add_model("simple-rich", init_simple_rich);
    setup.add_model("resnet", init_resnet);
    setup.add_model("gnn", init_gnn);
    setup.add_model("gnn2", init_gnn_with::<_, 101, 34>);
    setup.add_model("nexus", init_nexus_with::<_, 93, 7, 13>);

    // Configs
    setup.add_config(
        "small",
        TrainConfig {
            iterations: 1000,
            epochs: 2,
            replay_window: 10,
            bench_interval: 30,
            bench_sims: 50,
            ..TrainConfig::default()
        },
    );
    setup.add_config(
        "default",
        TrainConfig {
            iterations: 1000,
            epochs: 3,
            lr: 0.001,
            replay_window: 10,
            mcts_sims_start: 400,
            bench_games: 20,
            bench_interval: 10,
            bench_sims: 800,
            ..TrainConfig::default()
        },
    );
    setup.add_config(
        "resnet",
        TrainConfig {
            iterations: 1000,
            epochs: 2,
            lr: 0.0005,
            replay_window: 20,
            mcts_sims_start: 400,
            ..TrainConfig::default()
        },
    );
    setup.add_config(
        "nexus",
        TrainConfig {
            iterations: 1000,
            games_per_iter: 200,
            epochs: 2,
            lr: 0.0005,
            replay_window: 10,
            mcts_sims: 2500,
            mcts_sims_start: 400,
            leaf_batch_size: 32,
            bench_games: 0,
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

        if let Some(enc) = sub.get_one::<String>("encoder") {
            visualize::render_with_encoder_dispatch(enc, &game_log, &html_path);
            println!("Wrote {} (with {} features)", html_path.display(), enc);
            return;
        }

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
