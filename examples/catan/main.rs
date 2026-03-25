//! # Catan — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots playing Catan.
//!
//! ```text
//! cargo run --example catan -- --p1-sims 200 --p2-sims 1000
//! cargo run --example catan -- train --iterations 10 --games 20
//! ```

use std::sync::Arc;

use clap::Arg;

use canopy::cli::GameCli;
use canopy::train::TrainConfig;

mod encoder;
mod game;
mod heuristic;
mod model;
mod presenter;
mod visualize;

use encoder::NexusEncoder;
use game::dice::Dice;
use model::init_nexus;

fn main() {
    let mut setup = GameCli::new("catan", "Catan tournament between two MCTS bots");
    setup.add_evaluator("heuristic", heuristic::HeuristicEvaluator::default());

    // Encoders
    setup.add_encoder("nexus", Arc::new(NexusEncoder));

    // Models
    setup.add_model("nexus", |device, cfg| {
        init_nexus(device, cfg.aux_value_horizons.len())
    });

    // Configs
    setup.add_config(
        "nexus",
        TrainConfig {
            iterations: 1000,
            train_samples_per_iter: 100_000,
            replay_buffer_samples: 350_000,
            max_actions: 2000,
            epochs: 2,
            lr: 0.0002,
            mcts_sims: 800,
            train_batch_size: 1024,
            leaf_batch_size: 8,
            concurrent_games: 512,
            gumbel_m: 16,
            explore_actions: 16,
            q_weight_ramp_iters: 60,
            aux_value_horizons: vec![10, 30, 100],
            ..TrainConfig::default()
        },
    );

    let matches = setup
        .command()
        .arg(
            Arg::new("random-dice")
                .long("random-dice")
                .action(clap::ArgAction::SetTrue)
                .help("Use random dice instead of balanced (default)"),
        )
        .get_matches();

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
