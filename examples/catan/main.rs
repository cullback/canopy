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

mod colonist;
mod encoder;
mod game;
mod model;
mod presenter;
mod visualize;

use encoder::NexusEncoder;
use game::dice::Dice;
use model::init_nexus;

fn main() {
    let mut setup = GameCli::new("catan", "Catan tournament between two MCTS bots");
    setup.add_evaluator("rollout", canopy::eval::RolloutEvaluator::default());

    // Encoders
    setup.add_encoder("nexus", Arc::new(NexusEncoder));

    // Models
    setup.add_model("nexus", |device, cfg| {
        init_nexus::<_, 256, 96, 4>(device, cfg.aux_value_horizons.len())
    });
    setup.add_model("nexus-large", |device, cfg| {
        init_nexus::<_, 384, 128, 4>(device, cfg.aux_value_horizons.len())
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
            filter_legal: true,
            mcts_sims: 1200,
            train_batch_size: 1024,
            leaf_batch_size: 8,
            concurrent_games: 512,
            gumbel_m: 16,
            explore_actions: 16,
            q_weight_ramp_iters: 60,
            aux_value_horizons: vec![10, 50, 150],
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
        .arg(
            Arg::new("vp-limit")
                .long("vp-limit")
                .default_value("15")
                .help("Victory points needed to win"),
        )
        .arg(
            Arg::new("discard-threshold")
                .long("discard-threshold")
                .default_value("9")
                .help("Hand size above which players must discard on a 7"),
        )
        .subcommand(
            clap::Command::new("colonist")
                .about("Connect to colonist.io via CDP and read game log")
                .arg(
                    Arg::new("port")
                        .long("port")
                        .default_value("9223")
                        .help("Chrome debug port"),
                )
                .arg(
                    Arg::new("serve")
                        .long("serve")
                        .help("Serve board via web UI on this port"),
                )
                .arg(
                    Arg::new("leaf-batch")
                        .long("leaf-batch")
                        .default_value("1")
                        .help("Leaves per GPU batch (higher = better GPU utilization)"),
                ),
        )
        .get_matches();

    let dice = if matches.get_flag("random-dice") {
        Dice::Random
    } else {
        Dice::Balanced(game::dice::BalancedDice::new())
    };

    let vp_limit: u8 = matches
        .get_one::<String>("vp-limit")
        .unwrap()
        .parse()
        .expect("invalid vp-limit");
    let discard_threshold: u8 = matches
        .get_one::<String>("discard-threshold")
        .unwrap()
        .parse()
        .expect("invalid discard-threshold");

    // Colonist CDP subcommand
    if let Some(sub) = matches.subcommand_matches("colonist") {
        let cdp_port: u16 = sub
            .get_one::<String>("port")
            .unwrap()
            .parse()
            .expect("invalid port");
        if let Some(serve_port) = sub.get_one::<String>("serve") {
            let serve_port: u16 = serve_port.parse().expect("invalid serve port");
            setup.load_nn_evaluator(&matches);
            let has_nn = matches.get_one::<String>("nn-model").is_some();
            let (eval_name, evaluator) = if has_nn {
                ("nn", setup.evaluators().get_arc("nn"))
            } else {
                ("rollout", setup.evaluators().get_arc("rollout"))
            };
            let leaf_batch: u32 = sub
                .get_one::<String>("leaf-batch")
                .unwrap()
                .parse()
                .expect("invalid leaf-batch");
            colonist::run_serve(cdp_port, serve_port, evaluator, eval_name, leaf_batch);
        } else {
            colonist::run(cdp_port);
        }
        return;
    }

    // Serve subcommand
    if let Some(sub) = matches.subcommand_matches("serve") {
        let static_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/catan/web");
        let presenter = Arc::new(presenter::CatanPresenter::new(static_dir, dice));
        setup.run_serve(&matches, sub, presenter);
        return;
    }

    setup.run(&matches, move |seed| {
        game::new_game(seed, dice, vp_limit, discard_threshold)
    });
}
