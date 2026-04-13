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

use encoder::{NexusEncoderV1, NexusEncoderV2, NexusEncoderV3};
use game::dice::Dice;
use model::{init_nexus_v1, init_nexus_v2, init_nexus_v3};

fn main() {
    let mut setup = GameCli::new("catan", "Catan tournament between two MCTS bots");
    setup.add_evaluator("rollout", canopy::eval::RolloutEvaluator::default());

    // Encoders
    setup.add_encoder("nexus-v1", Arc::new(NexusEncoderV1));
    setup.add_encoder("nexus-v2", Arc::new(NexusEncoderV2));
    setup.add_encoder("nexus-v3", Arc::new(NexusEncoderV3));

    // Models
    setup.add_model("nexus-v1", |device, cfg| {
        init_nexus_v1::<_, 256, 96, 4>(device, cfg.aux_value_horizons.len())
    });
    setup.add_model("nexus-v2", |device, cfg| {
        init_nexus_v2(device, cfg.aux_value_horizons.len())
    });
    setup.add_model("nexus-v3", |device, cfg| {
        init_nexus_v3(device, cfg.aux_value_horizons.len())
    });

    // Configs
    setup.add_config(
        "nexus-v2",
        TrainConfig {
            iterations: 1000,
            train_samples_per_iter: 150_000,
            replay_buffer_samples: 500_000,
            max_actions: 2000,
            epochs: 2,
            lr: 0.0005,
            filter_legal: true,
            mcts_sims: 3200,
            inference_batch_size: 2048,
            train_batch_size: 1024,
            leaf_batch_size: 32,
            concurrent_games: 1536,
            gumbel_m: 16,
            explore_actions: 16,
            q_weight_ramp_iters: 60,
            aux_value_horizons: vec![10, 50, 150],
            ..TrainConfig::default()
        },
    );
    setup.add_config(
        "nexus-v3",
        TrainConfig {
            iterations: 1000,
            train_samples_per_iter: 150_000,
            replay_buffer_samples: 500_000,
            max_actions: 2000,
            epochs: 2,
            lr: 0.0001,
            filter_legal: true,
            mcts_sims: 1600,
            inference_batch_size: 2048,
            train_batch_size: 1024,
            leaf_batch_size: 32,
            concurrent_games: 768,
            gumbel_m: 6,
            explore_actions: 24,
            q_weight_ramp_iters: 60,
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
                    Arg::new("eval")
                        .long("eval")
                        .default_value("rollout")
                        .help("Evaluator (name or checkpoint path)"),
                )
                .arg(
                    Arg::new("leaf-batch")
                        .long("leaf-batch")
                        .default_value("1")
                        .help("Leaves per GPU batch (higher = better GPU utilization)"),
                )
                .arg(
                    Arg::new("gumbel-m")
                        .long("gumbel-m")
                        .default_value("4")
                        .help("Gumbel-Top-k sampled actions at root"),
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
            let eval_spec = sub.get_one::<String>("eval").unwrap().clone();
            setup.resolve_eval_spec(&eval_spec, "colonist-nn");
            let eval_name = if eval_spec.contains('/') || eval_spec.ends_with(".mpk") {
                "colonist-nn"
            } else {
                eval_spec.as_str()
            };
            let evaluator = setup.evaluators().get_arc(eval_name);
            let leaf_batch: u32 = sub
                .get_one::<String>("leaf-batch")
                .unwrap()
                .parse()
                .expect("invalid leaf-batch");
            let gumbel_m: u32 = sub
                .get_one::<String>("gumbel-m")
                .unwrap()
                .parse()
                .expect("invalid gumbel-m");
            colonist::run_serve(
                cdp_port, serve_port, evaluator, eval_name, leaf_batch, gumbel_m,
            );
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
        setup.run_serve(sub, presenter);
        return;
    }

    setup.run(&matches, move |seed| {
        game::new_game(seed, dice, vp_limit, discard_threshold)
    });
}
