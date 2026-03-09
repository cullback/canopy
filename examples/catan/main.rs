//! # Catan — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots playing Catan.
//!
//! ```text
//! cargo run --example catan -- --p1-simulations 200 --p2-simulations 1000
//! cargo run --example catan -- visualize logs/game_0.log
//! cargo run --example catan --features nn -- train --iterations 10 --games 20
//! ```

use clap::{Arg, Command};

use canopy2::cli;
use canopy2::eval::{Evaluator, RolloutEvaluator};
use canopy2::game_log::GameLog;
use canopy2::player::PerPlayer;
use canopy2::tournament;

mod game;
mod heuristic;
mod visualize;

#[cfg(feature = "nn")]
mod encoder;
#[cfg(feature = "nn")]
mod model;

use game::dice::Dice;
use game::state::GameState;

fn app() -> Command {
    let mut cmd = Command::new("catan")
        .about("Catan tournament between two MCTS bots")
        .subcommand(
            Command::new("visualize")
                .about("Convert a game log to an HTML visualization")
                .arg(Arg::new("log-file").required(true)),
        );

    // Train subcommand (always visible, but requires nn feature to run)
    #[cfg(feature = "nn")]
    {
        cmd = cmd.subcommand(train_command());
    }

    cmd = cmd
        .arg(
            Arg::new("num-games")
                .short('n')
                .long("num-games")
                .default_value("20"),
        )
        .arg(
            Arg::new("log-dir")
                .long("log-dir")
                .help("Directory to write game logs"),
        )
        .arg(
            Arg::new("p1-eval")
                .long("p1-eval")
                .default_value("rollout")
                .help("Evaluator for player 1: rollout, heuristic, or nn"),
        )
        .arg(
            Arg::new("p2-eval")
                .long("p2-eval")
                .default_value("rollout")
                .help("Evaluator for player 2: rollout, heuristic, or nn"),
        );

    #[cfg(feature = "nn")]
    {
        cmd = cmd
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
            );
    }

    cmd = cmd.arg(
        Arg::new("balanced")
            .long("balanced")
            .action(clap::ArgAction::SetTrue)
            .help("Use balanced dice instead of random"),
    );

    for arg in cli::config_args() {
        cmd = cmd.arg(arg);
    }
    cmd
}

#[cfg(feature = "nn")]
fn train_command() -> Command {
    cli::train_command()
        .arg(
            Arg::new("model")
                .long("model")
                .default_value("gnn")
                .help("Model architecture: simple, resnet, gnn, or gnn2"),
        )
        .arg(
            Arg::new("encoder")
                .long("encoder")
                .default_value("gnn")
                .help("Encoder: basic, rich, gnn, or gnn2"),
        )
        .arg(
            Arg::new("balanced")
                .long("balanced")
                .action(clap::ArgAction::SetTrue)
                .help("Use balanced dice instead of random"),
        )
}

fn main() {
    let matches = app().get_matches();

    // Visualize subcommand
    if let Some(sub) = matches.subcommand_matches("visualize") {
        let log_path = std::path::PathBuf::from(sub.get_one::<String>("log-file").unwrap());
        let game_log = GameLog::read(&log_path);
        let html_path = log_path.with_extension("html");
        visualize::render(&game_log, &html_path);
        println!("Wrote {}", html_path.display());
        return;
    }

    // Train subcommand
    #[cfg(feature = "nn")]
    if let Some(sub) = matches.subcommand_matches("train") {
        run_train(sub);
        return;
    }

    // Tournament mode
    run_tournament(&matches);
}

#[cfg(feature = "nn")]
fn train_config(model: &str) -> canopy2::train::TrainConfig {
    use canopy2::train::TrainConfig;

    match model {
        "simple" => TrainConfig {
            iterations: 1000,
            epochs: 2,
            batch_size: 128,
            replay_window: 10,
            games_per_iter: 100,
            bench_interval: 30,
            bench_baseline_sims: 50,
            ..TrainConfig::default()
        },
        "resnet" => TrainConfig {
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
        "gnn" | "gnn2" => TrainConfig {
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
        _ => TrainConfig::default(),
    }
}

#[cfg(feature = "nn")]
fn run_train(matches: &clap::ArgMatches) {
    use canopy2::game::Game;
    use canopy2::train::{BurnTrainableModel, default_device};

    let model_type = matches.get_one::<String>("model").unwrap().as_str();
    let encoder_type = matches.get_one::<String>("encoder").unwrap().as_str();
    let config = cli::parse_train_config(matches, train_config(model_type));
    let device = default_device();

    let dice = if matches.get_flag("balanced") {
        Dice::Balanced(game::dice::BalancedDice::new())
    } else {
        Dice::Random
    };

    let new_state = move |rng: &mut fastrand::Rng| game::new_game(rng.u64(..), dice);

    macro_rules! train {
        ($encoder:ty, $config_expr:expr) => {{
            let mc = $config_expr;
            let mut trainable =
                BurnTrainableModel::<GameState, $encoder, _>::new(move |dev| mc.init(dev), &device);
            canopy2::train::run_training::<GameState, _>(config, &mut trainable, new_state);
        }};
    }

    macro_rules! simple_cfg {
        ($enc:ty) => {
            model::CatanModelConfig::new(
                GameState::NUM_ACTIONS,
                <$enc>::NODES_F,
                <$enc>::EDGES_F,
                <$enc>::TILES_F,
                <$enc>::PORTS_F,
            )
        };
    }

    macro_rules! resnet_cfg {
        ($enc:ty) => {
            model::CatanResModelConfig::new(
                GameState::NUM_ACTIONS,
                <$enc>::NODES_F,
                <$enc>::EDGES_F,
                <$enc>::TILES_F,
                <$enc>::PORTS_F,
            )
        };
    }

    match (model_type, encoder_type) {
        ("simple", "basic") => {
            train!(encoder::BasicEncoder, simple_cfg!(encoder::BasicEncoder))
        }
        ("simple", "rich") => {
            train!(
                encoder::RichNodeEncoder,
                simple_cfg!(encoder::RichNodeEncoder)
            )
        }
        ("resnet", "basic") => {
            train!(encoder::BasicEncoder, resnet_cfg!(encoder::BasicEncoder))
        }
        ("resnet", "rich") => {
            train!(
                encoder::RichNodeEncoder,
                resnet_cfg!(encoder::RichNodeEncoder)
            )
        }
        ("gnn", "gnn") => {
            train!(
                encoder::GnnEncoder,
                model::CatanGnnModelConfig::new(GameState::NUM_ACTIONS)
            )
        }
        ("gnn2", "gnn2") => {
            let mc = model::CatanGnnModelConfig::new(GameState::NUM_ACTIONS);
            let mut trainable = BurnTrainableModel::<GameState, encoder::Gnn2Encoder, _>::new(
                move |dev| mc.init_with::<_, 101, 34>(dev),
                &device,
            );
            canopy2::train::run_training::<GameState, _>(config, &mut trainable, new_state);
        }
        (m, e) => panic!("unknown model '{m}' or encoder '{e}'"),
    }
}

fn run_tournament(matches: &clap::ArgMatches) {
    let num_games: u32 = matches
        .get_one::<String>("num-games")
        .unwrap()
        .parse()
        .unwrap();
    let log_dir = matches
        .get_one::<String>("log-dir")
        .map(std::path::PathBuf::from);

    let configs = cli::parse_configs(matches);

    let p1_eval_name = matches.get_one::<String>("p1-eval").unwrap().as_str();
    let p2_eval_name = matches.get_one::<String>("p2-eval").unwrap().as_str();

    let rollout = RolloutEvaluator { num_rollouts: 1 };
    let heuristic_eval = heuristic::HeuristicEvaluator {
        rollout: RolloutEvaluator { num_rollouts: 1 },
    };

    #[cfg(feature = "nn")]
    let nn_eval: Option<Box<dyn Evaluator<GameState>>> = {
        let nn_model_path = matches.get_one::<String>("nn-model");
        if p1_eval_name == "nn" || p2_eval_name == "nn" {
            let encoder_type = matches.get_one::<String>("encoder").unwrap().as_str();
            Some(load_nn_eval(
                nn_model_path.expect("--nn-model required when using nn evaluator"),
                encoder_type,
            ))
        } else {
            None
        }
    };

    let eval_ref = |name: &str| -> &dyn Evaluator<GameState> {
        match name {
            "rollout" => &rollout,
            "heuristic" => &heuristic_eval,
            #[cfg(feature = "nn")]
            "nn" => nn_eval.as_deref().unwrap(),
            other => {
                panic!("unknown evaluator '{other}', expected 'rollout', 'heuristic', or 'nn'")
            }
        }
    };

    let evaluators: PerPlayer<&dyn Evaluator<GameState>> =
        PerPlayer([eval_ref(p1_eval_name), eval_ref(p2_eval_name)]);

    let mut rng = fastrand::Rng::new();

    let dice = if matches.get_flag("balanced") {
        Dice::Balanced(game::dice::BalancedDice::new())
    } else {
        Dice::Random
    };

    println!(
        "=== Catan Tournament: {} vs {} simulations, P1 ({}) vs P2 ({}), {} games, dice: {} ===\n",
        configs.0[0].num_simulations,
        configs.0[1].num_simulations,
        p1_eval_name,
        p2_eval_name,
        num_games,
        if matches!(dice, Dice::Balanced(_)) {
            "balanced"
        } else {
            "random"
        },
    );

    let new_game = move |seed: u64| game::new_game(seed, dice);

    let game_logs = tournament::tournament(new_game, &evaluators, &configs, num_games, &mut rng);

    if let Some(log_dir) = log_dir {
        std::fs::create_dir_all(&log_dir).expect("failed to create log directory");
        for (i, log) in game_logs.iter().enumerate() {
            let log_path = log_dir.join(format!("game_{i}.log"));
            log.write(&log_path);
            println!("Wrote {}", log_path.display());
        }
    }
}

#[cfg(feature = "nn")]
fn load_nn_eval(checkpoint_path: &str, encoder_type: &str) -> Box<dyn Evaluator<GameState>> {
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
    use canopy2::game::Game;
    use canopy2::nn::NeuralEvaluator;
    use canopy2::train::{InferBackend, default_device};

    let device = default_device();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    macro_rules! load {
        ($encoder:ty) => {{
            let mc = model::CatanModelConfig::new(
                GameState::NUM_ACTIONS,
                <$encoder>::NODES_F,
                <$encoder>::EDGES_F,
                <$encoder>::TILES_F,
                <$encoder>::PORTS_F,
            );
            let model: model::CatanModel<InferBackend> = mc.init(&device);
            let model = model
                .load_file(checkpoint_path, &recorder, &device)
                .expect("failed to load nn checkpoint");
            Box::new(NeuralEvaluator::<InferBackend, $encoder, _>::new(
                model, device,
            ))
        }};
    }

    match encoder_type {
        "basic" => load!(encoder::BasicEncoder),
        "rich" => load!(encoder::RichNodeEncoder),
        "gnn" => {
            let mc = model::CatanGnnModelConfig::new(GameState::NUM_ACTIONS);
            let model: model::CatanGnnModel<InferBackend> = mc.init(&device);
            let model = model
                .load_file(checkpoint_path, &recorder, &device)
                .expect("failed to load nn checkpoint");
            Box::new(NeuralEvaluator::<InferBackend, encoder::GnnEncoder, _>::new(model, device))
        }
        "gnn2" => {
            let mc = model::CatanGnnModelConfig::new(GameState::NUM_ACTIONS);
            let model: model::CatanGnnModel<InferBackend, 101, 34> = mc.init_with(&device);
            let model = model
                .load_file(checkpoint_path, &recorder, &device)
                .expect("failed to load nn checkpoint");
            Box::new(NeuralEvaluator::<InferBackend, encoder::Gnn2Encoder, _>::new(model, device))
        }
        other => panic!("unknown encoder '{other}', expected 'basic', 'rich', 'gnn', or 'gnn2'"),
    }
}
