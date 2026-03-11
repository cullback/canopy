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
use canopy2::eval::{Evaluator, Evaluators, RolloutEvaluator};
use canopy2::game_log::GameLog;

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
    let mut viz = Command::new("visualize")
        .about("Convert a game log to an HTML visualization")
        .arg(Arg::new("log-file").required(true));

    #[cfg(feature = "nn")]
    {
        viz = viz.arg(
            Arg::new("encoder")
                .long("encoder")
                .help("Embed encoder features in the visualization (gnn or gnn2)"),
        );
    }

    let mut cmd =
        cli::tournament_command("catan", "Catan tournament between two MCTS bots").subcommand(viz);

    // Train subcommand (always visible, but requires nn feature to run)
    #[cfg(feature = "nn")]
    {
        cmd = cmd.subcommand(train_command());
    }

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

        #[cfg(feature = "nn")]
        if let Some(enc) = sub.get_one::<String>("encoder") {
            render_with_encoder_dispatch(enc, &game_log, &html_path);
            println!("Wrote {} (with {} features)", html_path.display(), enc);
            return;
        }

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
fn render_with_encoder_dispatch(encoder_type: &str, log: &GameLog, output: &std::path::Path) {
    let res = ["lumber", "brick", "wool", "grain", "ore"];

    // Phase labels (7)
    let phase_labels: Vec<String> = [
        "phase_settlement",
        "phase_road",
        "phase_pre_roll",
        "phase_discard",
        "phase_robber",
        "phase_main",
        "phase_road_building",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // Per-player standard labels (21)
    let player_std = |prefix: &str| -> Vec<String> {
        let mut v = Vec::new();
        for r in &res {
            v.push(format!("{prefix}_{r}"));
        }
        for d in ["dev_knight", "dev_vp", "dev_rb", "dev_yop", "dev_monopoly"] {
            v.push(format!("{prefix}_{d}"));
        }
        for d in [
            "played_knight",
            "played_vp",
            "played_rb",
            "played_yop",
            "played_monopoly",
        ] {
            v.push(format!("{prefix}_{d}"));
        }
        v.push(format!("{prefix}_settlements_left"));
        v.push(format!("{prefix}_cities_left"));
        v.push(format!("{prefix}_roads_left"));
        v.push(format!("{prefix}_longest_road"));
        v.push(format!("{prefix}_largest_army"));
        v.push(format!("{prefix}_road_length"));
        v
    };

    match encoder_type {
        "gnn" => {
            // Global: phase(7) + player_std(21×2) = 49
            let mut global_labels = phase_labels.clone();
            global_labels.extend(player_std("cur"));
            global_labels.extend(player_std("opp"));

            // Per-node (24): building(2) + prod(5) + robbed(5) + port(6) + road(6)
            let mut node_labels = vec!["building_cur".into(), "building_opp".into()];
            for r in &res {
                node_labels.push(format!("prod_{r}"));
            }
            for r in &res {
                node_labels.push(format!("robbed_{r}"));
            }
            for r in &res {
                node_labels.push(format!("port_{r}"));
            }
            node_labels.push("port_generic".into());
            for i in 0..3 {
                node_labels.push(format!("road{i}_cur"));
                node_labels.push(format!("road{i}_opp"));
            }

            visualize::render_with_encoder::<encoder::GnnEncoder>(
                log,
                output,
                global_labels,
                node_labels,
            );
        }
        "gnn2" => {
            // Global: phase(7) + player_std(21×2) + player_ext(26×2) = 101
            let mut global_labels = phase_labels;
            global_labels.extend(player_std("cur"));
            global_labels.extend(player_std("opp"));

            // Extended per-player labels (26)
            let player_ext = |prefix: &str| -> Vec<String> {
                let mut v = Vec::new();
                for r in &res {
                    v.push(format!("{prefix}_trade_{r}"));
                }
                for r in &res {
                    v.push(format!("{prefix}_prod_{r}"));
                }
                for r in &res {
                    v.push(format!("{prefix}_robbed_{r}"));
                }
                for n in 2..=12u8 {
                    v.push(format!("{prefix}_num_{n}"));
                }
                v
            };
            global_labels.extend(player_ext("cur"));
            global_labels.extend(player_ext("opp"));

            // Per-node (34): building(2) + tile_slot×3(21) + port(5) + road(6)
            let mut node_labels: Vec<String> = vec!["building_cur".into(), "building_opp".into()];
            for i in 0..3 {
                for r in &res {
                    node_labels.push(format!("t{i}_{r}"));
                }
                node_labels.push(format!("t{i}_number"));
                node_labels.push(format!("t{i}_robber"));
            }
            for r in &res {
                node_labels.push(format!("port_{r}"));
            }
            for i in 0..3 {
                node_labels.push(format!("road{i}_cur"));
                node_labels.push(format!("road{i}_opp"));
            }

            visualize::render_with_encoder::<encoder::Gnn2Encoder>(
                log,
                output,
                global_labels,
                node_labels,
            );
        }
        other => panic!("unknown encoder '{other}', expected 'gnn' or 'gnn2'"),
    }
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
    let opts = cli::parse_tournament(matches);

    let mut evals = Evaluators::new();
    evals.add("rollout", RolloutEvaluator { num_rollouts: 1 });
    evals.add(
        "heuristic",
        heuristic::HeuristicEvaluator {
            rollout: RolloutEvaluator { num_rollouts: 1 },
        },
    );

    #[cfg(feature = "nn")]
    if let Some(path) = matches.get_one::<String>("nn-model") {
        let encoder_type = matches.get_one::<String>("encoder").unwrap().as_str();
        evals.add_boxed("nn", load_nn_eval(path, encoder_type));
    }

    let dice = if matches.get_flag("balanced") {
        Dice::Balanced(game::dice::BalancedDice::new())
    } else {
        Dice::Random
    };

    opts.run(move |seed| game::new_game(seed, dice), &evals);
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
