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
        cmd = cmd.arg(
            Arg::new("nn-model")
                .long("nn-model")
                .help("Path to neural network checkpoint (for nn evaluator)"),
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
    Command::new("train")
        .about("Run AlphaZero-style self-play training")
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .default_value("1000"),
        )
        .arg(
            Arg::new("games")
                .long("games")
                .default_value("500")
                .help("Self-play games per iteration"),
        )
        .arg(
            Arg::new("train-mcts")
                .long("train-mcts")
                .default_value("800")
                .help("MCTS simulations per move during self-play"),
        )
        .arg(
            Arg::new("epochs")
                .long("epochs")
                .default_value("3")
                .help("Training epochs per iteration"),
        )
        .arg(
            Arg::new("batch-size")
                .long("batch-size")
                .default_value("256"),
        )
        .arg(Arg::new("lr").long("lr").default_value("0.001"))
        .arg(
            Arg::new("lr-final")
                .long("lr-final")
                .default_value("0.0001"),
        )
        .arg(
            Arg::new("lr-decay-after")
                .long("lr-decay-after")
                .default_value("300"),
        )
        .arg(
            Arg::new("replay-window")
                .long("replay-window")
                .default_value("40"),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .default_value("checkpoints"),
        )
        .arg(
            Arg::new("resume")
                .long("resume")
                .help("Resume training from checkpoint path (e.g. checkpoints/run/model_iter_10)"),
        )
        .arg(
            Arg::new("q-blend-gen")
                .long("q-blend-gen")
                .default_value("100"),
        )
        .arg(
            Arg::new("bench-games")
                .long("bench-games")
                .default_value("0")
                .help("Benchmark games vs rollout bot per iteration (0 to skip)"),
        )
        .arg(
            Arg::new("bench-mcts")
                .long("bench-mcts")
                .default_value("400"),
        )
        .arg(
            Arg::new("gumbel-m")
                .long("gumbel-m")
                .default_value("16")
                .help("Gumbel-Top-k sampled actions at root"),
        )
        .arg(Arg::new("c-visit").long("c-visit").default_value("50.0"))
        .arg(Arg::new("c-scale").long("c-scale").default_value("1.0"))
        .arg(
            Arg::new("explore-moves")
                .long("explore-moves")
                .default_value("30")
                .help("Early-game turns where action is sampled from improved policy"),
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
fn run_train(matches: &clap::ArgMatches) {
    use burn::backend::ndarray::NdArrayDevice;
    use canopy2::game::Game;
    use canopy2::nn::StateEncoder;
    use canopy2::train::{BurnTrainableModel, TrainConfig};

    let parse = |name: &str| -> String { matches.get_one::<String>(name).unwrap().clone() };

    let config = TrainConfig {
        iterations: parse("iterations").parse().unwrap(),
        games_per_iter: parse("games").parse().unwrap(),
        mcts_sims: parse("train-mcts").parse().unwrap(),
        epochs: parse("epochs").parse().unwrap(),
        batch_size: parse("batch-size").parse().unwrap(),
        lr: parse("lr").parse().unwrap(),
        lr_final: parse("lr-final").parse().unwrap(),
        lr_decay_after: parse("lr-decay-after").parse().unwrap(),
        replay_window: parse("replay-window").parse().unwrap(),
        output_dir: parse("output"),
        resume: matches.get_one::<String>("resume").cloned(),
        q_blend_generations: parse("q-blend-gen").parse().unwrap(),
        bench_games: parse("bench-games").parse().unwrap(),
        bench_mcts: parse("bench-mcts").parse().unwrap(),
        gumbel_m: parse("gumbel-m").parse().unwrap(),
        c_visit: parse("c-visit").parse().unwrap(),
        c_scale: parse("c-scale").parse().unwrap(),
        explore_moves: parse("explore-moves").parse().unwrap(),
    };

    let device = NdArrayDevice::Cpu;

    let model_config =
        model::CatanModelConfig::new(encoder::CatanEncoder::FEATURE_SIZE, GameState::NUM_ACTIONS);

    let mut trainable = BurnTrainableModel::<GameState, encoder::CatanEncoder, _>::new(
        move |dev| model_config.init(dev),
        &device,
    );

    let dice = if matches.get_flag("balanced") {
        Dice::Balanced(game::dice::BalancedDice::new())
    } else {
        Dice::Random
    };

    let new_state = move |rng: &mut fastrand::Rng| game::new_game(rng.u64(..), dice);

    canopy2::train::run_training::<GameState, encoder::CatanEncoder, _>(
        config,
        &mut trainable,
        new_state,
    );
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
    let nn_eval = {
        let nn_model_path = matches.get_one::<String>("nn-model");
        if p1_eval_name == "nn" || p2_eval_name == "nn" {
            Some(load_nn_eval(
                nn_model_path.expect("--nn-model required when using nn evaluator"),
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
            "nn" => nn_eval.as_ref().unwrap(),
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
fn load_nn_eval(
    checkpoint_path: &str,
) -> canopy2::nn::NeuralEvaluator<
    burn::backend::NdArray,
    encoder::CatanEncoder,
    model::CatanModel<burn::backend::NdArray>,
> {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
    use canopy2::game::Game;
    use canopy2::nn::{NeuralEvaluator, StateEncoder};

    let device = NdArrayDevice::Cpu;
    let model_config =
        model::CatanModelConfig::new(encoder::CatanEncoder::FEATURE_SIZE, GameState::NUM_ACTIONS);
    let model: model::CatanModel<NdArray> = model_config.init(&device);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(checkpoint_path, &recorder, &device)
        .expect("failed to load nn checkpoint");

    NeuralEvaluator::new(model, device)
}
