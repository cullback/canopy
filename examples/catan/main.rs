//! # Catan — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots playing Catan.
//!
//! ```text
//! cargo run --example catan -- --p1-simulations 200 --p2-simulations 1000
//! cargo run --example catan -- visualize logs/game_0.log
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

use game::dice::Dice;
use game::state::GameState;

fn app() -> Command {
    let mut cmd = Command::new("catan")
        .about("Catan tournament between two MCTS bots")
        .subcommand(
            Command::new("visualize")
                .about("Convert a game log to an HTML visualization")
                .arg(Arg::new("log-file").required(true)),
        )
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
                .help("Evaluator for player 1: rollout, heuristic, or policy"),
        )
        .arg(
            Arg::new("p2-eval")
                .long("p2-eval")
                .default_value("rollout")
                .help("Evaluator for player 2: rollout, heuristic, or policy"),
        );
    for arg in cli::config_args() {
        cmd = cmd.arg(arg);
    }
    cmd
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

    // Tournament mode
    let num_games: u32 = matches
        .get_one::<String>("num-games")
        .unwrap()
        .parse()
        .unwrap();
    let log_dir = matches
        .get_one::<String>("log-dir")
        .map(std::path::PathBuf::from);

    let configs = cli::parse_configs(&matches);

    let p1_eval_name = matches.get_one::<String>("p1-eval").unwrap().as_str();
    let p2_eval_name = matches.get_one::<String>("p2-eval").unwrap().as_str();

    let rollout = RolloutEvaluator { num_rollouts: 1 };
    let heuristic_eval = heuristic::HeuristicEvaluator;
    let policy_eval = heuristic::PolicyEvaluator {
        rollout: RolloutEvaluator { num_rollouts: 1 },
    };

    let eval_ref = |name: &str| -> &dyn Evaluator<GameState> {
        match name {
            "rollout" => &rollout,
            "heuristic" => &heuristic_eval,
            "policy" => &policy_eval,
            other => {
                panic!("unknown evaluator '{other}', expected 'rollout', 'heuristic', or 'policy'")
            }
        }
    };

    let evaluators: PerPlayer<&dyn Evaluator<GameState>> =
        PerPlayer([eval_ref(p1_eval_name), eval_ref(p2_eval_name)]);

    let mut rng = fastrand::Rng::new();

    println!(
        "=== Catan Tournament: {} vs {} simulations, {} ({}) vs {} ({}), {} games ===\n",
        configs.0[0].num_simulations,
        configs.0[1].num_simulations,
        "P1",
        p1_eval_name,
        "P2",
        p2_eval_name,
        num_games,
    );

    let new_game = |seed: u64| game::new_game(seed, Dice::default());

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
