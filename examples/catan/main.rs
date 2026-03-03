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

    let mut rng = fastrand::Rng::new();
    let eval = RolloutEvaluator { num_rollouts: 1 };
    let evaluators: PerPlayer<&dyn Evaluator<GameState>> = PerPlayer([&eval, &eval]);

    println!(
        "=== Catan Tournament: {} vs {} simulations, {} games ===\n",
        configs.0[0].num_simulations, configs.0[1].num_simulations, num_games,
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
