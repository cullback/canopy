//! # Catan — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots playing Catan.
//!
//! ```text
//! cargo run --example catan -- --p1-simulations 200 --p2-simulations 1000
//! ```

use clap::{Arg, Command};

use canopy2::cli;
use canopy2::eval::{Evaluator, RolloutEvaluator};
use canopy2::player::PerPlayer;
use canopy2::tournament;

mod game;

use game::dice::Dice;
use game::state::GameState;

fn app() -> Command {
    let mut cmd = Command::new("catan")
        .about("Catan tournament between two MCTS bots")
        .arg(
            Arg::new("num-games")
                .short('n')
                .long("num-games")
                .default_value("20"),
        );
    for arg in cli::config_args() {
        cmd = cmd.arg(arg);
    }
    cmd
}

fn main() {
    let matches = app().get_matches();
    let num_games: u32 = matches
        .get_one::<String>("num-games")
        .unwrap()
        .parse()
        .unwrap();

    let configs = cli::parse_configs(&matches);

    let mut rng = fastrand::Rng::new();
    let game = game::new_game(&mut rng, Dice::default());
    let eval = RolloutEvaluator { num_rollouts: 1 };
    let evaluators: PerPlayer<&dyn Evaluator<GameState>> = PerPlayer([&eval, &eval]);

    println!(
        "=== Catan Tournament: {} vs {} simulations, {} games ===\n",
        configs.0[0].num_simulations, configs.0[1].num_simulations, num_games,
    );

    tournament::tournament(&game, &evaluators, &configs, num_games, &mut rng);
}
