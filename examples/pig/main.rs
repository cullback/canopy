//! # Pig dice game — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots with different
//! configurations playing the Pig dice game.
//!
//! ```text
//! cargo run --example pig -- --p1-simulations 1000 --p2-simulations 5000
//! ```

use clap::Command;

use canopy2::cli;
use canopy2::eval::{Evaluator, RolloutEvaluator};
use canopy2::game::{Game, Status};
use canopy2::tournament;

mod game;

use game::{PigGame, Player};

impl Game for PigGame {
    const NUM_ACTIONS: usize = game::NUM_ACTIONS;

    fn status(&self) -> Status {
        match self.winner() {
            Some(Player::One) => Status::Terminal(1.0),
            Some(Player::Two) => Status::Terminal(-1.0),
            None => Status::Ongoing,
        }
    }

    fn current_sign(&self) -> f32 {
        match self.current_player() {
            Player::One => 1.0,
            Player::Two => -1.0,
        }
    }

    fn legal_actions(&self, buf: &mut Vec<usize>) {
        buf.push(game::ROLL);
        buf.push(game::HOLD);
    }

    fn apply_action(&mut self, action: usize) {
        if self.is_rolling() {
            self.apply_roll(action);
        } else {
            self.apply_decision(action);
        }
    }

    fn chance_outcomes(&self, buf: &mut Vec<(usize, u32)>) {
        if self.is_rolling() {
            for i in 0..game::NUM_DIE_FACES {
                buf.push((i, 1));
            }
        }
    }

    fn sample_chance(&self, rng: &mut fastrand::Rng) -> Option<usize> {
        if self.is_rolling() {
            Some(rng.usize(..game::NUM_DIE_FACES))
        } else {
            None
        }
    }
}

fn app() -> Command {
    let mut cmd = Command::new("pig").about("Pig dice game tournament between two MCTS bots");
    for arg in cli::tournament_args() {
        cmd = cmd.arg(arg);
    }
    cmd
}

fn main() {
    let matches = app().get_matches();
    let opts = cli::parse_tournament(&matches);

    let mut rng = fastrand::Rng::new();
    let eval = RolloutEvaluator { num_rollouts: 1 };
    let evaluators: [&dyn Evaluator<PigGame>; 2] = [&eval, &eval];

    println!(
        "=== Pig Tournament: {} vs {} simulations, {} games ===\n",
        opts.configs[0].num_simulations, opts.configs[1].num_simulations, opts.num_games,
    );

    let game_logs = tournament::tournament(
        |_seed| PigGame::new(100),
        &evaluators,
        &opts.configs,
        opts.num_games,
        &mut rng,
    );

    if let Some(dir) = &opts.log_dir {
        tournament::save_game_logs(&game_logs, dir);
    }
}
