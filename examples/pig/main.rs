//! # Pig dice game — tournament & training
//!
//! Runs a head-to-head tournament between two MCTS bots with different
//! configurations playing the Pig dice game.
//!
//! ```text
//! cargo run --example pig -- --p1-simulations 1000 --p2-simulations 5000
//! cargo run --example pig -- train --iterations 5 --games 20
//! ```

use canopy2::cli::GameSetup;
use canopy2::eval::RolloutEvaluator;
use canopy2::game::{Game, Status};

mod encoder;
mod game;
mod model;
mod strategy;

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

fn main() {
    use canopy2::train::TrainConfig;
    use std::sync::Arc;

    let mut setup = GameSetup::new("pig", "Pig dice game tournament between two MCTS bots");
    setup.add_evaluator("rollout", RolloutEvaluator::default());
    setup.add_evaluator("hold-at-20", strategy::HoldAt(20));

    setup.add_encoder("default", Arc::new(encoder::PigEncoder));
    setup.add_model("default", model::init_pig);
    setup.add_config(
        "default",
        TrainConfig {
            iterations: 200,
            games_per_iter: 300,
            mcts_sims: 200,
            mcts_sims_start: 50,
            epochs: 3,
            batch_size: 128,
            replay_window: 10,
            warmup_iters: 20,
            bench_games: 100,
            bench_interval: 5,
            bench_baseline_sims: 200,
            concurrent_games: 10,
            leaf_batch_size: 1,
            explore_moves: 10,
            ..TrainConfig::default()
        },
    );

    let matches = setup.command().get_matches();

    if let Some(sub) = matches.subcommand_matches("train") {
        setup.run_train(sub, |_rng| PigGame::new(100));
    } else {
        setup.run_tournament(&matches, |_| PigGame::new(100));
    }
}
