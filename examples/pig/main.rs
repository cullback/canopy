//! # Pig dice game — tournament & training
//!
//! Runs a head-to-head tournament between two MCTS bots with different
//! configurations playing the Pig dice game.
//!
//! ```text
//! cargo run --example pig -- --p1-sims 1000 --p2-sims 5000
//! cargo run --example pig -- train --iterations 5
//! ```

use canopy::cli::GameCli;
use canopy::game::{Game, Status};

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
    use canopy::train::TrainConfig;
    use std::sync::Arc;

    let mut setup = GameCli::new("pig", "Pig dice game tournament between two MCTS bots");
    setup.add_evaluator("hold-at-20", strategy::HoldAt(20));
    setup.add_evaluator("erkp", strategy::EndRaceKeepPace);

    setup.add_encoder("default", Arc::new(encoder::PigEncoder));
    setup.add_model("default", |device, _cfg| model::init_pig(device));
    setup.add_config(
        "default",
        TrainConfig {
            iterations: 100,
            train_samples_per_iter: 3_000,
            replay_buffer_samples: 30_000,
            // max_moves: 500,
            mcts_sims: 200,
            mcts_sims_start: 50,
            epochs: 3,
            train_batch_size: 128,
            warmup_iters: 20,
            concurrent_games: 100,
            // leaf_batch_size: 1,
            explore_moves: 10,
            ..TrainConfig::default()
        },
    );

    let matches = setup.command().get_matches();

    setup.run(&matches, |_seed| PigGame::new(100));
}
