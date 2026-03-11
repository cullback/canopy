//! # Pig dice game — tournament & training
//!
//! Runs a head-to-head tournament between two MCTS bots with different
//! configurations playing the Pig dice game.
//!
//! ```text
//! cargo run --example pig -- --p1-simulations 1000 --p2-simulations 5000
//! cargo run --example pig --features nn -- train --iterations 5 --games 20
//! ```

use canopy2::cli;
use canopy2::eval::{Evaluators, RolloutEvaluator};
use canopy2::game::{Game, Status};

mod game;
mod strategy;

#[cfg(feature = "nn")]
mod encoder;
#[cfg(feature = "nn")]
mod model;

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

fn app() -> clap::Command {
    let mut cmd = cli::tournament_command("pig", "Pig dice game tournament between two MCTS bots");

    #[cfg(feature = "nn")]
    {
        cmd = cmd.subcommand(cli::train_command());
    }

    cmd
}

#[cfg(feature = "nn")]
fn train_config() -> canopy2::train::TrainConfig {
    use canopy2::train::TrainConfig;

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
    }
}

#[cfg(feature = "nn")]
fn run_train(matches: &clap::ArgMatches) {
    use canopy2::game::Game;
    use canopy2::nn::StateEncoder;
    use canopy2::train::{BurnTrainableModel, default_device};

    let config = cli::parse_train_config(matches, train_config());
    let device = default_device();

    let new_state = |_rng: &mut fastrand::Rng| PigGame::new(100);

    let mc = model::PigModelConfig::new(PigGame::NUM_ACTIONS, encoder::PigEncoder::FEATURE_SIZE);
    let mut trainable = BurnTrainableModel::<PigGame, encoder::PigEncoder, _>::new(
        move |dev| mc.init(dev),
        &device,
    );
    canopy2::train::run_training::<PigGame, _>(config, &mut trainable, new_state);
}

fn main() {
    let matches = app().get_matches();

    #[cfg(feature = "nn")]
    if let Some(sub) = matches.subcommand_matches("train") {
        run_train(sub);
        return;
    }

    let opts = cli::parse_tournament(&matches);
    let mut evals = Evaluators::new();
    evals.add("rollout", RolloutEvaluator { num_rollouts: 1 });
    evals.add("hold-at-20", strategy::HoldAt(20));
    opts.run(|_| PigGame::new(100), &evals);
}
