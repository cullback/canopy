//! # Tenure game — tournament & training
//!
//! Spencer's Attacker-Defender (Tenure) Game. An attacker partitions pieces
//! and a defender destroys one partition. Uses micro-actions for linear action
//! space.
//!
//! ```text
//! cargo run --example tenure -- -n 20
//! cargo run --example tenure -- --p1-eval balanced-attacker --p2-eval optimal-defender -n 100
//! cargo run --example tenure -- train --iterations 5 --games 20
//! ```

use clap::{Arg, Command};

use canopy2::cli::GameCli;
use canopy2::eval::RolloutEvaluator;
use canopy2::game::{Game, Status};
use canopy2::game_log::GameLog;

mod encoder;
mod game;
mod model;
mod round_robin;
mod strategy;
mod visualize;

use game::{K, Phase, TenureGame};

impl Game for TenureGame {
    const NUM_ACTIONS: usize = game::NUM_ACTIONS;

    fn status(&self) -> Status {
        if self.is_terminal() {
            Status::Terminal(self.terminal_reward())
        } else {
            Status::Ongoing
        }
    }

    fn current_sign(&self) -> f32 {
        match self.phase {
            Phase::Attacker => 1.0,
            Phase::Defender => -1.0,
        }
    }

    fn legal_actions(&self, buf: &mut Vec<usize>) {
        match self.phase {
            Phase::Attacker => {
                for l in 0..K {
                    if self.board[l] > 0 {
                        buf.push(l);
                    }
                }
                buf.push(game::DONE);
            }
            Phase::Defender => {
                buf.push(game::DESTROY_A);
                buf.push(game::DESTROY_B);
            }
        }
    }

    fn apply_action(&mut self, action: usize) {
        match self.phase {
            Phase::Attacker => {
                if action == game::DONE {
                    self.attacker_done();
                } else {
                    self.attacker_move(action);
                }
            }
            Phase::Defender => {
                self.defender_choose(action == game::DESTROY_B);
            }
        }
    }
}

fn main() {
    use canopy2::train::TrainConfig;
    use std::sync::Arc;

    let mut setup = GameCli::new("tenure", "Spencer's Attacker-Defender (Tenure) Game");
    setup.add_evaluator("rollout", RolloutEvaluator::default());
    setup.add_evaluator("balanced-attacker", strategy::BalancedAttacker);
    setup.add_evaluator("optimal-defender", strategy::OptimalDefender);

    setup.add_encoder("default", Arc::new(encoder::TenureEncoder));
    setup.add_model("default", model::init_tenure);
    setup.add_config(
        "default",
        TrainConfig {
            iterations: 1000,
            games_per_iter: 400,
            mcts_sims: 200,
            mcts_sims_start: 50,
            epochs: 3,
            batch_size: 256,
            replay_window: 8,
            warmup_iters: 15,
            bench_games: 200,
            bench_interval: 10,
            bench_sims: 500,
            concurrent_games: 20,
            leaf_batch_size: 8,
            explore_moves: 10,
            ..TrainConfig::default()
        },
    );

    let viz = Command::new("visualize")
        .about("Convert a game log to an HTML visualization")
        .arg(Arg::new("log-file").required(true));

    let rr = Command::new("round-robin")
        .about("Round-robin tournament between all model checkpoints in a directory")
        .arg(Arg::new("checkpoint-dir").required(true))
        .arg(
            Arg::new("num-games")
                .short('n')
                .long("num-games")
                .default_value("1000"),
        )
        .arg(
            Arg::new("simulations")
                .long("simulations")
                .default_value("200"),
        );

    let matches = setup.command().subcommand(viz).subcommand(rr).get_matches();

    if let Some(sub) = matches.subcommand_matches("round-robin") {
        let dir = std::path::PathBuf::from(sub.get_one::<String>("checkpoint-dir").unwrap());
        let num_games: u32 = sub.get_one::<String>("num-games").unwrap().parse().unwrap();
        let sims: u32 = sub
            .get_one::<String>("simulations")
            .unwrap()
            .parse()
            .unwrap();
        round_robin::run(&dir, num_games, sims);
        return;
    }

    if let Some(sub) = matches.subcommand_matches("visualize") {
        let log_path = std::path::PathBuf::from(sub.get_one::<String>("log-file").unwrap());
        let game_log = GameLog::read(&log_path);
        let html_path = log_path.with_extension("html");
        visualize::render(&game_log, &html_path);
        println!("Wrote {}", html_path.display());
        return;
    }

    setup.run(&matches, |rng| TenureGame::random(rng));
}
