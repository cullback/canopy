//! # Pig dice game — tournament mode
//!
//! Runs a head-to-head tournament between two MCTS bots with different
//! configurations playing the Pig dice game.
//!
//! ```text
//! cargo run --example pig -- --p1-simulations 1000 --p2-simulations 5000
//! ```

use clap::{Arg, Command};

use canopy2::cli;
use canopy2::eval::{Evaluator, RolloutEvaluator};
use canopy2::game::{Game, Status};
use canopy2::player::{PerPlayer, Player};
use canopy2::tournament;

const ROLL: usize = 0;
const HOLD: usize = 1;

fn app() -> Command {
    let mut cmd = Command::new("pig")
        .about("Pig dice game tournament between two MCTS bots")
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

/// Phase within a turn.
#[derive(Clone, Debug)]
enum Phase {
    Decision,
    Rolling,
}

#[derive(Clone, Debug)]
struct PigGame {
    scores: PerPlayer<u32>,
    current: Player,
    turn_total: u32,
    phase: Phase,
    target: u32,
}

impl PigGame {
    fn new(target: u32) -> Self {
        Self {
            scores: PerPlayer::default(),
            current: Player::One,
            turn_total: 0,
            phase: Phase::Decision,
            target,
        }
    }

    fn pass_turn(&mut self) {
        self.current = self.current.opponent();
        self.turn_total = 0;
        self.phase = Phase::Decision;
    }

    /// Map chance outcome index to die face (1..=6).
    fn die_face(outcome: usize) -> u32 {
        outcome as u32 + 1
    }
}

impl Game for PigGame {
    const NUM_ACTIONS: usize = 2;

    fn status(&self) -> Status {
        if self.scores[Player::One] >= self.target {
            Status::Terminal(1.0)
        } else if self.scores[Player::Two] >= self.target {
            Status::Terminal(-1.0)
        } else {
            Status::Ongoing(self.current)
        }
    }

    fn legal_actions(&self, buf: &mut Vec<usize>) {
        buf.push(ROLL);
        buf.push(HOLD);
    }

    fn apply_action(&mut self, action: usize) {
        if matches!(self.phase, Phase::Rolling) {
            let face = Self::die_face(action);
            if face == 1 {
                self.pass_turn();
            } else {
                self.turn_total += face;
                self.phase = Phase::Decision;
            }
            return;
        }
        match action {
            ROLL => {
                self.phase = Phase::Rolling;
            }
            HOLD => {
                self.scores[self.current] += self.turn_total;
                self.pass_turn();
            }
            _ => panic!("invalid action {action}"),
        }
    }

    fn chance_outcomes(&self, buf: &mut Vec<(usize, f32)>) {
        if matches!(self.phase, Phase::Rolling) {
            for i in 0..6 {
                buf.push((i, 1.0 / 6.0));
            }
        }
    }
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
    let game = PigGame::new(100);
    let eval = RolloutEvaluator { num_rollouts: 1 };
    let evaluators: PerPlayer<&dyn Evaluator<PigGame>> = PerPlayer([&eval, &eval]);

    println!(
        "=== Pig Tournament: {} vs {} simulations, {} games ===\n",
        configs.0[0].num_simulations, configs.0[1].num_simulations, num_games,
    );

    tournament::tournament(&game, &evaluators, &configs, num_games, &mut rng);
}
