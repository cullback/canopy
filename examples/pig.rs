//! # Pig dice game with autoregressive actions
//!
//! Two players race to 100 points. On your turn you repeatedly roll a die:
//!   - Roll 2-6: add to your turn total, then decide to **roll** or **hold**.
//!   - Roll 1: lose your turn total, turn ends.
//!   - Hold: bank your turn total and pass the turn.
//!
//! ## Autoregressive action decomposition
//!
//! Instead of a single "Roll" action that simultaneously decides *and* samples,
//! we split each turn step into two phases:
//!
//! 1. **Decision node** (player chooses): `Roll` or `Hold`
//! 2. **Chance node** (die roll): outcome 0..=5 mapping to faces 1..=6
//!
//! Chance outcomes flow through `apply_action` — the game knows it's in a
//! chance state and interprets the `usize` as a die outcome.

use rand::Rng;

use canopy2::game::{Game, Status};
use canopy2::mcts::{Config, RolloutEvaluator, search};
use canopy2::player::{PerPlayer, Player};

const ROLL: usize = 0;
const HOLD: usize = 1;

const ACTION_NAMES: [&str; 2] = ["Roll", "Hold"];

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
    let mut rng = rand::rng();
    let mut game = PigGame::new(100);
    let simulations = 10_000;
    let evaluator = RolloutEvaluator { num_rollouts: 1 };

    println!("=== Pig (MCTS vs MCTS, target=100) ===\n");

    let mut turn_num = 0;
    let mut chance_buf = Vec::new();
    while let Status::Ongoing(player) = game.status() {
        chance_buf.clear();
        game.chance_outcomes(&mut chance_buf);
        if !chance_buf.is_empty() {
            let total: f32 = chance_buf.iter().map(|(_, p)| p).sum();
            let mut r: f32 = rng.random_range(0.0..total);
            let mut chosen = chance_buf[0].0;
            for &(o, p) in &chance_buf {
                r -= p;
                if r <= 0.0 {
                    chosen = o;
                    break;
                }
            }
            let face = PigGame::die_face(chosen);
            println!("  -> rolled {}", face);
            if face == 1 {
                turn_num += 1;
                println!("  PIG OUT! Turn passes.\n--- Turn {} ---", turn_num);
            }
            game.apply_action(chosen);
        } else {
            let config = Config {
                num_simulations: simulations,
                ..Default::default()
            };
            let result = search(&game, &evaluator, &config, &mut rng);

            let action = result
                .policy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            print!(
                "  {} (score={}, turn={}): ",
                player, game.scores[player], game.turn_total
            );
            for (a, name) in ACTION_NAMES.iter().enumerate() {
                let pct = result.policy[a] * 100.0;
                if pct > 0.0 {
                    print!("{}={:.0}%  ", name, pct);
                }
            }
            println!("-> {}", ACTION_NAMES[action]);

            game.apply_action(action);
        }

        if !matches!(game.phase, Phase::Rolling)
            && game.turn_total == 0
            && matches!(game.status(), Status::Ongoing(_))
        {
            turn_num += 1;
            println!("\n--- Turn {} ---", turn_num);
        }
    }

    println!("\n=== Final Scores ===");
    println!("  Player 1: {}", game.scores[Player::One]);
    println!("  Player 2: {}", game.scores[Player::Two]);
    if let Status::Terminal(reward) = game.status() {
        let winner = if reward > 0.0 {
            Player::One
        } else {
            Player::Two
        };
        println!("  {} wins!", winner);
    }
}
