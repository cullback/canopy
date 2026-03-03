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
//! 2. **Chance node** (die roll): outcome 1..=6
//!
//! This lets MCTS reason about the decision and stochasticity separately.

use std::collections::HashMap;
use std::hash::Hash;

use rand::Rng;

use canopy2::game::{Game, StochasticGame};
use canopy2::mcts::{Mcts, MctsConfig};

/// Actions in the game: player decisions + chance outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PigAction {
    Roll,
    Hold,
    /// Die face 1..=6. Only valid at chance nodes.
    DieRoll(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Player(u8);

/// Phase within a turn.
#[derive(Clone, Debug)]
enum Phase {
    /// Player decides to roll or hold.
    Decision,
    /// Die is being rolled (chance node).
    Rolling,
}

#[derive(Clone, Debug)]
struct PigGame {
    scores: [u32; 2],
    current: u8,
    turn_total: u32,
    phase: Phase,
    target: u32,
}

impl PigGame {
    fn new(target: u32) -> Self {
        Self {
            scores: [0; 2],
            current: 0,
            turn_total: 0,
            phase: Phase::Decision,
            target,
        }
    }

    fn opponent(&self) -> u8 {
        1 - self.current
    }

    fn pass_turn(&mut self) {
        self.current = self.opponent();
        self.turn_total = 0;
        self.phase = Phase::Decision;
    }
}

impl Game for PigGame {
    type Action = PigAction;
    type Player = Player;

    fn current_player(&self) -> Player {
        Player(self.current)
    }

    fn legal_actions(&self) -> Vec<PigAction> {
        match self.phase {
            Phase::Decision => vec![PigAction::Roll, PigAction::Hold],
            Phase::Rolling => (1..=6).map(PigAction::DieRoll).collect(),
        }
    }

    fn apply_action(&mut self, action: PigAction) {
        match action {
            PigAction::Roll => {
                // Player chose to roll — transition to chance node.
                self.phase = Phase::Rolling;
            }
            PigAction::Hold => {
                self.scores[self.current as usize] += self.turn_total;
                self.pass_turn();
            }
            PigAction::DieRoll(face) => {
                if face == 1 {
                    // Pig out — lose turn total.
                    self.pass_turn();
                } else {
                    self.turn_total += face as u32;
                    self.phase = Phase::Decision;
                }
            }
        }
    }

    fn is_terminal(&self) -> bool {
        self.scores[0] >= self.target || self.scores[1] >= self.target
    }

    fn rewards(&self) -> HashMap<Player, f32> {
        let mut m = HashMap::new();
        if self.scores[0] >= self.target {
            m.insert(Player(0), 1.0);
            m.insert(Player(1), 0.0);
        } else if self.scores[1] >= self.target {
            m.insert(Player(0), 0.0);
            m.insert(Player(1), 1.0);
        }
        m
    }

    fn state_key(&self) -> u64 {
        let mut h = self.scores[0] as u64;
        h = h * 200 + self.scores[1] as u64;
        h = h * 200 + self.turn_total as u64;
        h = h * 2 + self.current as u64;
        h = h * 2
            + match self.phase {
                Phase::Decision => 0,
                Phase::Rolling => 1,
            };
        h
    }

    fn action_index(action: &PigAction) -> usize {
        match action {
            PigAction::Roll => 0,
            PigAction::Hold => 1,
            PigAction::DieRoll(f) => 2 + (*f as usize - 1),
        }
    }

    fn action_space_size() -> usize {
        // Roll, Hold, DieRoll(1..=6)
        8
    }
}

impl StochasticGame for PigGame {
    fn is_chance_node(&self) -> bool {
        matches!(self.phase, Phase::Rolling)
    }

    fn chance_outcomes(&self) -> Vec<(PigAction, f32)> {
        (1..=6)
            .map(|f| (PigAction::DieRoll(f), 1.0 / 6.0))
            .collect()
    }
}

fn main() {
    let mut rng = rand::rng();
    let mut game = PigGame::new(100);
    let config = MctsConfig {
        simulations: 10_000,
        ..Default::default()
    };

    println!("=== Pig (MCTS vs MCTS, target=100) ===\n");

    let mut turn_num = 0;
    while !game.is_terminal() {
        let player = game.current_player();

        match game.phase {
            Phase::Decision => {
                let mut mcts = Mcts::new(MctsConfig {
                    simulations: config.simulations,
                    ..Default::default()
                });
                let visits = mcts.search(&game, &mut rng);

                let action = visits[0].0;
                let total_visits: u32 = visits.iter().map(|(_, v)| v).sum();
                print!(
                    "  P{} (score={}, turn={}): ",
                    player.0, game.scores[player.0 as usize], game.turn_total
                );
                for (a, v) in &visits {
                    let pct = *v as f32 / total_visits as f32 * 100.0;
                    print!("{:?}={:.0}%  ", a, pct);
                }
                println!("-> {:?}", action);

                game.apply_action(action);
            }
            Phase::Rolling => {
                // Chance node — sample a die roll.
                let outcomes = game.chance_outcomes();
                let total: f32 = outcomes.iter().map(|(_, p)| p).sum();
                let mut r: f32 = rng.random_range(0.0..total);
                let mut chosen = outcomes[0].0;
                for (a, p) in &outcomes {
                    r -= p;
                    if r <= 0.0 {
                        chosen = *a;
                        break;
                    }
                }
                if let PigAction::DieRoll(face) = chosen {
                    println!("  -> rolled {}", face);
                    if face == 1 {
                        turn_num += 1;
                        println!("  PIG OUT! Turn passes.\n--- Turn {} ---", turn_num);
                    }
                }
                game.apply_action(chosen);

                // Check if hold just happened implicitly (it didn't, but check terminal).
                if game.is_terminal() {
                    break;
                }
            }
        }

        // Detect turn change for display.
        if matches!(game.phase, Phase::Decision) && game.turn_total == 0 {
            turn_num += 1;
            println!("\n--- Turn {} ---", turn_num);
        }
    }

    println!("\n=== Final Scores ===");
    println!("  Player 0: {}", game.scores[0]);
    println!("  Player 1: {}", game.scores[1]);
    let winner = if game.scores[0] >= 100 { 0 } else { 1 };
    println!("  Player {} wins!", winner);
}
