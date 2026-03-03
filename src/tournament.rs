use indicatif::{ProgressBar, ProgressStyle};

use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, Step};
use crate::player::{PerPlayer, Player};

/// Play a single match between two MCTS bots.
///
/// Returns the terminal reward from P1's perspective.
/// When `swap` is true, seat assignments are reversed:
/// the game's P1 uses `configs[Player::Two]` and vice versa.
pub fn play_match<G: Game>(
    game: &G,
    evaluators: &PerPlayer<&dyn Evaluator<G>>,
    configs: &PerPlayer<Config>,
    swap: bool,
    rng: &mut fastrand::Rng,
) -> f32 {
    let mut state = game.clone();
    let mut chance_buf = Vec::new();

    loop {
        let player = match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing(p) => p,
        };

        chance_buf.clear();
        state.chance_outcomes(&mut chance_buf);

        if !chance_buf.is_empty() {
            let action = sample_chance(&chance_buf, rng);
            state.apply_action(action);
        } else {
            let seat = if swap { player.opponent() } else { player };
            let eval = evaluators.0[seat as usize];
            let config = &configs[seat];
            let (mut search, mut step) = Search::new(&state, config, rng);
            let result = loop {
                step = match step {
                    Step::NeedsEval(s) => {
                        let output = eval.evaluate(&s, rng);
                        search.supply(output, config, rng)
                    }
                    Step::Done(r) => break r,
                };
            };

            let action = result
                .policy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;

            state.apply_action(action);
        }
    }
}

/// Summary of a tournament between two MCTS configurations.
pub struct TournamentResult {
    /// Wins per seat (indexed by the original `configs` ordering).
    pub wins: PerPlayer<u32>,
    pub draws: u32,
    pub total: u32,
}

/// Run a tournament of `num_games` matches, alternating sides.
///
/// Even-numbered games use the original seat assignment;
/// odd-numbered games swap which config plays as P1.
/// Prints a summary line at the end.
pub fn tournament<G: Game>(
    game: &G,
    evaluators: &PerPlayer<&dyn Evaluator<G>>,
    configs: &PerPlayer<Config>,
    num_games: u32,
    rng: &mut fastrand::Rng,
) -> TournamentResult {
    let mut wins = PerPlayer::default();
    let mut draws = 0u32;

    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} | W {msg} | {elapsed} < {eta}").unwrap(),
    );
    pb.set_message("0-0-0");

    for i in 0..num_games {
        let swap = i % 2 == 1;
        let reward = play_match(game, evaluators, configs, swap, rng);

        // Map reward back to seat 0's perspective
        let seat0_reward = if swap { -reward } else { reward };

        if seat0_reward > 0.0 {
            wins[Player::One] += 1;
        } else if seat0_reward < 0.0 {
            wins[Player::Two] += 1;
        } else {
            draws += 1;
        }

        pb.set_message(format!(
            "{}-{}-{}",
            wins[Player::One],
            wins[Player::Two],
            draws,
        ));
        pb.inc(1);
    }

    pb.finish_and_clear();

    let total = num_games;
    println!(
        "P1 {}/{} ({:.1}%) | P2 {}/{} ({:.1}%) | Draws {} ({:.1}%)",
        wins[Player::One],
        total,
        wins[Player::One] as f32 / total as f32 * 100.0,
        wins[Player::Two],
        total,
        wins[Player::Two] as f32 / total as f32 * 100.0,
        draws,
        draws as f32 / total as f32 * 100.0,
    );

    TournamentResult { wins, draws, total }
}

fn sample_chance(outcomes: &[(usize, f32)], rng: &mut fastrand::Rng) -> usize {
    let total: f32 = outcomes.iter().map(|(_, p)| p).sum();
    let mut r = rng.f32() * total;
    for &(outcome, p) in outcomes {
        r -= p;
        if r <= 0.0 {
            return outcome;
        }
    }
    outcomes.last().unwrap().0
}
