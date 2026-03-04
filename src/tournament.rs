use indicatif::{ProgressBar, ProgressStyle};

use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::game_log::GameLog;
use crate::mcts::{Config, Search, Step};
use crate::player::{PerPlayer, Player};

/// Play a single match between two MCTS bots.
///
/// Returns the terminal reward from P1's perspective and a log of every action
/// applied (both chance outcomes and player decisions).
/// When `swap` is true, seat assignments are reversed:
/// the game's P1 uses `configs[Player::Two]` and vice versa.
pub fn play_match<G: Game>(
    game: &G,
    evaluators: &PerPlayer<&dyn Evaluator<G>>,
    configs: &PerPlayer<Config>,
    swap: bool,
    rng: &mut fastrand::Rng,
) -> (f32, Vec<usize>) {
    let mut state = game.clone();
    let mut chance_buf = Vec::new();
    let mut actions = Vec::new();

    loop {
        let player = match state.status() {
            Status::Terminal(reward) => return (reward, actions),
            Status::Ongoing(p) => p,
        };

        chance_buf.clear();
        state.chance_outcomes(&mut chance_buf);

        if !chance_buf.is_empty() {
            let action = sample_chance(&chance_buf, rng);
            actions.push(action);
            state.apply_action(action);
        } else {
            let seat = if swap { player.opponent() } else { player };
            let eval = evaluators.0[seat as usize];
            let config = &configs[seat];
            let (mut search, mut step) = Search::start(&state, config, rng);
            let result = loop {
                step = match step {
                    Step::NeedsEval(pending) => {
                        let output = eval.evaluate(&pending.state, rng);
                        search.supply(output, pending, rng)
                    }
                    Step::Done(r) => break r,
                };
            };

            let action = result.selected_action;

            actions.push(action);
            state.apply_action(action);
        }
    }
}

/// Run a tournament of `num_games` matches, alternating sides.
///
/// `new_game` is a factory that creates a fresh game state from a seed.
/// Even-numbered games use the original seat assignment;
/// odd-numbered games swap which config plays as P1.
pub fn tournament<G: Game>(
    new_game: impl Fn(u64) -> G,
    evaluators: &PerPlayer<&dyn Evaluator<G>>,
    configs: &PerPlayer<Config>,
    num_games: u32,
    rng: &mut fastrand::Rng,
) -> Vec<GameLog> {
    let mut wins: PerPlayer<u32> = PerPlayer::default();
    let mut draws = 0u32;
    let mut game_logs = Vec::with_capacity(num_games as usize);

    let pb = ProgressBar::new(num_games as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:30} {pos}/{len} | W {msg} | {elapsed} < {eta}").unwrap(),
    );
    pb.set_message("0-0-0");

    for i in 0..num_games {
        let swap = i % 2 == 1;
        let seed = rng.u64(..);
        let game = new_game(seed);
        let (reward, actions) = play_match(&game, evaluators, configs, swap, rng);
        game_logs.push(GameLog { seed, actions });

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

    pb.finish();

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

    game_logs
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
