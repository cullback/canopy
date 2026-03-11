use indicatif::{ProgressBar, ProgressStyle};

use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::game_log::GameLog;
use crate::mcts::{Config, Search, SearchResult, Step};

/// Drive a search to completion using the provided evaluator.
fn run_to_completion<G: Game, E: Evaluator<G> + ?Sized>(
    search: &mut Search<G>,
    evaluator: &E,
    rng: &mut fastrand::Rng,
) -> SearchResult {
    let mut evals = vec![];
    loop {
        match search.step(&evals, rng) {
            Step::NeedsEval(states) => {
                let refs: Vec<&G> = states.iter().collect();
                evals = evaluator.evaluate_batch(&refs, rng);
            }
            Step::Done(result) => return result,
        }
    }
}

/// Play a single match between two MCTS bots.
///
/// Returns the terminal reward from P1's perspective and a log of every action
/// applied (both chance outcomes and player decisions).
/// When `swap` is true, seat assignments are reversed:
/// the game's P1 uses `configs[1]` and vice versa.
pub fn play_match<G: Game>(
    game: &G,
    evaluators: &[&dyn Evaluator<G>; 2],
    configs: &[Config; 2],
    swap: bool,
    rng: &mut fastrand::Rng,
) -> (f32, Vec<usize>) {
    let mut state = game.clone();
    let mut actions = Vec::new();

    loop {
        match state.status() {
            Status::Terminal(reward) => return (reward, actions),
            Status::Ongoing => {}
        };

        if let Some(action) = state.sample_chance(rng) {
            actions.push(action);
            state.apply_action(action);
        } else {
            // sign-to-index: 1.0 → 0, -1.0 → 1
            let idx = ((1.0 - state.current_sign()) / 2.0) as usize;
            let seat = idx ^ (swap as usize);
            let eval = evaluators[seat];
            let config = configs[seat].clone();
            let result = run_to_completion(&mut Search::new(state.clone(), config), eval, rng);
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
    evaluators: &[&dyn Evaluator<G>; 2],
    configs: &[Config; 2],
    num_games: u32,
    rng: &mut fastrand::Rng,
) -> Vec<GameLog> {
    let mut wins: [u32; 2] = [0, 0];
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
            wins[0] += 1;
        } else if seat0_reward < 0.0 {
            wins[1] += 1;
        } else {
            draws += 1;
        }

        pb.set_message(format!("{}-{}-{}", wins[0], wins[1], draws));
        pb.inc(1);
    }

    pb.finish();

    let total = num_games;
    println!(
        "W {}/{} ({:.1}%) | L {}/{} ({:.1}%) | D {}/{} ({:.1}%)",
        wins[0],
        total,
        wins[0] as f32 / total as f32 * 100.0,
        wins[1],
        total,
        wins[1] as f32 / total as f32 * 100.0,
        draws,
        total,
        draws as f32 / total as f32 * 100.0,
    );

    game_logs
}
