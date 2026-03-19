use std::path::PathBuf;

use indicatif::ProgressStyle;
use tracing::info;
use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::eval::{Evaluator, Evaluators};
use crate::game::{Game, Status};
use crate::game_log::GameLog;
use crate::mcts::{Config, Search, SearchResult, Step};

/// Parsed tournament settings.
pub struct TournamentOptions {
    pub num_games: u32,
    pub configs: [Config; 2],
    pub log_dir: Option<PathBuf>,
    pub eval_names: [String; 2],
}

impl TournamentOptions {
    /// Run a full tournament: print banner, play games, print results, save logs.
    pub fn run<G: Game>(
        &self,
        new_game: impl Fn(u64) -> G,
        registry: &Evaluators<G>,
    ) -> Vec<GameLog> {
        let mut rng = fastrand::Rng::new();

        let evaluators: [&dyn Evaluator<G>; 2] = [
            registry.get(&self.eval_names[0]),
            registry.get(&self.eval_names[1]),
        ];

        info!(
            "=== Tournament: {} ({}) vs {} ({}) simulations, {} games ===",
            self.eval_names[0],
            self.configs[0].num_simulations,
            self.eval_names[1],
            self.configs[1].num_simulations,
            self.num_games,
        );

        let logs = tournament(
            new_game,
            &evaluators,
            &self.configs,
            self.num_games,
            &mut rng,
        );

        if let Some(dir) = &self.log_dir {
            save_game_logs(&logs, dir);
        }

        logs
    }
}

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
    let mut action_buf = Vec::new();

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

            let action = if configs[seat].num_simulations == 0 {
                let eval_result = eval.evaluate(&state, rng);
                action_buf.clear();
                state.legal_actions(&mut action_buf);
                crate::utils::sample_policy(&eval_result.policy_logits, &action_buf, rng)
            } else {
                let config = configs[seat].clone();
                let result = run_to_completion(&mut Search::new(state.clone(), config), eval, rng);
                result.selected_action
            };

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

    let span = tracing::info_span!("tournament");
    span.pb_set_style(
        &ProgressStyle::with_template(
            "{bar:30} {pos}/{len} {per_sec} | W {msg} | [{elapsed} < {eta}]",
        )
        .unwrap(),
    );
    span.pb_set_length(num_games as u64);
    span.pb_set_message("0-0-0");
    span.pb_start();

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

        span.pb_set_message(&format!("{}-{}-{}", wins[0], wins[1], draws));
        span.pb_inc(1);
    }

    let total = num_games;
    let elapsed = crate::utils::HumanDuration(span.pb_elapsed());
    drop(span);

    info!(
        "W {}/{} ({:.1}%) | L {}/{} ({:.1}%) | D {}/{} ({:.1}%) | {elapsed}",
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

/// Write game logs to a directory, one file per game.
pub fn save_game_logs(logs: &[GameLog], dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("failed to create log directory");
    for (i, log) in logs.iter().enumerate() {
        let path = dir.join(format!("game_{i}.log"));
        log.write(&path);
        info!("Wrote {}", path.display());
    }
}
