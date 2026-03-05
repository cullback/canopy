use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::mcts::Config;
use crate::player::Player;

use super::self_play::{run_search, sample_chance};

/// Play benchmark games: NN evaluator vs RolloutEvaluator, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
pub(super) fn run_benchmark<G: Game, Ev: Evaluator<G>>(
    evaluator: &Ev,
    mcts_sims: u32,
    num_games: u32,
    rng: &mut fastrand::Rng,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> (u32, u32, u32) {
    use crate::eval::RolloutEvaluator;

    let nn_config = Config {
        num_simulations: mcts_sims,
        ..Default::default()
    };
    let baseline_config = Config {
        num_simulations: 200,
        ..Default::default()
    };
    let baseline_eval = RolloutEvaluator { num_rollouts: 1 };

    let mut nn_wins = 0u32;
    let mut nn_losses = 0u32;
    let mut draws = 0u32;

    let pb = indicatif::ProgressBar::new(num_games as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message("bench W:0 L:0 D:0".to_string());

    for i in 0..num_games {
        let mut state = new_state(rng);
        let nn_is_p1 = i % 2 == 0;
        let mut chance_buf = Vec::new();

        loop {
            // Resolve chance
            chance_buf.clear();
            state.chance_outcomes(&mut chance_buf);
            if !chance_buf.is_empty() {
                let action = sample_chance(&chance_buf, rng);
                state.apply_action(action);
                continue;
            }

            let current = match state.status() {
                Status::Terminal(_) => break,
                Status::Ongoing(p) => p,
            };

            let is_nn_turn = (current == Player::One) == nn_is_p1;
            let result = if is_nn_turn {
                run_search(&state, evaluator, &nn_config, rng)
            } else {
                run_search(&state, &baseline_eval, &baseline_config, rng)
            };

            state.apply_action(result.selected_action);
        }

        if let Status::Terminal(reward) = state.status() {
            // Map to nn's result
            let nn_reward = if nn_is_p1 { reward } else { -reward };
            if nn_reward > 0.0 {
                nn_wins += 1;
            } else if nn_reward < 0.0 {
                nn_losses += 1;
            } else {
                draws += 1;
            }
        }

        pb.set_message(format!("bench W:{nn_wins} L:{nn_losses} D:{draws}"));
        pb.inc(1);
    }
    pb.finish();

    (nn_wins, nn_losses, draws)
}
