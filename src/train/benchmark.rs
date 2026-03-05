use crate::eval::Evaluator;
use crate::game::Game;
use crate::mcts::Config;
use crate::player::Player;

use super::TrainConfig;
use super::self_play::{play_game, run_search};

/// Play benchmark games: NN evaluator vs RolloutEvaluator, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
pub(super) fn run_benchmark<G: Game, Ev: Evaluator<G>>(
    evaluator: &Ev,
    config: &TrainConfig,
    rng: &mut fastrand::Rng,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> (u32, u32, u32) {
    use crate::eval::RolloutEvaluator;

    let nn_config = Config {
        num_simulations: config.mcts_sims,
        ..Default::default()
    };
    let baseline_config = Config {
        num_simulations: config.bench_baseline_sims,
        ..Default::default()
    };
    let baseline_eval = RolloutEvaluator {
        num_rollouts: config.bench_baseline_rollouts,
    };

    let mut nn_wins = 0u32;
    let mut nn_losses = 0u32;
    let mut draws = 0u32;

    let pb = indicatif::ProgressBar::new(config.bench_games as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message("bench W:0 L:0 D:0".to_string());

    for i in 0..config.bench_games {
        let mut state = new_state(rng);
        let nn_player = Player::from(i as usize % 2);

        let reward = play_game(
            &mut state,
            |state, current, rng| {
                let result = if current == nn_player {
                    run_search(state, evaluator, &nn_config, rng)
                } else {
                    run_search(state, &baseline_eval, &baseline_config, rng)
                };
                result.selected_action
            },
            rng,
        );

        let nn_reward = reward * nn_player.sign();
        match nn_reward.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => nn_wins += 1,
            Some(std::cmp::Ordering::Less) => nn_losses += 1,
            _ => draws += 1,
        }

        pb.set_message(format!("bench W:{nn_wins} L:{nn_losses} D:{draws}"));
        pb.inc(1);
    }
    pb.finish();

    (nn_wins, nn_losses, draws)
}
