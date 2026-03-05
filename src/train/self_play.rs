use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::mcts::{Config, Search};
use crate::nn::StateEncoder;
use crate::player::Player;

use super::{Sample, TrainConfig};

/// Result of a single self-play game.
pub(super) struct GameResult {
    pub samples: Vec<Sample>,
    pub winner: Option<Player>,
    pub num_turns: u32,
}

/// Aggregated results from one self-play iteration (many games).
pub(super) struct IterGameResults {
    pub samples: Vec<Sample>,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub total_turns: u32,
    pub min_game_length: Option<u32>,
    pub max_game_length: u32,
}

pub(super) fn self_play_game<G, Ev, E>(
    evaluator: &Ev,
    config: &TrainConfig,
    effective_sims: u32,
    seed: u64,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> GameResult
where
    G: Game,
    Ev: Evaluator<G> + Clone,
    E: StateEncoder<G>,
{
    let mut rng = fastrand::Rng::with_seed(seed);
    let base_config = Config {
        num_simulations: effective_sims,
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
    };
    let fast_sims = config.playout_cap_fast_sims.min(effective_sims);

    let mut search = Search::new(new_state(&mut rng));
    let mut actions = Vec::new();
    let mut samples = Vec::new();
    let mut turn_count: u32 = 0;
    let mut last_player: Option<Player> = None;

    loop {
        // Resolve chance events
        if let Some(action) = search.state().sample_chance(&mut rng) {
            search.apply_action(action);
            continue;
        }

        let current = match search.state().status() {
            Status::Terminal(_) => break,
            Status::Ongoing(p) => p,
        };

        // Track turn count via player changes
        if last_player != Some(current) {
            turn_count += 1;
            last_player = Some(current);
        }

        actions.clear();
        search.state().legal_actions(&mut actions);

        // Skip forced moves (single legal action)
        if actions.len() == 1 {
            search.apply_action(actions[0]);
            continue;
        }

        // Playout cap randomization: coin flip for full vs fast search
        let is_full = rng.f32() < config.playout_cap_full_prob;
        let move_config = Config {
            num_simulations: if is_full { effective_sims } else { fast_sims },
            ..base_config.clone()
        };

        // Encode state from current player's perspective
        let mut features_buf = Vec::with_capacity(E::FEATURE_SIZE);
        E::encode(search.state(), &mut features_buf);

        let result = search.run_to_completion(&move_config, evaluator, &mut rng);

        // Root Q from current player's perspective.
        // result.value is in [-1,1] from P1's perspective.
        let q = result.value * current.sign();

        // Early-game: sample from improved policy for diversity.
        // After explore_moves: use SH survivor deterministically.
        let chosen = if turn_count <= config.explore_moves {
            sample_from_policy(&result.policy, &mut rng)
        } else {
            result.selected_action
        };

        // z stores the player's sign as a placeholder; multiplied by the
        // terminal reward once the game ends.
        samples.push(Sample {
            features: features_buf.into_boxed_slice(),
            policy_target: result.policy.into_boxed_slice(),
            z: current.sign(),
            q,
            full_search: is_full,
        });
        search.apply_action(chosen);
    }

    // Assign value targets from game outcome.
    // z currently holds the player's sign; multiply by terminal reward
    // to get the value target from each player's perspective.
    let terminal_value = match search.state().status() {
        Status::Terminal(reward) => reward,
        _ => 0.0,
    };
    for s in &mut samples {
        s.z *= terminal_value;
    }

    let winner = if terminal_value > 0.0 {
        Some(Player::One)
    } else if terminal_value < 0.0 {
        Some(Player::Two)
    } else {
        None
    };

    GameResult {
        samples,
        winner,
        num_turns: turn_count,
    }
}

fn sample_from_policy(policy: &[f32], rng: &mut fastrand::Rng) -> usize {
    let mut roll = rng.f32();
    for (i, &p) in policy.iter().enumerate() {
        roll -= p;
        if roll <= 0.0 {
            return i;
        }
    }
    // Fallback: return argmax
    policy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

/// Generic game-playing loop: resolve chance, check terminal, select action via closure.
/// Returns the terminal reward from Player::One's perspective.
pub(super) fn play_game<G: Game>(
    state: &mut G,
    mut select_action: impl FnMut(&G, Player, &mut fastrand::Rng) -> usize,
    rng: &mut fastrand::Rng,
) -> f32 {
    loop {
        if let Some(action) = state.sample_chance(rng) {
            state.apply_action(action);
            continue;
        }
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing(player) => {
                let action = select_action(state, player, rng);
                state.apply_action(action);
            }
        }
    }
}

/// Run self-play games in parallel and aggregate results.
pub(super) fn run_self_play_iteration<G, E, Ev>(
    evaluator: Ev,
    config: &TrainConfig,
    effective_sims: u32,
    iteration: usize,
    rng: &mut fastrand::Rng,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> IterGameResults
where
    G: Game,
    E: StateEncoder<G>,
    Ev: Evaluator<G> + Clone + Send,
{
    use rayon::prelude::*;

    let pb = indicatif::ProgressBar::new(config.games_per_iter as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message(format!(
        "iter {}/{} self-play (sims={})",
        iteration + 1,
        config.iterations,
        effective_sims,
    ));

    let seeds: Vec<u64> = (0..config.games_per_iter).map(|_| rng.u64(..)).collect();
    let completed = std::sync::atomic::AtomicU32::new(0);

    let results: Vec<GameResult> = seeds
        .into_par_iter()
        .map_with(evaluator, |ev, seed| {
            let result = self_play_game::<G, Ev, E>(ev, config, effective_sims, seed, new_state);
            let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            pb.set_position(done as u64);
            result
        })
        .collect();
    pb.finish();

    let mut samples = Vec::new();
    let mut p1_wins = 0u32;
    let mut p2_wins = 0u32;
    let mut draws = 0u32;
    let mut total_turns = 0u32;
    let mut min_game_length: Option<u32> = None;
    let mut max_game_length = 0u32;

    for game in results {
        total_turns += game.num_turns;
        min_game_length =
            Some(min_game_length.map_or(game.num_turns, |m: u32| m.min(game.num_turns)));
        max_game_length = max_game_length.max(game.num_turns);
        match game.winner {
            Some(Player::One) => p1_wins += 1,
            Some(Player::Two) => p2_wins += 1,
            None => draws += 1,
        }
        samples.extend(game.samples);
    }

    IterGameResults {
        samples,
        p1_wins,
        p2_wins,
        draws,
        total_turns,
        min_game_length,
        max_game_length,
    }
}
