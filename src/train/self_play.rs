use crate::eval::Evaluator;
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, SearchResult, Step};
use crate::nn::StateEncoder;
use crate::player::Player;

use super::{Sample, TrainConfig};

struct GameRecord {
    features: Box<[f32]>,
    policy_target: Box<[f32]>,
    player: Player,
    q: f32,
    full_search: bool,
}

pub(super) struct GameStats {
    pub winner: Option<Player>,
    pub num_turns: u32,
}

pub(super) struct IterGameResults {
    pub samples: Vec<Sample>,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub total_turns: u32,
    pub min_game_length: Option<u32>,
    pub max_game_length: u32,
}

/// Drive an in-progress MCTS search state machine to completion.
fn drive_search<G: Game, E: Evaluator<G>>(
    search: &mut Search<G>,
    mut step: Step<G>,
    evaluator: &E,
    rng: &mut fastrand::Rng,
) -> SearchResult {
    loop {
        match step {
            Step::NeedsEval(pending) => {
                let output = evaluator.evaluate(&pending.state, rng);
                step = search.supply(output, pending, rng);
            }
            Step::Done(result) => return result,
        }
    }
}

/// Create a fresh search and drive it to completion.
pub(super) fn run_search<G: Game, E: Evaluator<G>>(
    state: &G,
    evaluator: &E,
    config: &Config,
    rng: &mut fastrand::Rng,
) -> SearchResult {
    let (mut search, step) = Search::start(state, config, rng);
    drive_search(&mut search, step, evaluator, rng)
}

/// Wraps MCTS tree reuse bookkeeping: persists the search tree across moves
/// and tracks intermediate actions (chance outcomes, forced moves) so the
/// tree can be advanced to the current state.
struct ReusableSearch<G: Game> {
    search: Option<Search<G>>,
    pending_actions: Vec<usize>,
}

impl<G: Game> ReusableSearch<G> {
    fn new() -> Self {
        Self {
            search: None,
            pending_actions: Vec::new(),
        }
    }

    fn track_action(&mut self, action: usize) {
        self.pending_actions.push(action);
    }

    fn run<E: Evaluator<G>>(
        &mut self,
        state: &G,
        evaluator: &E,
        config: &Config,
        rng: &mut fastrand::Rng,
    ) -> SearchResult {
        match self.search {
            Some(ref mut s) => {
                let step = s.step_to(state, &self.pending_actions, config, rng);
                self.pending_actions.clear();
                drive_search(s, step, evaluator, rng)
            }
            None => {
                let (mut s, step) = Search::start(state, config, rng);
                let result = drive_search(&mut s, step, evaluator, rng);
                self.search = Some(s);
                self.pending_actions.clear();
                result
            }
        }
    }
}

pub(super) fn self_play_game<G, Ev, E>(
    evaluator: &Ev,
    config: &TrainConfig,
    effective_sims: u32,
    seed: u64,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> (Vec<Sample>, GameStats)
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

    let mut state = new_state(&mut rng);
    let mut actions = Vec::new();
    let mut chance_buf = Vec::new();
    let mut records = Vec::new();
    let mut features_buf = Vec::new();
    let mut turn_count: u32 = 0;
    let mut last_player: Option<Player> = None;
    let mut tree = ReusableSearch::new();

    loop {
        // Resolve chance events
        chance_buf.clear();
        state.chance_outcomes(&mut chance_buf);
        if !chance_buf.is_empty() {
            let action = sample_chance(&chance_buf, &mut rng);
            tree.track_action(action);
            state.apply_action(action);
            continue;
        }

        let current = match state.status() {
            Status::Terminal(_) => break,
            Status::Ongoing(p) => p,
        };

        // Track turn count via player changes
        if last_player != Some(current) {
            turn_count += 1;
            last_player = Some(current);
        }

        actions.clear();
        state.legal_actions(&mut actions);

        // Skip forced moves (single legal action)
        if actions.len() == 1 {
            tree.track_action(actions[0]);
            state.apply_action(actions[0]);
            continue;
        }

        // Playout cap randomization: coin flip for full vs fast search
        let is_full = rng.f32() < config.playout_cap_full_prob;
        let move_config = Config {
            num_simulations: if is_full { effective_sims } else { fast_sims },
            ..base_config.clone()
        };

        let result = tree.run(&state, evaluator, &move_config, &mut rng);

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

        // Encode state from current player's perspective
        E::encode(&state, &mut features_buf);

        records.push(GameRecord {
            features: features_buf.clone().into_boxed_slice(),
            policy_target: result.policy.into_boxed_slice(),
            player: current,
            q,
            full_search: is_full,
        });
        tree.track_action(chosen);
        state.apply_action(chosen);
    }

    // Assign value targets from game outcome
    let (terminal_value, winner) = match state.status() {
        Status::Terminal(reward) => {
            let winner = if reward > 0.0 {
                Some(Player::One)
            } else if reward < 0.0 {
                Some(Player::Two)
            } else {
                None
            };
            (reward, winner)
        }
        _ => (0.0, None),
    };

    let stats = GameStats {
        winner,
        num_turns: turn_count,
    };

    let samples = records
        .into_iter()
        .map(|r| {
            // terminal_value is from P1's perspective in [-1,1].
            // Convert to current player's perspective.
            let z = terminal_value * r.player.sign();
            Sample {
                features: r.features,
                policy_target: r.policy_target,
                z,
                q: r.q,
                full_search: r.full_search,
            }
        })
        .collect();
    (samples, stats)
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
    let mut chance_buf = Vec::new();
    loop {
        chance_buf.clear();
        state.chance_outcomes(&mut chance_buf);
        if !chance_buf.is_empty() {
            let action = sample_chance(&chance_buf, rng);
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

pub(super) fn sample_chance(outcomes: &[(usize, f32)], rng: &mut fastrand::Rng) -> usize {
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

    let results: Vec<(Vec<Sample>, GameStats)> = seeds
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

    for (game_samples, stats) in results {
        total_turns += stats.num_turns;
        min_game_length =
            Some(min_game_length.map_or(stats.num_turns, |m: u32| m.min(stats.num_turns)));
        max_game_length = max_game_length.max(stats.num_turns);
        match stats.winner {
            Some(Player::One) => p1_wins += 1,
            Some(Player::Two) => p2_wins += 1,
            None => draws += 1,
        }
        samples.extend(game_samples);
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
