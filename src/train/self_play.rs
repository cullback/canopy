use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::mpsc;

use crate::eval::{Evaluation, Evaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, PendingEval, Search, Step};
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
    pub num_workers: usize,
    pub total_batches: u64,
    pub total_evals: u64,
}

impl IterGameResults {
    pub fn avg_batch_size(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.total_evals as f64 / self.total_batches as f64
        }
    }
}

/// A leaf-node evaluation request sent from a worker thread to the batcher.
struct EvalRequest<G: Game> {
    pending: PendingEval<G>,
    response_tx: mpsc::SyncSender<(Evaluation, PendingEval<G>)>,
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

/// Worker thread: plays sequential games, sending leaf evals to the batcher.
fn worker_loop<G: Game, E: StateEncoder<G>>(
    request_tx: mpsc::Sender<EvalRequest<G>>,
    games_remaining: &AtomicU32,
    completed: &AtomicU32,
    pb: &indicatif::ProgressBar,
    config: &TrainConfig,
    base_config: &Config,
    effective_sims: u32,
    fast_sims: u32,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
    seed: u64,
) -> Vec<GameResult> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let (resp_tx, resp_rx) = mpsc::sync_channel::<(Evaluation, PendingEval<G>)>(1);
    let mut results = Vec::new();
    let mut actions_buf = Vec::new();

    loop {
        // Claim a game via atomic decrement
        let claimed = games_remaining
            .fetch_update(Relaxed, Relaxed, |n| if n > 0 { Some(n - 1) } else { None });
        if claimed.is_err() {
            break;
        }

        let mut search = Search::new(new_state(&mut rng));
        let mut samples: Vec<Sample> = Vec::new();
        let mut turn_count: u32 = 0;
        let mut last_player: Option<Player> = None;

        loop {
            // Resolve chance nodes
            if let Some(action) = search.state().sample_chance(&mut rng) {
                search.apply_action(action);
                continue;
            }

            // Terminal check
            if let Status::Terminal(reward) = search.state().status() {
                for s in &mut samples {
                    s.z *= reward;
                }
                let winner = if reward > 0.0 {
                    Some(Player::One)
                } else if reward < 0.0 {
                    Some(Player::Two)
                } else {
                    None
                };
                results.push(GameResult {
                    samples: std::mem::take(&mut samples),
                    winner,
                    num_turns: turn_count,
                });
                let done = completed.fetch_add(1, Relaxed) + 1;
                pb.set_position(done as u64);
                break;
            }

            let current = match search.state().status() {
                Status::Ongoing(p) => p,
                Status::Terminal(_) => unreachable!(),
            };

            // Track turn count via player changes
            if last_player != Some(current) {
                turn_count += 1;
                last_player = Some(current);
            }

            // Skip forced moves (single legal action)
            actions_buf.clear();
            search.state().legal_actions(&mut actions_buf);
            if actions_buf.len() == 1 {
                search.apply_action(actions_buf[0]);
                continue;
            }

            // Playout cap randomization
            let is_full_search = rng.f32() < config.playout_cap_full_prob;
            let move_config = Config {
                num_simulations: if is_full_search {
                    effective_sims
                } else {
                    fast_sims
                },
                ..base_config.clone()
            };

            // Encode features before search
            let mut features_buf = Vec::with_capacity(E::FEATURE_SIZE);
            E::encode(search.state(), &mut features_buf);

            // MCTS search with eval requests sent to batcher
            let mut step = search.run(&move_config, &mut rng);
            let result = loop {
                match step {
                    Step::NeedsEval(pending) => {
                        request_tx
                            .send(EvalRequest {
                                pending,
                                response_tx: resp_tx.clone(),
                            })
                            .unwrap();
                        let (eval, pending) = resp_rx.recv().unwrap();
                        step = search.supply(eval, pending, &mut rng);
                    }
                    Step::Done(result) => break result,
                }
            };

            // Create training sample
            let q = result.value * current.sign();
            let chosen = if turn_count <= config.explore_moves {
                sample_from_policy(&result.policy, &mut rng)
            } else {
                result.selected_action
            };

            samples.push(Sample {
                features: features_buf.into_boxed_slice(),
                policy_target: result.policy.into_boxed_slice(),
                z: current.sign(),
                q,
                full_search: is_full_search,
            });

            search.apply_action(chosen);
        }
    }

    results
}

/// Main-thread batcher: collects eval requests and runs batched NN inference.
/// Returns (total_batches, total_evals) for diagnostics.
fn batcher_loop<G: Game, Ev: Evaluator<G>>(
    request_rx: &mpsc::Receiver<EvalRequest<G>>,
    evaluator: &Ev,
    rng: &mut fastrand::Rng,
    batch_limit: usize,
) -> (u64, u64) {
    let mut batch: Vec<EvalRequest<G>> = Vec::with_capacity(batch_limit);
    let mut total_batches: u64 = 0;
    let mut total_evals: u64 = 0;

    loop {
        // Block for the first request (Err = all workers dropped their senders)
        let first = match request_rx.recv() {
            Ok(req) => req,
            Err(_) => return (total_batches, total_evals),
        };
        batch.push(first);

        // Drain additional requests without blocking, up to batch_limit
        while batch.len() < batch_limit {
            match request_rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        total_batches += 1;
        total_evals += batch.len() as u64;

        // Batched evaluation
        let states: Vec<&G> = batch.iter().map(|r| &r.pending.state).collect();
        let evals = evaluator.evaluate_batch(&states, rng);

        // Send results back to workers
        for (req, eval) in batch.drain(..).zip(evals) {
            // Ignore send errors (worker may have panicked)
            let _ = req.response_tx.send((eval, req.pending));
        }
    }
}

/// Run self-play games using worker threads with batched NN inference.
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
    Ev: Evaluator<G>,
{
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

    let base_config = Config {
        num_simulations: effective_sims,
        num_sampled_actions: config.gumbel_m,
        c_visit: config.c_visit,
        c_scale: config.c_scale,
    };
    let fast_sims = config.playout_cap_fast_sims.min(effective_sims);

    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(config.games_per_iter);
    let games_remaining = AtomicU32::new(config.games_per_iter as u32);
    let completed = AtomicU32::new(0);
    let (request_tx, request_rx) = mpsc::channel();

    let (game_results, total_batches, total_evals) = std::thread::scope(|scope| {
        // Spawn worker threads
        let games_ref = &games_remaining;
        let completed_ref = &completed;
        let pb_ref = &pb;
        let base_config_ref = &base_config;
        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let tx = request_tx.clone();
                let seed = rng.u64(..);
                scope.spawn(move || {
                    worker_loop::<G, E>(
                        tx,
                        games_ref,
                        completed_ref,
                        pb_ref,
                        config,
                        base_config_ref,
                        effective_sims,
                        fast_sims,
                        new_state,
                        seed,
                    )
                })
            })
            .collect();

        // Drop our copy so batcher sees disconnect when all workers finish
        drop(request_tx);

        // Batcher runs on main thread
        let (total_batches, total_evals) = batcher_loop(&request_rx, &evaluator, rng, num_workers);

        // Join workers, collect results
        let game_results: Vec<GameResult> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        (game_results, total_batches, total_evals)
    });

    pb.finish();

    // Aggregate results
    let mut samples = Vec::new();
    let mut p1_wins = 0u32;
    let mut p2_wins = 0u32;
    let mut draws = 0u32;
    let mut total_turns = 0u32;
    let mut min_game_length: Option<u32> = None;
    let mut max_game_length = 0u32;

    for game in game_results {
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
        num_workers,
        total_batches,
        total_evals,
    }
}
