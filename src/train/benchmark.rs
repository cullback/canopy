use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::mpsc;

use crate::eval::{Evaluator, RolloutEvaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, PendingEval, Search, Step};
use crate::player::Player;

use super::TrainConfig;
use super::self_play::{EvalRequest, batcher_loop};

/// Play benchmark games: NN evaluator vs RolloutEvaluator, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
pub(super) fn run_benchmark<G: Game, Ev: Evaluator<G>>(
    evaluator: &Ev,
    config: &TrainConfig,
    rng: &mut fastrand::Rng,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
) -> (u32, u32, u32) {
    let nn_config = Config {
        num_simulations: config.mcts_sims,
        leaf_batch_size: config.leaf_batch_size,
        ..Default::default()
    };
    let baseline_config = Config {
        num_simulations: config.bench_baseline_sims,
        ..Default::default()
    };
    let baseline_eval = RolloutEvaluator {
        num_rollouts: config.bench_baseline_rollouts,
    };

    let nn_wins = AtomicU32::new(0);
    let nn_losses = AtomicU32::new(0);
    let draws = AtomicU32::new(0);
    let game_counter = AtomicU32::new(0);

    let pb = indicatif::ProgressBar::new(config.bench_games as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message("bench W:0 L:0 D:0".to_string());

    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(config.bench_games as usize);

    let (request_tx, request_rx) = mpsc::channel();

    std::thread::scope(|scope| {
        let nn_wins_ref = &nn_wins;
        let nn_losses_ref = &nn_losses;
        let draws_ref = &draws;
        let game_counter_ref = &game_counter;
        let pb_ref = &pb;
        let nn_config_ref = &nn_config;
        let baseline_config_ref = &baseline_config;
        let baseline_eval_ref = &baseline_eval;

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let tx = request_tx.clone();
                let seed = rng.u64(..);
                scope.spawn(move || {
                    bench_worker::<G>(
                        tx,
                        game_counter_ref,
                        config.bench_games,
                        nn_config_ref,
                        baseline_config_ref,
                        baseline_eval_ref,
                        nn_wins_ref,
                        nn_losses_ref,
                        draws_ref,
                        pb_ref,
                        new_state,
                        seed,
                    );
                })
            })
            .collect();

        // Drop our sender so batcher sees disconnect when all workers finish
        drop(request_tx);

        // Batcher runs on main thread
        let batch_limit = num_workers * nn_config_ref.leaf_batch_size as usize;
        batcher_loop(&request_rx, evaluator, rng, batch_limit);

        // Join workers
        for h in handles {
            h.join().unwrap();
        }
    });

    pb.finish();

    (
        nn_wins.load(Relaxed),
        nn_losses.load(Relaxed),
        draws.load(Relaxed),
    )
}

/// Worker thread: plays benchmark games, sending NN evals to batcher,
/// running baseline evals locally.
fn bench_worker<G: Game>(
    request_tx: mpsc::Sender<EvalRequest<G>>,
    game_counter: &AtomicU32,
    bench_games: u32,
    nn_config: &Config,
    baseline_config: &Config,
    baseline_eval: &RolloutEvaluator,
    nn_wins: &AtomicU32,
    nn_losses: &AtomicU32,
    draws: &AtomicU32,
    pb: &indicatif::ProgressBar,
    new_state: &(impl Fn(&mut fastrand::Rng) -> G + Sync),
    seed: u64,
) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let (resp_tx, resp_rx) = mpsc::sync_channel::<(crate::eval::Evaluation, PendingEval<G>)>(
        nn_config.leaf_batch_size as usize,
    );

    loop {
        // Claim a game via atomic increment
        let i = game_counter.fetch_add(1, Relaxed);
        if i >= bench_games {
            break;
        }

        let nn_player = Player::from(i as usize % 2);
        let mut state = new_state(&mut rng);

        let reward = play_bench_game(
            &mut state,
            nn_player,
            nn_config,
            baseline_config,
            baseline_eval,
            &request_tx,
            &resp_tx,
            &resp_rx,
            &mut rng,
        );

        let nn_reward = reward * nn_player.sign();
        match nn_reward.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => {
                nn_wins.fetch_add(1, Relaxed);
            }
            Some(std::cmp::Ordering::Less) => {
                nn_losses.fetch_add(1, Relaxed);
            }
            _ => {
                draws.fetch_add(1, Relaxed);
            }
        }

        let w = nn_wins.load(Relaxed);
        let l = nn_losses.load(Relaxed);
        let d = draws.load(Relaxed);
        pb.set_message(format!("bench W:{w} L:{l} D:{d}"));
        pb.inc(1);
    }
}

/// Play a single benchmark game. NN turns use run/supply loop with batcher;
/// baseline turns use run_to_completion locally.
fn play_bench_game<G: Game>(
    state: &mut G,
    nn_player: Player,
    nn_config: &Config,
    baseline_config: &Config,
    baseline_eval: &RolloutEvaluator,
    request_tx: &mpsc::Sender<EvalRequest<G>>,
    resp_tx: &mpsc::SyncSender<(crate::eval::Evaluation, PendingEval<G>)>,
    resp_rx: &mpsc::Receiver<(crate::eval::Evaluation, PendingEval<G>)>,
    rng: &mut fastrand::Rng,
) -> f32 {
    loop {
        if let Some(action) = state.sample_chance(rng) {
            state.apply_action(action);
            continue;
        }
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing(current) => {
                let action = if current == nn_player {
                    // NN player: batched eval via batcher
                    let mut search = Search::new(state.clone());
                    let mut step = search.run(nn_config, rng);
                    let result = loop {
                        match step {
                            Step::NeedsEval(pendings) => {
                                let count = pendings.len();
                                for pending in pendings {
                                    request_tx
                                        .send(EvalRequest {
                                            pending,
                                            response_tx: resp_tx.clone(),
                                        })
                                        .unwrap();
                                }
                                let evals: Vec<_> =
                                    (0..count).map(|_| resp_rx.recv().unwrap()).collect();
                                step = search.supply(evals, rng);
                            }
                            Step::Done(result) => break result,
                        }
                    };
                    result.selected_action
                } else {
                    // Baseline player: local rollout eval
                    Search::new(state.clone())
                        .run_to_completion(baseline_config, baseline_eval, rng)
                        .selected_action
                };
                state.apply_action(action);
            }
        }
    }
}
