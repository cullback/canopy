use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::eval::{Evaluator, RolloutEvaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, Step};
use crate::player::Player;

use super::TrainConfig;
use super::self_play::{BatcherStats, EvalRequest, EvalSender, batcher_loop};

/// Async bench game task: plays games alternating NN (batched) and baseline (local) turns.
#[allow(clippy::too_many_arguments)]
async fn bench_game_task<G: Game>(
    request_tx: EvalSender<G>,
    game_counter: Arc<AtomicU32>,
    bench_games: u32,
    nn_config: Config,
    baseline_config: Config,
    baseline_eval: RolloutEvaluator,
    nn_wins: Arc<AtomicU32>,
    nn_losses: Arc<AtomicU32>,
    draws: Arc<AtomicU32>,
    pb: Arc<indicatif::ProgressBar>,
    new_state: Arc<dyn Fn(&mut fastrand::Rng) -> G + Send + Sync>,
    seed: u64,
) {
    let mut rng = fastrand::Rng::with_seed(seed);

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
            &nn_config,
            &baseline_config,
            &baseline_eval,
            &request_tx,
            &mut rng,
        )
        .await;

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

/// Play a single benchmark game. NN turns use async eval via batcher;
/// baseline turns use run_to_completion locally.
async fn play_bench_game<G: Game>(
    state: &mut G,
    nn_player: Player,
    nn_config: &Config,
    baseline_config: &Config,
    baseline_eval: &RolloutEvaluator,
    request_tx: &EvalSender<G>,
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
                                let mut receivers = Vec::with_capacity(pendings.len());
                                for pending in pendings {
                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    request_tx
                                        .send(EvalRequest {
                                            pending,
                                            response_tx: tx,
                                        })
                                        .await
                                        .unwrap();
                                    receivers.push(rx);
                                }
                                let mut evals = Vec::with_capacity(receivers.len());
                                for rx in receivers {
                                    evals.push(rx.await.unwrap());
                                }
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

/// Play benchmark games: NN evaluator vs RolloutEvaluator, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
pub(super) fn run_benchmark<G: Game + 'static, Ev: Evaluator<G> + 'static>(
    evaluator: Ev,
    config: &TrainConfig,
    rng: &mut fastrand::Rng,
    new_state: &Arc<dyn Fn(&mut fastrand::Rng) -> G + Send + Sync>,
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

    let nn_wins = Arc::new(AtomicU32::new(0));
    let nn_losses = Arc::new(AtomicU32::new(0));
    let draws = Arc::new(AtomicU32::new(0));
    let game_counter = Arc::new(AtomicU32::new(0));

    let pb = indicatif::ProgressBar::new(config.bench_games as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{bar:40.cyan/dim} {pos}/{len}  {msg}  [{elapsed_precise} elapsed, ETA {eta_precise}]",
        )
        .unwrap(),
    );
    pb.set_message("bench W:0 L:0 D:0".to_string());
    let pb = Arc::new(pb);

    let concurrent_games = config.concurrent_games.min(config.bench_games as usize);
    let max_batch_size = config.max_batch_size;

    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(2 * max_batch_size);

    // Seed batcher rng before spawning
    let batcher_seed = rng.u64(..);
    let stats = Arc::new(BatcherStats::new());
    let stats_ref = stats.clone();

    // Batcher on a plain std::thread (owns evaluator)
    let batcher_handle = std::thread::spawn(move || {
        let mut batcher_rng = fastrand::Rng::with_seed(batcher_seed);
        batcher_loop(
            &mut request_rx,
            &evaluator,
            &mut batcher_rng,
            max_batch_size,
            &stats_ref,
        );
    });

    // Tokio runtime for async bench tasks
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let mut join_set = tokio::task::JoinSet::new();

        for _ in 0..concurrent_games {
            let seed = rng.u64(..);
            join_set.spawn(bench_game_task::<G>(
                request_tx.clone(),
                game_counter.clone(),
                config.bench_games,
                nn_config.clone(),
                baseline_config.clone(),
                baseline_eval.clone(),
                nn_wins.clone(),
                nn_losses.clone(),
                draws.clone(),
                pb.clone(),
                new_state.clone(),
                seed,
            ));
        }

        // Drop our sender so batcher sees disconnect when all tasks finish
        drop(request_tx);

        while let Some(result) = join_set.join_next().await {
            result.unwrap();
        }
    });

    batcher_handle.join().unwrap();
    pb.finish();

    (
        nn_wins.load(Relaxed),
        nn_losses.load(Relaxed),
        draws.load(Relaxed),
    )
}
