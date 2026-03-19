use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::eval::{Evaluation, Evaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, Step};
use crate::nn::StateEncoder;

use super::TrainConfig;
use super::inference::{BatcherStats, InferRequest, InferSender, batcher_loop, gpu_worker_loop};

/// Async bench game task: plays games alternating NN (batched) and baseline (local) turns.
#[allow(clippy::too_many_arguments)]
async fn bench_game_task<G: Game>(
    request_tx: InferSender,
    encoder: Arc<dyn StateEncoder<G>>,
    game_counter: Arc<AtomicU32>,
    bench_games: u32,
    nn_config: Config,
    baseline_config: Config,
    baseline_eval: Arc<dyn Evaluator<G> + Sync>,
    nn_wins: Arc<AtomicU32>,
    nn_losses: Arc<AtomicU32>,
    draws: Arc<AtomicU32>,
    pb: Arc<indicatif::ProgressBar>,
    new_state: Arc<dyn Fn(u64) -> G + Send + Sync>,
    seed: u64,
) {
    let mut rng = fastrand::Rng::with_seed(seed);

    loop {
        // Claim a game via atomic increment
        let i = game_counter.fetch_add(1, Relaxed);
        if i >= bench_games {
            break;
        }

        let nn_sign = if i % 2 == 0 { 1.0f32 } else { -1.0 };
        let mut state = new_state(rng.u64(..));

        let reward = play_bench_game(
            &mut state,
            nn_sign,
            &nn_config,
            &baseline_config,
            &*baseline_eval,
            &*encoder,
            &request_tx,
            &mut rng,
        )
        .await;

        let nn_reward = reward * nn_sign;
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

/// Drive a search to completion using a local evaluator.
fn run_to_completion<G: Game>(
    search: &mut Search<G>,
    evaluator: &(dyn Evaluator<G> + Sync),
    rng: &mut fastrand::Rng,
) -> crate::mcts::SearchResult {
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

/// Play a single benchmark game. NN turns use async eval via batcher;
/// baseline turns use run_to_completion locally.
async fn play_bench_game<G: Game>(
    state: &mut G,
    nn_sign: f32,
    nn_config: &Config,
    baseline_config: &Config,
    baseline_eval: &(dyn Evaluator<G> + Sync),
    encoder: &dyn StateEncoder<G>,
    request_tx: &InferSender,
    rng: &mut fastrand::Rng,
) -> f32 {
    let feature_size = encoder.feature_size();
    let mut encode_buf = Vec::with_capacity(feature_size);

    loop {
        if let Some(action) = state.sample_chance(rng) {
            state.apply_action(action);
            continue;
        }
        match state.status() {
            Status::Terminal(reward) => return reward,
            Status::Ongoing => {
                let action = if state.current_sign() == nn_sign {
                    // NN player: batched eval via batcher
                    let mut search = Search::new(state.clone(), nn_config.clone());
                    let mut evals: Vec<Evaluation> = vec![];
                    let result = loop {
                        match search.step(&evals, rng) {
                            Step::NeedsEval(states) => {
                                let batch_size = states.len();
                                let num_actions = G::NUM_ACTIONS;
                                let mut signs = Vec::with_capacity(batch_size);
                                let mut flat_features =
                                    Vec::with_capacity(batch_size * feature_size);
                                for pending_state in states {
                                    let sign = match pending_state.status() {
                                        Status::Ongoing => pending_state.current_sign(),
                                        Status::Terminal(_) => 1.0,
                                    };
                                    signs.push(sign);
                                    encode_buf.clear();
                                    encoder.encode(pending_state, &mut encode_buf);
                                    flat_features.extend_from_slice(&encode_buf);
                                }
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                request_tx
                                    .send(InferRequest {
                                        flat_features,
                                        batch_size,
                                        response_tx: tx,
                                    })
                                    .await
                                    .unwrap();
                                let resp = rx.await.unwrap();
                                evals.clear();
                                for (i, &sign) in signs.iter().enumerate() {
                                    let start = i * num_actions;
                                    evals.push(Evaluation {
                                        policy_logits: resp.flat_policy_logits
                                            [start..start + num_actions]
                                            .to_vec(),
                                        value: resp.values[i] * sign,
                                    });
                                }
                            }
                            Step::Done(result) => break result,
                        }
                    };
                    result.selected_action
                } else {
                    // Baseline player: local rollout eval
                    run_to_completion(
                        &mut Search::new(state.clone(), baseline_config.clone()),
                        baseline_eval,
                        rng,
                    )
                    .selected_action
                };
                state.apply_action(action);
            }
        }
    }
}

/// Play benchmark games: NN evaluator vs baseline, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
pub(super) fn run_benchmark<G>(
    evaluators: Vec<Arc<dyn Evaluator<G> + Sync>>,
    encoder: Arc<dyn StateEncoder<G>>,
    config: &TrainConfig,
    rng: &mut fastrand::Rng,
    new_state: &Arc<dyn Fn(u64) -> G + Send + Sync>,
    baseline: Arc<dyn Evaluator<G> + Sync>,
) -> (u32, u32, u32)
where
    G: Game + 'static,
{
    let nn_config = Config {
        num_simulations: config.mcts_sims,
        leaf_batch_size: config.leaf_batch_size,
        ..Default::default()
    };
    let baseline_config = Config {
        num_simulations: config.bench_sims,
        ..Default::default()
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
    let num_actions = G::NUM_ACTIONS;

    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(2 * max_batch_size);

    let stats = Arc::new(BatcherStats::new());
    let stats_ref = stats.clone();

    // Batcher → GPU worker channel (crossbeam SPMC)
    let (work_tx, work_rx) = crossbeam_channel::bounded(2);

    let mut worker_handles = Vec::new();
    for evaluator in evaluators {
        let rx = work_rx.clone();
        worker_handles.push(std::thread::spawn(move || {
            gpu_worker_loop(
                rx,
                |features, batch_size, feature_size| {
                    evaluator.infer_features(features, batch_size, feature_size)
                },
                num_actions,
            );
        }));
    }
    drop(work_rx);

    let batcher_handle = std::thread::spawn(move || {
        batcher_loop(&mut request_rx, &work_tx, max_batch_size, &stats_ref);
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
            join_set.spawn(bench_game_task(
                request_tx.clone(),
                encoder.clone(),
                game_counter.clone(),
                config.bench_games,
                nn_config.clone(),
                baseline_config.clone(),
                baseline.clone(),
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
    for h in worker_handles {
        h.join().unwrap();
    }
    pb.finish();

    (
        nn_wins.load(Relaxed),
        nn_losses.load(Relaxed),
        draws.load(Relaxed),
    )
}
