use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::eval::{Evaluation, Evaluator, RolloutEvaluator};
use crate::game::{Game, Status};
use crate::mcts::{Config, Search, Step};
use crate::nn::StateEncoder;
use crate::player::Player;

use super::TrainConfig;
use super::self_play::{BatcherStats, InferRequest, InferSender, batcher_loop};

/// Async bench game task: plays games alternating NN (batched) and baseline (local) turns.
#[allow(clippy::too_many_arguments)]
async fn bench_game_task<G: Game, E: StateEncoder<G>>(
    request_tx: InferSender,
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

        let reward = play_bench_game::<G, E>(
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

/// Drive a search to completion using a local evaluator.
fn run_to_completion<G: Game>(
    search: &mut Search<G>,
    evaluator: &RolloutEvaluator,
    rng: &mut fastrand::Rng,
) -> crate::mcts::SearchResult {
    let mut evals = vec![];
    loop {
        match search.supply(&evals, rng) {
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
async fn play_bench_game<G: Game, E: StateEncoder<G>>(
    state: &mut G,
    nn_player: Player,
    nn_config: &Config,
    baseline_config: &Config,
    baseline_eval: &RolloutEvaluator,
    request_tx: &InferSender,
    rng: &mut fastrand::Rng,
) -> f32 {
    let mut encode_buf = Vec::with_capacity(E::FEATURE_SIZE);

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
                    let mut search = Search::new(state.clone(), nn_config.clone());
                    let mut evals: Vec<Evaluation> = vec![];
                    let result = loop {
                        match search.supply(&evals, rng) {
                            Step::NeedsEval(states) => {
                                let mut receivers = Vec::with_capacity(states.len());
                                for pending_state in states {
                                    let sign = match pending_state.status() {
                                        Status::Ongoing(p) => p.sign(),
                                        Status::Terminal(_) => 1.0,
                                    };
                                    encode_buf.clear();
                                    E::encode(pending_state, &mut encode_buf);
                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    request_tx
                                        .send(InferRequest {
                                            features: encode_buf.clone(),
                                            response_tx: tx,
                                        })
                                        .await
                                        .unwrap();
                                    receivers.push((rx, sign));
                                }
                                evals.clear();
                                for (rx, sign) in receivers {
                                    let resp = rx.await.unwrap();
                                    evals.push(Evaluation {
                                        policy_logits: resp.policy_logits,
                                        value: resp.value * sign,
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

/// Play benchmark games: NN evaluator vs RolloutEvaluator, alternating seats.
/// Returns (nn_wins, nn_losses, draws).
pub(super) fn run_benchmark<G, E, Ev>(
    evaluator: Ev,
    config: &TrainConfig,
    rng: &mut fastrand::Rng,
    new_state: &Arc<dyn Fn(&mut fastrand::Rng) -> G + Send + Sync>,
) -> (u32, u32, u32)
where
    G: Game + 'static,
    E: StateEncoder<G> + 'static,
    Ev: Evaluator<G> + 'static,
{
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
    let num_actions = G::NUM_ACTIONS;
    let feature_size = E::FEATURE_SIZE;

    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(2 * max_batch_size);

    let stats = Arc::new(BatcherStats::new());
    let stats_ref = stats.clone();

    // Batcher on a plain std::thread — receives pre-encoded features, runs GPU inference
    let batcher_handle = std::thread::spawn(move || {
        batcher_loop(
            &mut request_rx,
            |features, batch_size| evaluator.infer_features(features, batch_size, feature_size),
            num_actions,
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
            join_set.spawn(bench_game_task::<G, E>(
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
