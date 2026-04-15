use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use indicatif::ProgressStyle;
use tracing::info;
use tracing_indicatif::span_ext::IndicatifSpanExt;

use crate::eval::{Evaluator, Evaluators};
use crate::game::{Game, Status};
use crate::game_log::GameLog;
use crate::mcts::{Config, Search, SearchResult, Select};

/// Parsed tournament settings.
pub struct TournamentOptions {
    pub num_games: u32,
    pub configs: [Config; 2],
    pub log_dir: Option<PathBuf>,
    pub eval_names: [String; 2],
    /// Maximum actions (including chance) before a game is declared a draw.
    /// `0` disables the cap.
    pub max_actions: u32,
}

impl TournamentOptions {
    /// Run a full tournament: print banner, play games, print results, save logs.
    pub fn run<G: Game + std::fmt::Display>(
        &self,
        new_game: impl Fn(u64) -> G + Sync,
        registry: &Evaluators<G>,
    ) -> Vec<GameLog> {
        let mut rng = fastrand::Rng::new();

        let evaluators: [&(dyn Evaluator<G> + Sync); 2] = [
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
            self.max_actions,
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
    counters: &TournamentCounters,
) -> SearchResult {
    loop {
        match search.select(rng) {
            Select::Eval(leaf, ref state) => {
                counters.evals.fetch_add(1, Ordering::Relaxed);
                let eval = evaluator.evaluate(state, rng);
                search.backup(leaf, eval);
            }
            Select::Terminal(leaf, wdl) => {
                search.backup_terminal(leaf, wdl);
            }
            Select::Done => {
                let result = search.result();
                counters
                    .depth_sum
                    .fetch_add(result.pv_depth as u64, Ordering::Relaxed);
                counters
                    .depth_max
                    .fetch_max(result.max_depth, Ordering::Relaxed);
                counters.depth_count.fetch_add(1, Ordering::Relaxed);
                return result;
            }
        }
    }
}

/// Play a single match between two MCTS bots.
///
/// Returns the terminal reward from P1's perspective and a log of every action
/// applied (both chance outcomes and player decisions).
/// When `swap` is true, seat assignments are reversed:
/// the game's P1 uses `configs[1]` and vice versa.
/// Shared counters for tournament-wide statistics.
pub struct TournamentCounters {
    pub evals: AtomicU64,
    pub depth_sum: AtomicU64,
    pub depth_max: AtomicU32,
    pub depth_count: AtomicU32,
}

pub fn play_match<G: Game>(
    game: &G,
    evaluators: &[&(dyn Evaluator<G> + Sync); 2],
    configs: &[Config; 2],
    swap: bool,
    max_actions: u32,
    rng: &mut fastrand::Rng,
    counters: &TournamentCounters,
) -> (f32, Vec<usize>) {
    let mut actions = Vec::new();
    let mut action_buf = Vec::new();

    // Persistent search trees — one per seat. Reused across actions so
    // subtrees explored in earlier searches carry over.
    let mut searches: [Option<Search<G>>; 2] = [
        if configs[0].num_simulations > 0 {
            Some(Search::new(game.clone(), configs[0].clone()))
        } else {
            None
        },
        if configs[1].num_simulations > 0 {
            Some(Search::new(game.clone(), configs[1].clone()))
        } else {
            None
        },
    ];

    // Reference search index — whichever seat has a search tree, use it for
    // state queries. Falls back to a standalone state if neither uses search.
    let ref_idx = if searches[0].is_some() { 0 } else { 1 };

    loop {
        // Get state info without holding a borrow across the mutable section.
        let (status, chance) = {
            let state = searches[ref_idx].as_ref().unwrap().state();
            let status = state.status();
            let chance = if matches!(status, Status::Chance) {
                state.sample_chance(rng)
            } else {
                None
            };
            (status, chance)
        };

        match status {
            Status::Terminal(reward) => return (reward, actions),
            Status::Decision(_) | Status::Chance => {}
        };

        // Action cap: declare a draw once the game exceeds max_actions.
        if max_actions > 0 && actions.len() as u32 >= max_actions {
            return (0.0, actions);
        }

        if let Some(action) = chance {
            actions.push(action);
            for search in searches.iter_mut().flatten() {
                search.apply_action(action);
            }
        } else if let Status::Decision(sign) = status {
            // sign-to-index: 1.0 → 0, -1.0 → 1
            let idx = ((1.0 - sign) / 2.0) as usize;
            let seat = idx ^ (swap as usize);
            let eval = evaluators[seat];

            let action = if let Some(search) = &mut searches[seat] {
                let result = run_to_completion(search, eval, rng, counters);
                result.selected_action
            } else {
                let state = searches[ref_idx].as_ref().unwrap().state();
                let eval_result = eval.evaluate(state, rng);
                action_buf.clear();
                state.legal_actions(&mut action_buf);
                crate::utils::sample_policy(&eval_result.policy_logits, &action_buf, rng)
            };

            actions.push(action);
            for search in searches.iter_mut().flatten() {
                search.apply_action(action);
            }
        }
    }
}

/// Run a tournament of `num_games` matches, alternating sides.
///
/// `new_game` is a factory that creates a fresh game state from a seed.
/// Even-numbered games use the original seat assignment;
/// odd-numbered games swap which config plays as P1.
/// Games run in parallel across available CPU cores.
pub fn tournament<G: Game + std::fmt::Display>(
    new_game: impl Fn(u64) -> G + Sync,
    evaluators: &[&(dyn Evaluator<G> + Sync); 2],
    configs: &[Config; 2],
    num_games: u32,
    max_actions: u32,
    rng: &mut fastrand::Rng,
) -> Vec<GameLog> {
    let n = num_games as usize;

    // Pre-generate all seeds for deterministic results regardless of thread count.
    let seeds: Vec<u64> = (0..n).map(|_| rng.u64(..)).collect();

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

    let start = std::time::Instant::now();

    // Shared state for parallel workers.
    let next_game = AtomicUsize::new(0);
    let wins = [AtomicUsize::new(0), AtomicUsize::new(0)];
    let draw_count = AtomicUsize::new(0);
    let counters = TournamentCounters {
        evals: AtomicU64::new(0),
        depth_sum: AtomicU64::new(0),
        depth_max: AtomicU32::new(0),
        depth_count: AtomicU32::new(0),
    };
    let results: Mutex<Vec<Option<GameLog>>> = Mutex::new((0..n).map(|_| None).collect());

    let num_workers = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    let done = AtomicUsize::new(0);

    std::thread::scope(|s| {
        // Ticker thread: refresh progress bar message every second.
        s.spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if done.load(Ordering::Relaxed) >= n {
                    break;
                }
                let w0 = wins[0].load(Ordering::Relaxed);
                let w1 = wins[1].load(Ordering::Relaxed);
                let d = draw_count.load(Ordering::Relaxed);
                let evals = counters.evals.load(Ordering::Relaxed);
                let secs = start.elapsed().as_secs_f64();
                let eps = if secs > 0.0 { evals as f64 / secs } else { 0.0 };
                span.pb_set_message(&format!("{w0}-{w1}-{d} | {eps:.0} evals/s"));
            }
        });

        for _ in 0..num_workers {
            s.spawn(|| {
                loop {
                    let i = next_game.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let swap = i % 2 == 1;
                    let seed = seeds[i];
                    let mut thread_rng = fastrand::Rng::with_seed(seed);
                    let game = new_game(seed);
                    let (reward, actions) = play_match(
                        &game,
                        evaluators,
                        configs,
                        swap,
                        max_actions,
                        &mut thread_rng,
                        &counters,
                    );

                    results.lock().unwrap()[i] = Some(GameLog {
                        initial_state: game.to_string(),
                        actions,
                    });

                    // Update progress counters.
                    let seat0_reward = if swap { -reward } else { reward };
                    if seat0_reward > 0.0 {
                        wins[0].fetch_add(1, Ordering::Relaxed);
                    } else if seat0_reward < 0.0 {
                        wins[1].fetch_add(1, Ordering::Relaxed);
                    } else {
                        draw_count.fetch_add(1, Ordering::Relaxed);
                    }

                    let w0 = wins[0].load(Ordering::Relaxed);
                    let w1 = wins[1].load(Ordering::Relaxed);
                    let d = draw_count.load(Ordering::Relaxed);
                    span.pb_set_message(&format!("{w0}-{w1}-{d}"));
                    span.pb_inc(1);
                    done.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });

    let game_logs: Vec<GameLog> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|opt| opt.unwrap())
        .collect();

    let w0 = wins[0].load(Ordering::Relaxed);
    let w1 = wins[1].load(Ordering::Relaxed);
    let d = draw_count.load(Ordering::Relaxed);
    let total = num_games;
    let elapsed = crate::utils::HumanDuration(span.pb_elapsed());
    drop(span);

    let dc = counters.depth_count.load(Ordering::Relaxed);
    let pv_depth = if dc > 0 {
        counters.depth_sum.load(Ordering::Relaxed) as f64 / dc as f64
    } else {
        0.0
    };
    let max_depth = counters.depth_max.load(Ordering::Relaxed);

    info!(
        "W {}/{} ({:.1}%) | L {}/{} ({:.1}%) | D {}/{} ({:.1}%) | depth {:.1}/{} | {elapsed}",
        w0,
        total,
        w0 as f32 / total as f32 * 100.0,
        w1,
        total,
        w1 as f32 / total as f32 * 100.0,
        d,
        total,
        d as f32 / total as f32 * 100.0,
        pv_depth,
        max_depth,
    );

    game_logs
}

// ---------------------------------------------------------------------------
// BatchedEvaluator — transparent batching wrapper for nn evaluators
// ---------------------------------------------------------------------------

/// Wraps an nn evaluator + encoder pair, routing inference through a shared
/// batcher pipeline via blocking channel sends/receives. Implements
/// `Evaluator<G>` so it's transparent to `play_match` / `tournament`.
#[cfg(feature = "nn")]
pub struct BatchedEvaluator<G: Game> {
    evaluator: std::sync::Arc<dyn Evaluator<G> + Sync>,
    encoder: std::sync::Arc<dyn crate::nn::StateEncoder<G>>,
    request_tx: tokio::sync::mpsc::Sender<crate::train::inference::InferRequest>,
}

#[cfg(feature = "nn")]
impl<G: Game> BatchedEvaluator<G> {
    pub fn new(
        evaluator: std::sync::Arc<dyn Evaluator<G> + Sync>,
        encoder: std::sync::Arc<dyn crate::nn::StateEncoder<G>>,
        request_tx: tokio::sync::mpsc::Sender<crate::train::inference::InferRequest>,
    ) -> Self {
        Self {
            evaluator,
            encoder,
            request_tx,
        }
    }
}

#[cfg(feature = "nn")]
impl<G: Game> Evaluator<G> for BatchedEvaluator<G> {
    fn evaluate(&self, state: &G, rng: &mut fastrand::Rng) -> crate::eval::Evaluation {
        self.evaluate_batch(&[state], rng).pop().unwrap()
    }

    fn evaluate_batch(
        &self,
        states: &[&G],
        _rng: &mut fastrand::Rng,
    ) -> Vec<crate::eval::Evaluation> {
        use crate::eval::{Evaluation, Wdl};
        use crate::train::inference::{InferRequest, InferResponse};

        let feature_size = self.encoder.feature_size();
        let num_actions = G::NUM_ACTIONS;

        let mut signs = Vec::with_capacity(states.len());
        let mut nn_indices = Vec::new();
        let mut results: Vec<Option<Evaluation>> = (0..states.len()).map(|_| None).collect();

        for (i, state) in states.iter().enumerate() {
            match state.status() {
                Status::Terminal(reward) => {
                    results[i] = Some(Evaluation::uniform(num_actions, reward));
                }
                Status::Chance => {
                    results[i] = Some(Evaluation::uniform(num_actions, 0.0));
                }
                Status::Decision(sign) => {
                    signs.push(sign);
                    nn_indices.push(i);
                }
            }
        }

        if nn_indices.is_empty() {
            return results.into_iter().map(|r| r.unwrap()).collect();
        }

        // Encode decision states
        let n = nn_indices.len();
        let mut flat_features = Vec::with_capacity(n * feature_size);
        let mut buf = Vec::with_capacity(feature_size);
        for &i in &nn_indices {
            buf.clear();
            self.encoder.encode(states[i], &mut buf);
            flat_features.extend_from_slice(&buf);
        }

        // Send to batcher and wait for response
        let (tx, rx) = tokio::sync::oneshot::channel::<InferResponse>();
        self.request_tx
            .blocking_send(InferRequest {
                flat_features,
                batch_size: n,
                response_tx: tx,
            })
            .expect("batcher channel closed");

        let resp = rx.blocking_recv().expect("batcher response channel closed");

        // Unpack response with sign correction
        for (j, &i) in nn_indices.iter().enumerate() {
            let logits_start = j * num_actions;
            let logits = resp.flat_policy_logits[logits_start..logits_start + num_actions].to_vec();
            let wdl_start = j * 3;
            let wdl_raw = &resp.flat_wdl[wdl_start..wdl_start + 3];
            // Sign-flip for P1 perspective (swap W/L when sign < 0)
            let wdl_cp = Wdl {
                w: wdl_raw[0],
                d: wdl_raw[1],
                l: wdl_raw[2],
            };
            let wdl = if signs[j] > 0.0 {
                wdl_cp
            } else {
                wdl_cp.flip()
            };
            results[i] = Some(Evaluation {
                policy_logits: logits,
                wdl,
            });
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }

    fn infer_features(
        &self,
        features: Vec<f32>,
        batch_size: usize,
        feature_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.evaluator
            .infer_features(features, batch_size, feature_size)
    }
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
