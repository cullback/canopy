//! Single-game simulation: plays one self-play game or replays one for reanalyze.
//!
//! Pure game logic — no channels, batching, or GPU knowledge. Leaf evaluation
//! is delegated to a caller-provided async `infer` closure. Handles chance
//! nodes, forced moves, MCTS search, playout cap randomization, training
//! sample collection, and post-game processing (surprise weights, short-term
//! value EMA targets).

use std::future::Future;

use crate::eval::{Evaluation, flip_wdl};
use crate::game::{Game, Status};
use crate::mcts::{Search, Step};
use crate::nn::StateEncoder;

use super::Sample;

/// Per-game config knobs derived from [`super::TrainConfig`].
pub(super) struct ActorConfig {
    pub explore_moves: u32,
    pub playout_cap_full_prob: f32,
    pub playout_cap_fast_sims_base: u32,
    pub num_aux_targets: usize,
    pub aux_value_horizons: Vec<u32>,
    pub surprise_weight_fraction: f32,
}

/// Result of a single self-play game.
pub(super) struct GameResult {
    pub samples: Vec<Sample>,
    pub reward: f32,
    pub actions: Vec<usize>,
    pub seed: u64,
}

/// Play one full self-play game, returning collected training samples.
///
/// Pure game logic: chance nodes → forced moves → MCTS search → sample → apply.
/// Leaf evaluation is delegated to the caller-provided `infer` closure, keeping
/// this function free of any channel, batching, or GPU knowledge.
pub(super) async fn play_game<G, F, Fut>(
    search: &mut Search<G>,
    effective_sims: u32,
    fast_sims: u32,
    actor_config: &ActorConfig,
    encoder: &dyn StateEncoder<G>,
    infer: F,
    rng: &mut fastrand::Rng,
    game_seed: u64,
) -> GameResult
where
    G: Game,
    F: Fn(Vec<f32>, usize) -> Fut,
    Fut: Future<Output = (Vec<f32>, Vec<f32>)>,
{
    let feature_size = encoder.feature_size();
    let mut samples: Vec<Sample> = Vec::new();
    let mut turn_count: u32 = 0;
    let mut last_sign: Option<f32> = None;
    let mut actions_buf = Vec::new();
    let mut encode_buf = Vec::with_capacity(feature_size);
    let mut all_actions: Vec<usize> = Vec::new();

    loop {
        // Resolve chance nodes
        if let Some(action) = search.state().sample_chance(rng) {
            all_actions.push(action);
            search.apply_action(action);
            continue;
        }

        // Terminal check
        if let Status::Terminal(reward) = search.state().status() {
            // Compute short-term value EMA targets before z *= reward
            if !actor_config.aux_value_horizons.is_empty() {
                compute_short_term_values(&mut samples, &actor_config.aux_value_horizons);
            }

            // Normalize surprise weights per-game
            if actor_config.surprise_weight_fraction > 0.0 {
                normalize_surprise_weights(&mut samples, actor_config.surprise_weight_fraction);
            }

            for s in &mut samples {
                s.z *= reward;
                s.game_length = turn_count;
            }
            return GameResult {
                samples,
                reward,
                actions: all_actions,
                seed: game_seed,
            };
        }

        let sign = search.state().current_sign();

        // Track turn count via player changes
        if last_sign != Some(sign) {
            turn_count += 1;
            last_sign = Some(sign);
        }

        // Skip forced moves (single legal action)
        actions_buf.clear();
        search.state().legal_actions(&mut actions_buf);
        if actions_buf.len() == 1 {
            all_actions.push(actions_buf[0]);
            search.apply_action(actions_buf[0]);
            continue;
        }

        // Playout cap randomization
        let is_full_search = rng.f32() < actor_config.playout_cap_full_prob;
        search.set_num_simulations(if is_full_search {
            effective_sims
        } else {
            fast_sims
        });

        // Encode features before search (for training sample)
        let mut features_buf = Vec::with_capacity(feature_size);
        encoder.encode(search.state(), &mut features_buf);

        // MCTS search with eval requests sent via infer closure
        let mut evals: Vec<Evaluation> = vec![];
        let result = loop {
            match search.step(&evals, rng) {
                Step::NeedsEval(states) => {
                    let batch_size = states.len();
                    let mut signs = Vec::with_capacity(batch_size);
                    let mut flat_features = Vec::with_capacity(batch_size * feature_size);
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
                    let (flat_logits, flat_wdl) = infer(flat_features, batch_size).await;
                    evals.clear();
                    let num_actions = G::NUM_ACTIONS;
                    for (i, &sign) in signs.iter().enumerate() {
                        let logits_start = i * num_actions;
                        let policy_logits =
                            flat_logits[logits_start..logits_start + num_actions].to_vec();
                        let wdl_start = i * 3;
                        let wdl_raw = &flat_wdl[wdl_start..wdl_start + 3];
                        // Sign-flip for P1 perspective (swap W/L when sign < 0)
                        let wdl_cp = [wdl_raw[0], wdl_raw[1], wdl_raw[2]];
                        let wdl = if sign > 0.0 { wdl_cp } else { flip_wdl(wdl_cp) };
                        debug_assert!(wdl.iter().all(|&x| x >= 0.0 && x <= 1.0));
                        debug_assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
                        evals.push(Evaluation { policy_logits, wdl });
                    }
                }
                Step::Done(result) => break result,
            }
        };

        // Create training sample
        // result.wdl is P1 perspective; flip to current player (swap W/L when sign < 0)
        let q_wdl = if sign > 0.0 {
            result.wdl
        } else {
            flip_wdl(result.wdl)
        };
        debug_assert!(q_wdl.iter().all(|&x| x >= 0.0 && x <= 1.0));
        debug_assert!((q_wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        let chosen = if turn_count <= actor_config.explore_moves {
            sample_from_policy(&result.policy, rng)
        } else {
            result.selected_action
        };

        let root_v = result.wdl[0] - result.wdl[2];
        let value_correction = (root_v - result.network_value).abs();
        let q_std = if result.children_q.len() >= 2 {
            let mean = result.children_q.iter().map(|&(_, q)| q).sum::<f32>()
                / result.children_q.len() as f32;
            let var = result
                .children_q
                .iter()
                .map(|&(_, q)| (q - mean).powi(2))
                .sum::<f32>()
                / result.children_q.len() as f32;
            var.sqrt()
        } else {
            0.0
        };
        let prior_agrees = result.prior_top1_action == result.selected_action;

        // Compute raw KL(target || prior) for policy surprise weighting
        let policy_surprise = if is_full_search {
            compute_kl(&result.policy, &result.prior_policy)
        } else {
            0.0
        };

        samples.push(Sample {
            features: features_buf.into_boxed_slice(),
            policy_target: result.policy.into_boxed_slice(),
            z: sign,
            q_wdl,
            full_search: is_full_search,
            move_number: turn_count,
            game_length: 0, // backfilled at game end
            network_value: result.network_value * sign,
            value_correction,
            q_std,
            prior_agrees,
            policy_surprise,
            surprise_weight: 1.0,
            aux_targets: vec![0.0; actor_config.num_aux_targets].into_boxed_slice(),
        });

        all_actions.push(chosen);
        search.apply_action(chosen);
    }
}

/// Reanalyze a previously played game with the current network.
///
/// Actions are predetermined (replayed from the original game). Always uses
/// full search budget. Chance nodes are detected via `chance_outcomes()`.
pub(super) async fn reanalyze_game<G, F, Fut>(
    search: &mut Search<G>,
    effective_sims: u32,
    encoder: &dyn StateEncoder<G>,
    infer: F,
    rng: &mut fastrand::Rng,
    actions: &[usize],
    reward: f32,
    num_aux_targets: usize,
    aux_value_horizons: &[u32],
    surprise_weight_fraction: f32,
) -> Vec<Sample>
where
    G: Game,
    F: Fn(Vec<f32>, usize) -> Fut,
    Fut: Future<Output = (Vec<f32>, Vec<f32>)>,
{
    let feature_size = encoder.feature_size();
    let mut samples: Vec<Sample> = Vec::new();
    let mut turn_count: u32 = 0;
    let mut last_sign: Option<f32> = None;
    let mut actions_buf = Vec::new();
    let mut chance_buf = Vec::new();
    let mut encode_buf = Vec::with_capacity(feature_size);
    let mut action_idx = 0;

    loop {
        if action_idx >= actions.len() {
            break;
        }

        // Resolve chance nodes (detected via chance_outcomes)
        chance_buf.clear();
        search.state().chance_outcomes(&mut chance_buf);
        if !chance_buf.is_empty() {
            let action = actions[action_idx];
            action_idx += 1;
            search.apply_action(action);
            continue;
        }

        // Terminal check
        if let Status::Terminal(_) = search.state().status() {
            break;
        }

        let sign = search.state().current_sign();

        // Track turn count via player changes
        if last_sign != Some(sign) {
            turn_count += 1;
            last_sign = Some(sign);
        }

        // Skip forced moves (single legal action)
        actions_buf.clear();
        search.state().legal_actions(&mut actions_buf);
        if actions_buf.len() == 1 {
            let action = actions[action_idx];
            action_idx += 1;
            search.apply_action(action);
            continue;
        }

        // Always full search for reanalyze
        search.set_num_simulations(effective_sims);

        // Encode features before search
        let mut features_buf = Vec::with_capacity(feature_size);
        encoder.encode(search.state(), &mut features_buf);

        // MCTS search
        let mut evals: Vec<Evaluation> = vec![];
        let result = loop {
            match search.step(&evals, rng) {
                Step::NeedsEval(states) => {
                    let batch_size = states.len();
                    let mut signs = Vec::with_capacity(batch_size);
                    let mut flat_features = Vec::with_capacity(batch_size * feature_size);
                    for pending_state in states {
                        let s = match pending_state.status() {
                            Status::Ongoing => pending_state.current_sign(),
                            Status::Terminal(_) => 1.0,
                        };
                        signs.push(s);
                        encode_buf.clear();
                        encoder.encode(pending_state, &mut encode_buf);
                        flat_features.extend_from_slice(&encode_buf);
                    }
                    let (flat_logits, flat_wdl) = infer(flat_features, batch_size).await;
                    evals.clear();
                    let num_actions = G::NUM_ACTIONS;
                    for (i, &sign) in signs.iter().enumerate() {
                        let logits_start = i * num_actions;
                        let policy_logits =
                            flat_logits[logits_start..logits_start + num_actions].to_vec();
                        let wdl_start = i * 3;
                        let wdl_raw = &flat_wdl[wdl_start..wdl_start + 3];
                        let wdl_cp = [wdl_raw[0], wdl_raw[1], wdl_raw[2]];
                        let wdl = if sign > 0.0 { wdl_cp } else { flip_wdl(wdl_cp) };
                        evals.push(Evaluation { policy_logits, wdl });
                    }
                }
                Step::Done(result) => break result,
            }
        };

        // Create training sample — always full_search: true
        let q_wdl = if sign > 0.0 {
            result.wdl
        } else {
            flip_wdl(result.wdl)
        };

        let root_v = result.wdl[0] - result.wdl[2];
        let value_correction = (root_v - result.network_value).abs();
        let q_std = if result.children_q.len() >= 2 {
            let mean = result.children_q.iter().map(|&(_, q)| q).sum::<f32>()
                / result.children_q.len() as f32;
            let var = result
                .children_q
                .iter()
                .map(|&(_, q)| (q - mean).powi(2))
                .sum::<f32>()
                / result.children_q.len() as f32;
            var.sqrt()
        } else {
            0.0
        };
        let prior_agrees = result.prior_top1_action == result.selected_action;
        let policy_surprise = compute_kl(&result.policy, &result.prior_policy);

        samples.push(Sample {
            features: features_buf.into_boxed_slice(),
            policy_target: result.policy.into_boxed_slice(),
            z: sign,
            q_wdl,
            full_search: true,
            move_number: turn_count,
            game_length: 0,
            network_value: result.network_value * sign,
            value_correction,
            q_std,
            prior_agrees,
            policy_surprise,
            surprise_weight: 1.0,
            aux_targets: vec![0.0; num_aux_targets].into_boxed_slice(),
        });

        // Apply the predetermined action
        let action = actions[action_idx];
        action_idx += 1;
        search.apply_action(action);
    }

    // Post-process: short-term values, surprise weights, z *= reward
    if !aux_value_horizons.is_empty() {
        compute_short_term_values(&mut samples, aux_value_horizons);
    }
    if surprise_weight_fraction > 0.0 {
        normalize_surprise_weights(&mut samples, surprise_weight_fraction);
    }
    for s in &mut samples {
        s.z *= reward;
        s.game_length = turn_count;
    }

    samples
}

/// Compute KL(target || prior) = Σ target[a] * ln(target[a] / prior[a]).
fn compute_kl(target: &[f32], prior: &[f32]) -> f32 {
    const EPSILON: f32 = 1e-8;
    let mut kl = 0.0f32;
    for (&t, &p) in target.iter().zip(prior.iter()) {
        if t > 0.0 {
            kl += t * (t / p.max(EPSILON)).ln();
        }
    }
    kl
}

/// Normalize surprise weights across a game's samples.
///
/// Full-search samples: w = (1-f)/N + f * kl / total_kl
/// Fast-search samples: w = (1-f)/N
fn normalize_surprise_weights(samples: &mut [Sample], fraction: f32) {
    let n_full = samples.iter().filter(|s| s.full_search).count();
    if n_full == 0 {
        return;
    }
    let total_kl: f32 = samples
        .iter()
        .filter(|s| s.full_search)
        .map(|s| s.policy_surprise)
        .sum();
    let n = n_full as f32;
    let uniform = (1.0 - fraction) / n;
    for s in samples.iter_mut() {
        if s.full_search {
            let surprise_part = if total_kl > 0.0 {
                fraction * s.policy_surprise / total_kl
            } else {
                fraction / n
            };
            s.surprise_weight = uniform + surprise_part;
        } else {
            s.surprise_weight = (1.0 - fraction) / n;
        }
    }
}

/// Compute EMA short-term value targets, backwards through the game.
///
/// For each horizon h: alpha = 1 - exp(-1/h)
/// Backwards: ema_p1 = alpha * q_p1[t] + (1-alpha) * ema_p1
/// Store in current-player perspective: aux_targets[h_idx] = ema_p1 * z
///
/// Called before z *= reward, so z still holds the player sign.
fn compute_short_term_values(samples: &mut [Sample], horizons: &[u32]) {
    for (h_idx, &h) in horizons.iter().enumerate() {
        let alpha = 1.0 - (-1.0 / h as f32).exp();
        let mut ema_p1 = 0.0f32;
        for i in (0..samples.len()).rev() {
            // Convert Q to P1 perspective: q_cp * sign (z is the player sign)
            let q_cp = samples[i].q_wdl[0] - samples[i].q_wdl[2];
            let q_p1 = q_cp * samples[i].z;
            ema_p1 = alpha * q_p1 + (1.0 - alpha) * ema_p1;
            // Store in current-player perspective
            samples[i].aux_targets[h_idx] = ema_p1 * samples[i].z;
        }
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
