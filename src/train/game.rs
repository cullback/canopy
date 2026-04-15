//! Single-game simulation: plays one self-play game.
//!
//! Pure game logic — no channels, batching, or GPU knowledge. Leaf evaluation
//! is delegated to a caller-provided async `infer` closure. Handles chance
//! nodes, forced actions, MCTS search, playout cap randomization, training
//! sample collection, and post-game processing (short-term value EMA targets).

use std::future::Future;

use crate::eval::{Evaluation, Wdl};
use crate::game::{Game, Status};
use crate::mcts::{Search, SearchResult, Select};
use crate::nn::StateEncoder;

use super::Sample;
use super::replay_buffer::GameRecord;

/// Per-game config knobs derived from [`super::TrainConfig`].
pub(super) struct ActorConfig {
    pub max_actions: u32,
    pub explore_actions: u32,
    pub playout_cap_full_prob: f32,
    pub playout_cap_fast_sims_base: u32,
    pub num_aux_targets: usize,
    pub aux_value_horizons: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Run MCTS search from the current position, returning the search result.
///
/// Handles the step → encode → infer → parse loop.
async fn run_search<G, F, Fut>(
    search: &mut Search<G>,
    encoder: &dyn StateEncoder<G>,
    encode_buf: &mut Vec<f32>,
    infer: &F,
    rng: &mut fastrand::Rng,
) -> SearchResult
where
    G: Game,
    F: Fn(Vec<f32>, usize) -> Fut,
    Fut: Future<Output = (Vec<f32>, Vec<f32>)>,
{
    let num_actions = G::NUM_ACTIONS;
    loop {
        match search.select(rng) {
            Select::Eval(leaf, ref state) => {
                let sign = match state.status() {
                    Status::Decision(s) => s,
                    Status::Chance => 1.0,
                    Status::Terminal(_) => 1.0,
                };
                encode_buf.clear();
                encoder.encode(state, encode_buf);
                let flat_features = encode_buf.clone();
                let (flat_logits, flat_wdl) = infer(flat_features, 1).await;
                let policy_logits = flat_logits[..num_actions].to_vec();
                let wdl_cp = Wdl {
                    w: flat_wdl[0],
                    d: flat_wdl[1],
                    l: flat_wdl[2],
                };
                let wdl = if sign > 0.0 { wdl_cp } else { wdl_cp.flip() };
                search.backup(leaf, Evaluation { policy_logits, wdl });
            }
            Select::Terminal(leaf, wdl) => {
                search.backup_terminal(leaf, wdl);
            }
            Select::Done => return search.result(),
        }
    }
}

/// Create a training sample from a search result.
fn make_sample(
    result: SearchResult,
    features: Vec<f32>,
    sign: f32,
    is_full_search: bool,
    action_count: u32,
    num_aux_targets: usize,
) -> Sample {
    let q_wdl = if sign > 0.0 {
        result.wdl
    } else {
        result.wdl.flip()
    };

    let root_v = result.wdl.q();
    let value_correction = (root_v - result.network_value).abs();
    let q_std = if result.children_q.len() >= 2 {
        let mean =
            result.children_q.iter().map(|&(_, q)| q).sum::<f32>() / result.children_q.len() as f32;
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

    Sample {
        features: features.into_boxed_slice(),
        policy_target: result.policy.into_boxed_slice(),
        z: sign,
        q_wdl,
        full_search: is_full_search,
        action_number: action_count,
        game_length: 0, // backfilled at game end
        network_value: result.network_value * sign,
        value_correction,
        q_std,
        prior_agrees,
        aux_targets: vec![0.0; num_aux_targets].into_boxed_slice(),
        search_pv_depth: result.pv_depth as f32,
        search_depth_max: result.max_depth,
    }
}

/// Post-process samples after game completion: short-term value targets and z *= reward.
fn finalize_samples(
    samples: &mut [Sample],
    reward: f32,
    action_count: u32,
    aux_value_horizons: &[u32],
) {
    if !aux_value_horizons.is_empty() {
        compute_short_term_values(samples, aux_value_horizons);
    }
    for s in samples {
        s.z *= reward;
        s.game_length = action_count;
    }
}

// ---------------------------------------------------------------------------
// Self-play
// ---------------------------------------------------------------------------

/// Play one full self-play game, returning collected training samples.
///
/// Pure game logic: chance nodes → forced actions → MCTS search → sample → apply.
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
) -> GameRecord
where
    G: Game + std::fmt::Display,
    F: Fn(Vec<f32>, usize) -> Fut,
    Fut: Future<Output = (Vec<f32>, Vec<f32>)>,
{
    let feature_size = encoder.feature_size();
    let initial_state = search.state().to_string();
    let mut samples: Vec<Sample> = Vec::new();
    let mut action_log: Vec<usize> = Vec::new();
    let mut actions_buf = Vec::new();
    let mut encode_buf = Vec::with_capacity(feature_size);
    let mut action_count: u32 = 0;
    let max = actor_config.max_actions;

    loop {
        // Advance through chance nodes and forced actions
        loop {
            if max > 0 && action_count >= max {
                break;
            }
            if let Some(action) = search.state().sample_chance(rng) {
                action_count += 1;
                action_log.push(action);
                search.apply_action(action);
            } else {
                actions_buf.clear();
                search.state().legal_actions(&mut actions_buf);
                if actions_buf.len() == 1 {
                    action_count += 1;
                    action_log.push(actions_buf[0]);
                    search.apply_action(actions_buf[0]);
                } else {
                    break;
                }
            }
        }

        // Terminal / max-actions check
        let reward = match search.state().status() {
            Status::Terminal(r) => Some(r),
            _ if max > 0 && action_count >= max => Some(0.0),
            _ => None,
        };
        if let Some(reward) = reward {
            finalize_samples(
                &mut samples,
                reward,
                action_count,
                &actor_config.aux_value_horizons,
            );
            return GameRecord {
                reward,
                samples,
                initial_state: initial_state.clone(),
                actions: action_log,
            };
        }

        let sign = match search.state().status() {
            Status::Decision(s) => s,
            _ => 1.0,
        };

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

        // MCTS search
        let result = run_search(search, encoder, &mut encode_buf, &infer, rng).await;

        // Choose action (exploration vs exploitation)
        let exploring = action_count < actor_config.explore_actions;
        let chosen = if exploring {
            sample_from_policy(&result.policy, rng)
        } else {
            result.selected_action
        };

        samples.push(make_sample(
            result,
            features_buf,
            sign,
            is_full_search,
            action_count,
            actor_config.num_aux_targets,
        ));

        action_count += 1;
        action_log.push(chosen);
        search.apply_action(chosen);
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

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
            let q_cp = samples[i].q_wdl.q();
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
        if p > 0.0 {
            roll -= p;
            if roll < 0.0 {
                return i;
            }
        }
    }
    // Fallback: return argmax (only among positive entries)
    policy
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > 0.0)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
